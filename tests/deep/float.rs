//! Deep checks of the float glue (`cc-float`): every wrapper of the standard set (f32, bf16, f16,
//! E4M3FN, and the E5M2 conversions) around stand-in core instances is written as SMT-LIB by Yosys with
//! its cores cut, bound by their contracts and compared with `cc_smt`'s specification: every proof
//! `unsat`; the negative controls (a wrapper without its flush, an encoder rounding toward zero or
//! without its overflow case, a missing NaN or signed-zero case) `sat`. The compositions (power, tanh,
//! log1p, rsqrt at the inner precision the default budget sizes) are proven against their special values
//! and recipe with every core but the final conversion a free function. Runs in the build area's
//! `runs/deep-float/`, removed when the test passes (kept with `CC_FLOAT_KEEP=1`). With `CC_FLOAT_RECORDS=<file>` the records (verdict,
//! solver, time) are also written there.

use std::sync::Arc;

use cc_core::adapters::Accuracy;
use cc_core::runner::{BoundRunner, Caps, Runners, ToolLog};
use cc_core::{CancelToken, LoadedConfig};
use cc_float::compose::{self, Budget, Recipe};
use cc_float::glue::{self, Glue, stand_in};
use cc_float::proof::Prover;
use cc_float::rtl::Mutation;
use cc_smt::float::{Arith, Format};
use cc_smt::{Cache, Location, Solver};
use tests::repo_root;

fn built(g: &Glue, m: Mutation) -> cc_float::Built {
    let cores: Vec<_> = g.needs().unwrap().iter().map(stand_in).collect();
    g.build(&cores, m).unwrap()
}

#[test]
fn float_glue_proofs_hold_and_controls_fail() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let dir = std::path::Path::new(&area).join("runs").join("deep-float");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // CC_FLOAT_CACHE=<dir>: another answer cache (an empty one measures every solver call).
    let sets: Vec<String> = std::env::var("CC_FLOAT_CACHE")
        .map(|d| vec![format!("paths.smt_cache={d}")])
        .unwrap_or_default();
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &sets).unwrap();
    let only = std::env::var("CC_FLOAT_ONLY").ok();

    let mut set: Vec<cc_float::Built> =
        glue::standard(&[Format::F32, Format::BF16, Format::F16, Format::E4M3FN])
            .iter()
            .map(|g| built(g, Mutation::None))
            .collect();
    for from in [Format::F32, Format::F16, Format::BF16] {
        set.push(built(
            &Glue::Convert {
                from,
                to: Format::E5M2,
            },
            Mutation::None,
        ));
    }
    let controls = [
        (
            Glue::Arith {
                op: Arith::Mul,
                fmt: Format::F32,
            },
            Mutation::NoFlush,
        ),
        (
            Glue::Arith {
                op: Arith::Add,
                fmt: Format::BF16,
            },
            Mutation::NoFlush,
        ),
        (
            Glue::Convert {
                from: Format::F32,
                to: Format::E4M3FN,
            },
            Mutation::TruncatingEncoder,
        ),
        (
            Glue::Arith {
                op: Arith::Mul,
                fmt: Format::E4M3FN,
            },
            Mutation::NoOverflowCase,
        ),
        (
            Glue::MaxMin {
                max: true,
                fmt: Format::F32,
            },
            Mutation::NoNanCase,
        ),
        (
            Glue::MaxMin {
                max: false,
                fmt: Format::BF16,
            },
            Mutation::NoZeroOrder,
        ),
        (
            Glue::ToInt {
                fmt: Format::F32,
                signed: true,
                w: 32,
            },
            Mutation::NoNanCase,
        ),
        (Glue::Clamp { fmt: Format::F16 }, Mutation::NoNanCase),
        (
            Glue::Round {
                fmt: Format::F32,
                rnd: cc_core::adapters::Rounding::Ceil,
            },
            Mutation::NoZeroOrder,
        ),
        (
            Glue::FromInt {
                fmt: Format::E4M3FN,
                signed: true,
                w: 8,
                via: Format::F16,
            },
            Mutation::TruncatingEncoder,
        ),
    ];
    // Compositions at the inner precision the budget sizes for 2-ulp cores.
    let budget = Budget::cuda();
    for fmt in [Format::F32, Format::BF16, Format::F16, Format::E4M3FN] {
        for (recipe, roles) in [
            (Recipe::Power, &["log", "exp"][..]),
            (Recipe::Tanh, &["exp"][..]),
            (Recipe::LogPlusOne, &["log"][..]),
            (Recipe::Rsqrt, &["sqrt"][..]),
        ] {
            let cores = roles
                .iter()
                .map(|r| (r.to_string(), Accuracy::Ulp { ulp: 2.0 }))
                .collect();
            let op = cc_float::recipe::op_name(recipe);
            let k = compose::size(op, fmt, recipe, &budget, &cores).unwrap().k.max(1);
            set.push(built(
                &Glue::Compose {
                    recipe,
                    fmt,
                    q: fmt.m + 1 + k,
                },
                Mutation::None,
            ));
        }
    }
    // One power core with the special values as glue (faithful core: the budget's k).
    for fmt in [Format::F32, Format::BF16, Format::F16, Format::E4M3FN] {
        let cores = [("pow".to_string(), Accuracy::Faithful)].into();
        let k = compose::size("power", fmt, Recipe::PowerCore, &budget, &cores)
            .unwrap()
            .k;
        set.push(built(
            &Glue::Compose {
                recipe: Recipe::PowerCore,
                fmt,
                q: fmt.m + 1 + k,
            },
            Mutation::None,
        ));
    }
    for (g, m) in &controls {
        set.push(built(g, *m));
    }
    for recipe in [
        Recipe::Power,
        Recipe::LogPlusOne,
        Recipe::Rsqrt,
        Recipe::Tanh,
        Recipe::PowerCore,
    ] {
        set.push(built(
            &Glue::Compose {
                recipe,
                fmt: Format::F32,
                q: 30,
            },
            Mutation::NoNanCase,
        ));
    }
    if let Some(o) = &only {
        set.retain(|b| b.rtl.name().contains(o.as_str()));
    }

    let runners = Runners::from_config(&cfg);
    let runner = BoundRunner::new(
        Arc::clone(&runners.docker),
        Caps {
            memory_gb: 32.0,
            cpus: 8.0,
            timeout_s: 3600.0,
            pids_limit: 512,
        },
        ToolLog::new("deep-float", dir.join("logs")),
        CancelToken::new(),
    );
    let prover = Prover {
        runner,
        location: Location::Image,
        solver: Solver::parse(&cfg.config.tools.smt.solver).unwrap(),
        timeout_s: 1200,
        cache: Cache::new(cfg.smt_cache_dir()),
        jobs: 8,
    };
    let records = prover.prove(&dir, &set).unwrap();
    let json = serde_json::to_string_pretty(&records).unwrap();
    std::fs::write(dir.join("proofs.json"), &json).unwrap();
    if let Ok(p) = std::env::var("CC_FLOAT_RECORDS") {
        std::fs::write(p, &json).unwrap();
    }
    for r in &records {
        eprintln!(
            "{:<55} {:<8} {:<10} {:>8.1} s{}",
            r.name,
            r.solver.name(),
            match &r.verdict {
                cc_smt::Verdict::Sat { .. } => "sat".to_string(),
                v => format!("{v:?}").to_lowercase(),
            },
            r.seconds,
            if r.cached { " (cached)" } else { "" }
        );
    }
    let bad: Vec<_> = records
        .iter()
        .filter(|r| !r.ok())
        .map(|r| (r.name.clone(), r.verdict.clone()))
        .collect();
    assert!(bad.is_empty(), "{bad:#?}");
    assert_eq!(records.len(), set.len());
    if std::env::var("CC_FLOAT_KEEP").is_err() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
