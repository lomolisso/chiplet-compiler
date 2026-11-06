//! Deep checks of the float glue (`cc-float`): every wrapper of the standard set (f32, bf16, f16,
//! E4M3FN, and the E5M2 conversions) around stand-in core instances is written as SMT-LIB by Yosys with
//! its cores cut, bound by their contracts and compared with `cc_smt`'s specification: every proof
//! `unsat`; the negative controls (a wrapper without its flush, an encoder rounding toward zero or
//! without its overflow case, a missing NaN or signed-zero case) `sat`. Runs in the build area's
//! `runs/deep-float/`, removed when the test passes (kept with `CC_FLOAT_KEEP=1`). With `CC_FLOAT_RECORDS=<file>` the records (verdict,
//! solver, time) are also written there.

use std::sync::Arc;

use cc_core::runner::{BoundRunner, Caps, Runners, ToolLog};
use cc_core::{CancelToken, LoadedConfig};
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
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &[]).unwrap();
    let only = std::env::var("CC_FLOAT_ONLY").ok();

    let mut set: Vec<cc_float::Built> = glue::standard(&[
        Format::F32,
        Format::BF16,
        Format::F16,
        Format::E4M3FN,
    ])
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
    ];
    for (g, m) in &controls {
        set.push(built(g, *m));
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
