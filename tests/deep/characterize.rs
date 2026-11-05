//! Deep checks of `characterize`.
//!
//! - `chipc doctor` (the `.db` files and the merged ASAP7 Liberty) into the build area's ephemeral
//!   `runs/deep-characterize/`, then `characterize` on a reduced catalog (`add` and `multiply` at 8 bits and
//!   f32, their accumulators, a 2-input mux, a register, a 16-from-8 split multiplier) on both kits at both
//!   timing points, through every synthesis flow (the `SynthFlow` conformance suite: the same catalog,
//!   every check of `verify` green): Design Compiler without and then with Formality (the second run
//!   reuses every synthesis and runs only the check), and Yosys + ABC. A golden snapshot keeps the keys,
//!   and the directory is removed when the test passes.
//! - The job directories this code computes for the default configuration are the ones of the standing
//!   Design Compiler library in the build area's `build/.characterize/` (read only), so a library
//!   characterized by an earlier version is reused.

use cc_core::runner::Runners;
use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::characterize::LIBRARY;
use cc_passes::characterize::json::{Library, Summary};
use cc_pipeline::env::{DoctorOptions, doctor};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

const REDUCED: &[&str] = &[
    "passes.characterize.ops=[\"add\", \"multiply\"]",
    "passes.characterize.int_widths=[8]",
    "passes.characterize.float_formats=[\"f32\"]",
    "passes.characterize.mux_inputs=[2]",
    "passes.characterize.mux_widths=[8]",
    "passes.characterize.register_widths=[8]",
    "passes.characterize.split_multipliers=[[16, 8]]",
];

#[test]
fn characterize_reduced_catalog_on_every_flow() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-characterize");
    let base = vec![format!("paths.build_dir={}", build.display())];
    let load = |extra: &[String]| {
        let mut sets = base.clone();
        sets.extend(REDUCED.iter().map(|s| s.to_string()));
        sets.extend(extra.iter().cloned());
        LoadedConfig::load(&repo_root().join("config.toml"), &sets).unwrap()
    };
    let cfg = load(&[]);
    assert!(
        cfg.config.paths.eda_toggle.is_some(),
        "deep tests need config.local.toml"
    );

    let mut text = Vec::new();
    let env = doctor(
        &cfg,
        &Runners::from_config(&cfg),
        DoctorOptions::default(),
        &mut text,
    )
    .unwrap();
    for name in ["pdk.asap7_merged", "pdk.db", "dc.nangate45", "dc.asap7"] {
        let c = env.checks.iter().find(|c| c.name == name).unwrap();
        assert_eq!(
            c.status,
            CheckStatus::Pass,
            "{name}: {}",
            String::from_utf8_lossy(&text)
        );
    }

    let mut digest = Vec::new();
    for (tool, lec) in [("dc", false), ("dc", true), ("yosys", false)] {
        let extra = vec![
            format!("passes.characterize.tool=\"{tool}\""),
            format!("passes.characterize.lec={lec}"),
        ];
        let cfg = load(&extra);
        let p = Pipeline::new(cc_passes::registry(), cfg).unwrap();
        let s = p
            .run(&RunOptions {
                targets: vec!["characterize".into()],
                ..RunOptions::default()
            })
            .unwrap();
        let dir = p.store().pass_dir("characterize");
        let report = dir.read_verify().unwrap().unwrap();
        let failed: Vec<_> = report
            .checks
            .iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| (c.name.clone(), c.evidence.notes.clone()))
            .collect();
        assert!(failed.is_empty(), "{tool}: {failed:#?}");
        assert!(
            matches!(
                s.status_of("characterize"),
                Some(RunStatus::Ok | RunStatus::CacheHit)
            ),
            "{tool}: {s:?}"
        );
        let lib: Library = dir.kind(LIBRARY).read_json("library.json").unwrap();
        let summary: Summary = dir.kind(LIBRARY).read_json("summary.json").unwrap();
        // Proven, or out of time with no failing point (the row says so); never a failing point.
        let open: Vec<String> = lib
            .rows
            .iter()
            .filter(|r| r.lec.as_ref().is_some_and(|l| !l.equivalent))
            .map(|r| format!("{} {} {}", r.key, r.kit, r.target))
            .collect();
        assert!(
            lib.rows.iter().all(|r| match &r.lec {
                Some(l) => lec && l.failing == 0 && (l.equivalent || r.inconclusive),
                None => !lec,
            }),
            "{tool}: {open:?}"
        );
        if tool == "dc" && lec {
            // Turning the check on reran only Formality: every synthesis was reused.
            let log = std::fs::read_to_string(dir.logs_dir().join("tools.jsonl")).unwrap_or_default();
            assert!(!log.contains("\"dc_shell\""), "{tool}: synthesis ran again");
            assert!(summary.jobs_run > 0, "{tool}: {summary:?}");
        }
        // A wider multiply costs more than a narrower one, and a split 16-bit multiply has its pieces.
        for kit in ["nangate45", "asap7"] {
            let area = |key: &str| {
                lib.rows
                    .iter()
                    .find(|r| r.key == key && r.kit == kit && r.target == "clock")
                    .and_then(|r| r.measured.area_um2)
                    .unwrap()
            };
            assert!(area("multiply@i8") > area("add@i8"), "{tool} {kit}");
            assert!(area("multiply@f32") > area("add@f32"), "{tool} {kit}");
            if tool == "dc" {
                let split = lib
                    .rows
                    .iter()
                    .find(|r| r.key == "split@w16n8" && r.kit == kit && r.target == "clock")
                    .unwrap();
                // The three piece instances (u<i>_<j>), besides any DesignWare block of the sum.
                let pieces = split
                    .measured
                    .children
                    .keys()
                    .filter(|k| k.starts_with('u'))
                    .count();
                assert_eq!(pieces, 3, "{kit}: {:?}", split.measured.children);
            }
            let fast = lib
                .rows
                .iter()
                .find(|r| r.key == "multiply@i8" && r.kit == kit && r.target == "fast")
                .unwrap();
            let clock = lib
                .rows
                .iter()
                .find(|r| r.key == "multiply@i8" && r.kit == kit && r.target == "clock")
                .unwrap();
            assert!(
                fast.measured.delay_ps.unwrap() <= clock.measured.delay_ps.unwrap(),
                "{tool} {kit}"
            );
        }
        let mut keys: Vec<String> = lib
            .rows
            .iter()
            .map(|r| format!("{} {} {}", r.key, r.kit, r.target))
            .collect();
        keys.sort();
        digest.push(serde_json::json!({
            "tool": tool,
            "lec": lec,
            "rows": keys.len(),
            "inconclusive": open,
            "keys": keys,
        }));
    }
    insta::assert_json_snapshot!("characterize_reduced_keys", digest);
    std::fs::remove_dir_all(&build).unwrap();
}

/// The standing Design Compiler library (`<build area>/build/.characterize`, or `CC_STANDING_BUILD`'s): the
/// RTL and synthesis job directories that this code computes for the default configuration (Design
/// Compiler, the equivalence check off) are exactly the ones there. Nothing is written; without the
/// standing library the test has nothing to compare and passes with a note.
#[test]
fn characterize_job_keys_match_the_standing_library() {
    use cc_passes::characterize::{CharacterizeConfig, catalog, flow};
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::env::var("CC_STANDING_BUILD")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::Path::new(&area).join("build"));
    let eda = build.join(".characterize/eda");
    if !eda.is_dir() {
        eprintln!("no standing library at {}: nothing to compare", eda.display());
        return;
    }
    let lc = LoadedConfig::load(
        &repo_root().join("config.toml"),
        &[format!("paths.build_dir={}", build.display())],
    )
    .unwrap();
    let section = lc.pass_section("characterize").unwrap();
    let cfg: CharacterizeConfig = section.parse().unwrap();
    assert!(!cfg.lec && cfg.tool == "dc", "the default configuration");
    let d = &lc.config.tools.docker;
    let image = format!("{}:{}", d.image, d.tag);
    let cat = catalog::catalog(&cfg).unwrap();
    let rtl_dir = eda.join(flow::rtl_dir_name(&image, &cat));
    assert!(
        rtl_dir.join("rtl.done").is_file(),
        "{} is not in the standing library",
        rtl_dir.display()
    );
    let pdks: Vec<_> = cfg
        .pdks
        .iter()
        .map(|k| cc_core::pdk::load(&lc, k).unwrap())
        .collect();
    let env = cc_core::ArtifactStore::new(build.clone()).pass_dir(cc_pipeline::env::doctor::ENV_DIR);
    let kits = flow::kits_of(&env, &pdks, &cfg).unwrap();
    let designs: Vec<_> = cat.iter().filter(|p| p.source.is_some()).map(flow::design).collect();
    let jobs = flow::job_list(
        &cfg,
        &kits,
        &designs,
        &rtl_dir,
        &build.join(".characterize"),
        &image,
        flow::session_caps(&section.resources, &cfg),
    )
    .unwrap();
    let want: std::collections::BTreeSet<String> = jobs
        .iter()
        .map(|j| j.dir.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let have: std::collections::BTreeSet<String> = std::fs::read_dir(&eda)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with("rtl-"))
        .collect();
    let missing: Vec<&String> = want.difference(&have).collect();
    assert!(missing.is_empty(), "{} job directories are not in the standing library: {missing:?}", missing.len());
    assert!(
        jobs.iter().all(|j| j.dir.join("synth.done").is_file()),
        "every job is finished"
    );
    eprintln!("{} job directories found in {}", want.len(), eda.display());
}

/// Deterministic test vectors for `rescale`: edge cases and pseudo-random operands with scales over the
/// exponent range that matters (right shifts, ties, left shifts, saturation, subnormal and zero scales).
fn rescale_vectors(n: u32) -> Vec<(i64, u32)> {
    let lo = -(1i64 << (n - 1));
    let hi = (1i64 << (n - 1)) - 1;
    let mut v = Vec::new();
    for x in [0, 1, -1, 2, -2, 3, -3, 5, -5, lo, hi, lo + 1, hi - 1] {
        for s in [
            0.5f32,
            0.25,
            0.75,
            1.5,
            0.001,
            2f32.powi(-40),
            2f32.powi(23),
            2f32.powi(24),
            2f32.powi(30),
            2f32.powi(31),
            0.0,
            -0.0,
            -0.5,
            -0.75,
            f32::INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
        ] {
            v.push((x, s.to_bits()));
        }
        v.push((x, 1));
    }
    let mut st: u64 = 0x9e37_79b9_7f4a_7c15 ^ u64::from(n);
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    for _ in 0..3000 {
        let r = next();
        let x = ((r as i64) >> (64 - n)).clamp(lo, hi);
        let e = 90 + (next() % 70) as u32;
        let s = ((next() & 1) as u32) << 31 | e << 23 | (next() as u32 & 0x7f_ffff);
        v.push((x, s));
    }
    v
}

#[test]
fn rescale_rtl_matches_its_reference() {
    let env = tests::Env::new();
    let cfg: toml::Value =
        toml::from_str(&std::fs::read_to_string(repo_root().join("config.toml")).unwrap()).unwrap();
    let tag = cfg["tools"]["docker"]["tag"].as_str().unwrap().to_string();
    let lc = env.config("[models]\nkeys = []\n", &[&format!("tools.docker.tag={tag}")]);
    let dir = env.root().join("rescale");
    std::fs::create_dir_all(&dir).unwrap();
    let cat =
        cc_passes::characterize::catalog::catalog(&cc_passes::characterize::CharacterizeConfig::default())
            .unwrap();
    let runner = cc_core::runner::BoundRunner::new(
        Runners::from_config(&lc).docker,
        cc_core::runner::Caps {
            memory_gb: 2.0,
            cpus: 1.0,
            timeout_s: 300.0,
            pids_limit: 64,
        },
        cc_core::runner::ToolLog::new("rescale-rtl", dir.join("logs")),
        cc_core::CancelToken::new(),
    );
    for n in [8u32, 16, 32] {
        let p = cat.iter().find(|p| p.key == format!("rescale@i{n}")).unwrap();
        let Some(cc_passes::characterize::catalog::Source::Verilog(v)) = &p.source else {
            panic!("rescale@i{n}: no Verilog");
        };
        std::fs::write(dir.join(format!("rescale_i{n}.v")), v).unwrap();
        let vs = rescale_vectors(n);
        let mut hex = String::new();
        for &(x, s) in &vs {
            let want = cc_smt::ops::rescale_reference_int(x, s) as u32;
            hex.push_str(&format!("{:08x}{s:08x}{want:08x}\n", x as u32));
        }
        std::fs::write(dir.join(format!("vec_i{n}.hex")), hex).unwrap();
        let tb = format!(
            "module tb;\n  reg [95:0] mem [0:{last}];\n  reg [31:0] x, s, want;\n  wire [31:0] q;\n  integer i, bad;\n  rescale_core_i{n} dut(.a(x[{nh}:0]), .b(s), .q(q));\n  initial begin\n    $readmemh(\"/w/vec_i{n}.hex\", mem);\n    bad = 0;\n    for (i = 0; i <= {last}; i = i + 1) begin\n      {{x, s, want}} = mem[i];\n      #1;\n      if (q !== want) begin\n        bad = bad + 1;\n        if (bad < 10) $display(\"MISMATCH x=%h s=%h q=%h want=%h\", x, s, q, want);\n      end\n    end\n    $display(\"DONE %0d %0d\", {count}, bad);\n  end\nendmodule\n",
            last = vs.len() - 1,
            count = vs.len(),
            nh = n - 1
        );
        std::fs::write(dir.join(format!("tb_i{n}.v")), tb).unwrap();
        let inv = cc_core::runner::Invocation::new("iverilog", "sh")
            .args([
                "-c".to_string(),
                format!(
                    "iverilog -g2012 -s tb -o /tmp/tb.vvp /w/rescale_i{n}.v /w/tb_i{n}.v && vvp -n /tmp/tb.vvp"
                ),
            ])
            .mount(cc_core::runner::Mount {
                src: dir.clone(),
                dst: "/w".into(),
                readonly: true,
            });
        let o = runner.run(&inv).unwrap();
        let out = std::fs::read_to_string(&o.stdout).unwrap();
        let err = std::fs::read_to_string(&o.stderr).unwrap_or_default();
        assert!(
            out.contains(&format!("DONE {} 0", vs.len())),
            "rescale_i{n}: {out}\n{err}"
        );
    }
}
