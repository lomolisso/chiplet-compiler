//! deep checks against the real environment: the toolchain image, `eda-run` and the licenses.
//! They use the repository's `config.toml` + `config.local.toml` (machine paths) and refresh
//! `build/.env/` (the doctor's evidence, which later passes read).

use cc_core::runner::{
    Caps, EdaRunner, EdaSettings, Invocation, KillReason, Mount, Runners, ToolLog, ToolRunner,
};
use cc_core::{CancelToken, CheckStatus, LoadedConfig};
use cc_pipeline::env::{DoctorOptions, doctor};
use tests::repo_root;

fn repo_config() -> LoadedConfig {
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &[]).unwrap();
    assert!(
        cfg.config.paths.eda_toggle.is_some(),
        "deep tests need config.local.toml (copy config.local.toml.example)"
    );
    cfg
}

const CONTAINER: Caps = Caps {
    memory_gb: 2.0,
    cpus: 1.0,
    timeout_s: 300.0,
    pids_limit: 512,
};

#[test]
fn doctor_is_green_on_this_host() {
    let cfg = repo_config();
    let mut out = Vec::new();
    let report = doctor(
        &cfg,
        &Runners::from_config(&cfg),
        DoctorOptions::default(),
        &mut out,
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(report.passed(), "{text}");
    for must in [
        "image.selftest.xls",
        "image.selftest.xls_equiv",
        "image.selftest.grami",
        "image.selftest.allo",
        "pdk.db",
        "dc.nangate45",
        "dc.asap7",
        "license.genus",
        "license.lec",
    ] {
        let c = report
            .checks
            .iter()
            .find(|c| c.name == must)
            .unwrap_or_else(|| panic!("{must} missing"));
        assert_eq!(c.status, CheckStatus::Pass, "{must}: {text}");
    }
    let env = cfg.build_dir().join(".env");
    assert!(env.join("verify.json").is_file() && env.join("manifest.json").is_file());
    assert!(env.join("out/pdk_db/NangateOpenCellLibrary.db").is_file());
}

#[test]
fn eda_watchdog_kills_a_real_eda_run_process_over_its_cap() {
    let cfg = repo_config();
    let d = tempfile::tempdir().unwrap();
    let r = EdaRunner::new(EdaSettings::from_config(&cfg));
    let caps = Caps {
        memory_gb: 0.1,
        cpus: 1.0,
        timeout_s: 120.0,
        pids_limit: 64,
    };
    // Through the real eda-run (module loads included): the group's RSS passes 0.1 GB.
    let inv = Invocation::new("hog", "python3")
        .args(["-c", "import time; b = b'a' * (300 << 20); time.sleep(60)"])
        .cwd(d.path().join("job"));
    let out = r
        .run(
            &inv,
            &caps,
            &ToolLog::new("deep", d.path().join("logs")),
            &CancelToken::new(),
        )
        .unwrap();
    assert_eq!(out.killed, Some(KillReason::MemoryCap), "{out:#?}");
}

#[test]
fn grami_toy_graph_finds_exactly_the_frequent_patterns() {
    let cfg = repo_config();
    let d = tempfile::tempdir().unwrap();
    let runners = Runners::from_config(&cfg);
    let inv = Invocation::new("grami", "java")
        .args([
            "-Xmx512m",
            "-XX:ActiveProcessorCount=1",
            "-XX:+UseSerialGC",
            "-cp",
            "/opt/grami/GRAMI_DIRECTED_SUBGRAPHS/bin",
            "Dijkstra.main",
            "filename=two_paths.lg",
            "datasetFolder=/work/in/",
            "freq=2",
            "limitedtime=false",
            "maxLabelAppearance=-1",
            "approximate=1",
        ])
        .cwd("/work/out")
        .mount(Mount {
            src: repo_root().join("tests/fixtures/grami"),
            dst: "/work/in".into(),
            readonly: true,
        })
        .mount(Mount {
            src: d.path().to_path_buf(),
            dst: "/work/out".into(),
            readonly: false,
        });
    let out = runners
        .docker
        .run(
            &inv,
            &CONTAINER,
            &ToolLog::new("deep", d.path().join("logs")),
            &CancelToken::new(),
        )
        .unwrap()
        .check()
        .unwrap();
    let text = std::fs::read_to_string(d.path().join("Output.txt")).unwrap();
    // Two disjoint copies of 1 -> 2 -> 3: with MNI support >= 2 the frequent connected subgraphs are
    // the edges 1->2 and 2->3 and the path 1->2->3 (GraMi's MNI support, grami.md §1).
    let count: u64 = text.lines().nth(1).unwrap().trim().parse().unwrap();
    assert_eq!(count, 3, "{text}\n{}", out.stdout_string().unwrap());
    let edges_per_pattern: Vec<usize> = text
        .split(':')
        .skip(1)
        .map(|p| p.lines().filter(|l| l.starts_with("e ")).count())
        .collect();
    assert_eq!(edges_per_pattern.iter().filter(|&&e| e == 2).count(), 1, "{text}");
}

#[test]
fn xls_round_trip_in_the_container() {
    let cfg = repo_config();
    let d = tempfile::tempdir().unwrap();
    let runners = Runners::from_config(&cfg);
    let script = "set -euo pipefail; xlscc /work/in/add8.cc --top=add8 --include_dirs=\"$XLSCC_INCLUDE_DIRS\" --defines=\"$XLSCC_DEFINES\" --out=add8.ir; \
                  top=$(sed -n 's/^top fn \\([^(]*\\)(.*/\\1/p' add8.ir); \
                  opt_main add8.ir --top=\"$top\" > add8.opt.ir; \
                  codegen_main add8.opt.ir --top=\"$top\" --generator=pipeline --delay_model=asap7 --clock_period_ps=1000 \
                  --reset=rst --reset_asynchronous=false --use_system_verilog=false --module_name=add8 \
                  --output_verilog_path=add8.v --output_schedule_path=add8.sched.textproto";
    let inv = Invocation::new("xls", "bash")
        .args(["-c", script])
        .cwd("/work/out")
        .mount(Mount {
            src: repo_root().join("tests/fixtures/xls"),
            dst: "/work/in".into(),
            readonly: true,
        })
        .mount(Mount {
            src: d.path().to_path_buf(),
            dst: "/work/out".into(),
            readonly: false,
        });
    runners
        .docker
        .run(
            &inv,
            &CONTAINER,
            &ToolLog::new("deep", d.path().join("logs")),
            &CancelToken::new(),
        )
        .unwrap()
        .check()
        .unwrap();
    let v = std::fs::read_to_string(d.path().join("add8.v")).unwrap();
    assert!(v.contains("module add8("), "{v}");
    assert!(
        v.contains("input wire clk"),
        "pipeline generator emits a clocked module: {v}"
    );
    assert!(d.path().join("add8.sched.textproto").is_file());
}
