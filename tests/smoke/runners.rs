use std::time::Duration;

use cc_core::CancelToken;
use cc_core::config::DockerConfig;
use cc_core::runner::{Caps, DockerRunner, HostRunner, Invocation, KillReason, Mount, ToolLog, ToolRunner};
use cc_pipeline::{Pipeline, RunOptions};
use tests::{Env, Fixture, recording_runners, resources};

#[test]
fn docker_run_carries_the_caps() {
    let r = DockerRunner {
        uid: 1291371,
        gid: 1291371,
        ..DockerRunner::from_config(DockerConfig::default())
    };
    let inv = Invocation::new("xls", "codegen_main")
        .args(["--generator=pipeline", "in.ir"])
        .cwd("/work/out")
        .mount(Mount {
            src: "/repo".into(),
            dst: "/work/repo".into(),
            readonly: true,
        })
        .mount(Mount {
            src: "/repo/build/.xls".into(),
            dst: "/work/out".into(),
            readonly: false,
        });
    let caps = Caps {
        memory_gb: 16.0,
        cpus: 8.0,
        timeout_s: 600.0,
        pids_limit: 4096,
    };
    let a = r.docker_args("chipc-xls-1", None, &inv, &caps);
    let joined = a.join(" ");
    for needle in [
        "--memory 16384m",
        "--memory-swap 16384m",
        "--memory-swappiness 0",
        "--cpus 8",
        "--pids-limit 4096",
        "--ulimit core=0",
        "--network none",
        "--init",
        "--user 1291371:1291371",
        "--mount type=bind,src=/repo,dst=/work/repo,readonly",
        "--mount type=bind,src=/repo/build/.xls,dst=/work/out",
        "--workdir /work/out",
    ] {
        assert!(joined.contains(needle), "missing `{needle}` in `{joined}`");
    }
    assert!(
        !a.iter().any(|x| x == "--rm"),
        "named container kept until OOMKilled is read"
    );
    let image = a
        .iter()
        .position(|x| x == "chiplet-compiler/toolchain:dev")
        .unwrap();
    assert_eq!(&a[image + 1..], ["codegen_main", "--generator=pipeline", "in.ir"]);
}

#[test]
fn pass_docker_caps_are_clamped_to_the_pass_declaration() {
    let env = Env::new();
    let cfg_text = format!(
        "[tools.docker]\nmemory_gb = 8\ncpus = 4.0\n{}",
        resources("d", 2.0, 1)
    );
    let cfg = env.config(&cfg_text, &[]);
    let (runners, rec) = recording_runners(&cfg);
    let p = Pipeline::new(vec![Fixture::new("d").docker_call().boxed()], cfg)
        .unwrap()
        .with_runners(runners);
    assert!(p.run(&RunOptions::default()).unwrap().ok);
    let calls = rec.calls.lock().unwrap();
    let caps = calls[0].1;
    assert_eq!((caps.memory_gb, caps.cpus, caps.timeout_s), (2.0, 1.0, 60.0));
    let argv = DockerRunner::from_config(DockerConfig::default()).docker_args("n", None, &calls[0].0, &caps);
    assert!(argv.join(" ").contains("--memory 2048m --memory-swap 2048m"));
}

#[test]
fn host_runner_runs_logs_and_reports_failure() {
    let d = tempfile::tempdir().unwrap();
    let log = ToolLog::new("p", d.path());
    let caps = Caps {
        memory_gb: 1.0,
        cpus: 1.0,
        timeout_s: 10.0,
        pids_limit: 64,
    };
    let r = HostRunner::new(Duration::from_millis(100));
    let ok = r
        .run(
            &Invocation::new("echo", "/bin/echo").arg("x"),
            &caps,
            &log,
            &CancelToken::new(),
        )
        .unwrap();
    assert!(ok.success());
    let bad = r
        .run(
            &Invocation::new("false", "false"),
            &caps,
            &log,
            &CancelToken::new(),
        )
        .unwrap();
    assert_eq!(bad.exit_code, Some(1));
    assert!(bad.check().is_err());
    let lines = std::fs::read_to_string(d.path().join("tools.jsonl")).unwrap();
    assert_eq!(lines.lines().count(), 2);
}

#[test]
fn host_runner_cancel_kills_the_process_group() {
    let d = tempfile::tempdir().unwrap();
    let log = ToolLog::new("p", d.path());
    let caps = Caps {
        memory_gb: 1.0,
        cpus: 1.0,
        timeout_s: 60.0,
        pids_limit: 64,
    };
    let token = CancelToken::new();
    let t2 = token.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        t2.cancel();
    });
    let out = HostRunner::new(Duration::from_millis(100))
        .run(
            &Invocation::new("sh", "sh").args(["-c", "sleep 30 & sleep 30"]),
            &caps,
            &log,
            &token,
        )
        .unwrap();
    assert_eq!(out.killed, Some(KillReason::Cancelled));
    assert!(out.duration_ms < 5000);
}
