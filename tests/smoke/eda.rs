//! `EdaRunner` against a fake `eda-toggle.sh` whose `eda-run` just executes its arguments: the real
//! launch path (`bash --norc -i -c …`), watchdog, timeout, license retry and semaphore, without EDA.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cc_core::runner::{Caps, EdaRunner, EdaSettings, Invocation, KillReason, ToolLog, ToolRunner};
use cc_core::{CancelToken, CoreError};

fn fake_toggle(dir: &Path) -> std::path::PathBuf {
    let p = dir.join("eda-toggle.sh");
    std::fs::write(&p, "eda-run() { \"$@\"; }\n").unwrap();
    p
}

fn settings(dir: &Path) -> EdaSettings {
    EdaSettings {
        eda_toggle: Some(fake_toggle(dir)),
        scratch_dir: Some(dir.join("scratch")),
        watchdog_interval: Duration::from_millis(50),
        max_licenses: 2,
        license_retries: 1,
        license_backoff: Duration::from_millis(10),
        license_error_patterns: vec!["(SEC-12)".into()],
    }
}

fn caps(timeout_s: f64, memory_gb: f64) -> Caps {
    Caps {
        memory_gb,
        cpus: 1.0,
        timeout_s,
        pids_limit: 64,
    }
}

fn job(dir: &Path, tool: &str, script: &str) -> Invocation {
    Invocation::new(tool, "bash")
        .args(["-c", script])
        .cwd(dir.join("job").join(tool))
}

#[test]
fn eda_runner_runs_through_eda_run_and_filters_bash_noise() {
    let d = tempfile::tempdir().unwrap();
    let r = EdaRunner::new(settings(d.path()));
    let log = ToolLog::new("synth", d.path().join("logs"));
    let out = r
        .run(
            &job(d.path(), "dc", r#"echo "tmp=$TMPDIR"; pwd; echo real-err >&2"#),
            &caps(30.0, 1.0),
            &log,
            &CancelToken::new(),
        )
        .unwrap()
        .check()
        .unwrap();
    let stdout = out.stdout_string().unwrap();
    let tmp = d.path().join("scratch/chipc/eda-tmp/synth-0001-dc");
    assert!(stdout.contains(&format!("tmp={}", tmp.display())), "{stdout}");
    assert!(stdout.contains("job/dc"), "{stdout}");
    assert_eq!(std::fs::read_to_string(&out.stderr).unwrap(), "real-err\n");
    assert!(!tmp.exists(), "TMPDIR is removed after success");
    assert_eq!(out.argv[0], "bash");
    assert_eq!(out.argv[1..4], ["--norc", "-i", "-c"]);
}

#[test]
fn eda_runner_watchdog_kills_over_the_memory_cap() {
    let d = tempfile::tempdir().unwrap();
    let r = EdaRunner::new(settings(d.path()));
    let out = r
        .run(
            &job(
                d.path(),
                "hog",
                r#"python3 -c 'import time; b = b"a" * (300 << 20); time.sleep(20)'"#,
            ),
            &caps(30.0, 0.1),
            &ToolLog::new("p", d.path().join("logs")),
            &CancelToken::new(),
        )
        .unwrap();
    assert_eq!(out.killed, Some(KillReason::MemoryCap));
}

#[test]
fn eda_runner_timeout_kills_the_group() {
    let d = tempfile::tempdir().unwrap();
    let r = EdaRunner::new(settings(d.path()));
    let out = r
        .run(
            &job(d.path(), "slow", "sleep 30"),
            &caps(0.5, 1.0),
            &ToolLog::new("p", d.path().join("logs")),
            &CancelToken::new(),
        )
        .unwrap();
    assert_eq!(out.killed, Some(KillReason::Timeout));
    assert!(out.duration_ms < 10_000);
}

#[test]
fn eda_runner_retries_license_failures_then_reports_them() {
    let d = tempfile::tempdir().unwrap();
    let r = EdaRunner::new(settings(d.path()));
    let log_dir = d.path().join("logs");
    let e = r
        .run(
            &job(
                d.path(),
                "dc",
                "echo \"Error: Can't communicate with the license server. (SEC-12)\"",
            ),
            &caps(30.0, 1.0),
            &ToolLog::new("p", &log_dir),
            &CancelToken::new(),
        )
        .unwrap_err();
    match e {
        CoreError::License {
            attempts, message, ..
        } => {
            assert_eq!(attempts, 2, "one try + license_retries = 1");
            assert!(message.contains("SEC-12"), "{message}");
        }
        other => panic!("expected a license error, got {other}"),
    }
    let calls = std::fs::read_to_string(log_dir.join("tools.jsonl")).unwrap();
    assert_eq!(calls.lines().count(), 2, "every attempt is logged");
}

#[test]
fn eda_runner_limits_concurrent_calls() {
    let d = tempfile::tempdir().unwrap();
    let r = Arc::new(EdaRunner::new(EdaSettings {
        max_licenses: 1,
        ..settings(d.path())
    }));
    let log = ToolLog::new("p", d.path().join("logs"));
    let t0 = Instant::now();
    let handles: Vec<_> = (0..2)
        .map(|i| {
            let (r, log, dir) = (r.clone(), log.clone(), d.path().to_path_buf());
            std::thread::spawn(move || {
                r.run(
                    &job(&dir, &format!("t{i}"), "sleep 0.6"),
                    &caps(30.0, 1.0),
                    &log,
                    &CancelToken::new(),
                )
                .unwrap()
                .check()
                .unwrap();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert!(t0.elapsed() >= Duration::from_millis(1100), "{:?}", t0.elapsed());
}

#[test]
fn eda_runner_needs_a_job_dir_and_a_toggle() {
    let d = tempfile::tempdir().unwrap();
    let log = ToolLog::new("p", d.path().join("logs"));
    let r = EdaRunner::new(settings(d.path()));
    let no_cwd = r.run(
        &Invocation::new("dc", "dc_shell"),
        &caps(30.0, 1.0),
        &log,
        &CancelToken::new(),
    );
    assert!(matches!(no_cwd, Err(CoreError::Tool { .. })));
    let r = EdaRunner::new(EdaSettings {
        eda_toggle: None,
        ..settings(d.path())
    });
    let no_toggle = r.run(
        &job(d.path(), "dc", "true"),
        &caps(30.0, 1.0),
        &log,
        &CancelToken::new(),
    );
    let msg = no_toggle.unwrap_err().to_string();
    assert!(msg.contains("eda_toggle"), "{msg}");
}
