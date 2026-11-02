//! Deep suite (local only, `heavy` test group, one test at a time): the runners against real processes
//! and a real container. Check `free -g` first (design §8). Caps here are 64 MiB, so a broken
//! cap costs at most a few hundred MB before the test fails.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod components;
mod datapaths;
mod env;
mod export;
mod import;
mod iteration_space;
mod kernel_classes;
mod kernels;
mod normalize;

use std::time::Duration;

use cc_core::CancelToken;
use cc_core::config::DockerConfig;
use cc_core::runner::{Caps, DockerRunner, HostRunner, Invocation, KillReason, ToolLog, ToolRunner};

/// Already on the host (`docker images`, 2026-10-30): no pull needed.
const IMAGE: &str = "ubuntu:24.04";

const SMALL: Caps = Caps {
    memory_gb: 0.0625,
    cpus: 1.0,
    timeout_s: 60.0,
    pids_limit: 64,
};

/// `tail` buffers its whole input when it contains no newline: resident memory grows to ~N MiB.
fn hog(mib: u32) -> Invocation {
    Invocation::new("hog", "sh").args(["-c".to_string(), format!("head -c {mib}m /dev/zero | tail")])
}

#[test]
fn docker_memory_cap_kills_and_reports_oom() {
    let d = tempfile::tempdir().unwrap();
    let log = ToolLog::new("deep", d.path());
    let r = DockerRunner::from_config(DockerConfig::default());
    let out = r
        .run(&hog(300).image(IMAGE), &SMALL, &log, &CancelToken::new())
        .unwrap();
    assert_eq!(out.killed, Some(KillReason::OomKilled), "{out:#?}");
    assert_eq!(out.exit_code, Some(137), "{out:#?}");
    assert!(out.check().is_err());
    let argv = std::fs::read_to_string(d.path().join("tools.jsonl")).unwrap();
    assert!(argv.contains("--memory") && argv.contains("64m"));
}

#[test]
fn docker_runs_as_the_invoking_user() {
    let d = tempfile::tempdir().unwrap();
    let log = ToolLog::new("deep", d.path());
    let r = DockerRunner::from_config(DockerConfig::default());
    let inv = Invocation::new("id", "sh")
        .args(["-c", "id -u; id -g"])
        .image(IMAGE);
    let out = r
        .run(&inv, &SMALL, &log, &CancelToken::new())
        .unwrap()
        .check()
        .unwrap();
    let ids: Vec<u32> = out
        .stdout_string()
        .unwrap()
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    assert_eq!(ids, [r.uid, r.gid]);
}

#[test]
fn docker_timeout_kills_the_container() {
    let d = tempfile::tempdir().unwrap();
    let log = ToolLog::new("deep", d.path());
    let r = DockerRunner::from_config(DockerConfig::default());
    let caps = Caps {
        timeout_s: 2.0,
        ..SMALL
    };
    let out = r.run(
        &Invocation::new("sleep", "sleep").arg("120").image(IMAGE),
        &caps,
        &log,
        &CancelToken::new(),
    );
    let out = out.unwrap();
    assert_eq!(out.killed, Some(KillReason::Timeout));
    assert!(out.duration_ms < 30_000, "{out:#?}");
}

#[test]
fn host_watchdog_kills_a_process_group_over_budget() {
    let d = tempfile::tempdir().unwrap();
    let log = ToolLog::new("deep", d.path());
    let r = HostRunner::new(Duration::from_millis(50));
    let out = r.run(&hog(400), &SMALL, &log, &CancelToken::new()).unwrap();
    assert_eq!(out.killed, Some(KillReason::MemoryCap), "{out:#?}");
    assert!(out.peak_rss_bytes.unwrap() > SMALL.memory_bytes());
    assert_eq!(out.signal, Some(9));
}

#[test]
fn host_watchdog_leaves_small_processes_alone() {
    let d = tempfile::tempdir().unwrap();
    let log = ToolLog::new("deep", d.path());
    let r = HostRunner::new(Duration::from_millis(50));
    let out = r.run(&hog(16), &SMALL, &log, &CancelToken::new()).unwrap();
    assert!(out.success(), "{out:#?}");
}

#[test]
fn host_timeout_kills_sleep() {
    let d = tempfile::tempdir().unwrap();
    let log = ToolLog::new("deep", d.path());
    let caps = Caps {
        timeout_s: 1.0,
        ..SMALL
    };
    let out = HostRunner::new(Duration::from_millis(100))
        .run(
            &Invocation::new("sleep", "sleep").arg("60"),
            &caps,
            &log,
            &CancelToken::new(),
        )
        .unwrap();
    assert_eq!(out.killed, Some(KillReason::Timeout));
    assert!(out.duration_ms < 5000);
}
