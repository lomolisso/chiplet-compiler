//! `HostRunner`: a host process in its own process group, with an RSS watchdog over the whole group,
//! a wall-clock timeout, cancellation, and `RLIMIT_CORE = 0` (spec §3.6).
//!
//! `RLIMIT_AS` is deliberately not set: it limits virtual memory, which JVMs and thread pools reserve
//! far beyond their resident use. The watchdog measures resident memory instead.

use std::fs::File;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::cancel::CancelToken;
use crate::error::{CoreError, IoContext, Result};
use crate::runner::process::{self, Supervisor};
use crate::runner::{CallLog, Caps, Invocation, ToolLog, ToolOutcome, ToolRunner};

#[derive(Debug, Clone)]
pub struct HostRunner {
    pub watchdog_interval: Duration,
}

impl HostRunner {
    pub fn new(watchdog_interval: Duration) -> Self {
        Self { watchdog_interval }
    }
}

impl ToolRunner for HostRunner {
    fn name(&self) -> &'static str {
        "host"
    }

    fn run(&self, inv: &Invocation, caps: &Caps, log: &ToolLog, cancel: &CancelToken) -> Result<ToolOutcome> {
        cancel.check()?;
        let call = CallLog::start(log, &inv.tool)?;
        let mut cmd = Command::new(&inv.program);
        cmd.args(&inv.args).stdin(Stdio::null());
        cmd.stdout(File::create(&call.files.stdout).at(&call.files.stdout)?);
        cmd.stderr(File::create(&call.files.stderr).at(&call.files.stderr)?);
        for (k, v) in &inv.env {
            cmd.env(k, v);
        }
        if let Some(cwd) = &inv.cwd {
            cmd.current_dir(cwd);
        }
        process::isolate(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| CoreError::Tool {
            tool: inv.tool.clone(),
            message: format!("spawn `{}`: {e}", inv.program),
        })?;
        let pgid = child.id() as i32;
        let sample = move || process::group_rss_bytes(pgid);
        let kill = move || process::kill_group(pgid);
        let sup = Supervisor {
            timeout: caps.timeout(),
            cancel,
            sample_interval: self.watchdog_interval,
            cap_bytes: Some(caps.memory_bytes()),
            sample: &sample,
            kill: &kill,
        };
        let res = process::supervise(&mut child, &sup).at(&call.files.stderr)?;
        let (exit_code, signal) = process::exit_parts(&res.status);
        let mut argv = vec![inv.program.clone()];
        argv.extend(inv.args.iter().cloned());
        let outcome = ToolOutcome {
            runner: self.name().into(),
            tool: inv.tool.clone(),
            argv,
            exit_code,
            signal,
            killed: res.killed,
            duration_ms: call.t0.elapsed().as_millis() as u64,
            peak_rss_bytes: res.peak_bytes,
            stdout: call.files.stdout.clone(),
            stderr: call.files.stderr.clone(),
        };
        call.finish(caps, &outcome)?;
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> Caps {
        Caps {
            memory_gb: 1.0,
            cpus: 1.0,
            timeout_s: 10.0,
            pids_limit: 64,
        }
    }

    #[test]
    fn runs_and_logs() {
        let d = tempfile::tempdir().unwrap();
        let log = ToolLog::new("p", d.path());
        let r = HostRunner::new(Duration::from_millis(50));
        let out = r
            .run(
                &Invocation::new("echo", "/bin/echo").arg("hi"),
                &caps(),
                &log,
                &CancelToken::new(),
            )
            .unwrap()
            .check()
            .unwrap();
        assert_eq!(out.stdout_string().unwrap(), "hi\n");
        let jsonl = std::fs::read_to_string(d.path().join("tools.jsonl")).unwrap();
        assert!(jsonl.contains("\"tool\":\"echo\""));
    }

    #[test]
    fn timeout_kills() {
        let d = tempfile::tempdir().unwrap();
        let log = ToolLog::new("p", d.path());
        let c = Caps {
            timeout_s: 0.2,
            ..caps()
        };
        let out = HostRunner::new(Duration::from_millis(50))
            .run(
                &Invocation::new("sleep", "sleep").arg("30"),
                &c,
                &log,
                &CancelToken::new(),
            )
            .unwrap();
        assert_eq!(out.killed, Some(crate::runner::KillReason::Timeout));
        assert!(out.duration_ms < 5000);
    }

    #[test]
    fn cancelled_token_refuses_to_start() {
        let d = tempfile::tempdir().unwrap();
        let t = CancelToken::new();
        t.cancel();
        let r = HostRunner::new(Duration::from_millis(50));
        assert!(matches!(
            r.run(
                &Invocation::new("echo", "/bin/echo"),
                &caps(),
                &ToolLog::new("p", d.path()),
                &t
            ),
            Err(CoreError::Cancelled)
        ));
    }
}
