//! `EdaRunner`: licensed host EDA tools, only through `eda-run` in an interactive bash that inherits
//! the session environment (Lmod), never by sourcing `eda-on` in the driver
//! (docs/context/tooling/eda_environment.md §2–§5; spec §3.6).
//!
//! Per call: a job directory as CWD (tools drop logs there), `TMPDIR` on the scratch disk (`/tmp` is
//! 8 GB here), a `MemAvailable` check before launch, the process-group RSS watchdog and timeout of
//! `HostRunner`, at most `max_licenses` concurrent calls, and license failures retried with backoff
//! and reported as `CoreError::License`, never as a tool result.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::cancel::CancelToken;
use crate::config::LoadedConfig;
use crate::error::{CoreError, IoContext, Result};
use crate::runner::host::run_watched;
use crate::runner::process;
use crate::runner::{CallLog, Caps, Invocation, ToolLog, ToolOutcome, ToolRunner, sanitize};

/// What `bash -i` prints on stderr when it has no terminal (observed; eda_environment.md §2).
const BASH_NO_TTY: [&str; 2] = [
    "bash: cannot set terminal process group",
    "bash: no job control in this shell",
];

#[derive(Debug, Clone)]
pub struct EdaSettings {
    pub eda_toggle: Option<PathBuf>,
    /// `TMPDIR` parent; `None` falls back to `<job dir>/.tmp`.
    pub scratch_dir: Option<PathBuf>,
    pub watchdog_interval: Duration,
    pub max_licenses: usize,
    pub license_retries: u32,
    pub license_backoff: Duration,
    pub license_error_patterns: Vec<String>,
}

impl EdaSettings {
    pub fn from_config(cfg: &LoadedConfig) -> Self {
        let eda = &cfg.config.tools.eda;
        Self {
            eda_toggle: cfg.config.paths.eda_toggle.as_ref().map(|p| cfg.resolve(p)),
            scratch_dir: cfg.config.paths.scratch_dir.as_ref().map(|p| cfg.resolve(p)),
            watchdog_interval: Duration::from_secs_f64(eda.watchdog_interval_s),
            max_licenses: eda.max_licenses.max(1),
            license_retries: eda.license_retries,
            license_backoff: Duration::from_secs_f64(eda.license_backoff_s.max(0.0)),
            license_error_patterns: eda.license_error_patterns.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EdaRunner {
    pub settings: EdaSettings,
    slots: Arc<Slots>,
}

impl EdaRunner {
    pub fn new(settings: EdaSettings) -> Self {
        let slots = Arc::new(Slots::new(settings.max_licenses.max(1)));
        Self { settings, slots }
    }

    /// The verified form: `bash --norc -i -c '. "$EDA_TOGGLE"; eda-run "$@"' eda-run <tool> … </dev/null`
    /// with `EDA_TOGGLE` in the environment. Every tool fix for this host lives in the EDA scripts
    /// (`eda-toggle.sh`, `setup_env.sh`), not here.
    pub fn eda_command(&self, inv: &Invocation) -> Result<EdaCommand> {
        let toggle = self.settings.eda_toggle.as_ref().ok_or_else(|| CoreError::Tool {
            tool: inv.tool.clone(),
            message: "paths.eda_toggle is not set (config.local.toml)".into(),
        })?;
        let mut args: Vec<String> = vec![
            "--norc".into(),
            "-i".into(),
            "-c".into(),
            r#". "$EDA_TOGGLE"; eda-run "$@""#.into(),
            "eda-run".into(),
            inv.program.clone(),
        ];
        args.extend(inv.args.iter().cloned());
        Ok(EdaCommand {
            program: "bash".into(),
            args,
            env: vec![("EDA_TOGGLE".into(), toggle.display().to_string())],
        })
    }

    /// Per-call `TMPDIR`: `<scratch>/tmp/eda/<pass>-<n>-<tool>` (the build area's ephemeral `tmp/`, removed after
    /// a successful call).
    fn tmp_dir(&self, log: &ToolLog, cwd: &Path, n: u64, tool: &str) -> PathBuf {
        let leaf = format!("{}-{n:04}-{}", sanitize(log.pass()), sanitize(tool));
        match &self.settings.scratch_dir {
            Some(s) => s.join("tmp").join("eda").join(leaf),
            None => cwd.join(".tmp").join(leaf),
        }
    }

    /// The first output line matching a license-failure pattern.
    fn license_failure(&self, out: &ToolOutcome) -> Result<Option<String>> {
        if self.settings.license_error_patterns.is_empty() {
            return Ok(None);
        }
        for path in [&out.stdout, &out.stderr] {
            let text = std::fs::read(path).at(path)?;
            let text = String::from_utf8_lossy(&text);
            if let Some(line) = text.lines().find(|l| {
                self.settings
                    .license_error_patterns
                    .iter()
                    .any(|p| l.contains(p.as_str()))
            }) {
                return Ok(Some(line.trim().to_string()));
            }
        }
        Ok(None)
    }
}

/// The host command line `EdaRunner` executes for one invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdaCommand {
    pub program: String,
    pub args: Vec<String>,
    /// Added to the inherited environment (which must keep Lmod's `MODULEPATH` / `module`).
    pub env: Vec<(String, String)>,
}

impl ToolRunner for EdaRunner {
    fn name(&self) -> &'static str {
        "eda"
    }

    fn run(&self, inv: &Invocation, caps: &Caps, log: &ToolLog, cancel: &CancelToken) -> Result<ToolOutcome> {
        cancel.check()?;
        let tool_err = |message: String| CoreError::Tool {
            tool: inv.tool.clone(),
            message,
        };
        let cwd = inv
            .cwd
            .clone()
            .ok_or_else(|| tool_err("EdaRunner needs a job directory (Invocation::cwd)".into()))?;
        std::fs::create_dir_all(&cwd).at(&cwd)?;
        let cmd = self.eda_command(inv)?;
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            // Spec §8: check free memory before launching; the watchdog then enforces the cap.
            if let Some(avail) = process::mem_available_bytes()
                && avail < caps.memory_bytes()
            {
                return Err(tool_err(format!(
                    "MemAvailable {:.1} GB is below the call's cap {:.1} GB; not starting",
                    avail as f64 / (1u64 << 30) as f64,
                    caps.memory_gb
                )));
            }
            let slot = self.slots.acquire(cancel)?;
            let call = CallLog::start(log, &inv.tool)?;
            let tmp = self.tmp_dir(log, &cwd, call.files.n, &inv.tool);
            std::fs::create_dir_all(&tmp).at(&tmp)?;
            let mut env = cmd.env.clone();
            env.push(("TMPDIR".into(), tmp.display().to_string()));
            env.extend(inv.env.iter().cloned());
            let host_inv = Invocation {
                tool: inv.tool.clone(),
                program: cmd.program.clone(),
                args: cmd.args.clone(),
                env,
                cwd: Some(cwd.clone()),
                ..Invocation::default()
            };
            let outcome = run_watched(
                self.name(),
                &host_inv,
                caps,
                &call,
                cancel,
                self.settings.watchdog_interval,
            )?;
            drop(slot);
            filter_bash_noise(&outcome.stderr)?;
            call.finish(caps, &outcome)?;
            let license = self.license_failure(&outcome)?;
            // Keep a failed call's temp files for inspection; the scratch disk is cleaned per run.
            if outcome.success() || license.is_some() {
                let _ = std::fs::remove_dir_all(&tmp);
            }
            let Some(line) = license else {
                return Ok(outcome);
            };
            if attempt > self.settings.license_retries {
                return Err(CoreError::License {
                    tool: inv.tool.clone(),
                    attempts: attempt,
                    message: line,
                });
            }
            tracing::warn!(tool = %inv.tool, attempt, "license failure, retrying: {line}");
            sleep_cancellable(
                self.settings.license_backoff * 2u32.saturating_pow(attempt - 1),
                cancel,
            )?;
        }
    }
}

/// Drop bash's two no-TTY warnings from a stderr log, keep everything else.
fn filter_bash_noise(path: &Path) -> Result<()> {
    let text = std::fs::read(path).at(path)?;
    let text = String::from_utf8_lossy(&text);
    if !text.lines().any(|l| BASH_NO_TTY.iter().any(|p| l.starts_with(p))) {
        return Ok(());
    }
    let mut kept: String = text
        .lines()
        .filter(|l| !BASH_NO_TTY.iter().any(|p| l.starts_with(p)))
        .collect::<Vec<_>>()
        .join("\n");
    if !kept.is_empty() {
        kept.push('\n');
    }
    std::fs::write(path, kept).at(path)
}

fn sleep_cancellable(d: Duration, cancel: &CancelToken) -> Result<()> {
    let end = std::time::Instant::now() + d;
    while std::time::Instant::now() < end {
        cancel.check()?;
        std::thread::sleep(Duration::from_millis(100).min(end - std::time::Instant::now()));
    }
    cancel.check()
}

/// Counting semaphore over concurrent EDA calls (`tools.eda.max_licenses`), cancel-aware.
#[derive(Debug)]
struct Slots {
    max: usize,
    used: Mutex<usize>,
    cv: Condvar,
}

struct SlotGuard<'a>(&'a Slots);

impl Slots {
    fn new(max: usize) -> Self {
        Self {
            max,
            used: Mutex::new(0),
            cv: Condvar::new(),
        }
    }

    fn acquire(&self, cancel: &CancelToken) -> Result<SlotGuard<'_>> {
        let mut used = self.used.lock().unwrap_or_else(|p| p.into_inner());
        while *used >= self.max {
            cancel.check()?;
            used = self
                .cv
                .wait_timeout(used, Duration::from_millis(100))
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
        *used += 1;
        Ok(SlotGuard(self))
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        let mut used = self.0.used.lock().unwrap_or_else(|p| p.into_inner());
        *used = used.saturating_sub(1);
        self.0.cv.notify_one();
    }
}

/// The licensed tools and their batch-mode flags and native resource caps
/// (docs/context/tooling/synopsys.md §2, §8; cadence.md §2, §3, §6). Passes put `cap_preamble`
/// first in every script; the RSS watchdog stays the backstop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdaTool {
    Dc,
    Lc,
    Fm,
    Pt,
    Genus,
    Lec,
}

impl EdaTool {
    pub fn program(self) -> &'static str {
        match self {
            EdaTool::Dc => "dc_shell",
            EdaTool::Lc => "lc_shell",
            EdaTool::Fm => "fm_shell",
            EdaTool::Pt => "pt_shell",
            EdaTool::Genus => "genus",
            EdaTool::Lec => "lec",
        }
    }

    /// Arguments that run `script` in batch mode.
    pub fn batch_args(self, script: &str) -> Vec<String> {
        let v: &[&str] = match self {
            // -no_home_init: ~/.synopsys_dc.setup cannot change results (dc_shell(1)).
            EdaTool::Dc => &["-no_home_init", "-f", script],
            EdaTool::Lc | EdaTool::Fm | EdaTool::Pt => &["-f", script],
            EdaTool::Genus => &[
                "-batch",
                "-no_gui",
                "-disable_user_startup",
                "-abort_on_error",
                "-files",
                script,
            ],
            // -XL: datapath analysis needs the XL tier; -nolicwait: fail fast, EdaRunner retries.
            EdaTool::Lec => &["-XL", "-nogui", "-nolicwait", "-dofile", script],
        };
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Script lines that cap the tool's own threads (and memory where it has a cap).
    pub fn cap_preamble(self, caps: &Caps) -> Vec<String> {
        let n = (caps.cpus.floor() as u32).max(1);
        match self {
            // DC: default 1 core, at most 16 (set_host_options(2)).
            EdaTool::Dc => vec![format!("set_host_options -max_cores {}", n.min(16))],
            EdaTool::Pt => vec![format!("set_host_options -max_cores {n}")],
            // Formality: more than 4 cores takes a second license; -max_memory is in GB (set_host_options(2), fm).
            EdaTool::Fm => vec![format!(
                "set_host_options -max_cores {} -max_memory {}",
                n.min(4),
                (caps.memory_gb.floor() as u64).max(1)
            )],
            // Genus defaults to 8 CPUs (genus_attref: max_cpus_per_server).
            EdaTool::Genus => vec![format!("set_db / .max_cpus_per_server {n}")],
            // Each LEC thread may take a license (conformal.hlp: SET PARALLEL OPTION).
            EdaTool::Lec => vec![format!("set parallel option -threads {}", n.min(4))],
            EdaTool::Lc => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn settings(toggle: Option<PathBuf>) -> EdaSettings {
        EdaSettings {
            eda_toggle: toggle,
            scratch_dir: None,
            watchdog_interval: Duration::from_millis(50),
            max_licenses: 2,
            license_retries: 1,
            license_backoff: Duration::from_millis(10),
            license_error_patterns: vec!["(SEC-12)".into()],
        }
    }

    #[test]
    fn command_matches_verified_form() {
        let r = EdaRunner::new(settings(Some("/home/u/bin/eda-toggle.sh".into())));
        let c = r
            .eda_command(&Invocation::new("dc", "dc_shell").args(["-f", "x.tcl"]))
            .unwrap();
        assert_eq!(c.program, "bash");
        assert_eq!(
            c.args.join(" "),
            r#"--norc -i -c . "$EDA_TOGGLE"; eda-run "$@" eda-run dc_shell -f x.tcl"#
        );
        assert_eq!(
            c.env,
            [("EDA_TOGGLE".to_string(), "/home/u/bin/eda-toggle.sh".to_string())]
        );
    }

    #[test]
    fn caps_respect_tool_limits() {
        let caps = Caps {
            memory_gb: 12.5,
            cpus: 32.0,
            timeout_s: 60.0,
            pids_limit: 64,
        };
        assert_eq!(
            EdaTool::Dc.cap_preamble(&caps),
            ["set_host_options -max_cores 16"]
        );
        assert_eq!(
            EdaTool::Fm.cap_preamble(&caps),
            ["set_host_options -max_cores 4 -max_memory 12"]
        );
        assert_eq!(
            EdaTool::Genus.cap_preamble(&caps),
            ["set_db / .max_cpus_per_server 32"]
        );
        assert_eq!(
            EdaTool::Lec.cap_preamble(&caps),
            ["set parallel option -threads 4"]
        );
        assert!(
            EdaTool::Lec
                .batch_args("x.do")
                .contains(&"-nolicwait".to_string())
        );
    }

    #[test]
    fn bash_noise_is_filtered() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("e");
        std::fs::write(
            &p,
            "bash: cannot set terminal process group (-1): Inappropriate ioctl for device\n\
             bash: no job control in this shell\nreal error\n",
        )
        .unwrap();
        filter_bash_noise(&p).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "real error\n");
    }
}
