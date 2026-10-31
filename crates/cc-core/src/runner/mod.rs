//! Tool runners (design §3.6): the only way a pass starts an external process. Every call gets memory/CPU
//! caps and a timeout, honours the pass's cancel token, and is logged to `build/.<pass>/logs/`
//! (`tools.jsonl` plus one stdout/stderr file pair per call).

mod docker;
mod eda;
mod host;
mod process;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;

pub use docker::DockerRunner;
pub use eda::{EdaCommand, EdaRunner, EdaSettings, EdaTool};
pub use host::HostRunner;
pub use process::mem_available_bytes;

use crate::cancel::CancelToken;
use crate::canon;
use crate::config::LoadedConfig;
use crate::error::{CoreError, IoContext, Result};

/// One external command. `tool` is the logical name used in logs and version records.
#[derive(Debug, Clone, Default)]
pub struct Invocation {
    pub tool: String,
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: Option<PathBuf>,
    /// Docker only: bind mounts and an image other than `[tools.docker]`'s.
    pub mounts: Vec<Mount>,
    pub image: Option<String>,
}

impl Invocation {
    pub fn new(tool: impl Into<String>, program: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            program: program.into(),
            ..Self::default()
        }
    }
    pub fn arg(mut self, a: impl Into<String>) -> Self {
        self.args.push(a.into());
        self
    }
    pub fn args<I: IntoIterator<Item = S>, S: Into<String>>(mut self, it: I) -> Self {
        self.args.extend(it.into_iter().map(Into::into));
        self
    }
    pub fn env(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.env.push((k.into(), v.into()));
        self
    }
    pub fn cwd(mut self, p: impl Into<PathBuf>) -> Self {
        self.cwd = Some(p.into());
        self
    }
    pub fn mount(mut self, m: Mount) -> Self {
        self.mounts.push(m);
        self
    }
    pub fn image(mut self, image: impl Into<String>) -> Self {
        self.image = Some(image.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mount {
    pub src: PathBuf,
    pub dst: String,
    pub readonly: bool,
}

/// Per-call limits.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Caps {
    pub memory_gb: f64,
    pub cpus: f64,
    pub timeout_s: f64,
    pub pids_limit: i64,
}

impl Caps {
    pub fn memory_bytes(&self) -> u64 {
        (self.memory_gb * (1u64 << 30) as f64) as u64
    }
    pub fn timeout(&self) -> Duration {
        Duration::from_secs_f64(self.timeout_s.max(0.0))
    }
    /// Elementwise minimum: a pass may lower its caps, never raise them above its declaration.
    pub fn min(&self, other: &Caps) -> Caps {
        Caps {
            memory_gb: self.memory_gb.min(other.memory_gb),
            cpus: self.cpus.min(other.cpus),
            timeout_s: self.timeout_s.min(other.timeout_s),
            pids_limit: self.pids_limit.min(other.pids_limit),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KillReason {
    Timeout,
    /// Host watchdog: RSS of the process group exceeded the cap.
    MemoryCap,
    /// Docker: the kernel OOM killer fired inside the container's memory cgroup.
    OomKilled,
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolOutcome {
    pub runner: String,
    pub tool: String,
    /// The command line actually executed (for Docker: the full `docker run …`).
    pub argv: Vec<String>,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub killed: Option<KillReason>,
    pub duration_ms: u64,
    pub peak_rss_bytes: Option<u64>,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
}

impl ToolOutcome {
    pub fn success(&self) -> bool {
        self.killed.is_none() && self.exit_code == Some(0)
    }

    pub fn stdout_string(&self) -> Result<String> {
        std::fs::read_to_string(&self.stdout).at(&self.stdout)
    }

    /// `Err` unless the tool exited 0 without being killed.
    pub fn check(self) -> Result<Self> {
        if self.success() {
            return Ok(self);
        }
        if self.killed == Some(KillReason::Cancelled) {
            return Err(CoreError::Cancelled);
        }
        let why = match self.killed {
            Some(k) => format!("killed ({k:?})"),
            None => match (self.exit_code, self.signal) {
                (Some(c), _) => format!("exit code {c}"),
                (None, Some(s)) => format!("signal {s}"),
                _ => "unknown exit".to_string(),
            },
        };
        Err(CoreError::Tool {
            tool: self.tool.clone(),
            message: format!("{why}; see {}", self.stderr.display()),
        })
    }
}

/// The adapter every external tool goes through.
pub trait ToolRunner: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;
    fn run(&self, inv: &Invocation, caps: &Caps, log: &ToolLog, cancel: &CancelToken) -> Result<ToolOutcome>;
}

/// The per-pass call log: `logs/tools.jsonl` and `logs/<n>-<tool>.{out,err}`.
#[derive(Debug, Clone)]
pub struct ToolLog {
    pass: String,
    dir: PathBuf,
    counter: Arc<AtomicU64>,
    lock: Arc<Mutex<()>>,
}

#[derive(Debug, Clone)]
pub struct CallFiles {
    pub n: u64,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
}

#[derive(Debug, Serialize)]
struct CallRecord<'a> {
    n: u64,
    started_at: &'a str,
    caps: &'a Caps,
    #[serde(flatten)]
    outcome: &'a ToolOutcome,
}

impl ToolLog {
    pub fn new(pass: impl Into<String>, dir: impl Into<PathBuf>) -> Self {
        Self {
            pass: pass.into(),
            dir: dir.into(),
            counter: Arc::new(AtomicU64::new(0)),
            lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn pass(&self) -> &str {
        &self.pass
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn next(&self, tool: &str) -> Result<CallFiles> {
        std::fs::create_dir_all(&self.dir).at(&self.dir)?;
        let n = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        let stem = format!("{n:04}-{}", sanitize(tool));
        Ok(CallFiles {
            n,
            stdout: self.dir.join(format!("{stem}.out")),
            stderr: self.dir.join(format!("{stem}.err")),
        })
    }

    fn append(&self, n: u64, started_at: &str, caps: &Caps, outcome: &ToolOutcome) -> Result<()> {
        let rec = CallRecord {
            n,
            started_at,
            caps,
            outcome,
        };
        let mut line = canon::to_canonical_compact(&rec).map_err(|source| CoreError::Json {
            path: self.dir.join("tools.jsonl"),
            source,
        })?;
        line.push(b'\n');
        let path = self.dir.join("tools.jsonl");
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .at(&path)?;
        f.write_all(&line).at(&path)
    }
}

/// Keep `[A-Za-z0-9_.-]`, for file and container names.
pub fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "_.-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// A runner bound to one pass: its caps, log directory and cancel token.
#[derive(Debug, Clone)]
pub struct BoundRunner {
    runner: Arc<dyn ToolRunner>,
    caps: Caps,
    log: ToolLog,
    cancel: CancelToken,
}

impl BoundRunner {
    pub fn new(runner: Arc<dyn ToolRunner>, caps: Caps, log: ToolLog, cancel: CancelToken) -> Self {
        Self {
            runner,
            caps,
            log,
            cancel,
        }
    }

    pub fn caps(&self) -> &Caps {
        &self.caps
    }

    /// Lower the caps for subsequent calls (clamped to the pass's declaration).
    pub fn with_caps(mut self, caps: Caps) -> Self {
        self.caps = self.caps.min(&caps);
        self
    }

    pub fn run(&self, inv: &Invocation) -> Result<ToolOutcome> {
        self.runner.run(inv, &self.caps, &self.log, &self.cancel)
    }
}

/// The three runner implementations, shared by all passes of a run. Tests substitute their own.
#[derive(Debug, Clone)]
pub struct Runners {
    pub host: Arc<dyn ToolRunner>,
    pub docker: Arc<dyn ToolRunner>,
    pub eda: Arc<dyn ToolRunner>,
}

impl Runners {
    pub fn from_config(cfg: &LoadedConfig) -> Self {
        let tools = &cfg.config.tools;
        Self {
            host: Arc::new(HostRunner::new(Duration::from_secs_f64(
                tools.host.watchdog_interval_s,
            ))),
            docker: Arc::new(DockerRunner::from_config(tools.docker.clone())),
            eda: Arc::new(EdaRunner::new(EdaSettings::from_config(cfg))),
        }
    }
}

/// Shared bookkeeping of one call: log files, timing, record.
pub(crate) struct CallLog<'a> {
    pub log: &'a ToolLog,
    pub files: CallFiles,
    pub started_at: String,
    pub t0: std::time::Instant,
}

impl<'a> CallLog<'a> {
    pub fn start(log: &'a ToolLog, tool: &str) -> Result<Self> {
        Ok(Self {
            log,
            files: log.next(tool)?,
            started_at: canon::now_utc(),
            t0: std::time::Instant::now(),
        })
    }

    pub fn finish(&self, caps: &Caps, outcome: &ToolOutcome) -> Result<()> {
        self.log.append(self.files.n, &self.started_at, caps, outcome)
    }
}
