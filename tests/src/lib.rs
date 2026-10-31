//! Shared helpers of the smoke and deep suites: a temporary repo layout, fixture passes and a recording
//! tool runner.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cc_core::runner::{Caps, Invocation, Runners, ToolLog, ToolOutcome, ToolRunner};
use cc_core::{ArtifactKind, CancelToken, Check, CoreError, LoadedConfig, PassSection, VerifyReport};
use cc_pipeline::{Pass, PassContext, PassError, Pipeline};

/// The repository root (for the tracked `config.toml`, `models/`, `docs/schemas/`).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

/// A throw-away repository root with a `config.toml`.
#[derive(Debug)]
pub struct Env {
    pub dir: tempfile::TempDir,
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

impl Env {
    pub fn new() -> Self {
        Self {
            dir: tempfile::tempdir().unwrap(),
        }
    }

    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    pub fn config_path(&self) -> PathBuf {
        self.root().join("config.toml")
    }

    pub fn build_dir(&self) -> PathBuf {
        self.root().join("build")
    }

    /// Write `config.toml` and load it with `--set` overrides.
    pub fn config(&self, text: &str, sets: &[&str]) -> LoadedConfig {
        self.try_config(text, sets).unwrap()
    }

    pub fn try_config(&self, text: &str, sets: &[&str]) -> cc_core::Result<LoadedConfig> {
        std::fs::write(self.config_path(), text).unwrap();
        let sets: Vec<String> = sets.iter().map(|s| s.to_string()).collect();
        LoadedConfig::load(&self.config_path(), &sets)
    }

    /// The production registry (toy passes) on this environment.
    pub fn toy_pipeline(&self, text: &str, sets: &[&str]) -> Pipeline {
        Pipeline::new(cc_passes::registry(), self.config(text, sets)).unwrap()
    }

    pub fn pipeline(&self, passes: Vec<Box<dyn Pass>>, text: &str) -> Pipeline {
        Pipeline::new(passes, self.config(text, &[])).unwrap()
    }
}

/// A minimal config for the toy passes: small resources, empty model catalog.
pub const TOY_CONFIG: &str = r#"
[pipeline]
max_parallel_passes = 2
targets = ["hello_b", "hello_c"]
[resources]
memory_budget_gb = 8
threads_total = 8
[passes.hello_a.resources]
threads = 1
memory_gb = 1
[passes.hello_b.resources]
threads = 1
memory_gb = 1
[passes.hello_c.resources]
threads = 1
memory_gb = 1
"#;

/// What a fixture pass does when it runs.
#[derive(Debug, Clone, Default)]
pub struct Behaviour {
    pub sleep_ms: u64,
    pub fail: bool,
    pub panic: bool,
    /// Sleep without checking the cancel token.
    pub ignore_cancel: bool,
    pub fail_verify: bool,
    /// Call `ctx.docker()` once with this invocation (use with a `RecordingRunner`).
    pub docker_call: bool,
}

/// A configurable pass for scheduler tests. Resources come from `[passes.<name>.resources]`.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub name: &'static str,
    pub requires: Vec<ArtifactKind>,
    pub provides: Vec<ArtifactKind>,
    pub behaviour: Behaviour,
}

impl Fixture {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            requires: Vec::new(),
            provides: Vec::new(),
            behaviour: Behaviour::default(),
        }
    }
    pub fn provides(mut self, k: &'static str) -> Self {
        self.provides.push(ArtifactKind(k));
        self
    }
    pub fn requires(mut self, k: &'static str) -> Self {
        self.requires.push(ArtifactKind(k));
        self
    }
    pub fn sleep(mut self, ms: u64) -> Self {
        self.behaviour.sleep_ms = ms;
        self
    }
    pub fn fail(mut self) -> Self {
        self.behaviour.fail = true;
        self
    }
    pub fn panics(mut self) -> Self {
        self.behaviour.panic = true;
        self
    }
    pub fn ignore_cancel(mut self) -> Self {
        self.behaviour.ignore_cancel = true;
        self
    }
    pub fn fail_verify(mut self) -> Self {
        self.behaviour.fail_verify = true;
        self
    }
    pub fn docker_call(mut self) -> Self {
        self.behaviour.docker_call = true;
        self
    }
    pub fn boxed(self) -> Box<dyn Pass> {
        Box::new(self)
    }
}

impl Pass for Fixture {
    fn name(&self) -> &'static str {
        self.name
    }
    fn version(&self) -> u32 {
        1
    }
    fn requires(&self) -> &[ArtifactKind] {
        &self.requires
    }
    fn provides(&self) -> &[ArtifactKind] {
        &self.provides
    }
    fn model_dependent(&self) -> bool {
        false
    }
    fn check_config(&self, _section: &PassSection) -> Result<(), CoreError> {
        Ok(())
    }
    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let b = &self.behaviour;
        if b.ignore_cancel {
            std::thread::sleep(Duration::from_millis(b.sleep_ms));
        } else {
            ctx.cancel_token().sleep(Duration::from_millis(b.sleep_ms))?;
        }
        if b.panic {
            panic!("fixture {} panicked", self.name);
        }
        if b.fail {
            return Err(PassError::failed(format!("fixture {} failed", self.name)));
        }
        if b.docker_call {
            ctx.docker().run(&Invocation::new("probe", "true"))?.check()?;
        }
        for k in &self.requires {
            ctx.input(*k)?.read_json::<serde_json::Value>("value.json")?;
        }
        for k in &self.provides {
            ctx.output(*k)?.write_json(
                "value.json",
                &serde_json::json!({ "pass": self.name, "kind": k.name() }),
            )?;
        }
        Ok(())
    }
    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        let mut r = VerifyReport::new(ctx.name());
        r.push(Check::new("fixture", !self.behaviour.fail_verify));
        Ok(r)
    }
}

/// A `ToolRunner` that records what it was asked to run, with which caps, and succeeds.
#[derive(Debug, Default)]
pub struct RecordingRunner {
    pub calls: Mutex<Vec<(Invocation, Caps)>>,
}

impl ToolRunner for RecordingRunner {
    fn name(&self) -> &'static str {
        "recording"
    }
    fn run(
        &self,
        inv: &Invocation,
        caps: &Caps,
        log: &ToolLog,
        _cancel: &CancelToken,
    ) -> cc_core::Result<ToolOutcome> {
        self.calls.lock().unwrap().push((inv.clone(), *caps));
        let files = log.next(&inv.tool)?;
        std::fs::write(&files.stdout, "").unwrap();
        std::fs::write(&files.stderr, "").unwrap();
        Ok(ToolOutcome {
            runner: "recording".into(),
            tool: inv.tool.clone(),
            argv: vec![inv.program.clone()],
            exit_code: Some(0),
            signal: None,
            killed: None,
            duration_ms: 0,
            peak_rss_bytes: None,
            stdout: files.stdout,
            stderr: files.stderr,
        })
    }
}

/// Replays the another change exporter without Python: `python -m cc_export ... --model <m> --out <dir>` copies
/// `tests/fixtures/export/<m>/` to `<dir>`; `python -m cc_export.facts <dir>` prints `facts` (by default
/// `tests/fixtures/export/facts.json`). Records every invocation and its caps.
#[derive(Debug)]
pub struct ExportReplay {
    pub facts: Mutex<String>,
    pub calls: Mutex<Vec<(Invocation, Caps)>>,
}

impl Default for ExportReplay {
    fn default() -> Self {
        let facts = std::fs::read_to_string(repo_root().join("tests/fixtures/export/facts.json")).unwrap();
        Self {
            facts: Mutex::new(facts),
            calls: Mutex::new(Vec::new()),
        }
    }
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let p = e.unwrap().path();
        let to = dst.join(p.file_name().unwrap());
        if p.is_dir() {
            copy_dir(&p, &to);
        } else {
            std::fs::copy(&p, &to).unwrap();
        }
    }
}

impl ToolRunner for ExportReplay {
    fn name(&self) -> &'static str {
        "export-replay"
    }
    fn run(
        &self,
        inv: &Invocation,
        caps: &Caps,
        log: &ToolLog,
        _cancel: &CancelToken,
    ) -> cc_core::Result<ToolOutcome> {
        self.calls.lock().unwrap().push((inv.clone(), *caps));
        let files = log.next(&inv.tool)?;
        let arg = |flag: &str| {
            inv.args
                .iter()
                .position(|a| a == flag)
                .map(|i| inv.args[i + 1].clone())
        };
        let stdout = if inv.args.iter().any(|a| a == "cc_export.facts") {
            self.facts.lock().unwrap().clone()
        } else {
            let model = arg("--model").unwrap();
            let out = PathBuf::from(arg("--out").unwrap());
            copy_dir(&repo_root().join("tests/fixtures/export").join(&model), &out);
            String::new()
        };
        std::fs::write(&files.stdout, stdout).unwrap();
        std::fs::write(&files.stderr, "").unwrap();
        Ok(ToolOutcome {
            runner: "export-replay".into(),
            tool: inv.tool.clone(),
            argv: vec![inv.program.clone()],
            exit_code: Some(0),
            signal: None,
            killed: None,
            duration_ms: 0,
            peak_rss_bytes: None,
            stdout: files.stdout,
            stderr: files.stderr,
        })
    }
}

/// Runners whose Docker slot records instead of running.
pub fn recording_runners(cfg: &LoadedConfig) -> (Runners, Arc<RecordingRunner>) {
    let rec = Arc::new(RecordingRunner::default());
    let mut r = Runners::from_config(cfg);
    r.docker = rec.clone();
    (r, rec)
}

/// `[passes.<name>.resources]` lines.
pub fn resources(name: &str, memory_gb: f64, threads: usize) -> String {
    format!("[passes.{name}.resources]\nthreads = {threads}\nmemory_gb = {memory_gb}\ntimeout_min = 1\n")
}
