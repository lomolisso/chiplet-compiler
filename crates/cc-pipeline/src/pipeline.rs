//! The pipeline manager (spec §3.3): plan, cache check, Kahn scheduling with admission control, one
//! scoped OS thread and one rayon pool per running pass, failure propagation, cancellation, manifests
//! and the run summary.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use cc_core::canon::now_utc;
use cc_core::manifest::{InputRecord, MANIFEST_SCHEMA_VERSION};
use cc_core::runner::{Runners, ToolLog};
use cc_core::{
    ArtifactStore, CancelToken, Check, LoadedConfig, Manifest, PassResources, PassSection, PassStatus,
    VerifyReport,
};

use crate::admission::{Limits, Load, admissible, is_oversize};
use crate::context::PassContext;
use crate::dag::PassGraph;
use crate::error::{PassError, PipelineError};
use crate::pass::Pass;
use crate::summary::{PassRun, RUN_SUMMARY_SCHEMA_VERSION, RunStatus, RunSummary};
use crate::versions::ToolVersions;

/// What `chipc run` asked for.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Passes to reach; empty means `pipeline.targets`, and an empty config list means every pass.
    pub targets: Vec<String>,
    /// Force this pass and its dependents to rerun.
    pub from: Option<String>,
    /// Run only this pass; its inputs must already exist with status `ok`.
    pub only: Option<String>,
    /// Restrict the model set (`--models a,b`).
    pub models: Option<Vec<String>>,
}

/// One row of `chipc status`.
#[derive(Debug, Clone, PartialEq)]
pub struct StatusRow {
    pub pass: String,
    /// `ok`, `failed`, `cancelled`, `running` (driver died mid-pass) or `not run`.
    pub status: String,
    /// Whether a run now would be a cache hit (`None` when it cannot be decided, e.g. inputs missing).
    pub fresh: Option<bool>,
    pub finished_at: Option<String>,
    pub duration_ms: Option<u64>,
    pub cache_key: Option<String>,
    pub error: Option<String>,
}

pub struct Pipeline {
    passes: Vec<Box<dyn Pass>>,
    graph: PassGraph,
    config: Arc<LoadedConfig>,
    store: ArtifactStore,
    runners: Runners,
    versions: ToolVersions,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field(
                "passes",
                &self.passes.iter().map(|p| p.name()).collect::<Vec<_>>(),
            )
            .field("build_dir", &self.store.root())
            .finish()
    }
}

/// What one run considers: the targets' closure, the passes `--from`/`--only` force, the target names.
#[derive(Debug)]
struct Plan {
    passes: BTreeSet<usize>,
    forced: BTreeSet<usize>,
    targets: Vec<String>,
}

/// Everything decided before a pass starts.
#[derive(Debug)]
struct Prepared {
    section: PassSection,
    resources: PassResources,
    inputs: Vec<InputRecord>,
    tools: BTreeMap<String, String>,
    models: Vec<String>,
    key: String,
}

/// What a pass thread reports back.
#[derive(Debug)]
struct Finished {
    status: PassStatus,
    error: Option<String>,
    start_ms: u64,
    end_ms: u64,
    peak_rss: Option<u64>,
}

#[derive(Debug)]
struct Running {
    token: CancelToken,
    deadline: Instant,
    resources: PassResources,
    key: String,
    timed_out: bool,
}

/// Peak RSS of the driver process per running-pass window (process-wide: an upper bound when passes
/// overlap; tool children report their own peaks in `tools.jsonl`).
#[derive(Debug, Default)]
struct RssSampler {
    windows: Mutex<BTreeMap<usize, u64>>,
    stop: AtomicBool,
}

impl RssSampler {
    fn current() -> Option<u64> {
        memory_stats::memory_stats().map(|m| m.physical_mem as u64)
    }

    fn sample(&self) {
        if let Some(rss) = Self::current() {
            for v in self
                .windows
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .values_mut()
            {
                *v = (*v).max(rss);
            }
        }
    }

    fn run(&self) {
        while !self.stop.load(Ordering::SeqCst) {
            self.sample();
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    fn begin(&self, i: usize) {
        let rss = Self::current().unwrap_or(0);
        self.windows
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(i, rss);
    }

    fn end(&self, i: usize) -> Option<u64> {
        self.sample();
        self.windows.lock().unwrap_or_else(|p| p.into_inner()).remove(&i)
    }
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".into()
    }
}

impl Pipeline {
    /// Validate the registry (DAG), `pipeline.targets`, and every registered pass's config section.
    /// Sections without a registered pass only warn: planned passes' sections exist before their passes.
    pub fn new(passes: Vec<Box<dyn Pass>>, config: LoadedConfig) -> Result<Self, PipelineError> {
        let graph = PassGraph::build(&passes)?;
        for t in &config.config.pipeline.targets {
            graph.resolve(t, "pipeline.targets")?;
        }
        for p in &passes {
            let section = config.pass_section(p.name())?;
            p.check_config(&section)?;
        }
        let unregistered: Vec<&str> = config
            .config
            .passes
            .keys()
            .map(String::as_str)
            .filter(|n| graph.index_of(n).is_none())
            .collect();
        if !unregistered.is_empty() {
            tracing::debug!(sections = %unregistered.join(", "), "config sections without a registered pass (ignored)");
        }
        let runners = Runners::from_config(&config);
        let store = ArtifactStore::new(config.build_dir());
        Ok(Self {
            passes,
            graph,
            config: Arc::new(config),
            store,
            runners,
            versions: ToolVersions::default(),
        })
    }

    /// Substitute the tool runners (tests).
    pub fn with_runners(mut self, runners: Runners) -> Self {
        self.runners = runners;
        self
    }

    pub fn graph(&self) -> &PassGraph {
        &self.graph
    }

    pub fn store(&self) -> &ArtifactStore {
        &self.store
    }

    pub fn config(&self) -> &LoadedConfig {
        &self.config
    }

    /// `[passes.<name>]` sections with no registered pass.
    pub fn unregistered_sections(&self) -> Vec<String> {
        self.config
            .config
            .passes
            .keys()
            .filter(|n| self.graph.index_of(n).is_none())
            .cloned()
            .collect()
    }

    fn limits(&self) -> Limits {
        let c = &self.config.config;
        Limits {
            max_parallel: c.pipeline.max_parallel_passes,
            memory_budget_gb: c.resources.memory_budget_gb,
            threads_total: c.resources.threads_total,
        }
    }

    fn plan(&self, opts: &RunOptions) -> Result<Plan, PipelineError> {
        if let Some(only) = &opts.only {
            let i = self.graph.resolve(only, "--only")?;
            return Ok(Plan {
                passes: BTreeSet::from([i]),
                forced: BTreeSet::from([i]),
                targets: vec![only.clone()],
            });
        }
        let mut targets = if opts.targets.is_empty() {
            self.config.config.pipeline.targets.clone()
        } else {
            opts.targets.clone()
        };
        if targets.is_empty() {
            targets = (0..self.graph.len())
                .map(|i| self.graph.name(i).to_string())
                .collect();
        }
        let mut roots = Vec::new();
        for t in &targets {
            roots.push(self.graph.resolve(t, "targets")?);
        }
        let from = opts
            .from
            .as_ref()
            .map(|f| self.graph.resolve(f, "--from"))
            .transpose()?;
        roots.extend(from);
        let plan = self.graph.with_ancestors(roots);
        let forced = match from {
            Some(f) => self
                .graph
                .with_descendants([f])
                .intersection(&plan)
                .copied()
                .collect(),
            None => BTreeSet::new(),
        };
        Ok(Plan {
            passes: plan,
            forced,
            targets,
        })
    }

    fn prepare(&self, i: usize, models: &[String]) -> Result<Prepared, String> {
        let pass = &self.passes[i];
        let name = pass.name();
        let section = self.config.pass_section(name).map_err(|e| e.to_string())?;
        pass.check_config(&section).map_err(|e| e.to_string())?;
        let resources = pass.resources(&section.resources);
        let mut kinds: Vec<_> = pass.requires().to_vec();
        kinds.sort();
        let mut inputs = Vec::new();
        for k in kinds {
            let prod = self
                .graph
                .producer(k.name())
                .ok_or_else(|| format!("no producer of `{k}`"))?;
            let pname = self.graph.name(prod);
            let m = self
                .store
                .pass_dir(pname)
                .read_manifest()
                .map_err(|e| e.to_string())?;
            let hash = match &m {
                Some(m) if m.status == PassStatus::Ok => m.output(k.name()).map(|o| o.hash.clone()),
                _ => None,
            }
            .ok_or_else(|| format!("input `{k}` from `{pname}` is not available (no ok manifest)"))?;
            inputs.push(InputRecord {
                kind: k.name().into(),
                producer: pname.into(),
                hash,
            });
        }
        let log = ToolLog::new("pipeline", self.store.pipeline_dir().join("logs"));
        let tools = self
            .versions
            .resolve(pass.tools(), &self.runners, &log)
            .map_err(|e| e.to_string())?;
        let models = if pass.model_dependent() {
            models.to_vec()
        } else {
            Vec::new()
        };
        let config_json = section.options_json();
        let key = cc_core::cache::cache_key(&cc_core::cache::CacheKeyInput {
            pass: name,
            pass_version: pass.version(),
            inputs: inputs.iter().map(|r| (r.kind.clone(), r.hash.clone())).collect(),
            config: &config_json,
            models: &models,
            tools: &tools,
        });
        Ok(Prepared {
            section,
            resources,
            inputs,
            tools,
            models,
            key,
        })
    }

    /// The outputs on disk still hash to what the manifest recorded.
    fn outputs_match(&self, i: usize, m: &Manifest) -> bool {
        let dir = self.store.pass_dir(self.graph.name(i));
        self.passes[i].provides().iter().all(|k| {
            let recorded = m.output(k.name()).map(|o| o.hash.as_str());
            let actual = dir.kind(*k).record().ok().map(|r| r.hash);
            recorded.is_some() && recorded == actual.as_deref()
        })
    }

    fn is_hit(&self, i: usize, p: &Prepared) -> bool {
        match self.store.pass_dir(self.graph.name(i)).read_manifest() {
            Ok(Some(m)) => m.status == PassStatus::Ok && m.cache_key == p.key && self.outputs_match(i, &m),
            _ => false,
        }
    }

    fn context(&self, i: usize, p: &Prepared, cancel: CancelToken) -> PassContext {
        let pass = &self.passes[i];
        let dir = self.store.pass_dir(pass.name());
        let inputs = pass
            .requires()
            .iter()
            .filter_map(|k| {
                let prod = self.graph.producer(k.name())?;
                Some((k.name(), self.store.pass_dir(self.graph.name(prod)).kind(*k)))
            })
            .collect();
        let outputs = pass.provides().iter().map(|k| (k.name(), dir.kind(*k))).collect();
        PassContext {
            pass: pass.name(),
            log: ToolLog::new(pass.name(), dir.logs_dir()),
            dir,
            section: p.section.clone(),
            resources: p.resources,
            config: self.config.clone(),
            models: p.models.clone(),
            inputs,
            outputs,
            runners: self.runners.clone(),
            cancel,
        }
    }

    fn base_manifest(&self, i: usize, p: &Prepared, started_at: String) -> Manifest {
        let pass = &self.passes[i];
        Manifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            pass: pass.name().into(),
            pass_version: pass.version(),
            cache_key: p.key.clone(),
            status: PassStatus::Running,
            inputs: p.inputs.clone(),
            outputs: Vec::new(),
            config: p.section.options_json(),
            resources: p.resources,
            tool_versions: p.tools.clone(),
            models: p.models.clone(),
            started_at,
            finished_at: None,
            duration_ms: None,
            peak_rss_bytes: None,
            error: None,
        }
    }

    /// `run` then `verify` on a dedicated rayon pool, panics caught.
    fn run_and_verify(&self, i: usize, ctx: &mut PassContext) -> Result<VerifyReport, PassError> {
        let pass = &self.passes[i];
        let name = pass.name();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(ctx.resources.threads)
            .thread_name(move |t| format!("{name}-{t}"))
            .build()
            .map_err(PassError::other)?;
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.install(|| -> Result<VerifyReport, PassError> {
                pass.run(ctx)?;
                ctx.check_cancelled()?;
                pass.verify(ctx)
            })
        }));
        r.unwrap_or_else(|p| Err(PassError::failed(format!("panic: {}", panic_message(p.as_ref())))))
    }

    fn execute(
        &self,
        i: usize,
        p: Prepared,
        token: CancelToken,
        t0: Instant,
        sampler: &RssSampler,
    ) -> Finished {
        let pass = &self.passes[i];
        let name = pass.name();
        let _span = tracing::info_span!("pass", pass = name).entered();
        let start_ms = t0.elapsed().as_millis() as u64;
        let started = Instant::now();
        let dir = self.store.pass_dir(name);
        let mut manifest = self.base_manifest(i, &p, now_utc());
        tracing::info!(
            threads = p.resources.threads,
            memory_gb = p.resources.memory_gb,
            "start"
        );
        sampler.begin(i);
        let result: Result<VerifyReport, PassError> = (|| {
            dir.reset()?;
            dir.write_manifest(&manifest)?;
            for k in pass.provides() {
                dir.kind(*k).create()?;
            }
            let mut ctx = self.context(i, &p, token.clone());
            self.run_and_verify(i, &mut ctx)
        })();
        let peak_rss = sampler.end(i);
        let (status, error) = match result {
            Ok(report) => match dir.write_verify(&report) {
                Err(e) => (PassStatus::Failed, Some(e.to_string())),
                Ok(()) if report.passed() => (PassStatus::Ok, None),
                Ok(()) => (
                    PassStatus::Failed,
                    Some(format!("verify failed: {}", report.failed_checks().join(", "))),
                ),
            },
            Err(e) if e.is_cancelled() => (PassStatus::Cancelled, Some("cancelled".into())),
            Err(e) => (PassStatus::Failed, Some(e.to_string())),
        };
        let (status, error) = if status == PassStatus::Ok {
            match pass
                .provides()
                .iter()
                .map(|k| dir.kind(*k).record())
                .collect::<cc_core::Result<Vec<_>>>()
            {
                Ok(outputs) => {
                    manifest.outputs = outputs;
                    (status, error)
                }
                Err(e) => (PassStatus::Failed, Some(e.to_string())),
            }
        } else {
            (status, error)
        };
        manifest.status = status;
        manifest.error = error.clone();
        manifest.finished_at = Some(now_utc());
        manifest.duration_ms = Some(started.elapsed().as_millis() as u64);
        manifest.peak_rss_bytes = peak_rss;
        let (status, error) = match dir.write_manifest(&manifest) {
            Ok(()) => (status, error),
            Err(e) => (PassStatus::Failed, Some(format!("writing manifest: {e}"))),
        };
        match status {
            PassStatus::Ok => tracing::info!(ms = manifest.duration_ms, "ok"),
            _ => tracing::warn!(status = ?status, error = error.as_deref().unwrap_or(""), "not ok"),
        }
        Finished {
            status,
            error,
            start_ms,
            end_ms: t0.elapsed().as_millis() as u64,
            peak_rss,
        }
    }

    /// Execute the plan. Returns the run summary (also written to `build/.pipeline/`); pass failures are
    /// reported in it, not as `Err`.
    pub fn run(&self, opts: &RunOptions) -> Result<RunSummary, PipelineError> {
        let started_at = now_utc();
        let t0 = Instant::now();
        let models = self.config.resolve_models(opts.models.as_deref())?;
        let Plan {
            passes: plan,
            forced,
            targets,
        } = self.plan(opts)?;
        let limits = self.limits();
        let stop_on_failure = self.config.config.pipeline.stop_on_failure;
        std::fs::create_dir_all(self.store.pipeline_dir()).map_err(|source| cc_core::CoreError::Io {
            path: self.store.pipeline_dir(),
            source,
        })?;

        let mut pending: BTreeSet<usize> = plan.clone();
        let mut prepared: BTreeMap<usize, Prepared> = BTreeMap::new();
        let mut running: BTreeMap<usize, Running> = BTreeMap::new();
        let mut done: BTreeMap<usize, RunStatus> = BTreeMap::new();
        let mut order: Vec<PassRun> = Vec::new();
        let mut stopping = false;
        let sampler = RssSampler::default();
        let (tx, rx) = mpsc::channel::<(usize, Finished)>();

        std::thread::scope(|s| {
            s.spawn(|| sampler.run());
            loop {
                let mut progressed = true;
                while progressed {
                    progressed = false;
                    let mut ready: Vec<usize> = pending.iter().copied().collect();
                    ready.sort_by_key(|&i| self.graph.name(i));
                    for i in ready {
                        let name = self.graph.name(i);
                        let preds: Vec<usize> = self
                            .graph
                            .predecessors(i)
                            .into_iter()
                            .filter(|p| plan.contains(p))
                            .collect();
                        if let Some(&bad) = preds
                            .iter()
                            .find(|p| done.get(p).is_some_and(|st| !st.is_success()))
                        {
                            let why = format!("dependency `{}` did not succeed", self.graph.name(bad));
                            order.push(PassRun::new(name, RunStatus::Skipped).with_message(why));
                            done.insert(i, RunStatus::Skipped);
                            pending.remove(&i);
                            progressed = true;
                            continue;
                        }
                        if !preds
                            .iter()
                            .all(|p| done.get(p).is_some_and(|st| st.is_success()))
                        {
                            continue;
                        }
                        if stopping {
                            let why = "stop_on_failure: an earlier pass failed";
                            order.push(PassRun::new(name, RunStatus::Skipped).with_message(why));
                            done.insert(i, RunStatus::Skipped);
                            pending.remove(&i);
                            progressed = true;
                            continue;
                        }
                        if let std::collections::btree_map::Entry::Vacant(e) = prepared.entry(i) {
                            match self.prepare(i, &models) {
                                Ok(p) => {
                                    e.insert(p);
                                }
                                Err(msg) => {
                                    tracing::warn!(pass = name, error = %msg, "cannot start");
                                    order.push(PassRun::new(name, RunStatus::Failed).with_message(msg));
                                    done.insert(i, RunStatus::Failed);
                                    pending.remove(&i);
                                    progressed = true;
                                    if stop_on_failure {
                                        stopping = true;
                                        running.values().for_each(|r| r.token.cancel());
                                    }
                                    continue;
                                }
                            }
                        }
                        let Some(p) = prepared.get(&i) else { continue };
                        if !forced.contains(&i) && self.is_hit(i, p) {
                            tracing::info!(pass = name, "cache hit");
                            let mut r = PassRun::new(name, RunStatus::CacheHit);
                            r.cache_key = Some(p.key.clone());
                            order.push(r);
                            done.insert(i, RunStatus::CacheHit);
                            pending.remove(&i);
                            prepared.remove(&i);
                            progressed = true;
                            continue;
                        }
                        let mut load = Load::default();
                        running.values().for_each(|r| load.add(&r.resources));
                        if !admissible(&load, &limits, &p.resources) {
                            continue;
                        }
                        if is_oversize(&limits, &p.resources) {
                            tracing::warn!(
                                pass = name,
                                "declared resources exceed the budget; running alone"
                            );
                        }
                        let Some(p) = prepared.remove(&i) else { continue };
                        let token = CancelToken::new();
                        let timeout = Duration::from_secs_f64(p.resources.timeout_min * 60.0);
                        running.insert(
                            i,
                            Running {
                                token: token.clone(),
                                deadline: Instant::now() + timeout,
                                resources: p.resources,
                                key: p.key.clone(),
                                timed_out: false,
                            },
                        );
                        pending.remove(&i);
                        progressed = true;
                        let tx = tx.clone();
                        let sampler = &sampler;
                        s.spawn(move || {
                            let fin = self.execute(i, p, token, t0, sampler);
                            let _ = tx.send((i, fin));
                        });
                    }
                }
                if running.is_empty() {
                    break;
                }
                if let Ok((i, fin)) = rx.recv_timeout(Duration::from_millis(10)) {
                    let Some(r) = running.remove(&i) else { continue };
                    let name = self.graph.name(i);
                    let (status, message) = match fin.status {
                        PassStatus::Ok => (RunStatus::Ok, None),
                        PassStatus::Cancelled if r.timed_out => {
                            let msg = format!("timeout after {} min", r.resources.timeout_min);
                            self.mark_failed(name, &msg);
                            (RunStatus::Failed, Some(msg))
                        }
                        PassStatus::Cancelled => (RunStatus::Cancelled, fin.error),
                        PassStatus::Failed | PassStatus::Running => (RunStatus::Failed, fin.error),
                    };
                    if status == RunStatus::Failed && stop_on_failure && !stopping {
                        stopping = true;
                        running.values().for_each(|r| r.token.cancel());
                    }
                    order.push(PassRun {
                        pass: name.into(),
                        status,
                        cache_key: Some(r.key),
                        start_offset_ms: Some(fin.start_ms),
                        end_offset_ms: Some(fin.end_ms),
                        duration_ms: Some(fin.end_ms.saturating_sub(fin.start_ms)),
                        peak_rss_bytes: fin.peak_rss,
                        resources: Some(r.resources),
                        message,
                    });
                    done.insert(i, status);
                }
                let now = Instant::now();
                for r in running.values_mut() {
                    if !r.timed_out && now > r.deadline {
                        r.timed_out = true;
                        r.token.cancel();
                    }
                }
            }
            sampler.stop.store(true, Ordering::SeqCst);
        });

        for i in pending {
            order.push(PassRun::new(self.graph.name(i), RunStatus::Skipped).with_message("not schedulable"));
        }
        let totals = RunSummary::totals_of(&order);
        let mut summary = RunSummary {
            schema_version: RUN_SUMMARY_SCHEMA_VERSION,
            started_at,
            finished_at: now_utc(),
            duration_ms: t0.elapsed().as_millis() as u64,
            targets,
            from: opts.from.clone(),
            only: opts.only.clone(),
            models,
            ok: order.iter().all(|p| p.status.is_success()),
            passes: order,
            totals,
            path: None,
        };
        summary.write(&self.store.pipeline_dir())?;
        Ok(summary)
    }

    fn mark_failed(&self, pass: &str, msg: &str) {
        let dir = self.store.pass_dir(pass);
        if let Ok(Some(mut m)) = dir.read_manifest() {
            m.status = PassStatus::Failed;
            m.error = Some(msg.into());
            let _ = dir.write_manifest(&m);
        }
    }

    /// `chipc verify <pass>`: re-run only the checks on existing artifacts, plus the framework check that
    /// the outputs still match the manifest. A failure marks the manifest `failed`.
    pub fn verify_pass(&self, name: &str) -> Result<VerifyReport, PipelineError> {
        let i = self.graph.resolve(name, "verify")?;
        let dir = self.store.pass_dir(name);
        let mut m = dir.read_manifest()?.ok_or_else(|| PipelineError::Pass {
            pass: name.into(),
            message: "no manifest: run the pass first".into(),
        })?;
        let p = self
            .prepare(i, &m.models)
            .map_err(|message| PipelineError::Pass {
                pass: name.into(),
                message,
            })?;
        let ctx = self.context(i, &p, CancelToken::new());
        let pass = &self.passes[i];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(p.resources.threads)
            .build()
            .map_err(|e| PipelineError::Pass {
                pass: name.into(),
                message: e.to_string(),
            })?;
        let r = catch_unwind(AssertUnwindSafe(|| pool.install(|| pass.verify(&ctx))))
            .unwrap_or_else(|p| Err(PassError::failed(format!("panic: {}", panic_message(p.as_ref())))));
        let mut report = r.map_err(|e| PipelineError::Pass {
            pass: name.into(),
            message: e.to_string(),
        })?;
        let intact = self.outputs_match(i, &m);
        report.push(
            Check::new("framework.outputs_match_manifest", intact)
                .count("kinds", pass.provides().len() as u64),
        );
        dir.write_verify(&report)?;
        if !report.passed() && m.status == PassStatus::Ok {
            m.status = PassStatus::Failed;
            m.error = Some(format!(
                "chipc verify: failed checks: {}",
                report.failed_checks().join(", ")
            ));
            dir.write_manifest(&m)?;
        }
        Ok(report)
    }

    /// `chipc status`: every registered pass in topological order, from its manifest.
    pub fn status(&self) -> Vec<StatusRow> {
        let models = self.config.resolve_models(None).unwrap_or_default();
        self.graph
            .topo_order()
            .into_iter()
            .map(|i| {
                let name = self.graph.name(i);
                match self.store.pass_dir(name).read_manifest() {
                    Ok(Some(m)) => {
                        let fresh = self.prepare(i, &models).ok().map(|p| self.is_hit(i, &p));
                        StatusRow {
                            pass: name.into(),
                            status: serde_json::to_value(m.status)
                                .ok()
                                .and_then(|v| v.as_str().map(str::to_string))
                                .unwrap_or_default(),
                            fresh,
                            finished_at: m.finished_at,
                            duration_ms: m.duration_ms,
                            cache_key: Some(m.cache_key),
                            error: m.error,
                        }
                    }
                    Ok(None) => StatusRow {
                        pass: name.into(),
                        status: "not run".into(),
                        fresh: None,
                        finished_at: None,
                        duration_ms: None,
                        cache_key: None,
                        error: None,
                    },
                    Err(e) => StatusRow {
                        pass: name.into(),
                        status: "unreadable".into(),
                        fresh: None,
                        finished_at: None,
                        duration_ms: None,
                        cache_key: None,
                        error: Some(e.to_string()),
                    },
                }
            })
            .collect()
    }

    /// `chipc clean [pass]`: remove one pass directory, or every registered pass's plus `.pipeline/`.
    /// Never deletes anything else under the build dir.
    pub fn clean(&self, pass: Option<&str>) -> Result<Vec<PathBuf>, PipelineError> {
        let names: Vec<&str> = match pass {
            Some(p) => vec![self.graph.name(self.graph.resolve(p, "clean")?)],
            None => (0..self.graph.len()).map(|i| self.graph.name(i)).collect(),
        };
        let mut removed = Vec::new();
        for n in names {
            let d = self.store.pass_dir(n);
            if d.path().exists() {
                d.remove()?;
                removed.push(d.path().to_path_buf());
            }
        }
        if pass.is_none() {
            let p = self.store.pipeline_dir();
            if p.exists() {
                std::fs::remove_dir_all(&p).map_err(|source| cc_core::CoreError::Io {
                    path: p.clone(),
                    source,
                })?;
                removed.push(p);
            }
        }
        Ok(removed)
    }
}
