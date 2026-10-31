//! What a running pass sees: its own directory and section, its declared inputs only, pre-capped tool
//! runners, the model set and the cancel token.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cc_core::config::{ModelConfigKind, PathsConfig, TestingConfig};
use cc_core::runner::{BoundRunner, Caps, Runners, ToolLog};
use cc_core::{
    ArtifactKind, CancelToken, CoreError, KindDir, LoadedConfig, PassDir, PassResources, PassSection,
};
use serde::de::DeserializeOwned;

use crate::error::PassError;

#[derive(Debug)]
pub struct PassContext {
    pub(crate) pass: &'static str,
    pub(crate) dir: PassDir,
    pub(crate) section: PassSection,
    pub(crate) resources: PassResources,
    pub(crate) config: Arc<LoadedConfig>,
    pub(crate) models: Vec<String>,
    pub(crate) inputs: BTreeMap<&'static str, KindDir>,
    pub(crate) outputs: BTreeMap<&'static str, KindDir>,
    pub(crate) runners: Runners,
    pub(crate) log: ToolLog,
    pub(crate) cancel: CancelToken,
}

impl PassContext {
    pub fn name(&self) -> &'static str {
        self.pass
    }

    pub fn dir(&self) -> &PassDir {
        &self.dir
    }

    pub fn section(&self) -> &PassSection {
        &self.section
    }

    /// Parse `[passes.<name>]` (without `resources`) into the pass's strict config type.
    pub fn options<T: DeserializeOwned>(&self) -> Result<T, PassError> {
        Ok(self.section.parse()?)
    }

    pub fn resources(&self) -> &PassResources {
        &self.resources
    }

    /// The model set (empty for passes that are not model-dependent).
    pub fn models(&self) -> &[String] {
        &self.models
    }

    pub fn testing(&self) -> &TestingConfig {
        &self.config.config.testing
    }

    /// `models.config`: which entry of the ports' `CONFIGS` the run uses (part of the cache key of
    /// model-dependent passes).
    pub fn model_config(&self) -> ModelConfigKind {
        self.config.config.models.config
    }

    /// The repository root (where `config.toml` lives).
    pub fn repo_root(&self) -> &Path {
        &self.config.root
    }

    pub fn paths(&self) -> &PathsConfig {
        &self.config.config.paths
    }

    /// Resolve a configured path against the repository root.
    pub fn resolve(&self, p: &Path) -> PathBuf {
        self.config.resolve(p)
    }

    /// A declared input. Reading anything else is an error (the pass rule, spec §3.1).
    pub fn input(&self, kind: ArtifactKind) -> Result<&KindDir, PassError> {
        self.inputs.get(kind.name()).ok_or_else(|| {
            PassError::Core(CoreError::Artifact {
                kind: kind.name().into(),
                message: format!("pass `{}` did not declare it in requires()", self.pass),
            })
        })
    }

    /// A declared output directory (created before `run`).
    pub fn output(&self, kind: ArtifactKind) -> Result<&KindDir, PassError> {
        self.outputs.get(kind.name()).ok_or_else(|| {
            PassError::Core(CoreError::Artifact {
                kind: kind.name().into(),
                message: format!("pass `{}` did not declare it in provides()", self.pass),
            })
        })
    }

    fn timeout_s(&self) -> f64 {
        self.resources.timeout_min * 60.0
    }

    /// Host processes: capped at the pass's memory and threads.
    pub fn host(&self) -> BoundRunner {
        let caps = Caps {
            memory_gb: self.resources.memory_gb,
            cpus: self.resources.threads as f64,
            timeout_s: self.timeout_s(),
            pids_limit: self.config.config.tools.docker.pids_limit,
        };
        BoundRunner::new(
            self.runners.host.clone(),
            caps,
            self.log.clone(),
            self.cancel.clone(),
        )
    }

    /// Containers: the `[tools.docker]` per-call caps, never above the pass's declaration.
    pub fn docker(&self) -> BoundRunner {
        let d = &self.config.config.tools.docker;
        let caps = Caps {
            memory_gb: d.memory_gb.min(self.resources.memory_gb),
            cpus: d.cpus.min(self.resources.threads as f64),
            timeout_s: self.timeout_s(),
            pids_limit: d.pids_limit,
        };
        BoundRunner::new(
            self.runners.docker.clone(),
            caps,
            self.log.clone(),
            self.cancel.clone(),
        )
    }

    /// Licensed host EDA tools through `eda-run`.
    pub fn eda(&self) -> BoundRunner {
        let caps = Caps {
            memory_gb: self.resources.memory_gb,
            cpus: self.resources.threads as f64,
            timeout_s: self.timeout_s(),
            pids_limit: self.config.config.tools.docker.pids_limit,
        };
        BoundRunner::new(
            self.runners.eda.clone(),
            caps,
            self.log.clone(),
            self.cancel.clone(),
        )
    }

    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancel
    }

    /// Call between work units: `Err` once the scheduler cancelled this pass.
    pub fn check_cancelled(&self) -> Result<(), PassError> {
        Ok(self.cancel.check()?)
    }
}
