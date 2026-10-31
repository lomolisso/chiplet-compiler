//! The pass contract (spec §3.1). A pass reads only its declared inputs, writes only into its own
//! `build/.<name>/`, and is a pure function of (inputs, config section, models, tool versions, version).

use std::collections::BTreeMap;

use cc_core::{ArtifactKind, CoreError, LoadedConfig, PassResources, PassSection, VerifyReport};

use crate::context::PassContext;
use crate::error::PassError;

/// Which runner a tool is called through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RunnerKind {
    Host,
    Docker,
    Eda,
}

/// An external tool a pass calls. Its version (first line of `program version_args…`) enters the
/// cache key, so upgrading a tool reruns the passes that use it.
#[derive(Debug, Clone, Copy)]
pub struct ToolSpec {
    pub name: &'static str,
    pub runner: RunnerKind,
    pub program: &'static str,
    pub version_args: &'static [&'static str],
}

pub trait Pass: Send + Sync {
    /// Equals the config section `[passes.<name>]` and the build directory `build/.<name>/`.
    fn name(&self) -> &'static str;

    /// Bump when the meaning of the outputs changes.
    fn version(&self) -> u32;

    fn requires(&self) -> &[ArtifactKind];

    fn provides(&self) -> &[ArtifactKind];

    /// Resources admitted by the scheduler; by default exactly `[passes.<name>.resources]`.
    fn resources(&self, declared: &PassResources) -> PassResources {
        *declared
    }

    fn tools(&self) -> &[ToolSpec] {
        &[]
    }

    /// Content hashes of inputs that are not artifacts, e.g. the source trees a pass reads (`models/`, a
    /// Python package, a lock file, a container image tag), as `name → hash`. They enter the cache
    /// key next to the tool versions (under `source:<name>`) and are recorded in the manifest, so editing
    /// such a source reruns the pass.
    fn fingerprints(
        &self,
        _config: &LoadedConfig,
        _section: &PassSection,
    ) -> Result<BTreeMap<String, String>, CoreError> {
        Ok(BTreeMap::new())
    }

    /// Whether outputs depend on the model set (`--models`). Passes that do not (e.g. a primitive
    /// library) return false so changing the model set keeps their cache.
    fn model_dependent(&self) -> bool {
        true
    }

    /// Parse the section strictly; called for every registered pass before anything runs, so a typo in
    /// any pass section fails the whole run up front.
    fn check_config(&self, section: &PassSection) -> Result<(), CoreError>;

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError>;

    /// Independent checks on the written artifacts (spec §5). Must not call the algorithm it checks.
    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError>;
}
