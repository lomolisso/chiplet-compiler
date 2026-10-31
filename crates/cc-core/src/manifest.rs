//! `build/.<pass>/manifest.json` (spec §3.4): cache key, inputs, outputs, config, tools, status, timings.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::config::PassResources;

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PassStatus {
    /// Written when the pass starts; seen afterwards only if the driver died mid-pass.
    Running,
    Ok,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InputRecord {
    pub kind: String,
    pub producer: String,
    pub hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FileRecord {
    /// Relative to `out/<kind>/`, `/`-separated.
    pub path: String,
    pub hash: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputRecord {
    pub kind: String,
    /// blake3 over the sorted `(path, file hash)` pairs of `files`.
    pub hash: String,
    pub files: Vec<FileRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub pass: String,
    pub pass_version: u32,
    /// blake3 of (inputs, config section, models, tool versions, pass version); see `cache::cache_key`.
    pub cache_key: String,
    pub status: PassStatus,
    pub inputs: Vec<InputRecord>,
    pub outputs: Vec<OutputRecord>,
    /// The pass's config section without `resources` (canonical JSON).
    pub config: serde_json::Value,
    /// Declared resources. Recorded, but not part of the cache key: they do not change outputs.
    pub resources: PassResources,
    pub tool_versions: BTreeMap<String, String>,
    pub models: Vec<String>,
    /// RFC 3339, UTC.
    pub started_at: String,
    pub finished_at: Option<String>,
    pub duration_ms: Option<u64>,
    /// Peak RSS of the driver process during the pass window (an upper bound when passes overlap).
    pub peak_rss_bytes: Option<u64>,
    pub error: Option<String>,
}

impl Manifest {
    pub fn output(&self, kind: &str) -> Option<&OutputRecord> {
        self.outputs.iter().find(|o| o.kind == kind)
    }
}
