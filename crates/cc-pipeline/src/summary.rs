//! `build/.pipeline/run-<timestamp>.json`: what one `chipc run` did (spec §3.3, observability).

use std::path::{Path, PathBuf};

use cc_core::PassResources;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const RUN_SUMMARY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Ran and verified.
    Ok,
    /// Skipped: an `ok` manifest with the same cache key and intact outputs exists.
    CacheHit,
    Failed,
    /// Cancelled by `stop_on_failure` after another pass failed.
    Cancelled,
    /// Not started: a dependency did not succeed, or the run was stopping.
    Skipped,
}

impl RunStatus {
    pub fn is_success(self) -> bool {
        matches!(self, RunStatus::Ok | RunStatus::CacheHit)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PassRun {
    pub pass: String,
    pub status: RunStatus,
    pub cache_key: Option<String>,
    /// Milliseconds since the run started (monotonic clock), for overlap analysis.
    pub start_offset_ms: Option<u64>,
    pub end_offset_ms: Option<u64>,
    pub duration_ms: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub resources: Option<PassResources>,
    pub message: Option<String>,
}

impl PassRun {
    pub fn new(pass: &str, status: RunStatus) -> Self {
        Self {
            pass: pass.into(),
            status,
            cache_key: None,
            start_offset_ms: None,
            end_offset_ms: None,
            duration_ms: None,
            peak_rss_bytes: None,
            resources: None,
            message: None,
        }
    }

    pub fn with_message(mut self, m: impl Into<String>) -> Self {
        self.message = Some(m.into());
        self
    }

    /// Whether this pass's execution interval intersects `other`'s.
    pub fn overlaps(&self, other: &PassRun) -> bool {
        match (
            self.start_offset_ms,
            self.end_offset_ms,
            other.start_offset_ms,
            other.end_offset_ms,
        ) {
            (Some(a0), Some(a1), Some(b0), Some(b1)) => a0 < b1 && b0 < a1,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Totals {
    pub ran: usize,
    pub cache_hits: usize,
    pub failed: usize,
    pub cancelled: usize,
    pub skipped: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunSummary {
    pub schema_version: u32,
    pub started_at: String,
    pub finished_at: String,
    pub duration_ms: u64,
    pub targets: Vec<String>,
    pub from: Option<String>,
    pub only: Option<String>,
    pub models: Vec<String>,
    /// In the order the passes finished (or were decided).
    pub passes: Vec<PassRun>,
    pub totals: Totals,
    pub ok: bool,
    #[serde(skip)]
    #[schemars(skip)]
    pub path: Option<PathBuf>,
}

impl RunSummary {
    pub fn pass(&self, name: &str) -> Option<&PassRun> {
        self.passes.iter().find(|p| p.pass == name)
    }

    pub fn status_of(&self, name: &str) -> Option<RunStatus> {
        self.pass(name).map(|p| p.status)
    }

    pub(crate) fn totals_of(passes: &[PassRun]) -> Totals {
        let mut t = Totals::default();
        for p in passes {
            match p.status {
                RunStatus::Ok => t.ran += 1,
                RunStatus::CacheHit => t.cache_hits += 1,
                RunStatus::Failed => {
                    t.ran += 1;
                    t.failed += 1
                }
                RunStatus::Cancelled => t.cancelled += 1,
                RunStatus::Skipped => t.skipped += 1,
            }
        }
        t
    }

    /// Write as `run-<UTC timestamp>.json` in `dir` (suffixed if a file of that name exists).
    pub(crate) fn write(&mut self, dir: &Path) -> cc_core::Result<PathBuf> {
        let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ").to_string();
        let mut path = dir.join(format!("run-{stamp}.json"));
        let mut n = 1;
        while path.exists() {
            path = dir.join(format!("run-{stamp}-{n}.json"));
            n += 1;
        }
        cc_core::canon::write_json_file(&path, self)?;
        self.path = Some(path.clone());
        Ok(path)
    }
}
