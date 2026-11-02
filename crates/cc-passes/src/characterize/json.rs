//! The characterize artifacts (`out/library/`):
//!
//! - `library.json` (`Library`): one row per (primitive, kit, target) with area, delay and leakage;
//! - `catalog.json`: the primitive set with its RTL sources (`catalog::Primitive`);
//! - `coverage.json` (`Coverage`): each primitive measured, provisional or pending;
//! - `rtl/<id>.{x,v}`, `netlist/<kit>/<target>/<id>.v`, `reports/<kit>/<target>/<id>.<what>.rpt`;
//! - `summary.json` (`Summary`).

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::catalog::{Kind, Primitive};
use crate::eda::parse::{Lec, Measured};

pub const LIBRARY_SCHEMA: &str = "cc.characterize.library/1";
pub const COVERAGE_SCHEMA: &str = "cc.characterize.coverage/1";
pub const SUMMARY_SCHEMA: &str = "cc.characterize.summary/1";

/// A kit as the rows were measured on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KitInfo {
    pub name: String,
    pub corner: String,
    pub lib_name: String,
    pub liberty: String,
    /// blake3 of the Liberty file.
    pub liberty_hash: String,
    pub db: String,
    pub db_hash: String,
    pub time_unit_ps: f64,
    pub driving_cell: String,
    /// Output load, in the library's capacitance unit.
    pub load: f64,
}

/// One timing point of one kit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TargetInfo {
    pub kit: String,
    pub target: String,
    pub period_ps: f64,
}

/// One measured primitive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub key: String,
    pub id: String,
    pub kind: Kind,
    pub op: String,
    pub variant: Option<String>,
    pub family: String,
    pub types: Vec<String>,
    pub width: u32,
    pub kit: String,
    pub corner: String,
    pub target: String,
    pub period_ps: f64,
    /// Area µm² (1x Liberty scale), delay and slack ps, leakage nW.
    pub measured: Measured,
    /// `dc`, `genus` or `yosys`.
    pub estimator: String,
    /// `formality`, `conformal` or `yosys-equiv`, and its outcome.
    pub lec_tool: String,
    pub lec: Lec,
    /// Design Compiler mapped arithmetic to DesignWare parts.
    pub designware: bool,
    pub provisional: bool,
    /// Relative to `out/library/`.
    pub netlist: String,
    pub netlist_hash: String,
    /// Report files (relative to `out/library/`) the values were read from.
    pub reports: Vec<String>,
}

/// `library.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Library {
    pub schema: String,
    pub tool: String,
    pub lec_tool: String,
    /// Versions of the tools that produced the rows (from `chipc doctor`'s manifest and the image).
    pub tool_versions: BTreeMap<String, String>,
    pub kits: Vec<KitInfo>,
    pub targets: Vec<TargetInfo>,
    pub rows: Vec<Row>,
}

/// What a primitive has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Measured,
    /// Measured on stand-in RTL (XLS `apfloat`) until the soft-float library replaces it.
    Provisional,
    /// No RTL yet.
    Pending,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoverageEntry {
    pub key: String,
    pub status: Status,
    pub reason: Option<String>,
    pub rows: u64,
}

/// `coverage.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Coverage {
    pub schema: String,
    pub primitives: Vec<CoverageEntry>,
}

/// `summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub schema: String,
    pub tool: String,
    pub primitives: u64,
    pub by_kind: BTreeMap<String, u64>,
    pub measured: u64,
    pub provisional: u64,
    pub pending: u64,
    pub rows: u64,
    pub lec_equivalent: u64,
    pub designware_rows: u64,
    /// Tool sessions run now and taken from earlier runs (`eda/` job directories).
    pub jobs_run: u64,
    pub jobs_reused: u64,
    /// Seconds spent in the tool sessions run now.
    pub tool_seconds: f64,
}

pub fn coverage(cat: &[Primitive], rows: &[Row]) -> Coverage {
    let mut n: BTreeMap<&str, u64> = BTreeMap::new();
    for r in rows {
        *n.entry(r.key.as_str()).or_default() += 1;
    }
    Coverage {
        schema: COVERAGE_SCHEMA.into(),
        primitives: cat
            .iter()
            .map(|p| CoverageEntry {
                key: p.key.clone(),
                status: if p.pending.is_some() {
                    Status::Pending
                } else if p.provisional {
                    Status::Provisional
                } else {
                    Status::Measured
                },
                reason: p.pending.clone(),
                rows: n.get(p.key.as_str()).copied().unwrap_or(0),
            })
            .collect(),
    }
}

pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "characterize_library.schema.json",
            serde_json::to_value(schemars::schema_for!(Library)).unwrap_or_default(),
        ),
        (
            "characterize_coverage.schema.json",
            serde_json::to_value(schemars::schema_for!(Coverage)).unwrap_or_default(),
        ),
        (
            "characterize_catalog.schema.json",
            serde_json::to_value(schemars::schema_for!(Vec<Primitive>)).unwrap_or_default(),
        ),
        (
            "characterize_summary.schema.json",
            serde_json::to_value(schemars::schema_for!(Summary)).unwrap_or_default(),
        ),
    ]
}
