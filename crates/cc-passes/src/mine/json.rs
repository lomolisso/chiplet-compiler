//! The mine artifacts (`out/mine/`):
//!
//! - `labels.json` (`Labels`): the integer node and edge labels of the GraMi graphs;
//! - `lg/<family>.lg` and `lg/<family>.map.json` (`LgMap`): one GraMi graph per family (the disjoint union
//!   of its units' datapaths) and the map from its vertices back to (unit, datapath vertex);
//! - `grami/<family>.json` (`GramiFile`): every pattern GraMi reported at the final τ, canonical;
//! - `patterns/<family>/<p>.json` (`PatternFile`): a kept pattern, its convex occurrences, its MIS per
//!   unit and its score;
//! - `selection/<family>.json` (`Selection`): the top-k patterns;
//! - `sweep/<family>.json` (`Sweep`): the τ sweep and why it stopped;
//! - `overlap.json` (`Overlap`): kept patterns found in several families;
//! - `summary.json` (`Summary`).

use std::collections::BTreeMap;

use cc_ir::absgraph::EdgeLabel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::datapaths::json::Stage;

pub const LABELS_SCHEMA: &str = "cc.mine.labels/1";
pub const LGMAP_SCHEMA: &str = "cc.mine.lg_map/1";
pub const GRAMI_SCHEMA: &str = "cc.mine.grami/1";
pub const PATTERN_SCHEMA: &str = "cc.mine.pattern/1";
pub const SELECTION_SCHEMA: &str = "cc.mine.selection/1";
pub const SWEEP_SCHEMA: &str = "cc.mine.sweep/1";
pub const OVERLAP_SCHEMA: &str = "cc.mine.overlap/1";
pub const SUMMARY_SCHEMA: &str = "cc.mine.summary/1";

/// A mining family: datapath classes by skeleton class, and the per-element stages of template classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Map,
    Reduce,
    Contraction,
    Stage,
}

impl Family {
    pub const ALL: [Family; 4] = [Family::Map, Family::Reduce, Family::Contraction, Family::Stage];

    pub fn name(self) -> &'static str {
        match self {
            Family::Map => "map",
            Family::Reduce => "reduce",
            Family::Contraction => "contraction",
            Family::Stage => "stage",
        }
    }
}

/// `labels.json`: node label `i` is `nodes[i]` (`op@family`, with `[…]` holding the attributes that change
/// what the op computes); edge label `j` is `edges[j]` (the merged ends of the parallel dataflow edges
/// between two vertices). Both lists are sorted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Labels {
    pub schema: String,
    pub nodes: Vec<String>,
    pub edges: Vec<EdgeLabel>,
}

/// One mining unit: a datapath class, or one stage (`prologue {port}` or `epilogue`) of a template class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Unit {
    /// `<class>`, `<class>/prologue<port>` or `<class>/epilogue`.
    pub id: String,
    pub class: String,
    pub stage: Option<Stage>,
    /// How often the unit runs over the model set: the class's weighted workload, or for a stage the
    /// trip-count-weighted number of elements of the core operand (prologue) or result (epilogue).
    pub workload: u128,
    /// Its first vertex in the `.lg` graph.
    pub first: u32,
    /// Its datapath vertices, in `.lg` order (constants are not vertices).
    pub vertices: Vec<u32>,
}

/// `lg/<family>.map.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LgMap {
    pub schema: String,
    pub family: Family,
    pub units: Vec<Unit>,
    pub vertices: u64,
    pub edges: u64,
}

/// A pattern GraMi reported within `max_vertices`, after canonicalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Reported {
    pub id: String,
    pub vertices: u32,
}

/// `grami/<family>.json`: the final τ's GraMi result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GramiFile {
    pub schema: String,
    pub family: Family,
    pub tau: Option<u64>,
    /// GraMi's own pattern count (its `Output.txt`).
    pub reported: u64,
    /// Patterns over `max_vertices` (counted, not canonicalized).
    pub dropped: u64,
    /// The others, sorted by id.
    pub patterns: Vec<Reported>,
}

/// One pattern edge: `src -> dst` with the merged ends.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct PatternEdge {
    pub src: u32,
    pub dst: u32,
    pub ends: EdgeLabel,
}

/// What feeds an operand of an occurrence from outside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Feed {
    /// A folded constant (a configuration register).
    Const,
    /// A datapath input.
    Input,
    /// An accumulator's current value.
    Acc,
    /// A template core's result.
    Result,
    /// A vertex outside the occurrence.
    Vertex,
}

/// Operand `arg` of pattern vertex `at` comes from outside the occurrence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OccInput {
    pub at: u32,
    pub arg: u32,
    pub feed: Feed,
}

/// A convex occurrence: pattern vertex `i` is datapath vertex `map[i]` of `unit`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Occurrence {
    pub unit: String,
    pub map: Vec<u32>,
    pub inputs: Vec<OccInput>,
}

/// The maximum independent set of one unit's occurrences (vertex overlap).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct UnitMis {
    pub unit: String,
    pub workload: u128,
    /// The unit's occurrences (indices into `occurrences`).
    pub occurrences: Vec<u32>,
    pub algorithm: String,
    /// The independent set (indices into `occurrences`).
    pub chosen: Vec<u32>,
    pub size: u64,
    /// `size` is the maximum.
    pub optimal: bool,
    /// An upper bound on the maximum (`size` when optimal).
    pub bound: u64,
}

/// `patterns/<family>/<id>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PatternFile {
    pub schema: String,
    /// `p` + 16 hex digits of the canonical code's hash.
    pub id: String,
    pub family: Family,
    /// The canonical code (vertex labels in canonical order, then the sorted edges).
    pub code: String,
    /// Node label per pattern vertex, in canonical order.
    pub labels: Vec<String>,
    pub edges: Vec<PatternEdge>,
    /// The τ it was mined at.
    pub tau: u64,
    /// MNI support over the family (min over pattern vertices of the distinct images).
    pub mni: u64,
    /// Embeddings (vertex maps) over the family, automorphic ones included.
    pub embeddings: u64,
    /// Distinct images that are not convex in their datapath (not stored).
    pub non_convex: u64,
    pub occurrences: Vec<Occurrence>,
    pub mis: Vec<UnitMis>,
    /// Σ over units of workload × MIS size.
    pub score: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Selected {
    pub id: String,
    pub vertices: u32,
    pub score: u128,
    pub mni: u64,
    /// Σ of the MIS sizes over units.
    pub mis: u64,
}

/// `selection/<family>.json`: the top-k by score (ties: more vertices, then id).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub schema: String,
    pub family: Family,
    pub tau: Option<u64>,
    pub top_k: u64,
    pub patterns: Vec<Selected>,
}

/// Why the sweep stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Stop {
    /// The selection equals the previous τ's.
    Stable,
    /// τ reached `tau_min`.
    TauMin,
    /// A GraMi run hit the time budget: the last finished τ is kept (depends on the machine).
    Budget,
    /// The family's graph has no edge: nothing to mine.
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Step {
    pub tau: u64,
    /// GraMi finished at this τ.
    pub finished: bool,
    /// Patterns GraMi reported, and those within `max_vertices`.
    pub reported: u64,
    pub kept: u64,
    /// Hash of the ordered selection.
    pub selection: String,
}

/// `sweep/<family>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sweep {
    pub schema: String,
    pub family: Family,
    pub steps: Vec<Step>,
    pub tau: Option<u64>,
    pub stopped_by: Stop,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Shared {
    pub id: String,
    pub families: Vec<Family>,
    /// Selected in these families.
    pub selected_in: Vec<Family>,
}

/// `overlap.json`: kept patterns (same canonical code) found in several families.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Overlap {
    pub schema: String,
    pub patterns: Vec<Shared>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FamilySummary {
    pub units: u64,
    pub vertices: u64,
    pub edges: u64,
    pub tau: Option<u64>,
    pub stopped_by: Stop,
    pub reported: u64,
    pub kept: u64,
    pub selected: u64,
    pub occurrences: u64,
    /// Σ workload of the units.
    pub workload: u128,
}

/// `summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub schema: String,
    pub families: BTreeMap<String, FamilySummary>,
    pub node_labels: u64,
    pub edge_labels: u64,
    pub overlap: u64,
    pub mis_algorithm: String,
}

pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    fn s<T: JsonSchema>() -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(T)).unwrap_or_default()
    }
    vec![
        ("mine_labels.schema.json", s::<Labels>()),
        ("mine_lg_map.schema.json", s::<LgMap>()),
        ("mine_grami.schema.json", s::<GramiFile>()),
        ("mine_pattern.schema.json", s::<PatternFile>()),
        ("mine_selection.schema.json", s::<Selection>()),
        ("mine_sweep.schema.json", s::<Sweep>()),
        ("mine_overlap.schema.json", s::<Overlap>()),
        ("mine_summary.schema.json", s::<Summary>()),
    ]
}
