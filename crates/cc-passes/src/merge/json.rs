//! The merge artifacts (`out/merge/`):
//!
//! - `fus/<F>.json` (`FuFile`): a functional unit: blocks (operators, constant registers, ports,
//!   accumulator registers) with their operand sources (two or more sources make a multiplexer), its
//!   host skeleton, the units it covers, area and critical path;
//! - `configs/<F>/<unit>/<t>.json` (`ConfigFile`): the configuration of one covered unit and width tuple:
//!   the block of every datapath vertex, the multiplexer selects, the width mode of every block;
//! - `proofs/<F>/<unit>/<t>.smt2` and `.result`: the unit under that configuration equals the datapath;
//! - `arrays/<A>.json` (`ArrayFile`): a parameterized array unit and the template classes it serves;
//! - `proofs/arrays/<A>/<z>/<t>.smt2` and `.result`: its processing element equals the class's core step;
//! - `trace/<family>.json` (`Trace`): every agglomeration step (the area-vs-units curve);
//! - `cliques/<family>/<step>.json` (`CliqueCert`): small compatibility graphs with their solution;
//! - `coverage.json` (`CoverageFile`), `summary.json` (`Summary`).

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::mine::json::Family;

pub const FU_SCHEMA: &str = "cc.merge.fu/1";
pub const CONFIG_SCHEMA: &str = "cc.merge.config/1";
pub const ARRAY_SCHEMA: &str = "cc.merge.array/1";
pub const TRACE_SCHEMA: &str = "cc.merge.trace/1";
pub const CLIQUE_SCHEMA: &str = "cc.merge.clique/1";
pub const COVERAGE_SCHEMA: &str = "cc.merge.coverage/1";
pub const SUMMARY_SCHEMA: &str = "cc.merge.summary/1";

/// What a block is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    /// An element-wise operator.
    Op,
    /// A constant register (its value is set per configuration) or a literal zero.
    Const,
    /// An input port (a kernel input, or a template's core result).
    Input,
    /// An accumulator register: ports `next` (0) and `init` (1); result 0 is its current value, result 1
    /// its final value (what follows the reduction reads).
    Acc,
    /// An output port.
    Output,
}

/// One operand source: result `res` of block `block`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct Source {
    pub block: u32,
    pub res: u32,
}

/// The price of a block or a multiplexer in the library.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Price {
    /// Library keys (several for a complex operator), empty for wiring.
    pub keys: Vec<String>,
    pub area_um2: f64,
    pub delay_ps: f64,
    /// Some key has no measured row (a pending primitive, or a size outside the library).
    pub estimated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Block {
    pub id: u32,
    pub kind: BlockKind,
    /// The op (without its prefix) for an operator; `const`, `input`, `acc`, `output` otherwise.
    pub op: String,
    /// Semantic attributes (`comparison_direction`, `compare_type`); a literal constant's `value`.
    pub attrs: BTreeMap<String, String>,
    /// The widest type served at every position: operand types then the result type (an operator), the
    /// value type otherwise.
    pub types: Vec<String>,
    /// Per operand port, its sources (sorted); two or more make a multiplexer.
    pub sources: Vec<Vec<Source>>,
    /// The mined-pattern occurrence this block belongs to (`groups`), and its vertex in the pattern.
    pub group: Option<u32>,
    pub member: Option<u32>,
    pub price: Price,
}

/// A multiplexer in front of one operand port.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Mux {
    pub block: u32,
    pub port: u32,
    pub inputs: u32,
    pub width: u32,
    pub price: Price,
}

/// Blocks that implement one mined pattern (merged only with blocks of the same pattern).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Group {
    pub id: u32,
    pub pattern: String,
}

/// A unit (a datapath class, or one stage of a template class) and its configurations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Covered {
    pub unit: String,
    pub class: String,
    pub tuples: Vec<String>,
}

/// `fus/<F>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FuFile {
    pub schema: String,
    pub id: String,
    pub family: Family,
    /// The class whose skeleton every covered class embeds into (⊑).
    pub host: String,
    pub blocks: Vec<Block>,
    pub groups: Vec<Group>,
    pub muxes: Vec<Mux>,
    pub covers: Vec<Covered>,
    /// Blocks plus multiplexers, µm².
    pub area_um2: f64,
    /// Critical path (operators and multiplexers at the `fast` target), ps.
    pub depth_ps: f64,
    pub bound_ps: f64,
    pub estimated: bool,
}

/// How a block runs for one configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// At the block's own types.
    Native,
    /// Narrower integers extended by their signedness, the result truncated back.
    Widen,
}

/// What one block does in one configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Binding {
    pub block: u32,
    /// The datapath node: `v<id>`, `in:<name>`, `acc<i>`, `out:<name>`.
    pub node: String,
    /// The SMT variable of a source (input name, constant column, `acc<i>`/`res<i>`), the output's
    /// name; `None` for operators and literal constants.
    pub var: Option<String>,
    /// The types at this tuple (operands then result, as `Block::types`).
    pub types: Vec<String>,
    pub mode: Mode,
    /// For an operator: the block port of each datapath operand (a permutation of a commutative pair).
    pub ports: Vec<u32>,
}

/// The select of one multiplexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Select {
    pub block: u32,
    pub port: u32,
    /// Index into the port's sources.
    pub select: u32,
}

/// `configs/<F>/<unit>/<t>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConfigFile {
    pub schema: String,
    pub fu: String,
    pub unit: String,
    pub class: String,
    pub tuple: String,
    pub bindings: Vec<Binding>,
    pub selects: Vec<Select>,
}

/// How an array serves one width tuple of a class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArrayMode {
    Native,
    /// Narrower integer operands, extended.
    Widen,
    /// A wider integer operand split into pieces the processing element multiplies one per pass.
    Compose,
}

/// One width tuple of a covered class on an array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ArrayTuple {
    pub tuple: String,
    /// Core operand and result types of the class at this tuple.
    pub types: Vec<String>,
    pub mode: ArrayMode,
    /// Passes per multiply-accumulate (1 unless composed): the throughput penalty.
    pub passes: u32,
    /// Bits of each piece of the composed operand, low piece first (empty unless composed).
    pub pieces: Vec<u32>,
    /// The composed operand (`lhs` or `rhs`).
    pub split: Option<String>,
}

/// A template class served by an array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ArrayCover {
    pub class: String,
    /// The class embeds into the host (an `embedding.json` pair), unless it is the host.
    pub embedded: bool,
    /// The class's selected space-time mapping (`templates/<z>.json`).
    pub candidate: u32,
    pub tuples: Vec<ArrayTuple>,
    pub workload: u64,
}

/// `arrays/<A>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrayFile {
    pub schema: String,
    pub id: String,
    pub dataflow: String,
    pub host: String,
    /// Array dims as functions of the host's group extents (parameters of the unit).
    pub dims: Vec<String>,
    /// Processing element: lhs, rhs and accumulator types.
    pub pe: Vec<String>,
    /// One processing element: two converts, a multiply, an add and the accumulator register.
    pub pe_price: Price,
    pub covers: Vec<ArrayCover>,
}

/// One agglomeration step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Step {
    pub step: u32,
    /// The merged units (by their first covered unit) and the area saved.
    pub a: String,
    pub b: String,
    pub gain_um2: f64,
    /// The family after the step.
    pub units: u64,
    pub area_um2: f64,
    pub depth_ps: f64,
    pub clique: CliqueStat,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CliqueStat {
    pub algorithm: String,
    pub vertex_mappings: u64,
    pub arc_mappings: u64,
    pub objective: f64,
    pub optimal: bool,
    /// Solves after adding cycle no-goods.
    pub rounds: u32,
    /// Why another algorithm answered (`size`, `timeout`).
    pub fallback: Option<String>,
    /// The stored instance (`cliques/<family>/<step>.json`).
    pub certificate: Option<String>,
}

/// `trace/<family>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub schema: String,
    pub family: Family,
    pub period_ps: f64,
    pub bound_ps: f64,
    pub initial_units: u64,
    pub initial_area_um2: f64,
    pub steps: Vec<Step>,
    /// `gain`: no admissible pair saves area; `target`: `target_units` reached; `none`: no pair.
    pub stopped_by: String,
    /// Exact gains evaluated (compatibility graphs solved).
    pub evaluated: u64,
}

/// One node of a stored compatibility graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CertNode {
    pub weight: f64,
    /// `vertex` (a block pair) or `arc` (a wire pair).
    pub kind: String,
    pub what: String,
}

/// `cliques/<family>/<step>.json`: the maximum-weight problem of one step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CliqueCert {
    pub schema: String,
    pub nodes: Vec<CertNode>,
    /// Pairs that cannot both be chosen.
    pub conflicts: Vec<(u32, u32)>,
    /// `(arc, vertex)`: the arc needs the vertex mapping.
    pub implies: Vec<(u32, u32)>,
    /// Sets that cannot all be chosen (cycles found in earlier rounds).
    pub nogoods: Vec<Vec<u32>>,
    pub chosen: Vec<u32>,
    pub objective: f64,
    pub algorithm: String,
    pub optimal: bool,
}

/// How one class is covered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ClassCover {
    pub class: String,
    pub route: String,
    /// `(unit, fu)` for every unit of the class (the class, or its stages).
    pub units: Vec<(String, String)>,
    pub array: Option<String>,
    /// Template stages without vertices (nothing to cover).
    pub empty_stages: Vec<String>,
}

/// `coverage.json`: Z_compute ⊆ ⋃ cover(F).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoverageFile {
    pub schema: String,
    pub classes: Vec<ClassCover>,
    pub total: u64,
    pub covered: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FamilySummary {
    pub units: u64,
    pub fus: u64,
    pub initial_area_um2: f64,
    pub area_um2: f64,
    pub steps: u64,
    pub evaluated: u64,
    pub bound_ps: f64,
    pub stopped_by: String,
    pub macro_groups: u64,
}

/// `summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub schema: String,
    pub kit: String,
    pub requant_variant: String,
    pub algorithm: String,
    pub families: BTreeMap<String, FamilySummary>,
    pub fus: u64,
    pub arrays: u64,
    pub configs: u64,
    pub proofs: u64,
    pub array_proofs: u64,
    /// Steps by the algorithm that answered, and those not proven optimal.
    pub cliques: BTreeMap<String, u64>,
    pub not_optimal: u64,
    pub estimated_blocks: u64,
    pub classes: u64,
    pub covered: u64,
}

pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    fn s<T: JsonSchema>() -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(T)).unwrap_or_default()
    }
    vec![
        ("merge_fu.schema.json", s::<FuFile>()),
        ("merge_config.schema.json", s::<ConfigFile>()),
        ("merge_array.schema.json", s::<ArrayFile>()),
        ("merge_trace.schema.json", s::<Trace>()),
        ("merge_clique.schema.json", s::<CliqueCert>()),
        ("merge_coverage.schema.json", s::<CoverageFile>()),
        ("merge_summary.schema.json", s::<Summary>()),
    ]
}
