//! The iteration_space artifacts (`out/iteration_space/`):
//!
//! - `skeletons/<z>.json` (`Skeleton`): iteration dims, the access of every operand and every member's
//!   values, the class (map, reduce, contraction), reuse, and the prologue and epilogue stages;
//! - `dependences/<z>.json` (`Dependences`): incidence groups and dependence vectors under ω;
//! - `templates/<z>.json` (`TemplateFile`): the valid space-time mappings of a template class, ranked;
//! - `embedding.json` (`Embedding`): the relation ⊑ with one certificate per pair;
//! - `routing.json` (`Routing`): every class → `template` | `datapath` with its reason;
//! - `smt/<file>.smt2` and `smt/<file>.result` (`ProofResult`): the access and mapping obligations;
//! - `summary.json` (`Summary`).

use std::collections::BTreeMap;

use cc_ir::ops::iter::Tag;
use cc_smt::{Solver, Verdict};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::normalize::json::Order;

pub const SKELETON_SCHEMA: &str = "cc.iteration_space.skeleton/1";
pub const DEPENDENCES_SCHEMA: &str = "cc.iteration_space.dependences/1";
pub const TEMPLATE_SCHEMA: &str = "cc.iteration_space.template/1";
pub const EMBEDDING_SCHEMA: &str = "cc.iteration_space.embedding/1";
pub const ROUTING_SCHEMA: &str = "cc.iteration_space.routing/1";
pub const SUMMARY_SCHEMA: &str = "cc.iteration_space.summary/1";
pub const RESULT_SCHEMA: &str = "cc.iteration_space.proof_result/1";

/// What a class computes, from its accesses (design §4.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SkelClass {
    /// No reduction dim.
    Map,
    /// Reduction dims, and every core input is indexed by every parallel dim (no reuse).
    Reduce,
    /// Reduction dims, and some core input is not indexed by some parallel dim (reuse).
    Contraction,
}

impl SkelClass {
    pub fn name(self) -> &'static str {
        match self {
            SkelClass::Map => "map",
            SkelClass::Reduce => "reduce",
            SkelClass::Contraction => "contraction",
        }
    }
}

/// A symbolic factor of a coefficient, bound to the class: a value column of the instance table
/// (`n<node>.<attribute>`, entry `index`), an entry of a shape operand (known only at run time), or the
/// extent of an iteration dim.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Factor {
    Column { column: String, index: u32 },
    Runtime { node: u32, port: u32, index: u32 },
    Extent { dim: u32 },
}

/// `k · Π factors · i_dim` (no dim: a constant term).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct Term {
    pub k: i64,
    pub factors: Vec<Factor>,
    pub dim: Option<u32>,
}

/// One index of a value: `(Σ terms) / divisor`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Row {
    pub terms: Vec<Term>,
    /// Exact division (base or lhs dilation): iterations where it does not divide read no element.
    pub divisor: Option<Factor>,
    /// Out-of-range indices read the padding value.
    pub padded: bool,
    /// The index also depends on data (scatter); the terms are its affine part.
    pub data: bool,
}

/// An access: one row per axis of the value.
pub type Access = Vec<Row>;

/// The extent of an iteration dim.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Extent {
    /// An axis class of the kernel class (the instance table's `extents[class]`).
    Class { class: u32 },
    /// A literal.
    Const { value: i64 },
    /// A value column entry (a window size).
    Column { column: String, index: u32 },
    /// A shape-operand entry, known only at run time.
    Runtime { node: u32, port: u32, index: u32 },
    /// An axis class divided by a factor (a convolution's output features per group).
    Quot { class: u32, by: Factor },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Dim {
    pub tag: Tag,
    pub extent: Extent,
    /// The core op's name for it (`batch`, `lhs_free`, `contract`, `out_spatial`, `window`, …).
    pub role: String,
}

/// ω on the iteration dims: the order and the reduction dims, outermost first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DimOrder {
    pub order: Order,
    pub dims: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    In,
    Out,
}

/// One operand of the skeleton: a core operand or result, or a kernel input use or output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Operand {
    /// `lhs`, `rhs`, `input<i>`, `init<i>`, `indices`, `updates<i>`, `result<i>` (core), `in<j>`, `out<j>`
    /// (kernel interface).
    pub name: String,
    pub direction: Direction,
    /// A core operand (or result).
    pub core: bool,
    /// The kernel node whose value it is (an input node, a member, or the member an output reads).
    pub node: u32,
    pub res: u32,
    /// Axis classes of the value, and its element class.
    pub axes: Vec<u32>,
    pub elem: u32,
    pub access: Access,
    /// Iteration dims that do not index it (reuse along `e_d`).
    pub reuse: Vec<u32>,
}

/// The accesses of one member node: per result and per operand port (`None`: a shape or folded port).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct NodeAccess {
    pub node: u32,
    pub results: Vec<Access>,
    pub operands: Vec<Option<Access>>,
}

/// Element-wise stages around a template's core (handed to `datapaths` like map kernels).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Stages {
    /// Per core input port: the prologue members on its path (once per element of that operand).
    pub prologue: BTreeMap<u32, Vec<u32>>,
    /// The epilogue members (once per element of the result).
    pub epilogue: Vec<u32>,
}

/// `skeletons/<z>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Skeleton {
    pub schema: String,
    pub id: String,
    pub class: SkelClass,
    /// The kernel_classes kind and core key.
    pub kind: String,
    pub core: String,
    pub core_node: Option<u32>,
    /// Every access is affine in the iteration vector (no data-dependent index).
    pub affine: bool,
    pub dims: Vec<Dim>,
    pub omega: Option<DimOrder>,
    pub operands: Vec<Operand>,
    pub accesses: Vec<NodeAccess>,
    pub stages: Stages,
    /// The access proof (`smt/<file>`).
    pub proof: String,
}

/// Which operands index an iteration dim, for a contraction (`a`, `b` the core inputs, `c` the result).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    /// a, b, c: an outer loop.
    Batch,
    /// a, c.
    M,
    /// b, c.
    N,
    /// a, b: the reduction.
    K,
    /// Reduction: the kept dims.
    Par,
    /// Reduction: the reduced dims.
    Red,
}

impl Group {
    pub fn name(self) -> &'static str {
        match self {
            Group::Batch => "batch",
            Group::M => "m",
            Group::N => "n",
            Group::K => "k",
            Group::Par => "par",
            Group::Red => "red",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VectorKind {
    /// A reused input passed on to the next iteration along the vector (localization).
    Propagate,
    /// The accumulator chain of ω.
    Accumulate,
}

/// One dependence vector over the space (grouped, linearized dims).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Vector {
    /// The skeleton operand it carries.
    pub operand: String,
    pub kind: VectorKind,
    pub v: Vec<i64>,
}

/// `dependences/<z>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Dependences {
    pub schema: String,
    pub id: String,
    /// The dependences are constant vectors (a uniform recurrence, Karp, Miller and Winograd 1967).
    pub uniform: bool,
    pub reason: String,
    /// Iteration dims per group, in linearization order (ω's order for the reduction).
    pub groups: BTreeMap<Group, Vec<u32>>,
    /// Dims that fit no group (e.g. a reduction dim indexing one input only).
    pub other: Vec<u32>,
    /// The space: the groups the recurrence runs over, in order (batch dims are outer loops).
    pub space: Vec<Group>,
    pub vectors: Vec<Vector>,
}

/// One link of a mapping: the dependence's displacement on the array and its delay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Link {
    pub operand: String,
    /// `S d`.
    pub step: Vec<i64>,
    /// `λ · d` registers.
    pub delay: i64,
}

/// One valid space-time mapping (λ, u).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Candidate {
    pub lambda: Vec<i64>,
    /// Projection direction: iterations on one line `z + t·u` share a PE.
    pub u: Vec<i64>,
    /// Allocation: `S u = 0`, PE coordinates `S z`.
    pub s: Vec<Vec<i64>>,
    pub links: Vec<Link>,
    /// The operand whose dependence does not move (`S d = 0`), if any.
    pub stationary: Option<String>,
    /// `output_stationary`, `weight_stationary`, `input_stationary` or `systolic` (nothing stationary).
    pub dataflow: String,
    /// Array dims as functions of the group extents (`E(m)`, `E(m) + E(k) - 1`, …).
    pub array: Vec<String>,
    /// Workload-weighted PE utilization |D| / (#PE · T) at the rows' extents.
    pub utilization: f64,
    /// Workload-weighted inter-PE link bits per PE per step.
    pub link_bits: f64,
    /// 0 = best.
    pub rank: u32,
}

/// `templates/<z>.json` (template classes only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TemplateFile {
    pub schema: String,
    pub id: String,
    pub space: Vec<Group>,
    /// Per space group: its dims (its extent is their product).
    pub groups: BTreeMap<Group, Vec<u32>>,
    pub candidates: Vec<Candidate>,
    pub selected: u32,
    /// Element families of the core inputs and result.
    pub families: Vec<String>,
    /// The mapping proof (`smt/<file>`).
    pub proof: String,
}

/// A host parameter's value in an embedding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Value {
    Zero,
    One,
    /// The guest's factor.
    Guest(Factor),
}

/// One pair `guest ⊑ host`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Pair {
    pub guest: String,
    pub host: String,
    /// Host dim of every guest dim (φ); host dims outside the image have extent 1.
    pub dims: Vec<u32>,
    /// Host operand index of every guest operand (π, over the embedding operands).
    pub operands: Vec<u32>,
    /// Per guest operand: the host row of each guest row (other host rows vanish).
    pub rows: Vec<Vec<u32>>,
    /// Host factors and their values (others: unused).
    pub params: Vec<(Factor, Value)>,
}

/// `embedding.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Embedding {
    pub schema: String,
    /// Skeleton operands that take part, per class: core operands, or the interface of a map.
    pub pairs: Vec<Pair>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    Template,
    Datapath,
}

/// The numbers behind a contraction's routing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Reuse {
    /// Share of the weighted workload in rows where both inputs reach `min_reuse`.
    pub share: f64,
    /// Workload-weighted mean reuse factor of each input.
    pub lhs: f64,
    pub rhs: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Routed {
    pub id: String,
    pub route: Route,
    pub class: SkelClass,
    pub core: String,
    pub reason: String,
    pub reuse: Option<Reuse>,
    /// Template classes: the selected candidate's dataflow.
    pub dataflow: Option<String>,
    /// Template classes: the stages handed to `datapaths`.
    pub stages: Option<Stages>,
    pub workload: u64,
}

/// `routing.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Routing {
    pub schema: String,
    pub classes: Vec<Routed>,
}

/// `smt/<file>.result`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProofResult {
    pub schema: String,
    pub file: String,
    /// blake3 of the `.smt2` file.
    pub query_hash: String,
    pub solver: Solver,
    pub version: String,
    pub verdict: Verdict,
}

/// `summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub schema: String,
    pub classes: u64,
    /// Skeleton classes per kernel_classes kind: `<kind> -> <class>`.
    pub classes_by_kind: BTreeMap<String, u64>,
    pub uniform: u64,
    pub templates: u64,
    pub datapaths: u64,
    /// Template classes per core key.
    pub template_cores: BTreeMap<String, u64>,
    /// Template classes per selected dataflow.
    pub dataflows: BTreeMap<String, u64>,
    /// Datapath classes per reason kind.
    pub datapath_reasons: BTreeMap<String, u64>,
    /// Weighted workload routed to templates and to datapaths.
    pub template_workload: u64,
    pub datapath_workload: u64,
    pub embedding_pairs: u64,
    /// Embedding searches stopped by their step budget (counted as no embedding).
    pub embedding_over_budget: u64,
    pub proofs: u64,
}

pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "iteration_space_skeleton.schema.json",
            serde_json::to_value(schemars::schema_for!(Skeleton)).unwrap_or_default(),
        ),
        (
            "iteration_space_dependences.schema.json",
            serde_json::to_value(schemars::schema_for!(Dependences)).unwrap_or_default(),
        ),
        (
            "iteration_space_template.schema.json",
            serde_json::to_value(schemars::schema_for!(TemplateFile)).unwrap_or_default(),
        ),
        (
            "iteration_space_embedding.schema.json",
            serde_json::to_value(schemars::schema_for!(Embedding)).unwrap_or_default(),
        ),
        (
            "iteration_space_routing.schema.json",
            serde_json::to_value(schemars::schema_for!(Routing)).unwrap_or_default(),
        ),
        (
            "iteration_space_proof_result.schema.json",
            serde_json::to_value(schemars::schema_for!(ProofResult)).unwrap_or_default(),
        ),
        (
            "iteration_space_summary.schema.json",
            serde_json::to_value(schemars::schema_for!(Summary)).unwrap_or_default(),
        ),
    ]
}
