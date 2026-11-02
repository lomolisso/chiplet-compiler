//! The kernel_classes artifacts (`out/kernel_classes/`):
//!
//! - `classes/<z>.json` (`ClassFile`): one kernel class: its representative kernel, the identity graph,
//!   its classes of axes and element types (the parameters), iteration axes, tags and members;
//! - `instances/<z>.json` (`InstancesTable`): T_z, one row per kernel instance;
//! - `certificates/<z>.json` (`CertificatesFile`): per member kernel, its maps to the representative;
//! - `sharing.json` (`Sharing`) and `summary.json` (`ClassesSummary`).

use std::collections::BTreeMap;

use cc_ir::Attr;
use cc_ir::absgraph::{Graph, PortLabel};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::components::algo::graph::{Family, Kind};
use crate::normalize::json::Order;

pub const CLASS_SCHEMA: &str = "cc.kernel_classes.class/1";
pub const INSTANCES_SCHEMA: &str = "cc.kernel_classes.instances/1";
pub const CERTIFICATES_SCHEMA: &str = "cc.kernel_classes.certificates/1";
pub const SHARING_SCHEMA: &str = "cc.kernel_classes.sharing/1";
pub const SUMMARY_SCHEMA: &str = "cc.kernel_classes.summary/1";

/// The reuse structure of a convolution's group count (design: kept in identity because it changes the
/// reuse): 1, one group per input feature (depthwise), or anything between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    One,
    Depthwise,
    Grouped,
}

/// The group kinds of a kernel instance: per kernel node, per group-count attribute.
pub type Groups = BTreeMap<u32, BTreeMap<String, Group>>;

/// What a role node of the identity graph says about the axis class (or window) it points to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ZRole {
    /// An axis of result `res` of the source node.
    Axis { res: u32 },
    /// The element type of result `res` of the source node.
    Elem { res: u32 },
    /// `dot_general`: a batching axis (of the lhs; the rhs axis has the same class).
    Batch,
    /// `dot_general`: a contracting axis.
    Contract,
    /// `reduce`: a reduced axis.
    Reduced,
    /// `broadcast_in_dim`: a result axis that no operand axis maps to.
    Expand,
    /// `dynamic_broadcast_in_dim`: an operand axis in `known_expanding_dimensions`.
    KnownExpanding,
    /// `dynamic_broadcast_in_dim`: an operand axis in `known_nonexpanding_dimensions`.
    KnownNonexpanding,
    /// Convolution: the input's batch axis.
    InBatch,
    /// Convolution: the input's feature axis.
    InFeature,
    /// Convolution: the kernel's input-feature axis.
    KernelIn,
    /// Convolution: the kernel's output-feature axis.
    KernelOut,
    /// Convolution: the output's batch axis.
    OutBatch,
    /// Convolution: the output's feature axis.
    OutFeature,
    /// A window (one spatial dimension of a convolution, one axis of a windowed reduction).
    Spatial,
    /// The window's input (operand) axis.
    WinIn,
    /// The window's kernel axis (convolution).
    WinKernel,
    /// The window's output axis.
    WinOut,
    /// The `k`-th axis of the reduction order ω.
    Omega { k: u32 },
    /// Scatter dimension numbers: entry `i` of list `list` (`update_window_dims`, `inserted_window_dims`,
    /// `input_batching_dims`, `scatter_indices_batching_dims`, `scattered_dims_to_operand_dims`,
    /// `index_vector_dim`, in that order).
    Scatter { list: u32, i: u32 },
}

/// A node of a kernel's identity graph: the kernel's nodes without positions, widths or values, plus one
/// node per axis class and per element class, joined through role nodes (vertex colouring and edge
/// subdivision; McKay and Piperno, "Practical graph isomorphism, II", J. Symbolic Computation
/// 60:94-112, 2014).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ZNode {
    /// A member: its op (or region argument), identity attributes, result families, ω's order, folded
    /// operands (port and family) and group kinds.
    Op {
        kind: Kind,
        attrs: BTreeMap<String, Attr>,
        results: Vec<Family>,
        order: Option<Order>,
        splats: Vec<(PortLabel, Family)>,
        groups: BTreeMap<String, Group>,
    },
    /// A value read from outside the kernel.
    Input { family: Family },
    /// A value used outside the kernel.
    Output,
    /// One class of axes (equal extents in every instance).
    Axis,
    /// One class of element types.
    Elem { family: Family },
    /// A role edge, subdivided; `pos` is the axis position when the kernel needs positional roles (two
    /// axes of one value in one class).
    Role { role: ZRole, pos: Option<u32> },
    /// One window (spatial dimension).
    Window,
}

/// A member kernel: one kernel of one component, and the group kinds it is classified with (a kernel
/// whose instances differ in group kind is a member once per kind).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct MemberKey {
    pub component: String,
    pub kernel: String,
    pub groups: Groups,
}

/// The local classes of a kernel graph: per node, per result, per axis its dim class; per node, per result
/// its element class.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
pub struct LocalClasses {
    pub dims: Vec<Vec<Vec<u32>>>,
    pub elems: Vec<Vec<u32>>,
    pub nd: u32,
    pub ne: u32,
    /// Per dim class: it holds an axis that a broadcast expands (extent 1 in every instance).
    pub unit: Vec<bool>,
}

/// What a class's core is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClassKind {
    /// `dot_general`, `convolution`, `dynamic_conv`.
    Contraction,
    /// `reduce`, `reduce_window`, `dynamic_reduce_window`.
    Reduction,
    ScatterReduce,
    /// No core.
    Map,
}

/// Tags of a class (docs/passes/kernel_classes.md §4).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Tags {
    /// No instance's axis extent depends on an input symbol: constant work per invocation.
    pub size_independent: bool,
    /// The domain has rank at most 1.
    pub low_rank: bool,
    /// Members that read a composite call's result.
    pub call_fed: u64,
    /// The identity graph uses positional roles.
    pub positional: bool,
}

/// One parameter of a class: an axis class (extent), an element class (width or format), or a value of
/// a member node (a size attribute, a folded constant).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Params {
    /// Axis classes, with whether each is an expanded unit axis (always 1).
    pub axes: Vec<bool>,
    /// Element classes: their family.
    pub elems: Vec<Family>,
    /// Value columns of the instance table: `n<node>.<attribute>` or `n<node>.splat<port>`.
    pub values: Vec<String>,
}

/// `classes/<z>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ClassFile {
    pub schema: String,
    pub id: String,
    /// WL hash of the identity graph (hex).
    pub hash: String,
    pub kind: ClassKind,
    /// `<op>/<family>` of the core, or `map`.
    pub core: String,
    pub tags: Tags,
    /// The representative: the first member.
    pub representative: MemberKey,
    /// Member roles of the representative's nodes (`input`/`output` for interface nodes).
    pub roles: Vec<String>,
    pub local: LocalClasses,
    pub params: Params,
    /// Axis classes of the iteration domain, with multiplicity: workload = their extents' product
    /// (times the window sizes of a windowed reduction).
    pub iteration: Vec<u32>,
    /// The identity graph; nodes `0..kernel_nodes` are the representative's kernel nodes.
    pub kernel_nodes: u32,
    pub identity: Graph<ZNode>,
    pub members: Vec<MemberKey>,
    /// Models with an instance.
    pub models: Vec<String>,
    pub instances: u64,
    pub workload: u64,
}

/// One member's maps to the representative: `nodes[i]` is the member node that representative node `i`
/// maps to; `axes[c]`, `elems[e]` the member's class for representative class `c`, `e`. `local` are the
/// member's own local classes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Certificate {
    pub member: MemberKey,
    pub nodes: Vec<u32>,
    pub axes: Vec<u32>,
    pub elems: Vec<u32>,
    pub local: LocalClasses,
}

/// `certificates/<z>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CertificatesFile {
    pub schema: String,
    pub id: String,
    pub certificates: Vec<Certificate>,
}

/// One instance of a class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub model: String,
    pub phase: String,
    /// The component instance.
    pub instance: String,
    pub component: String,
    pub kernel: String,
    /// Index of the member in the class's `members`.
    pub member: u32,
    /// Per axis class: its extent at the scenario sizes.
    pub extents: Vec<u64>,
    /// Per axis class: its symbolic size.
    pub sizes: Vec<String>,
    /// Per element class: the element type (MLIR spelling).
    pub elems: Vec<String>,
    /// Per value column (`Params.values`); `null` where the member has no such value.
    pub values: Vec<Option<Attr>>,
    /// |iteration domain| for one execution of the phase.
    pub workload: u64,
    pub trip_count: u64,
    /// `workload × trip_count`.
    pub weighted: u64,
}

/// `instances/<z>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InstancesTable {
    pub schema: String,
    pub id: String,
    /// The reduction order of the core, if any.
    pub omega: Option<Order>,
    pub rows: Vec<Row>,
}

/// Sharing at one level (components or kernel classes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Level {
    /// Items per model.
    pub per_model: BTreeMap<String, u64>,
    /// Distinct items.
    pub items: u64,
    /// Items used by at least two models.
    pub shared: u64,
    /// Pairwise Jaccard index of the models' item sets, rows and columns in `models` order.
    pub jaccard: Vec<Vec<f64>>,
    /// Mean of the Jaccard index over all model pairs.
    pub mean_jaccard: f64,
    /// Per model: the share of its weighted workload in items used by at least two models.
    pub coverage: BTreeMap<String, f64>,
    /// The same over all models' workload together (dominated by the heaviest models).
    pub coverage_total: f64,
    /// The mean of the per-model shares (every model counts once).
    pub coverage_mean: f64,
}

/// One view of the sharing analysis (all kernels, or activation kernels only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Variant {
    pub components: Level,
    pub kernels: Level,
    /// Weighted workload per model.
    pub workload: BTreeMap<String, u64>,
}

/// `sharing.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sharing {
    pub schema: String,
    pub models: Vec<String>,
    /// Every kernel.
    pub all: Variant,
    /// Kernels whose class is not `size_independent`.
    pub activation: Variant,
    /// The answer on `activation`: kernel-level mean Jaccard and workload coverage both exceed the
    /// component level.
    pub kernel_sharing_exceeds_component_sharing: bool,
}

/// `summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ClassesSummary {
    pub schema: String,
    pub classes: u64,
    /// Member kernels (kernels, plus one per extra group kind).
    pub members: u64,
    /// Kernels of the input.
    pub kernels: u64,
    /// Kernels classified more than once because their instances differ in group kind.
    pub split_by_group: u64,
    pub rows: u64,
    /// Classes per kind.
    pub kinds: BTreeMap<String, u64>,
    /// Classes per core key.
    pub cores: BTreeMap<String, u64>,
    pub size_independent: u64,
    pub low_rank: u64,
    pub call_fed: u64,
    pub positional: u64,
    /// WL buckets that held more than one class.
    pub split_buckets: u64,
    /// Classes per number of models using them.
    pub models_per_class: BTreeMap<u64, u64>,
}

pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "kernel_classes_class.schema.json",
            serde_json::to_value(schemars::schema_for!(ClassFile)).unwrap_or_default(),
        ),
        (
            "kernel_classes_instances.schema.json",
            serde_json::to_value(schemars::schema_for!(InstancesTable)).unwrap_or_default(),
        ),
        (
            "kernel_classes_certificates.schema.json",
            serde_json::to_value(schemars::schema_for!(CertificatesFile)).unwrap_or_default(),
        ),
        (
            "kernel_classes_sharing.schema.json",
            serde_json::to_value(schemars::schema_for!(Sharing)).unwrap_or_default(),
        ),
        (
            "kernel_classes_summary.schema.json",
            serde_json::to_value(schemars::schema_for!(ClassesSummary)).unwrap_or_default(),
        ),
    ]
}
