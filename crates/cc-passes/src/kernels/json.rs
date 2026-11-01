//! The kernels artifacts (`out/kernels/`):
//!
//! - `components/<id>.json` (`KernelPartition`): the kernels of one component, its boundary ops and a
//!   topological order of the quotient graph;
//! - `kernels/<id>.postcard` (`KernelGraphs`) with `<id>.postcard.summary.json` (`PartitionSummary`): every
//!   kernel as a graph of its own, with its inputs and outputs as interface nodes;
//! - `summary.json` (`KernelsSummary`): the statistics over all components.

use std::collections::BTreeMap;

use cc_ir::absgraph::{EdgeEnd, Graph};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::components::algo::graph::{ADim, AType};
use crate::normalize::json::NNode;

pub const PARTITION_SCHEMA: &str = "cc.kernels.partition/1";
pub const GRAPHS_SCHEMA: &str = "cc.kernels.graphs/1";
pub const SUMMARY_SCHEMA: &str = "cc.kernels.component_summary/1";
pub const TOTALS_SCHEMA: &str = "cc.kernels.summary/1";

/// The role of a kernel member (docs/passes/kernels.md §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The contraction, reduction or scatter-reduce.
    Core,
    /// A node of the core's regions (region arguments, the combiner, the terminator).
    Body,
    /// Element-wise, on one core operand's path, with every consumer in the kernel.
    Prologue,
    /// Element-wise over the core result's domain.
    Epilogue,
    /// Element-wise member of a kernel without a core.
    Map,
    /// A `broadcast_in_dim` of a kernel input whose consumers are all in the kernel.
    Broadcast,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::Core => "core",
            Role::Body => "body",
            Role::Prologue => "prologue",
            Role::Epilogue => "epilogue",
            Role::Map => "map",
            Role::Broadcast => "broadcast",
        }
    }
}

/// Why a node is in no kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryKind {
    /// `input`, `output`, `size_in`.
    Interface,
    /// A composite call: its child component has kernels of its own.
    Call,
    /// `func.return` and other structure.
    Control,
    /// Index remapping (reshape, transpose, slice, pad, concatenate, …; broadcasts not absorbed).
    Layout,
    /// Gather, dynamic slices, scatter without arithmetic update.
    DataMovement,
    /// Size computation.
    Size,
    /// `constant`, `iota`.
    Constant,
    /// `sort`, `fft`.
    Irregular,
}

impl BoundaryKind {
    pub fn name(self) -> &'static str {
        match self {
            BoundaryKind::Interface => "interface",
            BoundaryKind::Call => "call",
            BoundaryKind::Control => "control",
            BoundaryKind::Layout => "layout",
            BoundaryKind::DataMovement => "data_movement",
            BoundaryKind::Size => "size",
            BoundaryKind::Constant => "constant",
            BoundaryKind::Irregular => "irregular",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Member {
    /// Node of the normalized component.
    pub node: u32,
    pub role: Role,
}

/// One dataflow edge end crossing a kernel's border.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct Crossing {
    pub src: u32,
    pub dst: u32,
    pub end: EdgeEnd,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Kernel {
    /// `k<i>`: kernels with a core first, in the SSA order of their cores, then map kernels.
    pub id: String,
    /// The core node, if any.
    pub core: Option<u32>,
    /// The dims of the core's (first) result, or of every map member.
    pub domain: Vec<ADim>,
    /// Members in ascending node id.
    pub members: Vec<Member>,
    /// Edge ends entering the kernel, sorted.
    pub inputs: Vec<Crossing>,
    /// Edge ends leaving the kernel, sorted.
    pub outputs: Vec<Crossing>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Boundary {
    pub node: u32,
    pub kind: BoundaryKind,
    /// For a node of a boundary op's regions (e.g. a `sort` comparator): the owner.
    pub owner: Option<u32>,
}

/// One node of the quotient graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    /// Index into `kernels`.
    Kernel(u32),
    /// A boundary op (with its regions, if any), by its node.
    Boundary(u32),
}

/// `components/<id>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KernelPartition {
    pub schema: String,
    pub id: String,
    pub kernels: Vec<Kernel>,
    /// Boundary nodes in ascending id.
    pub boundaries: Vec<Boundary>,
    /// Every part once, in a topological order of the quotient graph: the acyclicity certificate.
    pub order: Vec<Part>,
    /// Members banned from a kernel to break a cycle of the quotient graph.
    pub evictions: u32,
}

/// A node of an extracted kernel graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum KNode {
    /// A member, with the label of its normalized node.
    Member { origin: u32, role: Role, node: NNode },
    /// An outside value read by the kernel: result `res` of node `src`.
    Input { src: u32, res: u32, ty: AType },
    /// A member's result `res` used outside the kernel.
    Output { node: u32, res: u32, ty: AType },
}

/// One kernel as a graph: members in ascending origin, then inputs, then outputs. Edges from an input
/// keep the destination's port and use result 0; edges to an output keep the member's result and use
/// port 0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct KernelGraph {
    pub id: String,
    pub graph: Graph<KNode>,
}

/// `kernels/<id>.postcard`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct KernelGraphs {
    pub schema: String,
    pub id: String,
    pub kernels: Vec<KernelGraph>,
}

/// `kernels/<id>.postcard.summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PartitionSummary {
    pub schema: String,
    pub id: String,
    pub nodes: u64,
    pub kernels: u64,
    pub core_kernels: u64,
    pub map_kernels: u64,
    /// Kernels with a core, per `<op>/<family>` of the core.
    pub cores: BTreeMap<String, u64>,
    /// Members per role.
    pub roles: BTreeMap<String, u64>,
    /// Boundary nodes per kind.
    pub boundaries: BTreeMap<String, u64>,
    /// Map kernels that read a composite call's result (an epilogue across a component border).
    pub call_fed_maps: u64,
    /// Map kernels whose domain has rank at most 1 (parameter arithmetic).
    pub low_rank_maps: u64,
    pub evictions: u64,
    /// Instances of the component.
    pub instances: u64,
    /// Instances per model.
    pub models: BTreeMap<String, u64>,
}

/// `summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KernelsSummary {
    pub schema: String,
    pub components: u64,
    pub kernels: u64,
    pub core_kernels: u64,
    pub map_kernels: u64,
    /// Components per number of kernels.
    pub kernels_per_component: BTreeMap<u64, u64>,
    /// Per `<op>/<family>`: kernels, and kernels × component instances.
    pub cores: BTreeMap<String, (u64, u64)>,
    pub roles: BTreeMap<String, u64>,
    pub boundaries: BTreeMap<String, u64>,
    pub call_fed_maps: u64,
    pub low_rank_maps: u64,
    pub evictions: u64,
    /// Per model: kernels × instances, of which with a core.
    pub models: BTreeMap<String, (u64, u64)>,
}

pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "kernels_partition.schema.json",
            serde_json::to_value(schemars::schema_for!(KernelPartition)).unwrap_or_default(),
        ),
        (
            "kernels_graphs.schema.json",
            serde_json::to_value(schemars::schema_for!(KernelGraphs)).unwrap_or_default(),
        ),
        (
            "kernels_component_summary.schema.json",
            serde_json::to_value(schemars::schema_for!(PartitionSummary)).unwrap_or_default(),
        ),
        (
            "kernels_summary.schema.json",
            serde_json::to_value(schemars::schema_for!(KernelsSummary)).unwrap_or_default(),
        ),
    ]
}
