//! The components artifacts (`out/components/`):
//!
//! - `instances/<model>/<phase>.json` (`InstancesFile`);
//! - `components/<id>.postcard` (`StoredComponent`) with `<id>.postcard.summary.json` (`ComponentSummary`);
//! - `certificates/<model>/<phase>.json` (`CertificatesFile`).

use std::collections::BTreeMap;

use cc_ir::absgraph::Graph;
use cc_ir::{Attr, Dim};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::algo::classes::Iface;
use super::algo::graph::{ANode, NodeRef, Param};

pub const INSTANCES_SCHEMA: &str = "cc.components.instances/1";
pub const COMPONENT_SCHEMA: &str = "cc.components.component/1";
pub const SUMMARY_SCHEMA: &str = "cc.components.summary/1";
pub const CERTIFICATES_SCHEMA: &str = "cc.components.certificates/1";

/// `instances/<model>/<phase>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InstancesFile {
    pub schema: String,
    pub model: String,
    pub phase: String,
    /// Ops of the module.
    pub ops: u64,
    /// The entry's terminator: the one op outside every instance.
    pub root: Option<u32>,
    pub instances: Vec<InstanceRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InstanceKind {
    /// One executed call of a `cc.*` composite.
    Composite,
    /// A maximal weakly connected region of the entry's own ops.
    Anon,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InstanceRecord {
    /// `<model>/<phase>/<call op>/…` (composite) or `<model>/<phase>/anon:<hash>`.
    pub id: String,
    pub kind: InstanceKind,
    /// The composite's name (`cc.flax.Dense`).
    pub name: Option<String>,
    /// The composite's `composite_attributes` and `version`.
    pub attributes: BTreeMap<String, Attr>,
    pub position: Position,
    /// The decomposition function (module function index).
    pub func: Option<u32>,
    /// Owned op ids at `position.path`, as inclusive ranges. A composite instance also owns its call op,
    /// at its parent's path.
    pub ops: Vec<[u32; 2]>,
    pub counts: Counts,
    /// The component (`None`: the instance has no compute op).
    pub component: Option<String>,
    /// The values of the component's classes and parameters for this instance.
    pub bindings: Option<Bindings>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Position {
    /// The composite-call op ids from the entry (empty for anon regions).
    pub path: Vec<u32>,
    /// `cc_path` of the composite attributes (the source module path).
    pub cc_path: Option<String>,
    pub parent: Option<String>,
    pub children: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    /// Owned ops.
    pub ops: u64,
    /// Owned ops left out of the graph (size computation, shape-only constants).
    pub size: u64,
    /// Nodes of the graph.
    pub nodes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Bindings {
    /// Per dim class of the component.
    pub dims: Vec<Dim>,
    /// Per element class of the component, in MLIR spelling (`f32`, `bf16`, `i8`, `f8E4M3FN`).
    pub elems: Vec<String>,
    /// Per component node: its parameters (size-attribute entries, constant value, size value).
    pub params: Vec<Vec<Param>>,
}

/// `components/<id>.postcard`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StoredComponent {
    pub schema: String,
    pub id: String,
    /// WL hash (hex).
    pub hash: String,
    /// 0 without composite calls, else 1 + the children's maximum.
    pub height: u32,
    pub graph: Graph<ANode>,
    pub iface: Iface,
    /// Numbers of dim and element classes.
    pub dims: u32,
    pub elems: u32,
}

/// `components/<id>.postcard.summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComponentSummary {
    pub schema: String,
    pub id: String,
    pub hash: String,
    pub height: u32,
    pub nodes: u64,
    /// Dataflow edges (parallel edges counted).
    pub edges: u64,
    pub dims: u32,
    pub elems: u32,
    /// Ops per name.
    pub ops: BTreeMap<String, u64>,
    pub children: Vec<String>,
    /// Composite names of the instances, with counts (`anon` for anon regions).
    pub names: BTreeMap<String, u64>,
    /// Instances per model.
    pub models: BTreeMap<String, u64>,
    pub phases: u64,
    pub instances: Vec<String>,
}

/// `certificates/<model>/<phase>.json`: per instance, the image of every component node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CertificatesFile {
    pub schema: String,
    pub model: String,
    pub phase: String,
    pub certificates: BTreeMap<String, Vec<NodeRef>>,
}

/// Sorted ids as inclusive ranges.
pub fn ranges(ids: &[u32]) -> Vec<[u32; 2]> {
    let mut out: Vec<[u32; 2]> = Vec::new();
    for &i in ids {
        match out.last_mut() {
            Some(r) if r[1] + 1 == i => r[1] = i,
            _ => out.push([i, i]),
        }
    }
    out
}

pub fn expand(r: &[[u32; 2]]) -> impl Iterator<Item = u32> + '_ {
    r.iter().flat_map(|&[a, b]| a..=b)
}

pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "components_instances.schema.json",
            serde_json::to_value(schemars::schema_for!(InstancesFile)).unwrap_or_default(),
        ),
        (
            "components_component.schema.json",
            serde_json::to_value(schemars::schema_for!(StoredComponent)).unwrap_or_default(),
        ),
        (
            "components_summary.schema.json",
            serde_json::to_value(schemars::schema_for!(ComponentSummary)).unwrap_or_default(),
        ),
        (
            "components_certificates.schema.json",
            serde_json::to_value(schemars::schema_for!(CertificatesFile)).unwrap_or_default(),
        ),
    ]
}
