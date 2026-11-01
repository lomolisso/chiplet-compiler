//! The normalize artifacts (`out/normalize/`):
//!
//! - `components/<id>.postcard` (`NormalizedComponent`) with `<id>.postcard.summary.json` (`NormalizedSummary`);
//! - `ledger/<id>.json` (`LedgerFile`): every rewrite applied, in order;
//! - `rules/<rule>.smt2` (the obligation) and `rules/<rule>.result` (`RuleResult`, JSON).

use std::collections::BTreeMap;

use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};
use cc_smt::{Solver, Verdict};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::components::algo::classes::Iface;
use crate::components::algo::graph::{ANode, AType};

pub const COMPONENT_SCHEMA: &str = "cc.normalize.component/1";
pub const SUMMARY_SCHEMA: &str = "cc.normalize.summary/1";
pub const LEDGER_SCHEMA: &str = "cc.normalize.ledger/1";
pub const RESULT_SCHEMA: &str = "cc.normalize.rule_result/1";

/// A reduction order ω (docs/passes/normalize.md §4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    /// One accumulator: the init value, then the elements in ascending lexicographic order of `axes`.
    Seq,
    /// The elements in the same order, grouped by `fan_in` (each group left to right), level by level;
    /// the init value is combined at the root.
    Tree { fan_in: u32 },
}

/// The reduction order of one op: `order` over `axes` (outermost first) of the operand the op's
/// definition reduces (docs/passes/normalize.md §4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Omega {
    pub order: Order,
    pub axes: Vec<u32>,
}

/// An operand folded into its consumer: the uniform value of a splat constant, directly or through a
/// `broadcast_in_dim` of a scalar constant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Splat {
    /// The consumer's port (`commutative` for an operand of a commutative op).
    pub port: PortLabel,
    /// The constant's node in the input component (its value is that node's instance parameter).
    pub constant: u32,
    /// The broadcast's node in the input component, if any.
    pub broadcast: Option<u32>,
    /// The type of the folded operand.
    pub ty: AType,
}

/// A node of a normalized graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NNode {
    pub label: ANode,
    /// Set on every reduction (`reduce`, `reduce_window`, `dot_general`, `convolution`, `scatter`, …).
    pub omega: Option<Omega>,
    /// Folded operands, sorted by port.
    pub splats: Vec<Splat>,
    /// The node of the input component: instance bindings are read through it.
    pub origin: u32,
}

/// `components/<id>.postcard`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NormalizedComponent {
    pub schema: String,
    pub id: String,
    pub height: u32,
    pub graph: Graph<NNode>,
    /// The input component's interface (normalization keeps it).
    pub iface: Iface,
    /// Dim and element class counts of the input component (class ids are kept).
    pub dims: u32,
    pub elems: u32,
}

/// `components/<id>.postcard.summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NormalizedSummary {
    pub schema: String,
    pub id: String,
    pub height: u32,
    pub nodes_in: u64,
    pub nodes: u64,
    pub edges: u64,
    /// Rewrites applied, per rule.
    pub rules: BTreeMap<String, u64>,
    /// Reductions per order (`seq`, `tree:<b>`).
    pub omega: BTreeMap<String, u64>,
    /// Instances of the component (from the components summary).
    pub instances: u64,
    /// Instances per model.
    pub models: BTreeMap<String, u64>,
}

/// One edit of the working graph. Node ids are the input component's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Edit {
    /// Remove one edge end `src -> dst`.
    Unlink { src: u32, dst: u32, end: EdgeEnd },
    /// Add one edge end `src -> dst`.
    Link { src: u32, dst: u32, end: EdgeEnd },
    /// Replace a node's label, folded operands and the rest (never its origin).
    Relabel { node: u32, value: Box<NNode> },
    /// Remove a node without edges.
    Remove { node: u32 },
}

/// One rewrite: the rule, the nodes it matched (rule-specific order, docs/passes/normalize.md §4) and
/// its edits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Step {
    pub rule: String,
    pub matched: Vec<u32>,
    pub edits: Vec<Edit>,
}

/// `ledger/<id>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LedgerFile {
    pub schema: String,
    pub id: String,
    pub steps: Vec<Step>,
}

/// What a query must answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Expect {
    /// A rule's obligation (or the positive control): the negated equivalence is unsatisfiable.
    Unsat,
    /// The negative control (float reassociation): a counterexample exists.
    Sat,
}

/// `rules/<rule>.result`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RuleResult {
    pub schema: String,
    pub rule: String,
    pub expect: Expect,
    /// blake3 of the `.smt2` file.
    pub query_hash: String,
    pub solver: Solver,
    pub version: String,
    pub verdict: Verdict,
}

pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "normalize_component.schema.json",
            serde_json::to_value(schemars::schema_for!(NormalizedComponent)).unwrap_or_default(),
        ),
        (
            "normalize_summary.schema.json",
            serde_json::to_value(schemars::schema_for!(NormalizedSummary)).unwrap_or_default(),
        ),
        (
            "normalize_ledger.schema.json",
            serde_json::to_value(schemars::schema_for!(LedgerFile)).unwrap_or_default(),
        ),
        (
            "normalize_rule_result.schema.json",
            serde_json::to_value(schemars::schema_for!(RuleResult)).unwrap_or_default(),
        ),
    ]
}
