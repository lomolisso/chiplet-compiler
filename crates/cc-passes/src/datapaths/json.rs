//! The datapaths artifacts (`out/datapaths/`):
//!
//! - `datapaths/<z>.json` (`Datapath`): the scalar datapath of one iteration of class `z`: vertices
//!   `op@family` with width variables and ports, its inputs and outputs, the accumulator, the width tuples;
//! - `proofs/<z>/<t>.smt2` and `.result` (`ProofResult`): datapath = kernel semantics for tuple `t`;
//! - `proofs/<z>/<t>.order.smt2` and `.result`: whether the combiner is associative and commutative;
//! - `lemmas/<format>.<what>.smt2` and `.result`: the float encodings the proofs use;
//! - `summary.json` (`Summary`).

use std::collections::BTreeMap;

use cc_smt::{Solver, Verdict};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::iteration_space::json::{DimOrder, Route, SkelClass};

pub const DATAPATH_SCHEMA: &str = "cc.datapaths.datapath/1";
pub const RESULT_SCHEMA: &str = "cc.datapaths.proof_result/1";
pub const SUMMARY_SCHEMA: &str = "cc.datapaths.summary/1";

/// Where a vertex sits: once per element of a core operand (prologue of port `port`), once per iteration
/// (core: the multiply and the combiner), once per result element (epilogue), or a map kernel's body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Prologue { port: u32 },
    Core,
    Epilogue,
    Map,
}

/// A value a vertex reads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Src {
    /// Result `res` of vertex `v`.
    Vertex { v: u32, res: u32 },
    /// An input (by name).
    Input { name: String },
    /// The current value of accumulator `index` (the combiner's left operand).
    Acc { index: u32 },
    /// The final value of core result `index` (what the epilogue reads).
    Result { index: u32 },
}

/// An operand of a vertex: its source and port (`p<i>`, or `*` for a commutative op).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Arg {
    pub src: Src,
    pub port: String,
}

/// The kernel node a vertex was extracted from, and the access instance (an index into the skeleton's
/// `accesses`; `None` for region bodies and the core's own vertices).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct From {
    pub node: u32,
    pub access: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Vertex {
    pub id: u32,
    /// `<op>@<family>` (`add@sint`, `convert@float`, `const@sint`).
    pub label: String,
    /// The StableHLO op without its prefix, or `const`.
    pub op: String,
    pub family: String,
    /// The result's width variable (`w<e>`, `e` its element class).
    pub width: String,
    /// Literal attributes the semantics reads: `comparison_direction` (GT/GE are written as LT/LE with the
    /// operands exchanged), `compare_type`, `exponent_bits`, `mantissa_bits`; a constant's `column` (the
    /// instance table's value column) or `value` (a literal).
    pub attrs: BTreeMap<String, String>,
    pub stage: Stage,
    pub args: Vec<Arg>,
    pub from: From,
}

/// What an input is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InputKind {
    /// Read by a prologue.
    Prologue,
    /// A core operand read directly.
    CoreOperand,
    /// An outside value read by the epilogue (a bias, a scale, a residual).
    EpilogueParam,
    /// Read by a map kernel.
    Map,
    /// Data-dependent indices: address generation, not compute (no vertex reads it).
    Address,
    /// A template's core result (the epilogue's input).
    CoreResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Input {
    /// The skeleton operand's name (`in<j>`, `in<j>.<u>`, `indices`), or `core_result<i>`.
    pub name: String,
    pub kind: InputKind,
    /// The kernel node and result whose value it is.
    pub node: u32,
    pub res: u32,
    /// The iteration dims that index it.
    pub incidence: Vec<u32>,
    pub width: String,
}

/// What an output is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    /// A kernel output (`out<j>`).
    Output,
    /// A template's core operand (the end of a prologue stage).
    CoreOperand,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Output {
    /// `out<j>`, or the core port's name (`lhs`, `rhs`).
    pub name: String,
    pub kind: OutputKind,
    pub src: Src,
    pub width: String,
}

/// One accumulator: its width, its init value and its next value (one combiner step).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AccEntry {
    pub width: String,
    pub init: Src,
    pub next: Src,
}

/// Whether the combiner is associative and commutative at one tuple (proof `order_file`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OrderFree {
    pub tuple: String,
    /// `true`: any order gives the same bits (ω is free); `false`: ω binds (a counterexample is in the
    /// result file).
    pub free: bool,
}

/// The accumulator vertex: a tuple of accumulators updated by one combiner (variadic reductions have
/// several).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Accumulator {
    pub entries: Vec<AccEntry>,
    /// The combiner's vertices.
    pub combiner: Vec<u32>,
    pub omega: Option<DimOrder>,
    pub order: Vec<OrderFree>,
}

/// A width tuple: the element type of every element class, and the structural attributes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct Tuple {
    /// `t<k>`, by the sorted order of the tuples.
    pub name: String,
    /// Per element class, the MLIR element type (`i8`, `bf16`, `f8E4M3FN`).
    pub elems: Vec<String>,
    /// Per folded constant (value column), its element type.
    pub consts: BTreeMap<String, String>,
    /// Attributes that change an op's semantics (`n<i>.exponent_bits`, …).
    pub attrs: BTreeMap<String, i64>,
    /// Instance rows with this tuple.
    pub rows: u64,
}

/// `datapaths/<z>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Datapath {
    pub schema: String,
    pub id: String,
    pub route: Route,
    pub class: SkelClass,
    pub core: String,
    /// Per element class used: its width variable and family.
    pub widths: BTreeMap<String, String>,
    pub vertices: Vec<Vertex>,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub accumulator: Option<Accumulator>,
    pub tuples: Vec<Tuple>,
}

/// `proofs/<z>/<file>.result` and `lemmas/<file>.result`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProofResult {
    pub schema: String,
    pub file: String,
    pub query_hash: String,
    pub solver: Solver,
    pub version: String,
    pub verdict: Verdict,
    /// What the verdict must be (`unsat`), or `either` for an order query.
    pub expect: String,
    /// The reason when the class is in `allow_unknown`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed: Option<String>,
}

/// `summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub schema: String,
    pub classes: u64,
    pub by_route: BTreeMap<String, u64>,
    pub vertices: u64,
    /// Vertices by `op@family` (the census `mine` works on).
    pub labels: BTreeMap<String, u64>,
    pub tuples: u64,
    pub proofs: u64,
    pub order_queries: u64,
    /// Order queries: combiners proven free / bound by ω.
    pub order_free: u64,
    pub order_bound: u64,
    pub lemmas: u64,
    pub allowed_unknown: u64,
    /// Uninterpreted functions used (`exponential_bf16`, …).
    pub uninterpreted: Vec<String>,
}

pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "datapaths_datapath.schema.json",
            serde_json::to_value(schemars::schema_for!(Datapath)).unwrap_or_default(),
        ),
        (
            "datapaths_proof_result.schema.json",
            serde_json::to_value(schemars::schema_for!(ProofResult)).unwrap_or_default(),
        ),
        (
            "datapaths_summary.schema.json",
            serde_json::to_value(schemars::schema_for!(Summary)).unwrap_or_default(),
        ),
    ]
}
