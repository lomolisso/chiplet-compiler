//! The softfloat artifacts (`out/softfloat/`):
//!
//! - `ops/<id>/`: one primitive: `kernel.py` (the Allo function), `kernel.mlir` (after inlining),
//!   `xls/test_block.cpp` (Allo's XLS C++), `fu.ir` (xlscc), `fu.opt.ir` (opt_main), `rtl.v` and
//!   `rtl.sig.textproto` (codegen_main), `vectors.npz` (operands, reference, Allo `llvm` and XLS IR
//!   results), `result.json` (`BuildResult`), and, for exactly specified ops, `proof.smt2` with
//!   `proof.result` (`datapaths::json::ProofResult`);
//! - `coverage.json` (`CoverageFile`), `summary.json` (`Summary`).

use std::collections::BTreeMap;

use cc_smt::Verdict;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const COVERAGE_SCHEMA: &str = "cc.softfloat.coverage/1";
pub const SUMMARY_SCHEMA: &str = "cc.softfloat.summary/1";

/// How a primitive is known to compute its op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    /// Its optimized XLS IR is proven equal to the op's semantics (`cc_smt::ops`) for every input.
    Proven,
    /// Every input was evaluated on the optimized XLS IR and on Allo's `llvm` target and matched the
    /// reference (no proof: it ran out of time).
    Exhaustive,
    /// Edge cases and random inputs matched (no proof: it ran out of time).
    Tested,
}

/// One mismatch example from the build (bit patterns).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Example {
    pub args: Vec<u64>,
    pub want: u64,
    pub got: u64,
}

/// Reference comparison of the Allo `llvm` results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RefCheck {
    pub mismatches: u64,
    pub examples: Vec<Example>,
}

/// `ops/<id>/result.json`, written by `python -m cc_softfloat build`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BuildResult {
    pub key: String,
    pub id: String,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default)]
    pub vectors: u64,
    #[serde(default)]
    pub exhaustive: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<RefCheck>,
    /// Vectors whose XLS IR result differs from the `llvm` result.
    #[serde(default)]
    pub ir_vs_llvm: u64,
    #[serde(default)]
    pub ir_nodes: u64,
    #[serde(default)]
    pub seconds: BTreeMap<String, f64>,
}

/// The proof of one primitive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ProofRef {
    /// `ops/<id>/proof.smt2`.
    pub file: String,
    pub verdict: Verdict,
    /// The significand product is an uninterpreted function (the multiply proofs, docs §4).
    pub product_abstracted: bool,
}

/// One primitive of the library.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// The `characterize` key, e.g. `add@f32`, `convert@i8>f32`.
    pub key: String,
    /// Directory, Verilog module and Allo function name.
    pub id: String,
    pub op: String,
    pub types: Vec<String>,
    pub evidence: Evidence,
    pub vectors: u64,
    pub exhaustive: bool,
    pub ir_nodes: u64,
    pub proof: Option<ProofRef>,
}

/// `coverage.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoverageFile {
    pub schema: String,
    pub entries: Vec<Entry>,
    /// Catalog keys of ops whose accuracy is implementation-defined (exp, log, tanh, ...), not built here.
    pub approximate: Vec<String>,
}

/// `summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub schema: String,
    pub primitives: u64,
    pub proven: u64,
    pub exhaustive: u64,
    pub tested: u64,
    pub approximate: u64,
    pub vectors: u64,
    /// blake3 of `python/cc_softfloat` (`.py`, `.h`): `allo_emit` keys its cache on it.
    pub library_hash: String,
    pub image: String,
    pub solver: String,
}

pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "softfloat_coverage.schema.json",
            serde_json::to_value(schemars::schema_for!(CoverageFile)).unwrap_or_default(),
        ),
        (
            "softfloat_summary.schema.json",
            serde_json::to_value(schemars::schema_for!(Summary)).unwrap_or_default(),
        ),
        (
            "softfloat_result.schema.json",
            serde_json::to_value(schemars::schema_for!(BuildResult)).unwrap_or_default(),
        ),
    ]
}
