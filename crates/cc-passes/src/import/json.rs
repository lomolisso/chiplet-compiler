//! The import artifacts: `out/import/<model>/<phase>/ir.postcard` (a `cc_ir::Module`) with its
//! `ir.postcard.summary.json` (`Summary`), and `out/import/op_inventory.json` (`Inventory`).

use std::collections::BTreeMap;

use cc_ir::ops::OpSpec;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const SUMMARY_SCHEMA: &str = "cc.import.summary/1";
pub const INVENTORY_SCHEMA: &str = "cc.import.op_inventory/1";

/// What inlining removed, per op name: the inlined `func.call`s, the inlined composites and the
/// `func.return`s of the inlined bodies (counted per executed call site, as the export census counts).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Ledger {
    pub calls: u64,
    pub composites: BTreeMap<String, u64>,
    pub returns: u64,
}

/// `ir.postcard.summary.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub schema: String,
    pub model: String,
    pub phase: String,
    pub config: String,
    /// Structural hash of the module (`cc_ir::hash`): order- and numbering-independent.
    pub ir_hash: String,
    pub symbols: Vec<String>,
    pub constraints: Vec<String>,
    pub funcs: u64,
    pub ops: u64,
    pub values: u64,
    /// Ops per class of the op table (the stored module, decompositions once).
    pub by_class: BTreeMap<String, u64>,
    /// Ops per name (the stored module).
    pub by_op: BTreeMap<String, u64>,
    /// Size-computation ops and their share of all non-structural ops (rounded to 1e-4).
    pub size_ops: u64,
    pub size_share: f64,
    /// Operand edges where a dimension value enters compute, per consuming op name.
    pub dim_as_data: BTreeMap<String, u64>,
    /// Component composites (`cc.*`) by name.
    pub composites: BTreeMap<String, u64>,
    /// Values whose type has a symbolic dimension.
    pub symbolic_values: u64,
    pub ledger: Ledger,
}

/// One row of `op_inventory.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InventoryRow {
    /// `null` when the op has no table entry (import then failed for that phase).
    pub entry: Option<serde_json::Value>,
    /// Occurrences in the stored modules, summed over phases.
    pub count: u64,
    pub models: Vec<String>,
}

/// `out/import/op_inventory.json`: every op seen across the model set (key `name` or `name@target`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    pub schema: String,
    pub ops: BTreeMap<String, InventoryRow>,
    pub unknown: Vec<String>,
}

/// An op table entry as data.
pub fn entry_json(e: &OpSpec) -> serde_json::Value {
    serde_json::to_value(e).unwrap_or_default()
}

/// `(file name, schema)` pairs committed under `docs/schemas/`.
pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "import_summary.schema.json",
            serde_json::to_value(schemars::schema_for!(Summary)).unwrap_or_default(),
        ),
        (
            "import_op_inventory.schema.json",
            serde_json::to_value(schemars::schema_for!(Inventory)).unwrap_or_default(),
        ),
        (
            "import_ir.schema.json",
            serde_json::to_value(schemars::schema_for!(cc_ir::Module)).unwrap_or_default(),
        ),
    ]
}
