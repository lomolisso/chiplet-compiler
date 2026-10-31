//! The export artifacts as Rust types: the contract between `python/cc_export` (writer) and the Rust
//! passes (readers: this pass's `verify`, the `import` pass). Every type is strict (`deny_unknown_fields`);
//! the JSON Schemas in `docs/schemas/export_*.schema.json` are generated from them.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const GRAPH_SCHEMA: &str = "cc.export.graph/1";
pub const SYMBOLS_SCHEMA: &str = "cc.export.symbols/1";
pub const CENSUS_SCHEMA: &str = "cc.export.census/1";
pub const MODEL_SCHEMA: &str = "cc.export.model/1";

/// A value type. `shape` entries are `null` for dynamic (`?`) dims; their symbolic expressions follow from
/// the dimension-variable arguments (`jax.global_constant`) and the explicit size computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Type {
    Tensor { shape: Vec<Option<u64>>, elem: String },
    Function { inputs: Vec<Type>, results: Vec<Type> },
    Other { value: String },
}

/// A float attribute value; non-finite values are strings (`"inf"`, `"-inf"`, `"nan"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum FloatValue {
    Num(f64),
    Special(String),
}

/// An integer attribute value (unsigned 64-bit values above `i64::MAX` exist for `ui64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum IntValue {
    Signed(i64),
    Unsigned(u64),
}

/// A typed MLIR attribute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Attr {
    Int {
        value: IntValue,
        #[serde(rename = "type")]
        ty: String,
    },
    Float {
        value: FloatValue,
        #[serde(rename = "type")]
        ty: String,
    },
    Bool {
        value: bool,
    },
    Str {
        value: String,
    },
    Symbol {
        value: String,
    },
    Ints {
        value: Vec<i64>,
    },
    Bools {
        value: Vec<bool>,
    },
    /// Dense elements: the values when at most 1024 elements numpy can view, else a content digest.
    Dense {
        #[serde(rename = "type")]
        ty: Type,
        splat: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        values: Option<Vec<serde_json::Value>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        digest: Option<String>,
    },
    Array {
        value: Vec<Attr>,
    },
    Dict {
        value: BTreeMap<String, Attr>,
    },
    Type {
        value: Type,
    },
    Unit,
    /// A StableHLO enum attribute (`comparison_direction`, `precision`, ...).
    Enum {
        #[serde(rename = "enum")]
        enum_kind: String,
        value: String,
    },
    DotDimensionNumbers {
        lhs_batching_dimensions: Vec<i64>,
        rhs_batching_dimensions: Vec<i64>,
        lhs_contracting_dimensions: Vec<i64>,
        rhs_contracting_dimensions: Vec<i64>,
    },
    ConvDimensionNumbers {
        input_batch_dimension: i64,
        input_feature_dimension: i64,
        input_spatial_dimensions: Vec<i64>,
        kernel_input_feature_dimension: i64,
        kernel_output_feature_dimension: i64,
        kernel_spatial_dimensions: Vec<i64>,
        output_batch_dimension: i64,
        output_feature_dimension: i64,
        output_spatial_dimensions: Vec<i64>,
    },
    GatherDimensionNumbers {
        offset_dims: Vec<i64>,
        collapsed_slice_dims: Vec<i64>,
        operand_batching_dims: Vec<i64>,
        start_indices_batching_dims: Vec<i64>,
        start_index_map: Vec<i64>,
        index_vector_dim: i64,
    },
    ScatterDimensionNumbers {
        update_window_dims: Vec<i64>,
        inserted_window_dims: Vec<i64>,
        input_batching_dims: Vec<i64>,
        scatter_indices_batching_dims: Vec<i64>,
        scattered_dims_to_operand_dims: Vec<i64>,
        index_vector_dim: i64,
    },
    /// No structured form: the printed attribute.
    Opaque {
        value: String,
    },
}

impl Attr {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Attr::Str { value } | Attr::Symbol { value } => Some(value),
            _ => None,
        }
    }
}

pub type Attrs = BTreeMap<String, Attr>;

/// A defined value: a block argument or an op result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Value {
    pub v: u64,
    #[serde(rename = "type")]
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub id: u64,
    pub args: Vec<Value>,
    pub ops: Vec<Op>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Region {
    pub blocks: Vec<Block>,
}

/// One op. `operands` are value ids; an operand's port is its position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Op {
    pub id: u64,
    pub name: String,
    pub operands: Vec<u64>,
    pub results: Vec<Value>,
    pub attributes: Attrs,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<Region>,
    /// Source location (provenance; repository-relative paths).
    pub loc: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FuncType {
    pub inputs: Vec<Type>,
    pub results: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Func {
    pub name: String,
    pub visibility: String,
    #[serde(rename = "type")]
    pub ty: FuncType,
    /// Per argument; dimension variables carry `jax.global_constant = "<symbol>"`.
    pub arg_attrs: Vec<Attrs>,
    pub res_attrs: Vec<Attrs>,
    pub attributes: Attrs,
    pub body: Region,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Module {
    pub attributes: Attrs,
    pub funcs: Vec<Func>,
    /// Module-level ops other than `func.func` (none expected).
    pub other_ops: Vec<Op>,
}

/// A dimension of an input: concrete, or a symbolic expression over the model's symbols.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Dim {
    Const(u64),
    Sym(String),
}

/// One flattened argument of the phase function `fn(params, *inputs)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub flat_index: u64,
    /// Pytree key path (`[0]['params']['Dense_0']['kernel']` for parameters, `[1]` ... for inputs).
    pub path: String,
    /// `param` or `input`.
    pub kind: String,
    /// Index among `main`'s arguments; `null` when the export dropped the unused argument.
    pub main_arg: Option<u64>,
    pub shape: Vec<Dim>,
    pub dtype: String,
}

/// One composite call in the expansion from the entry function (a component instance).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompositeInstance {
    pub instance: u64,
    /// Id of the `stablehlo.composite` op.
    pub op: u64,
    /// Function holding the op.
    pub func: String,
    /// `cc.<framework>.<Class>[.<method>]`.
    pub name: String,
    /// Module path (provenance).
    pub path: String,
    pub decomposition: String,
    pub version: i64,
    /// Enclosing instance (`null` at the top).
    pub parent: Option<u64>,
    /// No composite inside.
    pub leaf: bool,
    /// Non-structural ops whose innermost enclosing composite is this one.
    pub own_ops: u64,
}

/// Fractions of the executed non-structural ops (call sites expanded) inside a composite / a leaf one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Coverage {
    pub ops: u64,
    pub in_composite: u64,
    pub in_leaf_composite: u64,
    /// Ops outside every composite (flagged `anon`; the `components` pass groups them into `anon:<hash>` components).
    pub anon: u64,
    pub any: f64,
    pub leaf: f64,
    pub instances: u64,
    pub leaf_instances: u64,
    pub by_name: BTreeMap<String, u64>,
    pub anon_by_op: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InterceptionFailure {
    pub name: String,
    pub path: String,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Interception {
    pub wrapped: u64,
    /// Values of the enclosing trace turned into composite operands (closure conversion).
    pub hoisted: u64,
    pub excluded: BTreeMap<String, u64>,
    pub failures: Vec<InterceptionFailure>,
}

/// `out/export/<model>/<phase>/graph.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GraphFile {
    pub schema: String,
    pub model: String,
    pub phase: String,
    pub config: String,
    pub entry: String,
    pub module: Module,
    pub inputs: Vec<Input>,
    pub composites: Vec<CompositeInstance>,
    pub coverage: Coverage,
    pub interception: Interception,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SymbolSite {
    pub path: String,
    pub axis: u64,
    pub expr: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConcreteAxis {
    pub path: String,
    pub axis: u64,
    pub size: u64,
}

/// A declared symbol: the config field holding its value, that value, and the perturbed `alt`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SymbolDecl {
    pub field: String,
    pub value: serde_json::Value,
    /// A number (a size, or e.g. a grid resolution the sizes derive from).
    pub alt: serde_json::Value,
    pub doc: String,
    /// Where the symbol appears among the phase's inputs (`symbols.json` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<Vec<SymbolSite>>,
}

/// `out/export/<model>/<phase>/symbols.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SymbolsFile {
    pub schema: String,
    pub model: String,
    pub phase: String,
    /// `Phase.shapes` as declared by the port (a pytree prefix of the inputs).
    pub shapes: serde_json::Value,
    pub symbols: BTreeMap<String, SymbolDecl>,
    /// The constraints of the symbolic scope (the sizes the hardware is valid for).
    pub constraints: Vec<String>,
    /// Every input axis left concrete (architecture dims).
    pub concrete_input_axes: Vec<ConcreteAxis>,
    pub trip_count: u64,
    pub trip_count_expr: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SigSummary {
    pub count: u64,
    /// `null` when a dim is symbolic.
    pub flops: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoreOp {
    /// `dot_general` or `convolution` (`dynamic_conv` counts as a convolution).
    pub kind: String,
    pub op: String,
    /// `<lhs elem>x<rhs elem>-><result elem>` (`verify/hlo.py`).
    pub sig: String,
    pub flops: Option<u64>,
    pub lhs_shape: Vec<Option<u64>>,
    pub rhs_shape: Vec<Option<u64>>,
    pub out_shape: Vec<Option<u64>>,
}

/// `out/export/<model>/<phase>/census.json`: per executed op (calls and composites expanded).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CensusFile {
    pub schema: String,
    pub model: String,
    pub phase: String,
    /// `"<kind> <sig>"` → count and FLOPs.
    pub summary: BTreeMap<String, SigSummary>,
    pub ops: Vec<CoreOp>,
    /// Executed op-name multiset.
    pub executed: BTreeMap<String, u64>,
}

/// Allowed signatures: `{signature: why}` or a plain list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Allow {
    Explained(BTreeMap<String, String>),
    List(Vec<String>),
}

impl Allow {
    pub fn contains(&self, sig: &str) -> bool {
        match self {
            Allow::Explained(m) => m.contains_key(sig),
            Allow::List(v) => v.iter().any(|s| s == sig),
        }
    }
}

/// A recipe's audit (`models/<key>/recipe.py`), as data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Audit {
    pub allow: Allow,
    #[serde(default)]
    pub require: Vec<String>,
    #[serde(default)]
    pub phases: BTreeMap<String, PhaseAudit>,
    /// `counts[config][phase][signature]` = exact op count.
    #[serde(default)]
    pub counts: BTreeMap<String, BTreeMap<String, BTreeMap<String, u64>>>,
    /// Parameter storage dtypes that must occur.
    #[serde(default)]
    pub params: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhaseAudit {
    #[serde(default)]
    pub allow: Option<Allow>,
    #[serde(default)]
    pub require: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecipeFacts {
    pub audit: Option<Audit>,
    pub param_dtypes: Vec<String>,
}

/// One phase in `export.json`. When the export failed, only `trip_count`, `size_variant_of`, `error` and
/// `traceback` are present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhaseRecord {
    pub trip_count: u64,
    pub size_variant_of: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub traceback: Option<String>,
    /// Canonical hash of the module JSON (provenance excluded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    #[serde(default)]
    pub roots: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<Coverage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub census: Option<BTreeMap<String, SigSummary>>,
    /// The exporter's own checks: `{name: {"ok": bool, ...evidence}}`.
    #[serde(default)]
    pub checks: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub files: BTreeMap<String, u64>,
}

/// `out/export/<model>/export.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelFile {
    pub schema: String,
    pub model: String,
    pub config: String,
    pub versions: BTreeMap<String, String>,
    pub symbols: BTreeMap<String, SymbolDecl>,
    pub constraints: Vec<String>,
    pub exclude: Vec<String>,
    pub phases: BTreeMap<String, PhaseRecord>,
    pub checks: BTreeMap<String, serde_json::Value>,
    pub recipe: RecipeFacts,
}

/// `(file name, schema)` pairs committed under `docs/schemas/`.
pub fn schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "export_graph.schema.json",
            serde_json::to_value(schemars::schema_for!(GraphFile)).unwrap_or_default(),
        ),
        (
            "export_symbols.schema.json",
            serde_json::to_value(schemars::schema_for!(SymbolsFile)).unwrap_or_default(),
        ),
        (
            "export_census.schema.json",
            serde_json::to_value(schemars::schema_for!(CensusFile)).unwrap_or_default(),
        ),
        (
            "export_model.schema.json",
            serde_json::to_value(schemars::schema_for!(ModelFile)).unwrap_or_default(),
        ),
    ]
}
