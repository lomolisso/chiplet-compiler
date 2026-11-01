//! The op-semantics table: one entry per op that occurs in the exported modules, encoding the StableHLO
//! specification only (https://openxla.org/stablehlo/spec, anchor `#<op>`): arity, commutativity,
//! element-wise or not, a class, the operands that carry shapes, the result element-type rule and the
//! result shape rule. It is data: passes look ops up here and never branch on op names for semantics.
//!
//! An op that is not in the table is an error (`lookup` returns `None`; the caller names model, phase,
//! function and op). `custom_call` entries are keyed by `call_target_name`.

pub mod dims;
pub mod shape;

use std::collections::HashMap;
use std::sync::OnceLock;

use schemars::JsonSchema;
use serde::Serialize;

pub use dims::{DimsFn, DimsIn, Rel, Side, Slot};
pub use shape::{ShapeFn, ShapeIn};

/// Op classes (design §4.5). `Constant` holds the ops that create values from attributes alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OpClass {
    /// A contraction or reduction: `dot_general`, `convolution`, `reduce`, `reduce_window`.
    Core,
    /// Element-wise arithmetic, logic, comparison and conversion.
    Elementwise,
    /// Index remapping without data-dependent addresses: reshape, transpose, broadcast, slice, pad, …
    Layout,
    /// Data-dependent addressing: gather, scatter, dynamic slices.
    DataMovement,
    /// Size computation by definition (`get_dimension_size`, shape assertions).
    Size,
    /// Structure: returns, calls, composites.
    Control,
    /// Outside the affine model and not element-wise: `sort`, `fft`.
    Irregular,
    /// Values from attributes: `constant`, `iota`.
    Constant,
}

/// How many operands (or results) an op takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Arity {
    Exactly(u32),
    AtLeast(u32),
}

impl Arity {
    pub fn admits(self, n: usize) -> bool {
        match self {
            Arity::Exactly(k) => n == k as usize,
            Arity::AtLeast(k) => n >= k as usize,
        }
    }
}

/// Operand ports that carry shapes (their values are size computation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShapePorts {
    None,
    /// These ports.
    Ports(&'static [u32]),
    /// The last `n` ports.
    Last(u32),
    /// Every port (`shape_assertion`).
    All,
    /// The ports bound to dimension-variable arguments of the callee (composites, calls).
    DimVarArgs,
}

impl ShapePorts {
    /// Whether port `p` of an op with `n` operands carries a shape (`DimVarArgs` is resolved by the caller).
    pub fn contains(self, p: usize, n: usize) -> bool {
        match self {
            ShapePorts::None | ShapePorts::DimVarArgs => false,
            ShapePorts::Ports(ps) => ps.contains(&(p as u32)),
            ShapePorts::Last(k) => p + k as usize >= n && p < n,
            ShapePorts::All => true,
        }
    }
}

/// The result element type in terms of the operands (spec constraints on `element_type(result)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ElemRule {
    /// Every result has the element type of this operand.
    SameAs(u32),
    /// `i1`.
    Bool,
    /// `i32` (`get_dimension_size`).
    I32,
    /// The component type of operand 0 (`real`, `imag`).
    ComponentOf0,
    /// `complex<element_type(operand 0)>` (`complex`).
    ComplexOf0,
    /// Unconstrained by the spec (conversions, `dot_general`'s preferred type, …) or given by the callee.
    Free,
}

/// Iteration-space semantics of a core op: the hook the iteration-space analysis fills in.
pub type IterationFn = fn(&ShapeIn<'_>) -> Option<serde_json::Value>;

/// One table entry.
#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
pub struct OpSpec {
    /// The MLIR op name, e.g. `stablehlo.add`.
    pub name: &'static str,
    /// For `stablehlo.custom_call`: the `call_target_name`.
    pub target: Option<&'static str>,
    /// Section anchor in the StableHLO spec (or the MLIR `func` dialect doc for `func.*`).
    pub spec: &'static str,
    pub class: OpClass,
    pub operands: Arity,
    pub results: Arity,
    pub elementwise: bool,
    /// Operands commute (spec: the op is commutative for every element type the op admits).
    pub commutative: bool,
    /// Number of regions.
    pub regions: u32,
    pub shape_operands: ShapePorts,
    pub elem: ElemRule,
    #[serde(skip)]
    #[schemars(skip)]
    pub shape: ShapeFn,
    #[serde(skip)]
    #[schemars(skip)]
    pub iteration: Option<IterationFn>,
    /// Dimension and element-type equalities between operands and results (`dims.rs`).
    #[serde(skip)]
    #[schemars(skip)]
    pub dims: DimsFn,
    /// Attributes whose integer entries are sizes, offsets, strides or counts along axes (as opposed to
    /// axis numbers): the components pass makes them instance parameters.
    #[serde(skip)]
    #[schemars(skip)]
    pub size_attrs: &'static [&'static str],
}

impl OpSpec {
    /// The table key: `name` or `name@target`.
    pub fn key(&self) -> String {
        match self.target {
            Some(t) => format!("{}@{}", self.name, t),
            None => self.name.to_string(),
        }
    }
}

use Arity::{AtLeast, Exactly};
use ElemRule as E;
use OpClass as C;
use shape as s;

const fn entry(
    name: &'static str,
    spec: &'static str,
    class: OpClass,
    operands: Arity,
    results: Arity,
    elem: ElemRule,
    shape: ShapeFn,
) -> OpSpec {
    OpSpec {
        name,
        target: None,
        spec,
        class,
        operands,
        results,
        elementwise: false,
        commutative: false,
        regions: 0,
        shape_operands: ShapePorts::None,
        elem,
        shape,
        iteration: None,
        dims: dims::fresh,
        size_attrs: &[],
    }
}

/// An element-wise op of `n` operands (spec: `type(result) = type(operands)` up to the element type).
const fn ew(name: &'static str, spec: &'static str, n: u32, elem: ElemRule, commutative: bool) -> OpSpec {
    let mut e = entry(
        name,
        spec,
        C::Elementwise,
        Exactly(n),
        Exactly(1),
        elem,
        s::elementwise,
    );
    e.elementwise = true;
    e.commutative = commutative;
    e.dims = dims::elementwise;
    e
}

const fn with_shape_ports(mut e: OpSpec, p: ShapePorts) -> OpSpec {
    e.shape_operands = p;
    e
}

const fn with_regions(mut e: OpSpec, n: u32) -> OpSpec {
    e.regions = n;
    e
}

const fn with_dims(mut e: OpSpec, d: DimsFn) -> OpSpec {
    e.dims = d;
    e
}

const fn with_size_attrs(mut e: OpSpec, a: &'static [&'static str]) -> OpSpec {
    e.size_attrs = a;
    e
}

const fn custom(target: &'static str, mut e: OpSpec) -> OpSpec {
    e.target = Some(target);
    e
}

/// The table.
pub static TABLE: &[OpSpec] = &[
    // Element-wise (spec: each op's C1, identical operand and result shapes).
    ew("stablehlo.abs", "abs", 1, E::Free, false),
    ew("stablehlo.add", "add", 2, E::SameAs(0), true),
    ew("stablehlo.and", "and", 2, E::SameAs(0), true),
    ew("stablehlo.ceil", "ceil", 1, E::SameAs(0), false),
    ew("stablehlo.compare", "compare", 2, E::Bool, false),
    ew("stablehlo.complex", "complex", 2, E::ComplexOf0, false),
    ew("stablehlo.convert", "convert", 1, E::Free, false),
    ew("stablehlo.cosine", "cosine", 1, E::SameAs(0), false),
    ew("stablehlo.divide", "divide", 2, E::SameAs(0), false),
    ew("stablehlo.exponential", "exponential", 1, E::SameAs(0), false),
    ew("stablehlo.floor", "floor", 1, E::SameAs(0), false),
    ew("stablehlo.imag", "imag", 1, E::ComponentOf0, false),
    ew("stablehlo.is_finite", "is_finite", 1, E::Bool, false),
    ew("stablehlo.log", "log", 1, E::SameAs(0), false),
    ew("stablehlo.log_plus_one", "log_plus_one", 1, E::SameAs(0), false),
    ew("stablehlo.maximum", "maximum", 2, E::SameAs(0), true),
    ew("stablehlo.minimum", "minimum", 2, E::SameAs(0), true),
    ew("stablehlo.multiply", "multiply", 2, E::SameAs(0), true),
    ew("stablehlo.negate", "negate", 1, E::SameAs(0), false),
    ew("stablehlo.not", "not", 1, E::SameAs(0), false),
    ew("stablehlo.or", "or", 2, E::SameAs(0), true),
    ew("stablehlo.power", "power", 2, E::SameAs(0), false),
    ew("stablehlo.real", "real", 1, E::ComponentOf0, false),
    ew(
        "stablehlo.reduce_precision",
        "reduce_precision",
        1,
        E::SameAs(0),
        false,
    ),
    ew("stablehlo.remainder", "remainder", 2, E::SameAs(0), false),
    ew(
        "stablehlo.round_nearest_even",
        "round_nearest_even",
        1,
        E::SameAs(0),
        false,
    ),
    ew("stablehlo.rsqrt", "rsqrt", 1, E::SameAs(0), false),
    ew("stablehlo.sign", "sign", 1, E::SameAs(0), false),
    ew("stablehlo.sine", "sine", 1, E::SameAs(0), false),
    ew("stablehlo.sqrt", "sqrt", 1, E::SameAs(0), false),
    ew("stablehlo.subtract", "subtract", 2, E::SameAs(0), false),
    ew("stablehlo.tanh", "tanh", 1, E::SameAs(0), false),
    ew("stablehlo.xor", "xor", 2, E::SameAs(0), true),
    // `select(pred, on_true, on_false)`: `pred` may be rank 0 (C2); `clamp(min, operand, max)`: `min`
    // and `max` may be rank 0 (C1).
    ew("stablehlo.select", "select", 3, E::SameAs(1), false),
    ew("stablehlo.clamp", "clamp", 3, E::SameAs(1), false),
    // Values from attributes.
    entry(
        "stablehlo.constant",
        "constant",
        C::Constant,
        Exactly(0),
        Exactly(1),
        E::Free,
        s::declared,
    ),
    entry(
        "stablehlo.iota",
        "iota",
        C::Constant,
        Exactly(0),
        Exactly(1),
        E::Free,
        s::declared,
    ),
    with_shape_ports(
        entry(
            "stablehlo.dynamic_iota",
            "dynamic_iota",
            C::Constant,
            Exactly(1),
            Exactly(1),
            E::Free,
            s::dynamic_iota,
        ),
        ShapePorts::Ports(&[0]),
    ),
    // Layout.
    with_dims(
        entry(
            "stablehlo.broadcast_in_dim",
            "broadcast_in_dim",
            C::Layout,
            Exactly(1),
            Exactly(1),
            E::SameAs(0),
            s::broadcast_in_dim,
        ),
        dims::broadcast_in_dim,
    ),
    with_dims(
        with_shape_ports(
            entry(
                "stablehlo.dynamic_broadcast_in_dim",
                "dynamic_broadcast_in_dim",
                C::Layout,
                Exactly(2),
                Exactly(1),
                E::SameAs(0),
                s::dynamic_broadcast_in_dim,
            ),
            ShapePorts::Ports(&[1]),
        ),
        dims::broadcast_in_dim,
    ),
    entry(
        "stablehlo.reshape",
        "reshape",
        C::Layout,
        Exactly(1),
        Exactly(1),
        E::SameAs(0),
        s::reshape,
    ),
    with_shape_ports(
        entry(
            "stablehlo.dynamic_reshape",
            "dynamic_reshape",
            C::Layout,
            Exactly(2),
            Exactly(1),
            E::SameAs(0),
            s::dynamic_reshape,
        ),
        ShapePorts::Ports(&[1]),
    ),
    with_dims(
        entry(
            "stablehlo.transpose",
            "transpose",
            C::Layout,
            Exactly(1),
            Exactly(1),
            E::SameAs(0),
            s::transpose,
        ),
        dims::transpose,
    ),
    with_dims(
        entry(
            "stablehlo.reverse",
            "reverse",
            C::Layout,
            Exactly(1),
            Exactly(1),
            E::SameAs(0),
            s::same0,
        ),
        dims::same0,
    ),
    with_size_attrs(
        entry(
            "stablehlo.slice",
            "slice",
            C::Layout,
            Exactly(1),
            Exactly(1),
            E::SameAs(0),
            s::slice,
        ),
        &["start_indices", "limit_indices", "strides"],
    ),
    with_shape_ports(
        entry(
            "stablehlo.real_dynamic_slice",
            "real_dynamic_slice",
            C::Layout,
            Exactly(4),
            Exactly(1),
            E::SameAs(0),
            s::real_dynamic_slice,
        ),
        ShapePorts::Ports(&[1, 2, 3]),
    ),
    with_dims(
        entry(
            "stablehlo.concatenate",
            "concatenate",
            C::Layout,
            AtLeast(1),
            Exactly(1),
            E::SameAs(0),
            s::concatenate,
        ),
        dims::concatenate,
    ),
    with_size_attrs(
        with_dims(
            entry(
                "stablehlo.pad",
                "pad",
                C::Layout,
                Exactly(2),
                Exactly(1),
                E::SameAs(0),
                s::pad,
            ),
            dims::pad,
        ),
        &["edge_padding_low", "edge_padding_high", "interior_padding"],
    ),
    with_dims(
        with_shape_ports(
            entry(
                "stablehlo.dynamic_pad",
                "dynamic_pad",
                C::Layout,
                Exactly(5),
                Exactly(1),
                E::SameAs(0),
                s::dynamic_pad,
            ),
            ShapePorts::Ports(&[2, 3, 4]),
        ),
        dims::pad,
    ),
    // Data movement (data-dependent addresses).
    with_size_attrs(
        entry(
            "stablehlo.gather",
            "gather",
            C::DataMovement,
            Exactly(2),
            Exactly(1),
            E::SameAs(0),
            s::gather,
        ),
        &["slice_sizes"],
    ),
    with_shape_ports(
        entry(
            "stablehlo.dynamic_gather",
            "dynamic_gather",
            C::DataMovement,
            Exactly(3),
            Exactly(1),
            E::SameAs(0),
            s::dynamic_gather,
        ),
        ShapePorts::Ports(&[2]),
    ),
    with_size_attrs(
        entry(
            "stablehlo.dynamic_slice",
            "dynamic_slice",
            C::DataMovement,
            AtLeast(1),
            Exactly(1),
            E::SameAs(0),
            s::dynamic_slice,
        ),
        &["slice_sizes"],
    ),
    with_dims(
        entry(
            "stablehlo.dynamic_update_slice",
            "dynamic_update_slice",
            C::DataMovement,
            AtLeast(2),
            Exactly(1),
            E::SameAs(0),
            s::same0,
        ),
        dims::dynamic_update_slice,
    ),
    // `scatter(inputs…, scatter_indices, updates…)` with an update computation; whether it is a
    // scatter-reduce depends on that body (design §4.5), not on the op.
    with_dims(
        with_regions(
            entry(
                "stablehlo.scatter",
                "scatter",
                C::DataMovement,
                AtLeast(3),
                AtLeast(1),
                E::SameAs(0),
                s::scatter,
            ),
            1,
        ),
        dims::scatter,
    ),
    // Cores.
    with_dims(
        entry(
            "stablehlo.dot_general",
            "dot_general",
            C::Core,
            Exactly(2),
            Exactly(1),
            E::Free,
            s::dot_general,
        ),
        dims::dot_general,
    ),
    with_size_attrs(
        with_dims(
            entry(
                "stablehlo.convolution",
                "convolution",
                C::Core,
                Exactly(2),
                Exactly(1),
                E::Free,
                s::convolution,
            ),
            dims::convolution,
        ),
        &[
            "window_strides",
            "padding",
            "lhs_dilation",
            "rhs_dilation",
            "feature_group_count",
            "batch_group_count",
        ],
    ),
    with_size_attrs(
        with_dims(
            with_shape_ports(
                entry(
                    "stablehlo.dynamic_conv",
                    "dynamic_conv",
                    C::Core,
                    Exactly(3),
                    Exactly(1),
                    E::Free,
                    s::dynamic_conv,
                ),
                ShapePorts::Ports(&[2]),
            ),
            dims::convolution,
        ),
        &[
            "window_strides",
            "lhs_dilation",
            "rhs_dilation",
            "feature_group_count",
            "batch_group_count",
        ],
    ),
    with_dims(
        with_regions(
            entry(
                "stablehlo.reduce",
                "reduce",
                C::Core,
                AtLeast(2),
                AtLeast(1),
                E::Free,
                s::reduce,
            ),
            1,
        ),
        dims::reduce,
    ),
    with_size_attrs(
        with_dims(
            with_regions(
                entry(
                    "stablehlo.reduce_window",
                    "reduce_window",
                    C::Core,
                    AtLeast(2),
                    AtLeast(1),
                    E::Free,
                    s::reduce_window,
                ),
                1,
            ),
            dims::same_inputs,
        ),
        &[
            "window_dimensions",
            "window_strides",
            "base_dilations",
            "window_dilations",
            "padding",
        ],
    ),
    // `custom_call @stablehlo.dynamic_reduce_window(inputs…, init_values…, window_dimensions,
    // window_strides, base_dilations, window_dilations, padding)`: JAX's encoding of the spec's
    // `dynamic_reduce_window` (stablehlo `docs/spec.md`, "dynamic_reduce_window" in the experimental ops
    // of `stablehlo/dialect/StablehloOps.td`), the reducer passed as a called computation.
    with_dims(
        custom(
            "stablehlo.dynamic_reduce_window",
            with_shape_ports(
                entry(
                    "stablehlo.custom_call",
                    "custom_call",
                    C::Core,
                    AtLeast(7),
                    AtLeast(1),
                    E::Free,
                    s::dynamic_reduce_window,
                ),
                ShapePorts::Last(5),
            ),
        ),
        dims::same_inputs,
    ),
    // Irregular.
    with_dims(
        with_regions(
            entry(
                "stablehlo.sort",
                "sort",
                C::Irregular,
                AtLeast(1),
                AtLeast(1),
                E::Free,
                s::same_all,
            ),
            1,
        ),
        dims::sort,
    ),
    with_size_attrs(
        with_dims(
            entry(
                "stablehlo.fft",
                "fft",
                C::Irregular,
                Exactly(1),
                Exactly(1),
                E::Free,
                s::fft,
            ),
            dims::fft,
        ),
        &["fft_length"],
    ),
    // Size computation.
    entry(
        "stablehlo.get_dimension_size",
        "get_dimension_size",
        C::Size,
        Exactly(1),
        Exactly(1),
        E::I32,
        s::scalar,
    ),
    // `custom_call @shape_assertion(pred, dim values…)`: JAX's run-time check of the symbolic shape
    // constraints (`jax/_src/export/shape_poly.py`, `shape_assertion`).
    custom(
        "shape_assertion",
        with_shape_ports(
            entry(
                "stablehlo.custom_call",
                "custom_call",
                C::Size,
                AtLeast(1),
                Exactly(0),
                E::Free,
                s::none,
            ),
            ShapePorts::All,
        ),
    ),
    // Structure.
    entry(
        "func.return",
        "func.return",
        C::Control,
        AtLeast(0),
        Exactly(0),
        E::Free,
        s::none,
    ),
    entry(
        "stablehlo.return",
        "return",
        C::Control,
        AtLeast(0),
        Exactly(0),
        E::Free,
        s::none,
    ),
    with_shape_ports(
        entry(
            "func.call",
            "func.call",
            C::Control,
            AtLeast(0),
            AtLeast(0),
            E::Free,
            s::callee,
        ),
        ShapePorts::DimVarArgs,
    ),
    with_shape_ports(
        entry(
            "stablehlo.composite",
            "composite",
            C::Control,
            AtLeast(0),
            AtLeast(0),
            E::Free,
            s::callee,
        ),
        ShapePorts::DimVarArgs,
    ),
];

fn index() -> &'static HashMap<String, &'static OpSpec> {
    static I: OnceLock<HashMap<String, &'static OpSpec>> = OnceLock::new();
    I.get_or_init(|| TABLE.iter().map(|e| (e.key(), e)).collect())
}

/// The entry of an op (`target` = `call_target_name` of a `custom_call`).
pub fn lookup(name: &str, target: Option<&str>) -> Option<&'static OpSpec> {
    let idx = index();
    match target {
        Some(t) if name == "stablehlo.custom_call" => idx.get(&format!("{name}@{t}")).copied(),
        _ => idx.get(name).copied(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_unique_and_specs_consistent() {
        let mut seen = std::collections::BTreeSet::new();
        for e in TABLE {
            assert!(seen.insert(e.key()), "duplicate {}", e.key());
            assert_eq!(e.elementwise, e.class == OpClass::Elementwise, "{}", e.key());
            if e.commutative {
                assert_eq!(e.operands, Arity::Exactly(2), "{}", e.key());
            }
        }
    }

    #[test]
    fn required_entries_exist() {
        for k in [
            "func.call",
            "stablehlo.composite",
            "stablehlo.fft",
            "stablehlo.real",
            "stablehlo.imag",
            "stablehlo.complex",
            "stablehlo.reduce_precision",
            "stablehlo.dynamic_broadcast_in_dim",
            "stablehlo.dynamic_conv",
            "stablehlo.dynamic_gather",
            "stablehlo.dynamic_iota",
            "stablehlo.dynamic_pad",
            "stablehlo.dynamic_reshape",
            "stablehlo.dynamic_slice",
            "stablehlo.dynamic_update_slice",
            "stablehlo.real_dynamic_slice",
            "stablehlo.get_dimension_size",
        ] {
            assert!(lookup(k, None).is_some(), "{k}");
        }
        assert!(lookup("stablehlo.custom_call", Some("shape_assertion")).is_some());
        assert!(lookup("stablehlo.custom_call", Some("stablehlo.dynamic_reduce_window")).is_some());
        assert!(lookup("stablehlo.custom_call", Some("Sharding")).is_none());
        assert!(lookup("stablehlo.custom_call", None).is_none());
        assert!(lookup("stablehlo.while", None).is_none());
    }
}
