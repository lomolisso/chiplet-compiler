//! The reduction order ω of every reduction (docs/passes/normalize.md §4). The axes follow each op's
//! definition in the StableHLO spec: the order in which its `reduce` (or, for `scatter`, its schedule)
//! visits the elements once that order is fixed to ascending lexicographic.

use cc_ir::Attr;

use super::rules::{lit_ints, op_name, op_target};
use super::work::Work;
use crate::components::algo::graph::{AAttr, Family};
use crate::normalize::config::Orders;
use crate::normalize::json::{NNode, Omega, Order};

/// The family key of `[passes.normalize.order]`.
pub fn family_key(f: &Family) -> &'static str {
    match f {
        Family::Bool => "bool",
        Family::Sint => "sint",
        Family::Uint => "uint",
        Family::Float => "float",
        Family::Complex(_) => "complex",
    }
}

/// The reduced axes of a reduction node, or `None` when the op is not a reduction.
/// - `reduce`: `dimensions`, ascending (spec `#reduce`: ascending lexicographic order of the slice index).
/// - `reduce_window`, `dynamic_reduce_window`: every window axis (the window is reduced over
///   `axes(inputs...)`).
/// - `dot_general`: `lhs_contracting_dimensions` in attribute order (spec: the slice is reshaped to
///   `dims(lhs, lhs_contracting_dimensions)` and reduced over all its axes).
/// - `convolution`, `dynamic_conv`: `kernel_spatial_dimensions`, then `kernel_input_feature_dimension`
///   (spec: a `dot_general` with `rhs_contracting_dimensions = kernel_spatial_dimensions +
///   [kernel_input_feature_dimension]`).
/// - `scatter`: every axis of `updates[0]` (the schedule is a permutation of its index space).
pub fn axes(w: &Work, n: u32) -> Option<Vec<u32>> {
    let x = w.node(n)?;
    let u = |v: &[i64]| v.iter().map(|&a| a as u32).collect::<Vec<_>>();
    match (op_name(x)?, op_target(x)) {
        ("stablehlo.reduce", _) => {
            let mut d = u(&lit_ints(x, "dimensions")?);
            d.sort_unstable();
            Some(d)
        }
        ("stablehlo.reduce_window", _)
        | ("stablehlo.custom_call", Some("stablehlo.dynamic_reduce_window")) => {
            let r = operand_rank(w, n, 0)?;
            Some((0..r).collect())
        }
        ("stablehlo.dot_general", _) => match x.label.attrs.get("dot_dimension_numbers")? {
            AAttr::Lit(Attr::DotDims(d)) => Some(u(&d.lhs_contracting)),
            _ => None,
        },
        ("stablehlo.convolution" | "stablehlo.dynamic_conv", _) => {
            match x.label.attrs.get("dimension_numbers")? {
                AAttr::Lit(Attr::ConvDims(d)) => {
                    let mut a = u(&d.kernel_spatial);
                    a.push(d.kernel_input_feature as u32);
                    Some(a)
                }
                _ => None,
            }
        }
        ("stablehlo.scatter", _) => {
            // Operands: `inputs` (one per result), `scatter_indices`, then `updates`.
            let k = x.label.results.len() as u32;
            let r = operand_rank(w, n, k + 1)?;
            Some((0..r).collect())
        }
        _ => None,
    }
}

/// The rank of operand `port` (from its producer's result type).
fn operand_rank(w: &Work, n: u32, port: u32) -> Option<u32> {
    let (u, e) = w
        .data_ins(n)
        .find(|(_, e)| e.port == cc_ir::absgraph::PortLabel::Port(port))
        .copied()?;
    Some(w.node(u)?.label.results.get(e.res as usize)?.dims.len() as u32)
}

/// The family that selects the order: a float or complex result wins over integer ones.
pub fn family(x: &NNode) -> Option<&'static str> {
    let keys: Vec<&str> = x.label.results.iter().map(|t| family_key(&t.family)).collect();
    ["complex", "float", "sint", "uint", "bool"]
        .into_iter()
        .find(|k| keys.contains(k))
}

/// ω of node `n`, or `None` when it is not a reduction. `scatter` is always `seq`: its updates are applied
/// one at a time.
pub fn omega(w: &Work, n: u32, orders: &Orders) -> Option<Omega> {
    let axes = axes(w, n)?;
    let x = w.node(n)?;
    let order = if op_name(x) == Some("stablehlo.scatter") {
        Order::Seq
    } else {
        orders.get(family(x)?)
    };
    Some(Omega { order, axes })
}

/// Whether the op is a reduction ω must cover (also when its axes cannot be read: that is an error).
pub fn is_reduction(x: &NNode) -> bool {
    matches!(
        (op_name(x), op_target(x)),
        (
            Some(
                "stablehlo.reduce"
                    | "stablehlo.reduce_window"
                    | "stablehlo.dot_general"
                    | "stablehlo.convolution"
                    | "stablehlo.dynamic_conv"
                    | "stablehlo.scatter"
            ),
            _
        ) | (
            Some("stablehlo.custom_call"),
            Some("stablehlo.dynamic_reduce_window")
        )
    )
}

/// The printed form of an order (`seq`, `tree:<b>`).
pub fn order_name(o: &Order) -> String {
    match o {
        Order::Seq => "seq".into(),
        Order::Tree { fan_in } => format!("tree:{fan_in}"),
    }
}
