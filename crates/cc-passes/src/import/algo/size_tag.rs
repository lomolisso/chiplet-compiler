//! Size-computation tagging (design §4.2).
//!
//! An op is size computation when it is a size op by definition (`get_dimension_size`, shape
//! assertions), or when it computes small integer values (static shape, at most `MAX_ELEMS` elements:
//! shape vectors and scalars) from a dimension value and every use of its results is a shape operand or
//! another size op (no use at all counts: dead size code). This is the greatest fixpoint of that rule;
//! constants are not dimension values and are never tagged.
//!
//! A size value (a dimension-variable argument or a size op's result) consumed at a non-shape port of
//! an op that is not size computation is recorded as a `dim_as_data` port of that op: the dimension
//! enters compute there as a run-time scalar.

use cc_ir::ops::{OpClass, ShapePorts};
use cc_ir::{Def, Module, Op, ValueId};

use super::size_value::tracked;

/// Whether port `p` of `op` carries a shape.
pub fn is_shape_port(m: &Module, op: &Op, p: usize) -> bool {
    let Some(spec) = cc_ir::lookup(&op.name, op.target.as_deref()) else {
        return false;
    };
    match spec.shape_operands {
        ShapePorts::DimVarArgs => op.callee.is_some_and(|f| {
            let args = &m.region(m.func(f).body).args;
            args.get(p).is_some_and(|&a| m.value(a).dim_var.is_some())
        }),
        sp => sp.contains(p, op.operands.len()),
    }
}

pub fn is_size_op(op: &Op) -> bool {
    cc_ir::lookup(&op.name, op.target.as_deref()).is_some_and(|s| s.class == OpClass::Size)
}

/// An op that may be size computation: it has results and all are small static integer tensors.
pub fn small_integer(m: &Module, op: &Op) -> bool {
    !op.results.is_empty()
        && op
            .results
            .iter()
            .all(|&r| m.value(r).ty.tensor().and_then(tracked).is_some())
}

/// Set `size` and `dim_as_data` on every op. Op ids order every definition before its uses (the builder
/// creates an op after its operands and before its users), so one pass forward and one backward reach
/// the fixpoints.
pub fn tag(m: &mut Module) {
    let users = m.users();
    // Forward: small integer values computed from a dimension value.
    let mut dep = vec![false; m.values.len()];
    for (i, v) in m.values.iter().enumerate() {
        dep[i] = v.dim_var.is_some();
    }
    let mut cand = vec![false; m.ops.len()];
    for (i, op) in m.ops.iter().enumerate() {
        let d = op.name == "stablehlo.get_dimension_size"
            || (small_integer(m, op) && op.operands.iter().any(|v| dep[v.index()]));
        cand[i] = d || is_size_op(op);
        if d {
            for r in &op.results {
                dep[r.index()] = true;
            }
        }
    }
    // Backward: a candidate stays size computation when every use is a shape port or a size op.
    let mut size = vec![false; m.ops.len()];
    for i in (0..m.ops.len()).rev() {
        let op = &m.ops[i];
        size[i] = is_size_op(op)
            || cand[i]
                && op.results.iter().all(|r| {
                    users[r.index()]
                        .iter()
                        .all(|&(u, p)| size[u.index()] || is_shape_port(m, m.op(u), p as usize))
                });
    }
    let size_value = |v: ValueId| -> bool {
        let val = m.value(v);
        val.dim_var.is_some()
            || match val.def {
                Def::Result { op, .. } => size[op.index()],
                Def::Arg { .. } => false,
            }
    };
    let crossings: Vec<Vec<u32>> = m
        .ops
        .iter()
        .enumerate()
        .map(|(i, op)| {
            if size[i] {
                return Vec::new();
            }
            op.operands
                .iter()
                .enumerate()
                .filter(|&(p, &v)| size_value(v) && !is_shape_port(m, op, p))
                .map(|(p, _)| p as u32)
                .collect()
        })
        .collect();
    for ((op, s), c) in m.ops.iter_mut().zip(size).zip(crossings) {
        op.size = s;
        op.dim_as_data = c;
    }
}
