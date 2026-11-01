//! Dimension and element-type equalities that the StableHLO spec forces between an op's operands and
//! results (its shape and type constraints, cited per rule). The components pass groups the dims and the
//! element types of a graph into classes with these relations; an op without a rule relates nothing
//! (`fresh`), which is sound and only gives fewer shared classes.
//!
//! A rule sees only what is structural in an abstract graph: the rank of every operand and result, which
//! axes are the literal dimension 1, and the op's literal attributes (axis numbers, dimension numbers).
//! Axes of size 1 are never related (they are literal in the graph).

use crate::attrs::{Attr, Attrs, AttrsExt};

/// An operand or a result of an op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Side {
    Operand(u32),
    Result(u32),
}

/// One axis of an operand or a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Slot {
    pub side: Side,
    pub axis: u32,
}

/// What a rule sees.
#[derive(Debug, Clone, Copy)]
pub struct DimsIn<'a> {
    /// Per operand port: for each axis whether it is the literal 1; `None` for a port that is not part
    /// of the graph (a shape operand).
    pub operands: &'a [Option<Vec<bool>>],
    /// Per result: for each axis whether it is the literal 1.
    pub results: &'a [Vec<bool>],
    /// The literal attributes.
    pub attrs: &'a Attrs,
}

/// Equal pairs of dims and of element types.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Rel {
    pub dims: Vec<(Slot, Slot)>,
    pub elems: Vec<(Side, Side)>,
}

pub type DimsFn = fn(&DimsIn<'_>) -> Rel;

fn o(i: usize, a: usize) -> Slot {
    Slot {
        side: Side::Operand(i as u32),
        axis: a as u32,
    }
}

fn r(i: usize, a: usize) -> Slot {
    Slot {
        side: Side::Result(i as u32),
        axis: a as u32,
    }
}

impl DimsIn<'_> {
    fn rank_o(&self, i: usize) -> Option<usize> {
        self.operands.get(i)?.as_ref().map(Vec::len)
    }
    fn rank_r(&self, i: usize) -> Option<usize> {
        self.results.get(i).map(Vec::len)
    }
    fn one(&self, s: Slot) -> bool {
        match s.side {
            Side::Operand(i) => self
                .operands
                .get(i as usize)
                .and_then(|x| x.as_ref())
                .and_then(|x| x.get(s.axis as usize))
                .copied()
                .unwrap_or(true),
            Side::Result(i) => self
                .results
                .get(i as usize)
                .and_then(|x| x.get(s.axis as usize))
                .copied()
                .unwrap_or(true),
        }
    }
}

impl Rel {
    /// Relate two slots that exist and are not the literal 1.
    fn dim(&mut self, x: &DimsIn<'_>, a: Slot, b: Slot) {
        if !x.one(a) && !x.one(b) {
            self.dims.push((a, b));
        }
    }
    /// Relate every axis of two sides of equal rank.
    fn same_shape(&mut self, x: &DimsIn<'_>, a: Side, b: Side) {
        let rank = |s: Side| match s {
            Side::Operand(i) => x.rank_o(i as usize),
            Side::Result(i) => x.rank_r(i as usize),
        };
        if let (Some(ra), Some(rb)) = (rank(a), rank(b))
            && ra == rb
        {
            for k in 0..ra as u32 {
                self.dim(x, Slot { side: a, axis: k }, Slot { side: b, axis: k });
            }
        }
    }
    fn elem(&mut self, a: Side, b: Side) {
        self.elems.push((a, b));
    }
}

fn ints(a: &Attrs, k: &str) -> Vec<usize> {
    a.ints(k)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|v| usize::try_from(v).ok())
        .collect()
}

/// No relations.
pub fn fresh(_: &DimsIn<'_>) -> Rel {
    Rel::default()
}

/// Element-wise ops (spec: each op's shape constraint, `shape(operand…) = shape(result)`): every
/// operand of positive rank has the result's shape (`select`'s predicate and `clamp`'s bounds may be
/// rank 0, `#select` C2, `#clamp` C1).
pub fn elementwise(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    for i in 0..x.operands.len() {
        if x.rank_o(i).is_some_and(|k| k > 0) {
            rel.same_shape(x, Side::Operand(i as u32), Side::Result(0));
        }
    }
    rel
}

/// The result has operand 0's shape (`#reverse` C1, `#dynamic_update_slice` C1).
pub fn same0(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    rel.same_shape(x, Side::Operand(0), Side::Result(0));
    rel
}

/// `#sort` C1–C2: all inputs have one shape and each result has its input's type.
pub fn sort(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    for i in 0..x.operands.len() {
        if i > 0 {
            rel.same_shape(x, Side::Operand(i as u32), Side::Operand(0));
        }
        rel.same_shape(x, Side::Operand(i as u32), Side::Result(i as u32));
        rel.elem(Side::Operand(i as u32), Side::Result(i as u32));
    }
    rel
}

/// `#broadcast_in_dim` C2–C5 (and `#dynamic_broadcast_in_dim`): operand axis `i` is result axis
/// `broadcast_dimensions[i]` unless the operand dim is 1.
pub fn broadcast_in_dim(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    for (i, &d) in ints(x.attrs, "broadcast_dimensions").iter().enumerate() {
        rel.dim(x, o(0, i), r(0, d));
    }
    rel
}

/// `#transpose` C2: `dim(result, i) = dim(operand, permutation[i])`.
pub fn transpose(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    for (i, &p) in ints(x.attrs, "permutation").iter().enumerate() {
        rel.dim(x, r(0, i), o(0, p));
    }
    rel
}

/// `#concatenate` C1, C6: all inputs and the result agree except on `dimension`; all element types are
/// equal (C5).
pub fn concatenate(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    let d = x.attrs.int("dimension").and_then(|v| usize::try_from(v).ok());
    let rank = x.rank_r(0).unwrap_or(0);
    for i in 0..x.operands.len() {
        for a in (0..rank).filter(|&a| Some(a) != d) {
            rel.dim(x, o(i, a), r(0, a));
        }
        rel.elem(Side::Operand(i as u32), Side::Result(0));
    }
    rel
}

/// `#pad` C1: the padding value has the operand's element type.
pub fn pad(_: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    rel.elem(Side::Operand(1), Side::Operand(0));
    rel
}

/// `#dot_general` C1–C4, C13: batching and contracting dims agree between the operands; the result is
/// `batch ++ lhs free ++ rhs free`.
pub fn dot_general(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    let Some(Attr::DotDims(dn)) = x.attrs.get("dot_dimension_numbers") else {
        return rel;
    };
    let u = |v: &[i64]| -> Vec<usize> { v.iter().filter_map(|&a| usize::try_from(a).ok()).collect() };
    let (lb, rb, lc, rc) = (
        u(&dn.lhs_batching),
        u(&dn.rhs_batching),
        u(&dn.lhs_contracting),
        u(&dn.rhs_contracting),
    );
    for (k, (&a, &b)) in lb.iter().zip(&rb).enumerate() {
        rel.dim(x, o(0, a), o(1, b));
        rel.dim(x, o(0, a), r(0, k));
    }
    for (&a, &b) in lc.iter().zip(&rc) {
        rel.dim(x, o(0, a), o(1, b));
    }
    let mut k = lb.len();
    for side in 0..2 {
        let (b, c) = if side == 0 { (&lb, &lc) } else { (&rb, &rc) };
        for a in 0..x.rank_o(side).unwrap_or(0) {
            if !b.contains(&a) && !c.contains(&a) {
                rel.dim(x, o(side, a), r(0, k));
                k += 1;
            }
        }
    }
    rel
}

/// `#convolution` C13–C15 and the output shape rule: the output batch is the input batch when
/// `batch_group_count` is 1, the output feature is the kernel output feature, and the input feature is
/// the kernel input feature when `feature_group_count` is 1 (a group count of 1 is literal in the graph;
/// absent from the literal attributes means it is a parameter, so not 1).
pub fn convolution(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    let Some(Attr::ConvDims(dn)) = x.attrs.get("dimension_numbers") else {
        return rel;
    };
    let u = |a: i64| usize::try_from(a).ok();
    let one = |k: &str| x.attrs.int(k) == Some(1);
    if let (Some(a), Some(b)) = (u(dn.kernel_output_feature), u(dn.output_feature)) {
        rel.dim(x, o(1, a), r(0, b));
    }
    if one("batch_group_count")
        && let (Some(a), Some(b)) = (u(dn.input_batch), u(dn.output_batch))
    {
        rel.dim(x, o(0, a), r(0, b));
    }
    if one("feature_group_count")
        && let (Some(a), Some(b)) = (u(dn.input_feature), u(dn.kernel_input_feature))
    {
        rel.dim(x, o(0, a), o(1, b));
    }
    rel
}

/// `#reduce` C1–C3, C7, C8: all inputs have one shape; result `j` is input `j` without `dimensions`;
/// results have the element types of the init values (C8, through the body), inputs and inits agree.
pub fn reduce(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    let n = x.operands.len() / 2;
    let dims = ints(x.attrs, "dimensions");
    for j in 0..n {
        if j > 0 {
            rel.same_shape(x, Side::Operand(j as u32), Side::Operand(0));
        }
        let kept: Vec<usize> = (0..x.rank_o(j).unwrap_or(0))
            .filter(|a| !dims.contains(a))
            .collect();
        for (k, &a) in kept.iter().enumerate() {
            rel.dim(x, o(j, a), r(j, k));
        }
        rel.elem(Side::Operand((n + j) as u32), Side::Result(j as u32));
    }
    rel
}

/// `#reduce_window` C1 and `custom_call @stablehlo.dynamic_reduce_window`: all inputs have one shape.
pub fn same_inputs(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    let n = x.results.len();
    for j in 0..n {
        if j > 0 {
            rel.same_shape(x, Side::Operand(j as u32), Side::Operand(0));
        }
        rel.elem(Side::Operand((n + j) as u32), Side::Result(j as u32));
    }
    rel
}

/// `#scatter` C1, C17: result `i` has input `i`'s type; updates have the inputs' element types.
pub fn scatter(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    let n = x.results.len();
    for i in 0..n {
        rel.same_shape(x, Side::Operand(i as u32), Side::Result(i as u32));
        if i > 0 {
            rel.same_shape(x, Side::Operand(i as u32), Side::Operand(0));
        }
        rel.elem(Side::Operand(i as u32), Side::Result(i as u32));
        rel.elem(Side::Operand((n + 1 + i) as u32), Side::Result(i as u32));
    }
    rel
}

/// `#fft` C2–C4: the leading `rank - len(fft_length)` dims are kept.
pub fn fft(x: &DimsIn<'_>) -> Rel {
    let mut rel = Rel::default();
    let k = x.attrs.ints("fft_length").map(|v| v.len()).unwrap_or(0);
    let lead = x.rank_o(0).unwrap_or(0).saturating_sub(k);
    for a in 0..lead {
        rel.dim(x, o(0, a), r(0, a));
    }
    rel
}

/// `#dynamic_update_slice` C1–C2: the result is the operand's type; the update has its element type.
pub fn dynamic_update_slice(x: &DimsIn<'_>) -> Rel {
    let mut rel = same0(x);
    rel.elem(Side::Operand(1), Side::Operand(0));
    rel
}

/// Every relation of an op: its rule, plus the element-type equalities of its table entry. With
/// `elem = same_as(i)` and one result, the result has operand `i`'s element type; for an element-wise
/// op, every operand of operand `i`'s family has it too (the spec requires equal operand types, except
/// for `select`'s predicate). `same_family(a, b)` tells whether operands `a` and `b` share a family.
pub fn relations(spec: &super::OpSpec, x: &DimsIn<'_>, same_family: impl Fn(usize, usize) -> bool) -> Rel {
    let mut rel = (spec.dims)(x);
    let present = |i: usize| x.operands.get(i).is_some_and(Option::is_some);
    match spec.elem {
        super::ElemRule::SameAs(i) if x.results.len() == 1 && present(i as usize) => {
            rel.elem(Side::Operand(i), Side::Result(0));
            if spec.elementwise {
                for k in (0..x.operands.len()).filter(|&k| present(k) && k != i as usize) {
                    if same_family(k, i as usize) {
                        rel.elem(Side::Operand(k as u32), Side::Operand(i));
                    }
                }
            }
        }
        _ if spec.elementwise && x.operands.len() > 1 && present(0) => {
            // `compare`, `complex`: the operands have one type (`#compare` C1, `#complex` C1).
            for k in (1..x.operands.len()).filter(|&k| present(k)) {
                if same_family(k, 0) {
                    rel.elem(Side::Operand(k as u32), Side::Operand(0));
                }
            }
        }
        _ => {}
    }
    rel
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attrs::DotDims;

    fn attrs(kv: &[(&str, Attr)]) -> Attrs {
        kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn elementwise_skips_scalars_and_ones() {
        let ops = [Some(vec![false, true]), Some(vec![])];
        let res = [vec![false, true]];
        let a = Attrs::new();
        let rel = elementwise(&DimsIn {
            operands: &ops,
            results: &res,
            attrs: &a,
        });
        assert_eq!(rel.dims, vec![(o(0, 0), r(0, 0))]);
    }

    #[test]
    fn broadcast_maps_non_unit_axes() {
        let ops = [Some(vec![true, false])];
        let res = [vec![false, false, false]];
        let a = attrs(&[("broadcast_dimensions", Attr::Ints(vec![0, 2]))]);
        let rel = broadcast_in_dim(&DimsIn {
            operands: &ops,
            results: &res,
            attrs: &a,
        });
        assert_eq!(rel.dims, vec![(o(0, 1), r(0, 2))]);
    }

    #[test]
    fn dot_general_batch_contract_free() {
        // [b, m, k] x [b, k, n] -> [b, m, n]
        let ops = [Some(vec![false; 3]), Some(vec![false; 3])];
        let res = [vec![false; 3]];
        let a = attrs(&[(
            "dot_dimension_numbers",
            Attr::DotDims(DotDims {
                lhs_batching: vec![0],
                rhs_batching: vec![0],
                lhs_contracting: vec![2],
                rhs_contracting: vec![1],
            }),
        )]);
        let rel = dot_general(&DimsIn {
            operands: &ops,
            results: &res,
            attrs: &a,
        });
        assert_eq!(
            rel.dims,
            vec![
                (o(0, 0), o(1, 0)),
                (o(0, 0), r(0, 0)),
                (o(0, 2), o(1, 1)),
                (o(0, 1), r(0, 1)),
                (o(1, 2), r(0, 2)),
            ]
        );
    }

    #[test]
    fn reduce_keeps_unreduced_axes() {
        let ops = [Some(vec![false; 3]), Some(vec![])];
        let res = [vec![false; 2]];
        let a = attrs(&[("dimensions", Attr::Ints(vec![1]))]);
        let rel = reduce(&DimsIn {
            operands: &ops,
            results: &res,
            attrs: &a,
        });
        assert_eq!(rel.dims, vec![(o(0, 0), r(0, 0)), (o(0, 2), r(0, 1))]);
        assert_eq!(rel.elems, vec![(Side::Operand(1), Side::Result(0))]);
    }

    #[test]
    fn transpose_and_concatenate() {
        let ops = [Some(vec![false; 2])];
        let res = [vec![false; 2]];
        let a = attrs(&[("permutation", Attr::Ints(vec![1, 0]))]);
        let rel = transpose(&DimsIn {
            operands: &ops,
            results: &res,
            attrs: &a,
        });
        assert_eq!(rel.dims, vec![(r(0, 0), o(0, 1)), (r(0, 1), o(0, 0))]);
        let ops = [Some(vec![false; 2]), Some(vec![false; 2])];
        let a = attrs(&[(
            "dimension",
            Attr::Int {
                value: 1,
                ty: "i64".into(),
            },
        )]);
        let rel = concatenate(&DimsIn {
            operands: &ops,
            results: &res,
            attrs: &a,
        });
        assert_eq!(rel.dims, vec![(o(0, 0), r(0, 0)), (o(1, 0), r(0, 0))]);
    }
}
