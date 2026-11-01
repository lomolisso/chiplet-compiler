//! Result-shape rules, one per op kind, transcribed from the StableHLO spec's shape constraints (cited per
//! rule). They are symbolic: dims are `Dim`s, shape operands arrive as evaluated `SymExpr` vectors, and
//! equalities between dims are decided with the scope's bounds (`SymCtx`).

use crate::attrs::{Attr, Attrs, AttrsExt};
use crate::sym::{SymCtx, SymExpr};
use crate::types::{Dim, TensorType};

/// What a shape rule sees.
#[derive(Debug, Clone, Copy)]
pub struct ShapeIn<'a> {
    pub operands: &'a [&'a TensorType],
    pub attrs: &'a Attrs,
    /// Per operand: its elements, row-major, when it is a computed integer vector (size computation).
    pub values: &'a [Option<Vec<SymExpr>>],
    /// Per result: the declared shape (`None` = `?`), for ops whose shape is part of their type.
    pub declared: &'a [Vec<Option<u64>>],
    /// The callee's result types (composites).
    pub callee: Option<&'a [TensorType]>,
    pub ctx: &'a SymCtx,
}

/// Result shapes from the operands, or why they cannot be derived.
pub type ShapeFn = fn(&ShapeIn<'_>) -> Result<Vec<Vec<Dim>>, String>;

type R = Result<Vec<Vec<Dim>>, String>;

fn dim(e: SymExpr) -> Result<Dim, String> {
    Dim::from_expr(e.clone()).ok_or_else(|| format!("negative dimension {e}"))
}

/// Two dims that must be equal (spec: shapes agree).
fn unify(a: &Dim, b: &Dim, ctx: &SymCtx) -> Result<Dim, String> {
    if a == b || ctx.eq(&a.expr(), &b.expr()) == Some(true) {
        Ok(a.clone())
    } else {
        Err(format!("dimensions {a} and {b} are not provably equal"))
    }
}

fn operand<'a>(x: &ShapeIn<'a>, i: usize) -> Result<&'a TensorType, String> {
    x.operands
        .get(i)
        .copied()
        .ok_or_else(|| format!("missing operand {i}"))
}

fn value<'a>(x: &ShapeIn<'a>, i: usize) -> Result<&'a [SymExpr], String> {
    x.values
        .get(i)
        .and_then(|v| v.as_deref())
        .ok_or_else(|| format!("shape operand {i} is not computable"))
}

fn dims_of(v: &[SymExpr]) -> Result<Vec<Dim>, String> {
    v.iter().cloned().map(dim).collect()
}

fn ints(a: &Attrs, k: &str) -> Result<Vec<i64>, String> {
    a.ints(k).ok_or_else(|| format!("missing attribute `{k}`"))
}

fn ints_or(a: &Attrs, k: &str, n: usize, default: i64) -> Result<Vec<i64>, String> {
    match a.get(k) {
        None => Ok(vec![default; n]),
        Some(_) => {
            let v = ints(a, k)?;
            if v.len() == n {
                Ok(v)
            } else {
                Err(format!("`{k}` has {} entries, expected {n}", v.len()))
            }
        }
    }
}

fn int(a: &Attrs, k: &str) -> Result<i64, String> {
    a.int(k).ok_or_else(|| format!("missing attribute `{k}`"))
}

fn idx(i: i64, n: usize) -> Result<usize, String> {
    usize::try_from(i)
        .ok()
        .filter(|&u| u < n)
        .ok_or_else(|| format!("dimension index {i} out of range {n}"))
}

/// No results.
pub fn none(_: &ShapeIn<'_>) -> R {
    Ok(Vec::new())
}

/// A rank-0 result.
pub fn scalar(_: &ShapeIn<'_>) -> R {
    Ok(vec![Vec::new()])
}

/// The result shape is operand 0's (`reverse`, `dynamic_update_slice`).
pub fn same0(x: &ShapeIn<'_>) -> R {
    Ok(vec![operand(x, 0)?.shape.clone()])
}

/// One result per operand, each of its operand's shape (`sort`).
pub fn same_all(x: &ShapeIn<'_>) -> R {
    Ok(x.operands.iter().map(|t| t.shape.clone()).collect())
}

/// The result shape is the declared one, which must be static (`constant`, `iota`).
pub fn declared(x: &ShapeIn<'_>) -> R {
    x.declared
        .iter()
        .map(|d| {
            d.iter()
                .map(|e| {
                    e.map(Dim::Const)
                        .ok_or_else(|| "declared result shape is not static".to_string())
                })
                .collect()
        })
        .collect()
}

/// The callee's result shapes (composites, calls).
pub fn callee(x: &ShapeIn<'_>) -> R {
    let r = x.callee.ok_or("no callee")?;
    Ok(r.iter().map(|t| t.shape.clone()).collect())
}

/// Element-wise ops: all operand shapes equal; `select`'s predicate and `clamp`'s bounds may be rank 0
/// (spec `#select` C2, `#clamp` C1).
pub fn elementwise(x: &ShapeIn<'_>) -> R {
    let full: Vec<&TensorType> = x.operands.iter().copied().filter(|t| t.rank() > 0).collect();
    let Some(first) = full.first() else {
        return Ok(vec![Vec::new()]);
    };
    let mut out = first.shape.clone();
    for t in &full[1..] {
        if t.rank() != out.len() {
            return Err(format!("rank {} vs {}", t.rank(), out.len()));
        }
        for (o, d) in out.iter_mut().zip(&t.shape) {
            *o = unify(o, d, x.ctx)?;
        }
    }
    Ok(vec![out])
}

/// Spec `#broadcast_in_dim` C2–C5: static result; each operand dim is 1 or equals its result dim.
pub fn broadcast_in_dim(x: &ShapeIn<'_>) -> R {
    let out = declared(x)?;
    let res = out.first().ok_or("no result")?;
    check_broadcast(x, res)?;
    Ok(out)
}

fn check_broadcast(x: &ShapeIn<'_>, res: &[Dim]) -> Result<(), String> {
    let op = operand(x, 0)?;
    let bd = ints(x.attrs, "broadcast_dimensions")?;
    if bd.len() != op.rank() {
        return Err("broadcast_dimensions size differs from the operand rank".into());
    }
    for (d, &r) in op.shape.iter().zip(&bd) {
        let r = &res[idx(r, res.len())?];
        if *d != Dim::Const(1) {
            unify(d, r, x.ctx)?;
        }
    }
    Ok(())
}

/// Spec `#dynamic_broadcast_in_dim`: the result shape is `output_dimensions`.
pub fn dynamic_broadcast_in_dim(x: &ShapeIn<'_>) -> R {
    let res = dims_of(value(x, 1)?)?;
    check_broadcast(x, &res)?;
    Ok(vec![res])
}

fn product(ds: &[Dim]) -> SymExpr {
    ds.iter().fold(SymExpr::constant(1), |a, d| a.mul(&d.expr()))
}

/// Spec `#reshape` C1: the declared static shape with the same number of elements.
pub fn reshape(x: &ShapeIn<'_>) -> R {
    let out = declared(x)?;
    let op = operand(x, 0)?;
    if x.ctx.eq(&product(&op.shape), &product(&out[0])) != Some(true) {
        return Err("element counts differ".into());
    }
    Ok(out)
}

/// Spec `#dynamic_reshape`: the result shape is `output_shape`.
pub fn dynamic_reshape(x: &ShapeIn<'_>) -> R {
    let res = dims_of(value(x, 1)?)?;
    let op = operand(x, 0)?;
    if x.ctx.eq(&product(&op.shape), &product(&res)) == Some(false) {
        return Err("element counts differ".into());
    }
    Ok(vec![res])
}

/// Spec `#dynamic_iota`: the result shape is `output_shape`.
pub fn dynamic_iota(x: &ShapeIn<'_>) -> R {
    Ok(vec![dims_of(value(x, 0)?)?])
}

/// Spec `#transpose` C3: `shape(result)[i] = shape(operand)[permutation[i]]`.
pub fn transpose(x: &ShapeIn<'_>) -> R {
    let op = operand(x, 0)?;
    let p = ints(x.attrs, "permutation")?;
    let out = p
        .iter()
        .map(|&i| Ok(op.shape[idx(i, op.rank())?].clone()))
        .collect::<Result<_, String>>()?;
    Ok(vec![out])
}

/// `ceil((limit - start) / stride)` (spec `#slice` C4).
fn slice_len(start: &SymExpr, limit: &SymExpr, stride: &SymExpr) -> SymExpr {
    limit.sub(start).add(stride).add_const(-1).floordiv(stride)
}

/// Spec `#slice`.
pub fn slice(x: &ShapeIn<'_>) -> R {
    let op = operand(x, 0)?;
    let n = op.rank();
    let (st, li) = (ints(x.attrs, "start_indices")?, ints(x.attrs, "limit_indices")?);
    let sd = ints_or(x.attrs, "strides", n, 1)?;
    if st.len() != n || li.len() != n {
        return Err("slice indices do not match the rank".into());
    }
    let out = (0..n)
        .map(|i| {
            dim(slice_len(
                &SymExpr::constant(st[i]),
                &SymExpr::constant(li[i]),
                &SymExpr::constant(sd[i]),
            ))
        })
        .collect::<Result<_, _>>()?;
    Ok(vec![out])
}

/// Spec `#real_dynamic_slice`: `ceil((limit_indices - start_indices) / strides)` from the operands.
pub fn real_dynamic_slice(x: &ShapeIn<'_>) -> R {
    let (st, li, sd) = (value(x, 1)?, value(x, 2)?, value(x, 3)?);
    let n = operand(x, 0)?.rank();
    if st.len() != n || li.len() != n || sd.len() != n {
        return Err("slice operands do not match the rank".into());
    }
    let out = (0..n)
        .map(|i| dim(slice_len(&st[i], &li[i], &sd[i])))
        .collect::<Result<_, _>>()?;
    Ok(vec![out])
}

/// Spec `#dynamic_slice`: the result shape is `slice_sizes`.
pub fn dynamic_slice(x: &ShapeIn<'_>) -> R {
    let s = ints(x.attrs, "slice_sizes")?;
    Ok(vec![
        s.into_iter()
            .map(|v| dim(SymExpr::constant(v)))
            .collect::<Result<_, _>>()?,
    ])
}

/// Spec `#concatenate` C6: sizes add along `dimension`, the other dims agree.
pub fn concatenate(x: &ShapeIn<'_>) -> R {
    let d = int(x.attrs, "dimension")?;
    let first = operand(x, 0)?;
    let d = idx(d, first.rank())?;
    let mut out = first.shape.clone();
    let mut sum = first.shape[d].expr();
    for t in &x.operands[1..] {
        if t.rank() != out.len() {
            return Err("ranks differ".into());
        }
        for (i, (o, td)) in out.iter_mut().zip(&t.shape).enumerate() {
            if i != d {
                *o = unify(o, td, x.ctx)?;
            }
        }
        sum = sum.add(&t.shape[d].expr());
    }
    out[d] = dim(sum)?;
    Ok(vec![out])
}

/// `lo + d + hi + max(d - 1, 0) * interior` (spec `#pad` C4).
fn padded(d: &Dim, lo: &SymExpr, hi: &SymExpr, interior: &SymExpr, ctx: &SymCtx) -> Result<Dim, String> {
    let e = d.expr();
    let gaps = ctx.max(&e.add_const(-1), &SymExpr::zero()).mul(interior);
    dim(e.add(lo).add(hi).add(&gaps))
}

/// Spec `#pad`.
pub fn pad(x: &ShapeIn<'_>) -> R {
    let op = operand(x, 0)?;
    let n = op.rank();
    let (lo, hi) = (
        ints(x.attrs, "edge_padding_low")?,
        ints(x.attrs, "edge_padding_high")?,
    );
    let it = ints_or(x.attrs, "interior_padding", n, 0)?;
    if lo.len() != n || hi.len() != n {
        return Err("padding does not match the rank".into());
    }
    let c = SymExpr::constant;
    let out = (0..n)
        .map(|i| padded(&op.shape[i], &c(lo[i]), &c(hi[i]), &c(it[i]), x.ctx))
        .collect::<Result<_, _>>()?;
    Ok(vec![out])
}

/// Spec `#dynamic_pad`: padding amounts from operands 2–4.
pub fn dynamic_pad(x: &ShapeIn<'_>) -> R {
    let op = operand(x, 0)?;
    let (lo, hi, it) = (value(x, 2)?, value(x, 3)?, value(x, 4)?);
    let n = op.rank();
    if lo.len() != n || hi.len() != n || it.len() != n {
        return Err("padding does not match the rank".into());
    }
    let out = (0..n)
        .map(|i| padded(&op.shape[i], &lo[i], &hi[i], &it[i], x.ctx))
        .collect::<Result<_, _>>()?;
    Ok(vec![out])
}

/// Spec `#dot_general` C12: batch dims, then lhs free dims, then rhs free dims.
pub fn dot_general(x: &ShapeIn<'_>) -> R {
    let (l, r) = (operand(x, 0)?, operand(x, 1)?);
    let Some(Attr::DotDims(dn)) = x.attrs.get("dot_dimension_numbers") else {
        return Err("missing dot_dimension_numbers".into());
    };
    if dn.lhs_batching.len() != dn.rhs_batching.len() || dn.lhs_contracting.len() != dn.rhs_contracting.len()
    {
        return Err("dimension number lists differ in length".into());
    }
    let mut out = Vec::new();
    for (&a, &b) in dn.lhs_batching.iter().zip(&dn.rhs_batching) {
        out.push(unify(
            &l.shape[idx(a, l.rank())?],
            &r.shape[idx(b, r.rank())?],
            x.ctx,
        )?);
    }
    for (&a, &b) in dn.lhs_contracting.iter().zip(&dn.rhs_contracting) {
        unify(&l.shape[idx(a, l.rank())?], &r.shape[idx(b, r.rank())?], x.ctx)?;
    }
    for (t, used) in [
        (l, [&dn.lhs_batching, &dn.lhs_contracting]),
        (r, [&dn.rhs_batching, &dn.rhs_contracting]),
    ] {
        for (i, d) in t.shape.iter().enumerate() {
            if !used.iter().any(|u| u.contains(&(i as i64))) {
                out.push(d.clone());
            }
        }
    }
    Ok(vec![out])
}

/// Number of windows along one spatial dim (spec `#convolution` C25 / `#reduce_window` C15):
/// `dilated = (d - 1) * base_dilation + 1`, `padded = lo + dilated + hi`,
/// `window = (w - 1) * window_dilation + 1`, `floor((padded - window) / stride) + 1` (0 if empty).
fn windows(
    d: &Dim,
    w: &SymExpr,
    stride: &SymExpr,
    (lo, hi): (&SymExpr, &SymExpr),
    base_dil: &SymExpr,
    win_dil: &SymExpr,
    ctx: &SymCtx,
) -> Result<Dim, String> {
    let dilated = d.expr().add_const(-1).mul(base_dil).add_const(1);
    let padded = dilated.add(lo).add(hi);
    let window = w.add_const(-1).mul(win_dil).add_const(1);
    let span = padded.sub(&window);
    if ctx.nonneg(&span) == Some(false) {
        return Ok(Dim::Const(0));
    }
    dim(span.floordiv(stride).add_const(1))
}

fn conv(x: &ShapeIn<'_>, padding: Vec<(SymExpr, SymExpr)>) -> R {
    let (l, k) = (operand(x, 0)?, operand(x, 1)?);
    let Some(Attr::ConvDims(dn)) = x.attrs.get("dimension_numbers") else {
        return Err("missing dimension_numbers".into());
    };
    let ns = dn.input_spatial.len();
    if padding.len() != ns {
        return Err("padding does not match the spatial rank".into());
    }
    let strides = ints_or(x.attrs, "window_strides", ns, 1)?;
    let lhs_dil = ints_or(x.attrs, "lhs_dilation", ns, 1)?;
    let rhs_dil = ints_or(x.attrs, "rhs_dilation", ns, 1)?;
    let bgc = x.attrs.int("batch_group_count").unwrap_or(1);
    let rank = l.rank();
    let mut out = vec![Dim::Const(0); rank];
    let c = SymExpr::constant;
    let batch = l.shape[idx(dn.input_batch, rank)?].expr().floordiv(&c(bgc));
    out[idx(dn.output_batch, rank)?] = dim(batch)?;
    out[idx(dn.output_feature, rank)?] = k.shape[idx(dn.kernel_output_feature, k.rank())?].clone();
    for i in 0..ns {
        let d = &l.shape[idx(dn.input_spatial[i], rank)?];
        let w = k.shape[idx(dn.kernel_spatial[i], k.rank())?].expr();
        let (lo, hi) = &padding[i];
        out[idx(dn.output_spatial[i], rank)?] = windows(
            d,
            &w,
            &c(strides[i]),
            (lo, hi),
            &c(lhs_dil[i]),
            &c(rhs_dil[i]),
            x.ctx,
        )?;
    }
    Ok(vec![out])
}

/// Spec `#convolution`.
pub fn convolution(x: &ShapeIn<'_>) -> R {
    let ns = match x.attrs.get("dimension_numbers") {
        Some(Attr::ConvDims(dn)) => dn.input_spatial.len(),
        _ => return Err("missing dimension_numbers".into()),
    };
    let p = match x.attrs.get("padding") {
        None => vec![(0, 0); ns],
        Some(_) => x.attrs.padding("padding").ok_or("bad padding")?,
    };
    conv(
        x,
        p.into_iter()
            .map(|(a, b)| (SymExpr::constant(a), SymExpr::constant(b)))
            .collect(),
    )
}

/// Spec `#dynamic_conv`: `padding` is operand 2 (`[spatial, 2]`).
pub fn dynamic_conv(x: &ShapeIn<'_>) -> R {
    let p = value(x, 2)?;
    if !p.len().is_multiple_of(2) {
        return Err("padding operand is not [N, 2]".into());
    }
    conv(x, p.chunks(2).map(|c| (c[0].clone(), c[1].clone())).collect())
}

/// Spec `#reduce` C7: each result is its input's shape without `dimensions`.
pub fn reduce(x: &ShapeIn<'_>) -> R {
    let n = x.operands.len() / 2;
    let dims = ints(x.attrs, "dimensions")?;
    let first = operand(x, 0)?;
    let keep: Vec<Dim> = first
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !dims.contains(&(*i as i64)))
        .map(|(_, d)| d.clone())
        .collect();
    for t in &x.operands[1..n] {
        if t.rank() != first.rank() {
            return Err("inputs differ in rank".into());
        }
        for (a, b) in t.shape.iter().zip(&first.shape) {
            unify(a, b, x.ctx)?;
        }
    }
    Ok(vec![keep; n])
}

fn reduce_window_with(
    x: &ShapeIn<'_>,
    n: usize,
    win: &[SymExpr],
    strides: &[SymExpr],
    base: &[SymExpr],
    wdil: &[SymExpr],
    pad: &[(SymExpr, SymExpr)],
) -> R {
    let first = operand(x, 0)?;
    let r = first.rank();
    if [win.len(), strides.len(), base.len(), wdil.len(), pad.len()]
        .iter()
        .any(|&l| l != r)
    {
        return Err("window attributes do not match the rank".into());
    }
    let out = (0..r)
        .map(|i| {
            windows(
                &first.shape[i],
                &win[i],
                &strides[i],
                (&pad[i].0, &pad[i].1),
                &base[i],
                &wdil[i],
                x.ctx,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(vec![out; n])
}

/// Spec `#reduce_window` C15.
pub fn reduce_window(x: &ShapeIn<'_>) -> R {
    let n = x.operands.len() / 2;
    let r = operand(x, 0)?.rank();
    let c = |v: Vec<i64>| v.into_iter().map(SymExpr::constant).collect::<Vec<_>>();
    let win = c(ints(x.attrs, "window_dimensions")?);
    let strides = c(ints_or(x.attrs, "window_strides", r, 1)?);
    let base = c(ints_or(x.attrs, "base_dilations", r, 1)?);
    let wdil = c(ints_or(x.attrs, "window_dilations", r, 1)?);
    let pad = match x.attrs.get("padding") {
        None => vec![(0, 0); r],
        Some(_) => x.attrs.padding("padding").ok_or("bad padding")?,
    };
    let pad: Vec<_> = pad
        .into_iter()
        .map(|(a, b)| (SymExpr::constant(a), SymExpr::constant(b)))
        .collect();
    reduce_window_with(x, n, &win, &strides, &base, &wdil, &pad)
}

/// `custom_call @stablehlo.dynamic_reduce_window(inputs…, inits…, window_dimensions, window_strides,
/// base_dilations, window_dilations, padding)`: the window attributes are operands.
pub fn dynamic_reduce_window(x: &ShapeIn<'_>) -> R {
    let m = x.operands.len();
    if m < 7 || !(m - 5).is_multiple_of(2) {
        return Err("unexpected operand count".into());
    }
    let n = (m - 5) / 2;
    let v = |i: usize| value(x, 2 * n + i).map(<[SymExpr]>::to_vec);
    let pad = v(4)?;
    if !pad.len().is_multiple_of(2) {
        return Err("padding operand is not [N, 2]".into());
    }
    let pad: Vec<_> = pad.chunks(2).map(|c| (c[0].clone(), c[1].clone())).collect();
    reduce_window_with(x, n, &v(0)?, &v(1)?, &v(2)?, &v(3)?, &pad)
}

/// Spec `#gather` C14–C16 (and `#dynamic_gather` with `slice_sizes` as an operand).
fn gather_with(x: &ShapeIn<'_>, slice_sizes: Vec<SymExpr>) -> R {
    let (_op, ix) = (operand(x, 0)?, operand(x, 1)?);
    let Some(Attr::GatherDims(dn)) = x.attrs.get("dimension_numbers") else {
        return Err("missing dimension_numbers".into());
    };
    let ivd = usize::try_from(dn.index_vector_dim).map_err(|_| "negative index_vector_dim")?;
    let batch: Vec<Dim> = ix
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != ivd)
        .map(|(_, d)| d.clone())
        .collect();
    let offsets: Vec<Dim> = slice_sizes
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let i = *i as i64;
            !dn.collapsed_slice_dims.contains(&i) && !dn.operand_batching_dims.contains(&i)
        })
        .map(|(_, s)| dim(s.clone()))
        .collect::<Result<_, _>>()?;
    if offsets.len() != dn.offset_dims.len() {
        return Err("offset_dims do not match slice_sizes".into());
    }
    let rank = batch.len() + offsets.len();
    let (mut bi, mut oi) = (batch.into_iter(), offsets.into_iter());
    let out = (0..rank)
        .map(|d| {
            if dn.offset_dims.contains(&(d as i64)) {
                oi.next()
            } else {
                bi.next()
            }
            .ok_or_else(|| "inconsistent gather dimension numbers".to_string())
        })
        .collect::<Result<_, _>>()?;
    Ok(vec![out])
}

pub fn gather(x: &ShapeIn<'_>) -> R {
    let s = ints(x.attrs, "slice_sizes")?;
    gather_with(x, s.into_iter().map(SymExpr::constant).collect())
}

pub fn dynamic_gather(x: &ShapeIn<'_>) -> R {
    gather_with(x, value(x, 2)?.to_vec())
}

/// Spec `#scatter` C17: the results have the inputs' shapes (`scatter(inputs…, indices, updates…)`).
pub fn scatter(x: &ShapeIn<'_>) -> R {
    let n = (x.operands.len() - 1) / 2;
    Ok(x.operands[..n].iter().map(|t| t.shape.clone()).collect())
}

/// Spec `#fft` C2–C5: `RFFT` halves the last transformed dim (`n / 2 + 1`), `IRFFT` restores
/// `fft_length`, `FFT` / `IFFT` keep the shape.
pub fn fft(x: &ShapeIn<'_>) -> R {
    let op = operand(x, 0)?;
    let ty = x.attrs.enum_value("fft_type").ok_or("missing fft_type")?;
    let len = ints(x.attrs, "fft_length")?;
    let last = *len.last().ok_or("empty fft_length")?;
    let mut out = op.shape.clone();
    let tail = out.last_mut().ok_or("rank-0 fft operand")?;
    match ty {
        "FFT" | "IFFT" => {}
        "RFFT" => *tail = dim(SymExpr::constant(last / 2 + 1))?,
        "IRFFT" => *tail = dim(SymExpr::constant(last))?,
        t => return Err(format!("unknown fft_type {t}")),
    }
    Ok(vec![out])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Elem;

    fn t(shape: Vec<Dim>) -> TensorType {
        TensorType {
            shape,
            elem: Elem::Float {
                exp: 8,
                mant: 23,
                flavour: crate::types::FloatFlavour::Ieee,
            },
        }
    }
    fn s(v: &str) -> Dim {
        Dim::from_expr(SymExpr::parse(v).unwrap()).unwrap()
    }
    fn c(v: u64) -> Dim {
        Dim::Const(v)
    }

    fn run(f: ShapeFn, ops: &[TensorType], attrs: Attrs, values: Vec<Option<Vec<SymExpr>>>) -> R {
        let ctx = SymCtx::new(&["B".into(), "S".into(), "H".into()], &[]);
        let refs: Vec<&TensorType> = ops.iter().collect();
        let values = if values.is_empty() {
            vec![None; ops.len()]
        } else {
            values
        };
        f(&ShapeIn {
            operands: &refs,
            attrs: &attrs,
            values: &values,
            declared: &[],
            callee: None,
            ctx: &ctx,
        })
    }

    #[test]
    fn dot_general_layout() {
        let mut a = Attrs::new();
        a.insert(
            "dot_dimension_numbers".into(),
            Attr::DotDims(crate::attrs::DotDims {
                lhs_batching: vec![0],
                rhs_batching: vec![0],
                lhs_contracting: vec![2],
                rhs_contracting: vec![1],
            }),
        );
        let r = run(
            dot_general,
            &[t(vec![s("B"), s("S"), c(64)]), t(vec![s("B"), c(64), c(16)])],
            a,
            vec![],
        )
        .unwrap();
        assert_eq!(r, vec![vec![s("B"), s("S"), c(16)]]);
    }

    #[test]
    fn conv_output_is_floordiv() {
        let mut a = Attrs::new();
        a.insert(
            "dimension_numbers".into(),
            Attr::ConvDims(crate::attrs::ConvDims {
                input_batch: 0,
                input_feature: 3,
                input_spatial: vec![1, 2],
                kernel_input_feature: 2,
                kernel_output_feature: 3,
                kernel_spatial: vec![0, 1],
                output_batch: 0,
                output_feature: 3,
                output_spatial: vec![1, 2],
            }),
        );
        a.insert("window_strides".into(), Attr::Ints(vec![2, 2]));
        a.insert(
            "padding".into(),
            Attr::Dense {
                ty: crate::types::Type::Tensor(TensorType {
                    shape: vec![c(2), c(2)],
                    elem: Elem::Int {
                        signed: true,
                        width: 64,
                    },
                }),
                splat: true,
                values: Some(vec![crate::attrs::Scalar::Int(1)]),
                digest: None,
            },
        );
        let r = run(
            convolution,
            &[
                t(vec![s("B"), s("32*H"), s("32*H"), c(3)]),
                t(vec![c(3), c(3), c(3), c(16)]),
            ],
            a,
            vec![],
        )
        .unwrap();
        // floor((32H + 2 - 3) / 2) + 1 = 16H
        assert_eq!(r, vec![vec![s("B"), s("16*H"), s("16*H"), c(16)]]);
    }

    #[test]
    fn dynamic_broadcast_and_mismatch() {
        let mut a = Attrs::new();
        a.insert("broadcast_dimensions".into(), Attr::Ints(vec![1]));
        let shape = vec![SymExpr::var("B"), SymExpr::constant(16)];
        let ops = [t(vec![c(16)]), t(vec![c(2)])];
        let r = run(dynamic_broadcast_in_dim, &ops, a.clone(), vec![None, Some(shape)]).unwrap();
        assert_eq!(r, vec![vec![s("B"), c(16)]]);
        let bad = vec![SymExpr::var("B"), SymExpr::constant(8)];
        assert!(run(dynamic_broadcast_in_dim, &ops, a, vec![None, Some(bad)]).is_err());
        assert!(
            run(
                elementwise,
                &[t(vec![s("B")]), t(vec![s("S")])],
                Attrs::new(),
                vec![]
            )
            .is_err()
        );
    }
}
