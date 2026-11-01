//! Abstract interpretation of size computation over `SymExpr`: the integer (and boolean) values of small
//! static tensors, element by element, with StableHLO's integer semantics (spec `#divide`, `#remainder`:
//! truncating; `#compare`; `#select`).
//!
//! Each element is a set of guarded alternatives `guard → value`, the guards (conjunctions of facts
//! `f >= 0`) partitioning the sizes the scope allows. A comparison or sign that the bounds cannot decide
//! splits into its cases; infeasible cases are dropped and alternatives that agree are merged. This
//! decides JAX's floor-division lowering (`select(sign(x) != sign(y) && rem != 0, div - 1, div)`,
//! `jax/_src/lax/lax.py` `_floor_divide`) for sizes that may be zero, such as `(H - 1) // 2` with
//! `H >= 1`. Integers the module computes from run-time data are opaque variables (`%<value>.<i>`);
//! a dimension that depends on one is rejected by the caller.

use cc_ir::{Attr, Attrs, AttrsExt, Elem, Scalar, SymCtx, SymExpr, TensorType};

/// One case of a value: under `guard` (facts `f >= 0`), the value is `v`.
#[derive(Debug, Clone, PartialEq)]
pub struct Alt {
    pub guard: Vec<SymExpr>,
    pub v: SymExpr,
}

/// The cases of one element; together their guards cover every allowed size.
pub type Gv = Vec<Alt>;

/// Elements of a value, row-major, or `None` when untracked.
pub type Sv = Option<Vec<Gv>>;

/// Values with more elements are not tracked (shape vectors are short).
pub const MAX_ELEMS: u64 = 64;
const MAX_ALTS: usize = 8;
const MAX_COMBOS: usize = 64;

pub fn single(v: SymExpr) -> Gv {
    vec![Alt { guard: Vec::new(), v }]
}

/// The elements when each has one value.
pub fn known(sv: &Sv) -> Option<Vec<SymExpr>> {
    sv.as_ref()?
        .iter()
        .map(|g| match g.as_slice() {
            [a] => Some(a.v.clone()),
            _ => None,
        })
        .collect()
}

/// Opaque elements for a value nothing is known about.
pub fn opaque(name: &str, n: usize) -> Sv {
    Some(
        (0..n)
            .map(|i| single(SymExpr::var(&format!("%{name}.{i}"))))
            .collect(),
    )
}

pub fn tracked(t: &TensorType) -> Option<usize> {
    let n = t.num_elements()?;
    (n <= MAX_ELEMS && matches!(t.elem, Elem::Int { .. } | Elem::Bool)).then_some(n as usize)
}

fn feasible(ctx: &SymCtx, guard: &[SymExpr]) -> bool {
    guard.is_empty() || !ctx.with(guard).contradictory()
}

fn merge(alts: Vec<Alt>) -> Option<Gv> {
    let first = alts.first()?.v.clone();
    if alts.iter().all(|a| a.v == first) {
        return Some(single(first));
    }
    let mut out: Vec<Alt> = Vec::new();
    for a in alts {
        if !out.contains(&a) {
            out.push(a);
        }
    }
    (out.len() <= MAX_ALTS).then_some(out)
}

/// Apply `f` to every combination of the operands' cases, under the combination's guard.
/// A case function: the cases of one element from its operands' values, under a context.
type CaseFn<'a> = &'a dyn Fn(&SymCtx, &[&SymExpr]) -> Option<Vec<Alt>>;

fn lift(ctx: &SymCtx, args: &[&Gv], f: CaseFn<'_>) -> Option<Gv> {
    let combos: usize = args.iter().map(|g| g.len()).product();
    if combos > MAX_COMBOS {
        return None;
    }
    let mut out = Vec::new();
    for k in 0..combos {
        let (mut rest, mut guard, mut vals) = (k, Vec::new(), Vec::with_capacity(args.len()));
        for g in args {
            let a = &g[rest % g.len()];
            rest /= g.len();
            guard.extend(a.guard.iter().cloned());
            vals.push(&a.v);
        }
        if !feasible(ctx, &guard) {
            continue;
        }
        let c = if guard.is_empty() {
            ctx.clone()
        } else {
            ctx.with(&guard)
        };
        for a in f(&c, &vals)? {
            let mut g = guard.clone();
            g.extend(a.guard);
            if feasible(ctx, &g) {
                out.push(Alt { guard: g, v: a.v });
            }
        }
    }
    merge(out)
}

/// Whether `e` involves a run-time value (an opaque variable). Cases are never split on those: their
/// alternatives would not merge into a closed form over the size symbols.
fn opaque_in(e: &SymExpr) -> bool {
    e.vars().iter().any(|v| v.starts_with('%'))
}

fn one(v: SymExpr) -> Option<Vec<Alt>> {
    Some(single(v))
}

fn case(guard: Vec<SymExpr>, v: SymExpr) -> Alt {
    Alt { guard, v }
}

fn bool_e(b: bool) -> SymExpr {
    SymExpr::constant(i64::from(b))
}

/// `x >= 0` decided, or split into `x >= 0` / `x <= -1`.
fn split_sign(c: &SymCtx, x: &SymExpr, pos: SymExpr, neg: SymExpr) -> Option<Vec<Alt>> {
    match c.nonneg(x) {
        Some(true) => Some(single(pos)),
        Some(false) => Some(single(neg)),
        None if opaque_in(x) => None,
        None => Some(vec![
            case(vec![x.clone()], pos),
            case(vec![x.neg().add_const(-1)], neg),
        ]),
    }
}

/// Truncating division and remainder (spec `#divide`, `#remainder`) for a positive divisor.
fn div_trunc(c: &SymCtx, a: &SymExpr, b: &SymExpr, rem: bool) -> Option<Vec<Alt>> {
    if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
        return (if rem { x.checked_rem(y) } else { x.checked_div(y) }).map(|v| single(SymExpr::constant(v)));
    }
    if c.ge(b, &SymExpr::constant(1)) != Some(true) {
        return None;
    }
    let (pos, neg) = if rem {
        (a.modulo(b), a.neg().modulo(b).neg())
    } else {
        (a.floordiv(b), a.neg().floordiv(b).neg())
    };
    split_sign(c, a, pos, neg)
}

fn sign(c: &SymCtx, x: &SymExpr) -> Option<Vec<Alt>> {
    let (one, zero) = (SymExpr::constant(1), SymExpr::zero());
    if c.ge(x, &one) == Some(true) {
        return Some(single(one));
    }
    if c.ge(&x.neg(), &one) == Some(true) {
        return Some(single(SymExpr::constant(-1)));
    }
    if c.eq(x, &zero) == Some(true) {
        return Some(single(zero));
    }
    if opaque_in(x) {
        return None;
    }
    Some(vec![
        case(vec![x.add_const(-1)], one),
        case(vec![x.neg().add_const(-1)], SymExpr::constant(-1)),
        case(vec![x.clone(), x.neg()], zero),
    ])
}

/// A comparison, decided or split into its cases.
fn compare(c: &SymCtx, dir: &str, a: &SymExpr, b: &SymExpr) -> Option<Vec<Alt>> {
    let one_ = SymExpr::constant(1);
    let decided = match dir {
        "GE" => c.ge(a, b),
        "GT" => c.ge(a, &b.add(&one_)),
        "LE" => c.ge(b, a),
        "LT" => c.ge(b, &a.add(&one_)),
        "EQ" => c.eq(a, b),
        "NE" => c.eq(a, b).map(|x| !x),
        _ => None,
    };
    if let Some(x) = decided {
        return one(bool_e(x));
    }
    let d = a.sub(b);
    if opaque_in(&d) {
        return None;
    }
    let (t, f) = (bool_e(true), bool_e(false));
    // (facts for true, facts for false…)
    let ge = |x: &SymExpr| vec![x.clone()];
    let (yes, no): (Vec<Vec<SymExpr>>, Vec<Vec<SymExpr>>) = match dir {
        "GE" => (vec![ge(&d)], vec![ge(&d.neg().add_const(-1))]),
        "GT" => (vec![ge(&d.add_const(-1))], vec![ge(&d.neg())]),
        "LE" => (vec![ge(&d.neg())], vec![ge(&d.add_const(-1))]),
        "LT" => (vec![ge(&d.neg().add_const(-1))], vec![ge(&d)]),
        "EQ" => (
            vec![vec![d.clone(), d.neg()]],
            vec![ge(&d.add_const(-1)), ge(&d.neg().add_const(-1))],
        ),
        "NE" => (
            vec![ge(&d.add_const(-1)), ge(&d.neg().add_const(-1))],
            vec![vec![d.clone(), d.neg()]],
        ),
        _ => return None,
    };
    // Decided when every case of one side is contradictory.
    let impossible = |gs: &[Vec<SymExpr>]| gs.iter().all(|g| c.with(g).contradictory());
    if impossible(&no) {
        return one(t);
    }
    if impossible(&yes) {
        return one(f);
    }
    Some(
        yes.into_iter()
            .map(|g| case(g, t.clone()))
            .chain(no.into_iter().map(|g| case(g, f.clone())))
            .collect(),
    )
}

fn constant(attrs: &Attrs, n: usize) -> Option<Vec<SymExpr>> {
    let Some(Attr::Dense {
        splat,
        values: Some(vals),
        ..
    }) = attrs.get("value")
    else {
        return None;
    };
    let ints: Option<Vec<SymExpr>> = vals
        .iter()
        .map(|s| match s {
            Scalar::Int(v) => i64::try_from(*v).ok().map(SymExpr::constant),
            Scalar::Bool(b) => Some(bool_e(*b)),
            _ => None,
        })
        .collect();
    let ints = ints?;
    if *splat {
        Some(vec![ints.first()?.clone(); n])
    } else {
        (ints.len() == n).then_some(ints)
    }
}

fn strides(shape: &[u64]) -> Vec<u64> {
    let mut s = vec![1u64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn dims(t: &TensorType) -> Option<Vec<u64>> {
    t.shape.iter().map(|d| d.as_const()).collect()
}

fn unravel(mut i: u64, shape: &[u64]) -> Vec<u64> {
    strides(shape)
        .iter()
        .map(|s| {
            let q = i / s;
            i %= s;
            q
        })
        .collect()
}

fn ravel(ix: &[u64], shape: &[u64]) -> u64 {
    strides(shape).iter().zip(ix).map(|(s, i)| s * i).sum()
}

/// Element-wise over equal-shaped (or rank-0) operands.
fn zip(ctx: &SymCtx, v: &[&Vec<Gv>], n: usize, f: CaseFn<'_>) -> Option<Vec<Gv>> {
    if v.iter().any(|x| x.len() != 1 && x.len() != n) {
        return None;
    }
    (0..n)
        .map(|i| {
            let args: Vec<&Gv> = v
                .iter()
                .map(|x| if x.len() == 1 { &x[0] } else { &x[i] })
                .collect();
            lift(ctx, &args, f)
        })
        .collect()
}

/// Rearrange elements: result element `i` is element `src(i)` of operand `k(i)`.
fn gather_elems(v: &[&Vec<Gv>], n: u64, f: impl Fn(u64) -> Option<(usize, u64)>) -> Option<Vec<Gv>> {
    (0..n)
        .map(|i| {
            let (k, j) = f(i)?;
            v.get(k)?.get(j as usize).cloned()
        })
        .collect()
}

/// The values of `op`'s results, given its operands' (each `None` when unknown or untracked).
pub fn eval(
    name: &str,
    attrs: &Attrs,
    operands: &[&TensorType],
    vals: &[Sv],
    results: &[TensorType],
    ctx: &SymCtx,
) -> Vec<Sv> {
    let mut out = vec![None; results.len()];
    let Some(res) = results.first() else {
        return out;
    };
    let Some(n) = tracked(res) else {
        return out;
    };
    out[0] = eval1(name, attrs, operands, vals, res, n, ctx).filter(|o| o.len() == n);
    out
}

fn eval1(
    name: &str,
    attrs: &Attrs,
    operands: &[&TensorType],
    vals: &[Sv],
    res: &TensorType,
    n: usize,
    ctx: &SymCtx,
) -> Option<Vec<Gv>> {
    match name {
        "stablehlo.constant" => return Some(constant(attrs, n)?.into_iter().map(single).collect()),
        "stablehlo.get_dimension_size" => {
            let d = usize::try_from(attrs.int("dimension")?).ok()?;
            return Some(vec![single(operands.first()?.shape.get(d)?.expr())]);
        }
        "stablehlo.iota" => {
            let rs = dims(res)?;
            let d = usize::try_from(attrs.int("iota_dimension")?).ok()?;
            return (0..n as u64)
                .map(|i| {
                    unravel(i, &rs)
                        .get(d)
                        .map(|&x| single(SymExpr::constant(x as i64)))
                })
                .collect();
        }
        _ => {}
    }
    let v: Vec<&Vec<Gv>> = vals.iter().map(Option::as_ref).collect::<Option<_>>()?;
    let is_bool = res.elem == Elem::Bool;
    match name {
        "stablehlo.convert" => match (&operands[0].elem, &res.elem) {
            (Elem::Int { .. } | Elem::Bool, Elem::Int { .. }) | (Elem::Bool, Elem::Bool) => {
                Some(v[0].clone())
            }
            (Elem::Int { .. }, Elem::Bool) => {
                zip(ctx, &v, n, &|c, x| compare(c, "NE", x[0], &SymExpr::zero()))
            }
            _ => None,
        },
        "stablehlo.reshape" => (v[0].len() == n).then(|| v[0].clone()),
        "stablehlo.broadcast_in_dim" => {
            let (os, rs, bd) = (
                dims(operands[0])?,
                dims(res)?,
                attrs.ints("broadcast_dimensions")?,
            );
            gather_elems(&v, n as u64, |i| {
                let ri = unravel(i, &rs);
                let oi: Vec<u64> = bd
                    .iter()
                    .enumerate()
                    .map(|(d, &r)| if os[d] == 1 { 0 } else { ri[r as usize] })
                    .collect();
                Some((0, ravel(&oi, &os)))
            })
        }
        "stablehlo.concatenate" => {
            let d = usize::try_from(attrs.int("dimension")?).ok()?;
            let rs = dims(res)?;
            let shapes: Vec<Vec<u64>> = operands.iter().map(|t| dims(t)).collect::<Option<_>>()?;
            gather_elems(&v, n as u64, |i| {
                let mut ix = unravel(i, &rs);
                let mut k = 0;
                while ix[d] >= shapes.get(k)?[d] {
                    ix[d] -= shapes[k][d];
                    k += 1;
                }
                Some((k, ravel(&ix, &shapes[k])))
            })
        }
        "stablehlo.slice" => {
            let (os, rs) = (dims(operands[0])?, dims(res)?);
            let st = attrs.ints("start_indices")?;
            let sd = attrs.ints("strides").unwrap_or_else(|| vec![1; os.len()]);
            gather_elems(&v, n as u64, |i| {
                let oi: Vec<u64> = unravel(i, &rs)
                    .iter()
                    .enumerate()
                    .map(|(d, &x)| st[d] as u64 + x * sd[d] as u64)
                    .collect();
                Some((0, ravel(&oi, &os)))
            })
        }
        "stablehlo.transpose" => {
            let (os, rs, p) = (dims(operands[0])?, dims(res)?, attrs.ints("permutation")?);
            gather_elems(&v, n as u64, |i| {
                let ri = unravel(i, &rs);
                let mut oi = vec![0; os.len()];
                for (r, &src) in p.iter().enumerate() {
                    oi[src as usize] = ri[r];
                }
                Some((0, ravel(&oi, &os)))
            })
        }
        "stablehlo.add" => zip(ctx, &v, n, &|_, x| one(x[0].add(x[1]))),
        "stablehlo.subtract" => zip(ctx, &v, n, &|_, x| one(x[0].sub(x[1]))),
        "stablehlo.multiply" => zip(ctx, &v, n, &|_, x| one(x[0].mul(x[1]))),
        "stablehlo.divide" => zip(ctx, &v, n, &|c, x| div_trunc(c, x[0], x[1], false)),
        "stablehlo.remainder" => zip(ctx, &v, n, &|c, x| div_trunc(c, x[0], x[1], true)),
        "stablehlo.maximum" => zip(ctx, &v, n, &|c, x| one(c.max(x[0], x[1]))),
        "stablehlo.minimum" => zip(ctx, &v, n, &|c, x| one(c.min(x[0], x[1]))),
        "stablehlo.negate" => zip(ctx, &v, n, &|_, x| one(x[0].neg())),
        "stablehlo.abs" => zip(ctx, &v, n, &|c, x| split_sign(c, x[0], x[0].clone(), x[0].neg())),
        "stablehlo.sign" => zip(ctx, &v, n, &|c, x| sign(c, x[0])),
        "stablehlo.clamp" => zip(ctx, &v, n, &|c, x| one(c.min(&c.max(x[1], x[0]), x[2]))),
        "stablehlo.compare" => {
            let dir = attrs.enum_value("comparison_direction")?.to_string();
            zip(ctx, &v, n, &|c, x| compare(c, &dir, x[0], x[1]))
        }
        "stablehlo.select" => zip(ctx, &v, n, &|_, x| match x[0].as_const() {
            Some(p) => one(if p != 0 { x[1].clone() } else { x[2].clone() }),
            None => (x[1] == x[2]).then(|| single(x[1].clone())),
        }),
        "stablehlo.and" | "stablehlo.or" | "stablehlo.xor" => zip(ctx, &v, n, &|_, x| {
            let (a, b) = (x[0].as_const()?, x[1].as_const()?);
            one(SymExpr::constant(match name {
                "stablehlo.and" => a & b,
                "stablehlo.or" => a | b,
                _ => a ^ b,
            }))
        }),
        "stablehlo.not" => zip(ctx, &v, n, &|_, x| {
            let a = x[0].as_const()?;
            one(SymExpr::constant(if is_bool { i64::from(a == 0) } else { !a }))
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cc_ir::Constraint;

    fn i32s() -> TensorType {
        TensorType {
            shape: Vec::new(),
            elem: Elem::Int {
                signed: true,
                width: 32,
            },
        }
    }

    /// JAX's `floor_divide(x, y)` on `x = H - 1`, `y = 2` (H >= 1): both cases of `sign(x)` give
    /// `floordiv(H - 1, 2)`.
    #[test]
    fn floor_divide_lowering_is_decided() {
        let ctx = SymCtx::new(&["H".into()], &[] as &[Constraint]);
        let t = i32s();
        let x: Sv = Some(vec![single(SymExpr::parse("H - 1").unwrap())]);
        let y: Sv = Some(vec![single(SymExpr::constant(2))]);
        let ev = |name: &str, attrs: &Attrs, a: &[&Sv], elem: Elem| -> Sv {
            let ops: Vec<&TensorType> = a.iter().map(|_| &t).collect();
            let vals: Vec<Sv> = a.iter().map(|s| (*s).clone()).collect();
            let r = TensorType {
                shape: Vec::new(),
                elem,
            };
            eval(name, attrs, &ops, &vals, &[r], &ctx).remove(0)
        };
        let int = Elem::Int {
            signed: true,
            width: 32,
        };
        let cmp = |d: &str| -> Attrs {
            let mut a = Attrs::new();
            a.insert(
                "comparison_direction".into(),
                Attr::Enum {
                    kind: "comparison_direction".into(),
                    value: d.into(),
                },
            );
            a
        };
        let none = Attrs::new();
        let div = ev("stablehlo.divide", &none, &[&x, &y], int.clone());
        let sx = ev("stablehlo.sign", &none, &[&x], int.clone());
        let sy = ev("stablehlo.sign", &none, &[&y], int.clone());
        let ne1 = ev("stablehlo.compare", &cmp("NE"), &[&sx, &sy], Elem::Bool);
        let rem = ev("stablehlo.remainder", &none, &[&x, &y], int.clone());
        let zero: Sv = Some(vec![single(SymExpr::zero())]);
        let ne2 = ev("stablehlo.compare", &cmp("NE"), &[&rem, &zero], Elem::Bool);
        let both = ev("stablehlo.and", &none, &[&ne1, &ne2], Elem::Bool);
        assert_eq!(known(&both), Some(vec![SymExpr::zero()]));
        let one: Sv = Some(vec![single(SymExpr::constant(1))]);
        let dm1 = ev("stablehlo.subtract", &none, &[&div, &one], int.clone());
        let r = ev("stablehlo.select", &none, &[&both, &dm1, &div], int);
        assert_eq!(
            known(&r),
            Some(vec![SymExpr::parse("floordiv(H - 1, 2)").unwrap()])
        );
    }
}

/// Floor division and floor modulo from their truncating lowering (JAX `lax.floor_divide` and Python
/// `%` on integers, `jax/_src/lax/lax.py`), by the identities, for `b != 0`:
///
/// - `floor(a / b) = trunc(a / b) - [sign(a) != sign(b) and rem(a, b) != 0]`
///   lowered as `select(and(sign(a) != sign(b), rem(a, b) != 0), divide(a, b) - 1, divide(a, b))`;
/// - `a mod b = rem(a, b) + [(rem(a, b) < 0) != (b < 0) and rem(a, b) != 0] * b`
///   lowered as `select(and((r < 0) != (b < 0), r != 0), r + b, r)` with `r = remainder(a, b)`.
///
/// `select` is given by its operand values; `def` returns the op defining a value. The result is the
/// Python floor division or modulo of the known operand values, when the divisor is provably positive.
pub fn floor_identity(
    sel: &[cc_ir::ValueId],
    def: &dyn Fn(cc_ir::ValueId) -> Option<(String, Vec<cc_ir::ValueId>, Attrs)>,
    val: &dyn Fn(cc_ir::ValueId) -> Option<SymExpr>,
    ctx: &SymCtx,
) -> Option<SymExpr> {
    let [p, t, f] = sel else { return None };
    let op = |v: cc_ir::ValueId, name: &str| -> Option<Vec<cc_ir::ValueId>> {
        let (n, args, _) = def(v)?;
        (n == name).then_some(args)
    };
    let cmp = |v: cc_ir::ValueId, dir: &str| -> Option<Vec<cc_ir::ValueId>> {
        let (n, args, attrs) = def(v)?;
        (n == "stablehlo.compare" && attrs.enum_value("comparison_direction") == Some(dir)).then_some(args)
    };
    let is_const = |v: cc_ir::ValueId, c: i64| val(v).and_then(|e| e.as_const()) == Some(c);
    let both = op(*p, "stablehlo.and")?;
    let positive = |b: &SymExpr| ctx.ge(b, &SymExpr::constant(1)) == Some(true);
    // Floor division.
    if let Some(ab) = op(*f, "stablehlo.divide")
        && let Some(tm) = op(*t, "stablehlo.subtract")
        && tm[0] == *f
        && is_const(tm[1], 1)
    {
        let (a, b) = (ab[0], ab[1]);
        let sign_ne = |v: cc_ir::ValueId| {
            cmp(v, "NE").is_some_and(|s| {
                op(s[0], "stablehlo.sign") == Some(vec![a]) && op(s[1], "stablehlo.sign") == Some(vec![b])
            })
        };
        let rem_ne = |v: cc_ir::ValueId| {
            cmp(v, "NE")
                .is_some_and(|s| op(s[0], "stablehlo.remainder") == Some(vec![a, b]) && is_const(s[1], 0))
        };
        if (sign_ne(both[0]) && rem_ne(both[1])) || (sign_ne(both[1]) && rem_ne(both[0])) {
            let (x, y) = (val(a)?, val(b)?);
            return positive(&y).then(|| x.floordiv(&y));
        }
    }
    // Floor modulo.
    if let Some(ab) = op(*f, "stablehlo.remainder")
        && let Some(tb) = op(*t, "stablehlo.add")
        && tb[0] == *f
        && tb[1] == ab[1]
    {
        let (a, b, r) = (ab[0], ab[1], *f);
        let lt0 = |v: cc_ir::ValueId, x: cc_ir::ValueId| {
            cmp(v, "LT").is_some_and(|s| s[0] == x && is_const(s[1], 0))
        };
        let signs_ne = |v: cc_ir::ValueId| cmp(v, "NE").is_some_and(|s| lt0(s[0], r) && lt0(s[1], b));
        let rem_ne = |v: cc_ir::ValueId| cmp(v, "NE").is_some_and(|s| s[0] == r && is_const(s[1], 0));
        if (signs_ne(both[0]) && rem_ne(both[1])) || (signs_ne(both[1]) && rem_ne(both[0])) {
            let (x, y) = (val(a)?, val(b)?);
            return positive(&y).then(|| x.modulo(&y));
        }
    }
    None
}

#[cfg(test)]
mod identities {
    /// The two identities of `floor_identity`, exhaustively over 8-bit operands (`b != 0`).
    #[test]
    fn floor_identities_hold_for_all_i8() {
        for a in -128i32..128 {
            for b in -128i32..128 {
                if b == 0 {
                    continue;
                }
                let (q, r) = (a / b, a % b);
                let fd = q - i32::from(a.signum() != b.signum() && r != 0);
                assert_eq!(
                    fd,
                    a.div_euclid(b) - i32::from(b < 0 && a.rem_euclid(b) != 0),
                    "{a} // {b}"
                );
                let fm = if (r < 0) != (b < 0) && r != 0 { r + b } else { r };
                assert_eq!(fm, a - b * fd, "{a} % {b}");
            }
        }
    }
}

/// The size values of a written module, recomputed as the builder computes them: dimension arguments are
/// their symbol, other small integer arguments of functions (and region arguments) are opaque, and every
/// op is evaluated in id order (definitions precede uses) with the floor-division identity on `select`.
/// Used by later passes to recover what a `dim_as_data` operand carries; only the ops `wanted` depends on
/// are evaluated (its backward cone), the other values stay `None`.
pub fn interpret(m: &cc_ir::Module, wanted: &[cc_ir::ValueId]) -> Vec<Sv> {
    use cc_ir::{Def, FuncKind, Owner};
    let ctx = SymCtx::new(&m.symbols, &m.constraints);
    let mut cone = vec![false; m.ops.len()];
    let mut work: Vec<cc_ir::ValueId> = wanted.to_vec();
    while let Some(v) = work.pop() {
        if let Def::Result { op, .. } = m.value(v).def
            && !std::mem::replace(&mut cone[op.index()], true)
        {
            work.extend(m.op(op).operands.iter().copied());
        }
    }
    let mut sv: Vec<Sv> = vec![None; m.values.len()];
    let ty = |v: cc_ir::ValueId| m.value(v).ty.tensor();
    for reg in &m.regions {
        let entry = matches!(reg.owner, Owner::Func(f) if m.func(f).kind == FuncKind::Entry);
        for &a in &reg.args {
            let val = m.value(a);
            sv[a.index()] = match &val.dim_var {
                Some(s) => Some(vec![single(SymExpr::var(s))]),
                None if !entry => ty(a).and_then(tracked).and_then(|n| opaque(&a.0.to_string(), n)),
                None => None,
            };
        }
    }
    for (i, op) in m.ops.iter().enumerate() {
        if !cone[i] {
            continue;
        }
        let types: Option<Vec<&TensorType>> = op.operands.iter().map(|&v| ty(v)).collect();
        let res: Option<Vec<TensorType>> = op.results.iter().map(|&v| ty(v).cloned()).collect();
        let (Some(types), Some(res)) = (types, res) else {
            continue;
        };
        let vals: Vec<Sv> = op.operands.iter().map(|v| sv[v.index()].clone()).collect();
        let mut out = eval(&op.name, &op.attrs, &types, &vals, &res, &ctx);
        if op.name == "stablehlo.select" && out.first().is_some_and(|o| known(o).is_none()) {
            let def = |v: cc_ir::ValueId| match m.value(v).def {
                Def::Result { op, .. } => {
                    let o = m.op(op);
                    Some((o.name.clone(), o.operands.clone(), o.attrs.clone()))
                }
                Def::Arg { .. } => None,
            };
            let val =
                |v: cc_ir::ValueId| known(&sv[v.index()]).and_then(|x| (x.len() == 1).then(|| x[0].clone()));
            if let Some(e) = floor_identity(&op.operands, &def, &val, &ctx) {
                out[0] = Some(vec![single(e)]);
            }
        }
        for (&v, s) in op.results.iter().zip(out) {
            sv[v.index()] = match s {
                Some(s) => Some(s),
                None => ty(v).and_then(tracked).and_then(|n| opaque(&v.0.to_string(), n)),
            };
        }
    }
    sv
}
