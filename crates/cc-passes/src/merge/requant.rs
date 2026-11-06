//! The `fixed` requantization variant (design §6.3, docs/passes/merge.md §4.11): a rescaling chain
//! `convert (signed integer → f32) → multiply by a scale → round_nearest_even` becomes one `rescale`
//! vertex (`cc_smt::ops::rescale`: an exact integer multiply by the scale's significand and a rounding
//! shift). The rewrite is structural; whether a class takes it is decided by its tolerance lemmas.

use std::collections::{BTreeMap, BTreeSet};

use cc_smt::float::Format;
use cc_smt::ops::{RESCALE_MAX_BITS, Ty};

use crate::datapaths::json::{Arg, Datapath, Src, Tuple, Vertex};
use crate::datapaths::spec::ty_of_width;

/// One rescaling chain: the vertex ids of its convert, multiply and round, the integer it converts and
/// the scale it multiplies by.
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub convert: u32,
    pub multiply: u32,
    pub round: u32,
    pub x: Src,
    pub scale: Src,
}

/// The width variable a source reads.
fn src_width(dp: &Datapath, s: &Src) -> Option<String> {
    match s {
        Src::Vertex { v, .. } => dp.vertices.get(*v as usize).map(|x| x.width.clone()),
        Src::Input { name } => dp
            .inputs
            .iter()
            .find(|i| &i.name == name)
            .map(|i| i.width.clone()),
        Src::Acc { index } => dp
            .accumulator
            .as_ref()
            .and_then(|a| a.entries.get(*index as usize))
            .map(|e| e.width.clone()),
        Src::Result { index } => match &dp.accumulator {
            Some(a) => a.entries.get(*index as usize).map(|e| e.width.clone()),
            None => dp
                .inputs
                .iter()
                .find(|i| i.name == format!("core_result{index}"))
                .map(|i| i.width.clone()),
        },
    }
}

fn ty(dp: &Datapath, t: &Tuple, s: &Src) -> Option<Ty> {
    ty_of_width(t, &src_width(dp, s)?).ok()
}

/// How many times each vertex is read (operands, outputs, accumulator entries).
fn readers(dp: &Datapath) -> Vec<u32> {
    let mut n = vec![0u32; dp.vertices.len()];
    let mut count = |s: &Src| {
        if let Src::Vertex { v, .. } = s
            && let Some(c) = n.get_mut(*v as usize)
        {
            *c += 1;
        }
    };
    for v in &dp.vertices {
        for a in &v.args {
            count(&a.src);
        }
    }
    for o in &dp.outputs {
        count(&o.src);
    }
    if let Some(a) = &dp.accumulator {
        for e in &a.entries {
            count(&e.next);
            count(&e.init);
        }
    }
    n
}

/// The rescaling chains of a datapath: `round_nearest_even(multiply(convert(x), s))` (either operand
/// order) where, at every width tuple, `x` is a signed integer of at most 32 bits and the convert, the
/// multiply, `s` and the round are f32; the convert and the multiply have no other reader, the three
/// vertices are in one stage and none is part of a combiner.
pub fn sites(dp: &Datapath) -> Vec<Site> {
    let f32t = Ty::Float(Format::F32);
    let n = readers(dp);
    let combiner: BTreeSet<u32> = dp
        .accumulator
        .as_ref()
        .map(|a| a.combiner.iter().copied().collect())
        .unwrap_or_default();
    let vertex = |s: &Src| match s {
        Src::Vertex { v, res: 0 } => dp.vertices.get(*v as usize),
        _ => None,
    };
    let mut out = Vec::new();
    for r in &dp.vertices {
        if r.op != "round_nearest_even" || r.args.len() != 1 {
            continue;
        }
        let Some(m) = vertex(&r.args[0].src) else { continue };
        if m.op != "multiply" || m.args.len() != 2 || m.stage != r.stage || n[m.id as usize] != 1 {
            continue;
        }
        let found = (0..2).find_map(|i| {
            let c = vertex(&m.args[i].src)?;
            (c.op == "convert" && c.args.len() == 1 && c.stage == r.stage && n[c.id as usize] == 1)
                .then(|| (c, m.args[1 - i].src.clone()))
        });
        let Some((c, scale)) = found else { continue };
        if [c.id, m.id, r.id].iter().any(|v| combiner.contains(v)) {
            continue;
        }
        let x = c.args[0].src.clone();
        let typed = dp.tuples.iter().all(|t| {
            let at = |w: &str| ty_of_width(t, w).ok();
            matches!(ty(dp, t, &x), Some(Ty::Int { signed: true, w }) if w <= RESCALE_MAX_BITS)
                && [&c.width, &m.width, &r.width].iter().all(|w| at(w) == Some(f32t))
                && ty(dp, t, &scale) == Some(f32t)
        });
        if typed {
            out.push(Site {
                convert: c.id,
                multiply: m.id,
                round: r.id,
                x,
                scale,
            });
        }
    }
    out
}

/// The integer's type of a site at every width tuple of its datapath.
pub fn site_types(dp: &Datapath, s: &Site) -> Vec<String> {
    let mut v: Vec<String> = dp
        .tuples
        .iter()
        .filter_map(|t| ty(dp, t, &s.x).map(|x| x.name()))
        .collect();
    v.sort();
    v.dedup();
    v
}

/// The datapath with every site rewritten: the round becomes `rescale(x, s)`, the convert and the
/// multiply are removed and the vertices renumbered. Returns the new datapath and, per old vertex id,
/// its new id (`None`: removed).
pub fn rewrite(dp: &Datapath, sites: &[Site]) -> Result<(Datapath, Vec<Option<u32>>), String> {
    let removed: BTreeSet<u32> = sites.iter().flat_map(|s| [s.convert, s.multiply]).collect();
    let by_round: BTreeMap<u32, &Site> = sites.iter().map(|s| (s.round, s)).collect();
    let mut map: Vec<Option<u32>> = Vec::with_capacity(dp.vertices.len());
    let mut next = 0u32;
    for v in &dp.vertices {
        if removed.contains(&v.id) {
            map.push(None);
        } else {
            map.push(Some(next));
            next += 1;
        }
    }
    let bad = std::cell::Cell::new(None);
    let re = |s: &Src| match s {
        Src::Vertex { v, res } => Src::Vertex {
            v: map.get(*v as usize).copied().flatten().unwrap_or_else(|| {
                bad.set(Some(*v));
                0
            }),
            res: *res,
        },
        other => other.clone(),
    };
    let mut out = dp.clone();
    out.vertices = dp
        .vertices
        .iter()
        .filter(|v| !removed.contains(&v.id))
        .map(|v| match by_round.get(&v.id) {
            Some(s) => Vertex {
                id: map[v.id as usize].unwrap_or_default(),
                label: "rescale@float".into(),
                op: "rescale".into(),
                family: "float".into(),
                width: v.width.clone(),
                attrs: BTreeMap::new(),
                stage: v.stage,
                args: vec![
                    Arg {
                        src: re(&s.x),
                        port: "p0".into(),
                    },
                    Arg {
                        src: re(&s.scale),
                        port: "p1".into(),
                    },
                ],
                from: v.from.clone(),
            },
            None => Vertex {
                id: map[v.id as usize].unwrap_or_default(),
                args: v
                    .args
                    .iter()
                    .map(|a| Arg {
                        src: re(&a.src),
                        port: a.port.clone(),
                    })
                    .collect(),
                ..v.clone()
            },
        })
        .collect();
    for o in &mut out.outputs {
        o.src = re(&o.src);
    }
    if let Some(a) = &mut out.accumulator {
        for e in &mut a.entries {
            e.next = re(&e.next);
            e.init = re(&e.init);
        }
        a.combiner = a.combiner.iter().filter_map(|v| map[*v as usize]).collect();
    }
    match bad.get() {
        Some(v) => Err(format!("{}: vertex {v} is removed but still read", dp.id)),
        None => Ok((out, map)),
    }
}

/// Below this magnitude `fixed` and `fp32` differ by at most 1 (the fp32 chain's two roundings stay
/// within 1/2 of the exact product there).
pub const TOLERANCE_BOUND: u32 = 1 << 22;
/// Above [`TOLERANCE_BOUND`], the fp32 result is at least this large with the same sign.
pub const SATURATION_BOUND: u32 = 1 << 21;

/// The exact-value format of the lemma (Float128): 113 significand bits hold every product `±x · M` (at
/// most 56 bits) and 15 exponent bits every `2^-k`.
const WIDE: cc_smt::term::Sort = cc_smt::term::Sort::Fp { e: 15, s: 113 };
const WIDE_BIAS: i64 = 16383;

/// `2^e` in [`WIDE`].
fn pow2(c: &mut cc_smt::term::Ctx, e: i64) -> Result<cc_smt::term::T, String> {
    let bits = ((WIDE_BIAS + e) as u128) << 112;
    cc_smt::float::lit(c, WIDE, bits).map_err(|x| x.0)
}

/// `x · 2^d` exactly for a finite [`WIDE`] value `x` whose exponent stays in the normal range (`d` a
/// 15-bit signed term): the exponent field moved by `d`, zero unchanged. No multiplier: an
/// `fp.mul` by a power of two is a full significand multiplier to a bit-blasting solver.
fn times_pow2(
    c: &mut cc_smt::term::Ctx,
    x: cc_smt::term::T,
    d: cc_smt::term::T,
) -> Result<cc_smt::term::T, String> {
    use cc_smt::term::Op;
    let e = |x: cc_smt::term::SortError| x.0;
    let b = c.bits_of(x).map_err(e)?;
    let sign = c.extract(b, 127, 127).map_err(e)?;
    let ex = c.extract(b, 126, 112).map_err(e)?;
    let frac = c.extract(b, 111, 0).map_err(e)?;
    let moved = c.op2(Op::BvAdd, ex, d).map_err(e)?;
    let lo = c.op2(Op::Concat, moved, frac).map_err(e)?;
    let bits = c.op2(Op::Concat, sign, lo).map_err(e)?;
    let y = c.app(Op::FpFromBits { e: 15, s: 113 }, &[bits]).map_err(e)?;
    let zero = c.op1(Op::FpIsZero, x).map_err(e)?;
    c.ite(zero, x, y).map_err(e)
}

/// `p · 2^-k` exactly in [`WIDE`] (`p` a signed bit-vector of at most 113 bits, `k` 10 signed bits in
/// -104..=149: the result stays normal).
fn scaled(
    c: &mut cc_smt::term::Ctx,
    p: cc_smt::term::T,
    k: cc_smt::term::T,
) -> Result<cc_smt::term::T, String> {
    use cc_smt::term::{Op, Rm};
    let e = |x: cc_smt::term::SortError| x.0;
    let pv = c
        .app(
            Op::FpFromSbv {
                e: 15,
                s: 113,
                rm: Rm::Rne,
            },
            &[p],
        )
        .map_err(e)?;
    let k15 = c.resize(k, 15, true).map_err(e)?;
    let d = c.op1(Op::BvNeg, k15).map_err(e)?;
    times_pow2(c, pv, d)
}

/// The tolerance lemma of `rescale` at one integer type (`lemmas/rescale_<type>.smt2`, expect `unsat`,
/// docs/passes/merge.md §4.11). Two parts, each refuted:
///
/// 1. the scale decode is exact: for every finite f32 `s`, `±M · 2^-k = s` (`cc_smt::ops::scale_parts`),
///    so `rescale(x, s)` is `rescale_shift(P, k)` of the exact product `x · s = P · 2^-k`;
/// 2. for every product `P` of the type (`|P| < 2^(n+23)`, and `P = 0` or `|P| >= 2^23` when `k < 149`:
///    the significand of a scale with exponent field `e >= 2` has its implicit bit), every shift `k` and
///    every f32 `p` that IEEE
///    754 allows for the fp32 chain's product, `|p - x·s| <= (|p| + |x·s|) · 2^-24 + 2^-150` (the
///    convert's and the multiply's roundings, each within half an ulp), an overflow to infinity of the
///    same sign, or a zero when `|x·s| < 2^-94` (the flush of design §6.4 and a subnormal scale read as
///    zero): with `Q = rescale_shift(P, k)` and `R = round_nearest_even(p)`, `|Q| < 2^22` implies
///    `|Q − R| <= 1`, and otherwise `R` has the sign of `Q` and `|R| >= 2^21` (any clip narrower than
///    that saturates both alike).
///
/// The product is a free variable: no multiplier is in the query.
pub fn lemma(int: Ty) -> Result<(String, String), String> {
    lemma_control(int, 1.0, 0)
}

/// [`lemma`] with another tolerance and the decoded shift moved by `k_offset`: negative controls (a
/// tolerance of 0, or a wrong decode, must have a counterexample: the hypotheses are satisfiable and
/// the claims are not vacuous).
pub fn lemma_control(int: Ty, tolerance: f32, k_offset: i64) -> Result<(String, String), String> {
    use cc_smt::ops::{self, Attrs, Val};
    use cc_smt::term::{Ctx, Op, Rm, Sort};
    let e = |x: cc_smt::term::SortError| x.0;
    let Ty::Int { signed: true, w: n } = int else {
        return Err(format!("rescale lemma: {} is not a signed integer", int.name()));
    };
    if n > RESCALE_MAX_BITS {
        return Err(format!(
            "rescale lemma: {} is wider than {RESCALE_MAX_BITS} bits",
            int.name()
        ));
    }
    let f32t = Ty::Float(Format::F32);
    let f32s = Sort::Fp { e: 8, s: 24 };
    let mut c = Ctx::new();
    let none = Attrs::default();

    // Part 1: the decode.
    let s = c.var("s", f32s).map_err(e)?;
    let (neg, mant, k) = ops::scale_parts(&mut c, s).map_err(e)?;
    let m26 = c.resize(mant, 26, false).map_err(e)?;
    let nm = c.op1(Op::BvNeg, m26).map_err(e)?;
    let sm = c.ite(neg, nm, m26).map_err(e)?;
    let k = if k_offset == 0 {
        k
    } else {
        let off = c.bvs(i128::from(k_offset), 10);
        c.op2(Op::BvAdd, k, off).map_err(e)?
    };
    let value = scaled(&mut c, sm, k)?;
    let sw = c
        .app(
            Op::FpToFp {
                e: 15,
                s: 113,
                rm: Rm::Rne,
            },
            &[s],
        )
        .map_err(e)?;
    let exact = c.op2(Op::FpEq, value, sw).map_err(e)?;
    let inf = c.op1(Op::FpIsInfinite, s).map_err(e)?;
    let nan = c.op1(Op::FpIsNaN, s).map_err(e)?;
    let special = c.or(&[inf, nan]).map_err(e)?;
    let finite = c.not(special).map_err(e)?;
    let wrong = c.not(exact).map_err(e)?;
    let decode_fails = c.and(&[finite, wrong]).map_err(e)?;

    // Part 2: the tolerance, for a free product and shift.
    let w = n + 25;
    let p = c.var("product", Sort::Bv(w)).map_err(e)?;
    let kk = c.var("k", Sort::Bv(10)).map_err(e)?;
    let lim = 1i128 << (n + 23);
    let hi = c.bvs(lim, w);
    let lo = c.bvs(-lim, w);
    let p_hi = c.op2(Op::BvSlt, p, hi).map_err(e)?;
    let p_lo = c.op2(Op::BvSlt, lo, p).map_err(e)?;
    let kmin = c.bvs(-104, 10);
    let kmax = c.bvs(149, 10);
    let k_lo = c.op2(Op::BvSle, kmin, kk).map_err(e)?;
    let k_hi = c.op2(Op::BvSle, kk, kmax).map_err(e)?;
    // A normal scale's significand is at least 2^23: a non-zero product is at least that large.
    let normal = c.op2(Op::BvSlt, kk, kmax).map_err(e)?;
    let m23 = c.bvs(1i128 << 23, w);
    let nm23 = c.bvs(-(1i128 << 23), w);
    let zero_w = c.bv(0, w);
    let p_zero = c.eq(p, zero_w).map_err(e)?;
    let p_big = c.op2(Op::BvSle, m23, p).map_err(e)?;
    let p_nbig = c.op2(Op::BvSle, p, nm23).map_err(e)?;
    let p_ok = c.or(&[p_zero, p_big, p_nbig]).map_err(e)?;
    let p_norm = c.app(Op::Implies, &[normal, p_ok]).map_err(e)?;
    let v = scaled(&mut c, p, kk)?;
    let fp = c.var("fp32_product", f32s).map_err(e)?;
    let pw = c
        .app(
            Op::FpToFp {
                e: 15,
                s: 113,
                rm: Rm::Rne,
            },
            &[fp],
        )
        .map_err(e)?;
    // Finite: within the bound (difference rounded toward zero, bound rounded up: implied by the exact
    // inequality).
    let d = c.op2(Op::FpSub(Rm::Rtz), pw, v).map_err(e)?;
    let da = c.op1(Op::FpAbs, d).map_err(e)?;
    let pa = c.op1(Op::FpAbs, pw).map_err(e)?;
    let va = c.op1(Op::FpAbs, v).map_err(e)?;
    let sum = c.op2(Op::FpAdd(Rm::Rtp), pa, va).map_err(e)?;
    let m24 = c.bvs(-24, 15);
    let rel = times_pow2(&mut c, sum, m24)?;
    let tiny = pow2(&mut c, -150)?;
    let bound = c.op2(Op::FpAdd(Rm::Rtp), rel, tiny).map_err(e)?;
    let within = c.op2(Op::FpLeq, da, bound).map_err(e)?;
    let p_inf = c.op1(Op::FpIsInfinite, fp).map_err(e)?;
    let p_nan = c.op1(Op::FpIsNaN, fp).map_err(e)?;
    let p_special = c.or(&[p_inf, p_nan]).map_err(e)?;
    let p_finite = c.not(p_special).map_err(e)?;
    let fin = c.and(&[p_finite, within]).map_err(e)?;
    // Overflow: only past 2^127, with the product's sign.
    let big = pow2(&mut c, 127)?;
    let huge = c.op2(Op::FpLeq, big, va).map_err(e)?;
    let pn = c.op1(Op::FpIsNegative, fp).map_err(e)?;
    let vn = c.op1(Op::FpIsNegative, v).map_err(e)?;
    let same_sign = c.eq(pn, vn).map_err(e)?;
    let ovf = c.and(&[p_inf, huge, same_sign]).map_err(e)?;
    // Flushed: a zero of either sign when |x·s| < 2^-94, which covers the f32 flush below 2^-126 and a
    // subnormal scale read as zero (|x·s| < 2^31 · 2^-126; design §6.4).
    let p_zero = c.op1(Op::FpIsZero, fp).map_err(e)?;
    let small_lim = pow2(&mut c, -94)?;
    let v_small = c.op2(Op::FpLt, va, small_lim).map_err(e)?;
    let flushed = c.and(&[p_zero, v_small]).map_err(e)?;
    let allowed = c.or(&[fin, ovf, flushed]).map_err(e)?;
    let r = ops::apply(&mut c, "round_nearest_even", &none, &[Val::new(f32t, fp)], f32t).map_err(e)?;
    let q = ops::rescale_shift(&mut c, p, n, kk).map_err(e)?;
    let lit = |c: &mut Ctx, v: f32| cc_smt::float::lit(c, f32s, u128::from(v.to_bits()));
    let tol = lit(&mut c, TOLERANCE_BOUND as f32).map_err(e)?;
    let edge = lit(&mut c, SATURATION_BOUND as f32).map_err(e)?;
    let nedge = lit(&mut c, -(SATURATION_BOUND as f32)).map_err(e)?;
    let one = lit(&mut c, tolerance).map_err(e)?;
    let zero = lit(&mut c, 0.0).map_err(e)?;
    let qa = c.op1(Op::FpAbs, q.t).map_err(e)?;
    let small = c.op2(Op::FpLt, qa, tol).map_err(e)?;
    let diff = c.op2(Op::FpSub(Rm::Rne), q.t, r.t).map_err(e)?;
    let dabs = c.op1(Op::FpAbs, diff).map_err(e)?;
    let close = c.op2(Op::FpLeq, dabs, one).map_err(e)?;
    let qpos = c.op2(Op::FpLt, zero, q.t).map_err(e)?;
    let rbig = c.op2(Op::FpLeq, edge, r.t).map_err(e)?;
    let rneg = c.op2(Op::FpLeq, r.t, nedge).map_err(e)?;
    let large = c.ite(qpos, rbig, rneg).map_err(e)?;
    let claim = c.ite(small, close, large).map_err(e)?;
    let broken = c.not(claim).map_err(e)?;
    let tolerance_fails = c
        .and(&[p_hi, p_lo, k_lo, k_hi, p_norm, allowed, broken])
        .map_err(e)?;

    let goal = c.or(&[decode_fails, tolerance_fails]).map_err(e)?;
    let mut sc = cc_smt::Script::new();
    let name = format!("rescale_{}", int.name());
    sc.comment(format!(
        "{name}: the scale decode is exact, and |rescale - fp32| <= 1 below 2^22, the same sign and at least 2^21 above (expect unsat)"
    ));
    c.render(&mut sc, &[goal]);
    Ok((format!("lemmas/{name}.smt2"), sc.render()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn v(id: u32, op: &str, w: &str, args: &[serde_json::Value]) -> serde_json::Value {
        let args: Vec<serde_json::Value> = args
            .iter()
            .enumerate()
            .map(|(i, s)| json!({"src": s, "port": format!("p{i}")}))
            .collect();
        json!({"id": id, "label": format!("{op}@x"), "op": op, "family": "x", "width": w, "attrs": {},
               "stage": "map", "args": args, "from": {"node": id, "access": null}})
    }

    fn vx(i: u32) -> serde_json::Value {
        json!({"vertex": {"v": i, "res": 0}})
    }

    fn inp(n: &str) -> serde_json::Value {
        json!({"input": {"name": n}})
    }

    /// `out0 = round(convert(x) · s) + zp`, `out1 = convert(x) · 2` (when `shared`, the convert has a
    /// second reader); x at `ty`.
    fn dp(ty: &str, shared: bool) -> Datapath {
        let mut vs = vec![
            v(0, "negate", "w0", &[inp("x")]),
            v(1, "convert", "w1", &[vx(0)]),
            v(2, "multiply", "w1", &[inp("s"), vx(1)]),
            v(3, "round_nearest_even", "w1", &[vx(2)]),
            v(4, "add", "w1", &[vx(3), inp("zp")]),
        ];
        let mut outs = vec![json!({"name": "out0", "kind": "output", "src": vx(4), "width": "w1"})];
        if shared {
            vs.push(v(5, "add", "w1", &[vx(1), vx(1)]));
            outs.push(json!({"name": "out1", "kind": "output", "src": vx(5), "width": "w1"}));
        }
        let input = |n: &str, w: &str| json!({"name": n, "kind": "map", "node": 0, "res": 0, "incidence": [], "width": w});
        serde_json::from_value(json!({
            "schema": "cc.datapaths.datapath/1", "id": "z", "route": "datapath", "class": "map", "core": "",
            "widths": {}, "vertices": vs,
            "inputs": [input("x", "w0"), input("s", "w1"), input("zp", "w1")],
            "outputs": outs, "accumulator": null,
            "tuples": [{"name": "t0", "elems": [ty, "f32"], "consts": {}, "attrs": {}, "rows": 1}]
        }))
        .unwrap()
    }

    #[test]
    fn a_rescaling_chain_is_found_and_rewritten() {
        let d = dp("i32", false);
        let s = sites(&d);
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].convert, s[0].multiply, s[0].round), (1, 2, 3));
        assert_eq!(s[0].x, Src::Vertex { v: 0, res: 0 });
        assert_eq!(s[0].scale, Src::Input { name: "s".into() });
        assert_eq!(site_types(&d, &s[0]), vec!["i32".to_string()]);
        let (n, map) = rewrite(&d, &s).unwrap();
        assert_eq!(map, vec![Some(0), None, None, Some(1), Some(2)]);
        assert_eq!(n.vertices.len(), 3);
        let r = &n.vertices[1];
        assert_eq!((r.id, r.op.as_str()), (1, "rescale"));
        let args: Vec<&Src> = r.args.iter().map(|a| &a.src).collect();
        assert_eq!(
            args,
            vec![&Src::Vertex { v: 0, res: 0 }, &Src::Input { name: "s".into() }]
        );
        assert_eq!(n.vertices[2].args[0].src, Src::Vertex { v: 1, res: 0 });
        assert_eq!(n.outputs[0].src, Src::Vertex { v: 2, res: 0 });
    }

    #[test]
    fn shared_converts_unsigned_and_wide_integers_are_not_chains() {
        assert!(sites(&dp("i32", true)).is_empty());
        assert!(sites(&dp("ui32", false)).is_empty());
        assert!(sites(&dp("i64", false)).is_empty());
        assert_eq!(sites(&dp("i8", false)).len(), 1);
    }

    #[test]
    fn the_lemma_has_no_multiplier() {
        let (f, text) = lemma(Ty::Int { signed: true, w: 32 }).unwrap();
        assert_eq!(f, "lemmas/rescale_i32.smt2");
        assert!(!text.contains("bvmul") && !text.contains("fp.mul"), "{text}");
        assert!(lemma(Ty::Int { signed: false, w: 8 }).is_err());
    }
}
