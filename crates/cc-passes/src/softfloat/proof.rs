//! The proof query of a primitive: its optimized XLS IR (`fu.opt.ir`) against the op's semantics.
//!
//! The circuit is translated by `cc_smt::xls`; the claim is `ops::same(spec, circuit)` for every input
//! (SMT `=` on IEEE sorts: NaN is one class, +0 and -0 differ; bit patterns with NaN as one class for the
//! carrier formats). The query asserts the claim's negation, so `unsat` proves the primitive.
//!
//! **Multiply** in IEEE formats (f16, f32): a direct query is multiplier equivalence between the
//! circuit's integer product and the solver's own FP multiplier, which bit-blasting does not finish. The
//! circuit's product is therefore an uninterpreted function `umul_<wa>_<wb>_<wr>` of its operands, and the
//! spec is IEEE 754's definition of multiplication (the exact product, rounded once): the same function of
//! the operands' significands, normalized exactly into a Float64 value (a priority chain finds the leading
//! one), times the exact power of two, rounded by `to_fp RNE`. The only fact about the function the
//! query assumes is true of the integer product: it is zero exactly when a factor is zero. A query proven
//! for every such function holds for the product, and the spec equals `fp.mul RNE` by the SMT-LIB
//! definition of `fp.mul` (the exact result, rounded).

use cc_smt::Script;
use cc_smt::float::{self, Format};
use cc_smt::ops::{self, Attrs, Ty, Val};
use cc_smt::term::{Ctx, Op, Rm, Sort, SortError, T};
use cc_smt::xls::{self, Multiply};

/// The query of one primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    pub text: String,
    pub product_abstracted: bool,
}

pub fn bits(t: Ty) -> u32 {
    match t {
        Ty::Bool => 1,
        Ty::Int { w, .. } => w,
        Ty::Float(f) | Ty::Complex(f) => f.bits(),
    }
}

fn e(x: SortError) -> String {
    x.0
}

/// The value of type `t` a circuit port carries (its bit pattern).
fn val_of(c: &mut Ctx, t: Ty, b: T) -> Result<Val, String> {
    Ok(match t {
        Ty::Bool => {
            let one = c.bv(1, 1);
            Val::new(t, c.eq(b, one).map_err(e)?)
        }
        Ty::Float(f) if f.native() => {
            Val::new(t, c.app(Op::FpFromBits { e: f.e, s: f.m + 1 }, &[b]).map_err(e)?)
        }
        _ => Val::new(t, b),
    })
}

/// The circuit's multiplier when it is the significand product the abstraction expects: one `umul` of two
/// `m + 1`-bit operands into at least `2m + 2` bits that fit a Float64 significand.
fn abstractable_product(f: &xls::Function, fmt: Format) -> Option<u32> {
    let muls: Vec<&xls::Node> = f
        .nodes
        .iter()
        .filter(|n| n.op == "umul" || n.op == "smul")
        .collect();
    let [n] = muls.as_slice() else { return None };
    if n.op != "umul" {
        return None;
    }
    let width = |name: &str| {
        f.params
            .iter()
            .find(|(p, _)| p == name)
            .map(|(_, w)| *w)
            .or_else(|| f.nodes.iter().find(|x| x.name == name).map(|x| x.width))
    };
    let (wa, wb) = (width(&n.operands[0])?, width(&n.operands[1])?);
    let m = fmt.m;
    (wa == m + 1 && wb == m + 1 && n.width >= 2 * m + 2 && n.width <= 53).then_some(n.width)
}

/// IEEE multiplication of the patterns `a`, `b` of `fmt` with the significand product the uninterpreted
/// `umul` (see the module comment); returns the value and the assumption about the function.
fn multiply_spec(c: &mut Ctx, fmt: Format, a: T, b: T, pw: u32) -> Result<(T, T), String> {
    let (eb, m, w) = (fmt.e, fmt.m, fmt.bits());
    let fs = fmt.fp_sort();
    let from = Op::FpFromBits { e: eb, s: m + 1 };
    let fa = c.app(from.clone(), &[a]).map_err(e)?;
    let fb = c.app(from, &[b]).map_err(e)?;
    const XW: u32 = 13;
    let decode = |c: &mut Ctx, v: T| -> Result<(T, T), String> {
        let ef = c.extract(v, w - 2, m).map_err(e)?;
        let fr = c.extract(v, m - 1, 0).map_err(e)?;
        let z0 = c.bv(0, eb);
        let sub = c.eq(ef, z0).map_err(e)?;
        let (h0, h1) = (c.bv(0, 1), c.bv(1, 1));
        let hid = c.ite(sub, h0, h1).map_err(e)?;
        let mant = c.app(Op::Concat, &[hid, fr]).map_err(e)?;
        let one = c.bv(1, XW);
        let ex = c.resize(ef, XW, false).map_err(e)?;
        let x = c.ite(sub, one, ex).map_err(e)?;
        Ok((mant, x))
    };
    let (ma, xa) = decode(c, a)?;
    let (mb, xb) = decode(c, b)?;
    let name = xls::umul_name(m + 1, m + 1, pw);
    c.fun(&name, &[Sort::Bv(m + 1), Sort::Bv(m + 1)], Sort::Bv(pw))
        .map_err(e)?;
    let p = c.app(Op::Uf(name), &[ma, mb]).map_err(e)?;
    // The product of naturals is zero exactly when a factor is.
    let pz0 = c.bv(0, pw);
    let pz = c.eq(p, pz0).map_err(e)?;
    let mz0 = c.bv(0, m + 1);
    let az = c.eq(ma, mz0).map_err(e)?;
    let bz = c.eq(mb, mz0).map_err(e)?;
    let fz = c.or(&[az, bz]).map_err(e)?;
    let axiom = c.eq(pz, fz).map_err(e)?;
    // Leading-zero count of the product by a priority chain, then the exact normalization.
    let mut lz = c.bv(u128::from(pw), XW);
    for i in 0..pw {
        let bit = c.extract(p, i, i).map_err(e)?;
        let one = c.bv(1, 1);
        let set = c.eq(bit, one).map_err(e)?;
        let k = c.bv(u128::from(pw - 1 - i), XW);
        lz = c.ite(set, k, lz).map_err(e)?;
    }
    let lzp = c.resize(lz, pw, false).map_err(e)?;
    let pn = c.op2(Op::BvShl, p, lzp).map_err(e)?;
    // Unbiased exponent of the leading one: xa + xb + (pw - 1) - lz - (2m + 2 bias).
    let sum = c.op2(Op::BvAdd, xa, xb).map_err(e)?;
    let top = c.bv(u128::from(pw - 1), XW);
    let sum = c.op2(Op::BvAdd, sum, top).map_err(e)?;
    let sum = c.op2(Op::BvSub, sum, lz).map_err(e)?;
    let off = c.bv((2 * i64::from(m) + 2 * fmt.bias()) as u128, XW);
    let ue = c.op2(Op::BvSub, sum, off).map_err(e)?;
    let b1023 = c.bv(1023, XW);
    let be = c.op2(Op::BvAdd, ue, b1023).map_err(e)?;
    let be = c.extract(be, 10, 0).map_err(e)?;
    let sa = c.extract(a, w - 1, w - 1).map_err(e)?;
    let sb = c.extract(b, w - 1, w - 1).map_err(e)?;
    let sgn = c.op2(Op::BvXor, sa, sb).map_err(e)?;
    let frac = c.extract(pn, pw - 2, 0).map_err(e)?;
    let pad = 52 - (pw - 1);
    let frac = if pad > 0 {
        let z = c.bv(0, pad);
        c.app(Op::Concat, &[frac, z]).map_err(e)?
    } else {
        frac
    };
    let hi = c.app(Op::Concat, &[sgn, be]).map_err(e)?;
    let wbits = c.app(Op::Concat, &[hi, frac]).map_err(e)?;
    let ws = Sort::Fp { e: 11, s: 53 };
    let wide = c.app(Op::FpFromBits { e: 11, s: 53 }, &[wbits]).map_err(e)?;
    let one1 = c.bv(1, 1);
    let neg = c.eq(sgn, one1).map_err(e)?;
    let pzero = float::lit(c, ws, 0).map_err(e)?;
    let nzero = c.op1(Op::FpNeg, pzero).map_err(e)?;
    let zero = c.ite(neg, nzero, pzero).map_err(e)?;
    let wide = c.ite(pz, zero, wide).map_err(e)?;
    let rounded = c
        .app(
            Op::FpToFp {
                e: eb,
                s: m + 1,
                rm: Rm::Rne,
            },
            &[wide],
        )
        .map_err(e)?;
    let an = c.op1(Op::FpIsNaN, fa).map_err(e)?;
    let bn = c.op1(Op::FpIsNaN, fb).map_err(e)?;
    let nan = c.or(&[an, bn]).map_err(e)?;
    let ai = c.op1(Op::FpIsInfinite, fa).map_err(e)?;
    let bi = c.op1(Op::FpIsInfinite, fb).map_err(e)?;
    let inf = c.or(&[ai, bi]).map_err(e)?;
    let azr = c.op1(Op::FpIsZero, fa).map_err(e)?;
    let bzr = c.op1(Op::FpIsZero, fb).map_err(e)?;
    let zero_in = c.or(&[azr, bzr]).map_err(e)?;
    let qnan = float::nan(c, fs).map_err(e)?;
    let pinf = float::lit(c, fs, ((1u128 << eb) - 1) << m).map_err(e)?;
    let ninf = c.op1(Op::FpNeg, pinf).map_err(e)?;
    let sinf = c.ite(neg, ninf, pinf).map_err(e)?;
    let inf_case = c.ite(zero_in, qnan, sinf).map_err(e)?;
    let r = c.ite(inf, inf_case, rounded).map_err(e)?;
    let r = c.ite(nan, qnan, r).map_err(e)?;
    Ok((r, axiom))
}

/// The query of primitive `key` (op `op` on `args` to `ret` with `attrs`) for the IR text `ir`.
pub fn query(key: &str, op: &str, args: &[Ty], ret: Ty, attrs: &Attrs, ir: &str) -> Result<Query, String> {
    let f = xls::parse(ir)?;
    if f.params.len() != args.len() || f.params.iter().zip(args).any(|((_, w), t)| *w != bits(*t)) {
        return Err(format!(
            "{key}: the IR's parameters {:?} do not match the operand types",
            f.params
        ));
    }
    if f.ret_width != bits(ret) {
        return Err(format!("{key}: the IR returns bits[{}]", f.ret_width));
    }
    let mut c = Ctx::new();
    let xs: Vec<T> = (0..args.len())
        .map(|i| c.var(&format!("x{i}"), Sort::Bv(bits(args[i]))).map_err(e))
        .collect::<Result<_, _>>()?;
    let product = match (op, ret) {
        ("multiply", Ty::Float(fmt)) if fmt.native() => abstractable_product(&f, fmt).map(|pw| (fmt, pw)),
        _ => None,
    };
    let mode = if product.is_some() {
        Multiply::Uninterpreted
    } else {
        Multiply::Exact
    };
    let circ = f.translate(&mut c, &xs, mode)?;
    let circ = val_of(&mut c, ret, circ)?;
    let mut assume = Vec::new();
    let spec = match product {
        Some((fmt, pw)) => {
            let (v, axiom) = multiply_spec(&mut c, fmt, xs[0], xs[1], pw)?;
            assume.push(axiom);
            Val::new(ret, v)
        }
        None => {
            let vals: Vec<Val> = args
                .iter()
                .zip(&xs)
                .map(|(&t, &x)| val_of(&mut c, t, x))
                .collect::<Result<_, _>>()?;
            ops::apply(&mut c, op, attrs, &vals, ret).map_err(e)?
        }
    };
    let claim = ops::same(&mut c, spec, circ).map_err(e)?;
    let broken = c.not(claim).map_err(e)?;
    assume.push(broken);
    let goal = if assume.len() == 1 {
        broken
    } else {
        c.and(&assume).map_err(e)?
    };
    let mut sc = Script::new();
    sc.comment(format!(
        "{key}: the optimized XLS IR equals the op's semantics for every input{} (expect unsat)",
        if product.is_some() {
            " (significand product abstracted)"
        } else {
            ""
        }
    ));
    c.render(&mut sc, &[goal]);
    Ok(Query {
        text: sc.render(),
        product_abstracted: product.is_some(),
    })
}
