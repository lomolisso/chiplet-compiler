//! Proofs of the glue (docs/framework/float.md, "Proofs"; design §4.12, §5).
//!
//! The wrapper's Verilog is read by Yosys with each core replaced by a blackbox, every core instance
//! cut (`cutpoint`: its result becomes a free value) and the design written as SMT-LIB 2 (`write_smt2
//! -stbv -wires`: the state, inputs and free values included, is one bit-vector). The query adds:
//!
//! - each core's contract on its wires: a correctly rounded core's result is the IEEE operation at the
//!   core's own widths (gradual underflow) on its operands; a conversion core's result is its operand
//!   rounded to nearest even; a less-than core's bit is the quiet `fp.lt`; a float-to-integer core's
//!   result is `cc_smt::ops::convert` for every operand that is not NaN; an approximate core is free;
//! - the negation of the claim: the wrapper's output differs from `cc_smt::ops::apply` for the format
//!   (design §6.4 as encoded on `main`), or, for an approximate operator, from the core's result rounded
//!   to the format (`flush`, saturation and NaN rules included), or the core's operands are not the
//!   format's operands (subnormals read as zero) exactly widened.
//!
//! `unsat` proves the wrapper. f32 multiply, divide and square root leave both the core's and the
//! specification's arithmetic out: each is the rounding of one value `v`, the exact result, constrained
//! to the binade range the operands give (as the encoding lemmas of `cc_smt` do); a solver relating a
//! 24-bit multiplier of one sort to one of another does not finish (Z3: no answer in 5 minutes, where
//! these queries take seconds). The narrower formats keep both operations: there the specification
//! rounds twice (to Float32, then to the format), which equals one rounding only for actual products,
//! quotients and roots, not for every `v`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use cc_core::runner::{BoundRunner, Caps, Invocation, Mount};
use cc_smt::float::{self, Arith, Flavour, Format};
use cc_smt::ops::{self, Attrs, Ty, Val};
use cc_smt::term::{Ctx, Op, Rm, Sort, T};
use cc_smt::{Cache, Location, Script, Solver, Verdict};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::FloatError;
use crate::compose::Recipe;
use crate::glue::{Built, Glue, Role, SignOp};
use crate::rtl::Mutation;

type Res<X> = Result<X, FloatError>;

const F64: Sort = Sort::Fp { e: 11, s: 53 };

/// The Yosys script that writes a wrapper as SMT-LIB with its cores cut.
pub fn yosys_script(b: &Built) -> String {
    let m = b.rtl.name();
    let cells: Vec<String> = b
        .rtl
        .core_wires()
        .iter()
        .map(|w| format!("{m}/{}", w.cell))
        .collect();
    let mut s = String::new();
    s.push_str("read_verilog -sv stubs.v\nread_verilog -sv glue.v\n");
    s.push_str(&format!("hierarchy -check -top {m}\nproc\nopt_clean\n"));
    if !cells.is_empty() {
        s.push_str(&format!("cutpoint {}\n", cells.join(" ")));
    }
    s.push_str("opt_clean\ncheck -assert\nwrite_smt2 -stbv -wires glue.smt2\n");
    s
}

/// The exact value of an encoding of `f` as an FP term (native: itself; IEEE carriers: their own
/// sort; E4M3FN: its Float32 decoding).
fn value(c: &mut Ctx, f: Format, t: T) -> Res<T> {
    Ok(if f.native() {
        t
    } else if f.flavour == Flavour::Ieee {
        c.app(Op::FpFromBits { e: f.e, s: f.m + 1 }, &[t])?
    } else {
        float::to_f32(c, f, t)?
    })
}

/// The exact value of a bit pattern of `f` as an FP term.
fn bits_value(c: &mut Ctx, f: Format, b: T) -> Res<T> {
    if f.native() {
        Ok(c.app(Op::FpFromBits { e: f.e, s: f.m + 1 }, &[b])?)
    } else {
        value(c, f, b)
    }
}

fn to_sort(c: &mut Ctx, s: Sort, x: T) -> Res<T> {
    if c.sort(x) == s {
        return Ok(x);
    }
    let Sort::Fp { e, s } = s else {
        return Err(FloatError::Proof("rounding into a non-FP sort".into()));
    };
    Ok(c.app(Op::FpToFp { e, s, rm: Rm::Rne }, &[x])?)
}

fn sort_of(f: Format) -> Sort {
    Sort::Fp { e: f.e, s: f.m + 1 }
}

/// A core's IEEE operation at its own widths.
fn ieee_op(c: &mut Ctx, op: &str, xs: &[T]) -> Res<T> {
    let rm = Rm::Rne;
    Ok(match (op, xs) {
        ("add", [x, y]) => c.op2(Op::FpAdd(rm), *x, *y)?,
        ("sub", [x, y]) => c.op2(Op::FpSub(rm), *x, *y)?,
        ("mul", [x, y]) => c.op2(Op::FpMul(rm), *x, *y)?,
        ("div", [x, y]) => c.op2(Op::FpDiv(rm), *x, *y)?,
        ("sqrt", [x]) => c.op1(Op::FpSqrt(rm), *x)?,
        _ => return Err(FloatError::Proof(format!("no IEEE operation {op}"))),
    })
}

fn catalog_name(op: Arith) -> &'static str {
    match op {
        Arith::Add => "add",
        Arith::Sub => "subtract",
        Arith::Mul => "multiply",
        Arith::Div => "divide",
        Arith::Sqrt => "sqrt",
    }
}

/// Whether a wrapper's arithmetic is stated through the exact value (see the module docs).
fn exact_value_form(g: &Glue) -> Option<Arith> {
    match g {
        Glue::Arith { op, fmt }
            if *fmt == Format::F32 && matches!(op, Arith::Mul | Arith::Div | Arith::Sqrt) =>
        {
            Some(*op)
        }
        _ => None,
    }
}

/// `2^k` as a Float64, `k` a 12-bit signed term in the normal range.
fn pow2_f64(c: &mut Ctx, k: T) -> Res<T> {
    let b = c.bv(1023, 12);
    let field = c.op2(Op::BvAdd, k, b)?;
    let field = c.extract(field, 10, 0)?;
    let z1 = c.bv(0, 1);
    let z52 = c.bv(0, 52);
    let hi = c.op2(Op::Concat, z1, field)?;
    let bits = c.op2(Op::Concat, hi, z52)?;
    Ok(c.app(Op::FpFromBits { e: 11, s: 53 }, &[bits])?)
}

/// The unbiased exponent of a normal Float64, as a 12-bit term.
fn exp_f64(c: &mut Ctx, x: T) -> Res<T> {
    let b = c.bits_of(x)?;
    let e = c.extract(b, 62, 52)?;
    let e = c.resize(e, 12, false)?;
    let bias = c.bv(1023, 12);
    Ok(c.op2(Op::BvSub, e, bias)?)
}

/// The constraint on `v`, the exact result of `op` on the Float64 values `xs`: IEEE's special results,
/// else the sign and the binade range the operands' exponents give.
fn exact_value(c: &mut Ctx, op: Arith, xs: &[T], v: T) -> Res<T> {
    let isnan = |c: &mut Ctx, x: T| c.op1(Op::FpIsNaN, x);
    let isinf = |c: &mut Ctx, x: T| c.op1(Op::FpIsInfinite, x);
    let iszero = |c: &mut Ctx, x: T| c.op1(Op::FpIsZero, x);
    let neg = |c: &mut Ctx, x: T| c.op1(Op::FpIsNegative, x);
    let pinf = float::lit(c, F64, 0x7ff0_0000_0000_0000)?;
    let ninf = c.op1(Op::FpNeg, pinf)?;
    let pz = float::lit(c, F64, 0)?;
    let nz = c.op1(Op::FpNeg, pz)?;
    let (nan, inf, zero, sign, lo, hi, lo_closed) = match (op, xs) {
        (Arith::Mul | Arith::Div, [x, y]) => {
            let (nx, ny) = (isnan(c, *x)?, isnan(c, *y)?);
            let (ix, iy) = (isinf(c, *x)?, isinf(c, *y)?);
            let (zx, zy) = (iszero(c, *x)?, iszero(c, *y)?);
            let (sx, sy) = (neg(c, *x)?, neg(c, *y)?);
            let sign = c.app(Op::Xor, &[sx, sy])?;
            let (ex, ey) = (exp_f64(c, *x)?, exp_f64(c, *y)?);
            if op == Arith::Mul {
                let a = c.and(&[zx, iy])?;
                let b = c.and(&[ix, zy])?;
                let nan = c.or(&[nx, ny, a, b])?;
                let inf = c.or(&[ix, iy])?;
                let zero = c.or(&[zx, zy])?;
                let k = c.op2(Op::BvAdd, ex, ey)?;
                let two = c.bv(2, 12);
                let k2 = c.op2(Op::BvAdd, k, two)?;
                (nan, inf, zero, sign, k, k2, true)
            } else {
                let a = c.and(&[zx, zy])?;
                let b = c.and(&[ix, iy])?;
                let nan = c.or(&[nx, ny, a, b])?;
                let inf = c.or(&[ix, zy])?;
                let zero = c.or(&[zx, iy])?;
                let d = c.op2(Op::BvSub, ex, ey)?;
                let one = c.bv(1, 12);
                let lo = c.op2(Op::BvSub, d, one)?;
                let hi = c.op2(Op::BvAdd, d, one)?;
                (nan, inf, zero, sign, lo, hi, false)
            }
        }
        (Arith::Sqrt, [x]) => {
            let nx = isnan(c, *x)?;
            let sx = neg(c, *x)?;
            let zx = iszero(c, *x)?;
            let nzx = c.not(zx)?;
            let negx = c.and(&[sx, nzx])?;
            let nan = c.or(&[nx, negx])?;
            let inf = isinf(c, *x)?;
            let ex = exp_f64(c, *x)?;
            let one = c.bv(1, 12);
            let half = c.op2(Op::BvAshr, ex, one)?;
            let hi = c.op2(Op::BvAdd, half, one)?;
            // sqrt(±0) = ±0: the sign of the zero.
            (nan, inf, zx, sx, half, hi, true)
        }
        _ => return Err(FloatError::Proof(format!("{}: no exact value", op.name()))),
    };
    let vnan = isnan(c, v)?;
    let sinf = c.ite(sign, ninf, pinf)?;
    let szero = c.ite(sign, nz, pz)?;
    let vinf = c.eq(v, sinf)?;
    let vzero = c.eq(v, szero)?;
    let l = pow2_f64(c, lo)?;
    let h = pow2_f64(c, hi)?;
    let av = c.op1(Op::FpAbs, v)?;
    let above = if lo_closed {
        c.op2(Op::FpLeq, l, av)?
    } else {
        c.op2(Op::FpLt, l, av)?
    };
    let below = c.op2(Op::FpLt, av, h)?;
    let vs = neg(c, v)?;
    let same_sign = c.eq(vs, sign)?;
    let vnan_not = c.not(vnan)?;
    let finite = c.and(&[vnan_not, above, below, same_sign])?;
    let r = c.ite(zero, vzero, finite)?;
    let r = c.ite(inf, vinf, r)?;
    Ok(c.ite(nan, vnan, r)?)
}

/// The value of a result of the core format `c` rounded to `f` as an arithmetic result (design §6.4),
/// as `f`'s encoding: FTZ below `2^emin` keeping the sign for a flushing `f` whose core has one more
/// exponent bit (no rounding happens), else `cc_smt`'s rounding (Float32 to a carrier: `from_f32`;
/// to a native format: the theory's rounding).
fn round_result(c: &mut Ctx, core: Format, f: Format, z: T) -> Res<T> {
    let y = c.app(
        Op::FpFromBits {
            e: core.e,
            s: core.m + 1,
        },
        &[z],
    )?;
    if core == f {
        return if f.native() { Ok(y) } else { Ok(z) };
    }
    if f.flushes() && core == crate::glue::wide(f) {
        let emin = f.emin();
        let bias = (1i64 << (core.e - 1)) - 1;
        let min = float::lit(c, sort_of(core), ((emin + bias) as u128) << core.m)?;
        let ay = c.op1(Op::FpAbs, y)?;
        let tiny = c.op2(Op::FpLt, ay, min)?;
        let pz = float::lit(c, sort_of(core), 0)?;
        let nz = c.op1(Op::FpNeg, pz)?;
        let neg = c.op1(Op::FpIsNegative, y)?;
        let z0 = c.ite(neg, nz, pz)?;
        let y = c.ite(tiny, z0, y)?;
        let r = to_sort(c, sort_of(f), y)?;
        // A carrier's encoding is the narrowed value's bits.
        return if f.native() { Ok(r) } else { Ok(c.bits_of(r)?) };
    }
    if core == Format::F32 {
        return if f.native() {
            to_sort(c, sort_of(f), y)
        } else {
            Ok(float::from_f32(c, f, y)?)
        };
    }
    Err(FloatError::Proof(format!(
        "no rounding of {} results to {}",
        core.name(),
        f.name()
    )))
}

/// The value of a composition (design §4.12) on the format's operands, with its cores as the functions
/// `core<k>` of their operands' encodings (in the order of `recipe::needs`): the special values of C11
/// Annex F (F.10.3.9 log1p, F.10.4.4 pow, F.10.2.6 tanh) and IEEE 754-2019 §9.2 (rSqrt), else the
/// recipe over the cores rounded once to the format as an arithmetic result. Stated on FP values with
/// the theory's predicates, independently of the glue's bit tests; an FP8 result that would be infinite
/// is NaN (design §6.4).
fn compose_spec(c: &mut Ctx, rec: Recipe, f: Format, q: u32, args: &[Val]) -> Res<T> {
    let qf = crate::recipe::inner(f, q);
    let sq = sort_of(qf);
    let mut xs = Vec::new();
    for a in args {
        let d = float::daz(c, f, a.t)?;
        xs.push(value(c, f, d)?);
    }
    let sf = c.sort(xs[0]);
    let x = xs[0];
    let fl = |c: &mut Ctx, s: Sort, v: f64| -> Res<T> {
        let Sort::Fp { e, s: sb } = s else {
            return Err(FloatError::Proof("a literal of a non-FP sort".into()));
        };
        Ok(float::lit(
            c,
            s,
            crate::recipe::encode(Format::ieee(e, sb - 1), v, false),
        )?)
    };
    let pz = fl(c, sf, 0.0)?;
    let nz = c.op1(Op::FpNeg, pz)?;
    let nan = float::nan(c, sf)?;
    let inf = |c: &mut Ctx, neg: bool| -> Res<T> {
        if f.flavour == Flavour::FiniteNan {
            return Ok(nan);
        }
        let Sort::Fp { e, s: sb } = sf else {
            return Err(FloatError::Proof("non-FP sort".into()));
        };
        let p = float::lit(c, sf, ((1u128 << e) - 1) << (sb - 1))?;
        Ok(if neg { c.op1(Op::FpNeg, p)? } else { p })
    };
    let one = fl(c, sf, 1.0)?;
    let one_q = c.bv(crate::recipe::encode(qf, 1.0, false), qf.bits());
    let uf = |c: &mut Ctx, k: usize, ops: &[T]| -> Res<T> {
        let name = format!("core{k}");
        let sorts: Vec<Sort> = ops.iter().map(|&t| c.sort(t)).collect();
        c.fun(&name, &sorts, Sort::Bv(qf.bits()))?;
        Ok(c.app(Op::Uf(name), ops)?)
    };
    // The encoding in Q of a value (exact: Q holds every value of the format).
    let enc_q = |c: &mut Ctx, v: T| -> Res<T> {
        let w = to_sort(c, sq, v)?;
        Ok(c.bits_of(w)?)
    };
    // One rounding of a Q encoding to the format, as an arithmetic result, as a value of `sf`.
    let round = |c: &mut Ctx, bits: T| -> Res<T> {
        let v = c.app(Op::FpFromBits { e: qf.e, s: qf.m + 1 }, &[bits])?;
        if f.flavour == Flavour::FiniteNan {
            let v32 = to_sort(c, Sort::Fp { e: 8, s: 24 }, v)?;
            let b = float::from_f32(c, f, v32)?;
            return value(c, f, b);
        }
        if f.flushes() {
            let wf = crate::glue::wide(f);
            let y = to_sort(c, sort_of(wf), v)?;
            let min = float::lit(c, sort_of(wf), ((f.emin() + wf.bias()) as u128) << wf.m)?;
            let ay = c.op1(Op::FpAbs, y)?;
            let tiny = c.op2(Op::FpLt, ay, min)?;
            let wz = float::lit(c, sort_of(wf), 0)?;
            let wnz = c.op1(Op::FpNeg, wz)?;
            let neg = c.op1(Op::FpIsNegative, y)?;
            let z0 = c.ite(neg, wnz, wz)?;
            let y = c.ite(tiny, z0, y)?;
            return to_sort(c, sf, y);
        }
        to_sort(c, sf, v)
    };
    let copysign = |c: &mut Ctx, r: T, neg: T| -> Res<T> {
        let a = c.op1(Op::FpAbs, r)?;
        let n = c.op1(Op::FpNeg, a)?;
        Ok(c.ite(neg, n, a)?)
    };
    let is = |c: &mut Ctx, o: Op, x: T| c.op1(o, x);
    let xnan = is(c, Op::FpIsNaN, x)?;
    let xinf = is(c, Op::FpIsInfinite, x)?;
    let xzero = is(c, Op::FpIsZero, x)?;
    let xneg = is(c, Op::FpIsNegative, x)?;
    let ax = c.op1(Op::FpAbs, x)?;
    Ok(match rec {
        Recipe::Power | Recipe::PowerCore => {
            let y = xs[1];
            let ynan = is(c, Op::FpIsNaN, y)?;
            let yinf = is(c, Op::FpIsInfinite, y)?;
            let yzero = is(c, Op::FpIsZero, y)?;
            let yneg = is(c, Op::FpIsNegative, y)?;
            // Integer: equal to its truncation (finite). Odd: an integer whose half is not one, the
            // half taken exactly on the exponent field for |y| >= 2.
            let yfin = {
                let a = c.or(&[ynan, yinf])?;
                c.not(a)?
            };
            let ty = c.app(Op::FpRoundToIntegral(Rm::Rtz), &[y])?;
            let yeq = c.op2(Op::FpEq, ty, y)?;
            let yint = c.and(&[yfin, yeq])?;
            let ay = c.op1(Op::FpAbs, y)?;
            let two = fl(c, sf, 2.0)?;
            let ge2 = c.op2(Op::FpLeq, two, ay)?;
            let yb = c.bits_of(ay)?;
            let Sort::Fp { e: se, s: ss } = sf else {
                return Err(FloatError::Proof("non-FP sort".into()));
            };
            let w = se + ss;
            let ex = c.extract(yb, w - 2, ss - 1)?;
            let one_e = c.bv(1, se);
            let exm = c.op2(Op::BvSub, ex, one_e)?;
            let man = c.extract(yb, ss - 2, 0)?;
            let hb = c.op2(Op::Concat, exm, man)?;
            let z1 = c.bv(0, 1);
            let hb = c.op2(Op::Concat, z1, hb)?;
            let half = c.app(Op::FpFromBits { e: se, s: ss }, &[hb])?;
            let th = c.app(Op::FpRoundToIntegral(Rm::Rtz), &[half])?;
            let hint = c.op2(Op::FpEq, th, half)?;
            let hnot = c.not(hint)?;
            let one_y = c.op2(Op::FpEq, ay, one)?;
            let big_odd = c.and(&[ge2, hnot])?;
            let oddish = c.or(&[one_y, big_odd])?;
            let yodd = c.and(&[yint, oddish])?;
            let xone = c.op2(Op::FpEq, x, one)?;
            let xnotz = c.not(xzero)?;
            let xone_pos = c.and(&[xone, xnotz])?;
            let lt1 = c.op2(Op::FpLt, ax, one)?;
            let ax1 = c.op2(Op::FpEq, ax, one)?;
            // Core path on |x| and y.
            let xq = enc_q(c, ax)?;
            let yq = enc_q(c, y)?;
            let ex2 = if rec == Recipe::PowerCore {
                uf(c, 0, &[xq, yq])?
            } else {
                let lg = uf(c, 0, &[xq])?;
                let t = uf(c, 1, &[yq, lg])?;
                uf(c, 2, &[t])?
            };
            let mag = round(c, ex2)?;
            let negr = c.and(&[xneg, yodd])?;
            let path = copysign(c, mag, negr)?;
            let pinf = inf(c, false)?;
            let ninf = inf(c, true)?;
            let xinf_r = {
                let sxinf = if f.flavour == Flavour::FiniteNan {
                    nan
                } else {
                    c.ite(xneg, ninf, pinf)?
                };
                let szero = c.ite(xneg, nz, pz)?;
                let neg_y = c.ite(yodd, szero, pz)?;
                let pos_y = c.ite(yodd, sxinf, pinf)?;
                let neg_x = c.ite(yneg, neg_y, pos_y)?;
                let pos_x = c.ite(yneg, pz, pinf)?;
                c.ite(xneg, neg_x, pos_x)?
            };
            let xzero_r = {
                let sxinf = if f.flavour == Flavour::FiniteNan {
                    nan
                } else {
                    c.ite(xneg, ninf, pinf)?
                };
                let szero = c.ite(xneg, nz, pz)?;
                let neg_y = c.ite(yodd, sxinf, pinf)?;
                let pos_y = c.ite(yodd, szero, pz)?;
                c.ite(yneg, neg_y, pos_y)?
            };
            let yinf_r = {
                let flip = c.app(Op::Xor, &[lt1, yneg])?;
                let r = c.ite(flip, pz, pinf)?;
                c.ite(ax1, one, r)?
            };
            let ynot_int = c.not(yint)?;
            let neg_nonint = c.and(&[xneg, xnotz, ynot_int])?;
            let r = c.ite(neg_nonint, nan, path)?;
            let r = c.ite(xinf, xinf_r, r)?;
            let r = c.ite(xzero, xzero_r, r)?;
            let r = c.ite(yinf, yinf_r, r)?;
            let anynan = c.or(&[xnan, ynan])?;
            let r = c.ite(anynan, nan, r)?;
            let r = c.ite(xone_pos, one, r)?;
            c.ite(yzero, one, r)?
        }
        Recipe::Tanh => {
            let aq = enc_q(c, ax)?;
            let s = uf(c, 0, &[aq, aq])?;
            let c3 = uf(c, 1, &[s, aq])?;
            let three = c.bv(crate::recipe::encode(qf, 3.0, false), qf.bits());
            let d = uf(c, 2, &[c3, three])?;
            let cubic = uf(c, 3, &[aq, d])?;
            let aqv = to_sort(c, sq, ax)?;
            let a2v = c.op2(Op::FpAdd(Rm::Rne), aqv, aqv)?;
            let a2 = c.bits_of(a2v)?;
            let big = uf(c, 4, &[a2])?;
            let num = uf(c, 5, &[big, one_q])?;
            let den = uf(c, 6, &[big, one_q])?;
            let mid = uf(c, 7, &[num, den])?;
            let thr = |c: &mut Ctx, v: f64, up: bool| -> Res<T> {
                let b = c.bv(crate::recipe::encode(f, v, up), f.bits());
                bits_value(c, f, b)
            };
            let x1 = thr(c, crate::compose::tanh_cubic_limit(f), true)?;
            let cl = thr(c, crate::compose::tanh_clamp(f), false)?;
            let small = c.op2(Op::FpLt, ax, x1)?;
            let sat = c.op2(Op::FpLt, cl, ax)?;
            let sat = c.or(&[sat, xinf])?;
            let v = c.ite(sat, one_q, mid)?;
            let v = c.ite(small, cubic, v)?;
            let r = round(c, v)?;
            let r = copysign(c, r, xneg)?;
            c.ite(xnan, nan, r)?
        }
        Recipe::LogPlusOne => {
            let xq = enc_q(c, x)?;
            let wv = uf(c, 0, &[one_q, xq])?;
            let lw = uf(c, 1, &[wv])?;
            let dm = uf(c, 2, &[wv, one_q])?;
            let ratio = uf(c, 3, &[lw, dm])?;
            let prod = uf(c, 4, &[xq, ratio])?;
            let wone = c.eq(wv, one_q)?;
            let v = c.ite(wone, xq, prod)?;
            let r = round(c, v)?;
            let m1 = c.op1(Op::FpNeg, one)?;
            let below = c.op2(Op::FpLt, x, m1)?;
            let is_m1 = c.op2(Op::FpEq, x, m1)?;
            let pinf = inf(c, false)?;
            let ninf = inf(c, true)?;
            let r = c.ite(xzero, x, r)?;
            let r = c.ite(xinf, pinf, r)?;
            let r = c.ite(is_m1, ninf, r)?;
            let bad = c.or(&[xnan, below])?;
            c.ite(bad, nan, r)?
        }
        Recipe::Rsqrt => {
            let xq = enc_q(c, x)?;
            let rec = uf(c, 0, &[one_q, xq])?;
            let root = uf(c, 1, &[rec])?;
            let r = round(c, root)?;
            let sinf = if f.flavour == Flavour::FiniteNan {
                nan
            } else {
                let p = inf(c, false)?;
                let n = inf(c, true)?;
                c.ite(xneg, n, p)?
            };
            let r = c.ite(xinf, pz, r)?;
            let r = c.ite(xzero, sinf, r)?;
            let xnotz = c.not(xzero)?;
            let neg = c.and(&[xneg, xnotz])?;
            let bad = c.or(&[xnan, neg])?;
            c.ite(bad, nan, r)?
        }
        Recipe::Wider => return Err(FloatError::Proof("no wider composition".into())),
    })
}

/// A float value of format `f` from an encoding term (`Val` as `cc_smt::ops` reads it).
fn fval(c: &mut Ctx, f: Format, bits: T) -> Res<Val> {
    let t = if f.native() {
        c.app(Op::FpFromBits { e: f.e, s: f.m + 1 }, &[bits])?
    } else {
        bits
    };
    Ok(Val::new(Ty::Float(f), t))
}

/// One query: its text and whether it needs float sorts beyond the interchange ones.
#[derive(Debug, Clone)]
pub struct Query {
    pub name: String,
    pub text: String,
    pub any_sort: bool,
}

/// The query for a wrapper, from Yosys' SMT-LIB of it.
pub fn query(b: &Built, yosys: &str) -> Res<Query> {
    let g = &b.glue;
    let m = b.rtl.name().to_string();
    let mut c = Ctx::new();
    let mut links: Vec<(String, String, u32)> = Vec::new();
    let mut var = |c: &mut Ctx, wire: &str, w: u32| -> Res<T> {
        let n = format!("rtl.{wire}");
        links.push((n.clone(), wire.to_string(), w));
        Ok(c.var(&n, Sort::Bv(w))?)
    };
    let f = g.format();
    let mut ins = Vec::new();
    for p in b.rtl.inputs() {
        ins.push(var(&mut c, &p.name, p.width)?);
    }
    let outp = b
        .rtl
        .outputs()
        .first()
        .ok_or_else(|| FloatError::Proof(format!("{m}: no output")))?
        .clone();
    let out = var(&mut c, &outp.name, outp.width)?;
    let mut cores = Vec::new();
    for w in b.rtl.core_wires() {
        let mut a = Vec::new();
        for (n, wd) in &w.operands {
            a.push(var(&mut c, n, *wd)?);
        }
        let z = var(&mut c, &w.result.0, w.result.1)?;
        cores.push((a, z));
    }
    let mut hyps: Vec<T> = Vec::new();
    let mut mismatch: Vec<T> = Vec::new();
    let core_fmt = |k: usize| Format::ieee(b.needs[k].exp_width, b.needs[k].sig_width);

    // The spec's operands (raw encodings) and their DAZ values.
    let mut args = Vec::new();
    for (j, &x) in ins.iter().enumerate() {
        args.push(match g.input_ty(j) {
            Ty::Bool => {
                let one = c.bv(1, 1);
                Val::new(Ty::Bool, c.eq(x, one)?)
            }
            ty @ Ty::Int { .. } => Val::new(ty, x),
            _ => fval(&mut c, f, x)?,
        });
    }
    let exact = exact_value_form(g);
    let v = if exact.is_some() {
        Some(c.var("v", F64)?)
    } else {
        None
    };

    // Core contracts.
    for (k, (a, z)) in cores.iter().enumerate() {
        let cf = core_fmt(k);
        let fp = |c: &mut Ctx, t: T, f: Format| -> Res<T> {
            Ok(c.app(Op::FpFromBits { e: f.e, s: f.m + 1 }, &[t])?)
        };
        let composed = matches!(g, Glue::Compose { .. });
        match &b.needs[k].role {
            // In a composition every core but the final conversion is a function of its operands'
            // encodings, whatever it computes: the proof covers the structure and the special values,
            // the sampler the accuracy.
            Role::Arith { .. } | Role::Approx { .. } if composed => {
                let name = format!("core{k}");
                let sorts: Vec<Sort> = a.iter().map(|&t| c.sort(t)).collect();
                let ret = c.sort(*z);
                c.fun(&name, &sorts, ret)?;
                let app = c.app(Op::Uf(name), a)?;
                hyps.push(c.eq(*z, app)?);
            }
            Role::Arith { op } => {
                let zf = fp(&mut c, *z, cf)?;
                let want = if let Some(v) = v {
                    to_sort(&mut c, sort_of(cf), v)?
                } else {
                    let mut xs = Vec::new();
                    for &t in a {
                        xs.push(fp(&mut c, t, cf)?);
                    }
                    ieee_op(&mut c, op, &xs)?
                };
                hyps.push(c.eq(zf, want)?);
            }
            Role::Approx { .. } => {}
            Role::Convert { from } => {
                let src = crate::format_by_name(from)
                    .ok_or_else(|| FloatError::Proof(format!("unknown format {from}")))?;
                let x = fp(&mut c, a[0], src)?;
                let zf = fp(&mut c, *z, cf)?;
                let want = to_sort(&mut c, sort_of(cf), x)?;
                hyps.push(c.eq(zf, want)?);
            }
            Role::Less => {
                let x = fp(&mut c, a[0], cf)?;
                let y = fp(&mut c, a[1], cf)?;
                let lt = c.op2(Op::FpLt, x, y)?;
                let one = c.bv(1, 1);
                let zero = c.bv(0, 1);
                let bit = c.ite(lt, one, zero)?;
                hyps.push(c.eq(*z, bit)?);
            }
            Role::ToInt { signed, w } => {
                let x = fval(&mut c, cf, a[0])?;
                let r = ops::convert(
                    &mut c,
                    x,
                    Ty::Int {
                        signed: *signed,
                        w: *w,
                    },
                )?;
                let xf = fp(&mut c, a[0], cf)?;
                let nan = c.op1(Op::FpIsNaN, xf)?;
                let eq = c.eq(*z, r.t)?;
                hyps.push(c.or(&[nan, eq])?);
            }
            Role::RoundToInt { rnd, signed, w } => {
                let rm = match rnd.as_str() {
                    "rne" => Rm::Rne,
                    "rtz" => Rm::Rtz,
                    "floor" => Rm::Rtn,
                    "ceil" => Rm::Rtp,
                    other => return Err(FloatError::Proof(format!("rounding {other}"))),
                };
                if !*signed {
                    return Err(FloatError::Proof("an unsigned rounding core".into()));
                }
                // Bound only where the glue uses it: |x| < 2^(w-2), the result in range.
                let x = fp(&mut c, a[0], cf)?;
                let lim = float::lit(
                    &mut c,
                    sort_of(cf),
                    crate::recipe::encode(cf, 2f64.powi(*w as i32 - 2), false),
                )?;
                let ax = c.op1(Op::FpAbs, x)?;
                let small = c.op2(Op::FpLt, ax, lim)?;
                let big = c.not(small)?;
                let conv = c.app(Op::FpToSbv { w: *w, rm }, &[x])?;
                let eq = c.eq(*z, conv)?;
                hyps.push(c.or(&[big, eq])?);
            }
            Role::FromInt { signed, w: _ } => {
                let o = if *signed {
                    Op::FpFromSbv {
                        e: cf.e,
                        s: cf.m + 1,
                        rm: Rm::Rne,
                    }
                } else {
                    Op::FpFromUbv {
                        e: cf.e,
                        s: cf.m + 1,
                        rm: Rm::Rne,
                    }
                };
                let want = c.app(o, &[a[0]])?;
                let zf = fp(&mut c, *z, cf)?;
                hyps.push(c.eq(zf, want)?);
            }
        }
    }

    // The operands the first core must see: the format's DAZ values, exactly widened.
    let check_operands = |c: &mut Ctx, mismatch: &mut Vec<T>, cf: Format, a: &[T]| -> Res<()> {
        for (j, &t) in a.iter().enumerate() {
            let d = float::daz(c, f, args[j].t)?;
            let x = value(c, f, d)?;
            let want = to_sort(c, sort_of(cf), x)?;
            let got = c.app(Op::FpFromBits { e: cf.e, s: cf.m + 1 }, &[t])?;
            let eq = c.eq(got, want)?;
            mismatch.push(c.not(eq)?);
        }
        Ok(())
    };

    // The claim.
    let attrs = Attrs::default();
    let (got, want) = match g {
        Glue::Arith { op, fmt } => {
            let got = fval(&mut c, *fmt, out)?;
            let want = if let (Some(op), Some(v)) = (exact, v) {
                check_operands(&mut c, &mut mismatch, core_fmt(0), &cores[0].0)?;
                let mut xs = Vec::new();
                for a in &args {
                    let d = float::daz(&mut c, *fmt, a.t)?;
                    let val = value(&mut c, *fmt, d)?;
                    xs.push(to_sort(&mut c, F64, val)?);
                }
                hyps.push(exact_value(&mut c, op, &xs, v)?);
                let raw: Vec<T> = args.iter().map(|a| a.t).collect();
                Val::new(Ty::Float(*fmt), float::arith_exact(&mut c, *fmt, op, &raw, v)?)
            } else {
                ops::apply(&mut c, catalog_name(*op), &attrs, &args, Ty::Float(*fmt))?
            };
            (got, want)
        }
        Glue::Approx { fmt, via, .. } => {
            check_operands(&mut c, &mut mismatch, *via, &cores[0].0)?;
            let got = fval(&mut c, *fmt, out)?;
            // Relative to the approximate core's result (a conversion core in the chain is bound by
            // its contract).
            let w = round_result(&mut c, *via, *fmt, cores[0].1)?;
            (got, Val::new(Ty::Float(*fmt), w))
        }
        Glue::Convert { to, .. } => {
            let got = fval(&mut c, *to, out)?;
            let want = ops::convert(&mut c, args[0], Ty::Float(*to))?;
            (got, want)
        }
        Glue::MaxMin { max, fmt } => {
            let got = fval(&mut c, *fmt, out)?;
            let op = if *max { "maximum" } else { "minimum" };
            (got, ops::apply(&mut c, op, &attrs, &args, Ty::Float(*fmt))?)
        }
        Glue::Compare { dir, .. } => {
            let one = c.bv(1, 1);
            let got = Val::new(Ty::Bool, c.eq(out, one)?);
            let a = Attrs {
                direction: Some(dir.name().into()),
                compare_type: Some("FLOAT".into()),
                precision: None,
            };
            (got, ops::apply(&mut c, "compare", &a, &args, Ty::Bool)?)
        }
        Glue::Sign { op, fmt } => {
            let got = fval(&mut c, *fmt, out)?;
            let name = match op {
                SignOp::Abs => "abs",
                SignOp::Negate => "negate",
                SignOp::Sign => "sign",
            };
            (got, ops::apply(&mut c, name, &attrs, &args, Ty::Float(*fmt))?)
        }
        Glue::ToInt { signed, w, .. } => {
            let ty = Ty::Int {
                signed: *signed,
                w: *w,
            };
            (Val::new(ty, out), ops::convert(&mut c, args[0], ty)?)
        }
        Glue::FromInt { fmt, .. } => {
            let got = fval(&mut c, *fmt, out)?;
            (got, ops::convert(&mut c, args[0], Ty::Float(*fmt))?)
        }
        Glue::Bool { fmt, to_float: true } => {
            let got = fval(&mut c, *fmt, out)?;
            (got, ops::convert(&mut c, args[0], Ty::Float(*fmt))?)
        }
        Glue::Bool { to_float: false, .. } => {
            let one = c.bv(1, 1);
            let got = Val::new(Ty::Bool, c.eq(out, one)?);
            (got, ops::convert(&mut c, args[0], Ty::Bool)?)
        }
        Glue::IsFinite { .. } => {
            let one = c.bv(1, 1);
            let got = Val::new(Ty::Bool, c.eq(out, one)?);
            (got, ops::apply(&mut c, "is_finite", &attrs, &args, Ty::Bool)?)
        }
        Glue::Round { fmt, rnd } => {
            let got = fval(&mut c, *fmt, out)?;
            let name = crate::glue::round_op(*rnd);
            (got, ops::apply(&mut c, name, &attrs, &args, Ty::Float(*fmt))?)
        }
        Glue::ReducePrecision { fmt, e, m } => {
            let got = fval(&mut c, *fmt, out)?;
            let a = Attrs {
                precision: Some((*e, *m)),
                ..Attrs::default()
            };
            (
                got,
                ops::apply(&mut c, "reduce_precision", &a, &args, Ty::Float(*fmt))?,
            )
        }
        Glue::Clamp { fmt } => {
            let got = fval(&mut c, *fmt, out)?;
            (got, ops::apply(&mut c, "clamp", &attrs, &args, Ty::Float(*fmt))?)
        }
        Glue::Compose { recipe, fmt, q } => {
            let got = bits_value(&mut c, *fmt, out)?;
            let want = compose_spec(&mut c, *recipe, *fmt, *q, &args)?;
            let eq = c.eq(got, want)?;
            let t = c.bool(true);
            (Val::new(Ty::Bool, eq), Val::new(Ty::Bool, t))
        }
    };
    let same = ops::same(&mut c, got, want)?;
    mismatch.push(c.not(same)?);
    let goal = c.or(&mismatch)?;
    let mut roots = hyps;
    roots.push(goal);

    let mut s = Script::new();
    s.comment(format!(
        "{m}{}: the wrapper (Yosys write_smt2, cores cut and bound by their contracts) differs from the specification (expect {})",
        if b.mutation == Mutation::None {
            String::new()
        } else {
            format!(", negative control {:?}", b.mutation)
        },
        if b.mutation == Mutation::None { "unsat" } else { "sat" }
    ));
    c.render(&mut s, &roots);
    let text = s.render();
    let head = "(set-option :produce-models true)\n";
    let (pre, rest) = text
        .split_once(head)
        .ok_or_else(|| FloatError::Proof("unexpected script layout".into()))?;
    let (body, tail) = rest
        .rsplit_once("(check-sat)")
        .ok_or_else(|| FloatError::Proof("unexpected script layout".into()))?;
    let mut extra = String::new();
    extra.push_str(&format!("(declare-const rtl.state |{m}_s|)\n"));
    let mut links_text = String::new();
    for (n, wire, w) in &links {
        if !body.contains(&format!("(declare-const {n} ")) {
            links_text.push_str(&format!("(declare-const {n} (_ BitVec {w}))\n"));
        }
        // Yosys writes 1-bit wires as Bool.
        let rhs = if *w == 1 {
            format!("(ite (|{m}_n {wire}| rtl.state) #b1 #b0)")
        } else {
            format!("(|{m}_n {wire}| rtl.state)")
        };
        links_text.push_str(&format!("(assert (= {n} {rhs}))\n"));
    }
    let text = format!("{pre}{head}{yosys}{extra}{body}{links_text}(check-sat){tail}");
    let any_sort = uses_other_sorts(&text);
    Ok(Query {
        name: format!(
            "{m}{}",
            if b.mutation == Mutation::None {
                String::new()
            } else {
                format!(".{}", mutation_name(b.mutation))
            }
        ),
        text,
        any_sort,
    })
}

pub fn mutation_name(m: Mutation) -> &'static str {
    match m {
        Mutation::None => "none",
        Mutation::NoFlush => "no_flush",
        Mutation::TruncatingEncoder => "truncating_encoder",
        Mutation::NoOverflowCase => "no_overflow_case",
        Mutation::NoNanCase => "no_nan_case",
        Mutation::NoZeroOrder => "no_zero_order",
    }
}

/// Whether a query names a float sort other than Float16, Float32, Float64 and Float128, as a sort or
/// as the indices of a conversion (Bitwuzla's release build has only those four; it parses a
/// conversion into another sort without the check its sort declarations have, so both are looked for).
pub fn uses_other_sorts(text: &str) -> bool {
    ["(_ FloatingPoint ", "(_ to_fp ", "(_ to_fp_unsigned "]
        .iter()
        .any(|pat| {
            text.match_indices(pat).any(|(i, _)| {
                let rest = &text[i + pat.len()..];
                let end = rest.find(')').unwrap_or(rest.len());
                let es: Vec<u32> = rest[..end]
                    .split_whitespace()
                    .filter_map(|x| x.parse().ok())
                    .collect();
                !matches!(es.as_slice(), [5, 11] | [8, 24] | [11, 53] | [15, 113])
            })
        })
}

/// The answer to one proof.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofRecord {
    pub name: String,
    pub module: String,
    pub mutation: Mutation,
    pub expect: String,
    pub solver: Solver,
    pub version: String,
    pub verdict: Verdict,
    /// Wall time of the solver call (a container start included), 0 when the answer was cached.
    pub seconds: f64,
    pub cached: bool,
    pub query_hash: String,
}

impl ProofRecord {
    /// `unsat` for a wrapper, `sat` for a negative control.
    pub fn ok(&self) -> bool {
        match self.mutation {
            Mutation::None => self.verdict == Verdict::Unsat,
            _ => matches!(self.verdict, Verdict::Sat { .. }),
        }
    }
}

/// Where and how the proofs run.
#[derive(Debug, Clone)]
pub struct Prover {
    /// A Docker runner of the toolchain image (Yosys and the solvers).
    pub runner: BoundRunner,
    pub location: Location,
    /// The configured solver; queries with other float sorts go to the one that has them.
    pub solver: Solver,
    pub timeout_s: u64,
    pub cache: Cache,
    /// Parallel solver calls.
    pub jobs: usize,
}

/// The solver for a query: the configured one, or the first that declares float sorts of any width.
pub fn solver_for(q: &Query, configured: Solver) -> Solver {
    if q.any_sort && !configured.adapter().capabilities().any_float_sort {
        Solver::ALL
            .into_iter()
            .find(|s| s.adapter().capabilities().any_float_sort)
            .unwrap_or(Solver::Z3)
    } else {
        configured
    }
}

impl Prover {
    /// Write each wrapper and its stubs under `dir/<name>/`, run Yosys on all of them in one container,
    /// build the queries (`dir/smt/<name>.smt2`), and answer them (cache first).
    pub fn prove(&self, dir: &Path, built: &[Built]) -> Res<Vec<ProofRecord>> {
        let mut names = Vec::new();
        for b in built {
            let n = query_dir_name(b);
            let d = dir.join(&n);
            std::fs::create_dir_all(&d).map_err(|e| io(&d, e))?;
            write(&d.join("glue.v"), &b.rtl.verilog())?;
            write(&d.join("stubs.v"), &b.rtl.stub_verilog())?;
            write(&d.join("run.ys"), &yosys_script(b))?;
            names.push(n);
        }
        let script = "for d in \"$@\"; do (cd \"$d\" && yosys -q -l yosys.log -s run.ys >/dev/null 2>&1 || echo \"yosys failed: $d\"); done";
        let inv = Invocation::new("yosys-glue", "bash")
            .arg("-c")
            .arg(script)
            .arg("bash")
            .args(names.iter().cloned())
            .cwd("/work")
            .mount(Mount {
                src: dir.to_path_buf(),
                dst: "/work".into(),
                readonly: false,
            });
        let o = self
            .runner
            .clone()
            .with_caps(Caps {
                cpus: 1.0,
                ..*self.runner.caps()
            })
            .run(&inv)?;
        let text = o.stdout_string()?;
        o.check()?;
        if let Some(l) = text.lines().find(|l| l.starts_with("yosys failed")) {
            return Err(FloatError::Proof(l.to_string()));
        }
        let smt = dir.join("smt");
        std::fs::create_dir_all(&smt).map_err(|e| io(&smt, e))?;
        let mut queries = Vec::new();
        for (b, n) in built.iter().zip(&names) {
            let p = dir.join(n).join("glue.smt2");
            let ys = std::fs::read_to_string(&p).map_err(|e| io(&p, e))?;
            let q = query(b, &ys)?;
            write(&smt.join(format!("{}.smt2", q.name)), &q.text)?;
            queries.push((b, q));
        }
        self.answer(&smt, &queries)
    }

    fn answer(&self, smt: &Path, queries: &[(&Built, Query)]) -> Res<Vec<ProofRecord>> {
        let mut versions: BTreeMap<Solver, String> = BTreeMap::new();
        for (_, q) in queries {
            let s = solver_for(q, self.solver);
            if let std::collections::btree_map::Entry::Vacant(e) = versions.entry(s) {
                e.insert(cc_smt::solver_version(&self.runner, &self.location, s)?);
            }
        }
        let jobs = self.jobs.max(1);
        let c = self.runner.caps();
        let share = Caps {
            memory_gb: c.memory_gb / jobs as f64,
            cpus: 1.0,
            ..*c
        };
        let runner = self.runner.clone().with_caps(share);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .map_err(|e| FloatError::Proof(e.to_string()))?;
        let errors = Mutex::new(Vec::new());
        let records: Vec<ProofRecord> = pool.install(|| {
            queries
                .par_iter()
                .filter_map(|(b, q)| {
                    let solver = solver_for(q, self.solver);
                    let version = versions.get(&solver).cloned().unwrap_or_default();
                    let hash = cc_smt::query_hash(&q.text);
                    let file = format!("{}.smt2", q.name);
                    let (verdict, seconds, cached) = match self.cache.get(solver, &version, &hash) {
                        Some(v) => (v, 0.0, true),
                        None => {
                            let t0 = Instant::now();
                            match cc_smt::run_batch(
                                &runner,
                                &self.location,
                                solver,
                                smt,
                                std::slice::from_ref(&file),
                                self.timeout_s,
                            ) {
                                Ok(batch) => {
                                    let v = batch.verdicts.get(&file).cloned().unwrap_or(Verdict::Error {
                                        message: "no answer".into(),
                                    });
                                    let _ = self.cache.put(solver, &version, &hash, &v);
                                    (v, t0.elapsed().as_secs_f64(), false)
                                }
                                Err(e) => {
                                    if let Ok(mut es) = errors.lock() {
                                        es.push(e.to_string());
                                    }
                                    return None;
                                }
                            }
                        }
                    };
                    Some(ProofRecord {
                        name: q.name.clone(),
                        module: b.rtl.name().to_string(),
                        mutation: b.mutation,
                        expect: if b.mutation == Mutation::None {
                            "unsat".into()
                        } else {
                            "sat".into()
                        },
                        solver,
                        version,
                        verdict,
                        seconds,
                        cached,
                        query_hash: hash,
                    })
                })
                .collect()
        });
        if let Some(e) = errors.into_inner().ok().and_then(|es| es.into_iter().next()) {
            return Err(FloatError::Proof(e));
        }
        let mut records = records;
        records.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(records)
    }
}

fn query_dir_name(b: &Built) -> String {
    if b.mutation == Mutation::None {
        b.rtl.name().to_string()
    } else {
        format!("{}.{}", b.rtl.name(), mutation_name(b.mutation))
    }
}

fn io(p: &Path, e: std::io::Error) -> FloatError {
    FloatError::Io(format!("{}: {e}", p.display()))
}

fn write(p: &PathBuf, text: &str) -> Res<()> {
    std::fs::write(p, text).map_err(|e| io(p, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_sorts_beyond_the_interchange_ones() {
        assert!(!uses_other_sorts(
            "(declare-const x (_ FloatingPoint 8 24)) ((_ to_fp 11 53) RNE x)"
        ));
        assert!(uses_other_sorts("((_ to_fp 9 24) RNE x)"));
        assert!(uses_other_sorts("(declare-const x (_ FloatingPoint 8 8))"));
        assert!(uses_other_sorts("((_ to_fp_unsigned 9 8) RNE b)"));
    }
}
