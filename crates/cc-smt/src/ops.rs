//! Scalar semantics of the StableHLO element-wise ops (docs/framework/smt.md, "Op semantics").
//!
//! One function, [`apply`], gives the value of an op on scalar operands of given element types, following
//! the StableHLO specification (openxla/stablehlo `docs/spec.md`). Where the spec leaves a result open
//! (inexact `convert`, integer division by zero and `MIN / -1`, out-of-range float to integer), the
//! encoding follows XLA's element-wise emitter (`xla/service/elemental_ir_emitter.cc`), which is what
//! the exported models compute; docs/framework/smt.md lists every such choice. Ops whose float results
//! the spec defines by an approximate library function (`exponential`, `log`, `tanh`, ...) are
//! uninterpreted functions per op and format: a proof is then valid for any implementation of them.

use cc_ir::Elem;

use crate::float::{self, Flavour, Format};
use crate::term::{Ctx, Op, Rm, Sort, SortError, T};

type Res<X> = Result<X, SortError>;

/// A scalar type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Ty {
    Bool,
    Int { signed: bool, w: u32 },
    Float(Format),
    Complex(Format),
}

impl Ty {
    pub fn of(e: &Elem) -> Res<Ty> {
        Ok(match e {
            Elem::Bool => Ty::Bool,
            Elem::Int { signed, width } => Ty::Int {
                signed: *signed,
                w: u32::from(*width),
            },
            Elem::Float { .. } => {
                Ty::Float(Format::of(e).ok_or_else(|| SortError(format!("{e}: no encoding")))?)
            }
            Elem::Complex(c) => match Ty::of(c)? {
                Ty::Float(f) => Ty::Complex(f),
                _ => return Err(SortError(format!("{e}: no encoding"))),
            },
        })
    }

    /// The sort of one component.
    pub fn sort(self) -> Sort {
        match self {
            Ty::Bool => Sort::Bool,
            Ty::Int { w, .. } => Sort::Bv(w),
            Ty::Float(f) | Ty::Complex(f) => f.sort(),
        }
    }

    pub fn name(self) -> String {
        match self {
            Ty::Bool => "i1".into(),
            Ty::Int { signed: true, w } => format!("i{w}"),
            Ty::Int { signed: false, w } => format!("ui{w}"),
            Ty::Float(f) => f.name(),
            Ty::Complex(f) => format!("c{}", f.name()),
        }
    }
}

/// A scalar value: one term, or two for a complex number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Val {
    pub ty: Ty,
    pub t: T,
    /// The imaginary part of a complex value.
    pub im: Option<T>,
}

impl Val {
    pub fn new(ty: Ty, t: T) -> Val {
        Val { ty, t, im: None }
    }
}

/// A free scalar input named `name` (complex: `name.re`, `name.im`).
pub fn input(c: &mut Ctx, name: &str, ty: Ty) -> Res<Val> {
    Ok(match ty {
        Ty::Complex(_) => Val {
            ty,
            t: c.var(&format!("{name}.re"), ty.sort())?,
            im: Some(c.var(&format!("{name}.im"), ty.sort())?),
        },
        _ => Val::new(ty, c.var(name, ty.sort())?),
    })
}

/// The zero of a type (the init value of a contraction's accumulator).
pub fn zero(c: &mut Ctx, ty: Ty) -> Res<Val> {
    Ok(match ty {
        Ty::Bool => Val::new(ty, c.bool(false)),
        Ty::Int { w, .. } => Val::new(ty, c.bv(0, w)),
        Ty::Float(f) => Val::new(ty, fzero(c, f)?),
        Ty::Complex(f) => Val {
            ty,
            t: fzero(c, f)?,
            im: Some(fzero(c, f)?),
        },
    })
}

fn fzero(c: &mut Ctx, f: Format) -> Res<T> {
    if f.native() {
        float::lit(c, f.fp_sort(), 0)
    } else {
        Ok(c.bv(0, f.bits()))
    }
}

/// Equality of two values as the result of an op: bit-exact, all NaNs equal (a quiet NaN's payload is
/// not specified).
pub fn same(c: &mut Ctx, a: Val, b: Val) -> Res<T> {
    if a.ty != b.ty {
        return Err(SortError(format!(
            "compared {} with {}",
            a.ty.name(),
            b.ty.name()
        )));
    }
    let part = |c: &mut Ctx, x: T, y: T| -> Res<T> {
        match a.ty {
            Ty::Float(f) | Ty::Complex(f) if !f.native() => float::same_bits(c, f, x, y),
            // SMT-LIB `=` on FP terms: one NaN, +0 ≠ -0.
            _ => c.eq(x, y),
        }
    };
    let re = part(c, a.t, b.t)?;
    match (a.im, b.im) {
        (Some(x), Some(y)) => {
            let im = part(c, x, y)?;
            c.and(&[re, im])
        }
        _ => Ok(re),
    }
}

/// Attributes an op's semantics reads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Attrs {
    /// `comparison_direction` (`EQ`, `NE`, `LT`, `LE`, `GT`, `GE`).
    pub direction: Option<String>,
    /// `compare_type` (`SIGNED`, `UNSIGNED`, `FLOAT`, `TOTALORDER`).
    pub compare_type: Option<String>,
    /// `reduce_precision`'s `exponent_bits`, `mantissa_bits`.
    pub precision: Option<(u32, u32)>,
}

/// The value of op `op` (without the `stablehlo.` prefix) on `args`, with result type `ret`.
pub fn apply(c: &mut Ctx, op: &str, a: &Attrs, args: &[Val], ret: Ty) -> Res<Val> {
    let bad = || {
        SortError(format!(
            "{op}: no encoding for ({}) -> {}",
            names(args),
            ret.name()
        ))
    };
    let n = args.len();
    let x = args.first().copied();
    let y = args.get(1).copied();
    match (op, n) {
        ("convert", 1) => convert(c, args[0], ret),
        ("select", 3) => {
            let p = args[0];
            if p.ty != Ty::Bool || args[1].ty != args[2].ty {
                return Err(bad());
            }
            let t = c.ite(p.t, args[1].t, args[2].t)?;
            let im = match (args[1].im, args[2].im) {
                (Some(u), Some(v)) => Some(c.ite(p.t, u, v)?),
                _ => None,
            };
            Ok(Val {
                ty: args[1].ty,
                t,
                im,
            })
        }
        ("clamp", 3) => {
            // `min(max(operand, min), max)` (spec `clamp`).
            let lo = apply(c, "maximum", a, &[args[1], args[0]], ret)?;
            apply(c, "minimum", a, &[lo, args[2]], ret)
        }
        ("compare", 2) => compare(c, a, args[0], args[1]).map(|t| Val::new(Ty::Bool, t)),
        ("complex", 2) => match (args[0].ty, args[1].ty, ret) {
            (Ty::Float(f), Ty::Float(g), Ty::Complex(h)) if f == g && g == h => Ok(Val {
                ty: ret,
                t: args[0].t,
                im: Some(args[1].t),
            }),
            _ => Err(bad()),
        },
        ("real", 1) | ("imag", 1) => match (args[0].ty, args[0].im) {
            (Ty::Complex(f), Some(im)) if ret == Ty::Float(f) => {
                Ok(Val::new(ret, if op == "real" { args[0].t } else { im }))
            }
            (Ty::Float(_), _) if op == "real" && ret == args[0].ty => Ok(args[0]),
            (Ty::Float(f), _) if op == "imag" && ret == args[0].ty => Ok(Val::new(ret, fzero(c, f)?)),
            _ => Err(bad()),
        },
        _ => {
            let x = x.ok_or_else(bad)?;
            if args.iter().any(|v| v.ty != x.ty) || (op != "is_finite" && ret != x.ty) {
                return Err(bad());
            }
            match x.ty {
                Ty::Bool => boolean(c, op, args).ok_or_else(bad)?.map(|t| Val::new(ret, t)),
                Ty::Int { signed, w } => integer(c, op, args, signed, w)
                    .ok_or_else(bad)?
                    .map(|t| Val::new(ret, t)),
                Ty::Float(f) => floating(c, op, a, args, f, ret).ok_or_else(bad)?,
                Ty::Complex(f) => complex(c, op, x, y, f).ok_or_else(bad)?,
            }
        }
    }
}

fn names(args: &[Val]) -> String {
    args.iter().map(|v| v.ty.name()).collect::<Vec<_>>().join(", ")
}

fn boolean(c: &mut Ctx, op: &str, a: &[Val]) -> Option<Res<T>> {
    let t: Vec<T> = a.iter().map(|v| v.t).collect();
    Some(match (op, t.as_slice()) {
        // `maximum`/`minimum` on booleans are OR/AND (spec).
        ("and" | "minimum" | "multiply", [x, y]) => c.and(&[*x, *y]),
        ("or" | "maximum" | "add", [x, y]) => c.or(&[*x, *y]),
        ("xor", [x, y]) => c.app(Op::Xor, &[*x, *y]),
        ("not", [x]) => c.not(*x),
        _ => return None,
    })
}

fn integer(c: &mut Ctx, op: &str, a: &[Val], signed: bool, w: u32) -> Option<Res<T>> {
    let t: Vec<T> = a.iter().map(|v| v.t).collect();
    let r = (|| -> Res<T> {
        Ok(match (op, t.as_slice()) {
            // Two's complement, wrapping (spec: integer add, subtract, multiply).
            ("add", [x, y]) => c.op2(Op::BvAdd, *x, *y)?,
            ("subtract", [x, y]) => c.op2(Op::BvSub, *x, *y)?,
            ("multiply", [x, y]) => c.op2(Op::BvMul, *x, *y)?,
            ("and", [x, y]) => c.op2(Op::BvAnd, *x, *y)?,
            ("or", [x, y]) => c.op2(Op::BvOr, *x, *y)?,
            ("xor", [x, y]) => c.op2(Op::BvXor, *x, *y)?,
            ("not", [x]) => c.op1(Op::BvNot, *x)?,
            ("negate", [x]) => c.op1(Op::BvNeg, *x)?,
            ("maximum" | "minimum", [x, y]) => {
                let lt = if signed {
                    c.op2(Op::BvSlt, *x, *y)?
                } else {
                    c.op2(Op::BvUlt, *x, *y)?
                };
                if op == "maximum" {
                    c.ite(lt, *y, *x)?
                } else {
                    c.ite(lt, *x, *y)?
                }
            }
            ("abs", [x]) if signed => {
                let z = c.bv(0, w);
                let neg = c.op2(Op::BvSlt, *x, z)?;
                let nx = c.op1(Op::BvNeg, *x)?;
                c.ite(neg, nx, *x)?
            }
            ("sign", [x]) => {
                let z = c.bv(0, w);
                let one = c.bv(1, w);
                let m1 = c.bvs(-1, w);
                let is0 = c.eq(*x, z)?;
                let pos = c.ite(is0, z, one)?;
                if signed {
                    let neg = c.op2(Op::BvSlt, *x, z)?;
                    c.ite(neg, m1, pos)?
                } else {
                    pos
                }
            }
            // XLA (`EmitIntegerDivide`): x / 0 = -1, INT_SMIN / -1 = INT_SMIN.
            ("divide", [x, y]) => {
                let z = c.bv(0, w);
                let m1 = c.bvs(-1, w);
                let by0 = c.eq(*y, z)?;
                if signed {
                    let min = c.bv(1u128 << (w - 1), w);
                    let xm = c.eq(*x, min)?;
                    let ym = c.eq(*y, m1)?;
                    let ovf = c.and(&[xm, ym])?;
                    let q = c.op2(Op::BvSdiv, *x, *y)?;
                    let q = c.ite(ovf, min, q)?;
                    c.ite(by0, m1, q)?
                } else {
                    let q = c.op2(Op::BvUdiv, *x, *y)?;
                    c.ite(by0, m1, q)?
                }
            }
            // XLA (`EmitIntegerRemainder`): x % 0 = x, INT_SMIN % -1 = 0; the sign follows the dividend.
            ("remainder", [x, y]) => {
                let z = c.bv(0, w);
                let by0 = c.eq(*y, z)?;
                if signed {
                    let min = c.bv(1u128 << (w - 1), w);
                    let m1 = c.bvs(-1, w);
                    let xm = c.eq(*x, min)?;
                    let ym = c.eq(*y, m1)?;
                    let ovf = c.and(&[xm, ym])?;
                    let r = c.op2(Op::BvSrem, *x, *y)?;
                    let r = c.ite(ovf, z, r)?;
                    c.ite(by0, *x, r)?
                } else {
                    let r = c.op2(Op::BvUrem, *x, *y)?;
                    c.ite(by0, *x, r)?
                }
            }
            _ => return Err(SortError(String::new())),
        })
    })();
    match r {
        Err(SortError(m)) if m.is_empty() => None,
        r => Some(r),
    }
}

/// A float value as a Float32 (carrier formats) or as itself (native formats).
fn wide(c: &mut Ctx, f: Format, t: T) -> Res<T> {
    if f.native() { Ok(t) } else { float::to_f32(c, f, t) }
}

/// Back from `wide`: round a Float32 result to `f`.
fn narrow(c: &mut Ctx, f: Format, t: T) -> Res<T> {
    if f.native() {
        Ok(t)
    } else {
        float::from_f32(c, f, t)
    }
}

fn floating(c: &mut Ctx, op: &str, a: &Attrs, args: &[Val], f: Format, ret: Ty) -> Option<Res<Val>> {
    let ty = Ty::Float(f);
    let rounded = |c: &mut Ctx, o: Op, k: &str, args: &[Val]| -> Res<Val> {
        let mut w = Vec::new();
        for v in args {
            w.push(wide(c, f, v.t)?);
        }
        let r = c.app(o, &w)?;
        if !f.native() {
            c.note(format!("{}.{k}", f.name()));
        }
        Ok(Val::new(ty, narrow(c, f, r)?))
    };
    let r = (|| -> Res<Val> {
        Ok(match (op, args) {
            ("add", [x, y]) => rounded(c, Op::FpAdd(Rm::Rne), "add", &[*x, *y])?,
            ("subtract", [x, y]) => rounded(c, Op::FpSub(Rm::Rne), "sub", &[*x, *y])?,
            ("multiply", [x, y]) => rounded(c, Op::FpMul(Rm::Rne), "mul", &[*x, *y])?,
            ("divide", [x, y]) => rounded(c, Op::FpDiv(Rm::Rne), "div", &[*x, *y])?,
            ("sqrt", [x]) => rounded(c, Op::FpSqrt(Rm::Rne), "sqrt", &[*x])?,
            // The integral value of a value of `f` is a value of `f`: no rounding back.
            ("round_nearest_even", [x]) => exact1(c, f, x.t, Op::FpRoundToIntegral(Rm::Rne))?,
            ("floor", [x]) => exact1(c, f, x.t, Op::FpRoundToIntegral(Rm::Rtn))?,
            ("ceil", [x]) => exact1(c, f, x.t, Op::FpRoundToIntegral(Rm::Rtp))?,
            ("negate", [x]) => Val::new(ty, sign_op(c, f, x.t, SignOp::Neg)?),
            ("abs", [x]) if ret == ty => Val::new(ty, sign_op(c, f, x.t, SignOp::Abs)?),
            ("maximum", [x, y]) => Val::new(ty, maxmin(c, f, x.t, y.t, true)?),
            ("minimum", [x, y]) => Val::new(ty, maxmin(c, f, x.t, y.t, false)?),
            ("is_finite", [x]) if ret == Ty::Bool => {
                let v = wide(c, f, x.t)?;
                let n = c.op1(Op::FpIsNaN, v)?;
                let i = c.op1(Op::FpIsInfinite, v)?;
                let s = c.or(&[n, i])?;
                Val::new(Ty::Bool, c.not(s)?)
            }
            ("sign", [x]) => {
                // Spec `sign`: NaN → NaN, ±0 → ±0, else ±1.0 (exact in every format).
                let v = wide(c, f, x.t)?;
                let ws = c.sort(v);
                let one = fp_one(c, ws)?;
                let mone = c.op1(Op::FpNeg, one)?;
                let neg = c.op1(Op::FpIsNegative, v)?;
                let unit = c.ite(neg, mone, one)?;
                let z = c.op1(Op::FpIsZero, v)?;
                let n = c.op1(Op::FpIsNaN, v)?;
                let r = c.ite(z, v, unit)?;
                let r = c.ite(n, v, r)?;
                Val::new(ty, narrow_exact(c, f, r)?)
            }
            ("remainder", [x, y]) => {
                // Spec: `lhs - d * rhs`, d = lhs / rhs rounded toward zero (C `fmod`, exact). From the
                // IEEE remainder `r` (exact): if r is nonzero with the sign opposite to lhs, add |rhs|
                // with lhs's sign (exact, as the fmod result is representable).
                let (u, v) = (wide(c, f, x.t)?, wide(c, f, y.t)?);
                let r = c.op2(Op::FpRem, u, v)?;
                let rn = c.op1(Op::FpIsNegative, r)?;
                let un = c.op1(Op::FpIsNegative, u)?;
                let differ = c.app(Op::Xor, &[rn, un])?;
                let rz = c.op1(Op::FpIsZero, r)?;
                let nz = c.not(rz)?;
                let fix = c.and(&[differ, nz])?;
                let av = c.op1(Op::FpAbs, v)?;
                let nav = c.op1(Op::FpNeg, av)?;
                let step = c.ite(un, nav, av)?;
                let r2 = c.op2(Op::FpAdd(Rm::Rne), r, step)?;
                let out = c.ite(fix, r2, r)?;
                // Zero results keep the dividend's sign.
                let oz = c.op1(Op::FpIsZero, out)?;
                let ws = c.sort(u);
                let pz = float::lit(c, ws, 0)?;
                let nzr = c.op1(Op::FpNeg, pz)?;
                let sz = c.ite(un, nzr, pz)?;
                let out = c.ite(oz, sz, out)?;
                Val::new(ty, narrow_exact(c, f, out)?)
            }
            ("reduce_precision", [x]) => {
                let (eb, mb) = a
                    .precision
                    .ok_or_else(|| SortError("reduce_precision without its bits".into()))?;
                Val::new(ty, reduce_precision(c, f, x.t, eb, mb)?)
            }
            (
                "exponential"
                | "log"
                | "log_plus_one"
                | "tanh"
                | "rsqrt"
                | "sine"
                | "cosine"
                | "power"
                | "logistic"
                | "exponential_minus_one"
                | "cbrt"
                | "tan"
                | "atan2",
                _,
            ) => {
                // Implementation-defined accuracy: an uninterpreted function of the operands' encodings.
                let name = format!("{op}_{}", f.name());
                let s = f.sort();
                c.fun(&name, &vec![s; args.len()], s)?;
                let ts: Vec<T> = args.iter().map(|v| v.t).collect();
                Val::new(ty, c.app(Op::Uf(name), &ts)?)
            }
            _ => return Err(SortError(String::new())),
        })
    })();
    match r {
        Err(SortError(m)) if m.is_empty() => None,
        r => Some(r),
    }
}

fn fp_one(c: &mut Ctx, s: Sort) -> Res<T> {
    let Sort::Fp { e, s: sb } = s else {
        return Err(SortError("one of a non-FP sort".into()));
    };
    float::lit(c, s, ((1u128 << (e - 1)) - 1) << (sb - 1))
}

/// Round back a result known to be a value of `f` (no new rounding happens).
fn narrow_exact(c: &mut Ctx, f: Format, t: T) -> Res<T> {
    narrow(c, f, t)
}

fn exact1(c: &mut Ctx, f: Format, t: T, o: Op) -> Res<Val> {
    let v = wide(c, f, t)?;
    let r = c.op1(o, v)?;
    Ok(Val::new(Ty::Float(f), narrow_exact(c, f, r)?))
}

enum SignOp {
    Neg,
    Abs,
}

fn sign_op(c: &mut Ctx, f: Format, t: T, o: SignOp) -> Res<T> {
    if f.native() {
        return match o {
            SignOp::Neg => c.op1(Op::FpNeg, t),
            SignOp::Abs => c.op1(Op::FpAbs, t),
        };
    }
    // On the bit pattern: flip or clear the sign.
    let top = c.bv(1u128 << (f.bits() - 1), f.bits());
    match o {
        SignOp::Neg => c.op2(Op::BvXor, t, top),
        SignOp::Abs => {
            let m = c.op1(Op::BvNot, top)?;
            c.op2(Op::BvAnd, t, m)
        }
    }
}

/// IEEE 754-2019 `maximum` / `minimum` (spec `maximum`, `minimum`): NaN if either operand is NaN,
/// `-0 < +0`. The result is one of the operands (or NaN), so carrier formats need no rounding.
fn maxmin(c: &mut Ctx, f: Format, x: T, y: T, max: bool) -> Res<T> {
    let (u, v) = (wide(c, f, x)?, wide(c, f, y)?);
    let un = c.op1(Op::FpIsNaN, u)?;
    let vn = c.op1(Op::FpIsNaN, v)?;
    let anynan = c.or(&[un, vn])?;
    let uz = c.op1(Op::FpIsZero, u)?;
    let vz = c.op1(Op::FpIsZero, v)?;
    let both0 = c.and(&[uz, vz])?;
    let uneg = c.op1(Op::FpIsNegative, u)?;
    // Among zeros: maximum takes +0 (x unless x is -0), minimum takes -0.
    let zero_pick = if max {
        c.ite(uneg, y, x)?
    } else {
        c.ite(uneg, x, y)?
    };
    let lt = c.op2(Op::FpLt, u, v)?;
    let pick = if max { c.ite(lt, y, x)? } else { c.ite(lt, x, y)? };
    let p = c.ite(both0, zero_pick, pick)?;
    let nan = if f.native() {
        float::nan(c, f.fp_sort())?
    } else {
        c.bv(nan_bits(f), f.bits())
    };
    c.ite(anynan, nan, p)
}

fn nan_bits(f: Format) -> u128 {
    let exp = ((1u128 << f.e) - 1) << f.m;
    match f.flavour {
        Flavour::Ieee => exp | 1u128 << (f.m - 1),
        Flavour::FiniteNan => exp | ((1u128 << f.m) - 1),
    }
}

/// Spec `reduce_precision` on a native format: round the significand to `mb` bits (ties to even), then
/// overflow to infinity / underflow to zero outside the range of `eb` exponent bits, keeping the sign.
fn reduce_precision(c: &mut Ctx, f: Format, t: T, eb: u32, mb: u32) -> Res<T> {
    if !f.native() || mb > f.m || eb > f.e {
        return Err(SortError(format!(
            "reduce_precision of {} to ({eb}, {mb})",
            f.name()
        )));
    }
    let w = f.bits();
    let b = c.bits_of(t)?;
    let k = f.m - mb;
    let r = if k == 0 {
        b
    } else {
        // Ties to even on the bit pattern: add (2^(k-1) - 1) + lsb, clear the low k bits.
        let half = c.bv((1u128 << (k - 1)) - 1, w);
        let ksh = c.bv(u128::from(k), w);
        let sh = c.op2(Op::BvLshr, b, ksh)?;
        let one = c.bv(1, w);
        let lsb = c.op2(Op::BvAnd, sh, one)?;
        let inc = c.op2(Op::BvAdd, half, lsb)?;
        let s = c.op2(Op::BvAdd, b, inc)?;
        let mask = c.bv(!((1u128 << k) - 1) & ((1u128 << w) - 1), w);
        c.op2(Op::BvAnd, s, mask)?
    };
    let rv = c.app(Op::FpFromBits { e: f.e, s: f.m + 1 }, &[r])?;
    // Exponent range of the reduced format (normal numbers only).
    let rbias = (1i64 << (eb - 1)) - 1;
    let fbias = f.bias();
    let emax = rbias;
    let emin = 1 - rbias;
    let ex = c.extract(r, f.e + f.m - 1, f.m)?;
    let e_hi = c.bv((emax + fbias) as u128, f.e);
    let e_lo = c.bv((emin + fbias) as u128, f.e);
    let over = c.op2(Op::BvUlt, e_hi, ex)?;
    let under = c.op2(Op::BvUlt, ex, e_lo)?;
    let neg = c.op1(Op::FpIsNegative, t)?;
    let s = f.fp_sort();
    let pinf = float::lit(c, s, ((1u128 << f.e) - 1) << f.m)?;
    let ninf = c.op1(Op::FpNeg, pinf)?;
    let inf = c.ite(neg, ninf, pinf)?;
    let pz = float::lit(c, s, 0)?;
    let nz = c.op1(Op::FpNeg, pz)?;
    let z = c.ite(neg, nz, pz)?;
    let isnan = c.op1(Op::FpIsNaN, t)?;
    let isinf = c.op1(Op::FpIsInfinite, t)?;
    let special = c.or(&[isnan, isinf])?;
    let v = c.ite(under, z, rv)?;
    let v = c.ite(over, inf, v)?;
    c.ite(special, t, v)
}

fn complex(c: &mut Ctx, op: &str, x: Val, y: Option<Val>, f: Format) -> Option<Res<Val>> {
    let ty = Ty::Complex(f);
    let re = Ty::Float(f);
    let r = (|| -> Res<Val> {
        let xi =
            x.im.ok_or_else(|| SortError("complex without an imaginary part".into()))?;
        Ok(match (op, y) {
            ("add" | "subtract", Some(y)) => {
                let yi =
                    y.im.ok_or_else(|| SortError("complex without an imaginary part".into()))?;
                let a = apply(
                    c,
                    op,
                    &Attrs::default(),
                    &[Val::new(re, x.t), Val::new(re, y.t)],
                    re,
                )?;
                let b = apply(
                    c,
                    op,
                    &Attrs::default(),
                    &[Val::new(re, xi), Val::new(re, yi)],
                    re,
                )?;
                Val {
                    ty,
                    t: a.t,
                    im: Some(b.t),
                }
            }
            ("negate", None) => {
                let a = apply(c, op, &Attrs::default(), &[Val::new(re, x.t)], re)?;
                let b = apply(c, op, &Attrs::default(), &[Val::new(re, xi)], re)?;
                Val {
                    ty,
                    t: a.t,
                    im: Some(b.t),
                }
            }
            ("multiply" | "divide", Some(y)) => {
                // The spec's complex multiplication and division leave the evaluation (and its
                // roundings) open: two uninterpreted functions of the four parts.
                let yi =
                    y.im.ok_or_else(|| SortError("complex without an imaginary part".into()))?;
                let s = f.sort();
                let mut parts = Vec::new();
                for p in ["re", "im"] {
                    let name = format!("c{op}_{p}_{}", f.name());
                    c.fun(&name, &[s, s, s, s], s)?;
                    parts.push(c.app(Op::Uf(name), &[x.t, xi, y.t, yi])?);
                }
                Val {
                    ty,
                    t: parts[0],
                    im: Some(parts[1]),
                }
            }
            _ => return Err(SortError(String::new())),
        })
    })();
    match r {
        Err(SortError(m)) if m.is_empty() => None,
        r => Some(r),
    }
}

/// Spec `compare` (directions, `compare_type`); complex values compare lexicographically. `GT` and `GE`
/// are `LT` and `LE` with exchanged operands: IEEE 754's compareQuietGreater(a, b) is
/// compareQuietLess(b, a) (and SMT-LIB defines `fp.gt` so), and likewise for integers.
fn compare(c: &mut Ctx, a: &Attrs, x: Val, y: Val) -> Res<T> {
    let dir = a
        .direction
        .as_deref()
        .ok_or_else(|| SortError("compare without a direction".into()))?;
    if x.ty != y.ty {
        return Err(SortError("compare of two types".into()));
    }
    let ct = a.compare_type.as_deref();
    let (dir, x, y) = match dir {
        "GT" => ("LT", y, x),
        "GE" => ("LE", y, x),
        d => (d, x, y),
    };
    match x.ty {
        Ty::Bool => {
            // false < true.
            let (u, v) = (x.t, y.t);
            let eq = c.eq(u, v)?;
            let nu = c.not(u)?;
            let lt = c.and(&[nu, v])?;
            match dir {
                "EQ" => Ok(eq),
                "NE" => c.not(eq),
                "LT" => Ok(lt),
                "LE" => c.or(&[lt, eq]),
                d => Err(SortError(format!("comparison direction {d}"))),
            }
        }
        Ty::Int { signed, .. } => {
            let signed = match ct {
                Some("SIGNED") => true,
                Some("UNSIGNED") => false,
                None => signed,
                Some(t) => return Err(SortError(format!("compare_type {t} on integers"))),
            };
            let (lt, le) = if signed {
                (Op::BvSlt, Op::BvSle)
            } else {
                (Op::BvUlt, Op::BvUle)
            };
            match dir {
                "EQ" => c.eq(x.t, y.t),
                "NE" => {
                    let e = c.eq(x.t, y.t)?;
                    c.not(e)
                }
                "LT" => c.op2(lt, x.t, y.t),
                "LE" => c.op2(le, x.t, y.t),
                d => Err(SortError(format!("comparison direction {d}"))),
            }
        }
        Ty::Float(f) => {
            if !matches!(ct, None | Some("FLOAT")) {
                return Err(SortError(format!("compare_type {ct:?} on floats is not encoded")));
            }
            let (u, v) = (wide(c, f, x.t)?, wide(c, f, y.t)?);
            // IEEE quiet predicates: false on NaN, except NE.
            match dir {
                "EQ" => c.op2(Op::FpEq, u, v),
                "NE" => {
                    let e = c.op2(Op::FpEq, u, v)?;
                    c.not(e)
                }
                "LT" => c.op2(Op::FpLt, u, v),
                "LE" => c.op2(Op::FpLeq, u, v),
                d => Err(SortError(format!("comparison direction {d}"))),
            }
        }
        Ty::Complex(f) => {
            let (xi, yi) = (
                x.im.ok_or_else(|| SortError("complex".into()))?,
                y.im.ok_or_else(|| SortError("complex".into()))?,
            );
            let re = Ty::Float(f);
            let mk = |c: &mut Ctx, d: &str, p: T, q: T| {
                compare(
                    c,
                    &Attrs {
                        direction: Some(d.into()),
                        compare_type: a.compare_type.clone(),
                        precision: None,
                    },
                    Val::new(re, p),
                    Val::new(re, q),
                )
            };
            let req = mk(c, "EQ", x.t, y.t)?;
            let ieq = mk(c, "EQ", xi, yi)?;
            match dir {
                "EQ" => c.and(&[req, ieq]),
                "NE" => {
                    let e = c.and(&[req, ieq])?;
                    c.not(e)
                }
                _ => {
                    // Lexicographic: strictly smaller real parts, or equal real parts and `dir` on the
                    // imaginary parts.
                    let rs = mk(c, "LT", x.t, y.t)?;
                    let id = mk(c, dir, xi, yi)?;
                    let tie = c.and(&[req, id])?;
                    c.or(&[rs, tie])
                }
            }
        }
    }
}

/// Spec `convert`; inexact cases as XLA computes them (docs/framework/smt.md).
pub fn convert(c: &mut Ctx, x: Val, to: Ty) -> Res<Val> {
    let from = x.ty;
    if from == to {
        return Ok(x);
    }
    let bad = || SortError(format!("convert {} -> {}: no encoding", from.name(), to.name()));
    Ok(match (from, to) {
        (Ty::Bool, Ty::Int { w, .. }) => {
            let one = c.bv(1, w);
            let z = c.bv(0, w);
            Val::new(to, c.ite(x.t, one, z)?)
        }
        (Ty::Bool, Ty::Float(f)) => {
            let one = if f.native() {
                fp_one(c, f.fp_sort())?
            } else {
                let one32 = fp_one(c, Sort::Fp { e: 8, s: 24 })?;
                float::from_f32(c, f, one32)?
            };
            let z = fzero(c, f)?;
            Val::new(to, c.ite(x.t, one, z)?)
        }
        (Ty::Int { .. }, Ty::Bool) => {
            let z = c.bv(0, from.sort().width());
            let e = c.eq(x.t, z)?;
            Val::new(to, c.not(e)?)
        }
        (Ty::Float(f), Ty::Bool) => {
            // Non-zero (NaN included) is true.
            let v = wide(c, f, x.t)?;
            let z = c.op1(Op::FpIsZero, v)?;
            Val::new(to, c.not(z)?)
        }
        // XLA: truncation or sign/zero extension by the source's signedness.
        (Ty::Int { signed, .. }, Ty::Int { w, .. }) => Val::new(to, c.resize(x.t, w, signed)?),
        (Ty::Int { signed, w }, Ty::Float(f)) => {
            // Round to nearest even (XLA `sitofp`/`uitofp`).
            let (e, s) = if f.native() { (f.e, f.m + 1) } else { (8, 24) };
            let o = if signed {
                Op::FpFromSbv { e, s, rm: Rm::Rne }
            } else {
                Op::FpFromUbv { e, s, rm: Rm::Rne }
            };
            if f.native() {
                Val::new(to, c.app(o, &[x.t])?)
            } else if w <= 24 {
                // Exact in Float32: one rounding.
                let v = c.app(o, &[x.t])?;
                Val::new(to, float::from_f32(c, f, v)?)
            } else {
                // Exact in Float64 (w <= 53), then one correct rounding.
                if w > 53 {
                    return Err(bad());
                }
                let o64 = if signed {
                    Op::FpFromSbv {
                        e: 11,
                        s: 53,
                        rm: Rm::Rne,
                    }
                } else {
                    Op::FpFromUbv {
                        e: 11,
                        s: 53,
                        rm: Rm::Rne,
                    }
                };
                let v = c.app(o64, &[x.t])?;
                Val::new(to, float::from_f64(c, f, v)?)
            }
        }
        (Ty::Float(f), Ty::Int { signed, w }) => {
            // Truncation toward zero (spec); out of range saturates and NaN gives 0 (XLA).
            let v = wide(c, f, x.t)?;
            let vs = c.sort(v);
            let Sort::Fp { e, s } = vs else { return Err(bad()) };
            let (lo, hi) = if signed {
                (-(1i128 << (w - 1)), (1i128 << (w - 1)) - 1)
            } else {
                (0, (1i128 << w) - 1)
            };
            let lo_bv = c.bvs(lo, w + 1);
            let hi_bv = c.bvs(hi, w + 1);
            // The bounds as the source type (`static_cast<float>(INT_MAX)` rounds).
            let flo = c.app(Op::FpFromSbv { e, s, rm: Rm::Rne }, &[lo_bv])?;
            let fhi = c.app(Op::FpFromSbv { e, s, rm: Rm::Rne }, &[hi_bv])?;
            let below = c.op2(Op::FpLeq, v, flo)?;
            let above = c.op2(Op::FpLeq, fhi, v)?;
            let isnan = c.op1(Op::FpIsNaN, v)?;
            let conv = if signed {
                c.app(Op::FpToSbv { w, rm: Rm::Rtz }, &[v])?
            } else {
                c.app(Op::FpToUbv { w, rm: Rm::Rtz }, &[v])?
            };
            let lo_t = c.bvs(lo, w);
            let hi_t = c.bvs(hi, w);
            let z = c.bv(0, w);
            let r = c.ite(above, hi_t, conv)?;
            let r = c.ite(below, lo_t, r)?;
            Val::new(to, c.ite(isnan, z, r)?)
        }
        (Ty::Float(f), Ty::Float(g)) => {
            // The exact source value: itself (native) or its Float32 decoding.
            let v = wide(c, f, x.t)?;
            let f32s = Sort::Fp { e: 8, s: 24 };
            let t = if g.native() {
                // Widening is exact; narrowing rounds to nearest even once.
                if c.sort(v) == g.fp_sort() {
                    v
                } else {
                    c.app(
                        Op::FpToFp {
                            e: g.e,
                            s: g.m + 1,
                            rm: Rm::Rne,
                        },
                        &[v],
                    )?
                }
            } else {
                let v32 = match c.sort(v) {
                    s if s == f32s => v,
                    Sort::Fp { e, s } if e <= 8 && s <= 24 => c.app(
                        Op::FpToFp {
                            e: 8,
                            s: 24,
                            rm: Rm::Rne,
                        },
                        &[v],
                    )?,
                    Sort::Fp { e: 11, s: 53 } => {
                        let r = float::from_f64(c, g, v)?;
                        return Ok(Val::new(to, r));
                    }
                    _ => return Err(bad()),
                };
                float::from_f32(c, g, v32)?
            };
            Val::new(to, t)
        }
        (Ty::Bool | Ty::Int { .. } | Ty::Float(_), Ty::Complex(g)) => {
            let re = convert(c, x, Ty::Float(g))?;
            Val {
                ty: to,
                t: re.t,
                im: Some(fzero(c, g)?),
            }
        }
        (Ty::Complex(f), Ty::Complex(g)) => {
            let re = convert(c, Val::new(Ty::Float(f), x.t), Ty::Float(g))?;
            let im = convert(c, Val::new(Ty::Float(f), x.im.ok_or_else(bad)?), Ty::Float(g))?;
            Val {
                ty: to,
                t: re.t,
                im: Some(im.t),
            }
        }
        (Ty::Complex(f), _) => convert(c, Val::new(Ty::Float(f), x.t), to)?,
        _ => return Err(bad()),
    })
}
