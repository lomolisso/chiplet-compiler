//! Binary floating-point formats in the solvers (docs/framework/smt.md, "Floats").
//!
//! Bitwuzla 0.9.1 has only the IEEE interchange sorts (Float16, Float32, Float64, Float128), and both
//! solvers must answer every obligation. So:
//!
//! - **native** formats (f16, f32, f64) are FP terms and use the theory's operations (Brain, Tinelli,
//!   Rümmer, Wahl, ARITH 2015), round to nearest even, no flush to zero;
//! - every other format (bf16, the f8 formats) is its bit pattern; a value is decoded exactly into a
//!   Float32 carrier ([`to_f32`]), computed there, and rounded back by an explicit round-to-nearest-even
//!   on the carrier's bits ([`from_f32`]), with the format's subnormals and its overflow rule: IEEE
//!   formats overflow to infinity, finite-NaN (`FN`) formats to their NaN pattern (OCP OFP8 E4M3,
//!   as ml_dtypes' default conversion does: `float8.h`, `ConvertFrom<kSaturate = false>`).
//!
//! Each carrier encoding is proven once against Z3's native sort ([`lemmas`]): bf16 against
//! `(_ FloatingPoint 8 8)`; E4M3FN, which no SMT-LIB sort models (its top binade is finite), piecewise
//! against `(_ FloatingPoint 4 4)` (equal grids below 2^8) and `(_ FloatingPoint 5 4)` (equal grids in
//! the normal range).

use cc_ir::Elem;
use cc_ir::types::FloatFlavour;

use crate::term::{Ctx, Op, Rm, Sort, SortError, T};

type R = Result<T, SortError>;

/// What a format's all-ones exponent means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Flavour {
    /// Infinities and NaNs (IEEE 754).
    Ieee,
    /// No infinities; the all-ones pattern (exponent and significand) is the only NaN (`FN`).
    FiniteNan,
}

/// A binary float format: `e` exponent bits, `m` trailing significand bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Format {
    pub e: u32,
    pub m: u32,
    pub flavour: Flavour,
}

impl Format {
    pub const F16: Format = Format::ieee(5, 10);
    pub const BF16: Format = Format::ieee(8, 7);
    pub const F32: Format = Format::ieee(8, 23);
    pub const F64: Format = Format::ieee(11, 52);
    pub const E4M3FN: Format = Format {
        e: 4,
        m: 3,
        flavour: Flavour::FiniteNan,
    };
    pub const E5M2: Format = Format::ieee(5, 2);

    pub const fn ieee(e: u32, m: u32) -> Format {
        Format {
            e,
            m,
            flavour: Flavour::Ieee,
        }
    }

    /// The format of a float element type (`None` for other types and unsupported flavours).
    pub fn of(elem: &Elem) -> Option<Format> {
        match elem {
            Elem::Float { exp, mant, flavour } => {
                let flavour = match flavour {
                    FloatFlavour::Ieee => Flavour::Ieee,
                    FloatFlavour::FiniteNan => Flavour::FiniteNan,
                    _ => return None,
                };
                Some(Format {
                    e: u32::from(*exp),
                    m: u32::from(*mant),
                    flavour,
                })
            }
            _ => None,
        }
    }

    pub fn bits(self) -> u32 {
        1 + self.e + self.m
    }

    pub fn bias(self) -> i64 {
        (1i64 << (self.e - 1)) - 1
    }

    /// An IEEE interchange format both solvers have.
    pub fn native(self) -> bool {
        self.flavour == Flavour::Ieee && matches!((self.e, self.m), (5, 10) | (8, 23) | (11, 52))
    }

    /// Fits the Float32 carrier exactly (every value, its subnormals included, is a Float32).
    pub fn fits_f32(self) -> bool {
        self.e <= 8 && self.m <= 23
    }

    /// The sort of its values: an FP sort if native, else its bit pattern.
    pub fn sort(self) -> Sort {
        if self.native() {
            Sort::Fp {
                e: self.e,
                s: self.m + 1,
            }
        } else {
            Sort::Bv(self.bits())
        }
    }

    /// The SMT-LIB sort with the same exponent and significand (IEEE conventions).
    pub fn fp_sort(self) -> Sort {
        Sort::Fp {
            e: self.e,
            s: self.m + 1,
        }
    }

    /// A short name (`f32`, `bf16`, `f8e4m3fn`, `f<e>m<m>`).
    pub fn name(self) -> String {
        match (self.e, self.m, self.flavour) {
            (5, 10, Flavour::Ieee) => "f16".into(),
            (8, 7, Flavour::Ieee) => "bf16".into(),
            (8, 23, Flavour::Ieee) => "f32".into(),
            (11, 52, Flavour::Ieee) => "f64".into(),
            (4, 3, Flavour::FiniteNan) => "f8e4m3fn".into(),
            (5, 2, Flavour::Ieee) => "f8e5m2".into(),
            (e, m, Flavour::Ieee) => format!("f{e}m{m}"),
            (e, m, Flavour::FiniteNan) => format!("f{e}m{m}fn"),
        }
    }

    fn nan_pattern(self) -> u128 {
        let exp = ((1u128 << self.e) - 1) << self.m;
        match self.flavour {
            Flavour::Ieee => exp | 1u128 << (self.m - 1),
            Flavour::FiniteNan => exp | ((1u128 << self.m) - 1),
        }
    }

    fn inf_pattern(self) -> Option<u128> {
        (self.flavour == Flavour::Ieee).then(|| ((1u128 << self.e) - 1) << self.m)
    }

    /// The largest finite magnitude, as bits without the sign.
    fn max_finite(self) -> u128 {
        match self.flavour {
            Flavour::Ieee => (((1u128 << self.e) - 2) << self.m) | ((1u128 << self.m) - 1),
            Flavour::FiniteNan => (((1u128 << self.e) - 1) << self.m) | ((1u128 << self.m) - 2),
        }
    }
}

const F32: Sort = Sort::Fp { e: 8, s: 24 };

/// An FP literal from its bit pattern.
pub fn lit(c: &mut Ctx, s: Sort, bits: u128) -> R {
    let Sort::Fp { e, s: sb } = s else {
        return Err(SortError("an FP literal of a non-FP sort".into()));
    };
    let b = c.bv(bits, e + sb);
    c.app(Op::FpFromBits { e, s: sb }, &[b])
}

/// `2^k` as a Float32 (normal or subnormal; `-149 <= k <= 127`).
fn pow2_f32(c: &mut Ctx, k: i64) -> R {
    let bits = if k >= -126 {
        ((k + 127) as u128) << 23
    } else {
        1u128 << (k + 149)
    };
    lit(c, F32, bits)
}

/// The quiet NaN of an FP sort.
pub fn nan(c: &mut Ctx, s: Sort) -> R {
    let Sort::Fp { e, s: sb } = s else {
        return Err(SortError("NaN of a non-FP sort".into()));
    };
    lit(c, s, ((1u128 << e) - 1) << (sb - 1) | 1u128 << (sb - 2))
}

fn ext32(c: &mut Ctx, x: T) -> R {
    c.resize(x, 32, false)
}

/// Whether a bit pattern of `f` is a NaN.
pub fn is_nan_bits(c: &mut Ctx, f: Format, b: T) -> R {
    let exp = c.extract(b, f.e + f.m - 1, f.m)?;
    let ones = c.bv((1u128 << f.e) - 1, f.e);
    let top = c.eq(exp, ones)?;
    let man = c.extract(b, f.m - 1, 0)?;
    match f.flavour {
        Flavour::Ieee => {
            let z = c.bv(0, f.m);
            let mz = c.eq(man, z)?;
            let nz = c.not(mz)?;
            c.and(&[top, nz])
        }
        Flavour::FiniteNan => {
            let all = c.bv((1u128 << f.m) - 1, f.m);
            let ma = c.eq(man, all)?;
            c.and(&[top, ma])
        }
    }
}

/// The exact Float32 value of a bit pattern of `f` (a format that fits Float32).
pub fn to_f32(c: &mut Ctx, f: Format, b: T) -> R {
    if !f.fits_f32() {
        return Err(SortError(format!("{} does not fit Float32", f.name())));
    }
    c.note(format!("{}.decode", f.name()));
    let w = f.bits();
    let s = c.extract(b, w - 1, w - 1)?;
    let ex = c.extract(b, f.e + f.m - 1, f.m)?;
    let man = c.extract(b, f.m - 1, 0)?;
    // Normal: rebias the exponent, widen the significand.
    let e32 = ext32(c, ex)?;
    let rebias = c.bv((127 - f.bias()) as u128, 32);
    let eb = c.op2(Op::BvAdd, e32, rebias)?;
    let e8 = c.extract(eb, 7, 0)?;
    let pad = c.bv(0, 23 - f.m);
    let man23 = c.op2(Op::Concat, man, pad)?;
    let se = c.op2(Op::Concat, s, e8)?;
    let nb = c.op2(Op::Concat, se, man23)?;
    let normal = c.app(Op::FpFromBits { e: 8, s: 24 }, &[nb])?;
    // Subnormal (and zero): M · 2^(emin - m), exact.
    let mi = c.app(
        Op::FpFromUbv {
            e: 8,
            s: 24,
            rm: Rm::Rne,
        },
        &[man],
    )?;
    let scale = pow2_f32(c, 1 - f.bias() - i64::from(f.m))?;
    let mag = c.op2(Op::FpMul(Rm::Rne), mi, scale)?;
    let neg = c.op1(Op::FpNeg, mag)?;
    let one = c.bv(1, 1);
    let sneg = c.eq(s, one)?;
    let sub = c.ite(sneg, neg, mag)?;
    let z = c.bv(0, f.e);
    let ez = c.eq(ex, z)?;
    let fin = c.ite(ez, sub, normal)?;
    let isnan = is_nan_bits(c, f, b)?;
    let qnan = nan(c, F32)?;
    let mut v = fin;
    if f.flavour == Flavour::Ieee {
        let ones = c.bv((1u128 << f.e) - 1, f.e);
        let top = c.eq(ex, ones)?;
        let inf_bits = c.bv(0x7f80_0000, 32);
        let pinf = c.app(Op::FpFromBits { e: 8, s: 24 }, &[inf_bits])?;
        let ninf = c.op1(Op::FpNeg, pinf)?;
        let inf = c.ite(sneg, ninf, pinf)?;
        v = c.ite(top, inf, v)?;
    }
    c.ite(isnan, qnan, v)
}

/// Round a Float32 to `f` (round to nearest, ties to even), as `f`'s bit pattern.
pub fn from_f32(c: &mut Ctx, f: Format, x: T) -> R {
    if !f.fits_f32() || f.m >= 23 {
        return Err(SortError(format!("{} is not narrower than Float32", f.name())));
    }
    c.note(format!("{}.round", f.name()));
    let b = c.bits_of(x)?;
    let s = c.extract(b, 31, 31)?;
    let e8 = c.extract(b, 30, 23)?;
    let m23 = c.extract(b, 22, 0)?;
    let ee = ext32(c, e8)?;
    let mm = ext32(c, m23)?;
    let k32 = |c: &mut Ctx, v: i64| c.bvs(i128::from(v), 32);
    let zero = k32(c, 0);
    let one = k32(c, 1);
    let ez = c.eq(ee, zero)?;
    let eeff = c.ite(ez, one, ee)?;
    let hidden = k32(c, 1 << 23);
    let mh = c.op2(Op::BvOr, mm, hidden)?;
    let sig = c.ite(ez, mm, mh)?;
    let emin = 1 - f.bias();
    let m = i64::from(f.m);
    // Normal result iff the unbiased exponent is >= emin.
    let thr = k32(c, emin + 127);
    let normal = c.op2(Op::BvUle, thr, eeff)?;
    let kn = k32(c, 23 - m);
    let base = k32(c, 150 + emin - m);
    let ks = c.op2(Op::BvSub, base, eeff)?;
    let k31 = k32(c, 31);
    let over31 = c.op2(Op::BvUlt, k31, ks)?;
    let ksc = c.ite(over31, k31, ks)?;
    let k = c.ite(normal, kn, ksc)?;
    let q = c.op2(Op::BvLshr, sig, k)?;
    let km1 = c.op2(Op::BvSub, k, one)?;
    let gsh = c.op2(Op::BvLshr, sig, km1)?;
    let g = c.op2(Op::BvAnd, gsh, one)?;
    let bit = c.op2(Op::BvShl, one, km1)?;
    let mask = c.op2(Op::BvSub, bit, one)?;
    let low = c.op2(Op::BvAnd, sig, mask)?;
    let sticky0 = c.eq(low, zero)?;
    let sticky = c.not(sticky0)?;
    let l = c.op2(Op::BvAnd, q, one)?;
    let g1 = c.eq(g, one)?;
    let l1 = c.eq(l, one)?;
    let sl = c.or(&[sticky, l1])?;
    let up = c.and(&[g1, sl])?;
    let inc = c.ite(up, one, zero)?;
    let n = c.op2(Op::BvAdd, q, inc)?;
    // Packed magnitude: ((exponent - 1) << m) + N carries into the exponent on a round-up.
    let rebias = k32(c, 127 - f.bias() + 1);
    let eb = c.op2(Op::BvSub, eeff, rebias)?;
    let msh = k32(c, m);
    let ebs = c.op2(Op::BvShl, eb, msh)?;
    let pn = c.op2(Op::BvAdd, ebs, n)?;
    let packed = c.ite(normal, pn, n)?;
    let maxf = c.bv(f.max_finite(), 32);
    let over = c.op2(Op::BvUlt, maxf, packed)?;
    let nanp = c.bv(f.nan_pattern(), 32);
    let big = match f.inf_pattern() {
        Some(i) => c.bv(i, 32),
        None => nanp,
    };
    let isinf = c.op1(Op::FpIsInfinite, x)?;
    let isnan = c.op1(Op::FpIsNaN, x)?;
    let ovf = c.or(&[isinf, over])?;
    let mag0 = c.ite(ovf, big, packed)?;
    let mag = c.ite(isnan, nanp, mag0)?;
    let mag = c.extract(mag, f.e + f.m - 1, 0)?;
    c.op2(Op::Concat, s, mag)
}

/// Round a Float64 to `f` (round to nearest, ties to even), through Float32 rounded to odd: truncate,
/// and set the last bit when inexact. With 24 >= m + 3 significand bits, rounding to odd first and to
/// nearest second equals one rounding to nearest (Boldo, Melquiond, "Emulation of FMA and correctly
/// rounded sums: proved algorithms using rounding to odd", IEEE TC 57(4), 2008); the lemma checks it.
pub fn from_f64(c: &mut Ctx, f: Format, x: T) -> R {
    c.note(format!("{}.from_f64", f.name()));
    let y = c.app(Op::FpToFp { e: 8, s: 24, rm: Rm::Rtz }, &[x])?;
    let back = c.app(Op::FpToFp { e: 11, s: 53, rm: Rm::Rne }, &[y])?;
    let exact = c.eq(back, x)?;
    let b = c.bits_of(y)?;
    let one = c.bv(1, 32);
    let odd = c.op2(Op::BvOr, b, one)?;
    let isnan = c.op1(Op::FpIsNaN, y)?;
    let keep = c.or(&[exact, isnan])?;
    let rb = c.ite(keep, b, odd)?;
    let y2 = c.app(Op::FpFromBits { e: 8, s: 24 }, &[rb])?;
    from_f32(c, f, y2)
}

/// Bit-pattern equality of two values of `f`, all NaNs equal.
pub fn same_bits(c: &mut Ctx, f: Format, a: T, b: T) -> R {
    if a == b {
        return Ok(c.bool(true));
    }
    let na = is_nan_bits(c, f, a)?;
    let nb = is_nan_bits(c, f, b)?;
    let both = c.and(&[na, nb])?;
    let eq = c.eq(a, b)?;
    c.or(&[both, eq])
}

/// The encoding lemmas for the encodings a set of queries used (`Ctx::used`): `(file name, query)`.
/// Each must be `unsat` (in Z3: they state the native sorts Bitwuzla does not have).
pub fn lemmas(used: &std::collections::BTreeSet<String>) -> Result<Vec<(String, String)>, SortError> {
    let mut out = Vec::new();
    for u in used {
        let Some((fmt, what)) = u.split_once('.') else {
            continue;
        };
        let f = by_name(fmt).ok_or_else(|| SortError(format!("{u}: no carrier format {fmt}")))?;
        let text = match (f.flavour, what) {
            (Flavour::Ieee, "decode") => ieee_decode(f)?,
            (Flavour::Ieee, "round") => ieee_round(f)?,
            (Flavour::Ieee, op @ ("add" | "sub" | "mul" | "div" | "sqrt")) => ieee_op(f, op)?,
            (Flavour::Ieee, "from_f64") => ieee_from_f64(f)?,
            (Flavour::FiniteNan, "decode") => fn_decode(f)?,
            (Flavour::FiniteNan, "round") => fn_round(f)?,
            // An encoding without a lemma is never trusted.
            _ => return Err(SortError(format!("{u}: no encoding lemma"))),
        };
        out.push((format!("{u}.smt2"), text));
    }
    Ok(out)
}

/// The formats with carrier encodings, by [`Format::name`].
pub fn by_name(n: &str) -> Option<Format> {
    [Format::BF16, Format::E4M3FN, Format::E5M2]
        .into_iter()
        .find(|f| f.name() == n)
}

fn finish(c: &Ctx, comment: &str, goal: T) -> Result<String, SortError> {
    let mut s = crate::Script::new();
    s.comment(comment);
    c.render(&mut s, &[goal]);
    Ok(s.render())
}

/// `to_f32` is the exact widening of the native sort.
fn ieee_decode(f: Format) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let b = c.var("b", Sort::Bv(f.bits()))?;
    let v = to_f32(&mut c, f, b)?;
    let (e, s) = (f.e, f.m + 1);
    let r = c.app(Op::FpFromBits { e, s }, &[b])?;
    let w = c.app(Op::FpToFp { e: 8, s: 24, rm: Rm::Rne }, &[r])?;
    let eq = c.eq(v, w)?;
    let goal = c.not(eq)?;
    finish(&c, &format!("{}: the Float32 decoding is the exact widening (expect unsat)", f.name()), goal)
}

/// `from_f32` is the theory's rounding to the native sort.
fn ieee_round(f: Format) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let x = c.var("x", F32)?;
    let b = from_f32(&mut c, f, x)?;
    let (e, s) = (f.e, f.m + 1);
    let got = c.app(Op::FpFromBits { e, s }, &[b])?;
    let want = c.app(Op::FpToFp { e, s, rm: Rm::Rne }, &[x])?;
    let eq = c.eq(got, want)?;
    let goal = c.not(eq)?;
    finish(&c, &format!("{}: rounding from Float32 equals to_fp RNE (expect unsat)", f.name()), goal)
}

/// An arithmetic operation computed in Float32 and rounded equals the native one (double rounding is
/// harmless here; the lemma decides it).
fn ieee_op(f: Format, op: &str) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let (e, s) = (f.e, f.m + 1);
    let a = c.var("a", Sort::Bv(f.bits()))?;
    let b = c.var("b", Sort::Bv(f.bits()))?;
    let da = to_f32(&mut c, f, a)?;
    let db = to_f32(&mut c, f, b)?;
    let na = c.app(Op::FpFromBits { e, s }, &[a])?;
    let nb = c.app(Op::FpFromBits { e, s }, &[b])?;
    let (wide, native) = match op {
        "add" => (c.op2(Op::FpAdd(Rm::Rne), da, db)?, c.op2(Op::FpAdd(Rm::Rne), na, nb)?),
        "sub" => (c.op2(Op::FpSub(Rm::Rne), da, db)?, c.op2(Op::FpSub(Rm::Rne), na, nb)?),
        "mul" => (c.op2(Op::FpMul(Rm::Rne), da, db)?, c.op2(Op::FpMul(Rm::Rne), na, nb)?),
        "div" => (c.op2(Op::FpDiv(Rm::Rne), da, db)?, c.op2(Op::FpDiv(Rm::Rne), na, nb)?),
        _ => (c.op1(Op::FpSqrt(Rm::Rne), da)?, c.op1(Op::FpSqrt(Rm::Rne), na)?),
    };
    let r = from_f32(&mut c, f, wide)?;
    let got = c.app(Op::FpFromBits { e, s }, &[r])?;
    let eq = c.eq(got, native)?;
    let goal = c.not(eq)?;
    finish(
        &c,
        &format!("{}: {op} in Float32 then rounded equals the native {op} (expect unsat)", f.name()),
        goal,
    )
}

/// Rounding a Float64 through Float32 rounded to odd equals the native rounding.
fn ieee_from_f64(f: Format) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let (e, s) = (f.e, f.m + 1);
    let x = c.var("x", Sort::Fp { e: 11, s: 53 })?;
    let r = from_f64(&mut c, f, x)?;
    let got = c.app(Op::FpFromBits { e, s }, &[r])?;
    let want = c.app(Op::FpToFp { e, s, rm: Rm::Rne }, &[x])?;
    let eq = c.eq(got, want)?;
    let goal = c.not(eq)?;
    finish(
        &c,
        &format!("{}: rounding from Float64 through Float32 rounded to odd equals to_fp RNE (expect unsat)", f.name()),
        goal,
    )
}

/// The `(_ FloatingPoint (e+1) (m+1))` value of a normal pattern of a finite-NaN format: its binades
/// shifted by one exponent bit (the same grid in the normal range).
fn fn_as_wider(c: &mut Ctx, f: Format, b: T) -> R {
    let w = f.bits();
    let s = c.extract(b, w - 1, w - 1)?;
    let ex = c.extract(b, f.e + f.m - 1, f.m)?;
    let man = c.extract(b, f.m - 1, 0)?;
    let ex1 = c.resize(ex, f.e + 1, false)?;
    // Wider bias 2^e - 1 versus 2^(e-1) - 1.
    let d = c.bv(((1i64 << f.e) - 1 - f.bias()) as u128, f.e + 1);
    let e2 = c.op2(Op::BvAdd, ex1, d)?;
    let se = c.op2(Op::Concat, s, e2)?;
    let nb = c.op2(Op::Concat, se, man)?;
    c.app(
        Op::FpFromBits {
            e: f.e + 1,
            s: f.m + 1,
        },
        &[nb],
    )
}

fn fn_decode(f: Format) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let b = c.var("b", Sort::Bv(f.bits()))?;
    let v = to_f32(&mut c, f, b)?;
    let (e, s) = (f.e, f.m + 1);
    let ex = c.extract(b, f.e + f.m - 1, f.m)?;
    let z = c.bv(0, f.e);
    let sub = c.eq(ex, z)?;
    // Subnormals: the IEEE sort of the same width has the same layout there.
    let ieee = c.app(Op::FpFromBits { e, s }, &[b])?;
    let w1 = c.app(Op::FpToFp { e: 8, s: 24, rm: Rm::Rne }, &[ieee])?;
    let wider = fn_as_wider(&mut c, f, b)?;
    let w2 = c.app(Op::FpToFp { e: 8, s: 24, rm: Rm::Rne }, &[wider])?;
    let isnan = is_nan_bits(&mut c, f, b)?;
    let vnan = c.op1(Op::FpIsNaN, v)?;
    let e1 = c.eq(v, w1)?;
    let e2 = c.eq(v, w2)?;
    let c1 = c.app(Op::Implies, &[sub, e1])?;
    let ns = c.not(sub)?;
    let nn = c.not(isnan)?;
    let normal = c.and(&[ns, nn])?;
    let c2 = c.app(Op::Implies, &[normal, e2])?;
    let c3 = c.eq(isnan, vnan)?;
    let all = c.and(&[c1, c2, c3])?;
    let goal = c.not(all)?;
    finish(
        &c,
        &format!(
            "{}: the Float32 decoding agrees with (_ FloatingPoint {} {}) on subnormals and with (_ FloatingPoint {} {}) on normals (expect unsat)",
            f.name(),
            e,
            s,
            e + 1,
            s
        ),
        goal,
    )
}

fn fn_round(f: Format) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let x = c.var("x", F32)?;
    let b = from_f32(&mut c, f, x)?;
    let (e, s) = (f.e, f.m + 1);
    let ax = c.op1(Op::FpAbs, x)?;
    // Values up to the IEEE sort's largest finite value round alike in both.
    let ieee_max = lit(&mut c, F32, f32_bits_of_ieee_max(f))?;
    let small = c.op2(Op::FpLeq, ax, ieee_max)?;
    let got1 = c.app(Op::FpFromBits { e, s }, &[b])?;
    let want1 = c.app(Op::FpToFp { e, s, rm: Rm::Rne }, &[x])?;
    let e1 = c.eq(got1, want1)?;
    let c1 = c.app(Op::Implies, &[small, e1])?;
    // From the smallest normal up, the wider sort has the same grid; results past the largest finite
    // pattern are the NaN pattern.
    let min_normal = pow2_f32(&mut c, 1 - f.bias())?;
    let isinf = c.op1(Op::FpIsInfinite, x)?;
    let ninf = c.not(isinf)?;
    let ge = c.op2(Op::FpLeq, min_normal, ax)?;
    let big = c.and(&[ge, ninf])?;
    let ws = Sort::Fp { e: e + 1, s };
    let y = c.app(
        Op::FpToFp {
            e: e + 1,
            s,
            rm: Rm::Rne,
        },
        &[x],
    )?;
    let ay = c.op1(Op::FpAbs, y)?;
    // The first wider value past the largest finite pattern: the NaN pattern's value.
    let limit = lit(&mut c, ws, wider_bits_of_nan_value(f))?;
    let past = c.op2(Op::FpLeq, limit, ay)?;
    let isnan = is_nan_bits(&mut c, f, b)?;
    let wider = fn_as_wider(&mut c, f, b)?;
    let eqy = c.eq(wider, y)?;
    let br = c.ite(past, isnan, eqy)?;
    let c2 = c.app(Op::Implies, &[big, br])?;
    // NaN and infinities give the NaN pattern.
    let xnan = c.op1(Op::FpIsNaN, x)?;
    let special = c.or(&[xnan, isinf])?;
    let c3 = c.app(Op::Implies, &[special, isnan])?;
    let all = c.and(&[c1, c2, c3])?;
    let goal = c.not(all)?;
    finish(
        &c,
        &format!(
            "{}: rounding from Float32 agrees with (_ FloatingPoint {e} {s}) up to its largest finite value, with (_ FloatingPoint {} {s}) from the smallest normal, and overflows to the NaN pattern (expect unsat)",
            f.name(),
            e + 1
        ),
        goal,
    )
}

/// Float32 bits of the largest finite value of the IEEE sort with `f`'s exponent and significand.
fn f32_bits_of_ieee_max(f: Format) -> u128 {
    let emax = (1i64 << f.e) - 2 - f.bias();
    let man = ((1u128 << f.m) - 1) << (23 - f.m);
    (((emax + 127) as u128) << 23) | man
}

/// Bits, in the `(e+1, m+1)` sort, of the value the NaN pattern of a finite-NaN format would have.
fn wider_bits_of_nan_value(f: Format) -> u128 {
    let top = (1i64 << f.e) - 1; // the all-ones exponent field
    let wide_bias = (1i64 << f.e) - 1;
    let ex = (top - f.bias() + wide_bias) as u128;
    (ex << f.m) | ((1u128 << f.m) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With `CC_SMT_DUMP=<dir>`, write every lemma there (for running the solvers by hand).
    #[test]
    fn lemmas_render() {
        let used: std::collections::BTreeSet<String> = [
            "bf16.decode",
            "bf16.round",
            "bf16.add",
            "bf16.sub",
            "bf16.mul",
            "bf16.div",
            "bf16.sqrt",
            "bf16.from_f64",
            "f8e4m3fn.decode",
            "f8e4m3fn.round",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let ls = lemmas(&used).unwrap();
        assert_eq!(ls.len(), used.len());
        if let Ok(d) = std::env::var("CC_SMT_DUMP") {
            std::fs::create_dir_all(&d).unwrap();
            for (f, t) in &ls {
                std::fs::write(std::path::Path::new(&d).join(f), t).unwrap();
            }
        }
        let bad: std::collections::BTreeSet<String> = ["f8e4m3fn.add".to_string()].into_iter().collect();
        assert!(lemmas(&bad).is_err());
    }
}
