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

    /// The subnormal rule of design §6.4: f32 and bf16 flush (DAZ and FTZ), every other format keeps
    /// gradual underflow.
    pub fn flushes(self) -> bool {
        self == Format::F32 || self == Format::BF16
    }

    /// Conversions into the format saturate (design §6.4: the 8-bit formats, as OCP OFP8's saturating
    /// mode).
    pub fn saturates(self) -> bool {
        self.bits() == 8
    }

    /// The smallest normal exponent.
    pub fn emin(self) -> i64 {
        1 - self.bias()
    }

    /// The flush threshold of a flushing format, as an `f64`: `2^emin - 2^(emin - m - 2)`. A value
    /// rounded to `m + 1` significand bits with an unbounded exponent is below `2^emin` exactly when its
    /// magnitude is below this (the midpoint under `2^emin` rounds up to it, its even neighbour).
    pub fn flush_threshold(self) -> f64 {
        let e = self.emin() as i32;
        2f64.powi(e) - 2f64.powi(e - self.m as i32 - 2)
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
const F64: Sort = Sort::Fp { e: 11, s: 53 };

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

/// Zero of the sort and sign of the FP term `x`.
fn signed_zero(c: &mut Ctx, x: T) -> R {
    let s = c.sort(x);
    let pz = lit(c, s, 0)?;
    let nz = c.op1(Op::FpNeg, pz)?;
    let neg = c.op1(Op::FpIsNegative, x)?;
    c.ite(neg, nz, pz)
}

/// `ite(fp.isSubnormal x, ±0, x)` on an FP term (DAZ, the sign kept).
fn daz_fp(c: &mut Ctx, x: T) -> R {
    let sub = c.op1(Op::FpIsSubnormal, x)?;
    let z = signed_zero(c, x)?;
    c.ite(sub, z, x)
}

/// A value of `f` (its FP term or bit pattern) as the format's operators read it: a subnormal reads as
/// zero of its sign in a flushing format (DAZ, design §6.4), unchanged otherwise.
pub fn daz(c: &mut Ctx, f: Format, t: T) -> R {
    if !f.flushes() {
        return Ok(t);
    }
    if f.native() {
        return daz_fp(c, t);
    }
    let w = f.bits();
    let ex = c.extract(t, f.e + f.m - 1, f.m)?;
    let z = c.bv(0, f.e);
    let ez = c.eq(ex, z)?;
    let s = c.extract(t, w - 1, w - 1)?;
    let zeros = c.bv(0, w - 1);
    let sz = c.op2(Op::Concat, s, zeros)?;
    c.ite(ez, sz, t)
}

/// FTZ of a native result that is exact whenever it is below `2^emin` (a sum or difference of two
/// values of the format, a remainder): such a result is flushed exactly when it is subnormal.
pub fn flush_exact(c: &mut Ctx, f: Format, r: T) -> R {
    if f.flushes() { daz_fp(c, r) } else { Ok(r) }
}

/// Round a Float64 to the native format `f` (round to nearest even, overflow to infinity), flushing
/// below `2^emin` when `f` flushes: zero of its sign when `|x|` is below [`Format::flush_threshold`].
pub fn native_from_f64(c: &mut Ctx, f: Format, x: T) -> R {
    if !f.native() {
        return Err(SortError(format!("{} is not native", f.name())));
    }
    if f == Format::F64 {
        return Ok(x);
    }
    let (e, s) = (f.e, f.m + 1);
    let r = c.app(Op::FpToFp { e, s, rm: Rm::Rne }, &[x])?;
    if !f.flushes() {
        return Ok(r);
    }
    c.note(format!("{}.from_f64", f.name()));
    let thr = lit(c, F64, u128::from(f.flush_threshold().to_bits()))?;
    let ax = c.op1(Op::FpAbs, x)?;
    let tiny = c.op2(Op::FpLt, ax, thr)?;
    let z = signed_zero(c, r)?;
    c.ite(tiny, z, r)
}

/// Move the exponent of a normal or zero FP term by `k` (no multiplier: `fp.mul` by a power of two is a
/// full significand multiplier to a bit-blasting solver). Zeros, infinities and NaN are kept; the caller
/// guarantees the result stays normal.
fn shift_exp(c: &mut Ctx, x: T, k: i64) -> R {
    let Sort::Fp { e, s } = c.sort(x) else {
        return Err(SortError("an exponent shift of a non-FP term".into()));
    };
    let b = c.bits_of(x)?;
    let w = e + s;
    let sign = c.extract(b, w - 1, w - 1)?;
    let ex = c.extract(b, w - 2, s - 1)?;
    let frac = c.extract(b, s - 2, 0)?;
    let d = c.bvs(i128::from(k), e);
    let moved = c.op2(Op::BvAdd, ex, d)?;
    let lo = c.op2(Op::Concat, moved, frac)?;
    let nb = c.op2(Op::Concat, sign, lo)?;
    let y = c.app(Op::FpFromBits { e, s }, &[nb])?;
    let z = c.op1(Op::FpIsZero, x)?;
    let n = c.op1(Op::FpIsNaN, x)?;
    let i = c.op1(Op::FpIsInfinite, x)?;
    let keep = c.or(&[z, n, i])?;
    c.ite(keep, x, y)
}

/// The scale of [`scaled_op`]: a product or quotient that may be tiny is computed `2^SCALE` larger.
const SCALE: i64 = 32;

/// A multiply or divide of DAZ operands of f32 under the flush rule, in Float32 alone. When the
/// exponent fields say the result may be below `2^-125` (`e_x + e_y < -125` for a product, `e_x - e_y <
/// -124` for a quotient, unbiased), the left operand is scaled by `2^32` (its exponent field moved:
/// exact, as it is then below `2^4`, and no infinity arises), so the result `r` is the rounding to 24 bits
/// of the exact value times `2^32`, in the normal range wherever the decision matters: the result is
/// zero of its sign when `|r| < 2^-94`, else `r · 2^-32` (normal, exact). Otherwise the result is at least
/// `2^-125` and the operation's own. One Float32 multiplier or divider, as the IEEE operation has; the
/// lemma ([`scaled_lemma`]) proves the decision for every exact value.
fn scaled_op(c: &mut Ctx, f: Format, op: Arith, x: T, y: T) -> R {
    let (tp, xa) = scaled_pre(c, f, op, x, y)?;
    let r = op.apply(c, &[xa, y])?;
    scaled_post(c, tp, r, 0)
}

/// Whether the result may be tiny (from the exponent fields), and the left operand to use.
fn scaled_pre(c: &mut Ctx, f: Format, op: Arith, x: T, y: T) -> Result<(T, T), SortError> {
    if f != Format::F32 {
        return Err(SortError(format!("{}: no scaled encoding", f.name())));
    }
    let bx = c.bits_of(x)?;
    let by = c.bits_of(y)?;
    let ex = c.extract(bx, 30, 23)?;
    let ey = c.extract(by, 30, 23)?;
    let ex = c.resize(ex, 10, false)?;
    let ey = c.resize(ey, 10, false)?;
    let tp = match op {
        // Biased: e_x + e_y < 2 · 127 - 125.
        Arith::Mul => {
            let sum = c.op2(Op::BvAdd, ex, ey)?;
            let lim = c.bv(129, 10);
            c.op2(Op::BvUlt, sum, lim)?
        }
        // Biased: e_x + 124 < e_y.
        _ => {
            let off = c.bv(124, 10);
            let l = c.op2(Op::BvAdd, ex, off)?;
            c.op2(Op::BvUlt, l, ey)?
        }
    };
    let xs = shift_exp(c, x, SCALE)?;
    let xa = c.ite(tp, xs, x)?;
    Ok((tp, xa))
}

/// The result from the Float32 result `r` of the (possibly scaled) operation (`off` moves the flush
/// limit by binades: 0, except in a negative control).
fn scaled_post(c: &mut Ctx, tp: T, r: T, off: i64) -> R {
    let lim = lit(c, F32, ((127 - 126 + SCALE + off) as u128) << 23)?;
    let ar = c.op1(Op::FpAbs, r)?;
    let tiny = c.op2(Op::FpLt, ar, lim)?;
    let z = signed_zero(c, r)?;
    let back = shift_exp(c, r, -SCALE)?;
    let small = c.ite(tiny, z, back)?;
    c.ite(tp, small, r)
}

/// The correctly rounded arithmetic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
}

impl Arith {
    pub fn name(self) -> &'static str {
        match self {
            Arith::Add => "add",
            Arith::Sub => "sub",
            Arith::Mul => "mul",
            Arith::Div => "div",
            Arith::Sqrt => "sqrt",
        }
    }

    pub fn by_name(n: &str) -> Option<Arith> {
        [Arith::Add, Arith::Sub, Arith::Mul, Arith::Div, Arith::Sqrt]
            .into_iter()
            .find(|a| a.name() == n)
    }

    pub fn arity(self) -> usize {
        if self == Arith::Sqrt { 1 } else { 2 }
    }

    /// The theory's operation (round to nearest even) on FP terms of one sort.
    fn apply(self, c: &mut Ctx, xs: &[T]) -> R {
        let rm = Rm::Rne;
        match (self, xs) {
            (Arith::Add, [x, y]) => c.op2(Op::FpAdd(rm), *x, *y),
            (Arith::Sub, [x, y]) => c.op2(Op::FpSub(rm), *x, *y),
            (Arith::Mul, [x, y]) => c.op2(Op::FpMul(rm), *x, *y),
            (Arith::Div, [x, y]) => c.op2(Op::FpDiv(rm), *x, *y),
            (Arith::Sqrt, [x]) => c.op1(Op::FpSqrt(rm), *x),
            _ => Err(SortError(format!("{}: wrong arity", self.name()))),
        }
    }
}

/// An arithmetic operation of format `f` on values of `f` (FP terms or bit patterns), as design §6.4
/// specifies it: the IEEE operation, rounded to nearest even; in a flushing format on DAZ operands,
/// computed with two more exponent bits (rounding with an unbounded exponent) and flushed to zero of its
/// sign below `2^emin`. Bitwuzla has no such sort, so the encodings below compute in interchange sorts
/// and the lemmas ([`lemmas`], Z3) prove each equal to that statement:
///
/// - gradual native formats: the theory's operation;
/// - f32 add, subtract: in Float32, flushed when subnormal (an underflowing sum is exact: Hauser, "Handling
///   floating-point exceptions in numeric programs", ACM TOPLAS 18(2), 1996; the lemma proves it);
/// - f32 multiply, divide: in Float32 with the left operand scaled when the result may be tiny
///   ([`scaled_op`]);
/// - f32 square root: in Float32 (the root of a normal f32 is normal: the lemma proves it is never
///   subnormal);
/// - carrier formats: in the Float32 carrier, then [`from_f32`] with the format's rule.
pub fn arith(c: &mut Ctx, f: Format, op: Arith, args: &[T]) -> R {
    let mut xs = Vec::with_capacity(args.len());
    for &a in args {
        xs.push(daz(c, f, a)?);
    }
    if f.native() {
        if !f.flushes() {
            return op.apply(c, &xs);
        }
        return match op {
            Arith::Add | Arith::Sub => {
                c.note(format!("{}.{}", f.name(), op.name()));
                let r = op.apply(c, &xs)?;
                flush_exact(c, f, r)
            }
            Arith::Sqrt => {
                c.note(format!("{}.{}", f.name(), op.name()));
                op.apply(c, &xs)
            }
            Arith::Mul | Arith::Div => {
                c.note(format!("{}.{}", f.name(), op.name()));
                scaled_op(c, f, op, xs[0], xs[1])
            }
        };
    }
    let mut ws = Vec::with_capacity(xs.len());
    for x in xs {
        ws.push(to_f32(c, f, x)?);
    }
    let r = op.apply(c, &ws)?;
    c.note(format!("{}.{}", f.name(), op.name()));
    from_f32(c, f, r)
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

/// Round a Float32 to `f` (round to nearest, ties to even), as `f`'s bit pattern: the rounding of an
/// arithmetic result. Overflow gives infinity (IEEE formats) or the NaN pattern (finite-NaN formats:
/// design §6.4, an E4M3FN result beyond 448 is NaN); a flushing format flushes below `2^emin`.
pub fn from_f32(c: &mut Ctx, f: Format, x: T) -> R {
    c.note(format!("{}.round", f.name()));
    round_f32(c, f, x, false)
}

/// Convert a Float32 to `f`: [`from_f32`], except that a saturating format (FP8) maps overflow and
/// infinities to its largest finite value of the same sign (design §6.4; OCP OFP8's saturating mode).
pub fn convert_from_f32(c: &mut Ctx, f: Format, x: T) -> R {
    if !f.saturates() {
        return from_f32(c, f, x);
    }
    c.note(format!("{}.saturate", f.name()));
    round_f32(c, f, x, true)
}

fn round_f32(c: &mut Ctx, f: Format, x: T, saturate: bool) -> R {
    if !f.fits_f32() || f.m >= 23 {
        return Err(SortError(format!("{} is not narrower than Float32", f.name())));
    }
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
    let big = match (saturate, f.inf_pattern()) {
        (true, _) => maxf,
        (false, Some(i)) => c.bv(i, 32),
        (false, None) => nanp,
    };
    let isinf = c.op1(Op::FpIsInfinite, x)?;
    let isnan = c.op1(Op::FpIsNaN, x)?;
    let ovf = c.or(&[isinf, over])?;
    let mag0 = c.ite(ovf, big, packed)?;
    let mag = c.ite(isnan, nanp, mag0)?;
    let mag = c.extract(mag, f.e + f.m - 1, 0)?;
    let r = c.op2(Op::Concat, s, mag)?;
    if !f.flushes() {
        return Ok(r);
    }
    // FTZ: below the threshold the value rounded with an unbounded exponent is below 2^emin.
    let thr = lit(c, F32, u128::from((f.flush_threshold() as f32).to_bits()))?;
    let ax = c.op1(Op::FpAbs, x)?;
    let tiny = c.op2(Op::FpLt, ax, thr)?;
    let zeros = c.bv(0, f.e + f.m);
    let z = c.op2(Op::Concat, s, zeros)?;
    c.ite(tiny, z, r)
}

/// Round a Float64 to `f` (round to nearest, ties to even), through Float32 rounded to odd: truncate,
/// and set the last bit when inexact. With 24 >= m + 3 significand bits, rounding to odd first and to
/// nearest second equals one rounding to nearest (Boldo, Melquiond, "Emulation of FMA and correctly
/// rounded sums: proved algorithms using rounding to odd", IEEE TC 57(4), 2008); the lemma checks it.
/// Saturating formats saturate ([`convert_from_f32`]).
pub fn from_f64(c: &mut Ctx, f: Format, x: T) -> R {
    c.note(format!("{}.from_f64", f.name()));
    let y = c.app(
        Op::FpToFp {
            e: 8,
            s: 24,
            rm: Rm::Rtz,
        },
        &[x],
    )?;
    let back = c.app(
        Op::FpToFp {
            e: 11,
            s: 53,
            rm: Rm::Rne,
        },
        &[y],
    )?;
    let exact = c.eq(back, x)?;
    let b = c.bits_of(y)?;
    let one = c.bv(1, 32);
    let odd = c.op2(Op::BvOr, b, one)?;
    let isnan = c.op1(Op::FpIsNaN, y)?;
    let keep = c.or(&[exact, isnan])?;
    let rb = c.ite(keep, b, odd)?;
    let y2 = c.app(Op::FpFromBits { e: 8, s: 24 }, &[rb])?;
    convert_from_f32(c, f, y2)
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
/// Each must be `unsat` (in Z3: they state sorts Bitwuzla does not have). Each compares an encoding with
/// design §6.4 stated directly in SMT-LIB ([`reference`]): the rounding in the format's own sort, for a
/// flushing format with two more exponent bits, flushed below `2^emin` and narrowed.
pub fn lemmas(used: &std::collections::BTreeSet<String>) -> Result<Vec<(String, String)>, SortError> {
    let mut out = Vec::new();
    for u in used {
        let Some((fmt, what)) = u.split_once('.') else {
            continue;
        };
        let f = by_name(fmt).ok_or_else(|| SortError(format!("{u}: no encoded format {fmt}")))?;
        let text = match (f.flavour, what) {
            (Flavour::Ieee, "decode") if !f.native() => ieee_decode(f)?,
            (Flavour::Ieee, "round") if !f.native() => ieee_round(f, false)?,
            (Flavour::Ieee, "saturate") if !f.native() => ieee_round(f, true)?,
            (Flavour::Ieee, "from_f64") => ieee_from_f64(f)?,
            (Flavour::Ieee, "sqrt") if f.native() => native_sqrt(f)?,
            (Flavour::Ieee, op @ ("mul" | "div")) if f.native() => {
                scaled_lemma(f, Arith::by_name(op).unwrap_or(Arith::Mul))?
            }
            (Flavour::Ieee, op @ ("add" | "sub")) if f.native() => {
                native_sum(f, Arith::by_name(op).unwrap_or(Arith::Add))?
            }
            (Flavour::Ieee, op) if Arith::by_name(op).is_some() => {
                ieee_op(f, Arith::by_name(op).unwrap_or(Arith::Add))?
            }
            (Flavour::FiniteNan, "decode") => fn_decode(f)?,
            (Flavour::FiniteNan, "round") => fn_round(f, false)?,
            (Flavour::FiniteNan, "saturate") => fn_round(f, true)?,
            // An encoding without a lemma is never trusted.
            _ => return Err(SortError(format!("{u}: no encoding lemma"))),
        };
        out.push((format!("{u}.smt2"), text));
    }
    Ok(out)
}

/// The formats with encodings that need lemmas (the carriers and flushing f32), by [`Format::name`].
pub fn by_name(n: &str) -> Option<Format> {
    [Format::BF16, Format::E4M3FN, Format::E5M2, Format::F32]
        .into_iter()
        .find(|f| f.name() == n)
}

fn finish(c: &Ctx, comment: &str, goal: T) -> Result<String, SortError> {
    let mut s = crate::Script::new();
    s.comment(comment);
    c.render(&mut s, &[goal]);
    Ok(s.render())
}

/// `x · 2^k` as a literal of an FP sort (a normal value).
fn pow2_in(c: &mut Ctx, s: Sort, k: i64) -> R {
    let Sort::Fp { e, s: sb } = s else {
        return Err(SortError("a power of two of a non-FP sort".into()));
    };
    let bias = (1i64 << (e - 1)) - 1;
    lit(c, s, ((k + bias) as u128) << (sb - 1))
}

/// The sort of format `f` with two more exponent bits: no add, subtract, multiply, divide or square root
/// of its values underflows or overflows there, so rounding in it is rounding with an unbounded exponent.
fn wide_sort(f: Format) -> Sort {
    Sort::Fp {
        e: f.e + 2,
        s: f.m + 1,
    }
}

/// Design §6.4's result for a value `y` of [`wide_sort`] (already rounded to `f`'s precision): in a
/// flushing format zero of its sign below `2^emin`; then narrowed to `f`'s sort (exact, or overflow to
/// infinity).
fn flush_narrow(c: &mut Ctx, f: Format, y: T) -> R {
    let ws = c.sort(y);
    let y = if f.flushes() {
        let min = pow2_in(c, ws, f.emin())?;
        let ay = c.op1(Op::FpAbs, y)?;
        let tiny = c.op2(Op::FpLt, ay, min)?;
        let z = signed_zero(c, y)?;
        c.ite(tiny, z, y)?
    } else {
        y
    };
    let (e, s) = (f.e, f.m + 1);
    c.app(Op::FpToFp { e, s, rm: Rm::Rne }, &[y])
}

/// Design §6.4's rounding of an FP term `x` to the IEEE format `f`, in `f`'s sort: round to nearest even
/// with an unbounded exponent, flush, narrow; a saturating conversion maps infinities to the largest
/// finite value of their sign (NaN stays NaN).
fn reference(c: &mut Ctx, f: Format, x: T, saturate: bool) -> R {
    let r = if f.flushes() {
        let Sort::Fp { e, s } = wide_sort(f) else {
            return Err(SortError("wide sort".into()));
        };
        let y = c.app(Op::FpToFp { e, s, rm: Rm::Rne }, &[x])?;
        flush_narrow(c, f, y)?
    } else {
        let (e, s) = (f.e, f.m + 1);
        c.app(Op::FpToFp { e, s, rm: Rm::Rne }, &[x])?
    };
    if !saturate {
        return Ok(r);
    }
    let max = lit(c, f.fp_sort(), f.max_finite())?;
    let nmax = c.op1(Op::FpNeg, max)?;
    let neg = c.op1(Op::FpIsNegative, r)?;
    let m = c.ite(neg, nmax, max)?;
    let inf = c.op1(Op::FpIsInfinite, r)?;
    c.ite(inf, m, r)
}

/// A value of `f` from its encoding (FP term or bit pattern) in `f`'s sort.
fn as_fp(c: &mut Ctx, f: Format, t: T) -> R {
    if f.native() {
        Ok(t)
    } else {
        c.app(Op::FpFromBits { e: f.e, s: f.m + 1 }, &[t])
    }
}

/// The policy a lemma states, for its comment.
fn policy(f: Format) -> &'static str {
    if f.flushes() {
        "with DAZ, two more exponent bits and FTZ below 2^emin"
    } else {
        "with gradual underflow"
    }
}

/// `to_f32` is the exact widening of the native sort.
fn ieee_decode(f: Format) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let b = c.var("b", Sort::Bv(f.bits()))?;
    let v = to_f32(&mut c, f, b)?;
    let (e, s) = (f.e, f.m + 1);
    let r = c.app(Op::FpFromBits { e, s }, &[b])?;
    let w = c.app(
        Op::FpToFp {
            e: 8,
            s: 24,
            rm: Rm::Rne,
        },
        &[r],
    )?;
    let eq = c.eq(v, w)?;
    let goal = c.not(eq)?;
    finish(
        &c,
        &format!(
            "{}: the Float32 decoding is the exact widening (expect unsat)",
            f.name()
        ),
        goal,
    )
}

/// `from_f32` (or the saturating conversion) is the specified rounding.
fn ieee_round(f: Format, saturate: bool) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let x = c.var("x", F32)?;
    let b = if saturate {
        convert_from_f32(&mut c, f, x)?
    } else {
        from_f32(&mut c, f, x)?
    };
    let got = as_fp(&mut c, f, b)?;
    let want = reference(&mut c, f, x, saturate)?;
    let eq = c.eq(got, want)?;
    let goal = c.not(eq)?;
    finish(
        &c,
        &format!(
            "{}: {} from Float32 equals round to nearest even {}{} (expect unsat)",
            f.name(),
            if saturate { "converting" } else { "rounding" },
            policy(f),
            if saturate {
                ", saturating at the largest finite value"
            } else {
                ""
            }
        ),
        goal,
    )
}

/// An arithmetic operation's encoding ([`arith`]) equals the operation stated in the format's own sort.
fn ieee_op(f: Format, op: Arith) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let mut xs = Vec::new();
    for v in ["a", "b"].into_iter().take(op.arity()) {
        xs.push(c.var(v, f.sort())?);
    }
    let got = arith(&mut c, f, op, &xs)?;
    let got = as_fp(&mut c, f, got)?;
    // Z3 has no addition in sorts with more exponent than significand bits (bf16 with two more is
    // (10, 8)); an underflowing sum is exact (Hauser, ACM TOPLAS 18(2), 1996), so for a sum the native
    // sort's operation flushed when subnormal states the same.
    let direct = matches!(op, Arith::Add | Arith::Sub) && f.e + 2 > f.m + 1;
    let want = if f.flushes() && direct {
        let mut ns = Vec::new();
        for &x in &xs {
            let v = as_fp(&mut c, f, x)?;
            ns.push(daz_fp(&mut c, v)?);
        }
        let r = op.apply(&mut c, &ns)?;
        daz_fp(&mut c, r)?
    } else if f.flushes() {
        let Sort::Fp { e, s } = wide_sort(f) else {
            return Err(SortError("wide sort".into()));
        };
        let mut ws = Vec::new();
        for &x in &xs {
            let v = as_fp(&mut c, f, x)?;
            let v = daz_fp(&mut c, v)?;
            ws.push(c.app(Op::FpToFp { e, s, rm: Rm::Rne }, &[v])?);
        }
        let r = op.apply(&mut c, &ws)?;
        flush_narrow(&mut c, f, r)?
    } else {
        let mut ns = Vec::new();
        for &x in &xs {
            ns.push(as_fp(&mut c, f, x)?);
        }
        op.apply(&mut c, &ns)?
    };
    let eq = c.eq(got, want)?;
    let goal = c.not(eq)?;
    finish(
        &c,
        &format!(
            "{}: the encoding of {} equals the operation in its own sort {} (expect unsat)",
            f.name(),
            op.name(),
            policy(f)
        ),
        goal,
    )
}

/// The square root of a flushing native format on a DAZ operand is never subnormal (the root of a normal
/// value is normal), so the theory's operation is the specified one: nothing is flushed.
fn native_sqrt(f: Format) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let x = c.var("a", f.sort())?;
    let r = arith(&mut c, f, Arith::Sqrt, &[x])?;
    let goal = c.op1(Op::FpIsSubnormal, r)?;
    finish(
        &c,
        &format!(
            "{}: the square root of a DAZ operand is subnormal (expect unsat)",
            f.name()
        ),
        goal,
    )
}

/// A sum or difference of two DAZ operands of a flushing native format that IEEE makes subnormal is
/// exact (rounding up and down agree; Hauser, ACM TOPLAS 18(2), 1996). So the encoding (the theory's
/// operation, flushed when subnormal) is the specified one: a subnormal result is the exact value, below
/// `2^emin` and with at most the format's precision, so flushed; any other result is the rounding with an
/// unbounded exponent. Z3 decides this in about a second, against about 45 s for the whole operation in
/// the sort with two more exponent bits.
fn native_sum(f: Format, op: Arith) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let a = c.var("a", f.sort())?;
    let b = c.var("b", f.sort())?;
    let xs = [daz(&mut c, f, a)?, daz(&mut c, f, b)?];
    let r = op.apply(&mut c, &xs)?;
    let sub = c.op1(Op::FpIsSubnormal, r)?;
    let (up, down) = match op {
        Arith::Sub => (
            c.op2(Op::FpSub(Rm::Rtp), xs[0], xs[1])?,
            c.op2(Op::FpSub(Rm::Rtn), xs[0], xs[1])?,
        ),
        _ => (
            c.op2(Op::FpAdd(Rm::Rtp), xs[0], xs[1])?,
            c.op2(Op::FpAdd(Rm::Rtn), xs[0], xs[1])?,
        ),
    };
    let same = c.eq(up, down)?;
    let inexact = c.not(same)?;
    let goal = c.and(&[sub, inexact])?;
    finish(
        &c,
        &format!(
            "{}: a {} of DAZ operands that is subnormal is inexact (expect unsat)",
            f.name(),
            op.name()
        ),
        goal,
    )
}

/// `2^k` as a Float64, `k` a 12-bit signed term in the normal range.
fn pow2_f64_term(c: &mut Ctx, k: T) -> R {
    let b = c.bv(1023, 12);
    let field = c.op2(Op::BvAdd, k, b)?;
    let field = c.extract(field, 10, 0)?;
    let z1 = c.bv(0, 1);
    let z52 = c.bv(0, 52);
    let hi = c.op2(Op::Concat, z1, field)?;
    let bits = c.op2(Op::Concat, hi, z52)?;
    c.app(Op::FpFromBits { e: 11, s: 53 }, &[bits])
}

/// [`scaled_op`]'s decision for every exact value. For DAZ operands that are normal, the exact product
/// or quotient `v` lies in `[2^(e_x + e_y), 2^(e_x + e_y + 2))` or `(2^(e_x - e_y - 1), 2^(e_x - e_y + 1))`
/// in magnitude, with the sign of the operands' product. The Float32 operation is by definition the
/// rounding of `v` (scaled by `2^32` when the operand was), so the encoding is a function of `v`; the
/// lemma compares it, for every `v` in that range, with `v` rounded in `(_ FloatingPoint 10 24)`, flushed
/// and narrowed. `v` ranges over Float64: both sides are constant between dyadic breakpoints of at most 26
/// significant bits, all Float64 values, so agreement there is agreement on every real. Zero, infinite
/// and NaN operands give zero, infinity or NaN, which are never flushed (smoke cases). No multiplier
/// is in the query: stated against the operation itself the solver relates two multipliers (or dividers),
/// which Z3 does not decide in 10 minutes.
fn scaled_lemma(f: Format, op: Arith) -> Result<String, SortError> {
    scaled_lemma_with(f, op, 0)
}

/// [`scaled_lemma`] with the flush limit moved by `off` binades: a negative control (`off != 0` must be
/// `sat`, so the lemma's hypotheses reach the boundary).
pub fn scaled_control(op: Arith, off: i64) -> Result<String, SortError> {
    scaled_lemma_with(Format::F32, op, off)
}

fn scaled_lemma_with(f: Format, op: Arith, off: i64) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let a = c.var("a", f.sort())?;
    let b = c.var("b", f.sort())?;
    let x = daz(&mut c, f, a)?;
    let y = daz(&mut c, f, b)?;
    let (tp, _) = scaled_pre(&mut c, f, op, x, y)?;
    let v = c.var("v", F64)?;
    // The range of the exact value.
    let bx = c.bits_of(x)?;
    let by = c.bits_of(y)?;
    let field = |c: &mut Ctx, bits: T| -> R {
        let e = c.extract(bits, 30, 23)?;
        let e = c.resize(e, 12, false)?;
        let bias = c.bv(127, 12);
        c.op2(Op::BvSub, e, bias)
    };
    let (ex, ey) = (field(&mut c, bx)?, field(&mut c, by)?);
    let (k_lo, k_hi) = match op {
        Arith::Mul => {
            let s = c.op2(Op::BvAdd, ex, ey)?;
            let two = c.bv(2, 12);
            (s, c.op2(Op::BvAdd, s, two)?)
        }
        _ => {
            let d = c.op2(Op::BvSub, ex, ey)?;
            let one = c.bv(1, 12);
            (c.op2(Op::BvSub, d, one)?, c.op2(Op::BvAdd, d, one)?)
        }
    };
    let lo = pow2_f64_term(&mut c, k_lo)?;
    let hi = pow2_f64_term(&mut c, k_hi)?;
    let av = c.op1(Op::FpAbs, v)?;
    let above = if op == Arith::Mul {
        c.op2(Op::FpLeq, lo, av)?
    } else {
        c.op2(Op::FpLt, lo, av)?
    };
    let below = c.op2(Op::FpLt, av, hi)?;
    let sx = c.op1(Op::FpIsNegative, x)?;
    let sy = c.op1(Op::FpIsNegative, y)?;
    let sv = c.op1(Op::FpIsNegative, v)?;
    let sxy = c.app(Op::Xor, &[sx, sy])?;
    let sign = c.eq(sv, sxy)?;
    let mut normal = Vec::new();
    for t in [x, y] {
        let z = c.op1(Op::FpIsZero, t)?;
        let n = c.op1(Op::FpIsNaN, t)?;
        let i = c.op1(Op::FpIsInfinite, t)?;
        let sp = c.or(&[z, n, i])?;
        normal.push(c.not(sp)?);
    }
    // The encoding as a function of v.
    let vs = shift_exp(&mut c, v, SCALE)?;
    let va = c.ite(tp, vs, v)?;
    let r = c.app(
        Op::FpToFp {
            e: 8,
            s: 24,
            rm: Rm::Rne,
        },
        &[va],
    )?;
    let got = scaled_post(&mut c, tp, r, off)?;
    let want = reference(&mut c, f, v, false)?;
    let eq = c.eq(got, want)?;
    let differ = c.not(eq)?;
    let goal = c.and(&[normal[0], normal[1], above, below, sign, differ])?;
    finish(
        &c,
        &format!(
            "{}: the scaled {} differs from the exact value rounded with two more exponent bits, flushed and narrowed (expect unsat)",
            f.name(),
            op.name()
        ),
        goal,
    )
}

/// Rounding a Float64 (through Float32 rounded to odd for a carrier) equals the specified rounding.
fn ieee_from_f64(f: Format) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let x = c.var("x", F64)?;
    let r = if f.native() {
        native_from_f64(&mut c, f, x)?
    } else {
        from_f64(&mut c, f, x)?
    };
    let got = as_fp(&mut c, f, r)?;
    let want = reference(&mut c, f, x, f.saturates())?;
    let eq = c.eq(got, want)?;
    let goal = c.not(eq)?;
    finish(
        &c,
        &format!(
            "{}: rounding from Float64 equals round to nearest even {} (expect unsat)",
            f.name(),
            policy(f)
        ),
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
    let w1 = c.app(
        Op::FpToFp {
            e: 8,
            s: 24,
            rm: Rm::Rne,
        },
        &[ieee],
    )?;
    let wider = fn_as_wider(&mut c, f, b)?;
    let w2 = c.app(
        Op::FpToFp {
            e: 8,
            s: 24,
            rm: Rm::Rne,
        },
        &[wider],
    )?;
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

/// `from_f32` of a finite-NaN format, piecewise against the two IEEE sorts whose grids it shares; past
/// the largest finite value, and for infinities, the NaN pattern (arithmetic) or the largest finite
/// pattern of the sign (`saturate`, a conversion).
fn fn_round(f: Format, saturate: bool) -> Result<String, SortError> {
    let mut c = Ctx::new();
    let x = c.var("x", F32)?;
    let b = if saturate {
        convert_from_f32(&mut c, f, x)?
    } else {
        from_f32(&mut c, f, x)?
    };
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
    // The overflow pattern: NaN, or the largest finite magnitude with x's sign.
    let over = if saturate {
        let w = f.bits();
        let sx = c.op1(Op::FpIsNegative, x)?;
        let one = c.bv(1, 1);
        let zero = c.bv(0, 1);
        let sb = c.ite(sx, one, zero)?;
        let mag = c.bv(f.max_finite(), w - 1);
        let maxp = c.op2(Op::Concat, sb, mag)?;
        c.eq(b, maxp)?
    } else {
        isnan
    };
    let wider = fn_as_wider(&mut c, f, b)?;
    let eqy = c.eq(wider, y)?;
    let br = c.ite(past, over, eqy)?;
    let c2 = c.app(Op::Implies, &[big, br])?;
    // NaN gives the NaN pattern; infinities overflow.
    let xnan = c.op1(Op::FpIsNaN, x)?;
    let c3 = c.app(Op::Implies, &[xnan, isnan])?;
    let c4 = c.app(Op::Implies, &[isinf, over])?;
    let all = c.and(&[c1, c2, c3, c4])?;
    let goal = c.not(all)?;
    finish(
        &c,
        &format!(
            "{}: {} from Float32 agrees with (_ FloatingPoint {e} {s}) up to its largest finite value, with (_ FloatingPoint {} {s}) from the smallest normal, and overflows to {} (expect unsat)",
            f.name(),
            if saturate { "converting" } else { "rounding" },
            e + 1,
            if saturate {
                "the largest finite pattern of its sign"
            } else {
                "the NaN pattern"
            }
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
            "f32.add",
            "f32.sub",
            "f32.mul",
            "f32.div",
            "f32.sqrt",
            "f32.from_f64",
            "f8e4m3fn.decode",
            "f8e4m3fn.round",
            "f8e4m3fn.saturate",
            "f8e5m2.saturate",
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
