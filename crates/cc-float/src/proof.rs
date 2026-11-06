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
//! `unsat` proves the wrapper. Multiply, divide and square root of the flushing formats leave both the
//! core's and the specification's arithmetic out: each is the rounding of one value `v`, the exact
//! result, constrained to the binade range the operands give (as the encoding lemmas of `cc_smt` do);
//! a solver relating a multiplier of one sort to a multiplier of another does not finish.

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
            if fmt.flushes() && matches!(op, Arith::Mul | Arith::Div | Arith::Sqrt) =>
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
    let y = c.app(Op::FpFromBits { e: core.e, s: core.m + 1 }, &[z])?;
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
    for &x in &ins {
        args.push(fval(&mut c, f, x)?);
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
        match &b.needs[k].role {
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
                                    let v = batch.verdicts.get(&file).cloned().unwrap_or(
                                        Verdict::Error {
                                            message: "no answer".into(),
                                        },
                                    );
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
        assert!(!uses_other_sorts("(declare-const x (_ FloatingPoint 8 24)) ((_ to_fp 11 53) RNE x)"));
        assert!(uses_other_sorts("((_ to_fp 9 24) RNE x)"));
        assert!(uses_other_sorts("(declare-const x (_ FloatingPoint 8 8))"));
        assert!(uses_other_sorts("((_ to_fp_unsigned 9 8) RNE b)"));
    }
}
