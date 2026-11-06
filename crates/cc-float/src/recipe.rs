//! The Verilog of the compositions (design §4.12; docs/framework/float.md, "Compositions"): cores at
//! an inner format `Q` with two more exponent bits than the format (no intermediate overflows or
//! underflows where the result is finite) and `q` significand bits, one rounding back to the format
//! (a conversion core, or the FP8 encoder), and the special values as glue: C11 Annex F (ISO/IEC
//! 9899:2011, F.10) for power, tanh, log1p; IEEE 754-2019 §9.2 `rSqrt` for rsqrt.

use cc_core::adapters::CoreInstance;
use cc_smt::float::{Flavour, Format};

use crate::FloatError;
use crate::compose::Recipe;
use crate::glue::{CoreNeed, Role, wide};
use crate::rtl::{Mutation, Rtl};

type Res<X> = Result<X, FloatError>;

/// Instantiates the next core on operands, with a result width.
type CoreFn<'a> = dyn FnMut(&mut Rtl, &[String], u32) -> Res<String> + 'a;

/// The inner format of a composition at `f` with `q` significand bits (the hidden bit included).
pub fn inner(f: Format, q: u32) -> Format {
    Format::ieee(f.e + 2, q - 1)
}

/// The catalog op of a recipe.
pub fn op_name(r: Recipe) -> &'static str {
    match r {
        Recipe::Power => "power",
        Recipe::Tanh => "tanh",
        Recipe::LogPlusOne => "log_plus_one",
        Recipe::Rsqrt => "rsqrt",
        Recipe::Wider => "wider",
    }
}

fn arith(op: &str, q: Format, arity: usize) -> CoreNeed {
    CoreNeed {
        role: Role::Arith { op: op.into() },
        exp_width: q.e,
        sig_width: q.m,
        arity,
    }
}

fn approx(op: &str, q: Format) -> CoreNeed {
    CoreNeed {
        role: Role::Approx { op: op.into() },
        exp_width: q.e,
        sig_width: q.m,
        arity: 1,
    }
}

/// The cores of a composition, in order (the last one rounds to the format, except for E4M3FN, which
/// the encoder rounds).
pub fn needs(r: Recipe, f: Format, q: u32) -> Res<Vec<CoreNeed>> {
    if q <= f.m + 1 && r != Recipe::Wider {
        return Err(FloatError::Compose(format!(
            "{}: {q} inner bits do not exceed {}'s {}",
            op_name(r),
            f.name(),
            f.m + 1
        )));
    }
    let qf = inner(f, q);
    let mut v = match r {
        Recipe::Power => vec![approx("log2", qf), arith("mul", qf, 2), approx("exp2", qf)],
        Recipe::Tanh => vec![
            arith("mul", qf, 2),
            arith("mul", qf, 2),
            arith("div", qf, 2),
            arith("sub", qf, 2),
            approx("exp", qf),
            arith("sub", qf, 2),
            arith("add", qf, 2),
            arith("div", qf, 2),
        ],
        Recipe::LogPlusOne => vec![
            arith("add", qf, 2),
            approx("log", qf),
            arith("sub", qf, 2),
            arith("div", qf, 2),
            arith("mul", qf, 2),
        ],
        Recipe::Rsqrt => vec![arith("div", qf, 2), arith("sqrt", qf, 1)],
        Recipe::Wider => {
            return Err(FloatError::Compose(
                "a wider core is an approximate wrapper (Glue::Approx)".into(),
            ));
        }
    };
    if f.flavour == Flavour::Ieee {
        let to = if f.flushes() { wide(f) } else { f };
        v.push(CoreNeed {
            role: Role::Convert { from: qf.name() },
            exp_width: to.e,
            sig_width: to.m,
            arity: 1,
        });
    }
    Ok(v)
}

/// The encoding of a positive finite `v` in `f`, rounded up or down (thresholds the glue compares a
/// magnitude with).
pub fn encode(f: Format, v: f64, up: bool) -> u128 {
    let emin = f.emin() as i32;
    let e = (v.log2().floor() as i32).max(emin);
    let scaled = v / 2f64.powi(e - f.m as i32);
    let n = if up { scaled.ceil() } else { scaled.floor() } as u128;
    if e == emin && n < (1u128 << f.m) {
        return n;
    }
    // n in [2^m, 2^(m+1)]: a carry to 2^(m+1) moves to the next binade.
    let (n, e) = if n >= 1u128 << (f.m + 1) {
        (n >> 1, e + 1)
    } else {
        (n, e)
    };
    (((e as i64 + f.bias()) as u128) << f.m) | (n & ((1u128 << f.m) - 1))
}

fn lit(v: u128, w: u32) -> String {
    format!("{w}'d{v}")
}

/// `1 when the magnitude of x is below / above the constant`.
fn mag_cmp(r: &mut Rtl, f: Format, x: &str, op: &str, c: u128) -> String {
    let w = f.bits();
    r.wire(1, format!("{x}[{}:0] {op} {}", w - 2, lit(c, w - 1)))
}

/// Integer and odd-integer tests of a value of `f` (zero is an even integer; infinities and NaN are
/// neither).
fn int_odd(r: &mut Rtl, f: Format, y: &str) -> (String, String) {
    let m = f.m;
    let e = format!("{y}[{}:{m}]", f.e + m - 1);
    let mm = format!("{y}[{}:0]", m - 1);
    let bias = f.bias() as u128;
    let sig = r.wire(m + 1, format!("{{1'b1, {mm}}}"));
    let sw = f.e + 2;
    // The fraction bits below the units place: (bias + m) - E of them.
    let sh = r.wire(sw, format!("{} - {{2'd0, {e}}}", lit(bias + u128::from(m), sw)));
    let mask = r.wire(m + 1, format!("({} << {sh}) - {}", lit(1, m + 1), lit(1, m + 1)));
    let frac0 = r.wire(1, format!("~(|({sig} & {mask}))"));
    let unit = r.wire(1, format!("({sig} >> {sh}) & {}", lit(1, m + 1)));
    let ge = r.wire(1, format!("{e} >= {}", lit(bias, f.e)));
    let big = r.wire(1, format!("{e} > {}", lit(bias + u128::from(m), f.e)));
    let zero = r.is_zero(f, y);
    let special = match f.flavour {
        Flavour::Ieee => format!("(&{e})"),
        Flavour::FiniteNan => r.is_nan(f, y),
    };
    let int = r.wire(1, format!("~{special} & ({zero} | {big} | ({ge} & {frac0}))"));
    let odd = r.wire(1, format!("~{special} & ~{big} & {ge} & {frac0} & {unit}[0]"));
    (int, odd)
}

/// The composition's output from its operands `xs` (format `f`) and its cores.
pub fn build(
    r: &mut Rtl,
    rec: Recipe,
    f: Format,
    q: u32,
    xs: &[String],
    cores: &[CoreInstance],
    mutation: Mutation,
) -> Res<String> {
    let qf = inner(f, q);
    let qb = qf.bits();
    let one_q = lit(encode(qf, 1.0, false), qb);
    let w = f.bits();
    let mut k = 0usize;
    let mut core = |r: &mut Rtl, ops: &[String], out_w: u32| -> Res<String> {
        let c = cores
            .get(k)
            .ok_or_else(|| FloatError::Core(format!("{}: too few cores", op_name(rec))))?;
        k += 1;
        Ok(r.core(c, ops, qb, out_w)?.1)
    };
    let xd: Vec<String> = xs.iter().map(|x| r.daz(f, x)).collect();
    let sx = r.sign(f, &xd[0]);
    let nanx = r.is_nan(f, &xd[0]);
    let zx = r.is_zero(f, &xd[0]);
    let infx = r.is_inf(f, &xd[0]);
    let nan_f = Rtl::nan_literal(f);
    let inf_of = |s: &str| match f.flavour {
        Flavour::Ieee => format!("{{{s}, {}}}", lit(((1u128 << f.e) - 1) << f.m, w - 1)),
        Flavour::FiniteNan => nan_f.clone(),
    };
    let zero_of = |s: &str| format!("{{{s}, {}}}", lit(0, w - 1));
    let one_f = lit(encode(f, 1.0, false), w - 1);
    // The core path's value in Q, then rounded to f (a conversion core and the flush, or the encoder).
    let round = |r: &mut Rtl, core: &mut CoreFn, v: &str| -> Res<String> {
        if f.flavour == Flavour::FiniteNan {
            return r.encode_fp8(qf, f, v, false, Mutation::None);
        }
        let to = if f.flushes() { wide(f) } else { f };
        let z = core(r, &[v.to_string()], to.bits())?;
        if f.flushes() {
            r.flush_narrow(to, f, &z, true)
        } else {
            Ok(z)
        }
    };
    let out = match rec {
        Recipe::Power => {
            let yd = &xd[1];
            let sy = r.sign(f, yd);
            let nany = r.is_nan(f, yd);
            let zy = r.is_zero(f, yd);
            let infy = r.is_inf(f, yd);
            let ax = r.wire(w, format!("{{1'b0, {}[{}:0]}}", xd[0], w - 2));
            let one_x = r.wire(1, format!("{ax}[{}:0] == {one_f}", w - 2));
            let xone = r.wire(1, format!("~{sx} & {one_x}"));
            let lt1 = mag_cmp(r, f, &ax, "<", encode(f, 1.0, false));
            let (yint, yodd) = int_odd(r, f, yd);
            let xq = r.widen(f, qf, &ax)?;
            let yq = r.widen(f, qf, yd)?;
            let lg = core(r, &[xq], qb)?;
            let t = core(r, &[yq, lg], qb)?;
            let ex = core(r, &[t], qb)?;
            let mag = round(r, &mut core, &ex)?;
            let neg = r.wire(1, format!("{sx} & {yodd}"));
            let path = r.wire(w, format!("{{{neg}, {mag}[{}:0]}}", w - 2));
            let one = format!("{{1'b0, {one_f}}}");
            // C11 F.10.4.4, first match wins.
            let pinf = inf_of("1'b0");
            let pz = zero_of("1'b0");
            let zero_x_neg_y = r.wire(w, format!("{yodd} ? {} : {pinf}", inf_of(&sx)));
            let zero_x_pos_y = r.wire(w, format!("{yodd} ? {} : {pz}", zero_of(&sx)));
            let zero_x = r.wire(w, format!("{sy} ? {zero_x_neg_y} : {zero_x_pos_y}"));
            let ninf_neg_y = r.wire(w, format!("{yodd} ? {} : {pz}", zero_of("1'b1")));
            let ninf_pos_y = r.wire(w, format!("{yodd} ? {} : {pinf}", inf_of("1'b1")));
            let inf_xv = r.wire(
                w,
                format!("{sx} ? ({sy} ? {ninf_neg_y} : {ninf_pos_y}) : ({sy} ? {pz} : {pinf})"),
            );
            let inf_y = r.wire(w, format!("{one_x} ? {one} : (({lt1} ^ {sy}) ? {pz} : {pinf})"));
            let neg_nonint = if mutation == Mutation::NoNanCase {
                r.wire(1, "1'b0")
            } else {
                r.wire(1, format!("{sx} & ~{yint}"))
            };
            r.wire(
                w,
                format!(
                    "{zy} ? {one} : ({xone} ? {one} : (({nanx} | {nany}) ? {nan_f} : ({infy} ? {inf_y} : \
                     ({zx} ? {zero_x} : ({infx} ? {inf_xv} : ({neg_nonint} ? {nan_f} : {path}))))))"
                ),
            )
        }
        Recipe::Tanh => {
            let a = r.wire(w, format!("{{1'b0, {}[{}:0]}}", xd[0], w - 2));
            let aq = r.widen(f, qf, &a)?;
            let s = core(r, &[aq.clone(), aq.clone()], qb)?;
            let c3 = core(r, &[s, aq.clone()], qb)?;
            let three = lit(encode(qf, 3.0, false), qb);
            let d = core(r, &[c3, three], qb)?;
            let cubic = core(r, &[aq.clone(), d], qb)?;
            // 2a: the exponent field plus one (exact; a is below the clamp where it is used).
            let ef = format!("{aq}[{}:{}]", qf.e + qf.m - 1, qf.m);
            let a2 = r.wire(
                qb,
                format!("{{1'b0, {ef} + {}, {aq}[{}:0]}}", lit(1, qf.e), qf.m - 1),
            );
            let big = core(r, &[a2], qb)?;
            let num = core(r, &[big.clone(), one_q.clone()], qb)?;
            let den = core(r, &[big, one_q.clone()], qb)?;
            let mid = core(r, &[num, den], qb)?;
            let x1 = encode(f, crate::compose::tanh_cubic_limit(f), true);
            let clamp = encode(f, crate::compose::tanh_clamp(f), false);
            let small = mag_cmp(r, f, &a, "<", x1);
            let sat = mag_cmp(r, f, &a, ">", clamp);
            let v = r.wire(qb, format!("{small} ? {cubic} : ({sat} ? {one_q} : {mid})"));
            let mag = round(r, &mut core, &v)?;
            let signed = r.wire(w, format!("{{{sx}, {mag}[{}:0]}}", w - 2));
            if mutation == Mutation::NoNanCase {
                signed
            } else {
                r.wire(w, format!("{nanx} ? {nan_f} : {signed}"))
            }
        }
        Recipe::LogPlusOne => {
            let xq = r.widen(f, qf, &xd[0])?;
            let wv = core(r, &[one_q.clone(), xq.clone()], qb)?;
            let lw = core(r, std::slice::from_ref(&wv), qb)?;
            let dm = core(r, &[wv.clone(), one_q.clone()], qb)?;
            let ratio = core(r, &[lw, dm], qb)?;
            let prod = core(r, &[xq.clone(), ratio], qb)?;
            let wone = r.wire(1, format!("{wv} == {one_q}"));
            let v = r.wire(qb, format!("{wone} ? {xq} : {prod}"));
            let res = round(r, &mut core, &v)?;
            let m1 = r.wire(1, format!("{} == {{1'b1, {one_f}}}", xd[0]));
            let below = r.wire(1, format!("{sx} & ({}[{}:0] > {one_f})", xd[0], w - 2));
            let below = if mutation == Mutation::NoNanCase {
                r.wire(1, "1'b0")
            } else {
                below
            };
            let pinf = inf_of("1'b0");
            r.wire(
                w,
                format!(
                    "({nanx} | {below}) ? {nan_f} : ({m1} ? {} : ({infx} ? {pinf} : ({zx} ? {} : {res})))",
                    inf_of("1'b1"),
                    xd[0]
                ),
            )
        }
        Recipe::Rsqrt => {
            let xq = r.widen(f, qf, &xd[0])?;
            let rec = core(r, &[one_q.clone(), xq], qb)?;
            let root = core(r, &[rec], qb)?;
            let res = round(r, &mut core, &root)?;
            let neg = if mutation == Mutation::NoNanCase {
                r.wire(1, "1'b0")
            } else {
                r.wire(1, format!("{sx} & ~{zx}"))
            };
            r.wire(
                w,
                format!(
                    "({nanx} | {neg}) ? {nan_f} : ({zx} ? {} : ({infx} ? {} : {res}))",
                    inf_of(&sx),
                    zero_of("1'b0")
                ),
            )
        }
        Recipe::Wider => {
            return Err(FloatError::Compose("no wider recipe here".into()));
        }
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_thresholds() {
        assert_eq!(encode(Format::F32, 1.0, false), 0x3f80_0000);
        assert_eq!(encode(Format::F32, 3.0, true), 0x4040_0000);
        assert_eq!(encode(Format::F32, 0.1, false), 0x3dcc_cccc);
        assert_eq!(encode(Format::F32, 0.1, true), 0x3dcc_cccd);
        assert_eq!(encode(Format::E4M3FN, 1.0, false), 0x38);
        assert_eq!(encode(Format::E4M3FN, 448.0, false), 0x7e);
        // Rounding up across a binade.
        assert_eq!(
            encode(Format::ieee(4, 2), 1.99, true),
            encode(Format::ieee(4, 2), 2.0, false)
        );
    }
}
