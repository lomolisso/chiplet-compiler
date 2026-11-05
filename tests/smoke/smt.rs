//! `cc-smt`'s scalar op semantics against concrete cases (the StableHLO spec's examples and edge cases,
//! XLA's choices where the spec leaves a result open, the OCP FP8 boundary cases), and two negative
//! controls (a swapped operand, a truncation instead of a rounding) that must have counterexamples. Both
//! solvers answer every query.

use cc_core::CancelToken;
use cc_core::runner::{BoundRunner, Caps, Runners, ToolLog};
use cc_smt::float::{self, Format};
use cc_smt::ops::{self, Attrs, Ty, Val};
use cc_smt::term::{Ctx, Op, Rm, Sort, T};
use cc_smt::{Location, Script, Solver, Verdict};
use tests::{Env, repo_root};

const F32: Sort = Sort::Fp { e: 8, s: 24 };
const I8: Ty = Ty::Int { signed: true, w: 8 };
const U8: Ty = Ty::Int { signed: false, w: 8 };
const I32: Ty = Ty::Int { signed: true, w: 32 };
const TF32: Ty = Ty::Float(Format::F32);
const TBF16: Ty = Ty::Float(Format::BF16);
const TF8: Ty = Ty::Float(Format::E4M3FN);
const TF8E5: Ty = Ty::Float(Format::E5M2);

fn int(c: &mut Ctx, ty: Ty, v: i128) -> Val {
    let Ty::Int { w, .. } = ty else { unreachable!() };
    Val::new(ty, c.bvs(v, w))
}

fn f32v(c: &mut Ctx, x: f32) -> Val {
    Val::new(TF32, float::lit(c, F32, u128::from(x.to_bits())).unwrap())
}

fn bits(c: &mut Ctx, ty: Ty, b: u128) -> Val {
    let Ty::Float(f) = ty else { unreachable!() };
    Val::new(ty, c.bv(b, f.bits()))
}

fn cmp(d: &str, t: Option<&str>) -> Attrs {
    Attrs {
        direction: Some(d.into()),
        compare_type: t.map(str::to_string),
        precision: None,
    }
}

/// A query asserting that `got` differs from `want` (expect unsat).
fn differs(name: &str, f: impl FnOnce(&mut Ctx) -> (Val, Val)) -> (String, String) {
    let mut c = Ctx::new();
    let (got, want) = f(&mut c);
    let eq = ops::same(&mut c, got, want).unwrap();
    let goal = c.not(eq).unwrap();
    let mut s = Script::new();
    s.comment(name);
    c.render(&mut s, &[goal]);
    (format!("{name}.smt2"), s.render())
}

fn bool_query(name: &str, f: impl FnOnce(&mut Ctx) -> T) -> (String, String) {
    let mut c = Ctx::new();
    let goal = f(&mut c);
    let mut s = Script::new();
    s.comment(name);
    c.render(&mut s, &[goal]);
    (format!("{name}.smt2"), s.render())
}

fn ap(c: &mut Ctx, op: &str, a: &Attrs, args: &[Val], ret: Ty) -> Val {
    ops::apply(c, op, a, args, ret).unwrap()
}

/// E4M3FN decoded independently: (-1)^s · (E = 0 ? M/8 · 2^-6 : (1 + M/8) · 2^(E-7)).
fn e4m3fn(b: u8) -> Option<f32> {
    let (s, e, m) = (b >> 7, (b >> 3) & 15, b & 7);
    if e == 15 && m == 7 {
        return None;
    }
    let v = if e == 0 {
        f32::from(m) / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + f32::from(m) / 8.0) * 2f32.powi(i32::from(e) - 7)
    };
    Some(if s == 1 { -v } else { v })
}

type Queries = Vec<(String, String)>;

fn cases() -> (Queries, Queries) {
    let none = Attrs::default();
    let mut unsat = vec![
        // Two's complement wrapping.
        differs("i8_add_wraps", |c| {
            let (a, b, w) = (int(c, I8, 127), int(c, I8, 1), int(c, I8, -128));
            (ap(c, "add", &none, &[a, b], I8), w)
        }),
        // Integer division truncates; x / 0 = -1, MIN / -1 = MIN; x % 0 = x, MIN % -1 = 0 (XLA).
        differs("i32_divide_truncates", |c| {
            let (a, b, w) = (int(c, I32, -7), int(c, I32, 2), int(c, I32, -3));
            (ap(c, "divide", &none, &[a, b], I32), w)
        }),
        differs("i32_divide_by_zero", |c| {
            let (a, b, w) = (int(c, I32, 5), int(c, I32, 0), int(c, I32, -1));
            (ap(c, "divide", &none, &[a, b], I32), w)
        }),
        differs("i32_divide_min_by_minus_one", |c| {
            let (a, b, w) = (
                int(c, I32, i128::from(i32::MIN)),
                int(c, I32, -1),
                int(c, I32, i128::from(i32::MIN)),
            );
            (ap(c, "divide", &none, &[a, b], I32), w)
        }),
        differs("i32_remainder_sign_of_dividend", |c| {
            let (a, b, w) = (int(c, I32, -7), int(c, I32, 2), int(c, I32, -1));
            (ap(c, "remainder", &none, &[a, b], I32), w)
        }),
        differs("i32_remainder_by_zero", |c| {
            let (a, b, w) = (int(c, I32, 9), int(c, I32, 0), int(c, I32, 9));
            (ap(c, "remainder", &none, &[a, b], I32), w)
        }),
        differs("i32_remainder_min_by_minus_one", |c| {
            let (a, b, w) = (int(c, I32, i128::from(i32::MIN)), int(c, I32, -1), int(c, I32, 0));
            (ap(c, "remainder", &none, &[a, b], I32), w)
        }),
        differs("u8_divide_by_zero", |c| {
            let (a, b, w) = (int(c, U8, 200), int(c, U8, 0), int(c, U8, 255));
            (ap(c, "divide", &none, &[a, b], U8), w)
        }),
        differs("i8_abs_min_wraps", |c| {
            let (a, w) = (int(c, I8, -128), int(c, I8, -128));
            (ap(c, "abs", &none, &[a], I8), w)
        }),
        differs("i8_sign", |c| {
            let (a, w) = (int(c, I8, -5), int(c, I8, -1));
            (ap(c, "sign", &none, &[a], I8), w)
        }),
        differs("u8_maximum_unsigned", |c| {
            let (a, b, w) = (int(c, U8, 200), int(c, U8, 100), int(c, U8, 200));
            (ap(c, "maximum", &none, &[a, b], U8), w)
        }),
        differs("i8_compare_signed_vs_unsigned", |c| {
            let (a, b) = (int(c, I8, -1), int(c, I8, 1));
            let s = ap(c, "compare", &cmp("LT", Some("SIGNED")), &[a, b], Ty::Bool);
            let u = ap(c, "compare", &cmp("LT", Some("UNSIGNED")), &[a, b], Ty::Bool);
            let nu = c.not(u.t).unwrap();
            let both = c.and(&[s.t, nu]).unwrap();
            (Val::new(Ty::Bool, both), Val::new(Ty::Bool, c.bool(true)))
        }),
        differs("i8_compare_gt_ge", |c| {
            let (a, b) = (int(c, I8, 3), int(c, I8, 3));
            let gt = ap(c, "compare", &cmp("GT", None), &[a, b], Ty::Bool);
            let ge = ap(c, "compare", &cmp("GE", None), &[a, b], Ty::Bool);
            let ngt = c.not(gt.t).unwrap();
            let both = c.and(&[ngt, ge.t]).unwrap();
            (Val::new(Ty::Bool, both), Val::new(Ty::Bool, c.bool(true)))
        }),
        // convert: truncation, sign and zero extension.
        differs("i32_to_i8_truncates", |c| {
            let (a, w) = (int(c, I32, 300), int(c, I8, 44));
            (ops::convert(c, a, I8).unwrap(), w)
        }),
        differs("u8_to_i32_zero_extends", |c| {
            let (a, w) = (int(c, U8, 255), int(c, I32, 255));
            (ops::convert(c, a, I32).unwrap(), w)
        }),
        // float to int: truncation toward zero, saturation, NaN to 0 (XLA).
        differs("f32_to_i32_truncates", |c| {
            let (a, w) = (f32v(c, -3.9), int(c, I32, -3));
            (ops::convert(c, a, I32).unwrap(), w)
        }),
        differs("f32_to_i32_saturates", |c| {
            let (a, w) = (f32v(c, 1e10), int(c, I32, i128::from(i32::MAX)));
            (ops::convert(c, a, I32).unwrap(), w)
        }),
        differs("f32_to_i8_saturates_low", |c| {
            let (a, w) = (f32v(c, -1000.0), int(c, I8, -128));
            (ops::convert(c, a, I8).unwrap(), w)
        }),
        differs("f32_nan_to_i32_is_zero", |c| {
            let (a, w) = (f32v(c, f32::NAN), int(c, I32, 0));
            (ops::convert(c, a, I32).unwrap(), w)
        }),
        // int to float rounds to nearest even.
        differs("i32_to_f32_rounds_even", |c| {
            let (a, w) = (int(c, I32, 16_777_217), f32v(c, 16_777_216.0));
            (ops::convert(c, a, TF32).unwrap(), w)
        }),
        differs("i32_to_bf16_one_rounding", |c| {
            // 2^24 + 2^16 + 1: through Float32 RNE it is a bf16 tie; the exact value rounds up.
            let (a, w) = (int(c, I32, (1 << 24) + (1 << 16) + 1), bits(c, TBF16, 0x4b81));
            (ops::convert(c, a, TBF16).unwrap(), w)
        }),
        // IEEE maximum / minimum: NaN propagates, -0 < +0.
        differs("f32_maximum_nan", |c| {
            let (a, b, w) = (f32v(c, f32::NAN), f32v(c, 1.0), f32v(c, f32::NAN));
            (ap(c, "maximum", &none, &[a, b], TF32), w)
        }),
        differs("f32_maximum_signed_zero", |c| {
            let (a, b, w) = (f32v(c, -0.0), f32v(c, 0.0), f32v(c, 0.0));
            (ap(c, "maximum", &none, &[a, b], TF32), w)
        }),
        differs("f32_minimum_signed_zero", |c| {
            let (a, b, w) = (f32v(c, 0.0), f32v(c, -0.0), f32v(c, -0.0));
            (ap(c, "minimum", &none, &[a, b], TF32), w)
        }),
        differs("f32_compare_nan", |c| {
            let (a, b) = (f32v(c, f32::NAN), f32v(c, 1.0));
            let lt = ap(c, "compare", &cmp("LT", Some("FLOAT")), &[a, b], Ty::Bool);
            let ge = ap(c, "compare", &cmp("GE", Some("FLOAT")), &[a, b], Ty::Bool);
            let ne = ap(c, "compare", &cmp("NE", Some("FLOAT")), &[a, b], Ty::Bool);
            let any = c.or(&[lt.t, ge.t]).unwrap();
            let nany = c.not(any).unwrap();
            let r = c.and(&[nany, ne.t]).unwrap();
            (Val::new(Ty::Bool, r), Val::new(Ty::Bool, c.bool(true)))
        }),
        // Spec `remainder` examples: the sign of the dividend.
        differs("f32_remainder", |c| {
            let (a, b, w) = (f32v(c, -5.5), f32v(c, 2.0), f32v(c, -1.5));
            (ap(c, "remainder", &none, &[a, b], TF32), w)
        }),
        differs("f32_remainder_positive", |c| {
            let (a, b, w) = (f32v(c, 5.5), f32v(c, 2.0), f32v(c, 1.5));
            (ap(c, "remainder", &none, &[a, b], TF32), w)
        }),
        differs("f32_round_nearest_even", |c| {
            let (a, w) = (f32v(c, 2.5), f32v(c, 2.0));
            (ap(c, "round_nearest_even", &none, &[a], TF32), w)
        }),
        differs("f32_floor", |c| {
            let (a, w) = (f32v(c, -1.5), f32v(c, -2.0));
            (ap(c, "floor", &none, &[a], TF32), w)
        }),
        differs("f32_sign_negative_zero", |c| {
            let (a, w) = (f32v(c, -0.0), f32v(c, -0.0));
            (ap(c, "sign", &none, &[a], TF32), w)
        }),
        // bf16: ties to even, arithmetic in the carrier.
        differs("f32_to_bf16_tie_down", |c| {
            let (a, w) = (f32v(c, f32::from_bits(0x3f80_8000)), bits(c, TBF16, 0x3f80));
            (ops::convert(c, a, TBF16).unwrap(), w)
        }),
        differs("f32_to_bf16_tie_up", |c| {
            let (a, w) = (f32v(c, f32::from_bits(0x3f81_8000)), bits(c, TBF16, 0x3f82));
            (ops::convert(c, a, TBF16).unwrap(), w)
        }),
        differs("bf16_add_rounds", |c| {
            // 1 + 2^-7 + 2^-8 = 1.01171875, a tie: to even 1.015625.
            let (a, b, w) = (
                bits(c, TBF16, 0x3f81),
                bits(c, TBF16, 0x3b80),
                bits(c, TBF16, 0x3f82),
            );
            (ap(c, "add", &none, &[a, b], TBF16), w)
        }),
        differs("bf16_negate", |c| {
            let (a, w) = (bits(c, TBF16, 0x3f80), bits(c, TBF16, 0xbf80));
            (ap(c, "negate", &none, &[a], TBF16), w)
        }),
        // OCP OFP8 E4M3FN: max 448, the tie at 464 goes to 448; a conversion saturates beyond it and
        // at infinity (design §6.4), NaN stays NaN; subnormals.
        differs("f32_to_e4m3fn_max", |c| {
            let (a, w) = (f32v(c, 448.0), bits(c, TF8, 0x7e));
            (ops::convert(c, a, TF8).unwrap(), w)
        }),
        differs("f32_to_e4m3fn_tie_464", |c| {
            let (a, w) = (f32v(c, 464.0), bits(c, TF8, 0x7e));
            (ops::convert(c, a, TF8).unwrap(), w)
        }),
        differs("f32_to_e4m3fn_overflow_saturates", |c| {
            let (a, w) = (f32v(c, 470.0), bits(c, TF8, 0x7e));
            (ops::convert(c, a, TF8).unwrap(), w)
        }),
        differs("f32_to_e4m3fn_inf_saturates", |c| {
            let (a, w) = (f32v(c, f32::NEG_INFINITY), bits(c, TF8, 0xfe));
            (ops::convert(c, a, TF8).unwrap(), w)
        }),
        differs("f32_to_e4m3fn_nan_stays_nan", |c| {
            let (a, w) = (f32v(c, f32::NAN), bits(c, TF8, 0x7f));
            (ops::convert(c, a, TF8).unwrap(), w)
        }),
        differs("f32_to_e5m2_overflow_saturates", |c| {
            let (a, w) = (f32v(c, 1e6), bits(c, TF8E5, 0x7b));
            (ops::convert(c, a, TF8E5).unwrap(), w)
        }),
        differs("f32_to_e5m2_inf_saturates", |c| {
            let (a, w) = (f32v(c, f32::INFINITY), bits(c, TF8E5, 0x7b));
            (ops::convert(c, a, TF8E5).unwrap(), w)
        }),
        // f32 and bf16 flush (design §6.4): subnormal operands read as zero of their sign, results
        // below 2^-126 after rounding to the format's precision become zero of their sign.
        differs("f32_multiply_flushes", |c| {
            let (a, b, w) = (f32v(c, 2f32.powi(-63)), f32v(c, 2f32.powi(-64)), f32v(c, 0.0));
            (ap(c, "multiply", &none, &[a, b], TF32), w)
        }),
        differs("f32_multiply_flush_keeps_sign", |c| {
            let (a, b, w) = (f32v(c, -(2f32.powi(-63))), f32v(c, 2f32.powi(-64)), f32v(c, -0.0));
            (ap(c, "multiply", &none, &[a, b], TF32), w)
        }),
        differs("f32_multiply_flushes_before_the_ieee_round_up", |c| {
            // (1 - 2^-24) · 2^-126: 24 bits with an unbounded exponent, below 2^-126, so flushed;
            // IEEE gradual underflow would round this tie up to 2^-126.
            let (a, b, w) = (
                f32v(c, f32::from_bits(0x3f7f_ffff)),
                f32v(c, f32::from_bits(0x0080_0000)),
                f32v(c, 0.0),
            );
            (ap(c, "multiply", &none, &[a, b], TF32), w)
        }),
        differs("f32_divide_flushes", |c| {
            let (a, b, w) = (f32v(c, 2f32.powi(-100)), f32v(c, 2f32.powi(30)), f32v(c, 0.0));
            (ap(c, "divide", &none, &[a, b], TF32), w)
        }),
        differs("f32_subtract_flushes", |c| {
            let (a, b, w) = (
                f32v(c, f32::from_bits(0x00c0_0000)),
                f32v(c, f32::from_bits(0x0080_0000)),
                f32v(c, 0.0),
            );
            (ap(c, "subtract", &none, &[a, b], TF32), w)
        }),
        differs("f32_add_reads_subnormal_as_zero", |c| {
            let (a, b, w) = (f32v(c, f32::from_bits(1)), f32v(c, 0.0), f32v(c, 0.0));
            (ap(c, "add", &none, &[a, b], TF32), w)
        }),
        differs("f32_negate_reads_subnormal_as_zero", |c| {
            let (a, w) = (f32v(c, f32::from_bits(1)), f32v(c, -0.0));
            (ap(c, "negate", &none, &[a], TF32), w)
        }),
        differs("f32_compare_subnormal_equals_zero", |c| {
            let (a, b) = (f32v(c, f32::from_bits(1)), f32v(c, -0.0));
            (
                ap(c, "compare", &cmp("EQ", Some("FLOAT")), &[a, b], Ty::Bool),
                Val::new(Ty::Bool, c.bool(true)),
            )
        }),
        differs("f32_multiply_min_normal_is_kept", |c| {
            let (a, b, w) = (
                f32v(c, 2f32.powi(-63)),
                f32v(c, 2f32.powi(-63)),
                f32v(c, 2f32.powi(-126)),
            );
            (ap(c, "multiply", &none, &[a, b], TF32), w)
        }),
        differs("bf16_multiply_flushes", |c| {
            let (a, b, w) = (
                bits(c, TBF16, 0x2000),
                bits(c, TBF16, 0x1f80),
                bits(c, TBF16, 0x0000),
            );
            (ap(c, "multiply", &none, &[a, b], TBF16), w)
        }),
        differs("bf16_add_reads_subnormal_as_zero", |c| {
            let (a, b, w) = (
                bits(c, TBF16, 0x8001),
                bits(c, TBF16, 0x0000),
                bits(c, TBF16, 0x0000),
            );
            (ap(c, "add", &none, &[a, b], TBF16), w)
        }),
        differs("bf16_multiply_flushes_before_the_ieee_round_up", |c| {
            // (2 - 2^-7) · 2^-63 · 2^-64 = 2^-126 - 2^-134: 8 bits, below 2^-126, so flushed (IEEE
            // gradual underflow: a tie, rounded up to 2^-126).
            let (a, b, w) = (
                bits(c, TBF16, 0x207f),
                bits(c, TBF16, 0x1f80),
                bits(c, TBF16, 0x0000),
            );
            (ap(c, "multiply", &none, &[a, b], TBF16), w)
        }),
        differs("bf16_multiply_rounds_up_to_min_normal", |c| {
            // (2 - 2^-7)(1 + 2^-7) · 2^-127 = 2^-126 - 2^-141 rounds to 2^-126 at 8 bits: kept.
            let (a, b, w) = (
                bits(c, TBF16, 0x207f),
                bits(c, TBF16, 0x1f81),
                bits(c, TBF16, 0x0080),
            );
            (ap(c, "multiply", &none, &[a, b], TBF16), w)
        }),
        differs("f32_to_bf16_reads_subnormal_as_zero", |c| {
            let (a, w) = (f32v(c, f32::from_bits(0x007f_e000)), bits(c, TBF16, 0x0000));
            (ops::convert(c, a, TBF16).unwrap(), w)
        }),
        differs("bf16_to_f32_reads_subnormal_as_zero", |c| {
            let (a, w) = (bits(c, TBF16, 0x8001), f32v(c, -0.0));
            (ops::convert(c, a, TF32).unwrap(), w)
        }),
        differs("f32_to_e4m3fn_min_subnormal", |c| {
            let (a, w) = (f32v(c, 2f32.powi(-9)), bits(c, TF8, 0x01));
            (ops::convert(c, a, TF8).unwrap(), w)
        }),
        differs("f32_to_e4m3fn_subnormal_tie", |c| {
            // 3 · 2^-10 lies between 2^-9 (0x01) and 2^-8 (0x02): to even, 0x02.
            let (a, w) = (f32v(c, 3.0 * 2f32.powi(-10)), bits(c, TF8, 0x02));
            (ops::convert(c, a, TF8).unwrap(), w)
        }),
        differs("f32_to_e4m3fn_underflow", |c| {
            let (a, w) = (f32v(c, 2f32.powi(-10)), bits(c, TF8, 0x00));
            (ops::convert(c, a, TF8).unwrap(), w)
        }),
    ];
    // `rescale` (the fixed-point requantization primitive) against its integer reference: scales with
    // a right shift, a tie, a left shift, saturation, zero, a subnormal, infinity, every width.
    for (i, &(x, s, w)) in [
        (5i64, 0.5f32, 32u32),
        (-5, 0.5, 32),
        (-7, -0.5, 32),
        (100, 0.75, 32),
        (i64::from(i32::MAX), 2f32.powi(-40), 32),
        (i64::from(i32::MIN), 0.001, 32),
        (1, 2f32.powi(23), 32),
        (3, 2f32.powi(31), 32),
        (-3, 2f32.powi(30), 32),
        (0, f32::INFINITY, 32),
        (7, f32::from_bits(1), 32),
        (-128, 0.0123, 8),
        (12345, 0.5, 16),
    ]
    .iter()
    .enumerate()
    {
        unsat.push(differs(&format!("rescale_{i}"), |c| {
            let ty = Ty::Int { signed: true, w };
            let a = int(c, ty, i128::from(x));
            let sv = f32v(c, s);
            let want = Val::new(
                TF32,
                float::lit(c, F32, u128::from(ops::rescale_reference(x, s.to_bits()))).unwrap(),
            );
            (ap(c, "rescale", &none, &[a, sv], TF32), want)
        }));
    }
    // The whole E4M3FN decoding table against the independent formula.
    unsat.push(bool_query("e4m3fn_decode_table", |c| {
        let mut eqs = Vec::new();
        for b in 0u8..=255 {
            let x = c.bv(u128::from(b), 8);
            let d = float::to_f32(c, Format::E4M3FN, x).unwrap();
            let want = match e4m3fn(b) {
                Some(v) => float::lit(c, F32, u128::from(v.to_bits())).unwrap(),
                None => float::nan(c, F32).unwrap(),
            };
            eqs.push(c.eq(d, want).unwrap());
        }
        let all = c.and(&eqs).unwrap();
        c.not(all).unwrap()
    }));
    // Negative controls: these differ for some input.
    let sat = vec![
        bool_query("control_swapped_subtract", |c| {
            let a = Val::new(I32, c.var("a", Sort::Bv(32)).unwrap());
            let b = Val::new(I32, c.var("b", Sort::Bv(32)).unwrap());
            let x = ap(c, "subtract", &none, &[a, b], I32);
            let y = ap(c, "subtract", &none, &[b, a], I32);
            let e = ops::same(c, x, y).unwrap();
            c.not(e).unwrap()
        }),
        // An IEEE multiplier (gradual underflow) does not meet the flushing spec, in f32 nor as a bf16
        // multiply on an f32 unit.
        bool_query("control_ieee_f32_multiply_is_not_ftz", |c| {
            let x = c.var("x", F32).unwrap();
            let y = c.var("y", F32).unwrap();
            let ieee = c.op2(Op::FpMul(Rm::Rne), x, y).unwrap();
            let spec = ap(
                c,
                "multiply",
                &none,
                &[Val::new(TF32, x), Val::new(TF32, y)],
                TF32,
            );
            let e = ops::same(c, spec, Val::new(TF32, ieee)).unwrap();
            c.not(e).unwrap()
        }),
        bool_query("control_ieee_f32_unit_is_not_ftz_bf16", |c| {
            let a = Val::new(TBF16, c.var("a", Sort::Bv(16)).unwrap());
            let b = Val::new(TBF16, c.var("b", Sort::Bv(16)).unwrap());
            let (x, y) = (
                float::to_f32(c, Format::BF16, a.t).unwrap(),
                float::to_f32(c, Format::BF16, b.t).unwrap(),
            );
            let p = c.op2(Op::FpMul(Rm::Rne), x, y).unwrap();
            let ieee = float::from_f32(c, Format::BF16, p).unwrap();
            let spec = ap(c, "multiply", &none, &[a, b], TBF16);
            let e = ops::same(c, spec, Val::new(TBF16, ieee)).unwrap();
            c.not(e).unwrap()
        }),
        bool_query("control_truncation_is_not_rounding", |c| {
            let x = c.var("x", F32).unwrap();
            let r = float::from_f32(c, Format::BF16, x).unwrap();
            let b = c.bits_of(x).unwrap();
            let t = c.extract(b, 31, 16).unwrap();
            let e = float::same_bits(c, Format::BF16, r, t).unwrap();
            c.not(e).unwrap()
        }),
    ];
    (unsat, sat)
}

#[test]
fn op_semantics_match_the_spec_cases() {
    let env = Env::new();
    let cfg: toml::Value =
        toml::from_str(&std::fs::read_to_string(repo_root().join("config.toml")).unwrap()).unwrap();
    let tag = cfg["tools"]["docker"]["tag"].as_str().unwrap().to_string();
    let host = std::env::var("CC_SMT_RUNNER").as_deref() == Ok("host");
    let mut sets = vec![format!("tools.docker.tag={tag}")];
    if host {
        sets.push("tools.smt.runner=host".into());
    }
    let sets: Vec<&str> = sets.iter().map(String::as_str).collect();
    let lc = env.config("[models]\nkeys = []\n", &sets);
    let dir = env.root().join("smt");
    std::fs::create_dir_all(&dir).unwrap();
    let (unsat, sat) = cases();
    let mut files = Vec::new();
    for (f, t) in unsat.iter().chain(&sat) {
        std::fs::write(dir.join(f), t).unwrap();
        files.push(f.clone());
    }
    let runners = Runners::from_config(&lc);
    let c = &lc.config.tools.smt;
    let (r, loc) = if host {
        (runners.host, cc_smt::host_paths(&c.z3, &c.bitwuzla))
    } else {
        (runners.docker, Location::Image)
    };
    let runner = BoundRunner::new(
        r,
        Caps {
            memory_gb: 2.0,
            cpus: 1.0,
            timeout_s: 300.0,
            pids_limit: 64,
        },
        ToolLog::new("smt-test", dir.join("logs")),
        CancelToken::new(),
    );
    for solver in Solver::ALL {
        let b = cc_smt::run_batch(&runner, &loc, solver, &dir, &files, 60).unwrap();
        for (f, _) in &unsat {
            assert_eq!(b.verdicts.get(f), Some(&Verdict::Unsat), "{solver:?} {f}");
        }
        for (f, _) in &sat {
            assert!(
                matches!(b.verdicts.get(f), Some(Verdict::Sat { .. })),
                "{solver:?} {f}: {:?}",
                b.verdicts.get(f)
            );
        }
    }
}

/// `merge`'s `rescale` tolerance lemma at i8 holds, and its negative controls (a tolerance of 0; a
/// decode off by one place) have counterexamples: the hypotheses are satisfiable, the claims not vacuous.
#[test]
fn rescale_lemma_holds_and_its_controls_fail() {
    use cc_passes::merge::requant::{lemma, lemma_control};
    let env = Env::new();
    let cfg: toml::Value =
        toml::from_str(&std::fs::read_to_string(repo_root().join("config.toml")).unwrap()).unwrap();
    let tag = cfg["tools"]["docker"]["tag"].as_str().unwrap().to_string();
    let host = std::env::var("CC_SMT_RUNNER").as_deref() == Ok("host");
    let mut sets = vec![format!("tools.docker.tag={tag}")];
    if host {
        sets.push("tools.smt.runner=host".into());
    }
    let sets: Vec<&str> = sets.iter().map(String::as_str).collect();
    let lc = env.config("[models]\nkeys = []\n", &sets);
    let dir = env.root().join("rescale");
    std::fs::create_dir_all(&dir).unwrap();
    let i8t = Ty::Int { signed: true, w: 8 };
    let qs = [
        ("holds.smt2", lemma(i8t).unwrap().1, true),
        ("tolerance0.smt2", lemma_control(i8t, 0.0, 0).unwrap().1, false),
        ("decode.smt2", lemma_control(i8t, 1.0, 1).unwrap().1, false),
    ];
    for (f, t, _) in &qs {
        std::fs::write(dir.join(f), t).unwrap();
    }
    let runners = Runners::from_config(&lc);
    let c = &lc.config.tools.smt;
    let (r, loc) = if host {
        (runners.host, cc_smt::host_paths(&c.z3, &c.bitwuzla))
    } else {
        (runners.docker, Location::Image)
    };
    let runner = BoundRunner::new(
        r,
        Caps {
            memory_gb: 4.0,
            cpus: 1.0,
            timeout_s: 600.0,
            pids_limit: 64,
        },
        ToolLog::new("rescale-lemma", dir.join("logs")),
        CancelToken::new(),
    );
    let files: Vec<String> = qs.iter().map(|q| q.0.to_string()).collect();
    for solver in Solver::ALL {
        let b = cc_smt::run_batch(&runner, &loc, solver, &dir, &files, 300).unwrap();
        for (f, _, holds) in &qs {
            let v = b.verdicts.get(*f);
            if *holds {
                assert_eq!(v, Some(&Verdict::Unsat), "{solver:?} {f}");
            } else {
                assert!(matches!(v, Some(Verdict::Sat { .. })), "{solver:?} {f}: {v:?}");
            }
        }
    }
}

/// The encoding lemmas that Z3 decides in seconds (every one except the bf16 multiply and divide, which
/// take about 45 s each and run in `datapaths` when a model uses them) hold, including those of formats
/// no model uses, and the f32 multiply and divide lemmas fail with a wrong flush limit.
#[test]
fn quick_encoding_lemmas_hold() {
    let env = Env::new();
    let cfg: toml::Value =
        toml::from_str(&std::fs::read_to_string(repo_root().join("config.toml")).unwrap()).unwrap();
    let tag = cfg["tools"]["docker"]["tag"].as_str().unwrap().to_string();
    let host = std::env::var("CC_SMT_RUNNER").as_deref() == Ok("host");
    let mut sets = vec![format!("tools.docker.tag={tag}")];
    if host {
        sets.push("tools.smt.runner=host".into());
    }
    let sets: Vec<&str> = sets.iter().map(String::as_str).collect();
    let lc = env.config("[models]\nkeys = []\n", &sets);
    let dir = env.root().join("lemmas");
    std::fs::create_dir_all(&dir).unwrap();
    let used: std::collections::BTreeSet<String> = [
        "bf16.decode",
        "bf16.round",
        "bf16.add",
        "bf16.sub",
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
        "f8e5m2.decode",
        "f8e5m2.round",
        "f8e5m2.saturate",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let lemmas = float::lemmas(&used).unwrap();
    let mut files = Vec::new();
    for (f, t) in &lemmas {
        std::fs::write(dir.join(f), t).unwrap();
        files.push(f.clone());
    }
    let runners = Runners::from_config(&lc);
    let c = &lc.config.tools.smt;
    let (r, loc) = if host {
        (runners.host, cc_smt::host_paths(&c.z3, &c.bitwuzla))
    } else {
        (runners.docker, Location::Image)
    };
    let runner = BoundRunner::new(
        r,
        Caps {
            memory_gb: 2.0,
            cpus: 1.0,
            timeout_s: 300.0,
            pids_limit: 64,
        },
        ToolLog::new("lemmas", dir.join("logs")),
        CancelToken::new(),
    );
    // Negative controls: the f32 multiply and divide lemmas with the flush limit one binade off.
    let mut controls = Vec::new();
    for (op, name) in [
        (float::Arith::Mul, "control_mul.smt2"),
        (float::Arith::Div, "control_div.smt2"),
    ] {
        std::fs::write(dir.join(name), float::scaled_control(op, -1).unwrap()).unwrap();
        files.push(name.to_string());
        controls.push(name.to_string());
    }
    let b = cc_smt::run_batch(&runner, &loc, Solver::Z3, &dir, &files, 60).unwrap();
    for f in &files {
        let v = b.verdicts.get(f);
        if controls.contains(f) {
            assert!(matches!(v, Some(Verdict::Sat { .. })), "{f}: {v:?}");
        } else {
            assert_eq!(v, Some(&Verdict::Unsat), "{f}");
        }
    }
}
