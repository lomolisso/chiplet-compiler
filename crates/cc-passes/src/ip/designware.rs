//! Synopsys DesignWare floating-point components (DesignWare Building Block IP datasheets, DWBB_202409.0),
//! for Design Compiler, which links them from its synthetic library `dw_foundation.sldb`. Every instance
//! is IEEE compliant (`ieee_compliance = 1`: subnormals and NaNs); the formats that flush (design §6.4)
//! wrap the core in glue (design §4.12). Rounding ports are tied to round to nearest even (`rnd` 0) unless
//! the op names a mode (DesignWare's encoding: 1 toward zero, 2 up, 3 down).
//!
//! | Op | Component | `sig_width` / `exp_width` | Contract (datasheet) |
//! |---|---|---|---|
//! | `add`, `sub`, `mul`, `div` | `DW_fp_add`, `DW_fp_sub`, `DW_fp_mult`, `DW_fp_div` | 2-253 / 3-31 | correctly rounded |
//! | `sqrt`, `rsqrt` | `DW_fp_sqrt`, `DW_fp_invsqrt` | 2-253 / 3-31 | correctly rounded |
//! | `exp`, `exp2` | `DW_fp_exp`, `DW_fp_exp2` (`arch = 0`) | 2-57, 2-58 / 3-31 | 2 ulps |
//! | `ln` | `DW_fp_ln` (`extra_prec = sig_width`, `arch = 0`) | 2-29 / 3-31 | 1 ulp for any input |
//! | `log2` | `DW_fp_log2` (`extra_prec = sig_width`, `arch = 0`) | 2-29 / 3-31 | 2 ulps for any input |
//! | `sin`, `cos` | `DW_fp_sincos` (radians, `arch = 2`, `err_range = 1`) | 2-32 / 3-31 | relative error not bounded |
//! | `compare` | `DW_fp_cmp` | 2-253 / 3-31 | exact (also `unordered`) |
//! | `to_int.<rnd>.s<w>` | `DW_fp_flt2i` (`isize = w`, 3-512) | 2-253 / 3-31 | rounded per `rnd`, saturating |
//! | `from_int.<s\|u><w>` | `DW_fp_i2flt` (`isize = w`, `isign`) | 2-253 / 3-31 | correctly rounded |
//!
//! `ln` and `log2` keep their bound for every input only with `extra_prec = sig_width`, which the
//! component limits to `59 - sig_width`: hence `sig_width <= 29`. `DW_fp_flt2i` converts to signed integers
//! only. No component converts between float formats (HardFloat does, in this flow too).

use cc_core::adapters::{
    Accuracy, CoreOffer, CoreOp, CoreSource, Encoding, IpCapabilities, IpProvider, Named, Rounding,
    Subnormals, Timing,
};

use super::{contract, instance, model};

/// DesignWare (Design Compiler).
#[derive(Debug, Clone, Copy)]
pub struct DesignWare;

/// The `rnd` port value of a rounding mode.
fn rnd(r: Rounding) -> &'static str {
    match r {
        Rounding::NearestEven => "3'b000",
        Rounding::TowardZero => "3'b001",
        Rounding::Ceil => "3'b010",
        Rounding::Floor => "3'b011",
    }
}

impl Named for DesignWare {
    fn name(&self) -> &'static str {
        "designware"
    }
}

impl IpProvider for DesignWare {
    fn capabilities(&self) -> IpCapabilities {
        IpCapabilities {
            flows: &["dc"],
            licence: "Synopsys DesignWare Building Block IP (proprietary; the DesignWare licence feature)",
            ops: &[
                "add", "sub", "mul", "div", "sqrt", "rsqrt", "exp", "exp2", "ln", "log2", "sin", "cos",
                "compare", "to_int", "from_int",
            ],
        }
    }

    fn core(&self, op: &str, e: u32, s: u32) -> Option<CoreOffer> {
        let op: CoreOp = op.parse().ok()?;
        let in_range = |max_sig: u32| (3..=31).contains(&e) && (2..=max_sig).contains(&s);
        let p = |k: &'static str, v: u32| (k, v.to_string());
        let widths = vec![p("sig_width", s), p("exp_width", e), p("ieee_compliance", 1)];
        let with = |extra: Vec<(&'static str, String)>| {
            let mut v = widths.clone();
            v.extend(extra);
            v
        };
        let cr = || contract(Accuracy::CorrectlyRounded, true, Subnormals::Ieee);
        let ulp = |u: f64| contract(Accuracy::Ulp { ulp: u }, true, Subnormals::Ieee);
        let binary = [
            ("a", "a0"),
            ("b", "a1"),
            ("rnd", "3'b000"),
            ("z", "z"),
            ("status", ""),
        ];
        let unary_rnd = [("a", "a0"), ("rnd", "3'b000"), ("z", "z"), ("status", "")];
        let unary = [("a", "a0"), ("z", "z"), ("status", "")];
        let (inst, c) = match op {
            CoreOp::Add | CoreOp::Sub | CoreOp::Mul if in_range(253) => {
                let m = match op {
                    CoreOp::Add => "DW_fp_add",
                    CoreOp::Sub => "DW_fp_sub",
                    _ => "DW_fp_mult",
                };
                (instance(m, &widths, &binary, e, s), cr())
            }
            CoreOp::Div if in_range(253) => (
                instance("DW_fp_div", &with(vec![p("faithful_round", 0)]), &binary, e, s),
                cr(),
            ),
            CoreOp::Sqrt if in_range(253) => (instance("DW_fp_sqrt", &widths, &unary_rnd, e, s), cr()),
            CoreOp::Rsqrt if in_range(253) => (instance("DW_fp_invsqrt", &widths, &unary_rnd, e, s), cr()),
            CoreOp::Exp if in_range(57) => (
                instance("DW_fp_exp", &with(vec![p("arch", 0)]), &unary, e, s),
                ulp(2.0),
            ),
            CoreOp::Exp2 if in_range(58) => (
                instance("DW_fp_exp2", &with(vec![p("arch", 0)]), &unary, e, s),
                ulp(2.0),
            ),
            CoreOp::Ln | CoreOp::Log2 if in_range(29) => {
                let (m, bound) = if op == CoreOp::Ln {
                    ("DW_fp_ln", 1.0)
                } else {
                    ("DW_fp_log2", 2.0)
                };
                (
                    instance(m, &with(vec![p("extra_prec", s), p("arch", 0)]), &unary, e, s),
                    ulp(bound),
                )
            }
            CoreOp::Sin | CoreOp::Cos if in_range(32) => {
                let sel = if op == CoreOp::Sin { "1'b0" } else { "1'b1" };
                (
                    instance(
                        "DW_fp_sincos",
                        &with(vec![p("pi_multiple", 0), p("arch", 2), p("err_range", 1)]),
                        &[("a", "a0"), ("sin_cos", sel), ("z", "z"), ("status", "")],
                        e,
                        s,
                    ),
                    contract(Accuracy::Unbounded, true, Subnormals::Ieee),
                )
            }
            CoreOp::Compare if in_range(253) => (
                instance(
                    "DW_fp_cmp",
                    &widths,
                    &[
                        ("a", "a0"),
                        ("b", "a1"),
                        ("zctr", "1'b0"),
                        ("altb", "lt"),
                        ("aeqb", "eq"),
                        ("agtb", "gt"),
                        ("unordered", "unordered"),
                        ("z0", ""),
                        ("z1", ""),
                        ("status0", ""),
                        ("status1", ""),
                    ],
                    e,
                    s,
                ),
                cr(),
            ),
            CoreOp::ToInt {
                rnd: r,
                signed: true,
                width,
            } if in_range(253) && (3..=512).contains(&width) => (
                instance(
                    "DW_fp_flt2i",
                    &[
                        p("sig_width", s),
                        p("exp_width", e),
                        p("isize", width),
                        p("ieee_compliance", 1),
                    ],
                    &[("a", "a0"), ("rnd", rnd(r)), ("z", "z"), ("status", "")],
                    e,
                    s,
                ),
                cr(),
            ),
            CoreOp::FromInt { signed, width }
                if in_range(253) && (3 + u32::from(signed)..=512).contains(&width) =>
            {
                (
                    instance(
                        "DW_fp_i2flt",
                        &[
                            p("sig_width", s),
                            p("exp_width", e),
                            p("isize", width),
                            p("isign", u32::from(signed)),
                        ],
                        &unary_rnd,
                        e,
                        s,
                    ),
                    cr(),
                )
            }
            _ => return None,
        };
        Some(CoreOffer {
            instance: inst,
            contract: c,
            delay_ps: None,
            model: model(&op),
            source: CoreSource::Library {
                name: "designware".into(),
            },
            encoding: Encoding::Ieee,
            timing: Timing::Combinational,
        })
    }
}
