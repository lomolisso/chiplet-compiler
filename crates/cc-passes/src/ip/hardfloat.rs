//! Berkeley HardFloat (J. R. Hauser; the Chisel repository github.com/ucb-bar/berkeley-hardfloat at the
//! pinned submodule, BSD-3-Clause), through our generator in the toolchain image (`hardfloat-gen`,
//! docs/toolchain-image.md): an IEEE-interface module around one HardFloat core, lowered to Verilog by
//! firtool. Every operation "fully conforms" to IEEE 754 with gradual underflow; the formats that flush
//! (design §6.4) wrap the core in glue (design §4.12).
//!
//! | Op | Core | Ports |
//! |---|---|---|
//! | `add`, `sub` | `AddRecFN` | `a`, `b`, `rnd`, `z`, `flags` |
//! | `mul` | `MulRecFN` | `a`, `b`, `rnd`, `z`, `flags` |
//! | `div`, `sqrt` | `DivSqrtRecFN_small` (iterative: at most p + 2 cycles) | `clock`, `reset`, `start`, `ready`, `a`, (`b`), `rnd`, `done`, `z`, `flags` |
//! | `compare` | `CompareRecFN` (quiet; `lt` is `z`) | `a`, `b`, `lt`, `eq`, `gt`, `flags` |
//! | `to_int.<rnd>.<s\|u><w>` | `RecFNToIN` (saturating; `w >= 3`) | `a`, `rnd`, `z`, `flags` |
//! | `from_int.<s\|u><w>` | `INToRecFN` | `a`, `rnd`, `z`, `flags` |
//! | `convert.e<E>m<M>` | `RecFNToRecFN` | `a`, `rnd`, `z`, `flags` |
//!
//! Formats: exponent width w >= 3, precision p = sig_width + 1 >= 3 and p <= 2^(w-2) + 3 (HardFloat
//! Release 1 documentation, section 2), for operands and results; tested by its authors on f16, f32, f64
//! and f128. The adder elaborates only where its alignment mask fits ([`add_ok`]: not at bf16's p = 8). `rnd` uses HardFloat's encoding: 0 nearest even, 1 toward zero, 2 down, 3 up. Round to integral
//! has no HardFloat module; it composes `to_int` and `from_int` with glue (`routes`), as in DesignWare.

use cc_core::adapters::{
    Accuracy, CoreOffer, CoreOp, CoreSource, Encoding, Generator, IpCapabilities, IpProvider, Named,
    Rounding, Subnormals, Timing,
};

use super::{contract, generated_name, instance, model};

/// Berkeley HardFloat (both flows).
#[derive(Debug, Clone, Copy)]
pub struct HardFloat;

/// A format HardFloat supports.
pub fn format_ok(e: u32, s: u32) -> bool {
    let p = u64::from(s) + 1;
    (3..=32).contains(&e) && p >= 3 && p <= (1u64 << (e - 2)) + 3
}

/// A precision HardFloat's adder elaborates at: `AddRawFN` masks the alignment distance's high bits
/// with `lowMask(alignDist(w - 1, 2), (p + 5) / 4, 0)`, `w = ceil(log2 p)`, whose bound must not exceed
/// the `2^(w - 2)` values of that field (`AddRecFN.scala`, `primitives.scala` at the pinned commit). It
/// fails, for instance, at bf16 (p = 8), at p = 3, 4, 7, 16 and 32.
pub fn add_ok(s: u32) -> bool {
    let p = u64::from(s) + 1;
    let w = 64 - (p - 1).leading_zeros() as u64;
    p >= 5 && (p + 5) / 4 <= 1u64 << (w - 2)
}

fn rnd(r: Rounding) -> &'static str {
    match r {
        Rounding::NearestEven => "3'b000",
        Rounding::TowardZero => "3'b001",
        Rounding::Floor => "3'b010",
        Rounding::Ceil => "3'b011",
    }
}

impl Named for HardFloat {
    fn name(&self) -> &'static str {
        "hardfloat"
    }
}

impl IpProvider for HardFloat {
    fn capabilities(&self) -> IpCapabilities {
        IpCapabilities {
            flows: &["dc", "yosys"],
            licence: "BSD-3-Clause (Berkeley HardFloat)",
            ops: &[
                "add", "sub", "mul", "div", "sqrt", "compare", "to_int", "from_int", "convert",
            ],
        }
    }

    fn core(&self, op: &str, e: u32, s: u32) -> Option<CoreOffer> {
        let op: CoreOp = op.parse().ok()?;
        if !format_ok(e, s) {
            return None;
        }
        let mut args: Vec<String> = vec![
            op.family().to_string(),
            "--exp".into(),
            e.to_string(),
            "--sig".into(),
            s.to_string(),
        ];
        let mut timing = Timing::Combinational;
        // The generated module serves every rounding mode (a port), so a `to_int` core's name has none.
        let name_op = match op {
            CoreOp::ToInt { signed, width, .. } => {
                format!("to_int_{}{width}", if signed { "s" } else { "u" })
            }
            _ => op.to_string(),
        };
        let module = generated_name("hardfloat", &name_op, e, s);
        let binary = [
            ("a", "a0"),
            ("b", "a1"),
            ("rnd", "3'b000"),
            ("z", "z"),
            ("flags", ""),
        ];
        let unary = [("a", "a0"), ("rnd", "3'b000"), ("z", "z"), ("flags", "")];
        let ports: Vec<(&str, &str)> = match op {
            CoreOp::Add | CoreOp::Sub if add_ok(s) => binary.to_vec(),
            CoreOp::Mul => binary.to_vec(),
            CoreOp::Div | CoreOp::Sqrt => {
                timing = Timing::Iterative { max_cycles: s + 3 };
                let mut v = vec![
                    ("clock", "clk"),
                    ("reset", "rst"),
                    ("start", "start"),
                    ("ready", ""),
                    ("done", "done"),
                    ("a", "a0"),
                    ("rnd", "3'b000"),
                    ("z", "z"),
                    ("flags", ""),
                ];
                if op == CoreOp::Div {
                    v.push(("b", "a1"));
                }
                v
            }
            CoreOp::Compare => vec![
                ("a", "a0"),
                ("b", "a1"),
                ("lt", "z"),
                ("eq", ""),
                ("gt", ""),
                ("flags", ""),
            ],
            CoreOp::ToInt {
                rnd: r,
                signed,
                width,
            } if (3..=256).contains(&width) => {
                args.extend([
                    "--int".into(),
                    width.to_string(),
                    "--signed".into(),
                    u32::from(signed).to_string(),
                ]);
                vec![("a", "a0"), ("rnd", rnd(r)), ("z", "z"), ("flags", "")]
            }
            CoreOp::FromInt { signed, width } if (2..=256).contains(&width) => {
                args.extend([
                    "--int".into(),
                    width.to_string(),
                    "--signed".into(),
                    u32::from(signed).to_string(),
                ]);
                unary.to_vec()
            }
            CoreOp::Convert { exp_width, sig_width }
                if format_ok(exp_width, sig_width) && (exp_width, sig_width) != (e, s) =>
            {
                args.extend([
                    "--out-exp".into(),
                    exp_width.to_string(),
                    "--out-sig".into(),
                    sig_width.to_string(),
                ]);
                unary.to_vec()
            }
            _ => return None,
        };
        Some(CoreOffer {
            instance: instance(&module, &[], &ports, e, s),
            contract: contract(Accuracy::CorrectlyRounded, true, Subnormals::Ieee),
            delay_ps: None,
            model: model(&op),
            source: CoreSource::Generated {
                generator: Generator::Hardfloat { args },
            },
            encoding: Encoding::Ieee,
            timing,
        })
    }
}
