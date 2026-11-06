//! FloPoCo 5.1 (gitlab.com/flopoco/flopoco, in the toolchain image; generated VHDL under LGPL-3.0),
//! combinational (`frequency=0`), read by Yosys through GHDL. Its operators work on FloPoCo's own format
//! (two exception bits, sign, exponent, significand); `InputIEEE` and `OutputIEEE`, FloPoCo operators too,
//! convert from and to IEEE bits. FloPoCo has no subnormals: `InputIEEE` flushes them to zero.
//!
//! | Op | Operator | Ports | Contract |
//! |---|---|---|---|
//! | `exp` | `FPExp` | `X`, `R` | faithful |
//! | `ln` | `FPLog` | `X`, `R` | faithful |
//! | `pow` | `FPPow` (`type=0`: pow) | `X`, `Y`, `R` | faithful; special cases not documented |
//! | `from_ieee`, `to_ieee` | `InputIEEE`, `OutputIEEE` (same widths) | `X`, `R` | exact |
//!
//! Widths: those FloPoCo's own unit tests run (`wE` 5 to 11, `wF` 5 to 52, `FPExp.cpp`, `FPLog.cpp`), and
//! for `FPExp` and `FPPow`, whose exponential needs `wF + g <= 2^(wE-1) - 1` (`FPExp.cpp`), with 4 guard
//! bits `g` assumed.

use cc_core::adapters::{
    Accuracy, CoreOffer, CoreOp, CoreSource, Encoding, Generator, IpCapabilities, IpProvider, Named,
    Subnormals, Timing,
};

use super::{contract, generated_name, instance, model};

/// FloPoCo (open flow).
#[derive(Debug, Clone, Copy)]
pub struct FloPoCo;

impl Named for FloPoCo {
    fn name(&self) -> &'static str {
        "flopoco"
    }
}

impl IpProvider for FloPoCo {
    fn capabilities(&self) -> IpCapabilities {
        IpCapabilities {
            flows: &["yosys"],
            licence: "LGPL-3.0 (the VHDL FloPoCo generates; FloPoCo itself is AGPL-3.0-based)",
            ops: &["exp", "ln", "pow", "from_ieee", "to_ieee"],
        }
    }

    fn core(&self, op: &str, e: u32, s: u32) -> Option<CoreOffer> {
        let op: CoreOp = op.parse().ok()?;
        if !(5..=11).contains(&e) || !(5..=52).contains(&s) {
            return None;
        }
        let exp_fits = u64::from(s) + 4 < (1u64 << (e - 1));
        let w = |k: &str, v: u32| format!("{k}={v}");
        let (args, ports, accuracy, specials): (Vec<String>, Vec<(&str, &str)>, Accuracy, bool) = match op {
            CoreOp::Exp if exp_fits => (
                vec!["FPExp".into(), w("wE", e), w("wF", s)],
                vec![("X", "a0"), ("R", "z")],
                Accuracy::Faithful,
                true,
            ),
            CoreOp::Ln => (
                vec!["FPLog".into(), w("wE", e), w("wF", s)],
                vec![("X", "a0"), ("R", "z")],
                Accuracy::Faithful,
                true,
            ),
            CoreOp::Pow if exp_fits => (
                vec!["FPPow".into(), w("wE", e), w("wF", s), "type=0".into()],
                vec![("X", "a0"), ("Y", "a1"), ("R", "z")],
                Accuracy::Faithful,
                false,
            ),
            CoreOp::FromIeee | CoreOp::ToIeee => {
                let name = if op == CoreOp::FromIeee {
                    "InputIEEE"
                } else {
                    "OutputIEEE"
                };
                (
                    vec![
                        name.into(),
                        w("wEIn", e),
                        w("wFIn", s),
                        w("wEOut", e),
                        w("wFOut", s),
                    ],
                    vec![("X", "a0"), ("R", "z")],
                    Accuracy::CorrectlyRounded,
                    true,
                )
            }
            _ => return None,
        };
        let mut args = args;
        args.push("frequency=0".into());
        let model = match op {
            CoreOp::FromIeee | CoreOp::ToIeee => format!("flopoco.{}", op.family()),
            _ => model(&op),
        };
        Some(CoreOffer {
            instance: instance(
                &generated_name("flopoco", &op.to_string(), e, s),
                &[],
                &ports,
                e,
                s,
            ),
            contract: contract(accuracy, specials, Subnormals::Flush),
            delay_ps: None,
            model,
            source: CoreSource::Generated {
                generator: Generator::Flopoco { args },
            },
            encoding: Encoding::Flopoco,
            timing: Timing::Combinational,
        })
    }
}
