//! Float IP cores (design §3.7, §4.12): a provider answers, for a core operation at given operand widths,
//! with one instance of a third-party core, its documented contract and how its RTL is obtained (a vendor
//! library the flow links, or a generator run in the toolchain image). Implementations: DesignWare,
//! Berkeley HardFloat, FloPoCo and XLS's float library (`cc_passes::ip`).
//!
//! Core operations are named as the cores compute them, not as StableHLO does (`cc_passes::ip::routes`
//! maps one to the other):
//!
//! | Op | Operands → result | Notes |
//! |---|---|---|
//! | `add`, `sub`, `mul`, `div` | `a0`, `a1` → `z` | |
//! | `sqrt`, `rsqrt`, `exp`, `exp2`, `ln`, `log2`, `sin`, `cos` | `a0` → `z` | radians |
//! | `pow` | `a0` (base), `a1` (exponent) → `z` | |
//! | `compare` | `a0`, `a1` → `z` = `a0 < a1` (one bit) | quiet (IEEE `compareQuietLess`) |
//! | `to_int.<rnd>.<s\|u><w>` | `a0` → `z`, a `w`-bit integer rounded per `rnd` | saturating |
//! | `from_int.<s\|u><w>` | `a0`, a `w`-bit integer → `z`, rounded to nearest even | |
//! | `convert.e<E>m<M>` | `a0` → `z` in the format with `E` exponent and `M` significand bits | nearest even |
//! | `from_ieee`, `to_ieee` | `a0` → `z` between IEEE and a core's own encoding (FloPoCo) | |
//!
//! `rnd` is `rne` (nearest even), `rtz` (toward zero), `floor`, `ceil`. The widths of a call are the
//! operand format's (for `from_int`, the result's): `exp_width` exponent bits and `sig_width` stored
//! significand bits, without the hidden bit (DesignWare's convention).
//!
//! An instance's ports map to operands, results, constants or nothing (`CoreInstance::ports`); an iterative
//! core also has `clk`, `rst`, `start` (operands valid, one cycle) and `done` (result valid).

use std::fmt;
use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::Named;
use super::contract::{Accuracy, Contract, CoreInstance};

/// Rounding of a conversion to an integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rounding {
    NearestEven,
    TowardZero,
    Floor,
    Ceil,
}

impl Rounding {
    pub const ALL: [Rounding; 4] = [
        Rounding::NearestEven,
        Rounding::TowardZero,
        Rounding::Floor,
        Rounding::Ceil,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Rounding::NearestEven => "rne",
            Rounding::TowardZero => "rtz",
            Rounding::Floor => "floor",
            Rounding::Ceil => "ceil",
        }
    }
}

/// A core operation (the table in the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoreOp {
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
    Rsqrt,
    Exp,
    Exp2,
    Ln,
    Log2,
    Sin,
    Cos,
    Pow,
    Compare,
    ToInt { rnd: Rounding, signed: bool, width: u32 },
    FromInt { signed: bool, width: u32 },
    Convert { exp_width: u32, sig_width: u32 },
    FromIeee,
    ToIeee,
}

impl CoreOp {
    /// The family name, without modifiers (`to_int`, `convert`): what capabilities list.
    pub fn family(&self) -> &'static str {
        match self {
            CoreOp::Add => "add",
            CoreOp::Sub => "sub",
            CoreOp::Mul => "mul",
            CoreOp::Div => "div",
            CoreOp::Sqrt => "sqrt",
            CoreOp::Rsqrt => "rsqrt",
            CoreOp::Exp => "exp",
            CoreOp::Exp2 => "exp2",
            CoreOp::Ln => "ln",
            CoreOp::Log2 => "log2",
            CoreOp::Sin => "sin",
            CoreOp::Cos => "cos",
            CoreOp::Pow => "pow",
            CoreOp::Compare => "compare",
            CoreOp::ToInt { .. } => "to_int",
            CoreOp::FromInt { .. } => "from_int",
            CoreOp::Convert { .. } => "convert",
            CoreOp::FromIeee => "from_ieee",
            CoreOp::ToIeee => "to_ieee",
        }
    }

    /// Operands of the op.
    pub fn arity(&self) -> usize {
        match self {
            CoreOp::Add | CoreOp::Sub | CoreOp::Mul | CoreOp::Div | CoreOp::Pow | CoreOp::Compare => 2,
            _ => 1,
        }
    }

    /// The op's result names (every op has one result, `z`; compare's is the less-than bit).
    pub fn results(&self) -> &'static [&'static str] {
        &["z"]
    }

    /// The exact operations of design §6.4 (arithmetic, conversions, compare): a core for them must be
    /// correctly rounded (exact for compare and the encoding changes).
    pub fn exact(&self) -> bool {
        !matches!(
            self,
            CoreOp::Rsqrt
                | CoreOp::Exp
                | CoreOp::Exp2
                | CoreOp::Ln
                | CoreOp::Log2
                | CoreOp::Sin
                | CoreOp::Cos
                | CoreOp::Pow
        )
    }
}

impl fmt::Display for CoreOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let su = |s: bool| if s { "s" } else { "u" };
        match self {
            CoreOp::ToInt { rnd, signed, width } => {
                write!(f, "to_int.{}.{}{width}", rnd.name(), su(*signed))
            }
            CoreOp::FromInt { signed, width } => write!(f, "from_int.{}{width}", su(*signed)),
            CoreOp::Convert { exp_width, sig_width } => write!(f, "convert.e{exp_width}m{sig_width}"),
            op => f.write_str(op.family()),
        }
    }
}

fn int_type(s: &str) -> Option<(bool, u32)> {
    let signed = match s.as_bytes().first()? {
        b's' => true,
        b'u' => false,
        _ => return None,
    };
    let w: u32 = s[1..].parse().ok()?;
    (w > 0).then_some((signed, w))
}

impl FromStr for CoreOp {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let bad = || format!("not a core op: {s}");
        let parts: Vec<&str> = s.split('.').collect();
        Ok(match parts.as_slice() {
            ["add"] => CoreOp::Add,
            ["sub"] => CoreOp::Sub,
            ["mul"] => CoreOp::Mul,
            ["div"] => CoreOp::Div,
            ["sqrt"] => CoreOp::Sqrt,
            ["rsqrt"] => CoreOp::Rsqrt,
            ["exp"] => CoreOp::Exp,
            ["exp2"] => CoreOp::Exp2,
            ["ln"] => CoreOp::Ln,
            ["log2"] => CoreOp::Log2,
            ["sin"] => CoreOp::Sin,
            ["cos"] => CoreOp::Cos,
            ["pow"] => CoreOp::Pow,
            ["compare"] => CoreOp::Compare,
            ["from_ieee"] => CoreOp::FromIeee,
            ["to_ieee"] => CoreOp::ToIeee,
            ["to_int", r, t] => {
                let rnd = Rounding::ALL
                    .into_iter()
                    .find(|x| x.name() == *r)
                    .ok_or_else(bad)?;
                let (signed, width) = int_type(t).ok_or_else(bad)?;
                CoreOp::ToInt { rnd, signed, width }
            }
            ["from_int", t] => {
                let (signed, width) = int_type(t).ok_or_else(bad)?;
                CoreOp::FromInt { signed, width }
            }
            ["convert", f] => {
                let rest = f.strip_prefix('e').ok_or_else(bad)?;
                let (e, m) = rest.split_once('m').ok_or_else(bad)?;
                CoreOp::Convert {
                    exp_width: e.parse().map_err(|_| bad())?,
                    sig_width: m.parse().map_err(|_| bad())?,
                }
            }
            _ => return Err(bad()),
        })
    }
}

/// A generator run in the toolchain image's RTL step (docs/passes/characterize.md §4.2). The module or
/// entity it writes is named after the primitive that holds it; the arguments never name it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "tool", rename_all = "snake_case", deny_unknown_fields)]
pub enum Generator {
    /// Our HardFloat generator (`hardfloat-gen <args> --module <id> --out <id>.v`): Berkeley HardFloat's
    /// Chisel cores behind an IEEE interface, lowered to Verilog by firtool.
    Hardfloat { args: Vec<String> },
    /// FloPoCo (`flopoco <args> name=<id> outputFile=<id>.vhdl`): VHDL, read by Yosys through GHDL.
    Flopoco { args: Vec<String> },
    /// A function of an XLS DSLX library under `$XLS_DSLX_PATH` (`file` relative to it, function `top`),
    /// converted to combinational Verilog like the catalog's DSLX.
    XlsDslx { file: String, top: String },
}

impl Generator {
    /// The extension of the file it writes: `v` or `vhdl`.
    pub fn extension(&self) -> &'static str {
        match self {
            Generator::Flopoco { .. } => "vhdl",
            _ => "v",
        }
    }
}

/// Where a core's RTL comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CoreSource {
    /// A vendor library that the synthesis flow links (`designware`: DC's synthetic library); nothing is
    /// generated, and only a flow that links the library can use the core.
    Library { name: String },
    /// RTL a generator writes in the RTL step.
    Generated { generator: Generator },
}

/// How the core encodes its operands and result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    /// IEEE 754 interchange bits: sign, biased exponent, stored significand.
    Ieee,
    /// FloPoCo's format: two exception bits, sign, exponent, significand (`exp_width + sig_width + 3`
    /// bits); `from_ieee` and `to_ieee` convert.
    Flopoco,
}

/// Combinational, or iterative with a handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Timing {
    Combinational,
    /// `start` to `done` in at most `max_cycles` clock cycles.
    Iterative {
        max_cycles: u32,
    },
}

/// What a provider answers for one core operation at one width.
#[derive(Debug, Clone, PartialEq)]
pub struct CoreOffer {
    pub instance: CoreInstance,
    pub contract: Contract,
    /// Combinational delay hint in ps, when the provider states one; `characterize` measures it.
    pub delay_ps: Option<u32>,
    /// The bit-level Python model that simulates the core (`ieee.<op>`: the IEEE operation, correctly
    /// rounded to the instance's widths under its subnormal mode; `reference.<op>`: the real function
    /// correctly rounded, which lies within every accuracy contract here).
    pub model: String,
    pub source: CoreSource,
    pub encoding: Encoding,
    pub timing: Timing,
}

/// What a provider supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IpCapabilities {
    /// The synthesis flows that can use its cores (`dc`, `yosys`).
    pub flows: &'static [&'static str],
    /// The licence its cores (or the code it generates) come under.
    pub licence: &'static str,
    /// The core operation families it may offer (`CoreOp::family`).
    pub ops: &'static [&'static str],
}

/// A provider of float IP cores.
pub trait IpProvider: Named {
    fn capabilities(&self) -> IpCapabilities;

    /// The core for `op` (`CoreOp` syntax) on operands with `exp_width` exponent and `sig_width` stored
    /// significand bits, or `None` outside the provider's documented range.
    fn core(&self, op: &str, exp_width: u32, sig_width: u32) -> Option<CoreOffer>;
}

/// The worst error of a core's finite results in ulps of its own format, when its contract states one
/// relative to the result (correct rounding: 1/2; faithful: below 1).
pub fn ulp_bound(a: &Accuracy) -> Option<f64> {
    match a {
        Accuracy::CorrectlyRounded => Some(0.5),
        Accuracy::Faithful => Some(1.0),
        Accuracy::Ulp { ulp } => Some(*ulp),
        Accuracy::Absolute { .. } | Accuracy::Unbounded => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_ops_round_trip_through_their_names() {
        for s in [
            "add",
            "compare",
            "to_int.rtz.s32",
            "to_int.floor.u9",
            "from_int.s32",
            "convert.e5m10",
            "from_ieee",
            "rsqrt",
        ] {
            let op: CoreOp = s.parse().unwrap();
            assert_eq!(op.to_string(), s);
        }
        for bad in [
            "",
            "to_int.rne",
            "to_int.up.s8",
            "from_int.x8",
            "convert.5m10",
            "add.rne",
            "tanh",
        ] {
            assert!(bad.parse::<CoreOp>().is_err(), "{bad}");
        }
        assert!(CoreOp::Add.exact() && !CoreOp::Exp.exact());
        assert_eq!(ulp_bound(&Accuracy::Ulp { ulp: 2.0 }), Some(2.0));
        assert_eq!(ulp_bound(&Accuracy::Unbounded), None);
    }
}
