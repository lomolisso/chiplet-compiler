//! The accuracy contract of a third-party float core and one instance of it (design §4.12). A core is
//! trusted to meet its contract; what is built around it (glue, compositions) is proven or measured
//! against the contract, never against the core itself.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// How accurate a core's finite results are, as its documentation states it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "class", rename_all = "snake_case")]
pub enum Accuracy {
    /// Correctly rounded: the exact value rounded to nearest, ties to even.
    CorrectlyRounded,
    /// One of the two representable neighbours of the exact value.
    Faithful,
    /// Within `ulp` units in the last place of the exact value.
    Ulp { ulp: f64 },
    /// `|result - exact| <= bound`.
    Absolute { bound: f64 },
    /// No bound documented.
    Unbounded,
}

/// What a core does with subnormal operands and results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Subnormals {
    /// Gradual underflow, as IEEE 754.
    Ieee,
    /// Subnormal operands read as zero and subnormal results flushed to zero, sign kept.
    Flush,
}

/// The documented behaviour of a core.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contract {
    pub accuracy: Accuracy,
    /// NaN, infinities and signed zeros as IEEE 754 and C11 Annex F specify them for the op.
    pub ieee_specials: bool,
    pub subnormals: Subnormals,
}

/// One instance of a core as Verilog writes it: `module #(params) name (ports)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreInstance {
    pub module: String,
    /// Parameter name -> value, as Verilog literals.
    pub params: BTreeMap<String, String>,
    /// Port name -> what drives or reads it: an operand `a0`, `a1`, ..., the result `z`, or a Verilog
    /// constant such as `3'b000`; an empty string leaves an output unconnected.
    pub ports: BTreeMap<String, String>,
    /// Exponent width and significand width (without the hidden bit) of the core's operands and result.
    pub exp_width: u32,
    pub sig_width: u32,
}
