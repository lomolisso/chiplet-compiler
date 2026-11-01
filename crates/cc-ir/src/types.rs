//! Element types, dimensions and value types.

use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::sym::SymExpr;

/// Special-value convention of a binary float format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FloatFlavour {
    /// IEEE 754 style: infinities and NaNs.
    Ieee,
    /// `FN`: finite, NaN only, no infinities (OCP OFP8 E4M3, MLIR `f8E4M3FN`).
    FiniteNan,
    /// `FNUZ`: finite, one NaN encoding (negative zero), no infinities (MLIR `f8E4M3FNUZ`, `f8E5M2FNUZ`).
    FiniteNanUnsignedZero,
    /// `FNU`: finite, NaN only, unsigned (MLIR `f8E8M0FNU`).
    FiniteNanUnsigned,
    /// `FN` with exponent bias 11 (MLIR `f8E4M3B11FNUZ`).
    FiniteNanUnsignedZeroBias11,
}

/// An element type. MLIR signless `iN` (N > 1) is a signed two's-complement integer in StableHLO
/// (spec "Types": `si`/`i` are signed, `ui` unsigned); `i1` is the boolean.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Elem {
    Bool,
    Int {
        signed: bool,
        width: u16,
    },
    Float {
        exp: u8,
        mant: u8,
        flavour: FloatFlavour,
    },
    Complex(Box<Elem>),
}

impl Elem {
    /// Parse an MLIR element spelling (`i8`, `ui32`, `bf16`, `f8E4M3FN`, `complex<f32>`, …).
    pub fn parse(s: &str) -> Option<Elem> {
        use FloatFlavour::*;
        if let Some(inner) = s.strip_prefix("complex<").and_then(|r| r.strip_suffix('>')) {
            return Some(Elem::Complex(Box::new(Elem::parse(inner)?)));
        }
        if s == "i1" {
            return Some(Elem::Bool);
        }
        let int =
            |rest: &str, signed: bool| rest.parse::<u16>().ok().map(|width| Elem::Int { signed, width });
        if let Some(r) = s.strip_prefix("ui") {
            return int(r, false);
        }
        if let Some(r) = s.strip_prefix("si") {
            return int(r, true);
        }
        if let Some(r) = s.strip_prefix('i')
            && r.chars().all(|c| c.is_ascii_digit())
        {
            return int(r, true);
        }
        let f = |exp, mant, flavour| Some(Elem::Float { exp, mant, flavour });
        match s {
            "f16" => f(5, 10, Ieee),
            "bf16" => f(8, 7, Ieee),
            "tf32" => f(8, 10, Ieee),
            "f32" => f(8, 23, Ieee),
            "f64" => f(11, 52, Ieee),
            "f8E5M2" => f(5, 2, Ieee),
            "f8E4M3" => f(4, 3, Ieee),
            "f8E3M4" => f(3, 4, Ieee),
            "f8E4M3FN" => f(4, 3, FiniteNan),
            "f8E4M3FNUZ" => f(4, 3, FiniteNanUnsignedZero),
            "f8E5M2FNUZ" => f(5, 2, FiniteNanUnsignedZero),
            "f8E4M3B11FNUZ" => f(4, 3, FiniteNanUnsignedZeroBias11),
            "f8E8M0FNU" => f(8, 0, FiniteNanUnsigned),
            "f6E2M3FN" => f(2, 3, FiniteNan),
            "f6E3M2FN" => f(3, 2, FiniteNan),
            "f4E2M1FN" => f(2, 1, FiniteNan),
            _ => None,
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Elem::Int { .. })
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Elem::Float { .. })
    }

    /// Storage width in bits.
    pub fn bits(&self) -> u32 {
        match self {
            Elem::Bool => 1,
            Elem::Int { width, .. } => u32::from(*width),
            Elem::Float { exp, mant, flavour } => {
                u32::from(*exp) + u32::from(*mant) + u32::from(*flavour != FloatFlavour::FiniteNanUnsigned)
            }
            Elem::Complex(e) => 2 * e.bits(),
        }
    }
}

impl fmt::Display for Elem {
    /// The MLIR spelling.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use FloatFlavour::*;
        match self {
            Elem::Bool => write!(f, "i1"),
            Elem::Int { signed: true, width } => write!(f, "i{width}"),
            Elem::Int { signed: false, width } => write!(f, "ui{width}"),
            Elem::Complex(e) => write!(f, "complex<{e}>"),
            Elem::Float { exp, mant, flavour } => match (exp, mant, flavour) {
                (5, 10, Ieee) => write!(f, "f16"),
                (8, 7, Ieee) => write!(f, "bf16"),
                (8, 10, Ieee) => write!(f, "tf32"),
                (8, 23, Ieee) => write!(f, "f32"),
                (11, 52, Ieee) => write!(f, "f64"),
                (e, m, fl) => {
                    let w = u32::from(*e) + u32::from(*m) + u32::from(*fl != FiniteNanUnsigned);
                    let suffix = match fl {
                        Ieee => "",
                        FiniteNan => "FN",
                        FiniteNanUnsignedZero => "FNUZ",
                        FiniteNanUnsigned => "FNU",
                        FiniteNanUnsignedZeroBias11 => "B11FNUZ",
                    };
                    write!(f, "f{w}E{e}M{m}{suffix}")
                }
            },
        }
    }
}

/// A dimension: a constant or a symbolic expression over the model's size symbols. A `Sym` is never a
/// constant expression (constructors fold it into `Const`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Dim {
    Const(u64),
    Sym(SymExpr),
}

impl Dim {
    /// From an expression; `None` for a negative constant.
    pub fn from_expr(e: SymExpr) -> Option<Dim> {
        match e.as_const() {
            Some(c) => u64::try_from(c).ok().map(Dim::Const),
            None => Some(Dim::Sym(e)),
        }
    }

    pub fn expr(&self) -> SymExpr {
        match self {
            Dim::Const(c) => SymExpr::constant(*c as i64),
            Dim::Sym(e) => e.clone(),
        }
    }

    pub fn as_const(&self) -> Option<u64> {
        match self {
            Dim::Const(c) => Some(*c),
            Dim::Sym(_) => None,
        }
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Const(c) => write!(f, "{c}"),
            Dim::Sym(e) => write!(f, "{e}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct TensorType {
    pub shape: Vec<Dim>,
    pub elem: Elem,
}

impl TensorType {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The number of elements when every dim is constant.
    pub fn num_elements(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(1u64, |a, d| a.checked_mul(d.as_const()?))
    }

    pub fn is_static(&self) -> bool {
        self.shape.iter().all(|d| matches!(d, Dim::Const(_)))
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor<")?;
        for d in &self.shape {
            match d {
                Dim::Const(c) => write!(f, "{c}x")?,
                Dim::Sym(e) => write!(f, "[{e}]x")?,
            }
        }
        write!(f, "{}>", self.elem)
    }
}

/// A value type. Only tensors occur in the exported modules; anything else is kept by its printed form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Type {
    Tensor(TensorType),
    Other(String),
}

impl Type {
    pub fn tensor(&self) -> Option<&TensorType> {
        match self {
            Type::Tensor(t) => Some(t),
            Type::Other(_) => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Tensor(t) => write!(f, "{t}"),
            Type::Other(s) => write!(f, "{s}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elem_spellings_round_trip() {
        for s in [
            "i1",
            "i8",
            "i32",
            "ui8",
            "f16",
            "bf16",
            "f32",
            "f64",
            "f8E4M3FN",
            "f8E5M2",
            "f8E4M3FNUZ",
            "f8E5M2FNUZ",
            "complex<f32>",
            "complex<f64>",
        ] {
            let e = Elem::parse(s).unwrap_or_else(|| panic!("{s}"));
            assert_eq!(e.to_string(), s);
        }
        assert_eq!(Elem::parse("f8E4M3FN").map(|e| e.bits()), Some(8));
        assert_eq!(Elem::parse("bf16").map(|e| e.bits()), Some(16));
        assert!(Elem::parse("index").is_none());
    }
}
