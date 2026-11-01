//! Typed op attributes: the StableHLO attribute kinds that the exported modules carry, with accessors
//! for the structured ones (dimension numbers, windows, padding, comparison direction).

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::types::Type;

/// One element of a dense constant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Scalar {
    Int(i128),
    Float(f64),
    Bool(bool),
    /// Non-finite floats (`inf`, `-inf`, `nan`) and anything without a numeric form.
    Text(String),
    /// A complex number `(re, im)`.
    Complex(Box<Scalar>, Box<Scalar>),
}

impl Scalar {
    pub fn as_int(&self) -> Option<i128> {
        match self {
            Scalar::Int(v) => Some(*v),
            Scalar::Bool(b) => Some(i128::from(*b)),
            _ => None,
        }
    }
}

/// `stablehlo.dot_general` dimension numbers (spec `#dot_general`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DotDims {
    pub lhs_batching: Vec<i64>,
    pub rhs_batching: Vec<i64>,
    pub lhs_contracting: Vec<i64>,
    pub rhs_contracting: Vec<i64>,
}

/// `stablehlo.convolution` dimension numbers (spec `#convolution`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ConvDims {
    pub input_batch: i64,
    pub input_feature: i64,
    pub input_spatial: Vec<i64>,
    pub kernel_input_feature: i64,
    pub kernel_output_feature: i64,
    pub kernel_spatial: Vec<i64>,
    pub output_batch: i64,
    pub output_feature: i64,
    pub output_spatial: Vec<i64>,
}

/// `stablehlo.gather` dimension numbers (spec `#gather`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GatherDims {
    pub offset_dims: Vec<i64>,
    pub collapsed_slice_dims: Vec<i64>,
    pub operand_batching_dims: Vec<i64>,
    pub start_indices_batching_dims: Vec<i64>,
    pub start_index_map: Vec<i64>,
    pub index_vector_dim: i64,
}

/// `stablehlo.scatter` dimension numbers (spec `#scatter`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ScatterDims {
    pub update_window_dims: Vec<i64>,
    pub inserted_window_dims: Vec<i64>,
    pub input_batching_dims: Vec<i64>,
    pub scatter_indices_batching_dims: Vec<i64>,
    pub scattered_dims_to_operand_dims: Vec<i64>,
    pub index_vector_dim: i64,
}

/// A typed attribute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Attr {
    Int {
        value: i128,
        ty: String,
    },
    Float {
        value: Scalar,
        ty: String,
    },
    Bool(bool),
    Str(String),
    Symbol(String),
    Ints(Vec<i64>),
    Bools(Vec<bool>),
    /// Dense elements: the values when the export kept them (at most 1024), else a content digest.
    Dense {
        ty: Type,
        splat: bool,
        values: Option<Vec<Scalar>>,
        digest: Option<String>,
    },
    Array(Vec<Attr>),
    Dict(BTreeMap<String, Attr>),
    Type(Type),
    Unit,
    /// A StableHLO enum (`comparison_direction`, `comparison_type`, `fft_type`, `precision`, …).
    Enum {
        kind: String,
        value: String,
    },
    DotDims(DotDims),
    ConvDims(ConvDims),
    GatherDims(GatherDims),
    ScatterDims(ScatterDims),
    /// No structured form: the printed attribute.
    Opaque(String),
}

pub type Attrs = BTreeMap<String, Attr>;

impl Attr {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Attr::Int { value, .. } => i64::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Attr::Str(s) | Attr::Symbol(s) => Some(s),
            _ => None,
        }
    }

    /// An integer list: `array<i64: …>` or a dense integer tensor (row-major; a splat is expanded).
    pub fn as_ints(&self) -> Option<Vec<i64>> {
        match self {
            Attr::Ints(v) => Some(v.clone()),
            Attr::Dense {
                ty,
                splat,
                values: Some(vals),
                ..
            } => {
                let ints: Option<Vec<i64>> = vals
                    .iter()
                    .map(|s| s.as_int().and_then(|v| i64::try_from(v).ok()))
                    .collect();
                let ints = ints?;
                let n = ty.tensor()?.num_elements()? as usize;
                if *splat {
                    Some(vec![*ints.first()?; n])
                } else {
                    (ints.len() == n).then_some(ints)
                }
            }
            _ => None,
        }
    }

    pub fn as_enum(&self) -> Option<&str> {
        match self {
            Attr::Enum { value, .. } => Some(value),
            _ => None,
        }
    }
}

/// Typed accessors on an op's attribute map.
pub trait AttrsExt {
    fn int(&self, k: &str) -> Option<i64>;
    fn ints(&self, k: &str) -> Option<Vec<i64>>;
    fn enum_value(&self, k: &str) -> Option<&str>;
    fn string(&self, k: &str) -> Option<&str>;
    /// `[[lo, hi], …]` from a dense `Nx2` padding attribute.
    fn padding(&self, k: &str) -> Option<Vec<(i64, i64)>>;
}

impl AttrsExt for Attrs {
    fn int(&self, k: &str) -> Option<i64> {
        self.get(k)?.as_i64()
    }
    fn ints(&self, k: &str) -> Option<Vec<i64>> {
        self.get(k)?.as_ints()
    }
    fn enum_value(&self, k: &str) -> Option<&str> {
        self.get(k)?.as_enum()
    }
    fn string(&self, k: &str) -> Option<&str> {
        self.get(k)?.as_str()
    }
    fn padding(&self, k: &str) -> Option<Vec<(i64, i64)>> {
        let v = self.ints(k)?;
        v.len()
            .is_multiple_of(2)
            .then(|| v.chunks(2).map(|c| (c[0], c[1])).collect())
    }
}
