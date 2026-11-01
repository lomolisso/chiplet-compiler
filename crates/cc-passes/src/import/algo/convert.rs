//! `graph.json` attributes and types → `cc_ir` ones.

use cc_ir::attrs::{ConvDims, DotDims, GatherDims, ScatterDims};
use cc_ir::{Attr, Attrs, Dim, Elem, Scalar, TensorType, Type};

use crate::export::json;

pub fn elem(s: &str) -> Result<Elem, String> {
    Elem::parse(s).ok_or_else(|| format!("unknown element type `{s}`"))
}

/// A tensor type as declared (`None` = `?`).
pub fn declared(t: &json::Type) -> Result<(Vec<Option<u64>>, Elem), String> {
    match t {
        json::Type::Tensor { shape, elem: e } => Ok((shape.clone(), elem(e)?)),
        other => Err(format!("non-tensor value type {other:?}")),
    }
}

/// A type with every dim known (attribute types); `?` dims or other kinds keep their printed form.
pub fn static_type(t: &json::Type) -> Type {
    match t {
        json::Type::Tensor { shape, elem: e } if shape.iter().all(Option::is_some) => match Elem::parse(e) {
            Some(elem) => Type::Tensor(TensorType {
                shape: shape.iter().flatten().map(|&d| Dim::Const(d)).collect(),
                elem,
            }),
            None => Type::Other(format!("{t:?}")),
        },
        _ => Type::Other(format!("{t:?}")),
    }
}

fn scalar(v: &serde_json::Value) -> Scalar {
    match v {
        serde_json::Value::Bool(b) => Scalar::Bool(*b),
        serde_json::Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => Scalar::Int(i128::from(i)),
            (_, Some(u), _) => Scalar::Int(i128::from(u)),
            (_, _, Some(f)) => Scalar::Float(f),
            _ => Scalar::Text(n.to_string()),
        },
        serde_json::Value::String(s) => Scalar::Text(s.clone()),
        serde_json::Value::Array(a) if a.len() == 2 => {
            Scalar::Complex(Box::new(scalar(&a[0])), Box::new(scalar(&a[1])))
        }
        other => Scalar::Text(other.to_string()),
    }
}

pub fn attr(a: &json::Attr) -> Attr {
    use json::Attr as J;
    match a {
        J::Int { value, ty } => Attr::Int {
            value: match value {
                json::IntValue::Signed(v) => i128::from(*v),
                json::IntValue::Unsigned(v) => i128::from(*v),
            },
            ty: ty.clone(),
        },
        J::Float { value, ty } => Attr::Float {
            value: match value {
                json::FloatValue::Num(f) => Scalar::Float(*f),
                json::FloatValue::Special(s) => Scalar::Text(s.clone()),
            },
            ty: ty.clone(),
        },
        J::Bool { value } => Attr::Bool(*value),
        J::Str { value } => Attr::Str(value.clone()),
        J::Symbol { value } => Attr::Symbol(value.clone()),
        J::Ints { value } => Attr::Ints(value.clone()),
        J::Bools { value } => Attr::Bools(value.clone()),
        J::Dense {
            ty,
            splat,
            values,
            digest,
        } => Attr::Dense {
            ty: static_type(ty),
            splat: *splat,
            values: values.as_ref().map(|v| v.iter().map(scalar).collect()),
            digest: digest.clone(),
        },
        J::Array { value } => Attr::Array(value.iter().map(attr).collect()),
        J::Dict { value } => Attr::Dict(attrs(value)),
        J::Type { value } => Attr::Type(static_type(value)),
        J::Unit => Attr::Unit,
        J::Enum { enum_kind, value } => Attr::Enum {
            kind: enum_kind.clone(),
            value: value.clone(),
        },
        J::DotDimensionNumbers {
            lhs_batching_dimensions,
            rhs_batching_dimensions,
            lhs_contracting_dimensions,
            rhs_contracting_dimensions,
        } => Attr::DotDims(DotDims {
            lhs_batching: lhs_batching_dimensions.clone(),
            rhs_batching: rhs_batching_dimensions.clone(),
            lhs_contracting: lhs_contracting_dimensions.clone(),
            rhs_contracting: rhs_contracting_dimensions.clone(),
        }),
        J::ConvDimensionNumbers {
            input_batch_dimension,
            input_feature_dimension,
            input_spatial_dimensions,
            kernel_input_feature_dimension,
            kernel_output_feature_dimension,
            kernel_spatial_dimensions,
            output_batch_dimension,
            output_feature_dimension,
            output_spatial_dimensions,
        } => Attr::ConvDims(ConvDims {
            input_batch: *input_batch_dimension,
            input_feature: *input_feature_dimension,
            input_spatial: input_spatial_dimensions.clone(),
            kernel_input_feature: *kernel_input_feature_dimension,
            kernel_output_feature: *kernel_output_feature_dimension,
            kernel_spatial: kernel_spatial_dimensions.clone(),
            output_batch: *output_batch_dimension,
            output_feature: *output_feature_dimension,
            output_spatial: output_spatial_dimensions.clone(),
        }),
        J::GatherDimensionNumbers {
            offset_dims,
            collapsed_slice_dims,
            operand_batching_dims,
            start_indices_batching_dims,
            start_index_map,
            index_vector_dim,
        } => Attr::GatherDims(GatherDims {
            offset_dims: offset_dims.clone(),
            collapsed_slice_dims: collapsed_slice_dims.clone(),
            operand_batching_dims: operand_batching_dims.clone(),
            start_indices_batching_dims: start_indices_batching_dims.clone(),
            start_index_map: start_index_map.clone(),
            index_vector_dim: *index_vector_dim,
        }),
        J::ScatterDimensionNumbers {
            update_window_dims,
            inserted_window_dims,
            input_batching_dims,
            scatter_indices_batching_dims,
            scattered_dims_to_operand_dims,
            index_vector_dim,
        } => Attr::ScatterDims(ScatterDims {
            update_window_dims: update_window_dims.clone(),
            inserted_window_dims: inserted_window_dims.clone(),
            input_batching_dims: input_batching_dims.clone(),
            scatter_indices_batching_dims: scatter_indices_batching_dims.clone(),
            scattered_dims_to_operand_dims: scattered_dims_to_operand_dims.clone(),
            index_vector_dim: *index_vector_dim,
        }),
        J::Opaque { value } => Attr::Opaque(value.clone()),
    }
}

pub fn attrs(a: &json::Attrs) -> Attrs {
    a.iter().map(|(k, v)| (k.clone(), attr(v))).collect()
}
