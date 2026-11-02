//! Width tuples (docs/passes/datapaths.md §4.2): the distinct element types of a class over its instance
//! table, with the types of its folded constants and the attributes that change an op's semantics.
//! Constant values are not part of a tuple: a proof treats them as free variables, so it holds for
//! every value.

use std::collections::BTreeMap;

use cc_ir::{Attr, Type};

use crate::datapaths::json::Tuple;
use crate::kernel_classes::json::{ClassFile, InstancesTable};

/// Columns whose value is part of the semantics (`reduce_precision`'s bit counts).
fn structural(col: &str) -> bool {
    col.ends_with(".exponent_bits") || col.ends_with(".mantissa_bits")
}

fn is_const(col: &str) -> bool {
    col.rsplit_once('.').is_some_and(|(_, k)| k.starts_with("splat"))
}

/// The element type of a folded constant's value.
pub fn value_type(a: &Attr) -> Option<String> {
    match a {
        Attr::Dense {
            ty: Type::Tensor(t), ..
        } => Some(t.elem.to_string()),
        Attr::Int { ty, .. } | Attr::Float { ty, .. } => Some(ty.clone()),
        Attr::Bool(_) => Some("i1".into()),
        _ => None,
    }
}

pub fn tuples(class: &ClassFile, table: &InstancesTable) -> Result<Vec<Tuple>, String> {
    type Key = (Vec<String>, BTreeMap<String, String>, BTreeMap<String, i64>);
    let mut seen: BTreeMap<Key, u64> = BTreeMap::new();
    for (ri, row) in table.rows.iter().enumerate() {
        let mut consts = BTreeMap::new();
        let mut attrs = BTreeMap::new();
        for (col, v) in class.params.values.iter().zip(&row.values) {
            let Some(v) = v else { continue };
            if is_const(col) {
                let t = value_type(v).ok_or_else(|| format!("row {ri}: {col} has no element type"))?;
                consts.insert(col.clone(), t);
            } else if structural(col) {
                let x = match v {
                    Attr::Int { value, .. } => i64::try_from(*value).ok(),
                    _ => None,
                }
                .ok_or_else(|| format!("row {ri}: {col} is not an integer"))?;
                attrs.insert(col.clone(), x);
            }
        }
        *seen.entry((row.elems.clone(), consts, attrs)).or_default() += 1;
    }
    Ok(seen
        .into_iter()
        .enumerate()
        .map(|(k, ((elems, consts, attrs), rows))| Tuple {
            name: format!("t{k}"),
            elems,
            consts,
            attrs,
            rows,
        })
        .collect())
}
