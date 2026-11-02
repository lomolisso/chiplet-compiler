//! Evaluation of a skeleton's extents and factors at one instance row (the class's instance table).

use std::collections::HashMap;

use crate::iteration_space::json::{Extent, Factor, Skeleton};
use crate::kernel_classes::json::Row;

/// An instance table with its value columns by name.
#[derive(Debug)]
pub struct Table<'a> {
    pub rows: &'a [Row],
    pub cols: HashMap<String, usize>,
}

impl<'a> Table<'a> {
    pub fn new(rows: &'a [Row], columns: &[String]) -> Self {
        Self {
            rows,
            cols: columns.iter().enumerate().map(|(i, c)| (c.clone(), i)).collect(),
        }
    }

    fn column(&self, r: &Row, column: &str, index: u32) -> Option<i64> {
        let v = r.values.get(*self.cols.get(column)?)?.as_ref()?;
        v.as_ints()
            .and_then(|x| x.get(index as usize).copied())
            .or_else(|| v.as_i64())
    }

    /// A factor's value at a row (`None`: known only at run time).
    pub fn factor(&self, s: &Skeleton, f: &Factor, r: &Row) -> Option<i64> {
        match f {
            Factor::Column { column, index } => self.column(r, column, *index),
            Factor::Runtime { .. } => None,
            Factor::Extent { dim } => self.extent(s, *dim, r).and_then(|v| i64::try_from(v).ok()),
        }
    }

    /// A dim's extent at a row.
    pub fn extent(&self, s: &Skeleton, dim: u32, r: &Row) -> Option<u64> {
        match &s.dims.get(dim as usize)?.extent {
            Extent::Class { class } => r.extents.get(*class as usize).copied(),
            Extent::Const { value } => u64::try_from(*value).ok(),
            Extent::Column { column, index } => {
                self.column(r, column, *index).and_then(|v| u64::try_from(v).ok())
            }
            Extent::Runtime { .. } => None,
            Extent::Quot { class, by } => {
                let e = *r.extents.get(*class as usize)?;
                let b = u64::try_from(self.factor(s, by, r)?).ok()?;
                (b > 0).then(|| e / b)
            }
        }
    }

    /// The product of some dims' extents.
    pub fn group_extent(&self, s: &Skeleton, dims: &[u32], r: &Row) -> Option<u64> {
        dims.iter()
            .try_fold(1u64, |a, &d| a.checked_mul(self.extent(s, d, r)?))
    }
}

/// The bit width of an element class at a row (0 when unknown).
pub fn row_width(r: &Row, elem: u32) -> u32 {
    r.elems
        .get(elem as usize)
        .and_then(|e| cc_ir::Elem::parse(e))
        .map_or(0, |e| e.bits())
}
