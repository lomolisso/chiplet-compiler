//! The datapath's values on symbolic scalars (docs/passes/datapaths.md §6): every vertex of a
//! `Datapath` evaluated in order with `cc_smt::ops`, under one width tuple. Inputs, constants,
//! accumulators and core results are the free variables `spec` names the same way.

use std::collections::BTreeMap;

use cc_smt::ops::{self, Val};
use cc_smt::term::Ctx;

use crate::datapaths::json::{Datapath, Src, Tuple};
use crate::datapaths::spec::{attrs, ty_of_width};

type R<X> = Result<X, String>;

/// The datapath's values by name: its outputs, and `next<i>`, `init<i>` of its accumulator.
pub fn values(c: &mut Ctx, dp: &Datapath, t: &Tuple) -> R<BTreeMap<String, Val>> {
    let mut vs: Vec<Val> = Vec::with_capacity(dp.vertices.len());
    let res_width = |i: u32| -> R<String> {
        if let Some(a) = &dp.accumulator {
            return a
                .entries
                .get(i as usize)
                .map(|e| e.width.clone())
                .ok_or_else(|| format!("result {i}: no accumulator"));
        }
        dp.inputs
            .iter()
            .find(|x| x.name == format!("core_result{i}"))
            .map(|x| x.width.clone())
            .ok_or_else(|| format!("result {i}: no core result input"))
    };
    let src = |c: &mut Ctx, vs: &[Val], s: &Src| -> R<Val> {
        match s {
            Src::Vertex { v, .. } => vs
                .get(*v as usize)
                .copied()
                .ok_or_else(|| format!("vertex {v} read before it is defined")),
            Src::Input { name } => {
                let i = dp
                    .inputs
                    .iter()
                    .find(|x| &x.name == name)
                    .ok_or_else(|| format!("input {name} is not declared"))?;
                ops::input(c, name, ty_of_width(t, &i.width)?).map_err(|e| e.0)
            }
            Src::Acc { index } => {
                let a = dp.accumulator.as_ref().ok_or("an accumulator read without one")?;
                let e = a.entries.get(*index as usize).ok_or("accumulator index")?;
                ops::input(c, &format!("acc{index}"), ty_of_width(t, &e.width)?).map_err(|e| e.0)
            }
            Src::Result { index } => {
                ops::input(c, &format!("res{index}"), ty_of_width(t, &res_width(*index)?)?).map_err(|e| e.0)
            }
        }
    };
    for (i, v) in dp.vertices.iter().enumerate() {
        if v.id as usize != i {
            return Err(format!("vertex {i} has id {}", v.id));
        }
        let ret = ty_of_width(t, &v.width)?;
        let val = if v.op == "const" {
            match (v.attrs.get("column"), v.attrs.get("value").map(String::as_str)) {
                (Some(col), _) => ops::input(c, col, ret).map_err(|e| e.0)?,
                (None, Some("0")) => ops::zero(c, ret).map_err(|e| e.0)?,
                _ => return Err(format!("vertex {i}: a constant without a column or value")),
            }
        } else {
            let mut args = Vec::new();
            for a in &v.args {
                args.push(src(c, &vs, &a.src)?);
            }
            let col = |k: &str| v.attrs.get(k).and_then(|c| t.attrs.get(c)).copied();
            let bits = match (col("exponent_bits"), col("mantissa_bits")) {
                (Some(e), Some(m)) => Some((e, m)),
                _ => None,
            };
            let a = attrs(
                v.attrs.get("comparison_direction").cloned(),
                v.attrs.get("compare_type").cloned(),
                bits,
            )?;
            ops::apply(c, &v.op, &a, &args, ret).map_err(|e| format!("vertex {i} ({}): {}", v.label, e.0))?
        };
        vs.push(val);
    }
    let mut out = BTreeMap::new();
    for o in &dp.outputs {
        out.insert(o.name.clone(), src(c, &vs, &o.src)?);
    }
    if let Some(a) = &dp.accumulator {
        for (i, e) in a.entries.iter().enumerate() {
            out.insert(format!("next{i}"), src(c, &vs, &e.next)?);
            out.insert(format!("init{i}"), src(c, &vs, &e.init)?);
        }
    }
    Ok(out)
}
