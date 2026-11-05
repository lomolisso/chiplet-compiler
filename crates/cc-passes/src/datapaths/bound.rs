//! The error bound of a float sum whose order binds (design §6.4, docs/framework/smt.md, "Reduction
//! order"): the hardware computes ω; another order of the same reduction (the model's, which XLA leaves
//! open) may round differently, by at most
//!
//! `|hw - other| <= (gamma(d) + gamma(n)) · S + n · 2^emin · (2 + gamma(d) + gamma(n))`
//!
//! with `n` the reduced elements (the init value is one more term), `S = |init| + Σ|x_i|` over the values
//! the operators read (DAZ applied), `u = 2^-p` the unit roundoff of the accumulator's format,
//! `gamma(k) = k·u / (1 - k·u)` and `d` the additions on the longest path of ω's schedule (`n` for `seq`,
//! `(b - 1)·ceil(log_b n) + 1` for `tree:<b>`). Any order has at most `n` additions on a path, so
//! `|other - exact| <= gamma(n) · S` and `|hw - exact| <= gamma(d) · S` (Higham, *Accuracy and Stability
//! of Numerical Algorithms*, 2nd ed., SIAM 2002, §4.2, eq. 4.4). A flushing format adds an absolute
//! error below `2^emin` per flushed addition, carried through at most `1 + gamma` of later roundings
//! (Higham §2.1, the underflow model); a gradual format's underflowing sums are exact (Hauser, ACM TOPLAS
//! 18(2), 1996), so that term is absent. Assumptions: no partial sum overflows, `n·u < 1`.

use cc_smt::ops::Ty;

use crate::datapaths::json::{Datapath, Src, Tuple};
use crate::datapaths::spec::ty_of_width;
use crate::iteration_space::json::DimOrder;
use crate::normalize::json::Order;

/// The bound of every accumulator entry at tuple `t` when the hardware computes `hw`: a float sum gets
/// the formula, any other combiner the reason it has none.
pub fn bounds(dp: &Datapath, t: &Tuple, hw: &DimOrder) -> Result<Vec<String>, String> {
    let a = dp.accumulator.as_ref().ok_or("no accumulator")?;
    let n = if hw.dims.is_empty() {
        "1".to_string()
    } else {
        hw.dims
            .iter()
            .map(|d| format!("|d{d}|"))
            .collect::<Vec<_>>()
            .join(" * ")
    };
    let (order, d) = match hw.order {
        Order::Seq => ("seq".to_string(), "n".to_string()),
        Order::Tree { fan_in } => (
            format!("tree:{fan_in}"),
            format!("{} * ceil(log_{fan_in}(n)) + 1", fan_in - 1),
        ),
    };
    let mut out = Vec::new();
    for (i, e) in a.entries.iter().enumerate() {
        let ty = ty_of_width(t, &e.width)?;
        let sum = match (&e.next, ty) {
            (Src::Vertex { v, .. }, Ty::Float(f)) => dp
                .vertices
                .get(*v as usize)
                .filter(|x| {
                    x.op == "add"
                        && x.args.len() == 2
                        && x.args.iter().any(|a| a.src == Src::Acc { index: i as u32 })
                })
                .map(|_| f),
            _ => None,
        };
        let Some(f) = sum else {
            out.push(format!(
                "acc{i}: no bound (not a float sum: {})",
                match &e.next {
                    Src::Vertex { v, .. } => dp
                        .vertices
                        .get(*v as usize)
                        .map(|x| x.label.clone())
                        .unwrap_or_default(),
                    s => format!("{s:?}"),
                }
            ));
            continue;
        };
        let p = f.m + 1;
        let flush = if f.flushes() {
            format!(" + n * 2^{} * (2 + gamma(d) + gamma(n))", f.emin())
        } else {
            String::new()
        };
        out.push(format!(
            "acc{i} ({}, {order}): |hw - other order| <= (gamma(d) + gamma(n)) * S{flush}; \
             S = |init| + sum |x_i|, gamma(k) = k * 2^-{p} / (1 - k * 2^-{p}), d = {d}, n = {n}; \
             no overflow, n * 2^-{p} < 1",
            f.name()
        ));
    }
    Ok(out)
}
