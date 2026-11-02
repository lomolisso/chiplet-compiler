//! Dependences under ω (docs/passes/iteration_space.md §4.2). A contraction's dims are grouped by the
//! core operands that index them (batch: all three; M: lhs and result; N: rhs and result; K: lhs and
//! rhs). K is linearized in ω's order: ω's `seq` visits the reduced elements in ascending lexicographic
//! order of its axes (outermost first), so the accumulator chain is the successor in that order, which is
//! the unit vector of the linearized K. M and N are linearized in result order. Localizing the reuse of
//! each input (Rajopadhye and Fujimoto 1990; Quinton 1984) gives the uniform recurrence
//! `a(m, n, k) = a(m, n − 1, k)`, `b(m, n, k) = b(m − 1, n, k)`, `c(m, n, k) = c(m, n, k − 1) + a·b`
//! over the nonempty groups, with constant dependence vectors (Karp, Miller and Winograd 1967). Batch
//! dims carry no dependence and are outer loops.

use std::collections::BTreeMap;

use cc_ir::ops::iter::Tag;

use crate::iteration_space::json::{
    DEPENDENCES_SCHEMA, Dependences, Group, Operand, SkelClass, Skeleton, Vector, VectorKind,
};
use crate::normalize::json::Order;

pub fn operand<'a>(s: &'a Skeleton, name: &str) -> Option<&'a Operand> {
    s.operands.iter().find(|o| o.core && o.name == name)
}

/// The reduction dims in ω's order (dims ω does not name keep their index order, after it).
fn omega_order(s: &Skeleton, red: &[u32]) -> Vec<u32> {
    let mut out: Vec<u32> = s
        .omega
        .as_ref()
        .map(|o| o.dims.iter().copied().filter(|d| red.contains(d)).collect())
        .unwrap_or_default();
    for &d in red {
        if !out.contains(&d) {
            out.push(d);
        }
    }
    out
}

fn tree(s: &Skeleton) -> Option<u32> {
    match s.omega.as_ref().map(|o| &o.order) {
        Some(Order::Tree { fan_in }) => Some(*fan_in),
        _ => None,
    }
}

pub fn dependences(s: &Skeleton) -> Dependences {
    let mut d = Dependences {
        schema: DEPENDENCES_SCHEMA.into(),
        id: s.id.clone(),
        uniform: true,
        reason: String::new(),
        groups: BTreeMap::new(),
        other: Vec::new(),
        space: Vec::new(),
        vectors: Vec::new(),
    };
    let par: Vec<u32> = (0..s.dims.len() as u32)
        .filter(|&i| s.dims[i as usize].tag == Tag::Par)
        .collect();
    let red: Vec<u32> = (0..s.dims.len() as u32)
        .filter(|&i| s.dims[i as usize].tag == Tag::Red)
        .collect();
    match s.class {
        SkelClass::Map => {
            d.groups.insert(Group::Par, par);
            d.reason = "map: no reduction dims, so no dependences".into();
        }
        SkelClass::Reduce if !s.affine => {
            d.uniform = false;
            d.reason = "data-dependent target index (scatter): not affine, dependences unknown".into();
        }
        SkelClass::Reduce => {
            d.groups.insert(Group::Par, par.clone());
            d.groups.insert(Group::Red, omega_order(s, &red));
            if let Some(b) = tree(s) {
                d.uniform = false;
                d.reason = format!("ω is tree:{b}, not one accumulator chain");
                return d;
            }
            d.space = if par.is_empty() {
                vec![Group::Red]
            } else {
                vec![Group::Par, Group::Red]
            };
            let mut v = vec![0; d.space.len()];
            *v.last_mut().unwrap_or(&mut 0) = 1;
            d.vectors.push(Vector {
                operand: "result0".into(),
                kind: VectorKind::Accumulate,
                v,
            });
            d.reason = "uniform: one accumulator chain along the reduced dims in ω's order".into();
        }
        SkelClass::Contraction => {
            let (Some(a), Some(b), Some(c)) = (operand(s, "lhs"), operand(s, "rhs"), operand(s, "result"))
            else {
                d.uniform = false;
                d.reason = "a contraction without lhs, rhs and result".into();
                return d;
            };
            let mut groups: BTreeMap<Group, Vec<u32>> = BTreeMap::new();
            for i in 0..s.dims.len() as u32 {
                let inc = (
                    !a.reuse.contains(&i),
                    !b.reuse.contains(&i),
                    !c.reuse.contains(&i),
                );
                let tag = s.dims[i as usize].tag;
                let g = match (inc, tag) {
                    ((true, true, true), Tag::Par) => Some(Group::Batch),
                    ((true, false, true), Tag::Par) => Some(Group::M),
                    ((false, true, true), Tag::Par) => Some(Group::N),
                    ((true, true, false), Tag::Red) => Some(Group::K),
                    _ => None,
                };
                match g {
                    Some(g) => groups.entry(g).or_default().push(i),
                    None => d.other.push(i),
                }
            }
            if let Some(k) = groups.get_mut(&Group::K) {
                *k = omega_order(s, k);
            }
            d.groups = groups;
            if let Some(&o) = d.other.first() {
                d.uniform = false;
                d.reason = format!(
                    "dim {o} ({}) is indexed by neither exactly two nor all three core operands",
                    s.dims[o as usize].role
                );
                return d;
            }
            if !d.groups.contains_key(&Group::K) {
                d.uniform = false;
                d.reason = "no reduction dim indexes both inputs".into();
                return d;
            }
            if let Some(f) = tree(s) {
                d.uniform = false;
                d.reason = format!("ω is tree:{f}, not one accumulator chain");
                return d;
            }
            d.space = [Group::M, Group::N, Group::K]
                .into_iter()
                .filter(|g| d.groups.contains_key(g))
                .collect();
            let unit = |g: Group| -> Vec<i64> { d.space.iter().map(|&x| i64::from(x == g)).collect() };
            if d.groups.contains_key(&Group::N) {
                d.vectors.push(Vector {
                    operand: "lhs".into(),
                    kind: VectorKind::Propagate,
                    v: unit(Group::N),
                });
            }
            if d.groups.contains_key(&Group::M) {
                d.vectors.push(Vector {
                    operand: "rhs".into(),
                    kind: VectorKind::Propagate,
                    v: unit(Group::M),
                });
            }
            d.vectors.push(Vector {
                operand: "result".into(),
                kind: VectorKind::Accumulate,
                v: unit(Group::K),
            });
            let names: Vec<&str> = d.space.iter().map(|g| g.name()).collect();
            d.reason = format!(
                "uniform over ({}): lhs reused along n, rhs along m, accumulation along k in ω's order",
                names.join(", ")
            );
        }
    }
    d
}
