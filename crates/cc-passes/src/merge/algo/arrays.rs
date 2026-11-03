//! Array units for template classes (docs/passes/merge.md §4.7). An array is the host class's selected
//! space-time mapping with its dims as parameters and one processing-element type `(lhs, rhs, acc)`. It
//! serves the (class, tuple) pairs with the same dataflow whose skeleton embeds into the host's (⊑) and
//! whose types the element runs: natively, with narrower integer operands extended (`widen`), or with
//! one wider integer operand split into pieces multiplied one per pass (`compose`, the TPU and Bit Fusion
//! scheme: low pieces unsigned, the top piece keeps the sign).
//!
//! Grouping, per dataflow: the element type is the tuple with the most work among the pairs left; the
//! pairs it can run join it; within a group, hosts are chosen greedily by the work that embeds into them.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::library::Lib;
use crate::merge::json::{ARRAY_SCHEMA, ArrayCover, ArrayFile, ArrayMode, ArrayTuple, Price};

/// A template class as the grouping sees it.
#[derive(Debug, Clone)]
pub struct TemplateIn {
    pub class: String,
    pub dataflow: String,
    pub workload: u64,
    pub candidate: u32,
    pub dims: Vec<String>,
    /// `(tuple, [lhs, rhs, acc], rows)`.
    pub tuples: Vec<(String, Vec<String>, u64)>,
}

fn int(t: &str) -> Option<(bool, u32)> {
    if t == "i1" {
        return None;
    }
    if let Some(w) = t.strip_prefix("ui") {
        return w.parse().ok().map(|w| (false, w));
    }
    t.strip_prefix('i')
        .and_then(|w| w.parse().ok())
        .map(|w| (true, w))
}

/// An operand of type `x` on an element operand of type `p`, extended: `None` if it does not fit.
fn fits(x: &str, p: &str) -> bool {
    match (int(x), int(p)) {
        (Some((xs, xw)), Some((ps, pw))) => (xs == ps && xw <= pw) || (!xs && ps && xw < pw),
        _ => x == p,
    }
}

/// Pieces of an operand of type `x` on an element operand of type `p` (low first), if it composes.
fn pieces(x: &str, p: &str) -> Option<Vec<u32>> {
    let ((xs, xw), (ps, pw)) = (int(x)?, int(p)?);
    if xs && !ps {
        return None;
    }
    let low = if ps { pw - 1 } else { pw };
    let top = if xs { pw } else { low };
    let mut v = Vec::new();
    let mut left = xw;
    while left > top {
        v.push(low);
        left -= low;
    }
    v.push(left);
    (v.len() > 1).then_some(v)
}

/// How the element `pe` runs a class tuple, if it can.
pub fn mode(pe: &[String], t: &[String]) -> Option<(ArrayMode, u32, Vec<u32>, Option<String>)> {
    if pe.len() != 3 || t.len() != 3 || pe[2] != t[2] {
        return None;
    }
    if pe == t {
        return Some((ArrayMode::Native, 1, Vec::new(), None));
    }
    let (fl, fr) = (fits(&t[0], &pe[0]), fits(&t[1], &pe[1]));
    if fl && fr {
        return Some((ArrayMode::Widen, 1, Vec::new(), None));
    }
    if fr && let Some(p) = pieces(&t[0], &pe[0]) {
        return Some((ArrayMode::Compose, p.len() as u32, p, Some("lhs".into())));
    }
    if fl && let Some(p) = pieces(&t[1], &pe[1]) {
        return Some((ArrayMode::Compose, p.len() as u32, p, Some("rhs".into())));
    }
    None
}

/// The price of one element: the two converts, the multiply and the add at the accumulator type, and
/// the accumulator register.
pub fn pe_price(lib: &Lib, pe: &[String]) -> Price {
    use crate::merge::json::BlockKind;
    let none = BTreeMap::new();
    let acc = pe[2].clone();
    let parts = [
        lib.block(
            BlockKind::Op,
            "convert",
            &none,
            &[pe[0].clone(), acc.clone()],
            false,
        ),
        lib.block(
            BlockKind::Op,
            "convert",
            &none,
            &[pe[1].clone(), acc.clone()],
            false,
        ),
        lib.block(
            BlockKind::Op,
            "multiply",
            &none,
            &[acc.clone(), acc.clone(), acc.clone()],
            false,
        ),
        lib.block(
            BlockKind::Op,
            "add",
            &none,
            &[acc.clone(), acc.clone(), acc.clone()],
            false,
        ),
        lib.block(BlockKind::Acc, "acc", &none, std::slice::from_ref(&acc), false),
    ];
    let mut p = Price {
        keys: Vec::new(),
        area_um2: 0.0,
        delay_ps: 0.0,
        estimated: false,
    };
    for q in parts {
        p.keys.extend(q.keys);
        p.area_um2 += q.area_um2;
        p.delay_ps += q.delay_ps;
        p.estimated |= q.estimated;
    }
    p
}

pub fn arrays(classes: &[TemplateIn], embeds: &HashSet<(String, String)>, lib: &Lib) -> Vec<ArrayFile> {
    let by_class: BTreeMap<&str, &TemplateIn> = classes.iter().map(|c| (c.class.as_str(), c)).collect();
    let mut out = Vec::new();
    let flows: BTreeSet<&str> = classes.iter().map(|c| c.dataflow.as_str()).collect();
    for flow in flows {
        // (class, tuple index) pairs left, with their work.
        let mut left: Vec<(&TemplateIn, usize, f64)> = Vec::new();
        for c in classes.iter().filter(|c| c.dataflow == flow) {
            let rows: u64 = c.tuples.iter().map(|t| t.2).sum::<u64>().max(1);
            for (k, t) in c.tuples.iter().enumerate() {
                left.push((c, k, c.workload as f64 * t.2 as f64 / rows as f64));
            }
        }
        while !left.is_empty() {
            let mut work: BTreeMap<&Vec<String>, f64> = BTreeMap::new();
            for (c, k, w) in &left {
                *work.entry(&c.tuples[*k].1).or_default() += w;
            }
            let pe = work
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(t, _)| (*t).clone())
                .unwrap_or_default();
            let (mut group, rest): (Vec<_>, Vec<_>) = left
                .into_iter()
                .partition(|(c, k, _)| mode(&pe, &c.tuples[*k].1).is_some());
            left = rest;
            // Hosts, greedily by the work that embeds.
            while !group.is_empty() {
                let members: BTreeSet<&str> = group.iter().map(|(c, _, _)| c.class.as_str()).collect();
                let covers = |h: &str, g: &str| g == h || embeds.contains(&(g.to_string(), h.to_string()));
                let host = members
                    .iter()
                    .map(|&h| {
                        let w: f64 = group
                            .iter()
                            .filter(|(c, _, _)| covers(h, &c.class))
                            .map(|x| x.2)
                            .sum();
                        (h, w)
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(a.0)))
                    .map(|x| x.0.to_string())
                    .unwrap_or_default();
                let (mine, others): (Vec<_>, Vec<_>) =
                    group.into_iter().partition(|(c, _, _)| covers(&host, &c.class));
                group = others;
                let mut per: BTreeMap<&str, Vec<ArrayTuple>> = BTreeMap::new();
                for (c, k, _) in &mine {
                    let (name, types, _) = &c.tuples[*k];
                    if let Some((m, passes, pcs, split)) = mode(&pe, types) {
                        per.entry(c.class.as_str()).or_default().push(ArrayTuple {
                            tuple: name.clone(),
                            types: types.clone(),
                            mode: m,
                            passes,
                            pieces: pcs,
                            split,
                        });
                    }
                }
                let h = by_class[host.as_str()];
                out.push(ArrayFile {
                    schema: ARRAY_SCHEMA.into(),
                    id: String::new(),
                    dataflow: flow.to_string(),
                    host: host.clone(),
                    dims: h.dims.clone(),
                    pe: pe.clone(),
                    pe_price: pe_price(lib, &pe),
                    covers: per
                        .into_iter()
                        .map(|(cl, mut tuples)| {
                            tuples.sort_by(|a, b| a.tuple.cmp(&b.tuple));
                            let ci = by_class[cl];
                            ArrayCover {
                                class: cl.to_string(),
                                embedded: cl != host,
                                candidate: ci.candidate,
                                tuples,
                                workload: ci.workload,
                            }
                        })
                        .collect(),
                });
            }
        }
    }
    out.sort_by(|a, b| (&a.dataflow, &a.pe, &a.host).cmp(&(&b.dataflow, &b.pe, &b.host)));
    for (i, a) in out.iter_mut().enumerate() {
        a.id = format!("array{i:03}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn modes_of_an_int8_element() {
        let pe = s(&["i8", "i8", "i32"]);
        assert_eq!(mode(&pe, &pe).map(|m| m.0), Some(ArrayMode::Native));
        assert_eq!(
            mode(&pe, &s(&["i4", "i8", "i32"])).map(|m| m.0),
            Some(ArrayMode::Widen)
        );
        let c = mode(&pe, &s(&["i16", "i8", "i32"])).expect("composes");
        assert_eq!(c.0, ArrayMode::Compose);
        assert_eq!(c.2, vec![7, 7, 2]);
        let u = mode(&pe, &s(&["ui8", "i8", "i32"])).expect("composes");
        assert_eq!(u.2, vec![7, 1]);
        assert!(mode(&pe, &s(&["i8", "i8", "i16"])).is_none());
        assert!(mode(&s(&["bf16", "bf16", "f32"]), &s(&["f16", "f16", "f32"])).is_none());
    }
}
