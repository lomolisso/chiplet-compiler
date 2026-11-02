//! Space-time mappings (docs/passes/iteration_space.md §4.3). For the uniform recurrence of a
//! contraction over its space (at most three dims), every schedule λ ∈ {-1, 0, 1}^n with λ·d ≥ 1 for
//! every dependence (causality, Quinton 1984 Thm 1) and every projection direction u ∈ {-1, 0, 1}^n
//! (first nonzero entry positive) with λ·u ≠ 0 (no two iterations on one PE at one time, Quinton 1984
//! Thm 5) whose allocation S keeps every dependence on a nearest-neighbour link (S d ∈ {-1, 0, 1}^(n-1))
//! is a valid mapping. Candidates are ranked by workload-weighted PE utilization |D| / (#PE · T), then
//! by inter-PE link bits per PE (Kung 1982: local communication; a moving accumulator costs its width),
//! then by (λ, u).

use std::collections::BTreeMap;

use crate::iteration_space::algo::eval::{Table, row_width};
use crate::iteration_space::json::{Candidate, Dependences, Group, Link, Skeleton};

fn dot(a: &[i64], b: &[i64]) -> i64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn tuples(n: usize) -> Vec<Vec<i64>> {
    let mut out = vec![Vec::new()];
    for _ in 0..n {
        out = out
            .into_iter()
            .flat_map(|v| {
                [-1i64, 0, 1].into_iter().map(move |x| {
                    let mut w = v.clone();
                    w.push(x);
                    w
                })
            })
            .collect();
    }
    out
}

/// An allocation for projection direction `u` (entries in {-1, 0, 1}, pivot `p` its first nonzero):
/// rows `u_p·e_j − u_j·e_p` for `j ≠ p`, so `S u = 0`, and S restricted to the other coordinates is
/// `±I`, so `[λ; S]` is nonsingular iff `λ·u ≠ 0`.
pub fn allocation(u: &[i64]) -> Vec<Vec<i64>> {
    let p = u.iter().position(|&x| x != 0).unwrap_or(0);
    (0..u.len())
        .filter(|&j| j != p)
        .map(|j| {
            let mut r = vec![0; u.len()];
            r[j] = u[p];
            r[p] -= u[j];
            r
        })
        .collect()
}

pub fn apply(s: &[Vec<i64>], d: &[i64]) -> Vec<i64> {
    s.iter().map(|r| dot(r, d)).collect()
}

/// `|S(D)|` for a box of extents `e`: the lines `z + t·u` that meet the box.
pub fn pe_count(e: &[u64], u: &[i64]) -> u128 {
    let all: u128 = e.iter().map(|&x| u128::from(x)).product();
    let shifted: u128 = e
        .iter()
        .zip(u)
        .map(|(&x, &y)| u128::from(x.saturating_sub(y.unsigned_abs())))
        .product();
    all - shifted
}

/// Schedule length `T = Σ |λ_i|·(E_i − 1) + 1`.
pub fn schedule_length(e: &[u64], lambda: &[i64]) -> u128 {
    e.iter()
        .zip(lambda)
        .map(|(&x, &l)| u128::from(l.unsigned_abs()) * u128::from(x.saturating_sub(1)))
        .sum::<u128>()
        + 1
}

/// `Σ_j |S_rj|·(E(g_j) − 1) + 1`, printed.
fn array_dims(s: &[Vec<i64>], space: &[Group]) -> Vec<String> {
    s.iter()
        .map(|r| {
            let gs: Vec<String> = r
                .iter()
                .zip(space)
                .filter(|(x, _)| **x != 0)
                .map(|(_, g)| format!("E({})", g.name()))
                .collect();
            match gs.len() {
                0 => "1".into(),
                1 => gs[0].clone(),
                n => format!("{} - {}", gs.join(" + "), n - 1),
            }
        })
        .collect()
}

/// Whether every row's extents of an operand's axes are integers (no input symbol): a weight.
fn size_independent(t: &Table<'_>, axes: &[u32]) -> bool {
    t.rows.iter().all(|r| {
        axes.iter()
            .all(|&c| r.sizes.get(c as usize).is_some_and(|s| s.parse::<i64>().is_ok()))
    })
}

/// The ranked candidates of a uniform contraction.
pub fn candidates(s: &Skeleton, d: &Dependences, t: &Table<'_>) -> Vec<Candidate> {
    let n = d.space.len();
    let ds: Vec<(&str, &[i64])> = d
        .vectors
        .iter()
        .map(|v| (v.operand.as_str(), v.v.as_slice()))
        .collect();
    // Per row: weight, group extents, widths of the operands carrying a vector.
    let mut rows: Vec<(f64, Vec<u64>, BTreeMap<&str, u32>)> = Vec::new();
    for r in t.rows {
        let w = r.weighted as f64;
        if w <= 0.0 {
            continue;
        }
        let Some(e) = d
            .space
            .iter()
            .map(|g| t.group_extent(s, &d.groups[g], r))
            .collect::<Option<Vec<u64>>>()
        else {
            continue;
        };
        let mut widths = BTreeMap::new();
        for (name, _) in &ds {
            if let Some(o) = s.operands.iter().find(|o| o.core && o.name == *name) {
                widths.insert(*name, row_width(r, o.elem));
            }
        }
        rows.push((w, e, widths));
    }
    let total: f64 = rows.iter().map(|x| x.0).sum();
    let mut out: Vec<Candidate> = Vec::new();
    for lambda in tuples(n) {
        if ds.iter().any(|(_, v)| dot(&lambda, v) < 1) {
            continue;
        }
        for u in tuples(n) {
            if u.iter().find(|&&x| x != 0).is_none_or(|&x| x < 0) || dot(&lambda, &u) == 0 {
                continue;
            }
            let sm = allocation(&u);
            let links: Vec<Link> = ds
                .iter()
                .map(|(name, v)| Link {
                    operand: (*name).to_string(),
                    step: apply(&sm, v),
                    delay: dot(&lambda, v),
                })
                .collect();
            if links.iter().any(|l| l.step.iter().any(|x| x.abs() > 1)) {
                continue;
            }
            let stationary: Vec<&Link> = links.iter().filter(|l| l.step.iter().all(|&x| x == 0)).collect();
            let st = stationary.first().map(|l| l.operand.clone());
            let dataflow = match st.as_deref() {
                None => "systolic".to_string(),
                Some("result") => "output_stationary".to_string(),
                Some(x) => {
                    let axes = s
                        .operands
                        .iter()
                        .find(|o| o.core && o.name == x)
                        .map(|o| o.axes.clone())
                        .unwrap_or_default();
                    if size_independent(t, &axes) {
                        "weight_stationary".to_string()
                    } else {
                        "input_stationary".to_string()
                    }
                }
            };
            let (mut util, mut bits) = (0.0, 0.0);
            for (w, e, widths) in &rows {
                let all: u128 = e.iter().map(|&x| u128::from(x)).product();
                let pes = pe_count(e, &u).max(1);
                let tt = schedule_length(e, &lambda);
                util += w * (all as f64) / ((pes as f64) * (tt as f64));
                let b: u32 = links
                    .iter()
                    .filter(|l| l.step.iter().any(|&x| x != 0))
                    .map(|l| widths.get(l.operand.as_str()).copied().unwrap_or(0))
                    .sum();
                bits += w * f64::from(b);
            }
            if total > 0.0 {
                util /= total;
                bits /= total;
            }
            out.push(Candidate {
                lambda: lambda.clone(),
                array: array_dims(&sm, &d.space),
                s: sm,
                u,
                links,
                stationary: st,
                dataflow,
                utilization: round6(util),
                link_bits: round6(bits),
                rank: 0,
            });
        }
    }
    out.sort_by(|a, b| {
        b.utilization
            .total_cmp(&a.utilization)
            .then(a.link_bits.total_cmp(&b.link_bits))
            .then(a.lambda.cmp(&b.lambda))
            .then(a.u.cmp(&b.u))
    });
    for (i, c) in out.iter_mut().enumerate() {
        c.rank = i as u32;
    }
    out
}

/// Rounded so that the JSON (and its digest) does not depend on summation order noise.
pub fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}
