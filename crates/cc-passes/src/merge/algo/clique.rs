//! Maximum-weight clique on a compatibility graph (docs/passes/merge.md §4.5), three strategies:
//!
//! - `exact_ilp`: maximize Σ wᵢxᵢ subject to one constraint per super-node of A and of B (its vertex
//!   mappings are pairwise incompatible: a clique of the conflict graph), `x_arc ≤ x_end` for both ends of
//!   an arc mapping, one orientation per commutative pair, cycle no-goods, and the order constraints
//!   added lazily when a solution violates one (microlp, branch and bound with a node budget);
//! - `ostergard`: Östergård's weighted branch and bound (Nordic J. Computing 8(4), 2001): vertices in a
//!   fixed order, `c(i)` the best clique within `{vᵢ, …, vₙ}` computed from the back, pruning by
//!   `weight + c(i)`;
//! - `greedy`: by weight, each node kept when compatible with every kept one.
//!
//! The exact strategies give the same optimum: the ILP's constraints are exactly the incompatibilities
//! (an arc's ends are compatible with everything the arc is compatible with, and weights are
//! non-negative, so a maximum clique can always be closed under the ends of its arcs).

use std::time::{Duration, Instant};

use microlp::{ComparisonOp, LinearExpr, OptimizationDirection, Problem, SolveOptions};

use super::compat::{Bits, Model};
use crate::merge::config::CliqueAlgorithm;

#[derive(Debug, Clone, PartialEq)]
pub struct Solved {
    /// Closed under arc ends, sorted.
    pub chosen: Vec<usize>,
    pub objective: f64,
    pub optimal: bool,
    pub algorithm: CliqueAlgorithm,
    /// Why `greedy` answered instead (`size`, `timeout`, `budget`).
    pub fallback: Option<String>,
}

/// Solve with `alg`; `nogoods` are sets that may not all be chosen.
pub fn solve(
    m: &Model,
    alg: CliqueAlgorithm,
    nogoods: &[Vec<usize>],
    node_budget: u64,
    timeout: Duration,
    max_nodes: usize,
) -> Solved {
    if alg != CliqueAlgorithm::Greedy && m.n() > max_nodes {
        let mut g = greedy(m, nogoods);
        g.fallback = Some("size".into());
        return g;
    }
    let r = match alg {
        CliqueAlgorithm::ExactIlp => ilp(m, nogoods, node_budget, timeout),
        CliqueAlgorithm::Ostergard => ostergard(m, nogoods, node_budget, timeout),
        CliqueAlgorithm::Greedy => return greedy(m, nogoods),
    };
    match r {
        Ok(s) => s,
        Err(why) => {
            let mut g = greedy(m, nogoods);
            g.fallback = Some(why);
            g
        }
    }
}

fn violates(chosen: &[usize], nogoods: &[Vec<usize>]) -> bool {
    nogoods
        .iter()
        .any(|g| g.iter().all(|x| chosen.binary_search(x).is_ok()))
}

/// Feasibility of a closed, sorted set (the independent check used by the tests and `verify`).
pub fn feasible(m: &Model, chosen: &[usize], nogoods: &[Vec<usize>]) -> bool {
    for (k, &i) in chosen.iter().enumerate() {
        for &j in &chosen[k + 1..] {
            if !m.compatible(i, j) {
                return false;
            }
        }
        if m.ends(i).iter().any(|e| chosen.binary_search(e).is_err()) {
            return false;
        }
    }
    !violates(chosen, nogoods)
}

pub fn greedy(m: &Model, nogoods: &[Vec<usize>]) -> Solved {
    let nv = m.vm.len();
    let mut order: Vec<usize> = (0..m.n()).collect();
    order.sort_by(|&a, &b| m.weight(b).total_cmp(&m.weight(a)).then(a.cmp(&b)));
    // Incremental state: the same decisions as checking every chosen node, without the quadratic scan.
    let mut chosen = vec![false; m.n()];
    let mut used_a = vec![false; m.sa.members.len()];
    let mut used_b = vec![false; m.sb.members.len()];
    let mut vms: Vec<usize> = Vec::new();
    let mut perm: std::collections::HashMap<u32, bool> = Default::default();
    let mut wires_a: std::collections::HashSet<(u32, u32, u32, u32)> = Default::default();
    let mut wires_b: std::collections::HashSet<(u32, u32, u32, u32)> = Default::default();
    let fits = |x: usize, used_a: &[bool], used_b: &[bool], vms: &[usize], chosen: &[bool]| -> bool {
        if chosen[x] {
            return true;
        }
        let v = &m.vm[x];
        !used_a[v.sa as usize] && !used_b[v.sb as usize] && vms.iter().all(|&y| !m.order_conflict(x, y))
    };
    for i in order {
        if chosen[i] {
            continue;
        }
        let ends = m.ends(i);
        let mut ok = ends.iter().all(|&x| fits(x, &used_a, &used_b, &vms, &chosen));
        if ok && ends.len() == 2 && ends[0] != ends[1] && !chosen[ends[0]] && !chosen[ends[1]] {
            ok = !m.vm_conflict(ends[0], ends[1]);
        }
        if ok && i >= nv {
            let a = &m.am[i - nv];
            ok = perm.get(&a.dst).is_none_or(|&p| p == a.perm)
                && !wires_a.contains(&a.a)
                && !wires_b.contains(&a.b);
        }
        if !ok {
            continue;
        }
        if !nogoods.is_empty() {
            let mut next: Vec<usize> = (0..m.n()).filter(|&x| chosen[x]).collect();
            next.extend(ends.iter().copied().chain([i]));
            next.sort_unstable();
            next.dedup();
            if violates(&next, nogoods) {
                continue;
            }
        }
        for x in ends.iter().copied().chain([i]) {
            if chosen[x] {
                continue;
            }
            chosen[x] = true;
            if x < nv {
                used_a[m.vm[x].sa as usize] = true;
                used_b[m.vm[x].sb as usize] = true;
                vms.push(x);
            } else {
                let a = &m.am[x - nv];
                perm.insert(a.dst, a.perm);
                wires_a.insert(a.a);
                wires_b.insert(a.b);
            }
        }
    }
    let chosen: Vec<usize> = (0..m.n()).filter(|&x| chosen[x]).collect();
    Solved {
        objective: m.objective(&chosen),
        chosen,
        optimal: false,
        algorithm: CliqueAlgorithm::Greedy,
        fallback: None,
    }
}

fn ilp(m: &Model, nogoods: &[Vec<usize>], node_budget: u64, timeout: Duration) -> Result<Solved, String> {
    let start = Instant::now();
    let nv = m.vm.len();
    let mut order_cuts: Vec<(usize, usize)> = Vec::new();
    loop {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x: Vec<_> = (0..m.n()).map(|i| p.add_binary_var(m.weight(i))).collect();
        let row = |p: &mut Problem, vars: &[usize], rhs: f64| {
            let mut e = LinearExpr::empty();
            for &v in vars {
                e.add(x[v], 1.0);
            }
            p.add_constraint(e, ComparisonOp::Le, rhs);
        };
        let mut by_a: Vec<Vec<usize>> = vec![Vec::new(); m.sa.members.len()];
        let mut by_b: Vec<Vec<usize>> = vec![Vec::new(); m.sb.members.len()];
        for (i, v) in m.vm.iter().enumerate() {
            by_a[v.sa as usize].push(i);
            by_b[v.sb as usize].push(i);
        }
        for g in by_a.iter().chain(&by_b).filter(|g| g.len() > 1) {
            row(&mut p, g, 1.0);
        }
        let mut perms: std::collections::BTreeMap<u32, (Vec<usize>, Vec<usize>)> = Default::default();
        for (k, a) in m.am.iter().enumerate() {
            let i = nv + k;
            for e in [a.src as usize, a.dst as usize] {
                let mut d = LinearExpr::empty();
                d.add(x[i], 1.0);
                d.add(x[e], -1.0);
                p.add_constraint(d, ComparisonOp::Le, 0.0);
            }
            let s = perms.entry(a.dst).or_default();
            if a.perm { s.1.push(i) } else { s.0.push(i) }
        }
        for (same, swapped) in perms.values() {
            if same.is_empty() || swapped.is_empty() {
                continue;
            }
            let z = p.add_binary_var(0.0);
            for &i in same {
                let mut d = LinearExpr::empty();
                d.add(x[i], 1.0);
                d.add(z, 1.0);
                p.add_constraint(d, ComparisonOp::Le, 1.0);
            }
            for &i in swapped {
                let mut d = LinearExpr::empty();
                d.add(x[i], 1.0);
                d.add(z, -1.0);
                p.add_constraint(d, ComparisonOp::Le, 0.0);
            }
        }
        // The same wire of A (or of B) in two arc mappings with the same ends.
        let mut wa: std::collections::HashMap<(u32, u32, u32, u32), Vec<usize>> = Default::default();
        let mut wb: std::collections::HashMap<(u32, u32, u32, u32), Vec<usize>> = Default::default();
        for (k, a) in m.am.iter().enumerate() {
            wa.entry(a.a).or_default().push(nv + k);
            wb.entry(a.b).or_default().push(nv + k);
        }
        let mut wires: Vec<&Vec<usize>> = wa.values().chain(wb.values()).filter(|g| g.len() > 1).collect();
        wires.sort();
        for g in wires {
            row(&mut p, g, 1.0);
        }
        for &(i, j) in &order_cuts {
            row(&mut p, &[i, j], 1.0);
        }
        for g in nogoods {
            row(&mut p, g, g.len() as f64 - 1.0);
        }
        let left = timeout.saturating_sub(start.elapsed());
        if left.is_zero() {
            return Err("timeout".into());
        }
        let mut opts = SolveOptions::default();
        opts.time_limit = Some(left);
        opts.node_limit = Some(node_budget);
        let out = p.solve_with(opts).map_err(|e| format!("exact_ilp: {e}"))?;
        let optimal = out.is_optimal();
        let reason = out.termination_reason();
        let sol = out.into_solution().map_err(|_| match reason {
            microlp::TerminationReason::TimeLimit => "timeout".to_string(),
            _ => "budget".to_string(),
        })?;
        let chosen: Vec<usize> = (0..m.n()).filter(|&i| sol.var_value(x[i]) > 0.5).collect();
        let vms: Vec<usize> = chosen.iter().copied().filter(|&i| i < nv).collect();
        let mut cuts = Vec::new();
        for (k, &i) in vms.iter().enumerate() {
            for &j in &vms[k + 1..] {
                if m.order_conflict(i, j) {
                    cuts.push((i, j));
                }
            }
        }
        if cuts.is_empty() {
            let chosen = m.close(&chosen);
            return Ok(Solved {
                objective: m.objective(&chosen),
                chosen,
                optimal,
                algorithm: CliqueAlgorithm::ExactIlp,
                fallback: None,
            });
        }
        order_cuts.extend(cuts);
    }
}

/// Östergård's algorithm on the compatibility graph (adjacency as bitsets).
fn ostergard(
    m: &Model,
    nogoods: &[Vec<usize>],
    node_budget: u64,
    timeout: Duration,
) -> Result<Solved, String> {
    let n = m.n();
    let start = Instant::now();
    // Vertex order: by weight, heaviest last (c(i) is built from the back, so light vertices are tried
    // with the strongest bounds first).
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|&a, &b| m.weight(a).total_cmp(&m.weight(b)).then(a.cmp(&b)));
    let pos: Vec<usize> = {
        let mut p = vec![0; n];
        for (k, &v) in ord.iter().enumerate() {
            p[v] = k;
        }
        p
    };
    let w: Vec<f64> = ord.iter().map(|&v| m.weight(v)).collect();
    let mut adj = vec![Bits::new(n); n];
    for i in 0..n {
        for j in i + 1..n {
            if m.compatible(ord[i], ord[j]) {
                adj[i].set(j);
                adj[j].set(i);
            }
        }
    }
    let _ = &pos;
    struct S<'a> {
        m: &'a Model,
        ord: &'a [usize],
        adj: &'a [Bits],
        w: &'a [f64],
        c: Vec<f64>,
        best: f64,
        best_set: Vec<usize>,
        cur: Vec<usize>,
        nodes: u64,
        budget: u64,
        ng: &'a [Vec<usize>],
        start: Instant,
        timeout: Duration,
        stop: Option<String>,
    }
    /// No no-good is contained in the current clique plus `add`, closed under arc ends.
    fn ok_ng(s: &S, add: usize) -> bool {
        if s.ng.is_empty() {
            return true;
        }
        let mut closed: Vec<usize> = s
            .cur
            .iter()
            .chain([&add])
            .flat_map(|&k| s.m.ends(s.ord[k]).into_iter().chain([s.ord[k]]))
            .collect();
        closed.sort_unstable();
        closed.dedup();
        !violates(&closed, s.ng)
    }
    fn expand(s: &mut S, cand: Vec<usize>, weight: f64) {
        if s.stop.is_some() {
            return;
        }
        s.nodes += 1;
        if s.nodes > s.budget {
            s.stop = Some("budget".into());
            return;
        }
        if s.nodes.is_multiple_of(4096) && s.start.elapsed() > s.timeout {
            s.stop = Some("timeout".into());
            return;
        }
        if cand.is_empty() {
            if weight > s.best + 1e-9 {
                s.best = weight;
                s.best_set = s.cur.clone();
            }
            return;
        }
        let mut cand = cand;
        while let Some(&v) = cand.first() {
            let rest: f64 = cand.iter().map(|&x| s.w[x]).sum();
            if weight + rest <= s.best + 1e-9 || weight + s.c[v] <= s.best + 1e-9 {
                return;
            }
            cand.remove(0);
            if !ok_ng(s, v) {
                continue;
            }
            let next: Vec<usize> = cand.iter().copied().filter(|&x| s.adj[v].get(x)).collect();
            s.cur.push(v);
            expand(s, next, weight + s.w[v]);
            s.cur.pop();
            if s.stop.is_some() {
                return;
            }
        }
        if weight > s.best + 1e-9 {
            s.best = weight;
            s.best_set = s.cur.clone();
        }
    }
    let mut s = S {
        m,
        ord: &ord,
        adj: &adj,
        w: &w,
        c: vec![0.0; n],
        best: 0.0,
        best_set: Vec::new(),
        cur: Vec::new(),
        nodes: 0,
        budget: node_budget,
        ng: nogoods,
        start,
        timeout,
        stop: None,
    };
    for i in (0..n).rev() {
        let cand: Vec<usize> = (i + 1..n).filter(|&x| adj[i].get(x)).collect();
        s.cur.push(i);
        if ok_ng(&s, i) {
            expand(&mut s, cand, w[i]);
        }
        s.cur.pop();
        s.c[i] = s.best;
        if let Some(why) = s.stop.clone() {
            return Err(why);
        }
    }
    let chosen: Vec<usize> = s.best_set.iter().map(|&k| ord[k]).collect();
    let chosen = m.close(&chosen);
    Ok(Solved {
        objective: m.objective(&chosen),
        chosen,
        optimal: true,
        algorithm: CliqueAlgorithm::Ostergard,
        fallback: None,
    })
}
