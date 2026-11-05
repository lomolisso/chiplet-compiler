//! Maximum independent set on an overlap graph (docs/passes/mine.md §4.4): one vertex per occurrence,
//! an edge when two occurrences share a datapath vertex. Strategies (`[passes.mine.mis] algorithm`):
//!
//! - `exact_ilp`: max Σ x_o subject to Σ_{o ∋ h} x_o ≤ 1 for every datapath vertex h (one clique
//!   constraint per vertex: the occurrences through h are pairwise adjacent), solved by microlp
//!   (branch and bound on the LP relaxation);
//! - `branch_and_bound`: branch and reduce (take vertices of degree 0 and 1, branch on a vertex of
//!   maximum degree; Tarjan and Trojanowski, SIAM J. Comput. 6(3), 1977), with a node budget;
//! - `greedy_min_degree`: take a vertex of minimum degree, delete its closed neighbourhood; ratio
//!   (Δ+2)/3 (Halldórsson and Radhakrishnan, Algorithmica 18(1), 1997), reported as the bound.
//!
//! Every strategy runs per connected component; isolated occurrences are all taken.

use cc_core::adapters::{Limits, MisSolver, Named, Registry, SelectCapabilities};
use good_lp::{Expression, ProblemVariables, Solution, SolverModel, constraint, microlp, variable};

use crate::mine::config::MisAlgorithm;

/// The MIS solvers, by `[passes.mine.mis] algorithm` (`MisAlgorithm::name`).
pub static SOLVERS: Registry<dyn MisSolver<Overlap, MisOut>> =
    Registry::new("MIS algorithm", &[&ExactIlp, &BranchAndBound, &GreedyMinDegree]);

/// `exact_ilp`.
#[derive(Debug, Clone, Copy)]
pub struct ExactIlp;
/// `branch_and_bound`.
#[derive(Debug, Clone, Copy)]
pub struct BranchAndBound;
/// `greedy_min_degree`.
#[derive(Debug, Clone, Copy)]
pub struct GreedyMinDegree;

impl Named for ExactIlp {
    fn name(&self) -> &'static str {
        MisAlgorithm::ExactIlp.name()
    }
}
impl Named for BranchAndBound {
    fn name(&self) -> &'static str {
        MisAlgorithm::BranchAndBound.name()
    }
}
impl Named for GreedyMinDegree {
    fn name(&self) -> &'static str {
        MisAlgorithm::GreedyMinDegree.name()
    }
}

impl MisSolver<Overlap, MisOut> for ExactIlp {
    fn capabilities(&self) -> SelectCapabilities {
        SelectCapabilities {
            exact: true,
            budgeted: false,
            timed: false,
        }
    }
    fn solve(&self, g: &Overlap, _l: &Limits) -> Result<MisOut, String> {
        per_component(self.name(), g, ilp)
    }
}

impl MisSolver<Overlap, MisOut> for BranchAndBound {
    fn capabilities(&self) -> SelectCapabilities {
        SelectCapabilities {
            exact: true,
            budgeted: true,
            timed: false,
        }
    }
    fn solve(&self, g: &Overlap, l: &Limits) -> Result<MisOut, String> {
        per_component(self.name(), g, |c| Ok(bnb(&c.adj, l.node_budget)))
    }
}

impl MisSolver<Overlap, MisOut> for GreedyMinDegree {
    fn capabilities(&self) -> SelectCapabilities {
        SelectCapabilities {
            exact: false,
            budgeted: false,
            timed: false,
        }
    }
    fn solve(&self, g: &Overlap, _l: &Limits) -> Result<MisOut, String> {
        per_component(self.name(), g, |c| Ok(greedy(&c.adj)))
    }
}

/// The overlap graph: `adj` sorted, `cliques` the occurrences through each shared vertex.
#[derive(Debug, Clone, Default)]
pub struct Overlap {
    pub adj: Vec<Vec<u32>>,
    pub cliques: Vec<Vec<u32>>,
}

/// The overlap graph of occurrences given by their images (any vertex ids).
pub fn overlap(images: &[&[u32]]) -> Overlap {
    let mut by_vertex: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    for (o, img) in images.iter().enumerate() {
        for &h in img.iter() {
            by_vertex.entry(h).or_default().push(o as u32);
        }
    }
    let mut adj = vec![Vec::new(); images.len()];
    let mut cliques = Vec::new();
    for (_, c) in by_vertex {
        if c.len() < 2 {
            continue;
        }
        for &a in &c {
            for &b in &c {
                if a != b {
                    adj[a as usize].push(b);
                }
            }
        }
        cliques.push(c);
    }
    for a in &mut adj {
        a.sort_unstable();
        a.dedup();
    }
    cliques.sort();
    cliques.dedup();
    Overlap { adj, cliques }
}

/// A solution: the chosen vertices (sorted), whether it is a maximum, and an upper bound on the maximum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisOut {
    pub chosen: Vec<u32>,
    pub optimal: bool,
    pub bound: u64,
}

/// A connected component with local ids.
struct Comp {
    verts: Vec<u32>,
    adj: Vec<Vec<u32>>,
    cliques: Vec<Vec<u32>>,
}

fn components(g: &Overlap) -> Vec<Comp> {
    let n = g.adj.len();
    let mut comp = vec![u32::MAX; n];
    let mut out = Vec::new();
    for s in 0..n {
        if comp[s] != u32::MAX {
            continue;
        }
        let id = out.len() as u32;
        let mut verts = vec![s as u32];
        comp[s] = id;
        let mut k = 0;
        while k < verts.len() {
            let v = verts[k] as usize;
            for &w in &g.adj[v] {
                if comp[w as usize] == u32::MAX {
                    comp[w as usize] = id;
                    verts.push(w);
                }
            }
            k += 1;
        }
        verts.sort_unstable();
        out.push(verts);
    }
    let mut local = vec![0u32; n];
    let mut comps: Vec<Comp> = out
        .into_iter()
        .map(|verts| {
            for (i, &v) in verts.iter().enumerate() {
                local[v as usize] = i as u32;
            }
            let adj = verts
                .iter()
                .map(|&v| g.adj[v as usize].iter().map(|&w| local[w as usize]).collect())
                .collect();
            Comp {
                verts,
                adj,
                cliques: Vec::new(),
            }
        })
        .collect();
    for c in &g.cliques {
        let k = comp[c[0] as usize] as usize;
        comps[k]
            .cliques
            .push(c.iter().map(|&v| local[v as usize]).collect());
    }
    comps
}

/// Solve with `alg`; `node_budget` bounds `branch_and_bound` per component.
pub fn solve(alg: MisAlgorithm, g: &Overlap, node_budget: u64) -> Result<MisOut, String> {
    SOLVERS.by_name(alg.name()).map_err(|e| e.to_string())?.solve(
        g,
        &Limits {
            node_budget,
            timeout: None,
        },
    )
}

/// `one` on every connected component (an isolated vertex is taken without it), each answer checked
/// to be independent, mapped back to the graph's ids.
fn per_component(
    name: &str,
    g: &Overlap,
    one: impl Fn(&Comp) -> Result<MisOut, String>,
) -> Result<MisOut, String> {
    let mut chosen = Vec::new();
    let mut optimal = true;
    let mut bound = 0u64;
    for c in components(g) {
        let r = if c.verts.len() == 1 {
            MisOut {
                chosen: vec![0],
                optimal: true,
                bound: 1,
            }
        } else {
            one(&c)?
        };
        for &v in &r.chosen {
            for &w in &c.adj[v as usize] {
                if r.chosen.binary_search(&w).is_ok() {
                    return Err(format!("{name}: chose adjacent occurrences"));
                }
            }
        }
        optimal &= r.optimal;
        bound += r.bound;
        chosen.extend(r.chosen.iter().map(|&v| c.verts[v as usize]));
    }
    chosen.sort_unstable();
    Ok(MisOut {
        chosen,
        optimal,
        bound,
    })
}

fn ilp(c: &Comp) -> Result<MisOut, String> {
    let n = c.verts.len();
    let mut pv = ProblemVariables::new();
    let x: Vec<_> = (0..n).map(|_| pv.add(variable().binary())).collect();
    let obj: Expression = x.iter().copied().sum();
    let mut m = pv.maximise(obj).using(microlp);
    for q in &c.cliques {
        let e: Expression = q.iter().map(|&v| x[v as usize]).sum();
        m = m.with(constraint!(e <= 1));
    }
    let sol = m.solve().map_err(|e| format!("exact_ilp: {e}"))?;
    let chosen: Vec<u32> = (0..n as u32)
        .filter(|&v| sol.value(x[v as usize]) > 0.5)
        .collect();
    let size = chosen.len() as u64;
    Ok(MisOut {
        chosen,
        optimal: true,
        bound: size,
    })
}

type Bits = Vec<u64>;

fn bits_of(n: usize, it: impl IntoIterator<Item = u32>) -> Bits {
    let mut b = vec![0u64; n.div_ceil(64)];
    for v in it {
        b[v as usize / 64] |= 1 << (v % 64);
    }
    b
}

fn count(b: &Bits) -> u32 {
    b.iter().map(|w| w.count_ones()).sum()
}

fn degree(nb: &Bits, cands: &Bits) -> u32 {
    nb.iter().zip(cands).map(|(a, b)| (a & b).count_ones()).sum()
}

fn ones(b: &Bits) -> impl Iterator<Item = usize> + '_ {
    b.iter().enumerate().flat_map(|(i, &w)| {
        (0..64)
            .filter(move |k| w & (1u64 << k) != 0)
            .map(move |k| i * 64 + k)
    })
}

fn bnb(adj: &[Vec<u32>], budget: u64) -> MisOut {
    let n = adj.len();
    let nb: Vec<Bits> = adj.iter().map(|a| bits_of(n, a.iter().copied())).collect();
    struct S<'a> {
        nb: &'a [Bits],
        best: Vec<u32>,
        nodes: u64,
        budget: u64,
        aborted: bool,
    }
    fn rec(s: &mut S, mut cands: Bits, cur: &mut Vec<u32>) {
        s.nodes += 1;
        if s.nodes > s.budget {
            s.aborted = true;
            return;
        }
        let mark = cur.len();
        // Reductions: a vertex of degree 0 or 1 is in some maximum independent set.
        loop {
            let pick = ones(&cands).find(|&v| degree(&s.nb[v], &cands) <= 1);
            let Some(v) = pick else { break };
            cur.push(v as u32);
            cands[v / 64] &= !(1 << (v % 64));
            for (a, b) in cands.iter_mut().zip(&s.nb[v]) {
                *a &= !b;
            }
        }
        let left = count(&cands) as usize;
        if left == 0 {
            if cur.len() > s.best.len() {
                s.best = cur.clone();
            }
        } else if cur.len() + left > s.best.len() {
            let v = ones(&cands)
                .max_by_key(|&v| (degree(&s.nb[v], &cands), std::cmp::Reverse(v)))
                .unwrap_or(0);
            let mut with = cands.clone();
            with[v / 64] &= !(1 << (v % 64));
            for (a, b) in with.iter_mut().zip(&s.nb[v]) {
                *a &= !b;
            }
            cur.push(v as u32);
            rec(s, with, cur);
            cur.pop();
            if !s.aborted {
                let mut without = cands;
                without[v / 64] &= !(1 << (v % 64));
                rec(s, without, cur);
            }
        }
        cur.truncate(mark);
    }
    let mut s = S {
        nb: &nb,
        best: Vec::new(),
        nodes: 0,
        budget,
        aborted: false,
    };
    let all = bits_of(n, 0..n as u32);
    rec(&mut s, all, &mut Vec::new());
    let mut chosen = s.best;
    chosen.sort_unstable();
    let size = chosen.len() as u64;
    MisOut {
        chosen,
        optimal: !s.aborted,
        bound: if s.aborted { n as u64 } else { size },
    }
}

fn greedy(adj: &[Vec<u32>]) -> MisOut {
    let n = adj.len();
    let nb: Vec<Bits> = adj.iter().map(|a| bits_of(n, a.iter().copied())).collect();
    let mut cands = bits_of(n, 0..n as u32);
    let mut chosen = Vec::new();
    while count(&cands) > 0 {
        let v = ones(&cands)
            .min_by_key(|&v| (degree(&nb[v], &cands), v))
            .unwrap_or(0);
        chosen.push(v as u32);
        cands[v / 64] &= !(1 << (v % 64));
        for (a, b) in cands.iter_mut().zip(&nb[v]) {
            *a &= !b;
        }
    }
    chosen.sort_unstable();
    let delta = adj.iter().map(Vec::len).max().unwrap_or(0) as u64;
    let size = chosen.len() as u64;
    let bound = (size * (delta + 2) / 3).clamp(size, n as u64);
    MisOut {
        chosen,
        optimal: bound == size,
        bound,
    }
}

/// Whether `chosen` is independent in `adj`.
pub fn independent(adj: &[Vec<u32>], chosen: &[u32]) -> bool {
    let set: std::collections::HashSet<u32> = chosen.iter().copied().collect();
    chosen
        .iter()
        .all(|&v| adj[v as usize].iter().all(|w| !set.contains(w)))
        && set.len() == chosen.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [MisAlgorithm; 3] = [
        MisAlgorithm::ExactIlp,
        MisAlgorithm::BranchAndBound,
        MisAlgorithm::GreedyMinDegree,
    ];

    /// The APEX 4-tap chain: occurrences A={p0,s1,s2}, B={p1,s1,s2}, C={p2,s2,s3}, D={p3,s3,y}:
    /// α = 2 (A or B, with D).
    #[test]
    fn apex_chain() {
        let (p0, p1, p2, p3, s1, s2, s3, y) = (0, 1, 2, 3, 4, 5, 6, 7);
        let occ = [
            vec![p0, s1, s2],
            vec![p1, s1, s2],
            vec![p2, s2, s3],
            vec![p3, s3, y],
        ];
        let refs: Vec<&[u32]> = occ.iter().map(Vec::as_slice).collect();
        let g = overlap(&refs);
        assert_eq!(g.adj, vec![vec![1, 2], vec![0, 2], vec![0, 1, 3], vec![2]]);
        for a in ALL {
            let r = solve(a, &g, 1000).unwrap();
            assert_eq!(r.chosen.len(), 2, "{a:?}");
            assert!(independent(&g.adj, &r.chosen));
        }
    }

    fn graph(n: u32, edges: &[(u32, u32)]) -> Overlap {
        // One shared vertex per edge.
        let mut occ = vec![vec![]; n as usize];
        for (k, &(a, b)) in edges.iter().enumerate() {
            occ[a as usize].push(1000 + k as u32);
            occ[b as usize].push(1000 + k as u32);
        }
        for (i, o) in occ.iter_mut().enumerate() {
            o.push(i as u32);
        }
        let refs: Vec<&[u32]> = occ.iter().map(Vec::as_slice).collect();
        overlap(&refs)
    }

    #[test]
    fn fixtures() {
        // (graph, α): a path of 7, a 5-clique, a star with 6 leaves, a 6-cycle, two components,
        // an interval graph, and an edgeless graph.
        let path: Vec<(u32, u32)> = (0..6).map(|i| (i, i + 1)).collect();
        let clique: Vec<(u32, u32)> = (0..5).flat_map(|a| (a + 1..5).map(move |b| (a, b))).collect();
        let star: Vec<(u32, u32)> = (1..7).map(|i| (0, i)).collect();
        let cycle: Vec<(u32, u32)> = (0..6).map(|i| (i, (i + 1) % 6)).collect();
        let two = vec![(0, 1), (1, 2), (3, 4)];
        let intervals = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (3, 5)];
        type Case = (u32, Vec<(u32, u32)>, usize);
        let cases: Vec<Case> = vec![
            (7, path, 4),
            (5, clique, 1),
            (7, star, 6),
            (6, cycle, 3),
            (5, two, 3),
            (6, intervals, 2),
            (4, vec![], 4),
        ];
        for (n, e, alpha) in cases {
            let g = graph(n, &e);
            for a in [MisAlgorithm::ExactIlp, MisAlgorithm::BranchAndBound] {
                let r = solve(a, &g, 1_000_000).unwrap();
                assert_eq!(r.chosen.len(), alpha, "{a:?} {e:?}");
                assert!(r.optimal && r.bound == alpha as u64);
            }
            let r = solve(MisAlgorithm::GreedyMinDegree, &g, 0).unwrap();
            assert!(independent(&g.adj, &r.chosen));
            assert!(r.bound >= alpha as u64, "{e:?}: bound {} < {alpha}", r.bound);
        }
    }

    /// Conformance of every registered solver (design §3.7): on the same graphs, an exact solver returns
    /// an independent set of maximum size claimed optimal, a heuristic an independent set no larger, with a
    /// bound no smaller than the maximum.
    #[test]
    fn every_registered_solver_conforms() {
        assert_eq!(SOLVERS.names(), ALL.map(MisAlgorithm::name));
        let mut st: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let limits = Limits {
            node_budget: 1_000_000,
            timeout: None,
        };
        for round in 0..40 {
            let n = 2 + (next() % 14) as u32;
            let edges: Vec<(u32, u32)> = (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
                .filter(|_| next() % 3 == 0)
                .collect();
            let g = graph(n, &edges);
            let mut alpha = None;
            for s in SOLVERS.iter().filter(|s| s.capabilities().exact) {
                let r = s.solve(&g, &limits).unwrap();
                assert!(independent(&g.adj, &r.chosen), "{} round {round}", s.name());
                assert!(r.optimal && r.bound == r.chosen.len() as u64, "{}", s.name());
                assert_eq!(*alpha.get_or_insert(r.chosen.len()), r.chosen.len(), "{} round {round}", s.name());
            }
            let alpha = alpha.expect("an exact solver");
            for s in SOLVERS.iter().filter(|s| !s.capabilities().exact) {
                let r = s.solve(&g, &limits).unwrap();
                assert!(independent(&g.adj, &r.chosen) && r.chosen.len() <= alpha, "{}", s.name());
                assert!(r.bound >= alpha as u64, "{}", s.name());
            }
        }
    }

    #[test]
    fn an_exhausted_budget_is_not_optimal() {
        let cycle: Vec<(u32, u32)> = (0..9).map(|i| (i, (i + 1) % 9)).collect();
        let r = solve(MisAlgorithm::BranchAndBound, &graph(9, &cycle), 1).unwrap();
        assert!(!r.optimal);
        assert_eq!(r.bound, 9);
    }
}
