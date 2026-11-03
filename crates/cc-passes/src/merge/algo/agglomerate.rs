//! Agglomeration of one family (docs/passes/merge.md §4.6): start with one unit per datapath unit;
//! repeatedly merge the admissible pair (hosts related by ⊑, no class in both) with the largest area gain whose merged
//! unit is acyclic and meets the depth bound; stop when no pair saves area, or at `target_units`.
//!
//! Lazy greedy: every pair enters a priority queue keyed by an upper bound on its gain (per label, the
//! best pairing of block prices, ignoring multiplexers: a merge saves at most the cheaper of two shared
//! blocks). The top entry's exact gain (a clique solve and a rebuild) replaces its bound; an entry is
//! accepted when its exact gain is still on top. Bounds of the top entries are resolved in parallel
//! batches, so the result does not depend on the thread count.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::time::Duration;

use rayon::prelude::*;

use super::clique::{Solved, solve};
use super::compat::{Model, model, supers};
use super::fu::{Costed, Fu, cost, topo};
use super::library::Lib;
use super::rebuild::{nogood, rebuild};
use super::unit::Unit;
use crate::merge::config::{CliqueAlgorithm, MergeConfig};
use crate::merge::json::{CLIQUE_SCHEMA, CertNode, CliqueCert, CliqueStat, Step};

/// The result of one family.
#[derive(Debug, Clone)]
pub struct Agglomerated {
    pub fus: Vec<Fu>,
    pub costs: Vec<Costed>,
    pub steps: Vec<Step>,
    pub certs: Vec<(u32, CliqueCert)>,
    pub bound_ps: f64,
    pub initial_area: f64,
    pub stopped_by: String,
    pub evaluated: u64,
}

/// An evaluated pair.
#[derive(Debug, Clone)]
struct Eval {
    fu: Fu,
    cost: Costed,
    gain: f64,
    stat: CliqueStat,
    cert: Option<CliqueCert>,
}

#[derive(Debug, Clone, PartialEq)]
struct Entry {
    key: f64,
    a: usize,
    b: usize,
    exact: bool,
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.key
            .total_cmp(&o.key)
            .then(o.a.cmp(&self.a))
            .then(o.b.cmp(&self.b))
            .then(self.exact.cmp(&o.exact))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Per label, the prices of the unit's super-nodes, largest first.
fn profile(fu: &Fu, c: &Costed) -> BTreeMap<String, Vec<f64>> {
    let s = supers(fu, c);
    let mut m: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (l, p) in s.labels.iter().zip(&s.price) {
        m.entry(l.clone()).or_default().push(*p);
    }
    for v in m.values_mut() {
        v.sort_by(|a, b| b.total_cmp(a));
    }
    m
}

fn bound(a: &BTreeMap<String, Vec<f64>>, b: &BTreeMap<String, Vec<f64>>) -> f64 {
    let mut s = 0.0;
    for (l, xa) in a {
        if let Some(xb) = b.get(l) {
            s += xa.iter().zip(xb).map(|(x, y)| x.min(*y)).sum::<f64>();
        }
    }
    s
}

#[derive(Debug)]
pub struct Ctx<'a> {
    pub units: &'a [Unit],
    pub lib: &'a Lib,
    pub cfg: &'a MergeConfig,
    /// `(guest, host)` pairs of the embedding relation.
    pub embeds: &'a HashSet<(String, String)>,
}

impl Ctx<'_> {
    /// The host of a merged pair, if the pair is admissible.
    fn host(&self, a: &str, b: &str) -> Option<String> {
        if a == b || self.embeds.contains(&(a.to_string(), b.to_string())) {
            Some(b.to_string())
        } else if self.embeds.contains(&(b.to_string(), a.to_string())) {
            Some(a.to_string())
        } else {
            None
        }
    }

    /// Two units may merge when their hosts are related by ⊑ and no class has a different unit on each
    /// side (the stages of one template class run at the same time, so they cannot share a unit; the
    /// parts of one unit split by tuple types never run together, so they may).
    fn admissible(&self, a: &Fu, b: &Fu) -> Option<String> {
        let units = |f: &Fu| -> BTreeSet<(&str, &str)> {
            f.cfgs
                .iter()
                .map(|c| {
                    let u = &self.units[c.unit as usize];
                    (u.class.as_str(), u.id.as_str())
                })
                .collect()
        };
        let (ua, ub) = (units(a), units(b));
        for (c, u) in &ua {
            if ub.iter().any(|(d, v)| d == c && v != u) {
                return None;
            }
        }
        self.host(&a.host, &b.host)
    }

    fn evaluate(&self, a: &Fu, ca: &Costed, b: &Fu, cb: &Costed, bound_ps: f64) -> Option<Eval> {
        let host = self.admissible(a, b)?;
        let mut prices = HashMap::new();
        let m = model(a, ca, b, cb, self.lib, &mut prices);
        let cq = &self.cfg.clique;
        let timeout = Duration::from_secs(cq.timeout_s);
        let mut nogoods: Vec<Vec<usize>> = Vec::new();
        let mut rounds = 0u32;
        loop {
            rounds += 1;
            let s = solve(
                &m,
                cq.algorithm,
                &nogoods,
                cq.node_budget,
                timeout,
                self.cfg.max_compat_nodes,
            );
            let fu = rebuild(a, b, &m, &s.chosen, &host);
            let c = cost(&fu, self.units, self.lib);
            match topo(&fu, &c.wiring) {
                Ok(_) => {
                    if c.depth > bound_ps + 1e-6 {
                        return None;
                    }
                    let gain = ca.area + cb.area - c.area;
                    let cert = (m.n() <= self.cfg.certify_max_nodes).then(|| certificate(&m, &nogoods, &s));
                    let stat = CliqueStat {
                        algorithm: s.algorithm.name().into(),
                        vertex_mappings: m.vm.len() as u64,
                        arc_mappings: m.am.len() as u64,
                        objective: s.objective,
                        optimal: s.optimal,
                        rounds,
                        fallback: s.fallback.clone(),
                        certificate: None,
                    };
                    return Some(Eval {
                        fu,
                        cost: c,
                        gain,
                        stat,
                        cert,
                    });
                }
                Err(cycle) => {
                    let g = nogood(a, &m, &s.chosen, &cycle);
                    if g.is_empty() || nogoods.contains(&g) || rounds > 200 {
                        return None;
                    }
                    nogoods.push(g);
                }
            }
        }
    }
}

/// The stored instance of a small step: weights, conflicts, implications, no-goods, the solution.
pub fn certificate(m: &Model, nogoods: &[Vec<usize>], s: &Solved) -> CliqueCert {
    let nv = m.vm.len();
    let mut nodes = Vec::new();
    for v in &m.vm {
        nodes.push(CertNode {
            weight: v.w,
            kind: "vertex".into(),
            what: format!("{}~{}", m.sa.labels[v.sa as usize], v.sb),
        });
    }
    for a in &m.am {
        nodes.push(CertNode {
            weight: a.w,
            kind: "arc".into(),
            what: format!("{:?}~{:?}{}", a.a, a.b, if a.perm { " swapped" } else { "" }),
        });
    }
    let mut conflicts = Vec::new();
    for i in 0..m.n() {
        for j in i + 1..m.n() {
            if !m.compatible(i, j) {
                conflicts.push((i as u32, j as u32));
            }
        }
    }
    let mut implies = Vec::new();
    for (k, a) in m.am.iter().enumerate() {
        implies.push(((nv + k) as u32, a.src));
        implies.push(((nv + k) as u32, a.dst));
    }
    CliqueCert {
        schema: CLIQUE_SCHEMA.into(),
        nodes,
        conflicts,
        implies,
        nogoods: nogoods
            .iter()
            .map(|g| g.iter().map(|&x| x as u32).collect())
            .collect(),
        chosen: s.chosen.iter().map(|&x| x as u32).collect(),
        objective: s.objective,
        algorithm: s.algorithm.name().into(),
        optimal: s.optimal,
    }
}

/// The depth bound of a family: `max_logic_depth`, or the deepest single unit rounded up to whole
/// clock periods.
pub fn depth_bound(costs: &[Costed], lib: &Lib, max_logic_depth: f64) -> f64 {
    if max_logic_depth > 0.0 {
        return max_logic_depth;
    }
    let worst = costs.iter().map(|c| c.depth).fold(0.0f64, f64::max);
    let p = lib.period_ps.max(1.0);
    ((worst / p).ceil().max(1.0)) * p
}

pub fn agglomerate(cx: &Ctx, units_idx: &[u32]) -> Result<Agglomerated, String> {
    let mut fus: Vec<Fu> = Vec::new();
    for &ui in units_idx {
        fus.push(Fu::single(&cx.units[ui as usize], ui)?);
    }
    let mut costs: Vec<Costed> = fus.par_iter().map(|f| cost(f, cx.units, cx.lib)).collect();
    for (f, c) in fus.iter().zip(&costs) {
        if let Err(cyc) = topo(f, &c.wiring) {
            return Err(format!(
                "unit {}: a cycle {cyc:?}",
                cx.units[f.cfgs[0].unit as usize].id
            ));
        }
    }
    let bound_ps = depth_bound(&costs, cx.lib, cx.cfg.max_logic_depth);
    let initial_area: f64 = costs.iter().map(|c| c.area).sum();
    let mut profiles: Vec<BTreeMap<String, Vec<f64>>> =
        fus.iter().zip(&costs).map(|(f, c)| profile(f, c)).collect();
    let mut alive: Vec<bool> = vec![true; fus.len()];
    let mut heap: BinaryHeap<Entry> = BinaryHeap::new();
    let n0 = fus.len();
    let pairs: Vec<Entry> = (0..n0)
        .into_par_iter()
        .flat_map_iter(|a| {
            let (fus, profiles) = (&fus, &profiles);
            (a + 1..n0).filter_map(move |b| {
                cx.admissible(&fus[a], &fus[b])?;
                let k = bound(&profiles[a], &profiles[b]);
                (k > 0.0).then_some(Entry {
                    key: k,
                    a,
                    b,
                    exact: false,
                })
            })
        })
        .collect();
    heap.extend(pairs);
    let mut cache: HashMap<(usize, usize), Eval> = HashMap::new();
    let mut steps = Vec::new();
    let mut certs = Vec::new();
    let mut evaluated = 0u64;
    let mut area = initial_area;
    let mut count = n0;
    let batch = rayon::current_num_threads().max(1) * 2;
    let stopped_by;
    loop {
        if cx.cfg.target_units > 0 && count <= cx.cfg.target_units {
            stopped_by = "target".to_string();
            break;
        }
        // Resolve the bounds on top (a batch of them), in parallel.
        let mut todo: Vec<Entry> = Vec::new();
        let mut top_exact: Option<Entry> = None;
        while let Some(e) = heap.pop() {
            if !alive[e.a] || !alive[e.b] {
                continue;
            }
            if e.exact {
                top_exact = Some(e);
                break;
            }
            todo.push(e);
            if todo.len() >= batch {
                break;
            }
        }
        if !todo.is_empty() {
            if let Some(e) = top_exact.take() {
                heap.push(e);
            }
            let evals: Vec<(Entry, Option<Eval>)> = todo
                .into_par_iter()
                .map(|e| {
                    let r = cx.evaluate(&fus[e.a], &costs[e.a], &fus[e.b], &costs[e.b], bound_ps);
                    (e, r)
                })
                .collect();
            for (e, r) in evals {
                evaluated += 1;
                if let Some(v) = r {
                    heap.push(Entry {
                        key: v.gain,
                        a: e.a,
                        b: e.b,
                        exact: true,
                    });
                    cache.insert((e.a, e.b), v);
                }
            }
            continue;
        }
        let Some(e) = top_exact else {
            stopped_by = if steps.is_empty() {
                "none".to_string()
            } else {
                "gain".to_string()
            };
            break;
        };
        if e.key <= 1e-9 {
            stopped_by = "gain".to_string();
            break;
        }
        let mut v = cache.remove(&(e.a, e.b)).ok_or("an evaluated pair is missing")?;
        alive[e.a] = false;
        alive[e.b] = false;
        let id = fus.len();
        area -= v.gain;
        count -= 1;
        let step_no = steps.len() as u32;
        if let Some(cert) = v.cert.take() {
            v.stat.certificate = Some(format!("{step_no}.json"));
            certs.push((step_no, cert));
        }
        steps.push(Step {
            step: step_no,
            a: cx.units[fus[e.a].cfgs[0].unit as usize].id.clone(),
            b: cx.units[fus[e.b].cfgs[0].unit as usize].id.clone(),
            gain_um2: v.gain,
            units: count as u64,
            area_um2: area,
            depth_ps: v.cost.depth,
            clique: v.stat.clone(),
        });
        profiles.push(profile(&v.fu, &v.cost));
        fus.push(v.fu);
        costs.push(v.cost);
        alive.push(true);
        cache.retain(|k, _| alive[k.0] && alive[k.1]);
        let fresh: Vec<Entry> = (0..id)
            .into_par_iter()
            .filter(|&x| alive[x])
            .filter_map(|x| {
                cx.admissible(&fus[x], &fus[id])?;
                let k = bound(&profiles[x], &profiles[id]);
                (k > 0.0).then_some(Entry {
                    key: k,
                    a: x,
                    b: id,
                    exact: false,
                })
            })
            .collect();
        heap.extend(fresh);
    }
    let mut out_fus = Vec::new();
    let mut out_costs = Vec::new();
    for (i, (f, c)) in fus.into_iter().zip(costs).enumerate() {
        if alive[i] {
            out_fus.push(f);
            out_costs.push(c);
        }
    }
    Ok(Agglomerated {
        fus: out_fus,
        costs: out_costs,
        steps,
        certs,
        bound_ps,
        initial_area,
        stopped_by,
        evaluated,
    })
}

/// The algorithm a step used, parsed back (for summaries).
pub fn algorithm_of(s: &CliqueStat) -> Option<CliqueAlgorithm> {
    CliqueAlgorithm::parse(&s.algorithm)
}
