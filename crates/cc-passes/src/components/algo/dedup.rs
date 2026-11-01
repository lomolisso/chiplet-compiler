//! Deduplication across all phases (docs/passes/components.md §4): graphs are processed by height (a
//! graph of height `h` calls components of height below `h`), bucketed by WL hash, and confirmed by
//! VF2 isomorphism against the bucket's representatives. Node predicate: equal keys, equal stable
//! colours and, for call nodes, equal child components.

use std::collections::BTreeMap;

use cc_ir::absgraph::{BudgetExceeded, Graph, Hash128, isomorphism};
use cc_ir::{Dim, Elem};
use rayon::prelude::*;

use super::PhaseOut;
use super::classes::Iface;
use super::graph::{ADim, ANode, NodeRef, Param};

/// A deduplicated component.
#[derive(Debug, Clone)]
pub struct Comp {
    pub id: String,
    pub hash: Hash128,
    pub height: u32,
    /// The representative (phase index, graph index).
    pub rep: (usize, u32),
    /// The representative's graph with child component ids.
    pub graph: Graph<ANode>,
    pub iface: Iface,
    pub dims: u32,
    pub elems: u32,
}

/// One graph's membership: component index, the map from component nodes to the graph's nodes, and the
/// graph's bindings in the component's numbering.
#[derive(Debug, Clone)]
pub struct Member {
    pub comp: u32,
    pub map: Vec<usize>,
    pub dims: Vec<Dim>,
    pub elems: Vec<Elem>,
    pub params: Vec<Vec<Param>>,
    pub refs: Vec<NodeRef>,
}

pub fn comp_id(hash: Hash128, k: usize) -> String {
    let h = (hash >> 64) as u64;
    if k == 0 {
        format!("c{h:016x}")
    } else {
        format!("c{h:016x}-{k}")
    }
}

type Key = (usize, u32);

/// Deduplicate every graph of every phase. `budget` bounds each VF2 search.
pub fn dedup(phases: &[PhaseOut], budget: u64) -> Result<(Vec<Comp>, Vec<Vec<Member>>), String> {
    let mut by_height: BTreeMap<u32, BTreeMap<Hash128, Vec<Key>>> = BTreeMap::new();
    for (pi, p) in phases.iter().enumerate() {
        for (gi, g) in p.graphs.iter().enumerate() {
            by_height
                .entry(g.height)
                .or_default()
                .entry(g.hash)
                .or_default()
                .push((pi, gi as u32));
        }
    }
    let mut comp_of: Vec<Vec<Option<u32>>> = phases.iter().map(|p| vec![None; p.graphs.len()]).collect();
    let mut maps: Vec<Vec<Option<Vec<usize>>>> = phases.iter().map(|p| vec![None; p.graphs.len()]).collect();
    let mut comps: Vec<Comp> = Vec::new();
    for buckets in by_height.values() {
        let bl: Vec<(&Hash128, &Vec<Key>)> = buckets.iter().collect();
        let done: Vec<Result<Reps, String>> = bl
            .par_iter()
            .map(|(_, members)| bucket(phases, &comp_of, members, budget))
            .collect();
        for ((&hash, _), res) in bl.into_iter().zip(done) {
            // Per representative: its members with their maps.
            let reps = res?;
            for (k, (_, members)) in reps.into_iter().enumerate() {
                let ci = comps.len() as u32;
                let (rp, rg) = members[0].0;
                let g = &phases[rp].graphs[rg as usize];
                let mut graph = g.graph.clone();
                for (n, c) in g.children.iter().enumerate() {
                    if let Some(c) = c {
                        let cc = comp_of[rp][*c as usize].ok_or("child component not assigned")?;
                        graph.nodes[n].child = Some(comps[cc as usize].id.clone());
                    }
                }
                comps.push(Comp {
                    id: comp_id(hash, k),
                    hash,
                    height: g.height,
                    rep: (rp, rg),
                    graph,
                    iface: g.iface.clone(),
                    dims: g.dims.len() as u32,
                    elems: g.elems.len() as u32,
                });
                for ((pi, gi), _, map) in members {
                    comp_of[pi][gi as usize] = Some(ci);
                    maps[pi][gi as usize] = Some(map);
                }
            }
        }
    }
    // Bindings in each component's numbering.
    let members: Vec<Vec<Member>> = phases
        .par_iter()
        .enumerate()
        .map(|(pi, p)| {
            p.graphs
                .iter()
                .enumerate()
                .map(|(gi, g)| {
                    let ci = comp_of[pi][gi].ok_or("graph not assigned")?;
                    let map = maps[pi][gi].clone().ok_or("graph without a map")?;
                    let c = &comps[ci as usize];
                    let mut dims: Vec<Option<Dim>> = vec![None; c.dims as usize];
                    let mut elems: Vec<Option<Elem>> = vec![None; c.elems as usize];
                    for (cn, node) in c.graph.nodes.iter().enumerate() {
                        let mn = &g.graph.nodes[map[cn]];
                        for (t, mt) in node.results.iter().zip(&mn.results) {
                            for (d, md) in t.dims.iter().zip(&mt.dims) {
                                if let (ADim::Var(k), ADim::Var(mk)) = (d, md) {
                                    dims[*k as usize] = Some(g.dims[*mk as usize].clone());
                                }
                            }
                            elems[t.elem as usize] = Some(g.elems[mt.elem as usize].clone());
                        }
                    }
                    Ok(Member {
                        comp: ci,
                        dims: dims
                            .into_iter()
                            .map(|d| d.ok_or("unbound dim class"))
                            .collect::<Result<_, _>>()?,
                        elems: elems
                            .into_iter()
                            .map(|e| e.ok_or("unbound element class"))
                            .collect::<Result<_, _>>()?,
                        params: map.iter().map(|&m| g.params[m].clone()).collect(),
                        refs: map.iter().map(|&m| g.refs[m]).collect(),
                        map,
                    })
                })
                .collect::<Result<Vec<_>, &str>>()
                .map_err(String::from)
        })
        .collect::<Result<_, _>>()?;
    Ok((comps, members))
}

type Reps = Vec<(usize, Vec<(Key, usize, Vec<usize>)>)>;

/// Split one bucket into isomorphism classes: for every member, the first representative it matches,
/// with the map from the representative's nodes to the member's.
fn bucket(
    phases: &[PhaseOut],
    comp_of: &[Vec<Option<u32>>],
    members: &[Key],
    budget: u64,
) -> Result<Reps, String> {
    let mut reps: Reps = Vec::new();
    for &(pi, gi) in members {
        let m = &phases[pi].graphs[gi as usize];
        let mut placed = false;
        for (ri, (_, list)) in reps.iter_mut().enumerate() {
            let (rp, rg) = list[0].0;
            let r = &phases[rp].graphs[rg as usize];
            let child =
                |p: usize, g: &super::PhaseGraph, n: usize| g.children[n].map(|c| comp_of[p][c as usize]);
            let eq = |i: usize, j: usize| {
                r.colors[i] == m.colors[j]
                    && r.graph.nodes[i].same_key(&m.graph.nodes[j])
                    && child(rp, r, i) == child(pi, m, j)
            };
            let t = std::time::Instant::now();
            let found = isomorphism(&r.graph, &m.graph, eq, budget);
            if t.elapsed().as_millis() > 500 {
                tracing::debug!(
                    nodes = m.graph.node_count(),
                    ms = t.elapsed().as_millis() as u64,
                    matched = matches!(found, Ok(Some(_))),
                    "slow isomorphism search"
                );
            }
            match found {
                Ok(Some(map)) => {
                    list.push(((pi, gi), ri, map));
                    placed = true;
                    break;
                }
                Ok(None) => {}
                Err(BudgetExceeded(b)) => {
                    return Err(format!(
                        "isomorphism search over {b} candidate pairs ({}/{} graph {gi}, {} nodes); raise vf2_budget",
                        phases[pi].model,
                        phases[pi].phase,
                        m.graph.node_count()
                    ));
                }
            }
        }
        if !placed {
            let id: Vec<usize> = (0..m.graph.node_count()).collect();
            let k = reps.len();
            reps.push((k, vec![((pi, gi), k, id)]));
        }
    }
    Ok(reps)
}
