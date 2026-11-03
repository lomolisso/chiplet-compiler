//! Re-matching (docs/passes/mine.md §4.3): every embedding of a pattern in every unit of its family by
//! VF2 in its non-induced mode (GraMi's semantics: labels and edge ends equal, extra edges between the
//! images allowed), the MNI support from all embeddings, and the convex occurrences (one per image).

use std::collections::{BTreeMap, HashMap, HashSet};

use cc_ir::absgraph::{Graph, monomorphisms};

use crate::datapaths::json::{Datapath, Src};
use crate::mine::json::{Feed, OccInput, Occurrence, Unit};

/// Where an operand of a unit vertex comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgSrc {
    /// A vertex of the unit (local index).
    Local(u32),
    Outside(Feed),
}

/// A unit prepared for matching.
#[derive(Debug, Clone)]
pub struct UnitCtx {
    pub unit: Unit,
    pub graph: Graph<u32>,
    /// Per local vertex, the source of each operand.
    pub args: Vec<Vec<ArgSrc>>,
    /// Node label → count.
    pub counts: HashMap<u32, u32>,
    /// Per local vertex, its descendants (bitset, itself excluded).
    pub desc: Vec<Vec<u64>>,
}

fn words(n: usize) -> usize {
    n.div_ceil(64)
}

impl UnitCtx {
    pub fn new(unit: Unit, graph: Graph<u32>, dp: &Datapath) -> Result<Self, String> {
        let local: HashMap<u32, u32> = unit
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| (*v, i as u32))
            .collect();
        let by_id: HashMap<u32, &crate::datapaths::json::Vertex> =
            dp.vertices.iter().map(|v| (v.id, v)).collect();
        let mut args = Vec::new();
        for v in &unit.vertices {
            let vx = by_id
                .get(v)
                .ok_or_else(|| format!("{}: vertex {v} not in the datapath", unit.id))?;
            args.push(
                vx.args
                    .iter()
                    .map(|a| match &a.src {
                        Src::Vertex { v: u, .. } => match local.get(u) {
                            Some(&l) => ArgSrc::Local(l),
                            None if by_id.get(u).is_some_and(|x| x.op == "const") => {
                                ArgSrc::Outside(Feed::Const)
                            }
                            None => ArgSrc::Outside(Feed::Vertex),
                        },
                        Src::Input { .. } => ArgSrc::Outside(Feed::Input),
                        Src::Acc { .. } => ArgSrc::Outside(Feed::Acc),
                        Src::Result { .. } => ArgSrc::Outside(Feed::Result),
                    })
                    .collect(),
            );
        }
        Self::with_args(unit, graph, args)
    }

    /// A unit from its graph alone: no operand sources (occurrences then list no feeds).
    pub fn bare(unit: Unit, graph: Graph<u32>) -> Result<Self, String> {
        let args = vec![Vec::new(); graph.node_count()];
        Self::with_args(unit, graph, args)
    }

    fn with_args(unit: Unit, graph: Graph<u32>, args: Vec<Vec<ArgSrc>>) -> Result<Self, String> {
        let mut counts = HashMap::new();
        for l in &graph.nodes {
            *counts.entry(*l).or_insert(0) += 1;
        }
        let n = graph.node_count();
        // Descendants in reverse topological order (the datapath lists vertices in topological order,
        // and a unit keeps that order).
        let mut desc = vec![vec![0u64; words(n)]; n];
        for v in (0..n).rev() {
            for (w, _) in &graph.out[v] {
                let w = *w as usize;
                if w <= v {
                    return Err(format!("{}: vertices not in topological order", unit.id));
                }
                let (lo, hi) = desc.split_at_mut(w);
                let dv = &mut lo[v];
                dv[w / 64] |= 1 << (w % 64);
                for (a, b) in dv.iter_mut().zip(&hi[0]) {
                    *a |= *b;
                }
            }
        }
        Ok(Self {
            unit,
            graph,
            args,
            counts,
            desc,
        })
    }

    /// Whether the vertex set `s` (sorted local ids) is convex: no path leaves it and comes back.
    pub fn convex(&self, s: &[u32]) -> bool {
        let n = self.graph.node_count();
        let mut inside = vec![0u64; words(n)];
        for &v in s {
            inside[v as usize / 64] |= 1 << (v % 64);
        }
        let mut below = vec![0u64; words(n)];
        for &v in s {
            for (a, b) in below.iter_mut().zip(&self.desc[v as usize]) {
                *a |= *b;
            }
        }
        for w in 0..n {
            let bit = 1u64 << (w % 64);
            if below[w / 64] & bit != 0 && inside[w / 64] & bit == 0 {
                let back = self.desc[w].iter().zip(&inside).any(|(a, b)| a & b != 0);
                if back {
                    return false;
                }
            }
        }
        true
    }
}

/// The re-match of one pattern over a family.
#[derive(Debug, Clone, Default)]
pub struct Matched {
    pub mni: u64,
    pub embeddings: u64,
    pub non_convex: u64,
    /// Convex occurrences, sorted by (unit, image).
    pub occurrences: Vec<Occurrence>,
    /// Per occurrence: its unit index and its image (sorted local ids).
    pub at: Vec<(usize, Vec<u32>)>,
}

/// Re-match pattern `p` (node labels are label ids) over `units`.
pub fn rematch(p: &Graph<u32>, units: &[UnitCtx], budget: u64) -> Result<Matched, String> {
    let mut need: HashMap<u32, u32> = HashMap::new();
    for l in &p.nodes {
        *need.entry(*l).or_insert(0) += 1;
    }
    let mut images: Vec<HashSet<(usize, u32)>> = vec![HashSet::new(); p.node_count()];
    let mut out = Matched::default();
    let mut occ: Vec<(usize, Vec<u32>, Occurrence)> = Vec::new();
    for (ui, u) in units.iter().enumerate() {
        if need
            .iter()
            .any(|(l, c)| u.counts.get(l).copied().unwrap_or(0) < *c)
        {
            continue;
        }
        // First map per image, in VF2 order.
        let mut first: BTreeMap<Vec<u32>, Vec<u32>> = BTreeMap::new();
        let n = monomorphisms(
            p,
            &u.graph,
            |i, j| p.nodes[i] == u.graph.nodes[j],
            budget,
            |m| {
                for (i, &j) in m.iter().enumerate() {
                    images[i].insert((ui, j as u32));
                }
                let map: Vec<u32> = m.iter().map(|&j| j as u32).collect();
                let mut img = map.clone();
                img.sort_unstable();
                first.entry(img).or_insert(map);
            },
        )
        .map_err(|b| format!("{}: VF2 over its budget ({} pairs)", u.unit.id, b.0))?;
        out.embeddings += n;
        for (img, map) in first {
            if !u.convex(&img) {
                out.non_convex += 1;
                continue;
            }
            let o = occurrence(p, u, &map);
            occ.push((ui, img, o));
        }
    }
    out.mni = images.iter().map(|s| s.len() as u64).min().unwrap_or(0);
    occ.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    for (ui, img, o) in occ {
        out.at.push((ui, img));
        out.occurrences.push(o);
    }
    Ok(out)
}

/// The occurrence of map `map` (local ids): its datapath vertices and the operands fed from outside
/// (an operand is inside iff the pattern has the edge from the vertex that feeds it).
pub fn occurrence(p: &Graph<u32>, u: &UnitCtx, map: &[u32]) -> Occurrence {
    let pre: HashMap<u32, u32> = map.iter().enumerate().map(|(i, &j)| (j, i as u32)).collect();
    let mut inputs = Vec::new();
    for (i, &j) in map.iter().enumerate() {
        for (k, a) in u.args[j as usize].iter().enumerate() {
            let feed = match a {
                ArgSrc::Local(l) => match pre.get(l) {
                    Some(&pi) if p.edge(pi, i as u32).is_some() => continue,
                    _ => Feed::Vertex,
                },
                ArgSrc::Outside(f) => *f,
            };
            inputs.push(OccInput {
                at: i as u32,
                arg: k as u32,
                feed,
            });
        }
    }
    Occurrence {
        unit: u.unit.id.clone(),
        map: map.iter().map(|&j| u.unit.vertices[j as usize]).collect(),
        inputs,
    }
}
