//! The component as a DAG of *units*: every op that owns regions together with its region nodes
//! (arguments, body ops, terminators) is one unit, every other node is a unit of its own. The `region`
//! edges (owner to argument) and `yield` edges (terminator to owner) then lie inside units, and the unit
//! graph is acyclic. Units are numbered in a topological order that takes the smallest node id first
//! (SSA order: ids order definitions before uses).

use std::collections::{BTreeSet, VecDeque};

use cc_ir::absgraph::{Graph, PortLabel};
use cc_ir::ops::{self, OpClass};

use crate::components::algo::graph::{ADim, Kind};
use crate::kernels::json::BoundaryKind;
use crate::normalize::json::NNode;

/// What a unit can become (design §4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UClass {
    /// A contraction, reduction or scatter-reduce.
    Core,
    /// An element-wise op.
    Elementwise,
    /// A `broadcast_in_dim` (static or dynamic): absorbed into a kernel or a layout boundary.
    Broadcast,
    Boundary(BoundaryKind),
}

#[derive(Debug, Clone)]
pub struct Units {
    /// The representative node of each unit (the owner for a region-owning op), in unit order.
    pub rep: Vec<u32>,
    /// The nodes of each unit, ascending.
    pub nodes: Vec<Vec<u32>>,
    /// The unit of every node.
    pub of: Vec<u32>,
    /// Predecessor and successor units (sorted, no duplicates, no self loops).
    pub pred: Vec<Vec<u32>>,
    pub succ: Vec<Vec<u32>>,
    pub class: Vec<UClass>,
}

impl Units {
    pub fn len(&self) -> usize {
        self.rep.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rep.is_empty()
    }
}

/// Whether an edge end is a region link (`region` from an owner, `yield` to an owner).
pub fn is_region_end(p: PortLabel) -> bool {
    matches!(p, PortLabel::Region { .. } | PortLabel::Yield { .. })
}

/// The owner of every node inside a region, `None` elsewhere. A region's nodes are those reachable from
/// its owner's arguments (forward) or reaching its terminators (backward), without passing the owner,
/// minus any node with a consumer outside (a value captured from the enclosing scope stays outside).
/// Nested regions belong to the outermost owner.
pub fn region_owners(g: &Graph<NNode>) -> Vec<Option<u32>> {
    let n = g.node_count();
    let mut owners: BTreeSet<u32> = BTreeSet::new();
    for (u, out) in g.out.iter().enumerate() {
        for (v, ends) in out {
            for e in ends {
                match e.port {
                    PortLabel::Region { .. } => {
                        owners.insert(u as u32);
                    }
                    PortLabel::Yield { .. } => {
                        owners.insert(*v);
                    }
                    _ => {}
                }
            }
        }
    }
    let mut body: Vec<(u32, BTreeSet<u32>)> = Vec::new();
    for &o in &owners {
        let mut set: BTreeSet<u32> = BTreeSet::new();
        let mut q: VecDeque<u32> = VecDeque::new();
        // Forward from the arguments.
        for (v, ends) in &g.out[o as usize] {
            if ends.iter().any(|e| matches!(e.port, PortLabel::Region { .. })) && set.insert(*v) {
                q.push_back(*v);
            }
        }
        while let Some(u) = q.pop_front() {
            for (v, ends) in &g.out[u as usize] {
                if *v != o
                    && !ends.iter().all(|e| matches!(e.port, PortLabel::Yield { .. }))
                    && set.insert(*v)
                {
                    q.push_back(*v);
                }
            }
        }
        // Backward from the terminators.
        for &t in &g.inc[o as usize] {
            let yields = g
                .edge(t, o)
                .is_some_and(|l| l.iter().any(|e| matches!(e.port, PortLabel::Yield { .. })));
            if yields && set.insert(t) {
                q.push_back(t);
            }
        }
        while let Some(v) = q.pop_front() {
            for &u in &g.inc[v as usize] {
                if u != o && set.insert(u) {
                    q.push_back(u);
                }
            }
        }
        // Interface nodes and captured values stay outside.
        loop {
            let drop: Vec<u32> = set
                .iter()
                .copied()
                .filter(|&u| {
                    matches!(
                        g.nodes[u as usize].label.kind,
                        Kind::Input { .. } | Kind::Output | Kind::SizeIn
                    ) || g.out[u as usize].iter().any(|(v, _)| *v != o && !set.contains(v))
                })
                .collect();
            if drop.is_empty() {
                break;
            }
            for u in drop {
                set.remove(&u);
            }
        }
        body.push((o, set));
    }
    // Outermost owner: an owner inside another owner's region takes that owner's.
    let mut direct: Vec<Option<u32>> = vec![None; n];
    for (o, set) in &body {
        for &u in set {
            // Keep the owner with the larger region (the outer one contains the inner).
            let replace = match direct[u as usize] {
                None => true,
                Some(p) => body
                    .iter()
                    .find(|(x, _)| *x == p)
                    .is_some_and(|(_, s)| s.len() < set.len()),
            };
            if replace {
                direct[u as usize] = Some(*o);
            }
        }
    }
    let mut top = direct.clone();
    for u in 0..n {
        let mut cur = direct[u];
        let mut guard = 0;
        while let Some(o) = cur {
            match direct[o as usize] {
                Some(p) if guard < n => {
                    cur = Some(p);
                    guard += 1;
                }
                _ => break,
            }
        }
        top[u] = cur;
    }
    top
}

/// Whether a `scatter`'s update computation is arithmetic: its terminator yields a value computed by
/// an op rather than a region argument (an overwrite yields the update argument itself).
pub fn scatter_reduces(g: &Graph<NNode>, s: u32) -> bool {
    g.inc[s as usize].iter().any(|&t| {
        let yields = g
            .edge(t, s)
            .is_some_and(|l| l.iter().any(|e| matches!(e.port, PortLabel::Yield { .. })));
        yields
            && g.inc[t as usize]
                .iter()
                .any(|&p| !matches!(g.nodes[p as usize].label.kind, Kind::RegionArg { .. }))
    })
}

/// The class of a unit from its representative's op-table entry.
pub fn classify(g: &Graph<NNode>, rep: u32) -> Result<UClass, String> {
    let n = &g.nodes[rep as usize].label;
    let (name, target) = match &n.kind {
        Kind::Input { .. } | Kind::Output | Kind::SizeIn => {
            return Ok(UClass::Boundary(BoundaryKind::Interface));
        }
        Kind::RegionArg { .. } => return Err(format!("node {rep}: region argument outside every region")),
        Kind::Op { name, target } => (name.as_str(), target.as_deref()),
    };
    if n.is_call() {
        return Ok(UClass::Boundary(BoundaryKind::Call));
    }
    let spec =
        ops::lookup(name, target).ok_or_else(|| format!("node {rep}: `{name}` is not in the op table"))?;
    Ok(match spec.class {
        OpClass::Core => UClass::Core,
        OpClass::Elementwise => UClass::Elementwise,
        OpClass::Layout if is_broadcast(name) => UClass::Broadcast,
        OpClass::Layout => UClass::Boundary(BoundaryKind::Layout),
        OpClass::DataMovement if name == "stablehlo.scatter" && scatter_reduces(g, rep) => UClass::Core,
        OpClass::DataMovement => UClass::Boundary(BoundaryKind::DataMovement),
        OpClass::Size => UClass::Boundary(BoundaryKind::Size),
        OpClass::Control => UClass::Boundary(BoundaryKind::Control),
        OpClass::Irregular => UClass::Boundary(BoundaryKind::Irregular),
        OpClass::Constant => UClass::Boundary(BoundaryKind::Constant),
    })
}

pub fn is_broadcast(name: &str) -> bool {
    matches!(
        name,
        "stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim"
    )
}

/// The dims of result `res` of a node.
pub fn dims(g: &Graph<NNode>, node: u32, res: u32) -> Option<&[ADim]> {
    g.nodes[node as usize]
        .label
        .results
        .get(res as usize)
        .map(|t| t.dims.as_slice())
}

/// Contract regions into their owners and order the units.
pub fn units(g: &Graph<NNode>) -> Result<Units, String> {
    let n = g.node_count();
    let owner = region_owners(g);
    let key: Vec<u32> = (0..n as u32).map(|u| owner[u as usize].unwrap_or(u)).collect();
    // Unit adjacency by representative.
    let reps: BTreeSet<u32> = key.iter().copied().collect();
    let mut succ_r: std::collections::BTreeMap<u32, BTreeSet<u32>> =
        reps.iter().map(|&r| (r, BTreeSet::new())).collect();
    let mut indeg: std::collections::BTreeMap<u32, usize> = reps.iter().map(|&r| (r, 0)).collect();
    for (u, out) in g.out.iter().enumerate() {
        let a = key[u];
        for (v, _) in out {
            let b = key[*v as usize];
            if a != b && succ_r.get_mut(&a).is_some_and(|s| s.insert(b)) {
                *indeg.entry(b).or_default() += 1;
            }
        }
    }
    // Kahn, smallest representative first.
    let mut ready: BTreeSet<u32> = indeg.iter().filter(|(_, d)| **d == 0).map(|(r, _)| *r).collect();
    let mut order: Vec<u32> = Vec::with_capacity(reps.len());
    while let Some(r) = ready.pop_first() {
        order.push(r);
        for &s in &succ_r[&r] {
            if let Some(d) = indeg.get_mut(&s) {
                *d -= 1;
                if *d == 0 {
                    ready.insert(s);
                }
            }
        }
    }
    if order.len() != reps.len() {
        return Err("the graph with regions contracted has a cycle".into());
    }
    let mut index = vec![u32::MAX; n];
    for (i, &r) in order.iter().enumerate() {
        index[r as usize] = i as u32;
    }
    let of: Vec<u32> = key.iter().map(|&r| index[r as usize]).collect();
    let mut nodes: Vec<Vec<u32>> = vec![Vec::new(); order.len()];
    for u in 0..n {
        nodes[of[u] as usize].push(u as u32);
    }
    let mut pred: Vec<Vec<u32>> = vec![Vec::new(); order.len()];
    let mut succ: Vec<Vec<u32>> = vec![Vec::new(); order.len()];
    for (i, &r) in order.iter().enumerate() {
        let mut s: Vec<u32> = succ_r[&r].iter().map(|b| index[*b as usize]).collect();
        s.sort_unstable();
        for &b in &s {
            pred[b as usize].push(i as u32);
        }
        succ[i] = s;
    }
    for p in &mut pred {
        p.sort_unstable();
    }
    let class = order
        .iter()
        .map(|&r| classify(g, r))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Units {
        rep: order,
        nodes,
        of,
        pred,
        succ,
        class,
    })
}
