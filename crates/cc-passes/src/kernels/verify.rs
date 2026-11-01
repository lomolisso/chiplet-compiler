//! Independent checks of kernels (design §4.5, §5). They read the normalized components and the written
//! artifacts and never call the partition code: they work on the node graph itself (no unit
//! contraction, no ancestor bitsets), with breadth-first reachability and their own op-table reading.
//!
//! - `complete`: every normalized component has a partition and kernel graphs.
//! - `<id>:partition`: every node is in exactly one kernel or is a boundary node; every `region` and
//!   `yield` edge lies inside one part; region nodes of a boundary op name it as owner.
//! - `<id>:classes`: every member's role fits its op class (core: a core op or an arithmetic scatter;
//!   prologue, epilogue, map: element-wise; broadcast: `broadcast_in_dim`), every boundary kind is the
//!   node's class, no core or element-wise node is a boundary.
//! - `<id>:k1`: at most one core per kernel, and only kernels with a core have prologues and epilogues.
//! - `<id>:k2`: every kernel is convex: no path leaves it and re-enters.
//! - `<id>:k3`: a member whose value is consumed inside the kernel has all its consumers inside.
//! - `<id>:k4`: epilogue members have a core result's dims and descend from the core inside the kernel;
//!   prologue members have every consumer inside, reach exactly one core operand port through prologue
//!   and broadcast members and have that operand's dims; map members have the kernel's domain;
//!   broadcasts read only outside values or other broadcasts and feed only members.
//! - `<id>:quotient`: the quotient graph (kernels and boundary ops as nodes) is acyclic and the stored
//!   order is a topological order of it.
//! - `<id>:reconstruct`: the kernel graphs (members with their labels, input and output nodes, their
//!   edges), the kernels' crossing edges and the boundary nodes rebuild the normalized graph exactly.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use cc_core::{Check, VerifyReport};
use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};
use cc_ir::ops::{self, OpClass};
use cc_pipeline::{PassContext, PassError};
use rayon::prelude::*;

use super::json::{BoundaryKind, Crossing, KNode, KernelGraphs, KernelPartition, Part, Role};
use super::{KERNELS, component_ids};
use crate::components::algo::graph::{ADim, Kind};
use crate::normalize::NORMALIZE;
use crate::normalize::json::{NNode, NormalizedComponent};

/// The checks per component, in report order.
pub const CHECKS: [&str; 8] = [
    "partition",
    "classes",
    "k1",
    "k2",
    "k3",
    "k4",
    "quotient",
    "reconstruct",
];

fn check(name: impl Into<String>, notes: &[String], counts: &[(&str, u64)]) -> Check {
    let mut c = Check::new(name, notes.is_empty());
    for (k, v) in counts {
        c = c.count(*k, *v);
    }
    for n in notes.iter().take(20) {
        c = c.note(n.clone());
    }
    c
}

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let out = ctx.output(KERNELS)?;
    let input = ctx.input(NORMALIZE)?;
    let mut r = VerifyReport::new(ctx.name());
    let ids = component_ids(ctx)?;
    let mut missing = Vec::new();
    for id in &ids {
        for f in [format!("components/{id}.json"), format!("kernels/{id}.postcard")] {
            if !out.path(&f)?.is_file() {
                missing.push(format!("{id}: no {f}"));
            }
        }
    }
    r.push(check("complete", &missing, &[("components", ids.len() as u64)]));
    let per: Vec<Vec<Check>> = ids
        .par_iter()
        .filter(|id| !missing.iter().any(|m| m.starts_with(&format!("{id}:"))))
        .map(|id| -> Result<Vec<Check>, PassError> {
            let c: NormalizedComponent = input.read_postcard(&format!("components/{id}.postcard"))?;
            let p: KernelPartition = out.read_json(&format!("components/{id}.json"))?;
            let kg: KernelGraphs = out.read_postcard(&format!("kernels/{id}.postcard"))?;
            let res = check_component(&c.graph, &p, &kg);
            Ok(CHECKS
                .iter()
                .map(|name| {
                    check(
                        format!("{id}:{name}"),
                        res.get(name).map(Vec::as_slice).unwrap_or(&[]),
                        &[("kernels", p.kernels.len() as u64)],
                    )
                })
                .collect())
        })
        .collect::<Result<_, _>>()?;
    for cs in per {
        for c in cs {
            r.push(c);
        }
    }
    Ok(r)
}

/// What a node is, read from the op table (this module's own reading).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NClass {
    Interface,
    RegionArg,
    Call,
    Table(OpClass),
    ScatterReduce,
}

fn is_yield(l: &[EdgeEnd]) -> bool {
    l.iter().any(|e| matches!(e.port, PortLabel::Yield { .. }))
}

fn nclass(g: &Graph<NNode>, n: u32) -> Result<NClass, String> {
    let x = &g.nodes[n as usize].label;
    match &x.kind {
        Kind::Input { .. } | Kind::Output | Kind::SizeIn => Ok(NClass::Interface),
        Kind::RegionArg { .. } => Ok(NClass::RegionArg),
        Kind::Op { name, target } => {
            if name == "stablehlo.composite" {
                return Ok(NClass::Call);
            }
            let e =
                ops::lookup(name, target.as_deref()).ok_or_else(|| format!("node {n}: `{name}` unknown"))?;
            if name == "stablehlo.scatter" {
                // The update computation: the value the scatter's terminator yields.
                for &t in &g.inc[n as usize] {
                    if g.edge(t, n).is_some_and(|l| is_yield(l))
                        && g.inc[t as usize]
                            .iter()
                            .any(|&v| !matches!(g.nodes[v as usize].label.kind, Kind::RegionArg { .. }))
                    {
                        return Ok(NClass::ScatterReduce);
                    }
                }
            }
            Ok(NClass::Table(e.class))
        }
    }
}

fn op_name(g: &Graph<NNode>, n: u32) -> &str {
    match &g.nodes[n as usize].label.kind {
        Kind::Op { name, .. } => name,
        _ => "",
    }
}

fn dims_of(g: &Graph<NNode>, n: u32, r: u32) -> Option<&Vec<ADim>> {
    g.nodes[n as usize].label.results.get(r as usize).map(|t| &t.dims)
}

/// Data consumers of `n`: out-neighbours over at least one end that is not a region link.
fn consumers(g: &Graph<NNode>, n: u32) -> impl Iterator<Item = u32> + '_ {
    g.out[n as usize]
        .iter()
        .filter(|(_, l)| {
            l.iter()
                .any(|e| !matches!(e.port, PortLabel::Region { .. } | PortLabel::Yield { .. }))
        })
        .map(|(v, _)| *v)
}

/// Run every check on one component; failures per check name.
#[derive(Default)]
struct Bad(BTreeMap<&'static str, Vec<String>>);

impl Bad {
    fn add(&mut self, c: &'static str, m: String) {
        self.0.entry(c).or_default().push(m);
    }
}

pub fn check_component(
    g: &Graph<NNode>,
    p: &KernelPartition,
    kg: &KernelGraphs,
) -> BTreeMap<&'static str, Vec<String>> {
    let mut bad = Bad::default();
    let n = g.node_count();

    // Partition.
    let mut part: Vec<Option<Part>> = vec![None; n];
    let mut role: Vec<Option<Role>> = vec![None; n];
    let mut place = |x: u32, pt: Part, bad: &mut Bad| {
        if x as usize >= n {
            bad.add("partition", format!("node {x} does not exist"));
        } else if part[x as usize].is_some() {
            bad.add("partition", format!("node {x} is placed twice"));
        } else {
            part[x as usize] = Some(pt);
        }
    };
    for (k, kr) in p.kernels.iter().enumerate() {
        for m in &kr.members {
            place(m.node, Part::Kernel(k as u32), &mut bad);
            if (m.node as usize) < n {
                role[m.node as usize] = Some(m.role);
            }
        }
    }
    let mut bkind: BTreeMap<u32, (BoundaryKind, Option<u32>)> = BTreeMap::new();
    for b in &p.boundaries {
        place(b.node, Part::Boundary(b.owner.unwrap_or(b.node)), &mut bad);
        bkind.insert(b.node, (b.kind, b.owner));
    }
    for (x, pt) in part.iter().enumerate() {
        if pt.is_none() {
            bad.add("partition", format!("node {x} is in no kernel and no boundary"));
        }
    }
    if bad.0.contains_key("partition") {
        return bad.0;
    }
    let part: Vec<Part> = part.into_iter().flatten().collect();
    for (&b, &(kind, owner)) in &bkind {
        if let Some(o) = owner {
            match bkind.get(&o) {
                Some(&(k, None)) if k == kind => {}
                _ => bad.add(
                    "partition",
                    format!("boundary {b}: owner {o} is not a boundary of kind {kind:?}"),
                ),
            }
        }
    }
    for (u, out) in g.out.iter().enumerate() {
        for (v, l) in out {
            let link = l
                .iter()
                .any(|e| matches!(e.port, PortLabel::Region { .. } | PortLabel::Yield { .. }));
            if link && part[u] != part[*v as usize] {
                bad.add("partition", format!("region edge {u} -> {v} crosses parts"));
            }
        }
    }
    for x in 0..n as u32 {
        if matches!(g.nodes[x as usize].label.kind, Kind::RegionArg { .. })
            && role[x as usize].is_none_or(|r| r != Role::Body)
            && bkind.get(&x).is_none_or(|b| b.1.is_none())
        {
            bad.add("partition", format!("region argument {x} is not with its owner"));
        }
    }

    // Classes.
    let mut cls: Vec<Option<NClass>> = vec![None; n];
    for (x, c) in cls.iter_mut().enumerate() {
        match nclass(g, x as u32) {
            Ok(k) => *c = Some(k),
            Err(e) => bad.add("classes", e),
        }
    }
    for (k, kr) in p.kernels.iter().enumerate() {
        for m in &kr.members {
            let c = cls[m.node as usize];
            let ok = match m.role {
                Role::Core => matches!(c, Some(NClass::Table(OpClass::Core) | NClass::ScatterReduce)),
                Role::Prologue | Role::Epilogue | Role::Map => c == Some(NClass::Table(OpClass::Elementwise)),
                Role::Broadcast => matches!(
                    op_name(g, m.node),
                    "stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim"
                ),
                Role::Body => true,
            };
            if !ok {
                bad.add(
                    "classes",
                    format!("k{k}: node {} as {:?} is {:?}", m.node, m.role, c),
                );
            }
        }
    }
    for (&b, &(kind, owner)) in &bkind {
        if owner.is_some() {
            continue;
        }
        let want = match cls[b as usize] {
            Some(NClass::Interface) => Some(BoundaryKind::Interface),
            Some(NClass::Call) => Some(BoundaryKind::Call),
            Some(NClass::Table(OpClass::Layout)) => Some(BoundaryKind::Layout),
            Some(NClass::Table(OpClass::DataMovement)) => Some(BoundaryKind::DataMovement),
            Some(NClass::Table(OpClass::Size)) => Some(BoundaryKind::Size),
            Some(NClass::Table(OpClass::Control)) => Some(BoundaryKind::Control),
            Some(NClass::Table(OpClass::Irregular)) => Some(BoundaryKind::Irregular),
            Some(NClass::Table(OpClass::Constant)) => Some(BoundaryKind::Constant),
            _ => None,
        };
        if want != Some(kind) {
            bad.add(
                "classes",
                format!("boundary {b}: kind {kind:?}, class {:?}", cls[b as usize]),
            );
        }
    }

    // K1 and the core field.
    for (k, kr) in p.kernels.iter().enumerate() {
        let cores: Vec<u32> = kr
            .members
            .iter()
            .filter(|m| m.role == Role::Core)
            .map(|m| m.node)
            .collect();
        if cores.len() > 1 {
            bad.add("k1", format!("k{k}: {} cores {cores:?}", cores.len()));
        }
        if kr.core != cores.first().copied() {
            bad.add(
                "k1",
                format!("k{k}: core field {:?}, core members {cores:?}", kr.core),
            );
        }
        let fused = kr
            .members
            .iter()
            .any(|m| matches!(m.role, Role::Prologue | Role::Epilogue));
        let mapped = kr.members.iter().any(|m| m.role == Role::Map);
        if cores.is_empty() && fused || !cores.is_empty() && mapped {
            bad.add("k1", format!("k{k}: roles do not match the presence of a core"));
        }
    }

    // K2: forward search from the kernel through outside nodes only must not enter it again.
    for (k, kr) in p.kernels.iter().enumerate() {
        let me = Part::Kernel(k as u32);
        let mut seen = vec![false; n];
        let mut q: VecDeque<u32> = VecDeque::new();
        for m in &kr.members {
            for (v, _) in &g.out[m.node as usize] {
                if part[*v as usize] != me && !seen[*v as usize] {
                    seen[*v as usize] = true;
                    q.push_back(*v);
                }
            }
        }
        let mut reenter = None;
        while let Some(x) = q.pop_front() {
            for (v, _) in &g.out[x as usize] {
                if part[*v as usize] == me {
                    reenter = Some((x, *v));
                    break;
                }
                if !seen[*v as usize] {
                    seen[*v as usize] = true;
                    q.push_back(*v);
                }
            }
            if reenter.is_some() {
                break;
            }
        }
        if let Some((x, v)) = reenter {
            bad.add(
                "k2",
                format!("k{k}: a path leaves the kernel and re-enters it through {x} -> {v}"),
            );
        }
    }

    // K3.
    for (k, kr) in p.kernels.iter().enumerate() {
        let me = Part::Kernel(k as u32);
        for m in &kr.members {
            let cs: Vec<u32> = consumers(g, m.node).collect();
            let inside = cs.iter().any(|&c| part[c as usize] == me);
            let outside: Vec<u32> = cs.iter().copied().filter(|&c| part[c as usize] != me).collect();
            if inside && !outside.is_empty() {
                bad.add(
                    "k3",
                    format!("k{k}: node {} is consumed inside and by {outside:?}", m.node),
                );
            }
        }
    }

    // K4.
    for (k, kr) in p.kernels.iter().enumerate() {
        let me = Part::Kernel(k as u32);
        let inside = |x: u32| part[x as usize] == me;
        match kr.core {
            Some(c) => {
                let domains: Vec<&Vec<ADim>> = g.nodes[c as usize]
                    .label
                    .results
                    .iter()
                    .map(|t| &t.dims)
                    .collect();
                if dims_of(g, c, 0) != Some(&kr.domain) {
                    bad.add("k4", format!("k{k}: domain is not the core's result dims"));
                }
                // Descendants of the core inside the kernel.
                let mut desc = BTreeSet::new();
                let mut q = VecDeque::from([c]);
                while let Some(x) = q.pop_front() {
                    for v in consumers(g, x) {
                        if inside(v) && desc.insert(v) {
                            q.push_back(v);
                        }
                    }
                }
                // Core ports reached by each prologue or broadcast member.
                let mut memo: BTreeMap<u32, Option<BTreeSet<u32>>> = BTreeMap::new();
                for m in &kr.members {
                    match m.role {
                        Role::Epilogue => {
                            if !dims_of(g, m.node, 0).is_some_and(|d| domains.contains(&d)) {
                                bad.add(
                                    "k4",
                                    format!("k{k}: epilogue {} is not over a core result's domain", m.node),
                                );
                            }
                            if !desc.contains(&m.node) {
                                bad.add(
                                    "k4",
                                    format!("k{k}: epilogue {} does not descend from the core", m.node),
                                );
                            }
                        }
                        Role::Prologue => {
                            let ports = core_ports(g, &role, c, m.node, &inside, &mut memo);
                            match ports.as_ref().map(|s| s.iter().copied().collect::<Vec<_>>()) {
                                Some(v) if v.len() == 1 => {
                                    let want = g.inc[c as usize].iter().find_map(|&s| {
                                        g.edge(s, c)?
                                            .iter()
                                            .find(|e| e.port == PortLabel::Port(v[0]))
                                            .and_then(|e| dims_of(g, s, e.res))
                                    });
                                    if dims_of(g, m.node, 0) != want {
                                        bad.add(
                                            "k4",
                                            format!(
                                                "k{k}: prologue {} does not have its operand's dims",
                                                m.node
                                            ),
                                        );
                                    }
                                }
                                v => bad.add(
                                    "k4",
                                    format!(
                                        "k{k}: prologue {} reaches core ports {v:?}, not exactly one",
                                        m.node
                                    ),
                                ),
                            }
                        }
                        _ => {}
                    }
                }
            }
            None => {
                for m in &kr.members {
                    if m.role == Role::Map && dims_of(g, m.node, 0) != Some(&kr.domain) {
                        bad.add(
                            "k4",
                            format!("k{k}: map member {} is not over the kernel's domain", m.node),
                        );
                    }
                }
            }
        }
        for m in &kr.members {
            if m.role == Role::Broadcast
                && (g.inc[m.node as usize]
                    .iter()
                    .any(|&s| inside(s) && role[s as usize] != Some(Role::Broadcast))
                    || consumers(g, m.node).any(|v| !inside(v))
                    || consumers(g, m.node).next().is_none())
            {
                bad.add(
                    "k4",
                    format!(
                        "k{k}: broadcast {} does not read an input into members only",
                        m.node
                    ),
                );
            }
        }
    }

    // Quotient.
    let parts: BTreeSet<Part> = part.iter().copied().collect();
    let mut succ: BTreeMap<Part, BTreeSet<Part>> = parts.iter().map(|q| (*q, BTreeSet::new())).collect();
    for (u, out) in g.out.iter().enumerate() {
        for (v, _) in out {
            let (a, b) = (part[u], part[*v as usize]);
            if a != b {
                succ.entry(a).or_default().insert(b);
            }
        }
    }
    let mut indeg: BTreeMap<Part, usize> = parts.iter().map(|q| (*q, 0)).collect();
    for s in succ.values() {
        for b in s {
            *indeg.entry(*b).or_default() += 1;
        }
    }
    let mut q: VecDeque<Part> = indeg.iter().filter(|(_, d)| **d == 0).map(|(q, _)| *q).collect();
    let mut seen = 0usize;
    while let Some(a) = q.pop_front() {
        seen += 1;
        for b in &succ[&a] {
            if let Some(d) = indeg.get_mut(b) {
                *d -= 1;
                if *d == 0 {
                    q.push_back(*b);
                }
            }
        }
    }
    if seen != parts.len() {
        bad.add(
            "quotient",
            format!(
                "the quotient graph has a cycle ({} parts on or after it)",
                parts.len() - seen
            ),
        );
    }
    let pos: BTreeMap<Part, usize> = p.order.iter().enumerate().map(|(i, q)| (*q, i)).collect();
    if p.order.len() != parts.len() || pos.len() != parts.len() || parts.iter().any(|q| !pos.contains_key(q))
    {
        bad.add(
            "quotient",
            "the stored order does not list every part once".into(),
        );
    } else {
        for (a, s) in &succ {
            for b in s {
                if pos[a] >= pos[b] {
                    bad.add(
                        "quotient",
                        format!("stored order puts {a:?} after its successor {b:?}"),
                    );
                }
            }
        }
    }

    // Reconstruction.
    reconstruct(g, p, kg, &part, &mut |c, m| bad.add(c, m));
    bad.0
}

/// The core operand ports a prologue or broadcast member reaches through prologue and broadcast members;
/// `None` when a consumer is anything else.
fn core_ports(
    g: &Graph<NNode>,
    role: &[Option<Role>],
    c: u32,
    x: u32,
    inside: &dyn Fn(u32) -> bool,
    memo: &mut BTreeMap<u32, Option<BTreeSet<u32>>>,
) -> Option<BTreeSet<u32>> {
    if let Some(r) = memo.get(&x) {
        return r.clone();
    }
    let mut ports = BTreeSet::new();
    let mut ok = true;
    let cs: Vec<u32> = consumers(g, x).collect();
    if cs.is_empty() {
        ok = false;
    }
    for v in cs {
        if v == c {
            for e in g.edge(x, c).into_iter().flatten() {
                if let PortLabel::Port(k) = e.port {
                    ports.insert(k);
                }
            }
        } else if inside(v) && matches!(role[v as usize], Some(Role::Prologue | Role::Broadcast)) {
            match core_ports(g, role, c, v, inside, memo) {
                Some(s) => ports.extend(s),
                None => ok = false,
            }
        } else {
            ok = false;
        }
    }
    let r = ok.then_some(ports);
    memo.insert(x, r.clone());
    r
}

fn reconstruct(
    g: &Graph<NNode>,
    p: &KernelPartition,
    kg: &KernelGraphs,
    part: &[Part],
    fail: &mut dyn FnMut(&'static str, String),
) {
    if kg.kernels.len() != p.kernels.len() {
        fail(
            "reconstruct",
            format!(
                "{} kernel graphs for {} kernels",
                kg.kernels.len(),
                p.kernels.len()
            ),
        );
        return;
    }
    let mut labels: BTreeMap<u32, &NNode> = BTreeMap::new();
    let mut edges: BTreeSet<(u32, u32, EdgeEnd)> = BTreeSet::new();
    for (k, (kr, gr)) in p.kernels.iter().zip(&kg.kernels).enumerate() {
        if gr.id != kr.id {
            fail("reconstruct", format!("k{k}: graph id {}", gr.id));
        }
        let me = Part::Kernel(k as u32);
        // The crossings recorded in the partition are exactly the graph's.
        let mut ins = Vec::new();
        let mut outs = Vec::new();
        for (u, out) in g.out.iter().enumerate() {
            for (v, l) in out {
                for e in l {
                    let c = Crossing {
                        src: u as u32,
                        dst: *v,
                        end: *e,
                    };
                    let (a, b) = (part[u] == me, part[*v as usize] == me);
                    if b && !a {
                        ins.push(c);
                    } else if a && !b {
                        outs.push(c);
                    }
                }
            }
        }
        ins.sort();
        outs.sort();
        if ins != kr.inputs || outs != kr.outputs {
            fail(
                "reconstruct",
                format!("{}: recorded crossings differ from the graph's", kr.id),
            );
        }
        let mut origin: Vec<u32> = Vec::new();
        let mut members = BTreeSet::new();
        for (i, x) in gr.graph.nodes.iter().enumerate() {
            match x {
                KNode::Member {
                    origin: o,
                    role,
                    node,
                } => {
                    origin.push(*o);
                    members.insert((*o, *role));
                    labels.insert(*o, node);
                }
                KNode::Input { src, .. } | KNode::Output { node: src, .. } => {
                    origin.push(*src);
                    let _ = i;
                }
            }
        }
        let want: BTreeSet<(u32, Role)> = kr.members.iter().map(|m| (m.node, m.role)).collect();
        if members != want {
            fail(
                "reconstruct",
                format!("{}: graph members differ from the partition's", kr.id),
            );
        }
        let outputs: BTreeSet<(u32, u32)> = kr.outputs.iter().map(|c| (c.src, c.end.res)).collect();
        let mut seen_out = BTreeSet::new();
        for (a, out) in gr.graph.out.iter().enumerate() {
            for (b, l) in out {
                for e in l {
                    match (&gr.graph.nodes[a], &gr.graph.nodes[*b as usize]) {
                        (KNode::Member { .. }, KNode::Member { .. }) => {
                            edges.insert((origin[a], origin[*b as usize], *e));
                        }
                        (KNode::Input { src, res, .. }, KNode::Member { .. }) => {
                            edges.insert((
                                *src,
                                origin[*b as usize],
                                EdgeEnd {
                                    res: *res,
                                    port: e.port,
                                },
                            ));
                        }
                        (KNode::Member { .. }, KNode::Output { node, res, .. }) => {
                            if *node != origin[a] || *res != e.res {
                                fail("reconstruct", format!("{}: output node mislinked", kr.id));
                            }
                            seen_out.insert((*node, *res));
                        }
                        _ => fail("reconstruct", format!("{}: edge between interface nodes", kr.id)),
                    }
                }
            }
        }
        if seen_out != outputs {
            fail(
                "reconstruct",
                format!("{}: output nodes differ from the crossings", kr.id),
            );
        }
        for c in &kr.outputs {
            if matches!(part[c.dst as usize], Part::Boundary(_)) {
                edges.insert((c.src, c.dst, c.end));
            }
        }
    }
    for (u, out) in g.out.iter().enumerate() {
        for (v, l) in out {
            if matches!(part[u], Part::Boundary(_)) && matches!(part[*v as usize], Part::Boundary(_)) {
                for e in l {
                    edges.insert((u as u32, *v, *e));
                }
            }
        }
    }
    let mut want = BTreeSet::new();
    for (u, out) in g.out.iter().enumerate() {
        for (v, l) in out {
            for e in l {
                want.insert((u as u32, *v, *e));
            }
        }
    }
    if edges != want {
        let extra = edges.difference(&want).count();
        let lost = want.difference(&edges).count();
        fail(
            "reconstruct",
            format!("rebuilt edges differ: {lost} missing, {extra} extra"),
        );
    }
    for (x, nd) in g.nodes.iter().enumerate() {
        if matches!(part[x], Part::Kernel(_)) && labels.get(&(x as u32)) != Some(&nd) {
            fail(
                "reconstruct",
                format!("node {x}: the kernel graph's label differs"),
            );
        }
    }
}
