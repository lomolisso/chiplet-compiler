//! The kernel partition of one normalized component (design §4.5, docs/passes/kernels.md §4).
//!
//! On the unit DAG (`units`): seed one kernel per core in SSA order, absorb epilogues to a fixpoint,
//! then prologues, group the remaining element-wise units into map kernels, absorb broadcasts of kernel
//! inputs, and leave everything else as boundaries. Every growth step keeps its kernel convex (K2) by
//! an ancestor-bitset test and adds a member's whole consumer set or nothing (K3). If the quotient graph
//! has a cycle, the latest kernel on it loses its most recently added member (it is banned from that
//! kernel) and the partition is recomputed.

pub mod build;
pub mod extract;
pub mod units;

use std::collections::{BTreeMap, BTreeSet};

use cc_ir::absgraph::Graph;

use crate::components::algo::graph::ADim;
use crate::kernels::json::{Boundary, BoundaryKind, Crossing, Kernel, Member, Part, Role};
use crate::normalize::json::NNode;
use units::{UClass, Units};

/// A fixed-size bitset over units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bits(Vec<u64>);

impl Bits {
    pub fn new(n: usize) -> Self {
        Bits(vec![0; n.div_ceil(64)])
    }

    pub fn set(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }

    pub fn get(&self, i: usize) -> bool {
        self.0[i / 64] >> (i % 64) & 1 == 1
    }

    pub fn or(&mut self, o: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&o.0) {
            *a |= b;
        }
    }

    pub fn intersects(&self, o: &Bits) -> bool {
        self.0.iter().zip(&o.0).any(|(a, b)| a & b != 0)
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0
            .iter()
            .enumerate()
            .flat_map(|(w, &x)| (0..64).filter(move |b| x >> b & 1 == 1).map(move |b| w * 64 + b))
    }
}

/// Strict ancestors of every unit (units are in topological order).
pub fn ancestors(u: &Units) -> Vec<Bits> {
    let n = u.len();
    let mut anc: Vec<Bits> = Vec::with_capacity(n);
    for i in 0..n {
        let mut b = Bits::new(n);
        for &p in &u.pred[i] {
            b.or(&anc[p as usize]);
            b.set(p as usize);
        }
        anc.push(b);
    }
    anc
}

/// A kernel under construction.
#[derive(Debug, Clone)]
struct Build {
    /// The seed unit (the core, or the first map member).
    seed: u32,
    core: bool,
    /// Units with their roles, in the order they were added.
    members: Vec<(u32, Role)>,
    set: Bits,
    anc: Bits,
}

impl Build {
    fn new(seed: u32, core: bool, n: usize, anc: &[Bits]) -> Self {
        let mut set = Bits::new(n);
        set.set(seed as usize);
        Build {
            seed,
            core,
            members: vec![(seed, if core { Role::Core } else { Role::Map })],
            set,
            anc: anc[seed as usize].clone(),
        }
    }

    fn has(&self, x: u32) -> bool {
        self.set.get(x as usize)
    }

    fn add(&mut self, x: u32, role: Role, anc: &[Bits]) {
        self.set.set(x as usize);
        self.anc.or(&anc[x as usize]);
        self.members.push((x, role));
    }
}

/// K2: no unit outside `set` is both an ancestor and a descendant of `set`.
fn convex(set: &Bits, ancset: &Bits, anc: &[Bits]) -> bool {
    ancset.iter().all(|x| set.get(x) || !anc[x].intersects(set))
}

/// The result of one component's partition.
#[derive(Debug, Clone, PartialEq)]
pub struct Partition {
    pub kernels: Vec<Kernel>,
    pub boundaries: Vec<Boundary>,
    pub order: Vec<Part>,
    pub evictions: u32,
}

struct Ctx<'a> {
    g: &'a Graph<NNode>,
    u: &'a Units,
    anc: &'a [Bits],
    /// (seed unit's representative, unit's representative) pairs that may not be joined.
    bans: &'a BTreeSet<(u32, u32)>,
    assigned: Vec<Option<usize>>,
}

impl Ctx<'_> {
    fn dims(&self, x: u32) -> Option<&[ADim]> {
        units::dims(self.g, self.u.rep[x as usize], 0)
    }

    fn free(&self, b: &Build, x: u32, class: UClass) -> bool {
        self.assigned[x as usize].is_none()
            && self.u.class[x as usize] == class
            && !self
                .bans
                .contains(&(self.u.rep[b.seed as usize], self.u.rep[x as usize]))
    }

    /// The consumers of `m` outside `b`, if absorbing all of them keeps K2-K4: each is a free
    /// element-wise unit with dims in `domains`, every unit it reads in the kernel or in the set has all
    /// its consumers in the kernel or the set (K3), and the grown kernel is convex.
    fn consumer_set(&self, b: &Build, m: u32, domains: &[&[ADim]]) -> Option<Vec<u32>> {
        let c: Vec<u32> = self.u.succ[m as usize]
            .iter()
            .copied()
            .filter(|&v| !b.has(v))
            .collect();
        if c.is_empty() {
            return None;
        }
        let mut set = b.set.clone();
        let mut ancset = b.anc.clone();
        for &v in &c {
            set.set(v as usize);
            ancset.or(&self.anc[v as usize]);
        }
        let ok = c.iter().all(|&v| {
            self.free(b, v, UClass::Elementwise)
                && self.dims(v).is_some_and(|d| domains.contains(&d))
                && self.u.pred[v as usize]
                    .iter()
                    .filter(|&&p| set.get(p as usize))
                    .all(|&p| self.u.succ[p as usize].iter().all(|&s| set.get(s as usize)))
        }) && convex(&set, &ancset, self.anc);
        ok.then_some(c)
    }

    /// Grow forward by whole consumer sets to a fixpoint.
    fn grow_forward(&self, b: &mut Build, domains: &[Vec<ADim>], role: Role) {
        let doms: Vec<&[ADim]> = domains.iter().map(Vec::as_slice).collect();
        let mut closed: BTreeSet<u32> = BTreeSet::new();
        loop {
            let mut changed = false;
            let ms: Vec<u32> = b.members.iter().map(|m| m.0).collect();
            for m in ms {
                if closed.contains(&m) {
                    continue;
                }
                match self.consumer_set(b, m, &doms) {
                    Some(c) => {
                        for v in c {
                            b.add(v, role, self.anc);
                        }
                        changed = true;
                    }
                    None => {
                        closed.insert(m);
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
}

/// The ports of `core`'s representative fed by unit `p`, with the producer's result index.
fn core_ports(g: &Graph<NNode>, u: &Units, core: u32, p: u32) -> Vec<(u32, u32, u32)> {
    let c = u.rep[core as usize];
    let mut ports = Vec::new();
    for &src in &u.nodes[p as usize] {
        if let Some(l) = g.edge(src, c) {
            for e in l {
                if let cc_ir::absgraph::PortLabel::Port(k) = e.port {
                    ports.push((k, src, e.res));
                }
            }
        }
    }
    ports
}

/// Absorb prologues of a core kernel: element-wise units whose consumers are all in the kernel and lead
/// to one core operand port, with that operand's dims. Consumers are decided before producers (reverse
/// unit order), so one pass suffices.
fn grow_prologue(cx: &Ctx<'_>, b: &mut Build) {
    let core = b.seed;
    // Port reached by each prologue member (the core itself has none).
    let mut port_of: BTreeMap<u32, u32> = BTreeMap::new();
    let mut operand_dims: BTreeMap<u32, Vec<ADim>> = BTreeMap::new();
    let mut queue: BTreeSet<u32> = cx.u.pred[core as usize].iter().copied().collect();
    while let Some(x) = queue.pop_last() {
        if b.has(x) || !cx.free(b, x, UClass::Elementwise) {
            continue;
        }
        let mut ports: BTreeSet<u32> = BTreeSet::new();
        let mut inside = true;
        for &s in &cx.u.succ[x as usize] {
            if s == core {
                for (k, src, res) in core_ports(cx.g, cx.u, core, x) {
                    ports.insert(k);
                    if let Some(d) = units::dims(cx.g, src, res) {
                        operand_dims.entry(k).or_insert_with(|| d.to_vec());
                    }
                }
            } else if let Some(&k) = port_of.get(&s) {
                ports.insert(k);
            } else {
                inside = false;
            }
        }
        if !inside || ports.len() != 1 {
            continue;
        }
        let Some(&k) = ports.first() else { continue };
        if cx.dims(x) != operand_dims.get(&k).map(Vec::as_slice) {
            continue;
        }
        b.add(x, Role::Prologue, cx.anc);
        port_of.insert(x, k);
        for &p in &cx.u.pred[x as usize] {
            queue.insert(p);
        }
    }
}

/// Map kernel growth: whole consumer sets forward, and producers whose consumers are all members.
fn grow_map(cx: &Ctx<'_>, b: &mut Build) {
    let Some(d) = cx.dims(b.seed).map(<[ADim]>::to_vec) else {
        return;
    };
    loop {
        let before = b.members.len();
        cx.grow_forward(b, std::slice::from_ref(&d), Role::Map);
        let ms: Vec<u32> = b.members.iter().map(|m| m.0).collect();
        for m in ms {
            for &p in &cx.u.pred[m as usize] {
                if !b.has(p)
                    && cx.free(b, p, UClass::Elementwise)
                    && cx.dims(p) == Some(d.as_slice())
                    && cx.u.succ[p as usize].iter().all(|&s| b.has(s))
                {
                    b.add(p, Role::Map, cx.anc);
                }
            }
        }
        if b.members.len() == before {
            break;
        }
    }
}

/// One attempt with a set of bans.
fn attempt(g: &Graph<NNode>, u: &Units, anc: &[Bits], bans: &BTreeSet<(u32, u32)>) -> Vec<Build> {
    let n = u.len();
    let mut cx = Ctx {
        g,
        u,
        anc,
        bans,
        assigned: vec![None; n],
    };
    let mut ks: Vec<Build> = Vec::new();
    for x in 0..n as u32 {
        if u.class[x as usize] == UClass::Core {
            cx.assigned[x as usize] = Some(ks.len());
            ks.push(Build::new(x, true, n, anc));
        }
    }
    // Epilogues, kernel by kernel in seed order (ties go to the earlier core).
    for (k, kb) in ks.iter_mut().enumerate() {
        let mut b = kb.clone();
        let rep = u.rep[b.seed as usize];
        let domains: Vec<Vec<ADim>> = g.nodes[rep as usize]
            .label
            .results
            .iter()
            .map(|t| t.dims.clone())
            .collect();
        cx.grow_forward(&mut b, &domains, Role::Epilogue);
        for &(x, _) in &b.members {
            cx.assigned[x as usize] = Some(k);
        }
        *kb = b;
    }
    // Prologues.
    for (k, kb) in ks.iter_mut().enumerate() {
        let mut b = kb.clone();
        grow_prologue(&cx, &mut b);
        for &(x, _) in &b.members {
            cx.assigned[x as usize] = Some(k);
        }
        *kb = b;
    }
    // Map kernels over the remaining element-wise units.
    for x in 0..n as u32 {
        if u.class[x as usize] != UClass::Elementwise || cx.assigned[x as usize].is_some() {
            continue;
        }
        let k = ks.len();
        let mut b = Build::new(x, false, n, anc);
        cx.assigned[x as usize] = Some(k);
        grow_map(&cx, &mut b);
        for &(y, _) in &b.members {
            cx.assigned[y as usize] = Some(k);
        }
        ks.push(b);
    }
    // Broadcasts of kernel inputs whose consumers all lie in one kernel; consumers first, so a chain of
    // broadcasts is absorbed whole.
    for x in (0..n as u32).rev() {
        if u.class[x as usize] != UClass::Broadcast {
            continue;
        }
        let succ = &u.succ[x as usize];
        let Some(&first) = succ.first() else { continue };
        let Some(k) = cx.assigned[first as usize] else {
            continue;
        };
        let b = &ks[k];
        if succ.iter().all(|&s| cx.assigned[s as usize] == Some(k))
            && u.pred[x as usize].iter().all(|&p| !b.has(p))
            && !bans.contains(&(u.rep[b.seed as usize], u.rep[x as usize]))
        {
            cx.assigned[x as usize] = Some(k);
            ks[k].add(x, Role::Broadcast, anc);
        }
    }
    ks
}

/// The quotient graph's parts: kernel index or boundary unit.
fn part_of(ks: &[Build], u: &Units) -> Vec<Part> {
    let mut p: Vec<Part> = (0..u.len()).map(|x| Part::Boundary(u.rep[x])).collect();
    for (k, b) in ks.iter().enumerate() {
        for &(x, _) in &b.members {
            p[x as usize] = Part::Kernel(k as u32);
        }
    }
    p
}

/// A topological order of the quotient (ready parts by their first unit), or one cycle of it.
fn quotient_order(u: &Units, part: &[Part]) -> Result<Vec<Part>, Vec<Part>> {
    let mut first: BTreeMap<Part, usize> = BTreeMap::new();
    for (x, p) in part.iter().enumerate() {
        first.entry(*p).or_insert(x);
    }
    let mut succ: BTreeMap<Part, BTreeSet<Part>> = first.keys().map(|p| (*p, BTreeSet::new())).collect();
    let mut pred: BTreeMap<Part, BTreeSet<Part>> = first.keys().map(|p| (*p, BTreeSet::new())).collect();
    for x in 0..u.len() {
        for &y in &u.succ[x] {
            let (a, b) = (part[x], part[y as usize]);
            if a != b && succ.get_mut(&a).is_some_and(|s| s.insert(b)) {
                pred.entry(b).or_default().insert(a);
            }
        }
    }
    let mut indeg: BTreeMap<Part, usize> = pred.iter().map(|(p, s)| (*p, s.len())).collect();
    let mut ready: BTreeSet<(usize, Part)> = indeg
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(p, _)| (first[p], *p))
        .collect();
    let mut order = Vec::with_capacity(first.len());
    while let Some((_, p)) = ready.pop_first() {
        order.push(p);
        for s in &succ[&p] {
            if let Some(d) = indeg.get_mut(s) {
                *d -= 1;
                if *d == 0 {
                    ready.insert((first[s], *s));
                }
            }
        }
    }
    if order.len() == first.len() {
        return Ok(order);
    }
    // Every part left has a predecessor left: walking predecessors must repeat a part.
    let left: BTreeSet<Part> = indeg.iter().filter(|(_, d)| **d > 0).map(|(p, _)| *p).collect();
    let mut path: Vec<Part> = Vec::new();
    let mut seen: BTreeMap<Part, usize> = BTreeMap::new();
    let mut cur = left.first().copied();
    while let Some(p) = cur {
        if let Some(&i) = seen.get(&p) {
            return Err(path[i..].to_vec());
        }
        seen.insert(p, path.len());
        path.push(p);
        cur = pred[&p].iter().find(|q| left.contains(q)).copied();
    }
    Err(path)
}

/// Partition one normalized component.
pub fn partition(g: &Graph<NNode>) -> Result<Partition, String> {
    let u = units::units(g)?;
    let anc = ancestors(&u);
    let mut bans: BTreeSet<(u32, u32)> = BTreeSet::new();
    let (ks, order) = loop {
        let ks = attempt(g, &u, &anc, &bans);
        let part = part_of(&ks, &u);
        match quotient_order(&u, &part) {
            Ok(order) => break (ks, order),
            Err(cycle) => {
                // The latest kernel on the cycle that has more than its seed loses its last member.
                let victim = cycle
                    .iter()
                    .filter_map(|p| match p {
                        Part::Kernel(k) if ks[*k as usize].members.len() > 1 => Some(*k as usize),
                        _ => None,
                    })
                    .max()
                    .ok_or("the quotient graph has a cycle that no eviction breaks")?;
                let b = &ks[victim];
                let last = b.members.last().map(|m| m.0).unwrap_or(b.seed);
                if !bans.insert((u.rep[b.seed as usize], u.rep[last as usize])) {
                    return Err("eviction repeated a ban".into());
                }
            }
        }
    };
    Ok(finish(g, &u, &ks, &order, bans.len() as u32))
}

fn finish(g: &Graph<NNode>, u: &Units, ks: &[Build], order: &[Part], evictions: u32) -> Partition {
    let part = part_of(ks, u);
    let node_part = |n: u32| part[u.of[n as usize] as usize];
    let mut kernels: Vec<Kernel> = Vec::with_capacity(ks.len());
    for (k, b) in ks.iter().enumerate() {
        let mut members: Vec<Member> = Vec::new();
        for &(x, role) in &b.members {
            let rep = u.rep[x as usize];
            for &n in &u.nodes[x as usize] {
                members.push(Member {
                    node: n,
                    role: if n == rep { role } else { Role::Body },
                });
            }
        }
        members.sort_by_key(|m| m.node);
        let core = b.core.then(|| u.rep[b.seed as usize]);
        let domain = units::dims(g, u.rep[b.seed as usize], 0)
            .map(<[ADim]>::to_vec)
            .unwrap_or_default();
        let me = Part::Kernel(k as u32);
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for m in &members {
            for &src in &g.inc[m.node as usize] {
                if node_part(src) != me {
                    for e in g.edge(src, m.node).into_iter().flatten() {
                        inputs.push(Crossing {
                            src,
                            dst: m.node,
                            end: *e,
                        });
                    }
                }
            }
            for (dst, ends) in &g.out[m.node as usize] {
                if node_part(*dst) != me {
                    for e in ends {
                        outputs.push(Crossing {
                            src: m.node,
                            dst: *dst,
                            end: *e,
                        });
                    }
                }
            }
        }
        inputs.sort();
        outputs.sort();
        kernels.push(Kernel {
            id: format!("k{k}"),
            core,
            domain,
            members,
            inputs,
            outputs,
        });
    }
    let mut boundaries = Vec::new();
    for (x, pt) in part.iter().enumerate() {
        if matches!(pt, Part::Kernel(_)) {
            continue;
        }
        let kind = match u.class[x] {
            UClass::Boundary(k) => k,
            _ => BoundaryKind::Layout,
        };
        let rep = u.rep[x];
        for &n in &u.nodes[x] {
            boundaries.push(Boundary {
                node: n,
                kind,
                owner: (n != rep).then_some(rep),
            });
        }
    }
    boundaries.sort_by_key(|b| b.node);
    Partition {
        kernels,
        boundaries,
        order: order.to_vec(),
        evictions,
    }
}

#[cfg(test)]
mod tests {
    use super::build::{GraphBuilder, ty};
    use super::*;
    use crate::components::algo::graph::Family;
    use crate::kernels::json::{GRAPHS_SCHEMA, KernelGraphs, KernelPartition, PARTITION_SCHEMA};
    use crate::kernels::verify::check_component;

    const F: Family = Family::Float;

    /// Per kernel: the core's `<op>/<family>` or `map`, and the members other than region nodes.
    type Shape = Vec<(String, Vec<(u32, Role)>)>;

    /// Partition, extract and check with the independent checker.
    fn run(g: &Graph<NNode>) -> (Partition, Shape) {
        let p = partition(g).unwrap();
        let kp = KernelPartition {
            schema: PARTITION_SCHEMA.into(),
            id: "t".into(),
            kernels: p.kernels.clone(),
            boundaries: p.boundaries.clone(),
            order: p.order.clone(),
            evictions: p.evictions,
        };
        let kg = KernelGraphs {
            schema: GRAPHS_SCHEMA.into(),
            id: "t".into(),
            kernels: p
                .kernels
                .iter()
                .map(|k| extract::extract(g, k).unwrap())
                .collect(),
        };
        let bad = check_component(g, &kp, &kg);
        assert!(bad.is_empty(), "{bad:#?}");
        let ks = p
            .kernels
            .iter()
            .map(|k| {
                let what = k
                    .core
                    .map_or("map".to_string(), |c| crate::kernels::core_key(g, c));
                let roles = k
                    .members
                    .iter()
                    .filter(|m| m.role != Role::Body)
                    .map(|m| (m.node, m.role))
                    .collect();
                (what, roles)
            })
            .collect();
        (p, ks)
    }

    #[test]
    fn softmax_is_two_reductions_and_two_maps() {
        let mut b = GraphBuilder::new();
        let x = b.input(ty(&[1, 2], F));
        let ninf = b.constant(ty(&[], F));
        let zero = b.constant(ty(&[], F));
        let m = b.reduce("stablehlo.maximum", ty(&[1], F), x, ninf);
        let bm = b.op("stablehlo.broadcast_in_dim", ty(&[1, 2], F), &[m]);
        let d = b.op("stablehlo.subtract", ty(&[1, 2], F), &[x, bm]);
        let e = b.op("stablehlo.exponential", ty(&[1, 2], F), &[d]);
        let s = b.reduce("stablehlo.add", ty(&[1], F), e, zero);
        let bs = b.op("stablehlo.broadcast_in_dim", ty(&[1, 2], F), &[s]);
        let y = b.op("stablehlo.divide", ty(&[1, 2], F), &[e, bs]);
        b.ret(&[y]);
        let g = b.finish();
        let (p, ks) = run(&g);
        assert_eq!(ks.len(), 4, "{ks:?}");
        assert_eq!(ks[0], ("reduce/float".into(), vec![(m, Role::Core)]));
        assert_eq!(ks[1], ("reduce/float".into(), vec![(s, Role::Core)]));
        assert_eq!(
            ks[2],
            (
                "map".into(),
                vec![(bm, Role::Broadcast), (d, Role::Map), (e, Role::Map)]
            )
        );
        assert_eq!(ks[3], ("map".into(), vec![(bs, Role::Broadcast), (y, Role::Map)]));
        // The reduction bodies stay with their owners.
        let body = p.kernels[0]
            .members
            .iter()
            .filter(|m| m.role == Role::Body)
            .count();
        assert_eq!(body, 4, "two arguments, the combiner, the terminator");
    }

    #[test]
    fn conv_bias_requant_relu_is_one_kernel() {
        let i = Family::Sint;
        let mut b = GraphBuilder::new();
        let x = b.input(ty(&[1, 2, 3, 4], i.clone()));
        let w = b.input(ty(&[5, 6, 4, 7], i.clone()));
        let bias = b.input(ty(&[7], i.clone()));
        let c = b.op("stablehlo.convolution", ty(&[1, 8, 9, 7], i.clone()), &[x, w]);
        let bb = b.op(
            "stablehlo.broadcast_in_dim",
            ty(&[1, 8, 9, 7], i.clone()),
            &[bias],
        );
        let a = b.op("stablehlo.add", ty(&[1, 8, 9, 7], i.clone()), &[c, bb]);
        let f = b.op("stablehlo.convert", ty(&[1, 8, 9, 7], F), &[a]);
        let sc = b.op("stablehlo.multiply", ty(&[1, 8, 9, 7], F), &[f]);
        let r = b.op("stablehlo.round_nearest_even", ty(&[1, 8, 9, 7], F), &[sc]);
        let relu = b.op("stablehlo.maximum", ty(&[1, 8, 9, 7], F), &[r]);
        let q = b.op("stablehlo.convert", ty(&[1, 8, 9, 7], i), &[relu]);
        b.ret(&[q]);
        let g = b.finish();
        let (_, ks) = run(&g);
        assert_eq!(ks.len(), 1, "{ks:?}");
        assert_eq!(ks[0].0, "convolution/sint");
        let epi: Vec<u32> = ks[0]
            .1
            .iter()
            .filter(|m| m.1 == Role::Epilogue)
            .map(|m| m.0)
            .collect();
        assert_eq!(epi, vec![a, f, sc, r, relu, q]);
        assert!(ks[0].1.contains(&(bb, Role::Broadcast)));
    }

    #[test]
    fn attention_fuses_the_normalization_into_the_second_gemm() {
        let mut b = GraphBuilder::new();
        let q = b.input(ty(&[1, 2, 3], F));
        let k = b.input(ty(&[1, 4, 3], F));
        let v = b.input(ty(&[1, 4, 5], F));
        let ninf = b.constant(ty(&[], F));
        let zero = b.constant(ty(&[], F));
        let s = b.op("stablehlo.dot_general", ty(&[1, 2, 4], F), &[q, k]);
        let sc = b.op("stablehlo.multiply", ty(&[1, 2, 4], F), &[s]);
        let m = b.reduce("stablehlo.maximum", ty(&[1, 2], F), sc, ninf);
        let bm = b.op("stablehlo.broadcast_in_dim", ty(&[1, 2, 4], F), &[m]);
        let d = b.op("stablehlo.subtract", ty(&[1, 2, 4], F), &[sc, bm]);
        let e = b.op("stablehlo.exponential", ty(&[1, 2, 4], F), &[d]);
        let sum = b.reduce("stablehlo.add", ty(&[1, 2], F), e, zero);
        let bs = b.op("stablehlo.broadcast_in_dim", ty(&[1, 2, 4], F), &[sum]);
        let pr = b.op("stablehlo.divide", ty(&[1, 2, 4], F), &[e, bs]);
        let o = b.op("stablehlo.dot_general", ty(&[1, 2, 5], F), &[pr, v]);
        b.ret(&[o]);
        let g = b.finish();
        let (_, ks) = run(&g);
        let what: Vec<&str> = ks.iter().map(|k| k.0.as_str()).collect();
        assert_eq!(
            what,
            [
                "dot_general/float",
                "reduce/float",
                "reduce/float",
                "dot_general/float",
                "map"
            ]
        );
        // The scale is the scores' only consumer, so it joins the first GEMM and is its output; the
        // normalization's only consumer is the second GEMM, so it is that GEMM's prologue.
        assert_eq!(ks[0].1, vec![(s, Role::Core), (sc, Role::Epilogue)]);
        assert_eq!(
            ks[3].1,
            vec![(bs, Role::Broadcast), (pr, Role::Prologue), (o, Role::Core)]
        );
        assert_eq!(
            ks[4].1,
            vec![(bm, Role::Broadcast), (d, Role::Map), (e, Role::Map)]
        );
    }

    #[test]
    fn scatter_with_arithmetic_update_is_a_core_and_overwrite_is_a_boundary() {
        let mut b = GraphBuilder::new();
        let base = b.input(ty(&[1, 2], F));
        let idx = b.input(ty(&[3, 4], Family::Sint));
        let upd = b.input(ty(&[3, 2], F));
        let add = b.with_region(
            "stablehlo.scatter",
            ty(&[1, 2], F),
            &[base, idx, upd],
            2,
            |b, a| vec![b.op("stablehlo.add", ty(&[], F), &[a[0], a[1]])],
        );
        let put = b.with_region(
            "stablehlo.scatter",
            ty(&[1, 2], F),
            &[add, idx, upd],
            2,
            |_, a| vec![a[1]],
        );
        b.ret(&[put]);
        let g = b.finish();
        let (p, ks) = run(&g);
        assert_eq!(ks.len(), 1);
        assert_eq!(ks[0].0, "scatter/float");
        assert_eq!(p.kernels[0].core, Some(add));
        let b0 = p.boundaries.iter().find(|x| x.node == put).unwrap();
        assert_eq!(b0.kind, BoundaryKind::DataMovement);
        assert!(p.boundaries.iter().filter(|x| x.owner == Some(put)).count() == 3);
    }

    #[test]
    fn a_residual_add_of_two_gemms_joins_the_first() {
        let mut b = GraphBuilder::new();
        let x = b.input(ty(&[1, 2], F));
        let w1 = b.input(ty(&[2, 3], F));
        let y = b.input(ty(&[1, 4], F));
        let w2 = b.input(ty(&[4, 3], F));
        let a = b.op("stablehlo.dot_general", ty(&[1, 3], F), &[x, w1]);
        let c = b.op("stablehlo.dot_general", ty(&[1, 3], F), &[y, w2]);
        let s = b.op("stablehlo.add", ty(&[1, 3], F), &[a, c]);
        b.ret(&[s]);
        let g = b.finish();
        let (_, ks) = run(&g);
        assert_eq!(ks[0].1, vec![(a, Role::Core), (s, Role::Epilogue)]);
        assert_eq!(ks[1].1, vec![(c, Role::Core)]);
    }

    #[test]
    fn an_escaping_intermediate_stops_the_epilogue() {
        let mut b = GraphBuilder::new();
        let x = b.input(ty(&[1, 2], F));
        let w = b.input(ty(&[2, 3], F));
        let d = b.op("stablehlo.dot_general", ty(&[1, 3], F), &[x, w]);
        let t = b.op("stablehlo.tanh", ty(&[1, 3], F), &[d]);
        let r = b.op("stablehlo.transpose", ty(&[3, 1], F), &[d]);
        b.ret(&[t, r]);
        let g = b.finish();
        let (_, ks) = run(&g);
        assert_eq!(
            ks[0].1,
            vec![(d, Role::Core)],
            "the transpose consumer keeps tanh out (K3)"
        );
        assert_eq!(ks[1].1, vec![(t, Role::Map)]);
    }

    #[test]
    fn the_quotient_check_finds_the_per_part_convexity_pitfall() {
        // x1 -> y1, x2 -> y2; parts {x1, y2} and {y1, x2}: each part is convex, the quotient is cyclic.
        let mut b = GraphBuilder::new();
        let i = b.input(ty(&[1], F));
        let x1 = b.op("stablehlo.negate", ty(&[1], F), &[i]);
        let y1 = b.op("stablehlo.negate", ty(&[1], F), &[x1]);
        let x2 = b.op("stablehlo.negate", ty(&[1], F), &[i]);
        let y2 = b.op("stablehlo.negate", ty(&[1], F), &[x2]);
        b.ret(&[y1, y2]);
        let g = b.finish();
        let u = units::units(&g).unwrap();
        let mut part: Vec<Part> = (0..u.len()).map(|x| Part::Boundary(u.rep[x])).collect();
        for (n, k) in [(x1, 0), (y2, 0), (y1, 1), (x2, 1)] {
            part[u.of[n as usize] as usize] = Part::Kernel(k);
        }
        let cycle = quotient_order(&u, &part).unwrap_err();
        let set: BTreeSet<Part> = cycle.into_iter().collect();
        assert_eq!(set, BTreeSet::from([Part::Kernel(0), Part::Kernel(1)]));
    }
}
