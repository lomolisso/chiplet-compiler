//! The compatibility graph of two units (Moreano et al., IEEE TCAD 24(7), 2005, Defs. 6–8;
//! docs/passes/merge.md §4.4).
//!
//! Nodes are *vertex mappings* (a super-node of A with one of B: a block, or a whole mined-pattern
//! group, with the same label and joinable types) and *arc mappings* (a wire of A with a wire of B whose
//! ends map and whose ports match, or are exchanged on a commutative operator). Two mappings are
//! compatible when they keep the block correspondence injective (Def. 7), agree on the orientation of
//! a commutative pair, and do not order two blocks oppositely in A and B (a merged graph with a cycle
//! of length two; longer cycles are found on the rebuilt graph). Weights are the area saved: two blocks
//! minus their join, or the multiplexer input a shared wire avoids.

use std::collections::{BTreeMap, HashMap};

use super::fu::{Costed, Fu, join_sig, port_type, widen_ok};
use super::library::{Lib, bits};
use crate::merge::json::BlockKind;

/// A bitset over `n` elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bits(pub Vec<u64>);

impl Bits {
    pub fn new(n: usize) -> Bits {
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
}

/// The super-nodes of a unit: whole groups, and the blocks outside groups.
#[derive(Debug, Clone)]
pub struct Supers {
    /// Blocks of each super-node, in member order.
    pub members: Vec<Vec<u32>>,
    pub labels: Vec<String>,
    /// Per block, its super-node.
    pub of: Vec<u32>,
    /// Per super-node, the super-nodes reachable from it (edges out of accumulators excluded).
    pub reach: Vec<Bits>,
    /// Price of each super-node (sum of its blocks).
    pub price: Vec<f64>,
}

pub fn supers(fu: &Fu, c: &Costed) -> Supers {
    let mut members: Vec<Vec<u32>> = Vec::new();
    let mut labels = Vec::new();
    let mut of = vec![u32::MAX; fu.blocks.len()];
    let mut by_group: BTreeMap<u32, u32> = BTreeMap::new();
    for (i, b) in fu.blocks.iter().enumerate() {
        match b.group {
            Some(g) => {
                let s = *by_group.entry(g).or_insert_with(|| {
                    members.push(Vec::new());
                    labels.push(format!("pattern:{}", fu.groups[g as usize]));
                    members.len() as u32 - 1
                });
                let m = &mut members[s as usize];
                let at = b.member as usize;
                if m.len() <= at {
                    m.resize(at + 1, u32::MAX);
                }
                m[at] = i as u32;
                of[i] = s;
            }
            None => {
                members.push(vec![i as u32]);
                labels.push(b.label.clone());
                of[i] = members.len() as u32 - 1;
            }
        }
    }
    let n = members.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, ps) in c.wiring.iter().enumerate() {
        for s in ps.iter().flatten() {
            let (x, y) = (of[s.block as usize] as usize, of[b] as usize);
            if x != y && fu.blocks[s.block as usize].kind != BlockKind::Acc {
                succ[x].push(y);
            }
        }
    }
    // Reachability in reverse topological order of the (acyclic) super-node graph.
    let mut order = Vec::with_capacity(n);
    let mut indeg = vec![0usize; n];
    for s in &mut succ {
        s.sort_unstable();
        s.dedup();
    }
    for s in &succ {
        for &y in s {
            indeg[y] += 1;
        }
    }
    let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    while let Some(x) = ready.pop() {
        order.push(x);
        for &y in &succ[x] {
            indeg[y] -= 1;
            if indeg[y] == 0 {
                ready.push(y);
            }
        }
    }
    let mut reach = vec![Bits::new(n); n];
    for &x in order.iter().rev() {
        let mut r = Bits::new(n);
        for &y in &succ[x] {
            r.set(y);
            r.or(&reach[y]);
        }
        reach[x] = r;
    }
    let price = members
        .iter()
        .map(|m| m.iter().map(|&b| c.blocks[b as usize].area_um2).sum())
        .collect();
    Supers {
        members,
        labels,
        of,
        reach,
        price,
    }
}

/// A vertex mapping: super-node `sa` of A with `sb` of B; per member, the joined types.
#[derive(Debug, Clone, PartialEq)]
pub struct VMap {
    pub sa: u32,
    pub sb: u32,
    pub w: f64,
    pub types: Vec<Vec<String>>,
}

/// An arc mapping: wire `(a_src, a_res) -> (a_dst, a_port)` of A with one of B.
#[derive(Debug, Clone, PartialEq)]
pub struct AMap {
    pub a: (u32, u32, u32, u32),
    pub b: (u32, u32, u32, u32),
    /// The vertex mappings of the ends.
    pub src: u32,
    pub dst: u32,
    /// B's commutative pair is exchanged on the merged block.
    pub perm: bool,
    pub w: f64,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub vm: Vec<VMap>,
    pub am: Vec<AMap>,
    pub sa: Supers,
    pub sb: Supers,
}

pub fn commutative(op: &str) -> bool {
    cc_ir::ops::TABLE
        .iter()
        .any(|s| s.elementwise && s.commutative && s.name.trim_start_matches("stablehlo.") == op)
}

/// Every wire of a unit: `(src, res, dst, port)`.
pub fn arcs(c: &Costed) -> Vec<(u32, u32, u32, u32)> {
    let mut v = Vec::new();
    for (b, ps) in c.wiring.iter().enumerate() {
        for (p, ss) in ps.iter().enumerate() {
            for s in ss {
                v.push((s.block, s.res, b as u32, p as u32));
            }
        }
    }
    v
}

#[allow(clippy::too_many_arguments)]
pub fn model(
    a: &Fu,
    ca: &Costed,
    b: &Fu,
    cb: &Costed,
    lib: &Lib,
    prices: &mut HashMap<String, f64>,
) -> Model {
    let sa = supers(a, ca);
    let sb = supers(b, cb);
    let mut by_label: HashMap<&str, Vec<u32>> = HashMap::new();
    for (i, l) in sb.labels.iter().enumerate() {
        by_label.entry(l.as_str()).or_default().push(i as u32);
    }
    let mut vm = Vec::new();
    let mut at: HashMap<(u32, u32), u32> = HashMap::new();
    for (i, l) in sa.labels.iter().enumerate() {
        let Some(js) = by_label.get(l.as_str()) else {
            continue;
        };
        let ma = &sa.members[i];
        for &j in js {
            let mb = &sb.members[j as usize];
            if ma.len() != mb.len() || ma.contains(&u32::MAX) || mb.contains(&u32::MAX) {
                continue;
            }
            let mut w = 0.0;
            let mut types = Vec::new();
            let mut ok = true;
            for (&x, &y) in ma.iter().zip(mb) {
                let (bx, by) = (&a.blocks[x as usize], &b.blocks[y as usize]);
                if bx.kind != by.kind || bx.op != by.op {
                    ok = false;
                    break;
                }
                let widen = widen_ok(bx.kind, &bx.op, &bx.attrs, &bx.types);
                let Some(t) = join_sig(&bx.types, &by.types, widen) else {
                    ok = false;
                    break;
                };
                let literal = bx.literal && by.literal && bx.attrs == by.attrs;
                let key = format!("{:?}|{}|{:?}|{}|{literal}", bx.kind, bx.op, bx.attrs, t.join(","));
                let pj = *prices
                    .entry(key)
                    .or_insert_with(|| lib.block(bx.kind, &bx.op, &bx.attrs, &t, literal).area_um2);
                w += ca.blocks[x as usize].area_um2 + cb.blocks[y as usize].area_um2 - pj;
                types.push(t);
            }
            if !ok || w < -1e-9 {
                continue;
            }
            at.insert((i as u32, j), vm.len() as u32);
            vm.push(VMap {
                sa: i as u32,
                sb: j,
                w: w.max(0.0),
                types,
            });
        }
    }
    // Arc mappings: join the wires on (labels of the ends, members, result).
    let member = |s: &Supers, blk: u32| -> u32 {
        let sup = s.of[blk as usize] as usize;
        s.members[sup].iter().position(|&x| x == blk).unwrap_or(0) as u32
    };
    let key = |s: &Supers, e: &(u32, u32, u32, u32)| {
        (
            s.labels[s.of[e.0 as usize] as usize].clone(),
            member(s, e.0),
            e.1,
            s.labels[s.of[e.2 as usize] as usize].clone(),
            member(s, e.2),
        )
    };
    let ea = arcs(ca);
    let eb = arcs(cb);
    let mut b_by: HashMap<(String, u32, u32, String, u32), Vec<usize>> = HashMap::new();
    for (k, e) in eb.iter().enumerate() {
        b_by.entry(key(&sb, e)).or_default().push(k);
    }
    let mut am = Vec::new();
    for e in &ea {
        let Some(ks) = b_by.get(&key(&sa, e)) else {
            continue;
        };
        for &k in ks {
            let f = &eb[k];
            let (Some(&src), Some(&dst)) = (
                at.get(&(sa.of[e.0 as usize], sb.of[f.0 as usize])),
                at.get(&(sa.of[e.2 as usize], sb.of[f.2 as usize])),
            ) else {
                continue;
            };
            let da = &a.blocks[e.2 as usize];
            let perm = if e.3 == f.3 {
                false
            } else if da.kind == BlockKind::Op
                && da.group.is_none()
                && super::fu::arity(da) == 2
                && commutative(&da.op)
            {
                true
            } else {
                continue;
            };
            let w = bits(port_type(da, e.3 as usize));
            am.push(AMap {
                a: *e,
                b: *f,
                src,
                dst,
                perm,
                w: lib.mux(2, w).area_um2,
            });
        }
    }
    Model { vm, am, sa, sb }
}

impl Model {
    pub fn n(&self) -> usize {
        self.vm.len() + self.am.len()
    }

    pub fn weight(&self, i: usize) -> f64 {
        if i < self.vm.len() {
            self.vm[i].w
        } else {
            self.am[i - self.vm.len()].w
        }
    }

    /// Two vertex mappings that cannot both hold: they share a super-node, or order two super-nodes
    /// oppositely.
    pub fn vm_conflict(&self, i: usize, j: usize) -> bool {
        if i == j {
            return false;
        }
        let (x, y) = (&self.vm[i], &self.vm[j]);
        if x.sa == y.sa || x.sb == y.sb {
            return true;
        }
        self.order_conflict(i, j)
    }

    pub fn order_conflict(&self, i: usize, j: usize) -> bool {
        let (x, y) = (&self.vm[i], &self.vm[j]);
        let (a1, a2, b1, b2) = (x.sa as usize, y.sa as usize, x.sb as usize, y.sb as usize);
        (self.sa.reach[a1].get(a2) && self.sb.reach[b2].get(b1))
            || (self.sa.reach[a2].get(a1) && self.sb.reach[b1].get(b2))
    }

    /// The vertex mappings a node needs (itself, or an arc's two ends).
    pub fn ends(&self, i: usize) -> Vec<usize> {
        if i < self.vm.len() {
            vec![i]
        } else {
            let a = &self.am[i - self.vm.len()];
            vec![a.src as usize, a.dst as usize]
        }
    }

    pub fn compatible(&self, i: usize, j: usize) -> bool {
        if i == j {
            return true;
        }
        let nv = self.vm.len();
        for x in self.ends(i) {
            for y in self.ends(j) {
                if self.vm_conflict(x, y) {
                    return false;
                }
            }
        }
        if i >= nv && j >= nv {
            let (x, y) = (&self.am[i - nv], &self.am[j - nv]);
            if x.dst == y.dst && x.perm != y.perm {
                return false;
            }
            if x.a == y.a || x.b == y.b {
                return false;
            }
        }
        true
    }

    /// The chosen set closed under the ends of its arcs, sorted.
    pub fn close(&self, chosen: &[usize]) -> Vec<usize> {
        let mut s: Vec<usize> = chosen
            .iter()
            .flat_map(|&i| self.ends(i).into_iter().chain([i]))
            .collect();
        s.sort_unstable();
        s.dedup();
        s
    }

    pub fn objective(&self, chosen: &[usize]) -> f64 {
        chosen.iter().map(|&i| self.weight(i)).sum()
    }
}
