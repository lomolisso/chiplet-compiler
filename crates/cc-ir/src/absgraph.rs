//! Port-labelled directed graphs for matching: 1-WL colour refinement and hashing (Shervashidze et al.,
//! "Weisfeiler-Lehman graph kernels", JMLR 12:2539-2561, 2011), graph isomorphism through the `vf2`
//! crate (Cordella et al., "A (sub)graph isomorphism algorithm for matching large graphs", IEEE TPAMI
//! 26(10):1367-1372, 2004) and an O(V + E) check of a stored vertex map.
//!
//! Nodes carry a caller-defined label `N`. Parallel edges are merged: an edge `u -> v` carries the sorted
//! list of its ends (`EdgeEnd`: result index of `u`, port of `v`), so `x * x` is one edge with two ends.
//! A commutative destination labels its ports `Commutative`, so permuting its operands keeps the graph.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The destination side of one dataflow edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PortLabel {
    /// Operand port of the destination.
    Port(u32),
    /// An operand of a commutative op: the ports are an unordered multiset.
    Commutative,
    /// Region `region`, argument `index` of the source (the owner) feeds the destination (the argument).
    Region { region: u32, index: u32 },
    /// The terminator of region `region` of the destination (the owner) yields to it.
    Yield { region: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct EdgeEnd {
    /// Result index of the source.
    pub res: u32,
    pub port: PortLabel,
}

/// Sorted ends of all parallel edges between two nodes.
pub type EdgeLabel = Vec<EdgeEnd>;

/// A directed graph with labelled nodes and merged, labelled edges. Adjacency lists are sorted by the
/// neighbour index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Graph<N> {
    pub nodes: Vec<N>,
    pub out: Vec<Vec<(u32, EdgeLabel)>>,
    pub inc: Vec<Vec<u32>>,
}

impl<N> Default for Graph<N> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            out: Vec::new(),
            inc: Vec::new(),
        }
    }
}

impl<N> Graph<N> {
    pub fn add_node(&mut self, n: N) -> u32 {
        self.nodes.push(n);
        self.out.push(Vec::new());
        self.inc.push(Vec::new());
        (self.nodes.len() - 1) as u32
    }

    /// Add one edge end `u -> v`; ends between the same pair are merged into one edge.
    pub fn add_edge(&mut self, u: u32, v: u32, end: EdgeEnd) {
        let out = &mut self.out[u as usize];
        match out.binary_search_by_key(&v, |e| e.0) {
            Ok(i) => {
                let l = &mut out[i].1;
                let at = l.partition_point(|x| x <= &end);
                l.insert(at, end);
            }
            Err(i) => {
                out.insert(i, (v, vec![end]));
                let inc = &mut self.inc[v as usize];
                let at = inc.partition_point(|&x| x < u);
                inc.insert(at, u);
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of merged edges.
    pub fn edge_count(&self) -> usize {
        self.out.iter().map(Vec::len).sum()
    }

    /// Number of edge ends (dataflow edges before merging).
    pub fn end_count(&self) -> usize {
        self.out.iter().flatten().map(|e| e.1.len()).sum()
    }

    pub fn edge(&self, u: u32, v: u32) -> Option<&EdgeLabel> {
        let out = &self.out[u as usize];
        out.binary_search_by_key(&v, |e| e.0).ok().map(|i| &out[i].1)
    }

    /// Map the node labels.
    pub fn map<M>(&self, f: impl Fn(&N) -> M) -> Graph<M> {
        Graph {
            nodes: self.nodes.iter().map(f).collect(),
            out: self.out.clone(),
            inc: self.inc.clone(),
        }
    }
}

/// A 128-bit hash.
pub type Hash128 = u128;

fn h128(parts: &[&[u8]]) -> Hash128 {
    let mut x = blake3::Hasher::new();
    for p in parts {
        x.update(&(p.len() as u64).to_le_bytes());
        x.update(p);
    }
    let b = x.finalize();
    let mut a = [0u8; 16];
    a.copy_from_slice(&b.as_bytes()[..16]);
    u128::from_le_bytes(a)
}

/// Hash of arbitrary bytes (node label keys).
pub fn hash_bytes(b: &[u8]) -> Hash128 {
    h128(&[b])
}

fn end_bytes(l: &EdgeLabel) -> Vec<u8> {
    postcard::to_stdvec(l).unwrap_or_default()
}

/// The result of colour refinement: the stable colour of every node and the graph's hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wl {
    pub colors: Vec<Hash128>,
    pub hash: Hash128,
    pub rounds: u32,
}

/// 1-WL colour refinement on the directed, edge-labelled graph: `c0(v) = init(v)`,
/// `c_{t+1}(v) = H(c_t(v), sorted {(out, label, c_t(w))}, sorted {(in, label, c_t(u))})`, until the number
/// of colours stops growing (the partition is stable) or `max_rounds`. The graph hash covers the sorted
/// colour multiset and the node and edge counts. Stable colours are isomorphism-invariant: an
/// isomorphism maps every node to a node of the same colour.
pub fn wl<N>(g: &Graph<N>, init: impl Fn(&N) -> Hash128, max_rounds: u32) -> Wl {
    let n = g.node_count();
    let mut c: Vec<Hash128> = g.nodes.iter().map(&init).collect();
    let distinct = |c: &[Hash128]| {
        let mut s = c.to_vec();
        s.sort_unstable();
        s.dedup();
        s.len()
    };
    let mut k = distinct(&c);
    let labels: Vec<Vec<(u32, Vec<u8>)>> = g
        .out
        .iter()
        .map(|o| o.iter().map(|(v, l)| (*v, end_bytes(l))).collect())
        .collect();
    let mut inl: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n];
    for (u, o) in g.out.iter().enumerate() {
        for (i, (v, _)) in o.iter().enumerate() {
            inl[*v as usize].push((u as u32, i));
        }
    }
    let mut rounds = 0;
    while rounds < max_rounds {
        rounds += 1;
        let next: Vec<Hash128> = (0..n)
            .map(|v| {
                let mut outs: Vec<(Vec<u8>, Hash128)> = labels[v]
                    .iter()
                    .map(|(w, l)| (l.clone(), c[*w as usize]))
                    .collect();
                outs.sort_unstable();
                let mut ins: Vec<(&[u8], Hash128)> = inl[v]
                    .iter()
                    .map(|&(u, i)| (labels[u as usize][i].1.as_slice(), c[u as usize]))
                    .collect();
                ins.sort_unstable();
                let mut buf = Vec::new();
                buf.extend_from_slice(&c[v].to_le_bytes());
                buf.push(b'o');
                for (l, x) in &outs {
                    buf.extend_from_slice(&(l.len() as u32).to_le_bytes());
                    buf.extend_from_slice(l);
                    buf.extend_from_slice(&x.to_le_bytes());
                }
                buf.push(b'i');
                for (l, x) in &ins {
                    buf.extend_from_slice(&(l.len() as u32).to_le_bytes());
                    buf.extend_from_slice(l);
                    buf.extend_from_slice(&x.to_le_bytes());
                }
                h128(&[&buf])
            })
            .collect();
        let k2 = distinct(&next);
        c = next;
        if k2 == k {
            break;
        }
        k = k2;
    }
    let mut s = c.clone();
    s.sort_unstable();
    let bytes: Vec<u8> = s.iter().flat_map(|x| x.to_le_bytes()).collect();
    let hash = h128(&[
        b"graph",
        &(n as u64).to_le_bytes(),
        &(g.end_count() as u64).to_le_bytes(),
        &bytes,
    ]);
    Wl {
        colors: c,
        hash,
        rounds,
    }
}

/// `vf2::Graph` over a `Graph`: the node label is the node index, so the caller's predicate can look at
/// anything indexed by node (labels, colours, child ids).
struct View<'a, N> {
    g: &'a Graph<N>,
    ids: Vec<usize>,
}

impl<N> vf2::Graph for View<'_, N> {
    type NodeLabel = usize;
    type EdgeLabel = EdgeLabel;

    fn is_directed(&self) -> bool {
        true
    }

    fn node_count(&self) -> usize {
        self.g.node_count()
    }

    fn node_label(&self, node: vf2::NodeIndex) -> Option<&usize> {
        self.ids.get(node)
    }

    fn neighbors(
        &self,
        node: vf2::NodeIndex,
        direction: vf2::Direction,
    ) -> impl Iterator<Item = vf2::NodeIndex> {
        let (o, i): (&[(u32, EdgeLabel)], &[u32]) = match direction {
            vf2::Direction::Outgoing => (&self.g.out[node], &[]),
            vf2::Direction::Incoming => (&[], &self.g.inc[node]),
        };
        o.iter()
            .map(|e| e.0 as usize)
            .chain(i.iter().map(|&u| u as usize))
    }

    fn contains_edge(&self, source: vf2::NodeIndex, target: vf2::NodeIndex) -> bool {
        self.g.edge(source as u32, target as u32).is_some()
    }

    fn edge_label(&self, source: vf2::NodeIndex, target: vf2::NodeIndex) -> Option<&EdgeLabel> {
        self.g.edge(source as u32, target as u32)
    }
}

/// The search gave up after `budget` candidate pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetExceeded(pub u64);

/// An isomorphism `a -> b` (`map[i]` is the node of `b` that node `i` of `a` maps to), or `None`.
/// `node_eq(i, j)` decides whether node `i` of `a` may map to node `j` of `b`; edges must have equal
/// labels. Node and edge counts are compared first: with equal counts a label-preserving monomorphism is
/// an isomorphism, and VF2 is run in its isomorphism mode. `budget` bounds the number of candidate pairs.
pub fn isomorphism<N, M>(
    a: &Graph<N>,
    b: &Graph<M>,
    node_eq: impl Fn(usize, usize) -> bool,
    budget: u64,
) -> Result<Option<Vec<usize>>, BudgetExceeded> {
    if a.node_count() != b.node_count() || a.edge_count() != b.edge_count() || a.end_count() != b.end_count()
    {
        return Ok(None);
    }
    if a.node_count() == 0 {
        return Ok(Some(Vec::new()));
    }
    let va = View {
        g: a,
        ids: (0..a.node_count()).collect(),
    };
    let vb = View {
        g: b,
        ids: (0..b.node_count()).collect(),
    };
    let calls = std::cell::Cell::new(0u64);
    let over = std::cell::Cell::new(false);
    let found = vf2::isomorphisms(&va, &vb)
        .node_eq(|&i: &usize, &j: &usize| {
            let k = calls.get() + 1;
            calls.set(k);
            if k > budget {
                over.set(true);
                return false;
            }
            node_eq(i, j)
        })
        .edge_eq(|x: &EdgeLabel, y: &EdgeLabel| x == y)
        .first();
    if over.get() {
        return Err(BudgetExceeded(budget));
    }
    Ok(found)
}

/// Check that `map` (node `i` of `a` -> node `map[i]` of `b`) is an isomorphism: a bijection, `node_eq`
/// on every pair, and every edge of `a` maps to an edge of `b` with an equal label, with equal edge counts
/// (so the edge map is a bijection too). O(V + E log d) and independent of the search.
pub fn check_isomorphism<N, M>(
    a: &Graph<N>,
    b: &Graph<M>,
    map: &[usize],
    mut node_eq: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    if a.node_count() != b.node_count() || map.len() != a.node_count() {
        return Err(format!(
            "node counts differ: {} vs {} (map {})",
            a.node_count(),
            b.node_count(),
            map.len()
        ));
    }
    if a.edge_count() != b.edge_count() || a.end_count() != b.end_count() {
        return Err(format!(
            "edge counts differ: {}/{} vs {}/{}",
            a.edge_count(),
            a.end_count(),
            b.edge_count(),
            b.end_count()
        ));
    }
    let mut seen = vec![false; b.node_count()];
    for (i, &j) in map.iter().enumerate() {
        if j >= b.node_count() || std::mem::replace(&mut seen[j], true) {
            return Err(format!("node {i} maps to {j}: not a bijection"));
        }
        node_eq(i, j).map_err(|e| format!("node {i} -> {j}: {e}"))?;
    }
    for (u, o) in a.out.iter().enumerate() {
        for (v, l) in o {
            match b.edge(map[u] as u32, map[*v as usize] as u32) {
                Some(m) if m == l => {}
                Some(m) => return Err(format!("edge {u}->{v}: label {l:?} maps to {m:?}")),
                None => return Err(format!("edge {u}->{v} has no image")),
            }
        }
    }
    Ok(())
}

/// Union-find over `0..n` (path halving, union by index: the smaller root wins, so roots are
/// deterministic).
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
        }
    }

    pub fn push(&mut self) -> u32 {
        let i = self.parent.len() as u32;
        self.parent.push(i);
        i
    }

    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    pub fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = p;
            x = p;
        }
        x
    }

    pub fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi as usize] = lo;
        }
    }

    /// Dense class numbers in order of first occurrence (`0..n`).
    pub fn classes(&mut self) -> (Vec<u32>, u32) {
        let mut ids: HashMap<u32, u32> = HashMap::new();
        let mut out = Vec::with_capacity(self.parent.len());
        for x in 0..self.parent.len() as u32 {
            let r = self.find(x);
            let k = ids.len() as u32;
            out.push(*ids.entry(r).or_insert(k));
        }
        (out, ids.len() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(res: u32, port: u32) -> EdgeEnd {
        EdgeEnd {
            res,
            port: PortLabel::Port(port),
        }
    }
    const STAR: EdgeEnd = EdgeEnd {
        res: 0,
        port: PortLabel::Commutative,
    };

    /// `t = mul(a, b); y = add(t, c)` with commutative ports.
    fn pq(order: &[&str]) -> Graph<&'static str> {
        let mut g = Graph::default();
        let mut id = HashMap::new();
        for &n in order {
            let l: &'static str = match n {
                "a" | "b" | "c" => "param",
                "mul" => "mul",
                _ => "add",
            };
            id.insert(n, g.add_node(l));
        }
        g.add_edge(id["a"], id["mul"], STAR);
        g.add_edge(id["b"], id["mul"], STAR);
        g.add_edge(id["mul"], id["add"], STAR);
        g.add_edge(id["c"], id["add"], STAR);
        g
    }

    fn iso(a: &Graph<&str>, b: &Graph<&str>) -> Option<Vec<usize>> {
        let (wa, wb) = (
            wl(a, |n| hash_bytes(n.as_bytes()), 16),
            wl(b, |n| hash_bytes(n.as_bytes()), 16),
        );
        isomorphism(a, b, |i, j| wa.colors[i] == wb.colors[j], 1 << 20).unwrap()
    }

    #[test]
    fn commutative_graphs_match_in_any_order() {
        let a = pq(&["a", "b", "c", "mul", "add"]);
        let b = pq(&["add", "c", "mul", "b", "a"]);
        let wa = wl(&a, |n| hash_bytes(n.as_bytes()), 16);
        let wb = wl(&b, |n| hash_bytes(n.as_bytes()), 16);
        assert_eq!(wa.hash, wb.hash);
        let m = iso(&a, &b).unwrap();
        check_isomorphism(&a, &b, &m, |i, j| {
            (a.nodes[i] == b.nodes[j])
                .then_some(())
                .ok_or_else(|| "label".into())
        })
        .unwrap();
    }

    #[test]
    fn ports_distinguish_non_commutative_operands() {
        // sub(x, y) vs sub(y, x) with x = neg(p), y = p.
        let mk = |swap: bool| {
            let mut g = Graph::default();
            let p = g.add_node("param");
            let n = g.add_node("neg");
            let s = g.add_node("sub");
            g.add_edge(p, n, e(0, 0));
            let (x, y) = if swap { (p, n) } else { (n, p) };
            g.add_edge(x, s, e(0, 0));
            g.add_edge(y, s, e(0, 1));
            g
        };
        assert!(iso(&mk(false), &mk(false)).is_some());
        assert!(iso(&mk(false), &mk(true)).is_none());
    }

    #[test]
    fn parallel_edges_merge_and_count() {
        // x * x vs x * y.
        let mut a = Graph::default();
        let x = a.add_node("p");
        let m = a.add_node("mul");
        let _y = a.add_node("p");
        a.add_edge(x, m, STAR);
        a.add_edge(x, m, STAR);
        let mut b = Graph::default();
        let x = b.add_node("p");
        let m = b.add_node("mul");
        let y = b.add_node("p");
        b.add_edge(x, m, STAR);
        b.add_edge(y, m, STAR);
        assert_eq!(a.edge_count(), 1);
        assert_eq!(a.end_count(), 2);
        assert!(iso(&a, &b).is_none());
    }

    #[test]
    fn automorphic_adder_tree_matches() {
        // ((a + b) + (c + d)) built in two orders.
        let mk = |rev: bool| {
            let mut g = Graph::default();
            let leaves: Vec<u32> = (0..4).map(|_| g.add_node("p")).collect();
            let s1 = g.add_node("add");
            let s2 = g.add_node("add");
            let r = g.add_node("add");
            let l = if rev { vec![3, 2, 1, 0] } else { vec![0, 1, 2, 3] };
            g.add_edge(leaves[l[0]], s1, STAR);
            g.add_edge(leaves[l[1]], s1, STAR);
            g.add_edge(leaves[l[2]], s2, STAR);
            g.add_edge(leaves[l[3]], s2, STAR);
            g.add_edge(s1, r, STAR);
            g.add_edge(s2, r, STAR);
            g
        };
        assert!(iso(&mk(false), &mk(true)).is_some());
    }

    #[test]
    fn a_wrong_map_is_rejected() {
        let a = pq(&["a", "b", "c", "mul", "add"]);
        let m = vec![0, 1, 2, 4, 3];
        let r = check_isomorphism(&a, &a, &m, |i, j| {
            (a.nodes[i] == a.nodes[j])
                .then_some(())
                .ok_or_else(|| "label".into())
        });
        assert!(r.is_err());
    }

    #[test]
    fn budget_is_enforced() {
        let a = pq(&["a", "b", "c", "mul", "add"]);
        assert_eq!(isomorphism(&a, &a, |_, _| true, 2), Err(BudgetExceeded(2)));
    }

    #[test]
    fn union_find_classes_are_first_occurrence() {
        let mut u = UnionFind::new(5);
        u.union(3, 1);
        u.union(4, 0);
        assert_eq!(u.classes(), (vec![0, 1, 2, 1, 0], 3));
    }
}
