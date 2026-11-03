//! A brute-force frequent subgraph miner with GraMi's semantics (directed, labelled vertices and
//! edges, non-induced embeddings, MNI support), for small graphs: it enumerates every connected edge
//! subset up to `max_vertices` vertices, groups them by canonical code and counts MNI by backtracking.
//! Tests use it as an oracle for GraMi and for the re-matcher, and as GraMi's stand-in in smoke runs.

use std::collections::{BTreeMap, BTreeSet, HashSet};

/// A graph as integer lists: node labels and `(src, dst, label)` edges (one per ordered pair).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct G {
    pub nodes: Vec<u32>,
    pub edges: Vec<(u32, u32, u32)>,
}

/// Parse a `.lg` text (header line, `v`, then `e` lines).
pub fn parse_lg(text: &str) -> G {
    let mut g = G::default();
    for l in text.lines().skip(1) {
        let t: Vec<u32> = l.split(' ').skip(1).map(|x| x.parse().unwrap()).collect();
        if l.starts_with('v') {
            assert_eq!(t[0] as usize, g.nodes.len());
            g.nodes.push(t[1]);
        } else {
            g.edges.push((t[0], t[1], t[2]));
        }
    }
    g
}

impl G {
    pub fn to_lg(&self) -> String {
        let mut s = String::from("# t 1\n");
        for (i, l) in self.nodes.iter().enumerate() {
            s.push_str(&format!("v {i} {l}\n"));
        }
        let mut e = self.edges.clone();
        e.sort_unstable();
        for (a, b, l) in e {
            s.push_str(&format!("e {a} {b} {l}\n"));
        }
        s
    }
}

/// The canonical code: the smallest sorted edge list over the vertex orders that sort the labels.
pub fn code(p: &G) -> String {
    let n = p.nodes.len();
    let mut sorted = p.nodes.clone();
    sorted.sort_unstable();
    let mut best: Option<Vec<(u32, u32, u32)>> = None;
    let mut perm: Vec<usize> = Vec::new();
    let mut used = vec![false; n];
    fn rec(
        p: &G,
        sorted: &[u32],
        perm: &mut Vec<usize>,
        used: &mut [bool],
        best: &mut Option<Vec<(u32, u32, u32)>>,
    ) {
        let i = perm.len();
        if i == p.nodes.len() {
            let mut pos = vec![0u32; i];
            for (k, &v) in perm.iter().enumerate() {
                pos[v] = k as u32;
            }
            let mut e: Vec<(u32, u32, u32)> =
                p.edges.iter().map(|&(a, b, l)| (pos[a as usize], pos[b as usize], l)).collect();
            e.sort_unstable();
            if best.as_ref().is_none_or(|b| e < *b) {
                *best = Some(e);
            }
            return;
        }
        for v in 0..p.nodes.len() {
            if !used[v] && p.nodes[v] == sorted[i] {
                used[v] = true;
                perm.push(v);
                rec(p, sorted, perm, used, best);
                perm.pop();
                used[v] = false;
            }
        }
    }
    rec(p, &sorted, &mut perm, &mut used, &mut best);
    format!("{sorted:?}{:?}", best.unwrap_or_default())
}

/// Every embedding of `p` in `g` (non-induced, labels and edge labels equal).
pub fn embeddings(p: &G, g: &G) -> Vec<Vec<u32>> {
    let ge: BTreeMap<(u32, u32), u32> = g.edges.iter().map(|&(a, b, l)| ((a, b), l)).collect();
    let mut out = Vec::new();
    let mut map = Vec::new();
    let mut used = vec![false; g.nodes.len()];
    fn rec(
        p: &G,
        g: &G,
        ge: &BTreeMap<(u32, u32), u32>,
        map: &mut Vec<u32>,
        used: &mut [bool],
        out: &mut Vec<Vec<u32>>,
    ) {
        let i = map.len() as u32;
        if i as usize == p.nodes.len() {
            out.push(map.clone());
            return;
        }
        for h in 0..g.nodes.len() as u32 {
            if used[h as usize] || g.nodes[h as usize] != p.nodes[i as usize] {
                continue;
            }
            let ok = p.edges.iter().all(|&(a, b, l)| {
                let (ma, mb) = (
                    if a == i { Some(h) } else { map.get(a as usize).copied() },
                    if b == i { Some(h) } else { map.get(b as usize).copied() },
                );
                match (ma, mb) {
                    (Some(x), Some(y)) if a <= i && b <= i => ge.get(&(x, y)) == Some(&l),
                    _ => true,
                }
            });
            if ok {
                used[h as usize] = true;
                map.push(h);
                rec(p, g, ge, map, used, out);
                map.pop();
                used[h as usize] = false;
            }
        }
    }
    rec(p, g, &ge, &mut map, &mut used, &mut out);
    out
}

/// MNI support of `p` in `g`.
pub fn mni(p: &G, g: &G) -> u64 {
    let mut img: Vec<HashSet<u32>> = vec![HashSet::new(); p.nodes.len()];
    for m in embeddings(p, g) {
        for (i, h) in m.into_iter().enumerate() {
            img[i].insert(h);
        }
    }
    img.iter().map(|s| s.len() as u64).min().unwrap_or(0)
}

/// Every connected pattern with at least one edge and at most `max_vertices` vertices that occurs in
/// `g`, one per canonical code, with its MNI.
pub fn patterns(g: &G, max_vertices: usize) -> BTreeMap<String, (G, u64)> {
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let mut frontier: Vec<Vec<usize>> = (0..g.edges.len()).map(|e| vec![e]).collect();
    let mut found: BTreeMap<String, G> = BTreeMap::new();
    while let Some(s) = frontier.pop() {
        if !seen.insert(s.clone()) {
            continue;
        }
        let verts: BTreeSet<u32> = s.iter().flat_map(|&e| [g.edges[e].0, g.edges[e].1]).collect();
        let idx: BTreeMap<u32, u32> = verts.iter().enumerate().map(|(i, v)| (*v, i as u32)).collect();
        let p = G {
            nodes: verts.iter().map(|&v| g.nodes[v as usize]).collect(),
            edges: s
                .iter()
                .map(|&e| (idx[&g.edges[e].0], idx[&g.edges[e].1], g.edges[e].2))
                .collect(),
        };
        found.entry(code(&p)).or_insert(p);
        for (k, &(a, b, _)) in g.edges.iter().enumerate() {
            if s.contains(&k) || !(verts.contains(&a) || verts.contains(&b)) {
                continue;
            }
            let grows = !verts.contains(&a) as usize + !verts.contains(&b) as usize;
            if verts.len() + grows > max_vertices {
                continue;
            }
            let mut t = s.clone();
            t.push(k);
            t.sort_unstable();
            if !seen.contains(&t) {
                frontier.push(t);
            }
        }
    }
    found
        .into_iter()
        .map(|(c, p)| {
            let m = mni(&p, g);
            (c, (p, m))
        })
        .collect()
}

/// GraMi's `Output.txt` for the patterns of `g` with MNI ≥ `tau` and at most `max_vertices` vertices.
pub fn output(g: &G, tau: u64, max_vertices: usize) -> String {
    let frequent: Vec<G> = patterns(g, max_vertices)
        .into_values()
        .filter(|(_, m)| *m >= tau)
        .map(|(p, _)| p)
        .collect();
    let mut s = format!("0.0\n{}\n", frequent.len());
    for (i, p) in frequent.iter().enumerate() {
        s.push_str(&format!("{i}:\n"));
        for (k, l) in p.nodes.iter().enumerate() {
            s.push_str(&format!("v {k} {l}\n"));
        }
        for (a, b, l) in &p.edges {
            s.push_str(&format!("e {a} {b} {l}\n"));
        }
    }
    s
}

/// A random DAG (edges only forward) with `n` vertices, labels below `labels`, edge labels below
/// `edge_labels`, each forward pair an edge with probability `num/den`; deterministic in `seed`.
pub fn random_dag(seed: u64, n: usize, labels: u32, edge_labels: u32, num: u64, den: u64) -> G {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        x >> 33
    };
    let nodes = (0..n).map(|_| (next() % labels as u64) as u32).collect();
    let mut edges = Vec::new();
    for a in 0..n as u32 {
        for b in a + 1..n as u32 {
            if next() % den < num {
                edges.push((a, b, (next() % edge_labels as u64) as u32));
            }
        }
    }
    G { nodes, edges }
}

/// The patterns of GraMi's `Output.txt`, by canonical code.
pub fn parse_output(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut lines = text.lines().skip(1);
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let mut cur: Option<G> = None;
    for l in lines {
        if l.ends_with(':') {
            if let Some(g) = cur.take() {
                out.insert(code(&g));
            }
            cur = Some(G::default());
            continue;
        }
        let g = cur.as_mut().unwrap();
        let t: Vec<u32> = l.split(' ').skip(1).map(|x| x.parse().unwrap()).collect();
        if l.starts_with('v') {
            g.nodes.push(t[1]);
        } else {
            g.edges.push((t[0], t[1], t[2]));
        }
    }
    if let Some(g) = cur.take() {
        out.insert(code(&g));
    }
    assert_eq!(out.len(), n, "GraMi listed a pattern twice or miscounted");
    out
}
