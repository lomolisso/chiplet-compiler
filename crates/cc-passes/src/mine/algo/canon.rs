//! The canonical code of a small pattern: among the vertex orders that sort the labels, the one whose
//! sorted edge list `(pos(src), pos(dst), ends)` is lexicographically smallest. Two patterns are
//! isomorphic (labels and edge ends preserved) iff their codes are equal. The search is over the
//! permutations inside each label class, which `max_vertices` keeps small (at most 8! orders).

use cc_ir::absgraph::{EdgeLabel, PortLabel, hash_bytes};

/// The text of merged edge ends: `<res>p<i>` or `<res>*`, joined by `+`.
pub fn ends_text(l: &EdgeLabel) -> String {
    let parts: Vec<String> = l
        .iter()
        .map(|e| match e.port {
            PortLabel::Port(i) => format!("{}p{i}", e.res),
            PortLabel::Commutative => format!("{}*", e.res),
            p => format!("{}{p:?}", e.res),
        })
        .collect();
    parts.join("+")
}

/// The canonical form: `order[pos]` is the original vertex at canonical position `pos`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Canon {
    pub order: Vec<usize>,
    pub code: String,
    pub id: String,
}

/// The id of a code: `p` and 16 hex digits of its hash.
pub fn code_id(code: &str) -> String {
    format!("p{:016x}", (hash_bytes(code.as_bytes()) >> 64) as u64)
}

/// The code text of a pattern given in canonical order.
pub fn code_text(labels: &[&str], edges: &[(u32, u32, String)]) -> String {
    let e: Vec<String> = edges.iter().map(|(a, b, l)| format!("{a}>{b}:{l}")).collect();
    format!("{};{}", labels.join(","), e.join(","))
}

/// Canonical form of a pattern with vertex labels `labels` and edges `(src, dst, ends)`.
///
/// The search is factorial in the size of a label class: callers bound the pattern size first.
pub fn canon(labels: &[String], edges: &[(u32, u32, EdgeLabel)]) -> Canon {
    let n = labels.len();
    assert!(n <= 12, "canonical code of a {n}-vertex pattern");
    let mut base: Vec<usize> = (0..n).collect();
    base.sort_by(|&a, &b| labels[a].cmp(&labels[b]).then(a.cmp(&b)));
    let texts: Vec<(u32, u32, String)> = edges.iter().map(|(a, b, l)| (*a, *b, ends_text(l))).collect();
    type Best = (Vec<(u32, u32, String)>, Vec<usize>);
    struct S<'a> {
        labels: &'a [String],
        base: &'a [usize],
        texts: &'a [(u32, u32, String)],
        order: Vec<usize>,
        used: Vec<bool>,
        best: Option<Best>,
    }
    // Depth first over the positions: position `p` takes an unused vertex with the label of
    // `base[p]`; every complete order is scored by its sorted edge list.
    fn rec(s: &mut S) {
        let p = s.order.len();
        if p == s.base.len() {
            let mut pos = vec![0u32; p];
            for (i, &v) in s.order.iter().enumerate() {
                pos[v] = i as u32;
            }
            let mut e: Vec<(u32, u32, String)> = s
                .texts
                .iter()
                .map(|(a, b, l)| (pos[*a as usize], pos[*b as usize], l.clone()))
                .collect();
            e.sort();
            if s.best.as_ref().is_none_or(|(b, _)| e < *b) {
                s.best = Some((e, s.order.clone()));
            }
            return;
        }
        let want = &s.labels[s.base[p]];
        for k in 0..s.base.len() {
            let v = s.base[k];
            if !s.used[v] && &s.labels[v] == want {
                s.used[v] = true;
                s.order.push(v);
                rec(s);
                s.order.pop();
                s.used[v] = false;
            }
        }
    }
    let mut st = S {
        labels,
        base: &base,
        texts: &texts,
        order: Vec::with_capacity(n),
        used: vec![false; n],
        best: None,
    };
    rec(&mut st);
    let (e, order) = st.best.unwrap_or_default();
    let ls: Vec<&str> = order.iter().map(|&v| labels[v].as_str()).collect();
    let code = code_text(&ls, &e);
    let id = code_id(&code);
    Canon { order, code, id }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cc_ir::absgraph::EdgeEnd;

    fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
        if items.len() <= 1 {
            return vec![items.to_vec()];
        }
        let mut out = Vec::new();
        for i in 0..items.len() {
            let mut rest = items.to_vec();
            let x = rest.remove(i);
            for mut p in permutations(&rest) {
                p.insert(0, x);
                out.push(p);
            }
        }
        out
    }

    fn star() -> EdgeLabel {
        vec![EdgeEnd {
            res: 0,
            port: PortLabel::Commutative,
        }]
    }

    fn port(i: u32) -> EdgeLabel {
        vec![EdgeEnd {
            res: 0,
            port: PortLabel::Port(i),
        }]
    }

    /// `mul -> add <- mul`, `add -> sub` (port 1), under every relabelling of the vertices.
    #[test]
    fn the_code_is_invariant_under_vertex_renaming() {
        let labels = ["mul", "add", "mul", "sub"];
        let edges = [(0u32, 1u32, star()), (2, 1, star()), (1, 3, port(1))];
        let first = canon(&labels.map(String::from), &edges);
        for p in permutations(&[0, 1, 2, 3]) {
            // Vertex v becomes p[v].
            let mut ls = vec![String::new(); 4];
            for v in 0..4 {
                ls[p[v]] = labels[v].to_string();
            }
            let es: Vec<_> = edges
                .iter()
                .map(|(a, b, l)| (p[*a as usize] as u32, p[*b as usize] as u32, l.clone()))
                .collect();
            let c = canon(&ls, &es);
            assert_eq!(c.code, first.code);
            assert_eq!(c.id, first.id);
        }
        // A different port is a different pattern.
        let other = canon(
            &labels.map(String::from),
            &[(0, 1, star()), (2, 1, star()), (1, 3, port(0))],
        );
        assert_ne!(other.code, first.code);
        assert_eq!(first.code, "add,mul,mul,sub;0>3:0p1,1>0:0*,2>0:0*");
    }
}
