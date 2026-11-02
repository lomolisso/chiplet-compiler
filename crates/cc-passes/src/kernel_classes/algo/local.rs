//! Local classes of a kernel graph (docs/passes/kernel_classes.md §4.1).
//!
//! A kernel's nodes keep the class ids of their component, but ops outside the kernel can make two
//! component classes one, so those ids are not a function of the kernel. The kernel's own classes are
//! recomputed here from the op table's rules on the kernel graph alone, the way `components` computes a
//! component's classes, with one change: the literal 1 is an extent like any other (an axis of extent 1
//! is a parameter), except for an operand axis that a broadcast expands, which the rules leave apart.

use cc_ir::absgraph::{Graph, PortLabel, UnionFind};
use cc_ir::ops::{DimsIn, Side, Slot, dims::relations};

use crate::components::algo::graph::{AAttr, ADim, AType, Kind, rule_attrs};
use crate::kernel_classes::json::LocalClasses;
use crate::kernels::json::KNode;
use cc_ir::Attr;

/// The result types of a kernel node: a member's results, an input's value. An output has none: it is
/// the use of a value, not a value.
pub fn results(n: &KNode) -> &[AType] {
    match n {
        KNode::Member { node, .. } => &node.label.results,
        KNode::Input { ty, .. } => std::slice::from_ref(ty),
        KNode::Output { .. } => &[],
    }
}

/// The operand producers of every node by port, `(node, result)`. Commutative operands take the ports
/// `0, 1, …` in edge order (the rules relate them symmetrically); folded operands have none.
pub fn operands<N>(g: &Graph<N>) -> Vec<Vec<Option<(u32, u32)>>> {
    let mut out: Vec<Vec<Option<(u32, u32)>>> = vec![Vec::new(); g.node_count()];
    let mut comm: Vec<u32> = vec![0; g.node_count()];
    for (u, o) in g.out.iter().enumerate() {
        for (v, ends) in o {
            for e in ends {
                let p = match e.port {
                    PortLabel::Port(p) => p,
                    PortLabel::Commutative => {
                        let c = &mut comm[*v as usize];
                        *c += 1;
                        *c - 1
                    }
                    _ => continue,
                };
                let ops = &mut out[*v as usize];
                if ops.len() <= p as usize {
                    ops.resize(p as usize + 1, None);
                }
                ops[p as usize] = Some((u as u32, e.res));
            }
        }
    }
    out
}

/// `(name, target)` of a member op.
pub fn op_of(n: &KNode) -> Option<(&str, Option<&str>)> {
    match n {
        KNode::Member { node, .. } => match &node.label.kind {
            Kind::Op { name, target } => Some((name.as_str(), target.as_deref())),
            _ => None,
        },
        _ => None,
    }
}

/// A literal integer list attribute of a member.
pub fn lit_ints(n: &KNode, key: &str) -> Option<Vec<i64>> {
    match n {
        KNode::Member { node, .. } => match node.label.attrs.get(key)? {
            AAttr::Lit(a) => a.as_ints(),
            _ => None,
        },
        _ => None,
    }
}

/// A literal attribute of a member.
pub fn lit<'a>(n: &'a KNode, key: &str) -> Option<&'a Attr> {
    match n {
        KNode::Member { node, .. } => match node.label.attrs.get(key)? {
            AAttr::Lit(a) => Some(a),
            _ => None,
        },
        _ => None,
    }
}

fn is_broadcast(name: &str) -> bool {
    matches!(
        name,
        "stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim"
    )
}

/// The operand axes of a broadcast member that it expands: the operand's literal 1 at an axis whose
/// result axis is not the literal 1.
pub fn expanding_axes(g: &Graph<KNode>, ops: &[Vec<Option<(u32, u32)>>], n: u32) -> Vec<u32> {
    let x = &g.nodes[n as usize];
    let Some((name, _)) = op_of(x) else {
        return Vec::new();
    };
    if !is_broadcast(name) {
        return Vec::new();
    }
    let Some(Some((pn, pr))) = ops[n as usize].first().copied() else {
        return Vec::new();
    };
    let (Some(src), Some(res)) = (
        results(&g.nodes[pn as usize]).get(pr as usize),
        results(x).first(),
    ) else {
        return Vec::new();
    };
    let bd = lit_ints(x, "broadcast_dimensions").unwrap_or_default();
    (0..src.dims.len())
        .filter(|&a| {
            src.dims[a] == ADim::One
                && bd
                    .get(a)
                    .and_then(|&d| res.dims.get(d as usize))
                    .is_some_and(|d| *d != ADim::One)
        })
        .map(|a| a as u32)
        .collect()
}

/// The local classes of a kernel graph. Classes are numbered by first occurrence in node, result and
/// axis order; they depend only on the labelled graph, so an isomorphism maps classes to classes.
pub fn classes(g: &Graph<KNode>) -> Result<LocalClasses, String> {
    let ops = operands(g);
    let mut dim0: Vec<Vec<u32>> = Vec::with_capacity(g.node_count());
    let mut elem: Vec<Vec<u32>> = Vec::with_capacity(g.node_count());
    let (mut nd, mut ne) = (0u32, 0u32);
    for n in &g.nodes {
        let (mut d, mut e) = (Vec::new(), Vec::new());
        for t in results(n) {
            d.push(nd);
            nd += t.dims.len() as u32;
            e.push(ne);
            ne += 1;
        }
        dim0.push(d);
        elem.push(e);
    }
    let mut ud = UnionFind::new(nd as usize);
    let mut ue = UnionFind::new(ne as usize);
    let mut expanded = vec![false; nd as usize];
    let dslot = |n: u32, r: u32, a: u32| -> Option<u32> {
        let t = results(&g.nodes[n as usize]).get(r as usize)?;
        ((a as usize) < t.dims.len()).then(|| dim0[n as usize][r as usize] + a)
    };
    for (ni, x) in g.nodes.iter().enumerate() {
        let n = ni as u32;
        let Some((name, target)) = op_of(x) else {
            continue;
        };
        let Some(spec) = cc_ir::lookup(name, target) else {
            return Err(format!("node {n}: `{name}` is not in the op table"));
        };
        let prod = &ops[ni];
        let exp = expanding_axes(g, &ops, n);
        // Only a broadcast's expanded operand axes count as the literal 1 for the rules.
        let opnd: Vec<Option<Vec<bool>>> = prod
            .iter()
            .enumerate()
            .map(|(port, p)| {
                p.and_then(|(pn, pr)| {
                    results(&g.nodes[pn as usize]).get(pr as usize).map(|t| {
                        (0..t.dims.len() as u32)
                            .map(|a| port == 0 && exp.contains(&a))
                            .collect()
                    })
                })
            })
            .collect();
        if let Some(Some((pn, pr))) = prod.first().copied() {
            for &a in &exp {
                if let Some(s) = dslot(pn, pr, a) {
                    expanded[s as usize] = true;
                }
            }
        }
        let res: Vec<Vec<bool>> = results(x).iter().map(|t| vec![false; t.dims.len()]).collect();
        let KNode::Member { node, .. } = x else {
            continue;
        };
        let attrs = rule_attrs(&node.label);
        let fam = |i: usize| {
            prod.get(i)
                .copied()
                .flatten()
                .and_then(|(pn, pr)| results(&g.nodes[pn as usize]).get(pr as usize).map(|t| &t.family))
        };
        let rel = relations(
            spec,
            &DimsIn {
                operands: &opnd,
                results: &res,
                attrs: &attrs,
            },
            |a, b| fam(a).is_some() && fam(a) == fam(b),
        );
        let side = |sd: Side| -> Option<(u32, u32)> {
            match sd {
                Side::Operand(i) => prod.get(i as usize).copied().flatten(),
                Side::Result(i) => Some((n, i)),
            }
        };
        let slot = |x: Slot| -> Option<u32> { side(x.side).and_then(|(a, b)| dslot(a, b, x.axis)) };
        for (a, b) in rel.dims {
            if let (Some(x), Some(y)) = (slot(a), slot(b)) {
                ud.union(x, y);
            }
        }
        for (a, b) in rel.elems {
            if let (Some((an, ar)), Some((bn, br))) = (side(a), side(b))
                && let (Some(x), Some(y)) = (
                    elem[an as usize].get(ar as usize),
                    elem[bn as usize].get(br as usize),
                )
            {
                ue.union(*x, *y);
            }
        }
    }
    let mut dnum: std::collections::HashMap<u32, u32> = Default::default();
    let mut enumb: std::collections::HashMap<u32, u32> = Default::default();
    let mut out = LocalClasses::default();
    for (ni, x) in g.nodes.iter().enumerate() {
        let mut rd = Vec::new();
        let mut re = Vec::new();
        for (ri, t) in results(x).iter().enumerate() {
            let mut axes = Vec::with_capacity(t.dims.len());
            for a in 0..t.dims.len() as u32 {
                let s = dim0[ni][ri] + a;
                let root = ud.find(s);
                let k = dnum.len() as u32;
                let c = *dnum.entry(root).or_insert(k);
                if c as usize >= out.unit.len() {
                    out.unit.push(false);
                }
                out.unit[c as usize] |= expanded[s as usize];
                axes.push(c);
            }
            rd.push(axes);
            let root = ue.find(elem[ni][ri]);
            let k = enumb.len() as u32;
            re.push(*enumb.entry(root).or_insert(k));
        }
        out.dims.push(rd);
        out.elems.push(re);
    }
    out.nd = dnum.len() as u32;
    out.ne = enumb.len() as u32;
    Ok(out)
}

/// Whether some value has two axes in one class (then axis positions must stay in the identity).
pub fn repeated(l: &LocalClasses) -> bool {
    l.dims.iter().flatten().any(|axes| {
        let mut v = axes.clone();
        v.sort_unstable();
        v.windows(2).any(|w| w[0] == w[1])
    })
}
