//! Dim and element classes of an abstract graph (docs/passes/components.md §4).
//!
//! A slot is one axis of one result of a node (dims) or one result (element types). The op table's
//! `dims` rules and element rules relate the slots of each op's operands (their producers' results) and
//! results; a composite call relates its operands and results as its child component's interface does.
//! The classes are the connected slots, numbered by first occurrence in node, result and axis order. They
//! depend only on the labelled graph, so an isomorphism maps classes to classes.

use cc_ir::absgraph::{Graph, PortLabel, UnionFind};
use cc_ir::ops::{DimsIn, Side, Slot, dims::relations};
use cc_ir::{Dim, Elem, SymCtx};
use serde::{Deserialize, Serialize};

use super::graph::{ADim, ANode, Kind, rule_attrs};

/// The classes of a component's interface: per input `k` and per output `j`, the class of every axis
/// (`None` for the literal 1) and of the element type.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Iface {
    pub inputs: Vec<PortClasses>,
    pub outputs: Vec<PortClasses>,
}

/// The classes of one interface value: per axis (`None` for the literal 1), and of its element type.
pub type PortClasses = (Vec<Option<u32>>, u32);

/// The operand producers of every node by port: `(node, result)`. Commutative operands take the ports
/// `0, 1, …` in edge order (the rules relate them symmetrically).
pub fn operands(g: &Graph<ANode>) -> Vec<Vec<Option<(u32, u32)>>> {
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

struct Slots {
    /// First dim slot of every (node, result).
    dim0: Vec<Vec<u32>>,
    /// Elem slot of every (node, result).
    elem: Vec<Vec<u32>>,
    nd: u32,
    ne: u32,
}

fn slots(g: &Graph<ANode>) -> Slots {
    let (mut nd, mut ne) = (0u32, 0u32);
    let mut dim0 = Vec::with_capacity(g.node_count());
    let mut elem = Vec::with_capacity(g.node_count());
    for n in &g.nodes {
        let mut d = Vec::new();
        let mut e = Vec::new();
        for t in &n.results {
            d.push(nd);
            nd += t.dims.len() as u32;
            e.push(ne);
            ne += 1;
        }
        dim0.push(d);
        elem.push(e);
    }
    Slots { dim0, elem, nd, ne }
}

/// Compute and store the classes of `g` (overwriting the class numbers in its result types). `iface`
/// gives the interface of a call node's child. Returns the numbers of dim and element classes.
pub fn assign(g: &mut Graph<ANode>, iface: &dyn Fn(u32) -> Option<Iface>) -> Result<(u32, u32), String> {
    let s = slots(g);
    let mut ud = UnionFind::new(s.nd as usize);
    let mut ue = UnionFind::new(s.ne as usize);
    let ops = operands(g);
    let is_one = |n: u32, r: u32, a: u32| -> bool {
        g.nodes[n as usize]
            .results
            .get(r as usize)
            .and_then(|t| t.dims.get(a as usize))
            .is_none_or(|d| *d == ADim::One)
    };
    let dslot = |n: u32, r: u32, a: u32| -> Option<u32> {
        let t = g.nodes[n as usize].results.get(r as usize)?;
        ((a as usize) < t.dims.len()).then(|| s.dim0[n as usize][r as usize] + a)
    };
    let eslot = |n: u32, r: u32| -> Option<u32> { s.elem[n as usize].get(r as usize).copied() };
    for (ni, node) in g.nodes.iter().enumerate() {
        let n = ni as u32;
        let Kind::Op { name, target } = &node.kind else {
            continue;
        };
        let prod = &ops[ni];
        if node.is_call() {
            let Some(ifc) = iface(n) else {
                return Err(format!("node {n}: no interface for the call's child"));
            };
            let mut dmap: std::collections::HashMap<u32, u32> = Default::default();
            let mut emap: std::collections::HashMap<u32, u32> = Default::default();
            let join =
                |map: &mut std::collections::HashMap<u32, u32>, uf: &mut UnionFind, k: u32, s: u32| match map
                    .get(&k)
                {
                    Some(&t) => uf.union(t, s),
                    None => {
                        map.insert(k, s);
                    }
                };
            for (k, (dims, e)) in ifc.inputs.iter().enumerate() {
                let Some(Some((pn, pr))) = prod.get(k).copied() else {
                    continue;
                };
                for (a, c) in dims.iter().enumerate() {
                    if let (Some(c), Some(sl)) = (c, dslot(pn, pr, a as u32))
                        && !is_one(pn, pr, a as u32)
                    {
                        join(&mut dmap, &mut ud, *c, sl);
                    }
                }
                if let Some(sl) = eslot(pn, pr) {
                    join(&mut emap, &mut ue, *e, sl);
                }
            }
            for (j, (dims, e)) in ifc.outputs.iter().enumerate() {
                for (a, c) in dims.iter().enumerate() {
                    if let (Some(c), Some(sl)) = (c, dslot(n, j as u32, a as u32))
                        && !is_one(n, j as u32, a as u32)
                    {
                        join(&mut dmap, &mut ud, *c, sl);
                    }
                }
                if let Some(sl) = eslot(n, j as u32) {
                    join(&mut emap, &mut ue, *e, sl);
                }
            }
            continue;
        }
        let Some(spec) = cc_ir::lookup(name, target.as_deref()) else {
            return Err(format!("node {n}: `{name}` is not in the op table"));
        };
        let opnd: Vec<Option<Vec<bool>>> = prod
            .iter()
            .map(|p| {
                p.and_then(|(pn, pr)| {
                    g.nodes[pn as usize]
                        .results
                        .get(pr as usize)
                        .map(|t| t.dims.iter().map(|d| *d == ADim::One).collect())
                })
            })
            .collect();
        let res: Vec<Vec<bool>> = node
            .results
            .iter()
            .map(|t| t.dims.iter().map(|d| *d == ADim::One).collect())
            .collect();
        let attrs = rule_attrs(node);
        let fam = |i: usize| {
            prod.get(i)
                .copied()
                .flatten()
                .and_then(|(pn, pr)| g.nodes[pn as usize].results.get(pr as usize).map(|t| &t.family))
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
        let slot = |x: Slot| -> Option<(u32, u32, u32)> { side(x.side).map(|(a, b)| (a, b, x.axis)) };
        for (a, b) in rel.dims {
            if let (Some((an, ar, aa)), Some((bn, br, ba))) = (slot(a), slot(b)) {
                let one_a = is_one(an, ar, aa);
                let one_b = is_one(bn, br, ba);
                if one_a != one_b {
                    return Err(format!("node {n}: rule relates a literal 1 with a variable dim"));
                }
                if let (false, Some(x), Some(y)) = (one_a, dslot(an, ar, aa), dslot(bn, br, ba)) {
                    ud.union(x, y);
                }
            }
        }
        for (a, b) in rel.elems {
            if let (Some((an, ar)), Some((bn, br))) = (side(a), side(b))
                && let (Some(x), Some(y)) = (eslot(an, ar), eslot(bn, br))
            {
                ue.union(x, y);
            }
        }
    }
    // Number the classes by first occurrence of a variable slot.
    let mut dnum: std::collections::HashMap<u32, u32> = Default::default();
    let mut enum_: std::collections::HashMap<u32, u32> = Default::default();
    let mut out: Vec<Vec<(Vec<ADim>, u32)>> = Vec::with_capacity(g.node_count());
    for (ni, node) in g.nodes.iter().enumerate() {
        let mut rs = Vec::new();
        for (ri, t) in node.results.iter().enumerate() {
            let dims = t
                .dims
                .iter()
                .enumerate()
                .map(|(a, d)| match d {
                    ADim::One => ADim::One,
                    ADim::Var(_) => {
                        let root = ud.find(s.dim0[ni][ri] + a as u32);
                        let k = dnum.len() as u32;
                        ADim::Var(*dnum.entry(root).or_insert(k))
                    }
                })
                .collect();
            let root = ue.find(s.elem[ni][ri]);
            let k = enum_.len() as u32;
            rs.push((dims, *enum_.entry(root).or_insert(k)));
        }
        out.push(rs);
    }
    for (node, rs) in g.nodes.iter_mut().zip(out) {
        for (t, (dims, e)) in node.results.iter_mut().zip(rs) {
            t.dims = dims;
            t.elem = e;
        }
    }
    Ok((dnum.len() as u32, enum_.len() as u32))
}

/// The interface of a decomposition graph: its `input k` nodes and the operands of its `func.return`.
pub fn iface(g: &Graph<ANode>) -> Iface {
    let classes = |n: u32, r: u32| -> PortClasses {
        let t = &g.nodes[n as usize].results[r as usize];
        (
            t.dims
                .iter()
                .map(|d| match d {
                    ADim::One => None,
                    ADim::Var(k) => Some(*k),
                })
                .collect(),
            t.elem,
        )
    };
    let mut inputs: Vec<(u32, PortClasses)> = g
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| match n.kind {
            Kind::Input { index: Some(k) } => Some((k, classes(i as u32, 0))),
            _ => None,
        })
        .collect();
    inputs.sort_by_key(|x| x.0);
    let ops = operands(g);
    let ret = g
        .nodes
        .iter()
        .position(|n| matches!(&n.kind, Kind::Op { name, .. } if name == "func.return"));
    let outputs = ret
        .map(|r| {
            ops[r]
                .iter()
                .map(|p| match p {
                    Some((pn, pr)) => classes(*pn, *pr),
                    None => (Vec::new(), u32::MAX),
                })
                .collect()
        })
        .unwrap_or_default();
    Iface {
        inputs: inputs.into_iter().map(|x| x.1).collect(),
        outputs,
    }
}

/// The bindings of a graph's classes from the concrete result types of its nodes. Every slot of a class
/// must carry the same dim (equal, or provably equal under the scope's bounds) and the same element type.
pub fn bind(
    g: &Graph<ANode>,
    types: &[Vec<cc_ir::TensorType>],
    nd: u32,
    ne: u32,
    ctx: &SymCtx,
) -> Result<(Vec<Dim>, Vec<Elem>), String> {
    let mut dims: Vec<Option<Dim>> = vec![None; nd as usize];
    let mut elems: Vec<Option<Elem>> = vec![None; ne as usize];
    for (ni, node) in g.nodes.iter().enumerate() {
        for (ri, t) in node.results.iter().enumerate() {
            let ct = &types[ni][ri];
            if ct.shape.len() != t.dims.len() {
                return Err(format!("node {ni} result {ri}: rank differs"));
            }
            for (d, c) in t.dims.iter().zip(&ct.shape) {
                match d {
                    ADim::One if *c != Dim::Const(1) => {
                        return Err(format!("node {ni} result {ri}: literal 1 bound to {c}"));
                    }
                    ADim::One => {}
                    ADim::Var(k) => match &dims[*k as usize] {
                        None => dims[*k as usize] = Some(c.clone()),
                        Some(b) if dim_eq(b, c, ctx) => {}
                        Some(b) => return Err(format!("dim class {k} bound to {b} and {c} (node {ni})")),
                    },
                }
            }
            match &elems[t.elem as usize] {
                None => elems[t.elem as usize] = Some(ct.elem.clone()),
                Some(b) if *b == ct.elem => {}
                Some(b) => {
                    return Err(format!(
                        "element class {} bound to {b} and {} (node {ni})",
                        t.elem, ct.elem
                    ));
                }
            }
        }
    }
    let dims = dims
        .into_iter()
        .enumerate()
        .map(|(k, d)| d.ok_or_else(|| format!("dim class {k} has no slot")))
        .collect::<Result<_, _>>()?;
    let elems = elems
        .into_iter()
        .enumerate()
        .map(|(k, e)| e.ok_or_else(|| format!("element class {k} has no slot")))
        .collect::<Result<_, _>>()?;
    Ok((dims, elems))
}

/// Equal dims: syntactically, or provably under the scope's bounds.
pub fn dim_eq(a: &Dim, b: &Dim, ctx: &SymCtx) -> bool {
    a == b || ctx.eq(&a.expr(), &b.expr()) == Some(true)
}
