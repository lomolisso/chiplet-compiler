//! The algorithms of `kernel_classes` (docs/passes/kernel_classes.md §4): local classes, the identity
//! graph, classification, and the instance rows.

pub mod classify;
pub mod encode;
pub mod local;
pub mod sharing;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use cc_ir::absgraph::Graph;
use cc_ir::{Attr, Dim, Elem};

use crate::components::algo::graph::{AAttr, ADim, ANode, instantiate};
use crate::components::json::Bindings;
use crate::kernel_classes::json::{ClassKind, Group, Groups, LocalClasses};
use crate::kernels::json::{KNode, Kernel};
use crate::normalize::json::{NNode, Order};
use local::{lit, op_of, operands, results};

/// What the pass needs of one kernel, once per kernel.
#[derive(Debug, Clone)]
pub struct KInfo {
    pub id: String,
    pub graph: Graph<KNode>,
    pub local: LocalClasses,
    /// The first slot `(node, result, axis)` of every local dim class.
    pub slot: Vec<(u32, u32, u32)>,
    /// The first slot `(node, result)` of every local element class.
    pub eslot: Vec<(u32, u32)>,
    /// Convolution nodes (their group counts are identity).
    pub convs: Vec<u32>,
    pub kind: ClassKind,
    /// `<op>/<family>` of the core, or `map`.
    pub core: String,
    pub omega: Option<Order>,
    /// Local classes of the iteration domain, with multiplicity.
    pub iteration: Vec<u32>,
    /// `reduce_window` nodes: their window sizes multiply the workload.
    pub windows: Vec<u32>,
    pub call_fed: bool,
    pub low_rank: bool,
    pub roles: Vec<String>,
}

/// The kind of a core op.
pub fn kind_of(name: &str, target: Option<&str>) -> ClassKind {
    match (name, target) {
        ("stablehlo.dot_general" | "stablehlo.convolution" | "stablehlo.dynamic_conv", _) => {
            ClassKind::Contraction
        }
        ("stablehlo.scatter", _) => ClassKind::ScatterReduce,
        _ => ClassKind::Reduction,
    }
}

/// Analyse one kernel of a component.
pub fn kernel_info(nc: &Graph<NNode>, k: &Kernel, g: Graph<KNode>) -> Result<KInfo, String> {
    let local = local::classes(&g)?;
    let mut slot = vec![(0, 0, 0); local.nd as usize];
    let mut seen = vec![false; local.nd as usize];
    let mut eslot = vec![(0, 0); local.ne as usize];
    let mut eseen = vec![false; local.ne as usize];
    for (n, rs) in local.dims.iter().enumerate() {
        for (r, axes) in rs.iter().enumerate() {
            for (a, &c) in axes.iter().enumerate() {
                if !std::mem::replace(&mut seen[c as usize], true) {
                    slot[c as usize] = (n as u32, r as u32, a as u32);
                }
            }
            let e = local.elems[n][r];
            if !std::mem::replace(&mut eseen[e as usize], true) {
                eslot[e as usize] = (n as u32, r as u32);
            }
        }
    }
    let ops = operands(&g);
    let classes_of = |n: u32, r: u32| -> Vec<u32> {
        local.dims[n as usize]
            .get(r as usize)
            .cloned()
            .unwrap_or_default()
    };
    let operand_classes = |n: u32, port: u32| -> Vec<u32> {
        ops[n as usize]
            .get(port as usize)
            .copied()
            .flatten()
            .map(|(pn, pr)| classes_of(pn, pr))
            .unwrap_or_default()
    };
    let core = k.core.and_then(|c| {
        g.nodes
            .iter()
            .position(|x| matches!(x, KNode::Member { origin, .. } if *origin == c))
    });
    let convs = g
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, x)| {
            matches!(
                op_of(x),
                Some(("stablehlo.convolution" | "stablehlo.dynamic_conv", _))
            )
        })
        .map(|(i, _)| i as u32)
        .collect();
    let (kind, core_key, omega, iteration, windows) = match core {
        Some(ci) => {
            let x = &g.nodes[ci];
            let (name, target) = op_of(x).ok_or("core is not an op")?;
            let ci = ci as u32;
            let omega = match x {
                KNode::Member { node, .. } => node.omega.as_ref().map(|o| o.order.clone()),
                _ => None,
            };
            let mut it = classes_of(ci, 0);
            let mut windows = Vec::new();
            match name {
                "stablehlo.dot_general" => {
                    if let Some(Attr::DotDims(d)) = lit(x, "dot_dimension_numbers") {
                        let lhs = operand_classes(ci, 0);
                        it.extend(
                            d.lhs_contracting
                                .iter()
                                .filter_map(|&a| lhs.get(a as usize).copied()),
                        );
                    }
                }
                "stablehlo.convolution" | "stablehlo.dynamic_conv" => {
                    if let Some(Attr::ConvDims(d)) = lit(x, "dimension_numbers") {
                        let ker = operand_classes(ci, 1);
                        it.extend(
                            d.kernel_spatial
                                .iter()
                                .filter_map(|&a| ker.get(a as usize).copied()),
                        );
                        it.extend(ker.get(d.kernel_input_feature as usize).copied());
                    }
                }
                "stablehlo.reduce" => it = operand_classes(ci, 0),
                "stablehlo.reduce_window" => windows.push(ci),
                "stablehlo.scatter" => it = operand_classes(ci, results(x).len() as u32 + 1),
                _ => {}
            }
            (
                kind_of(name, target),
                crate::kernels::core_key(nc, k.core.unwrap_or_default()),
                omega,
                it,
                windows,
            )
        }
        None => {
            // Every map member is over the domain (K4); take the widest.
            let it = g
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, x)| matches!(x, KNode::Member { .. }))
                .map(|(i, _)| classes_of(i as u32, 0))
                .max_by_key(Vec::len)
                .unwrap_or_default();
            (ClassKind::Map, "map".to_string(), None, it, Vec::new())
        }
    };
    let call_fed = g.nodes.iter().any(|x| {
        matches!(x, KNode::Input { src, .. } if nc.nodes.get(*src as usize).is_some_and(|s| s.label.is_call()))
    });
    let roles = g
        .nodes
        .iter()
        .map(|x| match x {
            KNode::Member { role, .. } => role.name().to_string(),
            KNode::Input { .. } => "input".into(),
            KNode::Output { .. } => "output".into(),
        })
        .collect();
    Ok(KInfo {
        id: k.id.clone(),
        low_rank: kind == ClassKind::Map && k.domain.len() <= 1,
        graph: g,
        local,
        slot,
        eslot,
        convs,
        kind,
        core: core_key,
        omega,
        iteration,
        windows,
        call_fed,
        roles,
    })
}

/// One instance's view of its component: the bindings with element types parsed, and the phase's
/// symbol sizes.
#[derive(Debug)]
pub struct Inst<'a> {
    pub b: &'a Bindings,
    pub elems: Vec<Elem>,
    pub env: &'a BTreeMap<String, i64>,
}

impl Inst<'_> {
    /// The extent of a component dim at the scenario sizes, and its symbolic form.
    pub fn dim(&self, d: &ADim) -> Result<(u64, Dim), String> {
        let dim = match d {
            ADim::One => Dim::Const(1),
            ADim::Var(c) => self
                .b
                .dims
                .get(*c as usize)
                .cloned()
                .ok_or_else(|| format!("dim class {c} unbound"))?,
        };
        let v = match &dim {
            Dim::Const(v) => *v,
            Dim::Sym(e) => {
                let v = e.eval(self.env).map_err(|x| format!("{e}: {x:?}"))?;
                u64::try_from(v).map_err(|_| format!("{e} = {v} at the scenario"))?
            }
        };
        Ok((v, dim))
    }

    /// The node's attributes with this instance's parameters.
    pub fn attrs(&self, label: &ANode, origin: u32) -> Result<cc_ir::Attrs, String> {
        let p = self
            .b
            .params
            .get(origin as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        instantiate(label, &self.b.dims, &self.elems, p).map(|c| c.attrs)
    }
}

/// A row before classification: in the member kernel's own numbering.
#[derive(Debug, Clone)]
pub struct RawRow {
    pub model: String,
    pub phase: String,
    pub instance: String,
    pub component: String,
    pub kernel: u32,
    pub groups: Groups,
    /// Per local dim class: extent, symbolic size.
    pub extents: Vec<(u64, String)>,
    pub symbolic: bool,
    pub elems: Vec<String>,
    /// `(node, key, value)`: size attributes, dropped literal parameters, folded constants.
    pub values: Vec<(u32, String, Attr)>,
    pub workload: u64,
    pub trip_count: u64,
}

/// The group kinds of a kernel's convolutions in one instance.
pub fn groups(k: &KInfo, inst: &Inst<'_>) -> Result<Groups, String> {
    let ops = operands(&k.graph);
    let mut out = Groups::new();
    for &n in &k.convs {
        let KNode::Member { node, .. } = &k.graph.nodes[n as usize] else {
            continue;
        };
        let attrs = inst.attrs(&node.label, node.origin)?;
        let Some(Attr::ConvDims(d)) = attrs.get("dimension_numbers") else {
            continue;
        };
        let mut gs = BTreeMap::new();
        if let Some(fgc) = attrs.get("feature_group_count").and_then(Attr::as_i64) {
            let c = ops[n as usize]
                .first()
                .copied()
                .flatten()
                .and_then(|(pn, pr)| results(&k.graph.nodes[pn as usize]).get(pr as usize).cloned())
                .and_then(|t| t.dims.get(d.input_feature as usize).copied())
                .ok_or("convolution without an input")?;
            let (cv, _) = inst.dim(&c)?;
            gs.insert(
                "feature_group_count".to_string(),
                match fgc {
                    1 => Group::One,
                    g if g as u64 == cv => Group::Depthwise,
                    _ => Group::Grouped,
                },
            );
        }
        if let Some(bgc) = attrs.get("batch_group_count").and_then(Attr::as_i64) {
            gs.insert(
                "batch_group_count".to_string(),
                if bgc == 1 { Group::One } else { Group::Grouped },
            );
        }
        out.insert(n, gs);
    }
    Ok(out)
}

/// The values of one instance: every size attribute (all entries), the dropped literal parameters, and
/// every folded constant.
pub fn values(k: &KInfo, inst: &Inst<'_>) -> Result<Vec<(u32, String, Attr)>, String> {
    let mut out = Vec::new();
    for (i, x) in k.graph.nodes.iter().enumerate() {
        let KNode::Member { node, .. } = x else { continue };
        let i = i as u32;
        let has = node
            .label
            .attrs
            .iter()
            .any(|(key, a)| matches!(a, AAttr::Sized { .. }) || encode::DROPPED.contains(&key.as_str()));
        if has {
            let attrs = inst.attrs(&node.label, node.origin)?;
            for (key, a) in &node.label.attrs {
                if (matches!(a, AAttr::Sized { .. })
                    || (encode::DROPPED.contains(&key.as_str()) && matches!(a, AAttr::Lit(_))))
                    && let Some(v) = attrs.get(key)
                {
                    out.push((i, key.clone(), v.clone()));
                }
            }
        }
        for (j, s) in node.splats.iter().enumerate() {
            let v = inst
                .b
                .params
                .get(s.constant as usize)
                .and_then(|p| {
                    p.iter().find_map(|x| match x {
                        crate::components::algo::graph::Param::Value(v) => Some(v.clone()),
                        _ => None,
                    })
                })
                .ok_or_else(|| format!("node {i}: folded constant {} has no value", s.constant))?;
            out.push((i, format!("splat{j}"), v));
        }
    }
    Ok(out)
}

/// The workload of one instance: the product of the iteration classes' extents, times the window sizes
/// of a `reduce_window`.
pub fn workload(k: &KInfo, extents: &[(u64, String)], values: &[(u32, String, Attr)]) -> Result<u64, String> {
    let mut w: u64 = 1;
    for &c in &k.iteration {
        w = w
            .checked_mul(extents[c as usize].0)
            .ok_or("workload overflows u64")?;
    }
    for &n in &k.windows {
        for (_, _, v) in values
            .iter()
            .filter(|(m, key, _)| *m == n && key == "window_dimensions")
        {
            for s in v.as_ints().unwrap_or_default() {
                w = w
                    .checked_mul(u64::try_from(s).unwrap_or(0))
                    .ok_or("workload overflows u64")?;
            }
        }
    }
    Ok(w)
}

/// One instance's row of a kernel, in the kernel's numbering.
/// Per local dim class its extent and symbolic size, whether any is symbolic, and per element class
/// its type.
pub type RowCore = (Vec<(u64, String)>, bool, Vec<String>);

pub fn raw_row(k: &KInfo, inst: &Inst<'_>) -> Result<RowCore, String> {
    let mut extents = Vec::with_capacity(k.slot.len());
    let mut symbolic = false;
    for &(n, r, a) in &k.slot {
        let t = &results(&k.graph.nodes[n as usize])[r as usize];
        let (v, d) = inst.dim(&t.dims[a as usize])?;
        if let Dim::Sym(e) = &d {
            symbolic |= !e.vars().is_empty();
        }
        extents.push((v, d.to_string()));
    }
    let mut elems = Vec::with_capacity(k.eslot.len());
    for &(n, r) in &k.eslot {
        let t = &results(&k.graph.nodes[n as usize])[r as usize];
        elems.push(
            inst.b
                .elems
                .get(t.elem as usize)
                .cloned()
                .ok_or_else(|| format!("element class {} unbound", t.elem))?,
        );
    }
    Ok((extents, symbolic, elems))
}
