//! The identity graph of a kernel (docs/passes/kernel_classes.md §4.2): the kernel graph without axis
//! positions, widths and values. Every local axis class and element class becomes a node, and every
//! statement about an axis ("this result has it", "this dot contracts it", "this window reads it") is a
//! role node between the op and the class (vertex colouring and edge subdivision; McKay and Piperno,
//! J. Symbolic Computation 60:94-112, 2014). Two kernels have isomorphic identity graphs iff they are
//! the same computation up to the order of each value's axes, the extents and the element widths.

use std::collections::BTreeMap;

use cc_ir::Attr;
use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};

use super::local::{expanding_axes, lit, lit_ints, op_of, operands, repeated, results};
use crate::components::algo::graph::{AAttr, Family};
use crate::kernel_classes::json::{Group, LocalClasses, ZNode, ZRole};
use crate::kernels::json::KNode;

/// Attributes that are not identity: hints, or values that only parameterize the datapath.
pub const DROPPED: [&str; 5] = [
    "precision_config",
    "indices_are_sorted",
    "unique_indices",
    "exponent_bits",
    "mantissa_bits",
];

/// The positional attributes of an op, which the encoding replaces by roles.
pub fn positional(name: &str) -> &'static [&'static str] {
    match name {
        "stablehlo.dot_general" => &["dot_dimension_numbers"],
        "stablehlo.convolution" | "stablehlo.dynamic_conv" => &["dimension_numbers"],
        "stablehlo.reduce" => &["dimensions"],
        "stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim" => &[
            "broadcast_dimensions",
            "known_expanding_dimensions",
            "known_nonexpanding_dimensions",
        ],
        "stablehlo.scatter" => &["scatter_dimension_numbers"],
        _ => &[],
    }
}

/// `compare` with `GT` or `GE`: written as `LT` / `LE` with its operands swapped (`a > b ⇔ b < a`, also
/// for NaN, IEEE 754-2019 §5.11).
pub fn swapped_compare(n: &KNode) -> bool {
    op_of(n).is_some_and(|(name, _)| name == "stablehlo.compare")
        && matches!(lit(n, "comparison_direction"), Some(Attr::Enum { value, .. }) if value == "GT" || value == "GE")
}

/// The canonical direction of a swapped comparison.
pub fn flipped(dir: &str) -> &str {
    match dir {
        "GT" => "LT",
        "GE" => "LE",
        d => d,
    }
}

/// A port of a swapped comparison: operands 0 and 1 exchange.
pub fn swap_port(p: PortLabel) -> PortLabel {
    match p {
        PortLabel::Port(0) => PortLabel::Port(1),
        PortLabel::Port(1) => PortLabel::Port(0),
        p => p,
    }
}

/// The operand that a reduction's ω axes index (docs/passes/normalize.md §4).
pub fn omega_port(name: &str, target: Option<&str>, results: usize) -> Option<u32> {
    match (name, target) {
        ("stablehlo.reduce" | "stablehlo.reduce_window" | "stablehlo.dot_general", _)
        | ("stablehlo.custom_call", Some("stablehlo.dynamic_reduce_window")) => Some(0),
        ("stablehlo.convolution" | "stablehlo.dynamic_conv", _) => Some(1),
        ("stablehlo.scatter", _) => Some(results as u32 + 1),
        _ => None,
    }
}

/// The identity label of a member node.
fn op_label(x: &KNode, groups: Option<&BTreeMap<String, Group>>) -> ZNode {
    let KNode::Member { node, .. } = x else {
        unreachable!("members only")
    };
    let swapped = swapped_compare(x);
    let name = op_of(x).map(|o| o.0).unwrap_or_default();
    let pos = positional(name);
    let mut attrs = BTreeMap::new();
    for (k, a) in &node.label.attrs {
        let AAttr::Lit(v) = a else { continue };
        if DROPPED.contains(&k.as_str()) || pos.contains(&k.as_str()) {
            continue;
        }
        let v = match v {
            Attr::Enum { kind, value } if swapped && k == "comparison_direction" => Attr::Enum {
                kind: kind.clone(),
                value: flipped(value).into(),
            },
            v => v.clone(),
        };
        attrs.insert(k.clone(), v);
    }
    let splats = node
        .splats
        .iter()
        .map(|s| {
            let p = if swapped { swap_port(s.port) } else { s.port };
            (p, s.ty.family.clone())
        })
        .collect();
    ZNode::Op {
        kind: node.label.kind.clone(),
        attrs,
        results: node.label.results.iter().map(|t| t.family.clone()).collect(),
        order: node.omega.as_ref().map(|o| o.order.clone()),
        splats,
        groups: groups.cloned().unwrap_or_default(),
    }
}

/// An encoded kernel: nodes `0..n` are the kernel's nodes, then `nd` axis classes, then `ne` element
/// classes, then role and window nodes.
#[derive(Debug, Clone)]
pub struct Encoded {
    pub graph: Graph<ZNode>,
    pub n: u32,
    pub nd: u32,
    pub ne: u32,
    pub positional: bool,
}

struct Builder<'a> {
    z: Graph<ZNode>,
    l: &'a LocalClasses,
    ops: Vec<Vec<Option<(u32, u32)>>>,
    n: u32,
    positional: bool,
}

const E0: EdgeEnd = EdgeEnd {
    res: 0,
    port: PortLabel::Port(0),
};

impl Builder<'_> {
    fn axis(&self, c: u32) -> u32 {
        self.n + c
    }
    fn elem(&self, e: u32) -> u32 {
        self.n + self.l.nd + e
    }
    /// A role node `from -> role -> to`.
    fn role(&mut self, from: u32, to: u32, role: ZRole, pos: Option<u32>) {
        let r = self.z.add_node(ZNode::Role {
            role,
            pos: if self.positional { pos } else { None },
        });
        self.z.add_edge(from, r, E0);
        self.z.add_edge(r, to, E0);
    }
    /// The class of axis `a` of operand `port` of node `n`.
    fn oc(&self, n: u32, port: u32, a: i64) -> Result<u32, String> {
        let (pn, pr) = self.ops[n as usize]
            .get(port as usize)
            .copied()
            .flatten()
            .ok_or_else(|| format!("node {n}: operand {port} missing"))?;
        self.l.dims[pn as usize]
            .get(pr as usize)
            .and_then(|r| r.get(usize::try_from(a).ok()?))
            .copied()
            .ok_or_else(|| format!("node {n}: operand {port} has no axis {a}"))
    }
    /// The class of axis `a` of result `r` of node `n`.
    fn rc(&self, n: u32, r: u32, a: i64) -> Result<u32, String> {
        self.l.dims[n as usize]
            .get(r as usize)
            .and_then(|x| x.get(usize::try_from(a).ok()?))
            .copied()
            .ok_or_else(|| format!("node {n}: result {r} has no axis {a}"))
    }
    fn operand_rank(&self, n: u32, port: u32) -> Option<usize> {
        let (pn, pr) = self.ops[n as usize].get(port as usize).copied().flatten()?;
        self.l.dims[pn as usize].get(pr as usize).map(Vec::len)
    }
    fn op_role(&mut self, n: u32, port: u32, a: i64, role: ZRole) -> Result<(), String> {
        let c = self.oc(n, port, a)?;
        let to = self.axis(c);
        self.role(n, to, role, Some(a as u32));
        Ok(())
    }
    fn res_role(&mut self, n: u32, a: i64, role: ZRole) -> Result<(), String> {
        let c = self.rc(n, 0, a)?;
        let to = self.axis(c);
        self.role(n, to, role, Some(a as u32));
        Ok(())
    }
    fn window(&mut self, n: u32, ends: &[(Option<u32>, i64, ZRole)]) -> Result<(), String> {
        let w = self.z.add_node(ZNode::Window);
        let to = w;
        self.role(n, to, ZRole::Spatial, None);
        for (port, a, role) in ends {
            let c = match port {
                Some(p) => self.oc(n, *p, *a)?,
                None => self.rc(n, 0, *a)?,
            };
            let to = self.axis(c);
            self.role(w, to, role.clone(), Some(*a as u32));
        }
        Ok(())
    }
}

/// Encode a kernel graph with its local classes; `groups` are the group kinds of its convolutions.
pub fn encode(
    g: &Graph<KNode>,
    l: &LocalClasses,
    groups: &BTreeMap<u32, BTreeMap<String, Group>>,
) -> Result<Encoded, String> {
    let n = g.node_count() as u32;
    let mut b = Builder {
        z: Graph::default(),
        l,
        ops: operands(g),
        n,
        positional: repeated(l),
    };
    let swapped: Vec<bool> = g.nodes.iter().map(swapped_compare).collect();
    for (i, x) in g.nodes.iter().enumerate() {
        let label = match x {
            KNode::Member { .. } => op_label(x, groups.get(&(i as u32))),
            KNode::Input { ty, .. } => ZNode::Input {
                family: ty.family.clone(),
            },
            KNode::Output { .. } => ZNode::Output,
        };
        b.z.add_node(label);
    }
    for _ in 0..l.nd {
        b.z.add_node(ZNode::Axis);
    }
    let mut efam: Vec<Option<Family>> = vec![None; l.ne as usize];
    for (i, x) in g.nodes.iter().enumerate() {
        for (r, t) in results(x).iter().enumerate() {
            efam[l.elems[i][r] as usize].get_or_insert_with(|| t.family.clone());
        }
    }
    for f in efam {
        b.z.add_node(ZNode::Elem {
            family: f.ok_or("element class without a value")?,
        });
    }
    for (u, o) in g.out.iter().enumerate() {
        for (v, ends) in o {
            for e in ends {
                let port = if swapped[*v as usize] {
                    swap_port(e.port)
                } else {
                    e.port
                };
                b.z.add_edge(u as u32, *v, EdgeEnd { res: e.res, port });
            }
        }
    }
    for (i, x) in g.nodes.iter().enumerate() {
        let i = i as u32;
        for (r, axes) in l.dims[i as usize].iter().enumerate() {
            for (a, &c) in axes.iter().enumerate() {
                let to = b.axis(c);
                b.role(i, to, ZRole::Axis { res: r as u32 }, Some(a as u32));
            }
            let to = b.elem(l.elems[i as usize][r]);
            b.role(i, to, ZRole::Elem { res: r as u32 }, None);
        }
        let Some((name, target)) = op_of(x) else { continue };
        match name {
            "stablehlo.dot_general" => {
                if let Some(Attr::DotDims(d)) = lit(x, "dot_dimension_numbers") {
                    for &a in &d.lhs_batching {
                        b.op_role(i, 0, a, ZRole::Batch)?;
                    }
                    for &a in &d.lhs_contracting {
                        b.op_role(i, 0, a, ZRole::Contract)?;
                    }
                }
            }
            "stablehlo.convolution" | "stablehlo.dynamic_conv" => {
                if let Some(Attr::ConvDims(d)) = lit(x, "dimension_numbers") {
                    b.op_role(i, 0, d.input_batch, ZRole::InBatch)?;
                    b.op_role(i, 0, d.input_feature, ZRole::InFeature)?;
                    b.op_role(i, 1, d.kernel_input_feature, ZRole::KernelIn)?;
                    b.op_role(i, 1, d.kernel_output_feature, ZRole::KernelOut)?;
                    b.res_role(i, d.output_batch, ZRole::OutBatch)?;
                    b.res_role(i, d.output_feature, ZRole::OutFeature)?;
                    for k in 0..d.input_spatial.len() {
                        b.window(
                            i,
                            &[
                                (Some(0), d.input_spatial[k], ZRole::WinIn),
                                (Some(1), d.kernel_spatial[k], ZRole::WinKernel),
                                (None, d.output_spatial[k], ZRole::WinOut),
                            ],
                        )?;
                    }
                }
            }
            "stablehlo.reduce" => {
                for a in lit_ints(x, "dimensions").unwrap_or_default() {
                    b.op_role(i, 0, a, ZRole::Reduced)?;
                }
            }
            "stablehlo.reduce_window" | "stablehlo.custom_call"
                if name == "stablehlo.reduce_window" || target == Some("stablehlo.dynamic_reduce_window") =>
            {
                for a in 0..b.operand_rank(i, 0).unwrap_or(0) as i64 {
                    b.window(i, &[(Some(0), a, ZRole::WinIn), (None, a, ZRole::WinOut)])?;
                }
            }
            "stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim" => {
                let exp = expanding_axes(g, &b.ops, i);
                let bd = lit_ints(x, "broadcast_dimensions").unwrap_or_default();
                let mapped: Vec<i64> = bd
                    .iter()
                    .enumerate()
                    .filter(|(a, _)| !exp.contains(&(*a as u32)))
                    .map(|(_, &d)| d)
                    .collect();
                let rank = l.dims[i as usize].first().map_or(0, Vec::len) as i64;
                for j in (0..rank).filter(|j| !mapped.contains(j)) {
                    b.res_role(i, j, ZRole::Expand)?;
                }
                for a in lit_ints(x, "known_expanding_dimensions").unwrap_or_default() {
                    b.op_role(i, 0, a, ZRole::KnownExpanding)?;
                }
                for a in lit_ints(x, "known_nonexpanding_dimensions").unwrap_or_default() {
                    b.op_role(i, 0, a, ZRole::KnownNonexpanding)?;
                }
            }
            "stablehlo.scatter" => {
                if let Some(Attr::ScatterDims(d)) = lit(x, "scatter_dimension_numbers") {
                    let k = results(x).len() as u32;
                    let (operand, indices, updates) = (0, k, k + 1);
                    let lists: [(u32, &[i64]); 5] = [
                        (updates, &d.update_window_dims),
                        (operand, &d.inserted_window_dims),
                        (operand, &d.input_batching_dims),
                        (indices, &d.scatter_indices_batching_dims),
                        (operand, &d.scattered_dims_to_operand_dims),
                    ];
                    for (li, (port, list)) in lists.iter().enumerate() {
                        for (j, &a) in list.iter().enumerate() {
                            b.op_role(
                                i,
                                *port,
                                a,
                                ZRole::Scatter {
                                    list: li as u32,
                                    i: j as u32,
                                },
                            )?;
                        }
                    }
                    if b.operand_rank(i, indices)
                        .is_some_and(|r| (d.index_vector_dim as usize) < r)
                    {
                        b.op_role(i, indices, d.index_vector_dim, ZRole::Scatter { list: 5, i: 0 })?;
                    }
                }
            }
            _ => {}
        }
        if let KNode::Member { node, .. } = x
            && let Some(om) = &node.omega
        {
            let port = omega_port(name, target, results(x).len())
                .ok_or_else(|| format!("node {i}: ω on `{name}`"))?;
            for (k, &a) in om.axes.iter().enumerate() {
                b.op_role(i, port, i64::from(a), ZRole::Omega { k: k as u32 })?;
            }
        }
    }
    Ok(Encoded {
        n,
        nd: l.nd,
        ne: l.ne,
        positional: b.positional,
        graph: b.z,
    })
}
