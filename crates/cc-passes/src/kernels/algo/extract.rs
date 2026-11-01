//! Every kernel as a graph of its own (`KernelGraph`): its members in ascending node id, one `input`
//! node per outside value it reads (`(src, res)`, sorted) and one `output` node per member result used
//! outside (sorted). Edges between members keep their ends; an edge from an input keeps the member's port
//! and uses result 0; an edge to an output keeps the member's result and uses port 0.

use std::collections::{BTreeMap, BTreeSet};

use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};

use crate::kernels::json::{KNode, Kernel, KernelGraph};
use crate::normalize::json::NNode;

pub fn extract(g: &Graph<NNode>, k: &Kernel) -> Result<KernelGraph, String> {
    let ty = |n: u32, r: u32| {
        g.nodes[n as usize]
            .label
            .results
            .get(r as usize)
            .cloned()
            .ok_or_else(|| format!("{}: node {n} has no result {r}", k.id))
    };
    let mut out: Graph<KNode> = Graph::default();
    let mut local: BTreeMap<u32, u32> = BTreeMap::new();
    for m in &k.members {
        let i = out.add_node(KNode::Member {
            origin: m.node,
            role: m.role,
            node: g.nodes[m.node as usize].clone(),
        });
        local.insert(m.node, i);
    }
    let ins: BTreeSet<(u32, u32)> = k.inputs.iter().map(|c| (c.src, c.end.res)).collect();
    let mut in_node: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for &(src, res) in &ins {
        let i = out.add_node(KNode::Input {
            src,
            res,
            ty: ty(src, res)?,
        });
        in_node.insert((src, res), i);
    }
    let outs: BTreeSet<(u32, u32)> = k.outputs.iter().map(|c| (c.src, c.end.res)).collect();
    let mut out_node: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for &(node, res) in &outs {
        let i = out.add_node(KNode::Output {
            node,
            res,
            ty: ty(node, res)?,
        });
        out_node.insert((node, res), i);
    }
    for m in &k.members {
        for (dst, ends) in &g.out[m.node as usize] {
            if let Some(&d) = local.get(dst) {
                for e in ends {
                    out.add_edge(local[&m.node], d, *e);
                }
            }
        }
    }
    for c in &k.inputs {
        out.add_edge(
            in_node[&(c.src, c.end.res)],
            local[&c.dst],
            EdgeEnd {
                res: 0,
                port: c.end.port,
            },
        );
    }
    for &(node, res) in &outs {
        out.add_edge(
            local[&node],
            out_node[&(node, res)],
            EdgeEnd {
                res,
                port: PortLabel::Port(0),
            },
        );
    }
    Ok(KernelGraph {
        id: k.id.clone(),
        graph: out,
    })
}
