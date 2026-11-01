//! Normalization of one component: the rewrite rules to a fixpoint (`rules`), then ω on every reduction
//! (`omega`).

pub mod omega;
pub mod rules;
pub mod work;

use cc_ir::absgraph::Graph;

use crate::components::algo::graph::ANode;
use crate::normalize::config::Orders;
use crate::normalize::json::{NNode, Step};
use work::Work;

/// The input graph as a normalized graph: every node its own origin, nothing folded, no ω.
pub fn initial(g: &Graph<ANode>) -> Graph<NNode> {
    let mut out: Graph<NNode> = Graph::default();
    for (i, n) in g.nodes.iter().enumerate() {
        out.add_node(NNode {
            label: n.clone(),
            omega: None,
            splats: Vec::new(),
            origin: i as u32,
        });
    }
    out.out = g.out.clone();
    out.inc = g.inc.clone();
    out
}

/// The normalized graph and the ledger of one component.
#[derive(Debug, Clone)]
pub struct Normalized {
    pub graph: Graph<NNode>,
    pub steps: Vec<Step>,
}

pub fn normalize(g: &Graph<ANode>, orders: &Orders) -> Result<Normalized, String> {
    let mut w = Work::from_graph(&initial(g));
    let steps = rules::rewrite(&mut w)?;
    let ids: Vec<u32> = w.live().collect();
    for n in ids {
        let Some(x) = w.node(n) else { continue };
        if !omega::is_reduction(x) {
            continue;
        }
        let o =
            omega::omega(&w, n, orders).ok_or_else(|| format!("node {n}: cannot read the reduced axes"))?;
        if let Some(Some(x)) = w.nodes.get_mut(n as usize) {
            x.omega = Some(o);
        }
    }
    Ok(Normalized {
        graph: w.compact(),
        steps,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use cc_ir::Attr;
    use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};
    use cc_ir::attrs::DotDims;

    use super::*;
    use crate::components::algo::graph::{AAttr, ADim, AType, Family, Kind};
    use crate::normalize::config::OrderConfig;
    use crate::normalize::json::Order;

    fn orders() -> Orders {
        OrderConfig {
            bool: None,
            sint: None,
            uint: None,
            float: Some("tree:4".into()),
            complex: Some("seq".into()),
        }
        .parse()
        .unwrap()
    }

    fn t(dims: &[u32], elem: u32) -> AType {
        AType {
            family: Family::Float,
            elem,
            dims: dims.iter().map(|&d| ADim::Var(d)).collect(),
        }
    }

    fn node(kind: Kind, attrs: &[(&str, AAttr)], results: Vec<AType>) -> ANode {
        ANode {
            kind,
            attrs: attrs
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect::<BTreeMap<_, _>>(),
            results,
            child: None,
        }
    }

    fn op(name: &str, attrs: &[(&str, AAttr)], results: Vec<AType>) -> ANode {
        node(
            Kind::Op {
                name: name.into(),
                target: None,
            },
            attrs,
            results,
        )
    }

    fn input(k: u32, ty: AType) -> ANode {
        node(Kind::Input { index: Some(k) }, &[], vec![ty])
    }

    fn perm(p: &[i64]) -> (&'static str, AAttr) {
        ("permutation", AAttr::Lit(Attr::Ints(p.to_vec())))
    }

    fn e(port: u32) -> EdgeEnd {
        EdgeEnd {
            res: 0,
            port: PortLabel::Port(port),
        }
    }

    fn rules(n: &Normalized) -> Vec<&str> {
        n.steps.iter().map(|s| s.rule.as_str()).collect()
    }

    #[test]
    fn transpose_pair_to_identity() {
        let mut g: Graph<ANode> = Graph::default();
        let x = g.add_node(input(0, t(&[0, 1], 0)));
        let a = g.add_node(op("stablehlo.transpose", &[perm(&[1, 0])], vec![t(&[1, 0], 0)]));
        let b = g.add_node(op("stablehlo.transpose", &[perm(&[1, 0])], vec![t(&[0, 1], 0)]));
        let r = g.add_node(op("func.return", &[], vec![]));
        g.add_edge(x, a, e(0));
        g.add_edge(a, b, e(0));
        g.add_edge(b, r, e(0));
        let n = normalize(&g, &orders()).unwrap();
        assert_eq!(rules(&n), ["transpose_collapse", "dce", "transpose_identity"]);
        assert_eq!(n.graph.node_count(), 2);
        assert_eq!(n.graph.out[0], vec![(1, vec![e(0)])]);
        assert!(rules(&n).contains(&"transpose_collapse") && rules(&n).contains(&"transpose_identity"));
    }

    #[test]
    fn reshape_chain_collapses() {
        let mut g: Graph<ANode> = Graph::default();
        let x = g.add_node(input(0, t(&[0, 1], 0)));
        let a = g.add_node(op("stablehlo.reshape", &[], vec![t(&[2], 0)]));
        let b = g.add_node(op("stablehlo.reshape", &[], vec![t(&[3, 4], 0)]));
        let r = g.add_node(op("func.return", &[], vec![]));
        g.add_edge(x, a, e(0));
        g.add_edge(a, b, e(0));
        g.add_edge(b, r, e(0));
        let n = normalize(&g, &orders()).unwrap();
        assert_eq!(rules(&n), ["reshape_collapse", "dce"]);
        assert_eq!(n.graph.node_count(), 3);
        assert_eq!(n.graph.nodes[1].origin, b);
    }

    #[test]
    fn splat_on_a_commutative_port() {
        let mut g: Graph<ANode> = Graph::default();
        let x = g.add_node(input(0, t(&[0], 0)));
        let c = g.add_node(op(
            "stablehlo.constant",
            &[("value", AAttr::Value { splat: true })],
            vec![t(&[], 0)],
        ));
        let b = g.add_node(op(
            "stablehlo.broadcast_in_dim",
            &[("broadcast_dimensions", AAttr::Lit(Attr::Ints(vec![])))],
            vec![t(&[0], 0)],
        ));
        let add = g.add_node(op("stablehlo.add", &[], vec![t(&[0], 0)]));
        let r = g.add_node(op("func.return", &[], vec![]));
        let comm = EdgeEnd {
            res: 0,
            port: PortLabel::Commutative,
        };
        g.add_edge(x, add, comm);
        g.add_edge(c, b, e(0));
        g.add_edge(b, add, comm);
        g.add_edge(add, r, e(0));
        let n = normalize(&g, &orders()).unwrap();
        assert_eq!(rules(&n), ["splat_fold", "dce", "dce"]);
        let a = n.graph.nodes.iter().find(|x| x.origin == add).unwrap();
        assert_eq!(a.splats.len(), 1);
        assert_eq!(a.splats[0].port, PortLabel::Commutative);
        assert_eq!((a.splats[0].constant, a.splats[0].broadcast), (c, Some(b)));
        assert_eq!(n.graph.node_count(), 3);
    }

    #[test]
    fn cse_merges_equal_ops_but_not_fresh_reshapes_or_region_nodes() {
        let mut g: Graph<ANode> = Graph::default();
        let x = g.add_node(input(0, t(&[0], 0)));
        let n1 = g.add_node(op("stablehlo.negate", &[], vec![t(&[0], 0)]));
        let n2 = g.add_node(op("stablehlo.negate", &[], vec![t(&[0], 0)]));
        let s1 = g.add_node(op("stablehlo.reshape", &[], vec![t(&[1], 0)]));
        let s2 = g.add_node(op("stablehlo.reshape", &[], vec![t(&[2], 0)]));
        // A reduce whose body reads a third negate of x (captured): that one stays.
        let n3 = g.add_node(op("stablehlo.abs", &[], vec![t(&[0], 0)]));
        let n4 = g.add_node(op("stablehlo.abs", &[], vec![t(&[0], 0)]));
        let red = g.add_node(op(
            "stablehlo.reduce",
            &[("dimensions", AAttr::Lit(Attr::Ints(vec![0])))],
            vec![t(&[], 0)],
        ));
        let ra = g.add_node(node(
            Kind::RegionArg { region: 0, index: 0 },
            &[],
            vec![t(&[], 0)],
        ));
        let body = g.add_node(op("stablehlo.add", &[], vec![t(&[], 0)]));
        let term = g.add_node(op("stablehlo.return", &[], vec![]));
        let r = g.add_node(op("func.return", &[], vec![]));
        for v in [n1, n2, s1, s2, n3, n4] {
            g.add_edge(x, v, e(0));
        }
        g.add_edge(n4, red, e(0));
        g.add_edge(
            red,
            ra,
            EdgeEnd {
                res: 0,
                port: PortLabel::Region { region: 0, index: 0 },
            },
        );
        g.add_edge(
            ra,
            body,
            EdgeEnd {
                res: 0,
                port: PortLabel::Commutative,
            },
        );
        g.add_edge(
            n3,
            body,
            EdgeEnd {
                res: 0,
                port: PortLabel::Commutative,
            },
        );
        g.add_edge(body, term, e(0));
        g.add_edge(
            term,
            red,
            EdgeEnd {
                res: 0,
                port: PortLabel::Yield { region: 0 },
            },
        );
        for (i, v) in [n1, n2, s1, s2, red].into_iter().enumerate() {
            g.add_edge(v, r, e(i as u32));
        }
        let n = normalize(&g, &orders()).unwrap();
        let cse: Vec<&Vec<u32>> = n
            .steps
            .iter()
            .filter(|s| s.rule == "cse")
            .map(|s| &s.matched)
            .collect();
        assert_eq!(cse, [&vec![n1, n2]]);
        let red_n = n.graph.nodes.iter().find(|x| x.origin == red).unwrap();
        assert_eq!(
            red_n.omega,
            Some(crate::normalize::json::Omega {
                order: Order::Tree { fan_in: 4 },
                axes: vec![0]
            })
        );
    }

    #[test]
    fn dce_keeps_inputs_and_calls() {
        let mut g: Graph<ANode> = Graph::default();
        let x = g.add_node(input(0, t(&[0], 0)));
        let _unused = g.add_node(input(1, t(&[1], 1)));
        let mut call = op("stablehlo.composite", &[], vec![t(&[0], 0)]);
        call.child = Some("c0".into());
        let call = g.add_node(call);
        let dead = g.add_node(op("stablehlo.negate", &[], vec![t(&[0], 0)]));
        let dot = g.add_node(op(
            "stablehlo.dot_general",
            &[(
                "dot_dimension_numbers",
                AAttr::Lit(Attr::DotDims(DotDims {
                    lhs_batching: vec![],
                    rhs_batching: vec![],
                    lhs_contracting: vec![0],
                    rhs_contracting: vec![0],
                })),
            )],
            vec![t(&[], 0)],
        ));
        let r = g.add_node(op("func.return", &[], vec![]));
        g.add_edge(x, call, e(0));
        g.add_edge(x, dead, e(0));
        g.add_edge(x, dot, e(0));
        g.add_edge(x, dot, e(1));
        g.add_edge(dot, r, e(0));
        let n = normalize(&g, &orders()).unwrap();
        assert_eq!(
            n.steps
                .iter()
                .map(|s| (s.rule.as_str(), s.matched.clone()))
                .collect::<Vec<_>>(),
            [("dce", vec![dead])]
        );
        assert_eq!(n.graph.node_count(), 5);
        let d = n.graph.nodes.iter().find(|x| x.origin == dot).unwrap();
        assert_eq!(d.omega.as_ref().map(|o| o.axes.clone()), Some(vec![0]));
    }
}
