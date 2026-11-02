//! Identity decisions on hand-built kernels (docs/passes/kernel_classes.md §4.2) and the invariance of
//! the class under axis permutations, node renumbering and operand order.

use std::collections::BTreeMap;

use cc_ir::Attr;
use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};
use cc_ir::attrs::{ConvDims, DotDims};

use super::{classify, encode, local};
use crate::components::algo::graph::{AAttr, ADim, ANode, AType, Family, Kind};
use crate::kernel_classes::json::Group;
use crate::kernels::json::{KNode, Role};
use crate::normalize::json::{NNode, Omega, Order};

/// A kernel graph under construction. Dims: `0` is the literal 1, `k > 0` the class variable `k`.
#[derive(Default)]
struct K {
    g: Graph<KNode>,
}

fn t(dims: &[u32], family: Family) -> AType {
    AType {
        family,
        elem: 0,
        dims: dims
            .iter()
            .map(|&d| if d == 0 { ADim::One } else { ADim::Var(d) })
            .collect(),
    }
}

fn f(dims: &[u32]) -> AType {
    t(dims, Family::Float)
}

impl K {
    fn input(&mut self, ty: AType) -> u32 {
        let src = self.g.node_count() as u32;
        self.g.add_node(KNode::Input { src, res: 0, ty })
    }
    fn op(&mut self, name: &str, ty: AType, ins: &[u32], attrs: &[(&str, Attr)], role: Role) -> u32 {
        let origin = self.g.node_count() as u32;
        let n = self.g.add_node(KNode::Member {
            origin,
            role,
            node: NNode {
                label: ANode {
                    kind: Kind::Op {
                        name: name.into(),
                        target: None,
                    },
                    attrs: attrs
                        .iter()
                        .map(|(k, v)| (k.to_string(), AAttr::Lit(v.clone())))
                        .collect(),
                    results: vec![ty],
                    child: None,
                },
                omega: None,
                splats: Vec::new(),
                origin,
            },
        });
        let comm = matches!(name, "stablehlo.add" | "stablehlo.multiply" | "stablehlo.maximum");
        for (p, &i) in ins.iter().enumerate() {
            let port = if comm {
                PortLabel::Commutative
            } else {
                PortLabel::Port(p as u32)
            };
            self.g.add_edge(i, n, EdgeEnd { res: 0, port });
        }
        n
    }
    fn omega(&mut self, n: u32, axes: Vec<u32>) {
        if let KNode::Member { node, .. } = &mut self.g.nodes[n as usize] {
            node.omega = Some(Omega {
                order: Order::Seq,
                axes,
            });
        }
    }
    fn sized(&mut self, n: u32, key: &str, entries: Vec<Option<i64>>) {
        if let KNode::Member { node, .. } = &mut self.g.nodes[n as usize] {
            node.label.attrs.insert(
                key.into(),
                AAttr::Sized {
                    form: crate::components::algo::graph::SizedForm::Ints,
                    entries,
                },
            );
        }
    }
    fn output(&mut self, v: u32) {
        let ty = local::results(&self.g.nodes[v as usize])[0].clone();
        let o = self.g.add_node(KNode::Output { node: v, res: 0, ty });
        self.g.add_edge(
            v,
            o,
            EdgeEnd {
                res: 0,
                port: PortLabel::Port(0),
            },
        );
    }
}

type G = BTreeMap<u32, BTreeMap<String, Group>>;

fn enc(g: &Graph<KNode>, groups: &G) -> encode::Encoded {
    let l = local::classes(g).unwrap();
    encode::encode(g, &l, groups).unwrap()
}

/// The number of classes the kernels fall into.
fn classes_of(gs: &[(Graph<KNode>, G)]) -> usize {
    let es: Vec<encode::Encoded> = gs.iter().map(|(g, gr)| enc(g, gr)).collect();
    let refs: Vec<&encode::Encoded> = es.iter().collect();
    classify::classify(&refs, 64, 1 << 24).unwrap().0.len()
}

fn same(a: Graph<KNode>, b: Graph<KNode>) -> bool {
    classes_of(&[(a, G::new()), (b, G::new())]) == 1
}

fn dot(lhs: &[u32], rhs: &[u32], out: &[u32], dd: DotDims, family: Family) -> Graph<KNode> {
    let mut k = K::default();
    let x = k.input(t(lhs, family.clone()));
    let w = k.input(t(rhs, family.clone()));
    let c = d_contract(&dd);
    let d = k.op(
        "stablehlo.dot_general",
        t(out, family),
        &[x, w],
        &[("dot_dimension_numbers", Attr::DotDims(dd))],
        Role::Core,
    );
    k.omega(d, c);
    k.output(d);
    k.g
}

fn d_contract(d: &DotDims) -> Vec<u32> {
    d.lhs_contracting.iter().map(|&a| a as u32).collect()
}

fn dd(lc: &[i64], rc: &[i64]) -> DotDims {
    DotDims {
        lhs_batching: vec![],
        rhs_batching: vec![],
        lhs_contracting: lc.to_vec(),
        rhs_contracting: rc.to_vec(),
    }
}

#[test]
fn x_w_and_x_w_transposed_are_one_class() {
    // [M, K] x [K, N] and [M, K] x [N, K].
    let a = dot(&[1, 2], &[2, 3], &[1, 3], dd(&[1], &[0]), Family::Float);
    let b = dot(&[1, 2], &[3, 2], &[1, 3], dd(&[1], &[1]), Family::Float);
    assert!(same(a, b));
}

#[test]
fn integer_and_float_gemms_differ() {
    let a = dot(&[1, 2], &[2, 3], &[1, 3], dd(&[1], &[0]), Family::Float);
    let b = dot(&[1, 2], &[2, 3], &[1, 3], dd(&[1], &[0]), Family::Sint);
    assert!(!same(a, b));
}

#[test]
fn decode_and_prefill_share_a_class() {
    // [B, 1, D] x [D, F] (decode) and [B, S, D] x [D, F] (prefill): a unit axis is an extent.
    let a = dot(&[1, 0, 2], &[2, 3], &[1, 0, 3], dd(&[2], &[0]), Family::Float);
    let b = dot(&[1, 4, 2], &[2, 3], &[1, 4, 3], dd(&[2], &[0]), Family::Float);
    assert!(same(a, b));
}

#[test]
fn contracting_over_a_different_axis_differs() {
    // [M, K] x [K, N] vs [M, K] x [K, N] contracting nothing of the lhs's second axis (outer product
    // over a batch is another computation): contract M with K's position instead.
    let a = dot(&[1, 2], &[2, 3], &[1, 3], dd(&[1], &[0]), Family::Float);
    let b = dot(&[1, 2], &[1, 3], &[2, 3], dd(&[0], &[0]), Family::Float);
    assert!(same(a.clone(), b.clone()), "a transposed lhs is a layout");
    let c = dot(&[1, 2, 4], &[2, 3], &[1, 4, 3], dd(&[1], &[0]), Family::Float);
    assert!(!same(a, c), "an extra free axis is structure");
}

fn conv(nhwc: bool, k: [u32; 2], groups: Option<Group>) -> (Graph<KNode>, G) {
    // Input N=1, H=2, W=3, C=4; kernel spatial k, Cin=4 (or 0 for depthwise), Cout=7; output N, H'=8, W'=9.
    let mut b = K::default();
    let (input, kernel, out, dn) = if nhwc {
        (
            vec![1, 2, 3, 4],
            vec![k[0], k[1], 4, 7],
            vec![1, 8, 9, 7],
            ConvDims {
                input_batch: 0,
                input_feature: 3,
                input_spatial: vec![1, 2],
                kernel_input_feature: 2,
                kernel_output_feature: 3,
                kernel_spatial: vec![0, 1],
                output_batch: 0,
                output_feature: 3,
                output_spatial: vec![1, 2],
            },
        )
    } else {
        (
            vec![1, 4, 2, 3],
            vec![7, 4, k[0], k[1]],
            vec![1, 7, 8, 9],
            ConvDims {
                input_batch: 0,
                input_feature: 1,
                input_spatial: vec![2, 3],
                kernel_input_feature: 1,
                kernel_output_feature: 0,
                kernel_spatial: vec![2, 3],
                output_batch: 0,
                output_feature: 1,
                output_spatial: vec![2, 3],
            },
        )
    };
    let x = b.input(f(&input));
    let w = b.input(f(&kernel));
    let c = b.op(
        "stablehlo.convolution",
        f(&out),
        &[x, w],
        &[("dimension_numbers", Attr::ConvDims(dn.clone()))],
        Role::Core,
    );
    b.sized(c, "window_strides", vec![None, None]);
    let fgc = match groups {
        Some(Group::One) | None => Some(1),
        _ => None,
    };
    b.sized(c, "feature_group_count", vec![fgc]);
    let mut om: Vec<u32> = dn.kernel_spatial.iter().map(|&a| a as u32).collect();
    om.push(dn.kernel_input_feature as u32);
    b.omega(c, om);
    b.output(c);
    let gr: G = groups
        .map(|g| BTreeMap::from([(c, BTreeMap::from([("feature_group_count".to_string(), g)]))]))
        .unwrap_or_default();
    (b.g, gr)
}

#[test]
fn conv_layouts_are_one_class() {
    let a = conv(true, [5, 6], Some(Group::One));
    let b = conv(false, [5, 6], Some(Group::One));
    assert_eq!(classes_of(&[a, b]), 1);
}

#[test]
fn one_by_one_and_three_by_three_convs_are_one_class() {
    let a = conv(true, [0, 0], Some(Group::One));
    let b = conv(true, [5, 6], Some(Group::One));
    assert_eq!(classes_of(&[a, b]), 1);
}

#[test]
fn depthwise_and_dense_convs_differ() {
    let a = conv(true, [5, 6], Some(Group::One));
    let b = conv(true, [5, 6], Some(Group::Depthwise));
    assert_eq!(classes_of(&[a, b]), 2);
}

fn cmp(dir: &str, swap: bool) -> Graph<KNode> {
    let mut k = K::default();
    let a = k.input(f(&[1, 2]));
    let b = k.input(t(&[1, 2], Family::Float));
    let n = k.op("stablehlo.negate", f(&[1, 2]), &[b], &[], Role::Map);
    let ins = if swap { [n, a] } else { [a, n] };
    let c = k.op(
        "stablehlo.compare",
        t(&[1, 2], Family::Bool),
        &ins,
        &[(
            "comparison_direction",
            Attr::Enum {
                kind: "comparison_direction".into(),
                value: dir.into(),
            },
        )],
        Role::Map,
    );
    k.output(c);
    k.g
}

#[test]
fn greater_than_is_less_than_swapped() {
    assert!(same(cmp("GT", false), cmp("LT", true)));
    assert!(same(cmp("GE", true), cmp("LE", false)));
    assert!(!same(cmp("GT", false), cmp("LT", false)));
}

fn bcast_add(expanding: bool) -> Graph<KNode> {
    // broadcast_in_dim(x [1, C] or [N, C]) -> [N, C], plus y [N, C].
    let mut k = K::default();
    let x = k.input(f(&[if expanding { 0 } else { 1 }, 2]));
    let y = k.input(f(&[1, 2]));
    let b = k.op(
        "stablehlo.broadcast_in_dim",
        f(&[1, 2]),
        &[x],
        &[("broadcast_dimensions", Attr::Ints(vec![0, 1]))],
        Role::Broadcast,
    );
    let s = k.op("stablehlo.add", f(&[1, 2]), &[b, y], &[], Role::Map);
    k.output(s);
    k.g
}

#[test]
fn an_expanding_broadcast_is_structure() {
    assert!(!same(bcast_add(true), bcast_add(false)));
}

/// A dot with a bias and a ReLU epilogue, its lhs axes permuted by `p` and its nodes added in `order`.
fn dot_epilogue(p: &[usize], order: &[usize], swap_add: bool) -> Graph<KNode> {
    // lhs logical axes (B, M, K); rhs (K, N); result (B, M, N).
    let logical = [1u32, 2, 3];
    let lhs: Vec<u32> = p.iter().map(|&i| logical[i]).collect();
    let pos = |axis: usize| p.iter().position(|&i| i == axis).unwrap() as i64;
    let dn = DotDims {
        lhs_batching: vec![],
        rhs_batching: vec![],
        lhs_contracting: vec![pos(2)],
        rhs_contracting: vec![0],
    };
    // The result's free axes follow the lhs order (B and M in their lhs order), then N.
    let mut res: Vec<u32> = lhs.iter().copied().filter(|&c| c != 3).collect();
    res.push(4);
    let mut k = K::default();
    let mut ids: BTreeMap<usize, u32> = BTreeMap::new();
    for &step in order {
        match step {
            0 => {
                ids.insert(0, k.input(f(&lhs)));
            }
            1 => {
                ids.insert(1, k.input(f(&[3, 4])));
            }
            2 => {
                ids.insert(2, k.input(f(&res)));
            }
            _ => {}
        }
    }
    let d = k.op(
        "stablehlo.dot_general",
        f(&res),
        &[ids[&0], ids[&1]],
        &[("dot_dimension_numbers", Attr::DotDims(dn))],
        Role::Core,
    );
    k.omega(d, vec![pos(2) as u32]);
    let ins = if swap_add { [ids[&2], d] } else { [d, ids[&2]] };
    let a = k.op("stablehlo.add", f(&res), &ins, &[], Role::Epilogue);
    let r = k.op("stablehlo.maximum", f(&res), &[a, ids[&2]], &[], Role::Epilogue);
    k.output(r);
    k.g
}

const PERMS: [[usize; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
const ORDERS: [[usize; 3]; 6] = PERMS;

/// Every axis permutation of the lhs, every node order and both operand orders of the residual add
/// (432 kernels) fall into one class.
#[test]
fn permutations_renumbering_and_operand_order_keep_the_class() {
    let mut gs = Vec::new();
    for p in &PERMS {
        for o in &ORDERS {
            for s in [false, true] {
                gs.push((dot_epilogue(p, o, s), G::new()));
            }
        }
    }
    assert_eq!(classes_of(&gs), 1);
}
