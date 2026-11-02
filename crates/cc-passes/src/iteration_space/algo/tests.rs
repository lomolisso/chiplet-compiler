//! The pass's decisions on hand-built kernels (docs/passes/iteration_space.md §4): classes from accesses,
//! the uniform recurrence of a contraction, the three axis projections of a GEMM, the embedding
//! relation, the routing thresholds, and the agreement of the iteration hooks with the transcribed spec.

use std::collections::BTreeMap;

use cc_ir::Attr;
use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};
use cc_ir::attrs::{ConvDims, DotDims, ScatterDims};

use super::eval::Table;
use super::{deps, embed, mapping, route, skeleton};
use crate::components::algo::graph::{AAttr, ADim, ANode, AType, Family, Kind, SizedForm};
use crate::iteration_space::TemplateConfig;
use crate::iteration_space::json::{Extent, Group, Route, SkelClass, Skeleton};
use crate::iteration_space::spec;
use crate::kernel_classes::json::{ClassFile, ClassKind, Group as GroupKind, MemberKey, Params, Row, Tags};
use crate::kernels::json::{KNode, Role};
use crate::normalize::json::{NNode, Omega, Order};

/// A kernel graph under construction. Dims: `0` is the literal 1, `k > 0` the class variable `k`.
#[derive(Default)]
struct K {
    g: Graph<KNode>,
    groups: BTreeMap<u32, BTreeMap<String, GroupKind>>,
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
    fn omega(&mut self, n: u32, axes: Vec<u32>, order: Order) {
        if let KNode::Member { node, .. } = &mut self.g.nodes[n as usize] {
            node.omega = Some(Omega { order, axes });
        }
    }
    fn sized(&mut self, n: u32, key: &str, entries: Vec<Option<i64>>) {
        if let KNode::Member { node, .. } = &mut self.g.nodes[n as usize] {
            node.label.attrs.insert(
                key.into(),
                AAttr::Sized {
                    form: SizedForm::Ints,
                    entries,
                },
            );
        }
    }
    fn output(&mut self, v: u32) {
        let ty = crate::kernel_classes::algo::local::results(&self.g.nodes[v as usize])[0].clone();
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
    fn class(self, id: &str, kind: ClassKind) -> (ClassFile, Graph<KNode>) {
        let local = crate::kernel_classes::algo::local::classes(&self.g).unwrap();
        let c = ClassFile {
            schema: String::new(),
            id: id.into(),
            hash: String::new(),
            kind,
            core: String::new(),
            tags: Tags::default(),
            representative: MemberKey {
                component: "c".into(),
                kernel: "k0".into(),
                groups: self.groups,
            },
            roles: Vec::new(),
            params: Params {
                axes: local.unit.clone(),
                elems: vec![Family::Float; local.ne as usize],
                values: Vec::new(),
            },
            local,
            iteration: Vec::new(),
            kernel_nodes: self.g.node_count() as u32,
            identity: Graph::default(),
            members: Vec::new(),
            models: Vec::new(),
            instances: 0,
            workload: 0,
        };
        (c, self.g)
    }
}

fn dd(lb: &[i64], rb: &[i64], lc: &[i64], rc: &[i64]) -> DotDims {
    DotDims {
        lhs_batching: lb.to_vec(),
        rhs_batching: rb.to_vec(),
        lhs_contracting: lc.to_vec(),
        rhs_contracting: rc.to_vec(),
    }
}

fn dot(
    id: &str,
    lhs: &[u32],
    rhs: &[u32],
    out: &[u32],
    d: DotDims,
    order: Order,
) -> (ClassFile, Graph<KNode>) {
    let mut k = K::default();
    let x = k.input(f(lhs));
    let w = k.input(f(rhs));
    let om: Vec<u32> = d.lhs_contracting.iter().map(|&a| a as u32).collect();
    let n = k.op(
        "stablehlo.dot_general",
        f(out),
        &[x, w],
        &[("dot_dimension_numbers", Attr::DotDims(d))],
        Role::Core,
    );
    k.omega(n, om, order);
    k.output(n);
    k.class(id, ClassKind::Contraction)
}

fn gemm(id: &str) -> (ClassFile, Graph<KNode>) {
    // [M, K] x [K, N].
    dot(
        id,
        &[1, 2],
        &[2, 3],
        &[1, 3],
        dd(&[], &[], &[1], &[0]),
        Order::Seq,
    )
}

/// An NHWC convolution: input [1, H, W, C], kernel [R, T, C or 1, F], output [1, H', W', F].
fn conv(id: &str, depthwise: bool) -> (ClassFile, Graph<KNode>) {
    let mut b = K::default();
    let kin = if depthwise { 0 } else { 4 };
    let x = b.input(f(&[0, 2, 3, 4]));
    let w = b.input(f(&[5, 6, kin, 7]));
    let dn = ConvDims {
        input_batch: 0,
        input_feature: 3,
        input_spatial: vec![1, 2],
        kernel_input_feature: 2,
        kernel_output_feature: 3,
        kernel_spatial: vec![0, 1],
        output_batch: 0,
        output_feature: 3,
        output_spatial: vec![1, 2],
    };
    let c = b.op(
        "stablehlo.convolution",
        f(&[0, 8, 9, 7]),
        &[x, w],
        &[("dimension_numbers", Attr::ConvDims(dn))],
        Role::Core,
    );
    b.sized(c, "window_strides", vec![None, None]);
    b.sized(c, "padding", vec![None, None, None, None]);
    b.sized(
        c,
        "feature_group_count",
        vec![if depthwise { None } else { Some(1) }],
    );
    b.omega(c, vec![0, 1, 2], Order::Seq);
    b.output(c);
    if depthwise {
        b.groups.insert(
            c,
            BTreeMap::from([("feature_group_count".to_string(), GroupKind::Depthwise)]),
        );
    }
    b.class(id, ClassKind::Contraction)
}

fn skel(c: &(ClassFile, Graph<KNode>)) -> Skeleton {
    skeleton::build(&c.0, &c.1).unwrap()
}

/// One row: every axis class at `e`, except `set` (class → extent); sizes symbolic for `symbolic`.
fn row(c: &ClassFile, e: u64, set: &[(u32, u64)], symbolic: &[u32]) -> Row {
    let nd = c.local.nd as usize;
    let mut extents = vec![e; nd];
    for &(k, v) in set {
        extents[k as usize] = v;
    }
    Row {
        model: "m".into(),
        phase: "p".into(),
        instance: "i".into(),
        component: "c".into(),
        kernel: "k0".into(),
        member: 0,
        sizes: (0..nd as u32)
            .map(|k| {
                if symbolic.contains(&k) {
                    "S".to_string()
                } else {
                    extents[k as usize].to_string()
                }
            })
            .collect(),
        extents,
        elems: vec!["f32".into(); c.local.ne as usize],
        values: Vec::new(),
        workload: 1,
        trip_count: 1,
        weighted: 1,
    }
}

fn class_of(s: &Skeleton, role: &str) -> u32 {
    s.dims
        .iter()
        .find_map(|d| match (&d.extent, d.role == role) {
            (Extent::Class { class }, true) => Some(*class),
            _ => None,
        })
        .unwrap()
}

#[test]
fn gemm_is_a_uniform_contraction_with_the_three_axis_projections() {
    let c = gemm("g");
    let s = skel(&c);
    assert_eq!(s.class, SkelClass::Contraction);
    let d = deps::dependences(&s);
    assert!(d.uniform, "{}", d.reason);
    assert_eq!(d.space, vec![Group::M, Group::N, Group::K]);
    let m = class_of(&s, "lhs_free");
    let rows = vec![row(&c.0, 64, &[], &[m])];
    let tb = Table::new(&rows, &[]);
    let cands = mapping::candidates(&s, &d, &tb);
    let find = |u: [i64; 3]| {
        cands
            .iter()
            .find(|x| x.lambda == vec![1, 1, 1] && x.u == u.to_vec())
            .unwrap()
    };
    // Projection along k keeps the result in place, along m the weights (B[k, n] has no m), along n
    // the lhs (an input: its m axis is symbolic).
    assert_eq!(find([0, 0, 1]).dataflow, "output_stationary");
    assert_eq!(find([1, 0, 0]).dataflow, "weight_stationary");
    assert_eq!(find([0, 1, 0]).dataflow, "input_stationary");
    assert!(
        cands.iter().all(|x| x.lambda.iter().all(|&l| l == 1)),
        "unit dependences need λ = (1, 1, 1)"
    );
    let r = route::route(&s, &d, &cands, &tb, &TemplateConfig::default(), 1);
    assert_eq!(r.route, Route::Template, "{}", r.reason);
}

#[test]
fn decode_fails_the_reuse_threshold() {
    let c = gemm("g");
    let s = skel(&c);
    let d = deps::dependences(&s);
    let m = class_of(&s, "lhs_free");
    let rows = vec![row(&c.0, 4096, &[(m, 1)], &[m])];
    let tb = Table::new(&rows, &[]);
    let cands = mapping::candidates(&s, &d, &tb);
    let r = route::route(&s, &d, &cands, &tb, &TemplateConfig::default(), 1);
    assert_eq!(r.route, Route::Datapath);
    assert!(r.reason.contains("reused"), "{}", r.reason);
}

#[test]
fn x_w_and_x_w_transposed_embed_both_ways() {
    let a = skel(&gemm("a"));
    let b = skel(&dot(
        "b",
        &[1, 2],
        &[3, 2],
        &[1, 3],
        dd(&[], &[], &[1], &[1]),
        Order::Seq,
    ));
    assert!(embed::embed(&a, &b, 1 << 20).0.is_some());
    assert!(embed::embed(&b, &a, 1 << 20).0.is_some());
}

#[test]
fn matmul_embeds_in_conv_but_not_back() {
    let g = skel(&gemm("g"));
    let c = skel(&conv("c", false));
    let (p, _) = embed::embed(&g, &c, 1 << 20);
    let p = p.expect("matmul ⊑ conv");
    // n goes to the output feature, k to a reduction dim (the input feature, or a window: a GEMM is also
    // a convolution with window K and output width 1), m to an output batch or spatial dim.
    assert_eq!(c.dims[p.dims[1] as usize].role, "out_feature");
    assert!(["in_feature", "window"].contains(&c.dims[p.dims[2] as usize].role.as_str()));
    assert!(["out_batch", "out_spatial"].contains(&c.dims[p.dims[0] as usize].role.as_str()));
    assert!(embed::embed(&c, &g, 1 << 20).0.is_none());
}

#[test]
fn depthwise_does_not_embed_in_dense_conv() {
    let dw = skel(&conv("dw", true));
    let dense = skel(&conv("dense", false));
    assert!(dw.dims.iter().any(|d| d.role == "group"));
    assert!(embed::embed(&dw, &dense, 1 << 20).0.is_none());
    // The group dim indexes input, kernel and result: a batch dim of the recurrence.
    let d = deps::dependences(&dw);
    assert_eq!(d.groups.get(&Group::Batch).map(Vec::len), Some(1));
}

#[test]
fn block_scaled_dot_is_a_batched_gemm() {
    // [kb, M, bk] x [kb, bk, N] -> [kb, M, N]: the block index is a batch dim, bk the reduction.
    let c = dot(
        "b",
        &[1, 2, 3],
        &[1, 3, 4],
        &[1, 2, 4],
        dd(&[0], &[0], &[2], &[1]),
        Order::Seq,
    );
    let s = skel(&c);
    let d = deps::dependences(&s);
    assert!(d.uniform);
    assert_eq!(d.groups[&Group::Batch], vec![0]);
    assert_eq!(d.groups[&Group::K], vec![3]);
}

#[test]
fn tree_order_is_not_one_accumulator_chain() {
    let c = dot(
        "t",
        &[1, 2],
        &[2, 3],
        &[1, 3],
        dd(&[], &[], &[1], &[0]),
        Order::Tree { fan_in: 4 },
    );
    let d = deps::dependences(&skel(&c));
    assert!(!d.uniform);
    assert!(d.reason.contains("tree"), "{}", d.reason);
}

#[test]
fn shared_tables_give_two_dims_of_one_extent() {
    // q = x * t and k = y * t with one table t, then q · kᵀ over D: both sequence axes are one class.
    let mut b = K::default();
    let x = b.input(f(&[1, 2]));
    let y = b.input(f(&[3, 2]));
    let tb = b.input(f(&[1, 2]));
    let q = b.op("stablehlo.multiply", f(&[1, 2]), &[x, tb], &[], Role::Prologue);
    let k = b.op("stablehlo.multiply", f(&[3, 2]), &[y, tb], &[], Role::Prologue);
    let d = b.op(
        "stablehlo.dot_general",
        f(&[1, 3]),
        &[q, k],
        &[("dot_dimension_numbers", Attr::DotDims(dd(&[], &[], &[1], &[1])))],
        Role::Core,
    );
    b.omega(d, vec![1], Order::Seq);
    b.output(d);
    let s = skel(&b.class("p", ClassKind::Contraction));
    assert_eq!(s.dims[0].extent, s.dims[1].extent);
    let generic = skel(&gemm("g"));
    assert!(
        embed::embed(&s, &generic, 1 << 20).0.is_some(),
        "a specialized GEMM runs on a general one"
    );
    assert!(
        embed::embed(&generic, &s, 1 << 20).0.is_none(),
        "not the other way round"
    );
}

#[test]
fn softmax_kernels_are_not_templates() {
    // max over the last axis (a reduction), then exp(x - m) (a map).
    let mut b = K::default();
    let x = b.input(f(&[1, 2]));
    let init = b.input(f(&[]));
    let r = b.op(
        "stablehlo.reduce",
        f(&[1]),
        &[x, init],
        &[("dimensions", Attr::Ints(vec![1]))],
        Role::Core,
    );
    b.omega(r, vec![1], Order::Seq);
    b.output(r);
    let s = skel(&b.class("r", ClassKind::Reduction));
    assert_eq!(s.class, SkelClass::Reduce);
    let d = deps::dependences(&s);
    assert!(d.uniform, "a reduction chain is uniform");
    let tb = Table::new(&[], &[]);
    let r = route::route(&s, &d, &[], &tb, &TemplateConfig::default(), 0);
    assert_eq!(r.route, Route::Datapath);

    let mut b = K::default();
    let x = b.input(f(&[1, 2]));
    let m = b.input(f(&[1]));
    let mb = b.op(
        "stablehlo.broadcast_in_dim",
        f(&[1, 2]),
        &[m],
        &[("broadcast_dimensions", Attr::Ints(vec![0]))],
        Role::Broadcast,
    );
    let sub = b.op("stablehlo.subtract", f(&[1, 2]), &[x, mb], &[], Role::Map);
    let e = b.op("stablehlo.exponential", f(&[1, 2]), &[sub], &[], Role::Map);
    b.output(e);
    let s = skel(&b.class("m", ClassKind::Map));
    assert_eq!(s.class, SkelClass::Map);
    // The broadcast input is read through a projection: it does not depend on dim 1.
    let inm = s.operands.iter().find(|o| o.name == format!("in{m}")).unwrap();
    assert_eq!(inm.reuse, vec![1]);
}

#[test]
fn data_dependent_scatter_is_never_a_template() {
    // scatter-add of updates [U, D] into operand [N, D] at indices [U, 1].
    let mut b = K::default();
    let op = b.input(f(&[1, 2]));
    let idx = b.input(t(&[3, 0], Family::Sint));
    let upd = b.input(f(&[3, 2]));
    let s = b.op(
        "stablehlo.scatter",
        f(&[1, 2]),
        &[op, idx, upd],
        &[(
            "scatter_dimension_numbers",
            Attr::ScatterDims(ScatterDims {
                update_window_dims: vec![1],
                inserted_window_dims: vec![0],
                input_batching_dims: vec![],
                scatter_indices_batching_dims: vec![],
                scattered_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            }),
        )],
        Role::Core,
    );
    b.omega(s, vec![0, 1], Order::Seq);
    b.output(s);
    let sk = skel(&b.class("s", ClassKind::ScatterReduce));
    assert!(!sk.affine);
    assert_eq!(sk.class, SkelClass::Reduce);
    let d = deps::dependences(&sk);
    assert!(!d.uniform);
    let r = route::route(&sk, &d, &[], &Table::new(&[], &[]), &TemplateConfig::default(), 1);
    assert_eq!(r.route, Route::Datapath);
}

#[test]
fn hooks_agree_with_the_transcribed_spec() {
    let check = |c: &(ClassFile, Graph<KNode>)| {
        let s = skel(c);
        let claims = spec::claims(&c.0, &s, &c.1).unwrap();
        assert!(!claims.is_empty());
        for cl in &claims {
            assert_eq!(embed::poly(&cl.stored), embed::poly(&cl.spec), "{}", cl.what);
            assert_eq!(cl.stored.divisor, cl.spec.divisor, "{}", cl.what);
        }
        s
    };
    check(&gemm("g"));
    check(&conv("dense", false));
    check(&conv("dw", true));
    // A mutated stride coefficient is caught.
    let c = conv("dense", false);
    let mut s = check(&c);
    let i = s
        .accesses
        .iter()
        .position(|a| Some(a.node) == s.core_node)
        .unwrap();
    let row = s.accesses[i].operands[0].as_mut().unwrap().get_mut(1).unwrap();
    row.terms[0].k = 2;
    let claims = spec::claims(&c.0, &s, &c.1);
    let bad = claims
        .map(|cs| {
            cs.iter()
                .any(|cl| embed::poly(&cl.stored) != embed::poly(&cl.spec))
        })
        .unwrap_or(true);
    assert!(bad);
}
