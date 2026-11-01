//! `kernels`: the toy export (replayed) through import, components, normalize and kernels with every
//! check green; every check of its `verify` fails on a mutated partition (hand-built graphs, and the
//! written artifacts); on random graphs the partition always passes the independent checker.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use cc_core::runner::{Caps, Invocation, Runners, ToolLog, ToolOutcome, ToolRunner};
use cc_core::{CancelToken, CheckStatus, LoadedConfig};
use cc_ir::absgraph::Graph;
use cc_passes::components::algo::graph::Family;
use cc_passes::kernels::algo::build::{GraphBuilder, ty};
use cc_passes::kernels::algo::{extract::extract, partition};
use cc_passes::kernels::json::{
    Boundary, BoundaryKind, GRAPHS_SCHEMA, KNode, KernelGraphs, KernelPartition, KernelsSummary, Member,
    PARTITION_SCHEMA, Part, Role,
};
use cc_passes::kernels::verify::check_component;
use cc_passes::normalize::json::NNode;
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use proptest::prelude::*;
use tests::{Env, ExportReplay, repo_root};

const CONFIG: &str = r#"
[pipeline]
targets = ["kernels"]
[resources]
memory_budget_gb = 32
threads_total = 8
[models]
keys = ["toy"]
config = "reduced"
[passes.export]
runner = "host"
host_python = ["python3"]
[passes.export.resources]
threads = 2
memory_gb = 8
timeout_min = 1
[passes.import.resources]
threads = 2
memory_gb = 2
timeout_min = 1
[passes.components.resources]
threads = 2
memory_gb = 2
timeout_min = 1
[passes.normalize.order]
float = "seq"
complex = "seq"
[passes.normalize.resources]
threads = 2
memory_gb = 2
timeout_min = 5
[passes.kernels.resources]
threads = 2
memory_gb = 2
timeout_min = 1
"#;

const F: Family = Family::Float;

/// Export calls are replayed from the fixture; solver calls (`smt-*`, for normalize) run for real.
#[derive(Default)]
struct Routed {
    replay: ExportReplay,
    real: Option<Arc<dyn ToolRunner>>,
}

impl std::fmt::Debug for Routed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Routed")
    }
}

impl ToolRunner for Routed {
    fn name(&self) -> &'static str {
        "routed"
    }
    fn run(
        &self,
        inv: &Invocation,
        caps: &Caps,
        log: &ToolLog,
        cancel: &CancelToken,
    ) -> cc_core::Result<ToolOutcome> {
        match (&self.real, inv.tool.starts_with("smt-")) {
            (Some(r), true) => r.run(inv, caps, log, cancel),
            _ => self.replay.run(inv, caps, log, cancel),
        }
    }
}

fn sets() -> Vec<String> {
    let cfg: toml::Value =
        toml::from_str(&std::fs::read_to_string(repo_root().join("config.toml")).unwrap()).unwrap();
    let mut v = vec![format!(
        "tools.docker.tag={}",
        cfg["tools"]["docker"]["tag"].as_str().unwrap()
    )];
    if std::env::var("CC_SMT_RUNNER").as_deref() == Ok("host") {
        v.push("tools.smt.runner=host".into());
    }
    v
}

struct Setup {
    env: Env,
}

impl Setup {
    fn new() -> Self {
        let env = Env::new();
        std::fs::create_dir_all(env.root().join("models/toy")).unwrap();
        std::fs::write(env.root().join("models/toy/recipe.py"), "RECIPE = {}\n").unwrap();
        Self { env }
    }

    fn config(&self) -> LoadedConfig {
        let sets = sets();
        let sets: Vec<&str> = sets.iter().map(String::as_str).collect();
        self.env.config(CONFIG, &sets)
    }

    fn pipeline(&self) -> Pipeline {
        let cfg = self.config();
        let mut runners = Runners::from_config(&cfg);
        runners.host = Arc::new(Routed {
            replay: ExportReplay::default(),
            real: Some(runners.host.clone()),
        });
        Pipeline::new(cc_passes::registry(), cfg)
            .unwrap()
            .with_runners(runners)
    }

    fn out(&self, rel: &str) -> PathBuf {
        self.env.build_dir().join(".kernels/out/kernels").join(rel)
    }

    fn failing(&self) -> Vec<String> {
        let r = self.pipeline().verify_pass("kernels").unwrap();
        r.checks
            .into_iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name)
            .collect()
    }

    fn ids(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(self.out("components"))
            .unwrap()
            .filter_map(|e| {
                e.unwrap()
                    .file_name()
                    .to_str()
                    .and_then(|n| n.strip_suffix(".json"))
                    .map(str::to_string)
            })
            .collect();
        v.sort();
        v
    }

    fn partition(&self, id: &str) -> KernelPartition {
        serde_json::from_str(&std::fs::read_to_string(self.out(&format!("components/{id}.json"))).unwrap())
            .unwrap()
    }

    fn write_partition(&self, p: &KernelPartition) {
        std::fs::write(
            self.out(&format!("components/{}.json", p.id)),
            serde_json::to_string(p).unwrap(),
        )
        .unwrap();
    }

    /// The component with the most kernels.
    fn busiest(&self) -> String {
        self.ids()
            .into_iter()
            .max_by_key(|id| (self.partition(id).kernels.len(), id.clone()))
            .unwrap()
    }
}

fn fresh() -> Setup {
    let s = Setup::new();
    let st = s.pipeline().run(&RunOptions::default()).unwrap();
    assert_eq!(st.status_of("kernels"), Some(RunStatus::Ok), "{st:?}");
    s
}

#[test]
fn partitions_and_verifies_the_toy() {
    let s = fresh();
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let t: KernelsSummary =
        serde_json::from_str(&std::fs::read_to_string(s.out("summary.json")).unwrap()).unwrap();
    assert_eq!(t.components as usize, s.ids().len());
    assert!(t.core_kernels > 0 && t.map_kernels > 0, "{t:?}");
    assert!(
        t.cores.keys().any(|k| k.starts_with("dot_general/")),
        "{:?}",
        t.cores
    );
    assert!(t.cores.keys().any(|k| k.starts_with("reduce/")), "{:?}", t.cores);
    assert!(t.call_fed_maps > 0, "the toy's relu reads a Dense call");
    let views = s.env.build_dir().join(".kernels/views");
    assert!(views.join("kernels.md").is_file());
    for id in s.ids() {
        assert!(views.join(format!("{id}.dot")).is_file());
        assert!(s.out(&format!("kernels/{id}.postcard.summary.json")).is_file());
    }
}

#[test]
fn a_node_placed_twice_fails_partition() {
    let s = fresh();
    let id = s.busiest();
    let mut p = s.partition(&id);
    let node = p.kernels[0].members[0].node;
    p.boundaries.push(Boundary {
        node,
        kind: BoundaryKind::Layout,
        owner: None,
    });
    s.write_partition(&p);
    let f = s.failing();
    assert!(f.contains(&format!("{id}:partition")), "{f:?}");
}

#[test]
fn an_edited_crossing_fails_reconstruct() {
    let s = fresh();
    let id = s.busiest();
    let mut p = s.partition(&id);
    let k = p.kernels.iter().position(|k| !k.inputs.is_empty()).unwrap();
    p.kernels[k].inputs.pop();
    s.write_partition(&p);
    let f = s.failing();
    assert!(f.contains(&format!("{id}:reconstruct")), "{f:?}");
}

#[test]
fn a_missing_partition_fails_complete() {
    let s = fresh();
    let id = s.ids()[0].clone();
    std::fs::remove_file(s.out(&format!("components/{id}.json"))).unwrap();
    let f = s.failing();
    assert!(f.contains(&"complete".to_string()), "{f:?}");
}

// Mutations of hand-built partitions.

fn artifacts(g: &Graph<NNode>) -> (KernelPartition, KernelGraphs) {
    let p = partition(g).unwrap();
    let kp = KernelPartition {
        schema: PARTITION_SCHEMA.into(),
        id: "t".into(),
        kernels: p.kernels.clone(),
        boundaries: p.boundaries.clone(),
        order: p.order.clone(),
        evictions: p.evictions,
    };
    let kg = graphs(g, &kp);
    (kp, kg)
}

/// Kernel graphs (and crossings) consistent with a possibly mutated partition, so that only the
/// mutated property fails.
fn graphs(g: &Graph<NNode>, p: &KernelPartition) -> KernelGraphs {
    KernelGraphs {
        schema: GRAPHS_SCHEMA.into(),
        id: p.id.clone(),
        kernels: p.kernels.iter().map(|k| extract(g, k).unwrap()).collect(),
    }
}

/// Move `nodes` into kernel `k` with `role` (from wherever they were), recompute the crossings and the
/// graphs, and return the failing checks.
fn mutate(g: &Graph<NNode>, p: &mut KernelPartition, k: usize, nodes: &[(u32, Role)]) -> Vec<&'static str> {
    for &(n, _) in nodes {
        for kr in &mut p.kernels {
            kr.members.retain(|m| m.node != n);
        }
        p.boundaries.retain(|b| b.node != n);
    }
    for &(n, role) in nodes {
        p.kernels[k].members.push(Member { node: n, role });
    }
    p.kernels[k].members.sort_by_key(|m| m.node);
    p.kernels.retain(|kr| !kr.members.is_empty() || kr.core.is_some());
    recross(g, p);
    let kg = graphs(g, p);
    check_component(g, p, &kg).into_keys().collect()
}

/// Recompute every kernel's crossings and a part order from the members (a valid one when it exists).
fn recross(g: &Graph<NNode>, p: &mut KernelPartition) {
    let mut part = vec![Part::Boundary(u32::MAX); g.node_count()];
    for b in &p.boundaries {
        part[b.node as usize] = Part::Boundary(b.owner.unwrap_or(b.node));
    }
    for (k, kr) in p.kernels.iter().enumerate() {
        for m in &kr.members {
            part[m.node as usize] = Part::Kernel(k as u32);
        }
    }
    for (k, kr) in p.kernels.iter_mut().enumerate() {
        let me = Part::Kernel(k as u32);
        kr.inputs.clear();
        kr.outputs.clear();
        for (u, out) in g.out.iter().enumerate() {
            for (v, l) in out {
                for e in l {
                    let c = cc_passes::kernels::json::Crossing {
                        src: u as u32,
                        dst: *v,
                        end: *e,
                    };
                    match (part[u] == me, part[*v as usize] == me) {
                        (false, true) => kr.inputs.push(c),
                        (true, false) => kr.outputs.push(c),
                        _ => {}
                    }
                }
            }
        }
        kr.inputs.sort();
        kr.outputs.sort();
    }
    // Parts in the order of their smallest node (topological for these graphs when acyclic).
    let mut first: Vec<(u32, Part)> = Vec::new();
    let mut seen = BTreeSet::new();
    for (n, q) in part.iter().enumerate() {
        if seen.insert(*q) {
            first.push((n as u32, *q));
        }
    }
    p.order = first.into_iter().map(|(_, q)| q).collect();
}

fn softmax() -> (Graph<NNode>, [u32; 5]) {
    let mut b = GraphBuilder::new();
    let x = b.input(ty(&[1, 2], F));
    let ninf = b.constant(ty(&[], F));
    let zero = b.constant(ty(&[], F));
    let m = b.reduce("stablehlo.maximum", ty(&[1], F), x, ninf);
    let bm = b.op("stablehlo.broadcast_in_dim", ty(&[1, 2], F), &[m]);
    let d = b.op("stablehlo.subtract", ty(&[1, 2], F), &[x, bm]);
    let e = b.op("stablehlo.exponential", ty(&[1, 2], F), &[d]);
    let s = b.reduce("stablehlo.add", ty(&[1], F), e, zero);
    let bs = b.op("stablehlo.broadcast_in_dim", ty(&[1, 2], F), &[s]);
    let y = b.op("stablehlo.divide", ty(&[1, 2], F), &[e, bs]);
    b.ret(&[y]);
    (b.finish(), [m, d, e, s, y])
}

fn two_gemms() -> (Graph<NNode>, [u32; 3]) {
    let mut b = GraphBuilder::new();
    let x = b.input(ty(&[1, 2], F));
    let w1 = b.input(ty(&[2, 3], F));
    let w2 = b.input(ty(&[3, 4], F));
    let a = b.op("stablehlo.dot_general", ty(&[1, 3], F), &[x, w1]);
    let t = b.op("stablehlo.tanh", ty(&[1, 3], F), &[a]);
    let r = b.op("stablehlo.transpose", ty(&[3, 1], F), &[a]);
    let c = b.op("stablehlo.dot_general", ty(&[1, 4], F), &[t, w2]);
    b.ret(&[c, r]);
    let _ = r;
    (b.finish(), [a, t, c])
}

#[test]
fn hand_built_partitions_pass_unchanged() {
    for g in [softmax().0, two_gemms().0] {
        let (p, kg) = artifacts(&g);
        assert!(check_component(&g, &p, &kg).is_empty());
    }
}

#[test]
fn a_non_convex_kernel_fails_k2() {
    // The divide joins the {subtract, exponential} map kernel: exp -> sum -> broadcast -> divide
    // leaves the kernel and re-enters it.
    let (g, [_, d, _, _, y]) = softmax();
    let (mut p, _) = artifacts(&g);
    let k = p
        .kernels
        .iter()
        .position(|k| k.members.iter().any(|m| m.node == d))
        .unwrap();
    let f = mutate(&g, &mut p, k, &[(y, Role::Map)]);
    assert!(f.contains(&"k2"), "{f:?}");
}

#[test]
fn two_cores_in_one_kernel_fail_k1() {
    let (g, [a, _, c]) = two_gemms();
    let (mut p, _) = artifacts(&g);
    let k = p.kernels.iter().position(|k| k.core == Some(a)).unwrap();
    let f = mutate(&g, &mut p, k, &[(c, Role::Core)]);
    assert!(f.contains(&"k1"), "{f:?}");
}

#[test]
fn an_escaping_intermediate_fails_k3() {
    // tanh joins the first GEMM although the GEMM's result also feeds the transpose.
    let (g, [a, t, _]) = two_gemms();
    let (mut p, _) = artifacts(&g);
    let k = p.kernels.iter().position(|k| k.core == Some(a)).unwrap();
    let f = mutate(&g, &mut p, k, &[(t, Role::Epilogue)]);
    assert!(f.contains(&"k3"), "{f:?}");
}

#[test]
fn a_misaligned_epilogue_fails_k4() {
    // The subtract is over the input's domain, not the max's: as the max's epilogue it is misaligned.
    let (g, [m, d, e, _, _]) = softmax();
    let (mut p, _) = artifacts(&g);
    let k = p.kernels.iter().position(|k| k.core == Some(m)).unwrap();
    let f = mutate(&g, &mut p, k, &[(d, Role::Epilogue), (e, Role::Epilogue)]);
    assert!(f.contains(&"k4"), "{f:?}");
}

#[test]
fn a_cyclic_quotient_fails_quotient() {
    // x1 -> y1, x2 -> y2 with kernels {x1, y2} and {y1, x2}: each convex, the quotient cyclic.
    let mut b = GraphBuilder::new();
    let i = b.input(ty(&[1], F));
    let x1 = b.op("stablehlo.negate", ty(&[1], F), &[i]);
    let y1 = b.op("stablehlo.negate", ty(&[1], F), &[x1]);
    let x2 = b.op("stablehlo.negate", ty(&[1], F), &[i]);
    let y2 = b.op("stablehlo.negate", ty(&[1], F), &[x2]);
    b.ret(&[y1, y2]);
    let g = b.finish();
    let (mut p, _) = artifacts(&g);
    let base = p.kernels[0].clone();
    p.kernels = vec![base.clone(), base];
    for (k, (kr, ms)) in p.kernels.iter_mut().zip([[x1, y2], [y1, x2]]).enumerate() {
        kr.id = format!("k{k}");
        kr.core = None;
        kr.members = ms
            .iter()
            .map(|&n| Member {
                node: n,
                role: Role::Map,
            })
            .collect();
        kr.members.sort_by_key(|m| m.node);
    }
    p.boundaries.retain(|b| ![x1, y1, x2, y2].contains(&b.node));
    recross(&g, &mut p);
    let kg = graphs(&g, &p);
    let f: Vec<&str> = check_component(&g, &p, &kg).into_keys().collect();
    assert!(f.contains(&"quotient"), "{f:?}");
    assert!(!f.contains(&"k2"), "each kernel is convex: {f:?}");
}

#[test]
fn a_layout_op_as_a_member_fails_classes() {
    let (g, [a, ..]) = two_gemms();
    let (mut p, _) = artifacts(&g);
    let r = p
        .boundaries
        .iter()
        .find(|b| b.kind == BoundaryKind::Layout)
        .map(|b| b.node)
        .unwrap();
    let k = p.kernels.iter().position(|k| k.core == Some(a)).unwrap();
    let f = mutate(&g, &mut p, k, &[(r, Role::Epilogue)]);
    assert!(f.contains(&"classes"), "{f:?}");
}

#[test]
fn a_dropped_edge_in_a_kernel_graph_fails_reconstruct() {
    let (g, _) = softmax();
    let (p, mut kg) = artifacts(&g);
    let k = kg
        .kernels
        .iter()
        .position(|k| {
            k.graph.edge_count() > 0
                && k.graph
                    .nodes
                    .iter()
                    .filter(|n| matches!(n, KNode::Member { .. }))
                    .count()
                    > 1
        })
        .unwrap();
    let gr = &mut kg.kernels[k].graph;
    let u = gr.out.iter().position(|o| !o.is_empty()).unwrap();
    gr.out[u].remove(0);
    let f: Vec<&str> = check_component(&g, &p, &kg).into_keys().collect();
    assert!(f.contains(&"reconstruct"), "{f:?}");
}

// Random graphs: the partition always passes the independent checker.

#[derive(Debug, Clone)]
enum Gen {
    Unary(usize),
    Binary(usize, usize),
    Dot(usize, usize),
    Reduce(usize),
    Bcast(usize),
    Transpose(usize),
}

fn gen_node() -> impl Strategy<Value = Gen> {
    prop_oneof![
        3 => any::<usize>().prop_map(Gen::Unary),
        3 => (any::<usize>(), any::<usize>()).prop_map(|(a, b)| Gen::Binary(a, b)),
        1 => (any::<usize>(), any::<usize>()).prop_map(|(a, b)| Gen::Dot(a, b)),
        1 => any::<usize>().prop_map(Gen::Reduce),
        1 => any::<usize>().prop_map(Gen::Bcast),
        1 => any::<usize>().prop_map(Gen::Transpose),
    ]
}

/// Values have one of two shapes: `[1, 2]` (full) and `[1]` (reduced). Operands are picked among the
/// earlier values of the shape the op needs.
fn random_graph(spec: &[Gen]) -> Graph<NNode> {
    let mut b = GraphBuilder::new();
    let mut full = vec![b.input(ty(&[1, 2], F)), b.input(ty(&[1, 2], F))];
    let mut low = vec![b.input(ty(&[1], F))];
    let init = b.constant(ty(&[], F));
    let pick = |v: &Vec<u32>, i: usize| v[i % v.len()];
    for s in spec {
        match *s {
            Gen::Unary(a) => {
                if a % 3 == 0 {
                    let x = b.op("stablehlo.negate", ty(&[1], F), &[pick(&low, a)]);
                    low.push(x);
                } else {
                    let x = b.op("stablehlo.exponential", ty(&[1, 2], F), &[pick(&full, a)]);
                    full.push(x);
                }
            }
            Gen::Binary(a, c) => {
                let x = b.op("stablehlo.add", ty(&[1, 2], F), &[pick(&full, a), pick(&full, c)]);
                full.push(x);
            }
            Gen::Dot(a, c) => {
                let x = b.op(
                    "stablehlo.dot_general",
                    ty(&[1, 2], F),
                    &[pick(&full, a), pick(&full, c)],
                );
                full.push(x);
            }
            Gen::Reduce(a) => {
                let x = b.reduce("stablehlo.add", ty(&[1], F), pick(&full, a), init);
                low.push(x);
            }
            Gen::Bcast(a) => {
                let x = b.op("stablehlo.broadcast_in_dim", ty(&[1, 2], F), &[pick(&low, a)]);
                full.push(x);
            }
            Gen::Transpose(a) => {
                let x = b.op("stablehlo.transpose", ty(&[1, 2], F), &[pick(&full, a)]);
                full.push(x);
            }
        }
    }
    let outs: Vec<u32> = full
        .iter()
        .rev()
        .take(2)
        .chain(low.iter().rev().take(1))
        .copied()
        .collect();
    b.ret(&outs);
    b.finish()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn random_graphs_partition_into_checked_kernels(spec in prop::collection::vec(gen_node(), 1..40)) {
        let g = random_graph(&spec);
        let (p, kg) = artifacts(&g);
        let bad = check_component(&g, &p, &kg);
        prop_assert!(bad.is_empty(), "{bad:#?}");
        // Deterministic.
        let (q, _) = artifacts(&g);
        prop_assert_eq!(p, q);
    }
}
