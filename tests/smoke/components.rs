//! `components`: the toy export (replayed) through import and components with every check green; a
//! shared decomposition gives one instance per call; every check of its `verify` fails on a mutated
//! artifact; component ids do not depend on op order, value numbering or the order of commutative
//! operands.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use cc_core::CheckStatus;
use cc_core::runner::Runners;
use cc_ir::Module;
use cc_ir::absgraph::PortLabel;
use cc_passes::components::algo::graph::ADim;
use cc_passes::components::json::{InstanceKind, InstancesFile, StoredComponent};
use cc_passes::export::json::{GraphFile, SymbolsFile};
use cc_passes::import::ImportConfig;
use cc_passes::import::algo::{build, size_tag};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use proptest::prelude::*;
use tests::{Env, ExportReplay, repo_root};

const CONFIG: &str = r#"
[pipeline]
targets = ["components"]
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
"#;

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

    fn pipeline(&self) -> Pipeline {
        let cfg = self.env.config(CONFIG, &[]);
        let mut runners = Runners::from_config(&cfg);
        runners.host = Arc::new(ExportReplay::default());
        Pipeline::new(cc_passes::registry(), cfg)
            .unwrap()
            .with_runners(runners)
    }

    fn run(&self, opts: &RunOptions) -> RunStatus {
        let s = self.pipeline().run(opts).unwrap();
        s.status_of("components").unwrap()
    }

    fn rerun_components(&self) -> RunStatus {
        self.run(&RunOptions {
            only: Some("components".into()),
            ..RunOptions::default()
        })
    }

    fn out(&self, rel: &str) -> PathBuf {
        self.env.build_dir().join(".components/out/components").join(rel)
    }

    fn ir(&self) -> PathBuf {
        self.env
            .build_dir()
            .join(".import/out/import/toy/forward/ir.postcard")
    }

    fn failing(&self) -> Vec<String> {
        let r = self.pipeline().verify_pass("components").unwrap();
        r.checks
            .into_iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name)
            .collect()
    }

    fn instances(&self) -> InstancesFile {
        serde_json::from_str(&std::fs::read_to_string(self.out("instances/toy/forward.json")).unwrap())
            .unwrap()
    }

    fn edit_module(&self, f: impl FnOnce(&mut Module)) {
        let mut m: Module = postcard::from_bytes(&std::fs::read(self.ir()).unwrap()).unwrap();
        f(&mut m);
        std::fs::write(self.ir(), postcard::to_stdvec(&m).unwrap()).unwrap();
    }

    fn edit_json(&self, p: PathBuf, f: impl FnOnce(&mut serde_json::Value)) {
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        f(&mut v);
        std::fs::write(&p, v.to_string()).unwrap();
    }
}

fn fresh() -> Setup {
    let s = Setup::new();
    assert_eq!(s.run(&RunOptions::default()), RunStatus::Ok);
    s
}

fn assert_fails(s: &Setup, check: &str) {
    let f = s.failing();
    assert!(
        f.iter().any(|n| n.ends_with(&format!(":{check}")) || n == check),
        "{check} should fail; failing: {f:?}"
    );
}

/// Component of every composite instance by name and source path.
fn by_path(f: &InstancesFile) -> BTreeMap<(String, String), String> {
    f.instances
        .iter()
        .filter(|i| i.kind == InstanceKind::Composite)
        .map(|i| {
            (
                (
                    i.name.clone().unwrap_or_default(),
                    i.position.cc_path.clone().unwrap_or_default(),
                ),
                i.component.clone().unwrap_or_default(),
            )
        })
        .collect()
}

#[test]
fn abstracts_and_verifies_the_toy() {
    let s = fresh();
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let f = s.instances();
    let comps = by_path(&f);
    // Five Dense calls of different widths are one component with different dim bindings.
    let dense: Vec<&String> = comps
        .iter()
        .filter(|((n, _), _)| n == "cc.flax.Dense")
        .map(|(_, c)| c)
        .collect();
    assert_eq!(dense.len(), 5, "{comps:?}");
    assert!(dense.iter().all(|c| *c == dense[0]), "{comps:?}");
    let dims: std::collections::BTreeSet<String> = f
        .instances
        .iter()
        .filter(|i| i.name.as_deref() == Some("cc.flax.Dense"))
        .map(|i| format!("{:?}", i.bindings.as_ref().unwrap().dims))
        .collect();
    assert!(dims.len() > 1, "{dims:?}");
    // The two blocks are one component whose call nodes name the Dense component.
    let blocks: Vec<&String> = comps
        .iter()
        .filter(|((n, _), _)| n == "cc.flax.Block")
        .map(|(_, c)| c)
        .collect();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0], blocks[1]);
    let block: StoredComponent =
        postcard::from_bytes(&std::fs::read(s.out(&format!("components/{}.postcard", blocks[0]))).unwrap())
            .unwrap();
    assert_eq!(block.height, 1);
    assert!(
        block
            .graph
            .nodes
            .iter()
            .any(|n| n.child.as_ref() == Some(dense[0]))
    );
    // Symbolic batch and sequence are bound per instance, not part of the component.
    let any_sym = f.instances.iter().any(|i| {
        i.bindings
            .as_ref()
            .is_some_and(|b| b.dims.iter().any(|d| matches!(d, cc_ir::Dim::Sym(_))))
    });
    assert!(any_sym);
    assert!(f.instances.iter().any(|i| i.kind == InstanceKind::Anon));
    assert!(
        s.env
            .build_dir()
            .join(".components/views/components.md")
            .is_file()
    );
    assert_eq!(s.run(&RunOptions::default()), RunStatus::CacheHit);
}

#[test]
fn a_shared_decomposition_gives_one_instance_per_call() {
    let s = fresh();
    s.edit_module(|m| {
        let calls: Vec<usize> = (0..m.ops.len())
            .filter(|&i| {
                m.ops[i].name == "stablehlo.composite"
                    && matches!(m.ops[i].attrs.get("name"), Some(cc_ir::Attr::Str(n)) if n == "cc.flax.Block")
            })
            .collect();
        assert_eq!(calls.len(), 2);
        m.ops[calls[1]].callee = m.ops[calls[0]].callee;
    });
    assert_eq!(s.rerun_components(), RunStatus::Ok);
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let f = s.instances();
    let blocks: Vec<_> = f
        .instances
        .iter()
        .filter(|i| i.name.as_deref() == Some("cc.flax.Block"))
        .collect();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].func, blocks[1].func);
    assert_ne!(blocks[0].id, blocks[1].id);
    assert_eq!(blocks[0].ops, blocks[1].ops);
    // The nested Dense calls exist once per executed path.
    let dense = f
        .instances
        .iter()
        .filter(|i| i.name.as_deref() == Some("cc.flax.Dense"))
        .count();
    assert_eq!(dense, 5);
}

#[test]
fn partition_fails_on_a_dropped_op() {
    let s = fresh();
    s.edit_json(s.out("instances/toy/forward.json"), |v| {
        let insts = v["instances"].as_array_mut().unwrap();
        let i = insts.iter_mut().find(|i| i["kind"] == "composite").unwrap();
        let r = i["ops"].as_array_mut().unwrap();
        let last = r.last_mut().unwrap();
        let hi = last[1].as_u64().unwrap();
        if last[0] == last[1] {
            r.pop();
        } else {
            last[1] = (hi - 1).into();
        }
    });
    assert_fails(&s, "partition");
}

#[test]
fn recoverability_fails_on_a_swapped_port() {
    let s = fresh();
    s.edit_module(|m| {
        let i = m
            .ops
            .iter()
            .position(|o| o.name == "stablehlo.dot_general")
            .unwrap();
        m.ops[i].operands.swap(0, 1);
    });
    assert_fails(&s, "recoverability");
}

#[test]
fn recoverability_fails_on_a_corrupted_binding() {
    let s = fresh();
    s.edit_json(s.out("instances/toy/forward.json"), |v| {
        let insts = v["instances"].as_array_mut().unwrap();
        let i = insts.iter_mut().find(|i| i["name"] == "cc.flax.Dense").unwrap();
        i["bindings"]["dims"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"const": 7}));
        let d = &mut i["bindings"]["dims"][0];
        *d = serde_json::json!({"const": 1234});
    });
    assert_fails(&s, "recoverability");
}

#[test]
fn classes_fail_on_an_edited_component() {
    let s = fresh();
    let f = s.instances();
    let id = f
        .instances
        .iter()
        .find(|i| i.name.as_deref() == Some("cc.flax.Dense"))
        .and_then(|i| i.component.clone())
        .unwrap();
    let p = s.out(&format!("components/{id}.postcard"));
    let mut c: StoredComponent = postcard::from_bytes(&std::fs::read(&p).unwrap()).unwrap();
    // Give one dim its own class: the graph no longer forces it.
    let k = c.dims;
    'outer: for n in &mut c.graph.nodes {
        for t in &mut n.results {
            for d in &mut t.dims {
                if let ADim::Var(_) = d {
                    *d = ADim::Var(k);
                    break 'outer;
                }
            }
        }
    }
    c.dims += 1;
    std::fs::write(&p, postcard::to_stdvec(&c).unwrap()).unwrap();
    assert_fails(&s, "classes");
}

#[test]
fn dedup_fails_on_a_wrong_component() {
    let s = fresh();
    let f = s.instances();
    let ids: std::collections::BTreeSet<String> =
        f.instances.iter().filter_map(|i| i.component.clone()).collect();
    assert!(ids.len() >= 2);
    let other = ids.iter().next().unwrap().clone();
    s.edit_json(s.out("instances/toy/forward.json"), |v| {
        let insts = v["instances"].as_array_mut().unwrap();
        let i = insts
            .iter_mut()
            .find(|i| i["component"].is_string() && i["component"] != other.as_str())
            .unwrap();
        i["component"] = other.clone().into();
    });
    assert_fails(&s, "dedup");
}

#[test]
fn sharing_table_fails_on_an_edited_view() {
    let s = fresh();
    let p = s.env.build_dir().join(".components/views/components.md");
    let t = std::fs::read_to_string(&p).unwrap();
    std::fs::write(&p, t.replace("components", "parts")).unwrap();
    assert_fails(&s, "sharing_table");
}

#[test]
fn a_commutative_edge_is_unordered() {
    // Every add, multiply, maximum of the toy's components labels its operands `*`.
    let s = fresh();
    let f = s.instances();
    for id in f.instances.iter().filter_map(|i| i.component.clone()) {
        let c: StoredComponent =
            postcard::from_bytes(&std::fs::read(s.out(&format!("components/{id}.postcard"))).unwrap())
                .unwrap();
        for (u, out) in c.graph.out.iter().enumerate() {
            for (v, l) in out {
                if let cc_passes::components::algo::graph::Kind::Op { name, .. } =
                    &c.graph.nodes[*v as usize].kind
                    && (name == "stablehlo.add" || name == "stablehlo.multiply")
                {
                    assert!(
                        l.iter().all(|e| e.port == PortLabel::Commutative),
                        "{id}: edge {u}->{v} {l:?}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------- order and numbering independence

fn fixture() -> (GraphFile, SymbolsFile) {
    let dir = repo_root().join("tests/fixtures/export/toy/forward");
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
    (
        serde_json::from_str(&read("graph.json")).unwrap(),
        serde_json::from_str(&read("symbols.json")).unwrap(),
    )
}

/// Components of the toy's instances (by name and source path; anon instances as a sorted list).
fn components_of(g: &GraphFile, syms: &SymbolsFile) -> (BTreeMap<(String, String), String>, Vec<String>) {
    use cc_passes::components::algo;
    let mut m = build::build(g, syms, &ImportConfig::default()).unwrap().module;
    size_tag::tag(&mut m);
    let out = algo::phase(&m, 64).unwrap();
    let (comps, members) = algo::dedup::dedup(std::slice::from_ref(&out), 1 << 24).unwrap();
    let id = |gi: Option<u32>| gi.map(|g| comps[members[0][g as usize].comp as usize].id.clone());
    let mut named = BTreeMap::new();
    let mut anon = Vec::new();
    for d in &out.instances {
        match &d.name {
            Some(n) => {
                let path = match d.attrs.get("composite_attributes") {
                    Some(cc_ir::Attr::Dict(x)) => x
                        .get("cc_path")
                        .and_then(cc_ir::Attr::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    _ => String::new(),
                };
                named.insert((n.clone(), path), id(d.graph).unwrap_or_default());
            }
            None => anon.push(id(d.graph).unwrap_or_default()),
        }
    }
    anon.sort();
    (named, anon)
}

fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

fn ops_mut(f: &mut cc_passes::export::json::Func) -> &mut Vec<cc_passes::export::json::Op> {
    &mut f.body.blocks[0].ops
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// Reordering every function's ops into another valid order keeps every component.
    #[test]
    fn op_order_keeps_the_components(mut seed in any::<u64>()) {
        let (g, syms) = fixture();
        let want = components_of(&g, &syms);
        let mut g2 = g.clone();
        for f in &mut g2.module.funcs {
            let args: std::collections::HashSet<u64> = f.body.blocks[0].args.iter().map(|a| a.v).collect();
            let ops = ops_mut(f);
            let mut pending = std::mem::take(ops);
            let term = pending.pop().unwrap();
            let mut avail = args;
            while !pending.is_empty() {
                let ready: Vec<usize> = (0..pending.len())
                    .filter(|&i| pending[i].operands.iter().all(|v| avail.contains(v)))
                    .collect();
                let pick = ready[(next(&mut seed) as usize) % ready.len()];
                let op = pending.remove(pick);
                avail.extend(op.results.iter().map(|r| r.v));
                ops.push(op);
            }
            ops.push(term);
        }
        prop_assert_eq!(components_of(&g2, &syms), want);
    }

    /// Renumbering values keeps every component.
    #[test]
    fn value_numbering_keeps_the_components(seed in any::<u64>()) {
        let (g, syms) = fixture();
        let want = components_of(&g, &syms);
        let s = serde_json::to_string(&g).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&s).unwrap();
        let k = (seed % 1000) + 7;
        fn walk(v: &mut serde_json::Value, k: u64) {
            match v {
                serde_json::Value::Object(o) => {
                    for (key, x) in o.iter_mut() {
                        match (key.as_str(), &mut *x) {
                            ("v", serde_json::Value::Number(n)) => {
                                *x = (n.as_u64().unwrap() * 3 + k).into();
                            }
                            ("operands", serde_json::Value::Array(a)) => {
                                for n in a.iter_mut() {
                                    *n = (n.as_u64().unwrap() * 3 + k).into();
                                }
                            }
                            _ => walk(x, k),
                        }
                    }
                }
                serde_json::Value::Array(a) => a.iter_mut().for_each(|x| walk(x, k)),
                _ => {}
            }
        }
        walk(&mut v, k);
        let g2: GraphFile = serde_json::from_value(v).unwrap();
        prop_assert_eq!(components_of(&g2, &syms), want);
    }

    /// Swapping the operands of commutative ops keeps every component.
    #[test]
    fn commutative_operand_order_keeps_the_components(mut seed in any::<u64>()) {
        let (g, syms) = fixture();
        let want = components_of(&g, &syms);
        let mut g2 = g.clone();
        for f in &mut g2.module.funcs {
            for op in ops_mut(f) {
                let comm = cc_ir::lookup(&op.name, None).is_some_and(|s| s.commutative);
                if comm && next(&mut seed).is_multiple_of(2) {
                    op.operands.swap(0, 1);
                }
            }
        }
        prop_assert_eq!(components_of(&g2, &syms), want);
    }
}
