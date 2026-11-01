//! `import`: the toy export (`tests/fixtures/export/`, replayed) imports with every check green, every
//! check of its `verify` fails on a mutated artifact, an op outside the table is a hard error, and the
//! module's hash does not depend on op order or value numbering.

use std::path::PathBuf;
use std::sync::Arc;

use cc_core::CheckStatus;
use cc_core::runner::Runners;
use cc_ir::{Dim, Module, SymExpr, Type};
use cc_passes::export::json::{GraphFile, SymbolsFile};
use cc_passes::import::ImportConfig;
use cc_passes::import::algo::{build, size_tag};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use proptest::prelude::*;
use tests::{Env, ExportReplay, repo_root};

const CONFIG: &str = r#"
[pipeline]
targets = ["import"]
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
        s.status_of("import").unwrap()
    }

    fn out(&self, rel: &str) -> PathBuf {
        self.env.build_dir().join(".import/out/import").join(rel)
    }

    fn export_out(&self, rel: &str) -> PathBuf {
        self.env.build_dir().join(".export/out/export/toy").join(rel)
    }

    fn failing(&self) -> Vec<String> {
        let r = self.pipeline().verify_pass("import").unwrap();
        r.checks
            .into_iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name)
            .collect()
    }

    fn module(&self) -> Module {
        postcard::from_bytes(&std::fs::read(self.out("toy/forward/ir.postcard")).unwrap()).unwrap()
    }

    fn edit_module(&self, f: impl FnOnce(&mut Module)) {
        let mut m = self.module();
        f(&mut m);
        std::fs::write(
            self.out("toy/forward/ir.postcard"),
            postcard::to_stdvec(&m).unwrap(),
        )
        .unwrap();
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

fn first_op(m: &Module, name: &str) -> usize {
    m.ops.iter().position(|o| o.name == name).unwrap()
}

#[test]
fn imports_and_verifies_the_toy_export() {
    let s = fresh();
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let m = s.module();
    // Every value's dims are symbolic where the export had `?`: the dot results are [B, S, 16].
    let dot = &m.ops[first_op(&m, "stablehlo.dot_general")];
    let t = m.value(dot.results[0]).ty.tensor().unwrap().clone();
    let sym = |x: &str| Dim::Sym(SymExpr::parse(x).unwrap());
    assert_eq!(t.shape, vec![sym("B"), sym("S"), Dim::Const(16)]);
    // The toy divides by its sequence length: `S` enters compute through a convert.
    let summary: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(s.out("toy/forward/ir.postcard.summary.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(summary["dim_as_data"]["stablehlo.convert"], 2, "{summary}");
    assert!(summary["size_ops"].as_u64().unwrap() > 0);
    assert_eq!(summary["ledger"]["calls"], 3, "{summary}");
    let inv: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(s.out("op_inventory.json")).unwrap()).unwrap();
    assert_eq!(inv["unknown"], serde_json::json!([]));
    assert!(s.env.build_dir().join(".import/views/toy/forward.dot").is_file());
    assert_eq!(s.run(&RunOptions::default()), RunStatus::CacheHit);
}

#[test]
fn an_op_outside_the_table_is_a_hard_error() {
    let s = fresh();
    s.edit_json(s.export_out("forward/graph.json"), |v| {
        for f in v["module"]["funcs"].as_array_mut().unwrap() {
            for op in f["body"]["blocks"][0]["ops"].as_array_mut().unwrap() {
                if op["name"] == "stablehlo.dot_general" {
                    op["name"] = "stablehlo.mystery".into();
                }
            }
        }
    });
    let opts = RunOptions {
        only: Some("import".into()),
        ..RunOptions::default()
    };
    let summary = s.pipeline().run(&opts).unwrap();
    assert_eq!(summary.status_of("import"), Some(RunStatus::Failed));
    let inv: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(s.out("op_inventory.json")).unwrap()).unwrap();
    assert_eq!(inv["unknown"], serde_json::json!(["stablehlo.mystery"]));
    let err = summary
        .pass("import")
        .and_then(|p| p.message.clone())
        .unwrap_or_default();
    assert!(
        err.contains("toy/forward")
            && err.contains("stablehlo.mystery")
            && err.contains("op-semantics table"),
        "{err}"
    );
}

#[test]
fn ssa_fails_on_a_use_before_its_definition() {
    let s = fresh();
    s.edit_module(|m| {
        let last = cc_ir::ValueId(m.values.len() as u32 - 1);
        let i = m.ops.iter().position(|o| !o.operands.is_empty()).unwrap();
        m.ops[i].operands[0] = last;
    });
    assert_fails(&s, "ssa");
}

#[test]
fn types_fail_on_a_wrong_result_dim() {
    let s = fresh();
    s.edit_module(|m| {
        let r = m.ops[first_op(m, "stablehlo.dot_general")].results[0];
        if let Type::Tensor(t) = &mut m.values[r.index()].ty {
            t.shape[2] = Dim::Const(7);
        }
    });
    assert_fails(&s, "types");
}

#[test]
fn sym_eval_fails_on_a_wrong_symbolic_dim() {
    let s = fresh();
    s.edit_module(|m| {
        let r = m.ops[first_op(m, "stablehlo.dynamic_broadcast_in_dim")].results[0];
        if let Type::Tensor(t) = &mut m.values[r.index()].ty {
            t.shape[0] = Dim::Sym(SymExpr::parse("2*B").unwrap());
        }
    });
    assert_fails(&s, "sym_eval");
}

#[test]
fn census_fails_on_a_wrong_ledger_or_count() {
    let s = fresh();
    s.edit_json(s.out("toy/forward/ir.postcard.summary.json"), |v| {
        v["ledger"]["calls"] = 99.into();
    });
    assert_fails(&s, "census");

    let s = fresh();
    s.edit_json(s.export_out("forward/census.json"), |v| {
        v["executed"]["stablehlo.add"] = 99.into();
    });
    assert_fails(&s, "census");
}

#[test]
fn coverage_fails_on_an_unknown_op() {
    let s = fresh();
    s.edit_module(|m| {
        let i = first_op(m, "stablehlo.maximum");
        m.ops[i].name = "stablehlo.mystery".into();
    });
    assert_fails(&s, "coverage");
}

#[test]
fn size_tag_fails_on_a_flipped_tag_or_a_missing_crossing() {
    let s = fresh();
    s.edit_module(|m| {
        let i = m
            .ops
            .iter()
            .position(|o| o.size && o.name == "stablehlo.concatenate")
            .unwrap();
        m.ops[i].size = false;
    });
    assert_fails(&s, "size_tag");

    let s = fresh();
    s.edit_module(|m| {
        let i = m.ops.iter().position(|o| !o.dim_as_data.is_empty()).unwrap();
        m.ops[i].dim_as_data.clear();
    });
    assert_fails(&s, "size_tag");
}

#[test]
fn summary_fails_on_an_edited_summary() {
    let s = fresh();
    s.edit_json(s.out("toy/forward/ir.postcard.summary.json"), |v| {
        v["ops"] = 1.into();
    });
    assert_fails(&s, "summary");
}

#[test]
fn inventory_and_imported_fail_on_missing_entries() {
    let s = fresh();
    s.edit_json(s.out("op_inventory.json"), |v| {
        v["ops"].as_object_mut().unwrap().remove("stablehlo.add");
    });
    assert_fails(&s, "inventory");

    let s = fresh();
    std::fs::remove_file(s.out("toy/forward_b1/ir.postcard")).unwrap();
    assert_fails(&s, "imported");
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

fn hash(g: &GraphFile, syms: &SymbolsFile) -> String {
    let mut m = build::build(g, syms, &ImportConfig::default()).unwrap().module;
    size_tag::tag(&mut m);
    cc_ir::hash::module_hash(&m)
}

/// A pseudo-random generator for the permutations (seeded by proptest).
fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Reordering every function's ops into another valid order (definitions before uses, the return
    /// last) gives the same module hash.
    #[test]
    fn op_order_does_not_change_the_hash(mut seed in any::<u64>()) {
        let (g, syms) = fixture();
        let want = hash(&g, &syms);
        let mut g2 = g.clone();
        for f in &mut g2.module.funcs {
            let block = &mut f.body.blocks[0];
            let defined: std::collections::HashSet<u64> = block.args.iter().map(|a| a.v).collect();
            let mut pending = std::mem::take(&mut block.ops);
            let term = pending.pop().unwrap();
            let mut avail = defined;
            while !pending.is_empty() {
                let ready: Vec<usize> = (0..pending.len())
                    .filter(|&i| pending[i].operands.iter().all(|v| avail.contains(v)))
                    .collect();
                let pick = ready[(next(&mut seed) as usize) % ready.len()];
                let op = pending.remove(pick);
                avail.extend(op.results.iter().map(|r| r.v));
                block.ops.push(op);
            }
            block.ops.push(term);
        }
        prop_assert_eq!(hash(&g2, &syms), want);
    }

    /// Renumbering every value of `graph.json` gives the same module hash.
    #[test]
    fn value_numbering_does_not_change_the_hash(seed in any::<u64>()) {
        let (g, syms) = fixture();
        let want = hash(&g, &syms);
        let text = serde_json::to_string(&g).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let shift = 1000 + seed % 100_000;
        fn renumber(v: &mut serde_json::Value, shift: u64) {
            match v {
                serde_json::Value::Object(o) => {
                    for (k, x) in o.iter_mut() {
                        match (k.as_str(), &mut *x) {
                            ("v", serde_json::Value::Number(n)) => *x = (n.as_u64().unwrap() * 7 + shift).into(),
                            ("operands", serde_json::Value::Array(a)) => {
                                for e in a.iter_mut() {
                                    *e = (e.as_u64().unwrap() * 7 + shift).into();
                                }
                            }
                            _ => renumber(x, shift),
                        }
                    }
                }
                serde_json::Value::Array(a) => a.iter_mut().for_each(|x| renumber(x, shift)),
                _ => {}
            }
        }
        renumber(&mut v, shift);
        let g2: GraphFile = serde_json::from_value(v).unwrap();
        prop_assert_eq!(hash(&g2, &syms), want);
    }
}
