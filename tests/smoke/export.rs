//! `export`: the pass drives the exporter (replayed from `tests/fixtures/export/`, a real export of
//! the toy Flax port, see `make.py` there), and every check of its `verify` fails on a mutated artifact.

use std::path::PathBuf;
use std::sync::Arc;

use cc_core::CheckStatus;
use cc_core::runner::Runners;
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::{Env, ExportReplay};

const CONFIG: &str = r#"
[pipeline]
targets = ["export"]
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
"#;

struct Setup {
    env: Env,
    replay: Arc<ExportReplay>,
}

impl Setup {
    fn new() -> Self {
        let env = Env::new();
        // The catalog: model directories holding a recipe.py.
        std::fs::create_dir_all(env.root().join("models/toy")).unwrap();
        std::fs::write(env.root().join("models/toy/recipe.py"), "RECIPE = {}\n").unwrap();
        Self {
            env,
            replay: Arc::new(ExportReplay::default()),
        }
    }

    fn pipeline(&self) -> Pipeline {
        let cfg = self.env.config(CONFIG, &[]);
        let mut runners = Runners::from_config(&cfg);
        runners.host = self.replay.clone();
        Pipeline::new(cc_passes::registry(), cfg)
            .unwrap()
            .with_runners(runners)
    }

    fn run(&self) -> RunStatus {
        let s = self.pipeline().run(&RunOptions::default()).unwrap();
        s.status_of("export").unwrap()
    }

    fn out(&self, rel: &str) -> PathBuf {
        self.env.build_dir().join(".export/out/export/toy").join(rel)
    }

    fn failing(&self) -> Vec<String> {
        let r = self.pipeline().verify_pass("export").unwrap();
        r.checks
            .into_iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name)
            .collect()
    }

    fn edit_json(&self, rel: &str, f: impl FnOnce(&mut serde_json::Value)) {
        let p = self.out(rel);
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        f(&mut v);
        std::fs::write(&p, v.to_string()).unwrap();
    }
}

fn fresh() -> Setup {
    let s = Setup::new();
    assert_eq!(s.run(), RunStatus::Ok);
    s
}

fn assert_fails(s: &Setup, check: &str) {
    let f = s.failing();
    assert!(
        f.contains(&format!("toy:{check}")),
        "{check} should fail; failing: {f:?}"
    );
}

#[test]
fn runs_verifies_and_caps_each_model() {
    let s = fresh();
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let calls = s.replay.calls.lock().unwrap();
    let (export, caps) = calls
        .iter()
        .find(|(i, _)| i.args.iter().any(|a| a == "--model"))
        .unwrap();
    assert_eq!(export.program, "python3");
    let a = export.args.join(" ");
    assert!(
        a.starts_with("-m cc_export --model toy --config reduced --out "),
        "{a}"
    );
    assert!(
        export
            .env
            .iter()
            .any(|(k, v)| k == "PYTHONPATH" && v.ends_with("/python:")
                || k == "PYTHONPATH" && v.contains("/python:"))
    );
    // memory_gb / threads per model process, one CPU
    assert_eq!(caps.memory_gb, 4.0);
    assert_eq!(caps.cpus, 1.0);
    assert!(s.env.build_dir().join(".export/views/coverage.md").is_file());
}

#[test]
fn cache_hit_then_source_edit_reruns() {
    let s = fresh();
    let p = s.pipeline().run(&RunOptions::default()).unwrap();
    assert_eq!(p.status_of("export"), Some(RunStatus::CacheHit));
    std::fs::write(s.env.root().join("models/toy/net.py"), "# edited\n").unwrap();
    let p = s.pipeline().run(&RunOptions::default()).unwrap();
    assert_eq!(
        p.status_of("export"),
        Some(RunStatus::Ok),
        "a port edit reruns export"
    );
}

#[test]
fn exported_fails_on_phase_error() {
    let s = fresh();
    s.edit_json("export.json", |v| {
        v["phases"]["forward"]["error"] = "Boom: symbolic".into()
    });
    assert_fails(&s, "exported");
}

#[test]
fn schema_fails_on_unknown_field() {
    let s = fresh();
    s.edit_json("forward/graph.json", |v| v["surprise"] = 1.into());
    assert_fails(&s, "schema");
}

#[test]
fn mlir_verifier_fails_when_the_module_does_not_verify() {
    let s = fresh();
    let mut facts: serde_json::Value = serde_json::from_str(&s.replay.facts.lock().unwrap()).unwrap();
    facts["forward"]["verified"] = false.into();
    *s.replay.facts.lock().unwrap() = facts.to_string();
    assert_fails(&s, "mlir_verifier");
}

/// The first op named `name` in the entry function's callee chain (any function).
fn first_op<'a>(v: &'a mut serde_json::Value, name: &str) -> Option<&'a mut serde_json::Value> {
    for f in v["module"]["funcs"].as_array_mut()? {
        for op in f["body"]["blocks"][0]["ops"].as_array_mut()? {
            if op["name"] == name {
                return Some(op);
            }
        }
    }
    None
}

#[test]
fn roundtrip_fails_on_a_dropped_op() {
    let s = fresh();
    s.edit_json("forward/graph.json", |v| {
        let funcs = v["module"]["funcs"].as_array_mut().unwrap();
        let f = funcs.iter_mut().find(|f| f["name"] == "main").unwrap();
        let ops = f["body"]["blocks"][0]["ops"].as_array_mut().unwrap();
        let i = ops
            .iter()
            .position(|o| o["name"] == "stablehlo.constant")
            .unwrap();
        ops.remove(i);
    });
    assert_fails(&s, "roundtrip");
}

#[test]
fn census_and_recipe_fail_on_a_wrong_count() {
    let s = fresh();
    s.edit_json("forward/census.json", |v| {
        v["summary"]["dot_general f32xf32->f32"]["count"] = 99.into();
    });
    assert_fails(&s, "census");

    let s = fresh();
    s.edit_json("export.json", |v| {
        v["recipe"]["audit"]["allow"] = serde_json::json!({"dot_general i8xi8->i32": "int8 only"});
    });
    assert_fails(&s, "recipe_audit");
}

#[test]
fn census_concrete_fails_without_flops_or_on_other_counts() {
    let s = fresh();
    s.edit_json("forward/census_concrete.json", |v| {
        v["flops"] = serde_json::Value::Null;
        v["unknown"] = 1.into();
    });
    assert_fails(&s, "census_concrete");

    let s = fresh();
    s.edit_json("forward/census_concrete.json", |v| {
        v["summary"]["dot_general f32xf32->f32"]["count"] = 6.into();
    });
    assert_fails(&s, "census_concrete");
}

#[test]
fn decompositions_fail_on_a_dangling_symbol() {
    let s = fresh();
    s.edit_json("forward/graph.json", |v| {
        let op = first_op(v, "stablehlo.composite").unwrap();
        op["attributes"]["decomposition"]["value"] = "nowhere".into();
    });
    assert_fails(&s, "decompositions");
}

#[test]
fn components_fail_on_interception_failure_or_wrong_coverage() {
    let s = fresh();
    s.edit_json("forward/graph.json", |v| {
        v["interception"]["failures"] = serde_json::json!([{"name": "cc.flax.X", "path": "x", "error": "E"}]);
    });
    assert_fails(&s, "components");

    let s = fresh();
    s.edit_json("forward/graph.json", |v| v["coverage"]["anon"] = 0.into());
    assert_fails(&s, "components");
}

#[test]
fn symbols_fail_when_undeclared_or_absent() {
    let s = fresh();
    s.edit_json("export.json", |v| {
        v["symbols"].as_object_mut().unwrap().remove("S");
    });
    assert_fails(&s, "symbols");

    let s = fresh();
    s.edit_json("forward/symbols.json", |v| {
        v["symbols"]["B"]["inputs"] = serde_json::json!([])
    });
    assert_fails(&s, "symbols");

    let s = fresh();
    s.edit_json("forward/symbols.json", |v| {
        v["symbols"]["S"].as_object_mut().unwrap().remove("size");
    });
    assert_fails(&s, "symbols");

    // `S >= 2` must hold at the sizes.
    let s = fresh();
    s.edit_json("forward/symbols.json", |v| v["symbols"]["S"]["size"] = 1.into());
    assert_fails(&s, "symbols");
}

#[test]
fn invariance_fails_when_the_perturbed_graph_differs() {
    let s = fresh();
    s.edit_json("export.json", |v| {
        v["phases"]["forward"]["checks"]["invariance"]["perturbed_hash"] = "blake2b:other".into();
    });
    assert_fails(&s, "invariance");

    let s = fresh();
    s.edit_json("export.json", |v| {
        v["phases"]["forward"]["checks"]["invariance"]["changed_inputs"] =
            serde_json::json!([{"path": "[1]", "base": [2, 8, 4], "perturbed": [3, 8, 4]}]);
    });
    assert_fails(&s, "invariance");
}

#[test]
fn size_variants_fail_when_graphs_differ() {
    let s = fresh();
    s.edit_json("export.json", |v| {
        v["phases"]["forward_b1"]["hash"] = "blake2b:other".into()
    });
    assert_fails(&s, "size_variants");
}
