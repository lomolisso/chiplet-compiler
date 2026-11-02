//! `datapaths`: the toy export (replayed) through every pass up to datapaths with every check green (one
//! class routed to a template by low thresholds, the others to datapaths), and every check of its
//! `verify` failing on a mutated artifact.

use std::path::PathBuf;
use std::sync::Arc;

use cc_core::runner::{Caps, Invocation, Runners, ToolLog, ToolOutcome, ToolRunner};
use cc_core::{CancelToken, CheckStatus, LoadedConfig};
use cc_passes::datapaths::json::{Datapath, Summary};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::{Env, ExportReplay, repo_root};

const CONFIG: &str = r#"
[pipeline]
targets = ["datapaths"]
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
[passes.kernel_classes.resources]
threads = 2
memory_gb = 2
timeout_min = 1
[passes.iteration_space.template]
min_reuse = 1
min_share = 0.0
[passes.iteration_space.resources]
threads = 2
memory_gb = 2
timeout_min = 5
[passes.datapaths.resources]
threads = 2
memory_gb = 2
timeout_min = 10
"#;

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

    fn config_with(&self, extra: &[&str]) -> LoadedConfig {
        let mut sets = sets();
        sets.extend(extra.iter().map(|s| s.to_string()));
        let sets: Vec<&str> = sets.iter().map(String::as_str).collect();
        self.env.config(CONFIG, &sets)
    }

    fn pipeline(&self) -> Pipeline {
        self.pipeline_with(&[])
    }

    fn pipeline_with(&self, extra: &[&str]) -> Pipeline {
        let cfg = self.config_with(extra);
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
        self.env.build_dir().join(".datapaths/out/datapaths").join(rel)
    }

    fn failing(&self) -> Vec<String> {
        self.failing_with(&[])
    }

    fn failing_with(&self, extra: &[&str]) -> Vec<String> {
        let r = self.pipeline_with(extra).verify_pass("datapaths").unwrap();
        r.checks
            .into_iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name)
            .collect()
    }

    fn ids(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(self.out("datapaths"))
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

    fn edit(&self, path: PathBuf, f: impl FnOnce(&mut serde_json::Value)) {
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        f(&mut v);
        std::fs::write(&path, v.to_string()).unwrap();
    }

    fn datapath(&self, id: &str) -> Datapath {
        serde_json::from_str(&std::fs::read_to_string(self.out(&format!("datapaths/{id}.json"))).unwrap())
            .unwrap()
    }

    /// A class whose datapath satisfies `f`.
    fn find(&self, f: impl Fn(&Datapath) -> bool) -> String {
        self.ids()
            .into_iter()
            .find(|id| f(&self.datapath(id)))
            .expect("the toy has such a class")
    }

    fn expect(&self, check: &str) {
        let f = self.failing();
        assert!(f.iter().any(|c| c == check), "{check} not in {f:?}");
    }
}

fn fresh() -> Setup {
    let s = Setup::new();
    let st = s.pipeline().run(&RunOptions::default()).unwrap();
    assert_eq!(st.status_of("datapaths"), Some(RunStatus::Ok), "{st:?}");
    s
}

#[test]
fn extracts_proves_and_verifies_the_toy() {
    let s = fresh();
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let t: Summary = serde_json::from_str(&std::fs::read_to_string(s.out("summary.json")).unwrap()).unwrap();
    assert_eq!(t.classes as usize, s.ids().len());
    assert_eq!(t.proofs, t.tuples);
    assert!(t.by_route.get("template").is_some_and(|n| *n > 0), "{t:?}");
    assert!(
        t.order_queries > 0 && t.order_free + t.order_bound == t.order_queries,
        "{t:?}"
    );
    for id in s.ids() {
        let d = s.datapath(&id);
        for tu in &d.tuples {
            let r: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(s.out(&format!("proofs/{id}/{}.result", tu.name))).unwrap(),
            )
            .unwrap();
            assert_eq!(r["verdict"], "unsat", "{id} {}", tu.name);
        }
    }
    let r = s.pipeline().verify_pass("datapaths").unwrap();
    let names: Vec<&str> = r.checks.iter().map(|c| c.name.as_str()).collect();
    for k in ["complete", "lemmas", "solver_sanity", "summary"] {
        assert!(names.contains(&k), "{k}");
    }
    assert!(s.env.build_dir().join(".datapaths/views/datapaths.md").is_file());
}

/// Exchanging the operands of a non-commutative vertex makes the obligation satisfiable (the verify's
/// sanity control runs it in both solvers) and the stored proof stale.
#[test]
fn a_swapped_port_fails_the_proof() {
    let s = fresh();
    let r = s.pipeline().verify_pass("datapaths").unwrap();
    let c = r.checks.iter().find(|c| c.name == "solver_sanity").unwrap();
    assert_eq!(c.status, CheckStatus::Pass, "{c:?}");
    let id = s.find(|d| {
        d.vertices.iter().any(|v| {
            matches!(v.op.as_str(), "subtract" | "divide")
                && v.args.len() == 2
                && v.args[0].src != v.args[1].src
        })
    });
    s.edit(s.out(&format!("datapaths/{id}.json")), |v| {
        let vs = v["vertices"].as_array_mut().unwrap();
        let x = vs
            .iter_mut()
            .find(|x| matches!(x["op"].as_str(), Some("subtract" | "divide")) && x["args"][0] != x["args"][1])
            .unwrap();
        x["args"].as_array_mut().unwrap().swap(0, 1);
    });
    s.expect(&format!("{id}:proof"));
}

/// A wrong rounding: a convert retargeted to another width variable changes the semantics.
#[test]
fn a_wrong_convert_fails_structure_and_proof() {
    let s = fresh();
    let id = s.find(|d| d.vertices.iter().any(|v| v.op == "convert"));
    s.edit(s.out(&format!("datapaths/{id}.json")), |v| {
        let vs = v["vertices"].as_array_mut().unwrap();
        let x = vs.iter_mut().find(|x| x["op"] == "convert").unwrap();
        x["op"] = "round_nearest_even".into();
        x["label"] = "round_nearest_even@float".into();
    });
    let f = s.failing();
    assert!(
        f.contains(&format!("{id}:structure")) && f.contains(&format!("{id}:proof")),
        "{f:?}"
    );
}

#[test]
fn an_edited_label_or_stage_fails_structure() {
    let s = fresh();
    let id = s.ids()[0].clone();
    s.edit(s.out(&format!("datapaths/{id}.json")), |v| {
        v["vertices"][0]["label"] = "nothing@float".into();
    });
    s.expect(&format!("{id}:structure"));

    let s = fresh();
    let id = s.find(|d| {
        d.vertices
            .iter()
            .any(|v| v.stage == cc_passes::datapaths::json::Stage::Epilogue)
    });
    s.edit(s.out(&format!("datapaths/{id}.json")), |v| {
        let vs = v["vertices"].as_array_mut().unwrap();
        let x = vs.iter_mut().find(|x| x["stage"] == "epilogue").unwrap();
        x["stage"] = "map".into();
    });
    s.expect(&format!("{id}:structure"));
}

#[test]
fn an_edited_query_or_order_flag_fails_the_proof() {
    let s = fresh();
    let id = s.ids()[0].clone();
    let p = s.out(&format!("proofs/{id}/t0.smt2"));
    let text = std::fs::read_to_string(&p).unwrap();
    std::fs::write(&p, format!("{text}; edited\n")).unwrap();
    s.expect(&format!("{id}:proof"));

    let s = fresh();
    let id = s.find(|d| d.accumulator.as_ref().is_some_and(|a| !a.order.is_empty()));
    s.edit(s.out(&format!("datapaths/{id}.json")), |v| {
        let o = &mut v["accumulator"]["order"][0]["free"];
        *o = (!o.as_bool().unwrap()).into();
    });
    s.expect(&format!("{id}:proof"));
}

#[test]
fn a_missing_result_fails_complete() {
    let s = fresh();
    let id = s.ids()[0].clone();
    std::fs::remove_file(s.out(&format!("proofs/{id}/t0.result"))).unwrap();
    s.expect("complete");
}

/// A proof that ends `unknown` fails unless `allow_unknown` lists the class with the stored reason.
#[test]
fn an_unknown_proof_needs_allow_unknown() {
    let s = fresh();
    let id = s.ids()[0].clone();
    s.edit(s.out(&format!("proofs/{id}/t0.result")), |v| {
        v["verdict"] = "unknown".into();
        v["allowed"] = "too hard for the toy".into();
    });
    let f = s.failing();
    assert!(f.contains(&format!("{id}:proof")), "{f:?}");
    let allow =
        format!("passes.datapaths.allow_unknown=[{{class = \"{id}\", reason = \"too hard for the toy\"}}]");
    let f = s.failing_with(&[allow.as_str()]);
    assert!(!f.contains(&format!("{id}:proof")), "{f:?}");
}

#[test]
fn an_edited_summary_fails_summary() {
    let s = fresh();
    s.edit(s.out("summary.json"), |v| {
        v["vertices"] = 0.into();
    });
    s.expect("summary");
}
