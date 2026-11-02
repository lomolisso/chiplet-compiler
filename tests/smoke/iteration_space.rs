//! `iteration_space`: the toy export (replayed) through every pass up to iteration_space with every check
//! green (templates forced on by low thresholds), and every check of its `verify` failing on a mutated
//! artifact.

use std::path::PathBuf;
use std::sync::Arc;

use cc_core::runner::{Caps, Invocation, Runners, ToolLog, ToolOutcome, ToolRunner};
use cc_core::{CancelToken, CheckStatus, LoadedConfig};
use cc_passes::iteration_space::json::{Routing, Summary};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::{Env, ExportReplay, repo_root};

const CONFIG: &str = r#"
[pipeline]
targets = ["iteration_space"]
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
        self.env
            .build_dir()
            .join(".iteration_space/out/iteration_space")
            .join(rel)
    }

    fn failing(&self) -> Vec<String> {
        let r = self.pipeline().verify_pass("iteration_space").unwrap();
        r.checks
            .into_iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name)
            .collect()
    }

    fn ids(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(self.out("skeletons"))
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

    fn routing(&self) -> Routing {
        serde_json::from_str(&std::fs::read_to_string(self.out("routing.json")).unwrap()).unwrap()
    }

    /// A class routed to a template.
    fn template(&self) -> String {
        self.routing()
            .classes
            .into_iter()
            .find(|c| c.route == cc_passes::iteration_space::json::Route::Template)
            .expect("the toy has a template class")
            .id
    }

    fn expect(&self, check: &str) {
        let f = self.failing();
        assert!(f.iter().any(|c| c == check), "{check} not in {f:?}");
    }
}

fn fresh() -> Setup {
    let s = Setup::new();
    let st = s.pipeline().run(&RunOptions::default()).unwrap();
    assert_eq!(st.status_of("iteration_space"), Some(RunStatus::Ok), "{st:?}");
    s
}

#[test]
fn routes_and_verifies_the_toy() {
    let s = fresh();
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let t: Summary = serde_json::from_str(&std::fs::read_to_string(s.out("summary.json")).unwrap()).unwrap();
    assert_eq!(t.classes as usize, s.ids().len());
    assert!(t.templates > 0 && t.datapaths > 0, "{t:?}");
    assert_eq!(s.routing().classes.len(), s.ids().len());
    let r = s.pipeline().verify_pass("iteration_space").unwrap();
    let names: Vec<&str> = r.checks.iter().map(|c| c.name.as_str()).collect();
    for id in s.ids() {
        for k in ["skeleton", "access", "dependence", "routing"] {
            assert!(names.contains(&format!("{id}:{k}").as_str()), "{id}:{k}");
        }
    }
    let id = s.template();
    assert!(names.contains(&format!("{id}:template").as_str()));
    let views = s.env.build_dir().join(".iteration_space/views");
    assert!(views.join("routing.md").is_file() && views.join("iteration_space.md").is_file());
}

#[test]
fn a_missing_skeleton_fails_complete() {
    let s = fresh();
    let id = s.ids()[0].clone();
    std::fs::remove_file(s.out(&format!("skeletons/{id}.json"))).unwrap();
    s.expect("complete");
}

#[test]
fn an_edited_reuse_fails_skeleton() {
    let s = fresh();
    let id = s.template();
    s.edit(s.out(&format!("skeletons/{id}.json")), |v| {
        v["operands"][0]["reuse"] = serde_json::json!([]);
    });
    s.expect(&format!("{id}:skeleton"));
}

#[test]
fn an_edited_access_or_obligation_fails_access() {
    let s = fresh();
    let id = s.template();
    s.edit(s.out(&format!("skeletons/{id}.json")), |v| {
        let a = v["accesses"].as_array_mut().unwrap();
        let x = a.iter_mut().find(|x| !x["operands"][0].is_null()).unwrap();
        x["operands"][0][0]["terms"][0]["k"] = 3.into();
    });
    s.expect(&format!("{id}:access"));

    let s = fresh();
    let id = s.ids()[0].clone();
    let p = s.out(&format!("smt/access.{id}.smt2"));
    let text = std::fs::read_to_string(&p).unwrap();
    std::fs::write(&p, format!("{text}; edited\n")).unwrap();
    s.expect(&format!("{id}:access"));
}

#[test]
fn an_edited_vector_fails_dependence() {
    let s = fresh();
    let id = s.template();
    s.edit(s.out(&format!("dependences/{id}.json")), |v| {
        let x = &mut v["vectors"][0]["v"];
        let n = x.as_array().unwrap().len();
        *x = serde_json::json!(vec![1; n]);
    });
    s.expect(&format!("{id}:dependence"));
}

#[test]
fn an_edited_mapping_fails_template() {
    let s = fresh();
    let id = s.template();
    s.edit(s.out(&format!("templates/{id}.json")), |v| {
        let l = &mut v["candidates"][0]["lambda"];
        let n = l.as_array().unwrap().len();
        *l = serde_json::json!(vec![0; n]);
    });
    s.expect(&format!("{id}:template"));

    let s = fresh();
    let id = s.template();
    s.edit(s.out(&format!("templates/{id}.json")), |v| {
        v["candidates"][0]["utilization"] = 0.0.into();
    });
    s.expect(&format!("{id}:template"));
}

#[test]
fn an_edited_route_fails_routing() {
    let s = fresh();
    let id = s.ids()[0].clone();
    s.edit(s.out("routing.json"), |v| {
        let c = v["classes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|c| c["id"] == id.as_str())
            .unwrap();
        c["route"] = if c["route"] == "template" {
            "datapath"
        } else {
            "template"
        }
        .into();
    });
    s.expect(&format!("{id}:routing"));
}

#[test]
fn an_edited_certificate_fails_embedding() {
    let s = fresh();
    s.edit(s.out("embedding.json"), |v| {
        let p = &mut v["pairs"][0];
        let d = p["dims"].as_array_mut().unwrap();
        if d.len() > 1 {
            d.swap(0, 1);
        } else {
            p["rows"][0] = serde_json::json!([99]);
        }
    });
    s.expect("embedding");
}
