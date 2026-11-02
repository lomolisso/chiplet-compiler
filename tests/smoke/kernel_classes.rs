//! `kernel_classes`: the toy export (replayed) through every pass up to kernel_classes with every check
//! green, and every check of its `verify` failing on a mutated artifact.

use std::path::PathBuf;
use std::sync::Arc;

use cc_core::runner::{Caps, Invocation, Runners, ToolLog, ToolOutcome, ToolRunner};
use cc_core::{CancelToken, CheckStatus, LoadedConfig};
use cc_passes::kernel_classes::json::{ClassesSummary, Sharing};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::{Env, ExportReplay, repo_root};

const CONFIG: &str = r#"
[pipeline]
targets = ["kernel_classes"]
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
            .join(".kernel_classes/out/kernel_classes")
            .join(rel)
    }

    fn failing(&self) -> Vec<String> {
        let r = self.pipeline().verify_pass("kernel_classes").unwrap();
        r.checks
            .into_iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name)
            .collect()
    }

    fn ids(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(self.out("classes"))
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

    /// The class with the most kernel nodes and a core (its certificate has the most to check).
    fn biggest(&self) -> String {
        self.ids()
            .into_iter()
            .max_by_key(|id| {
                let v: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(self.out(&format!("classes/{id}.json"))).unwrap(),
                )
                .unwrap();
                (v["kernel_nodes"].as_u64().unwrap(), id.clone())
            })
            .unwrap()
    }

    fn expect(&self, check: &str) {
        let f = self.failing();
        assert!(f.iter().any(|c| c == check), "{check} not in {f:?}");
    }
}

fn fresh() -> Setup {
    let s = Setup::new();
    let st = s.pipeline().run(&RunOptions::default()).unwrap();
    assert_eq!(st.status_of("kernel_classes"), Some(RunStatus::Ok), "{st:?}");
    s
}

#[test]
fn classifies_and_verifies_the_toy() {
    let s = fresh();
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let t: ClassesSummary =
        serde_json::from_str(&std::fs::read_to_string(s.out("summary.json")).unwrap()).unwrap();
    assert_eq!(t.classes as usize, s.ids().len());
    assert!(
        t.kinds.contains_key("contraction") && t.kinds.contains_key("map"),
        "{t:?}"
    );
    assert!(t.classes <= t.kernels && t.rows > t.kernels, "{t:?}");
    let sh: Sharing = serde_json::from_str(&std::fs::read_to_string(s.out("sharing.json")).unwrap()).unwrap();
    assert_eq!(sh.models, vec!["toy".to_string()]);
    let r = s.pipeline().verify_pass("kernel_classes").unwrap();
    let names: Vec<&str> = r.checks.iter().map(|c| c.name.as_str()).collect();
    for ph in ["forward", "forward_b1"] {
        assert!(
            names.contains(&format!("workload:toy/{ph}").as_str()),
            "{names:?}"
        );
    }
    let views = s.env.build_dir().join(".kernel_classes/views");
    assert!(views.join("sharing.md").is_file() && views.join("classes.md").is_file());
}

#[test]
fn a_dropped_row_fails_complete() {
    let s = fresh();
    let id = s.biggest();
    s.edit(s.out(&format!("instances/{id}.json")), |v| {
        v["rows"].as_array_mut().unwrap().pop();
    });
    s.expect("complete");
}

#[test]
fn a_misfiled_class_fails_ids() {
    let s = fresh();
    let id = s.biggest();
    s.edit(s.out(&format!("classes/{id}.json")), |v| {
        v["hash"] = "0".repeat(32).into()
    });
    s.expect("ids");
}

#[test]
fn an_edited_node_map_fails_certificate() {
    let s = fresh();
    let id = s.biggest();
    s.edit(s.out(&format!("certificates/{id}.json")), |v| {
        let m = v["certificates"][0]["nodes"].as_array_mut().unwrap();
        let last = m.len() - 1;
        m.swap(0, last);
    });
    s.expect(&format!("{id}:certificate"));
}

#[test]
fn an_edited_axis_map_or_local_class_fails_certificate() {
    let s = fresh();
    let id = s.biggest();
    s.edit(s.out(&format!("certificates/{id}.json")), |v| {
        let m = v["certificates"][0]["axes"].as_array_mut().unwrap();
        m[0] = m[1].clone();
    });
    s.expect(&format!("{id}:certificate"));

    let s = fresh();
    let id = s.biggest();
    s.edit(s.out(&format!("classes/{id}.json")), |v| {
        let nd = v["local"]["nd"].as_u64().unwrap();
        v["local"]["dims"][0][0][0] = nd.into();
    });
    s.expect(&format!("{id}:certificate"));
}

#[test]
fn an_edited_extent_or_workload_fails_rows() {
    let s = fresh();
    let id = s.biggest();
    s.edit(s.out(&format!("instances/{id}.json")), |v| {
        let e = &mut v["rows"][0]["extents"][0];
        *e = (e.as_u64().unwrap() + 1).into();
    });
    s.expect(&format!("{id}:rows"));

    let s = fresh();
    let id = s.biggest();
    s.edit(s.out(&format!("instances/{id}.json")), |v| {
        let w = &mut v["rows"][0]["workload"];
        *w = (w.as_u64().unwrap() * 2).into();
    });
    s.expect(&format!("{id}:rows"));
}

#[test]
fn an_edited_census_fails_workload() {
    let s = fresh();
    let p = s
        .env
        .build_dir()
        .join(".export/out/export/toy/forward/census_concrete.json");
    s.edit(p, |v| {
        let f = &mut v["flops"];
        *f = (f.as_u64().unwrap() + 2).into();
    });
    s.expect("workload:toy/forward");
}

#[test]
fn an_edited_report_or_view_fails_sharing() {
    let s = fresh();
    s.edit(s.out("sharing.json"), |v| {
        v["all"]["kernels"]["shared"] = 99.into()
    });
    s.expect("sharing");

    let s = fresh();
    std::fs::write(
        s.env.build_dir().join(".kernel_classes/views/sharing.md"),
        "edited\n",
    )
    .unwrap();
    s.expect("sharing");
}
