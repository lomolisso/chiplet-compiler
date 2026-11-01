//! `normalize`: the toy export (replayed) through import, components and normalize with every check
//! green and real solver runs (the toolchain image, or host binaries when `CC_SMT_RUNNER=host`, as in
//! CI); an unsound rule fails its proof; every check of its `verify` fails on a mutated artifact;
//! renumbering a component's nodes does not change what normalization does.

use std::path::PathBuf;
use std::sync::Arc;

use cc_core::runner::{BoundRunner, Caps, Invocation, Runners, ToolLog, ToolOutcome, ToolRunner};
use cc_core::{CancelToken, CheckStatus, LoadedConfig};
use cc_ir::absgraph::Graph;
use cc_passes::components::algo::graph::{ADim, ANode};
use cc_passes::components::json::StoredComponent;
use cc_passes::normalize::algo::normalize;
use cc_passes::normalize::config::{NormalizeConfig, Orders};
use cc_passes::normalize::json::{LedgerFile, NormalizedComponent};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use cc_smt::{Location, Solver, Verdict};
use proptest::prelude::*;
use tests::{Env, ExportReplay, repo_root};

const CONFIG: &str = r#"
[pipeline]
targets = ["normalize"]
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
[passes.normalize]
[passes.normalize.order]
float = "seq"
complex = "seq"
[passes.normalize.resources]
threads = 2
memory_gb = 2
timeout_min = 5
"#;

/// Export calls are replayed from the fixture; solver calls (`smt-*`) run for real.
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

/// The repository's image tag (the solvers' image), and `tools.smt.runner=host` when
/// `CC_SMT_RUNNER=host` (CI downloads the pinned binaries onto `PATH`).
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
        self.env.build_dir().join(".normalize/out/normalize").join(rel)
    }

    fn input(&self, rel: &str) -> PathBuf {
        self.env.build_dir().join(".components/out/components").join(rel)
    }

    fn failing(&self) -> Vec<String> {
        let r = self.pipeline().verify_pass("normalize").unwrap();
        r.checks
            .into_iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name)
            .collect()
    }

    fn ids(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(self.input("components"))
            .unwrap()
            .filter_map(|e| {
                e.unwrap()
                    .file_name()
                    .to_str()
                    .and_then(|n| n.strip_suffix(".postcard"))
                    .map(str::to_string)
            })
            .collect();
        v.sort();
        v
    }

    fn normalized(&self, id: &str) -> NormalizedComponent {
        postcard::from_bytes(&std::fs::read(self.out(&format!("components/{id}.postcard"))).unwrap()).unwrap()
    }

    fn write_normalized(&self, n: &NormalizedComponent) {
        std::fs::write(
            self.out(&format!("components/{}.postcard", n.id)),
            postcard::to_stdvec(n).unwrap(),
        )
        .unwrap();
    }

    fn ledger(&self, id: &str) -> LedgerFile {
        serde_json::from_str(&std::fs::read_to_string(self.out(&format!("ledger/{id}.json"))).unwrap())
            .unwrap()
    }

    /// The component with the most ledger steps.
    fn busiest(&self) -> String {
        self.ids()
            .into_iter()
            .max_by_key(|id| (self.ledger(id).steps.len(), id.clone()))
            .unwrap()
    }
}

fn fresh() -> Setup {
    let s = Setup::new();
    let st = s.pipeline().run(&RunOptions::default()).unwrap();
    assert_eq!(st.status_of("normalize"), Some(RunStatus::Ok), "{st:?}");
    s
}

fn assert_fails(s: &Setup, check: &str) {
    let f = s.failing();
    assert!(
        f.iter().any(|n| n.ends_with(&format!(":{check}")) || n == check),
        "{check} should fail; failing: {f:?}"
    );
}

#[test]
fn normalizes_and_verifies_the_toy() {
    let s = fresh();
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let mut rules = std::collections::BTreeMap::new();
    let mut reductions = 0;
    for id in s.ids() {
        for st in s.ledger(&id).steps {
            *rules.entry(st.rule).or_insert(0) += 1;
        }
        let n = s.normalized(&id);
        for x in &n.graph.nodes {
            if let cc_passes::components::algo::graph::Kind::Op { name, .. } = &x.label.kind
                && matches!(name.as_str(), "stablehlo.dot_general" | "stablehlo.reduce")
            {
                reductions += 1;
                assert!(x.omega.is_some(), "{name} without ω");
            }
        }
    }
    assert!(reductions > 0);
    assert!(rules.get("splat_fold").copied().unwrap_or(0) > 0, "{rules:?}");
    for r in [
        "dce",
        "cse",
        "splat_fold",
        "transpose_collapse",
        "reshape_collapse",
    ] {
        assert!(s.out(&format!("rules/{r}.smt2")).is_file());
        assert!(s.out(&format!("rules/{r}.result")).is_file());
    }
    assert!(s.env.build_dir().join(".normalize/views/normalize.md").is_file());
}

/// The negative control: float reassociation is not a valid rule; its obligation has a counterexample.
#[test]
fn an_unsound_rule_fails_its_proof() {
    let s = Setup::new();
    let cfg = s.config();
    let dir = s.env.root().join("smt");
    std::fs::create_dir_all(&dir).unwrap();
    let mut files = Vec::new();
    for (rule, _, text) in cc_passes::normalize::proofs::obligations() {
        if rule.starts_with("control_") {
            std::fs::write(dir.join(format!("{rule}.smt2")), text).unwrap();
            files.push(format!("{rule}.smt2"));
        }
    }
    let runners = Runners::from_config(&cfg);
    let c = &cfg.config.tools.smt;
    let (r, loc) = if c.runner == "host" {
        (runners.host, cc_smt::host_paths(&c.z3, &c.bitwuzla))
    } else {
        (runners.docker, Location::Image)
    };
    let caps = Caps {
        memory_gb: 2.0,
        cpus: 1.0,
        timeout_s: 120.0,
        pids_limit: 64,
    };
    let runner = BoundRunner::new(
        r,
        caps,
        ToolLog::new("smt-test", dir.join("logs")),
        CancelToken::new(),
    );
    for solver in Solver::ALL {
        let b = cc_smt::run_batch(&runner, &loc, solver, &dir, &files, 60).unwrap();
        assert!(
            matches!(&b.verdicts["control_float_reassociation.smt2"], Verdict::Sat { model } if !model.is_empty()),
            "{solver:?}: {b:?}"
        );
        assert_eq!(
            b.verdicts["control_int_reassociation.smt2"],
            Verdict::Unsat,
            "{solver:?}"
        );
        assert!(!b.version.is_empty());
    }
}

#[test]
fn a_wrong_proof_result_fails_its_rule_check() {
    let s = fresh();
    let p = s.out("rules/splat_fold.result");
    let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    v["verdict"] = serde_json::json!({"sat": {"model": "()"}});
    std::fs::write(&p, v.to_string()).unwrap();
    assert_fails(&s, "rules:splat_fold");
}

#[test]
fn an_edited_query_fails_its_rule_check() {
    let s = fresh();
    let p = s.out("rules/cse.smt2");
    let t = std::fs::read_to_string(&p)
        .unwrap()
        .replace("(not ", "(and true ");
    std::fs::write(&p, t).unwrap();
    assert_fails(&s, "rules:cse");
}

#[test]
fn an_edited_ledger_step_fails_replay() {
    let s = fresh();
    let id = s.busiest();
    let mut l = s.ledger(&id);
    assert!(!l.steps.is_empty());
    l.steps[0].matched[0] += 1;
    std::fs::write(
        s.out(&format!("ledger/{id}.json")),
        serde_json::to_string(&l).unwrap(),
    )
    .unwrap();
    assert_fails(&s, "replay");
}

#[test]
fn a_dropped_omega_fails_the_omega_check() {
    let s = fresh();
    let id = s
        .ids()
        .into_iter()
        .find(|id| s.normalized(id).graph.nodes.iter().any(|x| x.omega.is_some()))
        .unwrap();
    let mut n = s.normalized(&id);
    let x = n.graph.nodes.iter_mut().find(|x| x.omega.is_some()).unwrap();
    x.omega = None;
    s.write_normalized(&n);
    assert_fails(&s, "omega");
}

#[test]
fn a_leftover_pattern_fails_the_normal_form_check() {
    let s = fresh();
    let id = s.busiest();
    let mut n = s.normalized(&id);
    // An unused copy of an element-wise op: DCE matches it.
    let dead = n
        .graph
        .nodes
        .iter()
        .find(|x| {
            matches!(&x.label.kind, cc_passes::components::algo::graph::Kind::Op { name, .. }
                if name == "stablehlo.add" || name == "stablehlo.multiply" || name == "stablehlo.convert")
        })
        .cloned()
        .unwrap();
    let mut copy = dead;
    copy.origin = 1_000_000;
    n.graph.add_node(copy);
    s.write_normalized(&n);
    assert_fails(&s, "normal_form");
}

#[test]
fn a_merged_class_fails_the_classes_check() {
    let s = fresh();
    let id = s.busiest();
    let mut n = s.normalized(&id);
    // Two different stored dim classes on slots that the rules force equal.
    let mut classes: Vec<u32> = n
        .graph
        .nodes
        .iter()
        .flat_map(|x| x.label.results.iter().flat_map(|t| t.dims.iter()))
        .filter_map(|d| match d {
            ADim::Var(k) => Some(*k),
            ADim::One => None,
        })
        .collect();
    classes.sort_unstable();
    classes.dedup();
    assert!(classes.len() >= 2);
    let (a, b) = (classes[0], classes[1]);
    let mut done = false;
    for x in &mut n.graph.nodes {
        for t in &mut x.label.results {
            for d in &mut t.dims {
                if !done && *d == ADim::Var(a) {
                    *d = ADim::Var(b);
                    done = true;
                }
            }
        }
    }
    assert!(done);
    s.write_normalized(&n);
    assert_fails(&s, "classes");
}

#[test]
fn a_missing_ledger_fails_complete() {
    let s = fresh();
    let id = s.ids()[0].clone();
    std::fs::remove_file(s.out(&format!("ledger/{id}.json"))).unwrap();
    assert_fails(&s, "complete");
}

/// The solver pins in CI equal the image's pins in `config.toml`.
#[test]
fn ci_solver_pins_match_the_config() {
    let ci = std::fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).unwrap();
    let cfg: toml::Value =
        toml::from_str(&std::fs::read_to_string(repo_root().join("config.toml")).unwrap()).unwrap();
    let b = &cfg["tools"]["docker"]["build"];
    for k in ["z3_version", "z3_sha256", "bitwuzla_version", "bitwuzla_sha256"] {
        let v = b[k].as_str().unwrap();
        assert!(ci.contains(v), "{k} = {v} is not in ci.yml");
    }
}

fn orders() -> Orders {
    let c: NormalizeConfig = toml::from_str("[order]\nfloat = \"seq\"\ncomplex = \"seq\"\n").unwrap();
    c.order.parse().unwrap()
}

/// `g` with its nodes renumbered by `perm` (`perm[old] = new`).
fn renumber(g: &Graph<ANode>, perm: &[u32]) -> Graph<ANode> {
    let mut nodes = vec![None; g.node_count()];
    for (i, n) in g.nodes.iter().enumerate() {
        nodes[perm[i] as usize] = Some(n.clone());
    }
    let mut out: Graph<ANode> = Graph::default();
    for n in nodes {
        out.add_node(n.unwrap());
    }
    for (u, o) in g.out.iter().enumerate() {
        for (v, ends) in o {
            for e in ends {
                out.add_edge(perm[u], perm[*v as usize], *e);
            }
        }
    }
    out
}

/// The normalized graph, independent of node numbers: sizes and the multiset of labels (the order in
/// which a chain of rewrites fires may depend on numbering, the result may not).
fn shape(g: &Graph<ANode>) -> (usize, usize, Vec<String>) {
    let n = normalize(g, &orders()).unwrap();
    let mut labels: Vec<String> = n
        .graph
        .nodes
        .iter()
        .map(|x| format!("{:?} {:?} {}", x.label.kind, x.omega, x.splats.len()))
        .collect();
    labels.sort();
    (n.graph.node_count(), n.graph.end_count(), labels)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 8, ..ProptestConfig::default() })]

    #[test]
    fn renumbering_nodes_keeps_the_normalization(seed in any::<u64>()) {
        thread_local!(static TOY: Vec<StoredComponent> = {
            let s = fresh();
            s.ids().iter().map(|id| postcard::from_bytes(&std::fs::read(s.input(&format!("components/{id}.postcard"))).unwrap()).unwrap()).collect()
        });
        TOY.with(|comps| {
            for c in comps {
                let n = c.graph.node_count();
                let mut perm: Vec<u32> = (0..n as u32).collect();
                let mut x = seed | 1;
                for i in (1..n).rev() {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    perm.swap(i, (x % (i as u64 + 1)) as usize);
                }
                prop_assert_eq!(shape(&c.graph), shape(&renumber(&c.graph, &perm)), "{}", c.id);
            }
            Ok(())
        })?;
    }
}
