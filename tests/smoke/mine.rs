//! `mine` on synthetic datapaths (multiply-add chains as in APEX's example, a reduction, a non-convex
//! diamond, compares with different directions and a template class with stages), with a brute-force
//! miner standing in for GraMi (no Docker in smoke): every check green, the run reproducible, the
//! re-matcher equal to the brute force on random graphs, and every check of its `verify` failing on a
//! mutated artifact.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use cc_core::runner::{Caps, Invocation, Runners, ToolLog, ToolOutcome, ToolRunner};
use cc_core::{ArtifactKind, CancelToken, CheckStatus, CoreError, PassSection, VerifyReport};
use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};
use cc_passes::datapaths::DATAPATHS;
use cc_passes::iteration_space::ITERATION_SPACE;
use cc_passes::kernel_classes::KERNEL_CLASSES;
use cc_passes::mine::algo::lg::LgGraph;
use cc_passes::mine::algo::rematch::{UnitCtx, rematch};
use cc_passes::mine::algo::sweep::canonical;
use cc_passes::mine::json::{Labels, PatternFile, Summary, Unit};
use cc_pipeline::{Pass, PassContext, PassError, Pipeline, RunOptions, RunStatus};
use serde_json::{Value, json};
use tests::Env;
use tests::brute::{self, G};

const CONFIG: &str = r#"
[pipeline]
targets = ["mine"]
[resources]
memory_budget_gb = 32
threads_total = 8
[passes.synthetic.resources]
threads = 1
memory_gb = 1
timeout_min = 1
[passes.mine]
tau_start = 4
tau_min = 1
max_vertices = 4
heap_gb = 1
[passes.mine.resources]
threads = 2
memory_gb = 8
timeout_min = 10
"#;

/// A datapath vertex; a source is `v<id>`, `acc`, `result` or an input name.
fn vertex(id: u32, op: &str, family: &str, args: &[(&str, &str)], stage: Value, attrs: Value) -> Value {
    let args: Vec<Value> = args
        .iter()
        .map(|(src, port)| {
            let src = if let Some(v) = src.strip_prefix('v').and_then(|v| v.parse::<u32>().ok()) {
                json!({"vertex": {"v": v, "res": 0}})
            } else if *src == "acc" {
                json!({"acc": {"index": 0}})
            } else if *src == "result" {
                json!({"result": {"index": 0}})
            } else {
                json!({"input": {"name": src}})
            };
            json!({"src": src, "port": port})
        })
        .collect();
    json!({
        "id": id, "label": format!("{op}@{family}"), "op": op, "family": family, "width": "w0",
        "attrs": attrs, "stage": stage, "args": args, "from": {"node": id, "access": null}
    })
}

fn datapath(id: &str, route: &str, class: &str, vertices: Vec<Value>) -> Value {
    json!({
        "schema": "cc.datapaths.datapath/1", "id": id, "route": route, "class": class, "core": "",
        "widths": {}, "vertices": vertices, "inputs": [], "outputs": [], "accumulator": null, "tuples": []
    })
}

/// `(((x0·y0 + x1·y1) + x2·y2) + …) + c` with `taps` products; every op commutative.
fn chain(id: &str, taps: u32) -> Value {
    let m = json!("map");
    let mut v = Vec::new();
    for j in 0..taps {
        let (x, y) = (format!("x{j}"), format!("y{j}"));
        v.push(vertex(j, "multiply", "sint", &[(&x, "*"), (&y, "*")], m.clone(), json!({})));
    }
    let c = taps;
    v.push(vertex(c, "const", "sint", &[], m.clone(), json!({"column": "n0.value"})));
    let mut prev = "v0".to_string();
    for j in 1..taps {
        let id = taps + j;
        let p = format!("v{j}");
        v.push(vertex(id, "add", "sint", &[(&prev, "*"), (&p, "*")], m.clone(), json!({})));
        prev = format!("v{id}");
    }
    let cs = format!("v{c}");
    v.push(vertex(2 * taps, "add", "sint", &[(&prev, "*"), (&cs, "*")], m.clone(), json!({})));
    datapath(id, "datapath", "map", v)
}

/// `a = x·y; b = -a; c = a + b`, then two compares of `c` (LT and EQ): `multiply -> add` occurs only
/// non-convexly (through `negate`).
fn diamond(id: &str) -> Value {
    let m = json!("map");
    let cmp = |d: &str| json!({"comparison_direction": d, "compare_type": "SIGNED"});
    datapath(
        id,
        "datapath",
        "map",
        vec![
            vertex(0, "multiply", "sint", &[("x", "*"), ("y", "*")], m.clone(), json!({})),
            vertex(1, "negate", "sint", &[("v0", "p0")], m.clone(), json!({})),
            vertex(2, "add", "sint", &[("v0", "*"), ("v1", "*")], m.clone(), json!({})),
            vertex(3, "compare", "sint", &[("v2", "p0"), ("z", "p1")], m.clone(), cmp("LT")),
            vertex(4, "compare", "sint", &[("v2", "p0"), ("z", "p1")], m.clone(), cmp("EQ")),
        ],
    )
}

/// A multiply-accumulate reduction with a convert after it.
fn mac(id: &str) -> Value {
    let c = json!("core");
    datapath(
        id,
        "datapath",
        "reduce",
        vec![
            vertex(0, "multiply", "float", &[("lhs", "*"), ("rhs", "*")], c.clone(), json!({})),
            vertex(1, "add", "float", &[("acc", "*"), ("v0", "*")], c.clone(), json!({})),
            vertex(2, "convert", "float", &[("v1", "p0")], json!("epilogue"), json!({})),
        ],
    )
}

/// A template class: prologue 0 `convert -> multiply(·, s)`, epilogue `convert -> add -> maximum`.
fn template(id: &str) -> Value {
    let p0 = json!({"prologue": {"port": 0}});
    let e = json!("epilogue");
    datapath(
        id,
        "template",
        "contraction",
        vec![
            vertex(0, "convert", "float", &[("in0", "p0")], p0.clone(), json!({})),
            vertex(1, "multiply", "float", &[("v0", "*"), ("s", "*")], p0.clone(), json!({})),
            vertex(2, "convert", "float", &[("result", "p0")], e.clone(), json!({})),
            vertex(3, "add", "float", &[("v2", "*"), ("bias", "*")], e.clone(), json!({})),
            vertex(4, "maximum", "float", &[("v3", "*"), ("zero", "*")], e.clone(), json!({})),
        ],
    )
}

fn skeleton(id: &str) -> Value {
    let op = |name: &str, dir: &str, axes: Value| {
        json!({"name": name, "direction": dir, "core": true, "node": 0, "res": 0, "axes": axes, "elem": 0,
               "access": [], "reuse": []})
    };
    json!({
        "schema": "cc.iteration_space.skeleton/1", "id": id, "class": "contraction", "kind": "contraction",
        "core": "dot_general", "core_node": 0, "affine": true, "dims": [], "omega": null,
        "operands": [
            op("lhs", "in", json!([0, 2])),
            op("rhs", "in", json!([2, 1])),
            op("result0", "out", json!([0, 1]))
        ],
        "accesses": [], "stages": {"prologue": {}, "epilogue": []}, "proof": ""
    })
}

fn instances(id: &str, weighted: u64) -> Value {
    let row = json!({
        "model": "m", "phase": "p", "instance": "i", "component": "c", "kernel": "k", "member": 0,
        "extents": [4, 5, 6], "sizes": ["4", "5", "6"], "elems": ["i8"], "values": [],
        "workload": weighted / 2, "trip_count": 2, "weighted": weighted
    });
    json!({"schema": "cc.kernel_classes.instances/1", "id": id, "omega": null, "rows": [row]})
}

/// Writes the synthetic `datapaths`, `iteration_space` and `kernel_classes` artifacts.
#[derive(Debug)]
struct Synthetic;

impl Pass for Synthetic {
    fn name(&self) -> &'static str {
        "synthetic"
    }
    fn version(&self) -> u32 {
        1
    }
    fn requires(&self) -> &[ArtifactKind] {
        &[]
    }
    fn provides(&self) -> &[ArtifactKind] {
        &[DATAPATHS, ITERATION_SPACE, KERNEL_CLASSES]
    }
    fn model_dependent(&self) -> bool {
        false
    }
    fn check_config(&self, _: &PassSection) -> Result<(), CoreError> {
        Ok(())
    }
    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let dps = [
            chain("z01", 4),
            chain("z02", 4),
            chain("z03", 3),
            chain("z04", 5),
            diamond("z05"),
            diamond("z06"),
            mac("z07"),
            mac("z08"),
            template("z09"),
            template("z10"),
        ];
        let d = ctx.output(DATAPATHS)?;
        let k = ctx.output(KERNEL_CLASSES)?;
        let s = ctx.output(ITERATION_SPACE)?;
        for (i, dp) in dps.iter().enumerate() {
            let id = dp["id"].as_str().unwrap();
            d.write_json(&format!("datapaths/{id}.json"), dp)?;
            k.write_json(&format!("instances/{id}.json"), &instances(id, 100 * (i as u64 + 1)))?;
            if dp["route"] == "template" {
                s.write_json(&format!("skeletons/{id}.json"), &skeleton(id))?;
            }
        }
        Ok(())
    }
    fn verify(&self, _: &PassContext) -> Result<VerifyReport, PassError> {
        Ok(VerifyReport::new("synthetic"))
    }
}

fn outcome(inv: &Invocation, log: &ToolLog, stdout: &str, stderr: &str) -> cc_core::Result<ToolOutcome> {
    let files = log.next(&inv.tool)?;
    std::fs::write(&files.stdout, stdout).unwrap();
    std::fs::write(&files.stderr, stderr).unwrap();
    Ok(ToolOutcome {
        runner: "fake".into(),
        tool: inv.tool.clone(),
        argv: vec![inv.program.clone()],
        exit_code: Some(0),
        signal: None,
        killed: None,
        duration_ms: 0,
        peak_rss_bytes: None,
        stdout: files.stdout,
        stderr: files.stderr,
    })
}

/// GraMi answered by the brute-force miner (on `graph.lg` in the job directory, writing `Output.txt`
/// there), its version probe by a fixed line.
#[derive(Debug)]
struct FakeGrami;

impl ToolRunner for FakeGrami {
    fn name(&self) -> &'static str {
        "fake-grami"
    }
    fn run(&self, inv: &Invocation, _: &Caps, log: &ToolLog, _: &CancelToken) -> cc_core::Result<ToolOutcome> {
        match inv.tool.as_str() {
            "java-version" => outcome(inv, log, "", "openjdk version \"17.0.0\" (fake)\n"),
            "grami" => {
                let dir = &inv.mounts.iter().find(|m| m.dst == "/work").unwrap().src;
                let cmd = &inv.args[1];
                assert!(
                    cmd.contains("limitedtime=false") && cmd.contains("$GRAMI_DIRECTED_CP"),
                    "{cmd}"
                );
                let at = cmd.find("freq=").unwrap() + 5;
                let tau: u64 = cmd[at..].split(' ').next().unwrap().parse().unwrap();
                let g = brute::parse_lg(&std::fs::read_to_string(dir.join("graph.lg")).unwrap());
                // Up to 5 vertices: the pass keeps 4, so it must drop GraMi's larger patterns.
                std::fs::write(dir.join("Output.txt"), brute::output(&g, tau, 5)).unwrap();
                outcome(inv, log, "", "")
            }
            t => panic!("unexpected tool {t}"),
        }
    }
}

struct Setup {
    env: Env,
}

impl Setup {
    fn new() -> Self {
        Self { env: Env::new() }
    }

    fn pipeline_text(&self, text: &str) -> Pipeline {
        let cfg = self.env.config(text, &[]);
        let mut runners = Runners::from_config(&cfg);
        let fake: Arc<dyn ToolRunner> = Arc::new(FakeGrami);
        runners.docker = fake.clone();
        runners.host = fake;
        Pipeline::new(vec![Box::new(Synthetic), Box::new(cc_passes::mine::Mine)], cfg)
            .unwrap()
            .with_runners(runners)
    }

    fn pipeline(&self) -> Pipeline {
        self.pipeline_text(CONFIG)
    }

    fn out(&self, rel: &str) -> PathBuf {
        self.env.build_dir().join(".mine/out/mine").join(rel)
    }

    fn failing(&self) -> Vec<String> {
        let r = self.pipeline().verify_pass("mine").unwrap();
        r.checks
            .into_iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name)
            .collect()
    }

    fn expect(&self, check: &str) {
        let f = self.failing();
        assert!(f.iter().any(|c| c == check), "{check} not in {f:?}");
    }

    fn edit(&self, path: PathBuf, f: impl FnOnce(&mut Value)) {
        let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        f(&mut v);
        std::fs::write(&path, v.to_string()).unwrap();
    }

    /// A family's pattern files, sorted by path.
    fn patterns(&self, family: &str) -> Vec<(PathBuf, PatternFile)> {
        let Ok(rd) = std::fs::read_dir(self.out(&format!("patterns/{family}"))) else {
            return Vec::new();
        };
        let mut v: Vec<(PathBuf, PatternFile)> = rd
            .map(|e| {
                let p = e.unwrap().path();
                let f = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                (p, f)
            })
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// A pattern (family, path, file) satisfying `f`.
    fn find(&self, f: impl Fn(&PatternFile) -> bool) -> (String, PathBuf, PatternFile) {
        for fam in ["map", "reduce", "contraction", "stage"] {
            for (p, x) in self.patterns(fam) {
                if f(&x) {
                    return (fam.into(), p, x);
                }
            }
        }
        panic!("no such pattern")
    }
}

fn fresh() -> Setup {
    let s = Setup::new();
    let st = s.pipeline().run(&RunOptions::default()).unwrap();
    assert_eq!(st.status_of("mine"), Some(RunStatus::Ok), "{st:?}");
    s
}

#[test]
fn mines_and_verifies_synthetic_datapaths() {
    let s = fresh();
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let t: Summary = serde_json::from_str(&std::fs::read_to_string(s.out("summary.json")).unwrap()).unwrap();
    assert_eq!(t.families.len(), 4);
    for f in ["map", "reduce", "stage"] {
        assert!(t.families[f].kept > 0, "{f}: {t:?}");
    }
    assert_eq!(t.families["contraction"].units, 0);
    // The stage family: prologue 0 reads lhs (axes 0, 2: 4·6 = 24 elements), the epilogue writes
    // result0 (4·5 = 20), each times the trip count 2.
    let map: Value = serde_json::from_str(&std::fs::read_to_string(s.out("lg/stage.map.json")).unwrap()).unwrap();
    let w: Vec<(String, u64)> = map["units"]
        .as_array()
        .unwrap()
        .iter()
        .map(|u| (u["id"].as_str().unwrap().to_string(), u["workload"].as_u64().unwrap()))
        .collect();
    assert_eq!(
        w,
        vec![
            ("z09/epilogue".to_string(), 40),
            ("z09/prologue0".to_string(), 48),
            ("z10/epilogue".to_string(), 40),
            ("z10/prologue0".to_string(), 48)
        ]
    );
    let r = s.pipeline().verify_pass("mine").unwrap();
    let names: Vec<&str> = r.checks.iter().map(|c| c.name.as_str()).collect();
    for k in [
        "labels",
        "map:lg",
        "map:grami",
        "map:patterns",
        "map:selection",
        "map:sweep",
        "overlap",
        "summary",
    ] {
        assert!(names.contains(&k), "{k} not in {names:?}");
    }
    // GraMi's list was re-read from the cache.
    let g = r.checks.iter().find(|c| c.name == "map:grami").unwrap();
    assert!(g.evidence.counts["reread"] > 0, "{g:?}");
    assert!(s.env.build_dir().join(".mine/views/mine.md").is_file());
    // Labels keep the comparison direction; constants are not vertices.
    let labels: Labels = serde_json::from_str(&std::fs::read_to_string(s.out("labels.json")).unwrap()).unwrap();
    assert!(labels.nodes.contains(&"compare@sint[LT,SIGNED]".to_string()));
    assert!(labels.nodes.contains(&"compare@sint[EQ,SIGNED]".to_string()));
    assert!(!labels.nodes.iter().any(|l| l.starts_with("const")));
    // GraMi's 5-vertex patterns are dropped; every kept one is within 4 and frequent.
    let grami: Value = serde_json::from_str(&std::fs::read_to_string(s.out("grami/map.json")).unwrap()).unwrap();
    assert!(grami["dropped"].as_u64().unwrap() > 0, "{grami}");
    for f in ["map", "reduce", "stage"] {
        for (_, p) in s.patterns(f) {
            assert!(p.labels.len() <= 4 && p.mni >= p.tau, "{}", p.id);
        }
    }
    // `multiply -> add` in the diamonds is not convex; in the chains it is.
    let (_, _, p) = s.find(|p| p.code == "add@sint,multiply@sint;1>0:0*");
    assert_eq!(p.non_convex, 2, "{p:?}");
    assert!(p.occurrences.iter().all(|o| o.unit != "z05" && o.unit != "z06"));
    // A constant operand is an input of the occurrence, fed by `const`.
    let (_, _, p) = s.find(|p| p.code == "add@sint,add@sint;0>1:0*");
    assert!(
        p.occurrences
            .iter()
            .flat_map(|o| &o.inputs)
            .any(|i| i.feed == cc_passes::mine::json::Feed::Const)
    );
}

/// The APEX example: in a 4-tap chain, `multiply -> add -> add` has four occurrences, MIS 2.
#[test]
fn mni_and_mis_differ_on_the_chain() {
    let s = fresh();
    let (_, _, p) = s.find(|p| p.code == "add@sint,add@sint,multiply@sint;0>1:0*,2>0:0*");
    let z01 = p.mis.iter().find(|m| m.unit == "z01").unwrap();
    assert_eq!(z01.occurrences.len(), 4);
    assert_eq!(z01.size, 2);
    assert!(z01.optimal);
}

/// Same inputs, same τ: the same artifacts.
#[test]
fn mining_is_reproducible() {
    let s = fresh();
    let read = |s: &Setup| -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        for f in ["map", "reduce", "contraction", "stage"] {
            for (p, _) in s.patterns(f) {
                m.insert(p.display().to_string(), std::fs::read_to_string(&p).unwrap());
            }
            for k in ["selection", "sweep", "grami"] {
                let p = s.out(&format!("{k}/{f}.json"));
                m.insert(p.display().to_string(), std::fs::read_to_string(&p).unwrap());
            }
        }
        m
    };
    let a = read(&s);
    let st = s
        .pipeline()
        .run(&RunOptions {
            from: Some("mine".into()),
            ..RunOptions::default()
        })
        .unwrap();
    assert_eq!(st.status_of("mine"), Some(RunStatus::Ok));
    assert_eq!(a, read(&s));
}

/// Every MIS strategy gives a verified result.
#[test]
fn every_mis_strategy_verifies() {
    for alg in ["branch_and_bound", "greedy_min_degree"] {
        let s = Setup::new();
        let p = s.pipeline_text(&format!("{CONFIG}[passes.mine.mis]\nalgorithm = \"{alg}\"\n"));
        let st = p.run(&RunOptions::default()).unwrap();
        assert_eq!(st.status_of("mine"), Some(RunStatus::Ok), "{alg}: {st:?}");
        let r = p.verify_pass("mine").unwrap();
        let bad: Vec<_> = r.checks.iter().filter(|c| c.status == CheckStatus::Fail).collect();
        assert!(bad.is_empty(), "{alg}: {bad:?}");
    }
}

/// The re-matcher's MNI equals the brute force for every pattern of random graphs, so its frequent
/// set at every τ is the brute force's.
#[test]
fn rematching_equals_brute_force_on_random_graphs() {
    let labels = Labels {
        schema: String::new(),
        nodes: (0..3).map(|l| format!("l{l}")).collect(),
        edges: (0..2)
            .map(|l| {
                vec![EdgeEnd {
                    res: 0,
                    port: PortLabel::Port(l),
                }]
            })
            .collect(),
    };
    for seed in 0..12u64 {
        let g = brute::random_dag(seed, 10, 3, 2, 1, 4);
        let mut host: Graph<u32> = Graph::default();
        for &l in &g.nodes {
            host.add_node(l);
        }
        for &(a, b, l) in &g.edges {
            host.add_edge(
                a,
                b,
                EdgeEnd {
                    res: 0,
                    port: PortLabel::Port(l),
                },
            );
        }
        let unit = Unit {
            id: "u".into(),
            class: "u".into(),
            stage: None,
            workload: 1,
            first: 0,
            vertices: (0..g.nodes.len() as u32).collect(),
        };
        let units = vec![UnitCtx::bare(unit, host).unwrap()];
        for (code, (p, m)) in brute::patterns(&g, 4) {
            let c = canonical(
                &LgGraph {
                    nodes: p.nodes.clone(),
                    edges: p.edges.clone(),
                },
                &labels,
            )
            .unwrap();
            let r = rematch(&c.graph, &units, 1 << 30).unwrap();
            assert_eq!(r.mni, m, "seed {seed}: {code}");
            assert_eq!(r.embeddings as usize, brute::embeddings(&p, &g).len(), "seed {seed}: {code}");
        }
    }
}

/// The oracle itself on the APEX chain: `mul -> add -> add` has MNI 3 (the second add's images).
#[test]
fn brute_force_counts_the_apex_chain() {
    // mul = 0, add = 1; p0..p3 = 0..3, s1, s2, s3, y = 4..7.
    let g = G {
        nodes: vec![0, 0, 0, 0, 1, 1, 1, 1],
        edges: vec![(0, 4, 0), (1, 4, 0), (4, 5, 0), (2, 5, 0), (5, 6, 0), (3, 6, 0), (6, 7, 0)],
    };
    let p = G {
        nodes: vec![0, 1, 1],
        edges: vec![(0, 1, 0), (1, 2, 0)],
    };
    assert_eq!(brute::mni(&p, &g), 3);
    assert_eq!(brute::embeddings(&p, &g).len(), 4);
}

#[test]
fn an_edited_occurrence_fails() {
    let s = fresh();
    let (fam, path, _) = s.find(|p| !p.occurrences.is_empty());
    s.edit(path, |v| {
        let m = v["occurrences"][0]["map"].as_array_mut().unwrap();
        m.swap(0, 1);
    });
    s.expect(&format!("{fam}:patterns"));
}

#[test]
fn a_dependent_mis_fails() {
    let s = fresh();
    let (fam, path, p) = s.find(|p| p.mis.iter().any(|m| m.occurrences.len() > m.chosen.len()));
    let (ui, m) = p
        .mis
        .iter()
        .enumerate()
        .find(|(_, m)| m.occurrences.len() > m.chosen.len())
        .unwrap();
    let extra = m.occurrences.iter().find(|o| !m.chosen.contains(o)).copied().unwrap();
    s.edit(path, |v| {
        let e = &mut v["mis"][ui];
        e["chosen"].as_array_mut().unwrap().push(json!(extra));
        let n = e["chosen"].as_array().unwrap().len();
        e["size"] = json!(n);
        e["bound"] = json!(n);
    });
    s.expect(&format!("{fam}:patterns"));
}

#[test]
fn a_non_convex_occurrence_or_wrong_mni_fails() {
    let s = fresh();
    let (fam, path, _) = s.find(|p| p.non_convex > 0);
    s.edit(path, |v| {
        let n = v["non_convex"].as_u64().unwrap();
        v["non_convex"] = json!(n - 1);
    });
    s.expect(&format!("{fam}:patterns"));
    let s = fresh();
    let (fam, path, p) = s.find(|_| true);
    s.edit(path, |v| v["mni"] = json!(p.tau.saturating_sub(1)));
    s.expect(&format!("{fam}:patterns"));
}

#[test]
fn an_edited_score_or_selection_fails() {
    let s = fresh();
    let (fam, path, _) = s.find(|p| p.score > 0);
    s.edit(path, |v| v["score"] = json!(1u64 << 50));
    s.expect(&format!("{fam}:patterns"));
    s.expect(&format!("{fam}:selection"));
    let s = fresh();
    let (fam, _, _) = s.find(|p| p.score > 0);
    s.edit(s.out(&format!("selection/{fam}.json")), |v| {
        v["patterns"].as_array_mut().unwrap().pop();
    });
    s.expect(&format!("{fam}:selection"));
}

#[test]
fn an_edited_graph_label_sweep_or_summary_fails() {
    let s = fresh();
    let p = s.out("lg/map.lg");
    let t = std::fs::read_to_string(&p).unwrap().replacen("\nv 0 ", "\nv 0 9", 1);
    std::fs::write(&p, t).unwrap();
    s.expect("map:lg");

    let s = fresh();
    s.edit(s.out("labels.json"), |v| {
        v["nodes"].as_array_mut().unwrap().push(json!("zzz@float"));
    });
    s.expect("labels");

    let s = fresh();
    s.edit(s.out("sweep/map.json"), |v| v["stopped_by"] = json!("budget"));
    s.expect("map:sweep");

    let s = fresh();
    s.edit(s.out("grami/map.json"), |v| {
        v["patterns"].as_array_mut().unwrap().pop();
    });
    s.expect("map:grami");

    let s = fresh();
    s.edit(s.out("summary.json"), |v| v["families"]["map"]["kept"] = json!(12345));
    s.expect("summary");
}
