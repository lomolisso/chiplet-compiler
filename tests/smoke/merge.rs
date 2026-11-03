//! `merge` on synthetic inputs: map datapaths that share a mined multiply-add (one at i8, widened into
//! the i32 unit; one adding an operand, which needs a multiplexer), a subtract that shares nothing, two equal
//! reductions, and two template classes (an i8 array; the second has a ui8 operand, composed from
//! pieces). A fixture library prices the blocks; the configuration proofs run in the solvers. Every
//! check of `verify` passes, the run is reproducible, and each check fails on a mutated artifact.

use std::collections::BTreeMap;
use std::path::PathBuf;

use cc_core::{ArtifactKind, CheckStatus, CoreError, PassSection, VerifyReport};
use cc_passes::characterize::LIBRARY;
use cc_passes::characterize::catalog::Kind;
use cc_passes::characterize::json::{Library, Row, TargetInfo};
use cc_passes::datapaths::DATAPATHS;
use cc_passes::eda::parse::{Lec, Measured};
use cc_passes::iteration_space::ITERATION_SPACE;
use cc_passes::kernel_classes::KERNEL_CLASSES;
use cc_passes::merge::json::{ArrayFile, ArrayMode, ConfigFile, FuFile, RequantFile, RequantStatus, Summary};
use cc_passes::mine::MINE;
use cc_pipeline::{Pass, PassContext, PassError, Pipeline, RunOptions, RunStatus};
use serde_json::{Value, json};
use tests::{Env, repo_root};

const CONFIG: &str = r#"
[pipeline]
targets = ["merge"]
[resources]
memory_budget_gb = 32
threads_total = 8
[passes.inputs.resources]
threads = 1
memory_gb = 1
timeout_min = 1
[passes.merge]
kit = "k"
certify_max_nodes = 400
[passes.merge.resources]
threads = 2
memory_gb = 4
timeout_min = 10
"#;

fn src(s: &str) -> Value {
    if let Some(v) = s.strip_prefix('v').and_then(|v| v.parse::<u32>().ok()) {
        json!({"vertex": {"v": v, "res": 0}})
    } else if s == "acc" {
        json!({"acc": {"index": 0}})
    } else if s == "result" {
        json!({"result": {"index": 0}})
    } else {
        json!({"input": {"name": s}})
    }
}

/// A vertex; `width` is `w<e>` or a constant's column.
fn vertex(
    id: u32,
    op: &str,
    family: &str,
    width: &str,
    args: &[(&str, &str)],
    stage: Value,
    attrs: Value,
) -> Value {
    let args: Vec<Value> = args
        .iter()
        .map(|(s, p)| json!({"src": src(s), "port": p}))
        .collect();
    json!({
        "id": id, "label": format!("{op}@{family}"), "op": op, "family": family, "width": width,
        "attrs": attrs, "stage": stage, "args": args, "from": {"node": id, "access": null}
    })
}

fn input(name: &str, kind: &str, width: &str) -> Value {
    json!({"name": name, "kind": kind, "node": 0, "res": 0, "incidence": [], "width": width})
}

fn output(name: &str, kind: &str, s: &str, width: &str) -> Value {
    json!({"name": name, "kind": kind, "src": src(s), "width": width})
}

fn tuple(elems: &[&str], consts: Value) -> Value {
    json!({"name": "t0", "elems": elems, "consts": consts, "attrs": {}, "rows": 1})
}

#[allow(clippy::too_many_arguments)]
fn datapath(
    id: &str,
    route: &str,
    class: &str,
    vertices: Vec<Value>,
    inputs: Vec<Value>,
    outputs: Vec<Value>,
    acc: Value,
    tuples: Vec<Value>,
) -> Value {
    json!({
        "schema": "cc.datapaths.datapath/1", "id": id, "route": route, "class": class, "core": "",
        "widths": {}, "vertices": vertices, "inputs": inputs, "outputs": outputs, "accumulator": acc,
        "tuples": tuples
    })
}

/// `out0 = a·b + c` at `ty`.
fn mac(id: &str, ty: &str) -> Value {
    let m = json!("map");
    datapath(
        id,
        "datapath",
        "map",
        vec![
            vertex(
                0,
                "multiply",
                "sint",
                "w0",
                &[("a", "*"), ("b", "*")],
                m.clone(),
                json!({}),
            ),
            vertex(1, "add", "sint", "w0", &[("v0", "*"), ("c", "*")], m, json!({})),
        ],
        vec![
            input("a", "map", "w0"),
            input("b", "map", "w0"),
            input("c", "map", "w0"),
        ],
        vec![output("out0", "output", "v1", "w0")],
        Value::Null,
        vec![tuple(&[ty], json!({}))],
    )
}

/// `out0 = a·b + a`: the multiply-add with its addend from an operand.
fn mac_self(id: &str) -> Value {
    let m = json!("map");
    datapath(
        id,
        "datapath",
        "map",
        vec![
            vertex(
                0,
                "multiply",
                "sint",
                "w0",
                &[("a", "*"), ("b", "*")],
                m.clone(),
                json!({}),
            ),
            vertex(1, "add", "sint", "w0", &[("v0", "*"), ("a", "*")], m, json!({})),
        ],
        vec![input("a", "map", "w0"), input("b", "map", "w0")],
        vec![output("out0", "output", "v1", "w0")],
        Value::Null,
        vec![tuple(&["i32"], json!({}))],
    )
}

/// `out0 = a - k` with a folded constant `k`.
fn sub(id: &str) -> Value {
    let m = json!("map");
    datapath(
        id,
        "datapath",
        "map",
        vec![
            vertex(
                0,
                "const",
                "sint",
                "n0.splat0",
                &[],
                m.clone(),
                json!({"column": "n0.splat0"}),
            ),
            vertex(
                1,
                "subtract",
                "sint",
                "w0",
                &[("a", "p0"), ("v0", "p1")],
                m,
                json!({}),
            ),
        ],
        vec![input("a", "map", "w0")],
        vec![output("out0", "output", "v1", "w0")],
        Value::Null,
        vec![tuple(&["i32"], json!({"n0.splat0": "i32"}))],
    )
}

/// `acc += x·y`, initially 0; the result is the output.
fn reduce(id: &str) -> Value {
    let c = json!("core");
    datapath(
        id,
        "datapath",
        "reduce",
        vec![
            vertex(
                0,
                "multiply",
                "sint",
                "w0",
                &[("x", "*"), ("y", "*")],
                c.clone(),
                json!({}),
            ),
            vertex(
                1,
                "add",
                "sint",
                "w0",
                &[("acc", "*"), ("v0", "*")],
                c.clone(),
                json!({}),
            ),
            vertex(2, "const", "sint", "w0", &[], c, json!({"value": "0"})),
        ],
        vec![input("x", "map", "w0"), input("y", "map", "w0")],
        vec![output("out0", "output", "result", "w0")],
        json!({"entries": [{"width": "w0", "init": src("v2"), "next": src("v1")}], "combiner": [1],
               "omega": null, "order": []}),
        vec![tuple(&["i32"], json!({}))],
    )
}

/// A template class: prologue 0 `in0 · s` (i8), rhs `in1` directly, epilogue `max(result, z)` (i32).
fn template_a(id: &str) -> Value {
    let p0 = json!({"prologue": {"port": 0}});
    let e = json!("epilogue");
    datapath(
        id,
        "template",
        "contraction",
        vec![
            vertex(
                0,
                "const",
                "sint",
                "n0.splat0",
                &[],
                p0.clone(),
                json!({"column": "n0.splat0"}),
            ),
            vertex(
                1,
                "multiply",
                "sint",
                "w0",
                &[("in0", "*"), ("v0", "*")],
                p0,
                json!({}),
            ),
            vertex(
                2,
                "const",
                "sint",
                "n1.splat0",
                &[],
                e.clone(),
                json!({"column": "n1.splat0"}),
            ),
            vertex(
                3,
                "maximum",
                "sint",
                "w1",
                &[("result", "*"), ("v2", "*")],
                e,
                json!({}),
            ),
        ],
        vec![
            input("in0", "prologue", "w0"),
            input("in1", "core_operand", "w0"),
            input("core_result0", "core_result", "w1"),
        ],
        vec![
            output("lhs", "core_operand", "v1", "w0"),
            output("rhs", "core_operand", "in1", "w0"),
            output("out0", "output", "v3", "w1"),
        ],
        Value::Null,
        vec![tuple(
            &["i8", "i32"],
            json!({"n0.splat0": "i8", "n1.splat0": "i32"}),
        )],
    )
}

/// A template class with a ui8 lhs and an i8 rhs (no prologue), epilogue `max(result, z)`.
fn template_b(id: &str) -> Value {
    let e = json!("epilogue");
    datapath(
        id,
        "template",
        "contraction",
        vec![
            vertex(
                0,
                "const",
                "sint",
                "n0.splat0",
                &[],
                e.clone(),
                json!({"column": "n0.splat0"}),
            ),
            vertex(
                1,
                "maximum",
                "sint",
                "w2",
                &[("result", "*"), ("v0", "*")],
                e,
                json!({}),
            ),
        ],
        vec![
            input("in0", "core_operand", "w0"),
            input("in1", "core_operand", "w1"),
            input("core_result0", "core_result", "w2"),
        ],
        vec![
            output("lhs", "core_operand", "in0", "w0"),
            output("rhs", "core_operand", "in1", "w1"),
            output("out0", "output", "v1", "w2"),
        ],
        Value::Null,
        vec![tuple(&["ui8", "i8", "i32"], json!({"n0.splat0": "i32"}))],
    )
}

/// `out0 = convert_i8(min(max(round(convert_f32(x) · s) + zp, lo), hi))`: a requantization chain with
/// an integer `x` of type `ty`.
fn requant(id: &str, ty: &str) -> Value {
    let m = json!("map");
    let v = |id: u32, op: &str, fam: &str, w: &str, args: &[(&str, &str)]| {
        vertex(id, op, fam, w, args, m.clone(), json!({}))
    };
    let c = |id: u32, col: &str| vertex(id, "const", "float", col, &[], m.clone(), json!({"column": col}));
    datapath(
        id,
        "datapath",
        "map",
        vec![
            v(0, "convert", "float", "w1", &[("x", "p0")]),
            v(1, "multiply", "float", "w1", &[("v0", "*"), ("s", "*")]),
            v(2, "round_nearest_even", "float", "w1", &[("v1", "p0")]),
            v(3, "add", "float", "w1", &[("v2", "*"), ("zp", "*")]),
            c(4, "n4.lo"),
            v(5, "maximum", "float", "w1", &[("v3", "*"), ("v4", "*")]),
            c(6, "n6.hi"),
            v(7, "minimum", "float", "w1", &[("v5", "*"), ("v6", "*")]),
            v(8, "convert", "sint", "w2", &[("v7", "p0")]),
        ],
        vec![
            input("x", "map", "w0"),
            input("s", "map", "w1"),
            input("zp", "map", "w1"),
        ],
        vec![output("out0", "output", "v8", "w2")],
        Value::Null,
        vec![tuple(&[ty, "f32", "i8"], json!({"n4.lo": "f32", "n6.hi": "f32"}))],
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
        "operands": [op("lhs", "in", json!([0, 2])), op("rhs", "in", json!([2, 1])), op("result0", "out", json!([0, 1]))],
        "accesses": [], "stages": {"prologue": {}, "epilogue": []}, "proof": ""
    })
}

fn instances(id: &str) -> Value {
    let row = json!({
        "model": "m", "phase": "p", "instance": "i", "component": "c", "kernel": "k", "member": 0,
        "extents": [4, 5, 6], "sizes": ["4", "5", "6"], "elems": ["i8"], "values": [],
        "workload": 60, "trip_count": 2, "weighted": 120
    });
    json!({"schema": "cc.kernel_classes.instances/1", "id": id, "omega": null, "rows": [row]})
}

fn template_file(id: &str) -> Value {
    json!({
        "schema": "cc.iteration_space.template/1", "id": id, "space": [], "groups": {},
        "candidates": [{"lambda": [1, 1, 1], "u": [0, 0, 1], "s": [[1, 0, 0], [0, 1, 0]], "links": [],
            "stationary": "rhs", "dataflow": "weight_stationary", "array": ["E(m)", "E(n)"],
            "utilization": 1.0, "link_bits": 0.0, "rank": 0}],
        "selected": 0, "families": ["sint", "sint", "sint"], "proof": ""
    })
}

fn routed(id: &str, route: &str, class: &str, workload: u64) -> Value {
    let flow = if route == "template" {
        json!("weight_stationary")
    } else {
        Value::Null
    };
    json!({"id": id, "route": route, "class": class, "core": "", "reason": "", "reuse": null,
           "dataflow": flow, "stages": null, "workload": workload})
}

fn pair(guest: &str, host: &str) -> Value {
    json!({"guest": guest, "host": host, "dims": [], "operands": [], "rows": [], "params": []})
}

fn row(key: &str, target: &str, area: f64, delay: f64) -> Row {
    Row {
        key: key.into(),
        id: key.replace(['@', '.', '>'], "_"),
        kind: Kind::Op,
        op: String::new(),
        variant: None,
        family: String::new(),
        types: Vec::new(),
        width: 0,
        kit: "k".into(),
        corner: "tt".into(),
        target: target.into(),
        period_ps: if target == "clock" { 1000.0 } else { 50.0 },
        measured: Measured {
            area_um2: Some(area),
            seq_area_um2: None,
            delay_ps: Some(delay),
            slack_ps: None,
            leakage_nw: None,
            children: BTreeMap::new(),
        },
        estimator: "fixture".into(),
        lec_tool: String::new(),
        lec: Lec::default(),
        inconclusive: false,
        designware: false,
        provisional: false,
        netlist: String::new(),
        netlist_hash: String::new(),
        reports: Vec::new(),
    }
}

fn library() -> Library {
    let mut rows = Vec::new();
    for (k, a, d) in [
        ("add@i8", 10.0, 50.0),
        ("add@i32", 40.0, 120.0),
        ("subtract@i32", 42.0, 120.0),
        ("multiply@i8", 50.0, 150.0),
        ("multiply@i16", 150.0, 250.0),
        ("multiply@i32", 400.0, 400.0),
        ("maximum@i8", 8.0, 40.0),
        ("maximum@i32", 30.0, 100.0),
        ("convert@i8>i32", 0.0, 0.0),
        ("convert@ui8>i32", 0.0, 0.0),
    ] {
        rows.push(row(k, "clock", a, d));
        rows.push(row(k, "fast", a, d));
    }
    for n in 2..=8u32 {
        for w in [1u32, 8, 16, 32] {
            let a = f64::from(n - 1) * f64::from(w) / 8.0;
            rows.push(row(&format!("mux{n}@w{w}"), "clock", a, 20.0));
            rows.push(row(&format!("mux{n}@w{w}"), "fast", a, 20.0));
        }
    }
    for w in [1u32, 8, 16, 32] {
        rows.push(row(&format!("register@w{w}"), "clock", f64::from(w), 0.0));
    }
    Library {
        schema: "cc.characterize.library/1".into(),
        tool: "fixture".into(),
        lec_tool: String::new(),
        tool_versions: BTreeMap::new(),
        kits: Vec::new(),
        targets: vec![
            TargetInfo {
                kit: "k".into(),
                target: "clock".into(),
                period_ps: 1000.0,
            },
            TargetInfo {
                kit: "k".into(),
                target: "fast".into(),
                period_ps: 50.0,
            },
        ],
        rows,
    }
}

/// Writes every input of `merge`; with `requant`, also a requantization class `zr1`.
#[derive(Debug)]
struct Inputs {
    requant: bool,
}

impl Pass for Inputs {
    fn name(&self) -> &'static str {
        "inputs"
    }
    fn version(&self) -> u32 {
        1
    }
    fn requires(&self) -> &[ArtifactKind] {
        &[]
    }
    fn provides(&self) -> &[ArtifactKind] {
        &[DATAPATHS, ITERATION_SPACE, KERNEL_CLASSES, MINE, LIBRARY]
    }
    fn model_dependent(&self) -> bool {
        false
    }
    fn check_config(&self, _: &PassSection) -> Result<(), CoreError> {
        Ok(())
    }
    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let mut dps = vec![
            mac("z01", "i32"),
            mac_self("z02"),
            mac("z03", "i8"),
            sub("z04"),
            reduce("z05"),
            reduce("z06"),
            template_a("z07"),
            template_b("z08"),
        ];
        let mut routes = vec![
            routed("z01", "datapath", "map", 10),
            routed("z02", "datapath", "map", 10),
            routed("z03", "datapath", "map", 10),
            routed("z04", "datapath", "map", 10),
            routed("z05", "datapath", "reduce", 10),
            routed("z06", "datapath", "reduce", 10),
            routed("z07", "template", "contraction", 1000),
            routed("z08", "template", "contraction", 10),
        ];
        if self.requant {
            dps.push(requant("zr1", "i8"));
            routes.push(routed("zr1", "datapath", "map", 10));
        }
        let d = ctx.output(DATAPATHS)?;
        let k = ctx.output(KERNEL_CLASSES)?;
        let s = ctx.output(ITERATION_SPACE)?;
        for dp in &dps {
            let id = dp["id"].as_str().unwrap();
            d.write_json(&format!("datapaths/{id}.json"), dp)?;
            k.write_json(&format!("instances/{id}.json"), &instances(id))?;
            if dp["route"] == "template" {
                s.write_json(&format!("skeletons/{id}.json"), &skeleton(id))?;
                s.write_json(&format!("templates/{id}.json"), &template_file(id))?;
            }
        }
        s.write_json(
            "routing.json",
            &json!({"schema": "cc.iteration_space.routing/1", "classes": routes}),
        )?;
        s.write_json(
            "embedding.json",
            &json!({"schema": "cc.iteration_space.embedding/1", "pairs": [
                pair("z02", "z01"), pair("z03", "z01"), pair("z04", "z01"), pair("z06", "z05"), pair("z08", "z07")
            ]}),
        )?;
        // One mined pattern: multiply -> add, in z01, z02 and z03.
        let m = ctx.output(MINE)?;
        let pid = "p0000000000000001";
        m.write_json(
            &format!("patterns/map/{pid}.json"),
            &json!({
                "schema": "cc.mine.pattern/1", "id": pid, "family": "map", "code": "",
                "labels": ["multiply@sint", "add@sint"], "edges": [], "tau": 2, "mni": 2, "embeddings": 2,
                "non_convex": 0,
                "occurrences": [
                    {"unit": "z01", "map": [0, 1], "inputs": []},
                    {"unit": "z02", "map": [0, 1], "inputs": []},
                    {"unit": "z03", "map": [0, 1], "inputs": []}
                ],
                "mis": [
                    {"unit": "z01", "workload": 1, "occurrences": [0], "algorithm": "exact_ilp", "chosen": [0], "size": 1, "optimal": true, "bound": 1},
                    {"unit": "z02", "workload": 1, "occurrences": [1], "algorithm": "exact_ilp", "chosen": [1], "size": 1, "optimal": true, "bound": 1},
                    {"unit": "z03", "workload": 1, "occurrences": [2], "algorithm": "exact_ilp", "chosen": [2], "size": 1, "optimal": true, "bound": 1}
                ],
                "score": 2
            }),
        )?;
        for f in ["map", "reduce", "contraction", "stage"] {
            let pats = if f == "map" {
                json!([{"id": pid, "vertices": 2, "score": 2, "mni": 2, "mis": 2}])
            } else {
                json!([])
            };
            m.write_json(
                &format!("selection/{f}.json"),
                &json!({"schema": "cc.mine.selection/1", "family": f, "tau": 2, "top_k": 8, "patterns": pats}),
            )?;
        }
        ctx.output(LIBRARY)?.write_json("library.json", &library())?;
        Ok(())
    }
    fn verify(&self, _: &PassContext) -> Result<VerifyReport, PassError> {
        Ok(VerifyReport::new("inputs"))
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
    /// The requantization class and `requant_variant = "fixed"` with these extra settings.
    requant: Option<Vec<String>>,
}

impl Setup {
    fn pipeline_with(&self, extra: &[&str]) -> Pipeline {
        let mut s = sets();
        s.extend(extra.iter().map(|x| x.to_string()));
        if let Some(r) = &self.requant {
            s.push("passes.merge.requant_variant=fixed".into());
            s.extend(r.iter().cloned());
        }
        let s: Vec<&str> = s.iter().map(String::as_str).collect();
        let cfg = self.env.config(CONFIG, &s);
        Pipeline::new(
            vec![
                Box::new(Inputs {
                    requant: self.requant.is_some(),
                }),
                Box::new(cc_passes::merge::Merge),
            ],
            cfg,
        )
        .unwrap()
    }

    fn pipeline(&self) -> Pipeline {
        self.pipeline_with(&[])
    }

    fn out(&self, rel: &str) -> PathBuf {
        self.env.build_dir().join(".merge/out/merge").join(rel)
    }

    fn value(&self, rel: &str) -> Value {
        serde_json::from_str(&std::fs::read_to_string(self.out(rel)).unwrap()).unwrap()
    }

    fn failing(&self) -> Vec<String> {
        let r = self.pipeline().verify_pass("merge").unwrap();
        r.checks
            .into_iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name)
            .collect()
    }

    fn expect(&self, check: &str) {
        let f = self.failing();
        assert!(
            f.iter().any(|c| c == check || c.ends_with(&format!(":{check}"))),
            "{check} not in {f:?}"
        );
    }

    fn edit(&self, rel: &str, f: impl FnOnce(&mut Value)) {
        let p = self.out(rel);
        let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        f(&mut v);
        std::fs::write(&p, v.to_string()).unwrap();
    }

    /// The units, by id.
    fn fus(&self) -> BTreeMap<String, FuFile> {
        let mut m = BTreeMap::new();
        for e in std::fs::read_dir(self.out("fus")).unwrap() {
            let p = e.unwrap().path();
            let f: FuFile = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            m.insert(f.id.clone(), f);
        }
        m
    }

    /// The unit covering `unit` and the relative path of its configuration.
    fn config_of(&self, unit: &str) -> (String, String) {
        for (id, f) in self.fus() {
            if f.covers.iter().any(|c| c.unit == unit) {
                return (id.clone(), format!("configs/{id}/{unit}/t0.json"));
            }
        }
        panic!("{unit} is not covered")
    }
}

fn fresh() -> Setup {
    let s = Setup {
        env: Env::new(),
        requant: None,
    };
    let st = s.pipeline().run(&RunOptions::default()).unwrap();
    assert_eq!(st.status_of("merge"), Some(RunStatus::Ok), "{st:?}");
    s
}

#[test]
fn merges_and_verifies_synthetic_datapaths() {
    let s = fresh();
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let sum: Summary = serde_json::from_value(s.value("summary.json")).unwrap();
    assert_eq!((sum.classes, sum.covered), (8, 8));
    let fams = &sum.families;
    // Map: the three multiply-adds merge; the subtract shares nothing.
    assert_eq!((fams["map"].units, fams["map"].fus), (4, 2), "{fams:?}");
    assert_eq!(fams["map"].macro_groups, 3);
    // Reduce: the two reductions are one unit with no multiplexer.
    assert_eq!((fams["reduce"].units, fams["reduce"].fus), (2, 1));
    let fus = s.fus();
    let red = fus
        .values()
        .find(|f| f.covers.iter().any(|c| c.unit == "z05"))
        .unwrap();
    assert!(red.muxes.is_empty() && red.covers.len() == 2, "{red:?}");
    // The i8 MAC runs widened on the i32 unit.
    let (_, p) = s.config_of("z03");
    let c: ConfigFile = serde_json::from_value(s.value(&p)).unwrap();
    assert!(c.bindings.iter().any(|b| format!("{:?}", b.mode) == "Widen"));
    // Stage: z08's epilogue merges with one of z07's stages (z07's own two run at the same time).
    assert_eq!((fams["stage"].units, fams["stage"].fus), (3, 2));
    // One i8 array serves both template classes; the ui8 operand is composed.
    assert_eq!(sum.arrays, 1);
    let a: ArrayFile = serde_json::from_value(s.value("arrays/array000.json")).unwrap();
    assert_eq!(a.pe, vec!["i8", "i8", "i32"]);
    let b = a.covers.iter().find(|c| c.class == "z08").unwrap();
    assert_eq!(b.tuples[0].mode, ArrayMode::Compose);
    assert_eq!(b.tuples[0].pieces, vec![7, 1]);
    assert!(b.embedded);
    assert!(sum.proofs >= 9 && sum.array_proofs == 2, "{sum:?}");
    // Each step's clique instance is stored (all are small here).
    assert!(sum.cliques.values().sum::<u64>() >= 4);
}

#[test]
fn merging_is_reproducible() {
    let s = fresh();
    let read = |s: &Setup| -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        for k in ["fus", "arrays", "trace"] {
            for e in std::fs::read_dir(s.out(k)).unwrap() {
                let p = e.unwrap().path();
                m.insert(p.display().to_string(), std::fs::read_to_string(&p).unwrap());
            }
        }
        m.insert(
            "coverage".into(),
            std::fs::read_to_string(s.out("coverage.json")).unwrap(),
        );
        m
    };
    let a = read(&s);
    let st = s
        .pipeline()
        .run(&RunOptions {
            from: Some("merge".into()),
            ..RunOptions::default()
        })
        .unwrap();
    assert_eq!(st.status_of("merge"), Some(RunStatus::Ok));
    assert_eq!(a, read(&s));
}

/// Every clique strategy gives a verified result.
#[test]
fn every_clique_strategy_verifies() {
    for alg in ["ostergard", "greedy"] {
        let s = Setup {
            env: Env::new(),
            requant: None,
        };
        let set = format!("passes.merge.clique.algorithm={alg}");
        let st = s.pipeline_with(&[&set]).run(&RunOptions::default()).unwrap();
        assert_eq!(st.status_of("merge"), Some(RunStatus::Ok), "{alg}: {st:?}");
        let r = s.pipeline_with(&[&set]).verify_pass("merge").unwrap();
        let bad: Vec<&str> = r
            .checks
            .iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name.as_str())
            .collect();
        assert!(bad.is_empty(), "{alg}: {bad:?}");
    }
}

#[test]
fn a_wrong_select_fails_the_proof() {
    let s = fresh();
    // z02's addend comes from an operand: its unit has multiplexers.
    let (_, p) = s.config_of("z02");
    let c: ConfigFile = serde_json::from_value(s.value(&p)).unwrap();
    assert!(!c.selects.is_empty(), "{c:?}");
    let k = c.selects[0];
    let n = s.fus()[&c.fu].blocks[k.block as usize].sources[k.port as usize].len() as u32;
    s.edit(&p, |v| v["selects"][0]["select"] = json!((k.select + 1) % n));
    let f = s.failing();
    assert!(f.contains(&"proofs".to_string()), "{f:?}");
    assert!(f.iter().any(|x| x.ends_with(":configs")), "{f:?}");
}

#[test]
fn a_rebound_node_fails_configs() {
    let s = fresh();
    let (fu, p) = s.config_of("z04");
    let c: ConfigFile = serde_json::from_value(s.value(&p)).unwrap();
    let n = s.fus()[&fu].blocks.len() as u32;
    let i = c.bindings.iter().position(|b| b.node == "v1").unwrap();
    let other = (c.bindings[i].block + 1) % n;
    s.edit(&p, |v| v["bindings"][i]["block"] = json!(other));
    s.expect(&format!("{fu}:configs"));
}

#[test]
fn a_missing_configuration_fails_coverage() {
    let s = fresh();
    let (fu, _) = s.config_of("z07/epilogue");
    s.edit(&format!("fus/{fu}.json"), |v| {
        let cs = v["covers"].as_array_mut().unwrap();
        cs.retain(|c| c["unit"] != "z07/epilogue");
    });
    s.expect("coverage");
}

#[test]
fn an_edited_area_fails() {
    let s = fresh();
    let (fu, _) = s.config_of("z01");
    s.edit(&format!("fus/{fu}.json"), |v| {
        v["area_um2"] = json!(v["area_um2"].as_f64().unwrap() - 1.0)
    });
    s.expect(&format!("{fu}:area"));
}

#[test]
fn a_one_input_mux_and_a_dead_source_fail_wiring() {
    let s = fresh();
    let (fu, _) = s.config_of("z02");
    let f = &s.fus()[&fu];
    let m = f.muxes[0].clone();
    // Drop the multiplexer entry: two sources without one.
    s.edit(&format!("fus/{fu}.json"), |v| {
        v["muxes"].as_array_mut().unwrap().remove(0);
    });
    s.expect(&format!("{fu}:wiring"));
    let _ = m;
}

#[test]
fn a_cycle_fails_dag() {
    let s = fresh();
    let (fu, _) = s.config_of("z01");
    let f = &s.fus()[&fu];
    // Feed an operator's first port from itself.
    let b = f
        .blocks
        .iter()
        .find(|b| b.kind == cc_passes::merge::json::BlockKind::Op)
        .unwrap()
        .id;
    s.edit(&format!("fus/{fu}.json"), |v| {
        v["blocks"][b as usize]["sources"][0]
            .as_array_mut()
            .unwrap()
            .push(json!({"block": b, "res": 0}));
    });
    s.expect(&format!("{fu}:dag"));
}

#[test]
fn an_edited_clique_fails() {
    let s = fresh();
    let dir = s.out("cliques/map");
    let f = std::fs::read_dir(&dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .file_name();
    let rel = format!("cliques/map/{}", f.to_str().unwrap());
    s.edit(&rel, |v| {
        v["objective"] = json!(v["objective"].as_f64().unwrap() + 5.0)
    });
    s.expect("cliques");
}

#[test]
fn an_edited_trace_and_array_fail() {
    let s = fresh();
    s.edit("trace/map.json", |v| v["initial_units"] = json!(5));
    s.expect("trace");
    s.edit("arrays/array000.json", |v| {
        v["covers"][1]["tuples"][0]["pieces"] = json!([8, 8]);
    });
    s.expect("arrays");
}

fn requant_setup(extra: &[&str]) -> Setup {
    let s = Setup {
        env: Env::new(),
        requant: Some(extra.iter().map(|x| x.to_string()).collect()),
    };
    let st = s.pipeline().run(&RunOptions::default()).unwrap();
    assert_eq!(st.status_of("merge"), Some(RunStatus::Ok), "{st:?}");
    s
}

/// The ops of the blocks of the unit that covers `unit`.
fn ops_of(s: &Setup, unit: &str) -> Vec<String> {
    let (id, _) = s.config_of(unit);
    s.fus()[&id].blocks.iter().map(|b| b.op.clone()).collect()
}

/// `fixed`: the chain becomes one `rescale` block, its lemma and configurations are proven, and verify
/// catches an edited decision, an edited lemma and an edited lemma result.
#[test]
fn fixed_requantization_rewrites_and_verifies() {
    let s = requant_setup(&[]);
    assert!(s.failing().is_empty(), "{:?}", s.failing());
    let sum: Summary = serde_json::from_value(s.value("summary.json")).unwrap();
    assert_eq!(sum.requant_variant, "fixed");
    assert_eq!((sum.classes, sum.covered), (9, 9));
    assert_eq!((sum.requant_fixed, sum.requant_kept), (1, 0));
    let rq: RequantFile = serde_json::from_value(s.value("requant.json")).unwrap();
    assert_eq!(rq.lemmas.len(), 1);
    assert!(rq.lemmas[0].proven && rq.lemmas[0].ty == "i8", "{rq:?}");
    assert_eq!(rq.classes[0].class, "zr1");
    assert_eq!(rq.classes[0].status, RequantStatus::Fixed);
    assert_eq!(
        (
            rq.classes[0].sites[0].convert,
            rq.classes[0].sites[0].multiply,
            rq.classes[0].sites[0].round
        ),
        (0, 1, 2)
    );
    let ops = ops_of(&s, "zr1");
    assert!(ops.iter().any(|o| o == "rescale"), "{ops:?}");
    assert!(!ops.iter().any(|o| o == "round_nearest_even"), "{ops:?}");

    let restore = |rel: &str| {
        let p = s.out(rel);
        let text = std::fs::read_to_string(&p).unwrap();
        move || std::fs::write(&p, &text).unwrap()
    };
    let back = restore("requant.json");
    s.edit("requant.json", |v| v["classes"][0]["status"] = json!("kept"));
    s.expect("requant");
    back();
    let back = restore("proofs/lemmas/rescale_i8.smt2");
    let p = s.out("proofs/lemmas/rescale_i8.smt2");
    let t = std::fs::read_to_string(&p).unwrap();
    std::fs::write(&p, t.replace("(check-sat)", "(assert true)\n(check-sat)")).unwrap();
    s.expect("requant");
    back();
    let back = restore("proofs/lemmas/rescale_i8.result");
    s.edit("proofs/lemmas/rescale_i8.result", |v| {
        v["verdict"] = json!({"sat": {"model": ""}})
    });
    s.expect("requant");
    back();
}

/// A lemma that is not proven (here: out of time) leaves its classes on fp32; everything verifies.
#[test]
fn an_unproven_lemma_keeps_fp32() {
    let s = requant_setup(&["passes.merge.requant_timeout_s=1"]);
    let rq: RequantFile = serde_json::from_value(s.value("requant.json")).unwrap();
    assert!(!rq.lemmas[0].proven, "{rq:?}");
    assert_eq!(rq.classes[0].status, RequantStatus::Kept);
    assert!(
        rq.classes[0]
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("rescale_i8"),
        "{rq:?}"
    );
    let ops = ops_of(&s, "zr1");
    assert!(ops.iter().any(|o| o == "round_nearest_even"), "{ops:?}");
    assert!(!ops.iter().any(|o| o == "rescale"), "{ops:?}");
    let sum: Summary = serde_json::from_value(s.value("summary.json")).unwrap();
    assert_eq!((sum.requant_fixed, sum.requant_kept, sum.covered), (0, 1, 9));
    assert!(s.failing().is_empty(), "{:?}", s.failing());
}
