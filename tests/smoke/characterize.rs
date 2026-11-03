//! `characterize` on stand-in tools: the image's XLS calls write a module per DSLX file, `dc_shell` and
//! `fm_shell` write reports per design (from the scripts the pass generated) on a synthetic four-cell
//! library. The pass runs green, a rerun reuses its job directories, a failed equivalence fails the pass,
//! and every check of its `verify` fails on a mutated artifact.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cc_core::runner::{Caps, Invocation, Runners, ToolLog, ToolOutcome, ToolRunner};
use cc_core::{CancelToken, CheckStatus, LoadedConfig};
use cc_passes::characterize::catalog::catalog;
use cc_passes::characterize::json::{Library, Summary};
use cc_passes::characterize::{CharacterizeConfig, flow};
use cc_passes::eda::netlist::Dir;
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::{Env, repo_root};

const CONFIG: &str = r#"
[pipeline]
targets = ["characterize"]
[resources]
memory_budget_gb = 8
threads_total = 4
[models]
keys = []
[passes.characterize]
pdks = ["nangate45"]
targets = ["clock"]
int_widths = [4, 8]
float_formats = ["f16"]
mux_inputs = [2]
mux_widths = [2, 4, 8]
register_widths = [8]
split_multipliers = [[8, 4]]
driving_cell = { nangate45 = "BUF_X1" }
batch = 40
[passes.characterize.resources]
threads = 2
memory_gb = 2
timeout_min = 5
"#;

/// The stand-in tools: one runner for Docker and EDA, dispatching on the program.
#[derive(Debug, Default)]
struct Tools {
    /// Designs whose equivalence check fails.
    lec_fails: Mutex<Vec<String>>,
    /// Designs whose equivalence check ends inconclusive (no failing point, some unverified).
    lec_inconclusive: Mutex<Vec<String>>,
    calls: Mutex<Vec<String>>,
}

fn ids(script: &str) -> Vec<String> {
    script
        .lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix("echo \"@@begin ")
                .map(|r| r.trim_end_matches('"').to_string())
        })
        .collect()
}

/// A stand-in gate netlist with the RTL's ports, two gates, and one flop per bit of a clocked design's
/// output.
fn netlist(cwd: &Path, id: &str) -> String {
    let rtl = std::fs::read_to_string(cwd.join(format!("rtl/{id}.v"))).unwrap();
    let ports = cc_passes::eda::netlist::ports(&rtl, id).unwrap();
    let names: Vec<&str> = ports.keys().map(String::as_str).collect();
    let mut s = format!("module {id} ( {} );\n", names.join(", "));
    for (n, (d, w)) in &ports {
        let dir = match d {
            Dir::Input => "input",
            Dir::Output => "output",
            Dir::Inout => "inout",
        };
        if *w == 1 {
            s.push_str(&format!("  {dir} {n};\n"));
        } else {
            s.push_str(&format!("  {dir} [{}:0] {n};\n", w - 1));
        }
    }
    s.push_str("  NAND2_X1 U1 ( .A1(n1), .A2(n2), .ZN(n3) );\n  BUF_X1 U2 ( .A(n3), .Z(n4) );\n");
    if ports.contains_key("clk") {
        let w = ports.values().find(|(d, _)| *d == Dir::Output).map_or(0, |p| p.1);
        for i in 0..w {
            s.push_str(&format!("  DFF_X1 R{i} ( .D(n4), .CK(clk), .Q(q{i}) );\n"));
        }
    }
    s.push_str("endmodule\n");
    s
}

fn area_of(id: &str) -> f64 {
    // Deterministic, and growing with the width digits of the id.
    let w: f64 = id
        .rsplit(|c: char| !c.is_ascii_digit())
        .find(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    3.0 * w + id.len() as f64 * 0.01
}

impl ToolRunner for Tools {
    fn name(&self) -> &'static str {
        "stand-in"
    }
    fn run(
        &self,
        inv: &Invocation,
        _caps: &Caps,
        log: &ToolLog,
        _c: &CancelToken,
    ) -> cc_core::Result<ToolOutcome> {
        self.calls.lock().unwrap().push(inv.program.clone());
        let files = log.next(&inv.tool)?;
        let cwd = inv
            .mounts
            .iter()
            .find(|m| m.dst == "/work")
            .map(|m| m.src.clone())
            .or_else(|| inv.cwd.clone())
            .unwrap_or_default();
        let mut out = String::new();
        match inv.program.as_str() {
            "bash" => {
                for e in std::fs::read_dir(&cwd).unwrap() {
                    let p = e.unwrap().path();
                    if p.extension().is_some_and(|x| x == "x") {
                        let id = p.file_stem().unwrap().to_str().unwrap().to_string();
                        std::fs::write(
                            cwd.join(format!("{id}.v")),
                            format!("module {id}(input wire [7:0] a, input wire [7:0] b, output wire [7:0] out);\n  assign out = a & b;\nendmodule\n"),
                        )
                        .unwrap();
                    }
                }
                out.push_str("@@done\n");
            }
            "dc_shell" => {
                let f = inv
                    .args
                    .iter()
                    .position(|a| a == "-f")
                    .map(|i| inv.args[i + 1].clone())
                    .unwrap();
                let script = std::fs::read_to_string(cwd.join(f)).unwrap();
                for id in ids(&script) {
                    let a = area_of(&id);
                    let w = |f: &str, t: String| std::fs::write(cwd.join(format!("{id}.{f}")), t).unwrap();
                    w(
                        "area.rpt",
                        format!(
                            "Combinational area:  {a}\nNoncombinational area:  0.000000\nTotal cell area:   {a}\n"
                        ),
                    );
                    w(
                        "qor.rpt",
                        format!(
                            "  Critical Path Length:  {:.2}\n  Critical Path Slack:  0.10\n",
                            a / 100.0
                        ),
                    );
                    w(
                        "power.rpt",
                        format!("Cell Leakage Power     = {:.4} nW\n", a * 2.0),
                    );
                    w("timing.rpt", String::new());
                    w("check.rpt", String::new());
                    w("net.v", netlist(&cwd, &id));
                    out.push_str(&format!("@@begin {id}\nInformation: compiled {id}\n@@end {id}\n"));
                }
            }
            "fm_shell" => {
                let script = std::fs::read_to_string(cwd.join("lec.tcl")).unwrap();
                let fails = self.lec_fails.lock().unwrap().clone();
                let open = self.lec_inconclusive.lock().unwrap().clone();
                for id in ids(&script) {
                    let (v, p, f, u) = if fails.contains(&id) {
                        ("FAILED", 7, 1, 0)
                    } else if open.contains(&id) {
                        ("INCONCLUSIVE", 6, 0, 2)
                    } else {
                        ("SUCCEEDED", 8, 0, 0)
                    };
                    std::fs::write(
                        cwd.join(format!("{id}.lec.rpt")),
                        format!("Verification {v}\nPassing (equivalent)  0  0  0  0  {p}  0  0  {p}\nFailing (not equivalent)  0  0  0  0  {f}  0  0  {f}\nUnverified  0  0  0  0  {u}  0  0  {u}\n"),
                    )
                    .unwrap();
                    out.push_str(&format!("@@begin {id}\n@@end {id}\n"));
                }
            }
            // Version probes of the image's tools.
            _ => out.push_str("stand-in 1.0\n"),
        }
        std::fs::write(&files.stdout, out).unwrap();
        std::fs::write(&files.stderr, "").unwrap();
        Ok(ToolOutcome {
            runner: "stand-in".into(),
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
}

struct Setup {
    env: Env,
    tools: Arc<Tools>,
    /// `--set` overrides.
    sets: Mutex<Vec<String>>,
}

impl Setup {
    fn new() -> Self {
        let env = Env::new();
        let db = env.build_dir().join(".env/out/pdk_db/mini_lib.db");
        std::fs::create_dir_all(db.parent().unwrap()).unwrap();
        std::fs::write(&db, "stand-in db\n").unwrap();
        Self {
            env,
            tools: Arc::new(Tools::default()),
            sets: Mutex::new(Vec::new()),
        }
    }

    fn config(&self) -> LoadedConfig {
        let lib = repo_root().join("tests/fixtures/characterize/mini.lib");
        let text = format!("{CONFIG}\n[paths]\nnangate45_lib = \"{}\"\n", lib.display());
        let sets = self.sets.lock().unwrap().clone();
        let sets: Vec<&str> = sets.iter().map(String::as_str).collect();
        self.env.config(&text, &sets)
    }

    fn pipeline(&self) -> Pipeline {
        let cfg = self.config();
        let mut r = Runners::from_config(&cfg);
        r.docker = self.tools.clone();
        r.eda = self.tools.clone();
        Pipeline::new(cc_passes::registry(), cfg).unwrap().with_runners(r)
    }

    fn run(&self) -> cc_pipeline::RunSummary {
        self.pipeline().run(&RunOptions::default()).unwrap()
    }

    fn out(&self, rel: &str) -> PathBuf {
        self.env.build_dir().join(".characterize/out/library").join(rel)
    }

    fn library(&self) -> Library {
        serde_json::from_str(&std::fs::read_to_string(self.out("library.json")).unwrap()).unwrap()
    }

    fn checks(&self) -> Vec<(String, CheckStatus)> {
        let r = self.pipeline().verify_pass("characterize").unwrap();
        r.checks.into_iter().map(|c| (c.name, c.status)).collect()
    }

    fn failing(&self) -> Vec<String> {
        self.checks()
            .into_iter()
            .filter(|c| c.1 == CheckStatus::Fail)
            .map(|c| c.0)
            .collect()
    }

    fn expect(&self, check: &str) {
        let f = self.failing();
        assert!(f.iter().any(|c| c == check), "{check} not in {f:?}");
    }

    fn edit_json(&self, rel: &str, f: impl FnOnce(&mut serde_json::Value)) {
        let p = self.out(rel);
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        f(&mut v);
        std::fs::write(&p, v.to_string()).unwrap();
    }

    fn green() -> Self {
        let s = Self::new();
        let sum = s.run();
        assert!(sum.ok, "{sum:?}");
        s
    }
}

fn first_id(s: &Setup, prefix: &str) -> String {
    s.library()
        .rows
        .iter()
        .find(|r| r.id.starts_with(prefix))
        .unwrap()
        .id
        .clone()
}

#[test]
fn runs_green_on_stand_in_tools_and_reuses_its_jobs() {
    let s = Setup::green();
    let cfg: CharacterizeConfig = s.config().pass_section("characterize").unwrap().parse().unwrap();
    let cat = catalog(&cfg).unwrap();
    let lib = s.library();
    let with_rtl = cat.iter().filter(|p| p.source.is_some()).count();
    assert_eq!(lib.rows.len(), with_rtl);
    assert!(
        lib.rows
            .iter()
            .all(|r| r.lec.equivalent && r.measured.area_um2.is_some())
    );
    assert_eq!(lib.kits[0].lib_name, "mini_lib");
    assert!(
        (lib.kits[0].load - 4.0 * 0.97).abs() < 1e-9,
        "load = 4 x BUF_X1/A"
    );
    let failing = s.failing();
    assert!(failing.is_empty(), "{failing:?}");
    let checks = s.checks();
    assert!(checks.contains(&("coverage.datapaths".into(), CheckStatus::Skip)));
    let sum: Summary =
        serde_json::from_str(&std::fs::read_to_string(s.out("summary.json")).unwrap()).unwrap();
    assert!(sum.jobs_run > 0 && sum.jobs_reused == 0);
    assert!(
        sum.pending > 0,
        "float ops without an apfloat function are pending"
    );
    assert!(
        std::fs::read_to_string(s.env.build_dir().join(".characterize/views/library.md"))
            .unwrap()
            .contains("## Area against width")
    );

    // A forced rerun finds every job directory done.
    let calls = s.tools.calls.lock().unwrap().len();
    let again = s
        .pipeline()
        .run(&RunOptions {
            from: Some("characterize".into()),
            ..RunOptions::default()
        })
        .unwrap();
    assert!(again.ok, "{again:?}");
    let sum: Summary =
        serde_json::from_str(&std::fs::read_to_string(s.out("summary.json")).unwrap()).unwrap();
    assert_eq!(sum.jobs_run, 0);
    assert!(sum.jobs_reused > 0);
    let tool_calls: Vec<String> = s.tools.calls.lock().unwrap()[calls..].to_vec();
    assert!(
        !tool_calls.iter().any(|c| c == "dc_shell" || c == "fm_shell"),
        "{tool_calls:?}"
    );
}

#[test]
fn a_failed_equivalence_fails_the_run() {
    let s = Setup::new();
    s.tools.lec_fails.lock().unwrap().push("add_i8".into());
    let sum = s.run();
    assert!(!sum.ok);
    assert!(
        sum.passes
            .iter()
            .any(|p| p.pass == "characterize" && p.status == RunStatus::Failed)
    );
    let err = format!("{sum:?}");
    assert!(
        err.contains("add@i8") && err.contains("equivalence not proven"),
        "{err}"
    );
}

#[test]
fn verify_catches_a_failed_equivalence_report() {
    let s = Setup::green();
    let id = first_id(&s, "add_i8");
    std::fs::write(
        s.out(&format!("reports/nangate45/clock/{id}.lec.rpt")),
        "Verification FAILED\n",
    )
    .unwrap();
    s.expect("nangate45:clock:lec");
}

#[test]
fn verify_catches_a_latch_and_a_cell_outside_the_library() {
    let s = Setup::green();
    let id = first_id(&s, "add_i8");
    let net = format!("netlist/nangate45/clock/{id}.v");
    let p = s.out(&net);
    let text = std::fs::read_to_string(&p).unwrap().replace(
        "endmodule",
        "  DLH_X1 L1 ( .D(a[3]), .G(a[4]), .Q(out[2]) );\nendmodule",
    );
    std::fs::write(&p, text).unwrap();
    s.expect("nangate45:clock:sanity");
    s.expect("nangate45:clock:rows");

    let s = Setup::green();
    let p = s.out(&net);
    let text = std::fs::read_to_string(&p).unwrap().replace(
        "endmodule",
        "  GTECH_AND2 G1 ( .A(a[3]), .B(a[4]), .Z(out[2]) );\nendmodule",
    );
    std::fs::write(&p, text).unwrap();
    s.expect("nangate45:clock:sanity");
}

#[test]
fn verify_catches_an_unresolved_reference_and_a_tool_error() {
    let s = Setup::green();
    let id = first_id(&s, "add_i8");
    let log = s.out(&format!("reports/nangate45/clock/{id}.log"));
    let text = std::fs::read_to_string(&log).unwrap().replacen(
        "@@lec\n",
        "Warning: Design 'x' has '1' unresolved references. For more detailed information, use the \"link\" command. (UID-341)\n@@lec\n",
        1,
    );
    std::fs::write(&log, text).unwrap();
    s.expect("nangate45:clock:sanity");
    let s = Setup::green();
    std::fs::write(
        s.out(&format!("reports/nangate45/clock/{id}.log")),
        "Error: Cannot find the design 'x' in the library 'WORK'. (LBR-0)\n@@lec\n",
    )
    .unwrap();
    s.expect("nangate45:clock:sanity");
}

#[test]
fn verify_catches_an_edited_value_a_missing_row_and_an_edited_catalog() {
    let s = Setup::green();
    s.edit_json("library.json", |v| {
        let a = v["rows"][0]["measured"]["area_um2"].as_f64().unwrap();
        v["rows"][0]["measured"]["area_um2"] = (a + 1.0).into();
    });
    s.expect("nangate45:clock:rows");

    let s = Setup::green();
    s.edit_json("library.json", |v| {
        v["rows"].as_array_mut().unwrap().remove(0);
    });
    s.expect("complete");

    let s = Setup::green();
    s.edit_json("catalog.json", |v| {
        v[0]["width"] = 99.into();
    });
    s.expect("catalog");

    let s = Setup::green();
    let id = first_id(&s, "add_i8");
    std::fs::write(s.out(&format!("rtl/{id}.x")), "pub fn edited() {}\n").unwrap();
    s.expect("rtl");
}

#[test]
fn an_off_trend_point_is_a_note_not_a_failure() {
    let s = Setup::green();
    // mux2 at 8 bits gets less area than at 4 bits, in the report and the library alike.
    let lib = s.library();
    let i = lib.rows.iter().position(|r| r.key == "mux2@w8").unwrap();
    let id = lib.rows[i].id.clone();
    let rpt = s.out(&format!("reports/nangate45/clock/{id}.area.rpt"));
    std::fs::write(
        &rpt,
        "Combinational area:  1.0\nNoncombinational area:  0.000000\nTotal cell area:   1.0\n",
    )
    .unwrap();
    s.edit_json("library.json", |v| {
        v["rows"][i]["measured"]["area_um2"] = 1.0.into();
    });
    let checks = s.pipeline().verify_pass("characterize").unwrap();
    let sanity = checks.checks.iter().find(|c| c.name == "trends").unwrap();
    assert_eq!(sanity.status, CheckStatus::Pass);
    assert!(sanity.evidence.counts["flags"] >= 1, "{sanity:?}");
    // Only the framework's output hashes notice the edit.
    let own: Vec<String> = s
        .failing()
        .into_iter()
        .filter(|c| !c.starts_with("framework."))
        .collect();
    assert!(own.is_empty(), "{own:?}");
}

fn datapath(dir: &Path, id: &str, op: &str, elem: &str) {
    let dp = serde_json::json!({
        "schema": "cc.datapaths.datapath/1", "id": id, "route": "datapath", "class": "map", "core": "",
        "widths": {"w0": "sint"},
        "vertices": [{"id": 0, "label": format!("{op}@sint"), "op": op, "family": "sint", "width": "w0",
            "attrs": {}, "stage": "map",
            "args": [{"src": {"input": {"name": "in0"}}, "port": "p0"}, {"src": {"input": {"name": "in0"}}, "port": "p1"}],
            "from": {"node": 0, "access": null}}],
        "inputs": [{"name": "in0", "kind": "map", "node": 0, "res": 0, "incidence": [], "width": "w0"}],
        "outputs": [], "accumulator": null,
        "tuples": [{"name": "t0", "elems": [elem], "consts": {}, "attrs": {}, "rows": 1}]
    });
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(format!("{id}.json")), dp.to_string()).unwrap();
}

#[test]
fn verify_maps_every_datapath_vertex_to_the_library() {
    let s = Setup::green();
    let dir = s.env.build_dir().join(".datapaths/out/datapaths/datapaths");
    datapath(&dir, "z0", "add", "i8");
    datapath(&dir, "z1", "subtract", "i4");
    let failing = s.failing();
    assert!(failing.is_empty(), "{failing:?}");
    let checks = s.checks();
    assert!(checks.contains(&("coverage.datapaths".into(), CheckStatus::Pass)));
    // An integer width the catalog lacks is a gap.
    datapath(&dir, "z2", "add", "i64");
    s.expect("coverage.datapaths");
}

#[test]
fn catalog_is_deterministic_and_complete_over_the_op_table() {
    let cfg = CharacterizeConfig::default();
    let a = catalog(&cfg).unwrap();
    assert_eq!(a, catalog(&cfg).unwrap());
    let keys: std::collections::BTreeSet<&str> = a.iter().map(|p| p.key.as_str()).collect();
    assert_eq!(keys.len(), a.len());
    for k in [
        "add@i32",
        "multiply@ui8",
        "compare.LT@i8",
        "compare.LE@f32",
        "convert@f32>bf16",
        "convert@i8>f8e4m3fn",
        "maximum@bf16",
        "select@f16",
        "mux2@w1",
        "mux8@w32",
        "register@w16",
        "accumulator.add@i32",
        "accumulator.add@f32",
        "split@w32n8",
        "rescale@i32",
        "rescale@i8",
    ] {
        assert!(keys.contains(k), "{k}");
    }
    // The rescale primitives come last (earlier job batches stay as they are) and reuse the
    // characterized i32 -> f32 conversion.
    let first = a.iter().position(|p| p.op == "rescale").unwrap();
    assert!(a[first..].iter().all(|p| p.op == "rescale"));
    let r = a.iter().find(|p| p.key == "rescale@i32").unwrap();
    assert_eq!(r.deps, vec!["convert_i32_f32".to_string()]);
    let pending: Vec<&str> = a
        .iter()
        .filter(|p| p.pending.is_some())
        .map(|p| p.key.as_str())
        .collect();
    assert!(
        pending.contains(&"divide@f32") && pending.contains(&"exponential@bf16"),
        "{pending:?}"
    );
    assert!(
        !pending.iter().any(|k| k.ends_with("@i32")),
        "every integer op has RTL: {pending:?}"
    );
    // Integer divide follows XLA's results for a zero divisor and the overflow case.
    let div = a.iter().find(|p| p.key == "divide@i8").unwrap();
    let src = format!("{:?}", div.source);
    assert!(
        src.contains("sN[8]:-1") && src.contains("signed_min_value"),
        "{src}"
    );
    // GT and GE are LT and LE with the operands exchanged: no own rows.
    assert!(
        !keys
            .iter()
            .any(|k| k.starts_with("compare.GT") || k.starts_with("compare.GE"))
    );
}

#[test]
fn kits_need_the_environment() {
    let s = Setup::new();
    std::fs::remove_file(s.env.build_dir().join(".env/out/pdk_db/mini_lib.db")).unwrap();
    let sum = s.run();
    assert!(!sum.ok);
    assert!(format!("{sum:?}").contains("chipc doctor"));
    let _ = flow::design;
}

#[test]
fn an_inconclusive_check_passes_only_with_every_sanity_check_and_a_failing_one_never() {
    // Inconclusive, sanity checks green: the run and verify pass, the row is marked.
    let s = Setup::new();
    s.tools.lec_inconclusive.lock().unwrap().push("add_i8".into());
    let sum = s.run();
    assert!(sum.ok, "{sum:?}");
    let row = s.library().rows.into_iter().find(|r| r.key == "add@i8").unwrap();
    assert!(row.inconclusive && !row.lec.equivalent);
    let failing = s.failing();
    assert!(failing.is_empty(), "{failing:?}");
    let checks = s.pipeline().verify_pass("characterize").unwrap();
    let lec = checks
        .checks
        .iter()
        .find(|c| c.name == "nangate45:clock:lec")
        .unwrap();
    assert_eq!(lec.evidence.counts["inconclusive"], 1);
    let summary: Summary =
        serde_json::from_str(&std::fs::read_to_string(s.out("summary.json")).unwrap()).unwrap();
    assert_eq!(summary.lec_inconclusive, 1);

    // Inconclusive and a sanity check fails (a port the RTL does not have): verify fails.
    let net = s.out(&format!("netlist/nangate45/clock/{}.v", row.id));
    let text = std::fs::read_to_string(&net)
        .unwrap()
        .replacen(" );", ", extra );\n  output extra;", 1);
    std::fs::write(&net, text).unwrap();
    s.expect("nangate45:clock:sanity");

    // A failing compare point always fails the run.
    let s = Setup::new();
    s.tools.lec_fails.lock().unwrap().push("add_i8".into());
    assert!(!s.run().ok);
}

#[test]
fn verify_catches_ports_and_flip_flops_that_differ_from_the_rtl() {
    let s = Setup::green();
    let id = first_id(&s, "add_i8");
    let net = s.out(&format!("netlist/nangate45/clock/{id}.v"));
    let text = std::fs::read_to_string(&net)
        .unwrap()
        .replace("output [7:0] out", "output [6:0] out");
    std::fs::write(&net, text).unwrap();
    s.expect("nangate45:clock:sanity");

    // A register with one flop fewer than its width.
    let s = Setup::green();
    let id = first_id(&s, "register_w8");
    let net = s.out(&format!("netlist/nangate45/clock/{id}.v"));
    let text = std::fs::read_to_string(&net)
        .unwrap()
        .replace("  DFF_X1 R7 ( .D(n4), .CK(clk), .Q(q7) );\n", "");
    std::fs::write(&net, text).unwrap();
    s.expect("nangate45:clock:sanity");

    // A combinational design with a flop.
    let s = Setup::green();
    let id = first_id(&s, "add_i8");
    let net = s.out(&format!("netlist/nangate45/clock/{id}.v"));
    let text = std::fs::read_to_string(&net)
        .unwrap()
        .replace("endmodule", "  DFF_X1 R0 ( .D(n4), .CK(n1), .Q(n9) );\nendmodule");
    std::fs::write(&net, text).unwrap();
    s.expect("nangate45:clock:sanity");
}

/// Deterministic test vectors for `rescale`: edge cases and pseudo-random operands with scales over the
/// exponent range that matters (right shifts, ties, left shifts, saturation, subnormal and zero scales).
fn rescale_vectors(n: u32) -> Vec<(i64, u32)> {
    let lo = -(1i64 << (n - 1));
    let hi = (1i64 << (n - 1)) - 1;
    let mut v = Vec::new();
    for x in [0, 1, -1, 2, -2, 3, -3, 5, -5, lo, hi, lo + 1, hi - 1] {
        for s in [
            0.5f32,
            0.25,
            0.75,
            1.5,
            0.001,
            2f32.powi(-40),
            2f32.powi(23),
            2f32.powi(24),
            2f32.powi(30),
            2f32.powi(31),
            0.0,
            -0.0,
            -0.5,
            -0.75,
            f32::INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
        ] {
            v.push((x, s.to_bits()));
        }
        v.push((x, 1));
    }
    let mut st: u64 = 0x9e37_79b9_7f4a_7c15 ^ u64::from(n);
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    for _ in 0..3000 {
        let r = next();
        let x = ((r as i64) >> (64 - n)).clamp(lo, hi);
        let e = 90 + (next() % 70) as u32;
        let s = ((next() & 1) as u32) << 31 | e << 23 | (next() as u32 & 0x7f_ffff);
        v.push((x, s));
    }
    v
}

#[test]
fn rescale_rtl_matches_its_reference() {
    let env = Env::new();
    let cfg: toml::Value =
        toml::from_str(&std::fs::read_to_string(repo_root().join("config.toml")).unwrap()).unwrap();
    let tag = cfg["tools"]["docker"]["tag"].as_str().unwrap().to_string();
    let lc = env.config("[models]\nkeys = []\n", &[&format!("tools.docker.tag={tag}")]);
    let dir = env.root().join("rescale");
    std::fs::create_dir_all(&dir).unwrap();
    let cat = catalog(&CharacterizeConfig::default()).unwrap();
    let runner = cc_core::runner::BoundRunner::new(
        Runners::from_config(&lc).docker,
        Caps {
            memory_gb: 2.0,
            cpus: 1.0,
            timeout_s: 300.0,
            pids_limit: 64,
        },
        ToolLog::new("rescale-rtl", dir.join("logs")),
        CancelToken::new(),
    );
    for n in [8u32, 16, 32] {
        let p = cat.iter().find(|p| p.key == format!("rescale@i{n}")).unwrap();
        let Some(cc_passes::characterize::catalog::Source::Verilog(v)) = &p.source else {
            panic!("rescale@i{n}: no Verilog");
        };
        std::fs::write(dir.join(format!("rescale_i{n}.v")), v).unwrap();
        let vs = rescale_vectors(n);
        let mut hex = String::new();
        for &(x, s) in &vs {
            let want = cc_smt::ops::rescale_reference_int(x, s) as u32;
            hex.push_str(&format!("{:08x}{s:08x}{want:08x}\n", x as u32));
        }
        std::fs::write(dir.join(format!("vec_i{n}.hex")), hex).unwrap();
        let tb = format!(
            "module tb;\n  reg [95:0] mem [0:{last}];\n  reg [31:0] x, s, want;\n  wire [31:0] q;\n  integer i, bad;\n  rescale_core_i{n} dut(.a(x[{nh}:0]), .b(s), .q(q));\n  initial begin\n    $readmemh(\"/w/vec_i{n}.hex\", mem);\n    bad = 0;\n    for (i = 0; i <= {last}; i = i + 1) begin\n      {{x, s, want}} = mem[i];\n      #1;\n      if (q !== want) begin\n        bad = bad + 1;\n        if (bad < 10) $display(\"MISMATCH x=%h s=%h q=%h want=%h\", x, s, q, want);\n      end\n    end\n    $display(\"DONE %0d %0d\", {count}, bad);\n  end\nendmodule\n",
            last = vs.len() - 1,
            count = vs.len(),
            nh = n - 1
        );
        std::fs::write(dir.join(format!("tb_i{n}.v")), tb).unwrap();
        let inv = Invocation::new("iverilog", "sh")
            .args([
                "-c".to_string(),
                format!(
                    "iverilog -g2012 -s tb -o /tmp/tb.vvp /w/rescale_i{n}.v /w/tb_i{n}.v && vvp -n /tmp/tb.vvp"
                ),
            ])
            .mount(cc_core::runner::Mount {
                src: dir.clone(),
                dst: "/w".into(),
                readonly: true,
            });
        let o = runner.run(&inv).unwrap();
        let out = std::fs::read_to_string(&o.stdout).unwrap();
        let err = std::fs::read_to_string(&o.stderr).unwrap_or_default();
        assert!(
            out.contains(&format!("DONE {} 0", vs.len())),
            "rescale_i{n}: {out}\n{err}"
        );
    }
}
