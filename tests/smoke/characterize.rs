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
                    w(
                        "net.v",
                        format!(
                            "module {id} ( a, b, out );\n  input [7:0] a;\n  output [7:0] out;\n  NAND2_X1 U1 ( .A1(a[0]), .A2(a[1]), .ZN(out[0]) );\n  BUF_X1 U2 ( .A(a[2]), .Z(out[1]) );\nendmodule\n"
                        ),
                    );
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
    s.expect("nangate45:clock:log");
    s.expect("nangate45:clock:rows");

    let s = Setup::green();
    let p = s.out(&net);
    let text = std::fs::read_to_string(&p).unwrap().replace(
        "endmodule",
        "  GTECH_AND2 G1 ( .A(a[3]), .B(a[4]), .Z(out[2]) );\nendmodule",
    );
    std::fs::write(&p, text).unwrap();
    s.expect("nangate45:clock:log");
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
    s.expect("nangate45:clock:log");
    let s = Setup::green();
    std::fs::write(
        s.out(&format!("reports/nangate45/clock/{id}.log")),
        "Error: Cannot find the design 'x' in the library 'WORK'. (LBR-0)\n@@lec\n",
    )
    .unwrap();
    s.expect("nangate45:clock:log");
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
    let sanity = checks.checks.iter().find(|c| c.name == "sanity").unwrap();
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
    ] {
        assert!(keys.contains(k), "{k}");
    }
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

const ALLOW_ADD_I8: &str =
    "passes.characterize.allow_inconclusive=[{ key = \"add@i8\", reason = \"stand-in\" }]";

#[test]
fn an_inconclusive_check_needs_its_allowance_and_a_failing_one_never_has_one() {
    // Not listed: the run fails.
    let s = Setup::new();
    s.tools.lec_inconclusive.lock().unwrap().push("add_i8".into());
    assert!(!s.run().ok);

    // Listed: the run and verify pass, the row records the reason.
    let s = Setup::new();
    s.tools.lec_inconclusive.lock().unwrap().push("add_i8".into());
    s.sets.lock().unwrap().push(ALLOW_ADD_I8.into());
    let sum = s.run();
    assert!(sum.ok, "{sum:?}");
    let row = s.library().rows.into_iter().find(|r| r.key == "add@i8").unwrap();
    assert_eq!(row.allowed.as_deref(), Some("stand-in"));
    assert!(!row.lec.equivalent);
    let failing = s.failing();
    assert!(failing.is_empty(), "{failing:?}");
    let checks = s.pipeline().verify_pass("characterize").unwrap();
    let lec = checks
        .checks
        .iter()
        .find(|c| c.name == "nangate45:clock:lec")
        .unwrap();
    assert_eq!(lec.evidence.counts["allowed_inconclusive"], 1);

    // Removing the allowance afterwards fails verify (and the stored reason no longer matches).
    s.sets.lock().unwrap().clear();
    s.expect("nangate45:clock:lec");

    // A failing compare point is never allowed.
    let s = Setup::new();
    s.tools.lec_fails.lock().unwrap().push("add_i8".into());
    s.sets.lock().unwrap().push(ALLOW_ADD_I8.into());
    assert!(!s.run().ok);
}
