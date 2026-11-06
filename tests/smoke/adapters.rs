//! The adapter registries (design §3.7; docs/framework/adapters.md): every family's names are unique and
//! its capabilities declared, the SMT solvers give the same verdicts on the same queries (and exactly the
//! declared ones accept a float sort of any width), every synthesis flow writes the per-design files the
//! pass reads, the declared families refuse what they do not implement yet, and every float IP provider
//! offers well-formed instances within its documented widths under the documented contracts.

use std::collections::BTreeSet;
use std::path::Path;

use cc_core::adapters::{
    Accuracy, CoreOffer, CoreOp, CoreSource, Encoding, Generator, IpProvider, Subnormals, Timing,
};
use cc_core::adapters::{Design, Kit, LecOptions, LibraryFormat, SynthJob};
use cc_core::runner::{BoundRunner, Caps, Runners, ToolLog};
use cc_core::{CancelToken, CoreError};
use cc_passes::adapters::{
    CLIQUE_SOLVERS, HLS_FRONTENDS, IP_PROVIDERS, MINERS, MIS_SOLVERS, PNR_FLOWS, SMT_SOLVERS, SYNTH_FLOWS,
};
use cc_smt::{Location, Script, Solver, Verdict};
use tests::{Env, repo_root};

fn unique(names: Vec<&'static str>) {
    let set: BTreeSet<&str> = names.iter().copied().collect();
    assert_eq!(set.len(), names.len(), "{names:?}");
}

#[test]
fn every_family_has_unique_names_and_declared_capabilities() {
    unique(SYNTH_FLOWS.names());
    unique(SMT_SOLVERS.names());
    unique(CLIQUE_SOLVERS.names());
    unique(MIS_SOLVERS.names());
    unique(MINERS.names());
    unique(HLS_FRONTENDS.names());
    unique(IP_PROVIDERS.names());
    assert_eq!(SYNTH_FLOWS.names(), ["dc", "yosys"]);
    assert_eq!(
        IP_PROVIDERS.names(),
        ["designware", "hardfloat", "flopoco", "xls_math"]
    );
    assert_eq!(SMT_SOLVERS.names(), ["bitwuzla", "z3"]);
    assert_eq!(MINERS.names(), ["grami"]);
    // Two flows: the licensed industry flow on `.db`, the open one on Liberty.
    let dc = SYNTH_FLOWS.by_name("dc").unwrap().capabilities();
    let yosys = SYNTH_FLOWS.by_name("yosys").unwrap().capabilities();
    assert!(dc.licensed && dc.library == LibraryFormat::Db && dc.equivalence == "formality");
    assert!(!yosys.licensed && yosys.library == LibraryFormat::Liberty && yosys.equivalence == "yosys-equiv");
    let e = SYNTH_FLOWS.by_name("genus").err().unwrap().to_string();
    assert!(
        e.contains("synthesis flow `genus` is not available (available: dc, yosys)"),
        "{e}"
    );
    // Exactly one way to answer each selection: at least one exact algorithm and one heuristic.
    for caps in [
        CLIQUE_SOLVERS
            .iter()
            .map(|s| s.capabilities())
            .collect::<Vec<_>>(),
        MIS_SOLVERS.iter().map(|s| s.capabilities()).collect(),
    ] {
        assert!(caps.iter().any(|c| c.exact) && caps.iter().any(|c| !c.exact));
    }
    let g = MINERS.by_name("grami").unwrap().capabilities();
    assert!(g.directed && g.vertex_bound && g.mni_support);
    // The lemma solver declares the lemmas and float sorts of any width.
    let lemma = cc_smt::lemma_solver().adapter().capabilities();
    assert!(lemma.lemmas && lemma.any_float_sort);
}

#[test]
fn declared_families_refuse_what_they_do_not_implement() {
    let allo = HLS_FRONTENDS.by_name("allo").unwrap();
    assert!(!allo.capabilities().implemented && allo.capabilities().targets == ["xls"]);
    let env = Env::new();
    let lc = env.config("[models]\nkeys = []\n", &[]);
    let runner = BoundRunner::new(
        Runners::from_config(&lc).host,
        Caps {
            memory_gb: 1.0,
            cpus: 1.0,
            timeout_s: 10.0,
            pids_limit: 16,
        },
        ToolLog::new("adapters", env.root().join("logs")),
        CancelToken::new(),
    );
    let e = allo.generate(&runner, env.root(), "fu0", env.root()).unwrap_err();
    assert!(
        matches!(e, CoreError::Tool { .. }) && e.to_string().contains("not implemented yet"),
        "{e}"
    );
    assert!(PNR_FLOWS.names().is_empty());
    let e = PNR_FLOWS.by_name("innovus").err().unwrap().to_string();
    assert!(e.contains("none yet"), "{e}");
}

fn kit() -> Kit {
    Kit {
        name: "k".into(),
        corner: "typical".into(),
        lib_name: "lib".into(),
        liberty: "/pdk/k/lib.lib".into(),
        db: "/env/out/pdk_db/lib.db".into(),
        time_unit_ps: 1000.0,
        driving_cell: "BUF".into(),
        load: 3.0,
        cells: BTreeSet::new(),
        latches: BTreeSet::new(),
        flops: BTreeSet::new(),
    }
}

/// Every flow's scripts bracket each design with its markers and write the files the pass reads; the
/// equivalence check appears exactly when it is asked for.
#[test]
fn every_synthesis_flow_writes_what_the_pass_reads() {
    let kit = kit();
    let designs = [
        Design {
            id: "add_i8".into(),
            files: vec!["add_i8.v".into()],
            clocked: false,
            libraries: Vec::new(),
        },
        Design {
            id: "register_w8".into(),
            files: vec!["register_w8.v".into()],
            clocked: true,
            libraries: Vec::new(),
        },
    ];
    let lec = LecOptions {
        timeout_min: 2,
        strategies: Vec::new(),
    };
    for flow in SYNTH_FLOWS.iter() {
        for with_lec in [false, true] {
            let job = SynthJob {
                kit: &kit,
                designs: &designs,
                period_ps: 1000.0,
                caps: Caps {
                    memory_gb: 4.0,
                    cpus: 2.0,
                    timeout_s: 60.0,
                    pids_limit: 64,
                },
                cache_dir: Path::new("/b/.characterize/alib"),
                lec: with_lec.then_some(&lec),
            };
            let synth: String = flow.synth_scripts(&job).into_iter().map(|(_, t)| t).collect();
            let check: String = flow.lec_scripts(&job).into_iter().map(|(_, t)| t).collect();
            for d in &designs {
                let id = &d.id;
                for needle in [
                    format!("@@begin {id}"),
                    format!("@@end {id}"),
                    format!("{id}.net.v"),
                    format!("{id}.area.rpt"),
                ] {
                    assert!(synth.contains(&needle), "{} {needle}", flow.name());
                }
                let lec_rpt = format!("{id}.lec.rpt");
                assert_eq!(
                    synth.contains(&lec_rpt) || check.contains(&lec_rpt),
                    with_lec,
                    "{} lec {with_lec}",
                    flow.name()
                );
            }
        }
    }
}

fn query(name: &str, decls: &[(&str, &str)], assertion: &str) -> (String, String) {
    let mut s = Script::new();
    s.comment(name);
    for (n, sort) in decls {
        s.constant(n, sort);
    }
    s.assert(assertion);
    (format!("{name}.smt2"), s.render())
}

/// The same queries to every solver: the same verdict; a float sort of width (8, 8) is answered by the
/// solvers that declare `any_float_sort` and refused (never proven) by the others.
#[test]
fn smt_solvers_conform() {
    let bv = "(_ BitVec 8)";
    let common = [
        query(
            "bv_comm",
            &[("a", bv), ("b", bv)],
            "(not (= (bvadd a b) (bvadd b a)))",
        ),
        query("bv_sat", &[("a", bv)], "(= (bvadd a #x01) #x00)"),
        query(
            "fp_comm",
            &[("x", "Float32"), ("y", "Float32")],
            "(not (= (fp.add RNE x y) (fp.add RNE y x)))",
        ),
        query("fp_sat", &[("x", "Float32")], "(fp.isSubnormal x)"),
    ];
    let mut uf = Script::new();
    uf.comment("uf_congruence")
        .constant("a", bv)
        .constant("b", bv)
        .fun("f", &[bv.to_string()], bv)
        .assert("(and (= a b) (not (= (f a) (f b))))");
    let any = query(
        "any_sort",
        &[("x", "(_ FloatingPoint 8 8)")],
        "(not (= (fp.add RNE x x) (fp.mul RNE x ((_ to_fp 8 8) RNE 2.0))))",
    );
    let cfg: toml::Value =
        toml::from_str(&std::fs::read_to_string(repo_root().join("config.toml")).unwrap()).unwrap();
    let tag = cfg["tools"]["docker"]["tag"].as_str().unwrap().to_string();
    let host = std::env::var("CC_SMT_RUNNER").as_deref() == Ok("host");
    let mut sets = vec![format!("tools.docker.tag={tag}")];
    if host {
        sets.push("tools.smt.runner=host".into());
    }
    let sets: Vec<&str> = sets.iter().map(String::as_str).collect();
    let env = Env::new();
    let lc = env.config("[models]\nkeys = []\n", &sets);
    let dir = env.root().join("smt");
    std::fs::create_dir_all(&dir).unwrap();
    let mut files = Vec::new();
    let all: Vec<(String, String)> = common
        .iter()
        .cloned()
        .chain([("uf_congruence.smt2".to_string(), uf.render()), any.clone()])
        .collect();
    for (f, t) in &all {
        std::fs::write(dir.join(f), t).unwrap();
        files.push(f.clone());
    }
    let runners = Runners::from_config(&lc);
    let c = &lc.config.tools.smt;
    let (r, loc) = if host {
        (runners.host, cc_smt::host_paths(&c.z3, &c.bitwuzla))
    } else {
        (runners.docker, Location::Image)
    };
    let runner = BoundRunner::new(
        r,
        Caps {
            memory_gb: 2.0,
            cpus: 1.0,
            timeout_s: 300.0,
            pids_limit: 64,
        },
        ToolLog::new("adapters-smt", dir.join("logs")),
        CancelToken::new(),
    );
    for solver in Solver::ALL {
        assert_eq!(SMT_SOLVERS.by_name(solver.name()).unwrap().name(), solver.name());
        let b = cc_smt::run_batch(&runner, &loc, solver, &dir, &files, 60).unwrap();
        let v = |f: &str| b.verdicts.get(f).cloned();
        for f in ["bv_comm.smt2", "fp_comm.smt2", "uf_congruence.smt2"] {
            assert_eq!(v(f), Some(Verdict::Unsat), "{} {f}", solver.name());
        }
        for f in ["bv_sat.smt2", "fp_sat.smt2"] {
            assert!(
                matches!(v(f), Some(Verdict::Sat { .. })),
                "{} {f}: {:?}",
                solver.name(),
                v(f)
            );
        }
        let declared = solver.adapter().capabilities().any_float_sort;
        let got = v("any_sort.smt2");
        if declared {
            assert_eq!(got, Some(Verdict::Unsat), "{}", solver.name());
        } else {
            assert!(
                matches!(got, Some(Verdict::Error { .. })),
                "{}: {got:?}",
                solver.name()
            );
        }
    }
}

// -- IpProvider conformance ---------------------------------------------------------------------------

/// Every core op a provider may be asked for, with modifiers.
fn probe_ops() -> Vec<String> {
    let mut v: Vec<String> = [
        "add",
        "sub",
        "mul",
        "div",
        "sqrt",
        "rsqrt",
        "exp",
        "exp2",
        "ln",
        "log2",
        "sin",
        "cos",
        "pow",
        "compare",
        "from_ieee",
        "to_ieee",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for r in ["rne", "rtz", "floor", "ceil"] {
        v.push(format!("to_int.{r}.s32"));
    }
    v.extend(
        [
            "to_int.rtz.u8",
            "to_int.rtz.s3",
            "from_int.s32",
            "from_int.u16",
            "from_int.s4",
            "convert.e5m10",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    v.extend(
        ["convert.e8m7", "convert.e5m2", "convert.e8m23"]
            .iter()
            .map(|s| s.to_string()),
    );
    v
}

/// Formats probed: the catalog's, wide inner formats of compositions, and the edges of the documented
/// ranges.
const PROBE_FORMATS: &[(u32, u32)] = &[
    (8, 23),
    (5, 10),
    (8, 7),
    (5, 2),
    (4, 3),
    (11, 52),
    (8, 33),
    (3, 2),
    (3, 4),
    (3, 5),
    (5, 11),
    (5, 12),
    (31, 253),
    (8, 29),
    (8, 30),
    (8, 32),
    (8, 57),
    (8, 58),
    (8, 59),
    (2, 2),
    (8, 1),
    (32, 23),
    (8, 254),
];

fn is_const(s: &str) -> bool {
    let Some((w, v)) = s.split_once('\'') else {
        return false;
    };
    !w.is_empty()
        && w.chars().all(|c| c.is_ascii_digit())
        && v.len() > 1
        && matches!(v.as_bytes()[0], b'b' | b'd' | b'h')
        && v[1..].chars().all(|c| c.is_ascii_hexdigit() || c == '_')
}

/// An offer is well formed: a Verilog module name, the requested widths, every operand and result
/// connected exactly once, nothing but operands, results, constants or the handshake of an iterative
/// core, a source the provider's flows can use.
fn well_formed(p: &dyn IpProvider, op: &str, e: u32, s: u32, o: &CoreOffer) {
    let what = format!("{} {op} ({e}, {s})", p.name());
    let core: CoreOp = op.parse().unwrap();
    let caps = p.capabilities();
    assert!(caps.ops.contains(&core.family()), "{what}: op not declared");
    let m = &o.instance.module;
    assert!(
        !m.is_empty()
            && m.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && !m.starts_with(|c: char| c.is_ascii_digit()),
        "{what}: module {m:?}"
    );
    assert_eq!((o.instance.exp_width, o.instance.sig_width), (e, s), "{what}");
    let operands: Vec<String> = (0..core.arity()).map(|i| format!("a{i}")).collect();
    let mut results: Vec<&str> = core.results().to_vec();
    if core == CoreOp::Compare {
        results.push("unordered");
    }
    let handshake = ["clk", "rst", "start", "done"];
    for (port, to) in &o.instance.ports {
        let ok = to.is_empty()
            || operands.contains(to)
            || results.contains(&to.as_str())
            || is_const(to)
            || (matches!(o.timing, Timing::Iterative { .. }) && handshake.contains(&to.as_str()));
        assert!(ok, "{what}: port {port} -> {to:?}");
    }
    let count = |x: &str| o.instance.ports.values().filter(|v| *v == x).count();
    for a in &operands {
        assert_eq!(count(a), 1, "{what}: operand {a}");
    }
    for r in core.results() {
        assert_eq!(count(r), 1, "{what}: result {r}");
    }
    if let Timing::Iterative { max_cycles } = o.timing {
        assert!(
            max_cycles > 0 && handshake.iter().all(|h| count(h) == 1),
            "{what}"
        );
    }
    match &o.source {
        CoreSource::Library { name } => {
            assert_eq!(name, "designware", "{what}");
            assert_eq!(
                caps.flows,
                ["dc"],
                "{what}: a vendor library only in the flow that links it"
            );
        }
        CoreSource::Generated { generator } => match generator {
            Generator::Hardfloat { args } | Generator::Flopoco { args } => {
                assert!(
                    !args.is_empty() && !args.iter().any(|a| a.contains("name=") || a == "--module"),
                    "{what}"
                )
            }
            Generator::XlsDslx { file, top } => assert!(file.ends_with(".x") && !top.is_empty(), "{what}"),
        },
    }
    assert!(
        o.model.starts_with("ieee.") || o.model.starts_with("reference.") || o.model.starts_with("flopoco."),
        "{what}: {}",
        o.model
    );
    // Exact ops are correctly rounded (design §6.4).
    if core.exact() {
        assert_eq!(o.contract.accuracy, Accuracy::CorrectlyRounded, "{what}");
    }
}

/// The documented ranges (DesignWare datasheets DWBB_202409.0; HardFloat Release 1 section 2; FloPoCo's
/// unit-test widths; XLS f32 only): does the provider offer `op` at (e, s)?
fn documented(provider: &str, op: &str, e: u32, s: u32) -> bool {
    let core: CoreOp = op.parse().unwrap();
    let dw = |max_sig: u32| (3..=31).contains(&e) && (2..=max_sig).contains(&s);
    let hf_fmt =
        |e: u32, s: u32| (3..=32).contains(&e) && s + 1 >= 3 && u64::from(s + 1) <= (1u64 << (e - 2)) + 3;
    let fl = (5..=11).contains(&e) && (5..=52).contains(&s);
    let fl_exp = fl && u64::from(s) + 4 < (1u64 << (e - 1));
    match provider {
        "designware" => match core {
            CoreOp::Add
            | CoreOp::Sub
            | CoreOp::Mul
            | CoreOp::Div
            | CoreOp::Sqrt
            | CoreOp::Rsqrt
            | CoreOp::Compare => dw(253),
            CoreOp::Exp => dw(57),
            CoreOp::Exp2 => dw(58),
            CoreOp::Ln | CoreOp::Log2 => dw(29),
            CoreOp::Sin | CoreOp::Cos => dw(32),
            CoreOp::ToInt { signed, width, .. } => dw(253) && signed && (3..=512).contains(&width),
            CoreOp::FromInt { signed, width } => dw(253) && (3 + u32::from(signed)..=512).contains(&width),
            _ => false,
        },
        "hardfloat" => match core {
            CoreOp::Add | CoreOp::Sub | CoreOp::Mul | CoreOp::Div | CoreOp::Sqrt | CoreOp::Compare => {
                hf_fmt(e, s)
            }
            CoreOp::ToInt { width, .. } | CoreOp::FromInt { width, .. } => {
                hf_fmt(e, s) && (2..=256).contains(&width)
            }
            CoreOp::Convert { exp_width, sig_width } => {
                hf_fmt(e, s) && hf_fmt(exp_width, sig_width) && (exp_width, sig_width) != (e, s)
            }
            _ => false,
        },
        "flopoco" => match core {
            CoreOp::Exp | CoreOp::Pow => fl_exp,
            CoreOp::Ln | CoreOp::FromIeee | CoreOp::ToIeee => fl,
            _ => false,
        },
        "xls_math" => matches!(core, CoreOp::Sin | CoreOp::Cos) && (e, s) == (8, 23),
        _ => unreachable!(),
    }
}

/// The documented contract of `op` from `provider`.
fn documented_contract(provider: &str, core: CoreOp) -> (Accuracy, Subnormals, Encoding) {
    match (provider, core) {
        ("designware", CoreOp::Exp | CoreOp::Exp2 | CoreOp::Log2) => {
            (Accuracy::Ulp { ulp: 2.0 }, Subnormals::Ieee, Encoding::Ieee)
        }
        ("designware", CoreOp::Ln) => (Accuracy::Ulp { ulp: 1.0 }, Subnormals::Ieee, Encoding::Ieee),
        ("designware", CoreOp::Sin | CoreOp::Cos) => (Accuracy::Unbounded, Subnormals::Ieee, Encoding::Ieee),
        ("designware" | "hardfloat", _) => (Accuracy::CorrectlyRounded, Subnormals::Ieee, Encoding::Ieee),
        ("flopoco", CoreOp::FromIeee | CoreOp::ToIeee) => {
            (Accuracy::CorrectlyRounded, Subnormals::Flush, Encoding::Flopoco)
        }
        ("flopoco", _) => (Accuracy::Faithful, Subnormals::Flush, Encoding::Flopoco),
        ("xls_math", _) => (
            Accuracy::Absolute {
                bound: 2f64.powi(-22),
            },
            Subnormals::Flush,
            Encoding::Ieee,
        ),
        _ => unreachable!(),
    }
}

#[test]
fn ip_providers_conform() {
    let mut offered = 0;
    for p in IP_PROVIDERS.iter() {
        let caps = p.capabilities();
        assert!(
            !caps.flows.is_empty() && !caps.licence.is_empty() && !caps.ops.is_empty(),
            "{}",
            p.name()
        );
        assert!(
            caps.flows.iter().all(|f| SYNTH_FLOWS.by_name(f).is_ok()),
            "{}",
            p.name()
        );
        for op in probe_ops() {
            for &(e, s) in PROBE_FORMATS {
                let o = p.core(&op, e, s);
                assert_eq!(
                    o.is_some(),
                    documented(p.name(), &op, e, s),
                    "{} {op} ({e}, {s}): offered {}",
                    p.name(),
                    o.is_some()
                );
                let Some(o) = o else { continue };
                offered += 1;
                well_formed(p, &op, e, s, &o);
                let core: CoreOp = op.parse().unwrap();
                let (acc, sub, enc) = documented_contract(p.name(), core);
                assert_eq!(
                    (&o.contract.accuracy, o.contract.subnormals, o.encoding),
                    (&acc, sub, enc),
                    "{} {op}",
                    p.name()
                );
                // The DesignWare instances are IEEE compliant (the flushing formats wrap them in glue);
                // `DW_fp_i2flt` has no such parameter (an integer operand has no special values).
                if p.name() == "designware" {
                    let want = (o.instance.module != "DW_fp_i2flt").then_some("1");
                    assert_eq!(
                        o.instance.params.get("ieee_compliance").map(String::as_str),
                        want,
                        "{op}"
                    );
                    assert_eq!(o.instance.params["sig_width"], s.to_string());
                    assert_eq!(o.instance.params["exp_width"], e.to_string());
                }
                // A generated module is named per op and widths: two different requests never share one.
                if let CoreSource::Generated { .. } = o.source {
                    assert!(
                        o.instance.module.contains(&format!("e{e}m{s}")),
                        "{}",
                        o.instance.module
                    );
                }
            }
        }
        // Nonsense is refused, not guessed.
        assert!(p.core("tanh", 8, 23).is_none() && p.core("add.rne", 8, 23).is_none());
    }
    assert!(offered > 300, "{offered}");
    // The DesignWare logarithms keep their bound for every input: `extra_prec = sig_width`.
    let ln = IP_PROVIDERS
        .by_name("designware")
        .unwrap()
        .core("ln", 8, 23)
        .unwrap();
    assert_eq!(ln.instance.params["extra_prec"], "23");
    // HardFloat's divider is iterative; its rounding ports follow HardFloat's encoding (floor 2, ceil 3).
    let hf = IP_PROVIDERS.by_name("hardfloat").unwrap();
    assert!(matches!(
        hf.core("div", 8, 23).unwrap().timing,
        Timing::Iterative { max_cycles: 26 }
    ));
    assert_eq!(
        hf.core("to_int.floor.s32", 8, 23).unwrap().instance.ports["rnd"],
        "3'b010"
    );
    let dw = IP_PROVIDERS.by_name("designware").unwrap();
    assert_eq!(
        dw.core("to_int.floor.s32", 8, 23).unwrap().instance.ports["rnd"],
        "3'b011"
    );
    // The default providers of each flow serve it.
    let cfg = cc_passes::characterize::CharacterizeConfig::default();
    for flow in ["dc", "yosys"] {
        assert!(!cfg.providers_of(flow).unwrap().is_empty(), "{flow}");
    }
}
