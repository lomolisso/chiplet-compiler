//! The adapter registries (design §3.7; docs/framework/adapters.md): every family's names are unique and
//! its capabilities declared, the SMT solvers give the same verdicts on the same queries (and exactly the
//! declared ones accept a float sort of any width), every synthesis flow writes the per-design files the
//! pass reads, and the declared families refuse what they do not implement yet.

use std::collections::BTreeSet;
use std::path::Path;

use cc_core::adapters::{Design, Kit, LecOptions, LibraryFormat, SynthJob};
use cc_core::runner::{BoundRunner, Caps, Runners, ToolLog};
use cc_core::{CancelToken, CoreError};
use cc_passes::adapters::{
    CLIQUE_SOLVERS, HLS_FRONTENDS, MINERS, MIS_SOLVERS, PNR_FLOWS, SMT_SOLVERS, SYNTH_FLOWS,
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
    assert_eq!(SYNTH_FLOWS.names(), ["dc", "yosys"]);
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
        },
        Design {
            id: "register_w8".into(),
            files: vec!["register_w8.v".into()],
            clocked: true,
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
