//! Unit tests of the merge algorithms on hand-built units and a small fixture library.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Duration;

use super::agglomerate::{Ctx, agglomerate};
use super::clique::{feasible, solve};
use super::compat::model;
use super::fu::{Fu, cost, join_type, topo};
use super::library::Lib;
use super::rebuild::rebuild;
use super::unit::{UNode, Unit};
use crate::characterize::catalog::Kind;
use crate::characterize::json::{Library, Row, TargetInfo};
use crate::eda::parse::Measured;
use crate::merge::config::{CliqueAlgorithm, MergeConfig};
use crate::merge::json::BlockKind;
use crate::mine::json::Family;

fn row(key: &str, target: &str, area: f64, delay: f64) -> Row {
    Row {
        key: key.into(),
        id: key.into(),
        kind: Kind::Op,
        op: String::new(),
        variant: None,
        family: String::new(),
        types: Vec::new(),
        width: 0,
        kit: "k".into(),
        corner: String::new(),
        target: target.into(),
        period_ps: 1000.0,
        measured: Measured {
            area_um2: Some(area),
            seq_area_um2: None,
            delay_ps: Some(delay),
            slack_ps: None,
            leakage_nw: None,
            children: BTreeMap::new(),
        },
        estimator: String::new(),
        lec_tool: None,
        lec: None,
        inconclusive: false,
        designware: false,
        provisional: false,
        netlist: String::new(),
        netlist_hash: String::new(),
        reports: Vec::new(),
    }
}

/// Adders 10 (i8) / 40 (i32), multipliers 50 / 400, subtract like add, muxes 4 per input of 32 bits.
pub fn lib() -> Lib {
    let mut rows = Vec::new();
    for (k, a, d) in [
        ("add@i8", 10.0, 50.0),
        ("add@i32", 40.0, 120.0),
        ("subtract@i32", 42.0, 120.0),
        ("multiply@i8", 50.0, 150.0),
        ("multiply@i32", 400.0, 400.0),
        ("maximum@i32", 30.0, 100.0),
    ] {
        rows.push(row(k, "clock", a, d));
        rows.push(row(k, "fast", a, d));
    }
    for n in 2..=8u32 {
        for w in [1u32, 8, 32] {
            let a = f64::from(n - 1) * f64::from(w) / 8.0;
            rows.push(row(&format!("mux{n}@w{w}"), "clock", a, 20.0));
            rows.push(row(&format!("mux{n}@w{w}"), "fast", a, 20.0));
        }
    }
    for w in [1u32, 8, 32] {
        rows.push(row(&format!("register@w{w}"), "clock", f64::from(w), 0.0));
    }
    let l = Library {
        schema: String::new(),
        tool: String::new(),
        lec_tool: None,
        tool_versions: BTreeMap::new(),
        kits: Vec::new(),
        targets: vec![TargetInfo {
            kit: "k".into(),
            target: "clock".into(),
            period_ps: 1000.0,
        }],
        rows,
    };
    Lib::new(&l, "k").expect("library")
}

fn input(name: &str, t: &str) -> UNode {
    UNode {
        name: format!("in:{name}"),
        kind: BlockKind::Input,
        op: "input".into(),
        attrs: BTreeMap::new(),
        var: Some(name.into()),
        args: Vec::new(),
        commutative: false,
        sigs: vec![vec![t.into()]],
    }
}

fn op(id: u32, op: &str, args: &[u32], t: &str, commutative: bool) -> UNode {
    let mut sig: Vec<String> = args.iter().map(|_| t.to_string()).collect();
    sig.push(t.into());
    UNode {
        name: format!("v{id}"),
        kind: BlockKind::Op,
        op: op.into(),
        attrs: BTreeMap::new(),
        var: None,
        args: args.iter().map(|&a| (a, 0)).collect(),
        commutative,
        sigs: vec![sig],
    }
}

fn output(name: &str, src: u32, t: &str) -> UNode {
    UNode {
        name: format!("out:{name}"),
        kind: BlockKind::Output,
        op: "output".into(),
        attrs: BTreeMap::new(),
        var: Some(name.into()),
        args: vec![(src, 0)],
        commutative: false,
        sigs: vec![vec![t.into()]],
    }
}

fn unit(id: &str, nodes: Vec<UNode>) -> Unit {
    Unit {
        id: id.into(),
        class: id.into(),
        family: Family::Map,
        tuples: vec!["t0".into()],
        nodes,
        groups: Vec::new(),
    }
}

/// `out = (a * b) + c` at type `t`.
fn mac(id: &str, t: &str) -> Unit {
    unit(
        id,
        vec![
            input("a", t),
            input("b", t),
            input("c", t),
            op(0, "multiply", &[0, 1], t, true),
            op(1, "add", &[3, 2], t, true),
            output("out0", 4, t),
        ],
    )
}

/// `out = (a + b) * c`.
fn add_mul(id: &str, t: &str) -> Unit {
    unit(
        id,
        vec![
            input("a", t),
            input("b", t),
            input("c", t),
            op(0, "add", &[0, 1], t, true),
            op(1, "multiply", &[3, 2], t, true),
            output("out0", 4, t),
        ],
    )
}

fn merge_pair(us: &[Unit], alg: CliqueAlgorithm) -> (Fu, f64) {
    let lib = lib();
    let a = Fu::single(&us[0], 0).expect("unit");
    let b = Fu::single(&us[1], 1).expect("unit");
    let (ca, cb) = (cost(&a, us, &lib), cost(&b, us, &lib));
    let m = model(&a, &ca, &b, &cb, &lib, &mut HashMap::new());
    let s = solve(&m, alg, &[], 1_000_000, Duration::from_secs(60), 100_000);
    assert!(feasible(&m, &s.chosen, &[]), "{alg:?}: infeasible");
    let fu = rebuild(&a, &b, &m, &s.chosen, &a.host);
    let c = cost(&fu, us, &lib);
    (fu, ca.area + cb.area - c.area)
}

#[test]
fn integers_join_by_signedness() {
    assert_eq!(join_type("i8", "i32", true).as_deref(), Some("i32"));
    assert_eq!(join_type("ui8", "i32", true), None);
    assert_eq!(join_type("i8", "i32", false), None);
    assert_eq!(join_type("f32", "bf16", true), None);
}

#[test]
fn two_macs_share_everything() {
    let us = vec![mac("z0", "i32"), mac("z1", "i32")];
    let (fu, gain) = merge_pair(&us, CliqueAlgorithm::ExactIlp);
    assert_eq!(fu.blocks.len(), 6, "one copy of every block");
    assert!((gain - 440.0).abs() < 1e-6, "gain {gain}");
}

#[test]
fn mac_and_add_mul_share_the_operators_with_muxes() {
    let us = vec![mac("z0", "i32"), add_mul("z1", "i32")];
    for alg in [CliqueAlgorithm::ExactIlp, CliqueAlgorithm::Ostergard] {
        let (fu, gain) = merge_pair(&us, alg);
        // Sharing both operators would close a loop (multiply -> add in one, add -> multiply in the
        // other): only the multiply, the larger saving, is shared.
        let ops = fu.blocks.iter().filter(|b| b.kind == BlockKind::Op).count();
        assert_eq!(ops, 3, "{alg:?}: one operator shared");
        assert_eq!(
            fu.blocks.iter().filter(|b| b.op == "multiply").count(),
            1,
            "{alg:?}"
        );
        let lib = lib();
        let c = cost(&fu, &us, &lib);
        assert!(topo(&fu, &c.wiring).is_ok(), "{alg:?}: acyclic");
        assert!(
            !c.muxes.is_empty(),
            "{alg:?}: the shared multiply needs multiplexers"
        );
        assert!(gain > 0.0, "{alg:?}: gain {gain}");
    }
}

#[test]
fn exact_strategies_agree_and_greedy_is_feasible() {
    let us = vec![mac("z0", "i32"), add_mul("z1", "i32")];
    let lib = lib();
    let a = Fu::single(&us[0], 0).expect("unit");
    let b = Fu::single(&us[1], 1).expect("unit");
    let (ca, cb) = (cost(&a, &us, &lib), cost(&b, &us, &lib));
    let m = model(&a, &ca, &b, &cb, &lib, &mut HashMap::new());
    let t = Duration::from_secs(60);
    let x = solve(&m, CliqueAlgorithm::ExactIlp, &[], 1_000_000, t, 100_000);
    let y = solve(&m, CliqueAlgorithm::Ostergard, &[], 1_000_000, t, 100_000);
    let g = solve(&m, CliqueAlgorithm::Greedy, &[], 1_000_000, t, 100_000);
    assert!(x.optimal && y.optimal);
    assert!(
        (x.objective - y.objective).abs() < 1e-6,
        "{} vs {}",
        x.objective,
        y.objective
    );
    assert!(feasible(&m, &g.chosen, &[]));
    assert!(g.objective <= x.objective + 1e-6);
}

/// Conformance of every registered clique solver (design §3.7): on the same compatibility graphs, each
/// returns a feasible clique; the exact ones the same optimum, claimed optimal; a heuristic at most it.
#[test]
fn every_registered_clique_solver_conforms() {
    use super::clique::SOLVERS;
    use cc_core::adapters::Limits;
    assert_eq!(
        SOLVERS.names(),
        [
            CliqueAlgorithm::ExactIlp,
            CliqueAlgorithm::Ostergard,
            CliqueAlgorithm::Greedy
        ]
        .map(CliqueAlgorithm::name)
    );
    let lib = lib();
    let pairs = [
        (mac("z0", "i32"), mac("z1", "i32")),
        (mac("z0", "i32"), add_mul("z1", "i32")),
        (add_mul("z0", "i8"), add_mul("z1", "i32")),
        (mac("z0", "i32"), mac("z1", "i8")),
        (add_mul("z0", "i8"), mac("z1", "i8")),
    ];
    let limits = Limits {
        node_budget: 1_000_000,
        timeout: Some(Duration::from_secs(60)),
    };
    for (k, (u0, u1)) in pairs.into_iter().enumerate() {
        let us = vec![u0, u1];
        let a = Fu::single(&us[0], 0).expect("unit");
        let b = Fu::single(&us[1], 1).expect("unit");
        let (ca, cb) = (cost(&a, &us, &lib), cost(&b, &us, &lib));
        let m = model(&a, &ca, &b, &cb, &lib, &mut HashMap::new());
        let mut best: Option<f64> = None;
        for s in SOLVERS.iter().filter(|s| s.capabilities().exact) {
            let r = s.solve(&m, &[], &limits).expect("an answer");
            assert!(r.optimal && feasible(&m, &r.chosen, &[]), "{} pair {k}", s.name());
            let o = *best.get_or_insert(r.objective);
            assert!(
                (o - r.objective).abs() < 1e-6,
                "{} pair {k}: {} vs {o}",
                s.name(),
                r.objective
            );
        }
        let best = best.expect("an exact solver");
        for s in SOLVERS.iter().filter(|s| !s.capabilities().exact) {
            let r = s.solve(&m, &[], &limits).expect("an answer");
            assert!(
                feasible(&m, &r.chosen, &[]) && r.objective <= best + 1e-6,
                "{} pair {k}",
                s.name()
            );
        }
    }
}

#[test]
fn an_int8_add_widens_into_an_int32_adder() {
    let us = vec![mac("z0", "i32"), mac("z1", "i8")];
    let (fu, gain) = merge_pair(&us, CliqueAlgorithm::ExactIlp);
    assert!(fu.blocks.iter().all(|b| b.types.iter().all(|t| t == "i32")));
    // Saved: the i8 multiply and add (60) and nothing else (ports are free).
    assert!((gain - 60.0).abs() < 1e-6, "gain {gain}");
}

/// The four-vertex cycle the pairwise order constraint misses: A has chains p→q and r→s, B has q'→r'
/// and s'→p'. Every pair of mappings is order-consistent, the merged graph is a cycle.
#[test]
fn a_long_cycle_is_cut_by_a_nogood() {
    let t = "i32";
    let ua = unit(
        "z0",
        vec![
            input("a", t),
            op(0, "add", &[0, 0], t, true),
            op(1, "multiply", &[1, 0], t, true),
            op(2, "maximum", &[0, 0], t, true),
            op(3, "subtract", &[3, 0], t, false),
            output("o0", 2, t),
            output("o1", 4, t),
        ],
    );
    let ub = unit(
        "z1",
        vec![
            input("a", t),
            op(0, "multiply", &[0, 0], t, true),
            op(1, "maximum", &[1, 0], t, true),
            op(2, "subtract", &[0, 0], t, false),
            op(3, "add", &[3, 0], t, true),
            output("o0", 2, t),
            output("o1", 4, t),
        ],
    );
    let us = vec![ua, ub];
    let lib = lib();
    let cfg = MergeConfig {
        max_logic_depth: 1e9,
        ..MergeConfig::default()
    };
    let embeds: HashSet<(String, String)> = [("z1".to_string(), "z0".to_string())].into_iter().collect();
    let cx = Ctx {
        units: &us,
        lib: &lib,
        cfg: &cfg,
        embeds: &embeds,
    };
    let ag = agglomerate(&cx, &[0, 1]).expect("agglomerate");
    for (f, c) in ag.fus.iter().zip(&ag.costs) {
        assert!(topo(f, &c.wiring).is_ok(), "acyclic");
    }
    let s = ag.steps.first().expect("the two units merge");
    assert!(
        s.clique.rounds >= 2,
        "a no-good was needed (rounds {})",
        s.clique.rounds
    );
}

#[test]
fn agglomeration_stops_when_nothing_is_saved() {
    let t = "i32";
    let us = vec![
        mac("z0", t),
        mac("z1", t),
        mac("z2", t),
        unit(
            "z3",
            vec![
                input("a", t),
                op(0, "maximum", &[0, 0], t, true),
                output("o", 1, t),
            ],
        ),
    ];
    let lib = lib();
    let cfg = MergeConfig {
        max_logic_depth: 1e9,
        ..MergeConfig::default()
    };
    let embeds: HashSet<(String, String)> = ["z1", "z2", "z3"]
        .iter()
        .map(|g| (g.to_string(), "z0".to_string()))
        .collect();
    let cx = Ctx {
        units: &us,
        lib: &lib,
        cfg: &cfg,
        embeds: &embeds,
    };
    let ag = agglomerate(&cx, &[0, 1, 2, 3]).expect("agglomerate");
    assert_eq!(ag.fus.len(), 2, "three MACs become one; the max stays apart");
    assert_eq!(ag.steps.len(), 2);
    assert_eq!(ag.stopped_by, "gain");
    let total: f64 = ag.costs.iter().map(|c| c.area).sum();
    assert!((ag.initial_area - ag.steps.iter().map(|s| s.gain_um2).sum::<f64>() - total).abs() < 1e-6);
}
