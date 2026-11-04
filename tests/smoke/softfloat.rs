//! `softfloat` without the toolchain image: the primitive list against the catalog, and the proof queries
//! built from committed optimized XLS IR (`tests/fixtures/softfloat/<id>.opt.ir`, written by the pass) in
//! both solvers: each circuit is proven equal to its op's semantics, the f16 multiply with its product
//! abstracted, and a circuit with one `add` turned into a `sub` has a counterexample. The pass itself
//! (Allo, XLS) runs in the deep test.

use cc_core::CancelToken;
use cc_core::runner::{BoundRunner, Caps, Runners, ToolLog};
use cc_passes::characterize::catalog;
use cc_passes::softfloat::{SoftfloatConfig, primitives, proof};
use cc_smt::{Location, Solver, Verdict};
use tests::{Env, repo_root};

const FIXTURES: &[&str] = &[
    "add_f8e4m3fn",
    "multiply_f16",
    "convert_i8_f32",
    "compare_lt_bf16",
];

#[test]
fn the_library_covers_every_float_op_of_the_catalog() {
    let (built, approx) = primitives(&SoftfloatConfig::default()).unwrap();
    assert!(
        built
            .iter()
            .all(|p| p.types.iter().any(|t| catalog::format(t).is_some()))
    );
    for key in [
        "add@f32",
        "divide@bf16",
        "sqrt@f16",
        "remainder@f8e4m3fn",
        "convert@ui32>bf16",
        "compare.LE@f16",
    ] {
        assert!(built.iter().any(|p| p.key == key), "{key}");
    }
    assert!(approx.iter().any(|k| k == "exponential@f32"));
    assert!(approx.iter().all(|k| !built.iter().any(|p| &p.key == k)));
    assert!(
        !built
            .iter()
            .any(|p| p.op == "tanh" || p.kind != catalog::Kind::Op)
    );
}

fn query(id: &str, mutate: bool) -> (String, proof::Query) {
    let (built, _) = primitives(&SoftfloatConfig::default()).unwrap();
    let p = built.iter().find(|p| p.id == id).unwrap();
    let mut ir =
        std::fs::read_to_string(repo_root().join(format!("tests/fixtures/softfloat/{id}.opt.ir"))).unwrap();
    if mutate {
        assert!(ir.contains(" = add("), "{id}: no add node to mutate");
        ir = ir.replacen(" = add(", " = sub(", 1);
    }
    let (args, ret, attrs) = catalog::signature(p).unwrap();
    let q = proof::query(&p.key, &p.op, &args, ret, &attrs, &ir).unwrap();
    (format!("{id}{}.smt2", if mutate { "_mutated" } else { "" }), q)
}

#[test]
fn circuits_are_proven_and_a_mutated_one_is_refuted() {
    let env = Env::new();
    let cfg: toml::Value =
        toml::from_str(&std::fs::read_to_string(repo_root().join("config.toml")).unwrap()).unwrap();
    let tag = cfg["tools"]["docker"]["tag"].as_str().unwrap().to_string();
    let host = std::env::var("CC_SMT_RUNNER").as_deref() == Ok("host");
    let mut sets = vec![format!("tools.docker.tag={tag}")];
    if host {
        sets.push("tools.smt.runner=host".into());
    }
    let sets: Vec<&str> = sets.iter().map(String::as_str).collect();
    let lc = env.config("[models]\nkeys = []\n", &sets);
    let dir = env.root().join("softfloat");
    std::fs::create_dir_all(&dir).unwrap();
    let mut files = Vec::new();
    for id in FIXTURES {
        let (f, q) = query(id, false);
        assert_eq!(q.product_abstracted, *id == "multiply_f16", "{id}");
        std::fs::write(dir.join(&f), &q.text).unwrap();
        files.push((f, true));
    }
    let (f, q) = query("add_f8e4m3fn", true);
    std::fs::write(dir.join(&f), &q.text).unwrap();
    files.push((f, false));
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
            memory_gb: 4.0,
            cpus: 1.0,
            timeout_s: 600.0,
            pids_limit: 64,
        },
        ToolLog::new("softfloat-proofs", dir.join("logs")),
        CancelToken::new(),
    );
    let names: Vec<String> = files.iter().map(|f| f.0.clone()).collect();
    for solver in Solver::ALL {
        let b = cc_smt::run_batch(&runner, &loc, solver, &dir, &names, 300).unwrap();
        for (f, holds) in &files {
            let v = b.verdicts.get(f);
            if *holds {
                assert_eq!(v, Some(&Verdict::Unsat), "{solver:?} {f}");
            } else {
                assert!(matches!(v, Some(Verdict::Sat { .. })), "{solver:?} {f}: {v:?}");
            }
        }
    }
}
