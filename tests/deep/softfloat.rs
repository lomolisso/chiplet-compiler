//! Deep check: `softfloat` on a reduced set (add, multiply, divide, compare, conversions and the
//! approximate ops exponential, log, log_plus_one, tanh and rsqrt in f16 and f8e4m3fn, integers of 8 bits)
//! into the build area's ephemeral `runs/deep-softfloat/`, with Allo, XLS and both solvers for real.
//! `verify` passes; then each kind of tampering fails its check. The directory is removed when the test
//! passes.

use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::softfloat::SOFTFLOAT;
use cc_passes::softfloat::json::{Accuracy, CoverageFile, Evidence};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

const REDUCED: &[&str] = &[
    "passes.softfloat.ops=[\"add\", \"multiply\", \"divide\", \"compare\", \"convert\", \"exponential\", \"log\", \"log_plus_one\", \"tanh\", \"rsqrt\"]",
    "passes.softfloat.float_formats=[\"f16\", \"f8e4m3fn\"]",
    "passes.softfloat.int_widths=[8]",
    "passes.softfloat.random_vectors=5000",
    "passes.softfloat.proof_timeout_s=300",
    "passes.softfloat.recheck_timeout_s=600",
];

fn failing(p: &Pipeline) -> Vec<String> {
    p.verify_pass("softfloat")
        .unwrap()
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| c.name.clone())
        .collect()
}

/// Edit a file, check that `check` fails, restore it, and check that everything passes again.
fn tamper(p: &Pipeline, path: &std::path::Path, check: &str, edit: impl FnOnce(&std::path::Path)) {
    let tmp = path.with_extension("orig");
    std::fs::copy(path, &tmp).unwrap();
    edit(path);
    let f = failing(p);
    std::fs::rename(&tmp, path).unwrap();
    assert!(
        f.iter().any(|c| c == check),
        "{}: {check} not in {f:?}",
        path.display()
    );
}

#[test]
fn softfloat_reduced_set_builds_proves_and_verifies() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-softfloat");
    let mut sets = vec![format!("paths.build_dir={}", build.display())];
    sets.extend(REDUCED.iter().map(|s| s.to_string()));
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &sets).unwrap();
    let p = Pipeline::new(cc_passes::registry(), cfg).unwrap();
    let s = p
        .run(&RunOptions {
            targets: vec!["softfloat".into()],
            ..RunOptions::default()
        })
        .unwrap();
    assert!(
        matches!(
            s.status_of("softfloat"),
            Some(RunStatus::Ok | RunStatus::CacheHit)
        ),
        "{s:?}"
    );
    let dir = p.store().pass_dir("softfloat");
    let report = dir.read_verify().unwrap().unwrap();
    let failed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| (c.name.clone(), c.evidence.notes.clone()))
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");

    let kind = dir.kind(SOFTFLOAT);
    let cov: CoverageFile = kind.read_json("coverage.json").unwrap();
    let entry = |key: &str| cov.entries.iter().find(|e| e.key == key).unwrap();
    // 2 add, 2 multiply, 2 divide, 8 compare, 14 conversions (bool, i8, ui8 with each format; f16 <-> f8),
    // 10 approximate (5 ops in 2 formats).
    assert_eq!(
        cov.entries.len(),
        38,
        "{:?}",
        cov.entries.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
    assert_eq!(entry("add@f8e4m3fn").evidence, Evidence::Proven);
    assert!(entry("add@f8e4m3fn").exhaustive);
    let mul = entry("multiply@f16");
    assert_eq!(mul.evidence, Evidence::Proven);
    assert!(mul.proof.as_ref().unwrap().product_abstracted);
    assert_eq!(entry("convert@f16>f8e4m3fn").evidence, Evidence::Proven);
    let exp = entry("exponential@f16");
    assert_eq!(
        (exp.evidence, exp.accuracy),
        (Evidence::Exhaustive, Accuracy::Faithful)
    );
    assert!(exp.proof.is_none() && exp.max_ulp.unwrap() <= 1.0);
    assert_eq!(entry("rsqrt@f16").accuracy, Accuracy::CorrectlyRounded);
    assert_eq!(entry("log@f8e4m3fn").accuracy, Accuracy::CorrectlyRounded);
    assert!(cov.approximate.is_empty());

    let op = |id: &str, f: &str| kind.dir().join(format!("ops/{id}/{f}"));
    // An edited circuit: its query no longer matches.
    tamper(&p, &op("add_f8e4m3fn", "fu.opt.ir"), "queries", |f| {
        let t = std::fs::read_to_string(f).unwrap();
        std::fs::write(f, t.replacen(" = add(", " = sub(", 1)).unwrap();
    });
    // An edited verdict.
    tamper(&p, &op("add_f8e4m3fn", "proof.result"), "verdicts", |f| {
        let t = std::fs::read_to_string(f).unwrap();
        std::fs::write(f, t.replace("\"unsat\"", "\"timeout\"")).unwrap();
    });
    // Another primitive's vectors: the recomputed reference differs.
    tamper(&p, &op("add_f16", "vectors.npz"), "reference", |f| {
        std::fs::copy(op("multiply_f16", "vectors.npz"), f).unwrap();
    });
    // A build result that reports IR mismatches.
    tamper(&p, &op("multiply_f16", "result.json"), "builds", |f| {
        let t = std::fs::read_to_string(f).unwrap();
        std::fs::write(f, t.replace("\"ir_vs_llvm\": 0", "\"ir_vs_llvm\": 3")).unwrap();
    });
    // A float type in a module.
    tamper(&p, &op("divide_f16", "kernel.mlir"), "integer_only", |f| {
        let t = std::fs::read_to_string(f).unwrap();
        std::fs::write(f, format!("{t}\n// %x = arith.sitofp %a : i32 to f32\n")).unwrap();
    });
    // A Verilog file of another module.
    tamper(&p, &op("divide_f16", "rtl.v"), "rtl", |f| {
        std::fs::copy(op("add_f16", "rtl.v"), f).unwrap();
    });
    // An approximate op's recorded error beyond its limit.
    tamper(&p, &kind.dir().join("coverage.json"), "accuracy", |f| {
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
        for e in v["entries"].as_array_mut().unwrap() {
            if e["key"] == "tanh@f16" {
                e["max_ulp"] = serde_json::json!(1.5);
            }
        }
        std::fs::write(f, v.to_string()).unwrap();
    });
    // A primitive dropped from the coverage table.
    tamper(&p, &kind.dir().join("coverage.json"), "primitives", |f| {
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
        v["entries"].as_array_mut().unwrap().pop();
        std::fs::write(f, v.to_string()).unwrap();
    });
    assert!(failing(&p).is_empty());
    std::fs::remove_dir_all(&build).unwrap();
}
