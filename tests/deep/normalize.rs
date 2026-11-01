//! Deep check: `export` (toolchain image), `import`, `components` and `normalize` on `[testing]
//! deep_models` at the `full` config, with every check of normalize's `verify` (the rule proofs run with
//! both solvers in the image), a golden snapshot (rewrites per rule, reductions per order, node counts)
//! and byte-identical outputs on a forced rerun. Writes into the build area's ephemeral
//! `runs/deep-normalize/` and removes it when the test passes.

use std::collections::BTreeMap;

use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::normalize::NORMALIZE;
use cc_passes::normalize::json::NormalizedSummary;
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

#[test]
fn normalize_deep_models_at_full() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-normalize");
    let sets = [
        format!("paths.build_dir={}", build.display()),
        "models.config=full".to_string(),
        "passes.export.runner=docker".to_string(),
        "tools.smt.runner=docker".to_string(),
    ];
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &sets).unwrap();
    let models = cfg.config.testing.deep_models.clone();
    let p = Pipeline::new(cc_passes::registry(), cfg).unwrap();
    let opts = RunOptions {
        targets: vec!["normalize".into()],
        models: Some(models.clone()),
        ..RunOptions::default()
    };
    let s = p.run(&opts).unwrap();
    let report = p.store().pass_dir("normalize").read_verify().unwrap().unwrap();
    let failed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| (c.name.clone(), c.evidence.notes.clone()))
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");
    assert!(
        matches!(
            s.status_of("normalize"),
            Some(RunStatus::Ok | RunStatus::CacheHit)
        ),
        "{s:?}"
    );

    // Golden digest.
    let out = p.store().pass_dir("normalize").kind(NORMALIZE);
    let mut rules: BTreeMap<String, u64> = BTreeMap::new();
    let mut omega: BTreeMap<String, u64> = BTreeMap::new();
    let (mut components, mut nodes_in, mut nodes) = (0u64, 0u64, 0u64);
    for f in out.files().unwrap() {
        if f.path.ends_with(".postcard.summary.json") {
            let c: NormalizedSummary = out.read_json(&f.path).unwrap();
            components += 1;
            nodes_in += c.nodes_in;
            nodes += c.nodes;
            for (k, v) in c.rules {
                *rules.entry(k).or_default() += v;
            }
            for (k, v) in c.omega {
                *omega.entry(k).or_default() += v;
            }
        }
    }
    let digest = serde_json::json!({
        "components": components,
        "nodes_in": nodes_in,
        "nodes": nodes,
        "rules": rules,
        "omega": omega,
        "checks": report.checks.len(),
    });
    insta::assert_json_snapshot!("normalize_deep_digest", digest);

    // Determinism: a forced rerun writes the same bytes.
    let before = out.record().unwrap().hash;
    let again = p
        .run(&RunOptions {
            from: Some("normalize".into()),
            ..opts
        })
        .unwrap();
    assert_eq!(again.status_of("normalize"), Some(RunStatus::Ok), "{again:?}");
    assert_eq!(
        out.record().unwrap().hash,
        before,
        "outputs differ between two runs"
    );

    std::fs::remove_dir_all(&build).unwrap();
}
