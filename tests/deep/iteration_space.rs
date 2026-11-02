//! Deep check: every pass up to `iteration_space` on `[testing] deep_models` at the `full` config, with
//! every check of its `verify` (access proofs by both solvers, dependences, mappings, routing, the
//! embedding certificates), a golden snapshot of the summary and of the routing table of every class,
//! and byte-identical outputs on a forced rerun. Writes into the build area's ephemeral
//! `runs/deep-iteration-space/` and removes it when the test passes.

use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::iteration_space::ITERATION_SPACE;
use cc_passes::iteration_space::json::{Routing, Summary};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

#[test]
fn iteration_space_deep_models_at_full() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area)
        .join("runs")
        .join("deep-iteration-space");
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
        targets: vec!["iteration_space".into()],
        models: Some(models.clone()),
        ..RunOptions::default()
    };
    let s = p.run(&opts).unwrap();
    let report = p
        .store()
        .pass_dir("iteration_space")
        .read_verify()
        .unwrap()
        .unwrap();
    let failed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| (c.name.clone(), c.evidence.notes.clone()))
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");
    assert!(
        matches!(
            s.status_of("iteration_space"),
            Some(RunStatus::Ok | RunStatus::CacheHit)
        ),
        "{s:?}"
    );

    // Golden digest: the summary and every class's route.
    let out = p.store().pass_dir("iteration_space").kind(ITERATION_SPACE);
    let t: Summary = out.read_json("summary.json").unwrap();
    let r: Routing = out.read_json("routing.json").unwrap();
    let routes: Vec<serde_json::Value> = r
        .classes
        .iter()
        .map(|c| {
            serde_json::json!({
                "id": c.id,
                "core": c.core,
                "class": c.class,
                "route": c.route,
                "dataflow": c.dataflow,
                "reason": cc_passes::iteration_space::reason_kind(&c.reason),
            })
        })
        .collect();
    let digest = serde_json::json!({
        "summary": t,
        "routing": routes,
        "checks": report.checks.len(),
    });
    insta::assert_json_snapshot!("iteration_space_deep_digest", digest);

    // Determinism: a forced rerun writes the same bytes.
    let before = out.record().unwrap().hash;
    let again = p
        .run(&RunOptions {
            from: Some("iteration_space".into()),
            ..opts
        })
        .unwrap();
    assert_eq!(
        again.status_of("iteration_space"),
        Some(RunStatus::Ok),
        "{again:?}"
    );
    assert_eq!(
        out.record().unwrap().hash,
        before,
        "outputs differ between two runs"
    );

    std::fs::remove_dir_all(&build).unwrap();
}
