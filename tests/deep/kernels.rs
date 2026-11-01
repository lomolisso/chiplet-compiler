//! Deep check: `export` (toolchain image), `import`, `components`, `normalize` and `kernels` on
//! `[testing] deep_models` at the `full` config, with every check of kernels' `verify`, a golden snapshot
//! (kernels per core, roles, boundary kinds, kernels per component) and byte-identical outputs on a
//! forced rerun. Writes into the build area's ephemeral `runs/deep-kernels/` and removes it when the test
//! passes.

use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::kernels::KERNELS;
use cc_passes::kernels::json::KernelsSummary;
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

#[test]
fn kernels_deep_models_at_full() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-kernels");
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
        targets: vec!["kernels".into()],
        models: Some(models.clone()),
        ..RunOptions::default()
    };
    let s = p.run(&opts).unwrap();
    let report = p.store().pass_dir("kernels").read_verify().unwrap().unwrap();
    let failed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| (c.name.clone(), c.evidence.notes.clone()))
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");
    assert!(
        matches!(s.status_of("kernels"), Some(RunStatus::Ok | RunStatus::CacheHit)),
        "{s:?}"
    );

    // Golden digest.
    let out = p.store().pass_dir("kernels").kind(KERNELS);
    let t: KernelsSummary = out.read_json("summary.json").unwrap();
    let digest = serde_json::json!({
        "components": t.components,
        "kernels": t.kernels,
        "core_kernels": t.core_kernels,
        "map_kernels": t.map_kernels,
        "kernels_per_component": t.kernels_per_component,
        "cores": t.cores,
        "roles": t.roles,
        "boundaries": t.boundaries,
        "call_fed_maps": t.call_fed_maps,
        "low_rank_maps": t.low_rank_maps,
        "evictions": t.evictions,
        "checks": report.checks.len(),
    });
    insta::assert_json_snapshot!("kernels_deep_digest", digest);

    // Determinism: a forced rerun writes the same bytes.
    let before = out.record().unwrap().hash;
    let again = p
        .run(&RunOptions {
            from: Some("kernels".into()),
            ..opts
        })
        .unwrap();
    assert_eq!(again.status_of("kernels"), Some(RunStatus::Ok), "{again:?}");
    assert_eq!(
        out.record().unwrap().hash,
        before,
        "outputs differ between two runs"
    );

    std::fs::remove_dir_all(&build).unwrap();
}
