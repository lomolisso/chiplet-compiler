//! deep check: the real `export` pass through the toolchain image (`runner = "docker"`) on
//! `[testing] deep_models` at the `full` config (abstract lowering only; 8 GB per model process), with
//! every check of its `verify`, including the census against the recipes at full. Writes into a scratch
//! build directory, not the repository's `build/`.

use cc_core::{CheckStatus, LoadedConfig};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

#[test]
fn export_deep_models_at_full() {
    let scratch = std::env::var("CC_SCRATCH")
        .unwrap_or_else(|_| format!("/srv/scratch/{}", std::env::var("USER").unwrap_or_default()));
    let build = std::path::Path::new(&scratch).join("chipc-deep-export");
    let sets = [
        format!("paths.build_dir={}", build.display()),
        "models.config=full".to_string(),
        "passes.export.runner=docker".to_string(),
    ];
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &sets).unwrap();
    let models = cfg.config.testing.deep_models.clone();
    let p = Pipeline::new(cc_passes::registry(), cfg).unwrap();
    let s = p
        .run(&RunOptions {
            only: Some("export".into()),
            models: Some(models.clone()),
            ..RunOptions::default()
        })
        .unwrap();
    let report = p.store().pass_dir("export").read_verify().unwrap().unwrap();
    let failed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| (c.name.clone(), c.evidence.notes.clone()))
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");
    assert!(
        matches!(s.status_of("export"), Some(RunStatus::Ok | RunStatus::CacheHit)),
        "{s:?}"
    );
    for m in &models {
        assert!(
            report
                .checks
                .iter()
                .any(|c| c.name == format!("{m}:invariance") && c.status == CheckStatus::Pass),
            "{m}"
        );
    }
}
