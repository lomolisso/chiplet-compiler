//! Deep check: `export` (toolchain image) then `import` on `[testing] deep_models` at the `full` config,
//! with every check of import's `verify`, a golden snapshot of each phase's summary digest (`insta`,
//! the structural hash included) and byte-identical outputs on a forced rerun. Writes into the build
//! area's ephemeral `runs/deep-import/` and removes it when the test passes.

use std::collections::BTreeMap;

use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::import::IMPORT;
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

#[test]
fn import_deep_models_at_full() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-import");
    let sets = [
        format!("paths.build_dir={}", build.display()),
        "models.config=full".to_string(),
        "passes.export.runner=docker".to_string(),
    ];
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &sets).unwrap();
    let models = cfg.config.testing.deep_models.clone();
    let p = Pipeline::new(cc_passes::registry(), cfg).unwrap();
    let opts = RunOptions {
        targets: vec!["import".into()],
        models: Some(models.clone()),
        ..RunOptions::default()
    };
    let s = p.run(&opts).unwrap();
    let report = p.store().pass_dir("import").read_verify().unwrap().unwrap();
    let failed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| (c.name.clone(), c.evidence.notes.clone()))
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");
    assert!(
        matches!(s.status_of("import"), Some(RunStatus::Ok | RunStatus::CacheHit)),
        "{s:?}"
    );

    // Golden digest per phase.
    let out = p.store().pass_dir("import").kind(IMPORT);
    let mut digest: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for f in out.files().unwrap() {
        let Some(rel) = f.path.strip_suffix("/ir.postcard.summary.json") else {
            continue;
        };
        let v: serde_json::Value = out.read_json(&f.path).unwrap();
        let dad: u64 = v["dim_as_data"]
            .as_object()
            .unwrap()
            .values()
            .map(|x| x.as_u64().unwrap())
            .sum();
        digest.insert(
            rel.to_string(),
            serde_json::json!({
                "ops": v["ops"], "funcs": v["funcs"], "size_ops": v["size_ops"], "dim_as_data": dad,
                "symbolic_values": v["symbolic_values"], "ledger": v["ledger"], "ir_hash": v["ir_hash"],
            }),
        );
    }
    assert_eq!(
        digest.len(),
        report.checks.iter().filter(|c| c.name.ends_with(":ssa")).count()
    );
    insta::assert_json_snapshot!("import_deep_digest", digest);

    // Determinism: a forced rerun writes the same bytes.
    let before = out.record().unwrap().hash;
    let again = p
        .run(&RunOptions {
            from: Some("import".into()),
            ..opts
        })
        .unwrap();
    assert_eq!(again.status_of("import"), Some(RunStatus::Ok), "{again:?}");
    assert_eq!(
        out.record().unwrap().hash,
        before,
        "outputs differ between two runs"
    );

    std::fs::remove_dir_all(&build).unwrap();
}
