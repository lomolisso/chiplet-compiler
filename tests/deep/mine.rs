//! Deep check: every pass up to `mine` on `[testing] deep_models` at the `full` config, with real GraMi
//! runs in the toolchain image and every check of its `verify` (translation of the graphs, GraMi's list
//! re-read, independent re-matching, convexity, MIS and scores), a golden snapshot of the summary and
//! the selections, and byte-identical outputs on a forced rerun (same τ, same patterns). Writes into
//! the build area's ephemeral `runs/deep-mine/` and removes it when the test passes.

use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::mine::MINE;
use cc_passes::mine::json::{Selection, Summary, Sweep};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

#[test]
fn mine_deep_models_at_full() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-mine");
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
        targets: vec!["mine".into()],
        models: Some(models),
        ..RunOptions::default()
    };
    let s = p.run(&opts).unwrap();
    let report = p.store().pass_dir("mine").read_verify().unwrap().unwrap();
    let failed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| (c.name.clone(), c.evidence.notes.clone()))
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");
    assert!(
        matches!(s.status_of("mine"), Some(RunStatus::Ok | RunStatus::CacheHit)),
        "{s:?}"
    );

    // Golden digest: the summary, the sweeps and the selections.
    let out = p.store().pass_dir("mine").kind(MINE);
    let t: Summary = out.read_json("summary.json").unwrap();
    let mut fams = Vec::new();
    for f in ["map", "reduce", "contraction", "stage"] {
        let sw: Sweep = out.read_json(&format!("sweep/{f}.json")).unwrap();
        let sel: Selection = out.read_json(&format!("selection/{f}.json")).unwrap();
        fams.push(serde_json::json!({
            "family": f,
            "sweep": sw,
            "selection": sel.patterns.iter().map(|p| (p.id.clone(), p.vertices, p.mni, p.mis)).collect::<Vec<_>>(),
        }));
    }
    // Machine-dependent only if the budget stopped a sweep.
    for f in &fams {
        assert_ne!(f["sweep"]["stopped_by"], "budget", "{f}");
    }
    let digest = serde_json::json!({
        "summary": t,
        "families": fams,
        "checks": report.checks.len(),
    });
    insta::assert_json_snapshot!("mine_deep_digest", digest);

    // Reproducibility: a forced rerun writes the same bytes.
    let before = out.record().unwrap().hash;
    let again = p
        .run(&RunOptions {
            from: Some("mine".into()),
            ..opts
        })
        .unwrap();
    assert_eq!(again.status_of("mine"), Some(RunStatus::Ok), "{again:?}");
    assert_eq!(out.record().unwrap().hash, before, "outputs differ between two runs");

    std::fs::remove_dir_all(&build).unwrap();
}
