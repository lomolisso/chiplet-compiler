//! Deep check: every pass up to `merge` on `[testing] deep_models` at the `full` config, with a
//! license-free library (`chipc doctor`, then Yosys on Nangate45 over the whole catalog, about 20
//! minutes), real solvers for every configuration proof and every check of its
//! `verify`, a golden snapshot of the summary and the traces, and byte-identical outputs on a forced
//! rerun. Writes into the build area's ephemeral `runs/deep-merge/` and removes it when the test passes.

use cc_core::runner::Runners;
use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::merge::MERGE;
use cc_passes::merge::json::{Summary, Trace};
use cc_pipeline::env::{DoctorOptions, doctor};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

#[test]
fn merge_deep_models_at_full() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-merge");
    let sets = [
        format!("paths.build_dir={}", build.display()),
        "models.config=full".to_string(),
        "passes.export.runner=docker".to_string(),
        "tools.smt.runner=docker".to_string(),
        "passes.characterize.tool=yosys".to_string(),
        "passes.characterize.pdks=[\"nangate45\"]".to_string(),
        "passes.characterize.resources.threads=8".to_string(),
        "passes.characterize.resources.memory_gb=96".to_string(),
        "passes.merge.kit=nangate45".to_string(),
    ];
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &sets).unwrap();
    // `characterize` reads the libraries `chipc doctor` prepares.
    let mut text = Vec::new();
    let env = doctor(
        &cfg,
        &Runners::from_config(&cfg),
        DoctorOptions::default(),
        &mut text,
    )
    .unwrap();
    let db = env.checks.iter().find(|c| c.name == "pdk.db").unwrap();
    assert_eq!(db.status, CheckStatus::Pass, "{}", String::from_utf8_lossy(&text));
    let models = cfg.config.testing.deep_models.clone();
    let p = Pipeline::new(cc_passes::registry(), cfg).unwrap();
    let opts = RunOptions {
        targets: vec!["merge".into()],
        models: Some(models),
        ..RunOptions::default()
    };
    let s = p.run(&opts).unwrap();
    assert!(
        matches!(s.status_of("merge"), Some(RunStatus::Ok | RunStatus::CacheHit)),
        "{s:?}"
    );
    let report = p.store().pass_dir("merge").read_verify().unwrap().unwrap();
    let failed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| (c.name.clone(), c.evidence.notes.clone()))
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");

    // Golden digest: the summary and every family's steps.
    let out = p.store().pass_dir("merge").kind(MERGE);
    let t: Summary = out.read_json("summary.json").unwrap();
    assert_eq!(t.covered, t.classes, "{t:?}");
    let mut fams = Vec::new();
    for f in ["map", "reduce", "contraction", "stage"] {
        let tr: Trace = out.read_json(&format!("trace/{f}.json")).unwrap();
        fams.push(serde_json::json!({
            "family": f,
            "units": tr.initial_units,
            "steps": tr.steps.iter().map(|s| (s.a.clone(), s.b.clone(), s.units)).collect::<Vec<_>>(),
            "stopped_by": tr.stopped_by,
        }));
    }
    let digest = serde_json::json!({
        "summary": {
            "families": t.families.iter().map(|(k, v)| (k.clone(), (v.units, v.fus, v.steps))).collect::<std::collections::BTreeMap<_, _>>(),
            "fus": t.fus, "arrays": t.arrays, "configs": t.configs, "proofs": t.proofs,
            "array_proofs": t.array_proofs, "classes": t.classes, "covered": t.covered,
        },
        "families": fams,
        "checks": report.checks.len(),
    });
    insta::assert_json_snapshot!("merge_deep_digest", digest);

    // Reproducibility: a forced rerun writes the same bytes.
    let before = out.record().unwrap().hash;
    let again = p
        .run(&RunOptions {
            from: Some("merge".into()),
            ..opts
        })
        .unwrap();
    assert_eq!(again.status_of("merge"), Some(RunStatus::Ok), "{again:?}");
    assert_eq!(
        out.record().unwrap().hash,
        before,
        "outputs differ between two runs"
    );

    std::fs::remove_dir_all(&build).unwrap();
}
