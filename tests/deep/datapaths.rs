//! Deep check: every pass up to `datapaths` on `[testing] deep_models` at the `full` config, every width
//! tuple and float class included, with every check of its `verify` (structure, proofs by both solvers,
//! order queries, the encoding lemmas in Z3, the mutated control), a golden snapshot of the summary and
//! of every class's datapath shape, and byte-identical outputs on a forced rerun. Writes into the build
//! area's ephemeral `runs/deep-datapaths/` and removes it when the test passes.

use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::datapaths::DATAPATHS;
use cc_passes::datapaths::json::{Datapath, Summary};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

#[test]
fn datapaths_deep_models_at_full() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-datapaths");
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
        targets: vec!["datapaths".into()],
        models: Some(models.clone()),
        ..RunOptions::default()
    };
    let s = p.run(&opts).unwrap();
    let report = p.store().pass_dir("datapaths").read_verify().unwrap().unwrap();
    let failed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| (c.name.clone(), c.evidence.notes.clone()))
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");
    assert!(
        matches!(
            s.status_of("datapaths"),
            Some(RunStatus::Ok | RunStatus::CacheHit)
        ),
        "{s:?}"
    );

    // Golden digest: the summary and every class's datapath shape.
    let out = p.store().pass_dir("datapaths").kind(DATAPATHS);
    let t: Summary = out.read_json("summary.json").unwrap();
    let mut ids: Vec<String> = std::fs::read_dir(out.path("datapaths").unwrap())
        .unwrap()
        .filter_map(|e| {
            e.ok()?
                .file_name()
                .to_str()?
                .strip_suffix(".json")
                .map(str::to_string)
        })
        .collect();
    ids.sort();
    let classes: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            let d: Datapath = out.read_json(&format!("datapaths/{id}.json")).unwrap();
            let labels: Vec<&str> = d.vertices.iter().map(|v| v.label.as_str()).collect();
            serde_json::json!({
                "id": d.id,
                "route": d.route,
                "labels": labels.join(" "),
                "tuples": d.tuples.len(),
                "order": d.accumulator.as_ref().map(|a| a.order.iter().map(|o| o.free).collect::<Vec<_>>()),
            })
        })
        .collect();
    let digest = serde_json::json!({
        "summary": t,
        "classes": classes,
        "checks": report.checks.len(),
    });
    insta::assert_json_snapshot!("datapaths_deep_digest", digest);

    // Determinism: a forced rerun writes the same bytes.
    let before = out.record().unwrap().hash;
    let again = p
        .run(&RunOptions {
            from: Some("datapaths".into()),
            ..opts
        })
        .unwrap();
    assert_eq!(again.status_of("datapaths"), Some(RunStatus::Ok), "{again:?}");
    assert_eq!(
        out.record().unwrap().hash,
        before,
        "outputs differ between two runs"
    );

    std::fs::remove_dir_all(&build).unwrap();
}
