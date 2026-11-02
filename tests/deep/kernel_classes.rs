//! Deep check: every pass up to `kernel_classes` on `[testing] deep_models` at the `full` config, with
//! every check of kernel_classes' `verify` (certificates, rows, the workload against the concrete census,
//! the sharing report), a golden snapshot of the summary and the sharing digest, and byte-identical
//! outputs on a forced rerun. Writes into the build area's ephemeral `runs/deep-kernel-classes/` and
//! removes it when the test passes.

use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::kernel_classes::KERNEL_CLASSES;
use cc_passes::kernel_classes::json::{ClassesSummary, Level, Sharing};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

fn r6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

fn level(l: &Level) -> serde_json::Value {
    serde_json::json!({
        "items": l.items,
        "shared": l.shared,
        "per_model": l.per_model,
        "mean_jaccard": r6(l.mean_jaccard),
        "coverage_total": r6(l.coverage_total),
        "coverage_mean": r6(l.coverage_mean),
    })
}

#[test]
fn kernel_classes_deep_models_at_full() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area)
        .join("runs")
        .join("deep-kernel-classes");
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
        targets: vec!["kernel_classes".into()],
        models: Some(models.clone()),
        ..RunOptions::default()
    };
    let s = p.run(&opts).unwrap();
    let report = p
        .store()
        .pass_dir("kernel_classes")
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
            s.status_of("kernel_classes"),
            Some(RunStatus::Ok | RunStatus::CacheHit)
        ),
        "{s:?}"
    );

    // Golden digest.
    let out = p.store().pass_dir("kernel_classes").kind(KERNEL_CLASSES);
    let t: ClassesSummary = out.read_json("summary.json").unwrap();
    let sh: Sharing = out.read_json("sharing.json").unwrap();
    let digest = serde_json::json!({
        "summary": t,
        "sharing": {
            "models": sh.models,
            "all": {"components": level(&sh.all.components), "kernels": level(&sh.all.kernels)},
            "activation": {
                "components": level(&sh.activation.components),
                "kernels": level(&sh.activation.kernels),
            },
            "kernel_sharing_exceeds_component_sharing": sh.kernel_sharing_exceeds_component_sharing,
        },
        "checks": report.checks.len(),
    });
    insta::assert_json_snapshot!("kernel_classes_deep_digest", digest);

    // Determinism: a forced rerun writes the same bytes.
    let before = out.record().unwrap().hash;
    let again = p
        .run(&RunOptions {
            from: Some("kernel_classes".into()),
            ..opts
        })
        .unwrap();
    assert_eq!(
        again.status_of("kernel_classes"),
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
