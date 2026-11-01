//! Deep check: `export` (toolchain image), `import` and `components` on `[testing] deep_models` at the
//! `full` config, with every check of components' `verify`, a golden snapshot (instances per phase, the
//! component count, the most shared components) and byte-identical outputs on a forced rerun. Writes into
//! the build area's ephemeral `runs/deep-components/` and removes it when the test passes.

use std::collections::BTreeMap;

use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::components::COMPONENTS;
use cc_passes::components::json::{ComponentSummary, InstanceKind, InstancesFile};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

#[test]
fn components_deep_models_at_full() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-components");
    let sets = [
        format!("paths.build_dir={}", build.display()),
        "models.config=full".to_string(),
        "passes.export.runner=docker".to_string(),
    ];
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &sets).unwrap();
    let models = cfg.config.testing.deep_models.clone();
    let p = Pipeline::new(cc_passes::registry(), cfg).unwrap();
    let opts = RunOptions {
        targets: vec!["components".into()],
        models: Some(models.clone()),
        ..RunOptions::default()
    };
    let s = p.run(&opts).unwrap();
    let report = p.store().pass_dir("components").read_verify().unwrap().unwrap();
    let failed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| (c.name.clone(), c.evidence.notes.clone()))
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");
    assert!(
        matches!(
            s.status_of("components"),
            Some(RunStatus::Ok | RunStatus::CacheHit)
        ),
        "{s:?}"
    );

    // Golden digest.
    let out = p.store().pass_dir("components").kind(COMPONENTS);
    let mut phases: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut comps: Vec<ComponentSummary> = Vec::new();
    for f in out.files().unwrap() {
        if let Some(rel) = f
            .path
            .strip_prefix("instances/")
            .and_then(|r| r.strip_suffix(".json"))
        {
            let v: InstancesFile = out.read_json(&f.path).unwrap();
            let anon = v
                .instances
                .iter()
                .filter(|i| i.kind == InstanceKind::Anon)
                .count();
            let distinct: std::collections::BTreeSet<&String> =
                v.instances.iter().filter_map(|i| i.component.as_ref()).collect();
            phases.insert(
                rel.to_string(),
                serde_json::json!({
                    "instances": v.instances.len(), "anon": anon, "components": distinct.len(),
                }),
            );
        } else if f.path.ends_with(".postcard.summary.json") {
            comps.push(out.read_json(&f.path).unwrap());
        }
    }
    comps.sort_by(|a, b| {
        (b.models.len(), b.instances.len(), &a.id).cmp(&(a.models.len(), a.instances.len(), &b.id))
    });
    let shared: Vec<serde_json::Value> = comps
        .iter()
        .take(10)
        .map(|c| serde_json::json!({"id": c.id, "models": c.models, "nodes": c.nodes}))
        .collect();
    let digest = serde_json::json!({
        "phases": phases,
        "components": comps.len(),
        "shared_by_two_or_more": comps.iter().filter(|c| c.models.len() >= 2).count(),
        "most_shared": shared,
    });
    insta::assert_json_snapshot!("components_deep_digest", digest);

    // Determinism: a forced rerun writes the same bytes.
    let before = out.record().unwrap().hash;
    let again = p
        .run(&RunOptions {
            from: Some("components".into()),
            ..opts
        })
        .unwrap();
    assert_eq!(again.status_of("components"), Some(RunStatus::Ok), "{again:?}");
    assert_eq!(
        out.record().unwrap().hash,
        before,
        "outputs differ between two runs"
    );

    std::fs::remove_dir_all(&build).unwrap();
}
