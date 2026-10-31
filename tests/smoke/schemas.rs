//! The committed JSON Schemas in `docs/schemas/` match the Rust types. `CC_UPDATE_SCHEMAS=1` rewrites them.

use tests::repo_root;

#[test]
fn committed_schemas_are_current() {
    let dir = repo_root().join("docs/schemas");
    let update = std::env::var("CC_UPDATE_SCHEMAS").is_ok_and(|v| v == "1");
    let mut stale = Vec::new();
    for (name, schema) in cc_core::schema::core_schemas()
        .into_iter()
        .chain(cc_pipeline::pipeline_schemas())
        .chain(cc_passes::pass_schemas())
    {
        let want = cc_core::canon::to_canonical_pretty(&schema).unwrap();
        let path = dir.join(name);
        if update {
            std::fs::write(&path, &want).unwrap();
        } else if std::fs::read(&path).ok().as_deref() != Some(want.as_slice()) {
            stale.push(name);
        }
    }
    assert!(
        stale.is_empty(),
        "stale schemas {stale:?}: run with CC_UPDATE_SCHEMAS=1 and commit"
    );
}
