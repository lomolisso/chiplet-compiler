//! JSON Schemas of the framework's artifact kinds, committed under `docs/schemas/` (spec §3.4). A smoke
//! test regenerates them and fails on drift; set `CC_UPDATE_SCHEMAS=1` to rewrite the files.

use crate::manifest::Manifest;
use crate::verify::VerifyReport;

/// `(file name, schema)` pairs owned by `cc-core`.
pub fn core_schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "manifest.schema.json",
            serde_json::to_value(schemars::schema_for!(Manifest)).unwrap_or_default(),
        ),
        (
            "verify.schema.json",
            serde_json::to_value(schemars::schema_for!(VerifyReport)).unwrap_or_default(),
        ),
    ]
}
