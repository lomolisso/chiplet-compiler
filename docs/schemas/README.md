# schemas/

JSON Schemas (draft 2020-12, generated with `schemars` from the Rust types) of every artifact kind in
`build/`, generated from the Rust types; each pass adds its own.

| File | Rust type | Written to |
|---|---|---|
| `manifest.schema.json` | `cc_core::Manifest` | `build/.<pass>/manifest.json` |
| `verify.schema.json` | `cc_core::VerifyReport` | `build/.<pass>/verify.json` |
| `run_summary.schema.json` | `cc_pipeline::RunSummary` | `build/.pipeline/run-<ts>.json` |

The smoke test `schemas::committed_schemas_are_current` regenerates them and fails on drift. After
changing a type: `CC_UPDATE_SCHEMAS=1 cargo nextest run --profile smoke -E 'test(schemas)'`, then
commit the files.
