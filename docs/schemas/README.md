# Artifact schemas

JSON Schemas (draft 2020-12, generated with `schemars` from the Rust types) of every artifact kind in
`build/`; each pass adds the schemas of its artifact kinds.

| File | Rust type | Written to |
|---|---|---|
| `manifest.schema.json` | `cc_core::Manifest` | `build/.<pass>/manifest.json` |
| `verify.schema.json` | `cc_core::VerifyReport` | `build/.<pass>/verify.json` |
| `export_model.schema.json` | `cc_passes::export::json::ModelFile` | `build/.export/out/export/<model>/export.json` |
| `export_graph.schema.json` | `cc_passes::export::json::GraphFile` | `build/.export/out/export/<model>/<phase>/graph.json` |
| `export_symbols.schema.json` | `cc_passes::export::json::SymbolsFile` | `build/.export/out/export/<model>/<phase>/symbols.json` |
| `export_census.schema.json` | `cc_passes::export::json::CensusFile` | `build/.export/out/export/<model>/<phase>/census.json` |
| `import_ir.schema.json` | `cc_ir::Module` | `build/.import/out/import/<model>/<phase>/ir.postcard` (postcard; the schema of its JSON form) |
| `import_summary.schema.json` | `cc_passes::import::json::Summary` | `build/.import/out/import/<model>/<phase>/ir.postcard.summary.json` |
| `import_op_inventory.schema.json` | `cc_passes::import::json::Inventory` | `build/.import/out/import/op_inventory.json` |
| `components_instances.schema.json` | `cc_passes::components::json::InstancesFile` | `build/.components/out/components/instances/<model>/<phase>.json` |
| `components_component.schema.json` | `cc_passes::components::json::StoredComponent` | `build/.components/out/components/components/<id>.postcard` (postcard; the schema of its JSON form) |
| `components_summary.schema.json` | `cc_passes::components::json::ComponentSummary` | `build/.components/out/components/components/<id>.postcard.summary.json` |
| `components_certificates.schema.json` | `cc_passes::components::json::CertificatesFile` | `build/.components/out/components/certificates/<model>/<phase>.json` |
| `normalize_component.schema.json` | `cc_passes::normalize::json::NormalizedComponent` | `build/.normalize/out/normalize/components/<id>.postcard` (postcard; the schema of its JSON form) |
| `normalize_summary.schema.json` | `cc_passes::normalize::json::NormalizedSummary` | `build/.normalize/out/normalize/components/<id>.postcard.summary.json` |
| `normalize_ledger.schema.json` | `cc_passes::normalize::json::LedgerFile` | `build/.normalize/out/normalize/ledger/<id>.json` |
| `normalize_rule_result.schema.json` | `cc_passes::normalize::json::RuleResult` | `build/.normalize/out/normalize/rules/<rule>.result` |
| `run_summary.schema.json` | `cc_pipeline::RunSummary` | `build/.pipeline/run-<ts>.json` |

The smoke test `schemas::committed_schemas_are_current` regenerates them and fails on drift. After
changing a type: `CC_UPDATE_SCHEMAS=1 cargo nextest run --profile smoke -E 'test(schemas)'`, then
commit the files.
