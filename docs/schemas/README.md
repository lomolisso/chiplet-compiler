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
| `export_census_concrete.schema.json` | `cc_passes::export::json::CensusConcreteFile` | `build/.export/out/export/<model>/<phase>/census_concrete.json` |
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
| `kernels_partition.schema.json` | `cc_passes::kernels::json::KernelPartition` | `build/.kernels/out/kernels/components/<id>.json` |
| `kernels_graphs.schema.json` | `cc_passes::kernels::json::KernelGraphs` | `build/.kernels/out/kernels/kernels/<id>.postcard` (postcard; the schema of its JSON form) |
| `kernels_component_summary.schema.json` | `cc_passes::kernels::json::PartitionSummary` | `build/.kernels/out/kernels/kernels/<id>.postcard.summary.json` |
| `kernels_summary.schema.json` | `cc_passes::kernels::json::KernelsSummary` | `build/.kernels/out/kernels/summary.json` |
| `kernel_classes_class.schema.json` | `cc_passes::kernel_classes::json::ClassFile` | `build/.kernel_classes/out/kernel_classes/classes/<z>.json` |
| `kernel_classes_instances.schema.json` | `cc_passes::kernel_classes::json::InstancesTable` | `build/.kernel_classes/out/kernel_classes/instances/<z>.json` |
| `kernel_classes_certificates.schema.json` | `cc_passes::kernel_classes::json::CertificatesFile` | `build/.kernel_classes/out/kernel_classes/certificates/<z>.json` |
| `kernel_classes_sharing.schema.json` | `cc_passes::kernel_classes::json::Sharing` | `build/.kernel_classes/out/kernel_classes/sharing.json` |
| `kernel_classes_summary.schema.json` | `cc_passes::kernel_classes::json::ClassesSummary` | `build/.kernel_classes/out/kernel_classes/summary.json` |
| `iteration_space_skeleton.schema.json` | `cc_passes::iteration_space::json::Skeleton` | `build/.iteration_space/out/iteration_space/skeletons/<z>.json` |
| `iteration_space_dependences.schema.json` | `cc_passes::iteration_space::json::Dependences` | `build/.iteration_space/out/iteration_space/dependences/<z>.json` |
| `iteration_space_template.schema.json` | `cc_passes::iteration_space::json::TemplateFile` | `build/.iteration_space/out/iteration_space/templates/<z>.json` |
| `iteration_space_embedding.schema.json` | `cc_passes::iteration_space::json::Embedding` | `build/.iteration_space/out/iteration_space/embedding.json` |
| `iteration_space_routing.schema.json` | `cc_passes::iteration_space::json::Routing` | `build/.iteration_space/out/iteration_space/routing.json` |
| `iteration_space_proof_result.schema.json` | `cc_passes::iteration_space::json::ProofResult` | `build/.iteration_space/out/iteration_space/smt/<file>.result` |
| `iteration_space_summary.schema.json` | `cc_passes::iteration_space::json::Summary` | `build/.iteration_space/out/iteration_space/summary.json` |
| `run_summary.schema.json` | `cc_pipeline::RunSummary` | `build/.pipeline/run-<ts>.json` |

The smoke test `schemas::committed_schemas_are_current` regenerates them and fails on drift. After
changing a type: `CC_UPDATE_SCHEMAS=1 cargo nextest run --profile smoke -E 'test(schemas)'`, then
commit the files.
