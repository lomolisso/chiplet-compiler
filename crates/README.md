# crates/

Rust workspace, created by another change. Planned crates (spec §3, §10; `docs/conventions.md`):

| Crate | Role |
|---|---|
| `cc-ir` | IR types (`Module`, `Op`, `TensorType`, symbolic `Dim`), the op-semantics table (StableHLO semantics only) |
| `cc-core` | config loading, artifact store and cache keys, `ToolRunner` (Docker / EDA / host), errors |
| `cc-pipeline` | pass registry, DAG, scheduler with memory admission, resume |
| `cc-smt` | SMT-LIB term builder, scalar op semantics, solver runner |
| `cc-passes` | one module per pass, each with its `README.md` |
| `cc-cli` | the `chipc` binary |
