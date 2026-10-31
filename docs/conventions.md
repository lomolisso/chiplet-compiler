# Conventions

Rules every change follows. They complement `docs/spec.md`; when they disagree, the spec wins and this file is fixed in the same commit.

## Code

- **Separate responsibilities.**
  - One crate per layer: `cc-ir` (IR types, op-semantics table), `cc-core` (config, artifacts, tool runners, errors), `cc-pipeline` (registry, DAG, scheduler, cache), `cc-smt` (SMT-LIB generation and solver runner), `cc-passes` (one module per pass), `cc-cli` (the `chipc` binary).
  - A pass module never reaches into another pass's internals. It reads artifacts.
- **Patterns.**
  - `Pass` trait + registry.
  - Strategy traits for every config-selectable algorithm: one trait per decision, implementations named after their paper/method.
  - `ToolRunner` adapter for external tools.
  - Typed artifact repository.
  - Builders for configs and fixtures.
  - Prefer plain functions and data types to deep trait hierarchies.
- **Errors.** `thiserror` enums per crate, with context on what failed (model, phase, pass, artifact). `anyhow` only in `cc-cli`. No `unwrap()` outside tests.
- **Comments.** Short. Say *why*, cite the rule or paper (e.g. `// K2 convexity, spec §4.5`). No narration of what the code obviously does.
- **Style.** `cargo fmt` and `cargo clippy -D warnings` must pass. Python: `ruff`, type hints, no notebooks in `python/`.
- **No hard-coded mappings.** No table maps op names or model names to hardware. Op behaviour comes from the op-semantics table (`cc-ir`), which encodes StableHLO semantics only (arity, element-wise or not, commutativity, iteration semantics). Hardware decisions come from analyses (spec §7).
- **Determinism.** The same inputs and config give byte-identical artifacts. Sort before writing, seed every RNG from config, never iterate a `HashMap` into an output.

## Pass anatomy

```
crates/cc-passes/src/<pass>/
  mod.rs        Pass impl: requires / provides / resources / run / verify
  config.rs     #[derive(Deserialize)] #[serde(deny_unknown_fields)] section struct with defaults
  algo/         algorithms (Strategy impls), pure functions over cc-ir types
  verify.rs     independent checks (must not call the algorithm being checked)
  README.md     required (below)
```

**`README.md` of each pass.** Thorough, not fluffy:
1. Purpose and position in the pipeline.
2. Background: the theory and tools needed to understand it, with links to `docs/context/`.
3. Inputs and outputs: artifact kinds and the schema of each file in `build/.<pass>/out/`.
4. Algorithm, step by step, with complexity and the published source of every heuristic.
5. Config keys (and defaults).
6. Verification: what each check proves and what evidence it stores.
7. Resource behaviour (what scales with what).
8. Known limitations.

## Artifacts

- **Location.** Everything a pass writes is under `build/.<pass>/` (spec §3.4): `manifest.json`, `out/<kind>/` (one directory per artifact kind, through `ctx.output(kind)`), `views/`, `verify.json`, `logs/`.
- **Formats.** JSON (pretty, sorted keys) for anything a human or notebook reads; `postcard` for large binary graphs, always with a JSON summary next to it. Every artifact kind has a Rust type and a JSON Schema in `docs/schemas/<kind>.schema.json`.
- **Per-unit files.** Write one file per independent unit (model, phase, component, class, FU) so outputs stream to disk, partial results survive failures, and notebooks can load one item.
- **Certificates.** Every claim a later check relies on is written down: isomorphism mappings, partitions, clique solutions, configurations, SMT queries and solver answers.

## Configuration

- **Naming.** Sections are `[passes.<pass>]` and `[passes.<pass>.resources]`; the pass name is snake_case and equals the build directory name. Tool settings live in `[tools.<tool>]`; machine paths go to `config.local.toml`.
- **Contents.** Every key has a comment in `config.toml` with its default and meaning. New keys are added in the same commit as the code that reads them.
- **Strategy selection.** Selectable algorithms use `algorithm = "<name>"`, and a pass-specific sub-table holds that algorithm's parameters, e.g. `[passes.mine.mis] algorithm = "exact_ilp"`.

## Resources

- **Declaration.** Every pass declares `threads` and `memory_gb`; it never exceeds them. Iterate per unit and drop data before the next one.
- **No model-scale numerics.** Shapes are abstract. Numbers exist only at `testing.tiny_dim` scale (spec §8).
- **External calls.** Every external tool call goes through a `ToolRunner`, which applies caps and timeouts. Never call `eda-on` in-process; never `Command::new("dc_shell")` directly.
- **Long jobs on the shared machine.** Run them in the background with caps, and check `free -g` first.

## Tests

- **Placement:**
  - Unit tests live next to the code.
  - Integration tests live in `tests/smoke/` and `tests/deep/`, as the workspace test crate `tests/`.
  - Python tests live in `python/tests/`.
- **What each pass adds:**
  - **Smoke** (all 25 models, `reduced` config, CI budget: less than 2 min and 4 GB in total): preconditions, schema, invariants, and the pass run in full when that is seconds.
  - **Deep** (`testing.deep_models`, local): the full pass including tool calls, plus the pass's `verify`, plus golden snapshot checks (`insta`) of summary JSON.
- **Negative tests.** Every check in `verify.rs` has one: a mutated artifact must make it fail.
- **Running:** `cargo nextest run --profile smoke` / `--profile deep`, or `chipc test [--deep]`.

## Sources

- Primary sources only: upstream code and docs, papers.
