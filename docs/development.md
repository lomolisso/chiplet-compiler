# Development

How code, artifacts, configuration, machines and tests are organised. This file complements
[design.md](design.md); when they disagree, the design wins and this file is fixed in the same change. The
contribution workflow (branches, commits, pull requests, changelog) is in [CONTRIBUTING.md](../CONTRIBUTING.md).

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
- **Comments.** Short. Say *why*, cite the rule or paper (e.g. `// K2 convexity, design §4.5`). No narration of what the code obviously does.
- **Style.** `cargo fmt` and `cargo clippy -D warnings` must pass. Python: `ruff`, type hints, no notebooks in `python/`.
- **No hard-coded mappings.** No table maps op names or model names to hardware. Op behaviour comes from the op-semantics table (`cc-ir`), which encodes StableHLO semantics only (arity, element-wise or not, commutativity, iteration semantics). Hardware decisions come from analyses (design §7).
- **Determinism.** The same inputs and config give byte-identical artifacts. Sort before writing, seed every RNG from config, never iterate a `HashMap` into an output.

## Pass anatomy

```
crates/cc-passes/src/<pass>/
  mod.rs        Pass impl: requires / provides / resources / run / verify
  config.rs     #[derive(Deserialize)] #[serde(deny_unknown_fields)] section struct with defaults
  algo/         algorithms (Strategy impls), pure functions over cc-ir types
  verify.rs     independent checks (must not call the algorithm being checked)
docs/passes/<pass>.md   required (below)
```

**`docs/passes/<pass>.md` for each pass.** Thorough, not fluffy:
1. Purpose and position in the pipeline.
2. Background: the theory and tools needed to understand it, citing primary sources (papers, upstream docs)
   and linking `docs/tools/`.
3. Inputs and outputs: artifact kinds and the schema of each file in `build/.<pass>/out/`.
4. Algorithm, step by step, with complexity and the published source of every heuristic.
5. Config keys (and defaults).
6. Verification: what each check proves and what evidence it stores.
7. Resource behaviour (what scales with what).
8. Known limitations.

## Artifacts

- **Location.** Everything a pass writes is under `build/.<pass>/` (design §3.4): `manifest.json`, `out/<kind>/` (one directory per artifact kind, through `ctx.output(kind)`), `views/`, `verify.json`, `logs/`.
- **Formats.** JSON (pretty, sorted keys) for anything a human or notebook reads; `postcard` for large binary graphs, always with a JSON summary next to it. Every artifact kind has a Rust type and a JSON Schema in `docs/schemas/<kind>.schema.json`.
- **Per-unit files.** Write one file per independent unit (model, phase, component, class, FU) so outputs stream to disk, partial results survive failures, and notebooks can load one item.
- **Certificates.** Every claim a later check relies on is written down: isomorphism mappings, partitions, clique solutions, configurations, SMT queries and solver answers.

## Configuration

- **Naming.** Sections are `[passes.<pass>]` and `[passes.<pass>.resources]`; the pass name is snake_case and equals the build directory name. Tool settings live in `[tools.<tool>]`; machine paths go to `config.local.toml`.
- **Contents.** Every key has a comment in `config.toml` with its default and meaning. New keys are added in the same commit as the code that reads them.
- **Strategy selection.** Selectable algorithms use `algorithm = "<name>"`, and a pass-specific sub-table holds that algorithm's parameters, e.g. `[passes.mine.mis] algorithm = "exact_ilp"`.

## Resources

- **Declaration.** Every pass declares `threads` and `memory_gb`; it never exceeds them. Iterate per unit and drop data before the next one.
- **No model-scale numerics.** Shapes are abstract. Numbers exist only at `testing.tiny_dim` scale (design §8).
- **External calls.** Every external tool call goes through a `ToolRunner`, which applies caps and timeouts. Never call `eda-on` in-process; never `Command::new("dc_shell")` directly.
- **Long jobs on the shared machine.** Run them in the background with caps, and check `free -g` first.

## Build host and scratch

Two machines share the NFS home, so the checkout has the same path on both.

- **Orchestration host (iacoma227).** Editing, git and `gh`, reading results. Nothing
  compiles, builds an image, runs a container, runs `chipc` or runs a test there: its local scratch is ~95 % full
  with other users' data, and it is a Condor execute node.
- **Build host (iacoma231).** Everything else, through `scripts/on-build-host <command>` (ssh, same checkout path,
  the build environment below). Check it first with `scripts/on-build-host --status` (load, memory, free scratch).
- **Build area:** `/scratch/$USER/chiplet-compiler/` on the build host (local disk; `/scratch` = `/srv/scratch`, the
  same disk as Docker's `/srv`). `config.local.toml` points `paths.scratch_dir` and `paths.build_dir` there.

| Path | Kind | Holds |
|---|---|---|
| `target/` | persistent, rebuildable | `CARGO_TARGET_DIR` |
| `venv/` | persistent, rebuildable | the locked Python stack (`UV_PROJECT_ENVIRONMENT`) |
| `build/` | persistent, rebuildable | `paths.build_dir` of the main checkout: the latest pass artifacts |
| `logs/` | persistent (small) | `docker/build.sh` logs |
| `runs/<name>/` | ephemeral | a worktree's build dir (`scripts/build-area init`), ad-hoc runs, deep-test build dirs |
| `tmp/` | ephemeral | `TMPDIR`, per-call EDA temp (`tmp/eda/`) |
| image `chiplet-compiler/toolchain:<tag>` | persistent, rebuildable | the current tag only, in Docker's `/srv/docker` |

- **Cleanup is mandatory.** Delete an ephemeral run when it has served its purpose, at the latest when its
  worktree is removed (`scripts/build-area clean <name>`); tests that build into `runs/` remove their directory when they pass (a failed run is kept for
  inspection, then removed). When the image tag changes, remove the old tag by name. Never prune Docker globally
  (`docker system|image|builder prune`): the daemon is shared.
- **Disk guard.** Keep at least 100 GB free on the build host's `/scratch` (`docker/build.sh` refuses below that);
  if a run would go lower, stop and ask.
- **Moving the image to another host:** `docker save <image> | ssh <host> docker load` (no temp file), not a rebuild.

## Tests

- **Placement:**
  - Unit tests live next to the code.
  - Integration tests live in `tests/smoke/` and `tests/deep/`, as the workspace test crate `tests/`.
  - Python tests live in `python/tests/`.
- **What each pass adds:**
  - **Smoke** (all 25 models, `reduced` config, CI budget: less than 2 min and 4 GB in total): preconditions, schema, invariants, and the pass run in full when that is seconds.
  - **Deep** (`testing.deep_models`, local): the full pass including tool calls, plus the pass's `verify`, plus golden snapshot checks (`insta`) of summary JSON.
- **Negative tests.** Every check in `verify.rs` has one: a mutated artifact must make it fail.
- **Running** (on the build host): `scripts/on-build-host cargo nextest run --profile smoke` / `--profile deep`,
  or `scripts/on-build-host chipc test [--deep]`.

## Sources

- Primary sources only: upstream code and docs, papers, measurements on our machines (with the date).
- Cite them where the fact is used: the pass's `docs/passes/<pass>.md`, or the tool's note in `docs/tools/`.
