# cc-pipeline

The pipeline manager: the `Pass` trait, the pass DAG, the scheduler with admission control, caching and
resume (spec §3.1–§3.3). It knows nothing about any particular pass; `cc-passes` registers them.

## Modules

| Module | Role |
|---|---|
| `pass.rs` | `Pass` trait, `ToolSpec`, `RunnerKind` |
| `context.rs` | `PassContext`: what a running pass may see and do |
| `dag.rs` | `PassGraph`: DAG from `requires`/`provides`, validation, closures, Kahn order, DOT |
| `admission.rs` | the admission predicate (pure, unit-tested) |
| `pipeline.rs` | `Pipeline`: plan → cache check → schedule → manifests → run summary; `verify`, `status`, `clean` |
| `summary.rs` | `RunSummary` (`build/.pipeline/run-<ts>.json`, schema in `docs/schemas/run_summary.schema.json`) |
| `versions.rs` | tool version probes for cache keys, memoised per run |

## The pass contract

```rust
pub trait Pass: Send + Sync {
    fn name(&self) -> &'static str;          // == [passes.<name>] == build/.<name>/
    fn version(&self) -> u32;                 // bump when outputs change meaning
    fn requires(&self) -> &[ArtifactKind];
    fn provides(&self) -> &[ArtifactKind];
    fn resources(&self, declared: &PassResources) -> PassResources { *declared }
    fn tools(&self) -> &[ToolSpec] { &[] }    // versions enter the cache key
    fn model_dependent(&self) -> bool { true } // false: `--models` does not invalidate it
    fn check_config(&self, section: &PassSection) -> Result<(), CoreError>;  // strict parse, up front
    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError>;
    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError>;
}
```

Compared with the sketch in spec §3.1, three methods are added: `tools` (spec §3.3 puts tool versions in
the cache key, so the pass must name its tools), `model_dependent` (so `--models` does not rerun
model-independent passes such as `characterize`), and `check_config` (every section is parsed strictly
before anything runs, so a typo in a late pass's section fails the run immediately, not hours later).

`PassContext` enforces the pass rule: `ctx.input(kind)` errors for a kind not in `requires()`,
`ctx.output(kind)` for one not in `provides()`. Each kind owns `build/.<pass>/out/<kind>/`. Runners from
`ctx.host()`, `ctx.docker()`, `ctx.eda()` are pre-capped (below) and log into `build/.<pass>/logs/`.

## DAG

Nodes are passes; an edge `p → q` labelled `k` means `p` provides kind `k` and `q` requires it. Build
errors (`DagError`): bad pass name (not snake_case), bad kind name, duplicate pass, duplicate provider,
missing provider, cycle (reported with its members via Tarjan SCC; a pass requiring its own output is a
self-loop). `pipeline.targets`, `--target`, `--from`, `--only` must name registered passes.

## Plan

- **Targets:** `--target` if given, else `pipeline.targets`, else every pass. The plan is the targets plus
  all their ancestors.
- **`--from P`:** P and its descendants (within the plan) are *forced*: they run even on a cache hit.
- **`--only P`:** the plan is `{P}`, forced; its inputs must already have `ok` manifests, otherwise P fails
  with "input … is not available".

## Scheduler

Kahn-style over the plan, inside one `std::thread::scope`:

1. A pending pass whose planned predecessor did not succeed is **skipped** (transitively).
2. A pass whose predecessors all succeeded is **ready**. Ready passes are visited in name order
   (deterministic). For each: compute its cache key; if it is not forced and is a cache hit, it is done
   without running.
3. Otherwise it is **admitted** if the admission rule holds, and runs on its own scoped OS thread with its
   own `rayon::ThreadPool` of `resources.threads` threads; `run` then `verify` execute inside
   `pool.install`, so `par_iter` inside a pass uses exactly its declared threads (never the global pool).
4. The main loop waits for completion events (10 ms ticks), enforces timeouts, and repeats until nothing
   is pending or running.

A pass is **ok** only if `run` returned `Ok`, `verify` returned a report with no failed check, and its
outputs and manifest were written. A panic in the pass is caught and reported as a failure.

### Admission rule

With `L` the set of running passes, a ready pass `p` is admitted iff `L` is empty, or all of

- `|L| < pipeline.max_parallel_passes`,
- `Σ_{q∈L} memory_gb(q) + memory_gb(p) ≤ resources.memory_budget_gb`,
- `Σ_{q∈L} threads(q) + threads(p) ≤ resources.threads_total`.

The "`L` is empty" clause is the **oversize rule**: a pass whose own declaration exceeds a limit could
never satisfy it, so it waits until nothing else runs and then runs alone (with a warning). Ready passes
that do not fit are retried at the next event; a smaller ready pass may start ahead of a larger one
(backfilling). Admission uses *declared* resources (spec §8); passes must stay within them.

### Failure, cancellation, timeouts

- **Default (`stop_on_failure = false`):** a failed pass skips its dependents; independent branches run
  to completion.
- **`stop_on_failure = true`:** on the first failure, nothing new is admitted (remaining ready passes are
  skipped) and every running pass's `CancelToken` is set. Passes check it between work units
  (`ctx.check_cancelled()`, `token.sleep`), and runners kill their child process group / container. Such
  passes end `cancelled`.
- **Timeouts:** when a pass exceeds `resources.timeout_min`, its token is set the same way; it ends
  `failed` with "timeout after N min".
- Cancellation is cooperative: a pass that never checks the token and calls no runner runs to the end.

## Cache keys and resume

```
key = blake3( canonical_json {
        pass, pass_version,
        inputs:  sorted [(kind, content hash from the producer's manifest)],
        config:  the [passes.<name>] section without `resources`,
        models:  sorted model set (empty unless model_dependent),
        tools:   sorted {tool: first line of `<program> <version_args>`} } )
```

- Canonical JSON sorts every object key, so TOML key order never changes a key.
- `resources` is excluded: threads and memory never change outputs (determinism rule).
- A kind's content hash is blake3 over the sorted `(relative path, blake3(file))` pairs of
  `out/<kind>/`.

**Cache hit** ⇔ the pass's `manifest.json` has `status = ok`, the same key, *and* every output directory
still hashes to the recorded value (a hand-edited or partially deleted output reruns the pass).

**Resume** falls out of this: after a failure, a new `chipc run` skips everything still `ok` and current,
and reruns the failed pass and whatever depends on it. Each pass resets its own `out/`, `views/`, `logs/`
and `verify.json` before running, writes `status = running` first, and the final manifest last, so a
crash mid-pass is never mistaken for a success.

`chipc verify P` re-runs `P.verify` on the existing artifacts plus the framework check
`framework.outputs_match_manifest`; a failure sets the manifest to `failed`, which invalidates the cache.

## Observability

- `tracing` span `pass{pass=…}` around each execution; `chipc` writes human logs to stderr and JSON
  logs of a run to `build/.pipeline/logs/chipc-<ts>.jsonl`.
- Per pass: `manifest.json` (timings, peak RSS, error), `verify.json`, `logs/tools.jsonl` (every tool
  call: command line, caps, duration, exit, kill reason, peak RSS) with stdout/stderr files.
- Per run: `build/.pipeline/run-<ts>.json` with each pass's status, cache key, start/end offsets
  (monotonic ms since run start), duration, peak RSS and resources.
- Peak RSS of a pass is the driver process's RSS sampled during the pass window (25 ms period). With
  overlapping passes it is an upper bound; external tools report their own peaks in `tools.jsonl`.

## How to add a pass (checklist, spec §3.2)

1. `crates/cc-passes/src/<name>/` with `mod.rs` (the `Pass` impl), `config.rs`
   (`#[serde(deny_unknown_fields, default)]` struct), `algo/` (strategies), `verify.rs` (independent
   checks), `README.md` (the eight sections of `docs/conventions.md`). `hello_a`/`hello_b` are templates.
2. Pick artifact kinds (`ArtifactKind("<name>.<what>")`) for `provides()`; list the kinds you read in
   `requires()`. Read and write only through `ctx.input(kind)` / `ctx.output(kind)`.
3. Implement `check_config` as `section.parse::<YourConfig>().map(drop)`.
4. Declare tools in `tools()` and call them only through `ctx.host()`, `ctx.docker()`, `ctx.eda()`.
5. Return `false` from `model_dependent()` if the outputs do not depend on the model set.
6. One line in `crates/cc-passes/src/registry.rs`.
7. One `[passes.<name>]` section (with `[passes.<name>.resources]`) in `config.toml`, every key commented.
8. Schemas for new artifact kinds in `docs/schemas/`.
9. Smoke tests (`tests/smoke/`) and deep tests (`tests/deep/`), with a negative (mutation) test per
   `verify.rs` check.

Nothing else changes: the order comes from the DAG.
