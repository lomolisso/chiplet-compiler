# Tests (`tests/`)

Workspace integration tests: the package `tests`. Unit tests live next to the code.

| Path | Scope | Where it runs |
|---|---|---|
| `src/lib.rs` | shared helpers: `Env` (temporary repo root + `config.toml`), `Fixture` (configurable pass: sleep, fail, panic, ignore cancel, fail verify, call Docker), `RecordingRunner` (captures invocations and caps) | - |
| `smoke/` | every pass on **all 25 models** at the `reduced` config: preconditions, schemas, invariants; full runs only when they take seconds. Budget: < 2 min, < 4 GB total. Framework: DAG and config errors, cache keys, scheduler (overlap, admission, resume, cancellation, timeouts), runners, schemas, verify negative tests, CLI. | CI on every push |
| `deep/` | the full pass, including external tools, on `[testing] deep_models`. Framework: Docker memory cap / OOM, user mapping, timeout; host watchdog and timeout. | locally: `chipc test --deep` |
| `fixtures/` | small hand-made graphs, reports and artifacts used by both | - |

Every check in a pass's `verify.rs` has a negative test (a mutated artifact must fail it).

## Running

On the build host (docs/development.md, "Build host and scratch"):

```bash
scripts/on-build-host cargo nextest run --profile smoke   # or: scripts/on-build-host chipc test
scripts/on-build-host --status                            # before deep: load, memory, free scratch
scripts/on-build-host cargo nextest run --profile deep    # or: … chipc test --deep  (Docker; serialised in `heavy`)
```

Deep tests build into the ephemeral `runs/` of the build area and remove what they built when they pass.

Smoke tests that prove rules (`smoke/normalize.rs`) call real solvers: the toolchain image's on the build
host, or host binaries on `PATH` when `CC_SMT_RUNNER=host` (CI downloads the pinned z3 and bitwuzla
releases and sets it; docs/framework/smt.md).

Golden snapshots (`insta`, `tests/deep/snapshots/`) pin summary digests of deep runs. After an intended
change, rerun with `INSTA_UPDATE=always` (or review with `cargo insta review`) and commit the new `.snap`.

Profiles and test groups are in `.config/nextest.toml`: `smoke` = everything except the `deep` binary,
`deep` = only the `deep` binary, in the `heavy` group (`max-threads = 1`). Plain `cargo test` also runs
the deep binary; use nextest.
