# tests/

Workspace integration tests: the package `tests` (created by another change). Unit tests live next to the code.

| Path | Scope | Where it runs |
|---|---|---|
| `src/lib.rs` | shared helpers: `Env` (temporary repo root + `config.toml`), `Fixture` (configurable pass: sleep, fail, panic, ignore cancel, fail verify, call Docker), `RecordingRunner` (captures invocations and caps) | - |
| `smoke/` | every pass on **all 25 models** at the `reduced` config: preconditions, schemas, invariants; full runs only when they take seconds. Budget: < 2 min, < 4 GB total. another change: DAG and config errors, cache keys, scheduler (overlap, admission, resume, cancellation, timeouts), runners, schemas, verify negative tests, CLI. | CI on every push |
| `deep/` | the full pass, including external tools, on `[testing] deep_models`. another change: Docker memory cap / OOM, user mapping, timeout; host watchdog and timeout. | locally: `chipc test --deep` |
| `fixtures/` | small hand-made graphs, reports and artifacts used by both | - |

Every check in a pass's `verify.rs` has a negative test (a mutated artifact must fail it).

## Running

```bash
export CARGO_TARGET_DIR=/srv/scratch/$USER/chiplet-target
cargo nextest run --profile smoke            # or: chipc test
free -g && cargo nextest run --profile deep  # or: chipc test --deep  (needs Docker; serialised in `heavy`)
```

Profiles and test groups are in `.config/nextest.toml`: `smoke` = everything except the `deep` binary,
`deep` = only the `deep` binary, in the `heavy` group (`max-threads = 1`). Plain `cargo test` also runs
the deep binary; use nextest.
