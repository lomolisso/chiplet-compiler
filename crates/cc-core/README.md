# cc-core

Services every pass and the pipeline share. It knows no pass types (spec §3.4–§3.6).

| Module | Role |
|---|---|
| `config.rs` | layered config, strict fixed sections, opaque `[passes.<name>]` sections |
| `artifact.rs` | `ArtifactStore`, `PassDir`, `KindDir`: typed JSON / postcard I/O, content hashes |
| `manifest.rs` | `Manifest` (`build/.<pass>/manifest.json`) |
| `verify.rs` | `VerifyReport`, `Check`, `Evidence` (`build/.<pass>/verify.json`) |
| `cache.rs` | `cache_key` |
| `canon.rs` | canonical JSON, blake3 hashing, atomic writes |
| `cancel.rs` | `CancelToken` (cooperative cancellation) |
| `runner/` | `ToolRunner` trait; `HostRunner`, `DockerRunner`, `EdaRunner`; `EdaTool` (batch flags, native caps) |
| `schema.rs` | JSON Schemas of the manifest and verify report (`docs/schemas/`) |
| `error.rs` | `CoreError` (`thiserror`), `IoContext` |

## Configuration

- **Merge order:** built-in defaults (`Default` impls) < `config.toml` < `config.local.toml` (beside it)
  < `--set a.b.c=value`. Tables merge key by key; scalars and arrays are replaced. A `--set` value is
  parsed as TOML (`3`, `true`, `["a"]`), falling back to a plain string.
- **Strictness:** every fixed section is `deny_unknown_fields`. Each layer is first deserialized *alone*
  (all fields have defaults), so an unknown key or bad type is reported with the file that contains it
  and its key path (`serde_path_to_error`), e.g.
  `config /repo/config.local.toml: at \`paths\`: unknown field \`scratch\`, expected one of …`.
- **Pass sections:** `passes` is a map of opaque tables. `LoadedConfig::pass_section(name)` splits off
  `resources` (parsed strictly into `PassResources { threads, memory_gb, timeout_min }`, defaults 4/4/60)
  and returns the rest as a `PassSection`, which the pass parses into its own strict type
  (`section.parse::<T>()`). A section without a registered pass is not an error (planned passes' sections
  exist before their passes).
- **Paths** resolve against the directory of `config.toml`. **Models:** `models.keys = "all"` resolves to
  the sorted subdirectories of `paths.models_dir` holding a `recipe.py` (spec §2); explicit keys are
  checked against that catalog.

## Artifacts

```
build/.<pass>/manifest.json  verify.json  out/<kind>/…  views/  logs/
```

- Each artifact kind owns `out/<kind>/`; every file a pass writes there belongs to that kind.
- **JSON** (default): pretty, keys sorted recursively, trailing newline: byte-identical for equal values.
- **postcard** for large binary data: `write_postcard(rel, value, summary)` always
  writes `<rel>.summary.json` beside the binary file (conventions: formats).
- Writes are atomic (temporary sibling + rename; no fsync, which costs ~100 ms per file on the NFS home).
- Relative paths with `..` or a leading `/` are rejected: a pass writes only its own directory.
- **Content hash** of a kind: blake3 over its sorted `(relative path, blake3(file))` pairs.

## Tool runners

Every external process goes through a `ToolRunner` (spec §8). Each call:

- gets `Caps { memory_gb, cpus, timeout_s, pids_limit }`, a `CancelToken` and a `ToolLog`;
- writes `logs/<n>-<tool>.out|.err` and appends one JSON line to `logs/tools.jsonl` (command line, caps,
  start time, duration, exit code / signal, kill reason, peak memory);
- returns a `ToolOutcome`; `outcome.check()` turns a non-zero exit or a kill into an error.

| Runner | Isolation | Kill reasons |
|---|---|---|
| `HostRunner` | own process group (`process_group(0)`), `RLIMIT_CORE = 0`; watchdog sums `VmRSS`-equivalent RSS (`/proc/<pid>/stat` field 24) over all members of the group every `tools.host.watchdog_interval_s`, `killpg(SIGKILL)` over the cap | `memory_cap`, `timeout`, `cancelled` |
| `DockerRunner` | `docker run --init --name chipc-… --user uid:gid -e HOME=/tmp/home --memory M --memory-swap M --memory-swappiness 0 --cpus C --pids-limit P --ulimit core=0 --network none --mount type=bind,…` (`docs/context/tooling/docker_resources.md` §6). The container is named (not `--rm`) so `OOMKilled` can be read, then removed. Writable mount sources are created first (`--mount` does not create them). Peak memory from the container's cgroup (`--cidfile`). | `oom_killed` (kernel, inside the container's cgroup), `timeout` / `cancelled` (`docker kill`) |
| `EdaRunner` | host process group via the verified `bash --norc -i -c '. "$EDA_TOGGLE"; eda-run "$@"' eda-run <tool> …` (`docs/context/tooling/eda_environment.md` §2), inheriting the session env (Lmod); `Invocation::cwd` (a job dir) is required; `TMPDIR` under `paths.scratch_dir`; `MemAvailable` checked before launch; same RSS watchdog and timeout as `HostRunner`; at most `tools.eda.max_licenses` calls at once; bash's no-TTY lines dropped from stderr; output matching `tools.eda.license_error_patterns` is retried with doubling backoff and then reported as `CoreError::License` (never a tool result). `EdaTool::{batch_args, cap_preamble}` give each tool's batch flags and native thread/memory caps | `memory_cap`, `timeout`, `cancelled` |

`BoundRunner` (what `PassContext` hands out) fixes the caps, log and token for one pass;
`with_caps` can only lower them. `RLIMIT_AS` is not used: it limits virtual memory, which JVMs and
thread pools reserve far beyond their resident use.

## Patterns

Repository (`ArtifactStore`), Adapter (`ToolRunner`), Builder (`Invocation`, `Check`), typed errors
(`CoreError` with the failing path / pass / kind / tool). No `unwrap` outside tests; the only `unsafe` is
in `runner/process.rs` (libc: `setrlimit` in `pre_exec`, `killpg`, `sysconf`, `getuid`).
