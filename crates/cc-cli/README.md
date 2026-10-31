# cc-cli (`chipc`)

The command line. Logic is in `src/lib.rs` (`execute(&Cli, registry, &mut dyn Write)`) so tests drive it
in-process; `src/main.rs` parses arguments, sets up logging and calls it with `cc_passes::registry()`.
`anyhow` is used only here (conventions: errors).

## Usage

```
chipc [--config config.toml] [--set key=value]... [-v] <command>

  run [--target P]... [--from P | --only P] [--models a,b]
  verify <P>            re-run P's checks on its artifacts (+ outputs match manifest)
  status                one row per pass: status, fresh (would be a cache hit), finished, duration, key
  clean [P] [--yes]     remove build/.<P>/; without P (needs --yes) every pass dir and build/.pipeline/
  graph                 pass DAG as Graphviz DOT (`chipc graph | dot -Tsvg > dag.svg`)
  test [--deep] [--test-threads N] [-- <nextest args>]
  doctor                environment checks (stub; another change)
```

- `--config` defaults to `./config.toml`; `config.local.toml` beside it is merged if present.
- `--set` takes dotted keys, highest priority: `--set passes.hello_a.greeting=hi`,
  `--set resources.memory_budget_gb=16`.
- Logging: stderr, level `info` (`-v` debug, `-vv` trace, or `CHIPC_LOG=<filter>`). `run` also writes
  JSON logs to `build/.pipeline/logs/chipc-<ts>.jsonl`.

**Exit codes:** 0 success; 1 a pass failed (`run`) or a check failed (`verify`); 2 usage, config or DAG
error (printed as `error: …` with its cause chain).

## Building on the lab host

```bash
export CARGO_TARGET_DIR=/srv/scratch/$USER/chiplet-target   # local disk; /home is NFS, /tmp is 8 GB
cargo build -j 8 -p cc-cli && "$CARGO_TARGET_DIR/debug/chipc" run
```

## Adding a subcommand

Add a variant to `Command`, a match arm in `execute` that writes to `out` and returns an exit code, and a
case in `tests/smoke/cli.rs`.
