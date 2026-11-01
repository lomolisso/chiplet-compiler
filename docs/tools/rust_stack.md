# Rust stack (crates, toolchain, solvers)

Crate versions are the latest stable on crates.io on 2026-10-30 (crates.io API). Pin minor versions in the
workspace `Cargo.toml` (`[workspace.dependencies]`) and commit `Cargo.lock`. Crate layout per
`docs/development.md`: `cc-ir`, `cc-core`, `cc-pipeline`, `cc-smt`, `cc-passes`, `cc-cli`.

## 1. Crates

| Crate | Version (MSRV) | License | Why / how we use it |
|---|---|---|---|
| `petgraph` | 0.8.3 (1.64) | MIT/Apache-2.0 | Graph data structures (`StableGraph`/`Graph`), toposort, SCC, dominators. **Its VF2 subgraph matching is node-induced only** (source doc: "'subgraph' always means a 'node-induced subgraph'. Edge-induced subgraph isomorphisms are not directly supported"); not what MNI counting over dataflow needs. |
| `vf2` | 1.0.1 (–) | MIT/Apache-2.0 | `subgraph_isomorphisms` (non-induced / monomorphism), `induced_subgraph_isomorphisms`, `isomorphisms`; `node_eq`/`edge_eq` hooks; `.first()`, `.iter()` (use `next_ref` to avoid clones), `.vec()` (memory warning); directed or undirected. Its optional petgraph integration targets **petgraph ^0.6**, so with petgraph 0.8 implement `vf2::Graph` for our own adjacency wrapper (`cc_ir::absgraph`, built with `default-features = false`). Measured in 1.0.1's source: the look-ahead rules are not implemented (`rule_in`, `rule_out`, `rule_new` return true) and candidate nodes are found by a linear scan, so a search is at least O(V²); strong node labels (stable WL colours) keep it fast. Last release 2024-09. |
| `rayon` | 1.12.0 (1.80) | MIT/Apache-2.0 | Data parallelism. Per pass: `rayon::ThreadPoolBuilder::new().num_threads(n).build()?` then `pool.install(|| …)` so each pass honours `[passes.<p>].threads`; never rely on the global pool (it sizes to all 64 CPUs unless `RAYON_NUM_THREADS` is set). For uneven work items use `par_iter().with_min_len(k)` to bound task granularity (closest analogue to OpenMP `schedule(dynamic, k)`; rayon itself is work-stealing). |
| `serde` | 1.0.229 | MIT/Apache-2.0 | derive for all artifacts. |
| `serde_json` | 1.0.154 (1.71) | MIT/Apache-2.0 | JSON artifacts (pretty, sorted keys: serialize via `BTreeMap` or `serde_json::Value` canonicalisation). |
| `toml` | 1.1.8 (1.85) | MIT/Apache-2.0 | `config.toml` / `config.local.toml`. Note the 1.x line implements TOML spec 1.1. |
| `clap` | 4.6.7 (1.85) | MIT/Apache-2.0 | `chipc` CLI (derive API). |
| `postcard` | 1.1.3 | MIT/Apache-2.0 | Compact binary for large graphs (with a JSON summary beside it). Chosen over `bincode`: RUSTSEC-2025-0141 marks bincode **unmaintained** ("the bincode team has taken the decision to cease development permanently"; 1.3.3 declared final; suggested alternatives: wincode, postcard, bitcode, rkyv). A `bincode` 3.0.0 exists on crates.io (2025-12-16) but is covered by the same advisory: do not use. |
| `blake3` | 1.8.7 | CC0/Apache-2.0 | Content hashes for the pass cache key (inputs + config section + tool versions). Use `blake3::Hasher` with streamed file reads. Its `rayon` feature (multithreaded hashing) is unnecessary. |
| `easy-smt` | 0.3.2 | MIT/Apache-2.0 | SMT-LIB2 over a **solver subprocess** (crate: "interacting with an SMT solver subprocess"). `ContextBuilder::new().with_z3_defaults()` spawns `z3 -smt2 -in -v:0`; `with_cvc5_defaults()`; generic `.solver(prog).solver_args([...])` (source `src/context.rs` on main; confirm in 0.3.2); `.replay_file(Some(File))` records the session → our SMT certificate; `check()` → `Response`, `get_value()`. |
| `z3` (alternative) | 0.22.0 (1.88) | MIT | In-process bindings to libz3. Rejected as default: links a C++ library (build/ABI friction on RHEL 8, a solver crash takes down the pass, no external memory cap). Keep `cc-smt` solver-agnostic via SMT-LIB text. |
| `tracing` + `tracing-subscriber` | 0.1.44 / 0.3.23 (1.65) | MIT | Structured logs, spans per pass/model; JSON log layer into `build/.<pass>/log.jsonl`. |
| `thiserror` / `anyhow` | 2.0.21 / 1.0.104 | MIT/Apache-2.0 | errors per crate / CLI only (conventions). |
| `memory-stats` | 1.2.0 | MIT/Apache-2.0 | `memory_stats()` → `physical_mem` (RSS) and `virtual_mem`; reads `/proc/self/smaps` when available (expensive), feature `always_use_statm` uses `/proc/self/statm`. Use for in-process peak-RSS checks and test budgets (smoke < 4 GB). For child tools use `/proc/<pid>/status` (see `eda_environment.md`). |
| `insta` (dev) | 1.49.0 (1.66) | Apache-2.0 | Golden snapshots of summary JSON in deep tests (`assert_json_snapshot!`); review with `cargo insta review`. |
| `proptest` (dev) | 1.12.0 (1.88) | MIT/Apache-2.0 | Property tests (graph invariants, negative tests by mutation). Set `PROPTEST_CASES` low in smoke. |
| `cargo-nextest` (tool) | 0.9.150 (1.91 to build) | Apache-2.0/MIT | Test runner with profiles and test groups (below). Install the prebuilt binary rather than `cargo install`. |

MSRV implied by the set: **1.91** to build `cargo-nextest` from source; libraries need ≥ 1.88 (`proptest`, `z3`).
Use the current stable toolchain, pinned in `rust-toolchain.toml` (`channel = "1.xx"`, components `rustfmt`, `clippy`).

## 2. nextest configuration (`.config/nextest.toml`)

```toml
[test-groups]
heavy  = { max-threads = 1 }      # tests that spawn Docker/EDA tools or use > 1 GB
solver = { max-threads = 4 }

[profile.smoke]
default-filter = "kind(test) & package(tests) & test(/^smoke::/)"   # filter syntax (unverified); check nextest filtersets doc
fail-fast = false

[profile.deep]
slow-timeout = { period = "300s", terminate-after = 4 }

[[profile.deep.overrides]]
filter = "test(/::heavy::/)"
test-group = "heavy"
```
Test groups use `[test-groups] name = { max-threads = N }` and `[[profile.<p>.overrides]] filter = …,
test-group = …`; `threads-required` can be combined (nextest docs). Also cap the global runner with
`--test-threads N` on this shared host.

## 3. Toolchain install without root

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
# installs to $RUSTUP_HOME (~/.rustup) and $CARGO_HOME (~/.cargo); set both before running to relocate
. "$HOME/.cargo/env"
rustup component add rustfmt clippy
```
- `/home/rid2` is NFS at 97 %: keep `~/.rustup`/`~/.cargo` there (a few GB), but build on the build host's local
  disk: `scripts/on-build-host cargo …` sets `CARGO_TARGET_DIR=/scratch/$USER/chiplet-compiler/target`
  (docs/development.md, "Build host and scratch").
- Build parallelism: `cargo build -j N` / `CARGO_BUILD_JOBS=N` (default = all CPUs).
- rustup docs: https://rust-lang.github.io/rustup/installation/other.html (the `-y`, `--profile`, `--no-modify-path`
  flags of `rustup-init`).

## 4. SMT solver binaries for `easy-smt`

| Solver | Latest release (GitHub, 2026-10-30) | Host (glibc 2.28) | Docker |
|---|---|---|---|
| z3 | `z3-5.3.3` (2026-10-29), assets `z3-5.3.3-x64-glibc-2.39.zip`, `…-arm64-glibc-2.38.zip`, `…-linux-x64.jar` | **x64 zip needs glibc 2.39 → won't run on RHEL 8** | Ubuntu 24.04 (glibc 2.39) OK |
| bitwuzla | `0.9.1` (2026-05-21), `Bitwuzla-Linux-x86_64-static.zip` | not fully static: `objdump -p` lists `libgmp.so.10`, `libmpfr.so.6`, `libstdc++` → needs those on the host | OK (image installs `libgmp10`, `libmpfr6`) |

- Install to `~/.local/opt/<solver>-<ver>/` (host) or `/opt` (image); expose through `config.toml`
  `[tools.smt] z3 = "…", bitwuzla = "…"`, never via `PATH` lookup alone; record `--version` in certificates.
- Bitwuzla in `easy-smt`: `.solver("bitwuzla").solver_args([...])`; interactive/incremental SMT-LIB2 stdin flags
  for bitwuzla (e.g. `--lang smt2`) (unverified); test once and freeze in config.
- Memory: solver subprocesses go through `ToolRunner` (Docker or RSS watchdog); z3 also accepts `-memory:<MB>` and
  `-T:<sec>` command-line limits (z3 `-h`; (unverified) for 5.x).

## 5. Rationale summary

- Graph algorithms in Rust (petgraph + vf2) keep mining verification independent of GraMi's Java code.
- Solvers out of process → crash isolation, memory caps, replayable `.smt2` certificates.
- JSON for anything reviewed, postcard for bulk, blake3 for cache keys: deterministic and fast.

## Sources

- crates.io API (`https://crates.io/api/v1/crates/<name>`) for versions, licenses, MSRV, 2026-10-30.
- petgraph 0.8.3 `src/algo/isomorphism.rs` doc comment (node-induced); https://docs.rs/petgraph/0.8.3/petgraph/algo/isomorphism/
- vf2: https://docs.rs/vf2/latest/vf2/
- RUSTSEC-2025-0141: https://rustsec.org/advisories/RUSTSEC-2025-0141.html
- rayon `ThreadPoolBuilder`, `ThreadPool::install`, `IndexedParallelIterator::with_min_len`: https://docs.rs/rayon/1.12.0/rayon/
- easy-smt: https://docs.rs/easy-smt/0.3.2/easy_smt/ ; source https://github.com/elliottt/easy-smt/blob/main/src/context.rs
- memory-stats: https://docs.rs/memory-stats/1.2.0/memory_stats/
- nextest test groups: https://nexte.st/docs/configuration/test-groups/
- Z3 releases: https://github.com/Z3Prover/z3/releases ; Bitwuzla releases: https://github.com/bitwuzla/bitwuzla/releases
- Repo: `docs/development.md`, `docs/testing.md`.
