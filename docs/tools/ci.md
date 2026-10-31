# CI (GitHub Actions) and the CI / local test split

The repository is public, so CI runs on GitHub's standard hosted runner for public repositories. CI runs only
what fits on that runner and needs no licensed or heavy tools; everything else is a local "deep" test on the build
host.

## 1. Runner facts (https://docs.github.com/en/actions/reference/runners/github-hosted-runners)

| Repo type | vCPU | RAM | SSD | Labels |
|---|---|---|---|---|
| **Public** (ours) | **4** | **16 GB** | 14 GB | `ubuntu-latest` (= Ubuntu 24.04), `ubuntu-24.04`, `ubuntu-22.04`, `ubuntu-26.04` |
| Private (contrast) | 2 | 8 GB | 14 GB | same |

Consequences: the smoke budget "< 2 min, < 4 GB total" (`docs/testing.md`) leaves headroom for cargo + rustc; the
14 GB disk excludes Docker toolchain images (XLS+LLVM+Allo are tens of GB) and the JAX export of 25 full models.
Keep the workflow short and cached.

## 2. Workflow (`.github/workflows/ci.yml`)

Two jobs on `ubuntu-24.04`, triggered on every pull request (opened, updated, title edited) and on pushes to `main`:

| Job | Steps |
|---|---|
| `lint` | conventional-commit check of the PR title and commit subjects; a `CHANGELOG` entry for `feat`/`fix` changes; no em-dash characters; `cargo fmt --all --check`; `cargo clippy --workspace --all-targets -- -D warnings`; hadolint on the Dockerfile |
| `test` | `cargo nextest run --workspace --profile smoke`; Python tests (`pytest`) |

Merges to `main` require both jobs to pass (a repository ruleset).

Action facts:
- `actions/checkout`: `submodules: true|recursive` (default false); `fetch-depth` default 1. Submodules are
  updated with `git submodule update --init --force --depth=<fetch-depth>` when fetch-depth > 0
  (`src/git-command-manager.ts`) → shallow by default. Private submodules need a PAT/SSH key via `token`/`ssh-key`
  (`github.token` is scoped to the current repo). Only check out submodules the smoke tests need (e.g. small test
  fixtures); large third-party sources (XLS, Allo, LLVM) must not be fetched in CI: gate with `submodules: false`
  plus explicit `git submodule update --depth 1 <path>` for the few needed.
- `dtolnay/rust-toolchain`: the toolchain is selected by the action ref (`@stable`, `@1.89.0`) or by the
  `toolchain` input with `@master`; `components` comma-separated. Its README does not mention
  `rust-toolchain.toml`; rustup itself honours that file when `cargo` runs, so keep both in sync (or rely on the
  file and let rustup install it) (unverified interaction).
- `Swatinem/rust-cache@v2` (v2.9.3): caches `~/.cargo` and `./target` dependency artifacts; key includes rustc
  version → put it after the toolchain step; `save-if` limits cache writes to `main`; `cache-on-failure` default false.
- `astral-sh/setup-uv` (v10.3.3): `version`/`version-file`, `enable-cache` (`auto` default), `python-version`
  (sets `UV_PYTHON`), `cache-dependency-glob`; "setup-uv alone won't install any libraries" → `uv sync`.
  Local uv is 0.11.21; latest uv is 0.13.6 (2026-10-27); pin one version in `pyproject.toml`
  (`[tool.uv] required-version`) so CI and local agree.
- `taiki-e/install-action` (v2.88.1) installs prebuilt `cargo-nextest` (0.9.150) quickly; building nextest from
  source needs Rust ≥ 1.91 and minutes of CI time.

## 3. What runs in CI

| Check | Notes |
|---|---|
| `cargo fmt --check`, `cargo clippy -D warnings` | whole workspace |
| `cargo nextest run --profile smoke` | all passes on all 25 models at the `reduced` config, schema + invariant checks, negative (mutation) tests, fixtures in `tests/fixtures/`; no external tools |
| Python `pytest` (`python/tests`) | `cc_export` unit tests at `testing.tiny_dim` only, export of tiny synthetic modules, JSON schema checks; `cc_softfloat` tests that need Allo are skipped when `allo` is not importable |
| JSON Schema validation of committed fixtures/reports | cheap |

## 4. What does NOT run in CI, and why

| Not in CI | Reason |
|---|---|
| Docker toolchain (XLS, xlscc, Allo+LLVM, GraMi, Yosys, Verilator, z3/bitwuzla) | image size (tens of GB) vs 14 GB disk; LLVM/xlscc builds take hours (`xls.md`: up to 6 h on 8 cores) |
| Synopsys / Cadence (DC, Formality, Genus, LEC, VCS, Xcelium) | licensed, host-only, university license servers (`eda_environment.md`) |
| Full-size model export / anything at model scale | memory rule; 16 GB runner |
| Deep tests (`tests/deep/`, `[testing] deep_models`) | need the above tools and long runtimes |
| GraMi mining on real kernel graphs | runtime; Java + exponential search |

## 5. Deep tests locally (`chipc test --deep`)

- Command: `chipc test --deep` (wraps `cargo nextest run --profile deep`) on the shared host, using Docker
  `ToolRunner`s and `EdaRunner` with caps; run only after `free -g` shows headroom (design §8).
- Covers: full pass incl. tool calls, the pass's `verify`, `insta` snapshot checks of summary JSON.
- Use nextest test groups to serialise heavy tests (`heavy = { max-threads = 1 }`, `rust_stack.md` §2) and pass
  `--test-threads` explicitly; license-bound EDA tests in their own group with max 1–2.
- Deep tests run locally on the build host and are not part of CI. Large artifacts stay in `build/` (not
  committed).

## Sources

- GitHub-hosted runners: https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- https://github.com/actions/checkout (inputs) and `src/git-command-manager.ts` (`submoduleUpdate --depth`)
- https://github.com/dtolnay/rust-toolchain ; https://github.com/Swatinem/rust-cache ; https://github.com/astral-sh/setup-uv ;
  https://github.com/taiki-e/install-action
- Latest release tags via GitHub API (2026-10-30): checkout v7.0.1, rust-cache v2.9.3, setup-uv v10.3.3,
  install-action v2.88.1, nextest 0.9.150, uv 0.13.6.
- Repo: `docs/testing.md`, `docs/development.md`, `pyproject.toml`.
