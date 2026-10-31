# chiplet-compiler

A formal, fully automated compiler from quantized DNN models to verified functional-unit RTL.

chiplet-compiler takes quantized DNN models written in JAX, lowers them to StableHLO, and runs a pipeline of passes:
1. **Find kernels** with compiler theory: convex partitions and iteration-space analysis.
2. **Find what kernels share** across models: graph isomorphism, frequent subgraph mining.
3. **Merge their datapaths** into configurable functional units, one unit serving several kernels and widths.
4. **Generate and verify RTL** for those units: Allo, XLS, Verilog, synthesis on FreePDK45 and ASAP7.

Each step is checked formally or structurally (SMT proofs, isomorphism certificates, equivalence checking), never by
running the large models.

**Status:** the framework (pipeline manager, artifact store, capped tool runners, `chipc`), the toolchain image
and the `export` pass (JAX to StableHLO with components and symbolic sizes) are built; the remaining passes are
added one at a time. See [CHANGELOG.md](CHANGELOG.md).

## Documentation

- [docs/design.md](docs/design.md): the system design: passes, verification, datatypes, decisions.
- [docs/development.md](docs/development.md): code, artifacts, configuration, machines and tests.
- [docs/README.md](docs/README.md): index of everything else (framework, passes, tools, toolchain image).
- [CONTRIBUTING.md](CONTRIBUTING.md): branches, commits, pull requests, changelog.

## Layout

| Path | Contents |
|---|---|
| `models/` | The 25 quantized JAX ports (inputs). Each `recipe.py` declares its datatypes. |
| `verify/` | Python checks of the ports (dtype audit, GEMM/conv census, unit tests). |
| `crates/` | The Rust framework: IR, pipeline manager, passes, CLI `chipc`. |
| `python/` | The JAX export front end. |
| `tests/` | `smoke/` (all models, fast, CI) and `deep/` (representative subset, build host). |
| `docker/` | The toolchain image. |
| `scripts/` | Build host helpers: `on-build-host`, `build-area`. |
| `third_party/` | Submodules: GraMi, Allo, XLS. |
| `config.toml` | All settings; per-pass sections `[passes.<pass>]`. Machine paths go in `config.local.toml`. |
| `build/` | (ignored) intermediate results, one hidden directory per pass: `build/.<pass>/`. |

## Quick start

Builds and runs happen on the build host (see [docs/development.md](docs/development.md)):

```bash
cp config.local.toml.example config.local.toml          # adjust machine paths
scripts/on-build-host chipc doctor                      # check tools, licenses, PDKs, the image
scripts/on-build-host chipc run --only export --models resnet50   # results in build/.export/
scripts/on-build-host chipc test                        # smoke tests (chipc test --deep for in-depth)
```
