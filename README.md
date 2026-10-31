# chiplet-compiler

A formal, fully automated compiler from quantized DNN models to verified functional-unit RTL.

chiplet-compiler takes quantized DNN models written in JAX, lowers them to StableHLO, and runs a pipeline of passes:
1. **Find kernels** with compiler theory: convex partitions and iteration-space analysis.
2. **Find what kernels share** across models: graph isomorphism, frequent subgraph mining.
3. **Merge their datapaths** into configurable functional units, one unit serving several kernels and widths.
4. **Generate and verify RTL** for those units: Allo → XLS → Verilog → synthesis on FreePDK45 and ASAP7.

Each step is checked formally or structurally (SMT proofs, isomorphism certificates, equivalence checking), never by running the large models.

**Status:** the framework core is built (pipeline manager, artifact store, capped tool runners, `chipc`); the passes are added one at a time.

## Read
- [docs/spec.md](docs/spec.md): the system design: passes, verification, datatypes, decisions.
- [docs/conventions.md](docs/conventions.md): how code, artifacts, config and tests are organised.
- [docs/context/](docs/context/): background on the tools, the theory and the literature the passes rely on.

## Layout
| Path | Contents |
|---|---|
| `models/` | The 25 quantized JAX ports (inputs). Each `recipe.py` declares its datatypes. |
| `verify/` | Pre-framework Python checks of the ports (dtype audit, GEMM/conv census, unit tests). |
| `crates/` | The Rust framework : IR, pipeline manager, passes, CLI `chipc`. |
| `python/` | The JAX export front end and the soft-float Allo library. |
| `tests/` | `smoke/` (all models, fast, CI) and `deep/` (representative subset, local). |
| `docker/` | The toolchain image. |
| `third_party/` | Submodules: GraMi, Allo, XLS. |
| `config.toml` | All settings; per-pass sections `[passes.<pass>]`. Machine paths go in `config.local.toml`. |
| `build/` | (ignored) intermediate results, one hidden directory per pass: `build/.<pass>/`. |

## Quick start (once another change are done)
```bash
cp config.local.toml.example config.local.toml   # adjust machine paths
chipc doctor                                     # check tools, licenses, PDKs
chipc run --models resnet50                       # run the pipeline; results in build/.<pass>/
chipc test                                        # smoke tests (chipc test --deep for in-depth)
```

The Python checks work today:
```bash
uv sync
uv run python -m verify.dtype_audit               # every port computes in its recipe's datatypes
uv run python -m pytest verify/test_quant.py
```
