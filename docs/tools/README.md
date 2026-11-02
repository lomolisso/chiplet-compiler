# Tooling context

Reference notes on the external tools chiplet-compiler uses, written for engineers implementing passes.
Every file cites primary sources (official docs, source files, man pages on this host, release metadata); claims
that could not be checked are marked "(unverified)". Facts about this machine were checked on 2026-10-30.

| File | Contents |
|---|---|
| [jax_export.md](jax_export.md) | `jax.export` (Exported, symbolic shapes, shape assertions, dynamic ops), `lax.composite` rules, wrapping Flax modules, reading StableHLO via JAX's MLIR bindings, memory rules |
| [stablehlo.md](stablehlo.md) | Op semantics for kernel analysis: elementwise, `broadcast_in_dim`, `dot_general`, `convolution`, `reduce`/`reduce_window`, gather/scatter, dynamic ops, composite, quantized types and quant-to-math |
| [allo.md](allo.md) | Allo kernels, types, bit ops, schedule primitives, targets, `llvm` simulation, `xls` backend constraints, `allo.verify`, install/LLVM pin, soft-float plan |
| [xls.md](xls.md) | XLS releases vs Bazel tools, xlscc → IR → opt → codegen flags, scheduling, verification tools, apfloat stdlib coverage, delay/area models |
| [grami.md](grami.md) | GraMi: MNI frequent subgraph mining, `.lg` format, build (Java ≥11 import fix), direct Java invocation, output format, time-limit pitfall |
| [synopsys.md](synopsys.md) | Library Compiler, Design Compiler, Formality, PrimeTime, VCS, SpyGlass: batch scripts, flags, resource options |
| [cadence.md](cadence.md) | Genus, Conformal LEC, Xcelium (Innovus out of scope): batch scripts, flags, licenses, observed issues |
| [pdks.md](pdks.md) | FreePDK45/Nangate45 and ASAP7 on this host: files, corners, units, `.db` creation, env vars |
| [eda_environment.md](eda_environment.md) | `eda-toggle.sh` functions, why `eda-run` needs an interactive shell, `EdaRunner` contract, licenses, memory limiting on cgroups v1 |
| [docker_resources.md](docker_resources.md) | `docker run` caps and OOM behaviour, permissions/mounts, toolchain image layering, disk layout |
| [rust_stack.md](rust_stack.md) | Chosen crates and versions, nextest config, rustup without root, z3/bitwuzla binaries |
| [ci.md](ci.md) | GitHub Actions on GitHub-hosted runners, the `lint` and `test` jobs, what runs in CI vs locally |

Most important cross-cutting findings:
- `eda-run` only loads the EDA tools from an **interactive** bash (`set -i` is invalid in bash 4.4); use
  `bash --norc -i -c '. ~/bin/eda-toggle.sh; eda-run "$@"' eda-run <tool> …` (`eda_environment.md` §2).
- GraMi's default `limitedtime=true` can over-count support; always pass `limitedtime=false` (`grami.md` §4).
- Allo's `xls` backend rejects any `f16/f32/f64/bf16` token in the lowered MLIR → soft-float FUs must be integer-only
  (`allo.md` §5).
- XLS releases lack `xlscc`/`eval_ir_main`; `check_ir_equivalence_main` compares procs only for N unrolled
  activations without state (`xls.md` §5).
