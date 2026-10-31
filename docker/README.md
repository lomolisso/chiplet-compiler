# docker/

The toolchain image: Python 3.12 + uv (JAX, Allo), Allo with its pinned LLVM, XLS release binaries plus Bazel-built `xlscc` / `eval_ir_main` / `check_ir_equivalence_main`, Java 17 + GraMi, Yosys, iverilog, Verilator, z3, bitwuzla.

Every container call made by the pipeline gets `--memory`, `--memory-swap` (equal, no swap), `--cpus` and `--pids-limit` from `config.toml` `[tools.docker]` and the pass resources. This host uses cgroups v1, where Docker's limits are the reliable way to cap memory. Licensed EDA tools are not in the image; they run on the host through `eda-run` (see `docs/context/tooling/eda_environment.md`).
