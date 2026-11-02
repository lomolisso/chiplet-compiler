# Google XLS (DSLX, IR, opt, codegen, xlscc)

XLS is Google's HLS toolchain: front ends (DSLX, xlscc C++) → XLS IR → `opt_main` → `codegen_main` → Verilog.
In chiplet-compiler, Allo's `xls` backend emits xlscc C++ for each functional unit (FU); we run
xlscc → IR → opt → codegen inside the Docker toolchain, and use XLS's equivalence/evaluation tools as
certificates. Repo: https://github.com/google/xls ; docs: https://google.github.io/xls/

## 1. Versions and distribution

| Item | Fact (checked 2026-10-30) |
|---|---|
| Release cadence | ~daily tags `v0.0.0-<n>-g<sha>`; latest `v0.0.0-10898-g8ddae8bf4` (2026-10-24), asset `xls-v0.0.0-10898-g8ddae8bf4-linux-x64.tar.gz` (105,031,731 B) (GitHub releases API) |
| Tarball contents (listed) | `codegen_main`, `dslx_fmt`, `dslx_ls`, `interpreter_main`, `ir_converter_main`, `opt_main`, `parse_and_typecheck_dslx_main`, `proto_to_dslx_main`, `prove_quickcheck_main`, `xls/dslx/stdlib/*.x`, LICENSE, THIRD_PARTY_NOTICES |
| `dist/BUILD` on main | additionally packages `check_ir_equivalence_main`, `delay_info_main`, `print_bom` → expect them in releases after 10898 |
| Not in any release | `xlscc`, `eval_ir_main`, `eval_proc_main`, `simulate_module_main`, `benchmark_main`, `block_to_verilog_main` → Bazel build |
| Bazel | `.bazelversion` = `8.7.0` (use bazelisk); README still says "install bazel 7" |
| Build platform | README build instructions and CI target Ubuntu 22.04; `Dockerfile-ubuntu-22.04` provided |
| Build time | README: on an 8-core VM ~2 h without the C++ front end, **up to 6 h including xlscc** (it builds clang/LLVM) |

Host compatibility (surprise): `objdump -T` on the release `opt_main` shows max symbol version `GLIBC_2.25`
and no `libstdc++` NEEDED entry (only libc, libm, libdl, libpthread, ld-linux), so the release binaries may run on
the RHEL 8 host (glibc 2.28). Verified by running (2026-10-30): `ldd opt_main` resolves against the host
glibc and `opt_main --version` prints `v0.0.0-10898-g8ddae8bf4`; `codegen_main --helpshort` runs. Docker stays the
default (memory caps on cgroups v1); `HostRunner` may call the release binaries if a pass needs it.

Pin: one XLS commit for both the release tarball and the Bazel-built tools (`--version` on each must match).

## 2. Pipeline we run

```
xlscc fu.cc --top=fu --out fu.ir [--include_dirs=…] [--defines=…]   # C++ → IR (Allo-generated C++)
opt_main fu.ir --top=fu > fu.opt.ir                                  # IR optimisation, --opt_level=3 default
codegen_main fu.opt.ir --top=fu --generator=pipeline \
  --delay_model=asap7 --clock_period_ps=1000 [--pipeline_stages=N] \
  --reset=rst --reset_asynchronous=false --use_system_verilog=false \
  --module_name=fu --output_verilog_path=fu.v \
  --output_signature_path=fu.sig.textproto --output_schedule_path=fu.sched.textproto \
  --output_block_ir_path=fu.block.ir
```

DSLX path (for hand-written reference FUs / soft-float checks): `ir_converter_main fu.x --top=fu > fu.ir`.

## 3. codegen_main options we use (https://google.github.io/xls/codegen_options/)

| Flag | Meaning |
|---|---|
| `--generator=pipeline\|combinational` | pipeline runs the scheduler that partitions IR ops into stages |
| `--delay_model=` | delay model for scheduling: `asap7`, `sky130`, `unit` (files in `xls/estimators/delay_model/models/`) |
| `--clock_period_ps` / `--pipeline_stages` | at least one; with period only → minimum stages meeting it; with stages only → binary search for the minimum period (scheduler runs fully each step) (scheduling doc) |
| `--clock_margin_percent` | shrink effective period before scheduling |
| `--worst_case_throughput` | at most one result per N cycles; default 1 |
| `--reset`, `--reset_active_low`, `--reset_asynchronous`, `--reset_data_path` | no reset if unset; default active-high, **synchronous**, datapath reset true. Keep synchronous for ASAP7 (no async-reset DFF; `pdks.md`) |
| `--flop_inputs`, `--flop_outputs` | register I/O (pipeline generator only); affects area/timing comparability |
| `--input_valid_signal`, `--output_valid_signal` | valid pipeline for functions; no ready → no backpressure; output valid needs input valid + reset |
| `--streaming_channel_{data,valid,ready}_suffix` | port suffixes for proc ready/valid channels (e.g. `_vld`) |
| `--use_system_verilog` | default **true** (`always_ff`, asserts); set false if a downstream tool prefers Verilog-2001 |
| `--module_name`, `--top`, `--multi_proc` | naming / which entity; `--multi_proc` default true |
| `--output_{verilog,signature,schedule,block_ir}_path` | artefacts; the schedule textproto maps every IR op to a stage → certificate input |

Scheduling facts (https://google.github.io/xls/scheduling/): SDC-based scheduler minimising pipeline-register bits
(replaced an earlier min-cut scheduler); every IR node is assigned to exactly one cycle, so **a single IR op is
never split across stages**: a wide multiply is one node and bounds the minimum clock period. Values used in
later stages get pipeline registers.

## 4. IR, procs, DSLX in brief

- IR: SSA dataflow over bit vectors, arrays, tuples; `fn` (pure function), `proc` (stateful, communicates over
  channels), `block` (post-codegen, with registers/ports). Combinational FU = `fn`; streaming FU = `proc`.
- Procs: `next` state per activation, channel `send`/`recv`; codegen emits ready/valid ports with the suffixes above.
- DSLX: Rust-like typed DSL; `#[test]` and `#[quickcheck]` functions run by `interpreter_main`.

## 5. Verification tools

| Tool | Use | Notes |
|---|---|---|
| `interpreter_main f.x [--compare=…] [--dslx_path=…]` | run DSLX tests | `--compare` cross-checks interpreter vs IR conversion/JIT (tools doc; accepted values (unverified)) |
| `prove_quickcheck_main f.x [--test_filter=RE] [--solver_num_threads=N]` | prove a `#[quickcheck]` over the full input domain via SMT | returns error when unproven, prints no counterexample; use `--alsologtostderr --v=1` (tools doc). Default 1 thread |
| `check_ir_equivalence_main a.ir b.ir --top=F [--timeout=60s]` | prove two IR functions equal over all inputs | Source (`xls/dev_tools/check_ir_equivalence_main.cc`): inlines everything first (Z3 does not handle subroutines); `--top` "strongly recommended" ("a crash may result" otherwise); procs need `--activation_count=N` and are compared by unrolling N activations **without state** (`include_state=false`) → bounded, not a full proof; exit `--mismatch_exit_code` (255) when not proven, `--match_exit_code` (0); `--timeout` from `tool_timeout.cc`. Large multipliers make the SMT query hard; expect timeouts on wide FP datapaths |
| `eval_ir_main f.ir --input='…' / random inputs` | evaluate IR (interpreter/JIT) | Bazel-only; supports `--input_validator_expr`, traces, node coverage (tools doc) |
| `eval_proc_main` | simulate procs/blocks for N cycles with channel I/O | Bazel-only |
| `simulate_module_main` | run emitted Verilog in a simulator | Bazel-only |
| `smtlib_emitter_main` | IR fn → SMT-LIB2 (via its Z3 representation) | Bazel-only; lets us run bitwuzla/z3 ourselves (`cc-smt`) |

Our certificate chain per FU: Allo `llvm` sim ≡ XLS IR eval (vectors) → `check_ir_equivalence_main` unopt vs opt
IR (or bounded for procs) → RTL vs gate netlist with Formality/LEC (`synopsys.md`, `cadence.md`).

## 6. Floating point in the DSLX stdlib (`xls/dslx/stdlib/apfloat.x`)

Generic `APFloat<EXP_SZ, FRACTION_SZ>`; instances in `float32.x`, `float64.x`, `bfloat16.x`, `hfloat16.x`,
`float_instances.x`. Public functions on main (2026-10-30):

`tag qnan is_nan inf is_inf is_pos_inf is_neg_inf abs zero one max_normal negate max_normal_exp min_normal_exp
exponent_bias unbiased_exponent bias flatten unflatten cast_from_fixed_using_rne cast_from_fixed_using_rz
is_zero_or_subnormal subnormals_to_zero upcast_with_denorms upcast_daz downcast downcast_rne normalize ldexp
cast_to_fixed eq_2 gt_2 gte_2 lte_2 lt_2 to_int to_uint add sub mul raw_mul full_precision_mul fma
has_fractional_part has_negative_exponent ceil_with_denorms ceil_daz floor_with_denorms floor_daz trunc round`

Missing: **div, sqrt, rsqrt, exp, log, min/max** (compare via `lt_2`/`gt_2`), FP8 instances (any `EXP_SZ/FRACTION_SZ`
works generically, e.g. E4M3 = `APFloat<4,3>`, but E4M3FN's no-inf/NaN encoding is not modelled (unverified)).
`xls/modules/add_dual_path/` has a dual-path FP adder with quickcheck files for bf16/f32/f64; `xls/modules/ml/`
has `activation.x`, `ws_matmul.x`. These are references for our Allo soft-float library and test oracles; our FUs
come from Allo, not DSLX.

## 7. xlscc (C++ front end)

- **Invocation outside Bazel (verified 2026-10-30).** XLS's `xls_cc_ir` rule (`xls/contrib/xlscc/build_rules/xlscc_rules.bzl`) always adds the defines `__SYNTHESIS__,__xlscc__,__AC_OVERRIDE_OVF_UPDATE_BODY=,__AC_OVERRIDE_OVF_UPDATE2_BODY=`. Without `__SYNTHESIS__`, `xls_int.h` hits `static_assert(false, "This header is only for synthesis")` and `ac_int.h` pulls in `std::ios` code.
  - The include path must contain `synth_only/` and a root holding `ac_datatypes/include/ac_int.h` (ac_types 4.8.0, pinned in `dependency_support/com_github_hlslibs_ac_types/extension.bzl`).
  - The image exports both: `XLSCC_DEFINES` and `XLSCC_INCLUDE_DIRS` (`docs/toolchain-image.md`).
- **Version string.** The Bazel-built `xlscc`, `eval_ir_main` and `check_ir_equivalence_main` print `redacted` for `--version` (unstamped build). Their version is the commit they were built from.

- libclang-based, "supports most C++17"; never: pointers, function pointers, virtual methods; arbitrary-width
  ints via `synth_only/xls_int.h` (`xls/contrib/xlscc/README.md`). Maintained best-effort by its original author.
- Build: `bazel build -c opt //xls/contrib/xlscc:xlscc //xls/tools:opt_main //xls/tools:codegen_main`.
- Flags (from `main.cc`): `--top`, `--out`, `--package`, `--block_pb`/`--block_from_class` (for channel-based
  blocks), `--include_dirs`, `--defines`, `--clang_args_file`, `--meta_out`, `--max_unroll_iters`,
  `--warn_unroll_iters`, `--top_level_init_interval`, `--error_on_uninitialized`, `--channel_strictness`.
- `#pragma hls_top` marks the top function; loops are unrolled unless pipelined via pragmas
  (`#pragma hls_unroll yes`, `#pragma hls_pipeline_init_interval N`; tutorial https://google.github.io/xls/tutorials/xlscc_overview/; exact pragma set (unverified)).

## 8. Area / delay models

- Delay models: `asap7`, `sky130`, `unit` (textprotos in `xls/estimators/delay_model/models/`); `delay_info_main`
  dumps per-node delay and critical path.
- Area models: `asap7`, `sky130`, `unit`, `unit_with_default_recursive_op_handling` (`xls/estimators/area_model/models/`);
  `opt_main --area_model` (default `asap7`) is used by resource-sharing decisions (tools doc). `benchmark_main`
  reports delay/area metrics (Bazel-only).
- No FreePDK45 model: for Nangate45 runs schedule with `asap7` or `unit` and take real numbers from DC/Genus.
  Record which model scheduled each FU.

## 9. Resource notes

- `opt_main`/`codegen_main` are single process; memory scales with IR size (fully unrolled loops explode IR).
  Cap via Docker `--memory`; keep `--max_unroll_iters` bounded in xlscc.
- `prove_quickcheck_main --solver_num_threads` and SMT in `check_ir_equivalence_main`: always pass `--timeout`.
- Bazel builds: `--jobs=N --local_ram_resources=<MB> --local_cpu_resources=N` with absolute values
  (`docker_resources.md`); the xlscc build compiles LLVM/clang.

## Sources

- XLS README (install, build, timings): https://github.com/google/xls/blob/main/README.md ; `.bazelversion`; `dist/BUILD`
- Releases API: https://api.github.com/repos/google/xls/releases ; tarball listing and `objdump` of `opt_main` (2026-10-30)
- Codegen options: https://google.github.io/xls/codegen_options/ ; scheduling: https://google.github.io/xls/scheduling/ ; tools: https://google.github.io/xls/tools/
- `xls/dev_tools/check_ir_equivalence_main.cc`, `xls/dev_tools/tool_timeout.cc`, `xls/contrib/xlscc/{README.md,main.cc}`
- `xls/dslx/stdlib/apfloat.x`; directory listings of `xls/dslx/stdlib`, `xls/modules`, `xls/estimators/{delay,area}_model/models`
