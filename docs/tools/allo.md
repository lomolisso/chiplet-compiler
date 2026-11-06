# Allo (functional-unit description language)

Allo is a Python-embedded, MLIR-based accelerator design language and compiler (Cornell, PLDI'24). In
chiplet-compiler every functional unit (FU) is written as an Allo kernel over **integer types only**; float
operators are third-party IP cores (design §4.12). Flow per FU:
Allo `llvm` target (functional simulation) → Allo `xls` backend (xlscc C++) → xlscc → XLS IR → opt → codegen
(see `xls.md`). Repo: https://github.com/cornell-zhang/allo ; docs: https://cornell-zhang.github.io/allo

## 1. Versions and install

| Item | Fact (checked 2026-10-30) |
|---|---|
| Allo `main` | `ac21bae` (2026-10-12). No releases/tags used; pin a commit as a submodule in `third_party/` |
| LLVM pin | submodule `externals/llvm-project` @ `6b09f739c4d085dc39eb9ff220c786bc3aa8c7fb` (2025-12-21, LLVM 22.0.0git) |
| Other submodule | `externals/past-python-bindings` @ `65f989b` (PAST checker for `allo.verify`) |
| Python | "dependencies target CPython 3.12 on Linux x86-64" (setup docs) |
| Build deps | `pyproject.toml` build-system: `setuptools>=61`, `cmake>=3.20`, `ninja`, `nanobind>=2.9,<3.0` |
| Docker image | `chhzh123/allo:latest` (Docker Hub, updated 2025-12-21, ~15 GB compressed); LLVM at `/root/llvm-project` in the image. Whether it matches the current pin is (unverified); same date as the pin commit |

Source install (setup docs, verbatim flags):

```bash
git clone --recursive https://github.com/cornell-zhang/allo.git && cd allo
python3 -m pip install cmake ninja "nanobind>=2.9,<3.0"
cd externals/llvm-project && mkdir -p build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;openmp" -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE="$(command -v python3)"
ninja                                      # ← cap: ninja -j8 + -DLLVM_PARALLEL_LINK_JOBS=2
export LLVM_BUILD_DIR="$(pwd)"; export PATH="$LLVM_BUILD_DIR/bin:$PATH"
cd "$ALLO_ROOT" && python3 -m pip install -v -e .
```

Resource notes: building clang+MLIR+OpenMP at default `ninja` parallelism uses one job per host CPU (64 here) and
links are memory-heavy: always build inside a memory-capped container with explicit `-j` and
`LLVM_PARALLEL_LINK_JOBS` (`docker_resources.md`). Using `uv` instead of conda is our choice (docs use conda).

## 2. Programming model

```python
import allo
from allo.ir.types import int32, UInt, uint16

def kernel(A: uint16[64], B: uint16[64]) -> uint16[64]:   # type hints are mandatory
    C: uint16[64] = 0
    for i in range(64):
        C[i] = A[i] + B[i]
    return C

s = allo.customize(kernel)   # parse to MLIR (allo dialect); s.module is the IR
s.pipeline("i")              # schedule primitives mutate s
mod = s.build()              # default target "llvm"
```

- Types (`allo/ir/types.py`): `Int(n)`, `UInt(n)` (aliases `int1…int512`, `uint1…uint64`), `Fixed(w,f)`,
  `UFixed(w,f)`, `Float` (16/32/64 bits only: `float16`, `bfloat16`, `float32`, `float64`), `Index`, `Struct`, `Stream`.
  No FP8 type.
- Bit operations (`tests/test_bitop.py`): bit get `x[0]`, slice `x[lo:hi]` (e.g. `a[28:32]` = top nibble of int32),
  bit set `b[0] = …`, `.reverse()`. These handle values as bit patterns.
- Schedule primitives (`allo/customize.py`): `split(axis, factor)`, `reorder(*axes)`, `unroll(axis, factor=0)`,
  `fuse`, `pipeline(axis, initiation_interval=1)`, `parallel`, `partition(target, partition_type, dim, factor)`,
  `buffer_at`, `reuse_at`, `compute_at`, `reshape`, `inline`, `dataflow`, `to` (streams), `unfold`,
  `compose(schs)` (compose schedules of sub-kernels into a top schedule), `get_loops`.
- `build(target=None, mode=None, project=None, configs=None, wrap_io=True, use_memory=False)`.

## 3. Targets

| `target=` | Module | Output / use |
|---|---|---|
| `"llvm"` (default) | `LLVMModule` | JIT on CPU through MLIR `ExecutionEngine(opt_level=2)`; call with numpy arrays |
| `"xls"` / `"xlscc"` | `XLSCCModule` | xlscc-flavoured C++ (`test_block.cpp`), optional `rewrites.textproto` (RAM config); `mode="sw_emu"` builds a g++ binary |
| `"vhls"`, `"vivado_hls"`, `"vitis_hls"`, `"pynq"`, `"tapa"`, `"ihls"`, `"catapult"` | `HLSModule` | vendor HLS C++ / projects (not used; no FPGA tools in our flow) |
| AIE | `allo/backend/aie` | AMD NPU (not used) |

Code comment in `customize.py`: "TODO Add XLS DSLX Backend": only the xlscc path exists.

## 4. `llvm` functional simulation (our golden model for FUs)

- Integer arrays: widths ≤64 are packed to the next power-of-two numpy dtype (i1–i8→int8 … i33–i64→int64) with
  overflow handling (`allo/backend/llvm.py`). Wider ints use struct-array packing paths (support level (unverified)).
- Inputs must be C-contiguous numpy arrays (error otherwise). Scalars are passed by value.
- Floats: f16/bf16 passed as int16 views, f32/f64 as C float/double. We do not rely on these for FUs.
- Run only at `testing.tiny_dim` sizes with generated data (design §8).

## 5. `xls` backend details (`allo/backend/xls.py`)

- Pipeline: parse MLIR → `decompose_library_function` → lower linalg → `empty-tensor-to-alloc-tensor`,
  `convert-linalg-to-affine-loops` → mark top → `allo_d.emit_xhls` emits C++.
- **Validation rejects any line of the lowered MLIR matching `\b(f16|f32|f64|bf16)\b`** ("Floating-point type … is
  not supported by XLS [CC] backend") and any dynamic shape `memref/tensor<…?…>`. So an FU sent through it must not
  contain a single float-typed value, cast or constant. Errors go to `<project>/xls_errors.txt`.
- `Fixed`/`UFixed` are emitted via a generated `Fixed<WIDTH,FRAC,SIGNED>` C++ template over `ac_int`.
- No array arguments → plain combinational function. Array arguments → a `TestBlock` class with `#pragma hls_top`,
  input channels `v{i}_in`, output channel `out`; `use_memory=True` maps arrays to memories and emits
  `rewrites.textproto` for XLS RAM rewriting.
- `mode="sw_emu"`: compiles a harness with `g++ -std=c++17 -O2` against Allo's shim headers
  (`allo/harness/xlscc/{XlsInt.h,Channel.h,Memory.h}`); `is_available()` only checks for `g++` or `xlscc` on PATH.
  Pitfall: array I/O is marshalled as **`np.int32`** and scalars are parsed from stdout as Python `int` →
  elements wider than 32 bits are not round-tripped faithfully. Use sw_emu only as a smoke test of the emitted
  C++; the certificates come from `llvm` sim vs XLS IR evaluation.
- The backend does not call xlscc itself; our pass runs xlscc/opt/codegen (`xls.md`) on `test_block.cpp`.

## 5a. Our pinned patches to the XLS backend (`third_party/patches/allo`, `docs/third-party.md`)

The image's Allo carries four patches; their tests are `tests/test_xls_extensions.py`, run at image build with
`xlscc`, `opt_main` and `codegen_main` on PATH.

- **Calls.** A called function is emitted as a C++ function returning its (single, scalar) result, and the call as
  `T v = f(args);`. Unpatched, the emitter inherits the Vitis call, which passes the result through a pointer
  (`fmul(v5, v6, &v8)`) that the callee's signature does not have, so `xlscc` rejected every kernel with a call.
  Calls with array arguments or results are an emitter error (inline such callees). The emitted header no longer
  includes `xls_builtin.h`, which XLS does not ship (`xls_int.h` includes `/xls_builtin.h`, provided by `xlscc`).
- **External Verilog modules.** A function decorated with `allo.extern_verilog` is implemented by an existing
  Verilog module:

  ```python
  @allo.extern_verilog(
      "DW_fp_mult #(.sig_width(23), .exp_width(8), .ieee_compliance(0)) "
      "{fn} (.a({a}), .b({b}), .rnd(3'b000), .z({return}), .status())",
      delay_ps=800,      # the module's combinational delay for the XLS scheduler (optional)
      model=f32_mul,     # NumPy model for target="llvm" (optional; else the body is simulated)
  )
  def fmul(a: uint32, b: uint32) -> uint32:
      r: uint32 = a ^ b  # body: what xlscc translates and XLS IR evaluation runs
      return r
  ```

  The template uses XLS's foreign function syntax (`{fn}` instance name, `{return}`, `{<argument>}` by its Python
  name); `module=`, `params=` and `ports=` build the same template from a port map (`ports` values: an argument
  name, `"return"`, a Verilog constant, or `None` for unconnected). The declaration is checked when the function
  is decorated (unknown placeholders, missing `{fn}`, negative delay).
  - `target="xls"` emits `[[xlscc::hls_extern_verilog("<template>", <delay_ps>)]]` before the callee, with the
    placeholders renamed to the emitted C++ argument names; the patched `xlscc` turns it into the IR function's
    foreign function data (`xls.md` §7a), and `codegen_main` instantiates the module.
  - `target="llvm"` replaces the callee by a declaration bound to `model` (`ExecutionEngine.register_runtime`):
    the model gets one Python int per argument (unsigned arguments in `[0, 2^w)`, signed ones signed) and returns
    the result's bit pattern; integer ports of at most 64 bits. An exception in the model fails the call.
  - Only scalar arguments and one scalar result (XLS external modules are combinational functions); the
    function must be called, not the top.
- **Loop directives.** Each loop gets exactly one `xlscc` directive: `pipeline(axis, ii)` gives
  `hls_pipeline_init_interval ii` (`xlscc` implements 1; a longer interval comes from codegen's
  `--worst_case_throughput`), `unroll(axis, factor)` gives `hls_unroll factor` (`yes` for 0, full unrolling).
  Without a directive: loops using channels are pipelined; loops inside a pipelined or fully unrolled loop, and
  loops accessing a completely partitioned array, are unrolled; other loops of a block (a kernel with array
  arguments) run sequentially as pipelined loops of interval 1; loops of a combinational function (scalar
  arguments only) are unrolled, the only form `xlscc` accepts there. Unpatched, every loop without a pipeline
  directive was fully unrolled, the pipeline interval was dropped and `unroll` factors were emitted as
  `hls_unroll yes factor=N`.
- **Partitioning.** A local array partitioned completely in every dimension is declared with
  `[[clang::annotate_type("hls_array_as_tuple")]]` (separate registers; `xlscc` then needs constant indices) and
  stays a register array in memory mode (`use_memory=True`), where other arrays are `__xls_memory`. A block or
  cyclic partition has no `xlscc` equivalent: it is recorded as a comment and the array is unchanged. No Vitis
  pragmas are emitted.

## 6. `allo.verify(schedule_a, schedule_b)` (PAST, FPGA'24)

- Builds both schedules with `target="vhls"`, writes `tmp/a.c`, `tmp/b.c` **relative to CWD**, adds PoCC pragmas,
  rewrites types (`int32_t`/`int64_t`/`ap_int<N≤99>` → `int`, drops `(float)` casts), finds the live-out variable,
  and calls `past.verify`. Requires the `past` module from `externals/past-python-bindings`.
- Consequence: it proves **schedule transformations** (loop reorder/split/unroll …) preserve the program, not
  bit-precise arithmetic (types are erased to `int`). Use it for schedule certificates; bit-level FU semantics are
  checked by simulation + XLS/EDA equivalence.

## 7. Pitfalls

| Pitfall | Mitigation |
|---|---|
| Float token anywhere → xls validation error | lint our generated Allo for float types before building |
| `allo.verify` writes `./tmp` | run with CWD = pass job dir |
| sw_emu int32 marshalling | don't use for >32-bit FUs |
| LLVM build oversubscription | explicit `-j`, link jobs, Docker `--memory` |
| Upstream churn (main only) | pin commits for Allo and LLVM; record both in `verify.json` |
| `IPModule` (external C++ for the `llvm` simulator and Vitis HLS) is ignored by the XLS backend | use `allo.extern_verilog` (section 5a) |
| A completely partitioned array is a tuple in `xlscc`: an access with a run-time index fails | unroll the loops that index it (the emitter does so for loops without a directive) |
| `xlscc` pipelines a loop with interval 1 only | `codegen_main --worst_case_throughput=N` for a unit that takes a new input every N cycles |
| No dataflow regions in the XLS backend | not needed for v1 units |

## Sources

- Allo README & publications: https://github.com/cornell-zhang/allo/blob/main/README.md ; H. Chen et al.,
  "Allo: A Programming Model for Composable Accelerator Design," PLDI 2024, https://dl.acm.org/doi/10.1145/3656401
- Setup docs: https://cornell-zhang.github.io/allo/setup/index.html
- Source files on `main` (2026-10-30): `allo/customize.py` (`build`, primitives), `allo/backend/xls.py`,
  `allo/backend/llvm.py`, `allo/verify.py`, `allo/ir/types.py`, `tests/test_bitop.py`, `tests/test_xls.py`,
  `.gitmodules`, `pyproject.toml`; submodule SHAs via GitHub contents API.
- LLVM commit 6b09f739 (`cmake/Modules/LLVMVersion.cmake`: 22.0.0): https://github.com/llvm/llvm-project/commit/6b09f739c4d085dc39eb9ff220c786bc3aa8c7fb
- Docker Hub tags: https://hub.docker.com/r/chhzh123/allo/tags
- L.-N. Pouchet et al., "Formal Verification of Source-to-Source Transformations for HLS," FPGA 2024.
