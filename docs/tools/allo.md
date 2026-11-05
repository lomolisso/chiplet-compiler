# Allo (functional-unit description language)

Allo is a Python-embedded, MLIR-based accelerator design language and compiler (Cornell, PLDI'24). In
chiplet-compiler every functional unit (FU) is written as an Allo kernel over **integer types only**; floats are
handled by our soft-float library (bit patterns in `UInt`). Flow per FU:
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
  bit set `b[0] = …`, `.reverse()`. These are what the soft-float library is built from.
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
  not supported by XLS [CC] backend") and any dynamic shape `memref/tensor<…?…>`. So a soft-float FU must not
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

## 6. `allo.verify(schedule_a, schedule_b)` (PAST, FPGA'24)

- Builds both schedules with `target="vhls"`, writes `tmp/a.c`, `tmp/b.c` **relative to CWD**, adds PoCC pragmas,
  rewrites types (`int32_t`/`int64_t`/`ap_int<N≤99>` → `int`, drops `(float)` casts), finds the live-out variable,
  and calls `past.verify`. Requires the `past` module from `externals/past-python-bindings`.
- Consequence: it proves **schedule transformations** (loop reorder/split/unroll …) preserve the program, not
  bit-precise arithmetic (types are erased to `int`). Use it for schedule certificates; bit-level FU semantics are
  checked by simulation + XLS/EDA equivalence.

## 7. Soft-float plan (what we build on Allo)

- Representation: bf16/f16 = `UInt(16)`, f32 = `UInt(32)`, FP8 E4M3/E5M2 = `UInt(8)`; fields extracted with
  slices (`x[0:m]` mantissa, `x[m:m+e]` exponent, `x[n-1]` sign).
- Operations written in plain Allo with integer add/sub/mul, shifts, slices, compares; leading-zero count and
  rounding (RNE) as small integer loops/priority logic, fully unrolled. Same code feeds `llvm` sim and `xls`.
- Oracles: (a) host reference using `ml_dtypes`/numpy float types on tiny generated vectors; (b) XLS DSLX
  `apfloat` (`add`, `sub`, `mul`, `fma`, casts) for formats it models (`xls.md` §6). Divide/sqrt/exp are not in
  apfloat and need their own oracle.
- Merged FUs (Moreano-style) are also integer Allo kernels with mux-select inputs.

## 8. Pitfalls

| Pitfall | Mitigation |
|---|---|
| Float token anywhere → xls validation error | lint our generated Allo for float types before building |
| `allo.verify` writes `./tmp` | run with CWD = pass job dir |
| sw_emu int32 marshalling | don't use for >32-bit FUs |
| LLVM build oversubscription | explicit `-j`, link jobs, Docker `--memory` |
| Upstream churn (main only) | pin commits for Allo and LLVM; record both in `verify.json` |

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
