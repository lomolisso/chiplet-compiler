"""Build and check one primitive in the toolchain image (needs Allo and the XLS tools).

Steps, all files in the primitive's output directory:
1. `kernel.py`: the generated Allo function;
2. vectors (`vectors.npz`: operands, reference, Allo `llvm` and XLS IR results);
3. Allo `llvm` simulation of every vector, compared with the reference;
4. `kernel.mlir`: the Allo module after MLIR inlining (helper calls are emitted wrongly by Allo's XLS
   backend), checked for float types; `xls/test_block.cpp` from Allo's XLS backend;
5. `fu.ir` (xlscc), `fu.opt.ir` (opt_main), `rtl.v` (codegen_main, combinational);
6. `eval_ir_main` (LLVM JIT) on every vector, compared with the `llvm` results bit for bit;
7. `result.json`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import catalog, reference, vectors
from .formats import Float

CHUNK = 4096
FLOAT_TOKEN = re.compile(r"\b(f16|f32|f64|bf16)\b")
INCLUDE = Path(__file__).resolve().parent / "include"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _np_dtype(bits: int) -> type:
    return np.uint8 if bits <= 8 else np.uint16 if bits <= 16 else np.uint32


def _simulate(p: catalog.Prim, out: Path, name: str, args: list[np.ndarray]) -> np.ndarray:
    import allo  # noqa: F401  (imported for its side effect of registering the dialects)

    params = ", ".join(f"A{i}: UInt({t.bits})[{CHUNK}]" for i, t in enumerate(p.params))
    call = ", ".join(f"A{i}[i]" for i in range(len(p.params)))
    sim = (
        (out / "kernel.py").read_text()
        + f"\n\ndef sim({params}) -> UInt({p.ret.bits})[{CHUNK}]:\n"
        + f"    R: UInt({p.ret.bits})[{CHUNK}] = 0\n"
        + f"    for i in range({CHUNK}):\n"
        + f"        R[i] = {name}({call})\n"
        + "    return R\n"
    )
    (out / "sim.py").write_text(sim)
    mod = _load(out / "sim.py", f"cc_sf_sim_{name}")
    exe = allo.customize(mod.sim).build()
    n = len(args[0])
    pad = (-n) % CHUNK
    cols = [np.concatenate([a, np.zeros(pad, dtype=np.uint64)]).astype(_np_dtype(t.bits)) for a, t in zip(args, p.params)]
    res = np.empty(n + pad, dtype=np.uint64)
    for i in range(0, n + pad, CHUNK):
        r = exe(*[np.ascontiguousarray(c[i : i + CHUNK]) for c in cols])
        res[i : i + CHUNK] = np.asarray(r).astype(np.int64).astype(np.uint64) & np.uint64((1 << p.ret.bits) - 1)
    (out / "sim.py").unlink()
    return res[:n]


def _emit_xls(out: Path, name: str) -> str:
    import allo
    from allo._mlir.passmanager import PassManager

    mod = _load(out / "kernel.py", f"cc_sf_kernel_{name}")
    s = allo.customize(getattr(mod, name))
    with s.module.context:
        PassManager.parse("builtin.module(inline,symbol-dce)").run(s.module.operation)
    text = str(s.module)
    (out / "kernel.mlir").write_text(text)
    if FLOAT_TOKEN.search(text):
        raise RuntimeError("float type in the lowered module")
    s.build(target="xls", project=str(out / "xls"))
    return text


def _run(cmd: list[str], stdout: Path | None = None) -> None:
    with open(stdout, "w") if stdout else open(os.devnull, "w") as so:
        r = subprocess.run(cmd, stdout=so, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}): {r.stderr[-2000:]}")


def _xls(out: Path, name: str) -> None:
    defines = os.environ.get("XLSCC_DEFINES", "")
    includes = ",".join([str(INCLUDE), os.environ.get("XLSCC_INCLUDE_DIRS", "")])
    _run(
        [
            "xlscc",
            str(out / "xls" / "test_block.cpp"),
            f"--top={name}",
            f"--out={out / 'fu.ir'}",
            f"--defines={defines}",
            f"--include_dirs={includes}",
        ]
    )
    _run(["opt_main", str(out / "fu.ir"), f"--top={name}"], stdout=out / "fu.opt.ir")
    _run(
        [
            "codegen_main",
            str(out / "fu.opt.ir"),
            f"--top={name}",
            "--generator=combinational",
            "--use_system_verilog=false",
            f"--module_name={name}",
            f"--output_verilog_path={out / 'rtl.v'}",
            f"--output_signature_path={out / 'rtl.sig.textproto'}",
        ]
    )


def _eval_ir(out: Path, name: str, p: catalog.Prim, args: list[np.ndarray]) -> np.ndarray:
    lines = []
    for row in zip(*args):
        lines.append("; ".join(f"bits[{t.bits}]:{int(v)}" for v, t in zip(row, p.params)))
    inp = out / "ir_in.txt"
    inp.write_text("\n".join(lines) + "\n")
    res = out / "ir_out.txt"
    _run(["eval_ir_main", str(out / "fu.opt.ir"), f"--top={name}", f"--input_file={inp}", "--use_llvm_jit=true"], stdout=res)
    vals = [int(l.split(":", 1)[1].replace("_", ""), 0) for l in res.read_text().splitlines() if l.strip()]
    inp.unlink()
    res.unlink()
    return np.array(vals, dtype=np.uint64)


def _summary(p: catalog.Prim, args: list[np.ndarray], got: np.ndarray, want: np.ndarray) -> dict:
    ok = reference.same(p.ret, got, want)
    bad = np.nonzero(~ok)[0]
    return {
        "mismatches": int(len(bad)),
        "examples": [
            {"args": [int(a[i]) for a in args], "want": int(want[i]), "got": int(got[i])} for i in bad[:8]
        ],
    }


def build(key: str, name: str, out: Path, exhaustive_bits: int, random_n: int, precision: tuple[int, int]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    p = catalog.parse_key(key, precision)
    t0 = time.time()
    (out / "kernel.py").write_text(catalog.source(p, name))
    args = vectors.generate(p, exhaustive_bits, random_n)
    exhaustive = p.input_bits <= exhaustive_bits
    want = reference.evaluate(p, args)
    t1 = time.time()
    llvm = _simulate(p, out, name, args)
    t2 = time.time()
    _emit_xls(out, name)
    _xls(out, name)
    t3 = time.time()
    ir = _eval_ir(out, name, p, args)
    t4 = time.time()
    np.savez_compressed(out / "vectors.npz", *args, want=want, llvm=llvm, ir=ir)
    res = {
        "key": key,
        "id": name,
        "vectors": int(len(want)),
        "exhaustive": exhaustive,
        "reference": _summary(p, args, llvm, want),
        "ir_vs_llvm": int(np.count_nonzero(ir != llvm)),
        "ir_nodes": sum(1 for l in (out / "fu.opt.ir").read_text().splitlines() if " = " in l),
        "seconds": {"reference": round(t1 - t0, 2), "llvm": round(t2 - t1, 2), "xls": round(t3 - t2, 2), "ir_eval": round(t4 - t3, 2)},
    }
    res["ok"] = res["reference"]["mismatches"] == 0 and res["ir_vs_llvm"] == 0
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def recheck(key: str, out: Path, precision: tuple[int, int], sample: int) -> dict:
    """Recompute the reference on a sample of the recorded vectors (for `verify`)."""
    p = catalog.parse_key(key, precision)
    z = np.load(out / "vectors.npz")
    args = [z[f"arr_{i}"] for i in range(len(p.params))]
    n = len(args[0])
    idx = np.unique(np.linspace(0, n - 1, min(sample, n)).astype(np.int64))
    sub = [a[idx] for a in args]
    want = reference.evaluate(p, sub)
    return {
        "key": key,
        "checked": int(len(idx)),
        "want_differs": int(np.count_nonzero(~reference.same(p.ret, z["want"][idx], want))),
        "llvm_differs": int(np.count_nonzero(~reference.same(p.ret, z["llvm"][idx], want))),
        "ir_differs": int(np.count_nonzero(~reference.same(p.ret, z["ir"][idx], want))),
        "ret_is_float": isinstance(p.ret, Float),
    }
