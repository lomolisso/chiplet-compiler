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

from . import approx, catalog, reference, vectors
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


def _simulator(p: catalog.Prim, out: Path, name: str, chunk: int = CHUNK):
    """The Allo `llvm` build of `kernel.py` over `chunk` vectors per call, as a function of operand arrays."""
    import allo  # noqa: F401  (imported for its side effect of registering the dialects)

    params = ", ".join(f"A{i}: UInt({t.bits})[{chunk}]" for i, t in enumerate(p.params))
    call = ", ".join(f"A{i}[i]" for i in range(len(p.params)))
    sim = (
        (out / "kernel.py").read_text()
        + f"\n\ndef sim({params}) -> UInt({p.ret.bits})[{chunk}]:\n"
        + f"    R: UInt({p.ret.bits})[{chunk}] = 0\n"
        + f"    for i in range({chunk}):\n"
        + f"        R[i] = {name}({call})\n"
        + "    return R\n"
    )
    (out / "sim.py").write_text(sim)
    mod = _load(out / "sim.py", f"cc_sf_sim_{name}")
    exe = allo.customize(mod.sim).build()
    (out / "sim.py").unlink()

    def run(args: list[np.ndarray]) -> np.ndarray:
        n = len(args[0])
        pad = (-n) % chunk
        cols = [
            np.concatenate([a, np.zeros(pad, dtype=np.uint64)]).astype(_np_dtype(t.bits))
            for a, t in zip(args, p.params)
        ]
        res = np.empty(n + pad, dtype=np.uint64)
        for i in range(0, n + pad, chunk):
            r = exe(*[np.ascontiguousarray(c[i : i + chunk]) for c in cols])
            res[i : i + chunk] = np.asarray(r).astype(np.int64).astype(np.uint64) & np.uint64((1 << p.ret.bits) - 1)
        return res[:n]

    return run


def _simulate(p: catalog.Prim, out: Path, name: str, args: list[np.ndarray]) -> np.ndarray:
    return _simulator(p, out, name)(args)


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


def _limit(p: catalog.Prim) -> float:
    assert isinstance(p.ret, Float)
    return reference.LIMITS[approx.design(p.op, p.ret).accuracy]


def _agree(p: catalog.Prim, got: np.ndarray, want: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Per vector: does `got` match the reference (bit for bit, or within the accuracy limit for an
    approximate op), and the error in ulps (approximate ops only)."""
    if p.approximate:
        return reference.approx_check(p, _limit(p), got, want)
    return reference.same(p.ret, got, want), None


def _summary(p: catalog.Prim, args: list[np.ndarray], got: np.ndarray, want: np.ndarray) -> dict:
    ok, err = _agree(p, got, want)
    bad = np.nonzero(~ok)[0]
    # an approximate op's reference is a float64 value: its bit pattern stands in for `want`
    wbits = want.view(np.uint64) if want.dtype == np.float64 else want
    out = {
        "mismatches": int(len(bad)),
        "examples": [
            {"args": [int(a[i]) for a in args], "want": int(wbits[i]), "got": int(got[i])} for i in bad[:8]
        ],
    }
    if err is not None:
        out["max_ulp"] = float(np.nanmax(err)) if np.isfinite(err).any() else 0.0
    return out


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
    ref = _summary(p, args, llvm, want)
    res = {
        "key": key,
        "id": name,
        "vectors": int(len(want)),
        "exhaustive": exhaustive,
        "reference": ref,
        "ir_vs_llvm": int(np.count_nonzero(ir != llvm)),
        "ir_nodes": sum(1 for l in (out / "fu.opt.ir").read_text().splitlines() if " = " in l),
        "seconds": {"reference": round(t1 - t0, 2), "llvm": round(t2 - t1, 2), "xls": round(t3 - t2, 2), "ir_eval": round(t4 - t3, 2)},
    }
    if p.approximate:
        assert isinstance(p.ret, Float)
        dz = approx.design(p.op, p.ret)
        res["accuracy"] = dz.accuracy
        res["bound_ulp"] = dz.bound_ulp
        res["max_ulp"] = ref.pop("max_ulp")
    else:
        res["accuracy"] = "exact"
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
    if p.approximate:
        # the stored float64 reference may differ from a recomputation by NumPy's own error
        stored = z["want"][idx]
        with np.errstate(all="ignore"):
            near = (stored == want) | (np.isnan(stored) & np.isnan(want))
            near |= np.abs(stored - want) <= np.abs(want) * 2.0**-48
        want_differs = int(np.count_nonzero(~near))
    else:
        want_differs = int(np.count_nonzero(~reference.same(p.ret, z["want"][idx], want)))
    return {
        "key": key,
        "checked": int(len(idx)),
        "want_differs": want_differs,
        "llvm_differs": int(np.count_nonzero(~_agree(p, z["llvm"][idx], want)[0])),
        "ir_differs": int(np.count_nonzero(~_agree(p, z["ir"][idx], want)[0])),
        "ret_is_float": isinstance(p.ret, Float),
    }


def sweep(key: str, name: str, out: Path, part: int, parts: int, block: int = 1 << 20) -> dict:
    """Every input of an approximate op with 32 input bits in all (a unary op on a 32-bit format, a binary
    one on a 16-bit format: index `i` is the operand pair `(i >> 16, i & 0xffff)`), in share `part` of
    `parts`, on the Allo `llvm` build, against the reference: the inputs outside the accuracy limit and the
    largest error in ulps (a one-off check, docs/passes/softfloat.md §7; nothing is stored)."""
    p = catalog.parse_key(key)
    if not p.approximate or p.input_bits != 32:
        raise ValueError(f"{key}: not an approximate op with 32 input bits")
    two = len(p.params) == 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(catalog.source(p, name))
    sim = _simulator(p, out, name, chunk=1 << 16)
    total = 1 << 32
    lo, hi = total * part // parts, total * (part + 1) // parts
    t0 = time.time()
    bad, worst, worst_at, examples = 0, 0.0, -1, []
    for start in range(lo, hi, block):
        x = np.arange(start, min(start + block, hi), dtype=np.uint64)
        args = [x >> np.uint64(16), x & np.uint64(0xFFFF)] if two else [x]
        got = sim(args)
        want = reference.evaluate(p, args)
        ok, err = _agree(p, got, want)
        bad += int(np.count_nonzero(~ok))
        for i in np.nonzero(~ok)[0][: max(0, 8 - len(examples))]:
            examples.append({"args": [int(a[i]) for a in args], "want": float(want[i]), "got": int(got[i])})
        if err is not None and np.isfinite(err).any():
            i = int(np.nanargmax(err))
            if err[i] > worst:
                worst, worst_at = float(err[i]), int(x[i])
    return {
        "key": key,
        "part": part,
        "parts": parts,
        "inputs": hi - lo,
        "bad": bad,
        "max_ulp": worst,
        "max_ulp_at": worst_at,
        "examples": examples,
        "seconds": round(time.time() - t0, 1),
    }
