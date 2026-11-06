"""bf16 and f16 sine and cosine through XLS's `fp_sincos_32`: every input of the format widened to f32
(exactly; bf16 subnormals read as zero and are left out), the real IR evaluated by `eval_ir_main`, one
round to nearest even back to the format, the error against float64 `sin`/`cos` (docs/framework/float.md,
"Sine and cosine").

    python -m cc_float.sincos inputs DIR        # DIR/in.txt: one f32 argument per line, IR syntax
    eval_ir_main --input_file=DIR/in.txt sincos.opt.ir > DIR/out.txt     (in the toolchain image)
    python -m cc_float.sincos analyse DIR --out sincos.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from .fpmodel import BF16, F16, all_values, round_result, ulp_error

FORMATS = (BF16, F16)


def inputs(d: Path) -> np.ndarray:
    xs = np.unique(np.concatenate([all_values(f) for f in FORMATS]))
    xs = xs[xs != 0]
    bits = xs.astype(np.float32).view(np.uint32)
    lines = [
        f"(bits[1]:{b >> 31:#x}, bits[8]:{(b >> 23) & 0xFF:#x}, bits[23]:{b & 0x7FFFFF:#x})"
        for b in bits.tolist()
    ]
    d.mkdir(parents=True, exist_ok=True)
    (d / "in.txt").write_text("\n".join(lines) + "\n")
    return bits


_TUPLE = re.compile(r"bits\[1\]:(0x[0-9a-f_]+), bits\[8\]:(0x[0-9a-f_]+), bits\[23\]:(0x[0-9a-f_]+)")


def parse(text: str) -> tuple[np.ndarray, np.ndarray]:
    sins, coss = [], []
    for line in text.splitlines():
        m = _TUPLE.findall(line)
        if len(m) != 2:
            continue
        for (s, e, f), out in zip(m, (sins, coss), strict=True):
            out.append((int(s.replace("_", ""), 16) << 31) | (int(e.replace("_", ""), 16) << 23) | int(f.replace("_", ""), 16))
    to = lambda v: np.array(v, dtype=np.uint32).view(np.float32).astype(np.float64)  # noqa: E731
    return to(sins), to(coss)


def analyse(d: Path) -> dict:
    lines = (d / "in.txt").read_text().splitlines()
    x = np.array(
        [
            (int(a, 16) << 31) | (int(b, 16) << 23) | int(c, 16)
            for a, b, c in (re.findall(r":(0x[0-9a-f]+)", line_) for line_ in lines)
        ],
        dtype=np.uint32,
    ).view(np.float32).astype(np.float64)
    s32, c32 = parse((d / "out.txt").read_text())
    if len(s32) != len(x):
        raise SystemExit(f"{len(s32)} results for {len(x)} inputs")
    out = {"inputs_f32": int(len(x)), "formats": {}}
    for f in FORMATS:
        vals = all_values(f)
        vals = vals[(vals != 0)]
        # in.txt lists the inputs in increasing order (np.unique).
        idx = np.searchsorted(x, vals)
        res = {}
        for name, got32, ref in (("sine", s32, np.sin), ("cosine", c32, np.cos)):
            r = round_result(f, got32[idx])
            v = ref(vals)
            e = ulp_error(f, r, v)
            cr = round_result(f, v)
            not_cr = int(np.sum(r != cr))
            small = np.abs(vals) < 2.0**-4
            i = int(np.argmax(e))
            res[name] = {
                "inputs": int(len(vals)),
                "max_ulp": float(e[i]),
                "worst_input": float(vals[i]),
                "max_ulp_below_2^-4": float(e[small].max()) if small.any() else 0.0,
                "not_correctly_rounded": not_cr,
                "within_0.51_ulp": bool(e.max() <= 0.51),
            }
        out["formats"][f.name] = res
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inputs")
    a.add_argument("dir", type=Path)
    b = sub.add_parser("analyse")
    b.add_argument("dir", type=Path)
    b.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "inputs":
        n = len(inputs(args.dir))
        print(f"{n} inputs")
        return 0
    res = analyse(args.dir)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
