"""`python -m cc_softfloat build|recheck|source|sweep ...` (run by the `softfloat` pass in the toolchain image;
`sweep` is a one-off check of every input of a 32-bit approximate op, run by hand)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import catalog


def _precision(s: str) -> tuple[int, int]:
    e, m = s.split(",")
    return int(e), int(m)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cc_softfloat")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate, simulate, emit and check primitives")
    b.add_argument("--key", action="append", required=True)
    b.add_argument("--id", action="append", required=True)
    b.add_argument("--out", required=True, help="directory holding one subdirectory per id")
    b.add_argument("--exhaustive-bits", type=int, default=16)
    b.add_argument("--random", type=int, default=100000)
    b.add_argument("--precision", type=_precision, default=catalog.DEFAULT_PRECISION)
    r = sub.add_parser("recheck", help="recompute the reference on a sample of recorded vectors")
    r.add_argument("--key", action="append", required=True)
    r.add_argument("--id", action="append", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--sample", type=int, default=2000)
    r.add_argument("--precision", type=_precision, default=catalog.DEFAULT_PRECISION)
    s = sub.add_parser("source", help="print the generated Allo source of a primitive")
    s.add_argument("--key", required=True)
    s.add_argument("--id", required=True)
    w = sub.add_parser("sweep", help="check every input of a unary approximate op (one share of them)")
    w.add_argument("--key", required=True)
    w.add_argument("--id", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--part", type=int, default=0)
    w.add_argument("--parts", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "source":
        sys.stdout.write(catalog.source(catalog.parse_key(a.key), a.id))
        return 0
    if a.cmd == "sweep":
        from .build import sweep

        res = sweep(a.key, a.id, Path(a.out) / f"{a.id}.{a.part}", a.part, a.parts)
        print("@@" + json.dumps(res, sort_keys=True), flush=True)
        return 0 if res["bad"] == 0 else 1
    if len(a.key) != len(a.id):
        ap.error("--key and --id must pair up")
    status = 0
    for key, ident in zip(a.key, a.id):
        out = Path(a.out) / ident
        try:
            if a.cmd == "build":
                from .build import build

                res = build(key, ident, out, a.exhaustive_bits, a.random, a.precision)
            else:
                from .build import recheck

                res = recheck(key, out, a.precision, a.sample)
        except Exception as e:  # noqa: BLE001  (reported per primitive; the pass fails on it)
            res = {"key": key, "id": ident, "ok": False, "error": f"{type(e).__name__}: {e}"}
            out.mkdir(parents=True, exist_ok=True)
            if a.cmd == "build":
                (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
            status = 1
        print("@@" + json.dumps(res, sort_keys=True), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
