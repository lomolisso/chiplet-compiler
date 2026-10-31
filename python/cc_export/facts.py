"""python -m cc_export.facts <model_dir>

MLIR-side facts for the export pass's `verify` (a fresh process, reading only files on disk): for every
`<model_dir>/<phase>/module.mlir`, re-parse it with JAX's bundled MLIR bindings, run the MLIR verifier and
count the op names of every op nested in the module. Prints one JSON object {phase: {"verified", "op_multiset"}}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jax._src.interpreters import mlir as jmlir
from jax._src.lib.mlir import ir


def facts(path: Path) -> dict:
    with jmlir.make_ir_context():
        m = ir.Module.parse(path.read_text())
        verified = bool(m.operation.verify())
        counts: dict[str, int] = {}

        def rec(op):
            for r in op.regions:
                for b in r.blocks:
                    for o in b.operations:
                        n = o.operation.name
                        counts[n] = counts.get(n, 0) + 1
                        rec(o.operation)

        rec(m.operation)
    return {"verified": verified, "op_multiset": dict(sorted(counts.items()))}


def main(argv: list[str]) -> int:
    root = Path(argv[1])
    out = {}
    for mlir in sorted(root.glob("*/module.mlir")):
        try:
            out[mlir.parent.name] = facts(mlir)
        except Exception as e:  # noqa: BLE001 - reported as a failed parse
            out[mlir.parent.name] = {"verified": False, "op_multiset": {}, "error": f"{type(e).__name__}: {e}"[:2000]}
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
