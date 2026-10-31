"""python -m cc_export --model <key> [--phase <p> ...] --config reduced|full --out <dir>

Exports one model (all phases unless --phase is given) into <dir>. Exit status 0 when every phase exported
(checks are recorded in <dir>/export.json and judged by the Rust pass's verify), 3 when a phase failed to
export (also recorded there), 2 on usage errors, 1 on a crash. `--version` prints the library versions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cc_export")
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--phase", action="append")
    ap.add_argument("--config", default="reduced")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--exclude", action="append", default=[], help="class name never wrapped")
    ap.add_argument("--no-invariance", action="store_true", help="skip the perturbed re-export (debug only)")
    a = ap.parse_args(argv)

    from cc_export import exporter

    if a.version:
        print(" ".join(f"{k}={v}" for k, v in exporter.versions().items()))
        return 0
    if not a.model or not a.out:
        ap.error("--model and --out are required")
    s = exporter.export_model(
        a.model, a.config, a.out, phases=a.phase, exclude=tuple(a.exclude), invariance=not a.no_invariance
    )
    failed = [p for p, r in s["phases"].items() if "error" in r]
    bad_checks = {
        p: sorted(k for k, c in r.get("checks", {}).items() if not c.get("ok"))
        for p, r in s["phases"].items()
        if "checks" in r
    }
    print(
        json.dumps(
            {
                "model": a.model,
                "phases": len(s["phases"]),
                "failed": failed,
                "failing_checks": {p: c for p, c in bad_checks.items() if c},
                "model_checks": {k: v["ok"] for k, v in s["checks"].items()},
            }
        )
    )
    return 3 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
