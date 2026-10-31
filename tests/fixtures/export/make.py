"""Regenerate tests/fixtures/export/: a real export of the toy Flax port (python/tests/toys.py) as model
`toy`, plus the facts the export pass's verify reads from `python -m cc_export.facts`. The Rust smoke tests
replay these files instead of running Python.

    PYTHONPATH=python:.:python/tests uv run --frozen python tests/fixtures/export/make.py
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "python" / "tests"), str(ROOT / "python"), str(ROOT)]

import conftest  # noqa: E402,F401  (environment)
import toys  # noqa: E402

from cc_export import exporter  # noqa: E402

out = HERE / "toy"
shutil.rmtree(out, ignore_errors=True)
s = exporter.export_model("toy", "reduced", out, port=toys.flax_port())
# A toy recipe audit (ports have models/<key>/recipe.py; the toy has none).
s["recipe"]["audit"] = {
    "allow": {"dot_general f32xf32->f32": "toy dense layers and attention"},
    "require": ["dot_general f32xf32->f32"],
}
s["model"] = "toy"
(out / "export.json").write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
for ph in s["phases"]:
    (out / ph / "exported.bin").unlink()  # not read by verify; keeps the fixture small
# Source locations carry machine paths: keep them relative (repo files) or under `site-packages/`.
site = re.compile(r'"[^"]*/site-packages/')
for p in sorted(out.rglob("*")):
    if p.suffix in (".mlir", ".json"):
        t = p.read_text().replace(str(ROOT) + "/", "")
        p.write_text(site.sub('"site-packages/', t))
facts = subprocess.run(
    [sys.executable, "-m", "cc_export.facts", str(out)],
    check=True, capture_output=True, text=True, cwd=ROOT,
    env={"PYTHONPATH": f"{ROOT / 'python'}:{ROOT}", "JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin"},
).stdout
(HERE / "facts.json").write_text(facts)
print({p: sum(f.stat().st_size for f in (out / p).iterdir()) for p in s["phases"]})
