#!/usr/bin/python3.12
"""Report every tool's version as JSON.

At image build time `cc-probe --write /opt/cc/versions.json --pin k=v ...` records the versions
and the pins the image was built from. `chipc doctor` runs `cc-probe` again in a container and
compares the `tools` map with the recorded file, so a tool that no longer starts is caught.

Output: {"schema_version": 1, "pins": {...}, "tools": {name: {"argv": [...], "exit_code": n,
"version": "<first non-empty output line>"}}}. Only the standard library is used.
"""

import argparse
import json
import subprocess
import sys

# (name, argv). The version is the first non-empty line of stdout, else stderr.
PROBES = [
    ("python", ["python3", "--version"]),
    ("uv", ["uv", "--version"]),
    ("allo", ["python3", "-c", "import importlib.metadata as m, allo; print(m.version('allo'))"]),
    ("llvm", ["llvm-config", "--version"]),
    ("mlir-opt", ["mlir-opt", "--version"]),
    ("xlscc", ["xlscc", "--version"]),
    ("opt_main", ["opt_main", "--version"]),
    ("codegen_main", ["codegen_main", "--version"]),
    ("ir_converter_main", ["ir_converter_main", "--version"]),
    ("interpreter_main", ["interpreter_main", "--version"]),
    ("prove_quickcheck_main", ["prove_quickcheck_main", "--version"]),
    ("eval_ir_main", ["eval_ir_main", "--version"]),
    ("check_ir_equivalence_main", ["check_ir_equivalence_main", "--version"]),
    ("java", ["java", "-version"]),
    ("grami", ["test", "-f", "/opt/grami/GRAMI_DIRECTED_SUBGRAPHS/bin/Dijkstra/main.class"]),
    ("yosys", ["yosys", "-V"]),
    ("iverilog", ["iverilog", "-V"]),
    ("verilator", ["verilator", "--version"]),
    ("z3", ["z3", "--version"]),
    ("bitwuzla", ["bitwuzla", "--version"]),
    ("gcc", ["gcc", "--version"]),
]


def probe(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        return {"argv": argv, "exit_code": 127, "version": ""}
    except subprocess.TimeoutExpired:
        return {"argv": argv, "exit_code": 124, "version": ""}
    lines = [l.strip() for l in (p.stdout.splitlines() + p.stderr.splitlines()) if l.strip()]
    version = lines[0] if lines else ("present" if p.returncode == 0 else "")
    return {"argv": argv, "exit_code": p.returncode, "version": version}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", help="also write the JSON to this path")
    ap.add_argument("--pin", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    pins = {}
    for kv in a.pin:
        k, sep, v = kv.partition("=")
        if not sep or not v:
            sys.exit(f"cc-probe: bad or empty pin {kv!r}")
        pins[k] = v
    report = {"schema_version": 1, "pins": pins, "tools": {n: probe(argv) for n, argv in PROBES}}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if a.write:
        with open(a.write, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
