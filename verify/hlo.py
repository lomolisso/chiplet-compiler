"""Abstract lowering of a port's phases and a census of its GEMMs / convolutions.

Nothing is executed: every phase is lowered with `jax.jit(fn).lower(...)` on
`ShapeDtypeStruct`s, and the StableHLO text is parsed. Each `dot_general` /
`convolution` is reported with its operand element types, accumulator (result)
type and FLOPs (2 * MACs), with calls to private functions expanded.

Also reads modules from symbolic export with component composites: composite
decompositions are expanded like calls, `dynamic_conv` counts as a convolution, and a
dynamic (`?`) dim leaves the FLOPs unknown (None).

    python -m verify.hlo resnet50 [reduced|full]
"""

from __future__ import annotations

import collections
import importlib
import re
import sys
from dataclasses import dataclass

import jax

_TENSOR = re.compile(r"tensor<([^>]*)>")
_FUNC = re.compile(r"^\s*func\.func\s+(?:public\s+|private\s+)?@([\w.$-]+)")
_CALL = re.compile(r"(?:(?:func\.)?call\s+|decomposition = )@([\w.$-]+)")


def _ttype(s: str):
    parts = s.split("x")
    dims = [None if p == "?" else int(p) for p in parts[:-1]]
    return tuple(dims), parts[-1]


@dataclass(frozen=True)
class Op:
    kind: str           # "dot_general" | "convolution"
    lhs: str            # element types
    rhs: str
    out: str
    flops: int | None   # None when a dim is dynamic
    lhs_shape: tuple
    rhs_shape: tuple
    out_shape: tuple

    @property
    def sig(self) -> str:
        return f"{self.lhs}x{self.rhs}->{self.out}"


def _mul(n, d):
    return None if n is None or d is None else n * d


def _parse_op(line: str) -> Op | None:
    if "stablehlo.dot_general" in line:
        kind = "dot_general"
    elif "stablehlo.convolution" in line or "stablehlo.dynamic_conv" in line:
        kind = "convolution"
    else:
        return None
    sig = line.rsplit(" : ", 1)[1]
    ts = [_ttype(t) for t in _TENSOR.findall(sig)]
    (ls, lt), (rs, rt), (os_, ot) = ts[0], ts[1], ts[-1]
    out_n = 1
    for d in os_:
        out_n = _mul(out_n, d)
    if kind == "dot_general":
        m = re.search(r"contracting_dims = \[([\d, ]*)\] x", line)
        k = 1
        for i in (int(t) for t in m.group(1).split(",") if t.strip()):
            k = _mul(k, ls[i])
    else:
        m = re.search(r"x\[([^\]]*)\]->", line)
        labels = [t.strip() for t in m.group(1).split(",")]
        k = 1
        for lab, d in zip(labels, rs):
            if lab != "o":
                k = _mul(k, d)
    flops = None if out_n is None or k is None else 2 * out_n * k
    return Op(kind, lt, rt, ot, flops, ls, rs, os_)


def census(text: str) -> list[Op]:
    """Every dot_general / convolution executed by `main`, private calls expanded."""
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        m = _FUNC.match(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur is not None:
            funcs[cur].append(line)
    memo: dict[str, list[Op]] = {}

    def expand(name: str) -> list[Op]:
        if name in memo:
            return memo[name]
        ops: list[Op] = []
        for line in funcs.get(name, []):
            op = _parse_op(line)
            if op is not None:
                ops.append(op)
            for callee in _CALL.findall(line):
                ops.extend(expand(callee))
        memo[name] = ops
        return ops

    return expand("main")


def load(key: str):
    return importlib.import_module(f"models.{key}.model")


def lower_phases(key: str, cfg_name: str = "reduced", phases=None) -> dict[str, str]:
    """{phase name: StableHLO text} for every phase of `key` at CONFIGS[cfg_name]."""
    mod = load(key)
    cfg = mod.CONFIGS[cfg_name]
    spec = mod.build(cfg)
    out = {}
    for ph in spec.phases:
        if phases is not None and ph.name not in phases:
            continue
        params = spec.param_specs
        if ph.param_keys is not None:
            params = {k: params[k] for k in ph.param_keys}
        out[ph.name] = jax.jit(ph.fn).lower(params, *ph.inputs).as_text()
    return out


def summarize(ops: list[Op]) -> dict[str, dict]:
    """{signature: {"count", "flops"}} grouped by kind + operand/accumulator types."""
    agg: dict[str, dict] = collections.defaultdict(lambda: {"count": 0, "flops": 0})
    for op in ops:
        a = agg[f"{op.kind} {op.sig}"]
        a["count"] += 1
        a["flops"] = None if a["flops"] is None or op.flops is None else a["flops"] + op.flops
    return dict(sorted(agg.items()))


def main(argv):
    key = argv[1]
    cfg_name = argv[2] if len(argv) > 2 else "reduced"
    spec = load(key).build(load(key).CONFIGS[cfg_name])
    trips = {p.name: p.trip_count for p in spec.phases}
    for name, text in lower_phases(key, cfg_name).items():
        print(f"== {key}/{cfg_name} phase {name} (trip {trips[name]})")
        for sig, a in summarize(census(text)).items():
            print(f"   {sig:55s} n={a['count']:5d}  GFLOP={a['flops'] / 1e9:12.4f}")


if __name__ == "__main__":
    main(sys.argv)
