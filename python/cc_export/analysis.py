"""Facts derived from the module JSON: composite instances, coverage, GEMM/conv census, canonical hash.

Everything here follows the static call graph from the entry function: `func.call` and composite
decompositions are expanded at each call site, as `verify/hlo.py`'s census does, so counts are per executed
op (per invocation of the phase).
"""

from __future__ import annotations

import collections
import hashlib
import json
from typing import Any

from .composite import PATH_ATTR

# Ops that only delimit structure; coverage counts every other op.
STRUCTURAL = frozenset({"func.return", "stablehlo.return", "func.call", "stablehlo.composite"})
CORE_KINDS = {
    "stablehlo.dot_general": "dot_general",
    "stablehlo.convolution": "convolution",
    "stablehlo.dynamic_conv": "convolution",
}


def _funcs(module: dict) -> dict[str, dict]:
    return {f["name"]: f for f in module["funcs"]}


def _types(module: dict) -> dict[int, dict]:
    t: dict[int, dict] = {}

    def region(r):
        for b in r["blocks"]:
            for a in b["args"]:
                t[a["v"]] = a["type"]
            for op in b["ops"]:
                for res in op["results"]:
                    t[res["v"]] = res["type"]
                for rr in op.get("regions", []):
                    region(rr)

    for f in module["funcs"]:
        region(f["body"])
    return t


def _callee(op: dict) -> str | None:
    a = op["attributes"]
    if op["name"] == "func.call":
        return a["callee"]["value"]
    if op["name"] == "stablehlo.composite":
        return a["decomposition"]["value"]
    return None


def walk_expanded(module: dict, entry: str = "main"):
    """Yield (func name, op, composite instance stack) for every op executed from `entry`.

    The stack holds instance ids (ints) of the enclosing composites, innermost last. Instances are numbered
    in expansion order; `instances` collects their records.
    """
    funcs = _funcs(module)
    instances: list[dict] = []

    def region_ops(r):
        for b in r["blocks"]:
            for op in b["ops"]:
                yield op
                for rr in op.get("regions", []):
                    yield from region_ops(rr)

    def visit(fname: str, stack: tuple[int, ...], depth: int):
        if depth > 10_000:
            raise RecursionError(f"call depth over 10000 at {fname}")
        for op in region_ops(funcs[fname]["body"]):
            yield fname, op, stack
            callee = _callee(op)
            if callee is None:
                continue
            if op["name"] == "stablehlo.composite":
                a = op["attributes"]
                attrs = a.get("composite_attributes", {"value": {}})["value"]
                inst = len(instances)
                instances.append(
                    {
                        "instance": inst,
                        "op": op["id"],
                        "func": fname,
                        "name": a["name"]["value"],
                        "path": attrs.get(PATH_ATTR, {}).get("value", ""),
                        "decomposition": callee,
                        "version": a.get("version", {"value": 0})["value"],
                        "parent": stack[-1] if stack else None,
                    }
                )
                yield from visit(callee, stack + (inst,), depth + 1)
            else:
                yield from visit(callee, stack, depth + 1)

    return visit(entry, (), 0), instances


def composites_and_coverage(module: dict, entry: str = "main") -> tuple[list[dict], dict]:
    gen, instances = walk_expanded(module, entry)
    own: collections.Counter[int] = collections.Counter()
    total = anon = 0
    anon_names: collections.Counter[str] = collections.Counter()
    rows = []
    for _fname, op, stack in gen:
        if op["name"] in STRUCTURAL:
            continue
        total += 1
        if stack:
            own[stack[-1]] += 1
        else:
            anon += 1
            anon_names[op["name"]] += 1
        rows.append(stack[-1] if stack else None)
    has_child = {i["parent"] for i in instances if i["parent"] is not None}
    for i in instances:
        i["leaf"] = i["instance"] not in has_child
        i["own_ops"] = own[i["instance"]]
    in_leaf = sum(1 for r in rows if r is not None and instances[r]["leaf"])
    names = collections.Counter(i["name"] for i in instances)
    cov = {
        "ops": total,
        "in_composite": total - anon,
        "in_leaf_composite": in_leaf,
        "anon": anon,
        "any": (total - anon) / total if total else 1.0,
        "leaf": in_leaf / total if total else 1.0,
        "instances": len(instances),
        "leaf_instances": sum(1 for i in instances if i["leaf"]),
        "by_name": dict(sorted(names.items())),
        "anon_by_op": dict(sorted(anon_names.items())),
    }
    return instances, cov


def _sig_elem(t: dict) -> str:
    return t["elem"] if t["kind"] == "tensor" else str(t)


def _shape(t: dict) -> list:
    return t["shape"] if t["kind"] == "tensor" else []


def _prod(xs) -> int | None:
    n = 1
    for x in xs:
        if x is None:
            return None
        n *= x
    return n


def census(module: dict, entry: str = "main") -> dict:
    """GEMM / convolution census with `verify/hlo.py`'s signatures, plus the executed op multiset."""
    types = _types(module)
    gen, _ = walk_expanded(module, entry)
    ops: list[dict] = []
    executed: collections.Counter[str] = collections.Counter()
    for _fname, op, _stack in gen:
        executed[op["name"]] += 1
        kind = CORE_KINDS.get(op["name"])
        if kind is None:
            continue
        lt, rt = types[op["operands"][0]], types[op["operands"][1]]
        ot = op["results"][0]["type"]
        ls, rs, os_ = _shape(lt), _shape(rt), _shape(ot)
        a = op["attributes"]
        if kind == "dot_general":
            dn = a["dot_dimension_numbers"]
            k = _prod(ls[i] for i in dn["lhs_contracting_dimensions"])
        else:
            dn = a["dimension_numbers"]
            out_f = dn["kernel_output_feature_dimension"]
            k = _prod(d for i, d in enumerate(rs) if i != out_f)
        out_n = _prod(os_)
        flops = 2 * out_n * k if out_n is not None and k is not None else None
        ops.append(
            {
                "kind": kind,
                "op": op["name"],
                "sig": f"{_sig_elem(lt)}x{_sig_elem(rt)}->{_sig_elem(ot)}",
                "flops": flops,
                "lhs_shape": ls,
                "rhs_shape": rs,
                "out_shape": os_,
            }
        )
    return {"summary": summarize(ops), "ops": ops, "executed": dict(sorted(executed.items()))}


def summarize(ops: list[dict]) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for op in ops:
        a = agg.setdefault(f"{op['kind']} {op['sig']}", {"count": 0, "flops": 0})
        a["count"] += 1
        a["flops"] = None if a["flops"] is None or op["flops"] is None else a["flops"] + op["flops"]
    return dict(sorted(agg.items()))


_PROVENANCE = frozenset({"loc", PATH_ATTR})


def _strip(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in _PROVENANCE}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def first_difference(a: Any, b: Any, path: str = "", limit: int = 5) -> list[dict]:
    """Up to `limit` leaf-level differences between two module JSONs (locations ignored)."""
    out: list[dict] = []

    def rec(x, y, p):
        if len(out) >= limit:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k in _PROVENANCE:
                    continue
                if k not in x or k not in y:
                    out.append({"path": f"{p}/{k}", "base": _clip(x.get(k)), "perturbed": _clip(y.get(k))})
                else:
                    rec(x[k], y[k], f"{p}/{k}")
                if len(out) >= limit:
                    return
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                out.append({"path": f"{p}[len]", "base": len(x), "perturbed": len(y)})
            for i, (u, v) in enumerate(zip(x, y)):
                rec(u, v, f"{p}[{i}]")
                if len(out) >= limit:
                    return
        elif x != y:
            out.append({"path": p, "base": _clip(x), "perturbed": _clip(y)})

    rec(a, b, path)
    return out


def _clip(x: Any) -> Any:
    s = json.dumps(x, sort_keys=True)
    return x if len(s) <= 300 else s[:300] + "…"


def describe_path(module: dict, path: str) -> str:
    """Name the op a `first_difference` path points into (func and op name), for humans."""
    import re

    m = re.match(r"/funcs\[(\d+)\]", path)
    if not m:
        return path
    f = module["funcs"][int(m.group(1))]
    ops = re.findall(r"/ops\[(\d+)\]", path)
    node: Any = f["body"]
    desc = f["name"]
    for part in re.finditer(r"/(blocks|ops|regions)\[(\d+)\]", path):
        kind, idx = part.group(1), int(part.group(2))
        node = node[kind][idx] if isinstance(node, dict) else node
        if kind == "ops":
            desc += f" > {node['name']} (op {node['id']}, {node.get('loc', '')[:160]})"
    return desc if ops else path


def canonical_hash(module: dict) -> str:
    """Hash of the module JSON without provenance (source locations, module paths): equal computations ⇔
    equal hashes. Paths are names (Keras auto-names come from process-global counters), not semantics."""
    blob = json.dumps(_strip(module), sort_keys=True, separators=(",", ":")).encode()
    return "blake2b:" + hashlib.blake2b(blob, digest_size=32).hexdigest()
