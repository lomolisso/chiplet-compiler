"""StableHLO module → `graph.json` (schema `docs/schemas/export_graph.schema.json`).

The module is walked with JAX's bundled MLIR Python bindings (`jax._src.lib.mlir.ir`); MLIR text is never
parsed by hand here. Ids are assigned in walk order, so the same module always gives the same JSON.

- Values: integer ids, numbered in definition order (block arguments, then op results). An operand is the id
  of the value it reads; its port is its position in `operands`.
- Types are tagged by `kind`: `{"kind": "tensor", "shape": [16, null], "elem": "f32"}` (`null` = dynamic,
  `?` in MLIR), `{"kind": "function", "inputs": [...], "results": [...]}`, `{"kind": "other", "value": ...}`.
  The symbolic expression of a dynamic dim is not in the type: it follows from the dimension-variable
  arguments (`jax.global_constant`) through the explicit size computation (the `import` pass).
- Attributes are tagged by `kind` too: `{"kind": "int", "value": 3, "type": "i64"}`, `{"kind": "ints",
  "value": [...]}`, `{"kind": "dot_dimension_numbers", "lhs_batching_dimensions": [...], ...}`, ...; anything
  without a structured form is `{"kind": "opaque", "value": "<printed attribute>"}`.
The Rust types (and the JSON Schema) are in `crates/cc-passes/src/export/json.rs`.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np
from jax._src.interpreters import mlir as jmlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo

SCHEMA = "cc.export.graph/1"
DENSE_VALUES_MAX = 1024  # constants with more elements are recorded by digest only

# StableHLO attribute wrappers and the properties that define them (jaxlib 0.12 bindings).
_HLO_ATTRS: list[tuple[str, Any, tuple[str, ...]]] = [
    (
        "dot_dimension_numbers",
        hlo.DotDimensionNumbers,
        (
            "lhs_batching_dimensions",
            "rhs_batching_dimensions",
            "lhs_contracting_dimensions",
            "rhs_contracting_dimensions",
        ),
    ),
    (
        "conv_dimension_numbers",
        hlo.ConvDimensionNumbers,
        (
            "input_batch_dimension",
            "input_feature_dimension",
            "input_spatial_dimensions",
            "kernel_input_feature_dimension",
            "kernel_output_feature_dimension",
            "kernel_spatial_dimensions",
            "output_batch_dimension",
            "output_feature_dimension",
            "output_spatial_dimensions",
        ),
    ),
    (
        "gather_dimension_numbers",
        hlo.GatherDimensionNumbers,
        (
            "offset_dims",
            "collapsed_slice_dims",
            "operand_batching_dims",
            "start_indices_batching_dims",
            "start_index_map",
            "index_vector_dim",
        ),
    ),
    (
        "scatter_dimension_numbers",
        hlo.ScatterDimensionNumbers,
        (
            "update_window_dims",
            "inserted_window_dims",
            "input_batching_dims",
            "scatter_indices_batching_dims",
            "scattered_dims_to_operand_dims",
            "index_vector_dim",
        ),
    ),
]
_HLO_ENUMS: list[tuple[str, Any]] = [
    ("comparison_direction", hlo.ComparisonDirectionAttr),
    ("comparison_type", hlo.ComparisonTypeAttr),
    ("precision", hlo.PrecisionAttr),
    ("fft_type", hlo.FftTypeAttr),
    ("transpose", hlo.TransposeAttr),
    ("rng_algorithm", hlo.RngAlgorithmAttr),
    ("rng_distribution", hlo.RngDistributionAttr),
]
_ENUM_VALUE = re.compile(r"^#stablehlo<\s*\w+\s+(.*)>$")


def type_json(t: ir.Type) -> dict:
    if isinstance(t, ir.RankedTensorType):
        return {
            "kind": "tensor",
            "shape": [None if ir.ShapedType.is_dynamic_size(d) else int(d) for d in t.shape],
            "elem": str(t.element_type),
        }
    if isinstance(t, ir.FunctionType):
        return {
            "kind": "function",
            "inputs": [type_json(x) for x in t.inputs],
            "results": [type_json(x) for x in t.results],
        }
    return {"kind": "other", "value": str(t)}


def _float(x: float) -> float | str:
    return x if math.isfinite(x) else repr(x)


def _dense(a: ir.DenseElementsAttr) -> dict:
    out: dict[str, Any] = {"kind": "dense", "type": type_json(a.type), "splat": bool(a.is_splat)}
    try:
        arr = np.asarray(a)
    except Exception:  # noqa: BLE001 - element types numpy cannot view (bf16, f8, complex): digest only
        arr = None
    if arr is not None and arr.size <= DENSE_VALUES_MAX:
        vals = arr.reshape(-1).tolist()
        if arr.dtype.kind == "f":
            vals = [_float(v) for v in vals]
        elif arr.dtype.kind == "b":
            vals = [bool(v) for v in vals]
        out["values"] = vals
    else:
        payload = arr.tobytes() if arr is not None else str(a).encode()
        out["digest"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    return out


def attr_json(a: ir.Attribute) -> dict:
    for key, wrapper, props in _HLO_ATTRS:
        if wrapper.isinstance(a):
            w = wrapper(a)
            return {"kind": key, **{p: _plain(getattr(w, p)) for p in props}}
    for key, wrapper in _HLO_ENUMS:
        if wrapper.isinstance(a):
            m = _ENUM_VALUE.match(str(a))
            return {"kind": "enum", "enum": key, "value": m.group(1).strip() if m else str(a)}
    if isinstance(a, ir.BoolAttr):
        return {"kind": "bool", "value": bool(a.value)}
    if isinstance(a, ir.IntegerAttr):
        return {"kind": "int", "value": int(a.value), "type": str(a.type)}
    if isinstance(a, ir.FloatAttr):
        return {"kind": "float", "value": _float(float(a.value)), "type": str(a.type)}
    if isinstance(a, ir.StringAttr):
        return {"kind": "str", "value": a.value}
    if isinstance(a, ir.FlatSymbolRefAttr):
        return {"kind": "symbol", "value": a.value}
    if isinstance(a, ir.DenseI64ArrayAttr) or isinstance(a, ir.DenseI32ArrayAttr):
        return {"kind": "ints", "value": [int(x) for x in a]}
    if isinstance(a, ir.DenseBoolArrayAttr):
        return {"kind": "bools", "value": [bool(x) for x in a]}
    if isinstance(a, ir.DenseElementsAttr):
        return _dense(a)
    if isinstance(a, ir.ArrayAttr):
        return {"kind": "array", "value": [attr_json(x) for x in a]}
    if isinstance(a, ir.DictAttr):
        return {"kind": "dict", "value": dict_json(a)}
    if isinstance(a, ir.TypeAttr):
        return {"kind": "type", "value": type_json(a.value)}
    if isinstance(a, ir.UnitAttr):
        return {"kind": "unit"}
    return {"kind": "opaque", "value": str(a)}


def _plain(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return int(v) if hasattr(v, "__int__") else str(v)


def dict_json(d: ir.DictAttr) -> dict:
    return {d[i].name: attr_json(d[i].attr) for i in range(len(d))}


def op_attrs(op: ir.Operation) -> dict:
    attrs = op.attributes
    return {attrs[i].name: attr_json(attrs[i].attr) for i in range(len(attrs))}


class _Walker:
    def __init__(self, loc_strip: list[str]):
        self.values: dict[ir.Value, int] = {}
        self.next_value = 0
        self.next_op = 0
        self.next_block = 0
        self.loc_strip = loc_strip

    def value(self, v: ir.Value) -> int:
        vid = self.values.get(v)
        if vid is None:
            raise KeyError("use of an undefined value (not dominated in walk order)")
        return vid

    def define(self, v: ir.Value) -> int:
        vid = self.next_value
        self.next_value += 1
        self.values[v] = vid
        return vid

    def loc(self, op: ir.Operation) -> str:
        s = str(op.location)
        for p in self.loc_strip:
            s = s.replace(p, "")
        return s

    def region(self, r: ir.Region) -> dict:
        blocks = []
        for b in r.blocks:
            bid = self.next_block
            self.next_block += 1
            args = [{"v": self.define(a), "type": type_json(a.type)} for a in b.arguments]
            ops = [self.op(o.operation) for o in b.operations]
            blocks.append({"id": bid, "args": args, "ops": ops})
        return {"blocks": blocks}

    def op(self, op: ir.Operation) -> dict:
        oid = self.next_op
        self.next_op += 1
        rec = {
            "id": oid,
            "name": op.name,
            "operands": [self.value(v) for v in op.operands],
        }
        # Results are defined before the regions are walked: region bodies may not use them (SSA), but a
        # fixed order keeps ids stable.
        rec["results"] = [{"v": self.define(r), "type": type_json(r.type)} for r in op.results]
        rec["attributes"] = op_attrs(op)
        if len(op.regions):
            rec["regions"] = [self.region(r) for r in op.regions]
        rec["loc"] = self.loc(op)
        return rec

    def func(self, f: ir.Operation) -> dict:
        attrs = op_attrs(f)
        name = attrs.pop("sym_name")["value"]
        vis = attrs.pop("sym_visibility", {"value": "public"})["value"]
        ftype = attrs.pop("function_type")["value"]
        arg_attrs = [a.get("value", {}) for a in attrs.pop("arg_attrs", {"value": []})["value"]]
        res_attrs = [a.get("value", {}) for a in attrs.pop("res_attrs", {"value": []})["value"]]
        body = self.region(f.regions[0])
        return {
            "name": name,
            "visibility": vis,
            "type": {"inputs": ftype["inputs"], "results": ftype["results"]},
            "arg_attrs": arg_attrs,
            "res_attrs": res_attrs,
            "attributes": attrs,
            "body": body,
        }


def walk(text: str, loc_strip: list[str] | None = None) -> tuple[dict, dict]:
    """Parse and verify `text`; return (module JSON, MLIR-side facts for the checks)."""
    with jmlir.make_ir_context():
        m = ir.Module.parse(text)
        verified = bool(m.operation.verify())
        w = _Walker(loc_strip or [])
        funcs, other = [], []
        for o in m.body.operations:
            op = o.operation
            if op.name == "func.func":
                funcs.append(w.func(op))
            else:
                other.append(w.op(op))
        mod_attrs = op_attrs(m.operation)
        facts = {"verified": verified, "op_multiset": _multiset(m.operation)}
    return {"attributes": mod_attrs, "funcs": funcs, "other_ops": other}, facts


def _multiset(root: ir.Operation) -> dict[str, int]:
    """Op-name multiset over every op nested in `root` (excluding `root`)."""
    counts: dict[str, int] = {}

    def rec(op):
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    n = o.operation.name
                    counts[n] = counts.get(n, 0) + 1
                    rec(o.operation)

    rec(root)
    return dict(sorted(counts.items()))


def iter_ops(region: dict):
    """Every op in a region JSON, nested regions included (pre-order)."""
    for b in region["blocks"]:
        for op in b["ops"]:
            yield op
            for r in op.get("regions", []):
                yield from iter_ops(r)


def graph_multiset(module: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in module["funcs"]:
        counts["func.func"] = counts.get("func.func", 0) + 1
        for op in iter_ops(f["body"]):
            counts[op["name"]] = counts.get(op["name"], 0) + 1
    for op in module["other_ops"]:
        counts[op["name"]] = counts.get(op["name"], 0) + 1
    return dict(sorted(counts.items()))
