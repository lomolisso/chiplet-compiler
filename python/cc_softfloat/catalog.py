"""Primitive keys (as `characterize` names them: `<op>[.<variant>]@<type>[><type>]`) to generated sources."""

from __future__ import annotations

from dataclasses import dataclass

from . import ops
from .formats import BOOL, Float, Int, parse

#: `reduce_precision`'s (exponent bits, mantissa bits) of the catalog primitive.
DEFAULT_PRECISION = (5, 10)


@dataclass(frozen=True)
class Prim:
    key: str
    op: str
    variant: str | None
    params: list[Float | Int]
    ret: Float | Int
    precision: tuple[int, int] = DEFAULT_PRECISION

    @property
    def input_bits(self) -> int:
        return sum(p.bits for p in self.params)


def parse_key(key: str, precision: tuple[int, int] = DEFAULT_PRECISION) -> Prim:
    head, _, types = key.partition("@")
    op, _, variant = head.partition(".")
    names = types.split(">")
    if op == "convert":
        src, dst = parse(names[0]), parse(names[1])
        return Prim(key, op, None, [src], dst)
    t = parse(names[0])
    if not isinstance(t, Float):
        raise ValueError(f"{key}: not a float primitive")
    if op == "compare":
        return Prim(key, op, variant, [t, t], BOOL)
    if op == "select":
        return Prim(key, op, None, [BOOL, t, t], t)
    if op == "is_finite":
        return Prim(key, op, None, [t], BOOL)
    if op == "clamp":
        return Prim(key, op, None, [t, t, t], t)
    if op in {"add", "subtract", "multiply", "divide", "remainder", "maximum", "minimum"}:
        return Prim(key, op, None, [t, t], t)
    if op in {
        "sqrt",
        "negate",
        "abs",
        "sign",
        "round_nearest_even",
        "floor",
        "ceil",
        "reduce_precision",
    }:
        return Prim(key, op, None, [t], t, precision)
    raise ValueError(f"{key}: no generator for op {op!r}")


def is_float_key(key: str) -> bool:
    try:
        p = parse_key(key)
    except ValueError:
        return False
    return any(isinstance(t, Float) for t in [*p.params, p.ret])


def source(p: Prim, name: str) -> str:
    """The Allo source of primitive `p` as a function called `name`."""
    t = p.params[-1]
    match p.op:
        case "convert":
            return ops.convert(name, p.params[0], p.ret)
        case "add" | "subtract":
            return ops.add(name, t, p.op == "subtract")
        case "multiply":
            return ops.multiply(name, t)
        case "divide":
            return ops.divide(name, t)
        case "sqrt":
            return ops.sqrt(name, t)
        case "remainder":
            return ops.remainder(name, t)
        case "negate" | "abs" | "sign":
            return ops.sign_ops(name, t, p.op)
        case "compare":
            assert p.variant is not None
            return ops.compare(name, t, p.variant)
        case "maximum" | "minimum":
            return ops.maxmin(name, t, p.op == "maximum")
        case "clamp":
            return ops.clamp(name, t)
        case "select":
            return ops.select(name, t)
        case "is_finite":
            return ops.is_finite(name, t)
        case "round_nearest_even" | "floor" | "ceil":
            return ops.to_integral(name, t, p.op)
        case "reduce_precision":
            return ops.reduce_precision(name, t, *p.precision)
    raise ValueError(p.key)


__all__ = ["BOOL", "DEFAULT_PRECISION", "Float", "Int", "Prim", "is_float_key", "parse_key", "source"]
