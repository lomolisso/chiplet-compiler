"""Symbolic input shapes from a port's declaration (`ModelSpec.symbols`, `Phase.shapes`).

Every phase of a model shares one `SymbolicScope` built from the model's constraints (restricted to the
constraints whose symbols the phase uses: a scope may not mention a variable no input carries).
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import jax
from jax import export

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FUNCS = frozenset({"mod", "floordiv", "max", "min", "non_negative"})


def names_in(expr: str) -> set[str]:
    return {t for t in _IDENT.findall(expr) if t not in _FUNCS and t != "_"}


def spec_leaves(shapes: Any) -> list[str]:
    if shapes is None:
        return []
    return [s for s in jax.tree.leaves(shapes) if isinstance(s, str)]


def phase_symbols(shapes: Any) -> set[str]:
    out: set[str] = set()
    for s in spec_leaves(shapes):
        out |= names_in(s)
    return out


def symbolic_args(args: tuple, shapes: Any, constraints: tuple[str, ...]):
    """`args` with the dims named in `shapes` replaced by symbols. `shapes` is a prefix of `args`."""
    if shapes is None:
        return args, ()
    used = phase_symbols(shapes)
    cons = tuple(c for c in constraints if names_in(c) <= used)
    scope = export.SymbolicScope(cons)
    return export.symbolic_args_specs(args, shapes, scope=scope), cons


def perturbed(cfg: Any, symbols: dict) -> Any:
    """`cfg` with every symbol's field set to its `alt` value (dotted fields reach nested configs)."""
    for sym in symbols.values():
        cfg = _replace(cfg, sym.field.split("."), sym.alt)
    return cfg


def _replace(obj: Any, path: list[str], value: Any) -> Any:
    head, rest = path[0], path[1:]
    if rest:
        value = _replace(getattr(obj, head), rest, value)
    if dataclasses.is_dataclass(obj):
        return dataclasses.replace(obj, **{head: value})
    if isinstance(obj, dict):
        return {**obj, head: value}
    raise TypeError(f"cannot set field {head!r} on {type(obj).__name__}")


def symbolic_fields(cfg: Any, symbols: dict) -> dict[type, dict[str, str]]:
    """{config class: {field: symbol}} for the composite attributes (several symbols may share a field)."""
    out: dict[type, dict[str, str]] = {}
    for name in sorted(symbols):
        *head, last = symbols[name].field.split(".")
        owner = cfg
        for p in head:
            owner = owner[p] if isinstance(owner, dict) else getattr(owner, p)
        names = out.setdefault(type(owner), {})
        names[last] = f"{names[last]},{name}" if last in names else name
    return out


def field_value(cfg: Any, field: str) -> Any:
    for p in field.split("."):
        cfg = cfg[p] if isinstance(cfg, dict) else getattr(cfg, p)
    return cfg


def input_axes(flat_paths: list[str], avals: list, kinds: list[str]) -> tuple[dict, list]:
    """Where each symbol appears ({sym: [{path, axis, expr}]}) and every concrete axis of a non-param input."""
    where: dict[str, list] = {}
    concrete = []
    for path, aval, kind in zip(flat_paths, avals, kinds):
        for ax, d in enumerate(aval.shape):
            if isinstance(d, int):
                if kind == "input":
                    concrete.append({"path": path, "axis": ax, "size": d})
                continue
            expr = str(d)
            for name in names_in(expr):
                where.setdefault(name, []).append({"path": path, "axis": ax, "expr": expr})
    return where, concrete
