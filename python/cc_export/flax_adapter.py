"""Flax linen: every submodule method call becomes a composite.

Mechanism (docs/context/tooling/jax_export.md §4):
- `nn.intercept_methods` sees every wrapped module method call without touching model code.
- The module's variables (`module.variables`) are explicit operands; the decomposition re-applies an unbound
  clone (`module.clone(parent=None)`) to them. A bound module cannot be used inside another trace (Flax
  checks the trace level), hence the clone.
- Mutable collections (e.g. a KV `cache`): the clone is applied with the scope's own `mutable` filter; the
  variables it changes or creates are extra composite results and are written back into the module's scope
  after the call, so the caller sees exactly what an unwrapped call would have done.
- Re-entrancy: inside the decomposition the clone is the root of its own `apply`, so its scope path is `()`
  and the interceptor passes it through; its children have non-empty paths and get their own nested
  composites. The root of the phase's own `apply` is passed through the same way (the phase root is not a
  component).
"""

from __future__ import annotations

import contextvars
import dataclasses
from collections.abc import Mapping

import flax.linen as nn

from . import composite

_PREFIX: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("cc_flax_prefix", default=())

_SKIP_METHODS = frozenset({"setup", "__post_init__"})


def _fields(m: nn.Module) -> dict:
    return {f.name: getattr(m, f.name, None) for f in dataclasses.fields(m) if f.name not in ("parent", "name")}


def _changed(new: Mapping, old: Mapping) -> dict:
    """The leaves of `new` that are not the very same value as in `old` (new or updated variables)."""
    out = {}
    for k, v in new.items():
        o = old.get(k) if isinstance(old, Mapping) else None
        if isinstance(v, Mapping):
            sub = _changed(v, o if isinstance(o, Mapping) else {})
            if sub:
                out[k] = sub
        elif v is not o:
            out[k] = v
    return out


def _write_back(scope, col: str, tree: Mapping) -> None:
    for k, v in tree.items():
        if isinstance(v, Mapping):
            _write_back(scope.push(k, reuse=True), col, v)
        else:
            scope.put_variable(col, k, v)


def interceptor(next_fun, args, kwargs, context):
    m = context.module
    method = context.method_name
    if method in _SKIP_METHODS or (method.startswith("_") and method != "__call__"):
        return next_fun(*args, **kwargs)
    scope = m.scope
    if scope is None or not scope.path or composite.current() is None:
        return next_fun(*args, **kwargs)

    path = _PREFIX.get() + tuple(scope.path)
    cls = type(m).__name__
    name = f"{composite.NAMESPACE}.flax.{cls}" + ("" if method == "__call__" else f".{method}")
    variables = m.variables
    unbound = m.clone(parent=None)
    apply_method = None if method == "__call__" else method
    mutable = scope.mutable

    def call(v, *a, **k):
        tok = _PREFIX.set(path)
        try:
            if not mutable:
                return unbound.apply(v, *a, method=apply_method, **k), {}
            out, new_vars = unbound.apply(v, *a, method=apply_method, mutable=mutable, **k)
            return out, _changed(dict(new_vars), dict(v))
        finally:
            _PREFIX.reset(tok)

    def fallback():
        return next_fun(*args, **kwargs), None

    out, updates = composite.wrap(
        name,
        cls,
        "/".join(path),
        composite.attributes(_fields(m)),
        variables,
        call,
        args,
        kwargs,
        fallback,
    )
    for col, tree in (updates or {}).items():
        _write_back(scope, col, tree)
    return out


def intercept():
    """Context manager installing the Flax interceptor."""
    return nn.intercept_methods(interceptor)
