"""Haiku: every non-root module method call becomes a composite.

Mechanism:
- `hk.intercept_methods` sees every module method call (`MethodContext.module`, `.method_name`).
- The decomposition calls the intercepted method itself. A Haiku module reads its parameters from the
  transform frame (`hk.get_parameter`, through every custom getter the port installs); inside the
  decomposition those reads are values of the enclosing trace, which `composite.cc_composite` hoists into
  leading operands (closure conversion). So the parameters are the composite's operands, read exactly as the
  port reads them.
- Root: a discovery trace records top-level calls (module names without `/`); when the phase makes exactly
  one, that module is the phase root and passes through (`Session.roots`). Several top-level calls are all
  components.
"""

from __future__ import annotations

import haiku as hk

from . import composite

_SKIP_ATTRS = frozenset({"module_name", "name"})


def _fields(m: hk.Module) -> dict:
    return {k: v for k, v in vars(m).items() if not k.startswith("_") and k not in _SKIP_ATTRS}


def interceptor(next_f, args, kwargs, context):
    m = context.module
    method = context.method_name
    # Haiku also intercepts its own public methods (`params_dict`, `state_dict`): only methods the model
    # defines are components.
    if (method.startswith("_") and method != "__call__") or (method != "__call__" and hasattr(hk.Module, method)):
        return next_f(*args, **kwargs)
    s = composite.current()
    if s is None:
        return next_f(*args, **kwargs)
    full = m.module_name
    if "/" not in full and s.discover:
        s.top_calls.append(full)
    cls = type(m).__name__
    name = f"{composite.NAMESPACE}.haiku.{cls}" + ("" if method == "__call__" else f".{method}")
    return composite.wrap(
        name,
        cls,
        full,
        composite.attributes(_fields(m)),
        (),
        lambda _extra, *a, **k: next_f(*a, **k),
        args,
        kwargs,
        lambda: next_f(*args, **kwargs),
    )


def intercept():
    """Context manager installing the Haiku interceptor."""
    return hk.intercept_methods(interceptor)
