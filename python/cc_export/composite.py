"""Shared composite machinery for the framework adapters.

A module call becomes a composite `cc.<framework>.<Class>[.<method>]` (`cc_composite`: `lax.composite` plus
closure conversion). The array leaves of the call's arguments become operands; every other leaf (Python ints,
flags, `None`, callables) stays static and is closed over by the decomposition. Adapters may pass the
module's variables as an explicit first operand (a pytree); any other value of the enclosing trace the
decomposition reads becomes a leading operand.

Static attributes are the module's configuration, made deterministic: values `mlir.ir_attribute` accepts are
kept, everything else becomes a stable string (never a `repr` with a memory address).
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import enum
import functools
import os
import re
from collections.abc import Callable, Mapping
from typing import Any

import jax
import numpy as np
from jax._src import api_util, core, tree_util
from jax._src import flattree as ft
from jax._src.hashable_array import HashableArray
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.lax import composite_p

NAMESPACE = "cc"
PATH_ATTR = "cc_path"  # provenance (module path); not part of the component's semantics

_I64 = (-(2**63), 2**63 - 1)
_SMALL_ARRAY = 16
_MAX_DEPTH = 4


@dataclasses.dataclass
class Session:
    """What one export records about interception.

    `discover` sessions wrap nothing: adapters only record top-level module calls (`top_calls`), from which
    the exporter derives the phase root (a single top-level call covering the whole phase). `roots` are the
    paths an export session passes through unwrapped.
    """

    exclude: frozenset[str] = frozenset()
    # Config fields that are symbols of the export: {config class: {field: symbol}}. In attributes their
    # value is the symbol name, not the scenario's number (the graph does not depend on it).
    symbolic_fields: Mapping[type, Mapping[str, str]] = dataclasses.field(default_factory=dict)
    discover: bool = False
    roots: frozenset[str] = frozenset()
    top_calls: list[str] = dataclasses.field(default_factory=list)
    wrapped: int = 0
    hoisted: int = 0  # closed-over tracers turned into composite operands (closure conversion)
    excluded: dict[str, int] = dataclasses.field(default_factory=dict)
    failures: list[dict[str, str]] = dataclasses.field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "wrapped": self.wrapped,
            "hoisted": self.hoisted,
            "excluded": dict(sorted(self.excluded.items())),
            "failures": self.failures,
        }


_SESSION: contextvars.ContextVar[Session | None] = contextvars.ContextVar("cc_export_session", default=None)


@contextlib.contextmanager
def session(exclude=(), *, discover: bool = False, roots=(), symbolic_fields=None):
    """Enable interception for the duration of one export (or one discovery trace)."""
    s = Session(
        exclude=frozenset(exclude),
        discover=discover,
        roots=frozenset(roots),
        symbolic_fields=symbolic_fields or {},
    )
    tok = _SESSION.set(s)
    try:
        yield s
    finally:
        _SESSION.reset(tok)


def current() -> Session | None:
    return _SESSION.get()


def is_array(x: Any) -> bool:
    return isinstance(x, (jax.Array, jax.core.Tracer))


def cc_composite(decomposition: Callable, name: str, version: int = 0) -> Callable:
    """`lax.composite` with closure conversion.

    Same construction as `jax/_src/lax/lax.py` `composite` (jax 0.12.0): trace the decomposition to a jaxpr,
    absorb its constants into leading invars, bind `composite_p` with the constants as leading operands.
    `lax.composite` refuses constants that are tracers (values of the enclosing trace the decomposition
    closes over); binding them as operands is exactly closure conversion (as `jax.closure_convert` does,
    without a second trace), so here they become operands too. That covers modules that read arrays from a
    closure, a global context or the enclosing module, and Haiku parameters read through the frame (with
    every custom getter of the port applied, inside the decomposition).
    """

    def bind(*args, **kwargs):
        dbg = api_util.debug_info("composite", decomposition, args, {})
        flat_args, in_tree = tree_util.tree_flatten(args)
        in_avals = tuple(core.typeof(x) for x in flat_args)
        if any(isinstance(v, core.Tracer) for v in jax.tree.leaves(kwargs)):
            raise TypeError(f"composite {name}: a tracer in the static attributes")
        fun = functools.partial(decomposition, **kwargs)

        def flat_fun(*xs):
            return fun(*tree_util.tree_unflatten(in_tree, xs))

        closed, out_avals = pe.trace_to_jaxpr(flat_fun, ft.flatten_args(*in_avals), dbg)
        # Concrete constants (e.g. jax.Arrays a port computed at trace time) go in as numpy values, which are
        # embedded in the module like any literal; as jax.Arrays, jit would lower them as extra `main`
        # arguments that `jax.export` does not expect.
        consts = [c if isinstance(c, core.Tracer) else np.asarray(c) for c in closed.consts]
        s = _SESSION.get()
        if s is not None:
            s.hoisted += sum(isinstance(c, core.Tracer) for c in consts)
        closed = pe.convert_constvars_jaxpr(closed)
        attributes = []
        for k, v in kwargs.items():
            leaves, treedef = tree_util.tree_flatten(v)
            leaves = tuple(HashableArray(x) if isinstance(x, np.ndarray) else x for x in leaves)
            attributes.append((k, leaves, treedef))
        out = composite_p.bind(
            *core.auto_insert_reshard(*consts, *flat_args),
            name=name,
            attributes=tuple(attributes),
            version=version,
            call_jaxpr=closed,
        )
        return tree_util.tree_unflatten(out_avals.tree, out)

    return bind


def _composite_vmap(axis_data, args, dims, *, call_jaxpr, **params):
    """Batching rule for `composite_p` (none in jax 0.12): vmap of a composite is the composite of the vmapped
    decomposition, under the same name, attributes and version. Same construction as JAX's `remat_vmap`
    (`jax/_src/ad_checkpoint.py`). Ports vmap over modules (alphafold2's per-template stacks)."""
    batched, out_batched = batching.batch_jaxpr_axes(
        call_jaxpr, axis_data, dims, [batching.zero_if_mapped] * len(call_jaxpr.outvars)
    )
    consts = batched.consts
    jaxpr = pe.convert_constvars_jaxpr(batched) if consts else batched
    outs = composite_p.bind(*consts, *args, call_jaxpr=jaxpr, **params)
    return outs, [0 if b else None for b in out_batched]


batching.fancy_primitive_batchers.setdefault(composite_p, _composite_vmap)


def _composite_dce(used_outputs, live_ins, eqn):
    """DCE rule for `composite_p` (none in jax 0.12, so composites were opaque to dead-code elimination:
    every input stayed live and dead code inside a component stayed in the module). Same construction as
    JAX's `remat_dce` (`jax/_src/ad_checkpoint.py`): DCE the decomposition for the used outputs, drop the
    unused inputs; a composite with no used output (and no effects) disappears."""
    if not any(used_outputs) and not pe.has_effects(eqn, live_ins):
        return [False] * len(eqn.invars), None
    jaxpr = eqn.params["call_jaxpr"]
    new_jaxpr, used_inputs = pe.dce_jaxpr(jaxpr, used_outputs, live_inputs=live_ins)
    if new_jaxpr is jaxpr and all(used_inputs) and all(used_outputs):
        return used_inputs, eqn
    new_invars = [v for v, u in zip(eqn.invars, used_inputs) if u]
    new_eqn = pe.new_jaxpr_eqn(
        new_invars,
        [v for v, u in zip(eqn.outvars, used_outputs) if u],
        eqn.primitive,
        dict(eqn.params, call_jaxpr=new_jaxpr),
        core.eqn_effects(new_jaxpr, new_invars),
        eqn.source_info,
        eqn.ctx,
    )
    return used_inputs, new_eqn


pe.dce_rules.setdefault(composite_p, _composite_dce)


def wrap(
    name: str,
    cls_name: str,
    path: str,
    attrs: Mapping[str, Any],
    extra: Any,
    call: Callable[..., Any],
    args: tuple,
    kwargs: Mapping[str, Any],
    fallback: Callable[[], Any],
) -> Any:
    """Run `call(extra, *args, **kwargs)` as a composite named `name`.

    `extra` is a pytree of arrays passed as operands ahead of the call's own array arguments (the module's
    variables). If wrapping fails, the failure is recorded and `fallback()` runs the call unwrapped, so the
    export still completes; `verify` reports every failure.
    """
    s = _SESSION.get()
    if s is None or s.discover or path in s.roots:
        return fallback()
    if cls_name in s.exclude:
        s.excluded[cls_name] = s.excluded.get(cls_name, 0) + 1
        return fallback()

    leaves, tree = jax.tree.flatten((tuple(args), dict(kwargs)))
    is_op = [is_array(x) for x in leaves]
    static = [None if o else x for x, o in zip(leaves, is_op)]
    ops = [x for x, o in zip(leaves, is_op) if o]

    def decomposition(extra_in, *ops_in, **_attrs):
        it = iter(ops_in)
        full = [next(it) if o else st for o, st in zip(is_op, static)]
        a, k = jax.tree.unflatten(tree, full)
        return call(extra_in, *a, **k)

    attributes = dict(attrs)
    attributes[PATH_ATTR] = path
    try:
        out = cc_composite(decomposition, name)(extra, *ops, **attributes)
    except Exception as e:  # noqa: BLE001 - every failure is recorded and surfaces in verify
        if os.environ.get("CC_EXPORT_STRICT"):
            raise
        s.failures.append({"name": name, "path": path, "error": _short(e)})
        return fallback()
    s.wrapped += 1
    return out


def _short(e: BaseException) -> str:
    msg = f"{type(e).__name__}: {e}"
    msg = re.sub(r"0x[0-9a-f]+", "0x?", msg)
    return msg[:500]


# --- static attributes ------------------------------------------------------------------------------


def qualname(f: Any) -> str:
    if isinstance(f, functools.partial):
        return f"partial({qualname(f.func)})"
    mod = getattr(f, "__module__", None) or ""
    name = getattr(f, "__qualname__", None) or getattr(f, "__name__", None) or type(f).__name__
    return f"{mod}.{name}" if mod else name


def _dtype(v: Any) -> str | None:
    if isinstance(v, np.dtype):
        return str(v)
    if isinstance(v, type) or type(v).__name__ == "_ScalarMeta":
        try:
            return str(np.dtype(v))
        except TypeError:
            return None
    return None


def attr_value(v: Any, depth: int = 0) -> Any:
    """A deterministic value `mlir.ir_attribute` accepts, or `None` (dropped)."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        i = int(v)
        return i if _I64[0] <= i <= _I64[1] else str(i)
    if isinstance(v, (float, np.floating)):
        return float(v)
    if isinstance(v, str):
        return v
    if isinstance(v, enum.Enum):
        return f"{type(v).__name__}.{v.name}"
    d = _dtype(v)
    if d is not None:
        return d
    if isinstance(v, jax.core.Tracer):
        return "tracer"
    if isinstance(v, (np.ndarray, jax.Array)):
        if v.size <= _SMALL_ARRAY and v.ndim <= 1:
            return tuple(attr_value(x, depth + 1) for x in np.asarray(v).reshape(-1).tolist())
        return f"array<{'x'.join(map(str, v.shape))}x{v.dtype}>"
    if depth >= _MAX_DEPTH:
        return type(v).__name__
    if isinstance(v, (tuple, list)):
        return tuple(_none_str(attr_value(x, depth + 1)) for x in v)
    if isinstance(v, Mapping):
        out = {}
        for k in sorted(v, key=str):
            x = attr_value(v[k], depth + 1)
            if x is not None:
                out[str(k)] = x
        return out
    if _is_module(v):
        return type(v).__name__
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        s = _SESSION.get()
        syms = s.symbolic_fields.get(type(v), {}) if s is not None else {}
        return attr_value(
            {
                f.name: syms.get(f.name, getattr(v, f.name, None))
                for f in dataclasses.fields(v)
                if f.name not in ("parent", "name")
            },
            depth + 1,
        )
    if callable(v):
        return qualname(v)
    return type(v).__name__


def _none_str(x: Any) -> Any:
    return "None" if x is None else x


def _is_module(v: Any) -> bool:
    for base in type(v).__mro__:
        mod = base.__module__ or ""
        if base.__name__ in ("Module", "Layer") and (
            mod.startswith("flax.") or mod.startswith("haiku.") or mod.startswith("keras.")
        ):
            return True
    return False


def attributes(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Composite attributes from a module's configuration fields (sorted, `None` dropped)."""
    out = {}
    for k in sorted(fields):
        x = attr_value(fields[k])
        if x is not None:
            out[k] = x
    return out
