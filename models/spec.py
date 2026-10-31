"""Port interface types shared by every `models/<key>/model.py`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax


@dataclass
class Phase:
    name: str
    fn: Callable[..., Any]                 # fn(params, *inputs)
    inputs: tuple                          # pytree of ShapeDtypeStruct / arrays
    trip_count: int = 1                    # times this phase runs per model invocation
    note: str = ""
    # Top-level param keys this phase consumes (None = all). For models whose
    # phases are disjoint components (e.g. SDXL text encoders / UNet / VAE);
    # fn then receives {k: params[k] for k in param_keys}.
    param_keys: tuple[str, ...] | None = None
    # Trip count as an expression in the model's symbols (e.g. "T", "ceil(T/2)") when it
    # depends on an input size; `trip_count` stays its value at the scenario.
    trip_count_expr: str | None = None
    # Symbolic shapes of `inputs` for the export: a pytree prefix of `inputs` whose
    # leaves are JAX shape specs over the model's `symbols`, "_" keeping an (architecture)
    # dim concrete, e.g. ("B, S", "B, S + 128, _, _"). None = every input dim is concrete.
    shapes: Any = None
    # Name of a phase this one differs from only in input sizes (e.g. a batch-128 variant):
    # both must export to the same graph.
    size_variant_of: str | None = None


@dataclass(frozen=True)
class Sym:
    """An input-dependent size: one dimension variable of the symbolic export.

    `field` is the config field holding its value at the scenario (dotted for nested
    configs); `alt` is another valid value. The exporter re-exports with every symbol's
    field set to `alt` and requires the identical graph, so nothing may branch on it.
    """
    field: str
    alt: int
    doc: str = ""


@dataclass
class ModelSpec:
    name: str
    variant: str
    phases: list[Phase]
    param_specs: Any                       # pytree of ShapeDtypeStruct (full size)
    excluded: list[str] = field(default_factory=list)   # host-side pre/post-processing
    # Input-dependent sizes: symbol name -> Sym; constraints in JAX's syntax
    # ("S >= 16", "mod(H, 32) == 0"). Every symbol must appear in some phase's `shapes`.
    symbols: dict[str, Sym] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()


def as_spec(tree):
    return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree)
