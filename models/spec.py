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


@dataclass
class ModelSpec:
    name: str
    variant: str
    phases: list[Phase]
    param_specs: Any                       # pytree of ShapeDtypeStruct (full size)
    excluded: list[str] = field(default_factory=list)   # host-side pre/post-processing


def as_spec(tree):
    return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree)
