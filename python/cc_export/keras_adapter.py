"""Keras 3 (JAX backend): every layer call on arrays becomes a composite.

Mechanism:
- Keras has no method-interception API, so `keras.layers.Layer.__call__` is patched for the duration of an
  export (restored on exit). Model code is untouched.
- Under `Layer.stateless_call` a layer reads its variables through the active `StatelessScope`'s
  `state_mapping`; inside a decomposition those values would be tracers of the outer trace. So the layer's
  variables (`layer.variables`, which include its sub-layers') are passed as explicit operands at their
  current values, and the decomposition calls the layer under a nested `StatelessScope` that maps each
  variable to its operand (`initialize_variables=False`: nothing is ever materialized).
- Only calls on JAX arrays are wrapped: building a model on `KerasTensor`s (shape inference) passes through.
- Root: `Model.stateless_call` runs the top model's `call` directly (not `__call__`), so the model itself is
  never wrapped; its layers are.
"""

from __future__ import annotations

import contextlib
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax  # noqa: E402

from . import composite  # noqa: E402

_SKIP_CONFIG = frozenset({"name", "trainable"})


def _fields(layer) -> dict:
    try:
        cfg = dict(layer.get_config())
    except Exception:  # noqa: BLE001 - custom layers without get_config: no attributes
        cfg = {}
    return {k: v for k, v in cfg.items() if k not in _SKIP_CONFIG}


@contextlib.contextmanager
def intercept():
    import keras

    layer_cls = keras.layers.Layer
    orig = layer_cls.__call__

    def patched(self, *args, **kwargs):
        if composite.current() is None or not any(
            composite.is_array(x) for x in jax.tree.leaves((args, kwargs))
        ):
            return orig(self, *args, **kwargs)
        variables = list(self.variables)
        values = [v.value for v in variables]
        cls = type(self).__name__

        def call(vals, *a, **k):
            mapping = list(zip(variables, vals))
            with keras.StatelessScope(state_mapping=mapping, initialize_variables=False):
                return orig(self, *a, **k)

        return composite.wrap(
            f"{composite.NAMESPACE}.keras.{cls}",
            cls,
            getattr(self, "path", None) or self.name,
            composite.attributes(_fields(self)),
            values,
            call,
            args,
            kwargs,
            lambda: orig(self, *args, **kwargs),
        )

    layer_cls.__call__ = patched
    try:
        yield
    finally:
        layer_cls.__call__ = orig
