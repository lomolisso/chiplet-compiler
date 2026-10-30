"""MobileNetV4 capture spec, shared by models/mobilenetv4_conv_m and mobilenetv4_hybrid_m.

The network lives in net.py; everything here is parameterized by `net.Config`,
so each variant directory only defines its CONFIGS.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, as_spec
from models.mobilenetv4.net import Config, MobileNetV4


def input_spec(cfg: Config):
    # TFLite full-integer model: uint8 image (mean / std folded into the input scale and
    # zero point), requantized to int8 by the first QUANTIZE.
    return (jax.ShapeDtypeStruct((cfg.batch, cfg.image, cfg.image, 3), jnp.uint8),)


def init_variables(cfg: Config, rng):
    return MobileNetV4(cfg).init(rng, jnp.zeros(input_spec(cfg)[0].shape, jnp.uint8))


def example_inputs(cfg: Config, rng):
    x = jax.random.randint(rng, input_spec(cfg)[0].shape, 0, 256, jnp.int32).astype(jnp.uint8)
    return {"forward": (x,)}


def intermediates(cfg: Config, variables, x) -> dict:
    _, state = MobileNetV4(cfg).apply(variables, x, capture_intermediates=True,
                                      mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        flat["/".join(k for k in keys if k != "__call__")] = v[0]
    return flat


def build(cfg: Config) -> ModelSpec:
    module = MobileNetV4(cfg)

    def forward(variables, x):
        return module.apply(variables, x)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    return ModelSpec(
        name=cfg.name,
        variant=f"{cfg.timm_name or 'reduced ' + cfg.name} (timm 1.0.30), {cfg.image}^2, "
                f"batch {cfg.batch}, TFLite full-integer INT8 (per-channel int8 weights, "
                f"asymmetric int8 activations, int8 softmax, uint8 I/O)",
        phases=[Phase("forward", forward, input_spec(cfg))],
        param_specs=as_spec(variables),
        excluded=["image decode / resize / crop (preprocessing; mean-std normalization is "
                  "folded into the uint8 input's scale / zero point)",
                  "argmax / top-k over the uint8 probabilities (postprocessing; the softmax "
                  "is in the graph, as in the TFLite model)"],
    )
