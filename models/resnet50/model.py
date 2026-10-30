"""ResNet-50 v1.5 INT8 (MLPerf NVIDIA TensorRT recipe) capture spec; network in net.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, as_spec
from models.resnet50.net import Config, ResNet50

CONFIGS = {"full": Config(), "reduced": Config(batch=2)}


def input_spec(cfg: Config):
    # MLPerf NVIDIA feeds preprocessed images already quantized to INT8.
    return (jax.ShapeDtypeStruct((cfg.batch, cfg.image, cfg.image, 3), jnp.int8),)


def init_variables(cfg: Config, rng):
    return ResNet50(cfg).init(rng, jnp.zeros(input_spec(cfg)[0].shape, jnp.int8))


def example_inputs(cfg: Config, rng):
    x = jax.random.randint(rng, input_spec(cfg)[0].shape, -127, 128, jnp.int32).astype(jnp.int8)
    return {"forward": (x,)}


def intermediates(cfg: Config, variables, x) -> dict:
    _, state = ResNet50(cfg).apply(variables, x, capture_intermediates=True,
                                   mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        flat["/".join(k for k in keys if k != "__call__")] = v[0]
    return flat


def build(cfg: Config) -> ModelSpec:
    module = ResNet50(cfg)

    def forward(variables, x):
        return module.apply(variables, x)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    return ModelSpec(
        name="resnet50",
        variant=(f"ResNet-50 v1.5, {cfg.image}^2, batch {cfg.batch}, INT8 W8A8 (TensorRT, MLPerf "
                 "NVIDIA): per-channel weights, per-tensor static activations, BN folded"),
        phases=[Phase("forward", forward, input_spec(cfg))],
        param_specs=as_spec(variables),
        excluded=["image decode / resize / normalization / INT8 quantization (preprocessing)",
                  "argmax / top-k (postprocessing; softmax removed as in MLPerf NVIDIA)"],
    )
