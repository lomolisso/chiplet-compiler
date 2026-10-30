"""YOLO11-L (MLPerf Inference edge object detection), Ultralytics TensorRT INT8 export; network in net.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, as_spec
from models.yolo11.net import Config, DetectionModel

# full: yolo11l.yaml scale "l" = [depth 1.00, width 1.00, max_channels 512], 640x640.
# reduced: same block types (C3k2 with C3k, SPPF, multi-head C2PSA, Detect + DFL),
# depth 0.5 (one block per C3k2/C2PSA), width 0.25, 160x160 input.
CONFIGS = {
    "full": Config(),
    "reduced": Config(image=160, depth=0.5, width=0.25, max_channels=1024),
}


def input_spec(cfg: Config):
    return (jax.ShapeDtypeStruct((cfg.batch, cfg.image, cfg.image, 3), jnp.float32),)


def init_variables(cfg: Config, rng):
    return DetectionModel(cfg).init(rng, jnp.zeros(input_spec(cfg)[0].shape))


def example_inputs(cfg: Config, rng):
    # Letterboxed RGB scaled to [0, 1], like Ultralytics' preprocess output.
    return {"forward": (jax.random.uniform(rng, input_spec(cfg)[0].shape),)}


def intermediates(cfg: Config, variables, x) -> dict:
    _, state = DetectionModel(cfg).apply(variables, x, capture_intermediates=True,
                                         mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        flat["/".join(k for k in keys if k != "__call__")] = v[0]
    return flat


def build(cfg: Config) -> ModelSpec:
    module = DetectionModel(cfg)

    def forward(variables, x):
        return module.apply(variables, x)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    return ModelSpec(
        name="yolo11",
        variant=(f"YOLO11-{cfg.scale.upper()} (depth {cfg.depth}, width {cfg.width}, "
                 f"max_ch {cfg.max_channels}), {cfg.image}^2, batch {cfg.batch}, TensorRT INT8 "
                 "(Ultralytics export): BN folded, int8 W8A8 body, fp16 final head convs + DFL, "
                 "fp32 sigmoid"),
        phases=[Phase("forward", forward, input_spec(cfg),
                      note="raw eval-mode Detect output (B, 4+nc, anchors): xywh boxes "
                           "(pixels) + sigmoid class scores, before NMS")],
        param_specs=as_spec(variables),
        excluded=["image decode, letterbox resize/pad to 640x640, BGR->RGB, /255 (preprocessing)",
                  "confidence filtering + NMS (non_max_suppression), box rescale to the "
                  "original image (postprocessing)"],
    )
