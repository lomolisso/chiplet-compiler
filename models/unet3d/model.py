"""MLPerf 3D U-Net (KiTS19), TensorRT INT8 (MLPerf NVIDIA) capture spec (network in net.py)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, Sym, as_spec
from models.unet3d.net import Config, Unet3D

# The reference hard-codes filters [32, 64, 128, 256, 320], so "reduced" keeps
# every width (and every block) and only shrinks the ROI: 64^3 still leaves a
# 2^3 bottleneck for the instance norms.
CONFIGS = {"full": Config(), "reduced": Config(grid=2)}


def input_spec(cfg: Config):
    # MLPerf NVIDIA builds the engine with an INT8 LINEAR input (preprocessed, quantized ROI).
    return (jax.ShapeDtypeStruct((cfg.batch, cfg.roi, cfg.roi, cfg.roi, cfg.in_channels),
                                 jnp.int8),)


def init_variables(cfg: Config, rng):
    return Unet3D(cfg).init(rng, jnp.zeros(input_spec(cfg)[0].shape, jnp.int8))


def example_inputs(cfg: Config, rng):
    x = jax.random.randint(rng, input_spec(cfg)[0].shape, -127, 128, jnp.int32).astype(jnp.int8)
    return {"forward": (x,)}


def intermediates(cfg: Config, variables, x) -> dict:
    _, state = jax.jit(lambda v, x: Unet3D(cfg).apply(
        v, x, capture_intermediates=True, mutable=["intermediates"]))(variables, x)
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        flat["/".join(k for k in keys if k != "__call__")] = v[0]
    return flat


def build(cfg: Config) -> ModelSpec:
    module = Unet3D(cfg)

    def forward(variables, x):
        return module.apply(variables, x)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    s = cfg.stride
    return ModelSpec(
        name="unet3d",
        variant=(f"MLPerf 3D U-Net KiTS19, one {cfg.roi}^3 sliding-window ROI, batch {cfg.batch}, "
                 "TensorRT INT8 (MLPerf NVIDIA): int8 convs, deconvs as 1x1x1 conv + pixel "
                 "shuffle, int8 InstanceNorm+ReLU, int8 input, fp16 logits"),
        phases=[Phase("forward", forward, input_spec(cfg), trip_count=1,
                      note=("one network call on one 128^3 ROI; the host-side sliding window "
                            "calls it prod_i((D_i - 128) // 64 + 1) times per case (overlap 0.5), "
                            "e.g. 3*4*4 = 48 for a 256x320x320 preprocessed volume; MLPerf does "
                            "not document a typical count, so trip_count is recorded as 1"),
                      # the ROI is a whole number of stride-32 cells per axis (Config.grid)
                      shapes=(f"B, {s}*D, {s}*H, {s}*W, _",))],
        param_specs=as_spec(variables),
        symbols={"B": Sym("batch", 3, "ROIs per call"),
                 "D": Sym("grid", 3, f"ROI depth / {s} (stride-{s} cells)"),
                 "H": Sym("grid", 3, f"ROI height / {s} (stride-{s} cells)"),
                 "W": Sym("grid", 3, f"ROI width / {s} (stride-{s} cells)")},
        excluded=[
            "KiTS19 preprocessing: resample to 1.6x1.2x1.2 mm, clip [-79, 304], "
            "normalize (x-101)/76.9, pad to >=128 and to a multiple of 64 (preprocess.py), "
            "INT8 quantization with the calibrated input scale",
            "sliding-window driver over the case volume (128^3 ROI, overlap 0.5; base_SUT.py, "
            "runs on the host)",
            "Gaussian importance blending: logits * cbrt(3D gaussian, std 0.125*128) "
            "accumulated with a norm map, then divided (inference_utils.py)",
            "argmax over the 3 classes (postprocessing)",
        ],
    )
