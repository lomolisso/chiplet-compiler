"""RT-DETRv2-R50 (COCO detection, 640x640), INT8 W8A8 (TensorRT INT8 platform fallback) capture
spec; network and the quantization recipe in net.py, recipe table in recipe.py.

The phase input is the fp32 preprocessed image (the first conv quantizes it); parameters are
int8 weights with per-channel fp32 scales, folded-BN fp32 biases, per-tensor activation
scales, fp32 LayerNorm affines and the fp32 `n_points_scale` constants."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, as_spec
from models.rtdetr_v2.net import Config, RTDetrV2ForObjectDetection

# full: PekingU/rtdetr_v2_r50vd config.json (== rtdetrv2_r50vd_6x_coco.yml), 640x640, batch 1.
# reduced: same block types at small widths. 448x448 keeps the invalid-anchor border
# of level 0 (56x56: centre 0.5/56 < 0.01), so the valid_mask / finfo.max path runs.
CONFIGS = {
    "full": Config(),
    "reduced": Config(image=448, embedding_size=32, hidden_sizes=(64, 128, 256, 512),
                      depths=(2, 1, 2, 2), encoder_hidden_dim=64, encoder_layers=2,
                      encoder_ffn_dim=128, encoder_attention_heads=4, d_model=64,
                      num_queries=60, decoder_layers=3, decoder_ffn_dim=128,
                      decoder_attention_heads=4, num_labels=16),
}


def input_spec(cfg: Config):
    return (jax.ShapeDtypeStruct((cfg.batch, cfg.image, cfg.image, cfg.num_channels),
                                 jnp.float32),)


def init_variables(cfg: Config, rng):
    return RTDetrV2ForObjectDetection(cfg).init(rng, jnp.zeros(input_spec(cfg)[0].shape))


def example_inputs(cfg: Config, rng):
    # RTDetrImageProcessor output: resized RGB rescaled by 1/255 (do_normalize=False).
    return {"forward": (jax.random.uniform(rng, input_spec(cfg)[0].shape),)}


def intermediates(cfg: Config, variables, x) -> dict:
    """Module outputs; for a module called several times (decoder.query_pos_head) the LAST
    call is kept, like a torch forward hook that overwrites its entry."""
    _, state = RTDetrV2ForObjectDetection(cfg).apply(variables, x, capture_intermediates=True,
                                                     mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        flat["/".join(k for k in keys if k != "__call__")] = v[-1]
    return flat


def build(cfg: Config) -> ModelSpec:
    module = RTDetrV2ForObjectDetection(cfg)

    def forward(variables, pixel_values):
        return module.apply(variables, pixel_values)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    S = sum(h * w for h, w in cfg.spatial_shapes())
    return ModelSpec(
        name="rtdetr_v2",
        variant=(f"RT-DETRv2-R50 (PekingU/rtdetr_v2_r50vd), {cfg.image}^2, batch {cfg.batch}, "
                 f"{cfg.num_queries} queries, {cfg.decoder_layers} decoder layers, eval; INT8 W8A8 "
                 "(TensorRT INT8 fallback: per-channel int8 weights, BN folded, per-tensor "
                 "static int8 activations; attention core, softmax, deformable sampling, "
                 "LayerNorm, top-k fp16)"),
        phases=[Phase("forward", forward, input_spec(cfg),
                      note=(f"backbone + hybrid encoder + query selection (top-{cfg.num_queries} "
                            f"of {S} tokens) + {cfg.decoder_layers}-layer deformable decoder -> "
                            "logits (B, Q, 80) and pred_boxes (B, Q, 4) cxcywh in [0, 1], plus "
                            "the other eval-mode output tensors (net.OUTPUT_NAMES)"))],
        param_specs=as_spec(variables),
        excluded=["image decode, resize to 640x640 (bilinear), rescale 1/255 "
                  "(RTDetrImageProcessor; do_normalize=False) (preprocessing)",
                  "post_process_object_detection: sigmoid, top-300 over Q x 80 scores, "
                  "cxcywh -> xyxy, scale to the image size, threshold (postprocessing)",
                  "contrastive denoising queries and denoising_class_embed (training only)",
                  "INT8 calibration (activation scales are parameters of the graph)"],
    )
