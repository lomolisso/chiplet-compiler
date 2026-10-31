"""Qwen3-VL-235B-A22B-Instruct-FP8 (official Qwen FP8 checkpoint; MLPerf Inference v6.1
multimodal/qwen3-vl model) capture spec.

Network: net.py (recipe in its docstring: FP8 E4M3 128x128-block weights with dynamic 1x128
activation groups on the text attention projections and the experts; vision tower,
router, embedding, lm_head, norms, KV cache and attention bf16). Phases (one network invocation each):
  vision_encoder  pixel patches of ONE fixed-size image (448x448 -> 28x28 patches ->
                  196 merged tokens) -> image embeddings [196, 4096] + 3 DeepStack
                  feature maps [3, 196, 4096].                       trip_count 1
  prefill         1024 prompt tokens (fixed layout: text, <vision_start>, 196 image
                  tokens, <vision_end>, 5 text tokens) + host M-RoPE position ids +
                  each sequence's vision outputs, static KV cache (capacity 1174) ->
                  last-position logits + cache.                       trip_count 1
  decode_step     one token at a runtime cache slot + its runtime M-RoPE position
                  -> logits + cache.                                  trip_count 150
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from models.spec import ModelSpec, Phase, Sym, as_spec
from models.qwen3vl.net import (Config, TextConfig, Transformer, VisionConfig, VisionTower,
                                vision_interpolation_table, vision_position_ids)

# MLPerf v6.1 Q3VL (mlcommons/endpoints @381d13bb examples/08_Qwen3-VL-235B-A22B_Example):
# model_params.max_new_tokens = 150 (offline, server and interactive configs).
MAX_OUTPUT_TOKENS = 150

_REDUCED = Config(
    vision=VisionConfig(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=4,
                        out_hidden_size=128, num_position_embeddings=16,
                        deepstack_visual_indexes=(0, 2)),
    # Every FP8 GEMM K and N is a multiple of the 128 block: D 128, q 4x64 = 256,
    # k/v 2x64 = 128, experts 2F = 256 / F = 128.
    text=TextConfig(vocab_size=256, emb_dim=128, num_decoder_layers=3, num_query_heads=4,
                    num_kv_heads=2, head_dim=64, num_experts=8, num_experts_per_tok=2,
                    moe_mlp_dim=128, mrope_section=(12, 10, 10)),
    image_size=(24, 24), image_tokens=9, max_prefill_predict_length=24, text_suffix_len=3, max_target_length=32,
    image_token_id=250, vision_start_token_id=252, vision_end_token_id=253, dtype="bfloat16")

CONFIGS = {
    # Qwen/Qwen3-VL-235B-A22B-Instruct @710c1386 config.json (MLPerf-pinned revision).
    "full": Config(image_tokens=196),          # 448x448 -> 28x28 patches -> 14x14 merged
    # Every block type (ViT blocks with/without DeepStack tap, post-shuffle and
    # final mergers, interpolated pos-embed 4x4 -> 6x6, 2-D RoPE; MoE decoder layers
    # with and without DeepStack injection, GQA, qk-norm, interleaved M-RoPE).
    "reduced": _REDUCED,
}


# ----------------------------------------------------------------------------
# Host-side static layout
# ----------------------------------------------------------------------------

def token_type_ids(cfg: Config) -> np.ndarray:
    """[P] 0 = text, 1 = image placeholder (HF `mm_token_type_ids`)."""
    tt = np.zeros(cfg.max_prefill_predict_length, np.int32)
    tt[cfg.image_start:cfg.image_start + cfg.num_image_tokens] = 1
    return tt


def mrope_position_ids(cfg: Config) -> tuple[np.ndarray, int]:
    """[3, 1, P] M-RoPE ids for the fixed layout and rope_delta (HF get_rope_index):
    text tokens advance all three axes together; the image span gets
    (t, h, w) = start + (0, row, col) on the merged grid, and text after it
    resumes at start + max(h, w)."""
    m = cfg.vision.spatial_merge_size
    _, gh, gw = cfg.patch_grid
    gh, gw = gh // m, gw // m
    s, n = cfg.image_start, cfg.num_image_tokens
    pre = np.broadcast_to(np.arange(s), (3, s))
    hh, ww = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    img = np.stack([np.zeros(n, np.int64), hh.reshape(-1), ww.reshape(-1)]) + s
    rest = cfg.max_prefill_predict_length - s - n
    post = np.broadcast_to(np.arange(rest) + s + max(gh, gw), (3, rest))
    pos = np.concatenate([pre, img, post], axis=1).astype(np.int32)
    delta = int(pos.max()) + 1 - cfg.max_prefill_predict_length
    return pos[:, None, :], delta


def vision_host_inputs(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The vision tower's host tables for the image size: patch (row, col) [N, 2] int32,
    position-embedding interpolation indices [N, 4] int32 and weights [N, 4] fp32."""
    idx, wts = vision_interpolation_table(cfg)
    return vision_position_ids(cfg), idx, wts


def decode_position_ids(cfg: Config, cache_index: int) -> np.ndarray:
    """[3, 1, 1]: text after the prompt -> slot + rope_delta on all three axes."""
    _, delta = mrope_position_ids(cfg)
    return np.full((3, cfg.batch, 1), cache_index + delta, np.int32)


def decode_position(cfg: Config) -> int:
    """Example runtime cache slot: a few steps into decoding (any value < S works)."""
    return cfg.max_prefill_predict_length + 3


# ----------------------------------------------------------------------------
# Specs / phase functions
# ----------------------------------------------------------------------------

def _d(cfg):
    return jnp.dtype(cfg.dtype)


def cache_spec(cfg: Config):
    t = cfg.text
    leaf = jax.ShapeDtypeStruct((cfg.batch, cfg.max_target_length, t.num_kv_heads, t.head_dim), _d(cfg))
    return {f"layers_{i}": {"key": leaf, "value": leaf} for i in range(t.num_decoder_layers)}


def input_specs(cfg: Config) -> dict:
    v, t, i32 = cfg.vision, cfg.text, jnp.int32
    patch_dim = v.in_channels * v.temporal_patch_size * v.patch_size ** 2
    P, n, D = cfg.max_prefill_predict_length, cfg.num_image_tokens, t.emb_dim
    N = cfg.num_patches
    return {
        "vision_encoder": (jax.ShapeDtypeStruct((N, patch_dim), jnp.float32),
                           jax.ShapeDtypeStruct((N, 2), i32),            # patch (row, col)
                           jax.ShapeDtypeStruct((N, 4), i32),            # interpolation taps
                           jax.ShapeDtypeStruct((N, 4), jnp.float32)),   # and their weights
        "prefill": (jax.ShapeDtypeStruct((cfg.batch, P), i32),
                    jax.ShapeDtypeStruct((3, cfg.batch, P), i32),
                    jax.ShapeDtypeStruct((cfg.batch, n, D), _d(cfg)),   # one image per sequence
                    jax.ShapeDtypeStruct((len(v.deepstack_visual_indexes), cfg.batch, n, D),
                                         _d(cfg)),
                    jax.ShapeDtypeStruct((), i32),                       # image_start
                    cache_spec(cfg)),
        "decode_step": (jax.ShapeDtypeStruct((cfg.batch, 1), i32),
                        jax.ShapeDtypeStruct((), i32),
                        jax.ShapeDtypeStruct((3, cfg.batch, 1), i32),
                        cache_spec(cfg)),
    }


def vision_fn(cfg: Config):
    def vision_encoder(params, pixel_patches, pos_ids, interp_idx, interp_wts):
        return VisionTower(cfg).apply({"params": params["vision"]}, pixel_patches, pos_ids,
                                      interp_idx, interp_wts)
    return vision_encoder


def prefill_fn(cfg: Config):
    def prefill(params, tokens, position_ids, image_embeds, deepstack, image_start, cache):
        return Transformer(cfg).apply({"params": params["text"]}, tokens, position_ids, cache,
                                      jnp.int32(0), image_embeds, deepstack, image_start)
    return prefill


def decode_fn(cfg: Config):
    def decode_step(params, token, cache_index, position_ids, cache):
        return Transformer(cfg).apply({"params": params["text"]}, token, position_ids, cache,
                                      cache_index)
    return decode_step


def init_variables(cfg: Config, rng):
    kv, kt = jax.random.split(rng)
    sp = input_specs(cfg)
    z = lambda s: jnp.zeros(s.shape, s.dtype)
    vis = VisionTower(cfg).init(kv, *jax.tree.map(z, sp["vision_encoder"]))
    tok, pos, img, ds, start, cache = jax.tree.map(z, sp["prefill"])
    txt = Transformer(cfg).init(kt, tok, pos, cache, jnp.int32(0), img, ds, start)
    return {"vision": vis["params"], "text": txt["params"]}


def example_tokens(cfg: Config, key) -> jnp.ndarray:
    """Random text ids (below every special id) in the fixed layout."""
    P, s, n = cfg.max_prefill_predict_length, cfg.image_start, cfg.num_image_tokens
    hi = min(cfg.image_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id)
    tok = np.array(jax.random.randint(key, (cfg.batch, P), 0, min(hi, cfg.text.vocab_size)))
    tok[:, s - 1] = cfg.vision_start_token_id
    tok[:, s:s + n] = cfg.image_token_id
    tok[:, s + n] = cfg.vision_end_token_id
    return jnp.asarray(tok, jnp.int32)


def example_inputs(cfg: Config, rng):
    ks = jax.random.split(rng, 6)
    sp = input_specs(cfg)
    pix = jax.random.normal(ks[0], sp["vision_encoder"][0].shape, jnp.float32)
    pos, _ = mrope_position_ids(cfg)
    img = jax.random.normal(ks[1], sp["prefill"][2].shape, _d(cfg))
    ds = jax.random.normal(ks[2], sp["prefill"][3].shape, _d(cfg))
    zero_cache = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), sp["prefill"][5])
    # Decode: every slot holds random data, so a wrong valid-mask changes the result.
    leaves, treedef = jax.tree.flatten(sp["decode_step"][3])
    keys = jax.random.split(ks[3], len(leaves))
    full_cache = jax.tree.unflatten(treedef, [jax.random.normal(k, s.shape, s.dtype)
                                              for k, s in zip(keys, leaves)])
    token = jax.random.randint(ks[4], (cfg.batch, 1), 0, 200, jnp.int32)
    ci = decode_position(cfg)
    return {"vision_encoder": (pix, *map(jnp.asarray, vision_host_inputs(cfg))),
            "prefill": (example_tokens(cfg, ks[5]), jnp.asarray(pos), img, ds,
                        jnp.int32(cfg.image_start), zero_cache),
            "decode_step": (token, jnp.int32(ci), jnp.asarray(decode_position_ids(cfg, ci)),
                            full_cache)}


def _flatten_intermediates(state) -> dict:
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        out = v[0]
        if isinstance(out, tuple):          # (output, kv_cache) / (embeds, deepstack) modules
            out = out[0]
        if isinstance(out, (list, tuple)):  # VisionEncoder (x, [deep feats]) handled above
            continue
        flat["/".join(k for k in keys if k != "__call__")] = out
    return flat


def intermediates(cfg: Config, variables, *inputs, phase: str | None = None) -> dict:
    kw = dict(capture_intermediates=True, mutable=["intermediates"])
    if phase is None:
        phase = ("prefill" if len(inputs) == 6 else
                 "vision_encoder" if inputs[0].dtype == jnp.float32 else "decode_step")
    if phase == "vision_encoder":
        _, st = VisionTower(cfg).apply({"params": variables["vision"]}, *inputs, **kw)
    elif phase == "prefill":
        tok, pos, img, ds, start, cache = inputs
        _, st = Transformer(cfg).apply({"params": variables["text"]}, tok, pos, cache,
                                       jnp.int32(0), img, ds, start, **kw)
    else:
        tok, ci, pos, cache = inputs
        _, st = Transformer(cfg).apply({"params": variables["text"]}, tok, pos, cache, ci, **kw)
    return _flatten_intermediates(st)


def config_name(cfg: Config) -> str:
    return next((k for k, v in CONFIGS.items() if v == cfg), "custom")


def build(cfg: Config) -> ModelSpec:
    specs = input_specs(cfg)
    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    name = {"full": "Qwen3-VL-235B-A22B-Instruct-FP8"}.get(config_name(cfg), "Qwen3-VL-MoE reduced")
    P, S, n = cfg.max_prefill_predict_length, cfg.max_target_length, cfg.num_image_tokens
    t = cfg.text
    return ModelSpec(
        name="qwen3vl",
        variant=(f"{name} (MLPerf v6.1 multimodal/qwen3-vl), {cfg.dtype}, batch {cfg.batch}, "
                 f"one {cfg.image_size[0]}x{cfg.image_size[1]} image ({n} visual tokens), "
                 f"prefill {P}, static KV cache {S}; MoE {t.num_experts} experts top-"
                 f"{t.num_experts_per_tok}, static dispatch capacity = T; Qwen official FP8: "
                 "E4M3 128x128-block weights, dynamic 1x128 activations on text q/k/v/o + "
                 "experts; vision, router, embedding, lm_head, KV cache, attention bf16"),
        phases=[
            Phase("vision_encoder", vision_fn(cfg), specs["vision_encoder"], trip_count=1,
                  param_keys=("vision",), shapes=("4*I, _",) * 4,
                  note=f"ViT on one fixed {cfg.image_size[0]}x{cfg.image_size[1]} image: "
                       f"{cfg.num_patches} patches -> {n} merged tokens + "
                       f"{len(cfg.vision.deepstack_visual_indexes)} DeepStack maps"),
            Phase("prefill", prefill_fn(cfg), specs["prefill"], trip_count=1,
                  param_keys=("text",),
                  shapes=("B, P", "_, B, P", "B, I, _", "_, B, I, _", "", "B, S, _, _"),
                  note=f"{P} prompt tokens; image tokens at [image_start, image_start + {n}) "
                       f"(host input; {cfg.image_start} in the example layout); DeepStack into layers 0.."
                       f"{len(cfg.vision.deepstack_visual_indexes) - 1}; writes cache slots "
                       f"0..{P - 1}; last-position logits"),
            Phase("decode_step", decode_fn(cfg), specs["decode_step"],
                  trip_count=MAX_OUTPUT_TOKENS, param_keys=("text",),
                  shapes=("B, _", "", "_, B, _", "B, S, _, _"),
                  note="one token at a runtime int32 cache slot with runtime M-RoPE ids "
                       f"(slot + rope_delta); attends over all {S} slots with a causal/valid "
                       f"mask. Trip count = MLPerf max_new_tokens {MAX_OUTPUT_TOKENS} (the first "
                       "output token comes from prefill, so at most 149 decode calls are "
                       "strictly needed)."),
        ],
        param_specs=as_spec(variables),
        symbols={"B": Sym("batch", 3, "sequences per call"),
                 "P": Sym("max_prefill_predict_length", 20, "prompt tokens (prefill)"),
                 "S": Sym("max_target_length", 40, "static KV-cache capacity"),
                 "I": Sym("image_tokens", 12, "merged visual tokens of one image "
                                              "(spatial_merge_size**2 = 4 patches each)")},
        # the prompt holds <vision_start>, the image span and <vision_end>
        constraints=("S >= P", "P >= I + 2"),
        excluded=["image preprocessing (resize to the fixed size, normalize, patchify: "
                  "Qwen2VLImageProcessor) and the chat-template / tokenization / detokenization",
                  "host-side M-RoPE position-id computation (get_rope_index) -- passed as inputs",
                  "vision patch position ids, position-embedding interpolation table "
                  "(get_vision_position_ids / get_vision_interpolation_indices_and_weights) and "
                  "the image-span offset image_start -- passed as inputs",
                  "sampling (MLPerf: temperature 0.7, top-p 0.8/1, top-k 20) and the outer "
                  "decode loop",
                  "vLLM scheduler / paged attention / prefix caching (disabled by MLPerf rules)"],
    )

