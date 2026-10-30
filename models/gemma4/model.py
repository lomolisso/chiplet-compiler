"""Gemma 4 26B-A4B QAT Q4_0 (official GGUF, llama.cpp arithmetic) capture spec; network in
net.py / the gemma library (quantization written into models/gemma4/upstream)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from models.spec import ModelSpec, Phase, as_spec
from models.gemma4 import net


@dataclass(frozen=True)
class Config:
    name: str
    tcfg: Any = field(repr=False)          # gemma _config.TransformerConfig
    dtype: Any = jnp.bfloat16              # text activations (vision stays f32, as in the library)
    prompt_len: int = 2048                 # prefill tokens
    decode_steps: int = 128                # decode_step trip count
    image_grid_hw: tuple[int, int] = (42, 60)   # patches; 42*60 = 2520 = 280 tokens * 3^2

    @property
    def cache_len(self) -> int:            # static decode cache: prompt + generated
        return self.prompt_len + self.decode_steps

    @property
    def reduced(self) -> bool:
        return self.name == "reduced"


CONFIGS = {
    "full": Config("full", net.FULL_CONFIG),
    "reduced": Config("reduced", net.reduced_config(), prompt_len=24,
                      decode_steps=8, image_grid_hw=(6, 9)),
}


def _models(cfg: Config):
    return net.text_model(cfg.tcfg, cfg.dtype), net.vision_tower(cfg.tcfg)


def _cache_spec(cfg: Config):
    return jax.eval_shape(lambda: cfg.tcfg.init_cache(batch_size=1, dtype=cfg.dtype,
                                                      cache_length=cfg.cache_len))


def input_specs(cfg: Config) -> dict:
    ve = cfg.tcfg.vision_encoder
    h, w = cfg.image_grid_hw
    i32 = jnp.int32
    return {
        "prefill": (jax.ShapeDtypeStruct((1, cfg.prompt_len), i32),),
        "decode_step": (jax.ShapeDtypeStruct((1, 1), i32), jax.ShapeDtypeStruct((1,), i32),
                        _cache_spec(cfg)),
        "vision_encoder": (jax.ShapeDtypeStruct((1, h * w, ve.patch_size ** 2 * 3), jnp.float32),
                           jax.ShapeDtypeStruct((1, h * w, 2), i32)),
    }


def image_positions(cfg: Config):
    h, w = cfg.image_grid_hw
    xy = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(1, h * w, 2)
    return jnp.asarray(xy, jnp.int32)          # (x, y) per patch, row-major (library patchify)


def init_variables(cfg: Config, rng):
    text, tower = _models(cfg)
    k1, k2 = jax.random.split(rng)
    ex = input_specs(cfg)
    tok = jnp.ones(ex["prefill"][0].shape, jnp.int32)
    with net.patched():
        tv = text.init(k1, tok)["params"]
        vi = net._transformer.PreprocessedVisionInput(
            patches=jnp.zeros(ex["vision_encoder"][0].shape, jnp.float32),
            positions_xy=image_positions(cfg),
            soft_token_counts=(cfg.tcfg.vision_encoder.num_mm_tokens_per_image,))
        vv = tower.init(k2, vi, method="_encode_vision")["params"]
    return {"text": tv, "vision": vv}


def example_inputs(cfg: Config, rng):
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    V = cfg.tcfg.num_embed
    tokens = jax.random.randint(k1, (1, cfg.prompt_len), 1, V, jnp.int32)   # no PAD (=0)
    # Decode: static cache holding a P-token context, next token at position P.
    P = cfg.prompt_len
    cache = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), _cache_spec(cfg))
    for i, (name, c) in enumerate(sorted(cache.items())):
        if cfg.reduced:     # random context K/V (any values: both sides get the same)
            kk, kv = jax.random.split(jax.random.fold_in(k2, i))
            valid = (jnp.arange(cfg.cache_len) < P)[None, :, None, None]
            c["k"] = jnp.where(valid, jax.random.normal(kk, c["k"].shape), 0).astype(c["k"].dtype)
            c["v"] = jnp.where(valid, jax.random.normal(kv, c["v"].shape), 0).astype(c["v"].dtype)
        c["positions"] = jnp.where(jnp.arange(cfg.cache_len) < P, jnp.arange(cfg.cache_len),
                                   0)[None].astype(jnp.int32)
        c["end_index"] = jnp.full((1,), P, jnp.int32)
    token = jax.random.randint(k3, (1, 1), 1, V, jnp.int32)
    position = jnp.full((1,), P, jnp.int32)
    ps = input_specs(cfg)["vision_encoder"][0].shape
    patches = jax.random.uniform(k4, ps, jnp.float32)          # pixels in [0, 1]
    return {"prefill": (tokens,), "decode_step": (token, position, cache),
            "vision_encoder": (patches, image_positions(cfg))}


def _phase_fns(cfg: Config):
    text, tower = _models(cfg)
    return {
        "prefill": lambda p, tokens: net.prefill(text, p["text"], tokens),
        "decode_step": lambda p, token, pos, cache: net.decode_step(text, p["text"], token, pos,
                                                                    cache),
        "vision_encoder": lambda p, patches, xy: net.vision_encode(tower, p["vision"], patches, xy),
    }


_CAPTURE = ("__call__", "encode", "decode", "encode_vision")


def intermediates(cfg: Config, variables, *inputs, phase: str | None = None) -> dict:
    """Per-module outputs (flax capture_intermediates) for `phase`."""
    text, tower = _models(cfg)
    flt = lambda mdl, method: method in _CAPTURE
    if phase is None:
        phase = {1: "prefill", 3: "decode_step", 2: "vision_encoder"}[len(inputs)]
    with net.patched():
        if phase == "prefill":
            (tokens,) = inputs
            cache = cfg.tcfg.init_cache(batch_size=1, dtype=cfg.dtype, cache_length=tokens.shape[1])
            _, st = text.apply({"params": variables["text"]}, tokens, cache=cache,
                               return_last_only=True, capture_intermediates=flt,
                               mutable=["intermediates"])
        elif phase == "decode_step":
            token, pos, cache = inputs
            _, st = text.apply({"params": variables["text"]}, token, cache=cache,
                               positions=pos[:, None],
                               attention_mask=net.step_attention_mask(pos, cfg.cache_len),
                               return_last_only=True, capture_intermediates=flt,
                               mutable=["intermediates"])
        else:                                     # vision_encoder
            patches, xy = inputs
            vi = net._transformer.PreprocessedVisionInput(
                patches=patches, positions_xy=xy,
                soft_token_counts=(cfg.tcfg.vision_encoder.num_mm_tokens_per_image,))
            _, st = tower.apply({"params": variables["vision"]}, vi, method="_encode_vision",
                                capture_intermediates=flt, mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            st["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        name = "/".join(k for k in keys if k != "__call__")
        # Last call wins: FeedForward shares its scope with its two Einsums
        # (nn.share_scope), so its own output is recorded after theirs.
        out = v[-1]
        if "/stacked_layers/block" in name:      # vision scan: [L, ...] -> per layer
            for i in range(cfg.tcfg.vision_encoder.num_layers):
                flat[name.replace("stacked_layers/block", f"layer_{i}")] = jax.tree.map(
                    lambda a: a[i], out)
        else:
            flat[name] = out
    return flat


def build(cfg: Config) -> ModelSpec:
    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    fns, ins = _phase_fns(cfg), input_specs(cfg)
    phases = [
        Phase("prefill", fns["prefill"], ins["prefill"], trip_count=1, param_keys=("text",),
              note=f"prompt of {cfg.prompt_len} tokens; static KV cache of prompt length "
                   "(library prefill slices the cache to the padded prompt)"),
        Phase("decode_step", fns["decode_step"], ins["decode_step"], trip_count=cfg.decode_steps,
              param_keys=("text",),
              note=f"one token at runtime int32 position; static cache of {cfg.cache_len} slots"),
        Phase("vision_encoder", fns["vision_encoder"], ins["vision_encoder"], trip_count=1,
              param_keys=("vision",),
              note=f"one image, {cfg.image_grid_hw[0]}x{cfg.image_grid_hw[1]} patches of "
                   f"{cfg.tcfg.vision_encoder.patch_size}px -> "
                   f"{cfg.tcfg.vision_encoder.num_mm_tokens_per_image} soft tokens"),
    ]
    return ModelSpec(
        name="gemma4",
        variant=(f"Gemma 4 26B-A4B (MoE 128 experts top-8 + shared MLP, 30 layers 5:1 "
                 f"local/global), {cfg.name}, batch 1, official QAT Q4_0 GGUF on llama.cpp: "
                 "Q4_0 x Q8_0 int8 dots (attention, dense MLP, experts), Q6_K x Q8_K tied lm_head, "
                 f"router / attention BMMs {jnp.dtype(cfg.dtype).name}, vision float32"),
        phases=phases,
        param_specs=as_spec(variables),
        excluded=["offline weight conversion (QAT bf16 -> GGUF Q4_0 / Q6_K, llama-quantize)",
                  "tokenization / detokenization and chat template",
                  "sampling (greedy/top-k/top-p) and the decode loop itself (trip_count)",
                  "sampler bookkeeping between phases: merging the prefill cache into the "
                  "full-length cache, end_index / done-mask updates",
                  "image decode + resize + patchify (gm.vision preprocessing)",
                  "merging vision soft tokens into the text embedding sequence "
                  "(_merge_mm_embeddings scatter) and the bidirectional image mask"],
    )

