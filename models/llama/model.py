"""Llama 3.1 (MLPerf Inference llama3.1-8b) FP8 capture spec; network lives in net.py.

Recipe: ModelOpt FP8 W8A8 per-tensor static + FP8 KV cache + FP8 attention (TensorRT-LLM,
MLPerf Inference Hopper); see net.py and recipe.py. The KV cache is stored FP8 E4M3.

Phases:
  prefill      tokens [b, P] at positions 0..P-1, static KV cache (capacity S)
               in -> last-position logits + updated cache.         trip_count 1
  decode_step  one token at a runtime int32 position, cache in -> logits +
               updated cache (dynamic_update_slice).               trip_count 128
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.quant import FP8
from models.spec import ModelSpec, Phase, Sym, as_spec
from models.llama.net import Config, Transformer

# MLPerf Inference llama3.1-8b: SUT_VLLM.py sampling params "max_tokens": 128.
MAX_OUTPUT_TOKENS = 128

CONFIGS = {
    # meta-llama/Llama-3.1-8B-Instruct config.json
    "full": Config(),
    "reduced": Config(vocab_size=256, emb_dim=64, mlp_dim=128, num_decoder_layers=2,
                      num_query_heads=4, num_kv_heads=2, head_dim=16,
                      max_prefill_predict_length=16, max_target_length=32),
}


def cache_spec(cfg: Config):
    shape = (cfg.batch, cfg.max_target_length, cfg.num_kv_heads, cfg.head_dim)
    leaf = jax.ShapeDtypeStruct(shape, FP8)        # FP8 E4M3 KV cache
    return {f"layers_{i}": {"key": leaf, "value": leaf} for i in range(cfg.num_decoder_layers)}


def input_specs(cfg: Config) -> dict:
    i32 = jnp.int32
    return {
        "prefill": (jax.ShapeDtypeStruct((cfg.batch, cfg.max_prefill_predict_length), i32),
                    cache_spec(cfg)),
        "decode_step": (jax.ShapeDtypeStruct((cfg.batch, 1), i32),
                        jax.ShapeDtypeStruct((), i32),
                        cache_spec(cfg)),
    }


def _apply(cfg: Config, variables, tokens, cache, cache_index, decode=False, **kw):
    return Transformer(cfg, decode).apply(variables, tokens, cache, cache_index, **kw)


def prefill_fn(cfg: Config):
    def prefill(variables, tokens, cache):
        return _apply(cfg, variables, tokens, cache, jnp.int32(0))
    return prefill


def decode_fn(cfg: Config):
    def decode_step(variables, token, position, cache):
        return _apply(cfg, variables, token, cache, position, decode=True)
    return decode_step


def init_variables(cfg: Config, rng):
    tok, cache = input_specs(cfg)["prefill"]
    cache = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), cache)
    return Transformer(cfg).init(rng, jnp.zeros(tok.shape, tok.dtype), cache, jnp.int32(0))


def decode_position(cfg: Config) -> int:
    """Example runtime position: a few steps into decoding (any value < S works)."""
    return cfg.max_prefill_predict_length + 3


def example_inputs(cfg: Config, rng):
    k_tok, k_dec, k_cache = jax.random.split(rng, 3)
    specs = input_specs(cfg)
    tokens = jax.random.randint(k_tok, specs["prefill"][0].shape, 0, cfg.vocab_size, jnp.int32)
    zero_cache = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), specs["prefill"][1])
    # Decode: every slot holds random data, so a wrong valid-mask changes the result.
    leaves, treedef = jax.tree.flatten(specs["decode_step"][2])
    keys = jax.random.split(k_cache, len(leaves))
    full_cache = jax.tree.unflatten(treedef, [(jax.random.normal(k, s.shape) * 64.0).astype(s.dtype)
                                              for k, s in zip(keys, leaves)])
    token = jax.random.randint(k_dec, (cfg.batch, 1), 0, cfg.vocab_size, jnp.int32)
    return {"prefill": (tokens, zero_cache),
            "decode_step": (token, jnp.int32(decode_position(cfg)), full_cache)}


def intermediates(cfg: Config, variables, *inputs) -> dict:
    if len(inputs) == 2:
        tokens, cache = inputs
        pos = jnp.int32(0)
    else:
        tokens, pos, cache = inputs
    _, state = _apply(cfg, variables, tokens, cache, pos, decode=len(inputs) == 3,
                      capture_intermediates=True, mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        out = v[0]
        if isinstance(out, tuple):          # (output, kv_cache) modules
            out = out[0]
        flat["/".join(k for k in keys if k != "__call__")] = out
    return flat


def build(cfg: Config) -> ModelSpec:
    specs = input_specs(cfg)
    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    name = "8B" if cfg == CONFIGS["full"] else "reduced"
    return ModelSpec(
        name="llama",
        variant=(f"Llama 3.1 {name} Instruct (MLPerf llama3.1-8b), ModelOpt FP8 E4M3 W8A8 per-tensor "
                 f"static + FP8 KV cache + FP8 attention (TensorRT-LLM, MLPerf Hopper); lm_head and "
                 f"activations {cfg.dtype}, batch {cfg.batch}, prefill {cfg.max_prefill_predict_length}, "
                 f"static FP8 KV cache {cfg.max_target_length}"),
        phases=[
            Phase("prefill", prefill_fn(cfg), specs["prefill"], trip_count=1,
                  shapes=("B, P", "B, S, _, _"),
                  note=f"{cfg.max_prefill_predict_length} prompt tokens at positions 0..; writes "
                       f"cache slots 0..{cfg.max_prefill_predict_length - 1}; last-position logits"),
            Phase("decode_step", decode_fn(cfg), specs["decode_step"],
                  trip_count=MAX_OUTPUT_TOKENS, shapes=("B, _", "", "B, S, _, _"),
                  note="one token at a runtime int32 position; dynamic_update_slice cache write; "
                       f"attends over all {cfg.max_target_length} slots with a causal/valid mask. "
                       "Trip count = MLPerf max_tokens 128 (the first output token comes from "
                       "prefill, so at most 127 decode calls are strictly needed)."),
        ],
        param_specs=as_spec(variables),
        symbols={"B": Sym("batch", 3, "sequences per call"),
                 "P": Sym("max_prefill_predict_length", 12, "prompt tokens (prefill)"),
                 "S": Sym("max_target_length", 40, "static KV-cache capacity")},
        constraints=("S >= P",),
        excluded=["tokenization / detokenization (pre/postprocessing)",
                  "sampling (MLPerf: greedy argmax over logits) and the outer decode loop",
                  "TensorRT-LLM paged KV cache / in-flight batching scheduler",
                  "offline FP8 calibration (ModelOpt max calibration on the MLPerf 1k "
                  "calibration set; scales are parameters here)"],
    )
