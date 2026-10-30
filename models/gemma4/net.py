"""Gemma 4 26B-A4B (MoE) on the canonical google-deepmind/gemma JAX library.

The network is the library's own Flax code (vendored at upstream/gemma):
`gemma.gm.nn.gemma4._gemma4.Gemma4_26B_A4B` and its submodules.
This file only adds what static graph capture needs:

1. `MoECapacityDispatch`: replaces the library's `_moe.MoERagged`
   (`jax.lax.ragged_dot` + argsort dispatch + `approx_max_k` custom call) with
   static top-k dispatch at capacity C = T tokens per expert -- exact, no
   drops: router -> top-k -> per-expert fixed-size gather of T token slots ->
   expert FFN (batched over experts) -> weighted scatter-add combine. Same
   parameters, same router, same math.
2. `VisionTransformerUnrolled`: the library's vision `nn.scan(unroll=1)` with
   `unroll=num_layers`, so no `while` is left in the graph.
3. `_VisionOnlyEmbedder` / `Gemma4VisionTower`: the vision phase applies the
   library's `Transformer._encode_vision` on a module that only owns the
   vision parameters (vision encoder + `embedder.mm_input_projection`), so
   the text weights are not dead inputs of that phase.
4. Phase entry points (`prefill`, `decode_step`, `vision_encode`) mirroring
   the library sampler (`gm.text._prefill.prefill`, `_sampler_loop._sample_step`).

Patches 1-2 are applied with `patched()` (a scoped monkeypatch of the two
library module attributes), so the *unpatched* library stays available.
They are not quantization.

Quantization: Google's official QAT Q4_0 GGUF release
(google/gemma-4-26B-A4B-it-qat-q4_0-gguf) as llama.cpp computes it, written
into the vendored library itself (every edit marked `# quantized`; recipe and
sources in recipe.py): `Gemma4_26B_A4B.config` sets weight_quant='q4_0' and
embed_quant='q6_k'.
  * Q4_0 (int4 values in [-8, 7], one fp16 scale per 32 along K): attention
    q / k / v / o, dense shared MLP, all experts. Activations are quantized in
    the graph to Q8_0 (int8 per 32 along K) and every 32-block is an
    i8 x i8 -> i32 dot scaled by d_w * d_x in fp32 (`_layers.q4_0_einsum`).
    The experts run batched over experts in `MoECapacityDispatch`.
  * Q6_K token embedding / tied lm_head: gathered rows dequantized to bf16;
    the lm_head is Q6_K x Q8_K (`_layers.q6_k_dot`).
  * Float: router, norms, attention QK^T / PV and softmax (bf16, KV cache
    bf16), vision tower and mm projector (fp32).
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
GEMMA_DIR = os.path.join(_HERE, "upstream")
if GEMMA_DIR not in sys.path:
    sys.path.insert(0, GEMMA_DIR)
_SHIMS = os.path.join(_HERE, "_shims")
if _SHIMS not in sys.path:
    sys.path.append(_SHIMS)      # import-only stand-ins; real installs win

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from gemma.gm.nn.gemma4 import _config, _gemma4, _layers, _modules, _moe, _transformer  # noqa: E402
from gemma.gm.nn.gemma4.vision import _encoder as _vision_encoder  # noqa: E402
from gemma.gm.nn.gemma4.vision import _transformer as _vision_transformer  # noqa: E402

LOCAL = _modules.AttentionType.LOCAL_SLIDING
GLOBAL = _modules.AttentionType.GLOBAL


# ----------------------------------------------------------------------------
# 1. MoE: static top-k dispatch, capacity = T (replaces MoERagged)
# ----------------------------------------------------------------------------

class MoECapacityDispatch(_moe.MoERagged):
    """`MoERagged` with exact static dispatch (same params / router / math).

    Every expert gets a fixed buffer of C = T token slots. A token routed to
    expert e lands in slot (number of earlier tokens routed to e), so with
    C = T no token is ever dropped and the result equals the ragged version.
    """

    def _router(self, router_logits: jax.Array):
        # Library: `jax.lax.approx_max_k` (an `ApproxTopK` custom_call; on CPU
        # its exact `is_fallback` path). `lax.top_k` is the exact top-k.
        router_logits = router_logits.astype(jnp.float32)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        _, choices = jax.lax.top_k(router_logits, k=self.num_experts_per_datapoint)
        weights = router_probs / _moe._renormalization_factor(router_probs, choices)
        return weights, choices

    def _run_ffw_and_routing(self, x, expert_choices, expert_weights):
        g, s, k = expert_choices.shape
        t, e, d = g * s, self.num_experts, self.features
        xf = x.reshape(t, d)
        # [T, E] routing indicator (top-k experts are distinct) and gate weight.
        sel = jax.nn.one_hot(expert_choices.reshape(t, k), e, dtype=jnp.int32).sum(axis=1)
        gate = sel.astype(jnp.float32) * expert_weights.reshape(t, e).astype(jnp.float32)
        # Slot of token t in expert e's buffer; unrouted pairs -> slot T (dropped).
        slot = jnp.where(sel > 0, jnp.cumsum(sel, axis=0) - 1, t)
        tok = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[:, None], (t, e))
        eid = jnp.broadcast_to(jnp.arange(e, dtype=jnp.int32)[None, :], (t, e))
        # Dispatch table [E, C]: token index held by each slot (T = empty slot).
        table = jnp.full((e, t), t, jnp.int32).at[eid, slot].set(tok, mode="drop")
        combine = jnp.zeros((e, t), jnp.float32).at[eid, slot].set(gate, mode="drop")
        # Per-expert fixed-size gather of T tokens: [E, C, D] (empty -> 0).
        xs = jnp.take(xf, table, axis=0, mode="fill", fill_value=0)

        # Expert FFN, batched over experts (library weights: [E,2,H,F], [E,H,F]).
        # Q4_0 release: Q4_0 x Q8_0 integer dots per 32-K block (library
        # `_layers.q4_0_einsum`, experts = shared batch axis).
        if self.quant is not None:
            gate_out = _layers.q4_0_einsum("ecd,ekhd->eckh", xs, *self.gating_einsum())
            activation = nn.gelu(gate_out[..., 0, :]) * gate_out[..., 1, :]
            expert_outputs = _layers.q4_0_einsum("ech,ehd->ecd", activation, *self.linear())
        else:
            gate_out = jnp.einsum("ecd,ekhd->eckh", xs, self.gating_einsum())
            activation = nn.gelu(gate_out[..., 0, :]) * gate_out[..., 1, :]
            expert_outputs = jnp.einsum("ech,ehd->ecd", activation, self.linear())
        per_expert = self.per_expert_scale.astype(expert_outputs.dtype)
        expert_outputs = expert_outputs * per_expert[:, None, None]

        # Weighted scatter-add combine (f32 accumulation, like the library's
        # promoted 'blkd,blk->bld' einsum), back to the activation dtype.
        weighted = expert_outputs.astype(jnp.float32) * combine[..., None]
        out = jnp.zeros((t, d), jnp.float32).at[table.reshape(-1)].add(
            weighted.reshape(e * t, d), mode="drop")
        return out.astype(x.dtype).reshape(g, s, d)


# ----------------------------------------------------------------------------
# 2. Vision transformer: fully unrolled layer scan
# ----------------------------------------------------------------------------

class VisionTransformerUnrolled(_vision_transformer.VisionTransformer):
    """Library `VisionTransformer.setup` verbatim, except `unroll=num_layers`
    and `intermediates` stacked like `metrics` (only read by the
    per-module activation dump; no effect on the captured computation)."""

    def setup(self):
        scan_init_fn = nn.scan(
            _vision_transformer.VisionBlock,
            length=self.num_layers,
            split_rngs={"params": True, "dropout": True},
            in_axes=(nn.broadcast, nn.broadcast),
            out_axes=0,
            variable_axes={"params": 0, "metrics": 0, "aux_losses": 0, "moe_aux": 0,
                           "intermediates": 0},
            metadata_params={"mesh_axis": None, "axis_type": "stacked",
                             "loss_reduction": "mean"},
            unroll=self.num_layers,
        )
        self.stacked_layers = scan_init_fn(
            d_model=self.d_model,
            ffw_hidden=self.ffw_hidden,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            key_size=self.d_model // self.num_heads,
            use_clipped_linears=self.use_clipped_linears,
        )


@contextlib.contextmanager
def patched(moe: bool = True, unroll_vision: bool = True):
    """Scoped monkeypatch of the two library attributes looked up at setup()."""
    saved = (_moe.MoERagged, _vision_encoder._transformer.VisionTransformer)
    if moe:
        _moe.MoERagged = MoECapacityDispatch
    if unroll_vision:
        _vision_encoder._transformer.VisionTransformer = VisionTransformerUnrolled
    try:
        yield
    finally:
        _moe.MoERagged, _vision_encoder._transformer.VisionTransformer = saved


# ----------------------------------------------------------------------------
# 3. Models (text backbone / vision tower)
# ----------------------------------------------------------------------------

FULL_CONFIG: _config.TransformerConfig = _gemma4.Gemma4_26B_A4B.config


def reduced_config() -> _config.TransformerConfig:
    """Same block types as 26B-A4B (local+global attention, k_eq_v global,
    p-RoPE, MoE + shared MLP, vision encoder w/ pooling + standardize) and the
    same quantization; every GEMM K is a multiple of the Q4_0 block (32) and
    embed_dim holds one Q6_K super-block (256)."""
    return dataclasses.replace(
        FULL_CONFIG,
        num_embed=512, embed_dim=256, hidden_dim=96, num_heads=4, head_dim=32,
        num_kv_heads=2, num_global_kv_heads=2, global_key_size=64,
        attention_types=(LOCAL, LOCAL, GLOBAL), sliding_window_size=8,
        num_experts=8, expert_dim=32, top_k_experts=2, moe_dense_hidden_dim=96,
        vision_encoder=_vision_encoder.VisionEncoder(
            d_model=32, num_layers=2, num_heads=2, ffw_hidden=64, patch_size=4,
            output_length=6, pos_emb_shape_yx=(16, 2), use_clipped_linears=False,
            standardize_embeddings=True),
    )


def text_model(config: _config.TransformerConfig, dtype) -> _transformer.Transformer:
    return _gemma4.Gemma4_26B_A4B(config=config, dtype=dtype, text_only=True)


class _VisionOnlyEmbedder(_modules.Embedder):
    """`Embedder` restricted to its multimodal-projection branch (library
    `Embedder.setup`, `if self.vision_proj_dim:` lines); `encode_vision` is the
    library method."""

    def setup(self):
        self.mm_input_projection = _layers.Einsum((self.vision_proj_dim, self.embed_dim))
        self.mm_pre_projection_norm = _layers.RMSNorm(with_scale=False)


class Gemma4VisionTower(_gemma4.Gemma4_26B_A4B):
    """Owns only the vision parameters; runs library `_encode_vision`."""

    text_only: bool = False

    def setup(self):
        self.embedder = _VisionOnlyEmbedder(
            vocab_size=self.config.num_embed, embed_dim=self.config.embed_dim,
            vision_proj_dim=self.config.vision_encoder.d_model)
        self.vision_encoder = self.config.vision_encoder


def vision_tower(config: _config.TransformerConfig) -> Gemma4VisionTower:
    return Gemma4VisionTower(config=config)


# ----------------------------------------------------------------------------
# 4. Phases (one network invocation each; sampler loop not traced)
# ----------------------------------------------------------------------------

def prefill(model, params, tokens):
    """Library prefill: model call on the prompt with a fresh static KV cache
    sliced to the (padded) prompt length, `return_last_only=True`."""
    cache = model.config.init_cache(batch_size=tokens.shape[0], dtype=model.dtype,
                                    cache_length=tokens.shape[1])
    with patched():
        out = model.apply({"params": params}, tokens, cache=cache, return_last_only=True)
    return out.logits, out.cache


def step_attention_mask(position, cache_length: int):
    """Sampler `attention_mask_for_step` for an unpadded prompt: every cache
    slot up to and including the current position is visible. [B, 1, L]."""
    return (jnp.arange(cache_length)[None, :] <= position[:, None])[:, None, :]


def decode_step(model, params, token, position, cache):
    """Library `_sample_step` network call: one token at a runtime position."""
    cache_length = cache["layer_0"]["k"].shape[1]
    with patched():
        out = model.apply({"params": params}, token, cache=cache,
                          positions=position[:, None],
                          attention_mask=step_attention_mask(position, cache_length),
                          return_last_only=True)
    return out.logits, out.cache


def vision_encode(tower, params, patches, positions_xy):
    """Library `Transformer._encode_vision`: encoder + pooling + standardize +
    projection into the text embedding space. Returns [1, n_img*tokens, D]."""
    n_tok = tower.config.vision_encoder.num_mm_tokens_per_image
    n_img = patches.shape[0]
    vi = _transformer.PreprocessedVisionInput(
        patches=patches.reshape(1, -1, patches.shape[-1]),
        positions_xy=positions_xy.reshape(1, -1, 2),
        soft_token_counts=(n_tok,) * n_img)
    with patched():
        return tower.apply({"params": params}, vi, method="_encode_vision")
