"""Llama 3.1 (MLPerf Inference llama3.1-8b) decoder, FP8 Flax linen port of MaxText.

Ported from MaxText (github.com/AI-Hypercomputer/maxtext) with `attention=dot_product`, `scan_layers=False`, single device:

  maxtext/models/llama2.py      LlamaDecoderLayer      -> LlamaDecoderLayer
  maxtext/layers/attentions.py  Attention              -> Attention
  maxtext/layers/attention_op.py apply_attention_dot / qk_product / wv_product
                                                       -> Attention.apply_attention_dot ...
  maxtext/layers/embeddings.py  Embed, RotaryEmbedding, LLaMARotaryEmbedding
                                                       -> Embed, LLaMARotaryEmbedding
  maxtext/layers/normalizations.py RMSNorm             -> RMSNorm
  maxtext/layers/linears.py     DenseGeneral, MlpBlock -> DenseGeneral, MlpBlock
  maxtext/layers/decoders.py    Decoder                -> Decoder
  maxtext/models/models.py      Transformer            -> Transformer

Module / parameter names follow MaxText (token_embedder, decoder/layers_{i},
pre_self_attention_layer_norm, self_attention/{query,key,value,out},
post_self_attention_layer_norm, mlp/{wi_0,wi_1,wo}, decoder_norm, logits_dense).
Numerics (where dtypes are cast) mirror the HF transformers LlamaForCausalLM
reference (eager attention). Sharding constraints, quantization, dropout (inference) and
all kernel paths (splash/flash/ragged/paged) are dropped.

KV cache: one static cache per layer, layout [batch, max_target_length,
num_kv_heads, head_dim] (MaxText's logical BSKD order), written with
`lax.dynamic_update_slice` at a runtime `cache_index`; attention runs over all
`max_target_length` slots with a causal/valid mask (HF StaticCache semantics).

Quantization: ModelOpt FP8 as built by TensorRT-LLM for MLPerf Inference on Hopper
(inference_results_v5.1 llama3_1-8b configs: precision 'fp8', kv_cache_dtype 'fp8',
use_fp8_context_fmha; checkpoint layout of nvidia/Llama-3.1-8B-Instruct-FP8):
  * q/k/v/o and gate/up/down: FP8 E4M3 weights with one fp32 scale per tensor, inputs
    quantized to FP8 E4M3 with one static (max-calibrated) scale per tensor
    (ModelOpt FP8_DEFAULT_CFG); f8 x f8 -> f32 GEMM, bf16 epilogue.
  * KV cache FP8 E4M3 with a static per-layer k_scale / v_scale (FP8_KV_CFG).
    TensorRT-LLM folds them into one kv_cache_scaling_factor = max(k_scale, v_scale);
    the checkpoint keeps both.
  * FP8 attention. Prefill (FP8 context FMHA): Q, K and V are quantized with the KV
    scale (unfusedAttentionKernels_2_template.h). Decode (XQA, mha_sm90.cu with an FP8
    cache): Q is cast to FP8 unscaled. In both, QK^T is f8 x f8 -> f32, the fp32
    softmax probabilities are multiplied by 448 and cast to FP8 (scale 1/448), and PV is
    f8 x f8 -> f32. The fp32 attention output is quantized by o_proj's input scale
    (TensorRT-LLM does that inside the attention kernel; fusion is not modelled).
  * Float: embedding gather and lm_head (ModelOpt '*lm_head*' disabled) bf16; RMSNorm
    fp32 math; RoPE fp32 angles; softmax fp32; SiLU * up and residual adds bf16.
TensorRT-LLM fuses q/k/v and gate/up into single GEMMs (max of the input scales);
kernel fusion is not modelled, the GEMM dtypes are the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from models import quant as q

# maxtext/common/common_types.py
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


@dataclass(frozen=True)
class Config:
    vocab_size: int = 128256
    emb_dim: int = 4096
    mlp_dim: int = 14336
    num_decoder_layers: int = 32
    num_query_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    normalization_layer_epsilon: float = 1e-5
    rope_max_timescale: float = 500000.0       # rope_theta
    # llama3 rope_scaling
    rope_factor: float = 8.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position: int = 8192
    batch: int = 1
    max_prefill_predict_length: int = 2048     # prefill tokens
    max_target_length: int = 4096              # static KV-cache capacity S
    dtype: str = "bfloat16"                    # activations, float weights, GEMM epilogues


# --------------------------------------------------------------------------
# layers/linears.py
# --------------------------------------------------------------------------

class DenseGeneral(nn.Module):
    """y = x @ kernel over the last `in_axes` axes.

    The kernel is stored flattened as [prod(in), prod(out)] (MaxText stores
    [*in, *out]); the dot_general is identical (same M, N, K).

    fp8=True (quantized: ModelOpt FP8): FP8 E4M3 kernel with a per-tensor
    `kernel_scale`, input quantized with the per-tensor static `input_scale`,
    f8 x f8 -> f32 dot, epilogue scaled by input_scale * kernel_scale in fp32, out in
    `dtype`. fp8=False: plain `dtype` GEMM (lm_head).
    """
    out_features_shape: tuple[int, ...]
    in_axes: int = 1
    dtype: str = "float32"
    fp8: bool = True

    @nn.compact
    def __call__(self, inputs):
        in_shape = inputs.shape[inputs.ndim - self.in_axes:]
        k_in, k_out = math.prod(in_shape), math.prod(self.out_features_shape)
        x = inputs.reshape(inputs.shape[:inputs.ndim - self.in_axes] + (k_in,))
        dims = (((x.ndim - 1,), (0,)), ((), ()))
        if self.fp8:
            kernel = self.param("kernel", q.rand_q("fp8"), (k_in, k_out), q.FP8)
            kernel_scale = self.param("kernel_scale", q.const(3.0 / math.sqrt(k_in) / 448.0),
                                      (), jnp.float32)
            input_scale = self.param("input_scale", q.const(4.0 / 448.0), (), jnp.float32)
            acc = lax.dot_general(q.quantize(x, input_scale, "fp8"), kernel, dims,
                                  preferred_element_type=jnp.float32)
            y = (acc * (input_scale * kernel_scale)).astype(self.dtype)
        else:
            kernel = self.param("kernel", nn.initializers.lecun_normal(), (k_in, k_out), self.dtype)
            x = jnp.asarray(x, self.dtype)
            y = lax.dot_general(x, jnp.asarray(kernel, self.dtype), dims)
        return y.reshape(y.shape[:-1] + tuple(self.out_features_shape))


class MlpBlock(nn.Module):
    """SwiGLU MLP: wo(silu(wi_0(x)) * wi_1(x)); activations=('silu','linear')."""
    intermediate_dim: int
    emb_dim: int
    dtype: str

    @nn.compact
    def __call__(self, inputs):
        activations = []
        for idx, act_fn in enumerate((nn.silu, lambda v: v)):
            x = DenseGeneral((self.intermediate_dim,), dtype=self.dtype, name=f"wi_{idx}")(inputs)
            activations.append(act_fn(x))
        x = (activations[0] * activations[1]).astype(self.dtype)
        return DenseGeneral((self.emb_dim,), dtype=self.dtype, name="wo")(x)


# --------------------------------------------------------------------------
# layers/normalizations.py
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    epsilon: float
    dtype: str

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],), self.dtype)
        x = jnp.asarray(x, jnp.float32)
        mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
        return y * jnp.asarray(scale, self.dtype)


# --------------------------------------------------------------------------
# layers/embeddings.py
# --------------------------------------------------------------------------

class Embed(nn.Module):
    num_embeddings: int
    num_features: int
    dtype: str

    @nn.compact
    def __call__(self, inputs):
        embedding = self.param("embedding", nn.initializers.normal(1.0),
                               (self.num_embeddings, self.num_features), self.dtype)
        return jnp.asarray(embedding, self.dtype)[inputs]


def llama3_inv_freq(cfg: Config) -> np.ndarray:
    """LLaMARotaryEmbedding.timescale (use_scale=True) as inverse frequencies.

    Evaluated in float32 with the op order of HF `_compute_llama3_parameters`
    (constant-folded, like HF's `inv_freq` buffer).
    """
    f32 = np.float32
    dim = cfg.head_dim
    exponent = np.arange(0, dim, 2).astype(f32) / f32(dim)
    # timescale = base ** fraction; evaluated in f64 and rounded (== torch's f32 pow here).
    timescale = (np.float64(cfg.rope_max_timescale) ** exponent.astype(np.float64)).astype(f32)
    inv = f32(1.0) / timescale
    low_wl = cfg.rope_original_max_position / cfg.rope_low_freq_factor
    high_wl = cfg.rope_original_max_position / cfg.rope_high_freq_factor
    wavelen = f32(2 * math.pi) / inv
    inv_l = np.where(wavelen > low_wl, inv / f32(cfg.rope_factor), inv)
    smooth = ((f32(cfg.rope_original_max_position) / wavelen - f32(cfg.rope_low_freq_factor))
              / f32(cfg.rope_high_freq_factor - cfg.rope_low_freq_factor))
    smoothed = (f32(1) - smooth) * inv_l / f32(cfg.rope_factor) + smooth * inv_l
    medium = ~(wavelen < high_wl) & ~(wavelen > low_wl)
    return np.where(medium, smoothed, inv_l).astype(f32)


class LLaMARotaryEmbedding(nn.Module):
    """RotaryEmbedding (half-split rotate) with llama3.1 frequency scaling."""
    cfg: Config

    def _rotate_half(self, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    def __call__(self, inputs, position):          # inputs [B, S, N, H], position [B, S]
        inv_freq = jnp.asarray(llama3_inv_freq(self.cfg))
        sinusoid_inp = position.astype(jnp.float32)[:, :, jnp.newaxis, jnp.newaxis] * inv_freq
        sin_half = jnp.sin(sinusoid_inp).astype(inputs.dtype)
        cos_half = jnp.cos(sinusoid_inp).astype(inputs.dtype)
        sin = jnp.concatenate([sin_half, sin_half], axis=-1)
        cos = jnp.concatenate([cos_half, cos_half], axis=-1)
        return (inputs * cos) + (self._rotate_half(inputs) * sin)


# --------------------------------------------------------------------------
# layers/attentions.py + layers/attention_op.py (dot_product path)
# --------------------------------------------------------------------------

class Attention(nn.Module):
    """GQA attention over an FP8 KV cache with FP8 QK^T / PV (see the module docstring)."""
    cfg: Config
    decode: bool = False            # XQA decode kernel (one token) vs context FMHA (prefill)

    def update_kv_caches(self, key, value, kv_cache, cache_index, k_scale, v_scale):
        # quantized: FP8 KV cache, per-layer static k_scale / v_scale
        idx = (jnp.int32(0), cache_index, jnp.int32(0), jnp.int32(0))
        cached_key = lax.dynamic_update_slice(kv_cache["key"], q.kv_store(key, k_scale), idx)
        cached_value = lax.dynamic_update_slice(kv_cache["value"], q.kv_store(value, v_scale), idx)
        return cached_key, cached_value

    def generate_attention_mask(self, q_positions, kv_len):
        """Causal over absolute positions: slot s is visible to query t iff s <= pos(t)."""
        kv_positions = jnp.arange(kv_len, dtype=jnp.int32)
        mask = kv_positions[jnp.newaxis, :] <= q_positions[:, :, jnp.newaxis]   # [b, t, s]
        return mask[:, jnp.newaxis, jnp.newaxis, :, :]                            # [b, 1, 1, t, s]

    def qk_product(self, query, key):
        b, t, n, d = query.shape
        n_kv = key.shape[-2]
        query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
        return jnp.einsum("btkgd,bskd->bkgts", query, key, preferred_element_type=jnp.float32)

    def wv_product(self, attn_weights, value):
        out = jnp.einsum("bkgts,bskd->btkgd", attn_weights, value,
                         preferred_element_type=jnp.float32)
        b, t, n_kv, g, d = out.shape
        return jnp.reshape(out, (b, t, n_kv * g, d))

    def apply_attention_dot(self, query, key, value, q_positions, k_scale, v_scale):
        """query bf16, key / value FP8 (cache). Returns the fp32 attention output."""
        # quantized: FP8 attention. Prefill (FP8 context FMHA) quantizes Q with the KV
        # scale; decode (XQA sm90, one query token) casts Q to FP8 unscaled. The kernel is
        # chosen by the phase (`decode`), never by the token count: the graph of each phase
        # must not depend on input sizes (symbolic export).
        q_scale = jnp.float32(1.0) if self.decode else k_scale
        attn_weights = self.qk_product(q.quantize(query, q_scale, "fp8"), key)
        attn_weights = attn_weights * (q_scale * k_scale * self.cfg.head_dim ** -0.5)
        mask = self.generate_attention_mask(q_positions, key.shape[1])
        attn_weights = jnp.where(mask, attn_weights, DEFAULT_MASK_VALUE)
        probs = jax.nn.softmax(attn_weights, axis=-1)
        # probabilities mapped 1 -> 448 before the FP8 cast (mha_sm90.cu kE4M3_MAX)
        p_scale = 1.0 / q.QMAX["fp8"]
        out = self.wv_product(q.quantize(probs, p_scale, "fp8"), value)
        return out * (p_scale * v_scale)

    @nn.compact
    def __call__(self, inputs_q, inputs_positions, kv_cache, cache_index):
        c = self.cfg
        query = DenseGeneral((c.num_query_heads, c.head_dim), dtype=c.dtype, name="query")(inputs_q)
        key = DenseGeneral((c.num_kv_heads, c.head_dim), dtype=c.dtype, name="key")(inputs_q)
        value = DenseGeneral((c.num_kv_heads, c.head_dim), dtype=c.dtype, name="value")(inputs_q)
        rope = LLaMARotaryEmbedding(c, name="rotary_embedding")
        query = rope(query, inputs_positions)
        key = rope(key, inputs_positions)
        k_scale = self.param("k_scale", q.const(8.0 / 448.0), (), jnp.float32)
        v_scale = self.param("v_scale", q.const(8.0 / 448.0), (), jnp.float32)
        cached_key, cached_value = self.update_kv_caches(key, value, kv_cache, cache_index,
                                                         k_scale, v_scale)
        out = self.apply_attention_dot(query, cached_key, cached_value, inputs_positions,
                                       k_scale, v_scale)
        out = DenseGeneral((c.emb_dim,), in_axes=2, dtype=c.dtype, name="out")(out)
        return out, {"key": cached_key, "value": cached_value}


# --------------------------------------------------------------------------
# models/llama2.py
# --------------------------------------------------------------------------

class LlamaDecoderLayer(nn.Module):
    cfg: Config
    decode: bool = False

    @nn.compact
    def __call__(self, inputs, decoder_positions, kv_cache, cache_index):
        c = self.cfg
        lnx = RMSNorm(c.normalization_layer_epsilon, c.dtype,
                      name="pre_self_attention_layer_norm")(inputs)
        attention_lnx, kv_cache = Attention(c, self.decode, name="self_attention")(
            lnx, decoder_positions, kv_cache, cache_index)
        intermediate_inputs = inputs + attention_lnx
        hidden_states = RMSNorm(c.normalization_layer_epsilon, c.dtype,
                                name="post_self_attention_layer_norm")(intermediate_inputs)
        mlp_lnx = MlpBlock(c.mlp_dim, c.emb_dim, c.dtype, name="mlp")(hidden_states)
        layer_output = mlp_lnx + intermediate_inputs
        return layer_output, kv_cache


# --------------------------------------------------------------------------
# layers/decoders.py + models/models.py
# --------------------------------------------------------------------------

class Decoder(nn.Module):
    cfg: Config
    decode: bool = False

    @nn.compact
    def __call__(self, y, decoder_positions, kv_caches, cache_index):
        c = self.cfg
        new_caches = {}
        for lyr in range(c.num_decoder_layers):
            name = f"layers_{lyr}"
            y, new_caches[name] = LlamaDecoderLayer(c, self.decode, name=name)(
                y, decoder_positions, kv_caches[name], cache_index)
        y = RMSNorm(c.normalization_layer_epsilon, c.dtype, name="decoder_norm")(y)
        # Only the last position's logits are needed for sampling (HF generate's
        # logits_to_keep=1; vLLM likewise computes only sampled positions).
        y = y[:, -1:, :]
        # lm_head stays bf16 (ModelOpt excludes *lm_head*)
        logits = DenseGeneral((c.vocab_size,), dtype=c.dtype, fp8=False, name="logits_dense")(y)
        return logits, new_caches


class Transformer(nn.Module):
    cfg: Config
    decode: bool = False            # decode_step phase (one token per call)

    @nn.compact
    def __call__(self, tokens, kv_caches, cache_index):
        """tokens [b, t] int32; cache_index: int32 scalar, absolute position of tokens[:, 0]."""
        c = self.cfg
        y = Embed(c.vocab_size, c.emb_dim, c.dtype, name="token_embedder")(tokens)
        positions = cache_index + jnp.arange(tokens.shape[1], dtype=jnp.int32)
        positions = jnp.broadcast_to(positions[jnp.newaxis, :], tokens.shape)
        return Decoder(c, self.decode, name="decoder")(y, positions, kv_caches, cache_index)
