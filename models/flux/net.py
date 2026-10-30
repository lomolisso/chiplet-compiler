"""FLUX.1 [schnell] text-to-image pipeline networks, Flax linen.

Four networks, one per captured phase:

* CLIP ViT-L/14 text encoder (pooled output)  -- re-implementation of HF
  transformers 5.20.2 `CLIPTextModel` (models/clip/modeling_clip.py); the Flax
  code follows the HF v4.57.6 `modeling_flax_clip.py` port used by models/sdxl.
* T5 v1.1 XXL encoder                         -- re-implementation of HF
  transformers 5.20.2 `T5EncoderModel` (models/t5/modeling_t5.py).
* FluxTransformer2DModel (MMDiT, 19 double-stream + 38 single-stream blocks)
  -- re-implementation of diffusers 0.41.0
  `models/transformers/transformer_flux.py` (+ the `normalization.py`,
  `embeddings.py`, `attention.py`, `activations.py` helpers it uses), quantized as
  BFL's official FP8 TensorRT export (see "Quantization" below).
* AutoencoderKL decoder (16 latent channels)  -- MaxDiffusion `vae_flax.py`
  linen decoder as ported in models/sdxl, with diffusers module names.

Module and attribute names follow diffusers / transformers (ModuleList index
"blocks.3" -> "blocks_3"), so parameter paths map 1:1 onto the torch
state_dict. Precision handling mirrors the torch code where it casts
explicitly (`.float()`, `.to(dtype)`, fp32 norm statistics); deviations are
marked `# DEVIATION`.

Quantization: the official BFL / NVIDIA FP8 TensorRT export
(black-forest-labs/FLUX.1-schnell-onnx `transformer.opt/fp8/model.onnx`, built with
ModelOpt; NVIDIA/TensorRT demo/Diffusion `utils_modelopt.py`; BFL `src/flux/trt`):
  * FP8 E4M3, one static (max-calibrated) scale per tensor, W8A8 on every linear of
    the 19 double-stream blocks (14 each: norm1.linear, norm1_context.linear, q/k/v,
    add_q/k/v_proj, to_out.0, to_add_out, ff and ff_context net.0.proj / net.2) and
    the 38 single-stream blocks (6 each: norm.linear, q/k/v, proj_mlp, proj_out):
    494 FP8 GEMMs per step, f8 x f8 -> f32 with a bf16 epilogue (scale, bias).
  * FP8 attention in all 57 blocks (ModelOpt FP8 MHA, sdpa decomposition): Q and K
    (after QK-RMSNorm and RoPE) are each multiplied by head_dim^-1/4 and quantized,
    V and the softmax probabilities are quantized; QK^T and PV are f8 x f8 -> f32,
    the softmax runs in fp32. The attention output is re-quantized by the input
    quantizers of to_out.0 / to_add_out / proj_out.
  * bf16 (unquantized, ModelOpt filter_func): x_embedder, context_embedder,
    time_text_embed (4 linears), norm_out.linear, the final proj_out, RoPE (fp32
    cos/sin), LayerNorm / RMSNorm statistics (fp32), AdaLN modulation, GELU, gates and
    residual adds.
  * CLIP, T5 and the VAE run unquantized in bf16 (BFL trt_manager / NVIDIA demo:
    clip.opt, t5.opt and vae.opt bf16 engines). The deployed T5 engine runs `wo` in
    bf16 (transformers' fp32 `wo`, `_keep_in_fp32_modules`, is not kept).
Graph form: explicit Q/DQ (models/quant.py); weights are stored fp8 with one fp32
kernel_scale, each linear owns an input_scale, each attention its q/k/v/softmax
`*_bmm_quant` / `softmax_quant` scales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from models import quant as q

F32 = jnp.float32

# Two-pass variance E[(x-mean)^2] as torch's LayerNorm / GroupNorm (see models/sdxl).
LayerNorm = partial(nn.LayerNorm, use_fast_variance=False)
GroupNorm = partial(nn.GroupNorm, use_fast_variance=False)


# =============================================================================
# Configs (field names follow the HF / diffusers config.json keys)
# =============================================================================

@dataclass(frozen=True)
class CLIPTextConfig:                       # text_encoder/config.json
    vocab_size: int = 49408
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5


@dataclass(frozen=True)
class T5Config:                             # text_encoder_2/config.json (google/t5-v1_1-xxl)
    vocab_size: int = 32128
    d_model: int = 4096
    d_kv: int = 64
    d_ff: int = 10240
    num_layers: int = 24
    num_heads: int = 64
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6        # feed_forward_proj "gated-gelu", dense_act_fn "gelu_new"


@dataclass(frozen=True)
class FluxConfig:                           # transformer/config.json
    in_channels: int = 64
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    axes_dims_rope: tuple = (16, 56, 56)    # FluxTransformer2DModel default (not in config.json)
    patch_size: int = 1
    mlp_ratio: float = 4.0

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


@dataclass(frozen=True)
class VAEConfig:                            # vae/config.json
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159


# =============================================================================
# Shared layers
# =============================================================================

def Dense(features, dtype, use_bias=True, name=None, param_dtype=None):
    return nn.Dense(features, use_bias=use_bias, dtype=dtype,
                    param_dtype=param_dtype or dtype, name=name)


def FP8Dense(features, dtype, name=None):
    """FP8 E4M3 W8A8 linear (per-tensor static scales), f32 accumulation, `dtype` epilogue."""
    return q.QDense(features, q.FP8_STATIC, use_bias=True, dtype=dtype, name=name)


def softmax_f32(logits, dtype):
    # torch softmax / sdpa accumulate reduced-precision inputs in fp32
    return jax.nn.softmax(logits.astype(F32), axis=-1).astype(dtype)


def gelu_tanh(x):
    # F.gelu(approximate="tanh") == transformers NewGELUActivation ("gelu_new")
    return nn.gelu(x, approximate=True)


def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)


# =============================================================================
# CLIP text encoder (transformers CLIPTextModel; FluxPipeline uses pooler_output)
# =============================================================================

class CLIPTextEmbeddings(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = F32

    def setup(self):
        c = self.config
        self.token_embedding = nn.Embed(c.vocab_size, c.hidden_size, dtype=self.dtype,
                                        param_dtype=self.dtype)
        self.position_embedding = nn.Embed(c.max_position_embeddings, c.hidden_size,
                                           dtype=self.dtype, param_dtype=self.dtype)

    def __call__(self, input_ids):
        position_ids = jnp.arange(input_ids.shape[-1])[None, :]     # position_ids buffer
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class CLIPAttention(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = F32

    def setup(self):
        c = self.config
        self.head_dim = c.hidden_size // c.num_attention_heads
        dense = partial(Dense, c.hidden_size, self.dtype)
        self.k_proj, self.v_proj, self.q_proj, self.out_proj = dense(), dense(), dense(), dense()

    def __call__(self, hidden_states):
        b, s, _ = hidden_states.shape
        split = lambda t: t.reshape(b, s, -1, self.head_dim)
        q = split(self.q_proj(hidden_states))
        k = split(self.k_proj(hidden_states))
        v = split(self.v_proj(hidden_states))
        # CLIP text attention is causal (create_causal_mask; no padding mask is passed)
        causal = jnp.tril(jnp.ones((s, s), bool))
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=F32) * self.head_dim ** -0.5
        logits = jnp.where(causal, logits, jnp.finfo(F32).min)
        probs = softmax_f32(logits, v.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", probs, v).reshape(b, s, -1)
        return (self.out_proj(out),)


class CLIPMLP(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = F32

    def setup(self):
        self.fc1 = Dense(self.config.intermediate_size, self.dtype)
        self.fc2 = Dense(self.config.hidden_size, self.dtype)

    def __call__(self, hidden_states):
        return self.fc2(quick_gelu(self.fc1(hidden_states)))    # hidden_act "quick_gelu"


class CLIPEncoderLayer(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = F32

    def setup(self):
        eps = self.config.layer_norm_eps
        self.self_attn = CLIPAttention(self.config, self.dtype)
        self.layer_norm1 = LayerNorm(epsilon=eps, dtype=self.dtype, param_dtype=self.dtype)
        self.mlp = CLIPMLP(self.config, self.dtype)
        self.layer_norm2 = LayerNorm(epsilon=eps, dtype=self.dtype, param_dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))[0]
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class CLIPEncoder(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = F32

    def setup(self):
        self.layers = [CLIPEncoderLayer(self.config, self.dtype, name=f"layers_{i}")
                       for i in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class CLIPTextModel(nn.Module):
    """input_ids [B, 77] -> pooler_output [B, hidden] (FluxPipeline._get_clip_prompt_embeds)."""
    config: CLIPTextConfig
    dtype: jnp.dtype = F32

    def setup(self):
        self.embeddings = CLIPTextEmbeddings(self.config, self.dtype)
        self.encoder = CLIPEncoder(self.config, self.dtype)
        self.final_layer_norm = LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype,
                                          param_dtype=self.dtype)

    def __call__(self, input_ids):
        last_hidden_state = self.final_layer_norm(self.encoder(self.embeddings(input_ids)))
        # config eos_token_id == 2 (legacy): the EOS token is the highest id; first argmax
        return last_hidden_state[jnp.arange(last_hidden_state.shape[0]), input_ids.argmax(axis=-1)]


# =============================================================================
# T5 v1.1 encoder (transformers T5EncoderModel; FluxPipeline uses last_hidden_state)
# =============================================================================

class T5LayerNorm(nn.Module):
    eps: float
    dtype: jnp.dtype = F32

    @nn.compact
    def __call__(self, hidden_states):
        weight = self.param("scale", nn.initializers.ones, (hidden_states.shape[-1],), self.dtype)
        variance = jnp.mean(jnp.square(hidden_states.astype(F32)), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        return weight * hidden_states.astype(weight.dtype)


def _relative_position_bucket(relative_position, num_buckets, max_distance):
    """T5Attention._relative_position_bucket, bidirectional (encoder)."""
    num_buckets //= 2
    relative_buckets = (relative_position > 0).astype(jnp.int32) * num_buckets
    relative_position = jnp.abs(relative_position)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_position_if_large = max_exact + (
        jnp.log(relative_position.astype(F32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(jnp.int32)
    relative_position_if_large = jnp.minimum(relative_position_if_large, num_buckets - 1)
    return relative_buckets + jnp.where(is_small, relative_position, relative_position_if_large)


def relative_position_buckets(c: T5Config, query_length, key_length):
    """T5Attention.compute_bias: bucketed (memory - context) positions [q, k]."""
    context_position = jnp.arange(query_length, dtype=jnp.int32)[:, None]
    memory_position = jnp.arange(key_length, dtype=jnp.int32)[None, :]
    return _relative_position_bucket(memory_position - context_position,
                                     c.relative_attention_num_buckets,
                                     c.relative_attention_max_distance)


class T5Attention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = F32

    def setup(self):
        c = self.config
        inner = c.num_heads * c.d_kv
        self.q = Dense(inner, self.dtype, use_bias=False)
        self.k = Dense(inner, self.dtype, use_bias=False)
        self.v = Dense(inner, self.dtype, use_bias=False)
        self.o = Dense(c.d_model, self.dtype, use_bias=False)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(c.relative_attention_num_buckets, c.num_heads,
                                                    dtype=self.dtype, param_dtype=self.dtype)

    def __call__(self, hidden_states, position_bias=None):
        b, s, _ = hidden_states.shape
        split = lambda t: t.reshape(b, s, -1, self.config.d_kv)
        q = split(self.q(hidden_states))
        k = split(self.k(hidden_states))
        v = split(self.v(hidden_states))
        if position_bias is None:                                 # layer 0: compute_bias(s, s)
            values = self.relative_attention_bias(relative_position_buckets(self.config, s, s))
            position_bias = jnp.transpose(values, (2, 0, 1))[None]  # [1, heads, q, k]
        # T5: no 1/sqrt(d) scaling; the relative position bias is added to the scores
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=F32)
        scores = scores + position_bias.astype(F32)
        probs = softmax_f32(scores, v.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", probs, v).reshape(b, s, -1)
        return self.o(out), position_bias


class T5LayerSelfAttention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = F32

    def setup(self):
        self.SelfAttention = T5Attention(self.config, self.has_relative_attention_bias, self.dtype)
        self.layer_norm = T5LayerNorm(self.config.layer_norm_epsilon, self.dtype)

    def __call__(self, hidden_states, position_bias=None):
        attention_output, position_bias = self.SelfAttention(self.layer_norm(hidden_states),
                                                             position_bias)
        return hidden_states + attention_output, position_bias


class T5DenseGatedActDense(nn.Module):
    config: T5Config
    dtype: jnp.dtype = F32

    def setup(self):
        c = self.config
        self.wi_0 = Dense(c.d_ff, self.dtype, use_bias=False)
        self.wi_1 = Dense(c.d_ff, self.dtype, use_bias=False)
        # Deployed TRT T5 (t5.opt, bf16): wo runs in bf16 like every other linear
        # (transformers keeps it fp32 via _keep_in_fp32_modules; the engine does not).
        self.wo = Dense(c.d_model, self.dtype, use_bias=False)

    def __call__(self, hidden_states):
        hidden_states = gelu_tanh(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
        return self.wo(hidden_states)


class T5LayerFF(nn.Module):
    config: T5Config
    dtype: jnp.dtype = F32

    def setup(self):
        self.DenseReluDense = T5DenseGatedActDense(self.config, self.dtype)
        self.layer_norm = T5LayerNorm(self.config.layer_norm_epsilon, self.dtype)

    def __call__(self, hidden_states):
        return hidden_states + self.DenseReluDense(self.layer_norm(hidden_states))


class T5Block(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = F32

    def setup(self):
        self.layer_0 = T5LayerSelfAttention(self.config, self.has_relative_attention_bias, self.dtype)
        self.layer_1 = T5LayerFF(self.config, self.dtype)

    def __call__(self, hidden_states, position_bias=None):
        hidden_states, position_bias = self.layer_0(hidden_states, position_bias)
        return self.layer_1(hidden_states), position_bias


class T5Stack(nn.Module):
    config: T5Config
    dtype: jnp.dtype = F32

    def setup(self):
        self.block = [T5Block(self.config, has_relative_attention_bias=(i == 0), dtype=self.dtype,
                              name=f"block_{i}") for i in range(self.config.num_layers)]
        self.final_layer_norm = T5LayerNorm(self.config.layer_norm_epsilon, self.dtype)

    def __call__(self, inputs_embeds):
        hidden_states, position_bias = inputs_embeds, None
        for layer in self.block:
            # the bias computed by block 0 is shared by every later block
            hidden_states, position_bias = layer(hidden_states, position_bias)
        return self.final_layer_norm(hidden_states)


class T5EncoderModel(nn.Module):
    """input_ids [B, max_sequence_length] -> last_hidden_state [B, S, d_model]."""
    config: T5Config
    dtype: jnp.dtype = F32

    def setup(self):
        # `shared` is also encoder.embed_tokens (one tied tensor)
        self.shared = nn.Embed(self.config.vocab_size, self.config.d_model, dtype=self.dtype,
                               param_dtype=self.dtype)
        self.encoder = T5Stack(self.config, self.dtype)

    def __call__(self, input_ids):
        return self.encoder(self.shared(input_ids))


# =============================================================================
# FluxTransformer2DModel (diffusers transformer_flux.py)
# =============================================================================

def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=1,
                           max_period=10000):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * jnp.arange(0, half_dim, dtype=F32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = timesteps[:, None].astype(F32) * jnp.exp(exponent)[None, :]
    # cat([sin, cos]) then flipped halves == cat([cos, sin]); Flux uses flip_sin_to_cos=True
    assert flip_sin_to_cos
    return jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)


class Timesteps(nn.Module):
    num_channels: int
    flip_sin_to_cos: bool
    downscale_freq_shift: float

    def __call__(self, timesteps):
        return get_timestep_embedding(timesteps, self.num_channels, self.flip_sin_to_cos,
                                      self.downscale_freq_shift)


class TimestepEmbedding(nn.Module):
    time_embed_dim: int
    dtype: jnp.dtype = F32

    def setup(self):
        self.linear_1 = Dense(self.time_embed_dim, self.dtype)
        self.linear_2 = Dense(self.time_embed_dim, self.dtype)

    def __call__(self, sample):
        return self.linear_2(nn.silu(self.linear_1(sample)))


class PixArtAlphaTextProjection(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = F32

    def setup(self):
        self.linear_1 = Dense(self.hidden_size, self.dtype)
        self.linear_2 = Dense(self.hidden_size, self.dtype)

    def __call__(self, caption):
        return self.linear_2(nn.silu(self.linear_1(caption)))     # act_fn="silu"


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """guidance_embeds=False (schnell): timestep + pooled CLIP text embedding."""
    embedding_dim: int
    dtype: jnp.dtype = F32

    def setup(self):
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(self.embedding_dim, self.dtype)
        self.text_embedder = PixArtAlphaTextProjection(self.embedding_dim, self.dtype)

    def __call__(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.astype(pooled_projection.dtype))
        return timesteps_emb + self.text_embedder(pooled_projection)


class FluxPosEmbed(nn.Module):
    """3D RoPE over (text/0, row, col) ids; returns (cos, sin) [S, sum(axes_dim)] fp32."""
    theta: int
    axes_dim: tuple

    def __call__(self, ids):
        pos = ids.astype(F32)
        cos_out, sin_out = [], []
        for i in range(ids.shape[-1]):
            dim = self.axes_dim[i]
            # DEVIATION (precision): diffusers computes the frequencies, the angles and
            # cos/sin in float64 (then .float()). JAX without x64 has no float64: the
            # frequencies are float64 host constants rounded to fp32, angles and cos/sin fp32.
            freqs = 1.0 / (self.theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
            freqs = pos[:, i][:, None] * jnp.asarray(freqs, F32)[None, :]       # torch.outer
            cos_out.append(jnp.repeat(jnp.cos(freqs), 2, axis=1))              # repeat_interleave
            sin_out.append(jnp.repeat(jnp.sin(freqs), 2, axis=1))
        return jnp.concatenate(cos_out, axis=-1), jnp.concatenate(sin_out, axis=-1)


def apply_rotary_emb(x, freqs_cis):
    """diffusers apply_rotary_emb(use_real=True, use_real_unbind_dim=-1, sequence_dim=1)."""
    cos, sin = freqs_cis
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_real, x_imag = pairs[..., 0], pairs[..., 1]
    x_rotated = jnp.stack([-x_imag, x_real], axis=-1).reshape(x.shape)
    return (x.astype(F32) * cos + x_rotated.astype(F32) * sin).astype(x.dtype)


class RMSNorm(nn.Module):
    """torch.nn.RMSNorm(dim, eps, elementwise_affine=True): fp32 math, cast back."""
    eps: float
    dtype: jnp.dtype = F32

    @nn.compact
    def __call__(self, x):
        weight = self.param("scale", nn.initializers.ones, (x.shape[-1],), self.dtype)
        xf = x.astype(F32)
        y = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + self.eps)
        return (y * weight.astype(F32)).astype(x.dtype)


def _no_affine_ln(dtype, name=None):
    return LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, dtype=dtype, name=name)


def scaled_dot_product_attention(query, key, value):
    """F.scaled_dot_product_attention on [B, S, H, D] (dispatch_attention_fn, native backend)."""
    scale = query.shape[-1] ** -0.5
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, preferred_element_type=F32) * scale
    probs = softmax_f32(logits, value.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", probs, value)


class FP8Attention(nn.Module):
    """ModelOpt FP8 MHA (sdpa on [B, S, H, D]): q and k each scaled by D^-1/4, then q, k, v
    and the fp32 softmax probabilities quantized to FP8 E4M3 with static per-tensor scales;
    QK^T and PV are f8 x f8 -> f32. Returns `dtype`."""
    dtype: jnp.dtype = F32

    @nn.compact
    def __call__(self, query, key, value):
        s = query.shape[-1] ** -0.25
        fp8 = lambda name, init: q.ActQuant("fp8", init_scale=init, name=name)
        qq, sq, _ = fp8("q_bmm_quant", 4.0 / 448)(query.astype(F32) * s)
        kq, sk, _ = fp8("k_bmm_quant", 4.0 / 448)(key.astype(F32) * s)
        vq, sv, _ = fp8("v_bmm_quant", 4.0 / 448)(value)
        logits = q.qeinsum("bqhd,bkhd->bhqk", qq, kq, sq, sk, "fp8", F32)
        pq, sp, _ = fp8("softmax_quant", 1.0 / 448)(jax.nn.softmax(logits, axis=-1))
        return q.qeinsum("bhqk,bkhd->bqhd", pq, vq, sp, sv, "fp8", self.dtype)


class FluxAttention(nn.Module):
    """FluxAttention + FluxAttnProcessor (unfused q/k/v projections); FP8 linears and FP8 MHA."""
    query_dim: int
    heads: int
    dim_head: int
    added_kv_proj_dim: int | None = None
    pre_only: bool = False
    eps: float = 1e-6
    dtype: jnp.dtype = F32

    def setup(self):
        inner = self.dim_head * self.heads
        dense = partial(FP8Dense, inner, self.dtype)
        self.attn_fp8 = FP8Attention(self.dtype)
        self.norm_q = RMSNorm(self.eps, self.dtype)
        self.norm_k = RMSNorm(self.eps, self.dtype)
        self.to_q, self.to_k, self.to_v = dense(), dense(), dense()
        if not self.pre_only:
            self.to_out_0 = FP8Dense(self.query_dim, self.dtype)
        if self.added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(self.eps, self.dtype)
            self.norm_added_k = RMSNorm(self.eps, self.dtype)
            self.add_q_proj, self.add_k_proj, self.add_v_proj = dense(), dense(), dense()
            self.to_add_out = FP8Dense(self.query_dim, self.dtype)

    def __call__(self, hidden_states, encoder_hidden_states=None, image_rotary_emb=None):
        heads = lambda t: t.reshape(*t.shape[:-1], -1, self.dim_head)     # unflatten(-1, (-1, D))
        query = self.norm_q(heads(self.to_q(hidden_states)))
        key = self.norm_k(heads(self.to_k(hidden_states)))
        value = heads(self.to_v(hidden_states))

        if encoder_hidden_states is not None:
            encoder_query = self.norm_added_q(heads(self.add_q_proj(encoder_hidden_states)))
            encoder_key = self.norm_added_k(heads(self.add_k_proj(encoder_hidden_states)))
            encoder_value = heads(self.add_v_proj(encoder_hidden_states))
            # joint attention: text tokens first
            query = jnp.concatenate([encoder_query, query], axis=1)
            key = jnp.concatenate([encoder_key, key], axis=1)
            value = jnp.concatenate([encoder_value, value], axis=1)

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = self.attn_fp8(query, key, value)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:2], -1)

        if encoder_hidden_states is not None:
            txt = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states[:, :txt], hidden_states[:, txt:]
            return self.to_out_0(hidden_states), self.to_add_out(encoder_hidden_states)
        return hidden_states


class AdaLayerNormZero(nn.Module):
    embedding_dim: int
    dtype: jnp.dtype = F32

    def setup(self):
        self.linear = FP8Dense(6 * self.embedding_dim, self.dtype)
        self.norm = _no_affine_ln(self.dtype)

    def __call__(self, x, emb):
        emb = self.linear(nn.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(emb, 6, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    embedding_dim: int
    dtype: jnp.dtype = F32

    def setup(self):
        self.linear = FP8Dense(3 * self.embedding_dim, self.dtype)
        self.norm = _no_affine_ln(self.dtype)

    def __call__(self, x, emb):
        emb = self.linear(nn.silu(emb))
        shift_msa, scale_msa, gate_msa = jnp.split(emb, 3, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class AdaLayerNormContinuous(nn.Module):
    embedding_dim: int
    dtype: jnp.dtype = F32

    def setup(self):
        self.linear = Dense(2 * self.embedding_dim, self.dtype)
        self.norm = _no_affine_ln(self.dtype)     # elementwise_affine=False, eps=1e-6

    def __call__(self, x, conditioning_embedding):
        emb = self.linear(nn.silu(conditioning_embedding).astype(x.dtype))
        scale, shift = jnp.split(emb, 2, axis=1)                 # note: scale first
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class GELU(nn.Module):
    """diffusers activations.GELU(approximate="tanh"): proj + tanh GELU."""
    dim_out: int
    dtype: jnp.dtype = F32

    def setup(self):
        self.proj = FP8Dense(self.dim_out, self.dtype)

    def __call__(self, hidden_states):
        return gelu_tanh(self.proj(hidden_states))


class FeedForward(nn.Module):
    """FeedForward(dim, dim_out=dim, activation_fn="gelu-approximate"), mult 4."""
    dim: int
    dtype: jnp.dtype = F32

    def setup(self):
        self.net_0 = GELU(4 * self.dim, self.dtype)
        self.net_2 = FP8Dense(self.dim, self.dtype)

    def __call__(self, hidden_states):
        return self.net_2(self.net_0(hidden_states))


class FluxTransformerBlock(nn.Module):
    """Double-stream (MMDiT) block; returns (encoder_hidden_states, hidden_states)."""
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    dtype: jnp.dtype = F32

    def setup(self):
        self.norm1 = AdaLayerNormZero(self.dim, self.dtype)
        self.norm1_context = AdaLayerNormZero(self.dim, self.dtype)
        self.attn = FluxAttention(self.dim, self.num_attention_heads, self.attention_head_dim,
                                  added_kv_proj_dim=self.dim, eps=1e-6, dtype=self.dtype)
        self.norm2 = _no_affine_ln(self.dtype)
        self.ff = FeedForward(self.dim, self.dtype)
        self.norm2_context = _no_affine_ln(self.dtype)
        self.ff_context = FeedForward(self.dim, self.dtype)

    def __call__(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
            self.norm1_context(encoder_hidden_states, temb)

        attn_output, context_attn_output = self.attn(norm_hidden_states, norm_encoder_hidden_states,
                                                     image_rotary_emb)

        hidden_states = hidden_states + gate_msa[:, None] * attn_output
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp[:, None] * self.ff(norm_hidden_states)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa[:, None] * context_attn_output
        norm_encoder_hidden_states = (self.norm2_context(encoder_hidden_states)
                                      * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None])
        encoder_hidden_states = (encoder_hidden_states
                                 + c_gate_mlp[:, None] * self.ff_context(norm_encoder_hidden_states))
        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream block: parallel attention + MLP on cat([text, image]), fused proj_out."""
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    mlp_ratio: float = 4.0
    dtype: jnp.dtype = F32

    def setup(self):
        self.norm = AdaLayerNormZeroSingle(self.dim, self.dtype)
        self.proj_mlp = FP8Dense(int(self.dim * self.mlp_ratio), self.dtype)
        self.proj_out = FP8Dense(self.dim, self.dtype)
        self.attn = FluxAttention(self.dim, self.num_attention_heads, self.attention_head_dim,
                                  pre_only=True, eps=1e-6, dtype=self.dtype)

    def __call__(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, temb)
        mlp_hidden_states = gelu_tanh(self.proj_mlp(norm_hidden_states))     # act_mlp
        attn_output = self.attn(norm_hidden_states, image_rotary_emb=image_rotary_emb)

        hidden_states = jnp.concatenate([attn_output, mlp_hidden_states], axis=2)
        hidden_states = residual + gate[:, None] * self.proj_out(hidden_states)
        return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]


class FluxTransformer2DModel(nn.Module):
    """One denoiser call: packed latents [B, (H/2)(W/2), 4C] -> velocity, same shape."""
    config: FluxConfig
    dtype: jnp.dtype = F32

    def setup(self):
        c = self.config
        d = c.inner_dim
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=c.axes_dims_rope)
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(d, self.dtype)
        self.context_embedder = Dense(d, self.dtype)
        self.x_embedder = Dense(d, self.dtype)
        self.transformer_blocks = [
            FluxTransformerBlock(d, c.num_attention_heads, c.attention_head_dim, self.dtype,
                                 name=f"transformer_blocks_{i}") for i in range(c.num_layers)]
        self.single_transformer_blocks = [
            FluxSingleTransformerBlock(d, c.num_attention_heads, c.attention_head_dim, c.mlp_ratio,
                                       self.dtype, name=f"single_transformer_blocks_{i}")
            for i in range(c.num_single_layers)]
        self.norm_out = AdaLayerNormContinuous(d, self.dtype)
        self.proj_out = Dense(c.patch_size * c.patch_size * c.in_channels, self.dtype)

    def __call__(self, hidden_states, encoder_hidden_states, pooled_projections, timestep,
                 img_ids, txt_ids):
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.astype(hidden_states.dtype) * 1000      # pipeline passes t / 1000
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = jnp.concatenate([txt_ids, img_ids], axis=0)
        image_rotary_emb = self.pos_embed(ids)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, temb,
                                                         image_rotary_emb)
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, temb,
                                                         image_rotary_emb)

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


# =============================================================================
# AutoencoderKL decoder (MaxDiffusion vae_flax.py port, diffusers names)
# =============================================================================

def Conv(features, kernel, dtype, name=None):
    pad = ((kernel // 2, kernel // 2),) * 2
    return nn.Conv(features, kernel_size=(kernel, kernel), strides=(1, 1), padding=pad,
                   dtype=dtype, param_dtype=dtype, name=name)


class Upsample2D(nn.Module):
    channels: int
    dtype: jnp.dtype = F32

    def setup(self):
        self.conv = Conv(self.channels, 3, self.dtype)

    def __call__(self, hidden_states):
        # nearest-neighbour 2x (F.interpolate(scale_factor=2.0, mode="nearest"))
        b, h, w, c = hidden_states.shape
        hidden_states = jnp.broadcast_to(hidden_states[:, :, None, :, None, :], (b, h, 2, w, 2, c))
        return self.conv(hidden_states.reshape(b, 2 * h, 2 * w, c))


class ResnetBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    groups: int = 32
    dtype: jnp.dtype = F32

    def setup(self):
        gn = partial(GroupNorm, num_groups=self.groups, epsilon=1e-6, dtype=self.dtype,
                     param_dtype=self.dtype)
        self.norm1 = gn()
        self.conv1 = Conv(self.out_channels, 3, self.dtype)
        self.norm2 = gn()
        self.conv2 = Conv(self.out_channels, 3, self.dtype)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = Conv(self.out_channels, 1, self.dtype)

    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = self.conv1(nn.swish(self.norm1(hidden_states)))
        hidden_states = self.conv2(nn.swish(self.norm2(hidden_states)))
        if self.in_channels != self.out_channels:
            residual = self.conv_shortcut(residual)
        return hidden_states + residual


class Attention(nn.Module):
    """VAE mid-block single-head spatial self-attention (diffusers Attention, AttnProcessor2_0)."""
    channels: int
    groups: int = 32
    dtype: jnp.dtype = F32

    def setup(self):
        dense = partial(Dense, self.channels, self.dtype)
        self.group_norm = GroupNorm(num_groups=self.groups, epsilon=1e-6, dtype=self.dtype,
                                    param_dtype=self.dtype)
        self.to_q, self.to_k, self.to_v = dense(), dense(), dense()
        self.to_out_0 = dense()

    def __call__(self, hidden_states):
        residual = hidden_states
        b, h, w, c = hidden_states.shape
        hidden_states = self.group_norm(hidden_states).reshape(b, h * w, c)
        heads = lambda t: t[:, :, None, :]                            # heads = 1 (head_dim = C)
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)
        out = scaled_dot_product_attention(heads(q), heads(k), heads(v)).reshape(b, h * w, c)
        out = out.astype(q.dtype)
        out = self.to_out_0(out).reshape(b, h, w, c)
        return out + residual            # residual_connection=True, rescale_output_factor=1


class UNetMidBlock2D(nn.Module):
    channels: int
    groups: int = 32
    dtype: jnp.dtype = F32

    def setup(self):
        self.resnets = [ResnetBlock2D(self.channels, self.channels, self.groups, self.dtype,
                                      name=f"resnets_{i}") for i in range(2)]
        self.attentions = [Attention(self.channels, self.groups, self.dtype, name="attentions_0")]

    def __call__(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.attentions[0](hidden_states)
        return self.resnets[1](hidden_states)


class UpDecoderBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    num_layers: int
    groups: int = 32
    add_upsample: bool = True
    dtype: jnp.dtype = F32

    def setup(self):
        self.resnets = [ResnetBlock2D(self.in_channels if i == 0 else self.out_channels,
                                      self.out_channels, self.groups, self.dtype, name=f"resnets_{i}")
                        for i in range(self.num_layers)]
        if self.add_upsample:
            self.upsamplers_0 = Upsample2D(self.out_channels, self.dtype)

    def __call__(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)
        return hidden_states


class Decoder(nn.Module):
    config: VAEConfig
    dtype: jnp.dtype = F32

    def setup(self):
        c = self.config
        ch = list(reversed(c.block_out_channels))
        self.conv_in = Conv(ch[0], 3, self.dtype)
        self.mid_block = UNetMidBlock2D(ch[0], c.norm_num_groups, self.dtype)
        self.up_blocks = [
            UpDecoderBlock2D(ch[max(i - 1, 0)], ch[i], c.layers_per_block + 1, c.norm_num_groups,
                             add_upsample=i < len(ch) - 1, dtype=self.dtype, name=f"up_blocks_{i}")
            for i in range(len(ch))]
        self.conv_norm_out = GroupNorm(num_groups=c.norm_num_groups, epsilon=1e-6, dtype=self.dtype,
                                       param_dtype=self.dtype)
        self.conv_out = Conv(c.out_channels, 3, self.dtype)

    def __call__(self, sample):
        sample = self.mid_block(self.conv_in(sample))
        for block in self.up_blocks:
            sample = block(sample)
        return self.conv_out(nn.swish(self.conv_norm_out(sample)))


class AutoencoderKL(nn.Module):
    """FluxPipeline post-loop: _unpack_latents, latents / scaling + shift, vae.decode(...).sample.

    use_post_quant_conv=False (FLUX.1 VAE), so decode == decoder.
    """
    config: VAEConfig
    dtype: jnp.dtype = F32

    def setup(self):
        self.decoder = Decoder(self.config, self.dtype)

    def __call__(self, latents):
        # latents [B, H/2, W/2, 4C] (the packed [B, (H/2)(W/2), 4C] tokens as their grid)
        # -> [B, C, H, W] (FluxPipeline._unpack_latents)
        b, hh, ww, c = latents.shape
        h, w = 2 * hh, 2 * ww
        latents = latents.reshape(b, h // 2, w // 2, c // 4, 2, 2)
        latents = jnp.transpose(latents, (0, 3, 1, 4, 2, 5)).reshape(b, c // 4, h, w)
        latents = latents / self.config.scaling_factor + self.config.shift_factor
        image = self.decoder(jnp.transpose(latents, (0, 2, 3, 1)))      # NCHW -> NHWC
        return jnp.transpose(image, (0, 3, 1, 2))
