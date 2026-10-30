"""Qwen3-VL-MoE (MLPerf Inference v6.1 `multimodal/qwen3-vl`), Flax linen port of MaxText.

Ported from MaxText (github.com/AI-Hypercomputer/maxtext @ 4a9985b8), which runs Qwen3-VL-MoE as `decoder_block: qwen3_moe` +
`vision_encoder_block: qwen3_vl` (configs/models/qwen3-vl-30b-a3b.yml):

  models/qwen3.py  Qwen3OmniMoeVisionPatchEmbed   -> VisionPatchEmbed
                   Qwen3OmniMoeVisionAttention    -> VisionAttention
                   Qwen3OmniMoeVisionBlock        -> VisionBlock
                   Qwen3OmniMoeVisionPatchMerger  -> VisionPatchMerger
                   Qwen3OmniMoeVisionEncoder      -> VisionEncoder (+ DeepStack taps)
                   Qwen3OmniMoeVisionProjector    -> VisionProjector
                   AttentionWithNorm, Qwen3MoeDecoderLayer -> Qwen3MoeDecoderLayer
  layers/embeddings.py Qwen3OmniMoeVisionRotaryEmbedding -> vision_rope_cos_sin
                   Qwen3OmniMoeVisionPosEmbedInterpolate -> VisionPosEmbedInterpolate
                   Qwen3OmniMoeThinkerTextRotaryEmbedding (interleaved M-RoPE) -> MRoPE
  layers/attentions.py Attention (dot_product, qk-norm, use_mrope) -> Attention
  layers/moe.py    RoutedMoE (+ GateLogit)        -> RoutedMoE (static capacity dispatch)
  layers/nnx_decoders.py deepstack_process        -> deepstack_process
  multimodal/utils.py merge_mm_embeddings         -> merge_mm_embeddings
  layers/normalizations.py RMSNorm                -> RMSNorm

Numerics (dtype casts) mirror the HF transformers `Qwen3VLMoeForConditionalGeneration`
reference (eager attention, eager experts). Sharding, MaxText's quantization, dropout,
remat, the Pallas/splash/ragged/megablox kernel paths and video/padding-mask
branches are dropped.

Quantization: Qwen's official FP8 checkpoint (Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 @
7fbcd8c9, served by vLLM / SGLang): config.json quantization_config {quant_method fp8,
fmt e4m3, weight_block_size [128, 128], activation_scheme dynamic}.
  * FP8 E4M3 weights with one fp32 scale (weight_scale_inv, a dequant multiplier) per
    128 x 128 block: text q/k/v/o projections (`QDense(FP8_BLOCK128)`, scales stored
    [K/128, N/128]) and the fused experts gate_up [E, D, 2F] / down [E, F, D] (scales
    [E, K/128, N/128], as in the checkpoint).
  * Activations FP8 E4M3, dynamic, one fp32 scale per token per 128-wide K group,
    computed in the graph (vLLM per_token_group_quant_fp8); each 128-K block is an
    f8 x f8 -> f32 partial product, scaled and summed in fp32 (`quant.block_scaled_dot`,
    DeepGEMM / CUTLASS blockwise kernels); bf16 output.
  * bf16 (the checkpoint's `ignored_layers` / non-linear tensors): the whole vision tower
    (patch embed, 27 ViT blocks, DeepStack mergers, merger), token embedding, lm_head,
    the MoE router (mlp.gate), every norm. No kv_cache_scheme: KV cache and attention
    QK^T / PV stay bf16 (softmax fp32).

Static layout: one image of fixed size (cfg.image_size) -> fixed patch grid;
its merged tokens occupy a fixed span [image_start, image_start + n_img) of a
fixed-length prompt. Vision grid / interpolation tables / RoPE position ids are
computed on the host (numpy) from the fixed size; the text M-RoPE position ids
are a host-computed graph input.

KV cache: one static cache per layer [batch, max_target_length, kv_heads,
head_dim], written with `lax.dynamic_update_slice` at `cache_index`
(HF StaticCache semantics, like models/llama).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from models import quant as q

# maxtext/common/common_types.py
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


@dataclass(frozen=True)
class VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 4096
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)
    rope_theta: float = 10000.0
    layer_norm_epsilon: float = 1e-6


@dataclass(frozen=True)
class TextConfig:
    vocab_size: int = 151936
    emb_dim: int = 4096
    num_decoder_layers: int = 94
    num_query_heads: int = 64
    num_kv_heads: int = 4
    head_dim: int = 128
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_mlp_dim: int = 1536
    normalization_layer_epsilon: float = 1e-6
    rope_max_timescale: float = 5000000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)


@dataclass(frozen=True)
class Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    image_size: tuple[int, int] = (448, 448)       # (H, W) pixels, multiple of patch*merge
    max_prefill_predict_length: int = 1024         # prompt tokens (text + image)
    text_suffix_len: int = 5                       # "<|im_end|>\n<|im_start|>assistant\n"
    max_target_length: int = 1174                  # static KV capacity = prefill + 150
    # Merged image-token count, if not derived from image_size: lets the vision tower
    # be exported with it as its own symbol (the patch count (H/16)(W/16) is not a linear size).
    image_tokens: int | None = None
    image_token_id: int = 151655
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    batch: int = 1
    dtype: str = "bfloat16"

    # ---- static layout (host) ----
    @property
    def patch_grid(self) -> tuple[int, int, int]:
        """(t, h, w) patch grid of the fixed image (one temporal patch for a still image)."""
        p = self.vision.patch_size
        return 1, self.image_size[0] // p, self.image_size[1] // p

    @property
    def num_patches(self) -> int:
        if self.image_tokens is not None:
            return self.image_tokens * self.vision.spatial_merge_size ** 2
        return math.prod(self.patch_grid)

    @property
    def num_image_tokens(self) -> int:
        if self.image_tokens is not None:
            return self.image_tokens
        return self.num_patches // self.vision.spatial_merge_size ** 2

    @property
    def image_start(self) -> int:
        """Prompt: prefix text, <vision_start>, image tokens, <vision_end>, suffix text."""
        return self.max_prefill_predict_length - self.text_suffix_len - 1 - self.num_image_tokens


# --------------------------------------------------------------------------
# Host-side tables (numpy, from the image size). HF computes them on the host too
# (get_vision_position_ids, get_vision_interpolation_indices_and_weights) and passes
# them to the vision tower; so does the vision_encoder phase (not trace-time constants
# frozen at one image size).
# --------------------------------------------------------------------------

def vision_position_ids(cfg: Config) -> np.ndarray:
    """(row, col) of every patch in spatial-merge-block order: [num_patches, 2]
    (embeddings.py Qwen3OmniMoeVisionRotaryEmbedding._generate_position_ids_single)."""
    _, h, w = cfg.patch_grid
    m = cfg.vision.spatial_merge_size
    row = (np.arange(h // m)[:, None, None, None] * m + np.arange(m)[None, None, :, None])
    col = (np.arange(w // m)[None, :, None, None] * m + np.arange(m)[None, None, None, :])
    row = np.broadcast_to(row, (h // m, w // m, m, m)).reshape(-1)
    col = np.broadcast_to(col, (h // m, w // m, m, m)).reshape(-1)
    return np.stack([row, col], axis=-1).astype(np.int32)


def vision_interpolation_table(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear (align_corners) resampling of the learned square position grid to the
    image's patch grid: gather indices [num_patches, 4] into the flattened table and
    weights [num_patches, 4] (fp32), in spatial-merge-block patch order.
    (embeddings.py Qwen3OmniMoeVisionPosEmbedInterpolate._interpolate_single.)"""
    f32 = np.float32
    side = int(cfg.vision.num_position_embeddings ** 0.5)
    _, h, w = cfg.patch_grid
    pos = vision_position_ids(cfg)

    def axis(index, size):
        src = index.astype(f32) * f32(side - 1) / f32(max(size - 1, 1))
        floor = np.floor(src)
        taps = np.clip(floor.astype(np.int64)[:, None] + np.arange(2), 0, side - 1)
        wts = np.clip(f32(1) - np.abs(src[:, None] - floor[:, None] - np.arange(2, dtype=f32)), 0, None)
        return taps, wts.astype(f32)

    ht, hw = axis(pos[:, 0], h)
    wt, ww = axis(pos[:, 1], w)
    idx = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4).astype(np.int32)
    wts = (hw[:, :, None] * ww[:, None, :]).reshape(-1, 4).astype(f32)
    return idx, wts


def rope_inv_freq(theta: float, dim: int) -> np.ndarray:
    """1 / theta ** (arange(0, dim, 2) / dim) in float32 (constant-folded, like HF's
    `inv_freq` buffer; pow evaluated in f64 and rounded -- equal to torch's vectorized
    f32 pow up to 1 ulp)."""
    f32 = np.float32
    exponent = np.arange(0, dim, 2).astype(f32) / f32(dim)
    return (f32(1.0) / (np.float64(theta) ** exponent.astype(np.float64)).astype(f32)).astype(f32)


# --------------------------------------------------------------------------
# Shared layers (layers/linears.py, layers/normalizations.py)
# --------------------------------------------------------------------------

class DenseGeneral(nn.Module):
    """y = x @ kernel (+ bias). Kernel stored flattened [in, out] (MaxText [*in, *out])."""
    features: int
    use_bias: bool = False
    dtype: str = "float32"

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param("kernel", nn.initializers.lecun_normal(),
                            (inputs.shape[-1], self.features), self.dtype)
        x = jnp.asarray(inputs, self.dtype)
        y = lax.dot_general(x, jnp.asarray(kernel, self.dtype), (((x.ndim - 1,), (0,)), ((), ())))
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), self.dtype)
            y = y + jnp.asarray(bias, self.dtype)
        return y


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


class LayerNorm(nn.Module):
    """nnx.LayerNorm (scale + bias); statistics and affine in fp32, output in dtype."""
    epsilon: float
    dtype: str

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones, (d,), self.dtype)
        bias = self.param("bias", nn.initializers.zeros, (d,), self.dtype)
        x = jnp.asarray(x, jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(lax.square(x - mean), axis=-1, keepdims=True)
        y = (x - mean) * lax.rsqrt(var + self.epsilon)
        y = y * jnp.asarray(scale, jnp.float32) + jnp.asarray(bias, jnp.float32)
        return jnp.asarray(y, self.dtype)


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


# --------------------------------------------------------------------------
# Vision tower (models/qwen3.py Qwen3OmniMoeVision*)
# --------------------------------------------------------------------------

class VisionPatchEmbed(nn.Module):
    """3-D conv over (temporal, h, w) patches, stride = kernel."""
    cfg: Config

    @nn.compact
    def __call__(self, pixel_patches):            # [N, C * Tp * P * P]
        v = self.cfg.vision
        k = (v.temporal_patch_size, v.patch_size, v.patch_size)
        x = pixel_patches.astype(self.cfg.dtype).reshape((-1, v.in_channels) + k)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))      # NDHWC
        x = nn.Conv(v.hidden_size, kernel_size=k, strides=k, padding="VALID", use_bias=True,
                    dtype=self.cfg.dtype, param_dtype=self.cfg.dtype, name="proj")(x)
        return x.reshape(-1, v.hidden_size)


class VisionPosEmbedInterpolate(nn.Module):
    """Learned square position grid, bilinearly resampled to the patch grid."""
    cfg: Config

    @nn.compact
    def __call__(self, idx, wts):
        """idx [N, 4] int32 / wts [N, 4] fp32: `vision_interpolation_table` (host input)."""
        v = self.cfg.vision
        table = self.param("pos_embed", nn.initializers.normal(v.hidden_size ** -0.5),
                           (v.num_position_embeddings, v.hidden_size), self.cfg.dtype)
        emb = jnp.asarray(table)[idx]                                  # [N, 4, D]
        return (emb.astype(jnp.float32) * wts[:, :, None]).sum(axis=1)


def vision_rope_cos_sin(cfg: Config, pos):
    """2-D axial RoPE tables [N, head_dim] (fp32): row freqs ++ col freqs, doubled.
    pos [N, 2] int32: `vision_position_ids` (host input)."""
    v = cfg.vision
    head_dim = v.hidden_size // v.num_heads
    inv_freq = jnp.asarray(rope_inv_freq(v.rope_theta, head_dim // 2))
    freqs = pos.astype(jnp.float32)[:, :, None] * inv_freq            # [N, 2, hd/4]
    cos, sin = jnp.cos(freqs), jnp.sin(freqs)
    cos = jnp.concatenate([cos[:, 0], cos[:, 1]], axis=-1)
    sin = jnp.concatenate([sin[:, 0], sin[:, 1]], axis=-1)
    return jnp.concatenate([cos, cos], axis=-1), jnp.concatenate([sin, sin], axis=-1)


class VisionAttention(nn.Module):
    """Full bidirectional self-attention over the image's patches (one image ->
    one attention segment), 2-D RoPE in fp32."""
    cfg: Config

    @nn.compact
    def __call__(self, x, cos, sin):
        v, dt = self.cfg.vision, self.cfg.dtype
        n, d = x.shape
        hd = d // v.num_heads
        qkv = DenseGeneral(3 * d, use_bias=True, dtype=dt, name="qkv")(x)
        q, k, val = jnp.split(qkv.reshape(n, 3, v.num_heads, hd), 3, axis=1)
        q, k, val = q[:, 0], k[:, 0], val[:, 0]                         # [N, H, hd]
        c, s = cos[:, None, :], sin[:, None, :]
        q = (q.astype(jnp.float32) * c + _rotate_half(q.astype(jnp.float32)) * s).astype(dt)
        k = (k.astype(jnp.float32) * c + _rotate_half(k.astype(jnp.float32)) * s).astype(dt)
        logits = jnp.einsum("qhd,khd->hqk", q, k) * hd ** -0.5
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(dt)
        out = jnp.einsum("hqk,khd->qhd", probs, val).reshape(n, d)
        return DenseGeneral(d, use_bias=True, dtype=dt, name="out")(out)


class VisionBlock(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, cos, sin):
        v, dt = self.cfg.vision, self.cfg.dtype
        x = x + VisionAttention(self.cfg, name="attn")(
            LayerNorm(v.layer_norm_epsilon, dt, name="ln1")(x), cos, sin)
        y = LayerNorm(v.layer_norm_epsilon, dt, name="ln2")(x)
        y = DenseGeneral(v.intermediate_size, use_bias=True, dtype=dt, name="mlp")(y)
        y = jax.nn.gelu(y, approximate=True)                            # gelu_pytorch_tanh
        y = DenseGeneral(v.hidden_size, use_bias=True, dtype=dt, name="mlp_out")(y)
        return x + y


class VisionPatchMerger(nn.Module):
    """Merge each spatial_merge_size**2 block of patches: LN -> Linear -> GELU -> Linear.
    `use_postshuffle_norm` (DeepStack mergers) normalizes the merged vector."""
    cfg: Config
    use_postshuffle_norm: bool = False

    @nn.compact
    def __call__(self, hidden):                                         # [N, D]
        v, dt = self.cfg.vision, self.cfg.dtype
        merged = v.hidden_size * v.spatial_merge_size ** 2
        ln = LayerNorm(v.layer_norm_epsilon, dt, name="ln_q")
        if self.use_postshuffle_norm:
            hidden = ln(hidden.reshape(-1, merged))
        else:
            hidden = ln(hidden).reshape(-1, merged)
        hidden = DenseGeneral(merged, use_bias=True, dtype=dt, name="mlp_0")(hidden)
        hidden = jax.nn.gelu(hidden, approximate=False)                 # nn.GELU() (erf)
        return DenseGeneral(v.out_hidden_size, use_bias=True, dtype=dt, name="mlp_2")(hidden)


class VisionEncoder(nn.Module):
    """Patch embed + interpolated position embedding + blocks; DeepStack taps are
    merged by their own post-shuffle-norm mergers (merger_{i})."""
    cfg: Config

    @nn.compact
    def __call__(self, pixel_patches, pos_ids, interp_idx, interp_wts):
        v = self.cfg.vision
        x = VisionPatchEmbed(self.cfg, name="patch_embed")(pixel_patches)
        pos = VisionPosEmbedInterpolate(self.cfg, name="pos_embed_interpolate")(interp_idx,
                                                                                interp_wts)
        x = x + pos.astype(x.dtype)
        cos, sin = vision_rope_cos_sin(self.cfg, pos_ids)
        deep_feats = []
        for i in range(v.depth):
            x = VisionBlock(self.cfg, name=f"blocks_{i}")(x, cos, sin)
            if i in v.deepstack_visual_indexes:
                j = v.deepstack_visual_indexes.index(i)
                deep_feats.append(VisionPatchMerger(self.cfg, use_postshuffle_norm=True,
                                                    name=f"merger_{j}")(x))
        return x, deep_feats


class VisionProjector(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden):
        return VisionPatchMerger(self.cfg, use_postshuffle_norm=False, name="merger")(hidden)


class VisionTower(nn.Module):
    """vision_encoder phase: pixel patches and the host tables (patch positions [N, 2],
    position-embedding interpolation indices / weights [N, 4]) -> (image embeddings
    [n_img, D_text], DeepStack features [n_deepstack, n_img, D_text])."""
    cfg: Config

    @nn.compact
    def __call__(self, pixel_patches, pos_ids, interp_idx, interp_wts):
        x, deep_feats = VisionEncoder(self.cfg, name="vision_encoder")(
            pixel_patches, pos_ids, interp_idx, interp_wts)
        image_embeds = VisionProjector(self.cfg, name="vision_projector")(x)
        return image_embeds, jnp.stack(deep_feats, axis=0)


# --------------------------------------------------------------------------
# Text decoder
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


class MRoPE(nn.Module):
    """Interleaved multimodal RoPE (embeddings.py Qwen3OmniMoeThinkerTextRotaryEmbedding):
    frequency j takes its angle from axis T, H or W in the pattern THWTHW...; the
    trailing frequencies (beyond 3 * section) stay T."""
    cfg: TextConfig

    def __call__(self, inputs, position_ids):      # inputs [B, S, N, H]; position_ids [3, B, S]
        c = self.cfg
        inv_freq = jnp.asarray(rope_inv_freq(c.rope_max_timescale, c.head_dim))
        freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq      # [3, B, S, hd/2]
        # Static axis of every frequency (MaxText: freqs_t.at[..., offset:3*sec:3].set(...));
        # a select instead of a strided scatter -- same values.
        axis_of = np.zeros(c.head_dim // 2, np.int32)
        for dim_idx, offset in enumerate((1, 2), start=1):                  # H, W
            axis_of[offset:c.mrope_section[dim_idx] * 3:3] = dim_idx
        axis_of = jnp.asarray(axis_of)
        freqs_t = jnp.where(axis_of == 1, freqs[1], jnp.where(axis_of == 2, freqs[2], freqs[0]))
        emb = jnp.concatenate([freqs_t, freqs_t], axis=-1)
        cos = jnp.cos(emb).astype(inputs.dtype)[:, :, None, :]
        sin = jnp.sin(emb).astype(inputs.dtype)[:, :, None, :]
        return inputs * cos + _rotate_half(inputs) * sin


def fp8_dense(features: int, dtype: str, name: str) -> q.QDense:
    """Text linear of the official FP8 checkpoint: E4M3 weights in 128x128 blocks,
    dynamic 1x128 activation groups (`quant.block_scaled_dot`), no bias."""
    return q.QDense(features, q.FP8_BLOCK128, use_bias=False, dtype=dtype, name=name)


class Attention(nn.Module):
    """GQA dot-product attention with per-head q/k RMSNorm, M-RoPE and a static KV cache."""
    cfg: TextConfig
    dtype: str

    def update_kv_caches(self, key, value, kv_cache, cache_index):
        idx = (jnp.int32(0), cache_index, jnp.int32(0), jnp.int32(0))
        cached_key = lax.dynamic_update_slice(kv_cache["key"], key.astype(kv_cache["key"].dtype), idx)
        cached_value = lax.dynamic_update_slice(kv_cache["value"], value.astype(kv_cache["value"].dtype), idx)
        return cached_key, cached_value

    def generate_attention_mask(self, slot_positions, kv_len):
        """Causal over cache slots: slot s is visible to query t iff s <= slot(t)."""
        kv_positions = jnp.arange(kv_len, dtype=jnp.int32)
        mask = kv_positions[jnp.newaxis, :] <= slot_positions[:, :, jnp.newaxis]   # [b, t, s]
        return mask[:, jnp.newaxis, jnp.newaxis, :, :]

    def qk_product(self, query, key):
        b, t, n, d = query.shape
        n_kv = key.shape[-2]
        query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
        return jnp.einsum("btkgd,bskd->bkgts", query, key)

    def wv_product(self, attn_weights, value):
        out = jnp.einsum("bkgts,bskd->btkgd", attn_weights, value)
        b, t, n_kv, g, d = out.shape
        return jnp.reshape(out, (b, t, n_kv * g, d))

    def apply_attention_dot(self, query, key, value, slot_positions):
        attn_weights = self.qk_product(query, key) * self.cfg.head_dim ** -0.5
        mask = self.generate_attention_mask(slot_positions, key.shape[1])
        attn_weights = jnp.where(mask, attn_weights, DEFAULT_MASK_VALUE)
        probs = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(query.dtype)
        return self.wv_product(probs, value)

    @nn.compact
    def __call__(self, x, position_ids, slot_positions, kv_cache, cache_index):
        c, dt = self.cfg, self.dtype
        b, t, _ = x.shape
        # FP8 block-128 W8A8 (official checkpoint): fp8 kernels [K, N] + [K/128, N/128] scales.
        query = fp8_dense(c.num_query_heads * c.head_dim, dt, "query")(x)
        key = fp8_dense(c.num_kv_heads * c.head_dim, dt, "key")(x)
        value = fp8_dense(c.num_kv_heads * c.head_dim, dt, "value")(x)
        query = query.reshape(b, t, c.num_query_heads, c.head_dim)
        key = key.reshape(b, t, c.num_kv_heads, c.head_dim)
        value = value.reshape(b, t, c.num_kv_heads, c.head_dim)
        query = RMSNorm(c.normalization_layer_epsilon, dt, name="query_norm")(query)
        key = RMSNorm(c.normalization_layer_epsilon, dt, name="key_norm")(key)
        rope = MRoPE(c, name="rotary_embedding")
        query = rope(query, position_ids)
        key = rope(key, position_ids)
        cached_key, cached_value = self.update_kv_caches(key, value, kv_cache, cache_index)
        out = self.apply_attention_dot(query, cached_key, cached_value, slot_positions)
        out = fp8_dense(c.emb_dim, dt, "out")(out.reshape(b, t, -1))
        return out, {"key": cached_key, "value": cached_value}


class RoutedMoE(nn.Module):
    """Qwen3 MoE: softmax router -> top-k (renormalized) -> SwiGLU experts.

    Static dispatch with capacity C = T tokens per expert (exact, no drops;
    same scheme as models/gemma4): each expert gets a fixed buffer of T slots; a
    token routed to expert e takes slot (#earlier tokens routed to e). Dispatch
    gather [E, C, D] -> expert FFN batched over experts -> combine by gathering
    each token's k expert outputs and summing them in ascending expert-id order
    in the activation dtype (the order and precision of HF's per-expert
    `index_add_` loop).
    """
    cfg: TextConfig
    dtype: str

    @nn.compact
    def __call__(self, inputs):
        c, dt = self.cfg, self.dtype
        b, s, d = inputs.shape
        t, e, k, f = b * s, c.num_experts, c.num_experts_per_tok, c.moe_mlp_dim
        x = inputs.reshape(t, d)

        # Router (GateLogit): fp32 softmax over all experts, top-k, renormalize.
        logits = DenseGeneral(e, dtype=dt, name="gate")(x)                      # [T, E]
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        top_w, top_i = lax.top_k(probs, k)
        top_w = (top_w / top_w.sum(axis=-1, keepdims=True)).astype(logits.dtype)
        top_i, top_w = lax.sort((top_i, top_w), dimension=-1, num_keys=1)      # ascending expert id

        # Dispatch table [E, C]: token held by each slot (T = empty slot).
        sel = jax.nn.one_hot(top_i, e, dtype=jnp.int32).sum(axis=1)            # [T, E]
        slot = jnp.where(sel > 0, jnp.cumsum(sel, axis=0) - 1, t)
        tok = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[:, None], (t, e))
        eid = jnp.broadcast_to(jnp.arange(e, dtype=jnp.int32)[None, :], (t, e))
        table = jnp.full((e, t), t, jnp.int32).at[eid, slot].set(tok, mode="drop")
        xs = jnp.take(x, table, axis=0, mode="fill", fill_value=0)              # [E, C, D]

        # Experts (fused gate/up kernel [E, D, 2F]; down [E, F, D]): FP8 E4M3 with one fp32
        # scale per 128x128 block [E, K/128, N/128] (checkpoint gate_up_proj / down_proj and
        # their _scale_inv); dynamic 1x128 activation groups, batched over the experts.
        bk, bn = q.FP8_BLOCK128.weight_block
        wi = self.param("wi", q.rand_q("fp8"), (e, d, 2 * f), q.FP8)
        wi_scale = self.param("wi_scale", nn.initializers.constant(1e-3),
                              (e, -(-d // bk), -(-2 * f // bn)), jnp.float32)
        wo = self.param("wo", q.rand_q("fp8"), (e, f, d), q.FP8)
        wo_scale = self.param("wo_scale", nn.initializers.constant(1e-3),
                              (e, -(-f // bk), -(-d // bn)), jnp.float32)
        gate_up = q.block_scaled_dot(xs, wi, wi_scale, (bk, bn), bk, batch_dims=1).astype(dt)
        h = nn.silu(gate_up[..., :f]) * gate_up[..., f:]
        ys = q.block_scaled_dot(h, wo, wo_scale, (bk, bn), bk, batch_dims=1).astype(dt)  # [E, C, D]

        # Combine: token t's j-th expert output sits at (top_i[t, j], slot[t, top_i[t, j]]).
        tok_slot = jnp.take_along_axis(slot, top_i, axis=1)                    # [T, k]
        picked = jnp.take(ys.reshape(e * t, d), top_i * t + tok_slot, axis=0)  # [T, k, D]
        weighted = picked * top_w[..., None]
        out = weighted[:, 0]
        for j in range(1, k):
            out = out + weighted[:, j]
        return out.reshape(b, s, d)


class Qwen3MoeDecoderLayer(nn.Module):
    cfg: TextConfig
    dtype: str

    @nn.compact
    def __call__(self, inputs, position_ids, slot_positions, kv_cache, cache_index):
        c, dt = self.cfg, self.dtype
        lnx = RMSNorm(c.normalization_layer_epsilon, dt, name="pre_self_attention_layer_norm")(inputs)
        attention_lnx, kv_cache = Attention(c, dt, name="self_attention")(
            lnx, position_ids, slot_positions, kv_cache, cache_index)
        intermediate_inputs = inputs + attention_lnx
        hidden_states = RMSNorm(c.normalization_layer_epsilon, dt,
                                name="post_self_attention_layer_norm")(intermediate_inputs)
        mlp_lnx = RoutedMoE(c, dt, name="moe_block")(hidden_states)
        return intermediate_inputs + mlp_lnx, kv_cache


def merge_mm_embeddings(text_embeddings, image_embeds, image_start):
    """Place the image embeddings on their token span [image_start, image_start + n_img);
    image_start is an int32 scalar from the host (the prompt layout)."""
    return lax.dynamic_update_slice(text_embeddings, image_embeds[None].astype(text_embeddings.dtype),
                                    (0, image_start, 0))


def deepstack_process(hidden_states, visual_embeds, image_start):
    """hidden[:, span] += visual_embeds on the image-token span."""
    n = visual_embeds.shape[0]
    span = lax.dynamic_slice_in_dim(hidden_states, image_start, n, axis=1)
    return lax.dynamic_update_slice(hidden_states, span + visual_embeds[None].astype(span.dtype),
                                    (0, image_start, 0))


class Decoder(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, y, position_ids, slot_positions, kv_caches, cache_index,
                 deepstack_visual_embeds=None, image_start=None):
        c, dt = self.cfg.text, self.cfg.dtype
        new_caches = {}
        for lyr in range(c.num_decoder_layers):
            name = f"layers_{lyr}"
            y, new_caches[name] = Qwen3MoeDecoderLayer(c, dt, name=name)(
                y, position_ids, slot_positions, kv_caches[name], cache_index)
            if deepstack_visual_embeds is not None and lyr < deepstack_visual_embeds.shape[0]:
                y = deepstack_process(y, deepstack_visual_embeds[lyr], image_start)
        y = RMSNorm(c.normalization_layer_epsilon, dt, name="decoder_norm")(y)
        y = y[:, -1:, :]                     # last-position logits only (HF logits_to_keep=1)
        logits = DenseGeneral(c.vocab_size, dtype=dt, name="logits_dense")(y)
        return logits, new_caches


class Transformer(nn.Module):
    """prefill / decode_step network call.

    tokens [b, t] int32; position_ids [3, b, t] int32 (host-computed M-RoPE ids);
    cache_index int32 scalar = cache slot of tokens[:, 0]; image_embeds
    [n_img, D], deepstack_visual_embeds [n_ds, n_img, D] and image_start (int32 scalar,
    first image-token position) are prefill only.
    """
    cfg: Config

    @nn.compact
    def __call__(self, tokens, position_ids, kv_caches, cache_index, image_embeds=None,
                 deepstack_visual_embeds=None, image_start=None):
        c = self.cfg
        y = Embed(c.text.vocab_size, c.text.emb_dim, c.dtype, name="token_embedder")(tokens)
        if image_embeds is not None:
            y = merge_mm_embeddings(y, image_embeds, image_start)
        slots = cache_index + jnp.arange(tokens.shape[1], dtype=jnp.int32)
        slots = jnp.broadcast_to(slots[jnp.newaxis, :], tokens.shape)
        return Decoder(c, name="decoder")(y, position_ids, slots, kv_caches, cache_index,
                                          deepstack_visual_embeds, image_start)
