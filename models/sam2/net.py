"""SAM 2.1 Hiera-L as Flax linen modules, INT8 W8A8: a port of facebookresearch/sam2
`sam2/modeling` (commit 2b90b9f).

Every module mirrors the torch module of the same name and keeps the torch layouts
(NCHW feature maps, OIHW conv kernels, [out, in] linear weights, seq-first memory
tensors), so the parameter tree is the torch state_dict with `.<i>` list indices
written `_<i>` (plus the quantization scales).

The model is split into five parameter groups, one per network invocation of the video
predictor (models/sam2/model.py builds the phases):

  SAM2ImageEncoder        image_encoder (Hiera trunk + FPN neck) and the mask decoder's
                          conv_s0 / conv_s1 (SAM2Base.forward_image)
  SAM2CondFrame           no_mem_embed (initial conditioning frame, directly added)
  SAM2MemoryConditioning  memory bank assembly + memory_attention
                          (SAM2Base._prepare_memory_conditioned_features)
  SAM2Heads               prompt encoder + mask decoder + object pointer
                          (SAM2Base._forward_sam_heads)
  SAM2MemoryEncoding      memory_encoder (SAM2Base._encode_new_memory)

Quantization (models/sam2/recipe.py): no official or production INT8 SAM 2.1 Hiera-L exists
(Meta ships fp32 weights + bf16 autocast; Qualcomm AI Hub's w8a8 release is the Tiny model,
encoder and decoder only; NVIDIA's documented path is fp16), so the edge INT8 W8A8 fallback
applies, in TensorRT INT8-engine form:
  * every conv and every Linear is INT8 (`QLinear`, `QConv2d`, `QConvTransposeD2S` below):
    weights int8 per-output-channel symmetric (`kernel_scale`), activations int8 per-tensor
    static symmetric (`input_scale`, or one `ActQuant` when a tensor feeds several INT8
    layers), i8 x i8 -> i32, fp32 epilogue (scale, bias) written as fp16;
  * the mask decoder's ConvTranspose2d (k2 s2, non-overlapping) is a 1x1 conv to 4 * Cout
    channels + depth-to-space (NVIDIA's deconv rewrite for 3D-UNet; the same math);
  * float carve-outs: attention cores QK^T / AV and the mask product hyper_in @ upscaled
    (fp16 operands, fp32 accumulation), softmax fp32; LayerNorm / LayerNorm2d (fp32
    statistics, fp16 out), GELU, residual adds fp16; RoPE, sine / random-Fourier position
    encodings (the coords @ gaussian GEMM included), the bicubic pos-embed and the mask
    interpolations, mask selection, sigmoids and score logits fp32;
  * tensors passed between phases are fp16 (vision features / pos, the memory bank:
    maskmem_features, maskmem_pos_enc, object pointers); masks and scores stay fp32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from models import quant

F32 = jnp.float32
F16 = jnp.float16
NO_OBJ_SCORE = -1024.0          # sam2_base.NO_OBJ_SCORE

# Constants hard-coded in SAM2Base._build_sam_heads / MaskDecoder / TwoWayTransformer.
NUM_MULTIMASK_OUTPUTS = 3
TWO_WAY_DEPTH = 2
TWO_WAY_HEADS = 8
TWO_WAY_MLP_DIM = 2048
ATTN_DOWNSAMPLE_RATE = 2
IOU_HEAD_DEPTH = 3
IOU_HEAD_HIDDEN = 256


@dataclass(frozen=True)
class Config:
    """Defaults are configs/sam2.1/sam2.1_hiera_l.yaml (+ SAM2Base / Hiera defaults)."""
    # Hiera trunk
    embed_dim: int = 144
    num_heads: int = 2
    stages: tuple = (2, 6, 36, 4)
    global_att_blocks: tuple = (23, 33, 43)
    window_pos_embed_bkg_spatial_size: tuple = (7, 7)
    window_spec: tuple = (8, 4, 16, 8)
    q_pool: int = 3
    q_stride: tuple = (2, 2)
    dim_mul: float = 2.0
    head_mul: float = 2.0
    mlp_ratio: float = 4.0
    # FPN neck
    d_model: int = 256
    fpn_top_down_levels: tuple = (2, 3)
    scalp: int = 1
    # memory attention
    mem_attn_layers: int = 4
    mem_attn_ffn: int = 2048
    rope_theta: float = 10000.0
    # memory encoder
    mem_dim: int = 64
    fuser_layers: int = 2
    cx_kernel: int = 7
    mask_ds_kernel: int = 3
    mask_ds_stride: int = 2
    mask_ds_padding: int = 1
    mask_ds_total_stride: int = 16
    # SAM2Base
    image_size: int = 1024
    backbone_stride: int = 16
    num_maskmem: int = 7
    max_obj_ptrs_in_encoder: int = 16
    sigmoid_scale_for_mem_enc: float = 20.0
    sigmoid_bias_for_mem_enc: float = -10.0
    # scenario (video predictor): video length, one object (batch 1)
    num_frames: int = 67
    batch: int = 1
    # Memory-bank sizes of the steady-state `memory_attention` phase (spatial memories,
    # object pointers), if not the full bank of the scenario (num_maskmem, min(num_frames,
    # max_obj_ptrs_in_encoder)): their own symbols, as the bank fills frame by frame (the
    # warm-up phases are size variants of the steady state).
    bank_mems: int | None = None
    bank_ptrs: int | None = None

    @property
    def feat_size(self) -> int:          # stride-16 feature map side (64 at 1024^2)
        return self.image_size // self.backbone_stride

    @property
    def channel_list(self) -> tuple:     # Hiera.channel_list (deepest stage first)
        dims = [int(self.embed_dim * self.dim_mul ** i) for i in range(len(self.stages))]
        return tuple(reversed(dims))


# ----------------------------------------------------------------------------
# primitives (torch layouts and parameter names), INT8 W8A8
# ----------------------------------------------------------------------------

_init = nn.initializers.normal(0.02)


def _kernel_scale(mod: nn.Module, n_out: int, fan_in: int):
    """Per-output-channel symmetric int8 weight scale (placeholder value; real scales
    are amax / 127 of the converted weights)."""
    return mod.param("kernel_scale", quant.const(3.0 / math.sqrt(max(fan_in, 1)) / quant.QMAX["int8"]),
                     (n_out,), F32)


def _quant_input(mod: nn.Module, x, x_scale):
    """(x_q, scale): an int8 tensor comes with its producer's scale; otherwise the layer
    owns a calibrated per-tensor `input_scale` and quantizes its input."""
    if x_scale is not None:
        assert x.dtype == quant.INT8, x.dtype
        return x, x_scale
    s = mod.param("input_scale", quant.const(4.0 / quant.QMAX["int8"]), (), F32)
    return quant.quantize(x, s, "int8"), s


class QLinear(nn.Module):
    """nn.Linear, INT8: weight int8 [out, in], i8 x i8 -> i32, fp32 epilogue -> `dtype`."""
    out_features: int
    dtype: object = F16

    @nn.compact
    def __call__(self, x, x_scale=None):
        k = x.shape[-1]
        w = self.param("weight", quant.rand_q("int8"), (self.out_features, k), quant.INT8)
        ws = _kernel_scale(self, self.out_features, k)
        b = self.param("bias", nn.initializers.zeros, (self.out_features,), F32)
        xq, xs = _quant_input(self, x, x_scale)
        acc = lax.dot_general(xq, w, (((x.ndim - 1,), (1,)), ((), ())),
                              preferred_element_type=jnp.int32)
        return (acc.astype(F32) * (xs * ws) + b).astype(self.dtype)


class QConv2d(nn.Module):
    """nn.Conv2d on NCHW with an OIHW kernel, INT8 (as QLinear)."""
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    groups: int = 1
    dtype: object = F16

    @nn.compact
    def __call__(self, x, x_scale=None):
        k, p, cin = self.kernel_size, self.padding, x.shape[1] // self.groups
        w = self.param("weight", quant.rand_q("int8"), (self.out_channels, cin, k, k), quant.INT8)
        ws = _kernel_scale(self, self.out_channels, cin * k * k)
        b = self.param("bias", nn.initializers.zeros, (self.out_channels,), F32)
        xq, xs = _quant_input(self, x, x_scale)
        acc = lax.conv_general_dilated(xq, w, (self.stride, self.stride), [(p, p), (p, p)],
                                       dimension_numbers=("NCHW", "OIHW", "NCHW"),
                                       feature_group_count=self.groups,
                                       preferred_element_type=jnp.int32)
        y = acc.astype(F32) * (xs * ws)[None, :, None, None] + b[None, :, None, None]
        return y.astype(self.dtype)


class QConvTransposeD2S(nn.Module):
    """nn.ConvTranspose2d(k=s, stride=s, padding=0), INT8, as a 1x1 conv to s*s*Cout
    channels + depth-to-space. Kernel == stride, so every output pixel receives exactly one
    tap: out[co, h*s + dy, w*s + dx] = sum_ci x[ci, h, w] W_torch[ci, co, dy, dx] + b[co].
    `weight` [s*s*Cout, Cin, 1, 1] with row co*s*s + dy*s + dx = W_torch[:, co, dy, dx]
    (NVIDIA's deconv -> conv + pixel-shuffle rewrite); per-channel scales per row."""
    out_channels: int
    stride: int
    dtype: object = F16

    @nn.compact
    def __call__(self, x, x_scale=None):
        s, co, cin = self.stride, self.out_channels, x.shape[1]
        w = self.param("weight", quant.rand_q("int8"), (co * s * s, cin, 1, 1), quant.INT8)
        ws = _kernel_scale(self, co * s * s, cin)
        b = self.param("bias", nn.initializers.zeros, (co,), F32)
        xq, xs = _quant_input(self, x, x_scale)
        acc = lax.conv_general_dilated(xq, w, (1, 1), [(0, 0), (0, 0)],
                                       dimension_numbers=("NCHW", "OIHW", "NCHW"),
                                       preferred_element_type=jnp.int32)
        y = acc.astype(F32) * (xs * ws)[None, :, None, None]
        n, _, h, wd = y.shape
        y = y.reshape(n, co, s, s, h, wd).transpose(0, 1, 4, 2, 5, 3).reshape(n, co, h * s, wd * s)
        return (y + b[None, :, None, None]).astype(self.dtype)


class LayerNorm(nn.Module):
    """nn.LayerNorm over the last dim (fp32 statistics, fp16 output)."""
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        c = x.shape[-1]
        w = self.param("weight", nn.initializers.ones, (c,))
        b = self.param("bias", nn.initializers.zeros, (c,))
        x = x.astype(F32)
        mu = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mu), axis=-1, keepdims=True)
        return ((x - mu) * lax.rsqrt(var + self.eps) * w + b).astype(F16)


class LayerNorm2d(nn.Module):
    """sam2_utils.LayerNorm2d: LayerNorm over the channel dim of NCHW (fp32 statistics,
    fp16 output)."""
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        c = x.shape[1]
        w = self.param("weight", nn.initializers.ones, (c,))
        b = self.param("bias", nn.initializers.zeros, (c,))
        x = x.astype(F32)
        u = jnp.mean(x, axis=1, keepdims=True)
        s = jnp.mean(jnp.square(x - u), axis=1, keepdims=True)
        x = (x - u) / jnp.sqrt(s + self.eps)
        return (w[:, None, None] * x + b[:, None, None]).astype(F16)


class Embedding(nn.Module):
    """nn.Embedding whose `.weight` is read directly (SAM tokens / prompt embeddings):
    `__call__(use)` returns use(weight), so the ops that read it carry this module's scope."""
    num: int
    dim: int

    @nn.compact
    def __call__(self, use):
        return use(self.param("weight", _init, (self.num, self.dim)))


def gelu(x):
    return jax.nn.gelu(x, approximate=False)      # nn.GELU() (erf)


def relu(x):
    return jax.nn.relu(x)


class MLP(nn.Module):
    """sam2_utils.MLP, INT8 Linears (each owns its input scale); `out_dtype` of the last
    layer (fp16, or fp32 for score heads); sigmoid in fp32."""
    hidden_dim: int
    output_dim: int
    num_layers: int
    act: str = "relu"
    sigmoid_output: bool = False
    out_dtype: object = F16

    def setup(self):
        h = [self.hidden_dim] * (self.num_layers - 1)
        dt = [F16] * (self.num_layers - 1) + [self.out_dtype]
        self.layers = [QLinear(k, d) for k, d in zip(h + [self.output_dim], dt)]

    def __call__(self, x, x_scale=None):
        act = {"relu": relu, "gelu": gelu}[self.act]
        for i, layer in enumerate(self.layers):
            x = layer(x, x_scale if i == 0 else None)
            x = act(x) if i < self.num_layers - 1 else x
        return jax.nn.sigmoid(x.astype(F32)) if self.sigmoid_output else x


def sdpa(q_, k, v):
    """F.scaled_dot_product_attention on [B, heads, L, D] (no mask, no dropout): fp16
    operands, fp32 accumulation and softmax, fp16 output."""
    s = jnp.einsum("bhqd,bhkd->bhqk", q_.astype(F16), k.astype(F16), preferred_element_type=F32)
    p = jax.nn.softmax(s * (1.0 / math.sqrt(q_.shape[-1])), axis=-1).astype(F16)
    return jnp.einsum("bhqk,bhkd->bhqd", p, v.astype(F16), preferred_element_type=F32).astype(F16)


# ----------------------------------------------------------------------------
# static resampling (F.interpolate, CPU kernel index/weight rules), gather-free
# ----------------------------------------------------------------------------

def _cubic_coeffs(t):
    a = np.float32(-0.75)                         # UpSample.h get_cubic_upsample_coefficients
    one = np.float32(1.0)
    cc1 = lambda x: ((a + 2) * x - (a + 3)) * x * x + 1
    cc2 = lambda x: ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return [cc2(t + one), cc1(t), cc1(one - t), cc2(one - t + one)]


def interp_taps(n_in: int, n_out: int, mode: str):
    """(indices, weights) per tap, as ATen's upsample kernels compute them for
    align_corners=False and an explicit output size (scale = n_in / n_out in fp32)."""
    i = np.arange(n_out, dtype=np.float32)
    if mode == "nearest":                         # nearest_idx: output size 2x input
        return [(np.arange(n_out) // (n_out // n_in), None)]
    scale = np.float32(n_in) / np.float32(n_out)
    src = scale * (i + np.float32(0.5)) - np.float32(0.5)
    if mode == "bilinear":
        src = np.maximum(src, np.float32(0.0))    # area_pixel_compute_source_index(cubic=False)
    idx = np.minimum(np.floor(src).astype(np.int64), n_in - 1)       # guard_index_and_lambda
    lam = np.clip(src - idx.astype(np.float32), np.float32(0), np.float32(1)).astype(np.float32)
    if mode == "bilinear":
        idx1 = idx + (idx < n_in - 1)
        return [(idx, (np.float32(1) - lam).astype(np.float32)), (idx1, lam)]
    return [(np.clip(idx - 1 + j, 0, n_in - 1), w.astype(np.float32))
            for j, w in enumerate(_cubic_coeffs(lam))]


def _shift(x, c: int, axis: int):
    """y[i] = x[clip(i + c, 0, n - 1)] along `axis`, from slices (no gather)."""
    n = x.shape[axis]
    edge = lambda lo, cnt: jnp.broadcast_to(
        lax.slice_in_dim(x, lo, lo + 1, axis=axis),
        x.shape[:axis] + (cnt,) + x.shape[axis + 1:])
    return jnp.concatenate([edge(0, max(-c, 0)), lax.slice_in_dim(x, max(c, 0), n + min(c, 0), axis=axis),
                            edge(n - 1, max(c, 0))], axis=axis)


def take_static(x, idx, axis: int, care=None):
    """x[..., idx, ...] for a static non-decreasing index vector, built from slices.

    Integer upsampling (n_out = f * n_in) whose phase p reads x[clip(k + c_p)] becomes
    f shifted copies interleaved by stack+reshape; anything else (bicubic 7 -> 256) is
    a run-length concatenation of broadcast slices. `care` marks the outputs whose
    index matters (a tap with weight 0 may read any element)."""
    idx = np.asarray(idx)
    n, m = x.shape[axis], idx.size
    care = np.ones(m, bool) if care is None else care
    f = m // n
    shifts = [int(idx[p::f][n // 2]) - n // 2 for p in range(f)] if m == f * n else None
    periodic = shifts is not None and all(
        np.array_equal(np.where(care[p::f], idx[p::f], 0),
                       np.where(care[p::f], np.clip(np.arange(n) + c, 0, n - 1), 0))
        for p, c in enumerate(shifts))
    if periodic:
        y = jnp.stack([_shift(x, c, axis) for c in shifts], axis=axis + 1)
        return y.reshape(x.shape[:axis] + (m,) + x.shape[axis + 1:])
    starts = np.flatnonzero(np.diff(idx, prepend=-1))
    counts = np.diff(np.append(starts, m))
    return jnp.concatenate([jnp.broadcast_to(
        lax.slice_in_dim(x, int(idx[s]), int(idx[s]) + 1, axis=axis),
        x.shape[:axis] + (int(c),) + x.shape[axis + 1:]) for s, c in zip(starts, counts)], axis=axis)


def resize_axis(x, axis: int, n_out: int, mode: str):
    taps = interp_taps(x.shape[axis], n_out, mode)
    shape = [1] * x.ndim
    shape[axis] = n_out
    out = None
    for idx, w in taps:
        t = take_static(x, idx, axis, None if w is None else w != 0)
        t = t if w is None else t * jnp.asarray(w).reshape(shape)
        out = t if out is None else out + t
    return out


def interpolate(x, size, mode: str):
    """F.interpolate(x NCHW, size, mode, align_corners=False): width pass, then height
    pass (the order of ATen's separable CPU kernel)."""
    x = resize_axis(x, 3, size[1], mode)
    return resize_axis(x, 2, size[0], mode)


# ----------------------------------------------------------------------------
# position encodings (parameter-free)
# ----------------------------------------------------------------------------

def sine_pe(b: int, h: int, w: int, num_pos_feats: int, temperature=10000.0):
    """position_encoding.PositionEmbeddingSine(num_pos_feats, normalize=True)._pe -> [B, C, H, W]."""
    npf = num_pos_feats // 2
    scale, eps = 2 * math.pi, 1e-6
    y = jnp.broadcast_to(jnp.arange(1, h + 1, dtype=F32)[:, None], (h, w))
    x = jnp.broadcast_to(jnp.arange(1, w + 1, dtype=F32)[None, :], (h, w))
    y = y / (y[-1:, :] + eps) * scale
    x = x / (x[:, -1:] + eps) * scale
    dim_t = jnp.arange(npf, dtype=F32)
    dim_t = temperature ** (2 * jnp.floor(dim_t / 2) / npf)
    px, py = x[:, :, None] / dim_t, y[:, :, None] / dim_t
    px = jnp.stack((jnp.sin(px[:, :, 0::2]), jnp.cos(px[:, :, 1::2])), axis=3).reshape(h, w, npf)
    py = jnp.stack((jnp.sin(py[:, :, 0::2]), jnp.cos(py[:, :, 1::2])), axis=3).reshape(h, w, npf)
    pos = jnp.concatenate((py, px), axis=2).transpose(2, 0, 1)
    return jnp.broadcast_to(pos[None], (b,) + pos.shape)


def get_1d_sine_pe(pos_inds, dim: int, temperature=10000.0):
    """sam2_utils.get_1d_sine_pe."""
    pe_dim = dim // 2
    dim_t = jnp.arange(pe_dim, dtype=F32)
    dim_t = temperature ** (2 * jnp.floor(dim_t / 2) / pe_dim)
    pe = pos_inds[:, None] / dim_t
    return jnp.concatenate([jnp.sin(pe), jnp.cos(pe)], axis=-1)


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float):
    """position_encoding.compute_axial_cis as (cos, sin), each [end_x * end_y, dim // 2]."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 4)[: dim // 4].astype(F32) / dim))
    t = jnp.arange(end_x * end_y, dtype=F32)
    t_x, t_y = t % end_x, jnp.floor(t / end_x)
    ang = jnp.concatenate([t_x[:, None] * freqs[None], t_y[:, None] * freqs[None]], axis=-1)
    return jnp.cos(ang), jnp.sin(ang)


def _rotate(x, cos, sin):
    """view_as_real(view_as_complex(x) * freqs_cis) on [B, H, L, D]."""
    xr = x.reshape(x.shape[:-1] + (-1, 2))
    a, b = xr[..., 0], xr[..., 1]
    return jnp.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(x.shape)


def apply_rotary_enc(xq, xk, cos, sin, repeat_freqs_k: bool):
    xq_out = _rotate(xq, cos, sin)
    r = xk.shape[-2] // xq.shape[-2]
    if repeat_freqs_k:                 # tile the query freqs over the r memory frames
        cos, sin = jnp.tile(cos, (r, 1)), jnp.tile(sin, (r, 1))
    return xq_out, _rotate(xk, cos, sin)


# ----------------------------------------------------------------------------
# Hiera trunk (backbones/hieradet.py) + FPN neck (backbones/image_encoder.py)
# ----------------------------------------------------------------------------

def window_partition(x, ws: int):
    b, h, w, c = x.shape
    assert h % ws == 0 and w % ws == 0, "window padding never occurs for the SAM2 configs"
    x = x.reshape(b, h // ws, ws, w // ws, ws, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws, ws, c)


def window_unpartition(windows, ws: int, hw):
    h, w = hw
    b = windows.shape[0] // (h * w // ws // ws)
    x = windows.reshape(b, h // ws, w // ws, ws, ws, -1).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, w, -1)


def max_pool(x, q_stride):
    """nn.MaxPool2d(q_stride, q_stride) applied to BHWC (do_pool permutes around it)."""
    s = (1,) + tuple(q_stride) + (1,)
    return lax.reduce_window(x, jnp.array(-jnp.inf, x.dtype), lax.max, s, s, "VALID")


class PatchEmbed(nn.Module):
    embed_dim: int

    def setup(self):
        self.proj = QConv2d(self.embed_dim, 7, stride=4, padding=3)      # quantizes the fp32 frame

    def __call__(self, x):
        return self.proj(x).transpose(0, 2, 3, 1)


class MultiScaleAttention(nn.Module):
    dim_out: int
    num_heads: int
    q_stride: tuple | None

    def setup(self):
        self.qkv = QLinear(self.dim_out * 3)
        self.proj = QLinear(self.dim_out)

    def __call__(self, x, x_scale):          # x: int8 normed block input (windowed)
        b, h, w, _ = x.shape
        qkv = self.qkv(x, x_scale).reshape(b, h * w, 3, self.num_heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.q_stride:
            q = max_pool(q.reshape(b, h, w, -1), self.q_stride)
            h, w = q.shape[1:3]
            q = q.reshape(b, h * w, self.num_heads, -1)
        x = sdpa(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3))
        x = x.transpose(0, 2, 1, 3).reshape(b, h, w, -1)
        return self.proj(x)


class MultiScaleBlock(nn.Module):
    dim: int
    dim_out: int
    num_heads: int
    mlp_ratio: float
    q_stride: tuple | None
    window_size: int

    def setup(self):
        self.norm1 = LayerNorm(1e-6)
        # norm1's output feeds qkv and (at a stage start) the shortcut proj: one int8 tensor
        self.norm1_quant = quant.ActQuant()
        self.attn = MultiScaleAttention(self.dim_out, self.num_heads, self.q_stride)
        self.norm2 = LayerNorm(1e-6)
        self.mlp = MLP(int(self.dim_out * self.mlp_ratio), self.dim_out, 2, act="gelu")
        if self.dim != self.dim_out:
            self.proj = QLinear(self.dim_out)

    def __call__(self, x):
        shortcut = x
        x, xs, _ = self.norm1_quant(self.norm1(x))
        if self.dim != self.dim_out:
            shortcut = max_pool(self.proj(x, xs), self.q_stride)
        window_size = self.window_size
        if window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x = window_partition(x, window_size)
        x = self.attn(x, xs)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            h, w = shortcut.shape[1:3]
        if self.window_size > 0:
            x = window_unpartition(x, window_size, (h, w))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class Hiera(nn.Module):
    cfg: Config

    def setup(self):
        c = self.cfg
        depth = sum(c.stages)
        self.stage_ends = [sum(c.stages[:i]) - 1 for i in range(1, len(c.stages) + 1)]
        q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:c.q_pool]
        self.patch_embed = PatchEmbed(c.embed_dim)
        self.pos_embed = self.param("pos_embed", _init, (1, c.embed_dim) + tuple(c.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = self.param("pos_embed_window", _init,
                                           (1, c.embed_dim, c.window_spec[0], c.window_spec[0]))
        embed_dim, num_heads, cur_stage, blocks = c.embed_dim, c.num_heads, 1, []
        for i in range(depth):
            dim_out = embed_dim
            # the window size lags by a block: the first block of a stage uses the
            # previous stage's window (hieradet.Hiera.__init__)
            window_size = 0 if i in c.global_att_blocks else c.window_spec[cur_stage - 1]
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * c.dim_mul)
                num_heads = int(num_heads * c.head_mul)
                cur_stage += 1
            blocks.append(MultiScaleBlock(embed_dim, dim_out, num_heads, c.mlp_ratio,
                                          tuple(c.q_stride) if i in q_pool_blocks else None,
                                          window_size))
            embed_dim = dim_out
        self.blocks = blocks

    def _get_pos_embed(self, hw):
        pos = interpolate(self.pos_embed, hw, "bicubic")
        win = self.pos_embed_window
        pos = pos + jnp.tile(win, [x // y for x, y in zip(pos.shape, win.shape)])
        return pos.transpose(0, 2, 3, 1).astype(F16)          # computed in fp32

    def __call__(self, x):
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.stage_ends:          # return_interm_layers=True
                outputs.append(x.transpose(0, 3, 1, 2))
        return outputs


class ConvSeq(nn.Module):
    """FpnNeck.convs[i]: nn.Sequential(conv=Conv2d(dim, d_model, 1))."""
    d_model: int

    def setup(self):
        self.conv = QConv2d(self.d_model, 1)

    def __call__(self, x):
        return self.conv(x)


class FpnNeck(nn.Module):
    cfg: Config

    def setup(self):
        self.convs = [ConvSeq(self.cfg.d_model) for _ in self.cfg.channel_list]

    def __call__(self, xs):
        n = len(self.convs) - 1
        out, pos = [None] * (n + 1), [None] * (n + 1)
        prev = None
        for i in range(n, -1, -1):
            lateral = self.convs[n - i](xs[i])
            if i in self.cfg.fpn_top_down_levels and prev is not None:
                b, _, h, w = prev.shape
                top_down = interpolate(prev, (2 * h, 2 * w), "nearest")    # scale_factor=2.0
                prev = lateral + top_down
            else:
                prev = lateral
            out[i] = prev
            b, _, h, w = prev.shape
            pos[i] = sine_pe(b, h, w, self.cfg.d_model).astype(F16)     # computed in fp32
        return out, pos


class SAM2ImageEncoder(nn.Module):
    """SAM2Base.forward_image: ImageEncoder(trunk, neck, scalp) + conv_s0 / conv_s1 of the
    mask decoder (precomputed high-res features). Returns (backbone_fpn, vision_pos_enc)."""
    cfg: Config

    def setup(self):
        d = self.cfg.d_model
        self.trunk = Hiera(self.cfg)
        self.neck = FpnNeck(self.cfg)
        self.conv_s0 = QConv2d(d // 8, 1)
        self.conv_s1 = QConv2d(d // 4, 1)

    def __call__(self, img):
        features, pos = self.neck(self.trunk(img))
        features, pos = features[: -self.cfg.scalp], pos[: -self.cfg.scalp]
        features[0] = self.conv_s0(features[0])
        features[1] = self.conv_s1(features[1])
        return tuple(features), tuple(pos)


# ----------------------------------------------------------------------------
# SAM prompt encoder + mask decoder (modeling/sam/*)
# ----------------------------------------------------------------------------

class PositionEmbeddingRandom(nn.Module):
    """`__call__(size=)` is PositionEmbeddingRandom.forward (dense grid), `__call__(coords=)`
    its forward_with_coords. (Flax scopes non-__call__ methods as separate names, so
    every parameter-owning module is entered through __call__ only.)"""
    num_pos_feats: int

    @nn.compact
    def __call__(self, size=None, coords=None, image_size=None):
        g = self.param("positional_encoding_gaussian_matrix", nn.initializers.normal(1.0),
                       (2, self.num_pos_feats))
        if coords is None:
            h, w = size
            y = (jnp.arange(1, h + 1, dtype=F32)[:, None] - 0.5) / h     # grid.cumsum(0) - 0.5
            x = (jnp.arange(1, w + 1, dtype=F32)[None, :] - 0.5) / w
            coords = jnp.stack([jnp.broadcast_to(x, (h, w)), jnp.broadcast_to(y, (h, w))], axis=-1)
        else:
            coords = coords / jnp.asarray([image_size[1], image_size[0]], F32)
        coords = 2 * coords - 1                                     # _pe_encoding
        coords = coords @ g                                         # fp32 (PE math)
        coords = 2 * np.pi * coords
        pe = jnp.concatenate([jnp.sin(coords), jnp.cos(coords)], axis=-1)
        return pe.transpose(2, 0, 1) if size is not None else pe


class PromptEncoder(nn.Module):
    """Point prompts only: the mask path (mask_downscaling) is never taken.
    Returns (sparse, dense, get_dense_pe()); the dense PE is computed here (first, so it
    is the pe_layer call recorded as its activation) instead of by the caller."""
    embed_dim: int
    image_embedding_size: tuple
    input_image_size: tuple

    def setup(self):
        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)
        self.point_embeddings = [Embedding(1, self.embed_dim) for _ in range(4)]
        self.not_a_point_embed = Embedding(1, self.embed_dim)
        self.no_mask_embed = Embedding(1, self.embed_dim)

    def __call__(self, points):
        dense_pe = self.pe_layer(size=self.image_embedding_size)[None]      # get_dense_pe
        coords, labels = points
        bs = coords.shape[0]
        # _embed_points(pad=True): a padding point with label -1; labels become float
        coords = coords + 0.5
        coords = jnp.concatenate([coords, jnp.zeros((bs, 1, 2), F32)], axis=1)
        labels = jnp.concatenate([labels.astype(F32), -jnp.ones((bs, 1), F32)], axis=1)
        pe = self.pe_layer(coords=coords, image_size=self.input_image_size)
        lab = labels[..., None]
        pe = jnp.where(lab == -1, self.not_a_point_embed(lambda e: jnp.zeros_like(pe) + e), pe)
        for i in range(4):
            pe = jnp.where(lab == i, self.point_embeddings[i](lambda e: pe + e), pe)
        h, w = self.image_embedding_size
        dense = self.no_mask_embed(
            lambda e: jnp.broadcast_to(e.reshape(1, -1, 1, 1), (bs, self.embed_dim, h, w)))
        return pe.astype(F16), dense.astype(F16), dense_pe.astype(F16)


class Attention(nn.Module):
    """sam/transformer.Attention, INT8 projections. `shared`: which of q / k / v are the
    same tensor at the call site ("qkv", "qk" or ""); a shared input is quantized once
    (`in_quant`) and passed to its projections with that scale."""
    embedding_dim: int
    num_heads: int
    downsample_rate: int = 1
    shared: str = ""

    def setup(self):
        self.internal_dim = self.embedding_dim // self.downsample_rate
        self.q_proj = QLinear(self.internal_dim)
        self.k_proj = QLinear(self.internal_dim)
        self.v_proj = QLinear(self.internal_dim)
        self.out_proj = QLinear(self.embedding_dim)
        if self.shared:
            self.in_quant = quant.ActQuant()

    def _project(self, q_in, k_in, v_in):
        if self.shared:
            x, s, _ = self.in_quant(q_in)
            v = self.v_proj(x, s) if self.shared == "qkv" else self.v_proj(v_in)
            return self.q_proj(x, s), self.k_proj(x, s), v
        return self.q_proj(q_in), self.k_proj(k_in), self.v_proj(v_in)

    def _separate_heads(self, x):
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(0, 2, 1, 3)

    @staticmethod
    def _recombine_heads(x):
        b, h, n, c = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, n, h * c)

    def __call__(self, q, k, v):
        q, k, v = map(self._separate_heads, self._project(q, k, v))
        return self.out_proj(self._recombine_heads(sdpa(q, k, v)))


class RoPEAttention(Attention):
    """sam/transformer.RoPEAttention (2D axial RoPE; keys repeat the query freqs)."""
    rope_theta: float = 10000.0
    rope_k_repeat: bool = False
    feat_sizes: tuple = (64, 64)

    def __call__(self, q, k, v, num_k_exclude_rope: int = 0):
        q, k, v = map(self._separate_heads, self._project(q, k, v))
        assert self.feat_sizes[0] * self.feat_sizes[1] == q.shape[-2]
        assert q.shape[-2] == k.shape[-2] or self.rope_k_repeat
        cos, sin = compute_axial_cis(self.internal_dim // self.num_heads, self.feat_sizes[0],
                                     self.feat_sizes[1], self.rope_theta)
        num_k_rope = k.shape[-2] - num_k_exclude_rope
        q, k_rope = apply_rotary_enc(q, k[:, :, :num_k_rope], cos, sin, self.rope_k_repeat)
        # RoPE in fp32 (fp16 projections promoted by the fp32 cos / sin tables)
        k = jnp.concatenate([k_rope, k[:, :, num_k_rope:].astype(k_rope.dtype)], axis=2)
        return self.out_proj(self._recombine_heads(sdpa(q, k, v)))


class TwoWayAttentionBlock(nn.Module):
    embedding_dim: int
    num_heads: int
    mlp_dim: int
    skip_first_layer_pe: bool

    def setup(self):
        d, h = self.embedding_dim, self.num_heads
        self.self_attn = Attention(d, h, shared="qkv" if self.skip_first_layer_pe else "qk")
        self.norm1 = LayerNorm()
        self.cross_attn_token_to_image = Attention(d, h, ATTN_DOWNSAMPLE_RATE)
        self.norm2 = LayerNorm()
        self.mlp = MLP(self.mlp_dim, d, 2, act="relu")
        self.norm3 = LayerNorm()
        self.norm4 = LayerNorm()
        self.cross_attn_image_to_token = Attention(d, h, ATTN_DOWNSAMPLE_RATE)

    def __call__(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q, q, queries)
        queries = self.norm1(queries)
        q, k = queries + query_pe, keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q, k, keys))
        queries = self.norm3(queries + self.mlp(queries))
        q, k = queries + query_pe, keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(k, q, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    embedding_dim: int

    def setup(self):
        d = self.embedding_dim
        self.layers = [TwoWayAttentionBlock(d, TWO_WAY_HEADS, TWO_WAY_MLP_DIM, i == 0)
                       for i in range(TWO_WAY_DEPTH)]
        self.final_attn_token_to_image = Attention(d, TWO_WAY_HEADS, ATTN_DOWNSAMPLE_RATE)
        self.norm_final_attn = LayerNorm()

    def __call__(self, image_embedding, image_pe, point_embedding):
        b, c, h, w = image_embedding.shape
        image_embedding = image_embedding.reshape(b, c, h * w).transpose(0, 2, 1)
        image_pe = image_pe.reshape(b, c, h * w).transpose(0, 2, 1)
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        q, k = queries + point_embedding, keys + image_pe
        queries = self.norm_final_attn(queries + self.final_attn_token_to_image(q, k, keys))
        return queries, keys


class MaskDecoder(nn.Module):
    """sam/mask_decoder.MaskDecoder with use_high_res_features, pred_obj_scores(_mlp),
    iou sigmoid and use_multimask_token_for_obj_ptr (the SAM 2.1 config); conv_s0 /
    conv_s1 live in SAM2ImageEncoder (forward_image applies them)."""
    transformer_dim: int

    def setup(self):
        d = self.transformer_dim
        self.num_mask_tokens = NUM_MULTIMASK_OUTPUTS + 1
        self.transformer = TwoWayTransformer(d)
        self.iou_token = Embedding(1, d)
        self.mask_tokens = Embedding(self.num_mask_tokens, d)
        self.obj_score_token = Embedding(1, d)
        # ConvTranspose2d(k2, s2): 1x1 conv to 4 * Cout + depth-to-space (QConvTransposeD2S)
        self.output_upscaling_0 = QConvTransposeD2S(d // 4, 2)
        self.output_upscaling_1 = LayerNorm2d()
        self.output_upscaling_3 = QConvTransposeD2S(d // 8, 2)
        self.output_hypernetworks_mlps = [MLP(d, d // 8, 3) for _ in range(self.num_mask_tokens)]
        self.iou_prediction_head = MLP(IOU_HEAD_HIDDEN, self.num_mask_tokens, IOU_HEAD_DEPTH,
                                       sigmoid_output=True, out_dtype=F32)
        self.pred_obj_score_head = MLP(d, 1, 3, out_dtype=F32)

    def __call__(self, image_embeddings, image_pe, sparse, dense, high_res_features):
        # predict_masks (inlined)
        s = 1
        expand = lambda e: jnp.broadcast_to(e[None], (sparse.shape[0],) + e.shape).astype(F16)
        output_tokens = jnp.concatenate(
            [self.obj_score_token(expand), self.iou_token(expand), self.mask_tokens(expand)], axis=1)
        tokens = jnp.concatenate((output_tokens, sparse), axis=1)
        assert image_embeddings.shape[0] == tokens.shape[0]          # repeat_image=False
        src = image_embeddings + dense
        pos_src = jnp.repeat(image_pe, tokens.shape[0], axis=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1: s + 1 + self.num_mask_tokens, :]
        src = src.transpose(0, 2, 1).reshape(b, c, h, w)
        feat_s0, feat_s1 = high_res_features
        up = gelu(self.output_upscaling_1(self.output_upscaling_0(src) + feat_s1))
        up = gelu(self.output_upscaling_3(up) + feat_s0)
        hyper_in = jnp.stack([self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                              for i in range(self.num_mask_tokens)], axis=1)
        b, c, h, w = up.shape
        # fp16 x fp16 -> fp32 mask logits (activation x activation, not quantized)
        masks = jnp.einsum("bnc,bcp->bnp", hyper_in, up.reshape(b, c, h * w),
                           preferred_element_type=F32).reshape(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        # multimask_output=True (SAM2Heads asserts it): masks / IoUs / tokens 1..3
        return masks[:, 1:], iou_pred[:, 1:], mask_tokens_out[:, 1:], object_score_logits


class SAM2Heads(nn.Module):
    """SAM2Base._forward_sam_heads(point_inputs, mask_inputs=None, high_res_features,
    multimask_output) for the SAM 2.1 config. Returns its 7-tuple."""
    cfg: Config
    num_points: int            # user clicks on this frame (0 while tracking)

    def setup(self):
        c = self.cfg
        d = c.d_model
        self.sam_prompt_encoder = PromptEncoder(d, (c.feat_size, c.feat_size), (c.image_size, c.image_size))
        self.sam_mask_decoder = MaskDecoder(d)
        self.obj_ptr_proj = MLP(d, d, 3)
        self.no_obj_ptr = self.param("no_obj_ptr", _init, (1, d))

    def __call__(self, backbone_features, feat_s0, feat_s1, point_coords=None, point_labels=None):
        b = backbone_features.shape[0]
        # SAM2Base._use_multimask: multimask_output_in_sam and (is_init_cond_frame or
        # multimask_output_for_tracking) and multimask_min_pt_num(0) <= num_pts <= 1
        assert 0 <= self.num_points <= 1, "SAM 2.1 config: multimask output on every frame"
        if point_coords is None:          # no clicks: one empty point with label -1
            point_coords = jnp.zeros((b, 1, 2), F32)
            point_labels = -jnp.ones((b, 1), jnp.int32)
        sparse, dense, dense_pe = self.sam_prompt_encoder((point_coords, point_labels))
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            backbone_features, dense_pe, sparse, dense, (feat_s0, feat_s1))
        is_obj_appearing = object_score_logits > 0
        low_res_multimasks = jnp.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)
        high_res_multimasks = interpolate(low_res_multimasks, (self.cfg.image_size,) * 2, "bilinear")
        # best mask by predicted IoU (in-graph argmax + gather)
        best = jnp.argmax(ious, axis=-1)
        bi = jnp.arange(b)
        low_res_masks = low_res_multimasks[bi, best][:, None]
        high_res_masks = high_res_multimasks[bi, best][:, None]
        sam_output_token = sam_output_tokens[bi, best]
        obj_ptr = self.obj_ptr_proj(sam_output_token)                     # fp16 (stored)
        lam = is_obj_appearing.astype(F16)         # fixed_no_obj_ptr, hard (not soft) choice
        obj_ptr = lam * obj_ptr
        obj_ptr = obj_ptr + (1 - lam) * self.no_obj_ptr.astype(F16)
        return (low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks,
                obj_ptr, object_score_logits)


class SAM2CondFrame(nn.Module):
    """_prepare_memory_conditioned_features on an initial conditioning frame with
    directly_add_no_mem_embed=True: pix_feat = vision_feats[-1] + no_mem_embed."""
    cfg: Config

    @nn.compact
    def __call__(self, vision_feat):
        b, c, h, w = vision_feat.shape
        no_mem_embed = self.param("no_mem_embed", _init, (1, 1, c))
        x = (vision_feat.reshape(b, c, h * w).transpose(2, 0, 1) + no_mem_embed).astype(F16)
        return x.transpose(1, 2, 0).reshape(b, c, h, w)


# ----------------------------------------------------------------------------
# memory attention (modeling/memory_attention.py) and memory bank assembly
# ----------------------------------------------------------------------------

class MemoryAttentionLayer(nn.Module):
    """d_model 256, ffn 2048, relu, pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=True,
    pos_enc_at_cross_attn_queries=False; dropout is identity in eval."""
    cfg: Config

    def setup(self):
        c = self.cfg
        fs = (c.feat_size, c.feat_size)
        self.self_attn = RoPEAttention(c.d_model, 1, shared="qkv", rope_theta=c.rope_theta,
                                       feat_sizes=fs)
        self.cross_attn_image = RoPEAttention(c.d_model, 1, rope_theta=c.rope_theta,
                                              rope_k_repeat=True, feat_sizes=fs)
        self.linear1 = QLinear(c.mem_attn_ffn)
        self.linear2 = QLinear(c.d_model)
        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()
        self.norm3 = LayerNorm()

    def __call__(self, tgt, memory, pos, query_pos, num_k_exclude_rope: int):
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.self_attn(tgt2, tgt2, tgt2)
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.cross_attn_image(tgt2, memory + pos, memory, num_k_exclude_rope=num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        return tgt + self.linear2(relu(self.linear1(tgt2)))


class MemoryAttention(nn.Module):
    cfg: Config

    def setup(self):
        self.layers = [MemoryAttentionLayer(self.cfg) for _ in range(self.cfg.mem_attn_layers)]
        self.norm = LayerNorm()

    def __call__(self, curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens: int):
        output = curr + 0.1 * curr_pos                  # pos_enc_at_input
        output = output.transpose(1, 0, 2)              # batch_first
        curr_pos = curr_pos.transpose(1, 0, 2)
        memory = memory.transpose(1, 0, 2)
        memory_pos = memory_pos.transpose(1, 0, 2)
        for layer in self.layers:
            output = layer(output, memory, memory_pos, curr_pos, num_obj_ptr_tokens)
        return self.norm(output).transpose(1, 0, 2)


class SAM2MemoryConditioning(nn.Module):
    """SAM2Base._prepare_memory_conditioned_features on a non-conditioning frame (tracking
    forward in time, one conditioning frame, memory_temporal_stride_for_eval=1).

    Memory bank (sizes from the inputs: n_mem = maskmem_features.shape[0] spatial memories,
    n_ptr = obj_ptrs.shape[0] pointers): maskmem_features[0] is the conditioning frame
    (t_pos 0), maskmem_features[i >= 1] the frame `n_mem - i` frames back (t_pos =
    num_maskmem - (n_mem - i)); obj_ptrs[0] is the conditioning frame's pointer (temporal
    offset = frame_idx, signed), obj_ptrs[j >= 1] the pointer of the frame j frames back.
    The per-memory loops of the reference are written over the bank axis (one gather of
    maskmem_tpos_enc, one reshape), so the graph does not depend on the bank sizes."""
    cfg: Config

    def setup(self):
        c = self.cfg
        self.memory_attention = MemoryAttention(c)
        self.obj_ptr_tpos_proj = QLinear(c.mem_dim)
        self.maskmem_tpos_enc = self.param("maskmem_tpos_enc", _init, (c.num_maskmem, 1, 1, c.mem_dim))

    def __call__(self, vision_feat, vision_pos, maskmem_features, maskmem_pos_enc, obj_ptrs, frame_idx):
        c = self.cfg
        b, C, h, w = vision_feat.shape
        n_mem, n_ptr, M = maskmem_features.shape[0], obj_ptrs.shape[0], c.mem_dim
        flat = lambda x: x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)   # BCHW -> (HW)BC
        curr, curr_pos = flat(vision_feat), flat(vision_pos)
        # spatial memories: memory i's (HW)BC tokens, i = 0..n_mem-1, concatenated; the memory
        # bank is stored fp16 (phase inputs) and attended in fp16
        memory = maskmem_features.reshape(n_mem, b, M, h * w).transpose(0, 3, 1, 2)
        memory = memory.reshape(n_mem * h * w, b, M)
        # maskmem_tpos_enc[num_maskmem - t_pos - 1]: num_maskmem - 1 for i = 0 (t_pos 0),
        # n_mem - 1 - i for i >= 1
        i = jnp.arange(n_mem, dtype=jnp.int32)
        tpos_idx = jnp.where(i == 0, c.num_maskmem - 1, i[::-1])
        tpos = jnp.take(self.maskmem_tpos_enc, tpos_idx, axis=0)                 # [n_mem, 1, 1, M]
        mem_pos = (flat(maskmem_pos_enc)[None] + tpos).astype(F16)                # [n_mem, HW, B, M]
        mem_pos = mem_pos.reshape(n_mem * h * w, b, M)
        # object pointers with a projected temporal sine encoding
        max_obj_ptrs = min(c.num_frames, c.max_obj_ptrs_in_encoder)
        j = jnp.arange(n_ptr, dtype=jnp.int32)
        pos = jnp.where(j == 0, frame_idx, j)                   # [frame_idx, 1, 2, ..., n_ptr - 1]
        obj_pos = get_1d_sine_pe(pos.astype(F32) / (max_obj_ptrs - 1), dim=C)     # fp32
        obj_pos = self.obj_ptr_tpos_proj(obj_pos)                                    # int8 -> fp16
        obj_pos = jnp.broadcast_to(obj_pos[:, None], (n_ptr, b, M))
        # mem_dim < C: split each pointer into C // mem_dim tokens
        ptrs = obj_ptrs.reshape(n_ptr, b, C // M, M).transpose(0, 2, 1, 3)
        ptrs = ptrs.reshape(n_ptr * (C // M), b, M)
        obj_pos = jnp.repeat(obj_pos, C // M, axis=0)
        memory = jnp.concatenate([memory, ptrs], axis=0)
        memory_pos = jnp.concatenate([mem_pos, obj_pos], axis=0)
        out = self.memory_attention(curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens=ptrs.shape[0])
        return out.transpose(1, 2, 0).reshape(b, C, h, w)


# ----------------------------------------------------------------------------
# memory encoder (modeling/memory_encoder.py)
# ----------------------------------------------------------------------------

class MaskDownSampler(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x):
        c = self.cfg
        num_layers = int(math.log2(c.mask_ds_total_stride) // math.log2(c.mask_ds_stride))
        ch, j = 1, 0
        for _ in range(num_layers):          # encoder.{3i}: Conv2d, {3i+1}: LayerNorm2d, {3i+2}: GELU
            ch = ch * c.mask_ds_stride ** 2
            x = QConv2d(ch, c.mask_ds_kernel, c.mask_ds_stride, c.mask_ds_padding, name=f"encoder_{j}")(x)
            x = gelu(LayerNorm2d(name=f"encoder_{j + 1}")(x))
            j += 3
        return QConv2d(c.d_model, 1, name=f"encoder_{j}")(x)


class CXBlock(nn.Module):
    dim: int
    kernel_size: int

    def setup(self):
        self.dwconv = QConv2d(self.dim, self.kernel_size, padding=self.kernel_size // 2, groups=self.dim)
        self.norm = LayerNorm2d()
        self.pwconv1 = QLinear(4 * self.dim)
        self.pwconv2 = QLinear(self.dim)
        self.gamma = self.param("gamma", nn.initializers.ones, (self.dim,))

    def __call__(self, x):
        inp = x
        x = self.norm(self.dwconv(x)).transpose(0, 2, 3, 1)
        x = self.gamma.astype(F16) * self.pwconv2(gelu(self.pwconv1(x)))
        return inp + x.transpose(0, 3, 1, 2)


class Fuser(nn.Module):
    cfg: Config

    def setup(self):
        self.layers = [CXBlock(self.cfg.d_model, self.cfg.cx_kernel) for _ in range(self.cfg.fuser_layers)]

    def __call__(self, x):
        for layer in self.layers:            # proj = Identity (input_projection=False)
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    cfg: Config

    def setup(self):
        c = self.cfg
        self.mask_downsampler = MaskDownSampler(c)
        self.pix_feat_proj = QConv2d(c.d_model, 1)
        self.fuser = Fuser(c)
        self.out_proj = QConv2d(c.mem_dim, 1)

    def __call__(self, pix_feat, masks):
        masks = self.mask_downsampler(masks)          # skip_mask_sigmoid=True; fp32 mask quantized
        x = self.fuser(self.pix_feat_proj(pix_feat) + masks)
        x = self.out_proj(x)
        b, _, h, w = x.shape
        return x, sine_pe(b, h, w, self.cfg.mem_dim).astype(F16)     # fp16 (stored)


class SAM2MemoryEncoding(nn.Module):
    """SAM2Base._encode_new_memory. `cond`: the initial conditioning frame, encoded in
    SAM2VideoPredictor.propagate_in_video_preflight from its low-res `pred_masks`
    (bilinear to image size; is_mask_from_pts=True, so binarized because the predictor
    sets binarize_mask_from_pts_for_mem_enc); otherwise the tracked frame's high-res mask
    goes through a sigmoid."""
    cfg: Config
    cond: bool

    def setup(self):
        self.memory_encoder = MemoryEncoder(self.cfg)
        self.no_obj_embed_spatial = self.param("no_obj_embed_spatial", _init, (1, self.cfg.mem_dim))

    def __call__(self, vision_feat, masks, object_score_logits):
        c = self.cfg
        if self.cond:
            masks = interpolate(masks, (c.image_size, c.image_size), "bilinear")
            mask_for_mem = (masks > 0).astype(F32)
        else:
            mask_for_mem = jax.nn.sigmoid(masks)
        mask_for_mem = mask_for_mem * c.sigmoid_scale_for_mem_enc + c.sigmoid_bias_for_mem_enc
        feats, pos = self.memory_encoder(vision_feat, mask_for_mem)
        is_obj_appearing = (object_score_logits > 0).astype(F16)
        feats = feats + (1 - is_obj_appearing[..., None, None]) * jnp.broadcast_to(
            self.no_obj_embed_spatial.astype(F16)[..., None, None], feats.shape)
        return feats, pos                                # fp16 memory-bank entries
