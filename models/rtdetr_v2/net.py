"""RT-DETRv2-R50 (PekingU/rtdetr_v2_r50vd), INT8 Flax linen port.

Mirrors HF transformers 5.20.2 `RTDetrV2ForObjectDetection` in eval mode
module-for-module (which itself ports lyuwenyu/RT-DETR `rtdetrv2_pytorch`).
Module names follow HF's, with list/Sequential indices joined to their parent
(`model.decoder.layers.3.encoder_attn` -> `model/decoder/layers_3/encoder_attn`,
`model.encoder_input_proj.0.1` -> `model/encoder_input_proj_0_1`):

  model/backbone/model/{embedder,encoder}     ResNet-50-vd, frozen BN
  model/encoder_input_proj_<l>_0              1x1 conv (BN folded) per level
  model/encoder/aifi_0                        AIFI transformer layer on S5
  model/encoder/{lateral_convs,fpn_blocks,downsample_convs,pan_blocks}_<i>   CCFM
  model/decoder_input_proj_<l>_0
  model/enc_output_{0,1}, enc_score_head, enc_bbox_head   query selection
  model/decoder/layers_<i>                    self-attn + MS deformable attn + FFN
  model/decoder/{query_pos_head,class_embed_<i>,bbox_embed_<i>}

Layout is NHWC for feature maps. Token sequences are flattened row-major over
(H, W), exactly like torch's `flatten(2).transpose(1, 2)` of NCHW. Dropout is
the identity at inference and is omitted. The deformable attention's
`grid_sample(bilinear, zeros, align_corners=False)` is re-implemented with
gathers (`grid_sample_bilinear`), following torch's CPU kernel formulas.

Quantization: INT8 W8A8 platform fallback (edge, TensorRT INT8). There is no official or
production INT8 RT-DETRv2 (lyuwenyu/RT-DETR tools/export_trt.py builds FP16 only; the HF
checkpoint is F32; NVIDIA TAO's RT-DETR allows implicit INT8 calibration without
per-layer carve-outs), so the edge INT8 recipe of models/quant_recipes.py applies:
  * every conv and every Linear is INT8 (models/quant.py QConv / QDense, INT8_STATIC):
    weights per-output-channel symmetric, activations per-tensor static symmetric
    (calibrated), i8 x i8 -> i32, fp32 epilogue. This includes sampling_offsets,
    attention_weights, query_pos_head and the score / box heads.
  * every BatchNorm is folded into its conv (TensorRT): the backbone's frozen BN, the
    input projections' BN and the CCFM ConvNormLayer norms; the convs gain a bias.
    RepVGG blocks keep their 3x3 and 1x1 branches (no re-parameterisation).
  * a tensor's scale is owned by its producer (`*_quant` ActQuant) when it stays int8
    between layers or feeds several int8 layers (ReLU outputs in the backbone, CSP
    inputs, x + pos for q/k, hidden + pos for offsets / weights, the encoder memory
    for the six value_proj, the decoder state for the heads and the next v_proj);
    otherwise the consuming layer owns its `input_scale`. The fp32 image is quantized
    by the first conv.
  * what TensorRT cannot run in INT8 stays float (TensorRT operator docs: GridSample,
    SoftMax, Resize fp16 / fp32 only; TopK no int8 type; activation x activation
    MatMul INT8 only with explicit Q/DQ): attention QK^T / AV are fp16 x fp16 with fp32
    accumulation, softmax fp32; deformable bilinear sampling and its weighted sum fp16;
    LayerNorm, GELU / SiLU on float tensors, nearest upsample, concat and top-k fp16;
    sine position embedding, anchors, sampling-location and sigmoid / inverse_sigmoid
    box math fp32. Other inter-layer float tensors are fp16 (TensorRT INT8 engine rule).
    Max / avg pooling in the backbone run on int8 tensors (int32 sum for the average).
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from models import quant as q

F16, F32 = jnp.float16, jnp.float32


@dataclass(frozen=True)
class Config:
    batch: int = 1
    image: int = 640                 # square input, H = W
    num_channels: int = 3
    # backbone: RTDetrResNetConfig (ResNet-50-vd)
    embedding_size: int = 64
    hidden_sizes: tuple[int, ...] = (256, 512, 1024, 2048)
    depths: tuple[int, ...] = (3, 4, 6, 3)
    # hybrid encoder
    encoder_hidden_dim: int = 256
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    encode_proj_layers: tuple[int, ...] = (2,)
    positional_encoding_temperature: int = 10000
    # decoder
    d_model: int = 256
    num_queries: int = 300
    decoder_layers: int = 6
    decoder_ffn_dim: int = 1024
    decoder_attention_heads: int = 8
    decoder_n_points: int = 4
    decoder_offset_scale: float = 0.5
    num_labels: int = 80
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    feat_strides: tuple[int, ...] = (8, 16, 32)

    @property
    def num_levels(self) -> int:
        return len(self.feat_strides)

    def spatial_shapes(self) -> list[tuple[int, int]]:
        return [(self.image // s, self.image // s) for s in self.feat_strides]


# -- shared layers -------------------------------------------------------------

def qconv(features: int, k: int, stride: int, pad: int, name: str, dtype=F32) -> q.QConv:
    """INT8 conv with its BatchNorm folded in (bias); fp32 epilogue, output in `dtype`."""
    return q.QConv(features, (k, k), q.INT8_STATIC, strides=(stride, stride),
                   padding=[(pad, pad)] * 2, use_bias=True, dtype=dtype, name=name)


def qdense(features: int, name: str, dtype=F16) -> q.QDense:
    """INT8 Linear; fp32 epilogue, output in `dtype`."""
    return q.QDense(features, q.INT8_STATIC, dtype=dtype, name=name)


def layer_norm(eps: float, name: str) -> nn.LayerNorm:
    # fp16 tensor (statistics reduced in fp32); use_fast_variance=False: two-pass
    # biased variance, as torch.
    return nn.LayerNorm(epsilon=eps, use_fast_variance=False, dtype=F16, name=name)


def gelu(x):
    """ACT2FN["gelu"] == torch.nn.functional.gelu (exact erf form)."""
    return x * 0.5 * (1.0 + jax.lax.erf(x * (2.0 ** -0.5)))


def attention_core(q_, k_, v_, scale: float):
    """softmax(q k^T * scale) v on [B, heads, L, hd]: fp16 operands, fp32 accumulation
    and softmax, fp16 output (TensorRT fp16 MHA; not INT8 without explicit Q/DQ)."""
    s = jnp.einsum("bhqd,bhkd->bhqk", q_, k_, preferred_element_type=F32) * scale
    p = jax.nn.softmax(s, axis=-1).astype(F16)
    return jnp.einsum("bhqk,bhkd->bhqd", p, v_, preferred_element_type=F32).astype(F16)


class MLPPredictionHead(nn.Module):
    """RTDetrV2MLPPredictionHead: INT8 Linear layers with ReLU between them."""
    hidden_dim: int
    output_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x, x_scale=None):
        dims = [self.hidden_dim] * (self.num_layers - 1) + [self.output_dim]
        for i, d in enumerate(dims):
            x = qdense(d, f"layers_{i}")(x, x_scale)
            x_scale = None
            x = nn.relu(x) if i < self.num_layers - 1 else x
        return x


# -- backbone: RTDetrResNetBackbone (ResNet-50-vd) -----------------------------

class RTDetrResNetConvLayer(nn.Module):
    """conv + frozen BN (folded) [+ ReLU, requantized to int8 -> (x_q, scale)]."""
    features: int
    kernel_size: int = 3
    stride: int = 1
    activation: bool = True          # relu

    @nn.compact
    def __call__(self, x, x_scale=None):
        k = self.kernel_size
        y = qconv(self.features, k, self.stride, k // 2, "convolution")(x, x_scale)
        if not self.activation:
            return y                                    # fp32 epilogue (residual add)
        y, s, _ = q.ActQuant(name="activation_quant")(nn.relu(y))
        return y, s


class RTDetrResNetShortCut(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, x_scale=None):
        return qconv(self.features, 1, 1, 0, "convolution")(x, x_scale)


class RTDetrResNetEmbeddings(nn.Module):
    """Deep stem: 3x3/2 (C/2), 3x3 (C/2), 3x3 (C), then max pool 3/2/1 on int8."""
    cfg: Config

    @nn.compact
    def __call__(self, x):                          # fp32 image, quantized by embedder_0
        c = self.cfg.embedding_size
        x, s = RTDetrResNetConvLayer(c // 2, 3, 2, name="embedder_0")(x)
        x, s = RTDetrResNetConvLayer(c // 2, 3, 1, name="embedder_1")(x, s)
        x, s = RTDetrResNetConvLayer(c, 3, 1, name="embedder_2")(x, s)
        x = lax.reduce_window(x, jnp.int8(-128), lax.max, (1, 3, 3, 1), (1, 2, 2, 1),
                              ((0, 0), (1, 1), (1, 1), (0, 0)))
        return x, s


class RTDetrResNetBottleNeckLayer(nn.Module):
    """Variant d: stride on the 3x3 conv; stride-2 shortcut = AvgPool(2, 2) + 1x1 conv."""
    in_channels: int
    out_channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, s):                       # int8 block input and its scale
        reduces = self.out_channels // 4
        if self.stride == 2:
            # AvgPool2d(2, 2, 0, ceil_mode=True): ceil_mode only matters for odd sizes.
            # INT8 average pool: int32 sum of the 2x2 window, scaled mean.
            assert x.shape[1] % 2 == 0 and x.shape[2] % 2 == 0, "odd size: ceil_mode differs"
            acc = lax.reduce_window(x.astype(jnp.int32), jnp.int32(0), lax.add,
                                    (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
            pooled = acc.astype(F32) * (s / 4)
            residual = RTDetrResNetShortCut(self.out_channels, name="shortcut_1")(pooled)
        elif self.in_channels != self.out_channels:
            residual = RTDetrResNetShortCut(self.out_channels, name="shortcut")(x, s)
        else:
            residual = q.dequantize(x, s)
        h, hs = RTDetrResNetConvLayer(reduces, 1, 1, name="layer_0")(x, s)
        h, hs = RTDetrResNetConvLayer(reduces, 3, self.stride, name="layer_1")(h, hs)
        h = RTDetrResNetConvLayer(self.out_channels, 1, 1, activation=False, name="layer_2")(h, hs)
        y, sy, _ = q.ActQuant(name="output_quant")(nn.relu(h + residual))
        return y, sy


class RTDetrResNetStage(nn.Module):
    in_channels: int
    out_channels: int
    stride: int
    depth: int

    @nn.compact
    def __call__(self, x, s):
        x, s = RTDetrResNetBottleNeckLayer(self.in_channels, self.out_channels, self.stride,
                                           name="layers_0")(x, s)
        for i in range(1, self.depth):
            x, s = RTDetrResNetBottleNeckLayer(self.out_channels, self.out_channels,
                                               name=f"layers_{i}")(x, s)
        return x, s


class RTDetrResNetEncoder(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, s):
        c = self.cfg
        ins = (c.embedding_size,) + tuple(c.hidden_sizes[:-1])
        outs = []
        for i, (cin, cout, d) in enumerate(zip(ins, c.hidden_sizes, c.depths)):
            # downsample_in_first_stage=False: stage 1 has stride 1.
            x, s = RTDetrResNetStage(cin, cout, 1 if i == 0 else 2, d, name=f"stages_{i}")(x, s)
            outs.append((x, s))
        return outs


class RTDetrResNetBackbone(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x):
        x, s = RTDetrResNetEmbeddings(self.cfg, name="embedder")(x)
        stages = RTDetrResNetEncoder(self.cfg, name="encoder")(x, s)
        return stages[1:]                       # out_features stage2..4, int8 + scale


class RTDetrV2ConvEncoder(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x):
        # The downsampled pixel_mask HF computes alongside is never consumed.
        return RTDetrResNetBackbone(self.cfg, name="model")(x)


# -- hybrid encoder ------------------------------------------------------------

class RTDetrV2ConvNormLayer(nn.Module):
    """conv + BN (folded) [+ SiLU]."""
    features: int
    kernel_size: int
    stride: int
    padding: int | None = None
    activation: bool = False         # silu
    dtype: object = F16

    @nn.compact
    def __call__(self, x, x_scale=None):
        k = self.kernel_size
        pad = (k - 1) // 2 if self.padding is None else self.padding
        x = qconv(self.features, k, self.stride, pad, "conv", self.dtype)(x, x_scale)
        return nn.silu(x) if self.activation else x


class RTDetrV2RepVggBlock(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, x_scale):                 # one int8 input for both branches
        c = self.cfg.encoder_hidden_dim
        y = (RTDetrV2ConvNormLayer(c, 3, 1, padding=1, dtype=F32, name="conv1")(x, x_scale)
             + RTDetrV2ConvNormLayer(c, 1, 1, padding=0, dtype=F32, name="conv2")(x, x_scale))
        return nn.silu(y)


class RTDetrV2CSPRepLayer(nn.Module):
    """hidden_expansion = 1.0, so conv3 is the identity (hidden == out channels)."""
    cfg: Config

    @nn.compact
    def __call__(self, x):                          # fp16 concat, quantized once for conv1 / conv2
        c = self.cfg.encoder_hidden_dim
        xq, xs, _ = q.ActQuant(name="input_quant")(x)
        h1 = RTDetrV2ConvNormLayer(c, 1, 1, activation=True, dtype=F32, name="conv1")(xq, xs)
        for i in range(3):
            hq, hs, _ = q.ActQuant(name=f"bottlenecks_{i}_quant")(h1)
            h1 = RTDetrV2RepVggBlock(self.cfg, name=f"bottlenecks_{i}")(hq, hs)
        h2 = RTDetrV2ConvNormLayer(c, 1, 1, activation=True, dtype=F32, name="conv2")(xq, xs)
        return (h1 + h2).astype(F16)


class RTDetrV2MLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    act: str                          # "gelu" | "relu"

    @nn.compact
    def __call__(self, x):
        x = qdense(self.intermediate_size, "fc1")(x)
        x = gelu(x) if self.act == "gelu" else nn.relu(x)
        return qdense(self.hidden_size, "fc2")(x)


class RTDetrV2SelfAttention(nn.Module):
    """Eager attention; position embeddings are added to queries and keys only. x + pos is
    quantized once for q_proj / k_proj; v_proj reads x (int8 if its producer quantized it)."""
    hidden_size: int
    num_heads: int

    @nn.compact
    def __call__(self, x, pos, x_q=None, x_scale=None):
        B, S, D = x.shape
        hd = D // self.num_heads
        qk, qks, _ = q.ActQuant(name="qk_quant")(x + pos)
        heads = lambda t: t.reshape(B, S, self.num_heads, hd).transpose(0, 2, 1, 3)
        q_ = heads(qdense(D, "q_proj")(qk, qks))
        k_ = heads(qdense(D, "k_proj")(qk, qks))
        v_ = heads(qdense(D, "v_proj")(x if x_q is None else x_q, x_scale))
        out = attention_core(q_, k_, v_, hd ** -0.5).transpose(0, 2, 1, 3).reshape(B, S, D)
        return qdense(D, "o_proj")(out)


class RTDetrV2EncoderLayer(nn.Module):
    """normalize_before = False (post-norm)."""
    cfg: Config

    @nn.compact
    def __call__(self, x, pos):
        c = self.cfg
        h = RTDetrV2SelfAttention(c.encoder_hidden_dim, c.encoder_attention_heads,
                                  name="self_attn")(x, pos)
        x = layer_norm(c.layer_norm_eps, "self_attn_layer_norm")(x + h)
        h = RTDetrV2MLP(c.encoder_hidden_dim, c.encoder_ffn_dim, "gelu", name="mlp")(x)
        return layer_norm(c.layer_norm_eps, "final_layer_norm")(x + h)


def sine_position_embedding(height: int, width: int, dim: int, temperature: float) -> np.ndarray:
    """build_2d_sinusoidal_position_embedding: [sin_h | cos_h | sin_w | cos_w], computed in
    float64 and cast to float32 (a shape-derived constant), shape (1, H*W, dim)."""
    pos_dim = dim // 4
    omega = 1.0 / temperature ** (np.arange(pos_dim, dtype=np.float64) / pos_dim)
    grid_h, grid_w = np.meshgrid(np.arange(height, dtype=np.float64),
                                 np.arange(width, dtype=np.float64), indexing="ij")
    emb_h = np.outer(grid_h.reshape(-1), omega)
    emb_w = np.outer(grid_w.reshape(-1), omega)
    pe = np.concatenate([np.sin(emb_h), np.cos(emb_h), np.sin(emb_w), np.cos(emb_w)], axis=1)
    return pe.astype(np.float32)[None]


class RTDetrV2AIFILayer(nn.Module):
    """eval_size = None, so the position embedding is built for the actual map size
    (fp32 constant, added to the fp16 tokens in fp16)."""
    cfg: Config

    @nn.compact
    def __call__(self, x):
        c = self.cfg
        B, H, W, C = x.shape
        h = x.reshape(B, H * W, C)
        pos = jnp.asarray(sine_position_embedding(H, W, C, c.positional_encoding_temperature)).astype(F16)
        for i in range(c.encoder_layers):
            h = RTDetrV2EncoderLayer(c, name=f"layers_{i}")(h, pos)
        return h.reshape(B, H, W, C)


def upsample_nearest_2x(x):
    """F.interpolate(scale_factor=2.0, mode="nearest"): src = floor(dst / 2)."""
    B, H, W, C = x.shape
    x = jnp.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C))
    return x.reshape(B, 2 * H, 2 * W, C)


class RTDetrV2HybridEncoder(nn.Module):
    """Takes the fp16 projected levels; returns the PAN outputs quantized once each
    (`pan_quant_<i>`: shared by the next downsample conv and the decoder input proj)."""
    cfg: Config

    @nn.compact
    def __call__(self, feats):
        c = self.cfg
        feats = list(feats)
        for i, idx in enumerate(c.encode_proj_layers):
            feats[idx] = RTDetrV2AIFILayer(c, name=f"aifi_{i}")(feats[idx])
        n = c.num_levels - 1
        d = c.encoder_hidden_dim
        # top-down FPN (fp16 maps; nearest upsample and concat in fp16)
        fpn = [feats[-1]]
        for i in range(n):
            low = feats[n - i - 1]
            top = RTDetrV2ConvNormLayer(d, 1, 1, activation=True, name=f"lateral_convs_{i}")(fpn[-1])
            fpn[-1] = top
            fused = jnp.concatenate([upsample_nearest_2x(top), low], axis=-1)
            fpn.append(RTDetrV2CSPRepLayer(c, name=f"fpn_blocks_{i}")(fused))
        fpn.reverse()
        # bottom-up PAN
        pan = [q.ActQuant(name="pan_quant_0")(fpn[0])[:2]]
        for i in range(n):
            down = RTDetrV2ConvNormLayer(d, 3, 2, activation=True,
                                         name=f"downsample_convs_{i}")(*pan[-1])
            fused = jnp.concatenate([down, fpn[i + 1]], axis=-1)
            out = RTDetrV2CSPRepLayer(c, name=f"pan_blocks_{i}")(fused)
            pan.append(q.ActQuant(name=f"pan_quant_{i + 1}")(out)[:2])
        return pan


# -- decoder -------------------------------------------------------------------

def grid_sample_bilinear(value, grid, height: int, width: int):
    """F.grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False).

    value: (N, H*W, C) (rows of the (H, W) map, row-major); grid: (N, Q, P, 2) with
    (x, y) in [-1, 1]. Returns (N, Q, P, C). Formulas follow torch's CPU kernel
    (GridSamplerKernel.cpp, ApplyGridSample<bilinear, zeros>): x = (gx + 1) * (W / 2) - 0.5,
    corner weights nw = s*e, ne = s*w, sw = n*e, se = n*w, out-of-bounds corners read 0.
    The four corner reads are one gather per level. Runs in the dtype of its inputs
    (fp16: TensorRT GridSample has no int8 type).
    """
    N, Q, P, _ = grid.shape
    C = value.shape[-1]
    x = (grid[..., 0] + 1.0) * (width * 0.5) - 0.5
    y = (grid[..., 1] + 1.0) * (height * 0.5) - 0.5
    x_w, y_n = jnp.floor(x), jnp.floor(y)
    w = x - x_w
    e = 1.0 - w
    n = y - y_n
    s = 1.0 - n
    weights = (s * e, s * w, n * e, n * w)                 # nw, ne, sw, se
    ix_w, iy_n = x_w.astype(jnp.int32), y_n.astype(jnp.int32)
    corners = ((ix_w, iy_n), (ix_w + 1, iy_n), (ix_w, iy_n + 1), (ix_w + 1, iy_n + 1))
    idx, valid = [], []
    for ix, iy in corners:
        valid.append((ix > -1) & (ix < width) & (iy > -1) & (iy < height))
        idx.append(jnp.clip(iy, 0, height - 1) * width + jnp.clip(ix, 0, width - 1))
    idx = jnp.stack(idx, axis=-1).reshape(N, Q * P * 4)
    vals = jax.vmap(lambda v, i: v.at[i].get(mode="promise_in_bounds"))(value, idx)
    vals = vals.reshape(N, Q, P, 4, C)
    vals = jnp.where(jnp.stack(valid, axis=-1)[..., None], vals, jnp.zeros((), vals.dtype))
    out = vals[..., 0, :] * weights[0][..., None]
    for k in range(1, 4):
        out = out + vals[..., k, :] * weights[k][..., None]
    return out


def multi_scale_deformable_attention_v2(value, spatial_shapes, sampling_locations,
                                        attention_weights, num_points_list):
    """value (B, S, nH, hd) fp16; sampling_locations (B, Q, nH, L*P, 2) in [0, 1] (fp32);
    attention_weights (B, Q, nH, L*P) fp16. Returns (B, Q, nH*hd) fp16. method="default"."""
    B, _, nH, hd = value.shape
    Q = sampling_locations.shape[1]
    grids = (2 * sampling_locations - 1).astype(value.dtype)
    grids = grids.transpose(0, 2, 1, 3, 4).reshape(B * nH, Q, -1, 2)
    sampled, start, p0 = [], 0, 0
    for (h, w), p in zip(spatial_shapes, num_points_list):
        v = value[:, start:start + h * w].transpose(0, 2, 1, 3).reshape(B * nH, h * w, hd)
        sampled.append(grid_sample_bilinear(v, grids[:, :, p0:p0 + p], h, w))
        start, p0 = start + h * w, p0 + p
    sampled = jnp.concatenate(sampled, axis=2)                       # (B*nH, Q, L*P, hd)
    aw = attention_weights.transpose(0, 2, 1, 3).reshape(B * nH, Q, -1, 1)
    out = (sampled * aw).sum(axis=2)                                 # (B*nH, Q, hd)
    return out.reshape(B, nH, Q, hd).transpose(0, 2, 1, 3).reshape(B, Q, nH * hd)


class RTDetrV2MultiscaleDeformableAttention(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden_states, pos, memory_q, memory_scale, reference_points):
        c = self.cfg
        nH, L, P = c.decoder_attention_heads, c.num_levels, c.decoder_n_points
        D = c.d_model
        n_points_list = [P] * L
        n_points_scale = self.variable(
            "constants", "n_points_scale",
            lambda: jnp.asarray([1 / n for n in n_points_list for _ in range(n)], jnp.float32)).value
        hidden_states = hidden_states + pos
        B, Q, _ = hidden_states.shape
        S = memory_q.shape[1]
        value = qdense(D, "value_proj")(memory_q, memory_scale).reshape(B, S, nH, D // nH)
        hq, hs, _ = q.ActQuant(name="query_quant")(hidden_states)   # offsets + weights
        offsets = qdense(nH * L * P * 2, "sampling_offsets")(hq, hs)
        offsets = offsets.reshape(B, Q, nH, L * P, 2).astype(F32)
        aw = qdense(nH * L * P, "attention_weights")(hq, hs).reshape(B, Q, nH, L * P)
        aw = jax.nn.softmax(aw.astype(F32), axis=-1).astype(F16)
        # reference_points (B, Q, 1, 4): cx, cy, w, h (fp32 box math)
        offset = (offsets * n_points_scale[:, None] * reference_points[:, :, None, :, 2:]
                  * c.decoder_offset_scale)
        loc = reference_points[:, :, None, :, :2] + offset
        out = multi_scale_deformable_attention_v2(value, c.spatial_shapes(), loc, aw, n_points_list)
        return qdense(D, "output_proj")(out)


class RTDetrV2DecoderLayer(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, x_q, x_scale, pos, memory_q, memory_scale, reference_points):
        c = self.cfg
        h = RTDetrV2SelfAttention(c.d_model, c.decoder_attention_heads,
                                  name="self_attn")(x, pos, x_q, x_scale)
        x = layer_norm(c.layer_norm_eps, "self_attn_layer_norm")(x + h)
        h = RTDetrV2MultiscaleDeformableAttention(c, name="encoder_attn")(
            x, pos, memory_q, memory_scale, reference_points)
        x = layer_norm(c.layer_norm_eps, "encoder_attn_layer_norm")(x + h)
        h = RTDetrV2MLP(c.d_model, c.decoder_ffn_dim, "relu", name="mlp")(x)
        return layer_norm(c.layer_norm_eps, "final_layer_norm")(x + h)


def inverse_sigmoid(x, eps=1e-5):
    x = jnp.clip(x, 0.0, 1.0)
    return jnp.log(jnp.maximum(x, eps) / jnp.maximum(1 - x, eps))


class RTDetrV2Decoder(nn.Module):
    """Iterative box refinement; class_embed / bbox_embed are the detection heads that
    RTDetrV2ForObjectDetection attaches to the decoder (applied after every layer).
    The decoder state after each layer is quantized once (`hidden_quant_<i>`) for the two
    heads and the next layer's v_proj; the memory once (`memory_quant`) for all value_proj."""
    cfg: Config

    @nn.compact
    def __call__(self, target, target_q, target_scale, memory, init_reference_points):
        c = self.cfg
        memory_q, memory_scale, _ = q.ActQuant(name="memory_quant")(memory)
        query_pos_head = MLPPredictionHead(2 * c.d_model, c.d_model, 2, name="query_pos_head")
        ref = jax.nn.sigmoid(init_reference_points)
        x, x_q, x_scale = target, target_q, target_scale
        hs, refs, logits = [], [], []
        for i in range(c.decoder_layers):
            pos = query_pos_head(ref)
            x = RTDetrV2DecoderLayer(c, name=f"layers_{i}")(x, x_q, x_scale, pos, memory_q,
                                                           memory_scale, ref[:, :, None, :])
            x_q, x_scale, _ = q.ActQuant(name=f"hidden_quant_{i}")(x)
            corners = MLPPredictionHead(c.d_model, 4, 3, name=f"bbox_embed_{i}")(x_q, x_scale)
            ref = jax.nn.sigmoid(corners.astype(F32) + inverse_sigmoid(ref))
            hs.append(x)
            refs.append(ref)
            logits.append(qdense(c.num_labels, f"class_embed_{i}")(x_q, x_scale))
        return jnp.stack(hs, 1), jnp.stack(logits, 1), jnp.stack(refs, 1)


def generate_anchors(cfg: Config, grid_size: float = 0.05):
    """RTDetrV2Model._cached_generate_anchors in float32 (anchor_image_size = None, so
    computed from the feature shapes): a shape-derived constant (1, S, 4) + valid mask."""
    anchors = []
    for level, (h, w) in enumerate(cfg.spatial_shapes()):
        gy, gx = np.meshgrid(np.arange(h).astype(np.float32), np.arange(w).astype(np.float32),
                             indexing="ij")
        xy = np.stack([gx, gy], -1)[None] + np.float32(0.5)
        xy[..., 0] /= np.float32(w)
        xy[..., 1] /= np.float32(h)
        wh = np.ones_like(xy) * np.float32(grid_size) * np.float32(2.0 ** level)
        anchors.append(np.concatenate([xy, wh], -1).reshape(-1, h * w, 4))
    anchors = np.concatenate(anchors, 1)
    eps = np.float32(1e-2)
    valid = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdims=True)
    anchors = np.log(anchors / (np.float32(1) - anchors))
    anchors = np.where(valid, anchors, np.finfo(np.float32).max).astype(np.float32)
    return anchors, valid


class RTDetrV2Model(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, pixel_values):
        c = self.cfg
        D = c.d_model
        feats = RTDetrV2ConvEncoder(c, name="backbone")(pixel_values)      # [(int8, scale)]
        proj = [qconv(c.encoder_hidden_dim, 1, 1, 0, f"encoder_input_proj_{l}_0", F16)(fq, fs)
                for l, (fq, fs) in enumerate(feats)]                      # BN folded
        enc = RTDetrV2HybridEncoder(c, name="encoder")(proj)              # [(int8, scale)]
        sources = []
        for l, (fq, fs) in enumerate(enc):
            f = qconv(D, 1, 1, 0, f"decoder_input_proj_{l}_0", F16)(fq, fs)
            B, H, W, _ = f.shape
            sources.append(f.reshape(B, H * W, D))
        source_flatten = jnp.concatenate(sources, axis=1)              # (B, S, D) fp16

        anchors, valid = generate_anchors(c)
        memory = jnp.asarray(valid.astype(np.float16)) * source_flatten
        output_memory = qdense(D, "enc_output_0")(memory)
        output_memory = layer_norm(c.layer_norm_eps, "enc_output_1")(output_memory)
        om_q, om_s, _ = q.ActQuant(name="enc_output_quant")(output_memory)   # both heads
        enc_outputs_class = qdense(c.num_labels, "enc_score_head")(om_q, om_s)
        enc_outputs_coord_logits = (MLPPredictionHead(D, 4, 3, name="enc_bbox_head")(om_q, om_s)
                                    .astype(F32) + jnp.asarray(anchors))

        # query selection: top-k (fp16) over the per-token max class logit
        _, topk_ind = jax.lax.top_k(enc_outputs_class.max(-1), c.num_queries)
        take = lambda a: jnp.take_along_axis(a, topk_ind[..., None], axis=1)
        reference_points_unact = take(enc_outputs_coord_logits)
        enc_topk_bboxes = jax.nn.sigmoid(reference_points_unact)
        enc_topk_logits = take(enc_outputs_class)
        target = take(output_memory)                                   # learn_initial_query=False
        target_q = take(om_q)                                          # int8 gather, same scale

        hs, logits, refs = RTDetrV2Decoder(c, name="decoder")(target, target_q, om_s,
                                                              source_flatten,
                                                              reference_points_unact)
        return {"intermediate_hidden_states": hs, "intermediate_logits": logits,
                "intermediate_reference_points": refs, "enc_topk_logits": enc_topk_logits,
                "enc_topk_bboxes": enc_topk_bboxes, "enc_outputs_class": enc_outputs_class}


OUTPUT_NAMES = ("logits", "pred_boxes", "intermediate_logits", "intermediate_reference_points",
                "intermediate_hidden_states", "enc_topk_logits", "enc_topk_bboxes",
                "enc_outputs_class")


class RTDetrV2ForObjectDetection(nn.Module):
    """Returns the eval-mode output tensors in OUTPUT_NAMES order (before postprocessing);
    logits / hidden states fp16, boxes fp32."""
    cfg: Config

    @nn.compact
    def __call__(self, pixel_values):                       # (B, H, W, 3), NHWC, fp32
        o = RTDetrV2Model(self.cfg, name="model")(pixel_values)
        o["logits"] = o["intermediate_logits"][:, -1]
        o["pred_boxes"] = o["intermediate_reference_points"][:, -1]
        return tuple(o[k] for k in OUTPUT_NAMES)
