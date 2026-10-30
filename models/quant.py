"""Quantized arithmetic shared by every port (the per-model recipes are in quant_recipes.py).

Every quantized GEMM / convolution is written in explicit-quantization (QDQ) form,
the form TensorRT, ONNX Runtime, ModelOpt and TFLite converters consume:

    x_q   = quantize(x, input_scale)                 # round + clip + convert (int8 / fp8)
    acc   = dot_general(x_q, w_q, preferred=int32 | float32)
    y     = acc * (input_scale * kernel_scale) + bias   # fp32 epilogue
    y     = y.astype(compute dtype)                  # or requantized to int8 ("out='int8'")

So the graph carries the exact operand / accumulator dtypes of the deployed kernel:
i8 x i8 -> i32, f8E4M3FN x f8E4M3FN -> f32, and an int4 / int8 weight-only dequant
followed by a 16-bit GEMM. Fusing a DQ -> Q pair into a single requant is a backend
scheduling choice and is not modelled; the arithmetic is the same.

Scales are parameters: real deployments calibrate activation scales offline (static
schemes) and store weight scales next to the weights. Dynamic schemes compute the
activation scale in the graph (amax reduction). Value convention: x ~= q * scale
(+ zero point for asymmetric int8, x ~= (q - zp) * scale).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

INT8 = jnp.int8
UINT8 = jnp.uint8
INT4 = jnp.int4
FP8 = jnp.float8_e4m3fn

QMAX = {"int8": 127.0, "int4": 7.0, "fp8": 448.0, "uint8": 255.0}
DTYPES = {"int8": INT8, "int4": INT4, "fp8": FP8, "uint8": UINT8}
ACC = {"int8": jnp.int32, "fp8": jnp.float32}


def _is_int(kind: str) -> bool:
    return kind in ("int8", "int4", "uint8")


# --------------------------------------------------------------------------
# element-wise quantize / dequantize
# --------------------------------------------------------------------------

def quantize(x, scale, kind: str, zero_point=None, narrow: bool = True):
    """x -> kind ("int8" | "int4" | "uint8" | "fp8") with x ~= (q - zp) * scale.

    Symmetric integers use the narrow range [-qmax, qmax] (TensorRT, TFLite weights);
    narrow=False keeps the full two's-complement range (e.g. GGML Q4_0: [-8, 7])."""
    y = x.astype(jnp.float32) / scale
    if kind == "fp8":
        return jnp.clip(y, -QMAX["fp8"], QMAX["fp8"]).astype(FP8)
    y = jnp.round(y)
    if zero_point is not None:
        y = y + zero_point
    lo, hi = {"int8": (-128, 127), "int4": (-8, 7), "uint8": (0, 255)}[kind]
    if zero_point is None and kind != "uint8" and narrow:
        lo = -hi                           # symmetric: restricted range [-127, 127]
    return jnp.clip(y, lo, hi).astype(DTYPES[kind])


def dequantize(q, scale, dtype=jnp.float32, zero_point=None):
    y = q.astype(jnp.float32)
    if zero_point is not None:
        y = y - zero_point
    return (y * scale).astype(dtype)


def dynamic_scale(x, kind: str, axis=-1):
    """Per-slice amax scale computed in the graph (dynamic quantization)."""
    amax = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=axis, keepdims=True)
    return jnp.maximum(amax, 1e-12) / QMAX[kind]


def qeinsum(spec: str, a, b, a_scale, b_scale, kind: str, dtype):
    """Activation x activation product (attention QK^T / PV) with both operands
    quantized per-tensor with static scales; dequantized epilogue in `dtype`."""
    aq = a if a.dtype == DTYPES[kind] else quantize(a, a_scale, kind)
    bq = b if b.dtype == DTYPES[kind] else quantize(b, b_scale, kind)
    acc = jnp.einsum(spec, aq, bq, preferred_element_type=ACC[kind])
    return (acc.astype(jnp.float32) * (a_scale * b_scale)).astype(dtype)


# --------------------------------------------------------------------------
# schemes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QScheme:
    """How one GEMM / conv is quantized.

    weight       "int8" | "fp8" | "int4"
    weight_gran  "tensor" | "channel" (one scale per output feature / row) | "block"
    weight_block (K, N) block of one weight scale when weight_gran == "block"
    act          "static" (per-tensor, calibrated scale parameter)
                 "token"  (dynamic, one scale per row of the GEMM's M axis)
                 "block"  (dynamic, one scale per 1 x act_block slice of K)
                 "none"   (weight-only: weights dequantized, 16-bit GEMM)
    act_zero_point  asymmetric int8 activations (TFLite); zero point folded into an int32 bias
    out          "float": dequantize in the epilogue to the module dtype
                 "int8":  requantize to int8 with an output scale / zero point (full-integer
                          pipelines; fp32 requantization as in XNNPACK)
    """
    weight: str
    weight_gran: str = "channel"
    weight_block: tuple[int, int] = (128, 128)
    act: str = "static"
    act_block: int = 128
    act_zero_point: bool = False
    out: str = "float"

    @property
    def act_kind(self) -> str:
        return "int8" if self.weight == "int4" else self.weight


# The recipes used by the ports.
INT8_STATIC = QScheme("int8")                                   # TensorRT / MLPerf INT8 W8A8
INT8_TENSOR_W = QScheme("int8", weight_gran="tensor")           # per-tensor weights
INT8_DYNAMIC_TOKEN = QScheme("int8", act="token")               # CTranslate2 / Intel W8A8 dynamic
INT8_TFLITE = QScheme("int8", act_zero_point=True, out="int8")  # TFLite full-integer
INT8_TFLITE_FC = QScheme("int8", weight_gran="tensor", act_zero_point=True, out="int8")
FP8_STATIC = QScheme("fp8", weight_gran="tensor")               # ModelOpt / TE FP8 per-tensor
FP8_BLOCK128 = QScheme("fp8", weight_gran="block", act="block") # DeepSeek-V3 / Qwen FP8 blocks


class ActQuant(nn.Module):
    """Quantizes one activation tensor with its calibrated per-tensor scale (and zero
    point); the producer owns the scale, consumers receive it with the tensor.
    Returns (x_q, scale, zero_point or None)."""
    kind: str = "int8"
    zero_point: bool = False
    init_scale: float = 4.0 / 127

    @nn.compact
    def __call__(self, x):
        s = self.param("scale", const(self.init_scale), (), jnp.float32)
        zp = self.param("zero_point", const(0), (), jnp.int32) if self.zero_point else None
        return quantize(x, s, self.kind, zp), s, zp


def rand_q(kind):
    """Random quantized-weight initializer (reduced configs only; real weights are converted)."""
    def init(key, shape, dtype):
        if _is_int(kind):
            hi = int(QMAX[kind])
            return jax.random.randint(key, shape, -hi, hi + 1, jnp.int32).astype(dtype)
        return jnp.clip(jax.random.normal(key, shape) * 64.0, -448, 448).astype(dtype)
    return init


def const(v):
    return lambda key, shape, dtype: jnp.full(shape, v, dtype)


class QBase(nn.Module):
    """Shared parameter declarations of QDense / QConv."""

    def _weight_scale(self, scheme: QScheme, k_in: int, k_out: int, fan_in: int):
        std = 1.0 / math.sqrt(max(fan_in, 1))
        v = 3.0 * std / QMAX[scheme.weight]
        if scheme.weight_gran == "tensor":
            shape = ()
        elif scheme.weight_gran == "channel":
            shape = (k_out,)
        else:
            bk, bn = scheme.weight_block
            shape = (-(-k_in // bk), -(-k_out // bn))
        return self.param("kernel_scale", const(v), shape, jnp.float32)

    def _input_quant(self, scheme: QScheme, x, x_scale=None, x_zero_point=None):
        """Returns (x_q, x_scale broadcastable to the accumulator, zero point or None).

        An input that is already quantized comes with its producer's scale (one scale
        per tensor, as in a TensorRT / TFLite calibration table); otherwise the layer
        owns a calibrated `input_scale` (static) or computes one (dynamic)."""
        kind = scheme.act_kind
        if x_scale is not None:
            assert x.dtype == DTYPES[kind], (x.dtype, kind)
            return x, x_scale, x_zero_point
        if scheme.act == "static":
            s = self.param("input_scale", const(4.0 / QMAX[kind]), (), jnp.float32)
            zp = None
            if scheme.act_zero_point:
                zp = self.param("input_zero_point", const(0), (), jnp.int32)
            if x.dtype == DTYPES[kind]:
                return x, s, zp                      # already quantized by its producer
            return quantize(x, s, kind, zp), s, zp
        if scheme.act == "token":
            s = dynamic_scale(x, kind, axis=-1)
            return quantize(x, s, kind), s, None
        raise ValueError(scheme.act)

    def _epilogue(self, scheme: QScheme, acc, scale, bias_shape, use_bias):
        if scheme.out == "int8":
            if use_bias or scheme.act_zero_point:
                # int32 bias in the accumulator domain (bias / (s_x s_w) - zp_x * sum_k w)
                acc = acc + self.param("bias", nn.initializers.zeros, bias_shape, jnp.int32)
            out_s = self.param("output_scale", const(4.0 / 127), (), jnp.float32)
            out_zp = self.param("output_zero_point", const(0), (), jnp.int32)
            y = acc.astype(jnp.float32) * (scale / out_s)
            y = jnp.round(y) + out_zp
            lo = -128 if scheme.act_zero_point else -127
            return jnp.clip(y, lo, 127).astype(INT8)
        y = acc.astype(jnp.float32) * scale
        if use_bias:
            y = y + self.param("bias", nn.initializers.zeros, bias_shape, jnp.float32)
        return y.astype(self.dtype)


class QDense(QBase):
    """Quantized y = x @ W (+ b) over the last axis; kernel [in, out] like nn.Dense."""
    features: int
    scheme: QScheme
    use_bias: bool = True
    dtype: str = "float32"          # epilogue / output dtype when scheme.out == "float"

    @nn.compact
    def __call__(self, x, x_scale=None, x_zero_point=None):
        sc = self.scheme
        k_in = x.shape[-1]
        if sc.act == "none":         # weight-only: dequantize, 16-bit GEMM
            w = self.param("kernel", rand_q(sc.weight), (k_in, self.features), DTYPES[sc.weight])
            ws = self._weight_scale(sc, k_in, self.features, k_in)
            wf = dequantize_weight(w, ws, sc, self.dtype)
            y = lax.dot_general(x.astype(self.dtype), wf, (((x.ndim - 1,), (0,)), ((), ())))
            if self.use_bias:
                y = y + self.param("bias", nn.initializers.zeros, (self.features,),
                                   jnp.float32).astype(self.dtype)
            return y
        w = self.param("kernel", rand_q(sc.weight), (k_in, self.features), DTYPES[sc.weight])
        ws = self._weight_scale(sc, k_in, self.features, k_in)
        if sc.act == "block":
            y = block_scaled_dot(x, w, ws, sc.weight_block, sc.act_block)
            if self.use_bias:
                y = y + self.param("bias", nn.initializers.zeros, (self.features,), jnp.float32)
            return y.astype(self.dtype)
        xq, xs, _ = self._input_quant(sc, x, x_scale, x_zero_point)
        acc = lax.dot_general(xq, w, (((x.ndim - 1,), (0,)), ((), ())),
                              preferred_element_type=ACC[sc.act_kind])
        return self._epilogue(sc, acc, xs * ws, (self.features,), self.use_bias)


def _conv_dims(ndim: int) -> lax.ConvDimensionNumbers:
    """Channels-last input/output (N, *spatial, C) and (*spatial, I, O) kernels (Flax layout)."""
    sp = tuple(range(1, ndim - 1))
    return lax.ConvDimensionNumbers((0, ndim - 1) + sp, (ndim - 1, ndim - 2) + tuple(range(ndim - 2)),
                                    (0, ndim - 1) + sp)


class QConv(QBase):
    """Quantized N-d convolution, channels-last, kernel (*k, in/groups, out) like nn.Conv.

    `padding` is explicit per spatial dim; `lhs_dilation` > 1 gives a transposed conv.
    With asymmetric activations the input is padded with its zero point (TFLite).
    """
    features: int
    kernel_size: Sequence[int]
    scheme: QScheme
    strides: Sequence[int] | None = None
    padding: Sequence[tuple[int, int]] | None = None
    lhs_dilation: Sequence[int] | None = None
    rhs_dilation: Sequence[int] | None = None
    feature_group_count: int = 1
    use_bias: bool = True
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, x_scale=None, x_zero_point=None):
        sc = self.scheme
        nsp = len(self.kernel_size)
        c_in = x.shape[-1]
        kshape = tuple(self.kernel_size) + (c_in // self.feature_group_count, self.features)
        w = self.param("kernel", rand_q(sc.weight), kshape, DTYPES[sc.weight])
        ws = self._weight_scale(sc, math.prod(kshape[:-1]), self.features, math.prod(kshape[:-1]))
        pad = list(self.padding) if self.padding is not None else [(0, 0)] * nsp
        strides = tuple(self.strides) if self.strides is not None else (1,) * nsp
        conv = lambda a, b, pref: lax.conv_general_dilated(
            a, b, strides, pad, lhs_dilation=self.lhs_dilation, rhs_dilation=self.rhs_dilation,
            dimension_numbers=_conv_dims(x.ndim), feature_group_count=self.feature_group_count,
            preferred_element_type=pref)
        if sc.act == "none":
            y = conv(x.astype(self.dtype), dequantize_weight(w, ws, sc, self.dtype), None)
            if self.use_bias:
                y = y + self.param("bias", nn.initializers.zeros, (self.features,),
                                   jnp.float32).astype(self.dtype)
            return y
        xq, xs, zp = self._input_quant(sc, x, x_scale, x_zero_point)
        if zp is not None and any(p != (0, 0) for p in pad):
            xq = jnp.pad(xq, [(0, 0)] + pad + [(0, 0)], constant_values=zp.astype(xq.dtype))
            pad = [(0, 0)] * nsp
        acc = conv(xq, w, ACC[sc.act_kind])
        return self._epilogue(sc, acc, xs * ws, (self.features,), self.use_bias)


# --------------------------------------------------------------------------
# block-scaled FP8 GEMM, weight-only dequant
# --------------------------------------------------------------------------

def block_scaled_dot(x, w, w_scale, weight_block=(128, 128), act_block=128, kind="fp8",
                     batch_dims: int = 0):
    """Block-scaled GEMM: dynamic 1 x act_block activation groups along K, weight scales
    per (bk, bn) block. Each K-block is a low-precision partial product (batched over the
    K-blocks), scaled by its activation-group and weight-block scales, and summed in fp32.

    kind="fp8":  f8 x f8 -> f32 (DeepSeek-V3 / Qwen3 FP8 checkpoints; DeepGEMM / CUTLASS
                 blockwise kernels, 128x128 weight blocks, per-128-K promotion).
    kind="int8": i8 x i8 -> i32 (GGML Q4_0 x Q8_0 / Q6_K x Q8_K dot products: int4 / int6
                 weights unpacked to int8, activations quantized per 32 (or 256) along K).

    x: [*B, ..., K] with `batch_dims` leading batch axes B shared with w: [*B, K, N]
    (e.g. MoE experts); w_scale: [*B, ceil(K/bk), ceil(N/bn)].
    """
    bk, bn = weight_block
    assert act_block == bk, "activation group must equal the weight K-block"
    nb = batch_dims
    B = w.shape[:nb]
    K, N = w.shape[nb:]
    kb = -(-K // bk)
    lead = x.shape[nb:-1]
    xf = x.reshape(B + (-1, kb, bk))                                 # [*B, M, kb, bk]
    xs = dynamic_scale(xf, kind, axis=-1)                            # [*B, M, kb, 1]
    xq = quantize(xf, xs, kind)
    if w.dtype != DTYPES[kind]:
        w = w.astype(DTYPES[kind])                                   # unpack int4 / int6 -> int8
    wk = w.reshape(B + (kb, bk, N))                                  # [*B, kb, bk, N]
    bd = tuple(range(nb))
    part = lax.dot_general(xq, wk, (((nb + 2,), (nb + 1,)), (bd + (nb + 1,), bd + (nb,))),
                           preferred_element_type=ACC[kind]).astype(jnp.float32)  # [*B, kb, M, N]
    ws = jnp.repeat(w_scale.astype(jnp.float32), bn, axis=-1)[..., :N]  # [*B, kb, N]
    part = part * jnp.swapaxes(xs, nb, nb + 1) * jnp.expand_dims(ws, nb + 1)
    return jnp.sum(part, axis=nb).reshape(B + lead + (N,))


def dequantize_weight(w, scale, scheme: QScheme, dtype):
    """Weight-only path: int4 / int8 weights -> `dtype` with per-channel or per-block scales."""
    wf = w.astype(jnp.float32)
    if scheme.weight_gran == "block":
        bk, bn = scheme.weight_block
        s = jnp.repeat(jnp.repeat(scale.astype(jnp.float32), bk, axis=0), bn, axis=1)
        wf = wf * s[: w.shape[-2], : w.shape[-1]]
    else:
        wf = wf * scale.astype(jnp.float32)
    return wf.astype(dtype)


# --------------------------------------------------------------------------
# quantized storage: KV cache, embedding tables, feature tables
# --------------------------------------------------------------------------

class QEmbed(nn.Module):
    """int8 embedding table with symmetric scales, one per table (MLPerf DLRM) or one per
    row (CTranslate2 int8 embeddings). Gathered rows are dequantized to `dtype`."""
    num_embeddings: int
    features: int
    dtype: str = "float32"
    per_row: bool = False

    @nn.compact
    def __call__(self, ids):
        table = self.param("embedding", rand_q("int8"), (self.num_embeddings, self.features), INT8)
        if self.per_row:
            s = self.param("embedding_scale", const(0.05 / 127), (self.num_embeddings, 1),
                           jnp.float32)
            return dequantize(jnp.take(table, ids, axis=0), jnp.take(s, ids, axis=0), self.dtype)
        s = self.param("embedding_scale", const(0.05 / 127), (), jnp.float32)
        return dequantize(jnp.take(table, ids, axis=0), s, self.dtype)


class RowwiseQEmbed(nn.Module):
    """FBGEMM rowwise-quantized embedding table: uint8 (or uint4) rows with a per-row
    fp16 scale and bias, x = q * scale + bias. Gathered rows are dequantized to `dtype`."""
    num_embeddings: int
    features: int
    bits: int = 8
    dtype: str = "float32"

    @nn.compact
    def __call__(self, ids):
        qd = UINT8 if self.bits == 8 else jnp.uint4
        hi = 2 ** self.bits - 1
        table = self.param("embedding",
                           lambda k, s, d: jax.random.randint(k, s, 0, hi + 1, jnp.int32).astype(d),
                           (self.num_embeddings, self.features), qd)
        sb = self.param("scale_bias", const(0.0), (self.num_embeddings, 2), jnp.float16)
        rows = jnp.take(table, ids, axis=0).astype(jnp.float32)
        rsb = jnp.take(sb, ids, axis=0).astype(jnp.float32)
        return (rows * rsb[..., 0:1] + rsb[..., 1:2]).astype(self.dtype)


def kv_store(x, scale):
    """Quantize a K or V slice for an FP8 KV cache (per-tensor calibrated scale)."""
    return quantize(x, scale, "fp8")


def kv_load(q, scale, dtype):
    return dequantize(q, scale, dtype)
