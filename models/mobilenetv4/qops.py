"""TFLite full-integer ops the shared library (models/quant.py) does not provide.

Every tensor between ops is int8 with a per-tensor (scale, zero point), as in a TFLite
full-integer model (x ~= (q - zp) * scale). Requantization uses fp32 scales, as
XNNPACK does (TFLite's reference kernels use the equivalent fixed-point multiplier).

  conv_out      a TFLite CONV_2D / DEPTHWISE_CONV_2D: q.QConv (INT8_TFLITE) plus its output
                (scale, zero point); a fused ReLU is the clamp at the output zero point
  qadd          ADD of two int8 tensors (residuals)
  qmean         MEAN over H, W (global average pool)
  QAffine       a BatchNorm that cannot be folded into a conv (it follows a residual add):
                int8 per-channel MUL then ADD constants, each op requantized
  qmatmul       BATCH_MATMUL of two int8 activations with zero points (gemmlowp form:
                i8 x i8 -> i32 product, zero-point corrections from int32 row / column sums)
  softmax_int8  SOFTMAX int8 -> int8 with the fixed output (1/256, -128)
  requant       QUANTIZE between int8 / uint8 (network input and output)
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

from models import quant as q

SOFTMAX_SCALE, SOFTMAX_ZP = 1.0 / 256.0, -128          # LiteRT quantization spec, SOFTMAX output


def conv_out(conv: q.QConv, x, s, zp, relu: bool):
    """Runs a TFLite int8 conv on (x, s, zp); returns its int8 output and (scale, zero point)."""
    y = conv(x, s, zp)
    so = conv.get_variable("params", "output_scale")
    zo = conv.get_variable("params", "output_zero_point")
    if relu:                                            # fused ReLU = clamp at the real 0
        y = jnp.maximum(y, jnp.clip(zo, -128, 127).astype(jnp.int8))
    return y, so, zo


def requant(x, s, zp, s_out, zp_out, kind: str = "int8"):
    return q.quantize(q.dequantize(x, s, zero_point=zp), s_out, kind, zp_out)


def qadd(a, b, out: q.ActQuant):
    """ADD: (qa - za) sa + (qb - zb) sb, requantized to the output (scale, zero point)."""
    (xa, sa, za), (xb, sb, zb) = a, b
    return out(q.dequantize(xa, sa, zero_point=za) + q.dequantize(xb, sb, zero_point=zb))


def qmean(x, s, zp, out: q.ActQuant, axis=(1, 2)):
    """MEAN: int32 sum of the int8 inputs, scaled by s / n, requantized."""
    n = 1
    for a in axis:
        n *= x.shape[a]
    acc = jnp.sum(x.astype(jnp.int32), axis=axis, keepdims=True)
    return out((acc.astype(jnp.float32) / n - zp) * s)


class QAffine(nn.Module):
    """Unfolded inference BatchNorm as TFLite runs it: MUL by an int8 per-channel constant,
    then ADD of an int8 per-channel constant (per-tensor constant scales), each requantized."""

    @nn.compact
    def __call__(self, x, s, zp):
        c = x.shape[-1]
        mul = self.param("mul", q.rand_q("int8"), (c,), jnp.int8)
        mul_s = self.param("mul_scale", q.const(1.0 / 127), (), jnp.float32)
        add = self.param("add", q.rand_q("int8"), (c,), jnp.int8)
        add_s = self.param("add_scale", q.const(1.0 / 127), (), jnp.float32)
        y, sy, zy = q.ActQuant(zero_point=True, name="mul_quant")(
            q.dequantize(x, s, zero_point=zp) * q.dequantize(mul, mul_s))
        return q.ActQuant(zero_point=True, name="add_quant")(
            q.dequantize(y, sy, zero_point=zy) + q.dequantize(add, add_s))


def qmatmul(a, b, out: q.ActQuant, transpose_b: bool):
    """BATCH_MATMUL of int8 activations a [..., M, K] and b [..., K, N] (or [..., N, K]
    with transpose_b), both asymmetric:
        sum_k (a - za)(b - zb) = a.b - za sum_k b - zb sum_k a + K za zb
    with a.b an i8 x i8 -> i32 GEMM; the int32 result is requantized to `out`."""
    (xa, sa, za), (xb, sb, zb) = a, b
    nb = xa.ndim - 2
    bd = tuple(range(nb))
    kb = nb + 1 if transpose_b else nb
    acc = lax.dot_general(xa, xb, (((nb + 1,), (kb,)), (bd, bd)), preferred_element_type=jnp.int32)
    sum_a = jnp.sum(xa.astype(jnp.int32), axis=-1, keepdims=True)              # [..., M, 1]
    sum_b = jnp.expand_dims(jnp.sum(xb.astype(jnp.int32), axis=kb), -2)        # [..., 1, N]
    k = xa.shape[-1]
    acc = acc - za * sum_b - zb * sum_a + k * za * zb
    return out(acc.astype(jnp.float32) * (sa * sb))


def softmax_int8(x, s, zp, axis=-1):
    """SOFTMAX int8 -> int8 (TFLite evaluates exp from a lookup table; fp32 here)."""
    p = jax.nn.softmax(q.dequantize(x, s, zero_point=zp), axis=axis)
    return q.quantize(p, SOFTMAX_SCALE, "int8", SOFTMAX_ZP), jnp.float32(SOFTMAX_SCALE), jnp.int32(SOFTMAX_ZP)
