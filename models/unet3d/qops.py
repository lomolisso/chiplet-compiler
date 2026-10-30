"""INT8 ops of NVIDIA's MLPerf 3D-UNet engine that models/quant.py does not provide.

Tensors between layers are int8 with one symmetric per-tensor scale, t = (x, scale).

  DeconvAsConv      ConvTranspose3d(k=2, s=2) as NVIDIA's UnetGraphSurgeon.conv_for_deconv
                    rewrites it: a 1x1x1 int8 conv to 8 * Cout channels (weights [Cin, Cout,
                    2, 2, 2] reshaped to [Cin, Cout * 8], bias repeated 8x), requantized to
                    int8, then PIXELSHUFFLE3D_TRT (int8 data movement)
  pixel_shuffle3d   channel c * 8 + (d * 4 + h * 2 + w) -> voxel (2D + d, 2H + h, 2W + w)
  InstanceNorm3dInt8  INSTNORM3D_TRT (InstanceNormalization_TRT v2, relu=1): int8 in, fp32
                    statistics and affine, fused ReLU, int8 out
  qcat              concat of int8 tensors requantized to one scale
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

from models import quant as q


def pixel_shuffle3d(x, r: int = 2):
    """[N, D, H, W, C * r^3] -> [N, rD, rH, rW, C] (NVIDIA's cdwh weight order)."""
    n, d, h, w, c8 = x.shape
    c = c8 // r ** 3
    x = x.reshape(n, d, h, w, c, r, r, r).transpose(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(n, d * r, h * r, w * r, c)


def qcat(ts, out: q.ActQuant):
    x, s, _ = out(jnp.concatenate([q.dequantize(x, s) for x, s in ts], axis=-1))
    return x, s


class DeconvAsConv(nn.Module):
    """ConvTranspose3d(k=2, s=2, p=0, bias) rewritten as 1x1x1 int8 conv + pixel shuffle.

    kernel: [1, 1, 1, Cin, Cout * 8] int8 (the torch weight [Cin, Cout, 2, 2, 2] reshaped),
    per-output-channel scales over the 8 * Cout channels; bias: torch's [Cout] (NVIDIA
    stores it repeated 8x; repeating it in the epilogue is the same arithmetic)."""
    features: int

    @nn.compact
    def __call__(self, t):
        x, s = t
        c8 = 8 * self.features
        w = self.param("kernel", q.rand_q("int8"), (1, 1, 1, x.shape[-1], c8), jnp.int8)
        ws = self.param("kernel_scale", q.const(3.0 / (127 * x.shape[-1] ** 0.5)), (c8,),
                        jnp.float32)
        bias = self.param("bias", nn.initializers.zeros, (self.features,), jnp.float32)
        acc = lax.conv_general_dilated(x, w, (1, 1, 1), [(0, 0)] * 3,
                                       dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
                                       preferred_element_type=jnp.int32)
        y = acc.astype(jnp.float32) * (s * ws) + jnp.repeat(bias, 8)
        y, sy, _ = q.ActQuant(name="out_quant")(y)
        return pixel_shuffle3d(y), sy


class InstanceNorm3dInt8(nn.Module):
    """InstanceNorm3d(affine=True, track_running_stats=False, eps 1e-5) + ReLU as the TRT
    plugin runs it: int8 input dequantized, per-sample per-channel mean / biased variance
    over D, H, W in fp32, affine, ReLU, requantized to int8 (`out_quant`)."""
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, t):
        x = q.dequantize(*t)
        c = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones, (c,))
        bias = self.param("bias", nn.initializers.zeros, (c,))
        mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
        d = x - mean
        var = jnp.mean(d * d, axis=(1, 2, 3), keepdims=True)
        y = jax.nn.relu(d * jax.lax.rsqrt(var + self.epsilon) * scale + bias)
        y, s, _ = q.ActQuant(name="out_quant")(y)
        return y, s
