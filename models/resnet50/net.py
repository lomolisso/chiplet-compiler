"""ResNet-50 v1.5 (MLPerf Inference image classification), INT8 Flax linen port.

Architecture mirrors torchvision.models.resnet50 exactly: stride on the 3x3 conv of each
bottleneck (v1.5), explicit symmetric padding (torch semantics -- *not* padding="SAME",
which pads stride-2 convs asymmetrically), inference-mode BN. Module names follow
torchvision's (dots -> underscores).

Quantization: the MLPerf NVIDIA TensorRT INT8 engine (inference_results_v5.0
closed/NVIDIA/code/resnet50/tensorrt: README "Calibration", calibrator.cache,
rn50_graphsurgeon.py):
  * TensorRT symmetric INT8; activations per-tensor, EntropyCalibration2 scales (one per
    tensor in calibrator.cache); conv weights per-output-channel symmetric (TRT default).
  * The network input is INT8 (preprocessed images, `input_tensor_0` scale).
  * BN folded into each conv (weights and bias); ReLU and the residual add run in the
    conv epilogue (fp32) and the result is requantized to INT8.
  * Max pool and global average pool run on INT8 tensors (pool1 shares the ReLU scale).
  * The FC is an INT8 GEMM (NVIDIA replaces it with a 1x1 conv, `fc_replaced`); softmax
    is removed and TopK follows (TopK is host-side here, see model.py).
Graph form: explicit Q/DQ (see models/quant.py); every conv / the FC is i8 x i8 -> i32.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp
from jax import lax

from models import quant as q


@dataclass(frozen=True)
class Config:
    batch: int = 1
    image: int = 224
    num_classes: int = 1000
    layers: tuple[int, ...] = (3, 4, 6, 3)
    width: int = 64


def conv(features: int, k: int, stride: int, name: str) -> q.QConv:
    """INT8 conv with the BN folded in (bias), per-channel weight scales."""
    return q.QConv(features, (k, k), q.INT8_STATIC, strides=(stride, stride),
                   padding=[(k // 2, k // 2)] * 2, use_bias=True, name=name)


class Bottleneck(nn.Module):
    planes: int
    stride: int
    downsample: bool

    @nn.compact
    def __call__(self, x, s):                    # x: int8 block input, s: its scale
        out = nn.relu(conv(self.planes, 1, 1, "conv1")(x, s))
        out, so, _ = q.ActQuant(name="relu1_quant")(out)
        out = nn.relu(conv(self.planes, 3, self.stride, "conv2")(out, so))
        out, so, _ = q.ActQuant(name="relu2_quant")(out)
        out = conv(self.planes * 4, 1, 1, "conv3")(out, so)
        if self.downsample:
            identity = conv(self.planes * 4, 1, self.stride, "downsample_0")(x, s)
        else:
            identity = q.dequantize(x, s)
        y, sy, _ = q.ActQuant(name="relu3_quant")(nn.relu(out + identity))
        return y, sy


class ResNet50(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x):                       # x: int8 [N, H, W, 3] (preprocessed, quantized)
        c = self.cfg
        s_in = self.param("input_scale", lambda *_: jnp.float32(2.64 / 127), (), jnp.float32)
        x = nn.relu(conv(c.width, 7, 2, "conv1")(x, s_in))
        x, s, _ = q.ActQuant(name="relu_quant")(x)
        x = lax.reduce_window(x, jnp.int8(-128), lax.max, (1, 3, 3, 1), (1, 2, 2, 1),
                              ((0, 0), (1, 1), (1, 1), (0, 0)))
        in_planes = c.width
        for li, n in enumerate(c.layers):
            planes = c.width * 2 ** li
            for bi in range(n):
                stride = 2 if (bi == 0 and li > 0) else 1
                ds = bi == 0 and (stride != 1 or in_planes != planes * 4)
                x, s = Bottleneck(planes, stride, ds, name=f"layer{li + 1}_{bi}")(x, s)
                in_planes = planes * 4
        # INT8 global average pool: int32 sum, scaled mean, requantized for the FC.
        hw = x.shape[1] * x.shape[2]
        acc = jnp.sum(x.astype(jnp.int32), axis=(1, 2))
        x, s, _ = q.ActQuant(name="avgpool_quant")(acc.astype(jnp.float32) * (s / hw))
        return q.QDense(c.num_classes, q.INT8_STATIC, name="fc")(x, s)
