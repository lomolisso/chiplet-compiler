"""MLPerf 3D U-Net (KiTS19), INT8 Flax linen port.

Mirrors `Unet3D` in mlcommons/inference
`vision/medical_imaging/3d-unet-kits19/pytorch_checkpoint_SUT.py` (identical
to mlcommons/training `image_segmentation/pytorch/model/unet3d.py` with
normalization="instancenorm", activation="relu"):

  * `conv_block_factory` = Conv3d (bias only when there is no norm) ->
    InstanceNorm3d(affine=True, eps=1e-5, no running stats) -> ReLU,
  * encoder downsampling by stride-2 3x3x3 convs (padding 1),
  * decoder upsampling by ConvTranspose3d(k=2, s=2, p=0, bias=True),
    then torch.cat((up, skip), channel) and two conv blocks,
  * 1x1x1 output conv with bias (n_class=3).

Layout is NDHWC (JAX convention); torch is NCDHW. Module names follow the
torch state-dict names with `Sequential` indices folded in
(`downsample.0.conv1.0` -> `downsample_0/conv1_0`).

Quantization: the MLPerf NVIDIA TensorRT INT8 engine (inference_results_v5.0
closed/NVIDIA/code/3d-unet/tensorrt: README, builder.py, calibrator.py; code/plugin):
  * INT8 meets both the 99% and 99.9% targets. Activations int8 per-tensor symmetric
    (MinMax calibration); the network input is INT8 (preprocessed, quantized volume).
  * Convs: int8 weights, per-output-channel scales (TensorRT), except the first conv
    (CONV3D3X3X3C1K32 plugin) and the output conv (CONV3D1X1X1K4 plugin), which use one
    weight scale per tensor. Each conv's int32 accumulator is scaled in fp32 and
    requantized to int8 (`<conv>_quant`).
  * InstanceNorm3d + ReLU: the INSTNORM3D plugin, int8 in / int8 out with fp32
    statistics (qops.InstanceNorm3dInt8).
  * Every ConvTranspose3d(2, 2) is rewritten as NVIDIA's conv_for_deconv does: a 1x1x1
    int8 conv to 8 * Cout channels + PIXELSHUFFLE3D (qops.DeconvAsConv); the concat with
    the skip is int8, requantized to one scale.
  * The output conv's epilogue writes fp16 logits (FP16 LINEAR network output).
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

from models import quant as q
from models.unet3d import qops


@dataclass(frozen=True)
class Config:
    batch: int = 1
    roi: int = 128                                   # ROI_SHAPE = [128, 128, 128]
    in_channels: int = 1
    n_class: int = 3
    filters: tuple[int, ...] = (32, 64, 128, 256, 320)


def conv_block(t, out_channels: int, name: str, stride: int = 1,
               first: bool = False):
    """`conv_block_factory` (regular conv, instancenorm, relu): Sequential(conv, norm, act)
    -> modules `<name>_0` (int8 conv, no bias), `<name>_1` (int8 InstanceNorm + ReLU)."""
    scheme = q.INT8_TENSOR_W if first else q.INT8_STATIC
    y = q.QConv(out_channels, (3, 3, 3), scheme, strides=(stride,) * 3, padding=[(1, 1)] * 3,
                use_bias=False, name=f"{name}_0")(*t)
    y, s, _ = q.ActQuant(name=f"{name}_0_quant")(y)
    return qops.InstanceNorm3dInt8(name=f"{name}_1")((y, s))


class DownsampleBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, t):
        t = conv_block(t, self.out_channels, "conv1", stride=2)
        return conv_block(t, self.out_channels, "conv2")


class UpsampleBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, t, skip):
        # upsample_conv: ConvTranspose3d(k2, s2) -> 1x1x1 conv (8x channels) + pixel shuffle
        t = qops.DeconvAsConv(self.out_channels, name="upsample_conv_0")(t)
        t = qops.qcat((t, skip), q.ActQuant(name="cat_quant"))   # torch.cat((x, skip), dim=1)
        t = conv_block(t, self.out_channels, "conv1")
        return conv_block(t, self.out_channels, "conv2")


class InputBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, t):
        t = conv_block(t, self.out_channels, "conv1", first=True)
        return conv_block(t, self.out_channels, "conv2")


class OutputLayer(nn.Module):
    n_class: int

    @nn.compact
    def __call__(self, t):
        # CONV3D1X1X1K4 plugin: int8 input, per-tensor int8 weights, fp32 bias, fp16 output
        return q.QConv(self.n_class, (1, 1, 1), q.INT8_TENSOR_W, use_bias=True,
                       dtype="float16", name="conv_0")(*t)


class Unet3D(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x):                            # x: int8 [N, D, H, W, 1]
        filters = self.cfg.filters
        inp, out = filters[:-1], filters[1:]
        s_in = self.param("input_scale", q.const(4.0 / 127), (), jnp.float32)
        t = InputBlock(filters[0], name="input_block")((x, s_in))
        outputs = [t]
        for i, o in enumerate(out):
            t = DownsampleBlock(o, name=f"downsample_{i}")(t)
            outputs.append(t)
        t = DownsampleBlock(filters[-1], name="bottleneck")(t)
        ups = [filters[-1]] + list(reversed(inp))
        for i, (o, skip) in enumerate(zip(ups, reversed(outputs))):
            t = UpsampleBlock(o, name=f"upsample_{i}")(t, skip)
        return OutputLayer(self.cfg.n_class, name="output")(t)
