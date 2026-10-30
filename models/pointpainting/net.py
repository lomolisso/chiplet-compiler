"""PointPainting (MLPerf Automotive 3D object detection), INT8 Flax linen port.

Two network components, ported from the MLPerf reference at
mlcommons/mlperf_automotive, automotive/3d-object-detection (commit 58100ef):

* `DeepLabV3Plus` -- model/segmentation/{modeling,_deeplab,utils}.py +
  backbone/resnet.py: `deeplabv3plus_resnet50(num_classes=19, output_stride=16)`
  (VainF DeepLabV3Plus-Pytorch; *not* torchvision's DeepLabV3). ResNet-50 with
  layer4 stride replaced by dilation 2, ASPP rates (6, 12, 18), low-level
  projection 256->48, 3x3 classifier on concat(48+256), final bilinear upsample
  to the input size (align_corners=False).
* `PointPillarsCore` -- model/pointpillars_core.py, painted=True:
  PillarEncoder (PFN: 16-ch features -> Conv1d(k=1) 64 -> BN -> ReLU -> max over
  points, scatter to the BEV canvas), 3-stage Backbone, ConvTranspose2d Neck,
  anchor Head (cls / reg / dir 1x1 convs).

Quantization (models/pointpainting/recipe.py): the automotive INT8 W8A8 fallback as a
TensorRT INT8 engine, with the PointPillars mixed-precision carve-outs of the only
published layer list (arXiv 2601.12638, Table III; CUDA-PointPillars and TAO PointPillars
are fp16-only, MLPerf PointPainting is fp32-only):
  * every conv is INT8: weights per-output-channel symmetric with the BatchNorm folded in
    (weights and bias), activations per-tensor static symmetric; i8 x i8 -> i32 with an
    fp32 epilogue (scale, bias, residual add, ReLU), requantized by the producer
    (`*_quant`) when the next layer is INT8, fp16 otherwise;
  * fp16 (TensorRT fp16 layers, BN folded): the PFN Conv1d, the second 3x3 conv of
    backbone stage 0 (`multi_blocks_0_3`) and the box-regression head conv (`conv_reg`);
    the PFN feature construction, max over points and the pillar scatter run in fp16;
  * TensorRT's Resize has no INT8 type: the bilinear upsamples run in fp16; branches that
    are concatenated (ASPP, decoder, neck) are fp16 and quantized once after the concat;
  * the neck's ConvTranspose2d(k = s, stride = s) is non-overlapping and is computed as a
    1x1 INT8 conv to s^2 * C_out channels + depth-to-space (NVIDIA's rewrite for 3D U-Net;
    same math, no zero-dilated input);
  * the camera image enters as the fp32 normalized tensor, quantized by `conv1`; the
    segmenter logits are returned as fp32, the head outputs in fp16.

Layout is NHWC (torch is NCHW); module names follow torch's with
`name.<int>` -> `name_<int>`; folded BatchNorms have no module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp
from jax import lax

from models import quant as q

F16 = jnp.float16


@dataclass(frozen=True)
class SegConfig:
    num_classes: int = 19
    output_stride: int = 16
    layers: tuple[int, ...] = (3, 4, 6, 3)


@dataclass(frozen=True)
class PPConfig:
    nclasses: int = 3
    voxel_size: tuple[float, float, float] = (0.32, 0.32, 6.0)
    point_cloud_range: tuple[float, ...] = (-74.88, -74.88, -2.0, 74.88, 74.88, 4.0)
    max_num_points: int = 20
    max_voxels: int = 32000
    point_dim: int = 11            # x, y, z, intensity, elongation + 6 painted class scores
    out_channel: int = 64
    backbone_out: tuple[int, ...] = (64, 128, 256)
    layer_nums: tuple[int, ...] = (3, 5, 5)
    layer_strides: tuple[int, ...] = (1, 2, 2)
    neck_strides: tuple[int, ...] = (1, 2, 4)
    neck_out: tuple[int, ...] = (128, 128, 128)
    # fp16 carve-outs (arXiv 2601.12638 Table III, mapped onto the MLPerf module names)
    fp16_convs: tuple[str, ...] = ("multi_blocks_0_3",)

    @property
    def x_l(self) -> int:
        r, v = self.point_cloud_range, self.voxel_size
        return math.ceil((r[3] - r[0]) / v[0])

    @property
    def y_l(self) -> int:
        r, v = self.point_cloud_range, self.voxel_size
        return math.ceil((r[4] - r[1]) / v[1])


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def qconv(features, k, stride=1, pad=0, dilation=1, dtype=jnp.float32, name=None):
    """INT8 conv with the BN folded in (bias), per-channel weight scales."""
    return q.QConv(features, (k, k), q.INT8_STATIC, strides=(stride, stride),
                   padding=[(pad, pad)] * 2, rhs_dilation=(dilation, dilation),
                   use_bias=True, dtype=dtype, name=name)


def fconv(features, k, stride=1, pad=0, name=None):
    """fp16 conv (TensorRT fp16 layer) with the BN folded in (bias)."""
    return nn.Conv(features, (k, k), (stride, stride), padding=[(pad, pad)] * 2,
                   use_bias=True, dtype=F16, param_dtype=F16, name=name)


def _linear_coeffs(in_size, out_size):
    """torch upsample_bilinear (align_corners=False, size given) source indices /
    weights along one axis, computed in float32 like ATen's CPU kernel.

    Computed on the device from the (possibly symbolic) sizes, as ATen does inside the
    upsample kernel (not a NumPy table frozen at one camera size)."""
    f32 = jnp.float32
    scale = jnp.asarray(in_size, f32) / jnp.asarray(out_size, f32)
    dst = jnp.arange(out_size, dtype=f32)
    src = jnp.maximum(scale * (dst + f32(0.5)) - f32(0.5), f32(0))
    last = jnp.asarray(in_size, jnp.int32) - 1
    i0 = jnp.minimum(jnp.floor(src).astype(jnp.int32), last)
    i1 = jnp.minimum(i0 + 1, last)
    lam = jnp.clip(src - i0.astype(f32), 0, 1)
    return i0, i1, (f32(1) - lam), lam


def _interp_axis(x, out_size, axis: int):
    i0, i1, w0, w1 = _linear_coeffs(x.shape[axis], out_size)
    shape = [1] * x.ndim
    shape[axis] = out_size
    a, b = jnp.take(x, i0, axis=axis), jnp.take(x, i1, axis=axis)
    return a * w0.reshape(shape).astype(x.dtype) + b * w1.reshape(shape).astype(x.dtype)


def interpolate_bilinear(x, size):
    """F.interpolate(x, size, mode='bilinear', align_corners=False) on NHWC, in x's dtype
    (fp16: TensorRT Resize). Separable: W first, then H (ATen's CPU accumulation order)."""
    if x.shape[1:3] == (1, 1):
        # ASPPPooling: upsampling a 1x1 map is a broadcast.
        return jnp.broadcast_to(x, (x.shape[0], *size, x.shape[3]))
    x = _interp_axis(x, size[1], axis=2)
    return _interp_axis(x, size[0], axis=1)


def depth_to_space(x, s: int):
    """[N, H, W, s*s*C] with channels ordered (dy, dx, c) -> [N, H*s, W*s, C]."""
    n, h, w, c = x.shape
    x = x.reshape(n, h, w, s, s, c // (s * s)).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(n, h * s, w * s, c // (s * s))


# ----------------------------------------------------------------------------
# DeepLabV3+ / ResNet-50 (output stride 16)
# ----------------------------------------------------------------------------

class Bottleneck(nn.Module):
    planes: int
    stride: int
    dilation: int
    downsample: bool

    @nn.compact
    def __call__(self, x, s):                    # x: int8 block input, s: its scale
        out = nn.relu(qconv(self.planes, 1, name="conv1")(x, s))
        out, so, _ = q.ActQuant(name="relu1_quant")(out)
        out = nn.relu(qconv(self.planes, 3, self.stride, self.dilation, self.dilation,
                            name="conv2")(out, so))
        out, so, _ = q.ActQuant(name="relu2_quant")(out)
        out = qconv(self.planes * 4, 1, name="conv3")(out, so)
        if self.downsample:
            identity = qconv(self.planes * 4, 1, self.stride, name="downsample_0")(x, s)
        else:
            identity = q.dequantize(x, s)
        y, sy, _ = q.ActQuant(name="relu3_quant")(nn.relu(out + identity))
        return y, sy


class ResNetBackbone(nn.Module):
    """backbone/resnet.py ResNet up to layer4 (IntermediateLayerGetter keeps
    layer1 -> 'low_level' and layer4 -> 'out'; avgpool / fc are dropped).
    Returns {name: (int8 tensor, scale)}."""
    layers: tuple[int, ...]
    replace_stride_with_dilation: tuple[bool, bool, bool]

    @nn.compact
    def __call__(self, x):                       # x: fp32 normalized image
        x = nn.relu(qconv(64, 7, 2, 3, name="conv1")(x))
        x, s, _ = q.ActQuant(name="relu_quant")(x)
        x = lax.reduce_window(x, jnp.int8(-128), lax.max, (1, 3, 3, 1), (1, 2, 2, 1),
                              ((0, 0), (1, 1), (1, 1), (0, 0)))
        inplanes, dilation, feats = 64, 1, []
        for li, n in enumerate(self.layers):
            planes = 64 * 2 ** li
            stride = 1 if li == 0 else 2
            previous_dilation = dilation
            if li > 0 and self.replace_stride_with_dilation[li - 1]:
                dilation *= stride
                stride = 1
            ds = stride != 1 or inplanes != planes * 4
            x, s = Bottleneck(planes, stride, previous_dilation, ds, name=f"layer{li + 1}_0")(x, s)
            inplanes = planes * 4
            for bi in range(1, n):
                x, s = Bottleneck(planes, 1, dilation, False, name=f"layer{li + 1}_{bi}")(x, s)
            feats.append((x, s))
        return {"low_level": feats[0], "out": feats[-1]}


class ASPP(nn.Module):
    atrous_rates: tuple[int, int, int]

    @nn.compact
    def __call__(self, x, s):                    # x: int8 layer4 output, s: its scale
        oc = 256
        # branch outputs are fp16 (concatenated, then quantized once for `project`)
        res = [nn.relu(qconv(oc, 1, dtype=F16, name="convs_0_0")(x, s))]
        for i, rate in enumerate(self.atrous_rates, start=1):      # ASPPConv
            res.append(nn.relu(qconv(oc, 3, 1, rate, rate, dtype=F16, name=f"convs_{i}_0")(x, s)))
        # ASPPPooling: AdaptiveAvgPool2d(1) on INT8 (int32 sum, scaled mean, requantized)
        # -> 1x1 conv -> ReLU -> upsample (a broadcast)
        hw = x.shape[1] * x.shape[2]
        acc = jnp.sum(x.astype(jnp.int32), axis=(1, 2), keepdims=True)
        p, sp, _ = q.ActQuant(name="pool_quant")(acc.astype(jnp.float32) * (s / hw))
        p = nn.relu(qconv(oc, 1, dtype=F16, name="convs_4_1")(p, sp))
        res.append(interpolate_bilinear(p, x.shape[1:3]))
        res = jnp.concatenate(res, axis=-1)
        # project: conv -> ReLU -> Dropout(0.1) (identity at inference); fp16 for the Resize
        return nn.relu(qconv(oc, 1, dtype=F16, name="project_0")(res))


class DeepLabHeadV3Plus(nn.Module):
    num_classes: int
    aspp_dilate: tuple[int, int, int]

    @nn.compact
    def __call__(self, feature):
        low = nn.relu(qconv(48, 1, dtype=F16, name="project_0")(*feature["low_level"]))
        out = ASPP(self.aspp_dilate, name="aspp")(*feature["out"])
        out = interpolate_bilinear(out, low.shape[1:3])
        x = jnp.concatenate([low, out], axis=-1)
        x = nn.relu(qconv(256, 3, 1, 1, name="classifier_0")(x))
        x, s, _ = q.ActQuant(name="classifier_quant")(x)
        return qconv(self.num_classes, 1, dtype=F16, name="classifier_3")(x, s)


class DeepLabV3Plus(nn.Module):
    """modeling.deeplabv3plus_resnet50 -> _SimpleSegmentationModel.forward."""
    cfg: SegConfig

    @nn.compact
    def __call__(self, x):                       # x: [N, H, W, 3] fp32 normalized image
        c = self.cfg
        if c.output_stride == 8:
            rswd, aspp_dilate = (False, True, True), (12, 24, 36)
        else:
            rswd, aspp_dilate = (False, False, True), (6, 12, 18)
        features = ResNetBackbone(c.layers, rswd, name="backbone")(x)
        y = DeepLabHeadV3Plus(c.num_classes, aspp_dilate, name="classifier")(features)
        return interpolate_bilinear(y, x.shape[1:3]).astype(jnp.float32)


# ----------------------------------------------------------------------------
# PointPillars core
# ----------------------------------------------------------------------------

class PillarEncoder(nn.Module):
    """pointpillars_core.PillarEncoder on a static, padded pillar set, in fp16.

    pillars [P, max_num_points, point_dim], coors [P, 3] (x, y, z voxel index,
    as produced by the voxelizer, batch column dropped -- batch 1),
    npoints [P] (0 for padding slots). Returns the fp16 BEV canvas [1, y_l, x_l, C].
    """
    cfg: PPConfig

    @nn.compact
    def __call__(self, pillars, coors, npoints):
        c = self.cfg
        pillars = pillars.astype(F16)            # fp32 voxelizer output -> fp16 engine input
        vx, vy = c.voxel_size[0], c.voxel_size[1]
        x_offset = c.voxel_size[0] / 2 + c.point_cloud_range[0]
        y_offset = c.voxel_size[1] / 2 + c.point_cloud_range[1]
        valid = npoints > 0
        # padding slots have npoints == 0; clamp so they stay finite (they are
        # dropped at the scatter). Identity for every real pillar (npoints >= 1).
        n = jnp.maximum(npoints, 1).astype(F16)

        # 1. offset to the points center
        offset_pt_center = pillars[:, :, :3] - jnp.sum(pillars[:, :, :3], axis=1,
                                                       keepdims=True) / n[:, None, None]
        # 2. offset to the pillar center
        cf = coors.astype(F16)
        x_offset_pi_center = pillars[:, :, :1] - (cf[:, None, 0:1] * vx + x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (cf[:, None, 1:2] * vy + y_offset)
        # 3. features = cat[pillars, offset_pt_center, x_off, y_off] with
        #    channels 0/1 overwritten by x_off / y_off (as in the reference)
        features = jnp.concatenate([x_offset_pi_center, y_offset_pi_center, pillars[:, :, 2:],
                                    offset_pt_center, x_offset_pi_center, y_offset_pi_center],
                                   axis=-1)
        # 4. zero the padded points
        voxel_ids = jnp.arange(pillars.shape[1])
        mask = voxel_ids[None, :] < npoints[:, None]
        features = features * mask[:, :, None].astype(features.dtype)
        # 5. embedding: fp16 Conv1d(k=1) with BN1d folded -> ReLU -> max over points
        features = nn.Conv(c.out_channel, (1,), use_bias=True, dtype=F16, param_dtype=F16,
                           name="conv")(features)
        pooling_features = jnp.max(nn.relu(features), axis=1)          # [P, C]
        # 6. pillar scatter (fp16, PPScatterPlugin). Static shape: padding slots are sent
        #    to a dummy row (index y_l * x_l) of a canvas with one extra row, sliced off.
        dummy = c.y_l * c.x_l
        flat_idx = jnp.where(valid, coors[:, 1] * c.x_l + coors[:, 0], dummy)
        canvas = jnp.zeros((dummy + 1, c.out_channel), F16)
        canvas = canvas.at[flat_idx].set(pooling_features)
        return canvas[:dummy].reshape(1, c.y_l, c.x_l, c.out_channel)


class Backbone(nn.Module):
    """Returns the stage outputs as (tensor, scale): int8 + scale, or fp16 + None."""
    cfg: PPConfig

    @nn.compact
    def __call__(self, x):                       # x: fp16 canvas
        c, outs, s = self.cfg, [], None
        for i, (oc, n, st) in enumerate(zip(c.backbone_out, c.layer_nums, c.layer_strides)):
            names = [f"multi_blocks_{i}_0"] + [f"multi_blocks_{i}_{3 * j + 3}" for j in range(n)]
            for j, name in enumerate(names):
                stride = st if j == 0 else 1
                nxt = names[j + 1] if j + 1 < n + 1 else None
                if name in c.fp16_convs:
                    x = nn.relu(fconv(oc, 3, stride, 1, name=name)(x))
                    s = None
                    if nxt is not None and nxt not in c.fp16_convs:
                        x, s, _ = q.ActQuant(name=f"{name}_quant")(x)
                    continue
                out_f16 = nxt in c.fp16_convs
                x = nn.relu(qconv(oc, 3, stride, 1, dtype=F16 if out_f16 else jnp.float32,
                                  name=name)(x, s))
                s = None
                if not out_f16:
                    x, s, _ = q.ActQuant(name=f"{name}_quant")(x)
            outs.append((x, s))
        return outs


class Neck(nn.Module):
    cfg: PPConfig

    @nn.compact
    def __call__(self, xs):
        c, outs = self.cfg, []
        for i, (st, oc) in enumerate(zip(c.neck_strides, c.neck_out)):
            # ConvTranspose2d(k=s, stride=s, bias=False) + BN: non-overlapping, so a 1x1
            # INT8 conv to s*s*oc channels (order dy, dx, oc; BN folded, bias repeated)
            # followed by depth-to-space.
            y = q.QConv(st * st * oc, (1, 1), q.INT8_STATIC, use_bias=True, dtype=F16,
                        name=f"decoder_blocks_{i}_0")(*xs[i])
            outs.append(depth_to_space(nn.relu(y), st))
        return jnp.concatenate(outs, axis=-1)


class Head(nn.Module):
    cfg: PPConfig

    @nn.compact
    def __call__(self, x):                       # x: fp16 neck output
        n_anchors = 2 * self.cfg.nclasses
        xq, s, _ = q.ActQuant(name="input_quant")(x)       # shared by conv_cls / conv_dir_cls
        cls = qconv(n_anchors * self.cfg.nclasses, 1, dtype=F16, name="conv_cls")(xq, s)
        reg = fconv(n_anchors * 7, 1, name="conv_reg")(x)  # fp16 carve-out
        dir_cls = qconv(n_anchors * 2, 1, dtype=F16, name="conv_dir_cls")(xq, s)
        return cls, reg, dir_cls


class PointPillarsCore(nn.Module):
    """PointPillarsCore.forward up to the head (anchor generation / decode /
    NMS are host post-processing)."""
    cfg: PPConfig

    @nn.compact
    def __call__(self, pillars, coors, npoints):
        pillar_features = PillarEncoder(self.cfg, name="pillar_encoder")(pillars, coors, npoints)
        xs = Backbone(self.cfg, name="backbone")(pillar_features)
        x = Neck(self.cfg, name="neck")(xs)
        return Head(self.cfg, name="head")(x)
