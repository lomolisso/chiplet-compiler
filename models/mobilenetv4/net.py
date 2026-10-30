"""MobileNetV4 (Conv-Medium / Hybrid-Medium), TFLite full-integer INT8 Flax linen port of timm 1.0.30.

Mirrors `timm.models.mobilenetv3._gen_mobilenet_v4` + `MobileNetV3`, with the
blocks of `timm.models._efficientnet_blocks` (ConvBnAct, EdgeResidual,
UniversalInvertedResidual, MobileAttention) and
`timm.layers.MultiQueryAttention2d` / `LayerScale2d` / `BatchNormAct2d`.

* The network is driven by timm's own block arch strings (copied verbatim
  below), decoded the same way as `_decode_block_str` / `decode_arch_def` /
  `EfficientNetBuilder` (only the first block of a stage keeps its stride).
* Padding: timm's default `pad_type=''`, i.e. explicit *symmetric*
  `((s - 1) + d * (k - 1)) // 2` padding, NOT TF "SAME" (which timm only uses
  for `tf_` models). BN eps = 1e-5 everywhere (`nn.BatchNorm2d` defaults).
* Module names follow timm's: `blocks.S.B` -> `blocks_S_B`; all other
  sub-module names (conv_stem, dw_start, pw_exp, dw_mid, pw_proj, dw_end,
  attn/query/proj, attn/key/down_conv, ...) are kept.
* Layout is NHWC (JAX convention); timm is NCHW.

Quantization: TFLite full-integer INT8 PTQ, the scheme of MLPerf Mobile's
`MobileNetV4-Conv-Large-int8-ptq.tflite` (mlcommons/mobile_app_open
tflite_settings_android.pbtxt, image_classification_v2; read from its flatbuffer) and of
the LiteRT quantization spec:
  * Every tensor is int8 per-tensor asymmetric (scale, zero point); the network input
    and output are uint8 (one QUANTIZE each). Requantization uses fp32 scales (XNNPACK).
  * Convs and depthwise convs: int8 per-output-channel symmetric weights, int32 bias
    (folded BN shift and input zero-point correction), int8 output; ReLU is the clamp at
    the output zero point. Inputs are padded with their zero point.
  * BN after a conv and LayerScale (a per-channel constant MUL after pw_proj / dw_end /
    output.proj, no activation between) are folded into that conv's weights / bias (the
    converter's conv + mul fusion). The query scale key_dim**-0.5 is folded into the query
    projection. The MQA pre-norm BN follows a residual add, so it stays an int8 MUL + ADD.
  * Residual ADD and global average pool (MEAN) are int8 ops with their own output scale.
  * Classifier: FULLY_CONNECTED with a per-tensor int8 weight.
  * Mobile-MQA (hybrid): q k^T and attn v are int8 x int8 BATCH_MATMULs with zero points;
    SOFTMAX is int8 -> int8 with the fixed output (1/256, -128).
  * The output softmax is in the graph (int8), followed by QUANTIZE to uint8.
The hybrid variant has no deployed quantized version; it takes the same converter recipe
(mobile fallback, models/mobilenetv4_hybrid_m/recipe.py). Local ops: models/mobilenetv4/qops.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

from models import quant as q
from models.mobilenetv4 import qops

# --- timm arch defs (verbatim from timm/models/mobilenetv3.py:_gen_mobilenet_v4) ---
CONV_MEDIUM_ARCH = (
    ('er_r1_k3_s2_e4_c48',),
    ('uir_r1_a3_k5_s2_e4_c80', 'uir_r1_a3_k3_s1_e2_c80'),
    ('uir_r1_a3_k5_s2_e6_c160', 'uir_r2_a3_k3_s1_e4_c160', 'uir_r1_a3_k5_s1_e4_c160',
     'uir_r1_a3_k3_s1_e4_c160', 'uir_r1_a3_k0_s1_e4_c160', 'uir_r1_a0_k0_s1_e2_c160',
     'uir_r1_a3_k0_s1_e4_c160'),
    ('uir_r1_a5_k5_s2_e6_c256', 'uir_r1_a5_k5_s1_e4_c256', 'uir_r2_a3_k5_s1_e4_c256',
     'uir_r1_a0_k0_s1_e4_c256', 'uir_r1_a3_k0_s1_e4_c256', 'uir_r1_a3_k5_s1_e2_c256',
     'uir_r1_a5_k5_s1_e4_c256', 'uir_r2_a0_k0_s1_e4_c256', 'uir_r1_a5_k0_s1_e2_c256'),
    ('cn_r1_k1_s1_c960',),
)
HYBRID_MEDIUM_ARCH = (
    ('er_r1_k3_s2_e4_c48',),
    ('uir_r1_a3_k5_s2_e4_c80', 'uir_r1_a3_k3_s1_e2_c80'),
    ('uir_r1_a3_k5_s2_e6_c160', 'uir_r1_a0_k0_s1_e2_c160', 'uir_r1_a3_k3_s1_e4_c160',
     'uir_r1_a3_k5_s1_e4_c160', 'mqa_r1_k3_h4_s1_v2_d64_c160', 'uir_r1_a3_k3_s1_e4_c160',
     'mqa_r1_k3_h4_s1_v2_d64_c160', 'uir_r1_a3_k0_s1_e4_c160', 'mqa_r1_k3_h4_s1_v2_d64_c160',
     'uir_r1_a3_k3_s1_e4_c160', 'mqa_r1_k3_h4_s1_v2_d64_c160', 'uir_r1_a3_k0_s1_e4_c160'),
    ('uir_r1_a5_k5_s2_e6_c256', 'uir_r1_a5_k5_s1_e4_c256', 'uir_r2_a3_k5_s1_e4_c256',
     'uir_r1_a0_k0_s1_e2_c256', 'uir_r1_a3_k5_s1_e2_c256', 'uir_r1_a0_k0_s1_e2_c256',
     'uir_r1_a0_k0_s1_e4_c256', 'mqa_r1_k3_h4_s1_d64_c256', 'uir_r1_a3_k0_s1_e4_c256',
     'mqa_r1_k3_h4_s1_d64_c256', 'uir_r1_a5_k5_s1_e4_c256', 'mqa_r1_k3_h4_s1_d64_c256',
     'uir_r1_a5_k0_s1_e4_c256', 'mqa_r1_k3_h4_s1_d64_c256', 'uir_r1_a5_k0_s1_e4_c256'),
    ('cn_r1_k1_s1_c960',),
)

BN_EPS = 1e-5          # BatchNorm2d eps, used when folding BN into the int8 convs


@dataclass(frozen=True)
class Config:
    name: str                                   # harness model name
    arch_def: tuple[tuple[str, ...], ...]
    timm_name: str | None = None                # timm model (None: custom reduced arch)
    stem_size: int = 32
    num_features: int = 1280                    # conv_head width
    num_classes: int = 1000
    layer_scale_init_value: float | None = None  # 1e-5 for hybrid, None for conv
    image: int = 256
    batch: int = 1


# --- timm helpers (timm.layers.helpers.make_divisible, timm.layers.padding.get_padding) ---
def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def decode_block_str(block_str: str) -> tuple[dict, int]:
    """Subset of timm `_decode_block_str` used by the MNv4 arch defs (er/cn/uir/mqa)."""
    ops = block_str.split('_')
    block_type, ops = ops[0], ops[1:]
    options = {}
    for op in ops:
        splits = re.split(r'(\d.*)', op)
        assert len(splits) >= 2, f"unsupported block option {op!r} in {block_str}"
        options[splits[0]] = splits[1]
    assert not set(options) - {'r', 'k', 's', 'e', 'c', 'a', 'p', 'h', 'v', 'd'}, block_str
    args = dict(block_type=block_type, out_chs=int(options['c']), stride=int(options['s']))
    if block_type == 'er':
        args.update(exp_kernel_size=int(options['k']), exp_ratio=float(options['e']))
    elif block_type == 'cn':
        args.update(kernel_size=int(options['k']))
    elif block_type == 'uir':
        args.update(dw_kernel_size_start=int(options.get('a', 0)),
                    dw_kernel_size_mid=int(options['k']),
                    dw_kernel_size_end=int(options.get('p', 0)),
                    exp_ratio=float(options['e']))
    else:
        assert block_type == 'mqa', block_str
        args.update(dw_kernel_size=int(options['k']), num_heads=int(options['h']),
                    key_dim=int(options['d']), value_dim=int(options['d']),
                    kv_stride=int(options.get('v', 1)))
    return args, int(options['r'])


def decode_arch_def(arch_def) -> list[list[dict]]:
    """timm decode_arch_def + EfficientNetBuilder stride rule (output_stride=32, no dilation)."""
    stages = []
    for stage_strs in arch_def:
        blocks = []
        for s in stage_strs:
            args, r = decode_block_str(s)
            blocks += [dict(args) for _ in range(r)]
        for bi, b in enumerate(blocks):
            if bi >= 1:          # only the first block in any stack can have a stride > 1
                b['stride'] = 1
        stages.append(blocks)
    return stages




# --- layers ----------------------------------------------------------------
# A quantized tensor travels as t = (x int8, scale, zero point).
def create_conv2d(features: int, k: int, stride: int = 1, groups: int = 1, *, name: str):
    """TFLite CONV_2D / DEPTHWISE_CONV_2D (BN folded): per-channel int8 weights, int32 bias."""
    p = get_padding(k, stride)
    return q.QConv(features, (k, k), q.INT8_TFLITE, strides=(stride, stride),
                   padding=[(p, p), (p, p)], feature_group_count=groups, use_bias=True, name=name)


class ConvNormAct(nn.Module):
    """timm.layers.ConvNormAct: conv -> bn (folded into the conv) -> optional ReLU (clamp)."""
    out_chs: int
    kernel_size: int
    stride: int = 1
    groups: int = 1
    apply_act: bool = True

    @nn.compact
    def __call__(self, t):
        conv = create_conv2d(self.out_chs, self.kernel_size, self.stride, self.groups, name='conv')
        return qops.conv_out(conv, *t, relu=self.apply_act)


def residual(t, shortcut, name: str):
    """Skip connection: int8 ADD with its own output scale / zero point."""
    return qops.qadd(t, shortcut, q.ActQuant(zero_point=True, name=name))


# --- blocks ----------------------------------------------------------------
class ConvBnAct(nn.Module):
    """'cn' block (no skip: MNv4 arch strings never set `skip`)."""
    out_chs: int
    kernel_size: int
    stride: int

    @nn.compact
    def __call__(self, t):
        conv = create_conv2d(self.out_chs, self.kernel_size, self.stride, name='conv')
        return qops.conv_out(conv, *t, relu=True)                  # bn1 folded


class EdgeResidual(nn.Module):
    """'er' block (FusedIB): conv_exp kxk -> bn1(relu) -> conv_pwl 1x1 -> bn2 (BNs folded)."""
    out_chs: int
    exp_kernel_size: int
    stride: int
    exp_ratio: float

    @nn.compact
    def __call__(self, t):
        in_chs = t[0].shape[-1]
        mid_chs = make_divisible(in_chs * self.exp_ratio)
        has_skip = in_chs == self.out_chs and self.stride == 1
        shortcut = t
        t = qops.conv_out(create_conv2d(mid_chs, self.exp_kernel_size, self.stride, name='conv_exp'),
                          *t, relu=True)
        t = qops.conv_out(create_conv2d(self.out_chs, 1, name='conv_pwl'), *t, relu=False)
        if has_skip:
            t = residual(t, shortcut, 'add_quant')  # er skip
        return t


class UniversalInvertedResidual(nn.Module):
    """'uir' block (UIB): [dw_start] -> pw_exp -> [dw_mid] -> pw_proj -> [dw_end]; layer scale
    folded into the last conv."""
    out_chs: int
    dw_kernel_size_start: int
    dw_kernel_size_mid: int
    dw_kernel_size_end: int
    stride: int
    exp_ratio: float
    layer_scale_init_value: float | None

    @nn.compact
    def __call__(self, t):
        in_chs = t[0].shape[-1]
        has_skip = in_chs == self.out_chs and self.stride == 1
        shortcut = t
        if self.dw_kernel_size_start:
            dw_start_stride = self.stride if not self.dw_kernel_size_mid else 1
            t = ConvNormAct(in_chs, self.dw_kernel_size_start, dw_start_stride, in_chs,
                            apply_act=False, name='dw_start')(t)
        mid_chs = make_divisible(in_chs * self.exp_ratio)
        t = ConvNormAct(mid_chs, 1, name='pw_exp')(t)
        if self.dw_kernel_size_mid:
            t = ConvNormAct(mid_chs, self.dw_kernel_size_mid, self.stride, mid_chs,
                            name='dw_mid')(t)
        t = ConvNormAct(self.out_chs, 1, apply_act=False, name='pw_proj')(t)
        if self.dw_kernel_size_end:
            dw_end_stride = (self.stride if not self.dw_kernel_size_start
                             and not self.dw_kernel_size_mid else 1)
            t = ConvNormAct(self.out_chs, self.dw_kernel_size_end, dw_end_stride, self.out_chs,
                            apply_act=False, name='dw_end')(t)
        if has_skip:
            t = residual(t, shortcut, 'add_quant')
        return t


class ProjSeq(nn.Module):
    """One of MultiQueryAttention2d's nn.Sequential branches: [down_conv dw -> norm] -> proj 1x1."""
    out_dim: int
    down_stride: int = 1
    dw_kernel_size: int = 3

    @nn.compact
    def __call__(self, t):
        if self.down_stride > 1:
            c = t[0].shape[-1]
            t = qops.conv_out(create_conv2d(c, self.dw_kernel_size, self.down_stride, c,
                                            name='down_conv'), *t, relu=False)   # norm folded
        return qops.conv_out(create_conv2d(self.out_dim, 1, name='proj'), *t, relu=False)


class MultiQueryAttention2d(nn.Module):
    """timm.layers.MultiQueryAttention2d (query_strides=1, as in all MNv4 arch strings)."""
    dim_out: int
    num_heads: int
    key_dim: int
    value_dim: int
    kv_stride: int
    dw_kernel_size: int

    @nn.compact
    def __call__(self, t):
        B, H, W, _ = t[0].shape
        nh, kd, vd = self.num_heads, self.key_dim, self.value_dim
        # query.proj carries the folded key_dim**-0.5 scale
        qx, qs, qz = ProjSeq(nh * kd, name='query')(t)                 # [B,H,W,nh*kd]
        qx = qx.reshape(B, H * W, nh, kd).transpose(0, 2, 1, 3)         # [B,nh,L,kd]
        kx, ks, kz = ProjSeq(kd, self.kv_stride, self.dw_kernel_size, name='key')(t)
        kx = kx.reshape(B, 1, -1, kd)                                   # [B,1,P,kd]
        vx, vs, vz = ProjSeq(vd, self.kv_stride, self.dw_kernel_size, name='value')(t)
        vx = vx.reshape(B, 1, -1, vd)                                   # [B,1,P,vd]
        # SDPA broadcasts the single K/V head over the query heads.
        kx = jnp.broadcast_to(kx, (B, nh) + kx.shape[2:])
        vx = jnp.broadcast_to(vx, (B, nh) + vx.shape[2:])
        attn = qops.qmatmul((qx, qs, qz), (kx, ks, kz),
                            q.ActQuant(zero_point=True, name='logits_quant'), transpose_b=True)
        attn = qops.softmax_int8(*attn, axis=-1)
        ox, os_, oz = qops.qmatmul(attn, (vx, vs, vz),
                                   q.ActQuant(zero_point=True, name='attn_v_quant'),
                                   transpose_b=False)                   # [B,nh,L,vd]
        ox = ox.transpose(0, 2, 1, 3).reshape(B, H, W, nh * vd)
        return ProjSeq(self.dim_out, name='output')((ox, os_, oz))     # layer scale folded


class MobileAttention(nn.Module):
    """'mqa' block: norm (BN, no act; int8 MUL + ADD) -> MQA -> [layer_scale folded] -> + shortcut."""
    out_chs: int
    stride: int
    dw_kernel_size: int
    num_heads: int
    key_dim: int
    value_dim: int
    kv_stride: int
    layer_scale_init_value: float | None

    @nn.compact
    def __call__(self, t):
        has_skip = self.stride == 1 and t[0].shape[-1] == self.out_chs
        shortcut = t
        t = qops.QAffine(name='norm')(*t)
        t = MultiQueryAttention2d(self.out_chs, self.num_heads, self.key_dim, self.value_dim,
                                  self.kv_stride, self.dw_kernel_size, name='attn')(t)
        if has_skip:
            t = residual(t, shortcut, 'add_quant')
        return t


def make_block(ba: dict, ls: float | None, name: str) -> nn.Module:
    bt = ba['block_type']
    out_chs = make_divisible(ba['out_chs'])          # round_chs_fn, multiplier 1.0
    if bt == 'er':
        return EdgeResidual(out_chs, ba['exp_kernel_size'], ba['stride'], ba['exp_ratio'],
                            name=name)
    if bt == 'cn':
        return ConvBnAct(out_chs, ba['kernel_size'], ba['stride'], name=name)
    if bt == 'uir':
        return UniversalInvertedResidual(out_chs, ba['dw_kernel_size_start'],
                                         ba['dw_kernel_size_mid'], ba['dw_kernel_size_end'],
                                         ba['stride'], ba['exp_ratio'], ls, name=name)
    return MobileAttention(out_chs, ba['stride'], ba['dw_kernel_size'], ba['num_heads'],
                           ba['key_dim'], ba['value_dim'], ba['kv_stride'], ls, name=name)


class MobileNetV4(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x):                                  # x: uint8 [N, H, W, 3]
        c = self.cfg
        # QUANTIZE uint8 -> int8: same scale, zero point shifted by 128
        s_in = self.param('input_scale', q.const(1.0 / 128), (), jnp.float32)
        z_in = self.param('input_zero_point', q.const(128), (), jnp.int32)
        t = (qops.requant(x, s_in, z_in, s_in, z_in - 128), s_in, z_in - 128)
        t = qops.conv_out(create_conv2d(make_divisible(c.stem_size), 3, 2, name='conv_stem'),
                          *t, relu=True)                    # bn1 folded
        for si, stage in enumerate(decode_arch_def(c.arch_def)):
            for bi, ba in enumerate(stage):
                t = make_block(ba, c.layer_scale_init_value, f'blocks_{si}_{bi}')(t)
        # MobileNetV3 'efficient head': global pool BEFORE conv_head (head_norm=True for MNv4)
        t = qops.qmean(*t, q.ActQuant(zero_point=True, name='global_pool_quant'))
        t = qops.conv_out(create_conv2d(c.num_features, 1, name='conv_head'), *t,
                          relu=True)                        # norm_head folded
        x, s, zp = t
        fc = q.QDense(c.num_classes, q.INT8_TFLITE_FC, name='classifier')
        x = fc(x.reshape(x.shape[0], -1), s, zp)
        p, sp, zpp = qops.softmax_int8(x, fc.get_variable('params', 'output_scale'),
                                       fc.get_variable('params', 'output_zero_point'))
        return qops.requant(p, sp, zpp, sp, zpp + 128, 'uint8')   # QUANTIZE int8 -> uint8
