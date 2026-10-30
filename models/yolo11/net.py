# SPDX-License-Identifier: AGPL-3.0
# Flax linen port of Ultralytics YOLO11 (https://github.com/ultralytics/ultralytics,
# AGPL-3.0). This file is a derivative work of `ultralytics/nn/modules/{conv,block,
# head}.py`, `ultralytics/nn/tasks.py`, `ultralytics/utils/tal.py`,
# `ultralytics/utils/export/engine.py` and `ultralytics/cfg/models/11/yolo11.yaml` at
# v8.4.183 and is distributed under AGPL-3.0.
"""YOLO11 object detector (MLPerf Inference edge detection: YOLO11-L), INT8 Flax linen.

Mirrors the Ultralytics module tree. `DetectionModel/model_<i>` is torch
`model.<i>`, and nested Sequential/ModuleList indices are joined with "_" (torch
`model.23.cv3.0.1.0` becomes `model_23/cv3_0_1_0`). Semantics follow Ultralytics:
- BN eps=1e-3 (`initialize_weights`); SiLU everywhere; autopad k//2.
- Nearest x2 upsampling; Concat in YAML `from` order.
- PSA attention scale key_dim**-0.5 and a depthwise 3x3 `pe` conv on V.
- Detect: the one-to-many head with DFL softmax-expectation decode, then the
  anchor/stride decode to xywh, and sigmoid class scores, giving (B, 4+nc, A) as
  in eval mode.
Layout is NHWC internally. The Detect head flattens to torch's (B, C, H*W) order,
so the head output equals Ultralytics' tensor exactly.

Quantization: the Ultralytics TensorRT INT8 export (`format=engine, quantize=8`;
ultralytics/utils/export/engine.py::onnx2engine, TensorRT 8-10 implicit INT8):
  * BN folded into every conv (`model.fuse()`); weights int8 per-output-channel
    symmetric, activations int8 per-tensor symmetric (MINMAX calibration). Each conv's
    epilogue (scale, folded-BN bias, SiLU, a following residual add) is fp32 and its
    output is requantized to int8 by the producer (`act_quant` / `add_quant`); concats
    requantize their inputs to one scale (`cat_quant`); max pool, upsample, split and
    chunk move int8 data. The fp32 network input is quantized by the first conv.
  * C2PSA attention: q^T k and v attn^T are i8 x i8 -> i32 GEMMs (the key_dim**-0.5
    scale only rescales q); the softmax runs in fp16 and its output is requantized.
  * Head carve-outs (OBEY_PRECISION_CONSTRAINTS in onnx2engine): the six final 1x1
    convs (`cv2.i.2`, `cv3.i.2`) and the DFL conv run in fp16 (fp16 weights), the class
    sigmoid in fp32; the DFL softmax and the anchor / stride decode run in fp16 (the
    FP16 builder flag); the output is fp32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

from models import quant as q

F16 = jnp.float16




@dataclass(frozen=True)
class Config:
    batch: int = 1
    image: int = 640
    nc: int = 80
    reg_max: int = 16
    scale: str = "l"            # C3k2 uses C3k blocks for scales m/l/x (parse_model)
    depth: float = 1.0          # yolo11.yaml scales["l"] = [1.00, 1.00, 512]
    width: float = 1.0
    max_channels: int = 512


# ultralytics/cfg/models/11/yolo11.yaml: backbone + head, [from, repeats, module, args]
LAYERS = (
    (-1, 1, "Conv", (64, 3, 2)),            # 0-P1/2
    (-1, 1, "Conv", (128, 3, 2)),           # 1-P2/4
    (-1, 2, "C3k2", (256, False, 0.25)),
    (-1, 1, "Conv", (256, 3, 2)),           # 3-P3/8
    (-1, 2, "C3k2", (512, False, 0.25)),
    (-1, 1, "Conv", (512, 3, 2)),           # 5-P4/16
    (-1, 2, "C3k2", (512, True)),
    (-1, 1, "Conv", (1024, 3, 2)),          # 7-P5/32
    (-1, 2, "C3k2", (1024, True)),
    (-1, 1, "SPPF", (1024, 5)),             # 9
    (-1, 2, "C2PSA", (1024,)),              # 10
    (-1, 1, "Upsample", (None, 2, "nearest")),
    ((-1, 6), 1, "Concat", (1,)),           # cat backbone P4
    (-1, 2, "C3k2", (512, False)),          # 13
    (-1, 1, "Upsample", (None, 2, "nearest")),
    ((-1, 4), 1, "Concat", (1,)),           # cat backbone P3
    (-1, 2, "C3k2", (256, False)),          # 16 (P3/8-small)
    (-1, 1, "Conv", (256, 3, 2)),
    ((-1, 13), 1, "Concat", (1,)),          # cat head P4
    (-1, 2, "C3k2", (512, False)),          # 19 (P4/16-medium)
    (-1, 1, "Conv", (512, 3, 2)),
    ((-1, 10), 1, "Concat", (1,)),          # cat head P5
    (-1, 2, "C3k2", (1024, True)),          # 22 (P5/32-large)
    ((16, 19, 22), 1, "Detect", ("nc",)),   # Detect(P3, P4, P5)
)


def autopad(k: int, p: int | None = None) -> int:
    """'same' padding for dilation 1 (all YOLO11 convs)."""
    return k // 2 if p is None else p


def make_divisible(x: float, divisor: int) -> int:
    return math.ceil(x / divisor) * divisor


def to_bcn(x):
    """NHWC -> torch's (B, C, H*W) `.view(b, c, -1)` order."""
    b, h, w, c = x.shape
    return x.reshape(b, h * w, c).transpose(0, 2, 1)


def to_bhwc(x, h: int, w: int):
    """(B, C, H*W) -> NHWC (inverse of to_bcn)."""
    b, c, _ = x.shape
    return x.transpose(0, 2, 1).reshape(b, h, w, c)




# --- INT8 tensors: t = (x int8, per-tensor scale) ------------------------------------
def qcat(ts, out: q.ActQuant):
    """Concat of int8 tensors with different scales: requantized to one output scale."""
    x, s, _ = out(jnp.concatenate([q.dequantize(x, s) for x, s in ts], axis=-1))
    return x, s


def qresidual(y, t, out: q.ActQuant):
    """Residual add fused into the producing conv's fp32 epilogue, then requantized."""
    x, s, _ = out(y + q.dequantize(*t))
    return x, s


class Conv(nn.Module):
    """nn/modules/conv.py::Conv with BN folded: INT8 conv -> fp32 epilogue (scale, bias,
    SiLU) -> requantized to int8 (`act_quant`). quant_out=False returns the fp32 epilogue
    for a following fused residual add. Input: (int8, scale), or the fp32 network input."""
    c2: int
    k: int = 1
    s: int = 1
    g: int | None = 1           # None: depthwise, g = gcd(c1, c2) (DWConv)
    act: bool = True
    quant_out: bool = True

    @nn.compact
    def __call__(self, t):
        x, s = t if isinstance(t, tuple) else (t, None)
        p = autopad(self.k)
        g = math.gcd(x.shape[-1], self.c2) if self.g is None else self.g
        y = q.QConv(self.c2, (self.k, self.k), q.INT8_STATIC, strides=(self.s, self.s),
                    padding=[(p, p), (p, p)], feature_group_count=g, use_bias=True,
                    name="conv")(x, s)                                  # bn folded
        if self.act:
            y = nn.silu(y)
        if not self.quant_out:
            return y
        x, s, _ = q.ActQuant(name="act_quant")(y)
        return x, s


class DWConv(Conv):
    """nn/modules/conv.py::DWConv: Conv with groups = gcd(c1, c2)."""
    g: int | None = None


class Bottleneck(nn.Module):
    """nn/modules/block.py::Bottleneck."""
    c2: int
    shortcut: bool = True
    g: int = 1
    k: tuple[int, int] = (3, 3)
    e: float = 0.5

    @nn.compact
    def __call__(self, t):
        c_ = int(self.c2 * self.e)
        add = self.shortcut and t[0].shape[-1] == self.c2
        y = Conv(self.c2, self.k[1], 1, g=self.g, quant_out=not add,
                 name="cv2")(Conv(c_, self.k[0], 1, name="cv1")(t))
        return qresidual(y, t, q.ActQuant(name="add_quant")) if add else y


class C3k(nn.Module):
    """nn/modules/block.py::C3k (a C3 whose Bottlenecks use k x k, e=1.0)."""
    c2: int
    n: int = 1
    shortcut: bool = True
    g: int = 1
    e: float = 0.5
    k: int = 3

    @nn.compact
    def __call__(self, t):
        c_ = int(self.c2 * self.e)
        a = Conv(c_, 1, 1, name="cv1")(t)
        for i in range(self.n):
            a = Bottleneck(c_, self.shortcut, self.g, k=(self.k, self.k), e=1.0, name=f"m_{i}")(a)
        b = Conv(c_, 1, 1, name="cv2")(t)
        return Conv(self.c2, 1, name="cv3")(qcat((a, b), q.ActQuant(name="cat_quant")))


class C3k2(nn.Module):
    """nn/modules/block.py::C3k2 (C2f forward; m[i] = C3k(c, c, 2) if c3k else Bottleneck)."""
    c2: int
    n: int = 1
    c3k: bool = False
    e: float = 0.5
    g: int = 1
    shortcut: bool = True

    @nn.compact
    def __call__(self, t):
        c = int(self.c2 * self.e)
        x, s = Conv(2 * c, 1, 1, name="cv1")(t)
        y = [(h, s) for h in jnp.split(x, 2, axis=-1)]                 # chunk(2): int8, one scale
        for i in range(self.n):
            m = (C3k(c, 2, self.shortcut, self.g, name=f"m_{i}") if self.c3k
                 else Bottleneck(c, self.shortcut, self.g, name=f"m_{i}"))
            y.append(m(y[-1]))
        return Conv(self.c2, 1, name="cv2")(qcat(y, q.ActQuant(name="cat_quant")))


def max_pool_int8(x, k: int):
    """MaxPool2d(k, 1, k//2) on int8 (pads with -128, the int8 -inf); the scale is unchanged."""
    p = k // 2
    return lax.reduce_window(x, jnp.int8(-128), lax.max, (1, k, k, 1), (1, 1, 1, 1),
                             ((0, 0), (p, p), (p, p), (0, 0)))


class SPPF(nn.Module):
    """nn/modules/block.py::SPPF. YOLO11 YAML rows are 'legacy' (args <= 3), for which
    parse_model sets cv1.act = SiLU (only YOLO26's SPPF leaves cv1 unactivated)."""
    c2: int
    k: int = 5
    n: int = 3

    @nn.compact
    def __call__(self, t):
        c_ = t[0].shape[-1] // 2
        x, s = Conv(c_, 1, 1, act=True, name="cv1")(t)
        y = [x]
        for _ in range(self.n):     # self.m = MaxPool2d(k, stride=1, padding=k//2), reused n times
            y.append(max_pool_int8(y[-1], self.k))
        # every concat input carries cv1's scale: an int8 concat, no requantization
        return Conv(self.c2, 1, 1, name="cv2")((jnp.concatenate(y, axis=-1), s))


class Attention(nn.Module):
    """nn/modules/block.py::Attention (non-coreml path). Returns proj's fp32 epilogue."""
    dim: int
    num_heads: int = 8
    attn_ratio: float = 0.5

    @nn.compact
    def __call__(self, t):
        B, H, W, C = t[0].shape
        N = H * W
        head_dim = self.dim // self.num_heads
        key_dim = int(head_dim * self.attn_ratio)
        scale = key_dim ** -0.5
        h = self.dim + key_dim * self.num_heads * 2
        x, s = Conv(h, 1, act=False, name="qkv")(t)
        qkv = to_bcn(x).reshape(B, self.num_heads, key_dim * 2 + head_dim, N)
        qh, kh, vh = jnp.split(qkv, [key_dim, 2 * key_dim], axis=2)
        # attn = ((q * scale).transpose(-2, -1) @ k).softmax(-1): int8 GEMM, fp16 softmax
        logits = q.qeinsum("bhkn,bhkm->bhnm", qh, kh, s * scale, s, "int8", F16)
        attn, sa, _ = q.ActQuant(name="attn_quant")(jax.nn.softmax(logits, axis=-1))
        # x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + pe(v.reshape(B, C, H, W))
        o = q.qeinsum("bhdm,bhnm->bhdn", vh, attn, s, sa, "int8", jnp.float32).reshape(B, C, N)
        pe = Conv(self.dim, 3, 1, g=self.dim, act=False, quant_out=False,
                  name="pe")((to_bhwc(vh.reshape(B, C, N), H, W), s))
        x, s, _ = q.ActQuant(name="pe_add_quant")(to_bhwc(o, H, W) + pe)
        return Conv(self.dim, 1, act=False, quant_out=False, name="proj")((x, s))


class PSABlock(nn.Module):
    """nn/modules/block.py::PSABlock."""
    c: int
    attn_ratio: float = 0.5
    num_heads: int = 4
    shortcut: bool = True

    @nn.compact
    def __call__(self, t):
        a = Attention(self.c, self.num_heads, self.attn_ratio, name="attn")(t)
        t = qresidual(a, t, q.ActQuant(name="attn_add_quant")) if self.shortcut else \
            q.ActQuant(name="attn_add_quant")(a)[:2]
        f = Conv(self.c, 1, act=False, quant_out=False,
                 name="ffn_1")(Conv(self.c * 2, 1, name="ffn_0")(t))
        return qresidual(f, t, q.ActQuant(name="ffn_add_quant")) if self.shortcut else \
            q.ActQuant(name="ffn_add_quant")(f)[:2]


class C2PSA(nn.Module):
    """nn/modules/block.py::C2PSA."""
    c2: int
    n: int = 1
    e: float = 0.5

    @nn.compact
    def __call__(self, t):
        c = int(t[0].shape[-1] * self.e)
        x, s = Conv(2 * c, 1, 1, name="cv1")(t)
        a, b = (x[..., :c], s), (x[..., c:], s)
        for i in range(self.n):
            b = PSABlock(c, attn_ratio=0.5, num_heads=max(c // 64, 1), name=f"m_{i}")(b)
        return Conv(self.c2, 1, name="cv2")(qcat((a, b), q.ActQuant(name="cat_quant")))


class Upsample(nn.Module):
    """torch.nn.Upsample(scale_factor=2, mode="nearest") == repeat each pixel (int8)."""
    scale_factor: int = 2

    @nn.compact
    def __call__(self, t):
        x, s = t
        b, h, w, c = x.shape
        f = self.scale_factor
        x = jnp.broadcast_to(x[:, :, None, :, None, :], (b, h, f, w, f, c))
        return x.reshape(b, h * f, w * f, c), s


class Concat(nn.Module):
    """nn/modules/conv.py::Concat along channels (NHWC axis -1 == NCHW dim 1), requantized
    to one scale."""

    @nn.compact
    def __call__(self, ts):
        return qcat(ts, q.ActQuant(name="cat_quant"))


class DFL(nn.Module):
    """nn/modules/block.py::DFL in fp16 (onnx2engine keeps '/dfl/' convs fp16): softmax over
    reg_max bins, then a fixed 1x1 conv with weight arange(c1) (an expectation). The weight
    is a (non-trainable) state_dict entry in Ultralytics, so it is a parameter here too."""
    c1: int = 16

    @nn.compact
    def __call__(self, x):          # (b, 4*c1, a) fp16
        b, _, a = x.shape
        # torch: conv(x.view(b, 4, c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        x = jax.nn.softmax(x.reshape(b, 4, self.c1, a).transpose(0, 1, 3, 2), axis=-1)
        conv = nn.Conv(1, (1, 1), use_bias=False, dtype=F16, param_dtype=F16, name="conv",
                       kernel_init=lambda _k, s, d=F16: jnp.arange(self.c1, dtype=d).reshape(s))
        return conv(x).reshape(b, 4, a)


def make_anchors(shapes, strides, dtype, grid_cell_offset: float = 0.5):
    """utils/tal.py::make_anchors (shape-only, so static); shapes are (h, w) per level."""
    anchor_points, stride_tensor = [], []
    for (h, w), stride in zip(shapes, strides):
        sx = jnp.arange(w, dtype=dtype) + grid_cell_offset
        sy = jnp.arange(h, dtype=dtype) + grid_cell_offset
        sy, sx = jnp.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(jnp.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(jnp.full((h * w, 1), stride, dtype=dtype))
    return jnp.concatenate(anchor_points), jnp.concatenate(stride_tensor)


def dist2bbox(distance, anchor_points, dim: int = 1):
    """utils/tal.py::dist2bbox with xywh=True (Detect, not end2end, xyxy=False)."""
    lt, rb = jnp.split(distance, 2, axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return jnp.concatenate(((x1y1 + x2y2) / 2, x2y2 - x1y1), axis=dim)


def head_conv(features: int, t, name: str):
    """A final head 1x1 nn.Conv2d (cv2.i.2 / cv3.i.2): kept out of INT8, fp16 weights and math."""
    return nn.Conv(features, (1, 1), dtype=F16, param_dtype=F16, name=name)(q.dequantize(*t, F16))


class Detect(nn.Module):
    """nn/modules/head.py::Detect (legacy=False, i.e. the YOLO11 DWConv cls head)."""
    nc: int = 80
    reg_max: int = 16
    stride: tuple[float, ...] = (8.0, 16.0, 32.0)

    @nn.compact
    def __call__(self, ts):
        c_in = ts[0][0].shape[-1]
        c2 = max(16, c_in // 4, self.reg_max * 4)
        c3 = max(c_in, min(self.nc, 100))
        boxes, scores = [], []
        for i, t in enumerate(ts):              # forward_head
            b = Conv(c2, 3, name=f"cv2_{i}_0")(t)
            b = Conv(c2, 3, name=f"cv2_{i}_1")(b)
            b = head_conv(4 * self.reg_max, b, f"cv2_{i}_2")
            s = DWConv(t[0].shape[-1], 3, name=f"cv3_{i}_0_0")(t)
            s = Conv(c3, 1, name=f"cv3_{i}_0_1")(s)
            s = DWConv(c3, 3, name=f"cv3_{i}_1_0")(s)
            s = Conv(c3, 1, name=f"cv3_{i}_1_1")(s)
            s = head_conv(self.nc, s, f"cv3_{i}_2")
            boxes.append(to_bcn(b))
            scores.append(to_bcn(s))
        boxes = jnp.concatenate(boxes, axis=-1)
        scores = jnp.concatenate(scores, axis=-1)
        # _inference / _get_decode_boxes: fp16 decode, fp32 sigmoid (onnx2engine constraint)
        shapes = [t[0].shape[1:3] for t in ts]
        anchors, strides = (a.T for a in make_anchors(shapes, self.stride, F16, 0.5))
        dbox = dist2bbox(DFL(self.reg_max, name="dfl")(boxes), anchors[None], dim=1) * strides
        return jnp.concatenate((dbox.astype(jnp.float32),
                                jax.nn.sigmoid(scores.astype(jnp.float32))), axis=1)


class DetectionModel(nn.Module):
    """nn/tasks.py::DetectionModel: parse_model(yolo11.yaml) + _predict_once."""
    cfg: Config

    @nn.compact
    def __call__(self, x):                      # x: fp32 [N, H, W, 3] in [0, 1]
        c = self.cfg
        h_in = x.shape[1]
        width = lambda c2: make_divisible(min(c2, c.max_channels) * c.width, 8)
        y = []
        for i, (f, n, m, args) in enumerate(LAYERS):
            n = max(round(n * c.depth), 1) if n > 1 else n          # depth gain
            name = f"model_{i}"
            xin = x if f == -1 else [x if j == -1 else y[j] for j in f]
            if m == "Conv":
                x = Conv(width(args[0]), args[1], args[2], name=name)(xin)
            elif m == "C3k2":
                c3k = c.scale in ("m", "l", "x") or args[1]          # parse_model: M/L/X force c3k
                e = args[2] if len(args) > 2 else 0.5
                x = C3k2(width(args[0]), n, c3k, e, name=name)(xin)
            elif m == "SPPF":
                x = SPPF(width(args[0]), args[1], name=name)(xin)
            elif m == "C2PSA":
                x = C2PSA(width(args[0]), n, name=name)(xin)
            elif m == "Upsample":
                x = Upsample(args[1], name=name)(xin)
            elif m == "Concat":
                x = Concat(name=name)(xin)
            elif m == "Detect":
                # DetectionModel.__init__: Detect.stride = input size / feature height
                stride = tuple(float(h_in // t[0].shape[1]) for t in xin)
                x = Detect(c.nc, c.reg_max, stride, name=name)(xin)
            y.append(x)
        return x
