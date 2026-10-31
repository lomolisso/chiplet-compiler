"""CenterPoint (OpenPCDet cbgs_voxel0075_res3d_centerpoint, nuScenes), Flax linen port.

Ported from open-mmlab/OpenPCDet @ 233f849:

* `MeanVFE`             models/backbones_3d/vfe/mean_vfe.py: mean of the points in each voxel.
* `VoxelResBackBone8x`  models/backbones_3d/spconv_backbone.py: spconv SubMConv3d /
  SparseConv3d (+ BatchNorm1d eps 1e-3, ReLU), SparseBasicBlocks, 8x downsampling,
  final SparseConv3d (3,1,1) stride (2,1,1).
* `HeightCompression`   models/backbones_2d/map_to_bev/height_compression.py:
  SparseConvTensor.dense() -> view(N, C * D, H, W).
* `BaseBEVBackbone`     models/backbones_2d/base_bev_backbone.py (ZeroPad2d + conv blocks,
  ConvTranspose2d deblocks, concat).
* `CenterHead`          models/dense_heads/center_head.py up to the per-head prediction
  dicts (shared conv + one SeparateHead per class group).

Sparse convolution is spconv's native gather-GEMM-scatter algorithm on a static,
padded rulebook (models/centerpoint/rulebook.py computes it on the host): one GEMM
per kernel offset per layer, [pairs, Cin] x [Cin, Cout]. Sparse feature tensors are
[N_l, C] with the valid voxels first; padding rows are never read by a valid pair.
Dense tensors are NHWC (torch is NCHW). Module names follow torch's state-dict
names with `name.<int>` -> `name_<int>`.

Quantization: NVIDIA CUDA-CenterPoint INT8 (NVIDIA-AI-IOT/Lidar_AI_Solution,
CUDA-CenterPoint/qat: tools/sparseconv_quantization.py, onnx_export/export-scn.py; README
timing table "RPN + Head FP16/INT8"), applied by layer position to this OpenPCDet model
(det3d SpMiddleResNetFHD there: conv_input, conv1..4, extra_conv = conv_out):
  * BN folded into every conv (funcs.layer_fusion_bn), ReLU in the epilogue.
  * Sparse convs INT8: weights per-output-channel symmetric (QUANT_DESC_8BIT_CONV2D_WEIGHT_
    PER_CHANNEL), activations per-tensor static (histogram calibration, MSE amax), each offset
    GEMM i8 x i8 -> i32 with the offsets accumulated in int32 (libspconv implicit GEMM), fp32
    epilogue, output requantized to INT8 (output_precision int8).
  * QuantAdd in every SparseBasicBlock: one per-tensor quantizer shared by both add inputs.
  * fp16 carve-outs (export-scn.py): conv_input and the whole first SparseBasicBlock
    (conv1[0].conv1 / conv2 / add, f16 x f16 -> f16), whose output is written INT8;
    extra_conv (conv_out) computes INT8 and writes fp16 into the dense BEV map.
  * Voxelization / MeanVFE fp32. RPN (BaseBEVBackbone) and CenterHead: TensorRT INT8, same
    scheme; the non-overlapping deblocks are 1x1 convs + depth-to-space; head outputs fp16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jax import lax

from models import quant as q

HEAD_ORDER = ("center", "center_z", "dim", "rot", "vel")
HEAD_OUT = {"center": 2, "center_z": 1, "dim": 3, "rot": 2, "vel": 2}


@dataclass(frozen=True)
class LevelCapacities:
    """Padded voxel capacities of sparse levels 2..5 (level 1 = Config.max_voxels)."""
    n2: int
    n3: int
    n4: int
    n5: int


@dataclass(frozen=True)
class Config:
    # cbgs_voxel0075_res3d_centerpoint.yaml DATA_CONFIG (test mode)
    point_cloud_range: tuple[float, ...] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
    voxel_size: tuple[float, float, float] = (0.075, 0.075, 0.2)
    max_points_per_voxel: int = 10
    max_voxels: int = 160000                    # MAX_NUMBER_OF_VOXELS['test']
    num_point_features: int = 5                 # x, y, z, intensity, timestamp (10 sweeps)
    # VoxelResBackBone8x (hard-coded in spconv_backbone.py)
    channels: tuple[int, ...] = (16, 32, 64, 128)
    out_channels: int = 128
    # strided SparseConv3d layers: (indice_key, kernel, stride, padding)
    down: tuple = (("spconv2", (3, 3, 3), (2, 2, 2), (1, 1, 1)),
                   ("spconv3", (3, 3, 3), (2, 2, 2), (1, 1, 1)),
                   ("spconv4", (3, 3, 3), (2, 2, 2), (0, 1, 1)),
                   ("spconv_down2", (3, 1, 1), (2, 1, 1), (0, 0, 0)))
    # SubM rulebook per level 1..4 (level 1: keys 'subm1' and 'res1' share it)
    subm_keys: tuple[str, ...] = ("subm1", "res2", "res3", "res4")
    # BACKBONE_2D
    layer_nums: tuple[int, ...] = (5, 5)
    layer_strides: tuple[int, ...] = (1, 2)
    num_filters: tuple[int, ...] = (128, 256)
    upsample_strides: tuple[int, ...] = (1, 2)
    num_upsample_filters: tuple[int, ...] = (256, 256)
    # DENSE_HEAD
    class_names_each_head: tuple = (("car",), ("truck", "construction_vehicle"),
                                    ("bus", "trailer"), ("barrier",),
                                    ("motorcycle", "bicycle"),
                                    ("pedestrian", "traffic_cone"))
    shared_conv_channel: int = 64
    num_hm_conv: int = 2
    head_num_conv: int = 2
    n_valid: int | None = None                  # valid voxels in example inputs (None: all)
    # Padded voxel capacity of levels 2..5 (level 1 = max_voxels) and the pair-buffer layout
    # of each strided layer, if not derived from max_voxels (level_bounds / down_layouts).
    # Pinned for symbolic export (model.pinned): the capacities are independent input sizes
    # (symbols) and the layouts must not follow them (the derivation branches on the bounds).
    level_capacities: LevelCapacities | None = None
    layouts: tuple[str, ...] | None = None
    sample: str = "saturated"                   # example-input generator (model.py)

    @property
    def grid_size(self) -> tuple[int, int, int]:          # (x, y, z), DataProcessor
        r, v = np.asarray(self.point_cloud_range), np.asarray(self.voxel_size)
        return tuple(int(g) for g in np.round((r[3:6] - r[0:3]) / v).astype(np.int64))

    @property
    def sparse_shape(self) -> tuple[int, int, int]:       # (z, y, x) = grid[::-1] + [1, 0, 0]
        g = self.grid_size
        return (g[2] + 1, g[1], g[0])


def _out_size(n, k, s, p):
    return (n + 2 * p - k) // s + 1


def level_shapes(cfg: Config) -> list[tuple[int, int, int]]:
    shapes = [cfg.sparse_shape]
    for _, k, s, p in cfg.down:
        shapes.append(tuple(_out_size(n, *a) for n, a in zip(shapes[-1], zip(k, s, p))))
    return shapes


def _reach_1d(positions: set[int], n_out: int, k: int, s: int, p: int) -> set[int]:
    out = set()
    for i in positions:
        for kk in range(k):
            num = i + p - kk
            if num >= 0 and num % s == 0 and num // s < n_out:
                out.add(num // s)
    return out


def level_bounds(cfg: Config) -> list[int]:
    """Worst-case active voxels per level for <= max_voxels input voxels:
    N_{l+1} <= min(|grid_{l+1}|, r_total * N_1, r_layer * N_l), where r_total
    (r_layer) is the largest number of level-(l+1) cells one level-1 (level-l)
    voxel can reach, the product of exact per-axis maxima."""
    shapes = level_shapes(cfg)
    bounds = [cfg.max_voxels]
    for l, (_, k, s, p) in enumerate(cfg.down):
        r_total = r_layer = 1
        for d in range(3):
            best_t = best_l = 0
            for i in range(shapes[0][d]):
                pos = {i}
                for m in range(l + 1):
                    _, km, sm, pm = cfg.down[m]
                    pos = _reach_1d(pos, shapes[m + 1][d], km[d], sm[d], pm[d])
                best_t = max(best_t, len(pos))
            for i in range(shapes[l][d]):
                best_l = max(best_l, len(_reach_1d({i}, shapes[l + 1][d], k[d], s[d], p[d])))
            r_total *= best_t
            r_layer *= best_l
        bounds.append(min(math.prod(shapes[l + 1]), r_total * bounds[0], r_layer * bounds[-1]))
    return bounds


def down_layouts(cfg: Config) -> list[str]:
    """Per strided layer: lay the pair buffer along the smaller side (see rulebook.py)."""
    b = level_bounds(cfg)
    return ["in" if b[l] <= b[l + 1] else "out" for l in range(len(cfg.down))]


def capacities(cfg: Config) -> list:
    """Padded voxel capacity per level: `Config.level_capacities`, else `level_bounds`."""
    lc = cfg.level_capacities
    if lc is None:
        return level_bounds(cfg)
    return [cfg.max_voxels, lc.n2, lc.n3, lc.n4, lc.n5]


def layouts(cfg: Config) -> list[str]:
    """Pair-buffer layout per strided layer: `Config.layouts`, else `down_layouts`."""
    return list(cfg.layouts) if cfg.layouts is not None else down_layouts(cfg)


# ----------------------------------------------------------------------------
# quantized sparse convolution (spconv native gather-GEMM-scatter, per offset)
# ----------------------------------------------------------------------------

F16 = jnp.float16
INT8 = q.INT8_STATIC          # per-output-channel symmetric W, per-tensor static A, i8 x i8 -> i32


def _with_dump_row(x):
    """Append the zero row that padded / absent pairs read."""
    return jnp.concatenate([x, jnp.zeros((1, x.shape[1]), x.dtype)], axis=0)


def _dot(a, w, acc):
    return lax.dot_general(a, w, (((1,), (0,)), ((), ())), preferred_element_type=acc)


class _SparseConv(q.QBase):
    """Parameters of a sparse conv: INT8 (`scheme`) or the fp16 carve-out (`scheme=None`:
    fp16 kernel, bias and operands, f16 x f16 -> f16). The kernel keeps the
    [kz, ky, kx, Cin, Cout] layout; BN is folded into kernel and bias (layer_fusion_bn)."""
    features: int
    kernel_size: tuple[int, int, int] = (3, 3, 3)
    scheme: q.QScheme | None = INT8
    dtype: str = "float32"            # INT8 epilogue output dtype

    def _setup(self, x, x_scale):
        """-> (operand, kernel [kv, Cin, Cout], accumulator dtype, epilogue)."""
        kv, cin, F = math.prod(self.kernel_size), x.shape[1], self.features
        kshape = (*self.kernel_size, cin, F)
        if self.scheme is None:
            w = self.param("kernel", nn.initializers.lecun_normal(), kshape, F16)
            b = self.param("bias", nn.initializers.zeros, (F,), F16)
            return x.astype(F16), w.reshape(kv, cin, F), F16, lambda acc: acc + b
        w = self.param("kernel", q.rand_q("int8"), kshape, jnp.int8)
        ws = self._weight_scale(self.scheme, kv * cin, F, kv * cin)
        xq, xs, _ = self._input_quant(self.scheme, x, x_scale)
        return (xq, w.reshape(kv, cin, F), jnp.int32,
                lambda acc: self._epilogue(self.scheme, acc, xs * ws, (F,), True))


class SubMConv3d(_SparseConv):
    """spconv.SubMConv3d(k=3, stride 1, padding 1): centre offset as one dense GEMM
    (spconv lists no pairs for it), then per offset gather -> GEMM -> accumulate (int32 for
    INT8). `nbr` [26, N]: input row per output row and non-centre offset (N = dump)."""

    @nn.compact
    def __call__(self, x, nbr, x_scale=None):
        x, w, acc_t, epilogue = self._setup(x, x_scale)
        acc = _dot(x, w[13], acc_t)
        xe = _with_dump_row(x)
        for r, k in enumerate([k for k in range(27) if k != 13]):
            acc = acc + _dot(jnp.take(xe, nbr[r], axis=0), w[k], acc_t)
        return epilogue(acc)


class SparseConv3d(_SparseConv):
    """spconv.SparseConv3d (strided): per kernel offset one GEMM over the pair buffer.
    layout "in":  idx [kv, N_in]  = output row per input row; scatter-add the GEMM.
    layout "out": idx [kv, N_out] = input row per output row; gather, GEMM, accumulate.
    The partial sums of all offsets accumulate in int32 before the epilogue."""
    layout: str = "in"
    n_out: int = 0                                # output level's capacity (may be symbolic)

    @nn.compact
    def __call__(self, x, idx, x_scale=None):
        x, w, acc_t, epilogue = self._setup(x, x_scale)
        kv = w.shape[0]
        if self.layout == "in":
            acc = jnp.zeros((self.n_out + 1, self.features), acc_t)
            for k in range(kv):
                acc = acc.at[idx[k]].add(_dot(x, w[k], acc_t))
            return epilogue(acc[:self.n_out])
        xe = _with_dump_row(x)
        acc = _dot(jnp.take(xe, idx[0], axis=0), w[0], acc_t)
        for k in range(1, kv):
            acc = acc + _dot(jnp.take(xe, idx[k], axis=0), w[k], acc_t)
        return epilogue(acc)


# ----------------------------------------------------------------------------
# VFE + sparse 3D backbone
# ----------------------------------------------------------------------------

class MeanVFE(nn.Module):
    """Voxelization output stays fp32 (CUDA-CenterPoint: Voxelization FP32)."""

    @nn.compact
    def __call__(self, voxels, voxel_num_points):
        points_mean = jnp.sum(voxels, axis=1)
        normalizer = jnp.maximum(voxel_num_points.reshape(-1, 1), 1).astype(voxels.dtype)
        return points_mean / normalizer


class SparseBasicBlockFP16(nn.Module):
    """conv1[0]: CUDA-CenterPoint keeps both SubM convs and the residual add in fp16
    (quant_add precision fp16) but writes the block output INT8 (output_precision int8),
    with the scale of the next layer's input quantizer (`relu_quant`)."""
    planes: int

    @nn.compact
    def __call__(self, x, nbr):                      # x: fp16 (conv_input output)
        out = nn.relu(SubMConv3d(self.planes, scheme=None, name="conv1")(x, nbr))
        out = SubMConv3d(self.planes, scheme=None, name="conv2")(out, nbr)
        y, s, _ = q.ActQuant(name="relu_quant")(nn.relu(out + x))
        return y, s


class SparseBasicBlock(nn.Module):
    """INT8 block: conv1 (+ folded bn1), ReLU, requantized; conv2 (+ folded bn2); QuantAdd:
    ONE per-tensor quantizer (`quant_add`) shared by both add inputs, so conv2's output and
    the identity are both requantized to its scale; add + ReLU, requantized."""
    planes: int

    @nn.compact
    def __call__(self, x, s, nbr):                   # x: int8 block input, s: its scale
        out = nn.relu(SubMConv3d(self.planes, name="conv1")(x, nbr, s))
        out, so, _ = q.ActQuant(name="relu1_quant")(out)
        out = SubMConv3d(self.planes, name="conv2")(out, nbr, so)
        a, s_add, _ = q.ActQuant(name="quant_add")(out)
        b = q.quantize(q.dequantize(x, s), s_add, "int8")
        y = nn.relu(q.dequantize(a, s_add) + q.dequantize(b, s_add))
        y, sy, _ = q.ActQuant(name="relu_quant")(y)
        return y, sy


class VoxelResBackBone8x(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, rb):
        c = self.cfg
        lay = layouts(c)
        # Level capacities from the rulebook shapes (subm rulebooks [26, N_l], level-5
        # coordinates [N_5, 3]), so that they can be symbolic sizes.
        bounds = [rb[k].shape[1] for k in c.subm_keys] + [rb["bev_coords"].shape[0]]
        # conv_input: fp16 compute and output (MeanVFE's fp32 features cast to fp16)
        x = nn.relu(SubMConv3d(16, scheme=None, name="conv_input_0")(x, rb["subm1"]))
        x, s = SparseBasicBlockFP16(16, name="conv1_0")(x, rb["subm1"])
        x, s = SparseBasicBlock(16, name="conv1_1")(x, s, rb["subm1"])
        for l, planes in enumerate(c.channels[1:], start=2):
            key, ks, _, _ = c.down[l - 2]
            x = nn.relu(SparseConv3d(planes, ks, layout=lay[l - 2], n_out=bounds[l - 1],
                                     name=f"conv{l}_0_0")(x, rb[key], s))
            x, s, _ = q.ActQuant(name=f"conv{l}_0_2_quant")(x)
            for b in (1, 2):
                x, s = SparseBasicBlock(planes, name=f"conv{l}_{b}")(x, s, rb[c.subm_keys[l - 1]])
        # extra_conv: INT8 compute, fp16 output into the dense BEV map
        key, ks, _, _ = c.down[-1]
        return nn.relu(SparseConv3d(c.out_channels, ks, layout=lay[-1], n_out=bounds[-1],
                                    dtype="float16", name="conv_out_0")(x, rb[key], s))


class HeightCompression(nn.Module):
    """dense() of the level-5 sparse tensor, then view(N, C * D, H, W): channel
    c * D + d. Padding rows carry coordinate (D, 0, 0) = the dump row. fp16 canvas."""
    cfg: Config

    @nn.compact
    def __call__(self, x, coords):
        D, H, W = level_shapes(self.cfg)[-1]
        C = x.shape[1]
        flat = (coords[:, 0] * H + coords[:, 1]) * W + coords[:, 2]
        canvas = jnp.zeros((D * H * W + 1, C), x.dtype).at[flat].set(x)
        dense = canvas[:D * H * W].reshape(D, H, W, C)
        return jnp.transpose(dense, (1, 2, 3, 0)).reshape(1, H, W, C * D)


# ----------------------------------------------------------------------------
# dense BEV part (TensorRT INT8)
# ----------------------------------------------------------------------------

def qconv(features, k, stride, name, dtype="float32"):
    """INT8 conv (padding k // 2) with the BN folded in (bias)."""
    return q.QConv(features, (k, k), INT8, strides=(stride, stride),
                   padding=[(k // 2, k // 2)] * 2, use_bias=True, dtype=dtype, name=name)


def depth_to_space(x, s: int):
    """[N, H, W, (dy, dx, C)] -> [N, H*s, W*s, C]."""
    n, h, w, c = x.shape
    x = x.reshape(n, h, w, s, s, c // (s * s)).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(n, h * s, w * s, c // (s * s))


class BaseBEVBackbone(nn.Module):
    """Every conv INT8 (BN folded, ReLU in the epilogue, requantized). The deblocks
    ConvTranspose2d(k = u, stride = u) do not overlap, so each is a 1x1 conv to u^2 * Cout
    channels + depth-to-space (kernel (1, 1, Cin, (dy, dx, Cout)), the folded-BN bias
    repeated per sub-pixel): the same arithmetic, the rewrite NVIDIA's TensorRT code uses for
    MLPerf 3D U-Net. The concat is requantized with one scale (TensorRT concat).
    Returns (x int8, scale, None)."""
    cfg: Config

    @nn.compact
    def __call__(self, x):                           # x: fp16 BEV canvas
        c, ups, s = self.cfg, [], None
        for i, (n, st, f) in enumerate(zip(c.layer_nums, c.layer_strides, c.num_filters)):
            # ZeroPad2d(1) + Conv2d(k=3, stride st, padding=0) == conv with padding 1
            x = nn.relu(qconv(f, 3, st, f"blocks_{i}_1")(x, s))
            x, s, _ = q.ActQuant(name=f"blocks_{i}_3_quant")(x)
            for k in range(n):
                x = nn.relu(qconv(f, 3, 1, f"blocks_{i}_{3 * k + 4}")(x, s))
                x, s, _ = q.ActQuant(name=f"blocks_{i}_{3 * k + 6}_quant")(x)
            u, uf = c.upsample_strides[i], c.num_upsample_filters[i]
            y = nn.relu(qconv(u * u * uf, 1, 1, f"deblocks_{i}_0")(x, s))
            ups.append(depth_to_space(y, u) if u > 1 else y)
        return q.ActQuant(name="concat_quant")(jnp.concatenate(ups, axis=-1))


class SeparateHead(nn.Module):
    """sep_head_dict order: HEAD_DICT (center, center_z, dim, rot, vel) then 'hm'.
    INT8 convs; the output convs write fp16."""
    num_classes: int
    num_conv: int
    num_hm_conv: int

    @nn.compact
    def __call__(self, x, s):
        ret = {}
        heads = [(n, HEAD_OUT[n], self.num_conv) for n in HEAD_ORDER]
        heads.append(("hm", self.num_classes, self.num_hm_conv))
        for name, out_ch, num_conv in heads:
            y, sy = x, s
            for k in range(num_conv - 1):
                y = nn.relu(qconv(x.shape[-1], 3, 1, f"{name}_{k}_0")(y, sy))
                y, sy, _ = q.ActQuant(name=f"{name}_{k}_2_quant")(y)
            ret[name] = qconv(out_ch, 3, 1, f"{name}_{num_conv - 1}", dtype="float16")(y, sy)
        return ret


class CenterHead(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, s):
        c = self.cfg
        x = nn.relu(qconv(c.shared_conv_channel, 3, 1, "shared_conv_0")(x, s))
        x, s, _ = q.ActQuant(name="shared_conv_2_quant")(x)
        return tuple(SeparateHead(len(names), c.head_num_conv, c.num_hm_conv,
                                  name=f"heads_list_{i}")(x, s)
                     for i, names in enumerate(c.class_names_each_head))


class CenterPoint(nn.Module):
    """Detector3DTemplate module_list [vfe, backbone_3d, map_to_bev_module,
    backbone_2d, dense_head] up to the head predictions."""
    cfg: Config

    @nn.compact
    def __call__(self, voxels, voxel_num_points, rb):
        c = self.cfg
        x = MeanVFE(name="vfe")(voxels, voxel_num_points)
        x = VoxelResBackBone8x(c, name="backbone_3d")(x, rb)
        x = HeightCompression(c, name="map_to_bev_module")(x, rb["bev_coords"])
        x, s, _ = BaseBEVBackbone(c, name="backbone_2d")(x)
        return CenterHead(c, name="dense_head")(x, s)
