"""Point Transformer V3 (PT-v3m1) + DefaultSegmentorV2 head, Flax linen port.

Ported from Pointcept @ 1342eda:
  pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py
  pointcept/models/default.py::DefaultSegmentorV2 (Linear seg head)
as built by configs/nuscenes/semseg-pt-v3m1-0-base.py with enable_flash=False.

Every point set (the reference's `Point` at one U-Net level) is a static array
of `capacity[s]` rows; the first n_s rows are the real points, in the
reference's row order. All data-dependent indexing is host-computed (host.py)
and fed as padded int32 inputs (model.py documents the padding conventions):

  SubMConv3d   spconv's gather -> GEMM -> scatter-add per kernel offset over a
               rulebook padded to `capacity` pairs per offset (padded pairs
               scatter out of range and are dropped); the center offset is one
               GEMM over all rows, as in spconv.
  attention    SerializedAttention: gather rows into serialized, padded patch
               order (serialized_order[slot][pad], host-composed) -> [P, H, K, K]
               attention per patch of K points -> gather back
               (unpad[serialized_inverse[slot]]); a key mask covers the n < K
               single-patch case (padded keys).
  pooling      SerializedPooling: proj -> gather by cluster-sorted rows ->
               segment max (torch_scatter.segment_csr; empty segments -> 0).
  unpooling    SerializedUnpooling: proj(coarse)[cluster] + proj_skip(skip); the
               next block's CPE sees proj_skip(skip) only (PT-v3m1 does not
               refresh sparse_conv_feat there, see SerializedUnpooling).

Module names follow torch's with `name.<int>` -> `name_<int>`.

Quantization: INT8 W8A8 platform fallback (automotive; no quantized PTv3 exists:
Pointcept ships none, Autoware's TensorRT PTv3 is fp16 only), in TensorRT INT8 engine
form (models/ptv3/recipe.py):
  * every Linear (qkv, proj, fc1, fc2, CPE Linear, pooling / unpooling proj, seg head)
    and every SubMConv3d (stem k=5, CPE k=3): int8 x int8 -> int32, per-output-channel
    symmetric weights, per-tensor static activation scales, fp32 epilogue (scale, bias);
  * BatchNorm folded into the Linear / conv right before it (stem conv, unpooling proj
    and proj_skip); the pooling BN follows a segment max, cannot be folded through it
    and runs as an fp16 elementwise op;
  * attention core in fp16 with fp32 accumulation and fp32 softmax (the released model's
    flash-attention numerics); LayerNorm, GELU, residual adds and the inter-layer tensors
    fp16;
  * an encoder level's output feeds two int8 layers (the next level's pooling proj and
    the decoder's proj_skip), so it is quantized once by its producer (`out_quant`).
The stem's input features are raw coordinates (m) + strength under one per-tensor scale,
coarse over the +-54 m range (the fallback has no carve-out evidence for it).
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

from models import quant as q


@dataclass(frozen=True)
class Config:
    in_channels: int = 4                         # coord (3) + strength (1)
    num_classes: int = 16
    orders: tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")
    stride: tuple[int, ...] = (2, 2, 2, 2)
    enc_depths: tuple[int, ...] = (2, 2, 2, 6, 2)
    enc_channels: tuple[int, ...] = (32, 64, 128, 256, 512)
    enc_num_head: tuple[int, ...] = (2, 4, 8, 16, 32)
    enc_patch_size: tuple[int, ...] = (1024,) * 5
    dec_depths: tuple[int, ...] = (2, 2, 2, 2)
    dec_channels: tuple[int, ...] = (64, 64, 128, 256)
    dec_num_head: tuple[int, ...] = (4, 4, 8, 16)
    dec_patch_size: tuple[int, ...] = (1024,) * 4
    mlp_ratio: int = 4
    stem_kernel: int = 5
    cpe_kernel: int = 3
    grid_size: float = 0.05
    # static point slots per U-Net level (worst-case bound on n_s)
    capacity: tuple[int, ...] = (34720,) * 5
    # Patch size K_s and patch count P_s per level, if not derived from the capacity:
    # P_s as its own symbol (min / ceil of a symbolic capacity is not a size).
    patch_sizes: tuple[int, ...] | None = None
    patch_counts: tuple | None = None

    @property
    def num_stages(self) -> int:
        return len(self.enc_depths)

    def patch(self, s: int) -> int:
        """Static patch size K_s = min(capacity, patch_size): the reference
        (enable_flash=False) uses min(n_s, patch_size)."""
        if self.patch_sizes is not None:
            return self.patch_sizes[s]
        return min(self.capacity[s], self.enc_patch_size[s])

    def n_patches(self, s: int) -> int:
        if self.patch_counts is not None:
            return self.patch_counts[s]
        return -(-self.capacity[s] // self.patch(s))


F16 = jnp.float16


def _bn(name):
    # nn.BatchNorm1d(eps=1e-3, momentum=0.01), inference mode; fp16 elementwise
    return nn.BatchNorm(use_running_average=True, epsilon=1e-3, momentum=0.99, dtype=F16,
                        name=name)


def _ln(name):
    # fp16 in / out (flax reduces the statistics in fp32)
    return nn.LayerNorm(epsilon=1e-5, use_fast_variance=False, dtype=F16, name=name)


def _dense(features, name):
    """INT8 nn.Linear (bias): per-channel weights, per-tensor input, fp16 output."""
    return q.QDense(features, q.INT8_STATIC, dtype=F16, name=name)


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)            # nn.GELU()


_GATHER = lax.GatherScatterMode.PROMISE_IN_BOUNDS    # host indices are in range
_DROP = lax.GatherScatterMode.FILL_OR_DROP           # padded pairs / rows: index = size


def _take(x, idx):
    """x[idx] along rows (one stablehlo.gather; indices are host-validated, so
    jnp's negative-index normalization is not emitted)."""
    dn = lax.GatherDimensionNumbers(offset_dims=tuple(range(1, x.ndim)),
                                    collapsed_slice_dims=(0,), start_index_map=(0,))
    return lax.gather(x, idx[:, None], dn, (1, *x.shape[1:]), mode=_GATHER)


def _scatter_dn(x):
    return lax.ScatterDimensionNumbers(update_window_dims=tuple(range(1, x.ndim)),
                                       inserted_window_dims=(0,),
                                       scatter_dims_to_operand_dims=(0,))


def _scatter_add(x, idx, y):
    """x.at[idx].add(y); out-of-range rows (padding) are dropped."""
    return lax.scatter_add(x, idx[:, None], y, _scatter_dn(x), mode=_DROP)


def _segment_max(x, segment, n):
    """Sorted segment max over rows (torch_scatter.segment_csr reduce="max");
    out-of-range segment ids (padding) are dropped, empty segments give 0."""
    init = jnp.full((n, *x.shape[1:]), -jnp.inf, x.dtype)
    out = lax.scatter_max(init, segment[:, None], x, _scatter_dn(init),
                          indices_are_sorted=True, mode=_DROP)
    return jnp.where(out == -jnp.inf, 0.0, out)


class SubMConv3d(q.QBase):
    """spconv.SubMConv3d(kernel_size=k) on rows with a precomputed rulebook, INT8.

    rule_in / rule_out: [k^3 - 1, capacity] per non-center offset (spconv's
    flat kernel order): out[rule_out] += x[rule_in] @ W[offset]. The input is
    quantized once (per-tensor `input_scale`), every offset GEMM is
    int8 x int8 -> int32 and the partial sums are accumulated / scatter-added in
    int32 (exact, the fused kernel's accumulator); one fp32 epilogue (scale
    s_x * s_w per output channel, bias) writes fp16."""
    features: int
    kernel: int
    use_bias: bool
    scheme: q.QScheme = q.INT8_STATIC
    dtype: str = F16

    @nn.compact
    def __call__(self, x, rule_in, rule_out, x_scale=None):
        k, cin, sc = self.kernel, x.shape[-1], self.scheme
        kv = k ** 3
        w = self.param("kernel", q.rand_q("int8"), (k, k, k, cin, self.features), q.INT8)
        ws = self._weight_scale(sc, kv * cin, self.features, kv * cin)
        xq, xs, _ = self._input_quant(sc, x, x_scale)
        w = w.reshape(kv, cin, self.features)
        dot = lambda a, b: lax.dot_general(a, b, (((1,), (0,)), ((), ())),
                                           preferred_element_type=jnp.int32)
        c = kv // 2
        acc = dot(xq, w[c])                             # center offset: every row
        for i in range(kv):
            if i == c:
                continue
            j = i if i < c else i - 1
            acc = _scatter_add(acc, rule_out[j], dot(_take(xq, rule_in[j]), w[i]))
        return self._epilogue(sc, acc, xs * ws, (self.features,), self.use_bias)


class Stem(nn.Module):
    """Embedding.stem: SubMConv3d(k=5, bias=False) -> BN -> GELU, the BN folded into
    the conv (its shift becomes the conv's bias)."""
    channels: int
    kernel: int

    @nn.compact
    def __call__(self, x, rule):
        x = SubMConv3d(self.channels, self.kernel, True, name="conv")(x, rule["in"], rule["out"])
        return _gelu(x)


class Embedding(nn.Module):
    channels: int
    kernel: int

    @nn.compact
    def __call__(self, x, rule):
        return Stem(self.channels, self.kernel, name="stem")(x, rule)


class SerializedAttention(nn.Module):
    channels: int
    num_heads: int

    @nn.compact
    def __call__(self, x, order, inverse, mask):
        """order: serialized_order[slot][pad] [P*K]; inverse:
        unpad[serialized_inverse[slot]] [capacity]; mask: valid keys [P, K]
        (P patches of K points: the sizes come from the mask's shape)."""
        H, C = self.num_heads, self.channels
        P, K = mask.shape
        scale = (C // H) ** -0.5
        qkv = _dense(3 * C, "qkv")(x)                   # fp16
        qkv = _take(qkv, order)                         # [P*K, 3C]
        qkv = qkv.reshape(P, K, 3, H, C // H).transpose(2, 0, 3, 1, 4)
        qh, kh, vh = qkv[0], qkv[1], qkv[2]             # [P, H, K, C/H]
        # fp16 operands, fp32 accumulation and softmax (flash-attention numerics)
        attn = jnp.einsum("phqd,phkd->phqk", qh * jnp.float16(scale), kh,
                          preferred_element_type=jnp.float32)
        attn = jnp.where(mask[:, None, None, :], attn, -jnp.inf)
        attn = jax.nn.softmax(attn, axis=-1).astype(F16)
        feat = jnp.einsum("phqk,phkd->phqd", attn, vh, preferred_element_type=jnp.float32)
        feat = feat.astype(F16).transpose(0, 2, 1, 3).reshape(P * K, C)
        feat = _take(feat, inverse)
        return _dense(C, "proj")(feat)


class MLP(nn.Module):
    hidden: int
    out: int

    @nn.compact
    def __call__(self, x):
        return _dense(self.out, "fc2")(_gelu(_dense(self.hidden, "fc1")(x)))


class Block(nn.Module):
    channels: int
    num_heads: int
    mlp_ratio: int
    order_index: int
    cpe_kernel: int

    @nn.compact
    def __call__(self, x, st, conv_feat=None):
        """conv_feat: the features the CPE's spconv sees if they differ from x
        (first decoder block, see SerializedUnpooling)."""
        C = self.channels
        shortcut = x
        h = SubMConv3d(C, self.cpe_kernel, True, name="cpe_0")(
            x if conv_feat is None else conv_feat, st["conv_in"], st["conv_out"])
        h = _ln("cpe_2")(_dense(C, "cpe_1")(h))
        x = shortcut + h
        shortcut = x
        h = _ln("norm1_0")(x)
        h = SerializedAttention(C, self.num_heads, name="attn")(
            h, st["attn_order"][self.order_index], st["attn_inverse"][self.order_index],
            st["mask"])
        x = shortcut + h
        shortcut = x
        h = MLP(C * self.mlp_ratio, C, name="mlp_0")(_ln("norm2_0")(x))
        return shortcut + h


class SerializedPooling(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, x_scale, pool, n_out):
        """x: int8 parent features with their producer's scale; n_out: capacity of the
        pooled level (from that level's input shapes). The BN follows the segment max, so
        it is not folded into proj (a negative BN scale does not commute with max)."""
        h = _dense(self.out_channels, "proj")(x, x_scale)
        h = _take(h, pool["indices"])                   # rows sorted by cluster
        h = _segment_max(h, pool["segment"], n_out)
        return _gelu(_bn("norm_0")(h))


class SerializedUnpooling(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, skip, skip_scale, cluster):
        """-> (feat, sparse_conv_feat features). PT-v3m1 sets parent.feat =
        proj_skip(parent).feat + point.feat[inverse] but leaves
        parent.sparse_conv_feat at proj_skip's output (v3m2 / v3m3 later add the
        replace_feature), so the next block's CPE convolves proj_skip(skip) only.
        proj / proj_skip: Linear + BN (folded into the Linear) + GELU; `skip` is the
        encoder level's int8 output with its producer's scale."""
        x = _gelu(_dense(self.out_channels, "proj_0")(x))
        skip = _gelu(_dense(self.out_channels, "proj_skip_0")(skip, skip_scale))
        return skip + _take(x, cluster), skip


def _block(cfg: Config, s: int, i: int, channels: int, heads: int) -> Block:
    return Block(channels, heads, cfg.mlp_ratio, i % len(cfg.orders), cfg.cpe_kernel,
                 name=f"block{i}")


class EncStage(nn.Module):
    """-> (x, x_q, x_scale): the level's output and, for every level but the deepest,
    its int8 copy (quantized once: read by the next pooling proj and by proj_skip)."""
    cfg: Config
    s: int

    @nn.compact
    def __call__(self, x, x_q, x_scale, index):
        cfg, s = self.cfg, self.s
        if s > 0:
            x = SerializedPooling(cfg.enc_channels[s], name="down")(
                x_q, x_scale, index[f"pool{s}"], index[f"stage{s}"]["attn_inverse"].shape[1])
        for i in range(cfg.enc_depths[s]):
            x = _block(cfg, s, i, cfg.enc_channels[s], cfg.enc_num_head[s])(
                x, index[f"stage{s}"])
        if s == cfg.num_stages - 1:
            return x, None, None
        x_q, x_scale, _ = q.ActQuant(name="out_quant")(x)
        return x, x_q, x_scale


class Encoder(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, index):
        skips, x_q, x_scale = [], None, None
        for s in range(self.cfg.num_stages):
            x, x_q, x_scale = EncStage(self.cfg, s, name=f"enc{s}")(x, x_q, x_scale, index)
            skips.append((x, x_q, x_scale))
        return skips


class DecStage(nn.Module):
    cfg: Config
    s: int

    @nn.compact
    def __call__(self, x, skip, index):
        cfg, s = self.cfg, self.s
        _, skip_q, skip_scale = skip
        x, conv_feat = SerializedUnpooling(cfg.dec_channels[s], name="up")(
            x, skip_q, skip_scale, index[f"pool{s + 1}"]["cluster"])
        for i in range(cfg.dec_depths[s]):
            x = _block(cfg, s, i, cfg.dec_channels[s], cfg.dec_num_head[s])(
                x, index[f"stage{s}"], conv_feat)
            conv_feat = None             # Block.forward re-syncs sparse_conv_feat at its end
        return x


class Decoder(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, skips, index):
        x = skips[-1][0]
        for s in reversed(range(self.cfg.num_stages - 1)):
            x = DecStage(self.cfg, s, name=f"dec{s}")(x, skips[s], index)
        return x


class PointTransformerV3(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, feat, index):
        cfg = self.cfg
        x = Embedding(cfg.enc_channels[0], cfg.stem_kernel, name="embedding")(feat, index["stem"])
        skips = Encoder(cfg, name="enc")(x, index)
        return Decoder(cfg, name="dec")(skips, index)


class DefaultSegmentorV2(nn.Module):
    """seg_head(backbone(point).feat); the pooling_parent loop of the reference
    does not run (the decoder returns the level-0 point)."""
    cfg: Config

    @nn.compact
    def __call__(self, feat, index):
        x = PointTransformerV3(self.cfg, name="backbone")(feat, index)
        return _dense(self.cfg.num_classes, "seg_head")(x)            # fp16 seg_logits
