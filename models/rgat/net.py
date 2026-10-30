"""R-GAT (MLPerf Inference graph/R-GAT on IGBH), FP8 Flax linen port.

Port of mlcommons/inference graph/R-GAT `dgl_utilities/components.py::RGAT_DGL`:
3 x `dgl.nn.HeteroGraphConv({etype: GATPatched(in, h // heads, heads)}, aggregate="sum")`,
i.e. one DGL `GATConv` per relation type, summed over the relations that end
in each destination node type; `view(N, heads*D)`; `leaky_relu` (slope 0.01)
and dropout (identity at eval) between layers; `Linear(h, classes)` on the
'paper' nodes of the last block. Module and parameter names follow torch's
(`layers.0.mods.cites.fc.weight` -> `layers_0/mods_cites/fc/kernel`).

Static-shape message-flow graphs (MFGs). Neighbour sampling (PyGSampler, fanouts
[5, 10, 15] reversed: the seeds get 15) and node dedup run on the host and are
not part of the graph. Each DGL block is given as padded per-relation COO edge
lists (`src`, `dst`, `mask`) whose sizes are the worst-case bounds derived from
the seed batch and fanouts (`mfg_bounds`); node arrays are padded at the end, so
block l's dst nodes stay the prefix of its src nodes, exactly as in a DGL block.
GAT attention uses the same edge-space formulation as DGL (u_add_v SDDMM, edge
softmax = segment max / exp / segment sum / div, u_mul_e + sum SpMM) with
static-`num_segments` segment ops; padded edges are masked out, so valid rows
are exact.

Quantization (see models/rgat/recipe.py):
  * Node-feature storage: FP8 E4M3 with one fixed scale 1.0 for every node type -- MLPerf
    NVIDIA (inference_results v5.0-v6.0 closed/NVIDIA/code/rgat/pytorch:
    preprocess_data.py fp8_conversion, --fp8_format e4m3 --scale 1.0; config.py
    fp8_embedding=True). The host gathers FP8 rows, so `feat` arrives as float8_e4m3fn.
  * GEMMs: FP8 E4M3 W8A8, per-tensor weight scale, per-tensor static activation scale, fp32
    accumulation, on every GATConv `fc` (applied to the src and to the dst rows, as DGL
    does) and on the classifier. This is the GPU-datacenter fallback: NVIDIA's deployed
    GEMMs are fp16 (rgat.py model.half()). The layer-0 `fc` consume the stored FP8
    features directly with their storage scale (no second rounding).
  * Everything else fp16 (NVIDIA model.half()): parameters attn_l / attn_r / bias, the
    GEMM outputs, relation sum, leaky_relu between layers, classifier logits. The
    attention logits, edge softmax and the aggregation accumulate in fp32 (cugraph-ops
    fused GAT; fp16 in / out).
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from models import quant as q

NTYPES = ("author", "conference", "fos", "institute", "journal", "paper")
# dgl.heterograph(IGBHeteroGraphStructure.edge_dict).canonical_etypes (sorted):
# 6 IGBH relations + rev_* reverses (paper-cites-paper is symmetrized instead).
CANONICAL_ETYPES = (
    ("author", "affiliated_to", "institute"),
    ("author", "rev_written_by", "paper"),
    ("conference", "rev_venue", "paper"),
    ("fos", "rev_topic", "paper"),
    ("institute", "rev_affiliated_to", "author"),
    ("journal", "rev_published", "paper"),
    ("paper", "cites", "paper"),
    ("paper", "published", "journal"),
    ("paper", "topic", "fos"),
    ("paper", "venue", "conference"),
    ("paper", "written_by", "author"),
)
PREDICT = "paper"
# FP8 node-feature storage scale: NVIDIA casts every node table with one fixed scale 1.0
# (preprocess_data.py --scale default 1.0, not calibrated).
FEAT_SCALE = 1.0
DTYPE = jnp.float16                     # NVIDIA model.half()


@dataclass(frozen=True)
class Config:
    batch: int = 1                      # seed 'paper' nodes (MLPerf --max-batchsize default 1)
    in_feats: int = 1024
    h_feats: int = 512
    n_heads: int = 4
    num_classes: int = 2983
    fanouts: tuple[int, ...] = (5, 10, 15)   # backend_dgl.fan_out; sampled reversed
    gat_negative_slope: float = 0.2     # GATConv default
    act_negative_slope: float = 0.01    # F.leaky_relu default


def mfg_bounds(cfg: Config):
    """Static worst-case sizes of the PyGSampler blocks, input layer first.

    Hop h (h = 1..L) samples `fanout_h = reversed(fanouts)[h-1]` in-edges per
    relation for every node of frontier F^{h-1} (F^0 = the seeds); its new
    sources form F^h. The block of layer l = L - i (i = 1..L, sampling order) has
    dst = F^0..F^{i-1}, src = F^0..F^i and edges E^1..E^i (PyGSampler re-adds all
    previously sampled edges). Dedup only shrinks these, so they are upper bounds.
    Returns [{"n_src": {t}, "n_dst": {t}, "n_edges": {etype}}] per layer.
    """
    fan = list(reversed(cfg.fanouts))
    L = len(fan)
    front = [{t: (cfg.batch if t == PREDICT else 0) for t in NTYPES}]
    hop_edges = []
    for f in fan:
        prev = front[-1]
        e = {r: f * prev[d] for s, r, d in CANONICAL_ETYPES}
        new = {t: 0 for t in NTYPES}
        for s, r, d in CANONICAL_ETYPES:
            new[s] += e[r]
        hop_edges.append(e)
        front.append(new)
    cum = lambda i: {t: sum(front[k][t] for k in range(i + 1)) for t in NTYPES}
    out = []
    for l in range(L):
        i = L - l
        out.append({"n_src": cum(i), "n_dst": cum(i - 1),
                    "n_edges": {r: sum(hop_edges[k][r] for k in range(i))
                                for _, r, _ in CANONICAL_ETYPES}})
    return out


def live_etypes(cfg: Config, layer: int) -> list[tuple[str, str, str]]:
    """Relations whose dst node type can be non-empty in block `layer`.

    With seeds of type 'paper' only, the last block's dst set is the seeds and the
    second-to-last block's dst set has no 'institute' (institutes are two hops
    from a paper). DGL still instantiates (and runs, on 0 dst nodes) a GATConv for
    every relation in every layer; those 7 of 33 never influence any output."""
    n_dst = mfg_bounds(cfg)[layer]["n_dst"]
    return [e for e in CANONICAL_ETYPES if n_dst[e[2]] > 0]


def _glorot(key, shape, dtype=DTYPE):
    return nn.initializers.xavier_uniform(in_axis=-2, out_axis=-1)(key, shape, dtype)


class GATConv(nn.Module):
    """dgl.nn.pytorch.GATConv(in, out, heads) on a block, feat=(src, dst) tuple:
    shared fc (no bias), feat_drop=attn_drop=0 (eval), negative_slope=0.2,
    residual=False, explicit bias, activation=None; zero in-degree allowed
    (HeteroGraphConv sets allow_zero_in_degree=True on every module)."""
    out_feats: int
    num_heads: int
    negative_slope: float

    @nn.compact
    def __call__(self, h_src, h_dst, src, dst, mask, x_scale=None):
        H, D = self.num_heads, self.out_feats
        n_dst = h_dst.shape[0]
        # quantized: FP8 W8A8 per-tensor (fallback); x_scale = FP8 feature storage scale
        fc = q.QDense(H * D, q.FP8_STATIC, use_bias=False, dtype=DTYPE, name="fc")
        feat_src = fc(h_src, x_scale).reshape(h_src.shape[0], H, D)
        feat_dst = fc(h_dst, x_scale).reshape(n_dst, H, D)
        attn_l = self.param("attn_l", _glorot, (1, H, D), DTYPE)
        attn_r = self.param("attn_r", _glorot, (1, H, D), DTYPE)
        bias = self.param("bias", nn.initializers.zeros, (H * D,), DTYPE)
        f32 = jnp.float32
        el = (feat_src.astype(f32) * attn_l.astype(f32)).sum(-1)[..., None]   # [Ns, H, 1]
        er = (feat_dst.astype(f32) * attn_r.astype(f32)).sum(-1)[..., None]   # [Nd, H, 1]
        # apply_edges(u_add_v("el", "er", "e")); leaky_relu
        e = nn.leaky_relu(el[src] + er[dst], self.negative_slope)    # [E, H, 1]
        # edge_softmax(graph, e) over the in-edges of every dst node
        valid = mask[:, None, None]
        e = jnp.where(valid, e, -jnp.inf)
        e_max = jax.ops.segment_max(e, dst, num_segments=n_dst)
        e_max = jnp.where(jnp.isfinite(e_max), e_max, 0.0)          # dst without in-edges
        p = jnp.where(valid, jnp.exp(e - e_max[dst]), 0.0)
        p_sum = jax.ops.segment_sum(p, dst, num_segments=n_dst)
        a = jnp.where(valid, p / jnp.where(valid, p_sum[dst], 1.0), 0.0)
        # update_all(u_mul_e("ft", "a", "m"), sum("m", "ft")): fp32 accumulation, fp16 out
        rst = jax.ops.segment_sum(feat_src[src].astype(f32) * a, dst, num_segments=n_dst)
        return rst.astype(DTYPE) + bias.reshape(1, H, D)


class HeteroGraphConv(nn.Module):
    """dgl.nn.pytorch.HeteroGraphConv(mods, aggregate="sum") on a block."""
    cfg: Config
    layer: int

    @nn.compact
    def __call__(self, h, block, x_scale=None):
        c = self.cfg
        n_dst = mfg_bounds(c)[self.layer]["n_dst"]
        dst_inputs = {t: v[: n_dst[t]] for t, v in h.items() if n_dst[t] > 0}  # src prefix
        outputs = {t: [] for t in NTYPES}
        # DGL also runs a GATConv for the relations into dst types that cannot occur
        # in this block (on 0 dst nodes, result discarded); see live_etypes.
        for s, r, d in live_etypes(c, self.layer):
            mod = GATConv(c.h_feats // c.n_heads, c.n_heads, c.gat_negative_slope,
                          name=f"mods_{r}")
            outputs[d].append(mod(h[s], dst_inputs[d], block[r]["src"], block[r]["dst"],
                                  block[r]["mask"], x_scale))
        return {t: jnp.stack(a, 0).sum(0) for t, a in outputs.items() if a}


class RGAT_DGL(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, feat, blocks):          # feat: {ntype: float8_e4m3fn rows}, scale FEAT_SCALE
        c = self.cfg
        L = len(c.fanouts)
        h = feat
        for l in range(L):
            x_scale = jnp.float32(FEAT_SCALE) if l == 0 else None
            h = HeteroGraphConv(c, l, name=f"layers_{l}")(h, blocks[l], x_scale)
            h = {t: x.reshape(x.shape[0], x.shape[1] * x.shape[2]) for t, x in h.items()}
            if l != L - 1:
                h = {t: nn.leaky_relu(x, c.act_negative_slope) for t, x in h.items()}
                # nn.Dropout(0.2): identity in eval mode
        # quantized: FP8 W8A8 per-tensor (fallback)
        return q.QDense(c.num_classes, q.FP8_STATIC, dtype=DTYPE, name="linear")(h[PREDICT])
