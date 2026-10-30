"""R-GAT capture spec (network definition lives in net.py).

Inputs of the single `forward` phase are the host-sampled, padded MFG:
  feat   : {ntype: f8E4M3FN[n_src(layer 0, ntype), in_feats]}   gathered node features,
           stored FP8 E4M3 with the fixed scale net.FEAT_SCALE = 1.0 (MLPerf NVIDIA)
  blocks : per layer (input first) {etype: {"src","dst": i32[E], "mask": bool[E]}}
with every size the static bound from `net.mfg_bounds`. Padded edges have
mask=False (src=dst=0); padded node rows are zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from models import quant as q
from models.spec import ModelSpec, Phase, as_spec
from models.rgat.net import (CANONICAL_ETYPES, FEAT_SCALE, NTYPES, PREDICT, Config, RGAT_DGL,
                             live_etypes, mfg_bounds)

CONFIGS = {
    "full": Config(),
    # every block / relation / node type kept; narrow features, small fanouts, batch 2
    "reduced": Config(batch=2, in_feats=32, h_feats=32, n_heads=4, num_classes=10,
                      fanouts=(2, 3, 4)),
}
ETYPES = tuple(r for _, r, _ in CANONICAL_ETYPES)


def live(cfg: Config, layer: int) -> list[str]:
    return [r for _, r, _ in live_etypes(cfg, layer)]


def input_spec(cfg: Config):
    b = mfg_bounds(cfg)
    feat = {t: jax.ShapeDtypeStruct((b[0]["n_src"][t], cfg.in_feats), q.FP8)
            for t in NTYPES}
    blocks = tuple({r: {"src": jax.ShapeDtypeStruct((bl["n_edges"][r],), jnp.int32),
                        "dst": jax.ShapeDtypeStruct((bl["n_edges"][r],), jnp.int32),
                        "mask": jax.ShapeDtypeStruct((bl["n_edges"][r],), jnp.bool_)}
                    for r in live(cfg, l)} for l, bl in enumerate(b))
    return feat, blocks


# ----------------------------------------------------------------------------
# Host-side MFGs (sampling is excluded from the graph)
# ----------------------------------------------------------------------------

def saturated_mfg(cfg: Config, seed: int):
    """The worst-case MFG: every sampled node has `fanout` distinct, previously
    unseen in-neighbours per relation, so every bound of `mfg_bounds` is met
    exactly (all masks True). It is a valid PyGSampler output (a sampled forest);
    node ids follow DGL's block layout (dst nodes = src prefix, cohorts in order)."""
    rng = np.random.default_rng(seed)
    fan = list(reversed(cfg.fanouts))
    L = len(fan)
    cohort = [{t: (cfg.batch if t == PREDICT else 0) for t in NTYPES}]
    start = [{t: 0 for t in NTYPES}]
    hop_edges = []
    nxt = dict(cohort[0])                     # next free id per type
    for f in fan:
        prev, prev_start = cohort[-1], start[-1]
        new_start = dict(nxt)
        edges = {}
        for s, r, d in CANONICAL_ETYPES:
            dst = np.repeat(np.arange(prev_start[d], prev_start[d] + prev[d]), f)
            src = np.arange(nxt[s], nxt[s] + dst.size)
            nxt[s] += dst.size
            edges[r] = (src, dst)
        cohort.append({t: nxt[t] - new_start[t] for t in NTYPES})
        start.append(new_start)
        hop_edges.append(edges)
    blocks = []
    for l in range(L):
        i = L - l
        bl = {}
        for r in ETYPES:
            src = np.concatenate([hop_edges[k][r][0] for k in range(i)])
            dst = np.concatenate([hop_edges[k][r][1] for k in range(i)])
            p = rng.permutation(src.size)
            bl[r] = (src[p], dst[p])
        blocks.append(bl)
    n_src0 = {t: nxt[t] for t in NTYPES}
    feat = {t: rng.standard_normal((n_src0[t], cfg.in_feats)).astype(np.float32)
            for t in NTYPES}
    return feat, blocks


def pad_mfg(cfg: Config, feat, blocks):
    """(features per type, [{etype: (src, dst)}] per layer) -> padded network inputs."""
    b = mfg_bounds(cfg)
    pf = {}
    for t in NTYPES:
        x = np.zeros((b[0]["n_src"][t], cfg.in_feats), np.float32)
        n = feat[t].shape[0]
        assert n <= x.shape[0], (t, n, x.shape)
        x[:n] = feat[t]
        pf[t] = x
    pb = []
    for l, bl in enumerate(blocks):
        out = {}
        for r in ETYPES:
            src, dst = (np.asarray(a, np.int64) for a in bl[r])
            if r not in live(cfg, l):
                assert src.size == 0, (l, r)
                continue
            E = b[l]["n_edges"][r]
            assert src.size <= E, (l, r, src.size, E)
            ps, pd, pm = (np.zeros(E, np.int32), np.zeros(E, np.int32), np.zeros(E, bool))
            ps[:src.size], pd[:src.size], pm[:src.size] = src, dst, True
            out[r] = {"src": ps, "dst": pd, "mask": pm}
        pb.append(out)
    return pf, tuple(pb)


def mfg_counts(cfg: Config, blocks):
    """Actual (unpadded) node counts per layer, recovered from the padded edges.

    In a DGL block src = dst nodes (prefix) + the remaining edge sources, and the
    dst nodes of block l are the src nodes of block l+1; the last block's dst
    nodes are the seeds. So n_src(l, t) = max(n_dst(l, t), 1 + max valid src id).
    """
    L = len(blocks)
    n_dst = {t: (cfg.batch if t == PREDICT else 0) for t in NTYPES}
    out = [None] * L
    for l in reversed(range(L)):
        n_src = dict(n_dst)
        for s, r, d in CANONICAL_ETYPES:
            if r not in blocks[l]:
                continue
            e = blocks[l][r]
            m = np.asarray(e["mask"])
            if m.any():
                n_src[s] = max(n_src[s], int(np.asarray(e["src"])[m].max()) + 1)
                assert int(np.asarray(e["dst"])[m].max()) < n_dst[d], (l, r)
        out[l] = {"n_src": n_src, "n_dst": n_dst}
        n_dst = n_src
    return out


def example_inputs(cfg: Config, rng):
    seed = int(jax.random.randint(rng, (), 0, 2 ** 31 - 1))
    feat, blocks = saturated_mfg(cfg, seed)
    pf, pb = pad_mfg(cfg, feat, blocks)
    to_j = lambda tree: jax.tree.map(jnp.asarray, tree)
    # FP8 feature storage (preprocess_data.py fp8_conversion: cast with scale 1.0)
    pf = {t: q.quantize(jnp.asarray(x), FEAT_SCALE, "fp8") for t, x in pf.items()}
    return {"forward": (pf, to_j(pb))}


def init_variables(cfg: Config, rng):
    feat, blocks = input_spec(cfg)
    z = lambda tree: jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), tree)
    return RGAT_DGL(cfg).init(rng, z(feat), z(blocks))


def intermediates(cfg: Config, variables, feat, blocks) -> dict:
    """GATConv outputs, per-type HeteroGraphConv outputs and the classifier,
    cut to the valid (unpadded) dst rows."""
    _, state = RGAT_DGL(cfg).apply(variables, feat, blocks, capture_intermediates=True,
                                   mutable=["intermediates"])
    it = state["intermediates"]
    counts = mfg_counts(cfg, jax.tree.map(np.asarray, blocks))
    out = {}
    for l in range(len(cfg.fanouts)):
        nd = counts[l]["n_dst"]
        lay = it[f"layers_{l}"]
        for s, r, d in live_etypes(cfg, l):
            out[f"layers_{l}/mods_{r}"] = np.asarray(lay[f"mods_{r}"]["__call__"][0])[:nd[d]]
        for t, v in lay["__call__"][0].items():
            out[f"layers_{l}/{t}"] = np.asarray(v)[:nd[t]]
    out["linear"] = np.asarray(it["linear"]["__call__"][0])
    return out


def build(cfg: Config) -> ModelSpec:
    module = RGAT_DGL(cfg)

    def forward(variables, feat, blocks):
        return module.apply(variables, feat, blocks)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    return ModelSpec(
        name="rgat",
        variant=(f"MLPerf R-GAT (IGBH), 3x HeteroGraphConv/GATConv {cfg.in_feats}->"
                 f"{cfg.h_feats} ({cfg.n_heads} heads) -> {cfg.num_classes}, "
                 f"fanouts {list(cfg.fanouts)}, {cfg.batch} seed paper node(s), padded worst-case "
                 "MFG; FP8 E4M3 node features (MLPerf NVIDIA, scale 1.0), FP8 W8A8 fc / "
                 "classifier (GPU fallback), fp16 elsewhere"),
        phases=[Phase("forward", forward, input_spec(cfg),
                      note="one RGAT_DGL.forward on the sampled blocks")],
        param_specs=as_spec(variables),
        excluded=["IGBH neighbour sampling (PyGSampler, fanouts [5,10,15]) and MFG/block "
                  "construction with node dedup (host side)",
                  "feature fetch: features[ntype][block[0].srcdata[NID]] (host gather of the "
                  "FP8-stored rows; the FP8 cast itself is offline preprocessing)",
                  "argmax over the 2983 classes (postprocessing)"],
    )
