"""Point Transformer V3 (base), nuScenes semantic segmentation, INT8 W8A8 (automotive
fallback, models/ptv3/recipe.py): capture spec.

Single phase `forward`: one DefaultSegmentorV2 call on one test fragment
(batch 1) of one nuScenes LIDAR_TOP sweep. Inputs (network definition in
net.py, host-side index generation in host.py; level s = 0..4 has
capacity C_s rows, n_s of them real, real rows first in the reference's order):

  feat                    f32[C_0, 4]   coord (x, y, z) + strength; padded rows 0
  index["stem"]           in/out i32[124, C_0]   SubMConv3d(k=5) rulebook
  index["stage{s}"]
    conv_in / conv_out    i32[26, C_s]  SubMConv3d(k=3) rulebook of indice_key stage{s}
    attn_order            i32[4, P_s, K_s]  serialized_order[slot][pad]
    attn_inverse          i32[4, C_s]   unpad[serialized_inverse[slot]]
    mask                  bool[P_s, K_s] valid attention keys
  index["pool{s}"] (s=1..4, level s-1 -> s)
    indices               i32[C_{s-1}]  parent rows sorted by cluster
    segment               i32[C_{s-1}]  cluster id of each sorted row (sorted)
    cluster               i32[C_{s-1}]  pooling_inverse (parent row -> cluster)

Padding: rulebook pairs (0, C_s) (scatter out of range, dropped); attention
positions past the reference's patches gather row min(p, C_s - 1) (dummy
patches, all keys valid; their outputs are never gathered back); padded rows'
attn_inverse / order are the identity; pool segment C_s (dropped) and cluster 0.
"""

from __future__ import annotations

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np

from models.spec import ModelSpec, Phase, Sym, as_spec
from models.ptv3 import host
from models.ptv3.net import Capacity, Config, DefaultSegmentorV2, PatchCount, PatchSize

# Worst-case bound: a nuScenes LIDAR_TOP sweep (Velodyne HDL-32E, 32 beams x 1085
# firings per 20 Hz revolution) has at most 34720 points; grid sampling only
# removes points, and pooling never increases the count, so 34720 bounds every
# level.
N_MAX = 34720



def pinned(cfg: Config) -> Config:
    """`cfg` with the patch sizes K_s and counts P_s fixed to their values derived from the
    capacities, so that they are independent sizes (symbols) of the export."""
    n = range(cfg.num_stages)
    return dataclasses.replace(cfg, patch_sizes=PatchSize(*(cfg.patch(s) for s in n)),
                               patch_counts=PatchCount(*(cfg.n_patches(s) for s in n)))


CONFIGS = {
    "full": pinned(Config(capacity=Capacity(*(N_MAX,) * 5))),
    # Same architecture and patch size (the unmodified config); only the point
    # bounds shrink. Example cloud: the 3500 voxels nearest the sensor of a
    # simulated sweep (levels hold ~3500/1500/600/180/60 points: multi-patch with tail
    # duplication at levels 0-1, single masked patches below, K_4 = 256 < 1024).
    "reduced": pinned(Config(capacity=Capacity(3500, 2048, 1024, 512, 256))),
}

KV_STEM = 5 ** 3 - 1
KV_CPE = 3 ** 3 - 1


# ----------------------------------------------------------------------------
# Padded inputs <-> host point cloud
# ----------------------------------------------------------------------------

def input_spec(cfg: Config):
    S = jax.ShapeDtypeStruct
    i32 = jnp.int32
    c = cfg.capacity
    index = {"stem": {"in": S((KV_STEM, c[0]), i32), "out": S((KV_STEM, c[0]), i32)}}
    for s in range(cfg.num_stages):
        P, K = cfg.n_patches(s), cfg.patch(s)
        index[f"stage{s}"] = {
            "conv_in": S((KV_CPE, c[s]), i32), "conv_out": S((KV_CPE, c[s]), i32),
            "attn_order": S((len(cfg.orders), P, K), i32),
            "attn_inverse": S((len(cfg.orders), c[s]), i32),
            "mask": S((P, K), jnp.bool_)}
        if s > 0:
            index[f"pool{s}"] = {k: S((c[s - 1],), i32) for k in ("indices", "segment", "cluster")}
    return S((c[0], cfg.in_channels), jnp.float32), index


def _rule(grid, k, cap):
    rb = host.rulebook(grid, k)
    rin = np.zeros((len(rb), cap), np.int32)
    rout = np.full((len(rb), cap), cap, np.int32)
    for j, (i, o) in enumerate(rb):
        rin[j, :len(i)], rout[j, :len(o)] = i, o
    return rin, rout


def _ident(n, cap):
    return np.arange(n, cap)


def build_inputs(cfg: Config, coord, strength, perms):
    """Host-side index generation for one fragment -> (feat, index) numpy arrays."""
    coord = np.asarray(coord, np.float32)
    n0 = coord.shape[0]
    c = cfg.capacity
    assert n0 <= c[0], (n0, c[0])
    grid = host.grid_coord_of(coord, cfg.grid_size)
    sts = host.stages(grid, perms, cfg.num_stages - 1, cfg.orders)
    feat = np.zeros((c[0], cfg.in_channels), np.float32)
    feat[:n0] = np.concatenate([coord, np.asarray(strength, np.float32)], 1)
    rin, rout = _rule(grid, cfg.stem_kernel, c[0])
    index = {"stem": {"in": rin, "out": rout}}
    for s, st in enumerate(sts):
        n, cap = st.n, c[s]
        assert n <= cap, f"level {s}: {n} points > capacity {cap}"
        P, K = cfg.n_patches(s), cfg.patch(s)
        pad, unpad, Kr = host.patches(n, cfg.enc_patch_size[s])
        assert Kr <= K
        L = P * K
        order = np.empty((len(cfg.orders), L), np.int32)
        inverse = np.empty((len(cfg.orders), cap), np.int32)
        for k in range(len(cfg.orders)):
            order[k, :len(pad)] = st.order[k][pad]
            order[k, len(pad):] = np.minimum(np.arange(len(pad), L), cap - 1)
            inverse[k, :n] = unpad[st.inverse[k]]
            inverse[k, n:] = _ident(n, cap)
        mask = np.ones((P, K), bool)
        if n <= K:
            mask[0, n:] = False
        rin, rout = _rule(st.grid, cfg.cpe_kernel, cap)
        index[f"stage{s}"] = {"conv_in": rin, "conv_out": rout,
                              "attn_order": order.reshape(len(cfg.orders), P, K),
                              "attn_inverse": inverse, "mask": mask}
        if s > 0:
            prev = c[s - 1]
            npv = sts[s - 1].n
            ind = np.concatenate([st.indices, _ident(npv, prev)]).astype(np.int32)
            seg = np.concatenate([st.cluster[st.indices], np.full(prev - npv, cap)]).astype(np.int32)
            clu = np.concatenate([st.cluster, np.zeros(prev - npv)]).astype(np.int32)
            index[f"pool{s}"] = {"indices": ind, "segment": seg, "cluster": clu}
    return feat, index


def level_counts(cfg: Config, index) -> list[int]:
    """Real point counts n_s, recovered from the padded pooling inputs."""
    counts = []
    for s in range(1, cfg.num_stages):
        seg = np.asarray(index[f"pool{s}"]["segment"])
        valid = seg < cfg.capacity[s]
        if s == 1:
            counts.append(int(valid.sum()))
        else:
            assert counts[-1] == int(valid.sum())
        counts.append(int(seg[valid].max()) + 1)
    return counts


# ----------------------------------------------------------------------------
# Example clouds
# ----------------------------------------------------------------------------

def nearest_voxels(n: int, seed: int):
    """The n voxels nearest the sensor of a simulated HDL-32E sweep after the
    test pipeline's GridSample(0.025) + GridSample(0.05) (fragment 0)."""
    rng = np.random.default_rng(seed)
    coord, strength = host.simulate_sweep(rng)
    coord, strength, _ = host.grid_sample(coord, strength, 0.025, rng)
    coord, strength, _ = host.grid_sample(coord, strength, 0.05)
    keep = np.argsort(np.linalg.norm(coord[:, :2], axis=1), kind="stable")[:n]
    keep = keep[rng.permutation(n)]
    return coord[keep], strength[keep]


def example_cloud(cfg: Config, seed: int):
    rng = np.random.default_rng(seed + 7)
    if cfg.capacity == CONFIGS["full"].capacity:
        coord, strength = host.saturated_cloud(cfg.capacity[0], rng, cfg.grid_size)
    else:
        coord, strength = nearest_voxels(cfg.capacity[0], seed)
    perms = [rng.permutation(len(cfg.orders)) for _ in range(cfg.num_stages)]
    return coord, strength, perms


def to_jax(feat, index):
    return jnp.asarray(feat), jax.tree.map(jnp.asarray, index)


def example_inputs(cfg: Config, rng):
    """full: the saturated worst case (every level holds exactly 34720 points,
    host.saturated_cloud). reduced: n_0 = capacity[0] voxels of a simulated sweep
    (deeper levels padded)."""
    seed = int(jax.random.randint(rng, (), 0, 2 ** 31 - 1)) % 1000
    coord, strength, perms = example_cloud(cfg, seed)
    return {"forward": to_jax(*build_inputs(cfg, coord, strength, perms))}


def init_variables(cfg: Config, rng):
    feat, index = input_spec(cfg)
    z = lambda tree: jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), tree)
    return DefaultSegmentorV2(cfg).init(rng, z(feat), z(index))


# ----------------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------------

def level_of(path: str) -> int:
    """U-Net level whose rows a module's output has."""
    m = re.search(r"/(enc|dec)(\d)(/|$)", path)
    if not m:
        return 0                     # embedding, backbone, dec, seg_head
    s = int(m.group(2))
    if m.group(1) == "enc" and "/down/proj" in path:
        return s - 1                 # pooling proj runs on the parent rows
    if m.group(1) == "dec" and re.search(r"/up/proj_\d", path):
        return s + 1                 # unpooling proj runs on the coarse rows
    return s


def intermediates(cfg: Config, variables, feat, index) -> dict:
    """Every module's output (but the root and the encoder's skip list), cut to
    the real rows n_s of its level."""
    _, state = DefaultSegmentorV2(cfg).apply(variables, feat, index, capture_intermediates=True,
                                             mutable=["intermediates"])
    counts = level_counts(cfg, index)
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        if not keys or keys[-1] != "__call__" or len(keys) == 1:
            continue
        out = v[0]
        if isinstance(out, tuple):   # SerializedUnpooling: (feat, sparse_conv_feat features)
            out = out[0]
        if not hasattr(out, "shape"):
            continue                 # Encoder: list of skips
        name = "/".join(keys[:-1])
        flat[name] = np.asarray(out)[:counts[level_of(name)]]
    return flat


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------

def input_shapes(cfg: Config):
    """Symbolic shapes of input_spec: level s has capacity C<s>, P<s> patches of K<s> points."""
    index = {"stem": "_, C0"}
    for s in range(cfg.num_stages):
        index[f"stage{s}"] = {"conv_in": f"_, C{s}", "conv_out": f"_, C{s}",
                              "attn_order": f"_, P{s}, K{s}", "attn_inverse": f"_, C{s}",
                              "mask": f"P{s}, K{s}"}
        if s > 0:
            index[f"pool{s}"] = f"C{s - 1}"
    return ("C0, _", index)


def build(cfg: Config) -> ModelSpec:
    module = DefaultSegmentorV2(cfg)

    def forward(variables, feat, index):
        return module.apply(variables, feat, index)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    c = cfg.capacity
    return ModelSpec(
        name="ptv3",
        variant=(f"Pointcept PT-v3m1 base + DefaultSegmentorV2 (nuScenes semseg, 16 classes), "
                 f"INT8 W8A8 fallback (TensorRT INT8 form: int8 Linears and SubMConv3d, "
                 f"per-channel weights, per-tensor static activations, BN folded; fp16 "
                 f"attention core with fp32 softmax, fp16 LayerNorm / GELU), batch 1, point "
                 f"capacity per level {list(c)}, patch {cfg.enc_patch_size[0]}"),
        phases=[Phase("forward", forward, input_spec(cfg), trip_count=40, shapes=input_shapes(cfg),
                      note=("one DefaultSegmentorV2.forward on one test fragment (fragment batch "
                            "size 1); SemSegTester runs it once per fragment: 10 test-time "
                            "augmentations x F GridSample(0.05, test) fragments, F data-dependent "
                            "(4 on a simulated sweep) -> trip_count 40. Output: "
                            f"seg_logits [{c[0]}, {cfg.num_classes}], rows >= n_0 are padding."))],
        param_specs=as_spec(variables),
        symbols={
            **{f"C{s}": Sym(f"capacity.l{s}", alt, f"level-{s} point capacity (padded rows)")
               for s, alt in enumerate((3000, 1800, 900, 400, 200))},
            **{f"P{s}": Sym(f"patch_counts.l{s}", alt, f"level-{s} attention patch count")
               for s, alt in enumerate((3, 3, 2, 2, 2))},
            **{f"K{s}": Sym(f"patch_sizes.l{s}", alt, f"level-{s} attention patch size "
                            f"(min(points, {cfg.enc_patch_size[s]}))")
               for s, alt in enumerate((1000, 900, 800, 400, 200))},
        },
        # the reference's patch size is min(n_s, patch_size)
        constraints=tuple(f"K{s} <= {k}" for s, k in enumerate(cfg.enc_patch_size)),
        excluded=[
            "nuScenes sweep loading (coord, strength / 255) and the test transform "
            "GridSample(0.025, train mode, fnv hash)",
            "test-time augmentation (10 x RandomScale in {0.9..1.1} x RandomFlip) and "
            "GridSample(0.05, test mode) fragmenting + grid_coord (host)",
            "Point.serialization: z / z-trans / hilbert / hilbert-trans codes, argsort orders "
            "and inverses, shuffle_orders randperm (host, host.py)",
            "SerializedPooling index part: code >> 3, torch.unique clusters, sort, head "
            "indices, re-serialization and randperm (host); the pooled `coord` "
            "(segment_csr mean) is never read by the network and is not computed",
            "SerializedAttention.get_padding_and_inverse (patch padding / unpadding indices) "
            "composed with the serialized orders (host)",
            "spconv SubMConv3d indice-pair (rulebook) generation for indice keys stem, "
            "stage0..stage4 (host)",
            "padding of every point set to its static capacity (max 34720 points)",
            "softmax, accumulation of fragment scores pred[index] += prob, argmax and the "
            "inverse map back to the raw points (SemSegTester post-processing; the graph "
            "returns fp16 seg_logits)",
        ],
    )

