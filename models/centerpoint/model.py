"""CenterPoint (OpenPCDet cbgs_voxel0075_res3d_centerpoint, nuScenes) capture spec, INT8
(NVIDIA CUDA-CenterPoint recipe, see net.py and recipe.py).

One phase, `forward`: the network on one voxelized 10-sweep sample, inputs
  voxels            f32 [N_1, 10, 5]   points per voxel (x, y, z, intensity, dt), zero padded
  voxel_num_points  i32 [N_1]          0 for padding slots (valid voxels first)
  rb                {key: i32 array}   host rulebook (rulebook.py), padded to the bounds
                                       of net.level_bounds
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np

from models.spec import ModelSpec, Phase, Sym, as_spec
from models.centerpoint import rulebook
from models.centerpoint.net import (CenterPoint, Config, LevelCapacities, capacities, down_layouts,
                                    layouts, level_bounds, level_shapes)


def pinned(cfg: Config) -> Config:
    """`cfg` with the level capacities and pair-buffer layouts fixed to their derived values
    (worst-case bounds for max_voxels), so that they are independent sizes of the export."""
    return dataclasses.replace(cfg, level_capacities=LevelCapacities(*level_bounds(cfg)[1:]),
                               layouts=tuple(down_layouts(cfg)))


CONFIGS = {
    "full": pinned(Config()),
    # Same weights, depth and width (the unmodified OpenPCDet constructors); only the
    # extents shrink: a 12 x 12 m range (grid 160 x 160 x 40, BEV 20 x 20) and 2000
    # voxel slots of which 1500 are valid, clustered so that every rulebook has
    # real neighbours, grid-boundary voxels and padding.
    "reduced": pinned(dataclasses.replace(Config(), point_cloud_range=(-6.0, -6.0, -5.0, 6.0, 6.0, 3.0),
                                          max_voxels=2000, n_valid=1500, sample="clustered")),
}


def rb_spec(cfg: Config) -> dict:
    b, lay = capacities(cfg), layouts(cfg)
    i32 = lambda *s: jax.ShapeDtypeStruct(s, jnp.int32)
    out = {key: i32(26, b[l]) for l, key in enumerate(cfg.subm_keys)}
    for l, (key, ks, _, _) in enumerate(cfg.down):
        out[key] = i32(math.prod(ks), b[l] if lay[l] == "in" else b[l + 1])
    out["bev_coords"] = i32(b[-1], 3)
    return out


def rb_shapes(cfg: Config) -> dict:
    """Symbolic shapes of the rulebook inputs (rb_spec): level l's capacity is symbol N<l>."""
    lay = layouts(cfg)
    out = {key: f"_, N{l + 1}" for l, key in enumerate(cfg.subm_keys)}
    for l, (key, _, _, _) in enumerate(cfg.down):
        out[key] = f"_, N{l + 1 if lay[l] == 'in' else l + 2}"
    out["bev_coords"] = f"N{len(cfg.down) + 1}, _"
    return out


def input_spec(cfg: Config):
    n = cfg.max_voxels
    return (jax.ShapeDtypeStruct((n, cfg.max_points_per_voxel, cfg.num_point_features), jnp.float32),
            jax.ShapeDtypeStruct((n,), jnp.int32), rb_spec(cfg))


def init_variables(cfg: Config, rng):
    zeros = lambda s: jnp.zeros(s.shape, s.dtype)
    return CenterPoint(cfg).init(rng, *jax.tree.map(zeros, input_spec(cfg)))


# ----------------------------------------------------------------------------
# Host side: example voxel sets, coordinates, rulebook (excluded from the graph)
# ----------------------------------------------------------------------------

def saturated_coords(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """max_voxels level-1 voxels for which every level bound is met exactly:
    a lattice with all coordinates = 1 mod 8 (each voxel reaches 8 level-2 and 8
    level-3 cells, disjoint from every other voxel's), which covers the whole
    level-4/5 grids. 162000 lattice points exist for the full grid; the ones
    dropped (m_z in 1..3, odd m_y and m_x) never leave a level-4 cell uncovered."""
    D, H, W = cfg.sparse_shape
    mz, my, mx = (np.arange((n - 2) // 8 + 1) for n in (D, H, W))
    lat = np.stack(np.meshgrid(mz, my, mx, indexing="ij"), -1).reshape(-1, 3)
    drop = (lat[:, 0] >= 1) & (lat[:, 0] <= 3) & (lat[:, 1] % 2 == 1) & (lat[:, 2] % 2 == 1)
    n_drop = len(lat) - cfg.max_voxels
    assert 0 <= n_drop <= drop.sum(), (len(lat), cfg.max_voxels)
    cand = np.nonzero(drop)[0]
    keep = np.ones(len(lat), bool)
    keep[rng.choice(cand, n_drop, replace=False)] = False
    c = 8 * lat[keep] + 1
    return c[rng.permutation(len(c))]


def clustered_coords(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Surface-like clusters plus scattered voxels and grid-boundary voxels."""
    shape = np.asarray(cfg.sparse_shape)
    n = cfg.n_valid or cfg.max_voxels
    keys, out = set(), []

    def add(c):
        c = np.clip(np.round(c).astype(np.int64), 0, shape - 1)
        for v in c:
            k = tuple(v)
            if k not in keys and len(out) < n:
                keys.add(k)
                out.append(v)
    add(np.array([[0, 0, 0], [shape[0] - 1, shape[1] - 1, shape[2] - 1],
                  [shape[0] - 1, 0, shape[2] - 1], [0, shape[1] - 1, 0]]))
    centers = rng.uniform([2, 0, 0], shape - [3, 1, 1], (12, 3))
    while len(out) < int(0.9 * n):
        c = centers[rng.integers(len(centers))]
        add(c + rng.normal(0, [1.5, 4.0, 4.0], (8, 3)))
    while len(out) < n:
        add(rng.uniform(0, shape - 1, (8, 3)))
    c = np.stack(out)
    return c[rng.permutation(len(c))]


def make_voxels(cfg: Config, coords: np.ndarray, rng: np.random.Generator):
    """Voxelizer-like output: npoints in [1, 10] points strictly inside each voxel,
    zero padded; padding slots are all-zero with npoints = 0 (valid slots first)."""
    N, P, F = cfg.max_voxels, cfg.max_points_per_voxel, cfg.num_point_features
    n = len(coords)
    r, v = np.asarray(cfg.point_cloud_range), np.asarray(cfg.voxel_size)
    npts = np.zeros(N, np.int32)
    npts[:n] = rng.integers(1, P + 1, n)
    vox = np.zeros((N, P, F), np.float32)
    u = rng.uniform(0.1, 0.9, (n, P, 3))
    xyz = r[:3] + (coords[:, None, ::-1] + u) * v           # coords are (z, y, x)
    vox[:n, :, :3] = xyz
    vox[:n, :, 3] = rng.uniform(0, 255, (n, P))              # intensity
    vox[:n, :, 4] = rng.integers(0, 10, (n, P)) * 0.05       # sweep time lag
    vox *= (np.arange(P)[None, :] < npts[:, None])[:, :, None]
    return vox, npts


def coords_from_voxels(cfg: Config, voxels, num_points) -> np.ndarray:
    """Recover the voxelizer's (z, y, x) coordinates from the first point of every
    valid voxel: floor((p - range_min) / voxel_size)."""
    voxels, num_points = np.asarray(voxels), np.asarray(num_points)
    n = int((num_points > 0).sum())
    assert np.all(num_points[:n] > 0) and np.all(num_points[n:] == 0), "valid voxels must come first"
    r, v = np.asarray(cfg.point_cloud_range, np.float64), np.asarray(cfg.voxel_size, np.float64)
    xyz = np.floor((voxels[:n, 0, :3].astype(np.float64) - r[:3]) / v).astype(np.int64)
    return xyz[:, ::-1].copy()


_HOST_CACHE: dict = {}


def host_rulebook(cfg: Config, voxels, num_points, rb=None) -> dict:
    """rulebook.build for the voxels of these inputs; if the graph's rulebook
    inputs `rb` are given they must equal it exactly."""
    h = hashlib.sha1(np.ascontiguousarray(np.asarray(voxels)).tobytes()).hexdigest()
    key = (cfg, h)
    if key not in _HOST_CACHE:
        _HOST_CACHE.clear()
        _HOST_CACHE[key] = rulebook.build(cfg, coords_from_voxels(cfg, voxels, num_points))
    res = _HOST_CACHE[key]
    if rb is not None:
        for k, a in res["inputs"].items():
            assert np.array_equal(np.asarray(rb[k]), a), f"rulebook input {k} differs from host rebuild"
    return res


def example_inputs(cfg: Config, rng):
    seed = int(jax.random.randint(rng, (), 0, 2 ** 31 - 1))
    g = np.random.default_rng(seed)
    coords = saturated_coords(cfg, g) if cfg.sample == "saturated" else clustered_coords(cfg, g)
    vox, npts = make_voxels(cfg, coords, g)
    rb = host_rulebook(cfg, vox, npts)
    spec = rb_spec(cfg)
    for k, s in spec.items():
        assert rb["inputs"][k].shape == s.shape, (k, rb["inputs"][k].shape, s.shape)
    return {"forward": (jnp.asarray(vox), jnp.asarray(npts),
                        {k: jnp.asarray(rb["inputs"][k]) for k in spec})}


# ----------------------------------------------------------------------------
# intermediates
# ----------------------------------------------------------------------------

def sparse_level(path: str) -> int | None:
    """Level (1..5) of a sparse-tensor-valued module, None for dense ones."""
    if path == "vfe":
        return 1
    if not path.startswith("backbone_3d/"):
        return None
    head = path.split("/")[1]
    if head.startswith(("conv_input", "conv1_")):
        return 1
    if head.startswith("conv_out"):
        return 5
    return int(head[len("conv")])


def intermediates(cfg: Config, variables, voxels, num_points, rb) -> dict:
    _, state = jax.jit(lambda v, *a: CenterPoint(cfg).apply(
        v, *a, capture_intermediates=True, mutable=["intermediates"]))(
            variables, voxels, num_points, rb)
    counts = [len(c) for c in host_rulebook(cfg, voxels, num_points, rb)["levels"]]
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        if not keys or keys[-1] != "__call__":
            continue
        name, out = "/".join(keys[:-1]), v[0]
        if isinstance(out, dict):                       # SeparateHead
            flat.update({f"{name}@{k}": a for k, a in out.items()})
        elif hasattr(out, "shape") and name:
            lvl = sparse_level(name)
            flat[name] = out if lvl is None else np.asarray(out)[:counts[lvl - 1]]
    return flat


# ----------------------------------------------------------------------------
# spec
# ----------------------------------------------------------------------------

def build(cfg: Config) -> ModelSpec:
    module = CenterPoint(cfg)
    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))

    def forward(params, voxels, voxel_num_points, rb):
        return module.apply(params, voxels, voxel_num_points, rb)

    b, shapes = capacities(cfg), level_shapes(cfg)
    D5, H5, W5 = shapes[-1]

    return ModelSpec(
        name="centerpoint",
        variant=(f"OpenPCDet CenterPoint cbgs_voxel0075_res3d (nuScenes, 10 sweeps): "
                 f"{cfg.max_voxels} voxels x {cfg.max_points_per_voxel} points, sparse grid "
                 f"{cfg.sparse_shape}, BEV {H5}x{W5}, batch 1; NVIDIA CUDA-CenterPoint INT8: "
                 f"libspconv INT8 sparse convs (per-channel W, per-tensor A, shared-scale QuantAdd; "
                 f"conv_input + first block fp16, extra_conv fp16 out) + TensorRT INT8 RPN / head"),
        phases=[Phase("forward", forward, input_spec(cfg), trip_count=1,
                      shapes=("N1, _, _", "N1", rb_shapes(cfg)),
                      note=(f"one CenterPoint forward on one voxelized sample: MeanVFE, "
                            f"VoxelResBackBone8x on padded voxel levels {b} (worst-case bounds "
                            f"for {cfg.max_voxels} input voxels, grids {shapes}), "
                            f"HeightCompression to [1,{H5},{W5},{128 * D5}] (NHWC), "
                            f"BaseBEVBackbone, CenterHead; output: 6 head dicts "
                            f"(center, center_z, dim, rot, vel, hm), NHWC"))],
        param_specs=as_spec(variables),
        symbols={"N1": Sym("max_voxels", 1800, "padded voxel slots (level-1 sparse capacity)"),
                 "N2": Sym("level_capacities.n2", 14000, "level-2 sparse capacity (padded voxels)"),
                 "N3": Sym("level_capacities.n3", 12000, "level-3 sparse capacity (padded voxels)"),
                 "N4": Sym("level_capacities.n4", 1500, "level-4 sparse capacity (padded voxels)"),
                 "N5": Sym("level_capacities.n5", 700, "level-5 sparse capacity (padded voxels)")},
        # a level cannot hold more active voxels than its grid has cells
        constraints=tuple(f"N{l + 1} <= {math.prod(s)}" for l, s in enumerate(shapes)),
        excluded=[
            "point loading and 10-sweep aggregation (nuScenes infos, time-lag channel) (host)",
            "mask_points_and_boxes_outside_range, shuffle_points (host)",
            "voxelization (spconv Point2VoxelCPU3d via VoxelGeneratorWrapper, voxel 0.075 x 0.075 "
            "x 0.2, <= 10 points, <= 160000 voxels) and padding to 160000 slots (host)",
            "sparse-conv rulebook generation (spconv get_indice_pairs per indice_key: output "
            "coordinates and per-offset pairs), padded to the level bounds (host, rulebook.py)",
            "CenterHead.generate_predicted_boxes: sigmoid, top-K (500) decode, score threshold, "
            "post-center range filter, class-agnostic NMS (iou3d_nms nms_gpu, 0.2, pre 1000, "
            "post 83) and label mapping (post-processing; CUDA-CenterPoint: fp16 decode + NMS "
            "kernels on the fp16 head outputs)",
            "INT8 calibration (histogram / MSE amax per tensor) and BN folding: offline; scales "
            "and folded weights are parameters",
        ],
    )

