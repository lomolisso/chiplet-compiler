"""PointPainting (MLPerf Automotive 3d-object-detection) capture spec, INT8 (TensorRT INT8
fallback with fp16 PointPillars carve-outs; see net.py and recipe.py).

Components (disjoint parameter subtrees, selected per phase via
`Phase.param_keys`):
  segmenter     DeepLabV3+ / ResNet-50 (OS16), 19 classes, one camera image
  pointpillars  PointPillars core on 32000 padded pillars, painted (11-dim points)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from models.spec import ModelSpec, Phase, Sym, as_spec
from models.pointpainting.net import DeepLabV3Plus, PointPillarsCore, PPConfig, SegConfig


@dataclass(frozen=True)
class Config:
    seg: SegConfig = field(default_factory=SegConfig)
    pp: PPConfig = field(default_factory=PPConfig)
    # Waymo camera images are fed at native resolution (ToTensor + Normalize,
    # no resize): FRONT / FRONT_LEFT / FRONT_RIGHT 1920x1280, SIDE_LEFT /
    # SIDE_RIGHT 1920x886.
    front_h: int = 1280
    front_w: int = 1920
    side_h: int = 886
    side_w: int = 1920
    batch: int = 1                      # camera images per segmenter call (MLPerf: 1)
    n_front: int = 3
    n_side: int = 2
    n_valid: int | None = None          # valid pillars in example inputs (None: all)

    @property
    def image_front(self) -> tuple[int, int]:
        return (self.front_h, self.front_w)

    @property
    def image_side(self) -> tuple[int, int]:
        return (self.side_h, self.side_w)


CONFIGS = {
    "full": Config(),
    # Same weights / depth / width (the unmodified MLPerf constructors); only
    # input extents shrink: small images (90 is not a multiple of 16 ->
    # non-integer bilinear scales, like 886), a 64x64 BEV canvas (+-10.24 m)
    # and 512 pillar slots of which 400 are valid.
    "reduced": Config(pp=PPConfig(point_cloud_range=(-10.24, -10.24, -2.0, 10.24, 10.24, 4.0),
                                  max_voxels=512),
                      front_h=128, front_w=192, side_h=90, side_w=192, n_valid=400),
}

PHASE_COMPONENT = {"segmenter": "segmenter", "segmenter_side": "segmenter",
                   "pointpillars": "pointpillars"}


def _modules(cfg: Config):
    return {"segmenter": DeepLabV3Plus(cfg.seg), "pointpillars": PointPillarsCore(cfg.pp)}


def input_specs(cfg: Config) -> dict:
    p = cfg.pp
    img = lambda hw: (jax.ShapeDtypeStruct((cfg.batch, *hw, 3), jnp.float32),)
    return {
        "segmenter": img(cfg.image_front),
        "segmenter_side": img(cfg.image_side),
        "pointpillars": (
            jax.ShapeDtypeStruct((p.max_voxels, p.max_num_points, p.point_dim), jnp.float32),
            jax.ShapeDtypeStruct((p.max_voxels, 3), jnp.int32),
            jax.ShapeDtypeStruct((p.max_voxels,), jnp.int32),
        ),
    }


def init_variables(cfg: Config, rng):
    specs, mods = input_specs(cfg), _modules(cfg)
    k1, k2 = jax.random.split(rng)
    zeros = lambda s: jnp.zeros(s.shape, s.dtype)
    return {
        "segmenter": mods["segmenter"].init(k1, *map(zeros, specs["segmenter"])),
        "pointpillars": mods["pointpillars"].init(k2, *map(zeros, specs["pointpillars"])),
    }


def _pillars(cfg: Config, rng: np.random.Generator):
    """Voxelizer-like padded pillars: unique (x, y) cells, npoints in [1, 20],
    zero-padded points, painted scores in [0, 1]; padding slots are all-zero
    with npoints = 0 and are scattered among the valid slots."""
    p = cfg.pp
    P, M = p.max_voxels, p.max_num_points
    n_valid = P if cfg.n_valid is None else cfg.n_valid
    cells = rng.choice(p.x_l * p.y_l, size=n_valid, replace=False)
    slots = np.sort(rng.choice(P, size=n_valid, replace=False))
    coors = np.zeros((P, 3), np.int32)
    coors[slots, 0] = cells % p.x_l
    coors[slots, 1] = cells // p.x_l
    npoints = np.zeros((P,), np.int32)
    npoints[slots] = rng.integers(1, M + 1, size=n_valid)
    vx, vy, vz = p.voxel_size
    r = p.point_cloud_range
    pts = np.zeros((P, M, p.point_dim), np.float32)
    x = r[0] + (coors[:, None, 0] + rng.random((P, M))) * vx
    y = r[1] + (coors[:, None, 1] + rng.random((P, M))) * vy
    z = r[2] + rng.random((P, M)) * vz
    pts[:, :, 0], pts[:, :, 1], pts[:, :, 2] = x, y, z
    pts[:, :, 3:5] = rng.standard_normal((P, M, 2))
    pts[:, :, 5:] = rng.random((P, M, p.point_dim - 5))
    pts *= (np.arange(M)[None, :] < npoints[:, None])[:, :, None]
    return jnp.asarray(pts), jnp.asarray(coors), jnp.asarray(npoints)


def example_inputs(cfg: Config, rng):
    k1, k2, k3 = jax.random.split(rng, 3)
    specs = input_specs(cfg)
    seed = int(jax.random.randint(k3, (), 0, 2 ** 31 - 1))
    return {
        "segmenter": (jax.random.normal(k1, specs["segmenter"][0].shape),),
        "segmenter_side": (jax.random.normal(k2, specs["segmenter_side"][0].shape),),
        "pointpillars": _pillars(cfg, np.random.default_rng(seed)),
    }


def intermediates(cfg: Config, variables, *inputs, phase: str | None = None) -> dict:
    if phase is not None:
        comp = PHASE_COMPONENT[phase]
    else:                                   # no phase given: infer from the signature
        comp = "segmenter" if len(inputs) == 1 else "pointpillars"
    _, state = _modules(cfg)[comp].apply(variables[comp], *inputs, capture_intermediates=True,
                                         mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        if not keys or keys[-1] != "__call__" or not hasattr(v[0], "shape"):
            continue
        flat["/".join([comp] + keys[:-1])] = v[0]
    if comp == "pointpillars":
        # Per-pillar activations: keep only the valid pillars.
        valid = np.asarray(inputs[2]) > 0
        for k in ("pointpillars/pillar_encoder/conv",):
            flat[k] = np.asarray(flat[k])[valid]
    return flat


def build(cfg: Config) -> ModelSpec:
    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    specs, mods = input_specs(cfg), _modules(cfg)

    def phase_fn(comp):
        def fn(params, *inputs):
            return mods[comp].apply(params[comp], *inputs)
        return fn

    p = cfg.pp
    phases = [
        Phase("segmenter", phase_fn("segmenter"), specs["segmenter"], trip_count=cfg.n_front,
              param_keys=("segmenter",), shapes=("B, H, W, _",),
              note=f"DeepLabV3+ R50 OS16 on one front camera image {cfg.image_front} (HxW, NHWC, "
                   "normalized fp32, quantized by conv1); 3 calls/frame (FRONT, FRONT_LEFT, "
                   "FRONT_RIGHT). Output: 19-class fp32 logits at input resolution."),
        Phase("segmenter_side", phase_fn("segmenter"), specs["segmenter_side"],
              trip_count=cfg.n_side, param_keys=("segmenter",), shapes=("B, H, W, _",),
              size_variant_of="segmenter",
              note=f"Same network on a side camera image {cfg.image_side}; 2 calls/frame "
                   "(SIDE_LEFT, SIDE_RIGHT)."),
        Phase("pointpillars", phase_fn("pointpillars"), specs["pointpillars"], trip_count=1,
              param_keys=("pointpillars",), shapes=("P, _, _", "P, _", "P"),
              note=f"PointPillarsCore up to the head on {p.max_voxels} padded pillars x "
                   f"{p.max_num_points} points x {p.point_dim} (painted) features, pillar "
                   f"coords (x, y, z) and npoints (0 = padding slot); BEV canvas "
                   f"{p.y_l}x{p.x_l}. Outputs NHWC fp16 (cls, reg, dir)."),
    ]
    return ModelSpec(
        name="pointpainting",
        variant=f"MLPerf Automotive PointPainting (Waymo): DeepLabV3+ R50 OS16 x5 cameras "
                f"{cfg.image_front}/{cfg.image_side} + PointPillars ({p.max_voxels} pillars, "
                f"{p.y_l}x{p.x_l} BEV), TensorRT INT8 W8A8 fallback (per-channel W, per-tensor "
                f"static A, BN folded; PFN conv, multi_blocks_0_3, conv_reg and resizes fp16), "
                f"batch 1",
        phases=phases,
        param_specs=as_spec(variables),
        symbols={"B": Sym("batch", 2, "camera images per segmenter call"),
                 "H": Sym("front_h", 100, "camera image height (front 1280, side 886 at full)"),
                 "W": Sym("front_w", 160, "camera image width"),
                 "P": Sym("pp.max_voxels", 640, "padded pillar slots")},
        constraints=("H >= 16", "W >= 16"),
        excluded=[
            "image decode + ToTensor + Normalize(ImageNet mean/std) (host preprocessing)",
            "Painter.get_score: softmax over the 19 logits and regrouping into 6 painting "
            "classes (host, part of painting)",
            "Painter.augment_lidar_class_scores_both: lidar->camera projection of every point "
            "into the 5 images, score lookup and multi-camera averaging (host painting)",
            "point_range_filter and hard voxelization (PillarLayer / numba "
            "_points_to_voxel_reverse_kernel; data-dependent pillar count, host)",
            "padding of the pillar set to max_voxels=32000 (host; padding slots have npoints=0)",
            "anchor generation, sigmoid, top-k (nms_pre=4096) and anchors2bboxes decode "
            "(PointPillarsCore.get_predicted_bboxes)",
            "per-class score threshold + rotated NMS (open3d ml3d.ops.nms), max_num=500, "
            "limit_period / direction fix (PointPillarsPos), keep_bbox_from_image_range",
        ],
    )
