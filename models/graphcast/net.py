"""GraphCast_operational (0.25 deg, 13 levels, precipitation output only):
one autoregressive 6 h step with a static graph.

The network is DeepMind's own JAX/Haiku code, vendored at upstream/graphcast
(google-deepmind/graphcast v0.2 @ 08cf7362; unmodified at 7b50334, since then edited in
place for FP8, see below). The predictor stack is the one the demo notebook
(`graphcast_demo.ipynb`, "Build jitted functions") runs:

    autoregressive.Predictor(
        normalization.InputsAndResiduals(
            casting.Bfloat16Cast(graphcast.GraphCast(model_config, task_config)),
            stddev_by_level, mean_by_level, diffs_stddev_by_level),
        gradient_checkpointing=True)

called by `rollout.chunked_prediction` once per 6 h step (num_steps_per_chunk=1).

Quantization (models/graphcast/recipe.py): FP8 E4M3 W8A8 with per-tensor static scales on
every MLP linear (the 86 GEMMs of a step), the GPU-datacenter fallback -- GraphCast has no
quantized release; DeepMind runs it in bf16 (Bfloat16Cast). Written into the vendored
source, each edit marked `# quantized`: graphcast/mlp.py `Fp8Linear` / `Fp8MLP` (haiku's
hk.Linear / hk.nets.MLP in explicit Q/DQ form: fp8 `w` + fp32 `w_scale`, fp32
`input_scale`, f8 x f8 -> f32 dot, fp32 scale + bias epilogue, bf16 out), used by
deep_typed_graph_net.py `DeepTypedGraphNet._networks_builder.build_mlp`. Stays float:
swish, LayerNorm, residual adds, gathers and the mesh / mesh2grid segment sums in bf16;
the grid2mesh segment sum (f32_aggregation), normalization, residual + unnormalization of
the predictions (InputsAndResiduals, outside the bf16 cast) in fp32.

What this file adds on top of the library -- scoped monkeypatches for the static graph
and unrolling (none for quantization):

  * `static_graph(graph)` -- while active, `GraphCast._maybe_init` takes the
    grid2mesh / mesh / mesh2grid graph structure (senders, receivers, edge
    features, grid2mesh node features) from `graph`, i.e. from phase inputs,
    instead of rebuilding it in numpy inside the trace. The arrays are exactly
    the ones `GraphCast._init_*_graph` builds (model.graph_inputs runs those
    library methods once on the host); the unpatched library builds them
    itself.
  * `unrolled()` -- while active, `hk.scan` is called with `unroll=True`, so
    the 1-step scan of `autoregressive.Predictor` lowers to straight-line code.
    The 16 processor steps are already a Python loop in deep_typed_graph_net.
  * xarray glue (`datasets`, `stats_datasets`, `targets_template`,
    `prediction_arrays`): the phase takes plain arrays and wraps them into the
    canonical xarray Datasets (variables, dims and coordinates of
    data_utils.extract_inputs_targets_forcings), so the whole xarray-level
    stack (normalization, residual, casting, stacking) is traced.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass

GC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")
if GC_DIR not in sys.path:
    sys.path.insert(0, GC_DIR)

import haiku as hk  # noqa: E402
import numpy as np  # noqa: E402
import xarray  # noqa: E402
from graphcast import autoregressive  # noqa: E402
from graphcast import casting  # noqa: E402
from graphcast import graphcast as gc  # noqa: E402
from graphcast import normalization  # noqa: E402
from graphcast import typed_graph  # noqa: E402
from graphcast import xarray_jax  # noqa: E402


@dataclass(frozen=True)
class Config:
    # graphcast.ModelConfig of the 0.25 deg models (Lam et al. 2023 and the
    # repo).
    resolution: float = 0.25
    mesh_size: int = 6
    latent_size: int = 512
    gnn_msg_steps: int = 16
    hidden_layers: int = 1
    radius_query_fraction_edge_length: float = 0.6
    mesh2grid_edge_normalization_factor: float | None = None
    # Bfloat16Cast enabled (demo); False runs the same stack in fp32.
    bfloat16: bool = True
    batch: int = 1
    # Autoregressive rollout: 40 x 6 h = 10 days (the paper's forecast length).
    rollout_steps: int = 40


TASK = gc.TASK_13_PRECIP_OUT
STEP = np.timedelta64(6, "h")
INPUT_STEPS = 2                         # input_duration "12h" at 6 h steps
STATIC_VARS = gc.STATIC_VARS
ATMOS_VARS = frozenset(gc.ALL_ATMOSPHERIC_VARS)


def model_config(cfg: Config) -> gc.ModelConfig:
    return gc.ModelConfig(
        resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.latent_size, gnn_msg_steps=cfg.gnn_msg_steps,
        hidden_layers=cfg.hidden_layers,
        radius_query_fraction_edge_length=cfg.radius_query_fraction_edge_length,
        mesh2grid_edge_normalization_factor=cfg.mesh2grid_edge_normalization_factor)


def grid(cfg: Config):
    """Equiangular lat/lon grid of the ERA5/HRES datasets (lat -90..90 incl.)."""
    n_lat = int(round(180 / cfg.resolution)) + 1
    n_lon = int(round(360 / cfg.resolution))
    lat = np.linspace(-90.0, 90.0, n_lat).astype(np.float32)
    lon = (np.arange(n_lon) * cfg.resolution).astype(np.float32)
    return lat, lon


def var_dims(name: str) -> tuple[str, ...]:
    """Dims of each variable as data_utils produces them (the example data)."""
    if name in STATIC_VARS:
        return ("lat", "lon")
    if name.startswith("year_progress"):
        return ("batch", "time")
    if name.startswith("day_progress"):
        return ("batch", "time", "lon")
    if name in ATMOS_VARS:
        return ("batch", "time", "level", "lat", "lon")
    return ("batch", "time", "lat", "lon")


def var_shape(cfg: Config, name: str, n_time: int) -> tuple[int, ...]:
    lat, lon = grid(cfg)
    sizes = {"batch": cfg.batch, "time": n_time, "level": len(TASK.pressure_levels),
             "lat": len(lat), "lon": len(lon)}
    return tuple(sizes[d] for d in var_dims(name))


def _coords(cfg: Config, times):
    lat, lon = grid(cfg)
    return {"lat": lat, "lon": lon, "level": np.array(TASK.pressure_levels),
            "time": np.asarray(times, "timedelta64[ns]")}


INPUT_TIMES = [-STEP, np.timedelta64(0, "h")]
TARGET_TIMES = [STEP]


def _dataset(cfg: Config, arrays: dict, times) -> xarray.Dataset:
    coords = _coords(cfg, times)
    data = {k: (var_dims(k), v) for k, v in arrays.items()}
    used = {d for k in arrays for d in var_dims(k)}
    return xarray_jax.Dataset(data, coords={k: v for k, v in coords.items() if k in used})


def datasets(cfg: Config, inputs: dict, forcings: dict):
    """(inputs, targets_template, forcings) Datasets of one 6 h step."""
    return (_dataset(cfg, inputs, INPUT_TIMES), targets_template(cfg),
            _dataset(cfg, forcings, TARGET_TIMES))


def targets_template(cfg: Config) -> xarray.Dataset:
    """Static template (`targets * nan` in the demo): only its variables,
    dims, coordinates and dtype are read."""
    arrays = {k: np.full(var_shape(cfg, k, 1), np.nan, np.float32)
              for k in TASK.target_variables}
    return _dataset(cfg, arrays, TARGET_TIMES)


def stats_datasets(stats: dict) -> dict:
    """{mean_by_level, stddev_by_level, diffs_stddev_by_level} -> Datasets."""
    out = {}
    for name, entries in stats.items():
        data = {k: (("level",) if k in ATMOS_VARS else (), v) for k, v in entries.items()}
        out[name] = xarray_jax.Dataset(data, coords={"level": np.array(TASK.pressure_levels)})
    return out


def prediction_arrays(predictions: xarray.Dataset) -> dict:
    """{variable: array} in the Dataset's own layout (time, batch, [level,] lat, lon)."""
    return {k: xarray_jax.unwrap_data(predictions[k]) for k in sorted(predictions.data_vars)}


def construct_wrapped_graphcast(model_cfg, task_cfg, stats: dict, bfloat16: bool = True):
    """The demo notebook's `construct_wrapped_graphcast` (graphcast_demo.ipynb),
    with Bfloat16Cast's own `enabled` switch exposed for fp32 configs."""
    # Deeper one-step predictor.
    predictor = gc.GraphCast(model_cfg, task_cfg)
    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor, enabled=bfloat16)
    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        stddev_by_level=stats["stddev_by_level"])
    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


def make_fn(cfg: Config, stats: dict):
    """hk.transform_with_state(run_forward) as in the demo, for given stats."""
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config(cfg), TASK, stats, cfg.bfloat16)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)
    return hk.transform_with_state(run_forward)


# ----------------------------------------------------------------------------
# Static graph structure as inputs
# ----------------------------------------------------------------------------

def _bipartite(graph: dict, name: str, senders_set: str, receivers_set: str,
               n_nodes: dict, node_features: dict) -> typed_graph.TypedGraph:
    g = graph[name]
    nodes = {k: typed_graph.NodeSet(n_node=np.array([n_nodes[k]]), features=node_features[k])
             for k in (senders_set, receivers_set)}
    edges = {typed_graph.EdgeSetKey(name, (senders_set, receivers_set)): typed_graph.EdgeSet(
        n_edge=np.array([g["senders"].shape[0]]),
        indices=typed_graph.EdgesIndices(senders=g["senders"], receivers=g["receivers"]),
        features=g["edge_features"])}
    return typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes, edges=edges)


@contextlib.contextmanager
def static_graph(graph: dict):
    """GraphCast._maybe_init with the three graph structures from `graph`.

    Mesh / grid properties (lat/lon of the nodes) are still initialized by the
    library. The node features of the mesh graph and the mesh2grid graph are
    replaced by latents before use in GraphCast._run_{mesh,mesh2grid}_gnn; they
    hold the same structural features as grid2mesh's (same lat/lon functions)."""
    orig = gc.GraphCast._maybe_init

    def _maybe_init(self, sample_inputs):
        if not self._initialized:   # (as the library: build once per predictor)
            self._init_mesh_properties()
            self._init_grid_properties(grid_lat=sample_inputs.lat, grid_lon=sample_inputs.lon)
            n_nodes = {"grid_nodes": self._num_grid_nodes, "mesh_nodes": self._num_mesh_nodes}
            feats = {"grid_nodes": graph["grid2mesh"]["grid_node_features"],
                     "mesh_nodes": graph["grid2mesh"]["mesh_node_features"]}
            self._grid2mesh_graph_structure = _bipartite(
                graph, "grid2mesh", "grid_nodes", "mesh_nodes", n_nodes, feats)
            self._mesh_graph_structure = _bipartite(
                graph, "mesh", "mesh_nodes", "mesh_nodes", n_nodes, feats)
            self._mesh2grid_graph_structure = _bipartite(
                graph, "mesh2grid", "mesh_nodes", "grid_nodes", n_nodes, feats)
            self._initialized = True

    gc.GraphCast._maybe_init = _maybe_init
    try:
        yield
    finally:
        gc.GraphCast._maybe_init = orig


_orig_scan = hk.scan


def _scan_unrolled(f, init, xs, length=None, reverse=False, unroll=1):
    del unroll
    return _orig_scan(f, init, xs, length=length, reverse=reverse, unroll=True)


@contextlib.contextmanager
def unrolled():
    """Fully unroll hk.scan (autoregressive.Predictor's 1-step scan) while active."""
    hk.scan = _scan_unrolled
    try:
        yield
    finally:
        hk.scan = _orig_scan
