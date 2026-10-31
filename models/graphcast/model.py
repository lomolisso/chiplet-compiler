"""GraphCast_operational capture spec: one autoregressive 6 h step.

Phase `forecast_step` = one call of the demo's jitted `run_forward`
(autoregressive.Predictor over InputsAndResiduals / Bfloat16Cast / GraphCast)
with a single target time, as `rollout.chunked_prediction` makes it per step:
normalization, bf16 cast, grid2mesh encoder, 16 multi-mesh processor steps,
mesh2grid decoder, residual + unnormalization. Every MLP linear is FP8 E4M3 W8A8
(per-tensor static; vendored graphcast/mlp.py Fp8Linear, see net.py). It runs `rollout_steps` = 40
times for a 10-day forecast; the input window shift between steps
(`rollout._get_next_inputs`) is host-side xarray concat/tail.
"""

from __future__ import annotations

import functools
import re
from collections import Counter

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax._src import source_info_util
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow.loops import _stack as _scan_stack

from models.spec import ModelSpec, Phase, Sym, as_spec
from models.graphcast import net
from models.graphcast.net import Config

from graphcast import graphcast as gc  # noqa: E402  (path set by net)

CONFIGS = {
    "full": Config(),
    # Every block type kept (grid2mesh encoder, multi-mesh processor, mesh2grid
    # decoder, all variables and 13 levels, bf16 cast): 6 deg grid (31 x 60),
    # mesh 0..2 (162 nodes), latent 32, 3 processor steps.
    "reduced": Config(resolution=6.0, mesh_size=2, latent_size=32, gnn_msg_steps=3),
}

PHASE = "forecast_step"
GNNS = ("grid2mesh_gnn", "mesh_gnn", "mesh2grid_gnn")

# mesh2grid_gnn builds a node update for every node set in node_latent_size
# (grid AND mesh nodes), but GraphCast only reads the grid nodes of its output:
# the mesh-node update MLP + LayerNorm of mesh2grid reach no output.
DEAD_RE = re.compile(r"^mesh2grid_gnn/~_networks_builder/processor_nodes_\d+_mesh_nodes_")


# ----------------------------------------------------------------------------
# Host-side static inputs: graph structure, normalization statistics
# ----------------------------------------------------------------------------

def compute_dtype(cfg: Config):
    return jnp.bfloat16 if cfg.bfloat16 else jnp.float32


@functools.lru_cache(maxsize=4)
def _library_graphs(cfg: Config):
    """The three TypedGraphs exactly as GraphCast._maybe_init builds them
    (numpy, host-side), for this config's grid."""
    lat, lon = net.grid(cfg)
    box = {}

    def f():
        m = gc.GraphCast(net.model_config(cfg), net.TASK)
        import xarray
        m._maybe_init(xarray.Dataset(coords={"lat": lat, "lon": lon}))
        box["g"] = (m._grid2mesh_graph_structure, m._mesh_graph_structure,
                    m._mesh2grid_graph_structure)
    hk.transform(f).init(None)
    return box["g"]


@functools.lru_cache(maxsize=4)
def graph_inputs(cfg: Config) -> dict:
    """Static graph inputs: indices as int32 (what x32 JAX makes of the numpy
    int64 constants) and structural features already cast to the activation
    dtype, as the library does in numpy (`features.astype(dtype)` on a numpy
    array) before broadcasting them in-graph."""
    dt = compute_dtype(cfg)
    out = {}
    for name, g in zip(("grid2mesh", "mesh", "mesh2grid"), _library_graphs(cfg)):
        (e,) = g.edges.values()
        out[name] = {"senders": np.asarray(e.indices.senders, np.int32),
                     "receivers": np.asarray(e.indices.receivers, np.int32),
                     "edge_features": np.asarray(e.features).astype(dt)}
    g2m = _library_graphs(cfg)[0]
    out["grid2mesh"]["grid_node_features"] = np.asarray(g2m.nodes["grid_nodes"].features).astype(dt)
    out["grid2mesh"]["mesh_node_features"] = np.asarray(g2m.nodes["mesh_nodes"].features).astype(dt)
    return out


def stats_keys() -> dict[str, list[str]]:
    """The normalization entries the model reads (any other entry of the
    stats files would be an unused input): mean/stddev for every input and
    forcing variable and for precipitation (a target that is not an input,
    unnormalized directly); diffs_stddev for the targets that are inputs
    (predicted as residuals)."""
    t = net.TASK
    direct = sorted(set(t.input_variables) | set(t.forcing_variables)
                    | (set(t.target_variables) - set(t.input_variables)))
    resid = sorted(set(t.target_variables) & set(t.input_variables))
    return {"mean_by_level": direct, "stddev_by_level": direct, "diffs_stddev_by_level": resid}


def input_spec(cfg: Config):
    t = net.TASK
    f32 = jnp.float32
    inputs = {k: jax.ShapeDtypeStruct(net.var_shape(cfg, k, net.INPUT_STEPS), f32)
              for k in t.input_variables}
    forcings = {k: jax.ShapeDtypeStruct(net.var_shape(cfg, k, 1), f32)
                for k in t.forcing_variables}
    nl = len(t.pressure_levels)
    stats = {s: {k: jax.ShapeDtypeStruct((nl,) if k in net.ATMOS_VARS else (), f32)
                 for k in keys} for s, keys in stats_keys().items()}
    graph = as_spec(graph_inputs(cfg))
    return inputs, forcings, stats, graph


def input_shapes(cfg: Config):
    """`Phase.shapes` of `input_spec`. Symbols: the batch B and everything that depends on
    the input grid (GraphCast builds its grid nodes and grid2mesh / mesh2grid graphs from
    the inputs' coordinates, `GraphCast._maybe_init`): H lat x W lon points, G = H*W grid
    nodes, Eg grid2mesh and Em mesh2grid edges. Concrete: the 2-step input window and the
    1 forcing step, the 13 levels, the statistics, feature widths, and the multi-mesh
    (mesh nodes / edges are fixed by ModelConfig.mesh_size)."""
    names = {"batch": "B", "lat": "H", "lon": "W"}
    spec = lambda k: ", ".join(names.get(d, "_") for d in net.var_dims(k))
    t = net.TASK
    graph = {"grid2mesh": {"senders": "Eg", "receivers": "Eg", "edge_features": "Eg, _",
                           "grid_node_features": "G, _", "mesh_node_features": None},
             "mesh": None,
             "mesh2grid": {"senders": "Em", "receivers": "Em", "edge_features": "Em, _"}}
    return ({k: spec(k) for k in t.input_variables},
            {k: spec(k) for k in t.forcing_variables}, None, graph)


def synthetic_inputs(cfg: Config, seed: int):
    """Random inputs (N(0,1) fields) and statistics (means N(0,1), scales in
    [0.5, 2]); the real ones are ERA5/HRES data files, not fetched."""
    rng = np.random.default_rng(seed)
    inputs, forcings, stats, _ = input_spec(cfg)

    def normal(s):
        return jnp.asarray(rng.standard_normal(s.shape, dtype=np.float32))

    inputs = {k: normal(s) for k, s in inputs.items()}
    forcings = {k: normal(s) for k, s in forcings.items()}
    stats = {n: {k: jnp.asarray((rng.standard_normal(s.shape) if n == "mean_by_level"
                                 else rng.uniform(0.5, 2.0, s.shape)).astype(np.float32))
                 for k, s in e.items()} for n, e in stats.items()}
    graph = jax.tree.map(jnp.asarray, graph_inputs(cfg))
    return inputs, forcings, stats, graph


def example_inputs(cfg: Config, rng):
    seed = int(jax.random.randint(rng, (), 0, 2 ** 30))
    return {PHASE: synthetic_inputs(cfg, seed)}


# ----------------------------------------------------------------------------
# Forward
# ----------------------------------------------------------------------------

def _module_scope(next_f, args, kwargs, context):
    """hk interceptor: run every module method under a name stack equal to the
    module's full haiku name (== its parameter path). Metadata only."""
    name = getattr(context.module, "module_name", None)
    if name is None:                    # __init__ runs before the name exists
        return next_f(*args, **kwargs)
    with source_info_util.reset_name_stack(), jax.named_scope(name):
        return next_f(*args, **kwargs)


def _clean_name_stack(ns):
    """Keep plain scopes; drop transform wrappers and `eval_jaxpr`."""
    return source_info_util.NameStack(tuple(
        e for e in ns.stack
        if isinstance(e, source_info_util.Scope) and e.name != "eval_jaxpr"))


def _is_absolute(ns) -> bool:
    return bool(ns.stack) and ns.stack[0].name.split("/")[0] in GNNS


def _clean_param(v):
    if isinstance(v, jax_core.Jaxpr):
        return _clean_jaxpr(v)
    if isinstance(v, tuple) and any(isinstance(x, jax_core.Jaxpr) for x in v):
        return tuple(_clean_param(x) for x in v)
    return v


def _clean_jaxpr(jaxpr):
    eqns = [e.replace(params={k: _clean_param(v) for k, v in e.params.items()},
                      source_info=e.source_info.replace(
                          name_stack=_clean_name_stack(e.source_info.name_stack)))
            for e in jaxpr.eqns]
    return jaxpr.replace(eqns=eqns)


def _eval(jaxpr, consts, args):
    """core.eval_jaxpr, plus: inline `eval_jaxpr` call eqns and expand fully
    unrolled scans (unroll >= length) the way JAX's own lowering of such a
    scan does (slice xs, run the body, stack ys). Same as models/alphafold2."""
    env = {}
    read = lambda v: v.val if isinstance(v, jax_core.Literal) else env[v]
    for v, x in zip(jaxpr.all_invars, [*consts, *args]):
        env[v] = x
    for eqn in jaxpr.eqns:
        invals = [read(v) for v in eqn.invars]
        ns = _clean_name_stack(eqn.source_info.name_stack)
        if not _is_absolute(ns):
            ns = source_info_util.current_name_stack() + ns
        p = eqn.params
        with source_info_util.user_context(eqn.source_info.traceback, name_stack=ns), \
                eqn.ctx.manager:
            if eqn.primitive is jax_core.eval_jaxpr_p:
                outs = _eval(p["call_jaxpr"], p["call_jaxpr"].consts, invals)
            elif eqn.primitive.name == "scan" and (p["unroll"] is True
                                                   or p["unroll"] >= p["length"]):
                cs, carry, xs = map(list, p["ft_in"].update(invals).unpack())
                body, ys = p["jaxpr"], []
                for i_ in range(p["length"]):
                    i = p["length"] - i_ - 1 if p["reverse"] else i_
                    x = [jax.lax.index_in_dim(v, i, keepdims=False) for v in xs]
                    carry_y = _eval(body, body.consts, [*cs, *carry, *x])
                    carry, y = map(list, p["ft_out"].update(carry_y).unpack())
                    ys.append(y)
                ys = list(reversed(ys)) if p["reverse"] else ys
                outs = [*carry, *map(_scan_stack, zip(*ys))]
            else:
                outs = eqn.primitive.bind(
                    *invals, **eqn.primitive.get_bind_params(
                        {k: _clean_param(v) for k, v in p.items()}))
        if not eqn.primitive.multiple_results:
            outs = [outs]
        for v, x in zip(eqn.outvars, outs):
            env[v] = x
    return [read(v) for v in jaxpr.outvars]


def dce(fn):
    """Trace `fn` to a jaxpr, drop dead equations, re-emit the rest.

    capture() lowers with keep_unused=True, which also skips JAX's equation
    DCE, so code whose results reach no output would stay in the graph (XLA
    deletes it at compile time; a plain jax.jit lowering drops it too). In
    GraphCast that is: the mesh2grid mesh-node update (DEAD_RE), the `sent`
    segment sums typed_graph_net._node_update computes for every
    sender node set (include_sent_messages_in_node_update=False drops them),
    the latents of node/edge sets the next stage replaces, and the
    autoregressive scan's next-inputs carry (concat + tail) that the 1-step
    rollout discards. Name stacks are set to haiku module paths.

    Two passes: the scan's DCE rule must keep a carry whose input the body
    reads (the loop needs it), so the next-inputs carry only becomes dead once
    the 1-step scan is expanded; the second pass removes it."""
    def _dce(f, leaves):
        closed = jax.make_jaxpr(f)(*leaves)
        jaxpr, _ = pe.dce_jaxpr(closed.jaxpr, [True] * len(closed.jaxpr.outvars),
                                instantiate=True)
        return jaxpr, closed.consts

    def wrapped(*args):
        leaves, out_shape = jax.tree.leaves(args), jax.eval_shape(fn, *args)
        j1, c1 = _dce(lambda *a: jax.tree.leaves(fn(*jax.tree.unflatten(
            jax.tree.structure(args), a))), leaves)
        j2, c2 = _dce(lambda *a: _eval(j1, c1, a), leaves)
        out = _eval(j2, c2, leaves)
        return jax.tree.unflatten(jax.tree.structure(out_shape), out)
    return wrapped


@functools.lru_cache(maxsize=4)
def full_param_spec(cfg: Config):
    """Haiku parameter tree of the canonical stack (dead modules included)."""
    inputs, forcings, stats, graph = input_spec(cfg)

    def init(inputs, forcings, stats, graph):
        with net.static_graph(graph):
            return transformed(cfg, stats).init(
                jax.random.PRNGKey(0), *net.datasets(cfg, inputs, forcings))[0]
    return jax.eval_shape(init, inputs, forcings, stats, graph)


def split_dead(params):
    live = {k: v for k, v in params.items() if not DEAD_RE.match(k)}
    dead = {k: v for k, v in params.items() if DEAD_RE.match(k)}
    return live, dead


def with_dead_params(cfg: Config, params):
    """Re-insert the dead mesh2grid mesh-node update parameters as zero
    constants of their own dtypes (fp8 `w`, fp32 scales / bias / LayerNorm):
    haiku's apply needs every parameter it creates; their
    consumers reach no output and are removed by `dce`."""
    _, dead = split_dead(full_param_spec(cfg))
    zeros = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), dead)
    return {**params, **zeros}


def transformed(cfg: Config, stats: dict):
    return net.make_fn(cfg, net.stats_datasets(stats))


def apply(cfg: Config, params, inputs, forcings, stats, graph):
    """Patched stack: static graph inputs, unrolled 1-step scan."""
    with net.static_graph(graph), net.unrolled():
        preds, _ = transformed(cfg, stats).apply(
            with_dead_params(cfg, params), {}, None, *net.datasets(cfg, inputs, forcings))
    return net.prediction_arrays(preds)


def make_forward(cfg: Config):
    def forward(params, inputs, forcings, stats, graph):
        with hk.intercept_methods(_module_scope):
            return apply(cfg, params, inputs, forcings, stats, graph)
    return dce(forward)


def init_variables(cfg: Config, rng):
    del rng           # the parameter tree (shapes) is what matters
    live, _ = split_dead(full_param_spec(cfg))
    return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), live)


# ----------------------------------------------------------------------------
# Per-module activations
# ----------------------------------------------------------------------------

def record_module_outputs(fn) -> dict:
    """{"<haiku module>#<call k><leaf path>:<shape>": array} for every module
    __call__ output (floating jax arrays only), in execution order. Runs
    op-by-op under jax.disable_jit(), so the autoregressive lax.scan body
    executes as a Python loop and every call is concrete."""
    counts, order, acts = Counter(), [], {}

    def interceptor(next_f, args, kwargs, context):
        out = next_f(*args, **kwargs)
        if context.method_name != "__call__":
            return out
        name = context.module.module_name
        leaves = [(p, x) for p, x in jax.tree_util.tree_flatten_with_path(out)[0]
                  if isinstance(x, jax.Array) and not isinstance(x, jax_core.Tracer)
                  and jnp.issubdtype(x.dtype, jnp.floating)]
        if not leaves:
            return out
        k = counts[name]
        counts[name] += 1
        for p, x in leaves:
            key = f"{name}#{k}{jax.tree_util.keystr(p)}:{'x'.join(map(str, x.shape))}"
            acts[key] = x
            order.append(key)
        return out

    with jax.disable_jit(), hk.intercept_methods(interceptor):
        fn()
    return {k: np.asarray(acts[k].astype(jnp.float32)) for k in order}


def intermediates(cfg: Config, variables, inputs, forcings, stats, graph) -> dict:
    return record_module_outputs(
        lambda: apply(cfg, variables, inputs, forcings, stats, graph))


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------

def build(cfg: Config) -> ModelSpec:
    params = jax.eval_shape(lambda: init_variables(cfg, None))
    inputs = input_spec(cfg)
    lat, lon = net.grid(cfg)
    g = graph_inputs(cfg)
    return ModelSpec(
        name="graphcast",
        variant=(f"GraphCast_operational (ERA5-HRES, precipitation output only), FP8 E4M3 "
                 f"W8A8 per-tensor MLP linears (GPU fallback), "
                 f"{'bf16' if cfg.bfloat16 else 'fp32'} activations, {cfg.resolution} deg grid "
                 f"{len(lat)}x{len(lon)}, {len(net.TASK.pressure_levels)} levels, "
                 f"mesh 0..{cfg.mesh_size} ({g['grid2mesh']['mesh_node_features'].shape[0]} "
                 f"nodes), latent {cfg.latent_size}, {cfg.gnn_msg_steps} processor steps, "
                 f"edges g2m/mesh/m2g = {g['grid2mesh']['senders'].shape[0]}/"
                 f"{g['mesh']['senders'].shape[0]}/{g['mesh2grid']['senders'].shape[0]}, "
                 f"batch {cfg.batch}"),
        phases=[Phase(PHASE, make_forward(cfg), inputs, trip_count=cfg.rollout_steps,
                      shapes=input_shapes(cfg),
                      note="one 6 h autoregressive step (rollout.chunked_prediction "
                           "chunk of 1); runs rollout_steps times, the predictions "
                           "shifted into the 2-step input window between calls")],
        param_specs=as_spec(params),
        # The grid sizes are all functions of the grid spacing `resolution` (the alt build
        # re-derives the grid and both grid graphs at 4 deg).
        symbols={"B": Sym("batch", 2, "forecasts (initial conditions) per call"),
                 "H": Sym("resolution", 4.0, "latitude points (180 / resolution + 1)"),
                 "W": Sym("resolution", 4.0, "longitude points (360 / resolution)"),
                 "G": Sym("resolution", 4.0, "grid nodes (H * W)"),
                 "Eg": Sym("resolution", 4.0, "grid2mesh edges (radius query, grid x mesh)"),
                 "Em": Sym("resolution", 4.0, "mesh2grid edges (3 per grid node: its "
                                              "containing mesh triangle)")},
        # The flattened grid ([B, T, H, W] -> [H*W, B, C]) meets the grid node features [G, 3].
        constraints=("H*W == G",),
        excluded=[
            "data pipeline: ERA5/HRES loading, data_utils.extract_inputs_targets_forcings, "
            "year/day progress features (data_utils.add_derived_vars) and TOA incident solar "
            "radiation (solar_radiation) for the target time, computed host-side in numpy",
            "graph construction (icosahedral multi-mesh, grid2mesh radius query, mesh2grid "
            "containing-triangle query, structural node/edge features): host-side numpy at "
            "model construction; fed as static inputs (`graph`)",
            "normalization statistics (mean/stddev/diffs_stddev_by_level .nc files): data, "
            "fed as static inputs (`stats`)",
            "rollout loop: rollout.chunked_prediction's Python loop (trip count = "
            "rollout_steps) and its rollout._get_next_inputs (xarray concat + tail of the "
            "input window, host-side data movement)",
            "losses (loss / loss_and_predictions), training",
        ],
    )

