"""MLPerf DLRMv2 (DLRM-DCNv2) INT8 capture spec (MLPerf NVIDIA dlrm-v2-99 recipe; network
definition lives in net.py and the edited vendored sources in upstream/)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from models.spec import ModelSpec, Phase, Sym
from models.dlrm import net
from models.dlrm.net import Config

CONFIGS = {
    "full": Config(),
    # Every block type kept (26 multi-hot tables incl. 100-hot, 3-layer bottom
    # MLP, low-rank DCN, ReLU top MLP + sigmoid head); vocab/width/depth shrunk.
    "reduced": Config(batch=2, batch_variant=3,
                      vocab_sizes=tuple(min(n, 97) for n in net.VOCAB_SIZES),
                      embedding_dim=16, bottom_mlp=(32, 24, 16), top_mlp=(48, 32, 1),
                      dcn_layers=2, dcn_low_rank=8),
}


def phase_batches(cfg: Config) -> dict[str, int]:
    return {"forward": cfg.batch, f"forward_b{cfg.batch_variant}": cfg.batch_variant}


def input_spec(cfg: Config, batch: int):
    return (jax.ShapeDtypeStruct((batch, cfg.dense_in), jnp.dtype(cfg.dtype)),
            tuple(jax.ShapeDtypeStruct((batch, h), jnp.int32) for h in cfg.multi_hot))


def param_specs(cfg: Config) -> dict:
    """Nested {layer: {weight: ShapeDtypeStruct}} of the (never initialized)
    Keras model's weights; paths come from `net.named_variables`."""
    tree: dict = {}
    with net.abstract_keras():
        model = net.build_keras_model(cfg, cfg.batch)
        for path, v in net.named_variables(model).items():
            *mods, leaf = path.split("/")
            node = tree
            for m in mods:
                node = node.setdefault(m, {})
            node[leaf] = jax.ShapeDtypeStruct(tuple(v.shape), jnp.dtype(v.dtype))
    return tree


def init_variables(cfg: Config, rng):
    """Random weights (only sensible for the reduced config: full is ~26 GB of int8):
    int8 codes for kernels / tables, small positive scales, N(0, 0.1) biases."""
    specs = param_specs(cfg)
    flat, treedef = jax.tree_util.tree_flatten_with_path(specs)
    keys = jax.random.split(rng, len(flat))

    def draw(k, path, s):
        if s.dtype == jnp.int8:
            return jax.random.randint(k, s.shape, -127, 128, jnp.int32).astype(jnp.int8)
        if str(path[-1].key).endswith("_scale"):
            return jnp.full(s.shape, 0.02, s.dtype)
        return 0.1 * jax.random.normal(k, s.shape, s.dtype)

    return jax.tree.unflatten(treedef, [draw(k, p, s) for k, (p, s) in zip(keys, flat)])


def example_inputs(cfg: Config, rng):
    out = {}
    for name, batch in phase_batches(cfg).items():
        rng, kd, *ks = jax.random.split(rng, 2 + len(cfg.multi_hot))
        dense = jax.random.uniform(kd, (batch, cfg.dense_in), jnp.dtype(cfg.dtype), 0.0, 5.0)
        sparse = tuple(jax.random.randint(k, (batch, h), 0, n, jnp.int32)
                       for k, n, h in zip(ks, cfg.vocab_sizes, cfg.multi_hot))
        out[name] = (dense, sparse)
    return out


def intermediates(cfg: Config, variables, dense, sparse) -> dict:
    rec: dict = {}
    net.forward(cfg, variables, dense, sparse, record=rec)
    return {k: np.asarray(v) for k, v in rec.items() if not isinstance(v, dict)}


def build(cfg: Config) -> ModelSpec:
    def make_fn():
        def fn(params, dense, sparse):
            return net.forward(cfg, params, dense, sparse)
        return fn

    phases = []
    for name, batch in phase_batches(cfg).items():
        note = ("single network invocation (one CTR prediction batch)" if name == "forward"
                else f"same network as 'forward' at batch {batch} (batch variant, "
                     "not an extra invocation)")
        phases.append(Phase(name, make_fn(), input_spec(cfg, batch), trip_count=1, note=note,
                            shapes=("B, _", "B, _"),
                            size_variant_of=None if name == "forward" else "forward"))
    n_tab = len(cfg.vocab_sizes)
    return ModelSpec(
        name="dlrm",
        variant=(f"MLPerf DLRMv2 (DLRM-DCNv2), Criteo 1TB multi-hot, {n_tab} tables x "
                 f"dim {cfg.embedding_dim}, batch {cfg.batch} (+ batch {cfg.batch_variant}), "
                 "INT8 (TensorRT, MLPerf NVIDIA dlrm-v2-99): int8 tables with per-table "
                 "scales, W8A8 MLPs (per-tensor W) and DCN (per-channel W), per-tensor static "
                 "activations, fp32 pooling / crossnet elementwise / sigmoid"),
        phases=phases,
        param_specs=param_specs(cfg),
        # dense [B, 13] and 26 x [B, hot_i]: the multi-hot sizes are fixed by the MLPerf
        # multi-hot Criteo dataset, so only the batch is a symbol (the batch variant binds B
        # to batch_variant).
        symbols={"B": Sym("batch", 5, "samples per call")},
        excluded=["Criteo preprocessing: log(x+1) of dense features, id hashing/modulo "
                  "(dataset side)",
                  "reshape of the CTR output to [B] and LoadGen accuracy/AUC postprocessing"],
    )

