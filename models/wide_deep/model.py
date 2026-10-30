"""Wide & Deep (NVIDIA DeepLearningExamples TF2, Outbrain) capture spec, recsys INT8 fallback
(INT8 W8A8 deep MLP, FBGEMM row-wise uint8 deep tables, fp16 wide part; see net.py)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from models.spec import ModelSpec, Phase, as_spec
from models.wide_deep.net import Config, WideDeepModel


def _reduced() -> Config:
    full = Config()
    # Every feature (16 categorical incl. 3 multi-hot, 13 numerical) and all 5+1
    # Dense layers are kept; only cardinalities, embedding dims and widths shrink.
    return dataclasses.replace(
        full, batch=1, eval_batch=16,
        onehot=tuple((k, min(n, 37 + 3 * i)) for i, (k, n) in enumerate(full.onehot)),
        multihot=tuple((k, min(n, 41 + 5 * i), h) for i, (k, n, h) in enumerate(full.multihot)),
        embedding_dims=tuple((k, max(2, d // 8)) for k, d in full.embedding_dims),
        deep_hidden_units=(64, 48, 64, 32, 64),
    )


CONFIGS = {"full": Config(), "reduced": _reduced()}

PHASES = (("forward", "batch"), ("eval", "eval_batch"))


def input_spec(cfg: Config, batch: int) -> dict:
    spec = {k: jax.ShapeDtypeStruct((batch, 1), jnp.int32) for k, _ in cfg.onehot}
    spec.update({k: jax.ShapeDtypeStruct((batch, h), jnp.int32) for k, _, h in cfg.multihot})
    spec.update({k: jax.ShapeDtypeStruct((batch, 1), jnp.float32) for k in cfg.numeric})
    return spec


def init_variables(cfg: Config, rng):
    x = {k: jnp.zeros(s.shape, s.dtype) for k, s in input_spec(cfg, 1).items()}
    return WideDeepModel(cfg).init(rng, x)


def _features(cfg: Config, batch: int, rng) -> dict:
    """Valid categorified ids; multi-hot rows padded with -1 (dataloader pad_batch)."""
    seed = int(jax.random.randint(rng, (), 0, 2**31 - 1))
    g = np.random.default_rng(seed + batch)
    out = {}
    for k, n in cfg.onehot:
        out[k] = jnp.asarray(g.integers(0, n, (batch, 1)), jnp.int32)
    for k, n, h in cfg.multihot:
        ids = g.integers(0, n, (batch, h))
        nnz = g.integers(0, h + 1, (batch,))           # includes empty rows
        nnz[0] = h
        ids = np.where(np.arange(h)[None, :] < nnz[:, None], ids, -1)
        out[k] = jnp.asarray(ids, jnp.int32)
    for k in cfg.numeric:
        out[k] = jnp.asarray(g.random((batch, 1)), jnp.float32)
    return out


def example_inputs(cfg: Config, rng):
    return {ph: (_features(cfg, getattr(cfg, attr), rng),) for ph, attr in PHASES}


def intermediates(cfg: Config, variables, x) -> dict:
    _, state = WideDeepModel(cfg).apply(variables, x, capture_intermediates=True,
                                        mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        flat["/".join(k for k in keys if k != "__call__")] = v[0]
    return flat


def build(cfg: Config) -> ModelSpec:
    module = WideDeepModel(cfg)

    def forward(variables, features):
        return module.apply(variables, features)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    n_cat = len(cfg.onehot) + len(cfg.multihot)
    return ModelSpec(
        name="wide_deep",
        variant=(f"Wide & Deep (NVIDIA DLE TF2, Outbrain spec): {n_cat} categorical + "
                 f"{len(cfg.numeric)} numerical features, deep MLP {list(cfg.deep_hidden_units)}, "
                 f"batch {cfg.batch} (eval phase batch {cfg.eval_batch}); recsys INT8 fallback: "
                 "W8A8 deep MLP (per-channel W, per-tensor static A), row-wise uint8 deep "
                 f"tables (fp16 scale + bias per row), {cfg.dtype} wide part, fp32 sigmoid"),
        phases=[Phase("forward", forward, (input_spec(cfg, cfg.batch),),
                      note="one inference call, batch 1"),
                Phase("eval", forward, (input_spec(cfg, cfg.eval_batch),),
                      note=f"same network at NVIDIA's documented --eval_batch_size "
                           f"{cfg.eval_batch} (alternative to 'forward', not an extra step)")],
        param_specs=as_spec(variables),
        excluded=[
            "NVTabular preprocessing: hash-bucketing / categorify of string ids into "
            "[0, cardinality) integer ids, multi-hot list padding with -1, numeric "
            "normalization, CTR / similarity / days-since feature engineering",
            "tf.feature_column parsing (identity/numeric columns only; no ops at inference)",
            "dataloader (parquet reading, batching)",
            "MAP@12 / AUC metrics and display_id grouping (postprocessing)",
        ],
    )
