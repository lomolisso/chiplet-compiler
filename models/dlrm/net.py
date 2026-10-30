"""MLPerf DLRMv2 (DLRM-DCNv2, Criteo 1TB multi-hot) on Keras 3 / JAX, INT8.

The network is the keras-rs MLPerf example model
(`examples/ml_perf/model.py`, vendored at `upstream/model.py`, class `DLRMDCNV2`, built on
`DistributedEmbedding` and `FeatureCross`). Quantization is written into the vendored
sources (edits marked `# quantized:`): `upstream/{base_distributed_embedding,embed_reduce,
feature_cross}.py` are copies of the keras-rs v0.4.0 layers edited to INT8, and
`upstream/quant_layers.py` holds the INT8 Dense. The recipe is MLPerf NVIDIA dlrm-v2-99
(inference_results_v4.1 closed/NVIDIA/code/dlrm-v2/tensorrt/builder.py, calibrator.cache):
  * 26 embedding tables int8, one symmetric scale per table (max|table| / 127); gathered
    rows dequantized and sum-pooled in fp32;
  * bottom MLP, top MLP and final linear INT8 W8A8 with per-tensor weight scales (TRT
    matrix-multiply layers); DCN V / W INT8 with per-output-channel weight scales (TRT 1x1
    convs); activations per-tensor static (entropy calibration); int32 accumulation, fp32
    epilogue (scale, bias, ReLU), crossnet elementwise ops in fp32;
  * sigmoid in fp32.
Every GEMM is i8 x i8 -> i32. This file only

  * builds that model with the MLPerf-inference config, with abstract
    (never initialized) Keras variables, so the ~26B-parameter tables are
    never materialized;
  * exposes it as a pure function `forward(cfg, params, dense, sparse)` via
    `keras.Model.stateless_call`, with params in a pytree whose paths are
    derived from the model structure (not from Keras' global auto-names).

All 26 tables use `placement="default_device"`, so `DistributedEmbedding`
dispatches to `keras_rs.layers.EmbedReduce` (keras `Embedding` -> `jnp.take`,
then a sum over the multi-hot axis): plain gather + reduce, no SparseCore ops.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import keras_rs  # noqa: E402
from keras.src.backend.common import global_state  # noqa: E402

if keras.backend.backend() != "jax":
    raise RuntimeError("models.dlrm needs KERAS_BACKEND=jax")

MLPERF_MODEL_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream",
                               "model.py")


# examples/ml_perf is not a package: load model.py by path.
_spec = importlib.util.spec_from_file_location("keras_rs_ml_perf_model", MLPERF_MODEL_PY)
mlperf_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlperf_model)

# MLPerf inference dlrm_v2 (main.py, dataset "multihot-criteo").
VOCAB_SIZES = (40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000, 3067956,
               405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000, 40000000, 40000000,
               590152, 12973, 108, 36)
# Multi-hot sizes of the MLPerf multi-hot Criteo dataset (tools/make_fake_criteo.sh).
MULTI_HOT = (3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1)
SEED = 1337  # keras-rs examples/ml_perf/main.py


@dataclass(frozen=True)
class Config:
    batch: int = 1
    batch_variant: int = 128             # second captured phase (same network, larger batch)
    vocab_sizes: tuple[int, ...] = VOCAB_SIZES
    multi_hot: tuple[int, ...] = MULTI_HOT
    dense_in: int = 13
    embedding_dim: int = 128
    bottom_mlp: tuple[int, ...] = (512, 256, 128)
    top_mlp: tuple[int, ...] = (1024, 1024, 512, 256, 1)
    dcn_layers: int = 3
    dcn_low_rank: int = 512
    dtype: str = "float32"


def feature_key(i: int) -> str:
    return f"{i}_id"                     # keras-rs ml_perf naming: f"{new_name}_id"


@contextlib.contextmanager
def abstract_keras():
    """Build/run Keras objects whose variables are never initialized.

    Variables created inside a StatelessScope are deferred; the outermost
    scope normally initializes them on exit (which would allocate ~100 GB
    here). `initialize_variables=False` plus restoring the global registry
    keeps them abstract forever. Name counters are reset so auto-names are
    deterministic; both globals are restored afterwards.
    """
    saved = {k: global_state.get_global_attribute(k)
             for k in ("uninitialized_variables", "object_name_uids")}
    global_state.set_global_attribute("uninitialized_variables", [])
    global_state.set_global_attribute("object_name_uids", None)
    try:
        with keras.StatelessScope(initialize_variables=False):
            yield
    finally:
        for k, v in saved.items():
            global_state.set_global_attribute(k, v)


def build_keras_model(cfg: Config, batch: int):
    """keras-rs `DLRMDCNV2` exactly as examples/ml_perf/main.py builds it,
    except that every table is placed on "default_device" (see module doc)."""
    feature_configs = {}
    for i, (vocab, hot) in enumerate(zip(cfg.vocab_sizes, cfg.multi_hot)):
        table = keras_rs.layers.TableConfig(
            name=str(i), vocabulary_size=vocab, embedding_dim=cfg.embedding_dim,
            initializer=keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_in", distribution="uniform", seed=SEED),
            combiner="sum", placement="default_device")
        feature_configs[feature_key(i)] = keras_rs.layers.FeatureConfig(
            name=str(i), table=table, input_shape=(batch, hot),
            output_shape=(batch, cfg.embedding_dim))
    model = mlperf_model.DLRMDCNV2(
        large_emb_feature_configs=feature_configs, small_emb_features=[],
        embedding_dim=cfg.embedding_dim, bottom_mlp_dims=list(cfg.bottom_mlp),
        top_mlp_dims=list(cfg.top_mlp), num_dcn_layers=cfg.dcn_layers,
        dcn_projection_dim=cfg.dcn_low_rank, seed=SEED, dtype=cfg.dtype, name="dlrm_dcn_v2")
    model(keras_inputs(cfg, batch, symbolic=True))      # build (shape inference only)
    return model


def keras_inputs(cfg: Config, batch: int, dense=None, sparse=None, symbolic=False):
    if symbolic:
        dense = keras.KerasTensor((batch, cfg.dense_in), cfg.dtype)
        sparse = [keras.KerasTensor((batch, h), "int32") for h in cfg.multi_hot]
    return {"dense_input": dense,
            "large_emb_inputs": {feature_key(i): x for i, x in enumerate(sparse)}}


def named_layers(model) -> dict[str, keras.layers.Layer]:
    """Stable structural names for every sub-layer that owns weights or whose
    output is compared against the reference."""
    out = {"bottom_mlp": model.bottom_mlp}
    for i, layer in enumerate(model.bottom_mlp.layers):
        out[f"bottom_mlp/dense_{i}"] = layer
    out["embedding_layer"] = model.embedding_layer
    tables = model.embedding_layer._default_device_embedding_layers
    for i in range(len(tables)):
        out[f"embedding_layer/table_{i}"] = tables[feature_key(i)]
    out["dcn_block"] = model.dcn_block
    for i, fc in enumerate(model.dcn_block.layers):
        out[f"dcn_block/feature_cross_{i}"] = fc
        out[f"dcn_block/feature_cross_{i}/down_proj_dense"] = fc.down_proj_dense
        out[f"dcn_block/feature_cross_{i}/dense"] = fc.dense
    out["top_mlp"] = model.top_mlp
    for i, layer in enumerate(model.top_mlp.layers):
        out[f"top_mlp/dense_{i}"] = layer
    return out


def named_variables(model) -> dict[str, keras.Variable]:
    """'<layer path>/<weight name>' -> Variable; must cover every model weight."""
    out = {}
    for name, layer in named_layers(model).items():
        for attr in ("kernel", "kernel_scale", "input_scale", "bias", "embeddings",
                     "embeddings_scale"):
            v = getattr(layer, attr, None)
            if isinstance(v, keras.Variable):
                out[f"{name}/{attr}"] = v
    if {id(v) for v in out.values()} != {id(v) for v in model.weights} or \
            len(out) != len(model.weights):
        raise AssertionError("named_variables does not cover the Keras model weights")
    return out


def _lookup(tree: dict, path: str):
    for k in path.split("/"):
        tree = tree[k]
    return tree


def forward(cfg: Config, params, dense, sparse, record=None):
    """Pure fn: params pytree + dense [B, 13] f32 + 26 x [B, hot_i] int32 -> CTR [B, 1].

    `record`: optional dict; if given, the output of every named layer is
    stored in it (for per-module activation checks).
    """
    batch = dense.shape[0]
    with abstract_keras():
        model = build_keras_model(cfg, batch)
        by_id = {id(v): p for p, v in named_variables(model).items()}
        values = [_lookup(params, by_id[id(v)]) for v in model.trainable_variables]
        # int8 kernels / tables and the scales are non-trainable Keras variables
        frozen = [_lookup(params, by_id[id(v)]) for v in model.non_trainable_variables]
        if record is not None:
            for name, layer in named_layers(model).items():
                _record_outputs(layer, name, record)
        out, _ = model.stateless_call(values, frozen, keras_inputs(cfg, batch, dense, sparse))
    return out


def _record_outputs(layer, name, record):
    call = layer.call

    def wrapped(*args, **kwargs):
        y = call(*args, **kwargs)
        record[name] = y
        return y

    layer.call = wrapped
