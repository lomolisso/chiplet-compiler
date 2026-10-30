"""Wide & Deep (Cheng et al. 2016), Flax linen port of NVIDIA DeepLearningExamples
TensorFlow2/Recommendation/WideAndDeep (Outbrain feature spec).

Structure and naming follow the NVIDIA model-construction code one-to-one:
  trainer/model/widedeep.py  get_feature_columns / wide_deep_model
  trainer/model/layers.py    LinearFeatures ("wide_linear"), DenseFeatures
                             ("deep_embedded"), _categorical_embedding_lookup
  tf.keras.experimental.WideDeepModel(linear, dnn, activation="sigmoid")
                             -> sigmoid(linear_output + dnn_output)

Feature columns are re-expressed as plain tuples. NVIDIA's layers sort their
columns by `feature_column.name` (`<key>` for identity/numeric columns,
`<key>_embedding` for embedding columns), which fixes the concat order of the
deep input and the summation order of the wide part; `_sort_columns` below
reproduces that exactly.

Inputs are already-categorified integer ids (NVTabular preprocessing): one-hot features int32 [B, 1], multi-hot int32 [B, hotness]
padded with -1, numerical float32 [B, 1].

Quantization: recsys fallback (NVIDIA deploys this model only in FP16: Triton TF-SavedModel
auto-FP16 or TensorRT FP16, triton/README.md; no INT8 path):
  * deep MLP (dense .. dense_5): INT8 W8A8, weights per-output-channel symmetric,
    activations per-tensor static symmetric, i8 x i8 -> i32, fp32 epilogue (scale, bias,
    ReLU) (`q.INT8_STATIC`);
  * deep embedding tables: FBGEMM row-wise INT8 (inference TBE SparseType.INT8 /
    TorchRec quantized embeddings): uint8 codes with an fp16 scale and bias per row,
    x = q * scale + bias, rows dequantized to fp16 (`q.RowwiseQEmbed`);
  * fp16: the wide part ([card, 1] tables, numerics @ W[13, 1], bias; row-wise uint8 would
    spend 4 bytes of scale / bias on 1 byte of data), the multi-hot masked sum pooling and
    the deep-input concat;
  * fp32: sigmoid(wide + deep).
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

from models import quant as q

# --- data/outbrain/features.py (NVIDIA), verified against it in tf_worker.py ---
NUMERIC_COLUMNS = (
    "document_id_document_id_promo_sim_categories",
    "document_id_document_id_promo_sim_topics",
    "document_id_document_id_promo_sim_entities",
    "document_id_promo_ctr",
    "publisher_id_promo_ctr",
    "source_id_promo_ctr",
    "document_id_promo_count",
    "publish_time_days_since_published",
    "ad_id_ctr",
    "advertiser_id_ctr",
    "campaign_id_ctr",
    "ad_id_count",
    "publish_time_promo_days_since_published",
)
ONEHOT_COLUMNS = (
    "ad_id", "document_id", "platform", "document_id_promo", "campaign_id",
    "advertiser_id", "source_id", "geo_location", "geo_location_country",
    "geo_location_state", "publisher_id", "source_id_promo", "publisher_id_promo",
)
MULTIHOT_COLUMNS = (("topic_id_list", 3), ("entity_id_list", 3), ("category_id_list", 3))
HASH_BUCKET_SIZES = {
    "document_id": 300000, "ad_id": 250000, "document_id_promo": 100000,
    "source_id_promo": 4000, "source_id": 4000, "geo_location": 2500,
    "advertiser_id": 2500, "geo_location_state": 2000, "publisher_id_promo": 1000,
    "publisher_id": 1000, "geo_location_country": 300, "platform": 4,
    "campaign_id": 5000, "topic_id_list": 350, "entity_id_list": 10000,
    "category_id_list": 100,
}
EMBEDDING_DIMENSIONS = {
    "document_id": 128, "ad_id": 128, "document_id_promo": 128, "source_id_promo": 64,
    "source_id": 64, "geo_location": 64, "advertiser_id": 64, "geo_location_state": 64,
    "publisher_id_promo": 64, "publisher_id": 64, "geo_location_country": 64,
    "platform": 19, "campaign_id": 128, "topic_id_list": 64, "entity_id_list": 64,
    "category_id_list": 64,
}


@dataclass(frozen=True)
class Config:
    batch: int = 1
    eval_batch: int = 131072                          # --eval_batch_size default
    numeric: tuple[str, ...] = NUMERIC_COLUMNS
    onehot: tuple[tuple[str, int], ...] = tuple((k, HASH_BUCKET_SIZES[k]) for k in ONEHOT_COLUMNS)
    multihot: tuple[tuple[str, int, int], ...] = tuple(
        (k, HASH_BUCKET_SIZES[k], h) for k, h in MULTIHOT_COLUMNS)  # (name, cardinality, hotness)
    embedding_dims: tuple[tuple[str, int], ...] = tuple(EMBEDDING_DIMENSIONS.items())
    deep_hidden_units: tuple[int, ...] = (1024, 1024, 1024, 1024, 1024)
    deep_dropout: float = 0.1
    combiner: str = "sum"
    dtype: str = "float16"          # non-quantized layers (wide part, pooling, concat)


# --- feature columns (tf.feature_column stand-ins) ------------------------------
# ("numeric", name, key) | ("identity", name, key, num_buckets)
# | ("embedding", name, key, num_buckets, dimension)

def get_feature_columns(cfg: Config):
    wide_columns, deep_columns = [], []
    dims = dict(cfg.embedding_dims)
    categorical = [(k, n) for k, n in cfg.onehot] + [(k, n) for k, n, _ in cfg.multihot]
    for key, num_buckets in categorical:
        wide_columns.append(("identity", key, key, num_buckets))
        deep_columns.append(("embedding", f"{key}_embedding", key, num_buckets, dims[key]))
    numerics = [("numeric", key, key) for key in cfg.numeric]
    wide_columns.extend(numerics)
    deep_columns.extend(numerics)
    return wide_columns, deep_columns


def _sort_columns(feature_columns):
    return sorted(feature_columns, key=lambda col: col[1])


def _categorical_embedding_lookup(table, inputs, feature_name, combiner):
    """`table` is a [card, dim] array or a module ids -> rows (row-wise INT8 tables)."""
    # gather mode="clip": XLA gather semantics (out-of-range ids clamp), which is
    # what TF's GatherV2 lowers to under XLA; the -1 padding rows are masked below.
    if isinstance(table, q.RowwiseQEmbed):
        lookup = lambda ids: table(jnp.clip(ids, 0, table.num_embeddings - 1))
    else:
        lookup = lambda ids: jnp.take(table, ids, axis=0, mode="clip")
    if inputs[feature_name].shape[1] > 1:
        # Multi-hot embedding lookup
        x = inputs[feature_name]
        embeddings = lookup(x)
        # Remove padded values (inverse of dataloader pad_batch, pad = -1)
        mask_array = (x >= 0).astype(embeddings.dtype)
        mask = jnp.expand_dims(mask_array, -1)
        embeddings = embeddings * mask
        # Sum aggregation
        embeddings = jnp.sum(embeddings, axis=1)
        # Divide by number of not zeros if mean aggregation
        if combiner == "mean":
            row_lengths = jnp.expand_dims(jnp.sum(mask_array, axis=1), -1)
            safe = jnp.where(row_lengths == 0, 1.0, row_lengths)          # tf.math.divide_no_nan
            embeddings = jnp.where(row_lengths == 0, 0.0, embeddings / safe)
    else:
        embeddings = lookup(inputs[feature_name][:, 0])
    return embeddings


class DenseFeatures(nn.Module):
    """trainer/model/layers.py::DenseFeatures (aggregation="concat"); row-wise INT8 tables
    (`<key>/embedding` uint8, `<key>/scale_bias` fp16), fp16 pooling and concat."""
    feature_columns: tuple
    combiner: str = "sum"
    dtype: str = "float16"

    @nn.compact
    def __call__(self, inputs):
        features = []
        for col in _sort_columns(self.feature_columns):
            if col[0] == "numeric":
                # _handle_continuous_feature, shape (1,)
                features.append(inputs[col[2]].astype(self.dtype))
            else:
                _, _, key, num_buckets, dimension = col
                table = q.RowwiseQEmbed(num_buckets, dimension, dtype=self.dtype, name=key)
                features.append(_categorical_embedding_lookup(table, inputs, key, self.combiner))
        return jnp.concatenate(features, axis=1)


class LinearFeatures(nn.Module):
    """trainer/model/layers.py::LinearFeatures (the wide part), fp16."""
    feature_columns: tuple
    dtype: str = "float16"

    @nn.compact
    def __call__(self, inputs):
        cols = _sort_columns(self.feature_columns)
        tables = {}
        numeric_kernel_dim = 0
        for col in cols:                                    # build()
            if col[0] == "numeric":
                numeric_kernel_dim += 1
                continue
            tables[col[2]] = self.param(col[2], nn.initializers.zeros, (col[3], 1), self.dtype)
        if numeric_kernel_dim > 0:
            tables["numeric"] = self.param("numeric", nn.initializers.zeros,
                                           (numeric_kernel_dim, 1), self.dtype)
        bias = self.param("bias", nn.initializers.zeros, (1,), self.dtype)

        x = bias                                            # call()
        numeric_inputs = []
        for col in cols:
            if col[0] == "numeric":
                numeric_inputs.append(inputs[col[2]].astype(self.dtype))
            else:
                x = x + _categorical_embedding_lookup(tables[col[2]], inputs, col[2], "sum")
        if len(numeric_inputs) > 0:
            numerics = jnp.concatenate(numeric_inputs, axis=1)
            x = x + numerics @ tables["numeric"]
        return x


def dense(units: int, name: str) -> q.QDense:
    """tf.keras.layers.Dense(units) as an INT8 W8A8 GEMM: input quantized with its calibrated
    per-tensor scale (`input_scale`), int8 kernel with per-output-channel `kernel_scale`,
    i8 x i8 -> i32, fp32 epilogue (scales + bias); Keras parameter names `kernel` / `bias`.
    The ReLU follows in fp32."""
    return q.QDense(units, q.INT8_STATIC, dtype="float32", name=name)


def _keras_name(base: str, i: int) -> str:
    return base if i == 0 else f"{base}_{i}"


class WideDeepModel(nn.Module):
    """widedeep.py::wide_deep_model + tf.keras.experimental.WideDeepModel."""
    cfg: Config

    @nn.compact
    def __call__(self, features):
        c = self.cfg
        wide_columns, deep_columns = get_feature_columns(c)
        # linear_model
        linear_output = LinearFeatures(tuple(wide_columns), c.dtype, name="wide_linear")(features)
        # dnn_model
        dnn = DenseFeatures(tuple(deep_columns), c.combiner, c.dtype,
                            name="deep_embedded")(features)
        for i, unit_size in enumerate(c.deep_hidden_units):
            dnn = nn.relu(dense(unit_size, _keras_name("dense", i))(dnn))
            dnn = nn.Dropout(rate=c.deep_dropout, deterministic=True,
                             name=_keras_name("dropout", i))(dnn)
        dnn_output = dense(1, _keras_name("dense", len(c.deep_hidden_units)))(dnn)
        # WideDeepModel.call: activation(linear_output + dnn_output), in fp32
        return nn.sigmoid(linear_output.astype(jnp.float32) + dnn_output)
