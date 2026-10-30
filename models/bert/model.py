"""BERT-Large SQuAD v1.1 (MLPerf Inference bert-99) INT8 capture spec (MLPerf NVIDIA TensorRT
recipe, see net.py and recipe.py); network in net.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from models.spec import ModelSpec, Phase, as_spec
from models.bert.net import BertForQuestionAnswering, Config

# MLPerf NVIDIA INT8 engine: INT8 GEMMs / attention, fp32 LayerNorm / softmax / GELU math.
CONFIGS = {
    "full": Config(),
    "reduced": Config(seq_len=32, vocab_size=128, hidden_size=64, num_hidden_layers=2,
                      num_attention_heads=4, intermediate_size=128, max_position_embeddings=64),
}

INPUT_NAMES = ("input_ids", "token_type_ids", "attention_mask")


def input_spec(cfg: Config):
    s = jax.ShapeDtypeStruct((cfg.batch, cfg.seq_len), jnp.int32)
    return (s, s, s)


def init_variables(cfg: Config, rng):
    return BertForQuestionAnswering(cfg).init(rng, *[jnp.zeros(s.shape, s.dtype)
                                                     for s in input_spec(cfg)])


def example_inputs(cfg: Config, rng):
    """[CLS] question [SEP] context [SEP] [PAD]...: segment ids 0/1, trailing padding."""
    B, S = cfg.batch, cfg.seq_len
    input_ids = jax.random.randint(rng, (B, S), 1, cfg.vocab_size, dtype=jnp.int32)
    q_len, n_valid = S // 4, (3 * S) // 4
    pos = np.arange(S)
    token_type_ids = np.broadcast_to(((pos >= q_len) & (pos < n_valid)).astype(np.int32), (B, S))
    attention_mask = np.broadcast_to((pos < n_valid).astype(np.int32), (B, S))
    input_ids = jnp.where(jnp.asarray(attention_mask) == 1, input_ids, 0)
    return {"forward": (input_ids, jnp.asarray(token_type_ids), jnp.asarray(attention_mask))}


def intermediates(cfg: Config, variables, *inputs) -> dict:
    _, state = BertForQuestionAnswering(cfg).apply(variables, *inputs, capture_intermediates=True,
                                                   mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        flat["/".join(k for k in keys if k != "__call__")] = _first_array(v[0])
    return flat


def _first_array(v):
    """Modules returning (int8 tensor, scale[, zero point]) record the tensor."""
    while isinstance(v, (tuple, list)):
        v = v[0]
    return v


def build(cfg: Config) -> ModelSpec:
    module = BertForQuestionAnswering(cfg)

    def forward(variables, input_ids, token_type_ids, attention_mask):
        return module.apply(variables, input_ids, token_type_ids, attention_mask)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    L = cfg.num_hidden_layers
    return ModelSpec(
        name="bert",
        variant=(f"BERT-Large SQuAD v1.1 QA (MLPerf bert-99), {L} layers, hidden "
                 f"{cfg.hidden_size}, seq {cfg.seq_len}, batch {cfg.batch}, INT8 W8A8 incl. "
                 "attention (TensorRT, MLPerf NVIDIA): per-row weights, per-tensor static "
                 "activations, INT8 residual stream, fp32 LayerNorm / softmax / tanh-GELU"),
        phases=[Phase("forward", forward, input_spec(cfg),
                      note="one encoder pass -> (start_logits, end_logits) in fp16")],
        param_specs=as_spec(variables),
        excluded=["WordPiece tokenization / SQuAD feature construction (doc-stride windows)",
                  "start/end span decoding (top-k, n-best, max answer length) (postprocessing)"],
    )
