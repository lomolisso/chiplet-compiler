"""BERT-Large for SQuAD v1.1 QA (MLPerf Inference `bert-99`), INT8 Flax linen port.

Mirrors HF transformers 5.20.2 `BertForQuestionAnswering` (eager attention,
encoder-only, no pooler) module-for-module; module names follow HF's
(`layer.N` -> `layer_N`), so the parameter mapping is mechanical:

  bert/embeddings/{word,position,token_type}_embeddings, LayerNorm
  bert/encoder/layer_N/attention/self/{query,key,value}
  bert/encoder/layer_N/attention/output/{dense,LayerNorm}
  bert/encoder/layer_N/intermediate/dense
  bert/encoder/layer_N/output/{dense,LayerNorm}
  qa_outputs

Float numerics mirror HF: scores = (q @ k^T) * d^-0.5 + mask, where the mask is
HF's extended additive mask (0 for kept keys, finfo(dtype).min for padded
keys); softmax over keys; LayerNorm eps 1e-12 with two-pass variance (as
torch). Dropout is identity at inference and is omitted.

Quantization: the MLPerf NVIDIA TensorRT INT8 engine for bert-99
(inference_results_v4.1 closed/NVIDIA/code/bert/tensorrt/network.py, class
BERTVarSeqLenINT8, built from MLCommons' QAT model bert_large_v1_1_fake_quant.onnx):
  * every GEMM is INT8 (QKV, attention-out, FC1, FC2, qa_outputs): weights per-row
    symmetric max-abs, activations per-tensor static (the QAT amax values);
  * the fused MHA plugin (CustomQKVToContextPluginDynamic, type int8) takes INT8 Q, K, V
    with one shared scale, computes Q K^T and P V in INT8, the softmax in fp32 and
    quantizes the probabilities to INT8 (scale amax / 127); its context output is INT8;
  * the residual stream is INT8 between ops and layers: the skip-LayerNorm plugins take the
    INT8 GEMM output and the INT8 residual, add and normalize in fp32 and emit INT8;
  * embeddings: fp32 tables, gathers + sum + LayerNorm in fp32 (EmbLayerNorm plugin), the
    result quantized for the first QKV GEMM;
  * GELU is the builder's tanh approximation in fp32 (`add_gelu`, FC1 epilogue), output INT8;
  * qa_outputs: INT8 conv with fp16 logits.
Graph form: explicit Q/DQ (models/quant.py); every dot_general is i8 x i8 -> i32.
NVIDIA packs variable-length sequences (no padding); here padded keys are masked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from models import quant as q


@dataclass(frozen=True)
class Config:
    batch: int = 1
    seq_len: int = 384
    vocab_size: int = 30522
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    num_labels: int = 2
    dtype: str = "float32"          # float math (LayerNorm, softmax, GELU, epilogues)


def gelu(x):
    """NVIDIA's INT8 BERT GELU (BERTNetwork.add_gelu, fp32):
    0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))."""
    return x * 0.5 * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(cfg: Config, name: str) -> nn.LayerNorm:
    # use_fast_variance=False: two-pass biased variance E[(x - mean)^2], as torch.
    return nn.LayerNorm(epsilon=cfg.layer_norm_eps, use_fast_variance=False,
                        dtype=cfg.dtype, param_dtype=cfg.dtype, name=name)


def dense(cfg: Config, features: int, name: str, dtype: str | None = None) -> q.QDense:
    """INT8 GEMM, per-row weight scales; the input arrives INT8 with its producer's scale."""
    return q.QDense(features, q.INT8_STATIC, dtype=dtype or cfg.dtype, name=name)


def embed(cfg: Config, num: int, name: str) -> nn.Embed:
    return nn.Embed(num, cfg.hidden_size, dtype=cfg.dtype, param_dtype=cfg.dtype, name=name)


def add_layer_norm(cfg: Config, y, residual, r_scale):
    """Skip-LayerNorm plugin: the GEMM output quantized to INT8 (`dense_quant`) + the INT8
    residual, added and normalized in fp32, INT8 output. Returns (int8, scale)."""
    yq, ys, _ = q.ActQuant(name="dense_quant")(y)
    h = q.dequantize(yq, ys, cfg.dtype) + q.dequantize(residual, r_scale, cfg.dtype)
    out, s, _ = q.ActQuant(name="LayerNorm_quant")(layer_norm(cfg, "LayerNorm")(h))
    return out, s


class BertEmbeddings(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, input_ids, token_type_ids):
        c = self.cfg
        position_ids = jnp.arange(input_ids.shape[1], dtype=jnp.int32)[None, :]
        inputs_embeds = embed(c, c.vocab_size, "word_embeddings")(input_ids)
        token_type_embeddings = embed(c, c.type_vocab_size, "token_type_embeddings")(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = embed(c, c.max_position_embeddings, "position_embeddings")(position_ids)
        embeddings = embeddings + position_embeddings
        out, s, _ = q.ActQuant(name="LayerNorm_quant")(layer_norm(c, "LayerNorm")(embeddings))
        return out, s


class BertSelfAttention(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden_states, h_scale, attention_mask):
        c = self.cfg
        head_dim = c.hidden_size // c.num_attention_heads
        hidden_shape = (*hidden_states.shape[:-1], c.num_attention_heads, head_dim)
        # Q, K, V: INT8 with one shared scale (dr_qkv) [B, S, H, D]
        qkv_quant = q.ActQuant(name="qkv_quant")
        query, s_qkv, _ = qkv_quant(dense(c, c.hidden_size, "query")(hidden_states, h_scale))
        key, _, _ = qkv_quant(dense(c, c.hidden_size, "key")(hidden_states, h_scale))
        value, _, _ = qkv_quant(dense(c, c.hidden_size, "value")(hidden_states, h_scale))
        query, key, value = (t.reshape(hidden_shape) for t in (query, key, value))
        # matmul(q, k^T) * scaling + mask -> softmax (fp32) -> INT8 probs @ v
        attn_weights = q.qeinsum("bqhd,bkhd->bhqk", query, key, s_qkv, s_qkv, "int8", c.dtype)
        attn_weights = attn_weights * (head_dim ** -0.5) + attention_mask
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        probs, s_p, _ = q.ActQuant(init_scale=1.0 / 127, name="probs_quant")(attn_weights)
        attn_output = q.qeinsum("bhqk,bkhd->bqhd", probs, value, s_p, s_qkv, "int8", c.dtype)
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        out, s, _ = q.ActQuant(name="context_quant")(attn_output)
        return out, s


class BertSelfOutput(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden_states, s, input_tensor, s_in):
        hidden_states = dense(self.cfg, self.cfg.hidden_size, "dense")(hidden_states, s)
        return add_layer_norm(self.cfg, hidden_states, input_tensor, s_in)


class BertAttention(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden_states, s, attention_mask):
        ctx, s_ctx = BertSelfAttention(self.cfg, name="self")(hidden_states, s, attention_mask)
        return BertSelfOutput(self.cfg, name="output")(ctx, s_ctx, hidden_states, s)


class BertIntermediate(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden_states, s):
        hidden_states = dense(self.cfg, self.cfg.intermediate_size, "dense")(hidden_states, s)
        out, s, _ = q.ActQuant(name="gelu_quant")(gelu(hidden_states))
        return out, s


class BertOutput(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden_states, s, input_tensor, s_in):
        hidden_states = dense(self.cfg, self.cfg.hidden_size, "dense")(hidden_states, s)
        return add_layer_norm(self.cfg, hidden_states, input_tensor, s_in)


class BertLayer(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden_states, s, attention_mask):
        attention_output, s_a = BertAttention(self.cfg, name="attention")(hidden_states, s,
                                                                          attention_mask)
        intermediate_output, s_i = BertIntermediate(self.cfg, name="intermediate")(
            attention_output, s_a)
        return BertOutput(self.cfg, name="output")(intermediate_output, s_i, attention_output, s_a)


class BertEncoder(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden_states, s, attention_mask):
        for i in range(self.cfg.num_hidden_layers):          # fully unrolled
            hidden_states, s = BertLayer(self.cfg, name=f"layer_{i}")(hidden_states, s,
                                                                     attention_mask)
        return hidden_states, s


def extended_attention_mask(attention_mask, dtype):
    """HF eager bidirectional mask: [B, S] {0,1} -> additive [B, 1, 1, S] (0 / finfo.min)."""
    keep = attention_mask[:, None, None, :].astype(jnp.bool_)
    return jnp.where(keep, jnp.zeros((), dtype), jnp.finfo(dtype).min).astype(dtype)


class BertModel(nn.Module):
    """BertModel(add_pooling_layer=False); returns the INT8 sequence output and its scale."""
    cfg: Config

    @nn.compact
    def __call__(self, input_ids, token_type_ids, attention_mask):
        embedding_output, s = BertEmbeddings(self.cfg, name="embeddings")(input_ids, token_type_ids)
        mask = extended_attention_mask(attention_mask, jnp.dtype(self.cfg.dtype))
        return BertEncoder(self.cfg, name="encoder")(embedding_output, s, mask)


class BertForQuestionAnswering(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, input_ids, token_type_ids, attention_mask):
        sequence_output, s = BertModel(self.cfg, name="bert")(input_ids, token_type_ids,
                                                             attention_mask)
        logits = dense(self.cfg, self.cfg.num_labels, "qa_outputs",
                       dtype="float16")(sequence_output, s)              # fp16 logits
        start_logits, end_logits = logits[..., 0], logits[..., 1]
        return start_logits, end_logits
