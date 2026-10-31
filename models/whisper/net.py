"""Whisper large-v3 (MLPerf Inference speech-to-text), Flax linen.

Port of HF transformers v4.57.6 `models/whisper/modeling_flax_whisper.py`
(commit 753d61104116eefc8ffc977327b441ee0c8d599f, Apache-2.0).
Class names, submodule names (and therefore the parameter tree), the
attention / cache code and the op order are kept from that file. Removed or
changed:

* the FlaxPreTrainedModel wrapper, ModelOutput dataclasses, output_attentions /
  output_hidden_states plumbing, dropout / LayerDrop / remat (training only) and
  the decoder padding-mask arguments (batch 1, unpadded prompt);
* cross-attention K/V caching (`cached_cross_key` / `cached_cross_value`) added
  so the decode step reuses the cross K/V computed at prefill, like the torch
  reference's EncoderDecoderCache (HF Flax recomputed them every step);
* `param_dtype` follows `dtype` for the float parameters.

Quantization: CTranslate2 `int8_float16` (the faster-whisper runtime; CTranslate2 v4.5.0
src/layers/common.cc Dense, src/ops/quantize.cc, python/ctranslate2/specs/model_spec.py,
src/models/model.cc):
* every Linear (q/k/v/out of self- and cross-attention, fc1, fc2) is W8A8 int8: weights
  per-output-row symmetric (127 / row amax), the input quantized on the fly with one scale
  per row (token), an int8 x int8 -> int32 GEMM, dequantized with input x weight scale and
  the bias into fp16 (`q.INT8_DYNAMIC_TOKEN`);
* the token embedding table is int8 with one scale per row; the tied LM head reuses it
  (CTranslate2 aliases `decoder/projection/weight{,_scale}` to the embeddings) as a W8A8
  GEMM with dynamic per-token activations;
* fp16 (CTranslate2's float type for int8_float16): the conv stem (no quantized conv on
  CUDA), attention QK^T / softmax / PV, LayerNorm, GELU, position embeddings, and the self /
  cross K/V caches.
CTranslate2 fuses the self-attention QKV (and cross-attention KV) projections into one GEMM;
the per-row weight scales and the per-token input scale are the same, so separate
projections compute the same arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from jax import lax

from models import quant as q


@dataclass(frozen=True)
class Config:
    """The `WhisperConfig` fields the network reads (defaults: openai/whisper-large-v3)."""
    num_mel_bins: int = 128
    d_model: int = 1280
    encoder_layers: int = 32
    decoder_layers: int = 32
    encoder_attention_heads: int = 20
    decoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    decoder_ffn_dim: int = 5120
    vocab_size: int = 51866
    max_source_positions: int = 1500
    max_target_positions: int = 448
    activation_function: str = "gelu"
    init_std: float = 0.02
    dtype: str = "float16"          # CTranslate2 int8_float16: float layers in fp16
    # MLPerf decode settings (not read by the network itself)
    batch: int = 1
    # static self-attention KV-cache capacity (HF Flax `init_cache(batch_size, max_length)`;
    # generation_config.json max_length = 448); at most max_target_positions
    max_length: int = 448
    prompt_len: int = 4            # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    max_new_tokens: int = 200      # vLLM SamplingParams(max_tokens=200, temperature=0)


ACT2FN = {"gelu": partial(nn.gelu, approximate=False)}


def sinusoidal_embedding_init(key, shape, dtype=jnp.float_) -> jax.Array:
    """Returns sinusoids for positional embedding"""
    length, channels = shape
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(10000) / (channels // 2 - 1)
    inv_timescales = jnp.exp(-log_timescale_increment * jnp.arange(channels // 2))
    scaled_time = jnp.arange(length).reshape(-1, 1) * inv_timescales.reshape(1, -1)
    return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1).astype(dtype)


class FlaxWhisperAttention(nn.Module):
    config: Config
    embed_dim: int
    num_heads: int
    causal: bool = False
    is_cross_attention: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        dense = partial(q.QDense, self.embed_dim, q.INT8_DYNAMIC_TOKEN, dtype=self.dtype)

        self.q_proj = dense(use_bias=self.bias)
        self.k_proj = dense(use_bias=False)
        self.v_proj = dense(use_bias=self.bias)
        self.out_proj = dense(use_bias=self.bias)

        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_target_positions), dtype="bool"), dtype="bool"
            )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states)
        query_states = self._split_heads(query_states)

        if self.is_cross_attention:
            if key_value_states is None:
                # decode step: reuse the cross-attention K/V cached at prefill
                key_states = self.variables["cache"]["cached_cross_key"]
                value_states = self.variables["cache"]["cached_cross_value"]
            else:
                key_states = self._split_heads(self.k_proj(key_value_states))
                value_states = self._split_heads(self.v_proj(key_value_states))
                if self.is_mutable_collection("cache"):
                    # prefill: store them for the decode steps
                    self.put_variable("cache", "cached_cross_key", key_states)
                    self.put_variable("cache", "cached_cross_value", value_states)
        else:
            key_states = self._split_heads(self.k_proj(hidden_states))
            value_states = self._split_heads(self.v_proj(hidden_states))

        attention_mask = None
        if self.causal:
            # the decoder always runs with the static KV cache (prefill and step)
            query_length = query_states.shape[1]
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
            attention_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

            # During fast autoregressive decoding, we feed one position at a time,
            # and cache the keys and values step by step.
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=None,
            dropout_rate=0.0,
            broadcast_dropout=True,
            deterministic=True,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

    def _split_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        num_updated_cache_vectors = query.shape[1]
        cache_index.value = cache_index.value + num_updated_cache_vectors
        # causal mask for cached decoder self-attention: our single query position should only
        # attend to those key positions that have already been generated and cached, not the
        # remaining zero elements.
        pad_mask = jnp.broadcast_to(
            jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
            tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
        )
        attention_mask = combine_masks(pad_mask, attention_mask)

        return key, value, attention_mask


class FlaxWhisperEncoderLayer(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dtype=self.dtype,
        )
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.dtype, epsilon=1e-05)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.fc1 = q.QDense(self.config.encoder_ffn_dim, q.INT8_DYNAMIC_TOKEN, dtype=self.dtype)
        self.fc2 = q.QDense(self.embed_dim, q.INT8_DYNAMIC_TOKEN, dtype=self.dtype)
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.dtype, epsilon=1e-05)

    def __call__(self, hidden_states: jnp.ndarray) -> tuple[jnp.ndarray]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class FlaxWhisperEncoderLayerCollection(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxWhisperEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.encoder_layers)
        ]

    def __call__(self, hidden_states):
        for encoder_layer in self.layers:     # Python loop: fully unrolled
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs[0]
        return (hidden_states,)


class FlaxWhisperDecoderLayer(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            causal=True,
            dtype=self.dtype,
        )
        self.activation_fn = ACT2FN[self.config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.dtype, epsilon=1e-05)
        self.encoder_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            is_cross_attention=True,
            dtype=self.dtype,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.dtype, epsilon=1e-05)
        self.fc1 = q.QDense(self.config.decoder_ffn_dim, q.INT8_DYNAMIC_TOKEN, dtype=self.dtype)
        self.fc2 = q.QDense(self.embed_dim, q.INT8_DYNAMIC_TOKEN, dtype=self.dtype)
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.dtype, epsilon=1e-05)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        # Cross-Attention Block (K/V from encoder_hidden_states at prefill, from the
        # cross cache at decode steps)
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class FlaxWhisperDecoderLayerCollection(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxWhisperDecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]

    def __call__(self, hidden_states, encoder_hidden_states: jnp.ndarray | None = None):
        for decoder_layer in self.layers:     # Python loop: fully unrolled
            layer_outputs = decoder_layer(hidden_states, encoder_hidden_states)
            hidden_states = layer_outputs[0]
        return (hidden_states,)


class FlaxWhisperEncoder(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.conv1 = nn.Conv(
            self.config.d_model,
            kernel_size=(3,),
            padding=1,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
            param_dtype=self.dtype,
        )
        self.conv2 = nn.Conv(
            self.config.d_model,
            kernel_size=(3,),
            strides=2,
            padding=1,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
            param_dtype=self.dtype,
        )

        self.layers = FlaxWhisperEncoderLayerCollection(self.config, dtype=self.dtype)

        self.embed_positions = nn.Embed(
            self.config.max_source_positions,
            self.config.d_model,
            dtype=self.dtype,
            param_dtype=self.dtype,
            embedding_init=sinusoidal_embedding_init,
        )

        self.layer_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.dtype, epsilon=1e-05)

    def __call__(self, input_features: jnp.ndarray) -> tuple[jnp.ndarray]:
        if input_features.shape[1:] != (self.config.num_mel_bins, self.config.max_source_positions * 2):
            raise ValueError(
                "input_features.shape[1:], must be equal to (self.config.num_mel_bins,"
                f" self.config.max_source_positions * 2) (got {input_features.shape[1:]}, but should be"
                f" ({self.config.num_mel_bins}, {self.config.max_source_positions * 2}))"
            )

        input_features = input_features.transpose(0, 2, 1)
        hidden_states = jax.nn.gelu(self.conv1(input_features), approximate=False)
        hidden_states = jax.nn.gelu(self.conv2(hidden_states), approximate=False)

        embed_positions = self.embed_positions(jnp.arange(self.config.max_source_positions))
        # freeze the sinusoidal embeddings by stopping the back-prop
        embed_positions = jax.lax.stop_gradient(embed_positions)
        hidden_states = hidden_states + embed_positions

        outputs = self.layers(hidden_states)

        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)
        return (last_hidden_states,)


class FlaxWhisperDecoder(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        # int8 table, one scale per row (CTranslate2 Embeddings + weight_scale)
        self.embed_tokens = q.QEmbed(self.config.vocab_size, self.config.d_model, dtype=self.dtype,
                                     per_row=True)
        self.embed_positions = nn.Embed(self.config.max_target_positions, self.config.d_model,
                                        dtype=self.dtype, param_dtype=self.dtype)

        self.layers = FlaxWhisperDecoderLayerCollection(self.config, dtype=self.dtype)

        self.layer_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.dtype, epsilon=1e-5)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        position_ids: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray]:
        input_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)

        hidden_states = input_embeds + position_embeds

        outputs = self.layers(hidden_states, encoder_hidden_states=encoder_hidden_states)

        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)
        return (last_hidden_states,)


class FlaxWhisperModule(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.encoder = FlaxWhisperEncoder(self.config, dtype=self.dtype)
        self.decoder = FlaxWhisperDecoder(self.config, dtype=self.dtype)


class FlaxWhisperForConditionalGenerationModule(nn.Module):
    """Entry points `encode` / `decode` mirror HF Flax `FlaxWhisperForConditionalGeneration.
    encode` (`_encoder_forward`) and `.decode` (`_decoder_forward`, tied lm head)."""
    config: Config

    def setup(self) -> None:
        dtype = jnp.dtype(self.config.dtype)
        self.model = FlaxWhisperModule(config=self.config, dtype=dtype)
        self.lm_head = q.QDense(self.config.vocab_size, q.INT8_DYNAMIC_TOKEN, use_bias=False,
                                dtype=dtype)

    def encode(self, input_features):
        return self.model.encoder(input_features)

    def decode(self, decoder_input_ids, decoder_position_ids, encoder_hidden_states=None):
        outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = outputs[0]

        # tie_word_embeddings=True for every Whisper checkpoint: the int8 table and its
        # per-row scales are the LM head's kernel and per-output-channel kernel_scale
        emb = self.model.decoder.embed_tokens.variables["params"]
        lm_logits = self.lm_head.apply(
            {"params": {"kernel": emb["embedding"].T, "kernel_scale": emb["embedding_scale"][:, 0]}},
            hidden_states)
        return lm_logits
