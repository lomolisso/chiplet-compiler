"""Stable Diffusion XL base 1.0 (MLPerf Inference text-to-image), Flax linen.

Four networks, one per captured phase:

* CLIP ViT-L/14 text encoder   -- ported from HF transformers v4.57.6
  `models/clip/modeling_flax_clip.py` (FlaxCLIPTextModule).
* OpenCLIP bigG text encoder + projection -- same file
  (FlaxCLIPTextModelWithProjectionModule).
* UNet2DConditionModel          -- ported from MaxDiffusion
  `models/unet_2d_condition_flax.py`, `unet_2d_blocks_flax.py`,
  `resnet_flax.py`, `embeddings_flax.py`, `attention_flax.py` (linen classes).
* AutoencoderKL decoder         -- ported from MaxDiffusion `models/vae_flax.py`.

Class, attribute and submodule names follow the upstream files so parameter
paths are the upstream Flax paths. Sharding constraints, MaxDiffusion's own
quantization / kernel plumbing and config branches SDXL never takes were removed;
three numerical fixes make the port compute what diffusers computes (`# DEVIATION`).

Quantization: the MLPerf NVIDIA Hopper (H100 / H200) TensorRT engines
(mlcommons/inference_results_v4.1-v5.1 closed/NVIDIA/code/stable-diffusion-xl:
modelopt/{utils,quantize_unet,quantize_vae,export_vae}.py, tensorrt/{builder,
sdxl_graphsurgeon}.py; see models/sdxl/recipe.py):
  * CLIP-L / CLIP-bigG: FP16 engines (`SDXLCLIPBuilder`, sm < 100); params fp16.
  * UNet: ModelOpt `SDXL_FP8_CFG` -- FP8 E4M3 W8A8, per-tensor weights and per-tensor
    static activations (max calibration, 500 captions x 20 steps), on every Linear and
    Conv2d except `filter_func_unet`'s carve-outs, which stay fp16: time_embedding,
    add_embedding, every time_emb_proj, conv_in, conv_out, every conv_shortcut,
    up_blocks.2.resnets.2.conv1/conv2. FP8 attention in every attn1 / attn2
    (`set_fmha` + q/k/v_bmm and softmax quantizers): Q, K, V and the softmax
    probabilities FP8 per-tensor, QK^T and PV f8 x f8 -> f32 (TensorRT fused FP8 MHA,
    softmax in fp32). Everything else (norms, SiLU, GEGLU, residuals, FP8 epilogues)
    fp16 (`trt_high_precision_dtype` Half); engine I/O fp16.
  * VAE decoder: the INT8 engine (`SDXLVAEBuilder` on Hopper). ModelOpt INT8 (quant level
    3, `get_int8_config`: weights per-output-channel, conv activations per-tensor static,
    max calibration over 64 captions) minus `filter_func`; the SmoothQuant step only acts
    on Linears, which are all in the disabled mid_block, so the result is INT8 W8A8 on 18
    convs (up_blocks.0-1, up_blocks.2.resnets.0 and every upsampler). The ONNX is fp32
    ("QDQ needs to be in FP32") with fp16 islands cast in by graphsurgeon
    (`insert_vae_fp16_conv_cast` / `insert_vae_fp16_linear_casts`); see `_vae_conv_mode`.
Graph form: explicit Q/DQ (models/quant.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import make_causal_mask
from flax.linen.attention import dot_product_attention_weights

from models import quant as q

FP16 = jnp.float16
# DEVIATION (numerics): two-pass variance E[(x-mean)^2] as torch's GroupNorm /
# LayerNorm, instead of flax's default E[x^2]-E[x]^2, which loses ~1e-4 relative
# accuracy in fp32 when |mean| >> std (e.g. GroupNorm right after the timestep
# embedding is added in every UNet resnet). Applies to every norm below.
GroupNorm = partial(nn.GroupNorm, use_fast_variance=False)
LayerNorm = partial(nn.LayerNorm, use_fast_variance=False)


# =============================================================================
# Configs (field names follow the HF / diffusers config.json keys)
# =============================================================================

@dataclass(frozen=True)
class CLIPTextConfig:
    vocab_size: int = 49408
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    projection_dim: int = 768
    eos_token_id: int = 2


CLIP_L = CLIPTextConfig()
CLIP_G = CLIPTextConfig(hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
                        num_attention_heads=20, hidden_act="gelu", projection_dim=1280)


@dataclass(frozen=True)
class UNetConfig:
    sample_size: int = 128
    in_channels: int = 4
    out_channels: int = 4
    down_block_types: tuple = ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D")
    up_block_types: tuple = ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D")
    block_out_channels: tuple = (320, 640, 1280)
    layers_per_block: int = 2
    attention_head_dim: tuple = (5, 10, 20)      # == number of heads (diffusers naming quirk)
    transformer_layers_per_block: tuple = (1, 2, 10)
    cross_attention_dim: int = 2048
    use_linear_projection: bool = True
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    addition_embed_type: str = "text_time"
    addition_time_embed_dim: int = 256
    projection_class_embeddings_input_dim: int = 2816
    norm_num_groups: int = 32
    norm_eps: float = 1e-5


@dataclass(frozen=True)
class VAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    up_block_types: tuple = ("UpDecoderBlock2D",) * 4
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    scaling_factor: float = 0.13025


# =============================================================================
# CLIP text encoders (HF transformers v4.57.6 modeling_flax_clip.py)
# =============================================================================

def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)


ACT2FN = {"gelu": partial(nn.gelu, approximate=False), "quick_gelu": quick_gelu}


class FlaxCLIPTextEmbeddings(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        self.token_embedding = nn.Embed(self.config.vocab_size, embed_dim, dtype=self.dtype,
                                        param_dtype=self.dtype,
                                        embedding_init=jax.nn.initializers.normal())
        self.position_embedding = nn.Embed(self.config.max_position_embeddings, embed_dim, dtype=self.dtype,
                                           param_dtype=self.dtype,
                                           embedding_init=jax.nn.initializers.normal())

    def __call__(self, input_ids, position_ids):
        input_embeds = self.token_embedding(input_ids.astype("i4"))
        position_embeds = self.position_embedding(position_ids.astype("i4"))
        embeddings = input_embeds + position_embeds
        return embeddings


class FlaxCLIPAttention(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = nn.Dense(self.embed_dim, dtype=self.dtype, param_dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.v_proj = nn.Dense(self.embed_dim, dtype=self.dtype, param_dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.q_proj = nn.Dense(self.embed_dim, dtype=self.dtype, param_dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.out_proj = nn.Dense(self.embed_dim, dtype=self.dtype, param_dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        # Text model: always causal (upstream `self.causal = isinstance(config, CLIPTextConfig)`).
        self.causal_mask = make_causal_mask(jnp.ones((1, self.config.max_position_embeddings), dtype="i4"))

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(self, hidden_states):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # SDXL calls the text encoders with input_ids only: the mask is the causal mask.
        query_length, key_length = query.shape[1], key.shape[1]
        attention_mask = self.causal_mask[:, :, key_length - query_length: key_length, :key_length]
        attention_bias = jax.lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )
        attn_weights = dot_product_attention_weights(
            query, key, bias=attention_bias, deterministic=True, dtype=self.dtype, precision=None,
            force_fp32_for_softmax=True).astype(self.dtype)   # TensorRT fp16 MHA: fp32 softmax

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return (attn_output,)


class FlaxCLIPMLP(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.activation_fn = ACT2FN[self.config.hidden_act]
        self.fc1 = nn.Dense(self.config.intermediate_size, dtype=self.dtype, param_dtype=self.dtype,
                            kernel_init=jax.nn.initializers.normal(0.01))
        self.fc2 = nn.Dense(self.config.hidden_size, dtype=self.dtype, param_dtype=self.dtype,
                            kernel_init=jax.nn.initializers.normal(0.01))

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FlaxCLIPEncoderLayer(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = FlaxCLIPAttention(self.config, dtype=self.dtype)
        self.layer_norm1 = LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype, param_dtype=self.dtype)
        self.mlp = FlaxCLIPMLP(self.config, dtype=self.dtype)
        self.layer_norm2 = LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype, param_dtype=self.dtype)

    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(hidden_states)
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)


class FlaxCLIPLayerCollection(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [FlaxCLIPEncoderLayer(self.config, name=str(i), dtype=self.dtype)
                       for i in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states):
        # output_hidden_states=True (SDXL reads hidden_states[-2]).
        all_hidden_states = ()
        for layer in self.layers:
            all_hidden_states += (hidden_states,)
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]
        all_hidden_states += (hidden_states,)
        return hidden_states, all_hidden_states


class FlaxCLIPEncoder(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = FlaxCLIPLayerCollection(self.config, dtype=self.dtype)

    def __call__(self, inputs_embeds):
        return self.layers(hidden_states=inputs_embeds)


class FlaxCLIPTextTransformer(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embeddings = FlaxCLIPTextEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxCLIPEncoder(self.config, dtype=self.dtype)
        self.final_layer_norm = LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype, param_dtype=self.dtype)
        self.eos_token_id = self.config.eos_token_id

    def __call__(self, input_ids, position_ids):
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        last_hidden_state, all_hidden_states = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # legacy CLIP configs (SDXL's): EOS is the highest token id
            pooled_output = last_hidden_state[jnp.arange(last_hidden_state.shape[0]),
                                              input_ids.argmax(axis=-1)]
        else:
            pooled_output = last_hidden_state[jnp.arange(last_hidden_state.shape[0]),
                                              (input_ids == self.eos_token_id).argmax(axis=-1)]
        return last_hidden_state, pooled_output, all_hidden_states


def _position_ids(input_ids):
    # FlaxCLIPTextPreTrainedModel.__call__ default position_ids
    return jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)


class FlaxCLIPTextModule(nn.Module):
    """CLIPTextModel (SDXL `text_encoder`). Returns (hidden_states[-2], pooler_output)."""
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.text_model = FlaxCLIPTextTransformer(self.config, dtype=self.dtype)

    def __call__(self, input_ids):
        _, pooled_output, hidden_states = self.text_model(input_ids, _position_ids(input_ids))
        return hidden_states[-2], pooled_output


class FlaxCLIPTextModelWithProjectionModule(nn.Module):
    """CLIPTextModelWithProjection (SDXL `text_encoder_2`). Returns (hidden_states[-2], text_embeds)."""
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.text_model = FlaxCLIPTextTransformer(self.config, dtype=self.dtype)
        self.text_projection = nn.Dense(self.config.projection_dim, use_bias=False, dtype=self.dtype,
                                        param_dtype=self.dtype)

    def __call__(self, input_ids):
        _, pooled_output, hidden_states = self.text_model(input_ids, _position_ids(input_ids))
        text_embeds = self.text_projection(pooled_output)
        return hidden_states[-2], text_embeds


# =============================================================================
# UNet (MaxDiffusion embeddings_flax.py / resnet_flax.py / attention_flax.py /
# unet_2d_blocks_flax.py / unet_2d_condition_flax.py)
# =============================================================================

def get_sinusoidal_embeddings(timesteps, embedding_dim, freq_shift=1, min_timescale=1,
                              max_timescale=1.0e4, flip_sin_to_cos=False, scale=1.0):
    num_timescales = float(embedding_dim // 2)
    # DEVIATION (numerics): exponent computed in diffusers' `get_timestep_embedding`
    # order (-log(max) * arange / (half - shift)) instead of MaxDiffusion's
    # `arange * -(log(max)/(half - shift))`. Same math; at t ~ 1000 a one-ulp
    # difference in a frequency shifts sin/cos by ~1e-4.
    exponent = -math.log(max_timescale / min_timescale) * jnp.arange(num_timescales, dtype=jnp.float32)
    exponent = exponent / (num_timescales - freq_shift)
    inv_timescales = min_timescale * jnp.exp(exponent)
    emb = jnp.expand_dims(timesteps, -1) * inv_timescales
    scaled_time = scale * emb
    if flip_sin_to_cos:
        signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=-1)
    else:
        signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
    return signal


def _fp8_dense(features, dtype, use_bias=True, name=None):
    """UNet Linear under SDXL_FP8_CFG: FP8 E4M3 per-tensor W8A8, fp16 epilogue."""
    return q.QDense(features, q.FP8_STATIC, use_bias=use_bias, dtype=dtype, name=name)


def _fp8_conv(features, stride, dtype):
    """UNet 3x3 Conv2d under SDXL_FP8_CFG: FP8 E4M3 per-tensor W8A8, fp16 epilogue."""
    return q.QConv(features, (3, 3), q.FP8_STATIC, strides=(stride, stride),
                   padding=((1, 1), (1, 1)), dtype=dtype)


class FlaxTimestepEmbedding(nn.Module):
    """time_embedding / add_embedding: fp16 (filter_func_unet carve-outs)."""
    time_embed_dim: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, temb):
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, param_dtype=self.dtype, name="linear_1")(temb)
        temb = nn.silu(temb)
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, param_dtype=self.dtype, name="linear_2")(temb)
        return temb


class FlaxTimesteps(nn.Module):
    dim: int = 32
    flip_sin_to_cos: bool = False
    freq_shift: float = 1.0

    @nn.compact
    def __call__(self, timesteps):
        return get_sinusoidal_embeddings(timesteps, embedding_dim=self.dim,
                                         flip_sin_to_cos=self.flip_sin_to_cos,
                                         freq_shift=self.freq_shift)


class FlaxUpsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = _fp8_conv(self.out_channels, 1, self.dtype)

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jnp.broadcast_to(hidden_states[:, :, None, :, None, :],
                                         (batch, height, 2, width, 2, channels))
        hidden_states = jnp.reshape(hidden_states, (batch, height * 2, width * 2, channels))
        hidden_states = self.conv(hidden_states)
        return hidden_states


class FlaxDownsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = _fp8_conv(self.out_channels, 2, self.dtype)

    def __call__(self, hidden_states):
        return self.conv(hidden_states)


class FlaxResnetBlock2D(nn.Module):
    in_channels: int
    out_channels: int = None
    dtype: jnp.dtype = jnp.float32
    norm_num_groups: int = 32
    eps: float = 1e-5
    fp8_convs: bool = True       # False: up_blocks.2.resnets.2 (filter_func_unet), fp16 convs

    def setup(self):
        out_channels = self.in_channels if self.out_channels is None else self.out_channels
        dt = self.dtype
        self.norm1 = GroupNorm(num_groups=self.norm_num_groups, epsilon=self.eps, dtype=dt, param_dtype=dt)
        self.norm2 = GroupNorm(num_groups=self.norm_num_groups, epsilon=self.eps, dtype=dt, param_dtype=dt)
        use_nin_shortcut = self.in_channels != out_channels
        self.conv_shortcut = None
        if use_nin_shortcut:     # fp16 carve-out
            self.conv_shortcut = nn.Conv(out_channels, kernel_size=(1, 1), strides=(1, 1),
                                         padding="VALID", dtype=dt, param_dtype=dt)
        if self.fp8_convs:
            self.conv1 = _fp8_conv(out_channels, 1, dt)
            self.conv2 = _fp8_conv(out_channels, 1, dt)
        else:
            self.conv1 = nn.Conv(out_channels, kernel_size=(3, 3), strides=(1, 1),
                                 padding=((1, 1), (1, 1)), dtype=dt, param_dtype=dt)
            self.conv2 = nn.Conv(out_channels, kernel_size=(3, 3), strides=(1, 1),
                                 padding=((1, 1), (1, 1)), dtype=dt, param_dtype=dt)
        self.time_emb_proj = nn.Dense(out_channels, dtype=dt, param_dtype=dt)   # fp16 carve-out

    def __call__(self, hidden_states, temb):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv1(hidden_states)
        temb = nn.swish(temb)
        temb = self.time_emb_proj(temb)
        temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return hidden_states + residual


def _apply_attention_dot(query, key, value, scales, softmax_quantizer, dtype, heads, dim_head, scale):
    """MaxDiffusion `_apply_attention_dot`, split_head_dim=True path (base_xl.yml), no mask,
    in TensorRT's fused FP8 MHA form: `query`/`key`/`value` arrive FP8 with their per-tensor
    scales; QK^T is f8 x f8 -> f32, the scaled softmax runs in fp32, its probabilities are
    quantized to FP8 (ModelOpt softmax_quantizer) and PV is f8 x f8 -> f32 -> `dtype`."""
    q_scale, k_scale, v_scale = scales
    b = key.shape[0]
    query_states = jnp.reshape(query, (b, -1, heads, dim_head))
    key_states = jnp.reshape(key, (b, -1, heads, dim_head))
    value_states = jnp.reshape(value, (b, -1, heads, dim_head))
    attention_scores = q.qeinsum("b t n h, b f n h -> b n f t", key_states, query_states,
                                 k_scale, q_scale, "fp8", jnp.float32)
    attention_scores = attention_scores * scale
    attention_probs = nn.softmax(attention_scores, axis=-1)
    probs_q, p_scale, _ = softmax_quantizer(attention_probs)
    hidden_states = q.qeinsum("b n f t, b t n h -> b f n h", probs_q, value_states,
                              p_scale, v_scale, "fp8", dtype)
    hidden_states = jnp.reshape(hidden_states, (b, -1, heads * dim_head))
    return hidden_states


class AttentionOp(nn.Module):
    scale: float
    heads: int
    dim_head: int
    dtype: jnp.dtype = jnp.float32

    def apply_attention(self, query, key, value, scales, softmax_quantizer):
        # attention_kernel="dot_product": the only kernel that lowers without custom_calls.
        return _apply_attention_dot(query, key, value, scales, softmax_quantizer, self.dtype,
                                    self.heads, self.dim_head, self.scale)


class FlaxAttention(nn.Module):
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim_head * self.heads
        scale = self.dim_head ** -0.5
        self.attention_op = AttentionOp(scale=scale, heads=self.heads, dim_head=self.dim_head,
                                        dtype=self.dtype)
        self.query = _fp8_dense(inner_dim, self.dtype, use_bias=False, name="to_q")
        self.key = _fp8_dense(inner_dim, self.dtype, use_bias=False, name="to_k")
        self.value = _fp8_dense(inner_dim, self.dtype, use_bias=False, name="to_v")
        self.proj_attn = _fp8_dense(self.query_dim, self.dtype, name="to_out_0")
        # FP8 MHA: ModelOpt _QuantAttention's per-tensor q/k/v_bmm and softmax quantizers
        self.q_bmm_quantizer = q.ActQuant("fp8", init_scale=4.0 / 448)
        self.k_bmm_quantizer = q.ActQuant("fp8", init_scale=4.0 / 448)
        self.v_bmm_quantizer = q.ActQuant("fp8", init_scale=4.0 / 448)
        self.softmax_quantizer = q.ActQuant("fp8", init_scale=1.0 / 448)

    def __call__(self, hidden_states, context=None):
        context = hidden_states if context is None else context
        query_proj, q_scale, _ = self.q_bmm_quantizer(self.query(hidden_states))
        key_proj, k_scale, _ = self.k_bmm_quantizer(self.key(context))
        value_proj, v_scale, _ = self.v_bmm_quantizer(self.value(context))
        hidden_states = self.attention_op.apply_attention(query_proj, key_proj, value_proj,
                                                          (q_scale, k_scale, v_scale),
                                                          self.softmax_quantizer)
        hidden_states = self.proj_attn(hidden_states)
        return hidden_states


class FlaxGEGLU(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = _fp8_dense(inner_dim * 2, self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        # DEVIATION (numerics): exact erf GELU as diffusers' GEGLU (F.gelu default);
        # MaxDiffusion uses flax's default tanh approximation (nn.gelu approximate=True).
        return hidden_linear * nn.gelu(hidden_gelu, approximate=False)


class FlaxFeedForward(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.net_0 = FlaxGEGLU(self.dim, self.dtype)
        self.net_2 = _fp8_dense(self.dim, self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class FlaxBasicTransformerBlock(nn.Module):
    dim: int
    n_heads: int
    d_head: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        dt = self.dtype
        self.attn1 = FlaxAttention(self.dim, self.n_heads, self.d_head, dtype=dt)
        self.attn2 = FlaxAttention(self.dim, self.n_heads, self.d_head, dtype=dt)
        self.ff = FlaxFeedForward(dim=self.dim, dtype=dt)
        self.norm1 = LayerNorm(epsilon=1e-5, dtype=dt, param_dtype=dt)
        self.norm2 = LayerNorm(epsilon=1e-5, dtype=dt, param_dtype=dt)
        self.norm3 = LayerNorm(epsilon=1e-5, dtype=dt, param_dtype=dt)

    def __call__(self, hidden_states, context):
        residual = hidden_states
        hidden_states = self.attn1(self.norm1(hidden_states))
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states))
        hidden_states = hidden_states + residual
        return hidden_states


class FlaxTransformer2DModel(nn.Module):
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dtype: jnp.dtype = jnp.float32
    norm_num_groups: int = 32

    def setup(self):
        # DEVIATION (numerics): eps=1e-6 as diffusers Transformer2DModel.norm
        # (MaxDiffusion uses 1e-5).
        self.norm = GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6, dtype=self.dtype,
                              param_dtype=self.dtype)
        inner_dim = self.n_heads * self.d_head
        # use_linear_projection=True (SDXL)
        self.proj_in = _fp8_dense(inner_dim, self.dtype)
        self.transformer_blocks = [
            FlaxBasicTransformerBlock(inner_dim, self.n_heads, self.d_head, dtype=self.dtype)
            for _ in range(self.depth)
        ]
        self.proj_out = _fp8_dense(inner_dim, self.dtype)
    def __call__(self, hidden_states, context):
        batch, height, width, channels = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.reshape(batch, height * width, channels)
        hidden_states = self.proj_in(hidden_states)
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, context)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, channels)
        hidden_states = hidden_states + residual
        return hidden_states


class FlaxCrossAttnDownBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    num_layers: int = 1
    num_attention_heads: int = 1
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32
    transformer_layers_per_block: int = 1
    norm_num_groups: int = 32

    def setup(self):
        resnets, attentions = [], []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            resnets.append(FlaxResnetBlock2D(in_channels=in_channels, out_channels=self.out_channels,
                                             dtype=self.dtype, norm_num_groups=self.norm_num_groups))
            attentions.append(FlaxTransformer2DModel(
                in_channels=self.out_channels, n_heads=self.num_attention_heads,
                d_head=self.out_channels // self.num_attention_heads,
                depth=self.transformer_layers_per_block, dtype=self.dtype,
                norm_num_groups=self.norm_num_groups))
        self.resnets = resnets
        self.attentions = attentions
        if self.add_downsample:
            self.downsamplers_0 = FlaxDownsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, temb, encoder_hidden_states):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
            output_states += (hidden_states,)
        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class FlaxDownBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    num_layers: int = 1
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32
    norm_num_groups: int = 32

    def setup(self):
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            resnets.append(FlaxResnetBlock2D(in_channels=in_channels, out_channels=self.out_channels,
                                             dtype=self.dtype, norm_num_groups=self.norm_num_groups))
        self.resnets = resnets
        if self.add_downsample:
            self.downsamplers_0 = FlaxDownsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, temb):
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)
        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class FlaxCrossAttnUpBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    prev_output_channel: int
    num_layers: int = 1
    num_attention_heads: int = 1
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32
    transformer_layers_per_block: int = 1
    norm_num_groups: int = 32

    def setup(self):
        resnets, attentions = [], []
        for i in range(self.num_layers):
            res_skip_channels = self.in_channels if (i == self.num_layers - 1) else self.out_channels
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels
            resnets.append(FlaxResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels,
                                             out_channels=self.out_channels, dtype=self.dtype,
                                             norm_num_groups=self.norm_num_groups))
            attentions.append(FlaxTransformer2DModel(
                in_channels=self.out_channels, n_heads=self.num_attention_heads,
                d_head=self.out_channels // self.num_attention_heads,
                depth=self.transformer_layers_per_block, dtype=self.dtype,
                norm_num_groups=self.norm_num_groups))
        self.resnets = resnets
        self.attentions = attentions
        if self.add_upsample:
            self.upsamplers_0 = FlaxUpsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)
        return hidden_states


class FlaxUpBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    prev_output_channel: int
    num_layers: int = 1
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32
    norm_num_groups: int = 32
    fp16_last_resnet: bool = False   # up_blocks.2.resnets.2.conv1/conv2 (filter_func_unet)

    def setup(self):
        resnets = []
        for i in range(self.num_layers):
            res_skip_channels = self.in_channels if (i == self.num_layers - 1) else self.out_channels
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels
            resnets.append(FlaxResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels,
                                             out_channels=self.out_channels, dtype=self.dtype,
                                             norm_num_groups=self.norm_num_groups,
                                             fp8_convs=not (self.fp16_last_resnet and i == self.num_layers - 1)))
        self.resnets = resnets
        if self.add_upsample:
            self.upsamplers_0 = FlaxUpsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, res_hidden_states_tuple, temb):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)
            hidden_states = resnet(hidden_states, temb)
        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)
        return hidden_states


class FlaxUNetMidBlock2DCrossAttn(nn.Module):
    in_channels: int
    num_layers: int = 1
    num_attention_heads: int = 1
    dtype: jnp.dtype = jnp.float32
    transformer_layers_per_block: int = 1
    norm_num_groups: int = 32

    def setup(self):
        resnets = [FlaxResnetBlock2D(in_channels=self.in_channels, out_channels=self.in_channels,
                                     dtype=self.dtype, norm_num_groups=self.norm_num_groups)]
        attentions = []
        for _ in range(self.num_layers):
            attentions.append(FlaxTransformer2DModel(
                in_channels=self.in_channels, n_heads=self.num_attention_heads,
                d_head=self.in_channels // self.num_attention_heads,
                depth=self.transformer_layers_per_block, dtype=self.dtype,
                norm_num_groups=self.norm_num_groups))
            resnets.append(FlaxResnetBlock2D(in_channels=self.in_channels, out_channels=self.in_channels,
                                             dtype=self.dtype, norm_num_groups=self.norm_num_groups))
        self.resnets = resnets
        self.attentions = attentions

    def __call__(self, hidden_states, temb, encoder_hidden_states):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class FlaxUNet2DConditionModel(nn.Module):
    config: UNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        block_out_channels = c.block_out_channels
        time_embed_dim = block_out_channels[0] * 4
        num_attention_heads = c.attention_head_dim
        transformer_layers_per_block = c.transformer_layers_per_block

        # conv_in / conv_out: fp16 carve-outs (filter_func_unet)
        self.conv_in = nn.Conv(block_out_channels[0], kernel_size=(3, 3), strides=(1, 1),
                               padding=((1, 1), (1, 1)), dtype=self.dtype, param_dtype=self.dtype)
        self.time_proj = FlaxTimesteps(block_out_channels[0], flip_sin_to_cos=c.flip_sin_to_cos,
                                       freq_shift=c.freq_shift)
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)
        # addition_embed_type == "text_time" (SDXL)
        self.add_time_proj = FlaxTimesteps(c.addition_time_embed_dim, c.flip_sin_to_cos, c.freq_shift)
        self.add_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(c.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if down_block_type == "CrossAttnDownBlock2D":
                down_block = FlaxCrossAttnDownBlock2D(
                    in_channels=input_channel, out_channels=output_channel,
                    num_layers=c.layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i], add_downsample=not is_final_block,
                    dtype=self.dtype, norm_num_groups=c.norm_num_groups)
            else:
                down_block = FlaxDownBlock2D(
                    in_channels=input_channel, out_channels=output_channel,
                    num_layers=c.layers_per_block, add_downsample=not is_final_block,
                    dtype=self.dtype, norm_num_groups=c.norm_num_groups)
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        self.mid_block = FlaxUNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1], num_attention_heads=num_attention_heads[-1],
            transformer_layers_per_block=transformer_layers_per_block[-1], dtype=self.dtype,
            norm_num_groups=c.norm_num_groups)

        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        output_channel = reversed_block_out_channels[0]
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        for i, up_block_type in enumerate(c.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final_block = i == len(block_out_channels) - 1
            if up_block_type == "CrossAttnUpBlock2D":
                up_block = FlaxCrossAttnUpBlock2D(
                    in_channels=input_channel, out_channels=output_channel,
                    prev_output_channel=prev_output_channel, num_layers=c.layers_per_block + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    add_upsample=not is_final_block, dtype=self.dtype,
                    norm_num_groups=c.norm_num_groups)
            else:
                up_block = FlaxUpBlock2D(
                    in_channels=input_channel, out_channels=output_channel,
                    prev_output_channel=prev_output_channel, num_layers=c.layers_per_block + 1,
                    add_upsample=not is_final_block, dtype=self.dtype,
                    norm_num_groups=c.norm_num_groups,
                    fp16_last_resnet=is_final_block)   # SDXL's UpBlock2D is up_blocks.2
            up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = up_blocks

        self.conv_norm_out = GroupNorm(num_groups=c.norm_num_groups, epsilon=c.norm_eps,
                                       dtype=self.dtype, param_dtype=self.dtype)
        self.conv_out = nn.Conv(c.out_channels, kernel_size=(3, 3), strides=(1, 1),
                                padding=((1, 1), (1, 1)), dtype=self.dtype, param_dtype=self.dtype)

    def __call__(self, sample, timesteps, encoder_hidden_states, text_embeds, time_ids):
        # 1. time
        # DEVIATION (structure): broadcast the scalar timestep to the batch before
        # the embedding, as diffusers does (`timesteps.expand(sample.shape[0])`);
        # MaxDiffusion embeds a [1] timestep and relies on broadcasting at the add.
        timesteps = jnp.broadcast_to(jnp.asarray(timesteps, jnp.float32), (sample.shape[0],))
        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        # additional embeddings ("text_time")
        time_embeds = self.add_time_proj(jnp.ravel(time_ids))  # (B, 6) => (6B,) => (6B, 256)
        time_embeds = jnp.reshape(time_embeds, (text_embeds.shape[0], -1))
        add_embeds = jnp.concatenate([text_embeds, time_embeds], axis=-1)
        aug_emb = self.add_embedding(add_embeds)
        t_emb = t_emb + aug_emb

        # 2. pre-process
        sample = jnp.transpose(sample, (0, 2, 3, 1))
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if isinstance(down_block, FlaxCrossAttnDownBlock2D):
                sample, res_samples = down_block(sample, t_emb, encoder_hidden_states)
            else:
                sample, res_samples = down_block(sample, t_emb)
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, t_emb, encoder_hidden_states)

        # 5. up
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-(self.config.layers_per_block + 1):]
            down_block_res_samples = down_block_res_samples[: -(self.config.layers_per_block + 1)]
            if isinstance(up_block, FlaxCrossAttnUpBlock2D):
                sample = up_block(sample, temb=t_emb, encoder_hidden_states=encoder_hidden_states,
                                  res_hidden_states_tuple=res_samples)
            else:
                sample = up_block(sample, temb=t_emb, res_hidden_states_tuple=res_samples)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)
        sample = jnp.transpose(sample, (0, 3, 1, 2))
        return sample


# =============================================================================
# VAE decoder (MaxDiffusion vae_flax.py). Upstream class names FlaxUpsample2D /
# FlaxResnetBlock2D collide with the UNet ones, hence the FlaxVAE prefix.
# =============================================================================

def _vae_conv_mode(up_block: int, resnet: int) -> str:
    """Precision of a VAE decoder up-block resnet's convs in NVIDIA's Hopper INT8 engine.

    INT8 unless modelopt/utils.py `filter_func` disables the quantizers (decoder.up_blocks.3,
    up_blocks.2.resnets.1/2, mid_block, conv_in / conv_out / quant_conv). Of the disabled
    convs, sdxl_graphsurgeon.insert_vae_fp16_conv_cast casts up_blocks.[23].resnets.[12],
    mid_block.resnets.[01] and conv_in to fp16; the rest (post_quant_conv,
    up_blocks.3.resnets.0 incl. conv_shortcut, conv_out) stay fp32.
    """
    if up_block in (0, 1) or (up_block == 2 and resnet == 0):
        return "int8"
    if up_block == 3 and resnet == 0:
        return "fp32"
    return "fp16"


def _vae_conv(mode: str, features: int, k: int = 3, name=None):
    """VAE conv in `mode`: INT8 W8A8 (per-output-channel weights, per-tensor static
    activations, fp32 epilogue), fp16 (output cast back to fp32 by the caller) or fp32."""
    pad = ((k // 2, k // 2), (k // 2, k // 2))
    if mode == "int8":
        return q.QConv(features, (k, k), q.INT8_STATIC, padding=pad, dtype=jnp.float32, name=name)
    dt = FP16 if mode == "fp16" else jnp.float32
    return nn.Conv(features, kernel_size=(k, k), strides=(1, 1), padding=pad, dtype=dt,
                   param_dtype=dt, name=name)


class FlaxVAEUpsample2D(nn.Module):
    """Every VAE upsampler conv (up_blocks.0-2) is INT8."""
    in_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = _vae_conv("int8", self.in_channels)

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jnp.broadcast_to(hidden_states[:, :, None, :, None, :],
                                         (batch, height, 2, width, 2, channels))
        hidden_states = jnp.reshape(hidden_states, (batch, height * 2, width * 2, channels))
        hidden_states = self.conv(hidden_states)
        return hidden_states


class FlaxVAEResnetBlock2D(nn.Module):
    in_channels: int
    out_channels: int = None
    groups: int = 32
    dtype: jnp.dtype = jnp.float32
    conv_mode: str = "fp32"          # "int8" | "fp16" | "fp32" (see _vae_conv_mode)
    fp16_norm_affine: bool = False   # up_blocks: GroupNorm affine in fp16 (graphsurgeon casts)

    def setup(self):
        out_channels = self.in_channels if self.out_channels is None else self.out_channels
        ndt = FP16 if self.fp16_norm_affine else self.dtype
        self.norm1 = GroupNorm(num_groups=self.groups, epsilon=1e-6, dtype=ndt, param_dtype=ndt)
        self.conv1 = _vae_conv(self.conv_mode, out_channels)
        self.norm2 = GroupNorm(num_groups=self.groups, epsilon=1e-6, dtype=ndt, param_dtype=ndt)
        self.conv2 = _vae_conv(self.conv_mode, out_channels)
        use_nin_shortcut = self.in_channels != out_channels
        self.conv_shortcut = None
        if use_nin_shortcut:
            self.conv_shortcut = _vae_conv(self.conv_mode, out_channels, k=1)

    def __call__(self, hidden_states):
        f32 = jnp.float32
        residual = hidden_states
        hidden_states = self.norm1(hidden_states).astype(f32)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv1(hidden_states).astype(f32)
        hidden_states = self.norm2(hidden_states).astype(f32)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv2(hidden_states).astype(f32)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual).astype(f32)
        return hidden_states + residual


class FlaxAttentionBlock(nn.Module):
    """VAE mid-block attention: GroupNorm fp32; to_q/k/v/to_out.0 and both BMMs fp16
    (insert_vae_fp16_linear_casts); softmax and the residual fp32."""
    channels: int
    num_head_channels: int = None
    num_groups: int = 32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_heads = self.channels // self.num_head_channels if self.num_head_channels is not None else 1
        dense = partial(nn.Dense, self.channels, dtype=FP16, param_dtype=FP16)
        self.group_norm = GroupNorm(num_groups=self.num_groups, epsilon=1e-6, dtype=self.dtype)
        self.query, self.key, self.value = (dense(), dense(), dense())
        self.proj_attn = dense()

    def transpose_for_scores(self, projection):
        new_projection_shape = projection.shape[:-1] + (self.num_heads, -1)
        new_projection = projection.reshape(new_projection_shape)
        new_projection = jnp.transpose(new_projection, (0, 2, 1, 3))
        return new_projection

    def __call__(self, hidden_states):
        residual = hidden_states
        batch, height, width, channels = hidden_states.shape
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.reshape((batch, height * width, channels)).astype(FP16)
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attn_weights = jnp.einsum("...qc,...kc->...qk", query * scale, key * scale)
        attn_weights = nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(FP16)
        hidden_states = jnp.einsum("...kc,...qk->...qc", value, attn_weights)
        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
        new_hidden_states_shape = hidden_states.shape[:-2] + (self.channels,)
        hidden_states = hidden_states.reshape(new_hidden_states_shape)
        hidden_states = self.proj_attn(hidden_states).astype(self.dtype)
        hidden_states = hidden_states.reshape((batch, height, width, channels))
        hidden_states = hidden_states + residual
        return hidden_states


class FlaxUpDecoderBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    num_layers: int = 1
    resnet_groups: int = 32
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32
    conv_modes: tuple = ()           # per resnet (see _vae_conv_mode)

    def setup(self):
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            resnets.append(FlaxVAEResnetBlock2D(in_channels=in_channels, out_channels=self.out_channels,
                                                groups=self.resnet_groups, dtype=self.dtype,
                                                conv_mode=self.conv_modes[i], fp16_norm_affine=True))
        self.resnets = resnets
        if self.add_upsample:
            self.upsamplers_0 = FlaxVAEUpsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)
        return hidden_states


class FlaxUNetMidBlock2D(nn.Module):
    """VAE mid block: fp16 resnet convs, fp32 norms (filter_func + insert_vae_fp16_conv_cast)."""
    in_channels: int
    num_layers: int = 1
    resnet_groups: int = 32
    num_attention_heads: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = [FlaxVAEResnetBlock2D(in_channels=self.in_channels, out_channels=self.in_channels,
                                        groups=self.resnet_groups, dtype=self.dtype, conv_mode="fp16")]
        attentions = []
        for _ in range(self.num_layers):
            attentions.append(FlaxAttentionBlock(channels=self.in_channels,
                                                 num_head_channels=self.num_attention_heads,
                                                 num_groups=self.resnet_groups, dtype=self.dtype))
            resnets.append(FlaxVAEResnetBlock2D(in_channels=self.in_channels,
                                                out_channels=self.in_channels,
                                                groups=self.resnet_groups, dtype=self.dtype,
                                                conv_mode="fp16"))
        self.resnets = resnets
        self.attentions = attentions

    def __call__(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)
        return hidden_states


class FlaxDecoder(nn.Module):
    config: VAEConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        block_out_channels = c.block_out_channels
        self.conv_in = _vae_conv("fp16", block_out_channels[-1])
        self.mid_block = FlaxUNetMidBlock2D(in_channels=block_out_channels[-1],
                                            resnet_groups=c.norm_num_groups,
                                            num_attention_heads=None, dtype=self.dtype)
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        up_blocks = []
        for i, _ in enumerate(c.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            up_blocks.append(FlaxUpDecoderBlock2D(
                in_channels=prev_output_channel, out_channels=output_channel,
                num_layers=c.layers_per_block + 1, resnet_groups=c.norm_num_groups,
                add_upsample=not is_final_block, dtype=self.dtype,
                conv_modes=tuple(_vae_conv_mode(i, j) for j in range(c.layers_per_block + 1))))
            prev_output_channel = output_channel
        self.up_blocks = up_blocks
        self.conv_norm_out = GroupNorm(num_groups=c.norm_num_groups, epsilon=1e-6, dtype=FP16,
                                       param_dtype=FP16)
        self.conv_out = _vae_conv("fp32", c.out_channels)

    def __call__(self, sample):
        sample = self.conv_in(sample.astype(FP16)).astype(jnp.float32)
        sample = self.mid_block(sample)
        for block in self.up_blocks:
            sample = block(sample)
        sample = self.conv_norm_out(sample).astype(jnp.float32)
        sample = nn.swish(sample)
        sample = self.conv_out(sample)
        return sample


class FlaxAutoencoderKL(nn.Module):
    """FlaxAutoencoderKL restricted to `decode` (text-to-image never runs the encoder).
    fp32 I/O (the INT8 engine is built strongly typed from an fp32 ONNX)."""
    config: VAEConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.decoder = FlaxDecoder(self.config, dtype=self.dtype)
        self.post_quant_conv = _vae_conv("fp32", self.config.latent_channels, k=1)

    def __call__(self, latents):
        # == FlaxAutoencoderKL.decode(latents).sample
        latents = jnp.transpose(latents, (0, 2, 3, 1))
        hidden_states = self.post_quant_conv(latents)
        hidden_states = self.decoder(hidden_states)
        hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))
        return hidden_states
