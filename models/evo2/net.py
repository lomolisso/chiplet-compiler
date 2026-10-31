"""Evo 2 7B (StripedHyena 2), Flax linen port of Vortex (Zymrael/vortex).

Line-by-line port of `vortex/model/{model,engine,layers,attention,rotary,utils}.py` at the
pinned commit, inference path with `inference_params` (what `Generator.generate` runs):

* prefill (`state is None`): `HyenaCascade.parallel_forward` through
  `HyenaInferenceEngine.parallel_fir` (featurizer short conv, HCS conv, HCM FFT conv) and
  `parallel_iir` (HCL FFT long conv + `prefill_via_modal_fft` for the modal state), MHA via
  vortex's torch `CrossAttention` over a fresh KV cache;
* decode (`state` given): `HyenaCascade.sequential_forward` (`step_fir` / `step_iir`) and
  one-token MHA at a runtime cache offset.

Module and parameter names follow vortex (`blocks_<i>/filter/log_poles`, ...). dtype casts
mirror vortex's (bf16 parameters and activations, fp32 FIR/FFT math, fp32 poles/residues,
fp32 recurrent state).

Quantization: the official evo2_7b inference path (evo2-7b-1m.yml
`use_fp8_input_projections: True`, vortex @ 8b00afe with Transformer Engine): the 27 Hyena
input `projections` are TE `Linear`s run under `fp8_autocast` with
`DelayedScaling(fp8_format=HYBRID, amax_history_len=16, amax_compute_algo="max")`. In
inference HYBRID means E4M3 forward tensors, so each projection is an FP8 E4M3 x E4M3 GEMM
with fp32 accumulation and a bf16 output, one per-tensor scale for the weight and one for
the input. The delayed scale the GEMM uses is fixed for the call (448 / max of the amax
history of earlier calls; the amax bookkeeping runs outside the GEMM), so it is a static
per-tensor `input_scale` here (`models.quant.FP8_STATIC`). TE casts the bf16 master weight
to FP8 each call with its own delayed scale; the port stores the FP8 weight and its
`kernel_scale` (same GEMM). vortex pads the projection input to a multiple of 16 rows for
the FP8 kernel (decode: 1 -> 16 rows); that is a kernel-alignment artifact and is not
modelled. Everything else (out_filter_dense, MLP, MHA Wqkv / out_proj, attention, unembed)
stays bf16 and the FIR / FFT / IIR math fp32, as in vortex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp
from jax import lax

from models import quant as q

F32 = jnp.float32


@dataclass(frozen=True)
class Config:
    # evo2/configs/evo2-7b-1m.yml (checked against the file in reference.vortex_config)
    vocab_size: int = 512
    hidden_size: int = 4096
    num_layers: int = 32
    hcs_layer_idxs: tuple = (0, 4, 7, 11, 14, 18, 21, 25, 28)
    hcm_layer_idxs: tuple = (1, 5, 8, 12, 15, 19, 22, 26, 29)
    hcl_layer_idxs: tuple = (2, 6, 9, 13, 16, 20, 23, 27, 30)
    attn_layer_idxs: tuple = (3, 10, 17, 24, 31)
    hcm_filter_length: int = 128
    hcm_filter_groups: int = 256
    hcs_filter_length: int = 7
    hcs_filter_groups: int = 256
    short_filter_length: int = 3
    num_attention_heads: int = 32
    eps: float = 1e-6
    state_size: int = 16
    rotary_emb_base: float = 10000
    rotary_emb_scaling_factor: float = 128
    inner_mlp_size: int = 11264
    # scenario
    batch: int = 1
    prompt_len: int = 8192           # prefill length (nt)
    n_tokens: int = 500              # Evo2.generate default; 1 from prefill + 499 decode steps
    decode_offset: int | None = None  # example decode offset (None: last cache slot)
    # FFT size of the Hyena long convolutions (None: 2 * prompt_len, as vortex). Any size
    # >= 2 * L gives the same linear convolution up to rounding; fixing it to an FFT
    # capacity lets the prompt length L can be symbolic (StableHLO's fft takes a static
    # length). Resolved at construction, so it stays fixed when prompt_len changes.
    fft_size: int | None = None
    # KV-cache capacity (None: Generator.generate's `tot_length = prompt + num_tokens`),
    # resolved at construction like fft_size: its own size, not recomputed from prompt_len.
    max_seqlen: int | None = None
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.fft_size is None:
            object.__setattr__(self, "fft_size", 2 * self.prompt_len)
        if self.max_seqlen is None:
            object.__setattr__(self, "max_seqlen", self.prompt_len + self.n_tokens)

    @property
    def jdtype(self):
        return jnp.dtype(self.dtype)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def fft_len(self) -> int:
        return self.fft_size

    def kind(self, i: int) -> str:
        """Block type of layer i: "hcs" / "hcm" / "hcl" / "attn" (KeyError if unlisted)."""
        return {j: k for k in ("hcs", "hcm", "hcl", "attn")
                for j in getattr(self, f"{k}_layer_idxs")}[i]


# ---------------------------------------------------------------------------
# helpers (vortex.model.utils.interleave, F.conv1d, repeat_interleave, F.gelu)
# ---------------------------------------------------------------------------

def interleave(z, axis: int):
    """cat([z[0::3], z[1::3], z[2::3]]) along the channel axis."""
    n = z.shape[axis]
    parts = [lax.slice_in_dim(z, s, n, stride=3, axis=axis) for s in range(3)]
    return jnp.concatenate(parts, axis=axis)


def depthwise_conv1d(x_ncl, weight, padding: int):
    """F.conv1d(x, weight, padding=padding, groups=C), weight [C, 1, K]."""
    return lax.conv_general_dilated(
        x_ncl, weight, window_strides=(1,), padding=[(padding, padding)],
        dimension_numbers=("NCH", "OIH", "NCH"), feature_group_count=x_ncl.shape[1])


def repeat_interleave0(h, repeats: int):
    """torch `h.repeat_interleave(repeats, 0)`."""
    return jnp.broadcast_to(h[:, None], (h.shape[0], repeats, *h.shape[1:])).reshape(
        h.shape[0] * repeats, *h.shape[1:])


def cdiv(z, n: int):
    """Complex tensor / real scalar, written per component: the graph builder cannot
    serialize complex-valued constants (MLIR bindings fail on dense<complex>)."""
    return lax.complex(jnp.real(z) / n, jnp.imag(z) / n)


def gelu(x):
    return nn.gelu(x.astype(F32), approximate=False).astype(x.dtype)


# ---------------------------------------------------------------------------
# layers.py
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """vortex RMSNorm: scale * x / (||x||_2 * d^-1/2 + eps)."""
    cfg: Config

    @nn.compact
    def __call__(self, x):
        c = self.cfg
        scale = self.param("scale", nn.initializers.ones, (c.hidden_size,), c.jdtype)
        norm = jnp.sqrt(jnp.sum(jnp.square(x.astype(F32)), axis=-1, keepdims=True)).astype(x.dtype)
        y = x / (norm * c.hidden_size ** (-1.0 / 2) + c.eps)
        return scale * y


class AddmmLinear(nn.Module):
    """nn.Linear with bias as torch runs it when the input is contiguous: `F.linear` folds
    the batch dims into one `addmm`, which adds the bias to the fp32 accumulator and rounds
    once. (Non-contiguous inputs take matmul + add, two roundings: flax `nn.Dense`.)"""
    features: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", nn.initializers.lecun_normal(), (x.shape[-1], self.features),
                            self.dtype)
        bias = self.param("bias", nn.initializers.zeros, (self.features,), self.dtype)
        y = lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())),
                            preferred_element_type=F32)
        return (y + bias.astype(F32)).astype(self.dtype)


def dense(cfg: Config, features: int, use_bias: bool, name: str, addmm: bool = False):
    if addmm:
        return AddmmLinear(features, cfg.jdtype, name=name)
    return nn.Dense(features, use_bias=use_bias, dtype=cfg.jdtype, param_dtype=cfg.jdtype,
                    name=name)


class ParallelGatedMLP(nn.Module):
    cfg: Config
    layer_idx: int

    @nn.compact
    def __call__(self, z):
        c = self.cfg
        z1 = dense(c, c.inner_mlp_size, False, "l1")(z)
        z2 = dense(c, c.inner_mlp_size, False, "l2")(z)
        # evo2_style_activations: gelu in layer 0, identity afterwards
        a = gelu(z1) if self.layer_idx == 0 else z1
        return dense(c, c.hidden_size, False, "l3")(a * z2)


class TiedUnembed(nn.Module):
    """`Lambda(embedding_layer.unembed)`: u @ weight.T with the embedding table."""

    @nn.compact
    def __call__(self, u, weight):
        return jnp.einsum("bld,vd->blv", u, weight)


# ---------------------------------------------------------------------------
# engine.py: HyenaInferenceEngine
# ---------------------------------------------------------------------------

def step_fir(u, fir_state, weight, bias=None, gated_bias=False, flip_filter=False):
    """HyenaInferenceEngine.step_fir (steady state: cache holds filter_length - 1 inputs)."""
    weight = jnp.squeeze(weight)
    cache_size = fir_state.shape[-1]
    if flip_filter:
        weight = jnp.flip(weight, -1)[..., -cache_size - 1:][None]
    else:
        weight = weight[..., :cache_size + 1][None]
    input_dtype = u.dtype
    weight = weight.astype(F32)
    u = u.astype(F32)
    fir_state = fir_state.astype(F32)
    h0, h = weight[..., -1], weight[..., :-1]
    y = h0 * u + jnp.sum(fir_state * h, axis=-1)
    if bias is not None:
        bias = bias.astype(F32)
        y = y + bias * u if gated_bias else y + bias
    # torch.roll(fir_state, -1, dims=2); fir_state[..., -1] = u
    fir_state = jnp.concatenate([fir_state[..., 1:], u[..., None]], axis=-1)
    return y.astype(input_dtype), fir_state


def step_iir(x2, x1, v, D, residues, log_poles, iir_state):
    """HyenaInferenceEngine.step_iir (modal recurrence, iir_groups=1)."""
    x1v = x1 * v
    poles = jnp.exp(log_poles)[..., 0][None]
    residues = residues[None]
    iir_state = poles * iir_state + x1v[..., None]
    res_state = jnp.sum(residues * iir_state, axis=-1)
    y = x2 * (res_state + D * x1v)
    return y, iir_state


def fftconv_func(u, k, D, fft_size):
    """engine.fftconv_func (non-bidirectional), u [B, D, L] fp32, k [D, 1, L] fp32.
    fft_size >= 2L (vortex: 2L; see Config.fft_size)."""
    seqlen = u.shape[-1]
    k_f = cdiv(jnp.fft.rfft(k, n=fft_size), fft_size)
    k_f = jnp.squeeze(k_f)[None]                    # adjust_filter_shape_for_broadcast
    u_f = jnp.fft.rfft(u.astype(k.dtype), n=fft_size)
    y = jnp.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    out = y + u * D[:, None]
    return out.astype(u.dtype)


# ---------------------------------------------------------------------------
# model.py: HyenaCascade
# ---------------------------------------------------------------------------

class HyenaCascade(nn.Module):
    """kind: "hcs" (explicit short FIR, groups `hcs_filter_groups`), "hcm" (explicit medium
    FIR via FFT, gated bias D), "hcl" (implicit modal long filter from poles/residues)."""
    cfg: Config
    kind: str

    @nn.compact
    def __call__(self, u, state=None):
        c = self.cfg
        pd, D_, S = c.jdtype, c.hidden_size, c.state_size
        K = c.short_filter_length
        p = {"short_w": self.param("short_filter_weight", nn.initializers.normal(1.0),
                                   (3 * D_, 1, K), pd)}
        if self.kind == "hcl":
            p["log_poles"] = self.param("log_poles", nn.initializers.normal(1.0), (D_, S, 1), F32)
            p["residues"] = self.param("residues", nn.initializers.normal(1.0), (D_, S), F32)
            p["D"] = self.param("D", nn.initializers.zeros, (D_,), pd)
        else:
            medium = self.kind == "hcm"
            groups = c.hcm_filter_groups if medium else c.hcs_filter_groups
            flen = c.hcm_filter_length if medium else c.hcs_filter_length
            h = self.param("h", nn.initializers.normal(1.0), (groups, 1, flen), pd)
            p["h"] = repeat_interleave0(h, D_ // groups)          # hyena_filter_groups > 1
            # fir_inner_filter_length >= 128 has the (gated) bias D; < 128 has D = None
            p["D"] = self.param("D", nn.initializers.zeros, (D_,), pd) if medium else None
        if state is None:
            return self.parallel_forward(u, **p)
        return self.sequential_forward(u, state, **p)

    def parallel_forward(self, u, short_w, D, h=None, log_poles=None, residues=None):
        c = self.cfg
        L = u.shape[1]
        K = c.short_filter_length
        # engine.parallel_fir(gate=False, dim_last=True): featurizer short conv
        u = jnp.swapaxes(u, 1, 2)                                   # B, 3D, L
        z_pre = depthwise_conv1d(u.astype(F32), short_w.astype(F32), K - 1)[..., :L]
        z_pre = z_pre.astype(u.dtype)
        state = {"fir_state": u[..., -K + 1:]}
        z_pre = interleave(z_pre, axis=1)
        x2, x1, v = jnp.split(z_pre, 3, axis=1)
        if self.kind == "hcl":
            y, state["iir_state"] = self.parallel_iir(x2, x1, v, D, log_poles, residues, L)
        else:
            # engine.parallel_fir(gate=True, dim_last=False)
            uu = x1 * v
            flen = h.shape[-1]
            if self.kind == "hcm":        # fir_length >= 128: fftconv_func, bias = D (gated)
                z = fftconv_func(uu.astype(F32), h[:, :, :L].astype(F32), D, c.fft_len)
                z = z.astype(uu.dtype)
            else:                         # depthwise conv1d in fp32, no bias
                z = depthwise_conv1d(uu.astype(F32), h.astype(F32), flen - 1)[..., :L]
                z = z.astype(uu.dtype)
            y = x2 * z
            state["fir_inner_state"] = uu[..., -flen + 1:]
        return jnp.swapaxes(y, 1, 2), state

    def parallel_iir(self, x2, x1, v, D, log_poles, residues, L):
        """engine.parallel_iir (prefill_style "fft") + prefill_via_modal_fft."""
        fft_size = self.cfg.fft_len                                 # >= 2L (vortex: 2L)
        t = jnp.arange(L, dtype=F32)[None, None]                  # HyenaCascade.update_time
        # HyenaCascade.compute_filter: h = sum_s R_s exp(log_pole_s * t), fp32 [1, D, L]
        h = jnp.sum(residues[..., None] * jnp.exp(log_poles * t), axis=1)[None]
        x1v = x1 * v
        H = cdiv(jnp.fft.rfft(h, n=fft_size), fft_size)
        X_s = jnp.fft.fft(x1v.astype(F32), n=fft_size)
        X = X_s[..., :H.shape[-1]]
        y = jnp.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]
        y = y.astype(x1v.dtype)
        y = (y + x1v * D[:, None]) * x2
        # prefill_via_modal_fft: state_d,s = sum_j x1v_j exp(log_pole_s (L-1-j))
        state_s = jnp.exp(log_poles.astype(F32) * t)                # D, S, L
        state_S = jnp.fft.fft(state_s, n=fft_size)
        state_S = jnp.broadcast_to(state_S[None], (x1v.shape[0], *state_S.shape))
        st = jnp.fft.ifft(X_s[..., None, :] * state_S, n=fft_size)
        last = lax.slice_in_dim(st, L - 1, L, axis=st.ndim - 1)[..., 0]  # st[..., L - 1]
        return y, jnp.real(last).astype(F32)

    def sequential_forward(self, u, state, short_w, D, h=None, log_poles=None, residues=None):
        data_dtype = u.dtype
        u = u[:, -1]
        z_pre, fir_state = step_fir(u, state["fir_state"], short_w)
        new = {"fir_state": fir_state}
        z_pre = interleave(z_pre, axis=-1)
        x2, x1, v = jnp.split(z_pre, 3, axis=1)
        if self.kind == "hcl":
            y, new["iir_state"] = step_iir(x2, x1, v, D, residues, log_poles, state["iir_state"])
        else:
            medium = self.kind == "hcm"
            y, new["fir_inner_state"] = step_fir(x1 * v, state["fir_inner_state"], h, bias=D,
                                                 flip_filter=medium, gated_bias=medium)
            y = y * x2
        y = y.astype(data_dtype)
        return y[:, None], new


class ParallelGatedConvBlock(nn.Module):
    cfg: Config
    layer_idx: int

    @nn.compact
    def __call__(self, u, offset=None, state=None):
        c = self.cfg
        # TE FP8 Linear (use_fp8_input_projections): E4M3 W8A8, per-tensor delayed scales
        z = q.QDense(3 * c.hidden_size, q.FP8_STATIC, use_bias=False, dtype=c.dtype,
                     name="projections")(RMSNorm(c, name="pre_norm")(u))
        decode = state is not None
        z, state = HyenaCascade(c, c.kind(self.layer_idx), name="filter")(z, state)
        # prefill: z is the permuted filter output (non-contiguous) -> matmul + add;
        # decode: z = y[:, None] is contiguous -> addmm
        z_in = dense(c, c.hidden_size, True, "out_filter_dense", addmm=decode)(z) + u
        mlp = ParallelGatedMLP(c, self.layer_idx, name="mlp")
        y = mlp(RMSNorm(c, name="post_norm")(z_in)) + z_in
        return y, state


# ---------------------------------------------------------------------------
# attention.py / rotary.py
# ---------------------------------------------------------------------------

class LinearlyScaledRotaryEmbedding(nn.Module):
    """positional_embeddings.LinearlyScaledRotaryEmbedding (+ the rotary kernel), applied to
    q and k of qkv [B, T, 3, H, hd] at positions offset .. offset + T - 1."""
    cfg: Config

    @nn.compact
    def __call__(self, qkv, offset):
        c = self.cfg
        dim, dtype = c.head_dim, qkv.dtype
        inv_freq = 1.0 / (float(c.rotary_emb_base) ** (jnp.arange(0, dim, 2, dtype=F32) / dim))
        # the inv_freq buffer is cast with the MHA (to(dtype)), then back to fp32
        inv_freq = inv_freq.astype(dtype).astype(F32)
        T = qkv.shape[1]
        t = (offset + jnp.arange(T, dtype=jnp.int32)).astype(F32) / float(c.rotary_emb_scaling_factor)
        freqs = t[:, None] * inv_freq[None, :]
        cos = jnp.cos(freqs).astype(dtype).astype(F32)[None, :, None, None, :]
        sin = jnp.sin(freqs).astype(dtype).astype(F32)[None, :, None, None, :]
        qk = qkv[:, :, :2].astype(F32)
        half = dim // 2
        x0, x1 = qk[..., :half], qk[..., half:]
        rot = jnp.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1).astype(dtype)
        return jnp.concatenate([rot, qkv[:, :, 2:]], axis=2)


class CrossAttention(nn.Module):
    """attention.CrossAttention (causal, softmax_scale 1/sqrt(d)); `n_keys` is vortex's
    `seqlen_k` (valid cached keys), which masks the unused cache slots in decode."""

    @nn.compact
    def __call__(self, q, kv, n_keys):
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
        k, v = kv[:, :, 0], kv[:, :, 1]
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = jnp.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        row_idx = jnp.arange(seqlen_q)[:, None]
        col_idx = jnp.arange(seqlen_k)
        causal_mask = col_idx > row_idx + n_keys - seqlen_q
        scores = jnp.where(causal_mask, jnp.asarray(-10000.0, scores.dtype), scores)
        attention = nn.softmax(scores.astype(F32), axis=-1).astype(v.dtype)
        return jnp.einsum("bhts,bshd->bthd", attention, v)


class MHA(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, offset, kv_cache):
        """prefill: offset None and `kv_cache` the fresh zero cache of capacity max_seqlen
        (vortex allocates it in InferenceParams; here it is an input, so its capacity is an
        input size); decode: runtime int32 offset and the cache."""
        c = self.cfg
        B, T, _ = x.shape
        H, hd = c.num_attention_heads, c.head_dim
        qkv = dense(c, 3 * c.hidden_size, False, "Wqkv")(x)
        qkv = qkv.reshape(B, T, 3, H, hd)                           # (three h d)
        if offset is None:          # prefill: seqlen_offset 0, fresh (zero) cache
            qkv = LinearlyScaledRotaryEmbedding(c, name="rotary_emb")(qkv, 0)
            kv_cache = lax.dynamic_update_slice(kv_cache, qkv[:, :, 1:], (0, 0, 0, 0, 0))
            kv = kv_cache[:, :T]                                    # kv_cache[:, :sequence_end]
            n_keys = T
        else:                       # decode: write slot `offset`, attend over the cache
            qkv = LinearlyScaledRotaryEmbedding(c, name="rotary_emb")(qkv, offset)
            kv_cache = lax.dynamic_update_slice(kv_cache, qkv[:, :, 1:].astype(kv_cache.dtype),
                                                (0, offset, 0, 0, 0))
            kv = kv_cache
            n_keys = offset + T
        context = CrossAttention(name="inner_cross_attn")(qkv[:, :, 0], kv, n_keys)
        # rearrange(context, "... h d -> ... (h d)") copies to a contiguous tensor -> addmm
        out = dense(c, c.hidden_size, True, "out_proj", addmm=True)(
            context.reshape(B, T, c.hidden_size))
        return out, kv_cache


class AttentionBlock(nn.Module):
    cfg: Config
    layer_idx: int

    @nn.compact
    def __call__(self, u, offset=None, state=None):
        c = self.cfg
        out, kv_cache = MHA(c, name="inner_mha_cls")(
            RMSNorm(c, name="pre_norm")(u), offset, state["kv_cache"])
        u = out + u
        u = ParallelGatedMLP(c, self.layer_idx, name="mlp")(RMSNorm(c, name="post_norm")(u)) + u
        return u, {"kv_cache": kv_cache}


# ---------------------------------------------------------------------------
# StripedHyena
# ---------------------------------------------------------------------------

class StripedHyena(nn.Module):
    cfg: Config

    def setup(self):
        c = self.cfg
        self.embedding_layer = nn.Embed(c.vocab_size, c.hidden_size, dtype=c.jdtype,
                                        param_dtype=c.jdtype, name="embedding_layer")
        self.blocks = [(AttentionBlock if c.kind(i) == "attn" else ParallelGatedConvBlock)(
            c, i, name=f"blocks_{i}") for i in range(c.num_layers)]
        self.norm = RMSNorm(c, name="norm")
        self.unembed = TiedUnembed(name="unembed")

    def __call__(self, input_ids, offset=None, state=None):
        """prefill: (ids [B, L], None, {attention block: {"kv_cache": zero cache}})
        -> (logits [B, L, V], states);
        decode: (ids [B, 1], offset int32, states) -> (logits [B, 1, V], states)."""
        x = self.embedding_layer(input_ids)
        new_state = {}
        for i, block in enumerate(self.blocks):
            name = f"blocks_{i}"
            x, new_state[name] = block(x, offset, state.get(name))
        x = self.norm(x)
        return self.unembed(x, self.embedding_layer.embedding), new_state
