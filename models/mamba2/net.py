"""Mamba-2 (state-spaces/mamba2-2.7b) causal LM, FP8 Flax linen port of HF transformers.

Line-by-line port of `transformers.models.mamba2.modeling_mamba2` (v5.20.2), pure
torch path (`mamba2_chunk_scan` for multi-token input, `mamba2_selective_state_update`
+ `causal_conv1d_update` for single-token decode with a cache). Module and parameter
names follow HF (`backbone/layers_<i>/mixer/in_proj`, ...). dtype casts mirror HF's
mixed-precision behaviour exactly:

* residual stream in fp32 (`residual_in_fp32`), RMSNorm math in fp32;
* the SSD scan runs in fp32 (x, B, C upcast), dt/softplus in the param dtype;
* the cached conv state has the activation dtype, the cached SSM state is fp32.

Prefill uses the chunked SSD algorithm with every chunk processed at once
(einsums over a chunk axis) and the inter-chunk recurrence in matrix form:
`new_states = exp(segsum(pad(A_cumsum[..., -1])))^T @ [0; chunk_states]`, a
(c+1)x(c+1) lower-triangular decay matrix contracted over the source chunk.
No loop, no while. Logits are computed for the last position only (`generate()` /
vLLM `logits_to_keep=1`; HF `model(input_ids)` would project every position).

Quantization (GPU-datacenter fallback: no official or widely deployed quantized
Mamba-2 exists and TensorRT-LLM serves Mamba-2 in FP16/BF16 only): FP8 E4M3 W8A8 with
per-tensor static scales on the mixer `in_proj` and `out_proj` -- NVIDIA's own FP8 form
for Mamba-2 mixers (ModelOpt FP8_DEFAULT_CFG as shipped in
nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8 and served by vLLM):
  * weights fp8 e4m3 with one fp32 scale per tensor; inputs quantized with a calibrated
    (max) per-tensor scale; f8 x f8 -> f32 GEMM, bf16 output;
  * float as in the recipe: embedding and tied lm_head (bf16; ModelOpt disables
    `*lm_head*` / nn.Embedding), depthwise conv1d + SiLU (bf16; ModelOpt disables
    `*mixer.conv1d*`), dt / softplus (bf16), SSD scan, A, D, recurrence and RMSNorms
    (fp32), decode bmm(state, C) (bf16), conv state (bf16) and SSM state (fp32).
See models/mamba2/recipe.py for the sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp
from jax import lax

from models import quant as q

F32 = jnp.float32


@dataclass(frozen=True)
class Config:
    vocab_size: int = 50288          # 50277 padded to a multiple of 16 (HF config)
    hidden_size: int = 2560
    state_size: int = 128
    num_heads: int = 80
    head_dim: int = 64
    expand: int = 2
    n_groups: int = 1
    conv_kernel: int = 4
    chunk_size: int = 256
    num_hidden_layers: int = 64
    layer_norm_epsilon: float = 1e-5
    time_step_limit: tuple[float, float] = (0.0, float("inf"))
    use_bias: bool = False
    use_conv_bias: bool = True
    residual_in_fp32: bool = True
    tie_word_embeddings: bool = True
    batch: int = 1
    seq_len: int = 2048              # prefill length
    decode_steps: int = 128          # trip count of decode_step
    dtype: str = "bfloat16"

    @property
    def intermediate_size(self) -> int:
        return int(self.expand * self.hidden_size)

    @property
    def conv_dim(self) -> int:
        return self.intermediate_size + 2 * self.n_groups * self.state_size

    @property
    def jdtype(self):
        return jnp.dtype(self.dtype)


# ---------------------------------------------------------------------------
# helpers (HF: pad_tensor_by_size, reshape_into_chunks, segment_sum, F.softplus)
# ---------------------------------------------------------------------------

def softplus(x):
    """torch.nn.functional.softplus (beta=1, threshold=20)."""
    return jnp.where(x > 20.0, x, jnp.log1p(jnp.exp(x)))


def pad_tensor_by_size(x, pad_size: int):
    """Zero-pad the seq dim (axis 1) at the end."""
    pad = [(0, 0)] * x.ndim
    pad[1] = (0, pad_size)
    return jnp.pad(x, pad)


def reshape_into_chunks(x, pad_size, chunk_size: int, n_chunks):
    x = pad_tensor_by_size(x, pad_size)
    return x.reshape(x.shape[0], n_chunks, chunk_size, *x.shape[2:])


def repeat_interleave_heads(x, num_heads: int):
    """torch `x.repeat_interleave(num_heads // g, dim=2)` for x [b, L, g, n] (expand + view)."""
    b, L, g, n = x.shape
    x = jnp.broadcast_to(x[:, :, :, None, :], (b, L, g, num_heads // g, n))
    return x.reshape(b, L, num_heads, n)


def segment_sum(x):
    """[..., T] -> [..., T, T]; out[..., i, j] = sum_{j < k <= i} x[k], -inf above diag."""
    T = x.shape[-1]
    x = jnp.broadcast_to(x[..., None], (*x.shape, T))
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x = jnp.where(mask, x, 0)
    seg = jnp.cumsum(x, axis=-2)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    return jnp.where(mask, seg, -jnp.inf)


def depthwise_conv1d(x_ncl, weight, bias, padding: int):
    """F.conv1d(x, weight, bias, padding=padding, groups=C); weight [C, 1, K]."""
    out = lax.conv_general_dilated(
        x_ncl, weight, window_strides=(1,), padding=[(padding, padding)],
        dimension_numbers=("NCH", "OIH", "NCH"), feature_group_count=x_ncl.shape[1])
    if bias is not None:
        out = out + bias[None, :, None]
    return out


# ---------------------------------------------------------------------------
# SSD chunk scan (HF mamba2_chunk_scan) and recurrent step (mamba2_selective_state_update)
# ---------------------------------------------------------------------------

def mamba2_chunk_scan(hidden_states, dt, A, B, C, chunk_size, D, dt_bias, dt_limit):
    """hidden_states [b,L,h,p], dt [b,L,h], A [h] fp32, B/C [b,L,g,n].

    Returns (y [b,L,h,p] fp32, final_state [b,h,p,n] fp32).
    """
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    num_groups = B.shape[2]

    dt = dt + dt_bias.astype(dt.dtype)
    dt = softplus(dt)
    dt = jnp.clip(dt, dt_limit[0], dt_limit[1])

    hidden_states = hidden_states.astype(F32)
    B = repeat_interleave_heads(B.astype(F32), num_heads)
    C = repeat_interleave_heads(C.astype(F32), num_heads)

    # HF: pad_size = (chunk_size - seq_len % chunk_size) % chunk_size. The same value, written
    # through the chunk count so that a symbolic seq_len reshapes provably.
    n_chunks = -(-seq_len // chunk_size)
    pad_size = n_chunks * chunk_size - seq_len
    D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

    # Discretize x and A
    hidden_states = hidden_states * dt[..., None].astype(F32)
    A = A.astype(hidden_states.dtype) * dt.astype(F32)

    # b c l h p / b c l h / b c l h n
    hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, chunk_size, n_chunks)
                              for t in (hidden_states, A, B, C)]

    A = jnp.transpose(A, (0, 3, 1, 2))                       # b h c l
    A_cumsum = jnp.cumsum(A, axis=-1)

    # 1. intra-chunk (diagonal blocks)
    L = jnp.exp(segment_sum(A))                              # b h c l s
    G = jnp.einsum("bclhn,bcshn->bclsh", C, B)
    M = G * jnp.transpose(L, (0, 2, 3, 4, 1))
    Y_diag = jnp.einsum("bclsh,bcshp->bclhp", M, hidden_states)

    # 2. per-chunk states (B terms)
    decay_states = jnp.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * jnp.transpose(decay_states, (0, 2, 3, 1))[..., None]
    states = jnp.einsum("bclhn,bclhp->bchpn", B_decay, hidden_states)

    # 3. inter-chunk recurrence in matrix form (A terms): zero initial state
    previous_states = jnp.zeros_like(states[:, :1])
    states = jnp.concatenate([previous_states, states], axis=1)          # b (c+1) h p n
    A_last = jnp.pad(A_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0)))    # b h (c+1)
    decay_chunk = jnp.swapaxes(jnp.exp(segment_sum(A_last)), 1, 3)       # b z c h
    new_states = jnp.einsum("bzch,bzhpn->bchpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. state -> output (C terms)
    state_decay_out = jnp.exp(A_cumsum)
    C_times_states = jnp.einsum("bclhn,bchpn->bclhp", C, states)
    Y_off = C_times_states * jnp.transpose(state_decay_out, (0, 2, 3, 1))[..., None]

    output = (Y_diag + Y_off).reshape(batch_size, -1, num_heads, head_dim)
    output = output + D_residual
    output = output[:, :seq_len]                             # cut padded chunks (no-op if 0)
    return output, final_state


def mamba2_selective_state_update(state, hidden_states, dt, A, B, C, D, dt_bias):
    """state [b,h,p,n] fp32, hidden_states/dt [b,h,p], A [h,p,n], B/C [b,g,n], D/dt_bias [h,p]."""
    batch_size, num_heads, head_dim = hidden_states.shape
    num_groups, state_size = B.shape[1], B.shape[-1]

    dt = dt + dt_bias.astype(dt.dtype)
    dt = softplus(dt)
    dt = dt[..., None]

    dA = jnp.exp(dt.astype(F32) * A.astype(F32))

    B = B.reshape(batch_size, num_groups, 1, state_size)
    B = jnp.broadcast_to(B, (batch_size, num_groups, num_heads // num_groups, state_size))
    B = B.reshape(batch_size, num_heads, 1, state_size)
    dB = dt * B
    dBx = dB * hidden_states[..., None]

    ssm_states = state * dA + dBx
    new_state = ssm_states.astype(state.dtype)

    C = C.reshape(batch_size, num_groups, 1, state_size)
    C = jnp.broadcast_to(C, (batch_size, num_groups, num_heads // num_groups, state_size))
    C = C.reshape(batch_size, num_heads, state_size)

    ssm_states = ssm_states.astype(C.dtype)
    out = jnp.matmul(ssm_states.reshape(batch_size * num_heads, head_dim, state_size),
                     C.reshape(batch_size * num_heads, state_size, 1))
    out = out.reshape(batch_size, num_heads, head_dim)
    out = (out + hidden_states * D).astype(out.dtype)
    return out.astype(hidden_states.dtype), new_state


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------

class Mamba2RMSNorm(nn.Module):
    eps: float
    param_dtype: jnp.dtype = F32

    @nn.compact
    def __call__(self, hidden_states):
        weight = self.param("weight", nn.initializers.ones, (hidden_states.shape[-1],),
                            self.param_dtype)
        input_dtype = hidden_states.dtype
        h = hidden_states.astype(F32)
        variance = jnp.mean(h ** 2, axis=-1, keepdims=True)
        h = h * lax.rsqrt(variance + self.eps)
        return weight * h.astype(input_dtype)


class MambaRMSNormGated(nn.Module):
    """norm(x * silu(gate)): gate applied BEFORE the norm (norm_before_gate=False)."""
    eps: float
    param_dtype: jnp.dtype = F32

    @nn.compact
    def __call__(self, hidden_states, gate):
        weight = self.param("weight", nn.initializers.ones, (hidden_states.shape[-1],),
                            self.param_dtype)
        input_dtype = hidden_states.dtype
        h = hidden_states.astype(F32)
        h = h * nn.silu(gate.astype(F32))
        variance = jnp.mean(h ** 2, axis=-1, keepdims=True)
        h = h * lax.rsqrt(variance + self.eps)
        return weight * h.astype(input_dtype)


class Conv1d(nn.Module):
    """HF `mixer.conv1d` (nn.Conv1d, depthwise, weight [C, 1, K]). HF never calls its forward;
    it applies `F.conv1d(x, conv1d.weight, conv1d.bias, padding=p, groups=C)` functionally,
    with p = K-1 for prefill (`causal_conv1d_fn`) and p = 0 for decode (`causal_conv1d_update`)."""
    features: int
    kernel_size: int
    use_bias: bool
    param_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x_ncl, padding: int):
        kernel = self.param("kernel", nn.initializers.lecun_normal(),
                            (self.features, 1, self.kernel_size), self.param_dtype)
        bias = (self.param("bias", nn.initializers.zeros, (self.features,), self.param_dtype)
                if self.use_bias else None)
        return depthwise_conv1d(x_ncl, kernel, bias, padding)


class Mamba2Mixer(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden_states, conv_state=None, ssm_state=None):
        """Prefill when the states are None (chunk form), else one decode step."""
        c = self.cfg
        pd = c.jdtype
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        d_inner, H, P, N, G = c.intermediate_size, c.num_heads, c.head_dim, c.state_size, c.n_groups
        decode = conv_state is not None

        dt_bias = self.param("dt_bias", nn.initializers.zeros, (H,), pd)
        A_log = self.param("A_log", nn.initializers.zeros, (H,), pd)
        D_param = self.param("D", nn.initializers.ones, (H,), pd)
        conv1d = Conv1d(c.conv_dim, c.conv_kernel, c.use_conv_bias, pd, name="conv1d")

        # 1. gated MLP's linear projection (FP8 W8A8 per-tensor static)
        projected_states = q.QDense(d_inner + c.conv_dim + H, q.FP8_STATIC, use_bias=c.use_bias,
                                    dtype=pd, name="in_proj")(hidden_states)
        A = -jnp.exp(A_log.astype(F32))
        gate, hidden_states_B_C, dt = jnp.split(
            projected_states, [d_inner, d_inner + c.conv_dim], axis=-1)

        # 2. convolution sequence transformation ([b, C, L] layout as in HF)
        hidden_states_B_C = jnp.swapaxes(hidden_states_B_C, 1, 2)
        if decode:     # causal_conv1d_update
            hs_new = jnp.concatenate([conv_state, hidden_states_B_C], axis=-1).astype(pd)
            new_conv_state = hs_new[:, :, -c.conv_kernel:]
            out = conv1d(hs_new, padding=0)[:, :, -seq_len:]
            hidden_states_B_C = nn.silu(out).astype(hidden_states_B_C.dtype)
        else:          # cache.update_conv_state (prefill: keep last K inputs) + causal_conv1d_fn
            # HF: F.pad(x, (K - L, 0)): crops to the last K inputs (L >= K) or zero-pads on the
            # left (L < K); lax.pad with the same (possibly negative) low padding, so the state is
            # [b, C, K] for every (symbolic) L, not [b, C, min(L, K)].
            new_conv_state = lax.pad(hidden_states_B_C, jnp.zeros((), hidden_states_B_C.dtype),
                                     [(0, 0, 0), (0, 0, 0), (c.conv_kernel - seq_len, 0, 0)])
            out = conv1d(hidden_states_B_C.astype(pd), padding=c.conv_kernel - 1)[:, :, :seq_len]
            hidden_states_B_C = nn.silu(out).astype(hidden_states_B_C.dtype)
            hidden_states_B_C = hidden_states_B_C[:, :, -seq_len:]

        # 3. SSM transformation
        hidden_states_B_C = jnp.swapaxes(hidden_states_B_C, 1, 2)
        x, B, C = jnp.split(hidden_states_B_C, [d_inner, d_inner + G * N], axis=-1)

        if decode:     # recurrent form
            x = x.reshape(batch_size, H, P)
            dt = jnp.broadcast_to(jnp.swapaxes(dt, 1, 2), (batch_size, H, P))
            A_e = jnp.broadcast_to(A[:, None, None], (H, P, N))
            B = B.reshape(batch_size, G, N)
            C = C.reshape(batch_size, G, N)
            D_e = jnp.broadcast_to(D_param[:, None], (H, P))
            dt_bias_e = jnp.broadcast_to(dt_bias[:, None], (H, P))
            scan_output, new_ssm_state = mamba2_selective_state_update(
                ssm_state, x, dt, A_e, B, C, D_e, dt_bias_e)
            scan_output = scan_output.reshape(batch_size, 1, -1)
        else:          # chunk form, zero initial state, returns the final state
            scan_output, new_ssm_state = mamba2_chunk_scan(
                x.reshape(batch_size, seq_len, H, P), dt, A,
                B.reshape(batch_size, seq_len, G, N), C.reshape(batch_size, seq_len, G, N),
                c.chunk_size, D_param, dt_bias, c.time_step_limit)
            scan_output = scan_output.reshape(batch_size, seq_len, -1)

        scan_output = MambaRMSNormGated(c.layer_norm_epsilon, pd, name="norm")(scan_output, gate)

        # 4. final linear projection (FP8 W8A8 per-tensor static)
        contextualized_states = q.QDense(c.hidden_size, q.FP8_STATIC, use_bias=c.use_bias,
                                         dtype=pd, name="out_proj")(scan_output.astype(dtype))
        return contextualized_states, new_conv_state, new_ssm_state


class Mamba2Block(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, hidden_states, conv_state=None, ssm_state=None):
        c = self.cfg
        residual = hidden_states
        hidden_states = Mamba2RMSNorm(c.layer_norm_epsilon, c.jdtype, name="norm")(
            hidden_states.astype(c.jdtype))
        if c.residual_in_fp32:
            residual = residual.astype(F32)
        hidden_states, conv_state, ssm_state = Mamba2Mixer(c, name="mixer")(
            hidden_states, conv_state, ssm_state)
        return residual + hidden_states, conv_state, ssm_state


class Mamba2Model(nn.Module):
    cfg: Config

    def setup(self):
        c = self.cfg
        self.embeddings = nn.Embed(c.vocab_size, c.hidden_size, dtype=c.jdtype,
                                   param_dtype=c.jdtype, name="embeddings")
        self.layers = [Mamba2Block(c, name=f"layers_{i}") for i in range(c.num_hidden_layers)]
        self.norm_f = Mamba2RMSNorm(c.layer_norm_epsilon, c.jdtype, name="norm_f")

    def __call__(self, input_ids, conv_states=None, ssm_states=None):
        n = self.cfg.num_hidden_layers
        conv_states = conv_states if conv_states is not None else [None] * n
        ssm_states = ssm_states if ssm_states is not None else [None] * n
        hidden_states = self.embeddings(input_ids)
        new_conv, new_ssm = [], []
        for layer, cs, ss in zip(self.layers, conv_states, ssm_states):
            hidden_states, cs, ss = layer(hidden_states, cs, ss)
            new_conv.append(cs)
            new_ssm.append(ss)
        hidden_states = self.norm_f(hidden_states)
        return hidden_states, new_conv, new_ssm

    def embedding_table(self):
        return self.embeddings.embedding


class TiedLMHead(nn.Module):
    """HF `lm_head` (nn.Linear, bias=False) whose weight is tied to the input embeddings."""

    @nn.compact
    def __call__(self, hidden_states, weight):
        return jnp.einsum("bld,vd->blv", hidden_states.astype(weight.dtype), weight)


class Mamba2ForCausalLM(nn.Module):
    cfg: Config

    def setup(self):
        assert self.cfg.tie_word_embeddings
        self.backbone = Mamba2Model(self.cfg, name="backbone")
        self.lm_head = TiedLMHead(name="lm_head")

    def __call__(self, input_ids, conv_states=None, ssm_states=None):
        """-> (logits fp32 [b, 1, V], conv_states [b, conv_dim, K] x n, ssm_states [b, h, p, n] x n)."""
        hidden_states, conv_states, ssm_states = self.backbone(input_ids, conv_states, ssm_states)
        # Only the last position is sampled (generate() / vLLM logits_to_keep=1).
        hidden_states = hidden_states[:, -1:]
        logits = self.lm_head(hidden_states, self.backbone.embedding_table()).astype(F32)
        return logits, conv_states, ssm_states
