"""Mamba-2 2.7B (state-spaces/mamba2-2.7b) FP8 capture spec; network in net.py.

Recipe: FP8 E4M3 W8A8 per-tensor static on the mixer in_proj / out_proj (GPU fallback in
NVIDIA's ModelOpt FP8 form for Mamba-2 mixers, see net.py and recipe.py); everything else
bf16 / fp32 as in HF.

Two phases, matching HF `Mamba2ForCausalLM`'s cache API:
  prefill      tokens [1, 2048] -> last-position logits [1, 1, V] + per-layer
               (conv_state, ssm_state)
  decode_step  token [1, 1] + per-layer (conv_state, ssm_state) -> logits [1, 1, V] + states
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, as_spec
from models.mamba2.net import Config, Mamba2ForCausalLM

# Full: Mamba-2 2.7B, bf16 activations (LLM inference dtype) + FP8 in_proj / out_proj.
# Reduced: every block type kept (multi-chunk SSD with inter-chunk recurrence, conv, gated
# norm, decode step), same dtypes as full.
CONFIGS = {
    "full": Config(),
    "reduced": Config(vocab_size=512, hidden_size=64, state_size=16, num_heads=8, head_dim=16,
                      chunk_size=32, num_hidden_layers=2, seq_len=128),
}


def state_specs(cfg: Config):
    conv = jax.ShapeDtypeStruct((cfg.batch, cfg.conv_dim, cfg.conv_kernel), cfg.jdtype)
    ssm = jax.ShapeDtypeStruct((cfg.batch, cfg.num_heads, cfg.head_dim, cfg.state_size),
                               jnp.float32)
    n = cfg.num_hidden_layers
    return [conv] * n, [ssm] * n


def input_specs(cfg: Config) -> dict:
    conv, ssm = state_specs(cfg)
    return {
        "prefill": (jax.ShapeDtypeStruct((cfg.batch, cfg.seq_len), jnp.int32),),
        "decode_step": (jax.ShapeDtypeStruct((cfg.batch, 1), jnp.int32), conv, ssm),
    }


def init_variables(cfg: Config, rng):
    return Mamba2ForCausalLM(cfg).init(rng, jnp.zeros((cfg.batch, cfg.seq_len), jnp.int32))


def example_inputs(cfg: Config, rng):
    k_tok, k_dec, k_conv, k_ssm = jax.random.split(rng, 4)
    conv, ssm = state_specs(cfg)
    kc = jax.random.split(k_conv, len(conv))
    ks = jax.random.split(k_ssm, len(ssm))
    return {
        "prefill": (jax.random.randint(k_tok, (cfg.batch, cfg.seq_len), 0, cfg.vocab_size,
                                       dtype=jnp.int32),),
        "decode_step": (
            jax.random.randint(k_dec, (cfg.batch, 1), 0, cfg.vocab_size, dtype=jnp.int32),
            [jax.random.normal(k, s.shape, jnp.float32).astype(s.dtype) for k, s in zip(kc, conv)],
            [jax.random.normal(k, s.shape, jnp.float32) for k, s in zip(ks, ssm)],
        ),
    }


def intermediates(cfg: Config, variables, *inputs) -> dict:
    _, state = Mamba2ForCausalLM(cfg).apply(variables, *inputs, capture_intermediates=True,
                                            mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            state["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        out = v[0]
        if isinstance(out, tuple):      # blocks / mixers return (hidden, conv_state, ssm_state)
            out = out[0]
        if out is None:
            continue
        flat["/".join(k for k in keys if k != "__call__")] = out
    return flat


def build(cfg: Config) -> ModelSpec:
    module = Mamba2ForCausalLM(cfg)

    def prefill(variables, input_ids):
        return module.apply(variables, input_ids)

    def decode_step(variables, input_ids, conv_states, ssm_states):
        return module.apply(variables, input_ids, conv_states, ssm_states)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    specs = input_specs(cfg)
    L = cfg.num_hidden_layers
    n_chunks = -(-cfg.seq_len // cfg.chunk_size)
    return ModelSpec(
        name="mamba2",
        variant=(f"Mamba-2 2.7B (state-spaces/mamba2-2.7b): d_model {cfg.hidden_size}, {L} layers, "
                 f"d_state {cfg.state_size}, headdim {cfg.head_dim}, chunk {cfg.chunk_size}, "
                 f"prefill {cfg.seq_len} ({n_chunks} chunks), batch {cfg.batch}, {cfg.dtype}; FP8 E4M3 "
                 "W8A8 per-tensor static in_proj / out_proj (ModelOpt FP8 form, GPU fallback)"),
        phases=[
            Phase("prefill", prefill, specs["prefill"],
                  note="chunked SSD over the prompt (all chunks unrolled, inter-chunk recurrence "
                       "as a (c+1)x(c+1) decay matrix) -> last-position logits + final "
                       "conv/SSM states"),
            Phase("decode_step", decode_step, specs["decode_step"], trip_count=cfg.decode_steps,
                  note="one token: causal_conv1d_update + selective_state_update recurrence"),
        ],
        param_specs=as_spec(variables),
        excluded=["GPT-NeoX tokenization (preprocessing)",
                  "FP8 calibration (offline; scales are parameters)",
                  "sampling / argmax over logits and the generate() loop (postprocessing; "
                  "decode_step trip_count records the loop)"],
    )
