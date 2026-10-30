"""Evo 2 7B (arcinstitute/evo2_7b, StripedHyena 2) capture spec; network in net.py.

Quantization: the official vortex + Transformer Engine path (FP8 E4M3 W8A8, per-tensor
delayed scales, on the 27 Hyena input projections only; see net.py).

Two phases, matching vortex `Generator.generate` with cached generation:
  prefill      tokens [1, 8192] -> logits [1, 8192, 512] + per-layer inference state
               (featurizer/inner FIR states, HCL modal states, KV caches of capacity 8692)
  decode_step  token [1, 1] at runtime offset + states -> logits [1, 1, 512] + states
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, as_spec
from models.evo2.net import Config, StripedHyena

F32 = jnp.float32

CONFIGS = {
    # evo2/configs/evo2-7b-1m.yml, bf16 (vortex load_checkpoint dtype protocol)
    "full": Config(),
    # Every block kind (HCS, HCM with its 128-tap FFT conv, HCL with modal FFT state,
    # MHA), layer-0 gelu, bf16 (the deployed dtypes, so the dtype audit sees the real
    # GEMM signatures). Prompt 160 >= hcm_filter_length so the HCM FIR state is
    # the full 127 taps; decode offset 163 < capacity - 1 so the KV valid-mask matters.
    "reduced": Config(hidden_size=64, num_layers=4, hcs_layer_idxs=(0,), hcm_layer_idxs=(1,),
                      hcl_layer_idxs=(2,), attn_layer_idxs=(3,), hcm_filter_groups=4,
                      hcs_filter_groups=4, num_attention_heads=4, state_size=8,
                      inner_mlp_size=176, prompt_len=160, n_tokens=8, decode_offset=163),
}


def state_specs(cfg: Config) -> dict:
    """decode_step state inputs. FIR states are fp32: vortex's step_fir returns them in
    fp32 (prefill emits them in the activation dtype; the upcast is exact)."""
    B, D = cfg.batch, cfg.hidden_size
    s = lambda *shape, dt=F32: jax.ShapeDtypeStruct(shape, dt)
    out = {}
    for i in range(cfg.num_layers):
        k = cfg.kind(i)
        if k == "attn":
            out[f"blocks_{i}"] = {"kv_cache": s(B, cfg.max_seqlen, 2, cfg.num_attention_heads,
                                                cfg.head_dim, dt=cfg.jdtype)}
            continue
        st = {"fir_state": s(B, 3 * D, cfg.short_filter_length - 1)}
        if k == "hcl":
            st["iir_state"] = s(B, D, cfg.state_size)
        else:
            flen = cfg.hcm_filter_length if k == "hcm" else cfg.hcs_filter_length
            st["fir_inner_state"] = s(B, D, flen - 1)
        out[f"blocks_{i}"] = st
    return out


def input_specs(cfg: Config) -> dict:
    i32 = jnp.int32
    return {
        "prefill": (jax.ShapeDtypeStruct((cfg.batch, cfg.prompt_len), i32),),
        "decode_step": (jax.ShapeDtypeStruct((cfg.batch, 1), i32),
                        jax.ShapeDtypeStruct((), i32), state_specs(cfg)),
    }


def init_variables(cfg: Config, rng):
    return StripedHyena(cfg).init(rng, jnp.zeros((cfg.batch, cfg.prompt_len), jnp.int32))


def decode_offset(cfg: Config) -> int:
    """Example decode offset. Full: the last cache slot (capacity - 1), where vortex attends
    over exactly the static capacity."""
    return cfg.max_seqlen - 1 if cfg.decode_offset is None else cfg.decode_offset


def example_inputs(cfg: Config, rng):
    k_tok, k_dec, k_st = jax.random.split(rng, 3)
    specs = state_specs(cfg)
    leaves, treedef = jax.tree.flatten(specs)
    keys = jax.random.split(k_st, len(leaves))
    state = jax.tree.unflatten(treedef, [jax.random.normal(k, s.shape, F32).astype(s.dtype)
                                         for k, s in zip(keys, leaves)])
    return {
        "prefill": (jax.random.randint(k_tok, (cfg.batch, cfg.prompt_len), 0, cfg.vocab_size,
                                       jnp.int32),),
        "decode_step": (jax.random.randint(k_dec, (cfg.batch, 1), 0, cfg.vocab_size, jnp.int32),
                        jnp.int32(decode_offset(cfg)), state),
    }


def intermediates(cfg: Config, variables, *inputs) -> dict:
    _, st = StripedHyena(cfg).apply(variables, *inputs, capture_intermediates=True,
                                    mutable=["intermediates"])
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            st["intermediates"], is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        out = v[0]
        if isinstance(out, tuple):      # blocks / filters / MHA return (output, state)
            out = out[0]
        flat["/".join(k for k in keys if k != "__call__")] = out
    return flat


def build(cfg: Config) -> ModelSpec:
    module = StripedHyena(cfg)

    def prefill(variables, input_ids):
        return module.apply(variables, input_ids)

    def decode_step(variables, input_ids, offset, state):
        return module.apply(variables, input_ids, offset, state)

    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    specs = input_specs(cfg)
    kinds = [cfg.kind(i) for i in range(cfg.num_layers)]
    n_attn, n_hcl, n_hcm, n_hcs = (kinds.count(k) for k in ("attn", "hcl", "hcm", "hcs"))
    full = cfg == CONFIGS["full"]
    return ModelSpec(
        name="evo2",
        variant=(f"Evo 2 {'7B' if full else 'reduced'} (arcinstitute/evo2_7b, StripedHyena 2, "
                 f"evo2-7b-1m.yml): d {cfg.hidden_size}, {cfg.num_layers} blocks "
                 f"({n_hcs} HCS / {n_hcm} HCM / {n_hcl} HCL / {n_attn} MHA), vocab "
                 f"{cfg.vocab_size}, prompt {cfg.prompt_len}, {cfg.n_tokens} generated tokens, "
                 f"KV capacity {cfg.max_seqlen}, batch {cfg.batch}, {cfg.dtype}; FP8 E4M3 "
                 f"W8A8 Hyena input projections (vortex + Transformer Engine, per-tensor "
                 f"delayed scaling), rest bf16 / fp32 filters"),
        phases=[
            Phase("prefill", prefill, specs["prefill"],
                  note="vortex parallel_forward with inference params: featurizer + HCS "
                       "depthwise convs, HCM/HCL FFT convolutions, HCL modal-FFT state "
                       "prefill, causal MHA writing a fresh KV cache -> logits + all states"),
            Phase("decode_step", decode_step, specs["decode_step"],
                  trip_count=cfg.n_tokens - 1,
                  note="one token at a runtime int32 offset: step_fir (featurizer, HCS, "
                       "HCM), step_iir (HCL modal recurrence), MHA with dynamic_update_slice "
                       "cache write over the static capacity (valid-key mask). Trip count = "
                       "n_tokens - 1 (the first token comes from the prefill logits)."),
        ],
        param_specs=as_spec(variables),
        excluded=["CharLevelTokenizer tokenization / detokenization (pre/postprocessing)",
                  "sampling (top-k / top-p / temperature over last-position logits) and the "
                  "Generator.generate loop (decode_step trip_count records it)",
                  "prompt forcing (force_prompt_threshold, off in Evo2.generate) and "
                  "multi-GPU layer placement",
                  "TE delayed-scaling bookkeeping (amax of the projection input / weight, "
                  "amax-history roll, scale update after each call): outside the GEMM path; "
                  "the scale each call uses is the `input_scale` / `kernel_scale` parameter",
                  "pad-to-16-rows of the FP8 projection input (kernel alignment)"],
    )

