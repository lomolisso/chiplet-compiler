"""Whisper large-v3 (MLPerf Inference speech2text) capture spec, CTranslate2 int8_float16
(faster-whisper: W8A8 int8 linears with dynamic per-token activations, int8 per-row token
embeddings / tied LM head, everything else and the K/V caches fp16); network in net.py.

Three phases, each one network invocation (the greedy decode loop is not traced):

* `encoder`          log-mel [1, 128, 3000] -> encoder states [1, 1500, 1280]          x1
* `decoder_prefill`  the 4 forced prompt tokens + encoder states -> logits, a static
                     self-attention KV cache (capacity 448) filled at positions 0..3,
                     and the cross-attention K/V of every layer                       x1
* `decoder_step`     one token at a runtime int32 position + both caches -> logits and
                     the updated self-attention cache                                 x199

Parameters are grouped at the top level by the phase that consumes them
(`Phase.param_keys`): the encoder, the decoder, and the decoder cross-attention
k_proj/v_proj, which only prefill uses (the step reads their output from the cross
cache). Inside each group the tree is the HF Flax Whisper tree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, as_spec
from models.whisper.net import Config, FlaxWhisperForConditionalGenerationModule as Whisper

# CTranslate2 int8_float16: non-quantized layers, activations and K/V caches in fp16
# (the MLPerf reference, vLLM, runs bf16; the quantized deployment is fp16).
CONFIGS = {
    "full": Config(),
    "reduced": Config(num_mel_bins=16, d_model=64, encoder_layers=2, decoder_layers=2,
                      encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=256,
                      decoder_ffn_dim=256, vocab_size=512, max_source_positions=64,
                      max_target_positions=32, dtype="float16"),
}

CROSS_KV = ("k_proj", "v_proj")


# ----------------------------------------------------------------------------
# shapes / params
# ----------------------------------------------------------------------------

def _dims(cfg: Config):
    H = cfg.decoder_attention_heads
    return H, cfg.d_model // H


def input_spec(cfg: Config) -> dict:
    dt = jnp.dtype(cfg.dtype)
    B, L = cfg.batch, cfg.decoder_layers
    H, D = _dims(cfg)
    S = jax.ShapeDtypeStruct
    self_kv = tuple(S((B, cfg.max_target_positions, H, D), dt) for _ in range(L))
    cross_kv = tuple(S((B, cfg.max_source_positions, H, D), dt) for _ in range(L))
    return {
        "encoder": (S((B, cfg.num_mel_bins, 2 * cfg.max_source_positions), dt),),
        "decoder_prefill": (S((B, cfg.prompt_len), jnp.int32),
                            S((B, cfg.max_source_positions, cfg.d_model), dt)),
        "decoder_step": (S((B, 1), jnp.int32), S((), jnp.int32),
                         self_kv, self_kv, cross_kv, cross_kv),
    }


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if k in out else v
    return out


def _split_params(params: dict) -> dict:
    """HF Flax tree {"model": {"encoder", "decoder"}} -> phase groups."""
    enc = params["model"]["encoder"]
    dec = jax.tree.map(lambda x: x, dict(params["model"]["decoder"]))
    dec["layers"] = {i: dict(l) for i, l in dec["layers"].items()}
    cross = {"layers": {}}
    for i, layer in dec["layers"].items():
        ea = dict(layer["encoder_attn"])
        cross["layers"][i] = {"encoder_attn": {k: ea.pop(k) for k in CROSS_KV}}
        layer["encoder_attn"] = ea
    return {"encoder": enc, "decoder": dec, "decoder_cross_kv": cross}


def init_variables(cfg: Config, rng):
    """Initialize through the same entry points the phases use (encode, then a
    cached decode), so every parameter is created exactly as it is consumed."""
    m = Whisper(cfg)
    r1, r2 = jax.random.split(rng)
    x = jnp.zeros(input_spec(cfg)["encoder"][0].shape, input_spec(cfg)["encoder"][0].dtype)
    (enc,), v1 = m.apply({}, x, method=Whisper.encode, rngs={"params": r1}, mutable=["params"])
    ids = jnp.zeros((cfg.batch, cfg.prompt_len), jnp.int32)
    _, v2 = m.apply({"cache": _zero_cache(cfg)}, ids, _prompt_positions(cfg), enc,
                    method=Whisper.decode, rngs={"params": r2}, mutable=["params", "cache"])
    return _split_params(_merge(v1["params"], v2["params"]))


# ----------------------------------------------------------------------------
# cache <-> flat tuples (HF Flax "cache" collection layout [B, L, H, D])
# ----------------------------------------------------------------------------

def _cache(self_k, self_v, index, cross_k=None, cross_v=None) -> dict:
    layers = {}
    for i in range(len(self_k)):
        d = {"self_attn": {"cached_key": self_k[i], "cached_value": self_v[i], "cache_index": index}}
        if cross_k is not None:
            d["encoder_attn"] = {"cached_cross_key": cross_k[i], "cached_cross_value": cross_v[i]}
        layers[str(i)] = d
    return {"model": {"decoder": {"layers": layers}}}


def _zero_cache(cfg: Config) -> dict:
    """HF Flax `init_cache`: zero self-attention cache of capacity max_target_positions."""
    s = input_spec(cfg)["decoder_step"][2][0]
    z = tuple(jnp.zeros(s.shape, s.dtype) for _ in range(cfg.decoder_layers))
    return _cache(z, z, jnp.array(0, jnp.int32))


def _prompt_positions(cfg: Config):
    return jnp.broadcast_to(jnp.arange(cfg.prompt_len, dtype=jnp.int32)[None],
                            (cfg.batch, cfg.prompt_len))


def _unpack(cache: dict, cfg: Config, cross: bool):
    layers = cache["model"]["decoder"]["layers"]
    rng = [str(i) for i in range(cfg.decoder_layers)]
    out = (tuple(layers[i]["self_attn"]["cached_key"] for i in rng),
           tuple(layers[i]["self_attn"]["cached_value"] for i in rng))
    if cross:
        out += (tuple(layers[i]["encoder_attn"]["cached_cross_key"] for i in rng),
                tuple(layers[i]["encoder_attn"]["cached_cross_value"] for i in rng))
    return out


# ----------------------------------------------------------------------------
# phases
# ----------------------------------------------------------------------------

def _run(cfg: Config, phase: str, p: dict, *inputs, capture: bool = False):
    """Returns (outputs, intermediates-or-None). `p` holds the phase's param groups."""
    m = Whisper(cfg)
    kw = dict(capture_intermediates=True) if capture else {}
    extra = ["intermediates"] if capture else []
    if phase == "encoder":
        (x,) = inputs
        variables = {"params": {"model": {"encoder": p["encoder"]}}}
        if capture:
            (h,), st = m.apply(variables, x, method=Whisper.encode, mutable=extra, **kw)
            return h, st["intermediates"]
        (h,) = m.apply(variables, x, method=Whisper.encode)
        return h, None
    if phase == "decoder_prefill":
        ids, enc = inputs
        dec = _merge(p["decoder"], p["decoder_cross_kv"])
        variables = {"params": {"model": {"decoder": dec}}, "cache": _zero_cache(cfg)}
        logits, st = m.apply(variables, ids, _prompt_positions(cfg), enc, method=Whisper.decode,
                             mutable=["cache"] + extra, **kw)
        return (logits, *_unpack(st["cache"], cfg, cross=True)), st.get("intermediates")
    if phase == "decoder_step":
        ids, position, self_k, self_v, cross_k, cross_v = inputs
        variables = {"params": {"model": {"decoder": p["decoder"]}},
                     "cache": _cache(self_k, self_v, position, cross_k, cross_v)}
        pos_ids = jnp.broadcast_to(position, ids.shape)
        logits, st = m.apply(variables, ids, pos_ids, None, method=Whisper.decode,
                             mutable=["cache"] + extra, **kw)
        return (logits, *_unpack(st["cache"], cfg, cross=False)), st.get("intermediates")
    raise KeyError(phase)


def phase_fn(cfg: Config, phase: str):
    def fn(p, *inputs):
        return _run(cfg, phase, p, *inputs)[0]
    fn.__name__ = phase
    return fn


def example_inputs(cfg: Config, rng):
    spec = input_spec(cfg)
    k = iter(jax.random.split(rng, 4 * cfg.decoder_layers + 8))
    rnd = lambda s, scale=1.0: (scale * jax.random.normal(next(k), s.shape, jnp.float32)).astype(s.dtype)
    tok = lambda shape: jax.random.randint(next(k), shape, 0, cfg.vocab_size, dtype=jnp.int32)
    ids = tok((cfg.batch, cfg.prompt_len)).at[:, 0].set(cfg.vocab_size - 1)
    pos = jnp.array(cfg.prompt_len + 3, jnp.int32)   # 3 tokens already generated
    st = spec["decoder_step"]
    return {
        "encoder": (rnd(spec["encoder"][0]),),
        "decoder_prefill": (ids, rnd(spec["decoder_prefill"][1])),
        "decoder_step": (tok((cfg.batch, 1)), pos,
                         tuple(rnd(s) for s in st[2]), tuple(rnd(s) for s in st[3]),
                         tuple(rnd(s) for s in st[4]), tuple(rnd(s) for s in st[5])),
    }


def _phase_of(inputs) -> str:
    return {1: "encoder", 2: "decoder_prefill", 6: "decoder_step"}[len(inputs)]


def _first_array(v):
    while isinstance(v, (tuple, list)):
        v = v[0]
    return v


def intermediates(cfg: Config, variables, *inputs) -> dict:
    phase = _phase_of(inputs)
    out, inter = _run(cfg, phase, variables, *inputs, capture=True)
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            inter, is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        flat["/".join(k for k in keys if k != "__call__")] = _first_array(v)
    if phase != "encoder":
        flat.setdefault("lm_head", out[0])     # tied head runs via lm_head.apply (not recorded)
    return flat


# ----------------------------------------------------------------------------
# spec
# ----------------------------------------------------------------------------

def build(cfg: Config) -> ModelSpec:
    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    spec = input_spec(cfg)
    Le, Ld = cfg.encoder_layers, cfg.decoder_layers
    n_step = cfg.max_new_tokens - 1
    phases = [
        Phase("encoder", phase_fn(cfg, "encoder"), spec["encoder"], trip_count=1,
              param_keys=("encoder",),
              note="log-mel [B, n_mels, 3000] -> encoder hidden states [B, 1500, d_model]"),
        Phase("decoder_prefill", phase_fn(cfg, "decoder_prefill"), spec["decoder_prefill"],
              trip_count=1, param_keys=("decoder", "decoder_cross_kv"),
              note=(f"{cfg.prompt_len} forced prompt tokens (<|startoftranscript|><|en|>"
                    "<|transcribe|><|notimestamps|>) + encoder states -> (logits, self K cache, "
                    f"self V cache [capacity {cfg.max_target_positions}], cross K, cross V); "
                    "its last-position argmax is generated token 1")),
        Phase("decoder_step", phase_fn(cfg, "decoder_step"), spec["decoder_step"],
              trip_count=n_step, param_keys=("decoder",),
              note=(f"one token at runtime int32 position + caches -> (logits, self K, self V); "
                    f"MLPerf max_tokens={cfg.max_new_tokens} (greedy, temperature 0) -> at most "
                    f"{n_step} steps after prefill (EOS ends earlier)")),
    ]
    return ModelSpec(
        name="whisper",
        variant=(f"Whisper large-v3 (MLPerf speech2text), {Le}+{Ld} layers, d_model {cfg.d_model}, "
                 f"vocab {cfg.vocab_size}, 30 s audio, batch {cfg.batch}, CTranslate2 "
                 f"int8_float16: int8 W8A8 linears + tied LM head (per-row weights, dynamic "
                 f"per-token activations), int8 per-row token embeddings, {cfg.dtype} conv stem / "
                 "attention / LayerNorm / K/V caches"),
        phases=phases,
        param_specs=as_spec(variables),
        excluded=["audio load / resample / zero-pad to 30 s and log-mel spectrogram "
                  "(WhisperFeatureExtractor; preprocessing)",
                  "greedy argmax / EOS check / token feedback of the decode loop (outer loop)",
                  "detokenization and text normalization (postprocessing)"],
    )
