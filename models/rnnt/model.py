"""RNN-T (MLPerf Inference speech recognition, retired) capture spec; network in net.py.

Quantization (MLPerf v4.0 NVIDIA, see net.py / recipe.py): encoder LSTMs INT8 (int8 layer
inputs and hidden states, per-row int8 weights, i32 GEMMs, fp16 gates and cell state),
prediction network and joint fp16, preprocessing fp32. Between encoder phases the hidden
sequences / states are int8 and the cell states fp16; the last encoder layer emits fp16.

One phase per distinct network invocation of the MLPerf PyTorch SUT
(pytorch_SUT.issue_queries -> AudioPreprocessing, then ScriptGreedyDecoder), with
the encoder split layer-major (so the frame count can be symbolic):

* `preprocess`         waveform [1, N] + length -> log-mel features [1, 240, T] + T   x1
* per encoder LSTM layer L (pre_l0, pre_l1, then post_l0, post_l1, post_l2):
  `encoder_L_in_proj`  the layer's input sequence [T_L, 1, in] -> x W_ih^T + b_ih
                       [T_L, 1, 4H] (one GEMM over the whole sequence)                x1
  `encoder_L_step`     one recurrence step: (xw_t [1, 4H] f16, h i8, c f16) -> (h, c)
                       (+ h in fp16 for the last layer, the encoder output)            x T_L
* `encoder_stack_time` pre_rnn output [T, 1, H] + T -> [ceil(T/2), 1, 2H] + ceil(T/2) x1
* `prediction_sos`     model.prediction(None, None): zero input, zero state
                       -> g [1, 1, 320], (h, c) [2, 1, 320]                           x1
* `prediction_step`    model.prediction(label [1, 1], (h, c)) -> g, (h, c)       x T/2 + U - 1
* `joint`              model.joint(f [1, 1, 1024], g [1, 1, 320]) -> logits [1,1,1,29] x T/2 + U

T_L is T for the pre-rnn layers and ceil(T/2) for the post-rnn layers. The encoder
time loop runs on the host like the greedy decoder's: it feeds xw[t] to each step
and stacks the step outputs into the next layer's input sequence. The greedy loop
(argmax, blank test, max_symbols_per_step, label/state feedback) runs on the host
between invocations and is not traced. Trip counts: every greedy iteration calls
the prediction network once and the joint once; each encoder frame ends with one
blank, so iterations = T/2 + U with U emitted symbols. The prediction network gets
the SOS form until the first symbol is emitted (assumed at frame 0).

Parameters are grouped at the top level by consumer (`Phase.param_keys`): per
encoder layer `encoder_L_ih` (weight_ih, weight_ih_scale, bias_ih, input_scale) and
`encoder_L_hh` (weight_hh, weight_hh_scale, bias_hh, hidden_scale), `prediction`
(the LSTM), `prediction_embed` (only the labelled step embeds) and `joint`. `preprocess` and `encoder_stack_time` have no parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.spec import ModelSpec, Phase, as_spec
from models.rnnt.net import RNNT, AudioPreprocessing, Config, LSTMLayer, StackTime

CONFIGS = {
    "full": Config(),
    # Same preprocessing (features fixed by the toml), short utterance with an odd
    # feature-frame count (35, exercises StackTime's zero padding), small widths.
    "reduced": Config(encoder_n_hidden=64, pred_n_hidden=32, joint_n_hidden=48,
                      n_samples=16500, n_symbols=12),
}


def enc_layers(cfg: Config) -> list[tuple[str, str, int]]:
    """(name, LstmDrop module, layer index) of every encoder LSTM layer, in order."""
    return ([(f"pre_l{k}", "pre_rnn", k) for k in range(cfg.encoder_pre_rnn_layers)]
            + [(f"post_l{k}", "post_rnn", k) for k in range(cfg.encoder_post_rnn_layers)])


def phase_names(cfg: Config) -> list[str]:
    names = ["preprocess"]
    for name, part, k in enc_layers(cfg):
        if part == "post_rnn" and k == 0:
            names.append("encoder_stack_time")
        names += [f"encoder_{name}_in_proj", f"encoder_{name}_step"]
    return names + ["prediction_sos", "prediction_step", "joint"]


def param_keys(cfg: Config) -> dict[str, tuple[str, ...]]:
    keys = {"preprocess": (), "encoder_stack_time": (), "prediction_sos": ("prediction",),
            "prediction_step": ("prediction", "prediction_embed"), "joint": ("joint",)}
    for name, _, _ in enc_layers(cfg):
        keys[f"encoder_{name}_in_proj"] = (f"encoder_{name}_ih",)
        keys[f"encoder_{name}_step"] = (f"encoder_{name}_hh",)
    return keys


# ----------------------------------------------------------------------------
# shapes / params
# ----------------------------------------------------------------------------

def in_features(cfg: Config) -> int:       # 80 mels x frame splicing 3
    return cfg.features * cfg.frame_splicing


def enc_frames(cfg: Config) -> int:        # StackTime: ceil(T / factor)
    if cfg.post_frames is not None:
        return cfg.post_frames
    return -(-cfg.feat_frames // cfg.encoder_stack_time_factor)


def _layer_io(cfg: Config, name: str) -> tuple[int, int]:
    """(sequence length, input width) of encoder layer `name`."""
    H, f = cfg.encoder_n_hidden, cfg.encoder_stack_time_factor
    if name.startswith("pre"):
        return cfg.feat_frames, (in_features(cfg) if name == "pre_l0" else H)
    return enc_frames(cfg), (H * f if name == "post_l0" else H)


def input_spec(cfg: Config) -> dict:
    """Preprocessing fp32; encoder features fp16 (NVIDIA input_dtype), int8 hidden
    sequences / states between encoder layers, fp16 cell states, xw, prediction state and
    joint inputs."""
    S, f32, i32, i8 = jax.ShapeDtypeStruct, jnp.float32, jnp.int32, jnp.int8
    f16 = jnp.dtype(cfg.dtype)
    B, Lp, Hp, H = cfg.batch, cfg.pred_rnn_layers, cfg.pred_n_hidden, cfg.encoder_n_hidden
    state = S((Lp, B, Hp), f16)
    spec = {
        "preprocess": (S((B, cfg.n_samples), f32), S((B,), i32)),
        "encoder_stack_time": (S((cfg.feat_frames, B, H), i8), S((B,), i32)),
        "prediction_sos": (),
        "prediction_step": (S((B, 1), i32), state, state),
        "joint": (S((B, 1, H), f16), S((B, 1, Hp), f16)),
    }
    for name, _, _ in enc_layers(cfg):
        T, n_in = _layer_io(cfg, name)
        spec[f"encoder_{name}_in_proj"] = (S((T, B, n_in), f16 if name == "pre_l0" else i8),)
        spec[f"encoder_{name}_step"] = (S((B, 4 * H), f16), S((B, H), i8), S((B, H), f16))
    return spec


def encoder_input_spec(cfg: Config) -> tuple:
    """Input of the whole encoder (`RNNT.encode`): fp16 features [T, B, 240] + lengths."""
    S = jax.ShapeDtypeStruct
    return (S((cfg.feat_frames, cfg.batch, in_features(cfg)), jnp.dtype(cfg.dtype)),
            S((cfg.batch,), jnp.int32))


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if k in out else v
    return out


# input-projection side of an INT8 encoder layer (the rest is the recurrent side)
IH_PARAMS = ("weight_ih", "weight_ih_scale", "bias_ih", "input_scale")


def _split_params(params: dict) -> dict:
    pred = dict(params["prediction"])
    embed = pred.pop("embed")
    out = {"prediction": pred, "prediction_embed": {"embed": embed}, "joint": params["joint"]}
    for part, mods in params["encoder"].items():          # pre_rnn / post_rnn
        for k, lp in mods["lstm"].items():                 # l0, l1, ...
            name = f"{part[:-4]}_{k}"                      # pre_l0, post_l2, ...
            out[f"encoder_{name}_ih"] = {k: v for k, v in lp.items() if k in IH_PARAMS}
            out[f"encoder_{name}_hh"] = {k: v for k, v in lp.items() if k not in IH_PARAMS}
    return out


def _join_params(p: dict) -> dict:
    out = {k: p[k] for k in ("joint",) if k in p}
    if "prediction" in p:
        out["prediction"] = dict(p["prediction"])
        if "prediction_embed" in p:
            out["prediction"]["embed"] = p["prediction_embed"]["embed"]
    enc: dict = {}
    for key, v in p.items():
        if key.startswith("encoder_"):
            part, k = key[len("encoder_"):-len("_ih")].split("_")
            layer = enc.setdefault(f"{part}_rnn", {"lstm": {}})["lstm"].setdefault(k, {})
            layer.update(v)
    if enc:
        out["encoder"] = enc
    return out


def init_variables(cfg: Config, rng):
    """Initialize through the entry points of the unsplit network (encode, a labelled
    prediction step, join), so every parameter is created as it is consumed; then
    group the parameters by phase."""
    m = RNNT(cfg)
    spec = input_spec(cfg)
    zeros = lambda s: jnp.zeros(s.shape, s.dtype)
    r1, r2, r3 = jax.random.split(rng, 3)
    enc_in = encoder_input_spec(cfg)
    v1 = m.init(r1, *map(zeros, enc_in), method=RNNT.encode)
    y, h, c = map(zeros, spec["prediction_step"])
    v2 = m.init(r2, y, (h, c), method=RNNT.predict)
    v3 = m.init(r3, *map(zeros, spec["joint"]), method=RNNT.join)
    return _split_params(_merge(_merge(v1["params"], v2["params"]), v3["params"]))


# ----------------------------------------------------------------------------
# phases
# ----------------------------------------------------------------------------

def _run(cfg: Config, phase: str, p: dict, *inputs, capture: bool = False):
    """Returns (outputs, intermediates-or-None)."""
    kw = dict(capture_intermediates=True, mutable=["intermediates"]) if capture else {}
    method = None
    if phase == "preprocess":
        m, variables, args = AudioPreprocessing(cfg), {}, inputs
    elif phase == "encoder_stack_time":
        m, variables, args = StackTime(cfg.encoder_stack_time_factor), {}, inputs
    elif phase.startswith("encoder_"):
        mode = "in_proj" if phase.endswith("_in_proj") else "step"
        (key,) = param_keys(cfg)[phase]
        last = phase == f"encoder_{enc_layers(cfg)[-1][0]}_step"     # emits the fp16 output
        m = LSTMLayer(cfg.encoder_n_hidden, mode, quantized=True, out_float=last, dtype=cfg.dtype)
        variables, args = {"params": p[key]}, inputs
    else:
        m, variables = RNNT(cfg), {"params": _join_params(p)}
        if phase == "prediction_sos":
            method, args = RNNT.predict, (None, None)
        elif phase == "prediction_step":
            y, h, c = inputs
            method, args = RNNT.predict, (y, (h, c))
        elif phase == "joint":
            method, args = RNNT.join, inputs
        else:
            raise KeyError(phase)
    if method is not None:
        kw["method"] = method
    if capture:
        out, st = m.apply(variables, *args, **kw)
        return out, st["intermediates"]
    return m.apply(variables, *args, **kw), None


def phase_fn(cfg: Config, phase: str):
    def fn(p, *inputs):
        return _run(cfg, phase, p, *inputs)[0]
    fn.__name__ = phase
    return fn


def example_inputs(cfg: Config, rng):
    spec = input_spec(cfg)
    k = iter(jax.random.split(rng, 16))

    def rnd(s, scale=1.0):              # int8 tensors: uniform codes; float: N(0, scale)
        if s.dtype == jnp.int8:
            return jax.random.randint(next(k), s.shape, -127, 128, jnp.int32).astype(jnp.int8)
        return (scale * jax.random.normal(next(k), s.shape, jnp.float32)).astype(s.dtype)

    B = cfg.batch
    # The draws of the unsplit port, in its order (the whole-encoder features second).
    pre = (rnd(spec["preprocess"][0], 0.1), jnp.full((B,), cfg.n_samples, jnp.int32))
    feats = (rnd(encoder_input_spec(cfg)[0]), jnp.full((B,), cfg.feat_frames, jnp.int32))
    out = {
        "preprocess": pre,
        "prediction_sos": (),
        "prediction_step": (jax.random.randint(next(k), (B, 1), 0, cfg.num_classes - 1, jnp.int32),
                            rnd(spec["prediction_step"][1], 0.5),
                            rnd(spec["prediction_step"][2], 0.5)),
        "joint": (rnd(spec["joint"][0], 0.5), rnd(spec["joint"][1], 0.5)),
        # not a phase: the input of the whole encoder, for regression_dump
        "encoder_features": feats,
    }
    # The split encoder's phases draw from their own key.
    k = iter(jax.random.split(jax.random.fold_in(rng, 1), 64))
    out["encoder_stack_time"] = (rnd(spec["encoder_stack_time"][0]),
                                 jnp.full((B,), cfg.feat_frames, jnp.int32))
    for name, _, _ in enc_layers(cfg):
        out[f"encoder_{name}_in_proj"] = (rnd(spec[f"encoder_{name}_in_proj"][0]),)
        out[f"encoder_{name}_step"] = tuple(rnd(s, 0.5) for s in spec[f"encoder_{name}_step"])
    return out


def _first_array(v):
    while isinstance(v, (tuple, list)):
        v = v[0]
    return v


def intermediates(cfg: Config, variables, *inputs, phase: str) -> dict:
    out, inter = _run(cfg, phase, variables, *inputs, capture=True)
    flat = {}
    for path, v in jax.tree_util.tree_flatten_with_path(
            inter, is_leaf=lambda t: isinstance(t, tuple))[0]:
        keys = [k.key for k in path if hasattr(k, "key")]
        flat["/".join(k for k in keys if k != "__call__")] = _first_array(v)
    if phase == "preprocess":
        flat["audio_preprocessor"] = flat.pop("")
    elif "" in flat:                         # encoder phases: a single root module
        flat[phase] = flat.pop("")
    return flat


def encode(cfg: Config, variables, x, x_lens, fn=jax.jit) -> tuple:
    """The whole encoder ([T, B, in] features -> ([B, T/2, H], T/2)) run as the host runs
    it: the split encoder's phases (each through `fn`) and the host time loop.
    Returns (outputs, per-module outputs keyed like the unsplit port's intermediates)."""
    H, B = cfg.encoder_n_hidden, x.shape[1]
    last = enc_layers(cfg)[-1][0]
    keys = param_keys(cfg)
    run = lambda ph, *a: fn(phase_fn(cfg, ph))({k: variables[k] for k in keys[ph]}, *a)
    inter = {}
    for name, part, k in enc_layers(cfg):
        if part == "post_rnn" and k == 0:
            x, x_lens = run("encoder_stack_time", x, x_lens)
            inter["encoder/stack_time"] = x
        xw = run(f"encoder_{name}_in_proj", x)
        h, c = jnp.zeros((B, H), jnp.int8), jnp.zeros((B, H), cfg.dtype)
        ys = []
        for t in range(xw.shape[0]):
            out = run(f"encoder_{name}_step", xw[t], h, c)
            h, c = out[:2]
            ys.append(out[2] if name == last else h)     # int8 h; fp16 encoder output
        x = jnp.stack(ys)
        inter[f"encoder/{part}/lstm/l{k}"] = x
        inter[f"encoder/{part}/lstm"] = inter[f"encoder/{part}"] = x
    enc = jnp.swapaxes(x, 0, 1)
    inter["encoder"] = enc
    return (enc, x_lens), inter


def regression_dump(cfg: Config, variables, inputs) -> dict:
    """The unsplit port's `encoder` phase results, from the split phases."""
    (enc, lens), inter = encode(cfg, variables, *inputs["encoder_features"])
    out = {"encoder/out0": enc, "encoder/out1": lens}
    out.update({f"encoder/inter/{k}": v for k, v in inter.items()})
    return out


# ----------------------------------------------------------------------------
# spec
# ----------------------------------------------------------------------------

def trip_counts(cfg: Config) -> dict:
    iters = enc_frames(cfg) + cfg.n_symbols
    trips = {"preprocess": 1, "encoder_stack_time": 1, "prediction_sos": 1,
             "prediction_step": iters - 1, "joint": iters}
    for name, _, _ in enc_layers(cfg):
        trips[f"encoder_{name}_in_proj"] = 1
        trips[f"encoder_{name}_step"] = _layer_io(cfg, name)[0]
    return trips


def trip_count_exprs(cfg: Config) -> dict[str, str]:
    """Trip counts in the feature-frame count T (U = emitted symbols, not a size)."""
    exprs = {"prediction_step": "ceil(T/2) + U - 1", "joint": "ceil(T/2) + U"}
    for name, _, _ in enc_layers(cfg):
        exprs[f"encoder_{name}_step"] = "T" if name.startswith("pre") else "ceil(T/2)"
    return exprs


def build(cfg: Config) -> ModelSpec:
    variables = jax.eval_shape(lambda: init_variables(cfg, jax.random.key(0)))
    spec, trips, exprs = input_spec(cfg), trip_counts(cfg), trip_count_exprs(cfg)
    T, Te, H = cfg.feat_frames, enc_frames(cfg), cfg.encoder_n_hidden
    Lpre, Lpost, Lp = cfg.encoder_pre_rnn_layers, cfg.encoder_post_rnn_layers, cfg.pred_rnn_layers
    notes = {
        "preprocess": (f"waveform [1, {cfg.n_samples}] ({cfg.n_samples} samples at "
                       f"{cfg.sample_rate} Hz) -> log-mel features [1, {in_features(cfg)}, {T}] "
                       "(80 mels x frame splicing 3, per-feature normalized) + length"),
        "encoder_stack_time": (f"StackTime x{cfg.encoder_stack_time_factor}: pre_rnn output "
                               f"[{T}, 1, {H}] -> [{Te}, 1, {cfg.encoder_stack_time_factor * H}]"
                               " (zero-pads an odd frame count)"),
        "prediction_sos": ("ScriptGreedyDecoder._pred_step(SOS, None): prediction(None, None), "
                           "zero input and zero LSTM state; runs until the first symbol is "
                           "emitted (assumed at encoder frame 0)"),
        "prediction_step": (f"prediction(label, (h, c)): embedding + {Lp}-layer LSTM step; "
                            f"one per greedy iteration: {Te} frames + {cfg.n_symbols} symbols "
                            "- 1 SOS call"),
        "joint": (f"joint(f_t, g): one per greedy iteration ({Te} encoder frames, each ending "
                  f"in a blank, + {cfg.n_symbols} emitted symbols); host argmax picks the next "
                  "label"),
    }
    for name, part, k in enc_layers(cfg):
        steps, n_in = _layer_io(cfg, name)
        notes[f"encoder_{name}_in_proj"] = (
            f"encoder {part} LSTM layer {k}: input projection of the whole sequence "
            f"[{steps}, 1, {n_in}] -> x W_ih^T + b_ih [{steps}, 1, {4 * H}] (one INT8 GEMM, "
            "fp16 out)")
        notes[f"encoder_{name}_step"] = (
            f"encoder {part} LSTM layer {k}: one recurrence step (h W_hh^T + b_hh + xw_t, "
            f"gates, cell update), {steps} steps; the host feeds xw[t] and stacks h")
    keys = param_keys(cfg)
    phases = [Phase(ph, phase_fn(cfg, ph), spec[ph], trip_count=trips[ph], note=notes[ph],
                    param_keys=keys[ph], trip_count_expr=exprs.get(ph))
              for ph in phase_names(cfg)]
    return ModelSpec(
        name="rnnt",
        variant=(f"RNN-T (MLPerf Inference speech recognition, LibriSpeech), encoder "
                 f"{Lpre}+{Lpost} x LSTM {cfg.encoder_n_hidden} (split layer-major into "
                 f"input-projection and step phases), prediction {Lp} x LSTM "
                 f"{cfg.pred_n_hidden}, joint {cfg.joint_n_hidden}, vocab {cfg.num_classes}, "
                 f"greedy decode, batch {cfg.batch}; MLPerf v4.0 NVIDIA: INT8 encoder LSTMs "
                 f"(per-row int8 weights, int8 inputs / hidden states), {cfg.dtype} prediction "
                 "network / joint / gate math, fp32 preprocessing"),
        phases=phases,
        param_specs=as_spec(variables),
        excluded=[
            "WAV decode and resampling into the in-memory QSL (dataset loading, untimed)",
            "feature.permute(2, 0, 1) between preprocess and encoder (layout glue in "
            "pytorch_SUT.issue_queries) and the fp32 -> fp16 cast of the features "
            "(NVIDIA input_dtype fp16)",
            "encoder time loop: feeding xw[t] of each layer's input projection to its step "
            "phase, stacking the step outputs into the next layer's input sequence, the "
            "initial zero LSTM state, and the final [T/2, B, H] -> [B, T/2, H] transpose "
            "(host glue of the layer-major split encoder)",
            "greedy-decode host loop: per-frame slicing of the encoder output, argmax, blank "
            "test, max_symbols_per_step=30, label shift (k > blank -> k - 1) and state "
            "feedback (ScriptGreedyDecoder._greedy_decode)",
            "labels -> characters (postprocessing, accuracy script)",
        ],
    )
