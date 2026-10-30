"""RNN-T (MLPerf Inference speech recognition, retired), Flax linen port.

Mirrors mlcommons/inference retired_benchmarks/speech_recognition/rnnt/pytorch:
  * preprocessing.py / parts/features.py  AudioPreprocessing -> FilterbankFeatures
    (preemphasis, STFT, power, + dither^2, mel filterbank, log, frame splicing x3,
    per-feature normalization)
  * model_separable_rnnt.py / rnn.py      RNNT = Encoder (pre_rnn LSTM, StackTime,
    post_rnn LSTM) + Prediction (Embedding + LSTM) + Joint (Linear, ReLU, Linear)

LSTMs follow torch.nn.LSTM exactly (gate order i, f, g, o; separate b_ih and b_hh;
zero initial state when none is given) and ATen's native CPU algorithm
(aten/src/ATen/native/RNN.cpp, FullLayer/LSTMCell): per layer, one input
projection over the whole sequence, then one hidden projection per time step.
Time is unrolled in Python, so the graph has no control flow. Module and parameter
names follow the PyTorch reference (`net.0` -> `net_0`, `weight_ih_l0` -> `l0/weight_ih`).

Quantization: the MLPerf Inference v4.0 NVIDIA engines (inference_results_v4.0
closed/NVIDIA/code/rnnt/tensorrt/rnnt_network.py, configs/rnnt):
  * encoder LSTMs INT8 (RNNTEncoderPlugin, dataType INT8): W_ih / W_hh int8 with per-row
    symmetric scales (calibration.md: plugins quantize weights per row, max-abs), the layer
    input and the hidden state h int8 with per-tensor static scales (min-max calibration:
    `input`, `encoder_reshape`, `lstmN_hidden` in calibrator.cache), i8 x i8 -> i32 GEMMs
    with an fp32 epilogue, gate math and cell state c in fp16; pre_rnn output and StackTime
    stay int8; the encoder output (last post_rnn layer) is fp16;
  * prediction network (RNNTDecoderPlugin, HALF) and joint: fp16;
  * preprocessing (DALI): fp32.
The plugin source is not published: the h scale (calibrated range ~1) and fp16 gate math are
inferred from what the builder passes (fp16 weights + one input scale per stack + INT8);
Intel's published v4.0 kernel (quant_lstm.py) has the same shape (int8 x and h, float c).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from models import quant as q


@dataclass(frozen=True)
class Config:
    # [input_eval] of configs/rnnt.toml
    sample_rate: int = 16000
    window_size: float = 0.02
    window_stride: float = 0.01
    n_fft: int = 512
    features: int = 80
    frame_splicing: int = 3
    dither: float = 0.00001
    preemph: float = 0.97          # FilterbankFeatures default (not set by the toml)
    # [rnnt]
    encoder_n_hidden: int = 1024
    encoder_pre_rnn_layers: int = 2
    encoder_stack_time_factor: int = 2
    encoder_post_rnn_layers: int = 3
    pred_n_hidden: int = 320
    pred_rnn_layers: int = 2
    joint_n_hidden: int = 512
    num_classes: int = 29          # 28 labels + blank
    # float type of the non-INT8 parts (MLPerf NVIDIA: encoder gate math / cell state,
    # prediction network and joint in fp16; preprocessing stays fp32)
    dtype: str = "float16"
    # scenario: batch 1, longest LibriSpeech dev-clean utterance kept by the MLPerf QSL
    batch: int = 1
    n_samples: int = 239920        # 2086-149220-0012.wav, 14.995 s
    n_symbols: int = 276           # characters of its transcript (greedy-decode trip count)
    # Feature-frame count of the encoder phases, if not derived from n_samples: lets them
    # be exported with the frame count as its own symbol (the derivation is a floor
    # division, not a linear size).
    frames: int | None = None
    # Encoder frame count after StackTime (post-rnn length), if not ceil(frames / factor);
    # its own symbol for the post-rnn phases (a ceil is not a linear size).
    post_frames: int | None = None

    @property
    def win_length(self) -> int:
        return int(self.sample_rate * self.window_size)

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.window_stride)

    @property
    def stft_frames(self) -> int:  # torch.stft(center=True)
        return 1 + self.n_samples // self.hop_length

    @property
    def feat_frames(self) -> int:  # FilterbankFeatures.get_seq_len of an unpadded utterance
        if self.frames is not None:
            return self.frames
        return feat_frames_of(self, self.n_samples)


def feat_frames_of(cfg: Config, n_samples):
    """get_seq_len of an unpadded utterance of n_samples samples (int or symbolic)."""
    seq = (n_samples + cfg.hop_length - 1) // cfg.hop_length
    return (seq + cfg.frame_splicing - 1) // cfg.frame_splicing


# ----------------------------------------------------------------------------
# constants of FilterbankFeatures (registered buffers in the reference)
# ----------------------------------------------------------------------------

def _hz_to_mel(f):
    """librosa.hz_to_mel(htk=False): Slaney's auditory toolbox scale."""
    f = np.asarray(f, np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz)
                    / logstep, f / f_sp)


def _mel_to_hz(m):
    m = np.asarray(m, np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) with its defaults
    (htk=False, norm='slaney', dtype=float32): [n_mels, 1 + n_fft // 2]."""
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return weights


def hann_window(n: int) -> np.ndarray:
    """torch.hann_window(n, periodic=False)."""
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / (n - 1))).astype(np.float32)


# ----------------------------------------------------------------------------
# AudioPreprocessing (preprocessing.py) / FilterbankFeatures (parts/features.py)
# ----------------------------------------------------------------------------

def get_seq_len(cfg: Config, seq_len):
    seq_len = (seq_len + cfg.hop_length - 1) // cfg.hop_length
    return (seq_len + cfg.frame_splicing - 1) // cfg.frame_splicing


def _stft_power(cfg: Config, x):
    """|torch.stft(x, n_fft, hop, win_length, window, center=True, pad_mode='reflect')|^2
    as [B, n_fft//2 + 1, frames]. Frames are cut with static slices of a
    [blocks, hop] view of the padded signal (torch: Tensor.unfold, also a view).

    The frame count comes from the signal's shape (symbolic lengths). The tail is
    zero-padded by the most any length needs (hop - n_fft % hop) and cut to n_blocks * hop;
    the old pad of n_blocks * hop - padded length was negative when N % hop > hop - rem."""
    n_fft, hop = cfg.n_fft, cfg.hop_length
    B, T = x.shape[0], 1 + x.shape[1] // hop                        # torch.stft(center=True)
    x = jnp.pad(x, ((0, 0), (n_fft // 2, n_fft // 2)), mode="reflect")
    full, rem = divmod(n_fft, hop)
    n_blocks = T + full
    x = jnp.pad(x, ((0, 0), (0, hop - rem)))[:, :n_blocks * hop]
    blocks = x.reshape(B, n_blocks, hop)
    parts = [blocks[:, i:i + T] for i in range(full)] + [blocks[:, full:full + T, :rem]]
    frames = jnp.concatenate(parts, axis=2)                        # [B, T, n_fft]
    left = (n_fft - cfg.win_length) // 2                           # torch pads the window
    window = np.zeros(n_fft, np.float32)
    window[left:left + cfg.win_length] = hann_window(cfg.win_length)
    spec = jnp.fft.rfft(frames * window, n=n_fft, axis=-1)          # [B, T, n_fft//2 + 1]
    power = jnp.real(spec) ** 2 + jnp.imag(spec) ** 2
    return jnp.swapaxes(power, 1, 2)


class FilterbankFeatures(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, seq_len):
        c = self.cfg
        # The utterance is unpadded (batch 1, length == n_samples), so get_seq_len(seq_len)
        # is static: the feature-frame count of the signal's length (a symbolic expression
        # for a symbolic length); it bounds the normalization and the final slice.
        n = feat_frames_of(c, x.shape[1])
        x = jnp.concatenate([x[:, :1], x[:, 1:] - c.preemph * x[:, :-1]], axis=1)
        x = _stft_power(c, x)
        x = x + c.dither ** 2                                       # deterministic dithering
        fb = mel_filterbank(c.sample_rate, c.n_fft, c.features, 0.0, c.sample_rate / 2)
        x = jnp.matmul(fb[None], x)                                 # [B, features, T]
        x = jnp.log(x + 1e-20)
        seq = [x] + [jnp.pad(x[:, :, k:], ((0, 0), (0, 0), (0, k)))
                     for k in range(1, c.frame_splicing)]
        x = jnp.concatenate(seq, axis=1)[:, :, ::c.frame_splicing]
        mean = jnp.mean(x[:, :, :n], axis=2)
        std = jnp.std(x[:, :, :n], axis=2, ddof=1) + 1e-5
        x = (x - mean[:, :, None]) / std[:, :, None]
        return x[:, :, :n]


class AudioPreprocessing(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, waveform, length):
        feats = FilterbankFeatures(self.cfg, name="featurizer")(waveform, length)
        return feats, get_seq_len(self.cfg, length)


# ----------------------------------------------------------------------------
# LSTM (torch.nn.LSTM, unrolled): INT8 encoder layers, fp16 prediction layers
# ----------------------------------------------------------------------------

def _uniform(scale: float):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -scale, scale)
    return init


def _int8_matmul(x_q, w_q, x_scale, w_scale, bias):
    """x_q [..., K] int8 @ w_q^T ([N, K] int8, torch layout) -> i32, fp32 epilogue
    (x_scale * per-row w_scale, + bias)."""
    acc = lax.dot_general(x_q, w_q, (((x_q.ndim - 1,), (1,)), ((), ())),
                          preferred_element_type=jnp.int32)
    return acc.astype(jnp.float32) * (x_scale * w_scale) + bias


class LSTMLayer(nn.Module):
    """One torch.nn.LSTM layer.

    quantized=True (encoder, MLPerf NVIDIA RNNTEncoderPlugin INT8): W_ih and W_hh int8 with
    per-row scales (`weight_ih_scale`, `weight_hh_scale`); the layer input x and the hidden
    state h are int8 with per-tensor static scales (`input_scale`, `hidden_scale`); both
    GEMMs are i8 x i8 -> i32 with an fp32 epilogue (scales, b_ih / b_hh); the gate math and
    the cell state c are fp16 (`dtype`); h is requantized to int8 for the recurrence and the
    next layer. out_float: the layer's output sequence is h in fp16 (the encoder output)
    instead of int8.
    quantized=False (prediction network, RNNTDecoderPlugin fp16): fp16 weights and math.
    """
    hidden: int
    # "sequence": the whole layer, unrolled over time (as the encoder module runs it).
    # "in_proj" / "step": its two parts as separate phases (the encoder is split
    # layer-major): the whole-sequence input projection, and one recurrence step.
    mode: str = "sequence"
    quantized: bool = False
    out_float: bool = False
    dtype: str = "float16"

    def _in_proj(self, xs):
        H = self.hidden
        init = _uniform(1.0 / math.sqrt(H))
        if not self.quantized:
            w_ih = self.param("weight_ih", init, (4 * H, xs.shape[-1]), self.dtype)
            b_ih = self.param("bias_ih", init, (4 * H,), self.dtype)
            return jnp.matmul(xs, w_ih.T) + b_ih
        w_ih = self.param("weight_ih", q.rand_q("int8"), (4 * H, xs.shape[-1]), jnp.int8)
        w_s = self.param("weight_ih_scale", q.const(3.0 / math.sqrt(H) / 127), (4 * H,),
                         jnp.float32)
        b_ih = self.param("bias_ih", init, (4 * H,), jnp.float32)
        x_s = self.param("input_scale", q.const(4.0 / 127), (), jnp.float32)
        if xs.dtype != jnp.int8:                    # stack input (features): calibrated scale
            xs = q.quantize(xs, x_s, "int8")
        return _int8_matmul(xs, w_ih, x_s, w_s, b_ih).astype(self.dtype)

    def _hh_params(self):
        H = self.hidden
        init = _uniform(1.0 / math.sqrt(H))
        if not self.quantized:
            return (self.param("weight_hh", init, (4 * H, H), self.dtype),
                    self.param("bias_hh", init, (4 * H,), self.dtype))
        return (self.param("weight_hh", q.rand_q("int8"), (4 * H, H), jnp.int8),
                self.param("weight_hh_scale", q.const(3.0 / math.sqrt(H) / 127), (4 * H,),
                           jnp.float32),
                self.param("bias_hh", init, (4 * H,), jnp.float32),
                self.param("hidden_scale", q.const(1.0 / 127), (), jnp.float32))

    def _step(self, p, xw_t, h, c):
        """One time step; returns (h for the recurrence (int8 if quantized), c, h in float)."""
        if not self.quantized:
            w_hh, b_hh = p
            gates = (jnp.matmul(h, w_hh.T) + b_hh) + xw_t
        else:
            w_hh, w_s, b_hh, h_s = p
            gates = (_int8_matmul(h, w_hh, h_s, w_s, b_hh) + xw_t).astype(self.dtype)
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i, f, g, o = nn.sigmoid(i), nn.sigmoid(f), jnp.tanh(g), nn.sigmoid(o)
        c = f * c + i * g
        h = o * jnp.tanh(c)
        if not self.quantized:
            return h, c, h
        return q.quantize(h, p[3], "int8"), c, h

    @nn.compact
    def __call__(self, xs, h=None, c=None):
        """sequence: xs [T, B, in], h/c [B, H] -> (ys [T, B, H], h_T, c_T).
        in_proj:  xs [T, B, in] -> xw [T, B, 4H] (only the W_ih side).
        step:     xs = xw_t [B, 4H], h/c [B, H] -> (h, c) (+ fp16 h if out_float; only the
                  W_hh side)."""
        if self.mode == "in_proj":
            return self._in_proj(xs)
        if self.mode == "step":
            h, c, y = self._step(self._hh_params(), xs, h, c)
            return (h, c, y) if self.out_float else (h, c)
        xw = self._in_proj(xs)                                      # whole-sequence projection
        p = self._hh_params()
        ys = []
        for t in range(xs.shape[0]):
            h, c, y = self._step(p, xw[t], h, c)
            ys.append(y if self.out_float else h)
        return jnp.stack(ys), h, c


class LSTM(nn.Module):
    hidden: int
    num_layers: int
    quantized: bool = False
    out_float: bool = False          # last layer's output sequence in fp16 (quantized stacks)
    dtype: str = "float16"

    @nn.compact
    def __call__(self, x, state=None):
        """x [T, B, in]; state (h0, c0) each [L, B, H] or None (zeros; h int8 when quantized)."""
        if state is None:
            B = x.shape[1]
            h0 = jnp.zeros((self.num_layers, B, self.hidden),
                           jnp.int8 if self.quantized else self.dtype)
            state = (h0, jnp.zeros((self.num_layers, B, self.hidden), self.dtype))
        hs, cs = [], []
        for k in range(self.num_layers):
            last = k == self.num_layers - 1
            x, h, c = LSTMLayer(self.hidden, quantized=self.quantized,
                                out_float=self.out_float and last, dtype=self.dtype,
                                name=f"l{k}")(x, state[0][k], state[1][k])
            hs.append(h)
            cs.append(c)
        return x, (jnp.stack(hs), jnp.stack(cs))


class LstmDrop(nn.Module):
    """rnn.LstmDrop: torch.nn.LSTM (+ inter-layer and in-place dropout, identity in eval)."""
    hidden: int
    num_layers: int
    quantized: bool = False
    out_float: bool = False
    dtype: str = "float16"

    @nn.compact
    def __call__(self, x, state=None):
        return LSTM(self.hidden, self.num_layers, self.quantized, self.out_float, self.dtype,
                    name="lstm")(x, state)


class StackTime(nn.Module):
    factor: int

    @nn.compact
    def __call__(self, x, x_lens):
        r = jnp.swapaxes(x, 0, 1)                                    # [B, T, H]
        B, T, H = r.shape
        Tp = -(-T // self.factor)            # ceil: the padded length, written provably
        r = jnp.pad(r, ((0, 0), (0, Tp * self.factor - T), (0, 0)))
        r = r.reshape(B, Tp, H * self.factor)
        x_lens = jnp.ceil(x_lens.astype(jnp.float32) / self.factor).astype(jnp.int32)
        return jnp.swapaxes(r, 0, 1), x_lens


# ----------------------------------------------------------------------------
# RNNT (model_separable_rnnt.py)
# ----------------------------------------------------------------------------

class Encoder(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, x, x_lens):
        """x [T, B, in_features] -> ([B, T', H], x_lens')."""
        c = self.cfg
        x, _ = LstmDrop(c.encoder_n_hidden, c.encoder_pre_rnn_layers, quantized=True,
                        dtype=c.dtype, name="pre_rnn")(x)                    # int8 output
        x, x_lens = StackTime(c.encoder_stack_time_factor, name="stack_time")(x, x_lens)
        x, _ = LstmDrop(c.encoder_n_hidden, c.encoder_post_rnn_layers, quantized=True,
                        out_float=True, dtype=c.dtype, name="post_rnn")(x)  # fp16 output
        return jnp.swapaxes(x, 0, 1), x_lens


class Prediction(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, y, state=None):
        """y [B, U] int labels, or None (start of sequence: a zero input, B = 1)."""
        c = self.cfg
        if y is None:
            y = jnp.zeros((1, 1, c.pred_n_hidden), c.dtype)
        else:
            y = nn.Embed(c.num_classes - 1, c.pred_n_hidden, dtype=c.dtype, param_dtype=c.dtype,
                         name="embed")(y)
        y = jnp.swapaxes(y, 0, 1)                                    # [U, B, H]
        g, hid = LstmDrop(c.pred_n_hidden, c.pred_rnn_layers, dtype=c.dtype,
                          name="dec_rnn")(y, state)
        return jnp.swapaxes(g, 0, 1), hid


class Joint(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, f, g):
        """f [B, T, H_enc], g [B, U, H_pred] -> logits [B, T, U, num_classes]."""
        B, T, H = f.shape
        _, U, H2 = g.shape
        f = jnp.broadcast_to(f[:, :, None, :], (B, T, U, H))
        g = jnp.broadcast_to(g[:, None, :, :], (B, T, U, H2))
        x = jnp.concatenate([f, g], axis=3)
        dense = partial(nn.Dense, dtype=self.cfg.dtype, param_dtype=self.cfg.dtype)
        x = nn.relu(dense(self.cfg.joint_n_hidden, name="net_0")(x))
        return dense(self.cfg.num_classes, name="net_3")(x)


class RNNT(nn.Module):
    cfg: Config

    def setup(self):
        self.encoder = Encoder(self.cfg)
        self.prediction = Prediction(self.cfg)
        self.joint = Joint(self.cfg)

    def encode(self, x, x_lens):
        return self.encoder(x, x_lens)

    def predict(self, y, state):
        return self.prediction(y, state)

    def join(self, f, g):
        return self.joint(f, g)
