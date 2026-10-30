# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base layers."""

import math

from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
from models import quant as _q  # quantized: GGUF Q4_0 / Q6_K (dnn-graph-decomp)

# quantized: GGUF block sizes (ggml-common.h QK4_0, QK_K and the Q6_K 16-weight
# sub-block). Gemma 4 QAT Q4_0 release: google/gemma-4-26B-A4B-it-qat-q4_0-gguf.
Q4_0_BLOCK = 32
Q6_K_BLOCK = 256
Q6_K_SUB = 16


def q4_0_init(key, shape, dtype=jnp.int4):
  """quantized: Q4_0 value init, x = d * q with q in [-8, 7] (ggml q - 8)."""
  return jax.random.randint(key, shape, -8, 8, jnp.int32).astype(dtype)


def q4_0_scale_shape(shape: tuple[int, ...], k_axis: int) -> tuple[int, ...]:
  """quantized: one fp16 scale d per 32 consecutive weights along K."""
  s = list(shape)
  s[k_axis] //= Q4_0_BLOCK
  return tuple(s)


def q4_0_einsum(eqn: str, x: jax.Array, w: jax.Array, w_scale: jax.Array):
  """quantized: GGUF Q4_0 weights x Q8_0 activations (llama.cpp ggml_vec_dot_q4_0_q8_0).

  The activation is quantized on the fly to int8 per 32 values along the
  contracted (K) axes (Q8_0: d = amax / 127), every 32-block is an
  int8 x int8 -> int32 dot (int4 values widened to int8), scaled in fp32 by
  d_w * d_x and summed. `w_scale` is `w`'s shape with the innermost contracted
  axis divided by 32 (GGUF blocks run along the flattened K of each row).
  Batch letters (in x, w and the output) are shared leading axes (experts).
  """
  ins, out = eqn.split('->')
  xs, ws = ins.split(',')
  n_ell = 0
  if xs.startswith('...'):
    xs, out = xs[3:], out[3:]
    n_ell = x.ndim - len(xs)
  batch = [c for c in ws if c in xs and c in out]
  contr = [c for c in ws if c in xs and c not in out]
  w_free = [c for c in ws if c not in xs]
  x_free = [c for c in xs if c not in ws]
  assert not batch or not n_ell
  # x -> [*B, ..., x_free, K]; w, scale -> [*B, K (/32), N]
  xp = ([n_ell + xs.index(c) for c in batch] + list(range(n_ell))
        + [n_ell + xs.index(c) for c in x_free + contr])
  xt = jnp.transpose(x, xp)
  k = math.prod(x.shape[n_ell + xs.index(c)] for c in contr)
  xt = xt.reshape(xt.shape[: xt.ndim - len(contr)] + (k,))
  wp = [ws.index(c) for c in batch + contr + w_free]
  nb = len(batch)
  bshape = tuple(w.shape[ws.index(c)] for c in batch)
  n = math.prod(w.shape[ws.index(c)] for c in w_free)
  wt = jnp.transpose(w, wp).reshape(bshape + (k, n))
  st = jnp.transpose(w_scale, wp).reshape(bshape + (k // Q4_0_BLOCK, n))
  y = _q.block_scaled_dot(xt, wt, st, weight_block=(Q4_0_BLOCK, 1),
                          act_block=Q4_0_BLOCK, kind='int8', batch_dims=nb)
  # [*B | ..., x_free, w_free] -> output letter order
  y = y.reshape(y.shape[:-1] + tuple(w.shape[ws.index(c)] for c in w_free))
  cur = batch + x_free + w_free
  yp = list(range(n_ell)) + [n_ell + cur.index(c) for c in out]
  return jnp.transpose(y, yp).astype(x.dtype)


def q6_k_init(key, shape, dtype=jnp.int8):
  """quantized: Q6_K values q - 32 in [-32, 31]."""
  return jax.random.randint(key, shape, -32, 32, jnp.int32).astype(dtype)


def q6_k_dequant(q: jax.Array, sc: jax.Array, d: jax.Array, dtype):
  """quantized: Q6_K rows -> dtype: x = d * sc * (q - 32) (ggml dequantize_row_q6_K)."""
  w = (q.astype(jnp.float32)
       * jnp.repeat(sc.astype(jnp.float32), Q6_K_SUB, axis=-1)
       * jnp.repeat(d.astype(jnp.float32), Q6_K_BLOCK, axis=-1))
  return w.astype(dtype)


def q6_k_dot(x: jax.Array, q: jax.Array, sc: jax.Array, d: jax.Array):
  """quantized: y = x @ W^T with W [N, K] in Q6_K and x in Q8_K (llama.cpp
  ggml_vec_dot_q6_K_q8_K): x quantized to int8 per 256 along K (d_x = amax /
  127); per 16-weight sub-block an int8 x int8 -> int32 dot, multiplied by the
  int8 sub-scale and summed over the 256 super-block in int32; then fp32
  d_w * d_x and the sum over super-blocks."""
  kdim, n = x.shape[-1], q.shape[0]
  kb, ns = kdim // Q6_K_BLOCK, Q6_K_BLOCK // Q6_K_SUB
  xf = x.reshape((-1, kb, Q6_K_BLOCK))
  m = xf.shape[0]
  xs = _q.dynamic_scale(xf, 'int8', axis=-1)                         # [M, kb, 1]
  xq = _q.quantize(xf, xs, 'int8').reshape(m, kb, ns, Q6_K_SUB)
  wq = q.reshape(n, kb, ns, Q6_K_SUB)
  isum = lax.dot_general(xq, wq, (((3,), (3,)), ((1, 2), (1, 2))),
                         preferred_element_type=jnp.int32)          # [kb, ns, M, N]
  scs = jnp.transpose(sc.reshape(n, kb, ns).astype(jnp.int32), (1, 2, 0))
  sumi = jnp.sum(isum * scs[:, :, None, :], axis=1)                # [kb, M, N] int32
  dw = jnp.transpose(d.reshape(n, kb).astype(jnp.float32))          # [kb, N]
  dx = jnp.transpose(xs[..., 0])                                    # [kb, M]
  y = sumi.astype(jnp.float32) * dx[:, :, None] * dw[:, None, :]
  return jnp.sum(y, axis=0).reshape(x.shape[:-1] + (n,))


class Einsum(nn.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  shape: tuple[int, ...]
  weight_name: str = 'w'
  initializer: nn.initializers.Initializer = nn.initializers.normal()
  dtype: jnp.dtype | None = None
  w_scale: float | None = None
  # quantized: 'q4_0' = GGUF Q4_0 weights (int4 + fp16 scale per 32 along K)
  # times Q8_0 activations; None = float (router, vision, projections).
  quant: str | None = None

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    if self.quant == 'q4_0':  # quantized: Q4_0 x Q8_0 integer dot
      ins, out = eqn.split('->')
      xs, ws = ins.split(',')
      k_axis = max(i for i, c in enumerate(ws) if c in xs and c not in out)
      w = self.param(self.weight_name, q4_0_init, self.shape, jnp.int4)
      if isinstance(w, dict) and 'w' in w:
        w = w['w']
      d = self.param(f'{self.weight_name}_scale',
                     nn.initializers.constant(2.0**-8),
                     q4_0_scale_shape(self.shape, k_axis), jnp.float16)
      return q4_0_einsum(eqn, x, w, d)
    w = self.param(
        self.weight_name,
        self.initializer,
        self.shape,
        self.dtype if self.dtype is not None else None,
    )
    # Workaround for behavior with nn.share_scope in parent modules:
    # self.param might return a dict {'w': tensor} instead of the bare tensor.
    # The key appears to be the default class weight_name 'w'.
    if isinstance(w, dict):
      if 'w' in w:
        w = w['w']
    if self.w_scale is not None:
      w *= self.w_scale
    return jnp.einsum(eqn, x, w)


class ClippedEinsum(nn.Module):
  """Einsum with input and output activation clamping."""

  shape: tuple[int, ...]
  weight_name: str = 'w'
  initializer: nn.initializers.Initializer = nn.initializers.normal()
  dtype: jnp.dtype | None = None
  w_scale: float | None = None

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    w = self.param(
        self.weight_name,
        self.initializer,
        self.shape,
        self.dtype if self.dtype is not None else None,
    )
    if isinstance(w, dict):
      if 'w' in w:
        w = w['w']
    if self.w_scale is not None:
      w *= self.w_scale

    inf = float('inf')
    clip_input_min = self.param(
        'clip_input_min', lambda key, shape, dtype=None: jnp.array(-inf), ()
    )
    clip_input_max = self.param(
        'clip_input_max', lambda key, shape, dtype=None: jnp.array(inf), ()
    )
    clip_output_min = self.param(
        'clip_output_min', lambda key, shape, dtype=None: jnp.array(-inf), ()
    )
    clip_output_max = self.param(
        'clip_output_max', lambda key, shape, dtype=None: jnp.array(inf), ()
    )

    x = jnp.clip(x, clip_input_min, clip_input_max)
    x = jnp.einsum(eqn, x, w)
    x = jnp.clip(x, clip_output_min, clip_output_max)
    return x


class RMSNorm(nn.Module):
  """RMSNorm layer."""

  with_scale: bool = True

  @nn.compact
  def __call__(self, x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

    # Jax.lax.rsqrt is used because it returns different floats than
    # jnp.reciprocal(jnp.sqrt(var + 1e-06))
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

    if self.with_scale:
      scale = self.param(
          'scale',
          nn.initializers.ones,
          (x.shape[-1]),
      )
      # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale
      # is a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
      # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
      scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
      normed_inputs = normed_inputs * scale
    return normed_inputs
