# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Constructors for MLPs."""

from typing import Callable, Iterable, Optional

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


# TODO(aelkadi): Move the mlp factory here from `deep_typed_graph_net.py`.


# quantized: FP8 W8A8 (E4M3, per-tensor static scales) -- GPU-datacenter fallback.
# Fp8Linear / Fp8MLP are haiku's hk.Linear / hk.nets.MLP (dm-haiku 0.0.17,
# haiku/_src/basic.py and haiku/_src/nets/mlp.py, Copyright 2019 DeepMind
# Technologies Limited, Apache-2.0) with the GEMM in explicit Q/DQ form:
# x -> fp8 with `input_scale`, w stored fp8 with `w_scale`, f8 x f8 -> f32 dot,
# fp32 scale + bias epilogue, output in the input dtype (bf16 under
# casting.Bfloat16Cast). Module and parameter names are haiku's
# (`<name>/~/linear_<i>/{w,b}`); `w_scale` / `input_scale` are new. `w` and the
# scales are requested in their own dtypes, so casting._bfloat16_creator (which
# only rewrites bfloat16 requests) leaves them as stored; `b` is fp32.
FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.


def _quantize_fp8(x, scale):
  return jnp.clip(x.astype(jnp.float32) / scale, -FP8_MAX, FP8_MAX).astype(FP8)


class Fp8Linear(hk.Module):
  """hk.Linear with FP8 E4M3 weights and input, per-tensor static scales."""

  def __init__(self,
               output_size: int,
               with_bias: bool = True,
               w_init: Optional[hk.initializers.Initializer] = None,
               b_init: Optional[hk.initializers.Initializer] = None,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros

  def __call__(self, inputs: jax.Array) -> jax.Array:
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    inputs = jnp.asarray(inputs)
    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    stddev = 1. / np.sqrt(self.input_size)
    w_init = self.w_init or hk.initializers.TruncatedNormal(stddev=stddev)
    w_scale_init = 3. * stddev / FP8_MAX
    w_scale = hk.get_parameter(
        "w_scale", [], jnp.float32,
        init=hk.initializers.Constant(w_scale_init))
    w = hk.get_parameter(
        "w", [input_size, output_size], FP8,
        init=lambda s, d: _quantize_fp8(w_init(s, jnp.float32), w_scale_init))
    input_scale = hk.get_parameter(
        "input_scale", [], jnp.float32,
        init=hk.initializers.Constant(4. / FP8_MAX))

    out = lax.dot_general(
        _quantize_fp8(inputs, input_scale), w,
        (((inputs.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32)
    out = out * (input_scale * w_scale)

    if self.with_bias:
      b = hk.get_parameter("b", [self.output_size], jnp.float32,
                           init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    return out.astype(dtype)


class Fp8MLP(hk.Module):
  """hk.nets.MLP built from Fp8Linear layers (no dropout: inference only)."""

  def __init__(self,
               output_sizes: Iterable[int],
               w_init: Optional[hk.initializers.Initializer] = None,
               b_init: Optional[hk.initializers.Initializer] = None,
               with_bias: bool = True,
               activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
               activate_final: bool = False,
               name: Optional[str] = None):
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init
    self.activation = activation
    self.activate_final = activate_final
    layers = []
    output_sizes = tuple(output_sizes)
    for index, output_size in enumerate(output_sizes):
      layers.append(Fp8Linear(output_size=output_size,
                              w_init=w_init,
                              b_init=b_init,
                              with_bias=with_bias,
                              name="linear_%d" % index))
    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None

  def __call__(self, inputs: jax.Array) -> jax.Array:
    num_layers = len(self.layers)

    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        out = self.activation(out)

    return out


class LinearNormConditioning(hk.Module):
  """Module for norm conditioning.

  Conditions the normalization of "inputs" by applying a linear layer to the
  "norm_conditioning" which produces the scale and variance which are applied to
  each channel (across the last dim) of "inputs".
  """

  def __init__(self, name="norm_conditioning"):
    super().__init__(name=name)

  def __call__(self, inputs: jax.Array, norm_conditioning: jax.Array):

    feature_size = inputs.shape[-1]
    conditional_linear_layer = hk.Linear(
        output_size=2 * feature_size,
        w_init=hk.initializers.TruncatedNormal(stddev=1e-8),
    )
    conditional_scale_offset = conditional_linear_layer(norm_conditioning)
    scale_minus_one, offset = jnp.split(conditional_scale_offset, 2, axis=-1)
    scale = scale_minus_one + 1.
    return inputs * scale + offset
