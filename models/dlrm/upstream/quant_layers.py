"""INT8 Keras layers for the quantized DLRMv2 (not part of keras-rs; written for this port).

`QDense` is the Keras counterpart of `models.quant.QDense` with the TensorRT implicit-INT8
arithmetic of MLPerf NVIDIA dlrm-v2-99 (inference_results_v4.1 closed/NVIDIA/code/dlrm-v2/
tensorrt/builder.py, calibrator.cache):

    x_q = quantize(x, input_scale)                       # per-tensor static (calibrated)
    acc = dot_general(x_q, kernel_int8, preferred=int32)  # i8 x i8 -> i32
    y   = activation(acc * (input_scale * kernel_scale) + bias)   # fp32 epilogue

Weights are int8 symmetric with one scale per tensor (`weight_scale="tensor"`, the MLPs:
TRT matrix-multiply layers with a calibrated constant-weight range) or one scale per output
channel (`"channel"`, the DCN V / W kernels: TRT 1x1 convolutions). The layers run on the
JAX backend only (models.dlrm requires it); the GEMM is `jax.lax.dot_general` so the graph
carries the int8 operands and the int32 accumulator.
"""

from __future__ import annotations

import keras
from jax import lax
import jax.numpy as jnp

from models import quant as q


class QDense(keras.layers.Layer):
    """INT8 W8A8 `keras.layers.Dense` (kernel [in, units], like Dense). The kernel and the
    scales are non-trainable variables (`kernel`, `kernel_scale`, `input_scale`); `bias` is
    fp32. `kernel_initializer` is accepted for signature compatibility with Dense and unused
    (the int8 kernel is converted from the float checkpoint)."""

    def __init__(self, units, activation=None, use_bias=True, weight_scale="tensor",
                 kernel_initializer=None, bias_initializer="zeros", kernel_regularizer=None,
                 bias_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        del kernel_initializer, kernel_regularizer, bias_regularizer
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.weight_scale = weight_scale
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        k_in = input_shape[-1]
        self.kernel = self.add_weight(name="kernel", shape=(k_in, self.units),
                                      initializer="zeros", dtype="int8", trainable=False)
        self.kernel_scale = self.add_weight(
            name="kernel_scale", shape=() if self.weight_scale == "tensor" else (self.units,),
            initializer=keras.initializers.Constant(0.05 / 127), dtype="float32",
            trainable=False)
        self.input_scale = self.add_weight(name="input_scale", shape=(),
                                           initializer=keras.initializers.Constant(4.0 / 127),
                                           dtype="float32", trainable=False)
        self.bias = None
        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(self.units,),
                                        initializer=self.bias_initializer, dtype="float32")
        self.built = True

    def call(self, inputs):
        s_in = keras.ops.convert_to_tensor(self.input_scale)
        s_w = keras.ops.convert_to_tensor(self.kernel_scale)
        x_q = q.quantize(inputs, s_in, "int8")
        acc = lax.dot_general(x_q, keras.ops.convert_to_tensor(self.kernel),
                              (((inputs.ndim - 1,), (0,)), ((), ())),
                              preferred_element_type=jnp.int32)
        y = acc.astype(jnp.float32) * (s_in * s_w)
        if self.bias is not None:
            y = y + keras.ops.convert_to_tensor(self.bias)
        return self.activation(y)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)
