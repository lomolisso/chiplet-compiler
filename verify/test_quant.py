"""Unit tests of models/quant.py on tiny arrays (the library, not the models).

Each quantized op must equal the same op in fp32 applied to the dequantized operands
(integer accumulation is exact; fp8 products are exact in fp32), and must lower to the
intended StableHLO operand / accumulator types.

    python -m pytest verify/test_quant.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from models import quant as q

RTOL = 1e-5


def _init_apply(mod, x, **overrides):
    v = mod.init(jax.random.key(0), x)
    p = jax.tree_util.tree_map(lambda a: a, v["params"])
    p.update(overrides)
    return {"params": p}, mod.apply({"params": p}, x)


def test_quantize_roundtrip():
    x = jax.random.normal(jax.random.key(1), (16, 8))
    for kind in ("int8", "fp8"):
        s = q.dynamic_scale(x, kind, axis=None)
        xr = q.dequantize(q.quantize(x, s, kind), s)
        bound = s / 2 if kind == "int8" else jnp.abs(x) * 2.0 ** -4 + 1e-6
        assert bool(jnp.all(jnp.abs(xr - x) <= bound + 1e-6)), kind


def test_asymmetric_int8():
    x = jax.random.uniform(jax.random.key(2), (32,), minval=0.0, maxval=6.0)  # post-ReLU6
    s, zp = 6.0 / 255, jnp.int32(-128)
    xq = q.quantize(x, s, "int8", zp)
    assert xq.dtype == jnp.int8 and int(xq.min()) >= -128
    assert float(jnp.max(jnp.abs(q.dequantize(xq, s, zero_point=zp) - x))) <= s / 2 + 1e-6


@pytest.mark.parametrize("scheme", [q.INT8_STATIC, q.INT8_TENSOR_W, q.INT8_DYNAMIC_TOKEN,
                                    q.FP8_STATIC])
def test_qdense_matches_dequantized(scheme):
    x = jax.random.normal(jax.random.key(3), (4, 64))
    mod = q.QDense(32, scheme)
    v, y = _init_apply(mod, x)
    p = v["params"]
    w = q.dequantize(p["kernel"], p["kernel_scale"])
    if scheme.act == "static":
        xs = p["input_scale"]
    else:
        xs = q.dynamic_scale(x, scheme.act_kind, -1)
    xd = q.dequantize(q.quantize(x, xs, scheme.act_kind), xs)
    np.testing.assert_allclose(y, xd @ w + p["bias"], rtol=RTOL, atol=1e-5)


def test_block_scaled_dot():
    K, N = 256, 256
    x = jax.random.normal(jax.random.key(4), (3, K))
    mod = q.QDense(N, q.FP8_BLOCK128, use_bias=False)
    v = mod.init(jax.random.key(5), x)
    ws = jax.random.uniform(jax.random.key(6), (2, 2), minval=1e-3, maxval=1e-2)
    v = {"params": {**v["params"], "kernel_scale": ws}}
    y = mod.apply(v, x)
    w = v["params"]["kernel"].astype(jnp.float32) * jnp.repeat(jnp.repeat(ws, 128, 0), 128, 1)
    xg = x.reshape(3, 2, 128)
    xs = q.dynamic_scale(xg, "fp8")
    xd = (q.quantize(xg, xs, "fp8").astype(jnp.float32) * xs).reshape(3, K)
    np.testing.assert_allclose(y, xd @ w, rtol=1e-4, atol=1e-4)


def test_block_scaled_dot_int8_batched():
    """GGML-style int4 weights x per-32 int8 activations, batched over experts."""
    E, M, K, N = 2, 3, 64, 8
    x = jax.random.normal(jax.random.key(20), (E, M, K))
    w = jax.random.randint(jax.random.key(21), (E, K, N), -8, 8).astype(jnp.int4)
    ws = jax.random.uniform(jax.random.key(22), (E, K // 32, N), minval=1e-3, maxval=1e-2)
    y = q.block_scaled_dot(x, w, ws, (32, 1), 32, kind="int8", batch_dims=1)
    xg = x.reshape(E, M, 2, 32)
    xs = q.dynamic_scale(xg, "int8")
    xd = (q.quantize(xg, xs, "int8").astype(jnp.float32) * xs).reshape(E, M, K)
    wd = w.astype(jnp.float32) * jnp.repeat(ws, 32, axis=1)
    np.testing.assert_allclose(y, jnp.einsum("emk,ekn->emn", xd, wd), rtol=1e-4, atol=1e-5)
    txt = jax.jit(lambda a, b, c: q.block_scaled_dot(a, b, c, (32, 1), 32, "int8", 1)).lower(x, w, ws).as_text()
    assert "xi8>) -> tensor<2x2x3x8xi32>" in txt


def test_int4_full_range():
    x = jnp.array([-1.0, 0.0, 0.875])
    assert q.quantize(x, 0.125, "int4", narrow=False).astype(jnp.int32).tolist() == [-8, 0, 7]
    assert q.quantize(x, 0.125, "int4").astype(jnp.int32).tolist() == [-7, 0, 7]


def test_qconv_tflite_zero_point_padding():
    """Asymmetric int8 conv padded with the zero point == float conv of dequantized x, 0-padded."""
    x = jax.random.uniform(jax.random.key(7), (1, 6, 6, 4), minval=0.0, maxval=6.0)
    s, zp = 6.0 / 255, -128
    for groups, cout in ((1, 8), (4, 4)):                  # dense and depthwise
        mod = q.QConv(cout, (3, 3), q.QScheme("int8", act_zero_point=True), strides=(2, 2),
                      padding=[(1, 1), (1, 1)], feature_group_count=groups, use_bias=False)
        v = mod.init(jax.random.key(8), x)
        p = {**v["params"], "input_scale": jnp.float32(s), "input_zero_point": jnp.int32(zp)}
        y = mod.apply({"params": p}, x)
        xd = q.dequantize(q.quantize(x, s, "int8", zp), s, zero_point=zp)
        w = q.dequantize(p["kernel"], p["kernel_scale"])
        ref = lax.conv_general_dilated(xd, w, (2, 2), [(1, 1), (1, 1)],
                                       dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                       feature_group_count=groups)
        # zero-point correction is folded into the int32 bias offline; here bias = 0, so
        # the conv sees (q - 0): compare against the conv of q*s, minus zp*s*sum(w) per window.
        corr = lax.conv_general_dilated(jnp.pad(jnp.full_like(x, zp * s), [(0, 0), (1, 1), (1, 1), (0, 0)],
                                                constant_values=zp * s), w, (2, 2), "VALID",
                                        dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                        feature_group_count=groups)
        np.testing.assert_allclose(y, ref + corr, rtol=1e-4, atol=1e-4)


def test_int8_output_requant():
    x = jax.random.normal(jax.random.key(9), (2, 16))
    mod = q.QDense(8, q.INT8_TFLITE_FC)
    v = mod.init(jax.random.key(10), x)
    y = mod.apply(v, x)
    assert y.dtype == jnp.int8


def test_qeinsum_attention():
    a = jax.random.normal(jax.random.key(11), (2, 4, 8))
    b = jax.random.normal(jax.random.key(12), (2, 5, 8))
    for kind in ("int8", "fp8"):
        sa, sb = q.dynamic_scale(a, kind, None), q.dynamic_scale(b, kind, None)
        y = q.qeinsum("bqd,bkd->bqk", a, b, sa, sb, kind, jnp.float32)
        ad = q.dequantize(q.quantize(a, sa, kind), sa)
        bd = q.dequantize(q.quantize(b, sb, kind), sb)
        np.testing.assert_allclose(y, jnp.einsum("bqd,bkd->bqk", ad, bd), rtol=1e-4, atol=1e-5)


def test_weight_only_int4_block():
    x = jax.random.normal(jax.random.key(13), (2, 64))
    sc = q.QScheme("int4", weight_gran="block", weight_block=(32, 1), act="none")
    mod = q.QDense(16, sc, use_bias=False, dtype="bfloat16")
    v = mod.init(jax.random.key(14), x)
    assert v["params"]["kernel"].dtype == jnp.int4
    assert v["params"]["kernel_scale"].shape == (2, 16)
    assert mod.apply(v, x).dtype == jnp.bfloat16


def test_lowered_types():
    """The GEMMs lower with low-precision operands and wide accumulators."""
    x = jnp.zeros((4, 64))
    for scheme, sig in ((q.INT8_STATIC, "tensor<4x64xi8>, tensor<64x32xi8>) -> tensor<4x32xi32>"),
                        (q.FP8_STATIC, "tensor<4x64xf8E4M3FN>, tensor<64x32xf8E4M3FN>) -> tensor<4x32xf32>")):
        mod = q.QDense(32, scheme)
        v = mod.init(jax.random.key(0), x)
        txt = jax.jit(mod.apply).lower(v, x).as_text()
        assert sig in txt, txt


def test_embeddings():
    ids = jnp.array([[0, 3], [5, 1]])
    for mod in (q.QEmbed(8, 4), q.QEmbed(8, 4, per_row=True), q.RowwiseQEmbed(8, 4),
                q.RowwiseQEmbed(8, 4, bits=4)):
        v = mod.init(jax.random.key(0), ids)
        assert mod.apply(v, ids).shape == (2, 2, 4)
