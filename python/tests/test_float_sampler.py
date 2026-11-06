"""The float models of `cc_float` (docs/framework/float.md): rounding against numpy and ml_dtypes casts,
the error measure, the sizing formula against the composition notes' predictions, and quick composition
runs (8-bit formats, exhaustive) within the budget."""

import ml_dtypes
import numpy as np

from cc_float import compositions as C
from cc_float.fpmodel import BF16, E4M3FN, F16, F32, all_values, rnd, round_result, ulp_error
from cc_float.sampler import fp8_arith, row


def test_rounding_matches_the_casts():
    rng = np.random.default_rng(1)
    x = np.ldexp(rng.uniform(-2, 2, 20000), rng.integers(-30, 30, 20000))
    assert np.array_equal(rnd(x, 24), x.astype(np.float32).astype(np.float64))
    with np.errstate(over="ignore"):
        assert np.array_equal(round_result(F16, x), x.astype(np.float16).astype(np.float64))
    y = x[np.abs(x) < 400]
    want = y.astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    assert np.array_equal(round_result(E4M3FN, y), want)
    # f32 below 2^-126 flushes (the value rounded with an unbounded exponent).
    tiny = np.array([2.0**-127, -(2.0**-127), 2.0**-126 * (1 - 2.0**-26)])
    assert np.array_equal(round_result(F32, tiny), [0.0, -0.0, 2.0**-126])
    assert np.signbit(round_result(F32, tiny)[1])
    # The smallest float64 magnitudes round without overflowing the scale.
    assert np.isfinite(rnd(np.array([1e-302]), 24)[0])


def test_every_value_and_the_error_measure():
    # Finite bf16 values without the subnormals (they read as zero): 2 * (254 * 128 + 1).
    assert len(all_values(BF16)) == 2 * (254 * 128 + 1)
    v = all_values(E4M3FN)
    assert len(v) == 254 and v.max() == 448.0
    assert ulp_error(F16, np.array([1.0 + 2.0**-10]), np.array([1.0]))[0] == 1.0
    # An overflow that agrees with the exact value is exact; one ulp beyond the largest is one ulp.
    assert ulp_error(F16, np.array([np.inf]), np.array([1e6]))[0] == 0.0
    assert ulp_error(E4M3FN, np.array([np.nan]), np.array([448.0]))[0] == 1.0


def test_sizing_matches_the_predictions():
    cr = {"log": 0.5, "exp": 0.5, "sqrt": 0.5}
    assert C.extra_bits(C.constant("power", F32, cr), 1.0) == 9
    assert C.extra_bits(C.constant("power", F16, cr), 1.0) == 6
    assert C.extra_bits(C.constant("tanh", F32, cr), 1.0) == 8
    assert C.extra_bits(C.constant("tanh", BF16, cr), 1.0) == 4
    assert C.extra_bits(C.constant("log_plus_one", F32, cr), 1.0) == 3
    assert C.extra_bits(C.constant("rsqrt", F32, cr), 1.0) == 2


def test_fp8_compositions_are_within_the_budget():
    for op, recipe in [("power", "power"), ("tanh", "tanh"), ("log_plus_one", "log_plus_one"), ("rsqrt", "rsqrt")]:
        for cls in ("correctly_rounded", "ulp2"):
            r = row(E4M3FN, op, recipe, cls, jobs=1)
            assert r["method"] == "exhaustive" and r["within_budget"], r


def test_fp8_arithmetic_on_f16_cores_is_correctly_rounded():
    for r in fp8_arith(E4M3FN):
        assert r["not_correctly_rounded"] == 0, r
