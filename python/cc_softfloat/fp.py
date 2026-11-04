"""Float64 helpers on the library's formats: decoding, exact rounding, ulps, and the accuracy check of the
approximate ops. Independent of the generated circuits (the reference and the ROM tables use them).

The approximate ops are compared with NumPy's float64 functions, whose own error is a few float64 ulps; the
check allows for it with a relative margin of `REF_REL_ERR` (2^-48, sixteen float64 ulps), so the claim it
checks is "error < limit + 2^-48 relative" (docs/passes/softfloat.md §4).
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

from .formats import Float

REF_REL_ERR = 2.0**-48

DTYPES = {
    "f32": np.float32,
    "f16": np.float16,
    "bf16": ml_dtypes.bfloat16,
    "f8e4m3fn": ml_dtypes.float8_e4m3fn,
    "f8e5m2": ml_dtypes.float8_e5m2,
}


def udtype(bits: int) -> type:
    return np.uint8 if bits <= 8 else np.uint16 if bits <= 16 else np.uint32


def to_f64(f: Float, bits: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):  # signaling NaN patterns
        return np.asarray(bits).astype(udtype(f.bits)).view(DTYPES[f.name]).astype(np.float64)


def max_value(f: Float) -> float:
    return float(to_f64(f, np.array([f.max_finite], dtype=np.uint64))[0])


def ulp(f: Float, v: np.ndarray) -> np.ndarray:
    """The format's ulp at `v` (Goldberg's definition: the spacing of the binade holding |v|, the subnormal
    spacing below the normal range)."""
    a = np.abs(np.asarray(v, dtype=np.float64))
    _, ex = np.frexp(np.where(a > 0, a, 1.0))
    e = np.maximum(ex - 1, 1 - f.bias)
    return np.ldexp(1.0, e - f.m)


def round_f64(f: Float, v: np.ndarray) -> np.ndarray:
    """Float64 values rounded once to `f` (to nearest, ties to even, gradual underflow; overflow to infinity,
    or to the canonical NaN in finite-NaN formats, which also stands for infinite values), as bit patterns. NumPy's casts to bfloat16 and float8 go through
    float32 and would round twice; here the rounding is exact: `v / ulp` is exact in float64 (a power-of-two
    scaling), `rint` rounds it to an integer, and the product is representable in `f`."""
    v = np.asarray(v, dtype=np.float64)
    with np.errstate(all="ignore"):
        u = ulp(f, v)
        r = np.rint(v / u) * u
        r = np.where(np.isfinite(v), r, v)
        big = np.abs(r) > max_value(f)
        out = r.astype(DTYPES[f.name]).view(udtype(f.bits)).astype(np.uint64)
    sign = np.where(np.signbit(v), np.uint64(1 << (f.bits - 1)), np.uint64(0))
    over = np.uint64(f.nan) if f.finite_nan else sign | np.uint64(f.inf)
    out = np.where(big | np.isinf(v), over, out)
    return np.where(np.isnan(v), np.uint64(f.nan), out).astype(np.uint64)


def check_ulps(f: Float, got: np.ndarray, v: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Is each result `got` (bit patterns) within `limit` ulps of the exact value, known as the float64 `v`?

    - NaN values need a NaN; infinite values need the rounded infinity (NaN in finite-NaN formats); an exact
      zero needs that zero with its sign.
    - Beyond the largest finite value, an overflow result (infinity, or NaN in finite-NaN formats) is
      accepted from max + ulp/2 on, and the largest finite value within the limit.
    - Otherwise the error |got - v| / ulp(v) must be at most `limit` plus the reference's own error, and a
      zero result must carry the sign of `v`.

    Returns the per-vector verdict and the error in ulps (NaN where it does not apply)."""
    v = np.asarray(v, dtype=np.float64)
    got = np.asarray(got).astype(np.uint64)
    with np.errstate(all="ignore"):
        g = to_f64(f, got)
        u = ulp(f, v)
        err = np.abs(g - v) / u
        margin = REF_REL_ERR * np.abs(v) / u
        fin = np.isfinite(v)
        same_sign = np.signbit(g) == np.signbit(v)
        want = round_f64(f, v)
        exact_special = (want == got) | (np.isnan(g) & np.isnan(to_f64(f, want)))
        ok = np.isnan(v) & np.isnan(g)
        ok |= np.isinf(v) & exact_special
        ok |= fin & (v == 0) & (g == 0) & same_sign
        top = max_value(f)
        overflow_ok = np.abs(v) >= (top + float(ulp(f, np.array([top]))[0]) / 2) * (1 - REF_REL_ERR)
        g_over = np.isinf(g) | (np.isnan(g) if f.finite_nan else False)
        ok |= fin & (v != 0) & g_over & overflow_ok & (same_sign | np.isnan(g))
        close = np.isfinite(g) & (err <= limit + margin) & ((g != 0) | same_sign)
        ok |= fin & (v != 0) & close
    return ok, np.where(fin & np.isfinite(g), err, np.nan)


__all__ = ["DTYPES", "REF_REL_ERR", "check_ulps", "max_value", "round_f64", "to_f64", "udtype", "ulp"]
