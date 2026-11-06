"""Float formats and cores at their contracts, bit-level on float64 carriers (docs/framework/float.md).

A value of every format used here (and every inner precision up to 52 bits) is a float64, so rounding to
`p` significant bits is a scale by a power of two, `numpy.rint` (ties to even) and the inverse scale, all
exact. Reference values are float64 library results (correctly rounded or within one float64 ulp), which
is 2^-29 ulp of f32 and finer for the narrower formats.
"""

from __future__ import annotations

from dataclasses import dataclass

import ml_dtypes
import numpy as np


@dataclass(frozen=True)
class Fmt:
    """A binary format: `e` exponent bits, `m` trailing significand bits."""

    name: str
    e: int
    m: int
    finite_nan: bool = False

    @property
    def p(self) -> int:
        return self.m + 1

    @property
    def bias(self) -> int:
        return (1 << (self.e - 1)) - 1

    @property
    def emin(self) -> int:
        return 1 - self.bias

    @property
    def emax(self) -> int:
        top = (1 << self.e) - 1
        return top - self.bias if self.finite_nan else top - 1 - self.bias

    @property
    def max(self) -> float:
        frac = 2.0 - 2.0 ** (1 - self.m) if self.finite_nan else 2.0 - 2.0 ** (-self.m)
        return frac * 2.0**self.emax

    @property
    def flushes(self) -> bool:
        """Design §6.4: f32 and bf16 read subnormals as zero and flush results below 2^emin."""
        return self.name in ("f32", "bf16")


F32 = Fmt("f32", 8, 23)
BF16 = Fmt("bf16", 8, 7)
F16 = Fmt("f16", 5, 10)
E4M3FN = Fmt("f8e4m3fn", 4, 3, finite_nan=True)
E5M2 = Fmt("f8e5m2", 5, 2)
FORMATS = {f.name: f for f in (F32, BF16, F16, E4M3FN, E5M2)}

_DTYPES = {
    "bf16": (np.uint16, ml_dtypes.bfloat16),
    "f16": (np.uint16, np.float16),
    "f8e4m3fn": (np.uint8, ml_dtypes.float8_e4m3fn),
    "f8e5m2": (np.uint8, ml_dtypes.float8_e5m2),
}


def all_values(f: Fmt) -> np.ndarray:
    """Every finite value of a 16-bit or 8-bit format as float64 (both zeros, subnormals of a flushing
    format excluded: they read as zero)."""
    raw, dt = _DTYPES[f.name]
    bits = np.arange(1 << (8 * np.dtype(raw).itemsize), dtype=np.uint64).astype(raw)
    with np.errstate(invalid="ignore"):
        v = bits.view(dt).astype(np.float64)
    v = v[np.isfinite(v)]
    if f.flushes:
        v = v[(v == 0) | (np.abs(v) >= 2.0**f.emin)]
    return v


def rnd(x: np.ndarray, p: int, emin: int | None = None) -> np.ndarray:
    """Round to nearest, ties to even, to `p` significant bits; below `2^emin` (when given) with the
    fixed spacing `2^(emin - p + 1)` (gradual underflow). Unbounded above. Zeros, infinities and NaN are
    kept."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        _, e = np.frexp(x)
        if emin is not None:
            e = np.maximum(e, emin + 1)
        # ldexp on x itself: 2^(p - e) alone overflows for the smallest float64 values.
        r = np.ldexp(np.rint(np.ldexp(x, p - e)), e - p)
    return np.where(np.isfinite(x), r, x)


def ulp(f: Fmt, v: np.ndarray) -> np.ndarray:
    """The spacing of `f` at the exact value `v` (below `2^emin` the subnormal spacing; design §6.4's
    ulp for flushing formats too)."""
    v = np.abs(np.asarray(v, dtype=np.float64))
    with np.errstate(divide="ignore"):
        _, e = np.frexp(v)
    e = np.maximum(e - 1, f.emin)
    return np.ldexp(1.0, e - f.m)


def round_result(f: Fmt, w: np.ndarray, *, flush: bool = True) -> np.ndarray:
    """An arithmetic result rounded to `f` (design §6.4): to nearest even; a flushing format rounds with
    an unbounded exponent and flushes below `2^emin` keeping the sign (`flush=False` keeps the value
    before the flush); beyond the largest finite value infinity (IEEE) or NaN (E4M3FN)."""
    if f.flushes:
        r = rnd(w, f.p)
        if flush:
            r = np.where(np.abs(r) < 2.0**f.emin, np.copysign(0.0, r), r)
    else:
        r = rnd(w, f.p, f.emin)
    over = np.abs(r) > f.max
    big = np.nan if f.finite_nan else np.inf
    return np.where(over, np.copysign(big, r), r)


def ulp_error(f: Fmt, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """|r - v| in ulps of `f` at `v`. An overflowed result (infinity, or E4M3FN's NaN) counts as the value
    one ulp beyond the largest finite one (the next binade of an unbounded exponent); a result that
    agrees with an overflowing exact value is exact. NaN where `v` is NaN."""
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    beyond = f.max + 2.0 ** (f.emax - f.m)
    rr = np.where(np.isfinite(r), r, np.copysign(beyond, np.where(np.isnan(r), v, r)))
    vv = np.where(np.isfinite(v), v, np.copysign(np.inf, v))
    with np.errstate(invalid="ignore", over="ignore"):
        rounded_v = round_result(f, vv)
        both_over = ~np.isfinite(rounded_v) & ~np.isfinite(r)
        err = np.abs(rr - vv) / ulp(f, vv)
    return np.where(both_over, 0.0, err)


def core(fn, x: np.ndarray, q: int, e: float, mode: str) -> np.ndarray:
    """A core of precision `q` (unbounded exponent) at its contract: `e` ulps (1/2: correctly rounded).
    `cr` rounds the reference; `+` and `-` move it by `e - 1/2` ulps of `q` first, so the result is up to
    `e` ulps away on that side (a core that errs systematically, the worst case for a composition)."""
    with np.errstate(all="ignore"):
        v = fn(x)
        if mode == "cr" or e <= 0.5:
            return rnd(v, q)
        _, ex = np.frexp(v)
        u = np.ldexp(1.0, ex - q)
        d = (e - 0.5) * u * (1.0 if mode == "+" else -1.0)
        return np.where(np.isfinite(v) & (v != 0), rnd(v + d, q), v)
