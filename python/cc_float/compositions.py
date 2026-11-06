"""Compositions of cores and their sizing (docs/framework/float.md, "Compositions"; the same formulas as
`cc_float::compose`).

Each composition computes in cores of precision `q = p + k` with an unbounded exponent (the core
instances have enough exponent bits) and is rounded once to the format. Special values (NaN, infinities,
zeros, the C11 Annex F cases) are glue, proven bit-exact separately; the models here take ordinary
operands only.
"""

from __future__ import annotations

import math

import numpy as np

from .fpmodel import Fmt, core, rnd

LN2 = math.log(2.0)

#: The default error budget (ulps): NVIDIA's CUDA math library table for f32, 1 ulp for the 16-bit and
#: 8-bit formats.
CUDA_F32 = {
    "exponential": 2.0,
    "log": 1.0,
    "log_plus_one": 1.0,
    "sine": 2.0,
    "cosine": 2.0,
    "tanh": 2.0,
    "rsqrt": 2.0,
    "power": 4.0,
}


def budget(op: str, f: Fmt) -> float:
    return CUDA_F32[op] if f.name == "f32" else 1.0


def tau(f: Fmt) -> float:
    if f.finite_nan:
        return max(math.log2(f.max), -(f.emin - f.m))
    return float(1 << (f.e - 1))


def tanh_cubic_limit(f: Fmt) -> float:
    return (7.5 * 2.0 ** -(f.p + 2)) ** 0.25


def tanh_clamp(f: Fmt) -> float:
    return (f.p + 2) * LN2 / 2.0


def constant(recipe: str, f: Fmt, err: dict[str, float]) -> float:
    if recipe == "power":
        return LN2 * tau(f) * (2.0 * err["log"] + 1.0) + 2.0 * err["exp"]
    if recipe == "tanh":
        return (2.0 * err["exp"] + 2.0) / (2.0 * tanh_cubic_limit(f))
    if recipe == "log_plus_one":
        return 2.0 * err["log"] + 3.0
    if recipe == "rsqrt":
        return 2.0 * err["sqrt"] + 0.5
    if recipe == "wider":
        return err["core"]
    raise ValueError(recipe)


def extra_bits(c: float, t: float) -> int:
    if t <= 0.5:
        raise ValueError("a budget of 1/2 ulp leaves nothing for the cores")
    k = math.ceil(math.log2(c / (t - 0.5)))
    return max(k, 0)


# ---- Recipes (values of the format in, the inner q-bit result out) ------------------------------------


def power(x, y, q, e, modes):
    """exp2(y * log2 x), x > 0: log2 and exp2 cores with errors e['log'], e['exp']; the multiply
    correctly rounded."""
    with np.errstate(all="ignore"):
        lg = core(np.log2, x, q, e["log"], modes[0])
        t = rnd(y * lg, q)
        return core(np.exp2, t, q, e["exp"], modes[1])


def tanh(x, q, e, modes, f: Fmt):
    """Odd symmetry; x - x^3/3 below the cubic limit; (E - 1)/(E + 1), E = e^(2|x|), up to the clamp; 1
    beyond it. Multiplies, adds and divides correctly rounded at q."""
    a = np.abs(x)
    with np.errstate(all="ignore"):
        s = rnd(a * a, q)
        c3 = rnd(s * a, q)
        d = rnd(c3 / 3.0, q)
        cubic = rnd(a - d, q)
        big = core(np.exp, 2.0 * a, q, e["exp"], modes[0])
        num = rnd(big - 1.0, q)
        den = rnd(big + 1.0, q)
        mid = rnd(num / den, q)
    r = np.where(a < tanh_cubic_limit(f), cubic, np.where(a > tanh_clamp(f), 1.0, mid))
    return np.copysign(r, x)


def log_plus_one(x, q, e, modes):
    """Goldberg (1991), Theorem 4: w = fl(1 + x); x if w = 1, else x ln(w) / (w - 1)."""
    with np.errstate(all="ignore"):
        w = rnd(1.0 + x, q)
        lw = core(np.log, w, q, e["log"], modes[0])
        d = rnd(w - 1.0, q)
        ratio = rnd(lw / d, q)
        r = rnd(x * ratio, q)
    return np.where(w == 1.0, x, r)


def rsqrt(x, q, e, modes):
    """sqrt(RN(1/x)): reciprocal correctly rounded, square root with error e['sqrt']."""
    with np.errstate(all="ignore"):
        r1 = rnd(1.0 / x, q)
        return core(np.sqrt, r1, q, e["sqrt"], modes[0])


REFERENCE = {
    "exponential": np.exp,
    "log": np.log,
    "log_plus_one": np.log1p,
    "sine": np.sin,
    "cosine": np.cos,
    "tanh": np.tanh,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
}


def wider(op, x, q, e, modes):
    """One core of a wider format (precision q, error e['core']), rounded once to the format."""
    return core(REFERENCE[op], x, q, e["core"], modes[0])
