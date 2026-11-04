"""Test vectors: every input combination when the operands have at most `exhaustive_bits` bits in total,
otherwise the cross product of each type's edge cases plus seeded random patterns (half of them, for binary
float ops, operand pairs with nearby exponents, where cancellation and ties happen). An approximate op also
gets the inputs within 3 ulps of each boundary of its design, and half of its random inputs drawn from the
range where its result is neither a constant nor an overflow (`_domain`)."""

from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np

from . import approx, fp
from .catalog import Prim
from .formats import Float, Int


def edges(t: Float | Int) -> list[int]:
    if isinstance(t, Int):
        if t.w == 1:
            return [0, 1]
        mask = (1 << t.w) - 1
        vals = {0, 1, 2, 3, t.lo, t.lo + 1, t.hi, t.hi - 1, -1, -2}
        for k in range(t.w):
            vals |= {(1 << k) - 1, 1 << k, (1 << k) + 1}
        return sorted({v & mask for v in vals})
    sign = 1 << (t.bits - 1)
    one, m = t.one, t.m
    vals = {
        0,
        1,
        2,
        t.man_mask,
        t.man_mask - 1,
        1 << m,
        (1 << m) | 1,
        one,
        one | 1,
        one - 1,
        one | t.man_mask,
        one - (1 << m),
        one + (1 << m),
        (one - (1 << m)) | (1 << (m - 1)) if m > 0 else one,
        t.max_finite,
        t.max_finite - 1,
        t.nan,
    }
    if not t.finite_nan:
        vals |= {t.inf, t.inf | 1, t.inf | (1 << (m - 1)) | 1}
    return sorted(vals | {v | sign for v in vals})


def _boundaries(op: str, f: Float) -> list[float]:
    """Inputs where an approximate op's design changes path or its result changes regime."""
    dz = approx.design(op, f)
    prm = dz.params
    p = f.m + 1
    top = fp.max_value(f)
    tiny = 2.0 ** (1 - f.bias - f.m)
    t = prm.get("t", 4)
    match op:
        case "exponential":
            ln2 = math.log(2)
            return [math.log(top), math.log(tiny), math.log(tiny / 2), 2.0 ** -(p + 1), 2.0 ** prm.get("int_bits", 7)] + [
                k * ln2 / (1 << (t + 1)) for k in range(1, 5)
            ]
        case "log":
            return [1.0, 1 - 2.0 ** -(t + 2), 1 + 2.0 ** -(t + 1), 2 - 2.0 ** -(t + 1), 0.5, 2.0, tiny, top]
        case "log_plus_one":
            return [-1.0, 2.0 ** -(t + 2), -(2.0 ** -(t + 2)), 2.0 ** -(t + 1), 2.0 ** (p + approx.G + 3), -0.5, tiny]
        case "tanh":
            return [2.0 ** -prm.get("small", p // 2), prm.get("sat", 9.0), 1.0, 0.5]
        case "rsqrt":
            return [1.0, 2.0, 4.0, 0.25, 0.5, top, tiny]
    raise ValueError(op)


def _domain(op: str, f: Float, rng: np.random.Generator, n: int) -> np.ndarray:
    """`n` float64 inputs where an approximate op does real work."""
    p = f.m + 1
    emin, emax = 1 - f.bias - f.m, f.bias + 1
    sign = rng.choice([-1.0, 1.0], n)
    u = rng.random(n)

    def logu(lo: float, hi: float) -> np.ndarray:
        return np.exp2(lo + (hi - lo) * rng.random(n))

    match op:
        case "exponential":
            hi = math.log2(math.log(fp.max_value(f)) + 2)
            return sign * logu(-(p + 2), hi)
        case "log" | "rsqrt":
            near_one = 1 + sign * logu(-(p + 1), -1)
            return np.where(u < 0.5, logu(emin, emax), near_one)
        case "log_plus_one":
            pos = logu(-(p + 3), 40)
            neg = -logu(-(p + 3), 0) * 0.999
            near_m1 = -1 + logu(-p, -1)
            return np.where(u < 0.4, pos, np.where(u < 0.8, neg, near_m1))
        case "tanh":
            return sign * logu(-(p // 2 + 2), math.log2(approx.design(op, f).params.get("sat", 9.0)) + 1)
    raise ValueError(op)


def _approx_extra(p: Prim, rng: np.random.Generator, random_n: int) -> np.ndarray:
    f = p.ret
    assert isinstance(f, Float)
    base = fp.round_f64(f, np.array(_boundaries(p.op, f)))
    mag = base & np.uint64(f.mag_mask)
    near = [mag + np.uint64(k) for k in range(4)] + [mag - np.uint64(k) for k in range(1, 4) if True]
    vals = np.concatenate(near) & np.uint64(f.mag_mask)
    vals = np.concatenate([vals, vals | np.uint64(1 << (f.bits - 1))])
    dom = fp.round_f64(f, _domain(p.op, f, rng, random_n // 2))
    return np.concatenate([vals, dom])


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def generate(p: Prim, exhaustive_bits: int, random_n: int) -> list[np.ndarray]:
    """One uint64 array per operand, all the same length."""
    if p.input_bits <= exhaustive_bits:
        grids = np.meshgrid(*[np.arange(1 << t.bits, dtype=np.uint64) for t in p.params], indexing="ij")
        return [g.ravel() for g in grids]
    cols: list[list[int]] = [[] for _ in p.params]
    for combo in itertools.product(*[edges(t) for t in p.params]):
        for i, v in enumerate(combo):
            cols[i].append(v)
    rng = np.random.default_rng(_seed(p.key))
    if p.approximate:
        extra = _approx_extra(p, rng, random_n)
        rand = [np.concatenate([extra, rng.integers(0, 1 << p.params[0].bits, random_n - random_n // 2, dtype=np.uint64)])]
        return [np.concatenate([np.array(c, dtype=np.uint64), r]) for c, r in zip(cols, rand)]
    rand = [rng.integers(0, 1 << t.bits, random_n, dtype=np.uint64) for t in p.params]
    fl = [i for i, t in enumerate(p.params) if isinstance(t, Float)]
    if len(fl) >= 2:
        i, j = fl[0], fl[1]
        t = p.params[i]
        half = random_n // 2
        assert isinstance(t, Float)
        near = rng.integers(0, 1 << (t.m + 2), half, dtype=np.uint64)
        flip = rng.integers(0, 2, half, dtype=np.uint64) << np.uint64(t.bits - 1)
        rand[j][:half] = (rand[i][:half] ^ near ^ flip) & np.uint64((1 << t.bits) - 1)
    return [np.concatenate([np.array(c, dtype=np.uint64), r]) for c, r in zip(cols, rand)]
