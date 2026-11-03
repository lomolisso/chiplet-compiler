"""Test vectors: every input combination when the operands have at most `exhaustive_bits` bits in total,
otherwise the cross product of each type's edge cases plus seeded random patterns (half of them, for binary
float ops, operand pairs with nearby exponents, where cancellation and ties happen)."""

from __future__ import annotations

import hashlib
import itertools

import numpy as np

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
