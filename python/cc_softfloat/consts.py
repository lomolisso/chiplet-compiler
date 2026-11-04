"""Constants of the approximate ops as fixed-point integers, computed with the standard library's `decimal`
at 120 significant digits (far beyond the widest fixed-point format used, about 2^-90).

A fixed-point integer `n` with `frac` fraction bits stands for `n / 2^frac`. `floor` and `nearest` say how
the exact value was rounded to it; the error bounds in `cc_softfloat.approx` use these roundings.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction

DIGITS = 120


def _dec(fn):
    with localcontext() as c:
        c.prec = DIGITS
        return fn()


def ln2() -> Decimal:
    return _dec(lambda: Decimal(2).ln())


def to_fixed(x: Decimal | Fraction, frac: int, mode: str = "nearest") -> int:
    """`x * 2^frac` rounded to an integer (`floor` or `nearest`, ties to even)."""
    if isinstance(x, Fraction):
        v = x * (1 << frac)
        q, r = divmod(v.numerator, v.denominator)
        if mode == "floor":
            return q
        twice = 2 * r
        if twice > v.denominator or (twice == v.denominator and q & 1):
            q += 1
        return q

    def go() -> int:
        v = x * (Decimal(2) ** frac)
        return int(v.to_integral_value(rounding=ROUND_FLOOR if mode == "floor" else ROUND_HALF_EVEN))

    return _dec(go)


def ln2_fixed(frac: int, mode: str = "nearest") -> int:
    return to_fixed(ln2(), frac, mode)


def inv_ln2_fixed(frac: int, mode: str = "floor") -> int:
    """`2^frac / ln 2`."""
    return to_fixed(_dec(lambda: 1 / ln2()), frac, mode)


def exp2_table(t: int, frac: int) -> list[int]:
    """`2^(j / 2^t)` for j = 0 .. 2^t - 1, rounded to nearest."""
    return [to_fixed(_dec(lambda j=j: (Decimal(j) / (1 << t) * ln2()).exp()), frac) for j in range(1 << t)]


def log_tables(t: int, fc: int, frac: int) -> tuple[list[int], list[int]]:
    """The reduction table of `log` (docs/passes/softfloat.md §4): for i = 0 .. 2^t, `c_i` ~ 1 / (1 + i/2^t)
    rounded to nearest at `fc` fraction bits (exactly 1 for i = 0 and 1/2 for i = 2^t), and `L_i = -ln(c_i)`
    of that rounded `c_i`, rounded to nearest at `frac` fraction bits."""
    cs, ls = [], []
    for i in range(1 << t | 1):
        c = to_fixed(Fraction(1 << t, (1 << t) + i), fc)
        cs.append(c)
        ls.append(to_fixed(_dec(lambda c=c: -(Decimal(c) / (Decimal(2) ** fc)).ln()), frac))
    return cs, ls


# ------------------------------------------------------------------------------------ pi, 2/pi (integers)

_GUARD = 64


def _atan_inv(x: int, bits: int) -> int:
    """`atan(1/x) * 2^bits`, each term truncated (error below the number of terms)."""
    total, k, power = 0, 0, (1 << bits) // x
    x2 = x * x
    while power:
        term = power // (2 * k + 1)
        total += -term if k & 1 else term
        power //= x2
        k += 1
    return total


def pi_fixed(frac: int) -> int:
    """`floor(pi * 2^frac)`, from Machin's formula `pi = 16 atan(1/5) - 4 atan(1/239)` with 64 guard bits
    (asserted far enough from an integer that the guard bits' error cannot change the floor)."""
    b = frac + _GUARD
    v = 16 * _atan_inv(5, b) - 4 * _atan_inv(239, b)
    low = v & ((1 << _GUARD) - 1)
    assert 1 << 16 < low < (1 << _GUARD) - (1 << 16), "pi: too close to an integer"
    return v >> _GUARD


def two_over_pi_fixed(frac: int) -> int:
    """`floor(2/pi * 2^frac)`: the first `frac` bits of 2/pi after the binary point."""
    pg = pi_fixed(frac + _GUARD)
    q = (1 << (2 * frac + 2 * _GUARD + 1)) // pg
    # pg is pi * 2^(frac+64) to within 1: q is within 2^-62 of the exact value, scaled by 2^64
    assert 1 << 3 < (q & ((1 << _GUARD) - 1)) < (1 << _GUARD) - (1 << 3), "2/pi: too close to an integer"
    return q >> _GUARD


def trig_worst_case(p: int, ex_lo: int, ex_hi: int) -> tuple[float, int, int]:
    """The smallest distance `||x * 2/pi||` (to the nearest integer) over the floats `x = M * 2^(ex-p+1)`,
    `M` in [2^(p-1), 2^p), `ex` in [ex_lo, ex_hi]: the argument reduction of sine and cosine must keep the
    fraction `x * 2/pi - q` to this many leading zeros plus the result's precision (Muller, *Elementary
    Functions*, ch. 11). Every such x is enumerated with 64 fraction bits of `2^E * 2/pi` (error below
    `M * 2^-64 < 2^-39`). Returns (a lower bound of the distance, the worst M, its ex)."""
    import numpy as np

    n = ex_hi + 64 + 8
    tpi = two_over_pi_fixed(n)
    m = np.arange(1 << (p - 1), 1 << p, dtype=np.uint64)
    best, arg = 2.0, (0, 0)
    for ex in range(ex_lo, ex_hi + 1):
        e = ex - (p - 1)
        # the 64 fraction bits of 2^e * 2/pi: bits e+1 .. e+64 of 2/pi
        a = (tpi >> (n - e - 64)) & ((1 << 64) - 1) if n - e - 64 >= 0 else (tpi << (e + 64 - n)) & ((1 << 64) - 1)
        hi, lo = np.uint64(a >> 32), np.uint64(a & 0xFFFFFFFF)
        with np.errstate(over="ignore"):
            fr = (m * hi << np.uint64(32)) + m * lo
            d = np.minimum(fr, np.uint64(0) - fr)
        i = int(np.argmin(d))
        v = float(d[i]) * 2.0**-64
        if v < best:
            best, arg = v, (int(m[i]), ex)
    return best - 2.0**-39, arg[0], arg[1]
