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
