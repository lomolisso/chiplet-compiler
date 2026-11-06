"""Bit-level models of the IP cores, by the model name a provider gives (`CoreOffer::model` in
`cc_core::adapters::ip`; docs/framework/float.md, "Core models").

`model(name, op, e, m)` returns a function from operand bit patterns (numpy `uint64` arrays, IEEE 754
encodings with `e` exponent and `m` stored significand bits, or a `w`-bit two's complement integer for
`from_int`) to the result's bit patterns:

- `ieee.<op>`: the IEEE operation, correctly rounded to nearest even with gradual underflow, computed
  exactly with rationals (any `e` <= 15, any `m`): `add`, `sub`, `mul`, `div`, `sqrt`, `compare` (the
  quiet less-than bit), `to_int` (rounded per the op's mode, saturating; NaN gives 0), `from_int`, `convert`
  (to the op's format).
- `reference.<op>`: the real function on the exact operand, from numpy's float64 functions, rounded once
  to the format: correctly rounded except within one float64 ulp of a tie, which is inside every
  accuracy contract of the providers (`exp`, `exp2`, `ln`, `log2`, `sin`, `cos`, `pow`, `rsqrt`).
- `flopoco.from_ieee`, `flopoco.to_ieee`: FloPoCo's encoding (two exception bits `00` zero, `01` normal,
  `10` infinity, `11` NaN; sign; exponent; significand) from and to IEEE bits of the same widths;
  subnormals flush to zero of their sign (FloPoCo's `InputIEEE`).

`MODEL_NAMES` lists every name; a test of the Rust providers checks that each name they give is here.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

import numpy as np

MODEL_NAMES = (
    "ieee.add",
    "ieee.sub",
    "ieee.mul",
    "ieee.div",
    "ieee.sqrt",
    "ieee.compare",
    "ieee.to_int",
    "ieee.from_int",
    "ieee.convert",
    "reference.exp",
    "reference.exp2",
    "reference.ln",
    "reference.log2",
    "reference.sin",
    "reference.cos",
    "reference.pow",
    "reference.rsqrt",
    "flopoco.from_ieee",
    "flopoco.to_ieee",
)


class Fmt:
    """An IEEE 754 binary format with `e` exponent and `m` stored significand bits."""

    def __init__(self, e: int, m: int):
        self.e, self.m = e, m
        self.bias = (1 << (e - 1)) - 1
        self.emin = 1 - self.bias
        self.emax = self.bias

    @property
    def bits(self) -> int:
        return 1 + self.e + self.m

    def nan(self) -> int:
        return (((1 << self.e) - 1) << self.m) | (1 << (self.m - 1))

    def inf(self, neg: bool) -> int:
        return (int(neg) << (self.e + self.m)) | (((1 << self.e) - 1) << self.m)

    def decode(self, b: int):
        """A value: a Fraction, or one of 'nan', '+inf', '-inf', '+0', '-0'."""
        s = (b >> (self.e + self.m)) & 1
        ex = (b >> self.m) & ((1 << self.e) - 1)
        fr = b & ((1 << self.m) - 1)
        if ex == (1 << self.e) - 1:
            return "nan" if fr else ("-inf" if s else "+inf")
        if ex == 0 and fr == 0:
            return "-0" if s else "+0"
        mag = Fraction(fr, 1 << self.m) * Fraction(2) ** (self.emin) if ex == 0 else (
            1 + Fraction(fr, 1 << self.m)
        ) * Fraction(2) ** (ex - self.bias)
        return -mag if s else mag

    def encode(self, v) -> int:
        """Round a Fraction (or a special) to nearest even, gradual underflow, overflow to infinity."""
        if isinstance(v, str):
            if v == "nan":
                return self.nan()
            neg = v.startswith("-")
            return self.inf(neg) if v.endswith("inf") else int(neg) << (self.e + self.m)
        if v == 0:
            return 0
        neg = v < 0
        a = -v if neg else v
        # Exponent k with 2^k <= a < 2^(k+1).
        k = a.numerator.bit_length() - a.denominator.bit_length()
        if Fraction(2) ** k > a:
            k -= 1
        k = max(k, self.emin)
        scaled = a / Fraction(2) ** (k - self.m)
        n = scaled.numerator // scaled.denominator
        rem = scaled - n
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2 == 1):
            n += 1
        if n >= 1 << (self.m + 1):
            n >>= 1
            k += 1
        sign = int(neg) << (self.e + self.m)
        if n < 1 << self.m:
            return sign | n  # subnormal (or zero after rounding)
        if k > self.emax:
            return self.inf(neg)
        return sign | ((k + self.bias) << self.m) | (n & ((1 << self.m) - 1))


def _value(v):
    """A special as a float for comparisons."""
    return {"+inf": float("inf"), "-inf": float("-inf"), "+0": 0.0, "-0": -0.0}.get(v, v)


def _parts(f: Fmt, b: int):
    """(sign, kind, magnitude): kind is 'nan', 'inf', 'zero' or 'fin'."""
    v = f.decode(b)
    s = (b >> (f.e + f.m)) & 1
    if v == "nan":
        return s, "nan", Fraction(0)
    if isinstance(v, str):
        return s, ("inf" if v.endswith("inf") else "zero"), Fraction(0)
    return s, "fin", abs(v)


def _signed(f: Fmt, s: int, kind: str, mag: Fraction = Fraction(0)) -> int:
    sign = s << (f.e + f.m)
    if kind == "inf":
        return f.inf(bool(s))
    if kind == "zero":
        return sign
    return f.encode(-mag if s else mag)


def _sqrt(f: Fmt, x: Fraction) -> Fraction:
    """A value strictly between the same rounding boundaries as sqrt(x) (exact when it is exact)."""
    k = 2 * (f.m + 8) - f.emin
    num = x * Fraction(4) ** k
    r = isqrt(num.numerator // num.denominator)
    v = Fraction(r, 1 << k)
    return v if r * r == num else v + Fraction(1, 1 << (k + 1))


def _arith(op: str, f: Fmt):
    def one(*bs):
        ps = [_parts(f, int(b)) for b in bs]
        if any(k == "nan" for _, k, _ in ps):
            return f.nan()
        if op == "sqrt":
            ((s, k, x),) = ps
            if k == "zero":
                return _signed(f, s, "zero")
            if s:
                return f.nan()
            return _signed(f, 0, "inf") if k == "inf" else f.encode(_sqrt(f, x))
        (sx, kx, x), (sy, ky, y) = ps
        if op in ("add", "sub"):
            if op == "sub":
                sy ^= 1
            if kx == "inf" or ky == "inf":
                if kx == "inf" and ky == "inf" and sx != sy:
                    return f.nan()
                return _signed(f, sx if kx == "inf" else sy, "inf")
            v = (-x if sx else x) + (-y if sy else y)
            if v == 0:
                # Two zeros of one sign keep it; any other exact zero sum is +0 (round to nearest).
                both = kx == "zero" and ky == "zero" and sx == sy
                return _signed(f, sx if both else 0, "zero")
            return f.encode(v)
        s = sx ^ sy
        if op == "mul":
            if (kx == "inf" and ky == "zero") or (kx == "zero" and ky == "inf"):
                return f.nan()
            if "inf" in (kx, ky):
                return _signed(f, s, "inf")
            if "zero" in (kx, ky):
                return _signed(f, s, "zero")
            return _signed(f, s, "fin", x * y)
        # div
        if (kx, ky) in (("zero", "zero"), ("inf", "inf")):
            return f.nan()
        if kx == "inf" or ky == "zero":
            return _signed(f, s, "inf")
        if kx == "zero" or ky == "inf":
            return _signed(f, s, "zero")
        return _signed(f, s, "fin", x / y)

    return one


def _compare(f: Fmt):
    def one(a, b):
        x, y = f.decode(int(a)), f.decode(int(b))
        if "nan" in (x, y):
            return 0
        return int(_value(x) < _value(y))

    return one


def _parse_int(t: str) -> tuple[bool, int]:
    return t[0] == "s", int(t[1:])


def _to_int(op: str, f: Fmt):
    _, rnd, t = op.split(".")
    signed, w = _parse_int(t)
    lo, hi = (-(1 << (w - 1)), (1 << (w - 1)) - 1) if signed else (0, (1 << w) - 1)

    def one(a):
        x = f.decode(int(a))
        if x == "nan":
            return 0
        x = _value(x)
        if isinstance(x, float):
            n = hi if x > 0 else (lo if x < 0 else 0)
        else:
            fl = x.numerator // x.denominator
            frac = x - fl
            n = {
                "floor": fl,
                "ceil": fl + (frac > 0),
                "rtz": fl + (frac > 0 and x < 0),
                "rne": fl + (frac > Fraction(1, 2) or (frac == Fraction(1, 2) and fl % 2 == 1)),
            }[rnd]
            n = min(max(n, lo), hi)
        return n & ((1 << w) - 1)

    return one


def _from_int(op: str, f: Fmt):
    signed, w = _parse_int(op.split(".")[1])

    def one(a):
        n = int(a) & ((1 << w) - 1)
        if signed and n >> (w - 1):
            n -= 1 << w
        return f.encode(Fraction(n)) if n else 0

    return one


def _convert(op: str, f: Fmt):
    tail = op.split(".")[1]
    e, m = tail[1:].split("m")
    g = Fmt(int(e), int(m))

    def one(a):
        x = f.decode(int(a))
        return g.encode(x)

    return one


_FUNCS = {
    "exp": np.exp,
    "exp2": np.exp2,
    "ln": np.log,
    "log2": np.log2,
    "sin": np.sin,
    "cos": np.cos,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
}


def _reference(fn: str, f: Fmt):
    def one(*bs):
        xs = [_value(f.decode(int(b))) for b in bs]
        if any(x == "nan" for x in xs):
            return f.nan()
        xs = [np.float64(float(x)) for x in xs]
        with np.errstate(all="ignore"):
            r = np.power(xs[0], xs[1]) if fn == "pow" else _FUNCS[fn](xs[0])
        if np.isnan(r):
            return f.nan()
        if np.isinf(r):
            return f.inf(r < 0)
        if r == 0:
            return f.encode("-0" if np.signbit(r) else "+0")
        return f.encode(Fraction(float(r)))

    return one


def _flopoco(direction: str, f: Fmt):
    w = f.e + f.m

    def from_ieee(a):
        b = int(a)
        s = (b >> w) & 1
        ex = (b >> f.m) & ((1 << f.e) - 1)
        fr = b & ((1 << f.m) - 1)
        if ex == (1 << f.e) - 1:
            exn = 0b11 if fr else 0b10
            return (exn << (w + 1)) | (s << w)
        if ex == 0:
            return s << w  # zero, subnormals flushed
        return (0b01 << (w + 1)) | (s << w) | (ex << f.m) | fr

    def to_ieee(a):
        b = int(a)
        exn = (b >> (w + 1)) & 0b11
        s = (b >> w) & 1
        body = b & ((1 << w) - 1)
        if exn == 0b00:
            return s << w
        if exn == 0b10:
            return f.inf(bool(s))
        if exn == 0b11:
            return f.nan()
        return (s << w) | body

    return from_ieee if direction == "from_ieee" else to_ieee


def model(name: str, op: str, e: int, m: int):
    """The model `name` of a core for the core op `op` (`CoreOp` syntax) at operand widths (e, m); a
    function of uint64 arrays."""
    if name not in MODEL_NAMES:
        raise KeyError(f"no model {name}")
    family, what = name.split(".")
    f = Fmt(e, m)
    if family == "ieee":
        one = {
            "compare": lambda: _compare(f),
            "to_int": lambda: _to_int(op, f),
            "from_int": lambda: _from_int(op, f),
            "convert": lambda: _convert(op, f),
        }.get(what, lambda: _arith(what, f))()
    elif family == "reference":
        one = _reference(what, f)
    else:
        one = _flopoco(what, f)

    def run(*xs):
        arrs = [np.atleast_1d(np.asarray(x, dtype=np.uint64)) for x in xs]
        return np.array([one(*vals) for vals in zip(*arrs, strict=True)], dtype=np.uint64)

    return run
