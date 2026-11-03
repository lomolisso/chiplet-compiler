"""Generators of the exactly specified float ops: one flat Allo function per (op, types).

The semantics are those of the datapath proofs (`cc_smt::ops`, docs/framework/smt.md): IEEE 754 with round to
nearest, ties to even, gradual underflow, StableHLO's op definitions, and XLA's choices where the spec leaves
one (float to integer truncates, saturates and maps NaN to 0). A NaN result is the format's canonical NaN
(`Float.nan`); the proofs compare NaN as one class.
"""

from __future__ import annotations

from .emit import HEADER, Src, clog2, decode, normalize, round_pack, shift_right_sticky, xw
from .formats import BOOL, Float, Int

GUARD = 3


def _ret(s: Src, out: str) -> str:
    s(f"return {out}")
    return HEADER + s.text()


def _put_inf(s: Src, f: Float, sign: str, out: str) -> None:
    """`out` = infinity with the sign bit `sign` (NaN in a finite-NaN format)."""
    t = s.tmp("isg")
    s(f"{t}: UInt({f.bits}) = {sign}")
    s(f"{out} = ({t} << {f.bits - 1}) | {f.inf}")


def add(name: str, f: Float, subtract: bool) -> str:
    """Single-path adder: order by magnitude, align with sticky, add or subtract, normalize (stopping at the
    subnormal exponent), round once."""
    w, m, xwid = f.bits, f.m, xw(f)
    sw = m + 1 + GUARD
    s = Src(name, [("a", w), ("b", w)], w, f"{'subtract' if subtract else 'add'} ({f.name})")
    decode(s, f, "a", "a", xwid)
    decode(s, f, "b", "b", xwid)
    if subtract:
        s("b_s = b_s ^ 1")
    s(f"swap: UInt(1) = 0")
    with s.if_(f"(b & {f.mag_mask}) > (a & {f.mag_mask})"):
        s("swap = 1")
    for v, t in (("s", "UInt(1)"), ("m", f"UInt({m + 1})"), ("x", f"Int({xwid})")):
        s(f"x_{v}: {t} = a_{v}")
        s(f"y_{v}: {t} = b_{v}")
    with s.if_("swap == 1"):
        for v in "smx":
            s(f"x_{v} = b_{v}")
            s(f"y_{v} = a_{v}")
    s(f"xa: UInt({sw + 1}) = x_m")
    s(f"xa = xa << {GUARD}")
    s(f"ya: UInt({sw + 1}) = y_m")
    s(f"ya = ya << {GUARD}")
    s(f"d: Int({xwid}) = x_x - y_x")
    shift_right_sticky(s, "ya", sw + 1, "d", "yb")
    s(f"r: UInt({sw + 1}) = 0")
    with s.if_("x_s == y_s"):
        s("r = xa + yb")
    with s.else_():
        s("r = xa - yb")
    s(f"e: Int({xwid}) = x_x")
    s(f"p: UInt({sw}) = 0")
    with s.if_(f"(r >> {sw}) == 1"):
        s("p = (r >> 1) | (r & 1)")
        s("e = e + 1")
    with s.else_():
        s(f"rl: UInt({sw}) = r")
        normalize(s, "rl", sw, "rn", "lzr", cap="e - 1")
        s("p = rn")
        s("e = e - lzr")
    s(f"res: UInt({w}) = 0")
    round_pack(s, f, "x_s", "e", "p", sw, "res", xwid, denorm=False)
    with s.if_("r == 0"):
        s(f"zs: UInt({w}) = x_s & y_s")
        s(f"res = zs << {w - 1}")
    with s.if_("a_nan == 1 or b_nan == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_inf == 1 and b_inf == 1"):
        with s.if_("a_s != b_s"):
            s(f"res = {f.nan}")
        with s.else_():
            _put_inf(s, f, "a_s", "res")
    with s.elif_("a_inf == 1"):
        _put_inf(s, f, "a_s", "res")
    with s.elif_("b_inf == 1"):
        _put_inf(s, f, "b_s", "res")
    return _ret(s, "res")


def multiply(name: str, f: Float) -> str:
    """Exact significand product, normalized, rounded once (denormalized first below the normal range)."""
    w, m, xwid = f.bits, f.m, xw(f)
    pw = 2 * m + 2
    s = Src(name, [("a", w), ("b", w)], w, f"multiply ({f.name})")
    decode(s, f, "a", "a", xwid)
    decode(s, f, "b", "b", xwid)
    s("sign: UInt(1) = a_s ^ b_s")
    s(f"am: UInt({pw}) = a_m")
    s(f"bm: UInt({pw}) = b_m")
    s(f"prod: UInt({pw}) = am * bm")
    normalize(s, "prod", pw, "pn", "lzp")
    s(f"e: Int({xwid}) = a_x + b_x - {f.bias - 1} - lzp")
    s(f"res: UInt({w}) = 0")
    round_pack(s, f, "sign", "e", "pn", pw, "res", xwid)
    with s.if_("a_zero == 1 or b_zero == 1"):
        s(f"zs: UInt({w}) = sign")
        s(f"res = zs << {w - 1}")
    with s.if_("a_nan == 1 or b_nan == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_inf == 1 or b_inf == 1"):
        with s.if_("a_zero == 1 or b_zero == 1"):
            s(f"res = {f.nan}")
        with s.else_():
            _put_inf(s, f, "sign", "res")
    return _ret(s, "res")


def divide(name: str, f: Float) -> str:
    """Restoring radix-2 division of the normalized significands: m + 3 quotient bits and the remainder as
    sticky, rounded once."""
    w, m, xwid = f.bits, f.m, xw(f)
    nq = m + 3
    s = Src(name, [("a", w), ("b", w)], w, f"divide ({f.name})")
    decode(s, f, "a", "a", xwid)
    decode(s, f, "b", "b", xwid)
    s("sign: UInt(1) = a_s ^ b_s")
    normalize(s, "a_m", m + 1, "an", "lza")
    normalize(s, "b_m", m + 1, "bn", "lzb")
    s(f"r: UInt({m + 3}) = an")
    s(f"bd: UInt({m + 3}) = bn")
    s(f"q: UInt({nq}) = 0")
    with s.for_("i", nq):
        s("q = q << 1")
        with s.if_("r >= bd"):
            s("r = r - bd")
            s("q = q | 1")
        s("r = r << 1")
    s(f"p: UInt({nq + 1}) = q")
    s("p = p << 1")
    with s.if_("r != 0"):
        s("p = p | 1")
    s(f"e: Int({xwid}) = a_x - lza - b_x + lzb + {f.bias}")
    with s.if_(f"(p >> {nq}) == 0"):
        s("p = p << 1")
        s("e = e - 1")
    s(f"res: UInt({w}) = 0")
    round_pack(s, f, "sign", "e", "p", nq + 1, "res", xwid)
    s(f"zs: UInt({w}) = sign")
    with s.if_("a_nan == 1 or b_nan == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_inf == 1 and b_inf == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_zero == 1 and b_zero == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_inf == 1 or b_zero == 1"):
        _put_inf(s, f, "sign", "res")
    with s.elif_("a_zero == 1 or b_inf == 1"):
        s(f"res = zs << {w - 1}")
    return _ret(s, "res")


def sqrt(name: str, f: Float) -> str:
    """Digit-by-digit square root of the normalized significand (exponent made even): m + 3 root bits and
    the remainder as sticky, rounded once; the result is never subnormal and never overflows."""
    w, m, xwid = f.bits, f.m, xw(f)
    nq = m + 3
    s = Src(name, [("a", w)], w, f"sqrt ({f.name})")
    decode(s, f, "a", "a", xwid)
    normalize(s, "a_m", m + 1, "an", "lza")
    s(f"ue: Int({xwid}) = a_x - lza - {f.bias}")
    s(f"v: UInt({2 * nq}) = an")
    with s.if_("(ue & 1) == 1"):
        s("v = v << 1")
        s("ue = ue - 1")
    s(f"v = v << {m + 4}")
    s(f"rem: UInt({nq + 2}) = 0")
    s(f"root: UInt({nq}) = 0")
    s(f"trial: UInt({nq + 2}) = 0")
    s(f"rr: UInt({nq + 2}) = 0")
    with s.for_("i", nq):
        s(f"rem = (rem << 2) | ((v >> ({2 * (nq - 1)} - 2 * i)) & 3)")
        s("rr = root")
        s("trial = (rr << 2) | 1")
        s("root = root << 1")
        with s.if_("rem >= trial"):
            s("rem = rem - trial")
            s("root = root | 1")
    s(f"p: UInt({nq + 1}) = root")
    s("p = p << 1")
    with s.if_("rem != 0"):
        s("p = p | 1")
    s(f"e: Int({xwid}) = (ue >> 1) + {f.bias}")
    s(f"res: UInt({w}) = 0")
    round_pack(s, f, "a_s", "e", "p", nq + 1, "res", xwid, denorm=False)
    with s.if_("a_nan == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_zero == 1"):
        s("res = a")
    with s.elif_("a_s == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_inf == 1"):
        s("res = a")
    return _ret(s, "res")


def remainder(name: str, f: Float) -> str:
    """C `fmod` (StableHLO `remainder`), exact: the divisor's significand is normalized, then one
    compare-subtract per binade of the exponent difference; the result keeps the dividend's sign."""
    w, m, xwid = f.bits, f.m, xw(f)
    dmax = f.emax_field + m - 2
    s = Src(name, [("a", w), ("b", w)], w, f"remainder ({f.name})")
    decode(s, f, "a", "a", xwid)
    decode(s, f, "b", "b", xwid)
    normalize(s, "b_m", m + 1, "bn", "lzb")
    s(f"by: Int({xwid}) = b_x - lzb")
    s(f"d: Int({xwid}) = a_x - by")
    s(f"r: UInt({m + 3}) = a_m")
    s(f"bd: UInt({m + 3}) = bn")
    with s.for_("i", dmax + 1):
        with s.if_("i <= d"):
            with s.if_("r >= bd"):
                s("r = r - bd")
            with s.if_("i < d"):
                s("r = r << 1")
    s(f"rl: UInt({m + 1}) = r")
    normalize(s, "rl", m + 1, "rn", "lzr")
    s(f"p: UInt({m + 3}) = rn")
    s("p = p << 2")
    s(f"e: Int({xwid}) = by - lzr")
    s(f"res: UInt({w}) = 0")
    round_pack(s, f, "a_s", "e", "p", m + 3, "res", xwid)
    s(f"zs: UInt({w}) = a_s")
    with s.if_("rl == 0"):
        s(f"res = zs << {w - 1}")
    with s.if_(f"(a & {f.mag_mask}) < (b & {f.mag_mask})"):
        s("res = a")
    with s.if_("a_nan == 1 or b_nan == 1 or a_inf == 1 or b_zero == 1"):
        s(f"res = {f.nan}")
    with s.elif_("b_inf == 1 or a_zero == 1"):
        s("res = a")
    return _ret(s, "res")


def sign_ops(name: str, f: Float, op: str) -> str:
    """`negate`, `abs` (on the sign bit), `sign` (NaN and zeros unchanged, else +-1)."""
    w = f.bits
    s = Src(name, [("a", w)], w, f"{op} ({f.name})")
    s(f"res: UInt({w}) = a")
    if op == "negate":
        s(f"res = a ^ {s.lit(1 << (w - 1), w)}")
    elif op == "abs":
        s(f"res = a & {f.mag_mask}")
    else:
        decode(s, f, "a", "a", xw(f))
        with s.if_("a_nan == 0 and a_zero == 0"):
            s(f"sg: UInt({w}) = a_s")
            s(f"res = (sg << {w - 1}) | {f.one}")
    return _ret(s, "res")


def _order_key(s: Src, f: Float, v: str, key: str) -> None:
    """An unsigned key with the order of the (non-NaN) values: -0 just below +0 (one spare top bit, so
    keys compare unsigned in both backends)."""
    w = f.bits
    s(f"{key}: UInt({w + 1}) = {v} | {s.lit(1 << (w - 1), w)}")
    with s.if_(f"({v} >> {w - 1}) == 1"):
        s(f"{key} = {s.inv(v, w)}")


def compare(name: str, f: Float, direction: str) -> str:
    """IEEE quiet predicates EQ, NE, LT, LE: false on NaN except NE; -0 == +0."""
    w = f.bits
    s = Src(name, [("a", w), ("b", w)], 1, f"compare {direction} ({f.name})")
    decode(s, f, "a", "a", xw(f))
    decode(s, f, "b", "b", xw(f))
    _order_key(s, f, "a", "ka")
    _order_key(s, f, "b", "kb")
    s("eq: UInt(1) = 0")
    with s.if_("a == b or (a_zero == 1 and b_zero == 1)"):
        s("eq = 1")
    s("lt: UInt(1) = 0")
    with s.if_("ka < kb and eq == 0"):
        s("lt = 1")
    s("nan: UInt(1) = a_nan | b_nan")
    s("res: UInt(1) = 0")
    expr = {"EQ": "eq", "NE": "eq ^ 1", "LT": "lt", "LE": "lt | eq"}[direction]
    s(f"res = {expr}")
    with s.if_("nan == 1"):
        s(f"res = {1 if direction == 'NE' else 0}")
    return _ret(s, "res")


def _maxmin(s: Src, f: Float, x: str, y: str, out: str, is_max: bool) -> None:
    """IEEE 754-2019 maximum/minimum: NaN if either is NaN; -0 < +0; one of the operands otherwise."""
    px, py = s.tmp("mx"), s.tmp("my")
    decode(s, f, x, px, xw(f))
    decode(s, f, y, py, xw(f))
    kx, ky = s.tmp("kx"), s.tmp("ky")
    _order_key(s, f, x, kx)
    _order_key(s, f, y, ky)
    s(f"{out}: UInt({f.bits}) = {x}")
    with s.if_(f"{px}_zero == 1 and {py}_zero == 1"):
        with s.if_(f"{px}_s == {1 if is_max else 0}"):
            s(f"{out} = {y}")
    with s.elif_(f"{kx} < {ky}"):
        s(f"{out} = {y if is_max else x}")
    with s.else_():
        s(f"{out} = {x if is_max else y}")
    with s.if_(f"{px}_nan == 1 or {py}_nan == 1"):
        s(f"{out} = {f.nan}")


def maxmin(name: str, f: Float, is_max: bool) -> str:
    w = f.bits
    s = Src(name, [("a", w), ("b", w)], w, f"{'maximum' if is_max else 'minimum'} ({f.name})")
    _maxmin(s, f, "a", "b", "res", is_max)
    return _ret(s, "res")


def clamp(name: str, f: Float) -> str:
    """StableHLO `clamp(min, operand, max)` = minimum(maximum(operand, min), max)."""
    w = f.bits
    s = Src(name, [("lo", w), ("x", w), ("hi", w)], w, f"clamp ({f.name})")
    _maxmin(s, f, "x", "lo", "t", True)
    _maxmin(s, f, "t", "hi", "res", False)
    return _ret(s, "res")


def select(name: str, f: Float) -> str:
    w = f.bits
    s = Src(name, [("p", 1), ("a", w), ("b", w)], w, f"select ({f.name})")
    s(f"res: UInt({w}) = b")
    with s.if_("p == 1"):
        s("res = a")
    return _ret(s, "res")


def is_finite(name: str, f: Float) -> str:
    s = Src(name, [("a", f.bits)], 1, f"is_finite ({f.name})")
    decode(s, f, "a", "a", xw(f))
    s("res: UInt(1) = 1")
    with s.if_("a_nan == 1 or a_inf == 1"):
        s("res = 0")
    return _ret(s, "res")


def to_integral(name: str, f: Float, mode: str) -> str:
    """`round_nearest_even`, `floor`, `ceil`: clear the fraction bits of the pattern, rounding its magnitude
    (a carry into the exponent is exact); values below 1 in magnitude become +-0 or +-1."""
    w, m = f.bits, f.m
    s = Src(name, [("a", w)], w, f"{mode} ({f.name})")
    decode(s, f, "a", "a", xw(f))
    s(f"ue: Int({xw(f)}) = a_x - {f.bias}")
    s(f"mag: UInt({w}) = a & {f.mag_mask}")
    s(f"sg: UInt({w}) = a_s")
    s(f"sgn: UInt({w}) = sg << {w - 1}")
    s(f"res: UInt({w}) = a")
    # Round the magnitude up: to nearest even, toward -inf for negatives (floor), toward +inf for positives
    # (ceil); otherwise truncate.
    up = {"round_nearest_even": None, "floor": "a_s == 1", "ceil": "a_s == 0"}[mode]
    with s.if_(f"a_nan == 1 or a_inf == 1 or a_zero == 1 or ue >= {m}"):
        s("res = a")
    with s.elif_("ue < 0"):
        s("res = sgn")
        if up is None:
            # |a| in (0.5, 1) rounds to 1; exactly 0.5 to 0 (even); smaller to 0.
            with s.if_(f"mag > {(f.bias - 1) << m}"):
                s(f"res = sgn | {f.one}")
        else:
            with s.if_(up):
                s(f"res = sgn | {f.one}")
    with s.else_():
        s(f"k: UInt({clog2(m)}) = {m} - ue")
        s(f"one: UInt({w}) = 1")
        s(f"low: UInt({w}) = (one << k) - 1")
        s(f"r: UInt({w}) = mag")
        if up is None:
            s(f"lsb: UInt({w}) = (mag >> k) & 1")
            s("r = mag + (low >> 1) + lsb")
        else:
            with s.if_(up):
                s("r = mag + low")
        s(f"res = sgn | (r & {s.inv('low', w)})")
    return _ret(s, "res")


def reduce_precision(name: str, f: Float, eb: int, mb: int) -> str:
    """XLA `reduce_precision` to (eb, mb): round the significand to `mb` bits on the pattern (ties to even),
    then overflow to +-inf and underflow (below the reduced format's normal range) to +-0; NaN and infinities
    unchanged."""
    w, m = f.bits, f.m
    assert not f.finite_nan and mb <= m and eb <= f.e
    s = Src(name, [("a", w)], w, f"reduce_precision {eb} {mb} ({f.name})")
    decode(s, f, "a", "a", xw(f))
    k = m - mb
    s(f"r: UInt({w}) = a")
    if k > 0:
        s(f"r = a + {(1 << (k - 1)) - 1} + ((a >> {k}) & 1)")
        s(f"r = r & {s.lit(~((1 << k) - 1) & ((1 << w) - 1), w)}")
    rbias = (1 << (eb - 1)) - 1
    s(f"ex: UInt({f.e}) = (r >> {m}) & {f.emax_field}")
    s(f"sg: UInt({w}) = a_s")
    s("res: UInt({0}) = r".format(w))
    with s.if_(f"ex > {rbias + f.bias}"):
        s(f"res = (sg << {w - 1}) | {f.inf}")
    with s.elif_(f"ex < {1 - rbias + f.bias}"):
        s(f"res = sg << {w - 1}")
    with s.if_("a_nan == 1 or a_inf == 1"):
        s("res = a")
    return _ret(s, "res")


def convert(name: str, src: Float | Int, dst: Float | Int) -> str:
    if isinstance(src, Int) and isinstance(dst, Float):
        return _int_to_float(name, src, dst)
    if isinstance(src, Float) and isinstance(dst, Int):
        return _float_to_int(name, src, dst)
    if isinstance(src, Float) and isinstance(dst, Float):
        return _float_to_float(name, src, dst)
    raise ValueError(f"convert {src.name} -> {dst.name} is not a float conversion")


def _int_to_float(name: str, t: Int, f: Float) -> str:
    """Round to nearest even; a bool is 0 or 1."""
    w = t.w
    s = Src(name, [("a", w)], f.bits, f"convert {t.name} -> {f.name}")
    if t == BOOL:
        s(f"res: UInt({f.bits}) = 0")
        with s.if_("a == 1"):
            s(f"res = {f.one}")
        return _ret(s, "res")
    xwid = max(xw(f), clog2(w) + 2)
    pw = max(w, f.m + 3)
    s("neg: UInt(1) = 0")
    s(f"mag: UInt({w}) = a")
    if t.signed:
        s(f"neg = (a >> {w - 1}) & 1")
        with s.if_("neg == 1"):
            s(f"mag = {s.inv('a', w)} + 1")
    s(f"pm: UInt({pw}) = mag")
    s(f"pm = pm << {pw - w}")
    normalize(s, "pm", pw, "pn", "lzp")
    s(f"e: Int({xwid}) = {f.bias + w - 1} - lzp")
    s(f"res: UInt({f.bits}) = 0")
    round_pack(s, f, "neg", "e", "pn", pw, "res", xwid, denorm=False)
    with s.if_("mag == 0"):
        s("res = 0")
    return _ret(s, "res")


def _float_to_int(name: str, f: Float, t: Int) -> str:
    """Truncate toward zero, saturate to the integer range, NaN to 0; to bool: non-zero (NaN included)."""
    w, m = f.bits, f.m
    s = Src(name, [("a", w)], t.w, f"convert {f.name} -> {t.name}")
    if t == BOOL:
        s("res: UInt(1) = 0")
        with s.if_(f"(a & {f.mag_mask}) != 0"):
            s("res = 1")
        return _ret(s, "res")
    xwid = max(xw(f), clog2(t.w) + 3)
    tw = max(m + 1, t.w + 1) + 1
    decode(s, f, "a", "a", xwid)
    s(f"ue: Int({xwid}) = a_x - {f.bias}")
    s("sat: UInt(1) = 0")
    s(f"mag: UInt({tw}) = 0")
    s(f"am: UInt({tw}) = a_m")
    with s.if_("a_inf == 1"):
        s("sat = 1")
    with s.elif_(f"ue > {t.w}"):
        s("sat = 1")
    with s.elif_(f"ue >= {m}"):
        s(f"mag = am << (ue - {m})")
    with s.elif_("ue >= 0"):
        s(f"mag = am >> ({m} - ue)")
    hi = s.lit(t.hi, tw)
    lo_mag = s.lit(-t.lo, tw)
    s(f"res: UInt({t.w}) = 0")
    s(f"rm: UInt({t.w}) = mag")
    with s.if_("a_s == 0"):
        with s.if_(f"sat == 1 or mag > {hi}"):
            s(f"res = {s.lit(t.hi, t.w)}")
        with s.else_():
            s("res = rm")
    with s.else_():
        if t.signed:
            with s.if_(f"sat == 1 or mag > {lo_mag}"):
                s(f"res = {s.lit(t.lo & ((1 << t.w) - 1), t.w)}")
            with s.else_():
                s(f"res = {s.inv('rm', t.w)} + 1")
        else:
            s("res = 0")
    with s.if_("a_nan == 1"):
        s("res = 0")
    return _ret(s, "res")


def _float_to_float(name: str, f: Float, g: Float) -> str:
    """Exact when `g` holds every value of `f`; otherwise one rounding (subnormal range included). Overflow
    and infinities give infinity, or NaN in a finite-NaN target."""
    xwid = max(xw(f), xw(g))
    pw = max(f.m + 1, g.m + 3)
    s = Src(name, [("a", f.bits)], g.bits, f"convert {f.name} -> {g.name}")
    decode(s, f, "a", "a", xwid)
    normalize(s, "a_m", f.m + 1, "an", "lza")
    s(f"pm: UInt({pw}) = an")
    s(f"pm = pm << {pw - (f.m + 1)}")
    s(f"e: Int({xwid}) = a_x - lza - {f.bias - g.bias}")
    s(f"res: UInt({g.bits}) = 0")
    round_pack(s, g, "a_s", "e", "pm", pw, "res", xwid)
    s(f"sg: UInt({g.bits}) = a_s")
    with s.if_("a_zero == 1"):
        s(f"res = sg << {g.bits - 1}")
    with s.if_("a_nan == 1"):
        s(f"res = {g.nan}")
    with s.elif_("a_inf == 1"):
        s(f"res = (sg << {g.bits - 1}) | {g.inf}")
    return _ret(s, "res")
