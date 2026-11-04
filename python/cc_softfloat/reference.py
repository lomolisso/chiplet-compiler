"""The reference: NumPy and ml_dtypes on bit patterns, independent of the generated circuits.

Values are decoded to float64, which holds every value of the library's formats exactly; an op is computed
there and rounded once to the result format by NumPy/ml_dtypes (round to nearest, ties to even, with
subnormals; overflow to infinity, or to NaN for E4M3FN). For +, -, *, / and sqrt one float64 operation
followed by that rounding is the correctly rounded result, because 53 >= 2p + 2 for every p <= 24 (Figueroa,
"When is double rounding innocuous?", SIGNUM Newsletter 30(3), 1995). `fmod` is exact in float64.
Everything else is a direct definition (NaN results are compared as one class by the checker).
"""

from __future__ import annotations

import numpy as np

from .catalog import Prim
from .formats import Float, Int
from .fp import DTYPES, check_ulps, to_f64, udtype

#: Accuracy limits in ulps by `approx.Design.accuracy`.
LIMITS = {"correctly_rounded": 0.5, "faithful": 1.0}


def approx_f64(op: str, x: np.ndarray) -> np.ndarray:
    """The float64 value of an approximate op (NumPy; a few float64 ulps from the exact value)."""
    with np.errstate(all="ignore"):
        match op:
            case "exponential":
                return np.exp(x)
            case "log":
                return np.log(x)
            case "log_plus_one":
                return np.log1p(x)
            case "tanh":
                return np.tanh(x)
            case "rsqrt":
                return 1.0 / np.sqrt(x)
    raise ValueError(op)


def from_f64(f: Float, x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return x.astype(DTYPES[f.name]).view(udtype(f.bits)).astype(np.uint64)


def to_int(t: Int, bits: np.ndarray) -> np.ndarray:
    v = bits.astype(np.int64)
    if t.signed:
        v = np.where(v >= (1 << (t.w - 1)), v - (1 << t.w), v)
    return v


def round_int(v: np.ndarray, p: int) -> np.ndarray:
    """Integers rounded to `p` significant bits, ties to even (exact integer arithmetic). ml_dtypes converts
    float64 to bfloat16 and float8 through float32, which would round a wide integer twice."""
    out = []
    for x in v.tolist():
        a = abs(x)
        shift = a.bit_length() - p
        if shift > 0:
            q, rem = a >> shift, a & ((1 << shift) - 1)
            half = 1 << (shift - 1)
            if rem > half or (rem == half and q & 1):
                q += 1
            a = q << shift
        out.append(-a if x < 0 else a)
    return np.array(out, dtype=np.float64)


def is_nan(t: Float | Int, bits: np.ndarray) -> np.ndarray:
    if isinstance(t, Int):
        return np.zeros(bits.shape, dtype=bool)
    return np.isnan(to_f64(t, bits))


def _maxmin(f: Float, a: np.ndarray, b: np.ndarray, is_max: bool) -> np.ndarray:
    x, y = to_f64(f, a), to_f64(f, b)
    both0 = (x == 0) & (y == 0)
    zero_pick = np.where(np.signbit(x) == is_max, b, a)
    lt = x < y
    pick = np.where(lt, b, a) if is_max else np.where(lt, a, b)
    out = np.where(both0, zero_pick, pick)
    return np.where(np.isnan(x) | np.isnan(y), np.uint64(f.nan), out).astype(np.uint64)


def _reduce_precision(f: Float, a: np.ndarray, eb: int, mb: int) -> np.ndarray:
    k = f.m - mb
    mask = (1 << f.bits) - 1
    r = a.astype(np.uint64)
    if k > 0:
        r = (r + ((1 << (k - 1)) - 1) + ((r >> np.uint64(k)) & np.uint64(1))) & np.uint64(mask)
        r = r & np.uint64(~((1 << k) - 1) & mask)
    rbias = (1 << (eb - 1)) - 1
    ex = (r >> np.uint64(f.m)) & np.uint64(f.emax_field)
    sign = a & np.uint64(1 << (f.bits - 1))
    out = np.where(ex > rbias + f.bias, sign | np.uint64(f.inf), r)
    out = np.where(ex < 1 - rbias + f.bias, sign, out)
    x = to_f64(f, a)
    return np.where(np.isnan(x) | np.isinf(x), a, out).astype(np.uint64)


def evaluate(p: Prim, args: list[np.ndarray]) -> np.ndarray:
    """The result bit patterns of `p` on the operand bit patterns `args` (uint64 arrays); for an approximate
    op, the float64 values (`approx_check` compares with them)."""
    op, ret = p.op, p.ret
    with np.errstate(all="ignore"):
        if op == "convert":
            src = p.params[0]
            if isinstance(src, Int) and isinstance(ret, Float):
                return from_f64(ret, round_int(to_int(src, args[0]), ret.m + 1))
            if isinstance(src, Float) and isinstance(ret, Int):
                x = to_f64(src, args[0])
                if ret.w == 1 and not ret.signed:
                    return (x != 0).astype(np.uint64)
                t = np.trunc(np.where(np.isnan(x), 0.0, x))
                t = np.clip(t, ret.lo, ret.hi).astype(np.int64)
                return (t & ((1 << ret.w) - 1)).astype(np.uint64)
            if isinstance(src, Float) and isinstance(ret, Float):
                return from_f64(ret, to_f64(src, args[0]))
            raise ValueError(p.key)
        f = p.params[-1]
        assert isinstance(f, Float)
        if op == "select":
            return np.where(args[0] == 1, args[1], args[2]).astype(np.uint64)
        if op in {"maximum", "minimum"}:
            return _maxmin(f, args[0], args[1], op == "maximum")
        if op == "clamp":
            lo, x, hi = args
            return _maxmin(f, _maxmin(f, x, lo, True), hi, False)
        if op == "reduce_precision":
            return _reduce_precision(f, args[0], *p.precision)
        xs = [to_f64(f, a) for a in args]
        if p.approximate:
            return approx_f64(op, xs[0])
        if op == "compare":
            x, y = xs
            r = {"EQ": x == y, "NE": x != y, "LT": x < y, "LE": x <= y}[p.variant or ""]
            return r.astype(np.uint64)
        if op == "is_finite":
            return np.isfinite(xs[0]).astype(np.uint64)
        if op == "sign":
            x = xs[0]
            r = np.where((x == 0) | np.isnan(x), x, np.where(x < 0, -1.0, 1.0))
            return from_f64(f, r)
        fn = {
            "add": np.add,
            "subtract": np.subtract,
            "multiply": np.multiply,
            "divide": np.divide,
            "remainder": np.fmod,
            "sqrt": np.sqrt,
            "negate": np.negative,
            "abs": np.abs,
            "round_nearest_even": np.rint,
            "floor": np.floor,
            "ceil": np.ceil,
        }[op]
        return from_f64(f, fn(*xs))


def approx_check(p: Prim, limit: float, got: np.ndarray, want: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For an approximate op: per vector, is `got` within `limit` ulps of the float64 value `want`
    (`fp.check_ulps`), and the error in ulps."""
    assert isinstance(p.ret, Float)
    return check_ulps(p.ret, got, want, limit)


def same(t: Float | Int, got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Equal bit patterns, any NaN equal to any NaN (the proofs' equality)."""
    eq = got.astype(np.uint64) == want.astype(np.uint64)
    if isinstance(t, Float):
        eq |= is_nan(t, got) & is_nan(t, want)
    return eq
