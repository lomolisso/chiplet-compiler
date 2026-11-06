"""The core models of `cc_float.models` (the names the IP providers give): exact IEEE arithmetic against
numpy and ml_dtypes, conversions, compare, integer conversions, the reference functions and FloPoCo's
encoding."""

import ml_dtypes
import numpy as np
import pytest

from cc_float.models import MODEL_NAMES, model

RNG = np.random.default_rng(7)


def f16_bits(n: int) -> np.ndarray:
    b = RNG.integers(0, 1 << 16, n, dtype=np.uint64)
    v = b.astype(np.uint16).view(np.float16)
    return b[np.isfinite(v)]


def as16(b) -> np.ndarray:
    return np.asarray(b, dtype=np.uint64).astype(np.uint16).view(np.float16)


def bits16(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float16).view(np.uint16).astype(np.uint64)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_name_resolves(name):
    op = {
        "ieee.to_int": "to_int.rtz.s8",
        "ieee.from_int": "from_int.s8",
        "ieee.convert": "convert.e8m7",
    }.get(name, name.split(".")[1])
    fn = model(name, op, 5, 10)
    args = 2 if name.split(".")[1] in ("add", "sub", "mul", "div", "compare", "pow") else 1
    out = fn(*([np.array([0x3C00], dtype=np.uint64)] * args))
    assert out.shape == (1,)


def test_f16_arithmetic_matches_numpy():
    a = f16_bits(3000)
    b = f16_bits(3000)[: len(a)]
    a = a[: len(b)]
    x, y = as16(a), as16(b)
    with np.errstate(all="ignore"):
        for op, ref in [("add", x + y), ("sub", x - y), ("mul", x * y), ("div", x / y)]:
            got = as16(model(f"ieee.{op}", op, 5, 10)(a, b))
            same = (got == ref) | (np.isnan(got) & np.isnan(ref))
            same &= np.signbit(got) == np.signbit(ref)
            same |= np.isnan(got) & np.isnan(ref)
            assert same.all(), op
        s = as16(model("ieee.sqrt", "sqrt", 5, 10)(a))
        ref = np.sqrt(x)
        assert ((s == ref) | (np.isnan(s) & np.isnan(ref))).all()
    lt = model("ieee.compare", "compare", 5, 10)(a, b)
    assert np.array_equal(lt.astype(bool), x < y)


def test_conversions():
    a = f16_bits(2000)
    x = as16(a).astype(np.float32)
    # f16 -> bf16 (convert.e8m7): one rounding, as ml_dtypes.
    got = model("ieee.convert", "convert.e8m7", 5, 10)(a).astype(np.uint16)
    want = x.astype(ml_dtypes.bfloat16).view(np.uint16)
    ok = (got == want) | (np.isnan(x))
    assert ok.all()
    # Integers: saturating, toward zero; NaN gives 0.
    t = model("ieee.to_int", "to_int.rtz.s8", 5, 10)
    vals = bits16([1.5, -1.5, 300.0, -300.0, np.nan, -0.0])
    assert [int(v) for v in t(vals)] == [1, 0xFF, 127, 0x80, 0, 0]
    r = model("ieee.to_int", "to_int.rne.s8", 5, 10)
    assert [int(v) for v in r(bits16([2.5, 3.5, -2.5]))] == [2, 4, 0xFE]
    fi = model("ieee.from_int", "from_int.s16", 5, 10)
    ints = np.array([0, 1, 0xFFFF, 2049, 0x7FFF], dtype=np.uint64)
    want = bits16(np.array([0, 1, -1, 2049, 32767], dtype=np.float64).astype(np.float16))
    assert np.array_equal(fi(ints), want)


def test_references_round_the_function():
    a = bits16([0.5, 1.0, 2.0, -1.0])
    e = as16(model("reference.exp", "exp", 5, 10)(a))
    assert np.array_equal(e, np.exp(as16(a).astype(np.float64)).astype(np.float16))
    p = as16(model("reference.pow", "pow", 5, 10)(bits16([2.0]), bits16([10.0])))
    assert p[0] == 1024.0


def test_flopoco_encoding_round_trips():
    a = f16_bits(2000)
    fin = model("flopoco.from_ieee", "from_ieee", 5, 10)
    fout = model("flopoco.to_ieee", "to_ieee", 5, 10)
    back = fout(fin(a))
    x = as16(a)
    sub = (np.abs(x) < 2.0**-14) & (x != 0)
    keep = ~sub & ~np.isnan(x)
    assert np.array_equal(back[keep], a[keep])
    # Subnormals flush to zero of their sign.
    assert (as16(back[sub]) == 0).all()
