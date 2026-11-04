"""cc_softfloat: the reference on hand-computed cases, the generators and the vectors (no Allo needed); the
generated functions on Allo's `llvm` target when Allo is importable (the toolchain image)."""

import importlib.util
import math
import struct

import numpy as np
import pytest

from cc_softfloat import catalog, reference, vectors
from cc_softfloat.formats import FLOATS, parse

F32 = FLOATS["f32"]
E4M3 = FLOATS["f8e4m3fn"]


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def run(key: str, *args: int) -> int:
    p = catalog.parse_key(key)
    return int(reference.evaluate(p, [np.array([a], dtype=np.uint64) for a in args])[0])


def test_reference_rounds_once_to_nearest_even():
    assert run("add@f32", f32(1.0), f32(2.0)) == f32(3.0)
    # 1 + 2^-24 is a tie between 1 and 1 + 2^-23: even is 1.
    assert run("add@f32", f32(1.0), f32(2.0**-24)) == f32(1.0)
    assert run("add@f32", f32(1.0), f32(3 * 2.0**-24)) == f32(1.0 + 2.0**-22)
    # Subnormal results stay (no flush to zero).
    assert run("multiply@f32", f32(2.0**-126), f32(0.5)) == f32(2.0**-127)
    assert run("divide@f32", f32(1.0), f32(3.0)) == f32(1.0 / 3.0)
    assert run("sqrt@f32", f32(2.0)) == f32(math.sqrt(2.0))


def test_reference_specials():
    nan = run("add@f32", f32(math.inf), f32(-math.inf))
    assert math.isnan(struct.unpack("<f", struct.pack("<I", nan))[0])
    assert run("multiply@f32", f32(-0.0), f32(5.0)) == f32(-0.0)
    assert run("add@f32", f32(-0.0), f32(-0.0)) == f32(-0.0)
    assert run("add@f32", f32(-0.0), f32(0.0)) == f32(0.0)
    assert run("remainder@f32", f32(-4.0), f32(2.0)) == f32(-0.0)
    assert run("remainder@f32", f32(7.5), f32(2.0)) == f32(1.5)


def test_e4m3fn_overflow_is_nan_and_448_is_the_max():
    one = E4M3.one
    assert E4M3.max_finite == 0x7E
    assert run("convert@f32>f8e4m3fn", f32(448.0)) == 0x7E
    assert run("convert@f32>f8e4m3fn", f32(464.0)) == 0x7E  # a tie, to the even 448 (OFP8 1.1, App. A)
    assert run("convert@f32>f8e4m3fn", f32(470.0)) & 0x7F == 0x7F  # past the tie: NaN
    assert run("add@f8e4m3fn", 0x7E, 0x7E) & 0x7F == 0x7F
    assert run("multiply@f8e4m3fn", one, one) == one


def test_wide_integers_round_once():
    # 0x837fffd9 rounds down to bf16 directly; through float32 it would become a tie and round up.
    assert run("convert@ui32>bf16", 0x837FFFD9) == 0x4F03
    assert run("convert@i32>f8e4m3fn", 470) & 0x7F == 0x7F  # rounds up past 448: NaN
    assert run("convert@i32>f8e4m3fn", 464) == 0x7E  # a tie, to the even 448


def test_float_to_int_truncates_saturates_and_maps_nan_to_zero():
    assert run("convert@f32>i8", f32(-3.9)) == 0xFD
    assert run("convert@f32>i8", f32(1000.0)) == 0x7F
    assert run("convert@f32>i8", f32(-1000.0)) == 0x80
    assert run("convert@f32>ui8", f32(-5.0)) == 0
    assert run("convert@f32>i32", f32(math.nan)) == 0
    assert run("convert@f32>i32", f32(2.0**31)) == 0x7FFFFFFF
    assert run("convert@f32>i1", f32(math.nan)) == 1


def test_maximum_minimum_order_zeros_and_propagate_nan():
    assert run("maximum@f32", f32(-0.0), f32(0.0)) == f32(0.0)
    assert run("minimum@f32", f32(0.0), f32(-0.0)) == f32(-0.0)
    assert run("maximum@f32", f32(math.nan), f32(1.0)) == F32.nan


def test_every_key_generates_valid_python():
    keys = [
        "add@f32", "subtract@bf16", "multiply@f16", "divide@f8e4m3fn", "sqrt@f32", "remainder@bf16",
        "negate@f16", "abs@f32", "sign@bf16", "maximum@f16", "minimum@f32", "clamp@bf16",
        "select@f8e4m3fn", "is_finite@f32", "round_nearest_even@f16", "floor@f32", "ceil@bf16",
        "reduce_precision@f32", "compare.EQ@f32", "compare.NE@f16", "compare.LT@bf16", "compare.LE@f8e4m3fn",
        "convert@i8>f32", "convert@ui32>f16", "convert@i1>bf16", "convert@f32>i32", "convert@f16>ui8",
        "convert@bf16>i1", "convert@f32>f8e4m3fn", "convert@f8e4m3fn>f32", "convert@f16>bf16",
    ]
    for k in keys:
        src = catalog.source(catalog.parse_key(k), "kernel_fn")
        compile(src, k, "exec")
        assert "~" not in src.split('"""')[-1], k


def test_exhaustive_and_random_vectors():
    p = catalog.parse_key("add@f8e4m3fn")
    a, b = vectors.generate(p, 16, 10)
    assert len(a) == 1 << 16 and len(set(zip(a.tolist(), b.tolist()))) == 1 << 16
    p = catalog.parse_key("add@f32")
    a, b = vectors.generate(p, 16, 1000)
    edges = vectors.edges(F32)
    assert len(a) == len(edges) ** 2 + 1000
    assert F32.inf in edges and F32.nan in edges and 1 in edges
    again = vectors.generate(p, 16, 1000)
    assert (again[0] == a).all() and (again[1] == b).all()
    assert parse("ui32").hi == 2**32 - 1 and parse("i8").lo == -128


@pytest.mark.skipif(importlib.util.find_spec("allo") is None, reason="needs Allo (the toolchain image)")
def test_generated_functions_match_the_reference_on_allo(tmp_path):
    from cc_softfloat.build import _simulate

    for key in ["multiply@f8e4m3fn", "convert@f8e4m3fn>f32", "compare.LT@f8e4m3fn"]:
        p = catalog.parse_key(key)
        name = key.replace("@", "_").replace(".", "_").replace(">", "_")
        out = tmp_path / name
        out.mkdir()
        (out / "kernel.py").write_text(catalog.source(p, name))
        args = vectors.generate(p, 16, 0)
        got = _simulate(p, out, name, args)
        assert reference.same(p.ret, got, reference.evaluate(p, args)).all(), key
