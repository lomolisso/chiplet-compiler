"""cc_softfloat's approximate ops: constants, exact rounding, the ulp check, the designs' error bounds, the
ROM tables and the vectors (no Allo needed); a ROM and a computed op on Allo's `llvm` target when Allo is
importable (the toolchain image)."""

import importlib.util
import math

import numpy as np
import pytest

from cc_softfloat import approx, catalog, consts, fp, reference, vectors
from cc_softfloat.formats import FLOATS

F32, F16, BF16, E4M3 = FLOATS["f32"], FLOATS["f16"], FLOATS["bf16"], FLOATS["f8e4m3fn"]
KEYS = [f"{op}@{f}" for op in approx.OPS for f in ("f32", "bf16", "f16", "f8e4m3fn")]


def bits(f, x: float) -> int:
    return int(fp.round_f64(f, np.array([x]))[0])


def test_constants_agree_with_math():
    assert abs(consts.ln2_fixed(80) / 2**80 - math.log(2)) < 2**-52
    assert abs(consts.inv_ln2_fixed(80) / 2**80 - 1 / math.log(2)) < 2**-52
    tab = consts.exp2_table(5, 60)
    assert tab[0] == 1 << 60
    assert all(abs(v / 2**60 - 2 ** (j / 32)) < 2**-51 for j, v in enumerate(tab))
    cs, ls = consts.log_tables(4, 9, 60)
    assert len(cs) == 17 and cs[0] == 1 << 9 and cs[16] == 1 << 8
    assert ls[0] == 0 and ls[16] == consts.ln2_fixed(60)
    assert all(abs(lv / 2**60 + math.log(c / 2**9)) < 2**-51 for c, lv in zip(cs, ls))


def test_round_f64_rounds_once():
    # 1 + 2^-8 + 2^-30 is above the bf16 tie between 1 and 1 + 2^-7: up (via float32 it would tie to 1)
    assert bits(BF16, 1 + 2**-8 + 2**-30) == 0x3F81
    assert bits(BF16, 1 + 2**-8) == 0x3F80
    assert bits(E4M3, 448.0) == 0x7E and bits(E4M3, 470.0) == E4M3.nan
    assert bits(F16, 2.0**-25) == 0 and bits(F16, 2.0**-25 * 1.01) == 1
    assert bits(F32, -math.inf) == 0xFF800000 and bits(E4M3, math.inf) == E4M3.nan


def test_check_ulps():
    one = np.array([bits(F32, 1.0)], dtype=np.uint64)
    up = one + np.uint64(1)

    def ok(got, v, limit):
        return bool(fp.check_ulps(F32, got, np.array([v]), limit)[0][0])

    assert ok(one, 1.0, 0.5) and ok(up, 1.0, 1.0) and not ok(up, 1.0, 0.5)
    assert not ok(up + np.uint64(1), 1.0, 1.0)
    assert ok(np.array([F32.nan], dtype=np.uint64), math.nan, 1.0)
    assert not ok(one, math.nan, 1.0)
    # exact zero: the same zero; a tiny value: a zero of its sign
    assert ok(np.array([0], dtype=np.uint64), 0.0, 1.0)
    assert not ok(np.array([0x80000000], dtype=np.uint64), 0.0, 1.0)
    assert ok(np.array([0x80000000], dtype=np.uint64), -1e-46, 1.0)
    # overflow: infinity from max + ulp/2 on
    inf = np.array([F32.inf], dtype=np.uint64)
    top = fp.max_value(F32)
    assert ok(inf, top * 1.0001, 1.0) and not ok(inf, top * 0.99999, 1.0)
    _, err = fp.check_ulps(F32, up, np.array([1.0]), 1.0)
    assert abs(err[0] - 1.0) < 1e-12


def test_designs_bound_their_error():
    for op in approx.OPS:
        for f in (F32, BF16, F16):
            d = approx.design(op, f)
            limit = reference.LIMITS[d.accuracy]
            assert d.bound_ulp <= limit, (op, f.name, d)
            assert d.accuracy == ("correctly_rounded" if op == "rsqrt" else "faithful")
        d = approx.design(op, E4M3)
        assert d.accuracy == "correctly_rounded" and d.params["rom"] == 256


def test_rom_tables():
    one, zero = E4M3.one, 0
    assert approx.rom_values("exponential", E4M3)[zero] == one
    assert approx.rom_values("log", E4M3)[one] == zero
    assert approx.rom_values("rsqrt", E4M3)[bits(E4M3, 4.0)] == bits(E4M3, 0.5)
    # log(0) = -inf and rsqrt(0) = inf: NaN in a format without infinities; exp overflows to NaN
    assert approx.rom_values("log", E4M3)[zero] == E4M3.nan
    assert approx.rom_values("rsqrt", E4M3)[zero] == E4M3.nan
    assert approx.rom_values("exponential", E4M3)[bits(E4M3, 7.0)] == E4M3.nan
    tanh = approx.rom_values("tanh", E4M3)
    x = fp.to_f64(E4M3, np.arange(256, dtype=np.uint64))
    assert (np.array(tanh, dtype=np.uint64) == fp.round_f64(E4M3, np.tanh(x))).all()


def test_every_approximate_key_generates_valid_python():
    for k in KEYS:
        p = catalog.parse_key(k)
        assert p.approximate and p.params == [p.ret]
        src = catalog.source(p, "kernel_fn")
        compile(src, k, "exec")
        assert "~" not in src.split('"""')[-1], k


def test_approximate_vectors():
    p = catalog.parse_key("exponential@f32")
    (a,) = vectors.generate(p, 16, 2000)
    assert len(a) > 2000 and (vectors.generate(p, 16, 2000)[0] == a).all()
    # the boundaries are there (ln of the largest finite value, with its neighbours)
    b = bits(F32, math.log(fp.max_value(F32)))
    assert {b - 1, b, b + 1} <= set(a.tolist())
    # half of the random inputs are in the range where exp does real work
    x = fp.to_f64(F32, a)
    assert np.count_nonzero((np.abs(x) < 104) & (np.abs(x) > 2.0**-26)) > 1000
    (h,) = vectors.generate(catalog.parse_key("tanh@f16"), 16, 10)
    assert len(h) == 1 << 16


@pytest.mark.skipif(importlib.util.find_spec("allo") is None, reason="needs Allo (the toolchain image)")
def test_approximate_ops_on_allo(tmp_path):
    from cc_softfloat.build import _simulate

    for key in ["exponential@f8e4m3fn", "rsqrt@bf16"]:
        p = catalog.parse_key(key)
        name = key.replace("@", "_")
        out = tmp_path / name
        out.mkdir()
        (out / "kernel.py").write_text(catalog.source(p, name))
        args = vectors.generate(p, 16, 0)
        got = _simulate(p, out, name, args)
        ok, _ = reference.approx_check(p, 0.5, got, reference.evaluate(p, args))
        assert ok.all(), key
