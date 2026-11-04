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
UNARY_KEYS = [k for k in KEYS if k.split("@")[0] in approx.UNARY]


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


def test_pi_and_two_over_pi():
    assert consts.pi_fixed(80) >> 29 == int(math.pi * 2**51)  # math.pi is pi rounded down to 53 bits
    assert abs(consts.two_over_pi_fixed(80) / 2**80 - 2 / math.pi) < 2**-52
    # 2/pi = 0.101000101111100110000011011011100100111001000100000101010010100111111100001001110101...
    assert consts.two_over_pi_fixed(24) == 0b101000101111100110000011


def test_trig_worst_cases_are_the_minimum():
    """The recorded worst cases are the minimum over every input (f32: about 10 s), and each one's distance
    is what its K claims."""
    for name, (m, ex, kd) in approx.TRIG_WORST.items():
        f = FLOATS[name]
        dist, wm, wex = consts.trig_worst_case(f.m + 1, -1, f.bias)
        assert (wm, wex) == (m, ex) and 2.0**-kd < dist, name
        approx._trig_check_worst(f)


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
        if op in approx.UNARY:
            assert d.accuracy == "correctly_rounded" and d.params["rom"] == 256
        else:
            assert d.accuracy == "faithful" and "rom" not in d.params and d.bound_ulp <= 1.0


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
        assert p.approximate and p.params == [p.ret] * (2 if p.op in approx.BINARY else 1)
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


def test_power_vectors_cover_the_special_cases():
    p = catalog.parse_key("power@f32")
    a, b = vectors.generate(p, 16, 4000)
    assert len(a) == len(b) > 4000
    pairs = set(zip(a.tolist(), b.tolist()))
    one, inf, nan = F32.one, F32.inf, F32.nan
    neg = 1 << 31
    for x, y in [(0, neg | bits(F32, 3.0)), (neg | one, inf), (one, nan), (nan, 0), (neg | inf, neg | bits(F32, 3.0))]:
        assert (x, y) in pairs, (hex(x), hex(y))


def test_power_reference_follows_annex_f():
    """NumPy's float64 power is the reference: it follows C11 Annex F.10.4.4 on the special cases."""
    inf, nan = math.inf, math.nan
    cases = [
        (nan, 0.0, 1.0), (1.0, nan, 1.0), (-1.0, inf, 1.0), (-1.0, -inf, 1.0),
        (0.0, -3.0, inf), (-0.0, -3.0, -inf), (-0.0, -2.0, inf), (-0.0, 3.0, -0.0), (-0.0, 2.0, 0.0),
        (0.5, -inf, inf), (2.0, -inf, 0.0), (0.5, inf, 0.0), (2.0, inf, inf),
        (-inf, -3.0, -0.0), (-inf, -2.0, 0.0), (-inf, 3.0, -inf), (-inf, 2.0, inf),
        (inf, -1.0, 0.0), (inf, 0.5, inf), (-2.0, 0.5, nan), (-2.0, 3.0, -8.0),
    ]
    for x, y, want in cases:
        got = float(reference.approx_f64("power", np.array([x]), np.array([y]))[0])
        assert (math.isnan(got) and math.isnan(want)) or (got == want and math.copysign(1, got) == math.copysign(1, want)), (x, y, got)


@pytest.mark.skipif(importlib.util.find_spec("allo") is None, reason="needs Allo (the toolchain image)")
def test_approximate_ops_on_allo(tmp_path):
    from cc_softfloat.build import _simulate

    for key in ["exponential@f8e4m3fn", "rsqrt@bf16", "sine@f16", "power@f8e4m3fn"]:
        p = catalog.parse_key(key)
        name = key.replace("@", "_")
        out = tmp_path / name
        out.mkdir()
        (out / "kernel.py").write_text(catalog.source(p, name))
        args = vectors.generate(p, 16, 0)
        got = _simulate(p, out, name, args)
        limit = reference.LIMITS[approx.design(p.op, p.ret).accuracy]
        ok, _ = reference.approx_check(p, limit, got, reference.evaluate(p, args))
        assert ok.all(), key
