"""Generators of the approximate float ops (exponential, log, log_plus_one, tanh, rsqrt).

The datapath semantics leave these ops' results to an implementation (uninterpreted functions in the proofs),
so their accuracy is this library's choice: within one ulp in every format (docs/passes/softfloat.md §4).

- Formats of at most `ROM_BITS` bits use a table of the correctly rounded results of every input.
- `rsqrt` is correctly rounded in every format: an exact integer division and an exact integer square root
  give `floor(2^N / sqrt(m))` and whether a remainder is left, which is all round-to-nearest needs.
- `exponential`, `log`, `log_plus_one` and `tanh` reduce the argument with a table (Tang's methods), evaluate
  a Taylor polynomial in fixed point, and round once. `design` picks each one's parameters and bounds the
  error of the approximation before that rounding (argument reduction, table and coefficient rounding,
  polynomial truncation, every truncated product); the bound is at most 2^-G ulp, so the rounded result is
  within 1/2 + 2^-G ulp of the exact value.

All arithmetic is unsigned fixed point (a value `n` with `F` fraction bits is `n / 2^F`), with signs kept
apart, so the `llvm` simulation and the XLS C++ agree without signed shifts and products.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import consts, fp
from .emit import Src, clog2, decode, normalize, round_pack, xw
from .formats import Float

G = 4
ROM_BITS = 8
UNARY = ("exponential", "log", "log_plus_one", "tanh", "rsqrt", "sine", "cosine")
BINARY = ("power",)
OPS = UNARY + BINARY


@dataclass
class Design:
    op: str
    fmt: str
    accuracy: str
    #: Bound of the result's error in ulps: 1/2 for a correctly rounded result, else 1/2 plus the bound of
    #: the approximation that is rounded.
    bound_ulp: float
    params: dict = field(default_factory=dict)


def _p(f: Float) -> int:
    return f.m + 1


def _ulps(f: Float, rel: float) -> float:
    """A relative error as ulps of the result (an ulp is at least 2^-p of the value)."""
    return rel * 2.0 ** _p(f)


# --------------------------------------------------------------------------------------------- exponential


def _exp_params(f: Float, frac: int, int_bits: int, t: int, exact_x: bool) -> tuple[dict, float]:
    """Parameters of the shared exp core for `|x| < 2^int_bits` with `frac` fraction bits, and the bound of
    its relative error."""
    ln2 = math.log(2)
    k_extra = int_bits + t + 3
    s_bits = int_bits + 4
    big_l = ln2 / (1 << t)
    nmax = (1 << int_bits) * (1 << t) / ln2 + 2
    e_l = nmax * 2.0 ** -(frac + k_extra)
    rmax = big_l + e_l + 2.0**-frac
    d = 1
    while rmax ** (d + 1) / math.factorial(d + 1) * math.exp(rmax) > 2.0 ** -(frac + 1):
        d += 1
    rel = (
        (0 if exact_x else 2.0**-frac)
        + e_l
        + 2.0**-frac
        + rmax ** (d + 1) / math.factorial(d + 1) * math.exp(rmax)
        + (1 / (1 - rmax) + d) * 2.0**-frac
        + 2.0**-frac
        + 2.0 ** -(frac - 1)
    ) * 1.01
    koff = math.ceil((1 << int_bits) / ln2) + 2
    return {
        "frac": frac,
        "int_bits": int_bits,
        "t": t,
        "d": d,
        "k_extra": k_extra,
        "s_bits": s_bits,
        "koff": koff,
    }, rel


EXP_T = {"f32": 6, "bf16": 4, "f16": 5}


def _exp_int_bits(f: Float) -> int:
    lo = -math.log(2.0) * (f.bias - 1 + f.m + 1)  # below: rounds to +0
    hi = math.log(fp.max_value(f)) + 1
    return max(1, math.ceil(math.log2(max(-lo, hi) + 1)))


def _exp_core(s: Src, f: Float, x: str, wx: int, neg: str, prm: dict, xwid: int) -> tuple[str, str, int]:
    """exp(+-x / 2^frac) for `x: UInt(wx)` (`x < 2^(int_bits + frac)`) and the sign `neg` (1: negative), as
    `2^k * q / 2^frac`: returns (k: Int(xwid), q, width of q); q / 2^frac lies in [1, 2 + 2^-frac)."""
    frac, ib, t, d = prm["frac"], prm["int_bits"], prm["t"], prm["d"]
    kx, sb, koff = prm["k_extra"], prm["s_bits"], prm["koff"]
    cinv = consts.inv_ln2_fixed(t + sb, "floor")
    lc = consts.ln2_fixed(frac + kx - t, "floor")
    wn = ib + t + 3
    wc = cinv.bit_length()
    # n = floor(x / (ln2 / 2^t)) from an estimate that is at most two below it
    xc = s.tmp("xc")
    s(f"{xc}: UInt({wx + wc}) = {x}")
    s(f"{xc} = {xc} * {s.lit(cinv, wx + wc)}")
    n = s.tmp("n")
    s(f"{n}: UInt({wn}) = {xc} >> {frac + sb}")
    wr = wx + kx + 2
    lcv = s.lit(lc, wr)
    r = s.tmp("r")
    nl = s.tmp("nl")
    s(f"{r}: UInt({wr}) = {x}")
    s(f"{r} = {r} << {kx}")
    s(f"{nl}: UInt({wr}) = {n}")
    s(f"{nl} = {nl} * {lcv}")
    s(f"{r} = {r} - {nl}")
    for _ in range(2):
        with s.if_(f"{r} >= {lcv}"):
            s(f"{r} = {r} - {lcv}")
            s(f"{n} = {n} + 1")
    # negative x: n -> -(n + 1), r -> L - r (unless r = 0); k = floor(n / 2^t), j = n mod 2^t
    noff = s.tmp("noff")
    s(f"{noff}: UInt({clog2(koff << (t + 1))}) = {koff << t}")
    with s.if_(f"{neg} == 1"):
        with s.if_(f"{r} != 0"):
            s(f"{r} = {lcv} - {r}")
            s(f"{n} = {n} + 1")
        s(f"{noff} = {noff} - {n}")
    with s.else_():
        s(f"{noff} = {noff} + {n}")
    k = s.tmp("k")
    s(f"{k}: Int({xwid}) = {noff} >> {t}")
    s(f"{k} = {k} - {koff}")
    j = s.tmp("j")
    s(f"{j}: UInt({t + 1}) = {noff} & {(1 << t) - 1}")
    # P(r) = sum r^i / i!, Horner in fixed point (frac bits)
    wrr = frac - t + 2
    rr = s.tmp("rr")
    s(f"{rr}: UInt({wrr}) = {r} >> {kx}")
    wa = frac + 2
    acc = s.tmp("acc")
    s(f"{acc}: UInt({wa}) = {s.lit(consts.to_fixed(consts.Fraction(1, math.factorial(d)), frac), wa)}")
    wp = wrr + wa
    for i in range(d - 1, -1, -1):
        pr = s.tmp("hp")
        s(f"{pr}: UInt({wp}) = {rr}")
        s(f"{pr} = {pr} * {acc}")
        c = consts.to_fixed(consts.Fraction(1, math.factorial(i)), frac)
        s(f"{acc} = ({pr} >> {frac}) + {s.lit(c, wa)}")
    tab = consts.exp2_table(t, frac)
    tv = s.tmp("tv")
    s.rom(j, tab, frac + 1, tv)
    wq = 2 * frac + 4
    q = s.tmp("q")
    s(f"{q}: UInt({wq}) = {tv}")
    s(f"{q} = ({q} * {acc}) >> {frac}")
    return k, q, wq


def _exp_design(f: Float) -> Design:
    p = _p(f)
    prm, rel = _exp_params(f, p + G + 4, _exp_int_bits(f), EXP_T[f.name], exact_x=False)
    return Design("exponential", f.name, "faithful", 0.5 + _ulps(f, rel), prm)


def _head(s: Src, f: Float, xwid: int) -> None:
    """Decode `a`, normalize its significand: `an` (p bits, top bit set unless zero), `ex` its unbiased
    exponent (`a = an / 2^(p-1) * 2^ex`)."""
    decode(s, f, "a", "a", xwid)
    normalize(s, "a_m", f.m + 1, "an", "lza")
    s(f"ex: Int({xwid}) = a_x - lza - {f.bias}")


def _fixed_from(s: Src, f: Float, frac: int, width: int, out: str) -> None:
    """`out: UInt(width)` = |a| as fixed point with `frac` fraction bits, truncated (`an`, `ex` from `_head`;
    the caller makes sure it fits)."""
    p = _p(f)
    s(f"{out}: UInt({width}) = an")
    sh = s.tmp("fsh")
    s(f"{sh}: Int({xw(f) + 2}) = ex + {frac - (p - 1)}")
    with s.if_(f"{sh} >= 0"):
        with s.if_(f"{sh} < {width}"):
            s(f"{out} = {out} << {sh}")
    with s.else_():
        nsh = s.tmp("fns")
        s(f"{nsh}: Int({xw(f) + 2}) = 0 - {sh}")
        with s.if_(f"{nsh} < {width}"):
            s(f"{out} = {out} >> {nsh}")
        with s.else_():
            s(f"{out} = 0")


def _pack_q(s: Src, f: Float, sign: str, k: str, q: str, wq: int, frac: int, out: str, xwid: int) -> None:
    """Round `2^k * q / 2^frac` (q / 2^frac in [1, 4)) into `out`, keeping p + G + 4 bits (at most the
    available ones) and a sticky bit."""
    pw = min(_p(f) + G + 4, frac + 1)
    pp = s.tmp("pp")
    s(f"{pp}: UInt({pw}) = 0")
    be = s.tmp("be")
    s(f"{be}: Int({xwid}) = {k} + {f.bias}")
    lo = s.tmp("lo")
    s(f"{lo}: UInt({wq}) = 0")
    with s.if_(f"({q} >> {frac + 1}) != 0"):
        s(f"{pp} = {q} >> {frac + 1 - (pw - 1)}")
        s(f"{lo} = {q} & {s.lit((1 << (frac + 2 - pw)) - 1, wq)}")
        s(f"{be} = {be} + 1")
    with s.else_():
        s(f"{pp} = {q} >> {frac - (pw - 1)}")
        s(f"{lo} = {q} & {s.lit((1 << (frac + 1 - pw)) - 1, wq)}")
    with s.if_(f"{lo} != 0"):
        s(f"{pp} = {pp} | 1")
    round_pack(s, f, sign, be, pp, pw, out, xwid)


def exponential(name: str, f: Float) -> str:
    dz = _exp_design(f)
    prm = dz.params
    w, xwid = f.bits, xw(f) + 2
    frac, ib = prm["frac"], prm["int_bits"]
    s = Src(name, [("a", w)], w, f"exponential ({f.name})")
    _head(s, f, xwid)
    s(f"res: UInt({w}) = 0")
    wx = ib + frac + 1
    _fixed_from(s, f, frac, wx, "xf")
    k, q, wq = _exp_core(s, f, "xf", wx, "a_s", prm, xwid)
    _pack_q(s, f, "0", k, q, wq, frac, "res", xwid)
    with s.if_(f"ex >= {ib}"):
        with s.if_("a_s == 1"):
            s("res = 0")
        with s.else_():
            s(f"res = {f.inf}")
    with s.if_("a_nan == 1"):
        s(f"res = {f.nan}")
    return _ret(s, "res")


_NATIVE = re.compile(r":\s*(U?Int)\((8|16|32|64)\)(\s*=)")


def _ret(s: Src, out: str) -> str:
    """The module, every scalar variable of 8, 16, 32 or 64 bits widened by one bit: Allo emits those as C++
    `uint32_t` etc., which xlscc cannot assign from a value wider than 64 bits (Allo widens a product to the
    sum of its operand widths); other widths are `ac_int`, which converts between any widths."""
    s(f"return {out}")
    return _NATIVE.sub(lambda m: f": {m.group(1)}({int(m.group(2)) + 1}){m.group(3)}", s.module())


# ------------------------------------------------------------------------------------------- log, log1p


LOG_T = {"f32": 7, "bf16": 4, "f16": 5}


def _q_coeffs(d: int, frac: int) -> list[int]:
    """Coefficients 1/(i+1) of Q(r) = log1p(r) / r, i = 0..d, at `frac` bits."""
    return [consts.to_fixed(consts.Fraction(1, i + 1), frac) for i in range(d + 1)]


def _log_params(f: Float, t: int, wm: int, extra: int = 0, exact_m: bool = False) -> tuple[dict, float]:
    """`extra`: fraction bits beyond what a result of the format needs (`power` takes the log with more);
    `exact_m`: a significand wider than the format's is exact (not a cut 1 + x)."""
    p = _p(f)
    fc = t + 5
    umax = 2.0 ** -(t + 1) * (1 + 2.0 ** -3)
    frac = p + G + 4 + extra
    d = 1
    while umax**d / (d + 1) / (1 - umax) > 2.0 ** -(frac - 1):
        d += 1
    # Q's relative error: Taylor tail, coefficient rounding, d truncations
    rel_q = (umax**d / (d + 1) / (1 - umax) + (d + 2) * 2.0**-frac) * 1.01
    fs = p + G + t + 5 + extra
    # general path: absolute errors over |result| >= 2^-(t+2) (bounded below in the docs)
    abs_err = (
        2.0 ** -(fs + 1)  # L_i
        + 160 * 2.0 ** -(fs + 8)  # |E| * ln2
        + umax * rel_q  # u * Q
        + 2.0**-fs  # u * Q truncated to fs bits
        + (2.0 ** -(wm - 1) if wm > p and not exact_m else 0.0)  # 1 + x truncated to wm bits (log_plus_one)
    )
    rel = max(rel_q + 2.0 ** -(frac - 1), abs_err / 2.0 ** -(t + 2) * 1.01)
    return {"t": t, "fc": fc, "frac": frac, "fs": fs, "d": d, "wm": wm, "int_bits": 8}, rel


def _log_core(s: Src, f: Float, am: str, wm: int, ex: str, prm: dict, xwid: int) -> list[tuple[str, str, str, int, str]]:
    """log(am / 2^(wm-1) * 2^ex) for a normalized `am: UInt(wm)`. Returns the two candidates for
    `_pack_choice`: the pure path (|x - 1| small: r * Q(r) alone, relative accuracy) and the general path
    (ex * ln2 + L_i + r * Q(r))."""
    t, fc, frac, fs, d, ib = prm["t"], prm["fc"], prm["frac"], prm["fs"], prm["d"], prm["int_bits"]
    cs, ls = consts.log_tables(t, fc, fs)
    top = s.tmp("top")
    s(f"{top}: UInt({t + 2}) = ({am} >> {wm - 2 - t}) & {(1 << (t + 1)) - 1}")
    s(f"{top} = ({top} + 1) >> 1")
    ci = s.tmp("ci")
    li = s.tmp("li")
    s.rom(top, cs, fc + 1, ci)
    s.rom(top, ls, fs + 1, li)
    fu = wm - 1 + fc
    wpr = wm + fc + 2
    pr = s.tmp("pr")
    s(f"{pr}: UInt({wpr}) = {am}")
    s(f"{pr} = {pr} * {ci}")
    one = s.lit(1 << fu, wpr)
    u = s.tmp("u")
    rs = s.tmp("rs")
    s(f"{u}: UInt({wpr}) = 0")
    s(f"{rs}: UInt(1) = 0")
    with s.if_(f"{pr} >= {one}"):
        s(f"{u} = {pr} - {one}")
    with s.else_():
        s(f"{u} = {one} - {pr}")
        s(f"{rs} = 1")
    # Q(r) = sum (-r)^i / (i+1): Horner on u = |r| with the sign of each step from rs
    wur = frac - t + 2
    ur = s.tmp("ur")
    if fu >= frac:
        s(f"{ur}: UInt({wur}) = {u} >> {fu - frac}")
    else:
        s(f"{ur}: UInt({wur}) = {u}")
        s(f"{ur} = {ur} << {frac - fu}")
    wa = frac + 2
    co = _q_coeffs(d, frac)
    acc = s.tmp("qa")
    s(f"{acc}: UInt({wa}) = {s.lit(co[d], wa)}")
    for i in range(d - 1, -1, -1):
        hp = s.tmp("hp")
        s(f"{hp}: UInt({wur + wa}) = {ur}")
        s(f"{hp} = ({hp} * {acc}) >> {frac}")
        with s.if_(f"{rs} == 1"):
            s(f"{acc} = {s.lit(co[i], wa)} + {hp}")
        with s.else_():
            s(f"{acc} = {s.lit(co[i], wa)} - {hp}")
    # u * Q, exact product (fu + frac fraction bits)
    wuq = (fu - t + 2) + wa
    uq = s.tmp("uq")
    s(f"{uq}: UInt({wuq}) = {u}")
    s(f"{uq} = {uq} * {acc}")
    # pure path
    pn, lzp = s.tmp("pn"), s.tmp("lzp")
    normalize(s, uq, wuq, pn, lzp)
    bp = s.tmp("bp")
    s(f"{bp}: Int({xwid}) = {f.bias + wuq - 1 - fu - frac} - {lzp}")
    # general path: A = ex * ln2 + L_i (sign-magnitude), B = u * Q at fs bits, S = A +- B
    ws = ib + fs + 2
    ea, asn = s.tmp("ea"), s.tmp("asn")
    s(f"{ea}: Int({xwid}) = {ex}")
    s(f"{asn}: UInt(1) = 0")
    with s.if_(f"{ex} < 0"):
        s(f"{ea} = 0 - {ex}")
        s(f"{asn} = 1")
    eu = s.tmp("eu")
    s(f"{eu}: UInt({ws}) = {ea}")
    s(f"{eu} = {eu} * {s.lit(consts.ln2_fixed(fs), ws)}")
    lv = s.tmp("lv")
    s(f"{lv}: UInt({ws}) = {li}")
    am_ = s.tmp("am")
    s(f"{am_}: UInt({ws}) = 0")
    with s.if_(f"{asn} == 0"):
        s(f"{am_} = {eu} + {lv}")
    with s.elif_(f"{lv} >= {eu}"):
        s(f"{am_} = {lv} - {eu}")
        s(f"{asn} = 0")
    with s.else_():
        s(f"{am_} = {eu} - {lv}")
    bm = s.tmp("bm")
    s(f"{bm}: UInt({ws}) = {uq} >> {fu + frac - fs}")
    sm, ss = s.tmp("sm"), s.tmp("ss")
    s(f"{sm}: UInt({ws}) = 0")
    s(f"{ss}: UInt(1) = {asn}")
    with s.if_(f"{asn} == {rs}"):
        s(f"{sm} = {am_} + {bm}")
    with s.elif_(f"{am_} >= {bm}"):
        s(f"{sm} = {am_} - {bm}")
    with s.else_():
        s(f"{sm} = {bm} - {am_}")
        s(f"{ss} = {rs}")
    gn, lzg = s.tmp("gn"), s.tmp("lzg")
    normalize(s, sm, ws, gn, lzg)
    bg = s.tmp("bg")
    s(f"{bg}: Int({xwid}) = {f.bias + ws - 1 - fs} - {lzg}")
    pure = s.tmp("pure")
    s(f"{pure}: UInt(1) = 0")
    with s.if_(f"{ex} == 0 and {top} == 0"):
        s(f"{pure} = 1")
    with s.if_(f"({ex} + 1) == 0 and {top} == {1 << t}"):
        s(f"{pure} = 1")
    return [(pure, rs, pn, wuq, bp), ("1", ss, gn, ws, bg)]


def _pack_choice(s: Src, f: Float, cands: list[tuple[str, str, str, int, str]], out: str, xwid: int) -> None:
    """Round the first candidate whose condition holds: (condition, sign, normalized value, its width,
    biased exponent of its top bit). Each is cut to p + G + 4 bits with a sticky bit."""
    p = _p(f)
    pw = p + G + 6
    pp, be, sg = s.tmp("pp"), s.tmp("be"), s.tmp("sg")
    s(f"{pp}: UInt({pw}) = 0")
    s(f"{be}: Int({xwid}) = 0")
    s(f"{sg}: UInt(1) = 0")
    first = True
    for cond, sign, val, wv, bexp in cands:
        ctx = s.if_(f"{cond} == 1") if first else (s.elif_(f"{cond} == 1") if cond != "1" else s.else_())
        first = False
        with ctx:
            s(f"{sg} = {sign}")
            s(f"{be} = {bexp}")
            if wv > pw:
                s(f"{pp} = {val} >> {wv - pw}")
                with s.if_(f"({val} & {s.lit((1 << (wv - pw)) - 1, wv)}) != 0"):
                    s(f"{pp} = {pp} | 1")
            else:
                s(f"{pp} = {val}")
                s(f"{pp} = {pp} << {pw - wv}")
    round_pack(s, f, sg, be, pp, pw, out, xwid)


def _log_design(f: Float, op: str) -> Design:
    t = LOG_T[f.name]
    p = _p(f)
    wm = p if op == "log" else p + G + t + 6
    prm, rel = _log_params(f, t, wm)
    return Design(op, f.name, "faithful", 0.5 + _ulps(f, rel), prm)


def log(name: str, f: Float) -> str:
    dz = _log_design(f, "log")
    w, xwid = f.bits, xw(f) + 2
    s = Src(name, [("a", w)], w, f"log ({f.name})")
    _head(s, f, xwid)
    s(f"res: UInt({w}) = 0")
    cands = _log_core(s, f, "an", _p(f), "ex", dz.params, xwid)
    _pack_choice(s, f, cands, "res", xwid)
    with s.if_("a_nan == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_zero == 1"):
        s(f"res = {f.inf | (1 << (w - 1))}")
    with s.elif_("a_s == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_inf == 1"):
        s(f"res = {f.inf}")
    return _ret(s, "res")


def log_plus_one(name: str, f: Float) -> str:
    dz = _log_design(f, "log_plus_one")
    prm = dz.params
    t, frac, d, wm = prm["t"], prm["frac"], prm["d"], prm["wm"]
    p, w, xwid = _p(f), f.bits, xw(f) + 2
    s = Src(name, [("a", w)], w, f"log_plus_one ({f.name})")
    _head(s, f, xwid)
    s(f"res: UInt({w}) = 0")
    # tiny |x| < 2^-(t+2): x * Q(x)
    wa = frac + 2
    wur = frac - t + 2
    s(f"tu: UInt({wur}) = an")
    s(f"tsh: Int({xwid}) = {p - 1 - frac} - ex")
    with s.if_("tsh >= 0"):
        with s.if_(f"tsh < {p}"):
            s("tu = an >> tsh")
        with s.else_():
            s("tu = 0")
    with s.else_():
        s("tln: Int({xwid}) = 0 - tsh".replace("{xwid}", str(xwid)))
        with s.if_(f"tln < {wur}"):
            s("tu = tu << tln")
    co = _q_coeffs(d, frac)
    s(f"ta: UInt({wa}) = {s.lit(co[d], wa)}")
    for i in range(d - 1, -1, -1):
        s(f"th{i}: UInt({wur + wa}) = tu")
        s(f"th{i} = (th{i} * ta) >> {frac}")
        with s.if_("a_s == 1"):
            s(f"ta = {s.lit(co[i], wa)} + th{i}")
        with s.else_():
            s(f"ta = {s.lit(co[i], wa)} - th{i}")
    wt = p + wa
    s(f"tp: UInt({wt}) = an")
    s("tp = tp * ta")
    normalize(s, "tp", wt, "tpn", "lzt")
    s(f"tbe: Int({xwid}) = ex + {f.bias + wt - 1 - (p - 1) - frac} - lzt")
    # u = 1 + x: exact in fixed point below 2^huge, x itself above (log(1+x) - log(x) < 2^-(p+G+3))
    huge = p + G + 3
    fx = t + p + 1
    wu = huge + 1 + fx + 1
    s(f"uf: UInt({wu}) = an")
    s(f"ush: Int({xwid}) = ex + {t + 2}")
    with s.if_("ush >= 0"):
        with s.if_(f"ush < {wu}"):
            s("uf = uf << ush")
    one = s.lit(1 << fx, wu)
    with s.if_("a_s == 1"):
        s(f"uf = {one} - uf")
    with s.else_():
        s(f"uf = {one} + uf")
    normalize(s, "uf", wu, "ufn", "lzu")
    s(f"um: UInt({wm}) = ufn >> {wu - wm}")
    s(f"ue: Int({xwid}) = {wu - 1 - fx} - lzu")
    with s.if_(f"ex >= {huge}"):
        s(f"um = an")
        s(f"um = um << {wm - p}")
        s("ue = ex")
    cands = _log_core(s, f, "um", wm, "ue", prm, xwid)
    s("tiny: UInt(1) = 0")
    with s.if_(f"(ex + {t + 2}) < 0"):
        s("tiny = 1")
    _pack_choice(s, f, [("tiny", "a_s", "tpn", wt, "tbe"), *cands], "res", xwid)
    neg_one = s.lit((1 << (w - 1)) | f.one, w + 1)
    with s.if_("a_nan == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_zero == 1"):
        s("res = a")
    with s.elif_(f"a == {neg_one}"):
        s(f"res = {f.inf | (1 << (w - 1))}")
    with s.elif_(f"a_s == 1 and (a & {f.mag_mask}) > {f.one}"):
        s(f"res = {f.nan}")
    with s.elif_("a_inf == 1"):
        s(f"res = {f.inf}")
    return _ret(s, "res")


# ------------------------------------------------------------------------------------------------- tanh


TANH_T = {"f32": 6, "bf16": 4, "f16": 4}


def _tanh_design(f: Float) -> Design:
    p = _p(f)
    small = (p + 1) // 2
    fn = p + G + small + 4
    sat = (p + 2) * math.log(2) / 2
    ib = max(1, math.ceil(math.log2(2 * sat + 1)))
    prm, rel_e = _exp_params(f, fn, ib, TANH_T[f.name], exact_x=True)
    # N = 1 - exp(-z) >= z (1 - z/2) >= 2^-(small-1) * 0.99; D = 2 - N in (1, 2); the quotient keeps
    # p + G + 2 bits with a sticky bit
    rel_n = rel_e / (2.0 ** -(small - 1) * 0.99) + 2.0 ** -(fn - small - 1)
    rel = (rel_n + rel_e + 2.0 ** -(fn - 1) + 2.0 ** -(p + G + 1)) * 1.01
    small_ulps = 2.0 ** -(2 * small) * 2.0**p / 3
    prm.update({"small": small, "sat": sat, "fn": fn})
    return Design("tanh", f.name, "faithful", max(0.5 + _ulps(f, rel), small_ulps), prm)


def tanh(name: str, f: Float) -> str:
    dz = _tanh_design(f)
    prm = dz.params
    fn, ib, small = prm["fn"], prm["int_bits"], prm["small"]
    p, w, xwid = _p(f), f.bits, xw(f) + 2
    s = Src(name, [("a", w)], w, f"tanh ({f.name})")
    _head(s, f, xwid)
    s(f"res: UInt({w}) = 0")
    # z = 2|a| exactly in fixed point (fn fraction bits): |a| >= 2^-small here, so the shift is left
    wx = ib + fn + 1
    s(f"zf: UInt({wx}) = an")
    s(f"zsh: Int({xwid}) = ex + {fn - (p - 1) + 1}")
    with s.if_(f"zsh >= 0 and zsh < {wx}"):
        s("zf = zf << zsh")
    k, q, wq = _exp_core(s, f, "zf", wx, "1", prm, xwid)
    # E = exp(-z) at fn bits, N = 1 - E, D = 2 - N
    s(f"nk: Int({xwid}) = 0 - {k}")
    s(f"ef: UInt({wq}) = 0")
    with s.if_(f"nk < {wq}"):
        s(f"ef = {q} >> nk")
    s(f"nf: UInt({fn + 2}) = {s.lit(1 << fn, fn + 2)} - ef")
    s(f"df: UInt({fn + 3}) = {s.lit(2 << fn, fn + 3)} - nf")
    normalize(s, "nf", fn + 2, "nn", "lzn")
    # nn has its top bit at fn + 1, df at fn: quotient nn / (2 df) in [1/2, 2) after the shift
    nq = p + G + 2
    s(f"qr: UInt({fn + 4}) = nn >> 1")
    s(f"qd: UInt({fn + 4}) = df")
    s(f"qq: UInt({nq}) = 0")
    with s.for_("i", nq):
        s("qq = qq << 1")
        with s.if_("qr >= qd"):
            s("qr = qr - qd")
            s("qq = qq | 1")
        s("qr = qr << 1")
    s(f"qp: UInt({nq + 1}) = qq")
    s("qp = qp << 1")
    with s.if_("qr != 0"):
        s("qp = qp | 1")
    s(f"qe: Int({xwid}) = {f.bias + 1} - lzn")
    with s.if_(f"(qp >> {nq}) == 0"):
        s("qp = qp << 1")
        s("qe = qe - 1")
    round_pack(s, f, "a_s", "qe", "qp", nq + 1, "res", xwid, denorm=False)
    sat_bits = int(fp.round_f64(f, np.array([prm["sat"]]))[0])
    while float(fp.to_f64(f, np.array([sat_bits], dtype=np.uint64))[0]) < prm["sat"]:
        sat_bits += 1
    s(f"sg: UInt({w}) = a_s")
    with s.if_(f"(ex + {small}) < 0"):
        s("res = a")
    with s.if_(f"(a & {f.mag_mask}) >= {sat_bits}"):
        s(f"res = (sg << {w - 1}) | {f.one}")
    with s.if_("a_nan == 1"):
        s(f"res = {f.nan}")
    return _ret(s, "res")


# ------------------------------------------------------------------------------------------------ rsqrt


def _rsqrt_design(f: Float) -> Design:
    return Design("rsqrt", f.name, "correctly_rounded", 0.5, {"n": _p(f) + 1})


def rsqrt(name: str, f: Float) -> str:
    """Correctly rounded 1/sqrt(a): with a = m * 2^(2e') (m in [1, 4)), D = floor(2^T / M) for the integer
    significand M = m * 2^fm and T = 2N + fm, then S = isqrt(D) = floor(2^N / sqrt(m)) (floor(sqrt(floor(y)))
    = floor(sqrt(y))); the result is S / 2^N * 2^-e', inexact exactly when either step leaves a remainder."""
    p, w, xwid = _p(f), f.bits, xw(f) + 2
    n = p + 1
    fm = p if p % 2 == 0 else p - 1  # even: m = M / 2^fm with M = an << (fm - (p - 1))
    tt = 2 * n + fm
    s = Src(name, [("a", w)], w, f"rsqrt ({f.name})")
    _head(s, f, xwid)
    s(f"res: UInt({w}) = 0")
    wm = fm + 3
    s(f"mi: UInt({wm}) = an")
    s(f"mi = mi << {fm - (p - 1)}")
    s(f"eh: Int({xwid}) = ex")
    with s.if_("(ex & 1) == 1"):
        s("mi = mi << 1")
        s("eh = ex - 1")
    # D = floor(2^tt / mi): restoring division, remainder < mi
    nq = tt - fm + 1
    s(f"dr: UInt({wm + 1}) = {1 << fm}")
    s(f"dd: UInt({wm + 1}) = mi")
    s(f"dq: UInt({nq + 1}) = 0")
    with s.for_("i", nq):
        s("dq = dq << 1")
        with s.if_("dr >= dd"):
            s("dr = dr - dd")
            s("dq = dq | 1")
        s("dr = dr << 1")
    # S = isqrt(D), D < 2^(2n+1): n + 1 root digits
    nd = n + 1
    s(f"sv: UInt({2 * nd}) = dq")
    s(f"srem: UInt({nd + 3}) = 0")
    s(f"sroot: UInt({nd + 1}) = 0")
    s(f"strial: UInt({nd + 3}) = 0")
    s(f"srr: UInt({nd + 3}) = 0")
    with s.for_("i", nd):
        s(f"srem = (srem << 2) | ((sv >> ({2 * (nd - 1)} - 2 * i)) & 3)")
        s("srr = sroot")
        s("strial = (srr << 2) | 1")
        s("sroot = sroot << 1")
        with s.if_("srem >= strial"):
            s("srem = srem - strial")
            s("sroot = sroot | 1")
    # value = sroot / 2^n * 2^(-eh/2), sroot in (2^(n-1), 2^n]
    pw = n + 2
    s(f"rp: UInt({pw}) = sroot")
    s("rp = rp << 1")
    with s.if_("dr != 0 or srem != 0"):
        s("rp = rp | 1")
    # -eh/2 without shifting a negative value: eh is even and eh + 1024 >= 0
    s(f"rh: Int({xwid}) = eh + 1024")
    s(f"re: Int({xwid}) = {f.bias + 512} - (rh >> 1)")
    with s.if_(f"(rp >> {pw - 1}) == 0"):
        s("rp = rp << 1")
        s("re = re - 1")
    round_pack(s, f, "0", "re", "rp", pw, "res", xwid, denorm=False)
    with s.if_("a_nan == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_zero == 1"):
        s(f"sz: UInt({w}) = a_s")
        s(f"res = (sz << {w - 1}) | {f.inf}")
    with s.elif_("a_s == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_inf == 1"):
        s("res = 0")
    return _ret(s, "res")


# ------------------------------------------------------------------------------------------ sine, cosine

#: The worst case of the trig argument reduction per format: (M, ex, K) with `x = M * 2^(ex-p+1)` the float
#: (ex >= -1) closest to a multiple of pi/2 and `||x * 2/pi|| > 2^-K` for every such float
#: (`consts.trig_worst_case`, an enumeration of every input; f32: 16367173 * 2^72, as in Muller, *Elementary
#: Functions*, ch. 11).
TRIG_WORST = {"f32": (16367173, 95, 30), "bf16": (189, 58, 14), "f16": (1420, 7, 17)}


def _trig_design(f: Float, op: str) -> Design:
    p = _p(f)
    _, _, kd = TRIG_WORST[f.name]
    fw = 2 * p + kd + G + 6  # window fraction bits: tail error < 2^(p - fw) <= 2^-(p+G+6) * 2^-kd
    wf = p + G + 8  # the reduced fraction, normalized
    wpi = p + G + 8  # pi/2
    wr = p + G + 7  # r, normalized
    fr = p + G + 6  # r as fixed point (for z = r^2)
    frac = p + G + 6  # the polynomial
    zmax = (math.pi / 4) ** 2 * 1.001
    d = 1
    while zmax ** (d + 1) / math.factorial(2 * d + 2) > 2.0 ** -(frac + 1):
        d += 1
    rel_r = 2.0 ** (p - fw + kd) + 2.0 ** -(wf - 1) + 2.0 ** -(wpi - 1) + 2.0 ** -(wr - 1)
    dz = 2.6 * 2.0**-fr + 1.3 * rel_r
    horner = d * 2.0**-frac + (d + 1) * 2.0 ** -(frac + 1)
    rem_s = zmax ** (d + 1) / math.factorial(2 * d + 3)
    rem_c = zmax ** (d + 1) / math.factorial(2 * d + 2)
    rel_s = rel_r + (dz / 6 + horner + rem_s) / 0.9
    rel_c = (dz / 2 + horner + rem_c) / math.cos(math.pi / 4)
    rel = max(rel_s, rel_c) * 1.01
    prm = {"kd": kd, "fw": fw, "wf": wf, "wpi": wpi, "wr": wr, "fr": fr, "frac": frac, "d": d}
    return Design(op, f.name, "faithful", 0.5 + _ulps(f, rel), prm)


def _trig_check_worst(f: Float) -> None:
    """The recorded worst case is a real float at the distance its K claims (exact integers)."""
    p = _p(f)
    m, ex, kd = TRIG_WORST[f.name]
    e = ex - (p - 1)
    n = ex + 2 * kd + 64
    t = consts.two_over_pi_fixed(n)
    v = (m * t) << e if e >= 0 else (m * t) >> -e  # x * 2/pi * 2^n, truncated
    fr = v & ((1 << n) - 1)
    dist = min(fr, (1 << n) - fr) / 2.0**n
    assert 2.0 ** -kd < dist < 2.0 ** -(kd - 1), (f.name, dist)


def _trig_window(f: Float, fw: int) -> list[int]:
    """For ex = -1 .. emax (index ex + 1): the bits i of 2/pi = sum b_i 2^-i with E - 1 <= i <= E + fw,
    E = ex - (p - 1), as an integer (bits below E - 1 add multiples of 4 to x * 2/pi)."""
    p = _p(f)
    emax = f.bias
    n = emax + fw + 8
    t = consts.two_over_pi_fixed(n)
    out = []
    for ex in range(-1, emax + 1):
        e = ex - (p - 1)
        lo, hi = max(1, e - 1), e + fw
        bits = t >> (n - hi)  # b_1 .. b_hi
        out.append(bits & ((1 << (hi - lo + 1)) - 1))
    return out


def _trig(name: str, f: Float, op: str) -> str:
    dz = _trig_design(f, op)
    _trig_check_worst(f)
    prm = dz.params
    fw, wf, wpi, wr, fr, frac, d = (prm[k] for k in ("fw", "wf", "wpi", "wr", "fr", "frac", "d"))
    p, w, xwid = _p(f), f.bits, xw(f) + 2
    s = Src(name, [("a", w)], w, f"{op} ({f.name})")
    _head(s, f, xwid)
    s(f"res: UInt({w}) = 0")
    # |a| < 1/2: r = |a| exactly, quadrant 0; else Payne-Hanek on |a| = an * 2^(ex - p + 1)
    s(f"rr: UInt({wr}) = an")
    s(f"rr = rr << {wr - p}")
    s(f"re: Int({xwid}) = ex")
    s("rs: UInt(1) = 0")
    s("qd: UInt(2) = 0")
    s("red: UInt(1) = 0")
    s(f"wi: UInt({clog2(f.bias + 1)}) = 0")
    with s.if_(f"(ex + 1) >= 0 and ex <= {f.bias}"):
        s("red = 1")
        s("wi = ex + 1")
    win = _trig_window(f, fw)
    s.rom("wi", win, fw + 2, "wv")
    s(f"pw: UInt({p + fw + 2}) = an")
    s("pw = pw * wv")
    s(f"pm: UInt({fw + 3}) = pw & {s.lit((1 << (fw + 2)) - 1, p + fw + 2)}")
    s(f"pq: UInt(3) = pm >> {fw}")
    s(f"pf: UInt({fw + 1}) = pm & {s.lit((1 << fw) - 1, fw + 3)}")
    s(f"pg: UInt({fw + 1}) = pf")
    s("ps: UInt(1) = 0")
    with s.if_(f"pf >= {s.lit(1 << (fw - 1), fw + 1)}"):
        s(f"pg = {s.lit(1 << fw, fw + 1)} - pf")
        s("pq = pq + 1")
        s("ps = 1")
    s(f"pfw: UInt({fw}) = pg")
    normalize(s, "pfw", fw, "pn", "plz")
    s(f"fn: UInt({wf}) = pn >> {fw - wf}")
    pio2 = consts.pi_fixed(wpi - 2)  # pi/2 with wpi - 1 fraction bits, in [1, 2)
    s(f"rp: UInt({wf + wpi}) = fn")
    s(f"rp = rp * {s.lit(pio2, wpi)}")
    # rp / 2^(wf + wpi - 2) in [pi/2, pi): r = that * 2^(-plz - 1)
    with s.if_("red == 1"):
        s("rs = ps")
        s("qd = pq & 3")
        with s.if_(f"(rp >> {wf + wpi - 1}) != 0"):
            s(f"rr = rp >> {wf + wpi - wr}")
            s("re = 0 - plz")
        with s.else_():
            s(f"rr = rp >> {wf + wpi - 1 - wr}")
            s("re = 0 - plz")
            s("re = re - 1")
    # rf = |r| as fixed point (fr bits; re <= -1), z = r^2
    s(f"rf: UInt({fr}) = 0")
    s(f"rsh: Int({xwid}) = {wr - 1 - fr} - re")
    with s.if_(f"rsh < {wr}"):
        s("rf = rr >> rsh")
    s(f"zz: UInt({2 * fr}) = rf")
    s(f"zz = zz * rf")
    s(f"zf: UInt({fr}) = zz >> {fr}")
    # which function of r: sine's quadrant k = q, cosine's q + 1; k odd: cos(r), else sin(r); k >= 2: negate
    s(f"kq: UInt(3) = qd + {1 if op == 'cosine' else 0}")
    s("kq = kq & 3")
    s("uc: UInt(1) = kq & 1")
    # Horner on z, coefficients 1/(2i+1)! (sin(r)/r) or 1/(2i)! (cos(r)), alternating signs
    wa = frac + 2
    cs_ = [consts.to_fixed(consts.Fraction(1, math.factorial(2 * i + 1)), frac) for i in range(d + 1)]
    cc_ = [consts.to_fixed(consts.Fraction(1, math.factorial(2 * i)), frac) for i in range(d + 1)]
    s(f"ha: UInt({wa}) = {s.lit(cs_[d], wa)}")
    with s.if_("uc == 1"):
        s(f"ha = {s.lit(cc_[d], wa)}")
    for i in range(d - 1, -1, -1):
        s(f"hz{i}: UInt({fr + wa}) = zf")
        s(f"hz{i} = (hz{i} * ha) >> {fr}")
        s(f"hc{i}: UInt({wa}) = {s.lit(cs_[i], wa)}")
        with s.if_("uc == 1"):
            s(f"hc{i} = {s.lit(cc_[i], wa)}")
        s(f"ha = hc{i} - hz{i}")
    # sin: r * S (exact product, value rr * ha / 2^(wr-1+frac) * 2^re, in [0.9, 2) * 2^re);
    # cos: C = ha / 2^frac in [0.7, 1]
    wsn = wr + wa
    s(f"sv: UInt({wsn}) = rr")
    s("sv = sv * ha")
    s(f"se: Int({xwid}) = re + {f.bias}")
    with s.if_(f"(sv >> {wr + frac - 1}) == 0"):
        s("sv = sv << 1")
        s("se = se - 1")
    s(f"sv = sv << {wsn - (wr + frac)}")
    s(f"cv: UInt({wa}) = ha")
    s(f"ce: Int({xwid}) = {f.bias}")
    with s.if_(f"(cv >> {frac}) == 0"):
        s("cv = cv << 1")
        s("ce = ce - 1")
    s(f"cv = cv << {wa - frac - 1}")
    sgn = "kq >> 1" if op == "cosine" else "(kq >> 1) ^ a_s"
    s(f"sgs: UInt(1) = ({sgn}) ^ rs")
    s(f"sgc: UInt(1) = {sgn}")
    _pack_choice(s, f, [("uc", "sgc", "cv", wa, "ce"), ("1", "sgs", "sv", wsn, "se")], "res", xwid)
    with s.if_("a_nan == 1 or a_inf == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_zero == 1"):
        s(f"res = {f.one if op == 'cosine' else 'a'}")
    return _ret(s, "res")


def sine(name: str, f: Float) -> str:
    return _trig(name, f, "sine")


def cosine(name: str, f: Float) -> str:
    return _trig(name, f, "cosine")


# ------------------------------------------------------------------------------------------------ power

#: (log table bits, exp table bits) of `power` per format.
POW_T = {"f32": (8, 6), "bf16": (4, 4), "f16": (5, 5), "f8e4m3fn": (3, 3)}


def _pow_design(f: Float) -> Design:
    """pow(x, y) = exp(y * ln|x|): ln|x| from the log core with `ib + 3` extra bits, because an error rho
    (relative) in it moves z = y ln|x| by |z| rho < 2^ib rho (beyond |z| >= 2^ib the result overflows or
    underflows), and exp(z) by that much relative."""
    p = _p(f)
    tl, te = POW_T[f.name]
    ib = _exp_int_bits(f)
    wm = max(p, tl + 2)  # the log core reads t + 2 bits of the significand
    prm_l, rho = _log_params(f, tl, wm, ib + 3, exact_m=True)
    fe = p + G + 6
    prm_e, rel_e = _exp_params(f, fe, ib, te, exact_x=False)
    wl = p + G + ib + 10  # ln|x| kept to wl bits (truncated)
    rel = (rel_e + 2.0**ib * (rho + 2.0 ** -(wl - 1))) * 1.01
    return Design("power", f.name, "faithful", 0.5 + _ulps(f, rel), {"log": prm_l, "exp": prm_e, "wl": wl, "ib": ib, "fe": fe})


def _select(s: Src, cands: list[tuple[str, str, str, int, str]], wl: int, xwid: int) -> tuple[str, str, str]:
    """The first candidate whose condition holds, as (sign, value cut to `wl` bits, biased exponent)."""
    v, be, sg = s.tmp("sv"), s.tmp("sbe"), s.tmp("ssg")
    s(f"{v}: UInt({wl}) = 0")
    s(f"{be}: Int({xwid}) = 0")
    s(f"{sg}: UInt(1) = 0")
    first = True
    for cond, sign, val, wv, bexp in cands:
        ctx = s.if_(f"{cond} == 1") if first else (s.elif_(f"{cond} == 1") if cond != "1" else s.else_())
        first = False
        with ctx:
            s(f"{sg} = {sign}")
            s(f"{be} = {bexp}")
            if wv > wl:
                s(f"{v} = {val} >> {wv - wl}")
            else:
                s(f"{v} = {val}")
                s(f"{v} = {v} << {wl - wv}")
    return sg, v, be


def _y_parity(s: Src, f: Float, xwid: int) -> None:
    """`yint`, `yodd` of the finite nonzero operand b (`bn` normalized, `ey` its exponent)."""
    p = _p(f)
    s("yint: UInt(1) = 0")
    s("yodd: UInt(1) = 0")
    with s.if_(f"ey >= {p - 1}"):
        s("yint = 1")
        with s.if_(f"ey == {p - 1}"):
            s("yodd = bn & 1")
    with s.elif_("ey >= 0"):
        s(f"ysh: Int({xwid}) = {p - 1} - ey")
        s(f"yhi: UInt({p}) = bn >> ysh")
        s(f"ylo: UInt({p}) = yhi << ysh")
        with s.if_("ylo == bn"):
            s("yint = 1")
            s("yodd = yhi & 1")


def power(name: str, f: Float) -> str:
    dz = _pow_design(f)
    prm = dz.params
    pl, pe, wl, ib, fe = prm["log"], prm["exp"], prm["wl"], prm["ib"], prm["fe"]
    p, w, xwid = _p(f), f.bits, xw(f) + 2
    s = Src(name, [("a", w), ("b", w)], w, f"power ({f.name}): a^b")
    _head(s, f, xwid)
    decode(s, f, "b", "b", xwid)
    normalize(s, "b_m", p, "bn", "lzb")
    s(f"ey: Int({xwid}) = b_x - lzb - {f.bias}")
    s(f"res: UInt({w}) = 0")
    # ln|a|, then z = b * ln|a| = zp / 2^(p + wl - 2) * 2^ze
    wm = pl["wm"]
    s(f"am: UInt({wm}) = an")
    s(f"am = am << {wm - p}")
    cands = _log_core(s, f, "am", wm, "ex", pl, xwid)
    ls, lv, lb = _select(s, cands, wl, xwid)
    s(f"zp: UInt({p + wl}) = bn")
    s(f"zp = zp * {lv}")
    s(f"ze: Int({xwid}) = ey + {lb} - {f.bias}")
    s(f"zs: UInt(1) = b_s ^ {ls}")
    # z as fixed point (fe fraction bits); |z| >= 2^ib overflows or underflows
    wz = ib + fe + 3  # a spare top bit for the comparison
    s(f"zw: UInt({wz}) = 0")
    s("zbig: UInt(1) = 0")
    with s.if_(f"ze > {ib}"):
        s("zbig = 1")
    with s.else_():
        s(f"zsh: Int({xwid}) = {p + wl - 2 - fe} - ze")
        with s.if_(f"zsh < {p + wl}"):
            s("zw = zp >> zsh")
        with s.if_(f"zw >= {s.lit(1 << (ib + fe), wz)}"):
            s("zbig = 1")
    wx = ib + fe + 1
    s(f"zf: UInt({wx}) = zw")
    with s.if_(f"(a & {f.mag_mask}) == {f.one}"):
        s("zbig = 0")  # ln|a| = 0: the exponent of the zero logarithm means nothing
        s("zf = 0")
    with s.if_("zbig == 1"):
        s("zf = 0")
    k, q, wq = _exp_core(s, f, "zf", wx, "zs", pe, xwid)
    _y_parity(s, f, xwid)
    s("rsg: UInt(1) = a_s & yodd")
    _pack_q(s, f, "rsg", k, q, wq, fe, "res", xwid)
    s(f"sgw: UInt({w}) = rsg")
    inf_s = f"{f.nan}" if f.finite_nan else f"(sgw << {w - 1}) | {f.inf}"
    with s.if_("zbig == 1"):
        with s.if_("zs == 0"):
            s(f"res = {inf_s}")
        with s.else_():
            s(f"res = sgw << {w - 1}")
    # special values, C11 Annex F.10.4.4, in order of precedence
    pinf = f"{f.inf}"
    s(f"xs: UInt({w}) = a_s")
    s("xsodd: UInt(1) = a_s & yodd")
    s(f"xo: UInt({w}) = xsodd")
    s(f"amag: UInt({w}) = a & {f.mag_mask}")
    s("alt: UInt(1) = 0")
    with s.if_(f"amag < {f.one}"):
        s("alt = 1")
    with s.if_("b_zero == 1"):
        s(f"res = {f.one}")
    with s.elif_(f"a == {f.one}"):
        s(f"res = {f.one}")
    with s.elif_("a_nan == 1 or b_nan == 1"):
        s(f"res = {f.nan}")
    with s.elif_("a_zero == 1"):
        with s.if_("b_s == 1"):
            s(f"res = {f.nan if f.finite_nan else f'(xo << {w - 1}) | {pinf}'}")
        with s.else_():
            s(f"res = xo << {w - 1}")
    with s.elif_("b_inf == 1"):
        with s.if_(f"amag == {f.one}"):
            s(f"res = {f.one}")
        with s.elif_("alt == b_s"):
            s(f"res = {pinf}")
        with s.else_():
            s("res = 0")
    with s.elif_("a_inf == 1"):
        with s.if_("b_s == 1"):
            s(f"res = xo << {w - 1}")
        with s.else_():
            s(f"res = (xo << {w - 1}) | {pinf}")
    with s.elif_("a_s == 1 and yint == 0"):
        s(f"res = {f.nan}")
    return _ret(s, "res")


# --------------------------------------------------------------------------------------------------- ROM


def rom_values(op: str, f: Float) -> list[int]:
    """The correctly rounded result of every input pattern of `f`; each float64 value is checked to be far
    enough from a rounding boundary that its own error cannot change the rounding."""
    from .reference import approx_f64

    bits = np.arange(1 << f.bits, dtype=np.uint64)
    v = approx_f64(op, fp.to_f64(f, bits))
    out = fp.round_f64(f, v)
    with np.errstate(all="ignore"):
        u = fp.ulp(f, v)
        frac = np.abs(v) / u - np.floor(np.abs(v) / u)
        near = np.isfinite(v) & (np.abs(frac - 0.5) < 2.0**-30) & (np.abs(v) <= fp.max_value(f) * 2)
    assert not near.any(), f"{op}@{f.name}: a value within 2^-30 ulp of a rounding boundary"
    return [int(x) for x in out]


def rom(name: str, f: Float, op: str) -> str:
    w = f.bits
    s = Src(name, [("a", w)], w, f"{op} ({f.name}), a table of the correctly rounded results")
    s.rom("a", rom_values(op, f), w, "res")
    return _ret(s, "res")


# ------------------------------------------------------------------------------------------------ public


def design(op: str, f: Float) -> Design:
    if f.bits <= ROM_BITS and op in UNARY:
        return Design(op, f.name, "correctly_rounded", 0.5, {"rom": 1 << f.bits})
    match op:
        case "exponential":
            return _exp_design(f)
        case "log" | "log_plus_one":
            return _log_design(f, op)
        case "tanh":
            return _tanh_design(f)
        case "rsqrt":
            return _rsqrt_design(f)
        case "sine" | "cosine":
            return _trig_design(f, op)
        case "power":
            return _pow_design(f)
    raise ValueError(op)


def source(op: str, f: Float, name: str) -> str:
    if f.bits <= ROM_BITS and op in UNARY:
        return rom(name, f, op)
    gen = {"exponential": exponential, "log": log, "log_plus_one": log_plus_one, "tanh": tanh, "rsqrt": rsqrt,
           "sine": sine, "cosine": cosine, "power": power}
    return gen[op](name, f)


__all__ = ["BINARY", "G", "OPS", "ROM_BITS", "UNARY", "Design", "design", "rom_values", "source"]
