"""Building blocks of the generated Allo functions.

Every op is one flat Allo function over `UInt(n)` bit patterns, written as Python source text by a generator
(`cc_softfloat.ops`) from the blocks here. Rules the blocks keep, so that Allo's `llvm` simulation and its
XLS C++ (xlscc) agree and the hardware stays small:

- bits are read with shifts and masks (Allo's slice syntax emits Vivado `ap_int` code that xlscc rejects);
- integer literals stay below 2^31 (Allo types a literal as a 32-bit signed integer: larger ones compare
  wrongly), so wider constants are built in a typed variable (`Src.lit`); `~` is a no-op in Allo, so bitwise
  NOT is an XOR with all ones (`Src.inv`);
- an ordered comparison (`<`, `>=`, ...) of two `UInt(n)` variables needs their top bit clear: Allo's `llvm`
  lowering compares them as signed (the XLS C++ compares unsigned), so such variables get a spare bit;
- no 64-bit types (`UInt(64) + 1` becomes a 66-bit value xlscc cannot convert back), minimal widths instead;
- every variable shift amount is guarded below the operand width (LLVM makes a larger shift poison, XLS 0);
- leading-zero counts and normalizing shifts are log-depth (one conditional shift per power of two);
- an operation rounds once, in `round_pack`, with guard and sticky bits (round to nearest, ties to even).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .formats import Float


def clog2(n: int) -> int:
    """Bits to hold the values 0..n."""
    return max(1, n.bit_length())


def xw(f: Float) -> int:
    """Width of the signed exponent arithmetic of a format (sums of two exponents, shift counts, offsets)."""
    return max(8, f.e + 5)


class Src:
    """Python source of one Allo function, built line by line."""

    def __init__(self, name: str, params: list[tuple[str, int]], ret: int, doc: str):
        self.name = name
        self.lines: list[str] = []
        self.depth = 1
        self.counter = 0
        sig = ", ".join(f"{p}: UInt({w})" for p, w in params)
        self.head = [f"def {name}({sig}) -> UInt({ret}):", f'    """{doc}"""']

    def __call__(self, line: str) -> None:
        self.lines.append("    " * self.depth + line)

    def tmp(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def lit(self, value: int, width: int) -> str:
        """The literal `value` (0 <= value < 2^width) as an expression of `width` bits: the number itself
        below 2^31, else a variable assembled from 31-bit pieces."""
        assert 0 <= value < (1 << width), (value, width)
        if value < (1 << 31):
            return str(value)
        name = self.tmp("lit")
        chunks = []
        v = value
        while v:
            chunks.append(v & ((1 << 31) - 1))
            v >>= 31
        chunks.reverse()
        self(f"{name}: UInt({width}) = {chunks[0]}")
        for c in chunks[1:]:
            self(f"{name} = ({name} << 31) | {c}")
        return name

    def inv(self, expr: str, width: int) -> str:
        """Bitwise NOT of a `width`-bit expression."""
        return f"({expr} ^ {self.lit((1 << width) - 1, width)})"

    @contextmanager
    def block(self, header: str) -> Iterator[None]:
        self(header)
        self.depth += 1
        yield
        self.depth -= 1

    def if_(self, cond: str):
        return self.block(f"if {cond}:")

    def elif_(self, cond: str):
        return self.block(f"elif {cond}:")

    def else_(self):
        return self.block("else:")

    def for_(self, var: str, n: int):
        return self.block(f"for {var} in range({n}):")

    def text(self) -> str:
        return "\n".join(self.head + self.lines) + "\n"


HEADER = "from allo.ir.types import Int, UInt\n\n\n"


def decode(s: Src, f: Float, v: str, p: str, xwidth: int) -> None:
    """Fields of the pattern `v`: `{p}_s` sign, `{p}_m` significand with the hidden bit (M+1 bits), `{p}_x`
    biased exponent (1 for subnormals, signed `xwidth` bits), flags `{p}_nan`, `{p}_inf`, `{p}_zero`."""
    m, w = f.m, f.bits
    s(f"{p}_s: UInt(1) = ({v} >> {w - 1}) & 1")
    s(f"{p}_ef: UInt({f.e}) = ({v} >> {m}) & {f.emax_field}")
    s(f"{p}_m: UInt({m + 1}) = {v} & {f.man_mask}")
    s(f"{p}_x: Int({xwidth}) = {p}_ef")
    s(f"{p}_nan: UInt(1) = 0")
    s(f"{p}_inf: UInt(1) = 0")
    s(f"{p}_zero: UInt(1) = 0")
    with s.if_(f"({v} & {f.mag_mask}) == 0"):
        s(f"{p}_zero = 1")
    if f.finite_nan:
        with s.if_(f"({v} & {f.mag_mask}) == {f.nan & f.mag_mask}"):
            s(f"{p}_nan = 1")
    else:
        with s.if_(f"{p}_ef == {f.emax_field}"):
            with s.if_(f"{p}_m != 0"):
                s(f"{p}_nan = 1")
            with s.else_():
                s(f"{p}_inf = 1")
    with s.if_(f"{p}_ef != 0"):
        s(f"{p}_m = {p}_m | {1 << m}")
    with s.else_():
        s(f"{p}_x = 1")


def normalize(s: Src, x: str, width: int, out: str, lz: str, cap: str | None = None) -> None:
    """`{out}` = `x` shifted left until its top bit (of `width`) is set, `{lz}` the shift. With `cap` (a signed
    expression >= 0), the shift stops at `cap` places: a sentinel bit `cap` places below the top limits the
    count, while the value itself is shifted by the same steps. Log-depth: one conditional shift per power of
    two. A zero `x` gives the largest shift reached."""
    lw = clog2(width)
    s(f"{out}: UInt({width}) = {x}")
    s(f"{lz}: UInt({lw}) = 0")
    probe = None
    if cap is not None:
        probe = s.tmp("nprobe")
        s(f"{probe}: UInt({width}) = {out}")
        with s.if_(f"({cap}) < {width}"):
            sh = s.tmp("ncap")
            one = s.tmp("none")
            s(f"{sh}: UInt({lw}) = {width - 1} - ({cap})")
            s(f"{one}: UInt({width}) = 1")
            s(f"{probe} = {probe} | ({one} << {sh})")
    k = 1 << (lw - 1)
    while k >= 1:
        if k < width:
            test = probe if probe is not None else out
            with s.if_(f"({test} >> {width - k}) == 0"):
                s(f"{out} = {out} << {k}")
                if probe is not None:
                    s(f"{probe} = {probe} << {k}")
                s(f"{lz} = {lz} + {k}")
        k //= 2


def shift_right_sticky(s: Src, x: str, width: int, amount: str, out: str) -> None:
    """`{out}` = `x >> amount` with the OR of the shifted-out bits in bit 0 (`amount` signed, >= 0)."""
    s(f"{out}: UInt({width}) = 0")
    with s.if_(f"{amount} < {width}"):
        q = s.tmp("srq")
        s(f"{q}: UInt({width}) = {x} >> {amount}")
        s(f"{out} = {q}")
        with s.if_(f"({q} << {amount}) != {x}"):
            s(f"{out} = {q} | 1")
    with s.else_():
        with s.if_(f"{x} != 0"):
            s(f"{out} = 1")


def round_pack(
    s: Src, f: Float, sign: str, be: str, p: str, pw: int, out: str, xwidth: int, denorm: bool = True
) -> None:
    """Round and pack a finite value into `{out}` (`UInt(bits)`).

    The value is `p / 2^(pw-1) * 2^(be - bias)`: `p` (`pw` bits, `pw >= m + 3`) is normalized with its top bit
    at `pw - 1`, or has leading zeros only when `be` is 1 (subnormal) or `p` is 0. Bits below the kept
    significand are guard and sticky (bit 0 may carry an OR of lower bits). With `denorm`, a `be` below 1 is
    first shifted into the subnormal position (sticky kept), so the value rounds once at subnormal precision.
    Overflow gives infinity (IEEE) or NaN (finite-NaN formats, non-saturating)."""
    m, w = f.m, f.bits
    d = pw - (m + 1)
    assert d >= 2, (pw, m)
    if denorm:
        with s.if_(f"{be} < 1"):
            amt = s.tmp("dsh")
            s(f"{amt}: Int({xwidth}) = 1 - {be}")
            shifted = s.tmp("dp")
            shift_right_sticky(s, p, pw, amt, shifted)
            s(f"{p} = {shifted}")
            s(f"{be} = 1")
    kept = s.tmp("kept")
    rest = s.tmp("rest")
    s(f"{kept}: UInt({m + 2}) = {p} >> {d}")
    s(f"{rest}: UInt({d}) = {p} & {(1 << d) - 1}")
    with s.if_(f"{rest} > {1 << (d - 1)}"):
        s(f"{kept} = {kept} + 1")
    with s.elif_(f"{rest} == {1 << (d - 1)}"):
        with s.if_(f"({kept} & 1) == 1"):
            s(f"{kept} = {kept} + 1")
    with s.if_(f"({kept} >> {m + 1}) == 1"):
        s(f"{kept} = {kept} >> 1")
        s(f"{be} = {be} + 1")
    with s.if_(f"({kept} >> {m}) == 0"):
        s(f"{be} = 0")
    over = s.tmp("over")
    s(f"{over}: UInt(1) = 0")
    if f.finite_nan:
        with s.if_(f"{be} > {f.emax_field}"):
            s(f"{over} = 1")
        with s.elif_(f"{be} == {f.emax_field}"):
            with s.if_(f"({kept} & {f.man_mask}) == {f.man_mask}"):
                s(f"{over} = 1")
    else:
        with s.if_(f"{be} >= {f.emax_field}"):
            s(f"{over} = 1")
    sw = s.tmp("sw")
    s(f"{sw}: UInt({w}) = {sign}")
    with s.if_(f"{over} == 1"):
        s(f"{out} = (({sw} << {w - 1}) | {f.inf})")
    with s.else_():
        bw = s.tmp("bw")
        kw = s.tmp("kw")
        s(f"{bw}: UInt({w}) = {be}")
        s(f"{kw}: UInt({w}) = {kept} & {f.man_mask}")
        s(f"{out} = ({sw} << {w - 1}) | ({bw} << {m}) | {kw}")
