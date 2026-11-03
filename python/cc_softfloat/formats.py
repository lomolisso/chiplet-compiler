"""Float and integer types of the library, named as `cc_smt` names them (`f32`, `bf16`, `f8e4m3fn`, `i8`, ...)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Float:
    """A binary float format: 1 sign bit, `e` exponent bits, `m` trailing significand bits.

    `finite_nan` formats (OCP E4M3FN) have no infinities: the top binade is ordinary except the all-ones
    significand, which is NaN. Every other format follows IEEE 754 for zeros, subnormals, infinities and NaN.
    """

    name: str
    e: int
    m: int
    finite_nan: bool = False

    @property
    def bits(self) -> int:
        return 1 + self.e + self.m

    @property
    def bias(self) -> int:
        return (1 << (self.e - 1)) - 1

    @property
    def emax_field(self) -> int:
        """The all-ones exponent field."""
        return (1 << self.e) - 1

    @property
    def mag_mask(self) -> int:
        return (1 << (self.bits - 1)) - 1

    @property
    def man_mask(self) -> int:
        return (1 << self.m) - 1

    @property
    def nan(self) -> int:
        """The NaN the library produces: positive, quiet (IEEE) or the only NaN magnitude (E4M3FN)."""
        if self.finite_nan:
            return (self.emax_field << self.m) | self.man_mask
        return (self.emax_field << self.m) | (1 << (self.m - 1))

    @property
    def inf(self) -> int:
        """+infinity; for `finite_nan` formats the value overflow gives (NaN, non-saturating)."""
        return self.nan if self.finite_nan else self.emax_field << self.m

    @property
    def max_finite(self) -> int:
        if self.finite_nan:
            return (self.emax_field << self.m) | (self.man_mask - 1)
        return ((self.emax_field - 1) << self.m) | self.man_mask

    @property
    def one(self) -> int:
        return self.bias << self.m


@dataclass(frozen=True)
class Int:
    """A two's-complement (`signed`) or unsigned integer of `w` bits; `w == 1` unsigned is `bool`."""

    name: str
    w: int
    signed: bool

    @property
    def bits(self) -> int:
        return self.w

    @property
    def lo(self) -> int:
        return -(1 << (self.w - 1)) if self.signed else 0

    @property
    def hi(self) -> int:
        return (1 << (self.w - 1)) - 1 if self.signed else (1 << self.w) - 1


FLOATS: dict[str, Float] = {
    f.name: f
    for f in (
        Float("f32", 8, 23),
        Float("f16", 5, 10),
        Float("bf16", 8, 7),
        Float("f8e4m3fn", 4, 3, finite_nan=True),
        Float("f8e5m2", 5, 2),
    )
}

BOOL = Int("i1", 1, False)


def parse(name: str) -> Float | Int:
    """A type by its `cc_smt` name: a float format, `i1` (bool), `i<w>` or `ui<w>`."""
    if name in FLOATS:
        return FLOATS[name]
    if name == "i1":
        return BOOL
    if name.startswith("ui") and name[2:].isdigit():
        return Int(name, int(name[2:]), False)
    if name.startswith("i") and name[1:].isdigit():
        return Int(name, int(name[1:]), True)
    raise ValueError(f"unknown type {name!r}")
