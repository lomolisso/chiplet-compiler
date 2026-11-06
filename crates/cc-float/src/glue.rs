//! The glue wrappers: what each needs from cores and the Verilog that wraps them (design §4.12,
//! docs/framework/float.md).
//!
//! A wrapper is one module per op and format around zero, one or two core instances. The format rules
//! are design §6.4's: f32 and bf16 read subnormal operands as zero and flush results whose value rounded
//! with an unbounded exponent is below `2^emin`, so their cores run with one more exponent bit and the
//! glue flushes and narrows exactly; f16 and the 8-bit formats keep gradual underflow; E4M3FN runs on
//! f16 (arithmetic) or f32 (approximate operators) cores through an exact decoder and a round-to-nearest
//! encoder.

use cc_core::adapters::{CoreInstance, Rounding};
use cc_smt::float::{Arith, Flavour, Format};
use cc_smt::ops::Ty;
use serde::{Deserialize, Serialize};

use crate::FloatError;
use crate::compose::Recipe;
use crate::rtl::{Mutation, Rtl};

type Res<X> = Result<X, FloatError>;

/// The comparison directions (`compare` with `FLOAT`; `GT`/`GE` are `LT`/`LE` with exchanged operands).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Dir {
    Eq,
    Ne,
    Lt,
    Le,
}

impl Dir {
    pub fn name(self) -> &'static str {
        match self {
            Dir::Eq => "EQ",
            Dir::Ne => "NE",
            Dir::Lt => "LT",
            Dir::Le => "LE",
        }
    }
}

/// The sign-bit operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SignOp {
    Abs,
    Negate,
    Sign,
}

impl SignOp {
    pub fn name(self) -> &'static str {
        match self {
            SignOp::Abs => "abs",
            SignOp::Negate => "negate",
            SignOp::Sign => "sign",
        }
    }
}

/// One wrapper: an op at a format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Glue {
    /// add, subtract, multiply, divide, sqrt (correctly rounded cores).
    Arith { op: Arith, fmt: Format },
    /// An approximate operator (catalog name: `exponential`, `log`, ...) of `arity` operands on a
    /// core computing in `via`: the format itself (f16), the format with one more exponent bit (f32,
    /// bf16), or f32 (16-bit and 8-bit formats, then rounded once).
    Approx {
        op: String,
        arity: usize,
        fmt: Format,
        via: Format,
    },
    /// `convert` between float formats: exact widening (glue only), to another IEEE format (a convert
    /// core: narrowing, or between f16 and bf16), to E4M3FN (the encoder, saturating), to E5M2 (a
    /// convert core, saturated).
    Convert { from: Format, to: Format },
    /// IEEE 754-2019 `maximum` / `minimum` around a less-than core.
    MaxMin { max: bool, fmt: Format },
    /// `compare` (quiet predicates) from less-than cores.
    Compare { dir: Dir, fmt: Format },
    /// abs, negate, sign: no core.
    Sign { op: SignOp, fmt: Format },
    /// `convert` to an integer around a float-to-integer core (rounding toward zero, saturating), with
    /// NaN to 0.
    ToInt { fmt: Format, signed: bool, w: u32 },
    /// A composition of cores at `q` significand bits (`crate::recipe`), its special values as glue.
    Compose { recipe: Recipe, fmt: Format, q: u32 },
    /// `convert` from a `w`-bit integer around an integer-to-float core of `via` (the format itself, or
    /// for E4M3FN a format that holds every integer of the width exactly), then the FP8 encoder
    /// (saturating) for E4M3FN: one rounding.
    FromInt {
        fmt: Format,
        signed: bool,
        w: u32,
        via: Format,
    },
    /// `convert` between bool and a float: to the float 0 or 1; to bool `x != 0` (NaN is true).
    Bool { fmt: Format, to_float: bool },
    /// `is_finite`: neither NaN nor an infinity.
    IsFinite { fmt: Format },
    /// floor, ceil, round_nearest_even: below `2^(p-1)` in magnitude the value through a to-integer
    /// core of that rounding and an integer-to-float core; above it (and for NaN and infinities) the
    /// operand itself; a zero result keeps the operand's sign.
    Round { fmt: Format, rnd: Rounding },
    /// `reduce_precision` to `e` exponent and `m` significand bits on the bit pattern (the StableHLO
    /// definition: round to nearest even on the pattern, beyond the exponent range infinity, below it
    /// zero, NaN and infinities kept).
    ReducePrecision { fmt: Format, e: u32, m: u32 },
    /// `clamp(min, x, max)` = `minimum(maximum(x, min), max)` from two less-than cores.
    Clamp { fmt: Format },
}

/// What a core computes in a wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Role {
    /// The IEEE operation, round to nearest even (operands and result in `format`).
    Arith { op: String },
    /// An approximate operator (operands and result in `format`).
    Approx { op: String },
    /// Conversion of an operand of `from` to `format`, round to nearest even.
    Convert { from: String },
    /// `a0 < a1` (IEEE quiet less-than) on operands of `format`; a 1-bit result.
    Less,
    /// Conversion of an operand of `format` to a `w`-bit integer, toward zero, saturating at the
    /// bounds; the result for NaN is not used.
    ToInt { signed: bool, w: u32 },
    /// Conversion of an operand of `format` to a `w`-bit integer rounded per `rnd` (`Rounding::name`);
    /// only operands below `2^(w-2)` in magnitude are used.
    RoundToInt { rnd: String, signed: bool, w: u32 },
    /// Conversion of a `w`-bit integer to `format`, round to nearest even.
    FromInt { signed: bool, w: u32 },
}

/// A core a wrapper needs: `CoreInstance::exp_width` and `sig_width` must be `format`'s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreNeed {
    pub role: Role,
    /// The core's float format: of its operands and result (arithmetic, approximate), of its result
    /// (convert), of its operands (less-than, to-integer).
    pub exp_width: u32,
    pub sig_width: u32,
    pub arity: usize,
}

/// A generated wrapper.
#[derive(Debug, Clone)]
pub struct Built {
    pub glue: Glue,
    pub rtl: Rtl,
    pub needs: Vec<CoreNeed>,
    pub mutation: Mutation,
}

/// The format of a core with one more exponent bit (design §6.4's rounding with an unbounded
/// exponent for a flushing format).
pub fn wide(f: Format) -> Format {
    Format::ieee(f.e + 1, f.m)
}

fn need(role: Role, f: Format, arity: usize) -> CoreNeed {
    CoreNeed {
        role,
        exp_width: f.e,
        sig_width: f.m,
        arity,
    }
}

/// The format of the operands a core sees for a format: E4M3FN decodes to f16, every other format is
/// its own.
fn decoded(f: Format) -> Format {
    if f.flavour == Flavour::FiniteNan {
        Format::F16
    } else {
        f
    }
}

impl Glue {
    /// The module name: `glue_<op>_<format>`.
    pub fn name(&self) -> String {
        match self {
            Glue::Arith { op, fmt } => format!("glue_{}_{}", op.name(), fmt.name()),
            Glue::Approx { op, fmt, via, .. } => {
                format!("glue_{op}_{}_via_{}", fmt.name(), via.name())
            }
            Glue::Convert { from, to } => format!("glue_convert_{}_{}", from.name(), to.name()),
            Glue::MaxMin { max, fmt } => {
                format!("glue_{}_{}", if *max { "maximum" } else { "minimum" }, fmt.name())
            }
            Glue::Compare { dir, fmt } => {
                format!("glue_compare_{}_{}", dir.name().to_lowercase(), fmt.name())
            }
            Glue::Sign { op, fmt } => format!("glue_{}_{}", op.name(), fmt.name()),
            Glue::ToInt { fmt, signed, w } => format!(
                "glue_convert_{}_{}{w}",
                fmt.name(),
                if *signed { "i" } else { "ui" }
            ),
            Glue::Compose { recipe, fmt, q } => {
                let core = if *recipe == Recipe::PowerCore { "_core" } else { "" };
                format!(
                    "glue_{}{core}_{}_q{q}",
                    crate::recipe::op_name(*recipe),
                    fmt.name()
                )
            }
            Glue::FromInt { fmt, signed, w, via } => {
                let i = if *signed { "i" } else { "ui" };
                if via == fmt {
                    format!("glue_convert_{i}{w}_{}", fmt.name())
                } else {
                    format!("glue_convert_{i}{w}_{}_via_{}", fmt.name(), via.name())
                }
            }
            Glue::Bool { fmt, to_float: true } => format!("glue_convert_bool_{}", fmt.name()),
            Glue::Bool { fmt, to_float: false } => format!("glue_convert_{}_bool", fmt.name()),
            Glue::IsFinite { fmt } => format!("glue_is_finite_{}", fmt.name()),
            Glue::Round { fmt, rnd } => format!("glue_{}_{}", round_op(*rnd), fmt.name()),
            Glue::ReducePrecision { fmt, e, m } => {
                format!("glue_reduce_precision_{}_e{e}m{m}", fmt.name())
            }
            Glue::Clamp { fmt } => format!("glue_clamp_{}", fmt.name()),
        }
    }

    /// The type of operand `j` (a float of the format, except for the conversions from integers and
    /// bool).
    pub fn input_ty(&self, j: usize) -> Ty {
        match self {
            Glue::FromInt { signed, w, .. } => Ty::Int {
                signed: *signed,
                w: *w,
            },
            Glue::Bool { to_float: true, .. } => Ty::Bool,
            _ => {
                let _ = j;
                Ty::Float(self.format())
            }
        }
    }

    /// The width of the result.
    pub fn output_width(&self) -> u32 {
        match self {
            Glue::Compare { .. } | Glue::IsFinite { .. } | Glue::Bool { to_float: false, .. } => 1,
            Glue::ToInt { w, .. } => *w,
            Glue::Convert { to, .. } => to.bits(),
            _ => self.format().bits(),
        }
    }

    /// The format of the operands.
    pub fn format(&self) -> Format {
        match self {
            Glue::Arith { fmt, .. }
            | Glue::Approx { fmt, .. }
            | Glue::MaxMin { fmt, .. }
            | Glue::Compare { fmt, .. }
            | Glue::Sign { fmt, .. }
            | Glue::ToInt { fmt, .. }
            | Glue::Compose { fmt, .. }
            | Glue::FromInt { fmt, .. }
            | Glue::Bool { fmt, .. }
            | Glue::IsFinite { fmt }
            | Glue::Round { fmt, .. }
            | Glue::ReducePrecision { fmt, .. }
            | Glue::Clamp { fmt } => *fmt,
            Glue::Convert { from, .. } => *from,
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Glue::Arith { op, .. } => op.arity(),
            Glue::Approx { arity, .. } => *arity,
            Glue::MaxMin { .. } | Glue::Compare { .. } => 2,
            Glue::Compose {
                recipe: Recipe::Power | Recipe::PowerCore,
                ..
            } => 2,
            Glue::Clamp { .. } => 3,
            _ => 1,
        }
    }

    /// The cores the wrapper instantiates, in order.
    pub fn needs(&self) -> Res<Vec<CoreNeed>> {
        let bad = || FloatError::Glue(format!("{}: no wrapper", self.name()));
        Ok(match self {
            Glue::Arith { op, fmt } => {
                let a = Role::Arith { op: op.name().into() };
                // A sum or difference that underflows is exact (Hauser, ACM TOPLAS 18(2), 1996): a
                // core of the format's own width rounds it like an unbounded exponent, and the glue
                // flushes it when subnormal. Products, quotients and roots need one more exponent bit.
                let cf = if fmt.flushes() && !matches!(op, Arith::Add | Arith::Sub) {
                    wide(*fmt)
                } else if fmt.flavour == Flavour::FiniteNan {
                    Format::F16
                } else {
                    *fmt
                };
                vec![need(a, cf, op.arity())]
            }
            Glue::Approx { op, arity, fmt, via } => {
                let a = need(Role::Approx { op: op.clone() }, *via, *arity);
                let direct = *via == *fmt || (fmt.flushes() && *via == wide(*fmt));
                if direct || *fmt == Format::E4M3FN {
                    vec![a]
                } else if fmt.bits() < via.bits() && fmt.flavour == Flavour::Ieee {
                    let to = if fmt.flushes() { wide(*fmt) } else { *fmt };
                    vec![a, need(Role::Convert { from: via.name() }, to, 1)]
                } else {
                    return Err(bad());
                }
            }
            Glue::Convert { from, to } => {
                if exact_widening(*from, *to) || *to == Format::E4M3FN && from.bits() > 8 {
                    vec![]
                } else if to.flavour == Flavour::Ieee && from.flavour == Flavour::Ieee && to != from {
                    // Narrowing, or neither containing the other (f16 and bf16): one rounding by a core.
                    let r = if to.flushes() { wide(*to) } else { *to };
                    vec![need(Role::Convert { from: from.name() }, r, 1)]
                } else {
                    return Err(bad());
                }
            }
            Glue::MaxMin { fmt, .. } => vec![need(Role::Less, decoded(*fmt), 2)],
            Glue::Compare { dir, fmt } => {
                let n = if *dir == Dir::Lt { 1 } else { 2 };
                vec![need(Role::Less, decoded(*fmt), 2); n]
            }
            Glue::Sign { .. } => vec![],
            Glue::ToInt { fmt, signed, w } => vec![need(
                Role::ToInt {
                    signed: *signed,
                    w: *w,
                },
                decoded(*fmt),
                1,
            )],
            Glue::Compose { recipe, fmt, q } => crate::recipe::needs(*recipe, *fmt, *q)?,
            Glue::FromInt { fmt, signed, w, via } => {
                let ok = if *fmt == Format::E4M3FN {
                    via.flavour == Flavour::Ieee && via.m > fmt.m && via.e >= fmt.e
                } else {
                    via == fmt
                };
                if !ok {
                    return Err(bad());
                }
                vec![need(
                    Role::FromInt {
                        signed: *signed,
                        w: *w,
                    },
                    *via,
                    1,
                )]
            }
            Glue::Bool { .. } | Glue::IsFinite { .. } | Glue::ReducePrecision { .. } => vec![],
            Glue::Round { fmt, rnd } => {
                let c = decoded(*fmt);
                let w = c.m + 2;
                vec![
                    need(
                        Role::RoundToInt {
                            rnd: rnd.name().into(),
                            signed: true,
                            w,
                        },
                        c,
                        1,
                    ),
                    need(Role::FromInt { signed: true, w }, c, 1),
                ]
            }
            Glue::Clamp { fmt } => vec![need(Role::Less, decoded(*fmt), 2); 2],
        })
    }

    /// The Verilog of the wrapper around `cores` (one per [`Glue::needs`] entry), with a deliberate fault
    /// for a negative control (`Mutation::None` otherwise).
    pub fn build(&self, cores: &[CoreInstance], mutation: Mutation) -> Res<Built> {
        let needs = self.needs()?;
        if cores.len() != needs.len() {
            return Err(FloatError::Core(format!(
                "{}: {} cores given, {} needed",
                self.name(),
                cores.len(),
                needs.len()
            )));
        }
        for (c, n) in cores.iter().zip(&needs) {
            if (c.exp_width, c.sig_width) != (n.exp_width, n.sig_width) {
                return Err(FloatError::Core(format!(
                    "{}: core {} computes in ({}, {}), the wrapper needs ({}, {})",
                    self.name(),
                    c.module,
                    c.exp_width,
                    c.sig_width,
                    n.exp_width,
                    n.sig_width
                )));
            }
        }
        let mut r = Rtl::new(self.name());
        let f = self.format();
        let xs: Vec<String> = (0..self.arity())
            .map(|j| {
                let w = match self.input_ty(j) {
                    Ty::Bool => 1,
                    Ty::Int { w, .. } => w,
                    _ => f.bits(),
                };
                r.input(&format!("a{j}"), w)
            })
            .collect();
        let core_fmt = |n: &CoreNeed| Format::ieee(n.exp_width, n.sig_width);
        let out = match self {
            Glue::Arith { fmt, .. } => {
                let cf = core_fmt(&needs[0]);
                let mut ops = Vec::new();
                for x in &xs {
                    let d = r.daz(*fmt, x);
                    ops.push(r.widen(*fmt, cf, &d)?);
                }
                let (_, z) = r.core(&cores[0], &ops, cf.bits(), cf.bits())?;
                if fmt.flushes() {
                    flush_narrow(&mut r, cf, *fmt, &z, mutation)?
                } else if *fmt == Format::E4M3FN {
                    r.encode_fp8(cf, *fmt, &z, false, mutation)?
                } else {
                    z
                }
            }
            Glue::Approx { fmt, via, .. } => {
                let mut ops = Vec::new();
                for x in &xs {
                    let d = r.daz(*fmt, x);
                    ops.push(r.widen(*fmt, *via, &d)?);
                }
                let (_, z) = r.core(&cores[0], &ops, via.bits(), via.bits())?;
                if via == fmt {
                    z
                } else if fmt.flushes() && *via == wide(*fmt) {
                    flush_narrow(&mut r, *via, *fmt, &z, mutation)?
                } else if *fmt == Format::E4M3FN {
                    r.encode_fp8(*via, *fmt, &z, false, mutation)?
                } else {
                    let to = core_fmt(&needs[1]);
                    let (_, z2) = r.core(&cores[1], &[z], via.bits(), to.bits())?;
                    if fmt.flushes() {
                        flush_narrow(&mut r, to, *fmt, &z2, mutation)?
                    } else {
                        z2
                    }
                }
            }
            Glue::Convert { from, to } => {
                let d = r.daz(*from, &xs[0]);
                if exact_widening(*from, *to) {
                    r.widen(*from, *to, &d)?
                } else if *to == Format::E4M3FN {
                    r.encode_fp8(*from, *to, &d, true, mutation)?
                } else {
                    let cf = core_fmt(&needs[0]);
                    let (_, z) = r.core(&cores[0], &[d], from.bits(), cf.bits())?;
                    if to.flushes() {
                        flush_narrow(&mut r, cf, *to, &z, mutation)?
                    } else if to.saturates() {
                        r.saturate(*to, &z)
                    } else {
                        z
                    }
                }
            }
            Glue::MaxMin { max, fmt } => {
                let cf = core_fmt(&needs[0]);
                max_min(&mut r, *fmt, cf, &cores[0], &xs[0], &xs[1], *max, mutation)?
            }
            Glue::Clamp { fmt } => {
                let cf = core_fmt(&needs[0]);
                // clamp(min, x, max) = minimum(maximum(x, min), max).
                let lo = max_min(&mut r, *fmt, cf, &cores[0], &xs[1], &xs[0], true, mutation)?;
                max_min(&mut r, *fmt, cf, &cores[1], &lo, &xs[2], false, mutation)?
            }
            Glue::FromInt { fmt, w, via, .. } => {
                let (_, z) = r.core(&cores[0], &[xs[0].clone()], *w, via.bits())?;
                if *fmt == Format::E4M3FN {
                    // A conversion: saturating (design §6.4).
                    r.encode_fp8(*via, *fmt, &z, true, mutation)?
                } else {
                    z
                }
            }
            Glue::Bool { fmt, to_float: true } => {
                let one = crate::recipe::encode(*fmt, 1.0, false);
                r.wire(
                    fmt.bits(),
                    format!("{} ? {}'d{one} : {}'d0", xs[0], fmt.bits(), fmt.bits()),
                )
            }
            Glue::Bool { fmt, to_float: false } => {
                let x = r.daz(*fmt, &xs[0]);
                let z = r.is_zero(*fmt, &x);
                r.wire(1, format!("~{z}"))
            }
            Glue::IsFinite { fmt } => {
                let x = r.daz(*fmt, &xs[0]);
                let n = r.is_nan(*fmt, &x);
                let i = r.is_inf(*fmt, &x);
                r.wire(1, format!("~({n} | {i})"))
            }
            Glue::Round { fmt, .. } => {
                let cf = core_fmt(&needs[0]);
                let w = match needs[0].role {
                    Role::RoundToInt { w, .. } => w,
                    _ => return Err(FloatError::Glue("round: no to-integer core".into())),
                };
                let x = r.daz(*fmt, &xs[0]);
                let wx = r.widen(*fmt, cf, &x)?;
                let (_, n) = r.core(&cores[0], &[wx], cf.bits(), w)?;
                let (_, y) = r.core(&cores[1], &[n], w, cf.bits())?;
                // Integral values below 2^(p-1) are exact in the format.
                let y = if cf == *fmt {
                    y
                } else {
                    r.encode_fp8(cf, *fmt, &y, false, mutation)?
                };
                let wd = fmt.bits();
                let limit = crate::recipe::encode(*fmt, 2f64.powi(fmt.m as i32), false);
                let big = r.wire(1, format!("{x}[{}:0] >= {}'d{limit}", wd - 2, wd - 1));
                let yz = r.is_zero(*fmt, &y);
                let s = r.sign(*fmt, &x);
                let zero = r.wire(wd, format!("{{{s}, {}'d0}}", wd - 1));
                if mutation == Mutation::NoZeroOrder {
                    r.wire(wd, format!("{big} ? {x} : {y}"))
                } else {
                    r.wire(wd, format!("{big} ? {x} : ({yz} ? {zero} : {y})"))
                }
            }
            Glue::ReducePrecision { fmt, e, m } => reduce_precision(&mut r, *fmt, &xs[0], *e, *m)?,
            Glue::Compare { dir, fmt } => {
                let cf = core_fmt(&needs[0]);
                let x = r.daz(*fmt, &xs[0]);
                let y = r.daz(*fmt, &xs[1]);
                let wx = r.widen(*fmt, cf, &x)?;
                let wy = r.widen(*fmt, cf, &y)?;
                let (_, lt) = r.core(&cores[0], &[wx.clone(), wy.clone()], cf.bits(), 1)?;
                if *dir == Dir::Lt {
                    lt
                } else {
                    let (_, gt) = r.core(&cores[1], &[wy, wx], cf.bits(), 1)?;
                    let nx = r.is_nan(*fmt, &x);
                    let ny = r.is_nan(*fmt, &y);
                    let un = if mutation == Mutation::NoNanCase {
                        r.wire(1, "1'b0")
                    } else {
                        r.wire(1, format!("{nx} | {ny}"))
                    };
                    match dir {
                        Dir::Le => r.wire(1, format!("~{gt} & ~{un}")),
                        Dir::Eq => r.wire(1, format!("~{lt} & ~{gt} & ~{un}")),
                        _ => r.wire(1, format!("{lt} | {gt} | {un}")),
                    }
                }
            }
            Glue::Sign { op, fmt } => {
                let x = r.daz(*fmt, &xs[0]);
                let w = fmt.bits();
                let s = r.sign(*fmt, &x);
                match op {
                    SignOp::Abs => r.wire(w, format!("{{1'b0, {x}[{}:0]}}", w - 2)),
                    SignOp::Negate => r.wire(w, format!("{{~{s}, {x}[{}:0]}}", w - 2)),
                    SignOp::Sign => {
                        let n = r.is_nan(*fmt, &x);
                        let z = r.is_zero(*fmt, &x);
                        let u = r.unit(*fmt, &x);
                        r.wire(w, format!("({n} | {z}) ? {x} : {u}"))
                    }
                }
            }
            Glue::Compose { recipe, fmt, q } => {
                crate::recipe::build(&mut r, *recipe, *fmt, *q, &xs, cores, mutation)?
            }
            Glue::ToInt { fmt, w, .. } => {
                let cf = core_fmt(&needs[0]);
                let x = r.daz(*fmt, &xs[0]);
                let wx = r.widen(*fmt, cf, &x)?;
                let (_, z) = r.core(&cores[0], &[wx], cf.bits(), *w)?;
                if mutation == Mutation::NoNanCase {
                    z
                } else {
                    let n = r.is_nan(*fmt, &x);
                    r.wire(*w, format!("{n} ? {w}'d0 : {z}"))
                }
            }
        };
        let ow = self.output_width();
        r.output("z", ow, &out);
        Ok(Built {
            glue: self.clone(),
            rtl: r,
            needs,
            mutation,
        })
    }
}

/// The catalog name of a rounding to integral.
pub fn round_op(rnd: Rounding) -> &'static str {
    match rnd {
        Rounding::Floor => "floor",
        Rounding::Ceil => "ceil",
        Rounding::NearestEven | Rounding::TowardZero => "round_nearest_even",
    }
}

/// IEEE 754-2019 `maximum` / `minimum` of `a` and `b` (raw encodings of `fmt`) around a less-than
/// core of `cf`: DAZ, NaN if either is NaN, `-0 < +0` among zeros, else the core's order.
#[allow(clippy::too_many_arguments)]
fn max_min(
    r: &mut Rtl,
    fmt: Format,
    cf: Format,
    core: &CoreInstance,
    a: &str,
    b: &str,
    max: bool,
    mutation: Mutation,
) -> Res<String> {
    let x = r.daz(fmt, a);
    let y = r.daz(fmt, b);
    let wx = r.widen(fmt, cf, &x)?;
    let wy = r.widen(fmt, cf, &y)?;
    let (_, lt) = r.core(core, &[wx, wy], cf.bits(), 1)?;
    let nx = r.is_nan(fmt, &x);
    let ny = r.is_nan(fmt, &y);
    let zx = r.is_zero(fmt, &x);
    let zy = r.is_zero(fmt, &y);
    let sx = r.sign(fmt, &x);
    let (pick, zpick) = if max {
        (
            r.wire(fmt.bits(), format!("{lt} ? {y} : {x}")),
            r.wire(fmt.bits(), format!("{sx} ? {y} : {x}")),
        )
    } else {
        (
            r.wire(fmt.bits(), format!("{lt} ? {x} : {y}")),
            r.wire(fmt.bits(), format!("{sx} ? {x} : {y}")),
        )
    };
    let ordered = if mutation == Mutation::NoZeroOrder {
        pick
    } else {
        r.wire(fmt.bits(), format!("({zx} & {zy}) ? {zpick} : {pick}"))
    };
    Ok(if mutation == Mutation::NoNanCase {
        ordered
    } else {
        r.wire(
            fmt.bits(),
            format!("({nx} | {ny}) ? {} : {ordered}", Rtl::nan_literal(fmt)),
        )
    })
}

/// StableHLO `reduce_precision` of an IEEE value of `f` to `eb` exponent and `mb` significand bits, on
/// its bit pattern after DAZ (`cc_smt::ops`' definition): ties to even by adding `2^(k-1) - 1` and the
/// kept least significant bit and clearing the `k = m - mb` low bits; an exponent above the reduced
/// range gives infinity, below it zero, both with the sign; NaN and infinities are kept.
fn reduce_precision(r: &mut Rtl, f: Format, a: &str, eb: u32, mb: u32) -> Res<String> {
    if f.flavour != Flavour::Ieee || mb > f.m || eb > f.e || eb < 2 {
        return Err(FloatError::Glue(format!(
            "no reduce_precision of {} to ({eb}, {mb})",
            f.name()
        )));
    }
    let w = f.bits();
    let x = r.daz(f, a);
    let k = f.m - mb;
    let rounded = if k == 0 {
        x.clone()
    } else {
        let half = (1u128 << (k - 1)) - 1;
        let sum = r.wire(w, format!("{x} + {w}'d{half} + {{{}'d0, {x}[{k}]}}", w - 1));
        let mask = !((1u128 << k) - 1) & ((1u128 << w) - 1);
        r.wire(w, format!("{sum} & {w}'d{mask}"))
    };
    let rbias = (1i64 << (eb - 1)) - 1;
    let hi = (rbias + f.bias()) as u128;
    let lo = (1 - rbias + f.bias()) as u128;
    let ex = format!("{rounded}[{}:{}]", f.e + f.m - 1, f.m);
    let over = r.wire(1, format!("{ex} > {}'d{hi}", f.e));
    let under = r.wire(1, format!("{ex} < {}'d{lo}", f.e));
    let s = r.sign(f, &x);
    let inf = r.wire(w, format!("{{{s}, {}'d{}}}", w - 1, ((1u128 << f.e) - 1) << f.m));
    let zero = r.wire(w, format!("{{{s}, {}'d0}}", w - 1));
    let n = r.is_nan(f, &x);
    let i = r.is_inf(f, &x);
    Ok(r.wire(
        w,
        format!("({n} | {i}) ? {x} : ({over} ? {inf} : ({under} ? {zero} : {rounded}))"),
    ))
}

/// `to` holds every value of `from` exactly (an IEEE `to` with at least `from`'s exponent and
/// significand bits; subnormals of `from` normal in `to` when the exponent fields differ).
pub fn exact_widening(from: Format, to: Format) -> bool {
    from != to
        && to.flavour == Flavour::Ieee
        && to.e >= from.e
        && to.m >= from.m
        && (to.e == from.e || from.flushes() || from.emin() - i64::from(from.m) >= to.emin())
}

fn flush_narrow(r: &mut Rtl, c: Format, f: Format, z: &str, m: Mutation) -> Res<String> {
    if c == f {
        // A core of the format's own width: a subnormal result (exact) is flushed.
        return Ok(if m == Mutation::NoFlush {
            z.to_string()
        } else {
            r.daz(f, z)
        });
    }
    if m == Mutation::NoFlush {
        // A negative control: results below 2^emin narrowed as if normal (exponent wrapped).
        let s = r.sign(c, z);
        let d = (c.bias() - f.bias()) as u128;
        let reb = r.wire(c.e, format!("{z}[{}:{}] - {}'d{d}", c.e + c.m - 1, c.m, c.e));
        let top = r.wire(1, format!("&{z}[{}:{}]", c.e + c.m - 1, c.m));
        let body = r.wire(
            f.bits(),
            format!("{{{s}, {reb}[{}:0], {z}[{}:0]}}", f.e - 1, c.m - 1),
        );
        let special = r.wire(
            f.bits(),
            format!(
                "{{{s}, {}'b{}, {z}[{}:0]}}",
                f.e,
                "1".repeat(f.e as usize),
                c.m - 1
            ),
        );
        return Ok(r.wire(f.bits(), format!("{top} ? {special} : {body}")));
    }
    r.flush_narrow(c, f, z, true)
}

/// The default wrapper set: every op the glue covers at each format of `formats` (the approximate
/// operators on cores of the format itself, of the format with one more exponent bit, or of f32).
pub fn standard(formats: &[Format]) -> Vec<Glue> {
    let mut out = Vec::new();
    for &f in formats {
        if f.flushes() || f == Format::E4M3FN || f == Format::F16 {
            for op in [Arith::Add, Arith::Sub, Arith::Mul, Arith::Div, Arith::Sqrt] {
                out.push(Glue::Arith { op, fmt: f });
            }
        }
        for max in [true, false] {
            out.push(Glue::MaxMin { max, fmt: f });
        }
        for dir in [Dir::Eq, Dir::Ne, Dir::Lt, Dir::Le] {
            out.push(Glue::Compare { dir, fmt: f });
        }
        for op in [SignOp::Abs, SignOp::Negate, SignOp::Sign] {
            out.push(Glue::Sign { op, fmt: f });
        }
        for (signed, w) in [(true, 32), (true, 8)] {
            out.push(Glue::ToInt { fmt: f, signed, w });
        }
        for &g in formats {
            if g != f && (exact_widening(f, g) || Glue::Convert { from: f, to: g }.needs().is_ok()) {
                out.push(Glue::Convert { from: f, to: g });
            }
        }
        out.push(Glue::Clamp { fmt: f });
        for (signed, w) in [(true, 32), (true, 8), (false, 8)] {
            out.push(Glue::FromInt {
                fmt: f,
                signed,
                w,
                via: int_via(f, signed, w),
            });
        }
        for to_float in [true, false] {
            out.push(Glue::Bool { fmt: f, to_float });
        }
        out.push(Glue::IsFinite { fmt: f });
        for rnd in [Rounding::Floor, Rounding::Ceil, Rounding::NearestEven] {
            out.push(Glue::Round { fmt: f, rnd });
        }
        if f.native() && f.e >= 5 && f.m >= 10 {
            out.push(Glue::ReducePrecision { fmt: f, e: 5, m: 10 });
        }
        let via = if f.flushes() {
            wide(f)
        } else if f == Format::F16 {
            f
        } else {
            Format::F32
        };
        out.push(Glue::Approx {
            op: "exponential".into(),
            arity: 1,
            fmt: f,
            via,
        });
        if f.bits() == 16 && via != Format::F32 {
            out.push(Glue::Approx {
                op: "exponential".into(),
                arity: 1,
                fmt: f,
                via: Format::F32,
            });
        }
    }
    out
}

/// The format of the integer-to-float core for `fmt`: the format itself, or for E4M3FN (no core has its
/// encoding) the first of f16, f32 and an IEEE format with 8 exponent bits that holds every integer of
/// the width exactly, so that the encoder rounds once.
pub fn int_via(fmt: Format, signed: bool, w: u32) -> Format {
    if fmt.flavour == Flavour::Ieee {
        return fmt;
    }
    let magnitude = if signed { w - 1 } else { w };
    [
        Format::F16,
        Format::F32,
        Format::ieee(8, magnitude.saturating_sub(1).max(2)),
    ]
    .into_iter()
    .find(|q| q.e >= fmt.e && q.m + 1 >= magnitude)
    .unwrap_or(Format::F32)
}

/// A placeholder instance for a need (tests and examples): module `core_<role>_<e>_<m>`, the
/// operands on ports `a`, `b`, the result on `z`, a rounding-mode port tied to round to nearest even.
pub fn stand_in(n: &CoreNeed) -> CoreInstance {
    let role = match &n.role {
        Role::Arith { op } | Role::Approx { op } => op.clone(),
        Role::Convert { from } => format!("from_{from}"),
        Role::Less => "lt".into(),
        Role::ToInt { signed, w } => format!("to_{}{w}", if *signed { "i" } else { "ui" }),
        Role::RoundToInt { rnd, signed, w } => {
            format!("to_{}{w}_{rnd}", if *signed { "i" } else { "ui" })
        }
        Role::FromInt { signed, w } => format!("from_{}{w}", if *signed { "i" } else { "ui" }),
    };
    let mut ports = std::collections::BTreeMap::new();
    for (j, p) in ["a", "b"].into_iter().take(n.arity).enumerate() {
        ports.insert(p.to_string(), format!("a{j}"));
    }
    ports.insert("z".into(), "z".into());
    ports.insert("rnd".into(), "3'b000".into());
    ports.insert("status".into(), String::new());
    CoreInstance {
        module: format!("core_{role}_{}_{}", n.exp_width, n.sig_width),
        params: [
            ("exp_width".to_string(), n.exp_width.to_string()),
            ("sig_width".to_string(), n.sig_width.to_string()),
        ]
        .into(),
        ports,
        exp_width: n.exp_width,
        sig_width: n.sig_width,
    }
}
