//! The glue wrappers: what each needs from cores and the Verilog that wraps them (design §4.12,
//! docs/framework/float.md).
//!
//! A wrapper is one module per op and format around zero, one or two core instances. The format rules
//! are design §6.4's: f32 and bf16 read subnormal operands as zero and flush results whose value rounded
//! with an unbounded exponent is below `2^emin`, so their cores run with one more exponent bit and the
//! glue flushes and narrows exactly; f16 and the 8-bit formats keep gradual underflow; E4M3FN runs on
//! f16 (arithmetic) or f32 (approximate operators) cores through an exact decoder and a round-to-nearest
//! encoder.

use cc_core::adapters::CoreInstance;
use cc_smt::float::{Arith, Flavour, Format};
use serde::{Deserialize, Serialize};

use crate::FloatError;
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
    /// `convert` between float formats: exact widening (glue only), narrowing to an IEEE format (a
    /// convert core), to E4M3FN (the encoder, saturating), to E5M2 (a convert core, saturated).
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
            Glue::MaxMin { max, fmt } => format!(
                "glue_{}_{}",
                if *max { "maximum" } else { "minimum" },
                fmt.name()
            ),
            Glue::Compare { dir, fmt } => {
                format!("glue_compare_{}_{}", dir.name().to_lowercase(), fmt.name())
            }
            Glue::Sign { op, fmt } => format!("glue_{}_{}", op.name(), fmt.name()),
            Glue::ToInt { fmt, signed, w } => format!(
                "glue_convert_{}_{}{w}",
                fmt.name(),
                if *signed { "i" } else { "ui" }
            ),
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
            | Glue::ToInt { fmt, .. } => *fmt,
            Glue::Convert { from, .. } => *from,
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Glue::Arith { op, .. } => op.arity(),
            Glue::Approx { arity, .. } => *arity,
            Glue::MaxMin { .. } | Glue::Compare { .. } => 2,
            _ => 1,
        }
    }

    /// The cores the wrapper instantiates, in order.
    pub fn needs(&self) -> Res<Vec<CoreNeed>> {
        let bad = || FloatError::Glue(format!("{}: no wrapper", self.name()));
        Ok(match self {
            Glue::Arith { op, fmt } => {
                let a = Role::Arith {
                    op: op.name().into(),
                };
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
            Glue::Approx {
                op,
                arity,
                fmt,
                via,
            } => {
                let a = need(Role::Approx { op: op.clone() }, *via, *arity);
                let direct = *via == *fmt || (fmt.flushes() && *via == wide(*fmt));
                if direct || *fmt == Format::E4M3FN {
                    vec![a]
                } else if fmt.bits() < via.bits() && fmt.flavour == Flavour::Ieee {
                    let to = if fmt.flushes() { wide(*fmt) } else { *fmt };
                    vec![
                        a,
                        need(
                            Role::Convert {
                                from: via.name(),
                            },
                            to,
                            1,
                        ),
                    ]
                } else {
                    return Err(bad());
                }
            }
            Glue::Convert { from, to } => {
                if exact_widening(*from, *to) || *to == Format::E4M3FN && from.bits() > 8 {
                    vec![]
                } else if to.bits() < from.bits() && to.flavour == Flavour::Ieee {
                    let r = if to.flushes() { wide(*to) } else { *to };
                    vec![need(
                        Role::Convert {
                            from: from.name(),
                        },
                        r,
                        1,
                    )]
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
            .map(|j| r.input(&format!("a{j}"), f.bits()))
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
                let x = r.daz(*fmt, &xs[0]);
                let y = r.daz(*fmt, &xs[1]);
                let wx = r.widen(*fmt, cf, &x)?;
                let wy = r.widen(*fmt, cf, &y)?;
                let (_, lt) = r.core(&cores[0], &[wx, wy], cf.bits(), 1)?;
                let nx = r.is_nan(*fmt, &x);
                let ny = r.is_nan(*fmt, &y);
                let zx = r.is_zero(*fmt, &x);
                let zy = r.is_zero(*fmt, &y);
                let sx = r.sign(*fmt, &x);
                let (pick, zpick) = if *max {
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
                    r.wire(
                        fmt.bits(),
                        format!("({zx} & {zy}) ? {zpick} : {pick}"),
                    )
                };
                if mutation == Mutation::NoNanCase {
                    ordered
                } else {
                    r.wire(
                        fmt.bits(),
                        format!(
                            "({nx} | {ny}) ? {} : {ordered}",
                            Rtl::nan_literal(*fmt)
                        ),
                    )
                }
            }
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
        let ow = match self {
            Glue::Compare { .. } => 1,
            Glue::ToInt { w, .. } => *w,
            Glue::Convert { to, .. } => to.bits(),
            _ => f.bits(),
        };
        r.output("z", ow, &out);
        Ok(Built {
            glue: self.clone(),
            rtl: r,
            needs,
            mutation,
        })
    }
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
            format!("{{{s}, {}'b{}, {z}[{}:0]}}", f.e, "1".repeat(f.e as usize), c.m - 1),
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

/// A placeholder instance for a need (tests and examples): module `core_<role>_<e>_<m>`, the
/// operands on ports `a`, `b`, the result on `z`, a rounding-mode port tied to round to nearest even.
pub fn stand_in(n: &CoreNeed) -> CoreInstance {
    let role = match &n.role {
        Role::Arith { op } | Role::Approx { op } => op.clone(),
        Role::Convert { from } => format!("from_{from}"),
        Role::Less => "lt".into(),
        Role::ToInt { signed, w } => format!("to_{}{w}", if *signed { "i" } else { "ui" }),
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
