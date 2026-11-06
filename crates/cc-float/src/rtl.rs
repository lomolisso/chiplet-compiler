//! A small Verilog-2001 writer and the bit-level glue it emits (docs/framework/float.md, "Glue").
//!
//! Every glue function takes the name of a wire holding an encoding of a format and returns the name of
//! a new wire. Values are IEEE 754 encodings (sign, biased exponent, trailing significand) of a
//! `cc_smt::float::Format`; a finite-NaN format (E4M3FN) has no infinities and one NaN pattern per sign.
//! The only significand arithmetic is the FP8 encoder ([`Rtl::encode_fp8`]), a round to nearest even.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use cc_core::adapters::CoreInstance;
use cc_smt::float::{Flavour, Format};

use crate::FloatError;

type Res<X> = Result<X, FloatError>;

/// A port of the generated module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Port {
    pub name: String,
    pub width: u32,
}

/// One module under construction: ports, wires, assignments, core instances.
#[derive(Debug, Clone)]
pub struct Rtl {
    name: String,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    body: Vec<String>,
    next: u32,
    /// Module names of the core instances with their parameter names and port widths (for the stubs
    /// the proofs read in their place).
    stubs: BTreeMap<String, Stub>,
    cores: u32,
    wires: Vec<CoreWires>,
}

/// The wires around one core instance (the proofs constrain them by the core's contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreWires {
    /// `(name, width)` of each operand wire `c<k>_a<j>`.
    pub operands: Vec<(String, u32)>,
    /// `(name, width)` of the result wire `c<k>_z`.
    pub result: (String, u32),
    /// The instance name `u_c<k>`.
    pub cell: String,
}

/// The interface of a core module as the glue uses it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stub {
    pub params: Vec<String>,
    /// Port name -> (is an output, width).
    pub ports: BTreeMap<String, (bool, u32)>,
}

/// The width of a sized Verilog literal such as `3'b000`.
fn literal_width(s: &str) -> Option<u32> {
    s.split_once('\'').and_then(|(w, _)| w.trim().parse().ok())
}

fn ones(w: u32) -> String {
    format!("{w}'b{}", "1".repeat(w as usize))
}

fn zeros(w: u32) -> String {
    format!("{w}'b{}", "0".repeat(w as usize))
}

fn lit(v: u128, w: u32) -> String {
    format!("{w}'d{v}")
}

/// `x` (of `from` bits) zero-extended to `to` bits.
fn zext(x: &str, from: u32, to: u32) -> String {
    if to <= from {
        x.to_string()
    } else {
        format!("{{{}, {x}}}", zeros(to - from))
    }
}

impl Rtl {
    pub fn new(name: impl Into<String>) -> Rtl {
        Rtl {
            name: name.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            body: Vec::new(),
            next: 0,
            stubs: BTreeMap::new(),
            cores: 0,
            wires: Vec::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn inputs(&self) -> &[Port] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Port] {
        &self.outputs
    }

    pub fn stubs(&self) -> &BTreeMap<String, Stub> {
        &self.stubs
    }

    /// The core instances, in order.
    pub fn core_wires(&self) -> &[CoreWires] {
        &self.wires
    }

    /// An input port.
    pub fn input(&mut self, name: &str, width: u32) -> String {
        self.inputs.push(Port {
            name: name.into(),
            width,
        });
        name.to_string()
    }

    /// An output port driven by `expr`.
    pub fn output(&mut self, name: &str, width: u32, expr: &str) {
        self.outputs.push(Port {
            name: name.into(),
            width,
        });
        self.body.push(format!("  assign {name} = {expr};"));
    }

    /// A fresh wire `g<k>` of `width` bits driven by `expr`.
    pub fn wire(&mut self, width: u32, expr: impl AsRef<str>) -> String {
        let n = format!("g{}", self.next);
        self.next += 1;
        self.body
            .push(format!("  wire [{}:0] {n} = {};", width - 1, expr.as_ref()));
        n
    }

    /// A named wire kept through synthesis and visible to the proofs (`(* keep *)`).
    pub fn named(&mut self, name: &str, width: u32, expr: impl AsRef<str>) -> String {
        self.body.push(format!(
            "  (* keep *) wire [{}:0] {name} = {};",
            width - 1,
            expr.as_ref()
        ));
        name.to_string()
    }

    /// A comment line in the body.
    pub fn note(&mut self, text: &str) {
        self.body.push(format!("  // {text}"));
    }

    /// Instantiate a core: its operands `a0, a1, ...` are the given wires (of `operand_width` bits),
    /// its result `z` a new named wire of `result_width` bits. Returns `(k, result wire)`; the operand
    /// and result wires are `c<k>_a<j>` and `c<k>_z`.
    pub fn core(
        &mut self,
        inst: &CoreInstance,
        operands: &[String],
        operand_width: u32,
        result_width: u32,
    ) -> Res<(u32, String)> {
        let k = self.cores;
        self.cores += 1;
        let mut names = Vec::new();
        for (j, o) in operands.iter().enumerate() {
            names.push(self.named(&format!("c{k}_a{j}"), operand_width, o));
        }
        let z = format!("c{k}_z");
        self.body
            .push(format!("  (* keep *) wire [{}:0] {z};", result_width - 1));
        let stub = self.stubs.entry(inst.module.clone()).or_default();
        let mut conns = Vec::new();
        let mut has_z = false;
        for (port, what) in &inst.ports {
            let (out, w, expr) = if what == "z" {
                has_z = true;
                (true, result_width, z.clone())
            } else if let Some(j) = what.strip_prefix('a').and_then(|j| j.parse::<usize>().ok()) {
                let n = names.get(j).ok_or_else(|| {
                    FloatError::Core(format!(
                        "{}: port {port} reads operand a{j}, the glue has {}",
                        inst.module,
                        names.len()
                    ))
                })?;
                (false, operand_width, n.clone())
            } else if what.is_empty() {
                (true, 1, String::new())
            } else {
                let w = literal_width(what).ok_or_else(|| {
                    FloatError::Core(format!(
                        "{}: port {port} = {what:?} is no operand, result or sized literal",
                        inst.module
                    ))
                })?;
                (false, w, what.clone())
            };
            let prev = stub.ports.insert(port.clone(), (out, w));
            if prev.is_some_and(|p| p != (out, w)) {
                return Err(FloatError::Core(format!(
                    "{}: port {port} used with two widths",
                    inst.module
                )));
            }
            conns.push(format!(".{port}({expr})"));
        }
        if !has_z {
            return Err(FloatError::Core(format!(
                "{}: no port is the result z",
                inst.module
            )));
        }
        for p in inst.params.keys() {
            if !stub.params.contains(p) {
                stub.params.push(p.clone());
            }
        }
        let params = if inst.params.is_empty() {
            String::new()
        } else {
            let ps: Vec<String> = inst.params.iter().map(|(p, v)| format!(".{p}({v})")).collect();
            format!(" #({})", ps.join(", "))
        };
        self.body.push(format!(
            "  {}{params} u_c{k} ({});",
            inst.module,
            conns.join(", ")
        ));
        self.wires.push(CoreWires {
            operands: names.iter().map(|n| (n.clone(), operand_width)).collect(),
            result: (z.clone(), result_width),
            cell: format!("u_c{k}"),
        });
        Ok((k, z))
    }

    /// The module text.
    pub fn verilog(&self) -> String {
        let mut s = String::new();
        let ports: Vec<String> = self
            .inputs
            .iter()
            .map(|p| format!("input [{}:0] {}", p.width - 1, p.name))
            .chain(
                self.outputs
                    .iter()
                    .map(|p| format!("output [{}:0] {}", p.width - 1, p.name)),
            )
            .collect();
        let _ = writeln!(s, "module {} (", self.name);
        let _ = writeln!(s, "  {}", ports.join(",\n  "));
        let _ = writeln!(s, ");");
        for l in &self.body {
            let _ = writeln!(s, "{l}");
        }
        let _ = writeln!(s, "endmodule");
        s
    }

    /// Blackbox modules standing for the cores (only their interface; the proofs cut them).
    pub fn stub_verilog(&self) -> String {
        let mut s = String::new();
        for (m, st) in &self.stubs {
            let _ = writeln!(s, "(* blackbox *)");
            let params: Vec<String> = st.params.iter().map(|p| format!("parameter {p} = 0")).collect();
            let ps = if params.is_empty() {
                String::new()
            } else {
                format!(" #({})", params.join(", "))
            };
            let ports: Vec<String> = st
                .ports
                .iter()
                .map(|(p, (out, w))| format!("{} [{}:0] {p}", if *out { "output" } else { "input" }, w - 1))
                .collect();
            let _ = writeln!(s, "module {m}{ps} ({});", ports.join(", "));
            let _ = writeln!(s, "endmodule");
        }
        s
    }

    // ---- Glue (bit level) -------------------------------------------------------------------------

    /// The sign bit.
    pub fn sign(&mut self, f: Format, x: &str) -> String {
        format!("{x}[{}]", f.bits() - 1)
    }

    fn exp(f: Format, x: &str) -> String {
        format!("{x}[{}:{}]", f.e + f.m - 1, f.m)
    }

    fn man(f: Format, x: &str) -> String {
        format!("{x}[{}:0]", f.m - 1)
    }

    /// 1 when `x` is a NaN of `f`.
    pub fn is_nan(&mut self, f: Format, x: &str) -> String {
        let e = Self::exp(f, x);
        let m = Self::man(f, x);
        match f.flavour {
            Flavour::Ieee => self.wire(1, format!("(&{e}) & (|{m})")),
            Flavour::FiniteNan => self.wire(1, format!("(&{e}) & (&{m})")),
        }
    }

    /// 1 when `x` is an infinity of `f` (never for a finite-NaN format).
    pub fn is_inf(&mut self, f: Format, x: &str) -> String {
        match f.flavour {
            Flavour::Ieee => {
                let e = Self::exp(f, x);
                let m = Self::man(f, x);
                self.wire(1, format!("(&{e}) & ~(|{m})"))
            }
            Flavour::FiniteNan => self.wire(1, "1'b0"),
        }
    }

    /// 1 when `x` is a zero of either sign.
    pub fn is_zero(&mut self, f: Format, x: &str) -> String {
        self.wire(1, format!("~(|{x}[{}:0])", f.bits() - 2))
    }

    /// The canonical NaN of `f` with sign 0 (IEEE: the quiet bit; FN: the all-ones pattern).
    pub fn nan_literal(f: Format) -> String {
        lit(nan_bits(f), f.bits())
    }

    /// Design §6.4's DAZ: in a flushing format a subnormal operand reads as zero of its sign.
    pub fn daz(&mut self, f: Format, x: &str) -> String {
        if !f.flushes() {
            return x.to_string();
        }
        let e = Self::exp(f, x);
        let s = self.sign(f, x);
        self.wire(
            f.bits(),
            format!("(~(|{e})) ? {{{s}, {}}} : {x}", zeros(f.bits() - 1)),
        )
    }

    /// The exact re-encoding of a value of `f` in the wider IEEE format `g` (`g.e >= f.e`,
    /// `g.m >= f.m`): exponent rebiased, significand padded, subnormals of `f` normalized when they
    /// are normal in `g`, infinities kept, NaN to `g`'s canonical NaN when `f` is finite-NaN.
    pub fn widen(&mut self, f: Format, g: Format, x: &str) -> Res<String> {
        if f == g {
            return Ok(x.to_string());
        }
        if g.flavour != Flavour::Ieee || g.e < f.e || g.m < f.m {
            return Err(FloatError::Glue(format!(
                "no exact widening from {} to {}",
                f.name(),
                g.name()
            )));
        }
        let s = self.sign(f, x);
        let e = Self::exp(f, x);
        let m = Self::man(f, x);
        let pad = g.m - f.m;
        let padm = if pad == 0 {
            m.clone()
        } else {
            format!("{{{m}, {}}}", zeros(pad))
        };
        if g.e == f.e && f.flavour == Flavour::Ieee {
            // Same exponent field: the encoding with the significand padded (subnormals included).
            return Ok(self.wire(g.bits(), format!("{{{x}, {}}}", zeros(pad))));
        }
        let d = (g.bias() - f.bias()) as u128;
        let reb = self.wire(g.e, format!("{{{}, {e}}} + {}", zeros(g.e - f.e), lit(d, g.e)));
        let normal = self.wire(g.bits(), format!("{{{s}, {reb}, {padm}}}"));
        let top = match f.flavour {
            Flavour::Ieee => self.wire(g.bits(), format!("{{{s}, {}, {padm}}}", ones(g.e))),
            Flavour::FiniteNan => {
                // The all-ones exponent is a normal binade except the NaN pattern.
                let nan = self.is_nan(f, x);
                self.wire(g.bits(), format!("{nan} ? {} : {normal}", Self::nan_literal(g)))
            }
        };
        // Subnormals: the leading one at bit i of the trailing significand gives 2^(emin - m + i).
        let sub = if f.flushes() {
            self.wire(g.bits(), format!("{{{s}, {}}}", zeros(g.bits() - 1)))
        } else {
            let lowest = f.emin() - i64::from(f.m);
            if lowest < g.emin() {
                return Err(FloatError::Glue(format!(
                    "subnormals of {} are not normal in {}",
                    f.name(),
                    g.name()
                )));
            }
            let mut expr = format!("{{{s}, {}}}", zeros(g.bits() - 1));
            for i in 0..f.m {
                let ex = (lowest + i64::from(i) + g.bias()) as u128;
                let frac = if i == 0 {
                    zeros(g.m)
                } else {
                    format!("{{{x}[{}:0], {}}}", i - 1, zeros(g.m - i))
                };
                expr = format!("{x}[{i}] ? {{{s}, {}, {frac}}} : ({expr})", lit(ex, g.e));
            }
            self.wire(g.bits(), expr)
        };
        let ez = self.wire(1, format!("~(|{e})"));
        let et = self.wire(1, format!("&{e}"));
        Ok(self.wire(g.bits(), format!("{ez} ? {sub} : ({et} ? {top} : {normal})")))
    }

    /// A result of the IEEE format `c` (`c.m == f.m`, `c.e >= f.e`) as a value of `f` under design
    /// §6.4: NaN and infinities kept, beyond `f`'s largest binade infinity, and below `2^emin` zero of
    /// its sign when `flush` (else an error: a gradual format runs its core at its own width). The core
    /// rounded to `f`'s precision with `c`'s exponent range, so no rounding happens here.
    pub fn flush_narrow(&mut self, c: Format, f: Format, z: &str, flush: bool) -> Res<String> {
        if c.m != f.m || c.e < f.e || f.flavour != Flavour::Ieee {
            return Err(FloatError::Glue(format!(
                "no flush-narrow from {} to {}",
                c.name(),
                f.name()
            )));
        }
        if c == f {
            return Ok(z.to_string());
        }
        if !flush {
            return Err(FloatError::Glue(format!(
                "{} keeps gradual underflow: its core runs at its own width",
                f.name()
            )));
        }
        let s = self.sign(c, z);
        let e = Self::exp(c, z);
        let m = Self::man(c, z);
        let d = (c.bias() - f.bias()) as u128;
        let lo = d + 1;
        let hi = d + (1u128 << f.e) - 2;
        let top = self.wire(1, format!("&{e}"));
        let big = self.wire(1, format!("{e} > {}", lit(hi, c.e)));
        let tiny = self.wire(1, format!("{e} < {}", lit(lo, c.e)));
        let reb = self.wire(c.e, format!("{e} - {}", lit(d, c.e)));
        let normal = self.wire(f.bits(), format!("{{{s}, {reb}[{}:0], {m}}}", f.e - 1));
        let special = self.wire(f.bits(), format!("{{{s}, {}, {m}}}", ones(f.e)));
        let inf = self.wire(f.bits(), format!("{{{s}, {}, {}}}", ones(f.e), zeros(f.m)));
        let zero = self.wire(f.bits(), format!("{{{s}, {}}}", zeros(f.bits() - 1)));
        Ok(self.wire(
            f.bits(),
            format!("{top} ? {special} : ({big} ? {inf} : ({tiny} ? {zero} : {normal}))"),
        ))
    }

    /// Round a value of the IEEE format `i` (wider than `o`) to the 8-bit format `o` to nearest, ties
    /// to even: `o`'s subnormal step, NaN kept; beyond the largest finite value and for infinities the
    /// largest finite value of the sign when `saturate` (a conversion, design §6.4), else infinity
    /// (IEEE `o`) or NaN (finite-NaN `o`: an E4M3FN arithmetic result beyond 448 is NaN). The same
    /// steps as `cc_smt::float::from_f32`, on `i`'s bits.
    pub fn encode_fp8(&mut self, i: Format, o: Format, x: &str, saturate: bool, m: Mutation) -> Res<String> {
        if i.flavour != Flavour::Ieee || o.bits() != 8 || i.m <= o.m || i.e < o.e {
            return Err(FloatError::Glue(format!(
                "no FP8 encoder from {} to {}",
                i.name(),
                o.name()
            )));
        }
        let s = self.sign(i, x);
        let e = Self::exp(i, x);
        let mm = Self::man(i, x);
        // Working width: the significand with room for the guard bit and the largest shift.
        let sw = i.m + 4;
        let kw = 8u32;
        let kmax = u128::from(i.m + 3);
        let ez = self.wire(1, format!("~(|{e})"));
        let eeff = self.wire(i.e, format!("{ez} ? {} : {e}", lit(1, i.e)));
        let sig = self.wire(sw, format!("{{{}, ~{ez}, {mm}}}", zeros(sw - i.m - 1)));
        // Normal in `o` iff the unbiased exponent is at least emin(o).
        let t = (o.emin() + i.bias()) as u128;
        let normal = self.wire(1, format!("{eeff} >= {}", lit(t, i.e)));
        let kn = u128::from(i.m - o.m);
        let ew = i.e.max(kw) + 1;
        let ks_full = self.wire(ew, format!("{} - {}", lit(kn + t, ew), zext(&eeff, i.e, ew)));
        let ks = self.wire(
            kw,
            format!(
                "({ks_full} > {}) ? {} : {ks_full}[{}:0]",
                lit(kmax, ew),
                lit(kmax, kw),
                kw - 1
            ),
        );
        let k = self.wire(kw, format!("{normal} ? {} : {ks}", lit(kn, kw)));
        let q = self.wire(sw, format!("{sig} >> {k}"));
        let km1 = self.wire(kw, format!("{k} - {}", lit(1, kw)));
        let g = self.wire(1, format!("({sig} >> {km1}) & {}", lit(1, sw)));
        let mask = self.wire(sw, format!("({} << {km1}) - {}", lit(1, sw), lit(1, sw)));
        let sticky = self.wire(1, format!("|({sig} & {mask})"));
        let up = if m == Mutation::TruncatingEncoder {
            self.wire(1, "1'b0")
        } else {
            self.wire(1, format!("{g} & ({sticky} | {q}[0])"))
        };
        let pw = (i.e + o.m + 3).max(sw);
        let n = self.wire(pw, format!("{} + {}", zext(&q, sw, pw), zext(&up, 1, pw)));
        let rebias = (i.bias() - o.bias() + 1) as u128;
        let eb = self.wire(pw, format!("{} - {}", zext(&eeff, i.e, pw), lit(rebias, pw)));
        let pn = self.wire(pw, format!("({eb} << {}) + {n}", o.m));
        let packed = self.wire(pw, format!("{normal} ? {pn} : {n}"));
        let maxf = max_finite(o);
        let nanm = lit(nan_bits(o), o.bits() - 1);
        let over = self.wire(1, format!("{packed} > {}", lit(maxf, pw)));
        let big = if saturate {
            lit(maxf, o.bits() - 1)
        } else {
            match o.flavour {
                Flavour::Ieee => lit(((1u128 << o.e) - 1) << o.m, o.bits() - 1),
                Flavour::FiniteNan => nanm.clone(),
            }
        };
        let inf = self.is_inf(i, x);
        let nan = self.is_nan(i, x);
        let ovf = if m == Mutation::NoOverflowCase {
            inf.clone()
        } else {
            self.wire(1, format!("{inf} | {over}"))
        };
        let mag = self.wire(
            o.bits() - 1,
            format!("{nan} ? {nanm} : ({ovf} ? {big} : {packed}[{}:0])", o.bits() - 2),
        );
        Ok(self.wire(o.bits(), format!("{{{s}, {mag}}}")))
    }

    /// An IEEE 8-bit result with infinities replaced by the largest finite value of their sign (the
    /// saturating conversion of design §6.4 after a round-to-nearest-even core).
    pub fn saturate(&mut self, f: Format, z: &str) -> String {
        let inf = self.is_inf(f, z);
        let s = self.sign(f, z);
        let max = lit(max_finite(f), f.bits() - 1);
        self.wire(f.bits(), format!("{inf} ? {{{s}, {max}}} : {z}"))
    }

    /// `+1.0` of `f` with the sign of `x` (the `sign` op on a finite non-zero value).
    pub fn unit(&mut self, f: Format, x: &str) -> String {
        let s = self.sign(f, x);
        let one = (f.bias() as u128) << f.m;
        self.wire(f.bits(), format!("{{{s}, {}}}", lit(one, f.bits() - 1)))
    }
}

/// The canonical NaN of `f` with sign 0, as bits.
pub fn nan_bits(f: Format) -> u128 {
    let e = (1u128 << f.e) - 1;
    let m = match f.flavour {
        Flavour::Ieee => 1u128 << (f.m - 1),
        Flavour::FiniteNan => (1u128 << f.m) - 1,
    };
    (e << f.m) | m
}

/// The largest finite magnitude of `f` (bits without the sign).
pub fn max_finite(f: Format) -> u128 {
    match f.flavour {
        Flavour::Ieee => (((1u128 << f.e) - 2) << f.m) | ((1u128 << f.m) - 1),
        Flavour::FiniteNan => (((1u128 << f.e) - 1) << f.m) | ((1u128 << f.m) - 2),
    }
}

/// Deliberate faults for the negative controls of the proofs (each must give `sat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mutation {
    None,
    /// The FTZ wrapper without its flush (results below `2^emin` narrowed as if normal).
    NoFlush,
    /// The FP8 encoder rounding toward zero.
    TruncatingEncoder,
    /// The FP8 encoder without its overflow case.
    NoOverflowCase,
    /// A missing NaN case (maximum/minimum without NaN propagation; float to integer without NaN to 0).
    NoNanCase,
    /// maximum/minimum without the signed-zero order.
    NoZeroOrder,
}
