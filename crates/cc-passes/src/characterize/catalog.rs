//! The primitive set (docs/passes/characterize.md §4.1): every element-wise op of the op table at every
//! configured type for which `cc_smt::ops` defines its semantics, plus the structural primitives the
//! merged units add (multiplexers, registers, accumulators, split multipliers). Each primitive carries its
//! RTL source: DSLX mirroring the SMT semantics, or a small Verilog template; float RTL comes from XLS's
//! `apfloat` library and is provisional; an op without an RTL source is pending.

use std::collections::BTreeSet;

use cc_ir::ops::{Arity, ElemRule, TABLE};
use cc_smt::float::{Flavour, Format};
use cc_smt::ops::{Attrs, Ty};
use cc_smt::term::Ctx;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::config::CharacterizeConfig;

/// Bump when a generator's output changes (enters the cache key through `catalog.json`).
pub const CATALOG_VERSION: u32 = 3;

/// Why a primitive has no RTL yet.
pub const PENDING_SOFTFLOAT: &str = "softfloat: no apfloat function for this op";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// One element-wise op at one type (or type pair, for `convert`).
    Op,
    /// An N-input multiplexer with a binary select.
    Mux,
    /// A register (D flip-flops, no reset).
    Register,
    /// A commutative combiner with its result register in the feedback loop.
    Accumulator,
    /// A wrapping w-bit multiply composed of n-bit unsigned piece multipliers and a shift-add tree.
    Split,
    /// The fixed-point requantization primitive (`cc_smt::ops::rescale`): an integer times an f32 scale
    /// by an exact integer multiply and a rounding shift, back to f32.
    Rescale,
}

/// The RTL of a primitive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", tag = "lang", content = "text")]
pub enum Source {
    /// A DSLX module; `ir_converter_main` → `opt_main` → `codegen_main` make `rtl/<id>.v`.
    Dslx(String),
    /// Verilog written as is to `rtl/<id>.v`.
    Verilog(String),
}

/// One primitive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Primitive {
    /// Stable key: `<op>[.<variant>]@<type>[><type>]`, e.g. `add@i32`, `compare.LT@ui8`,
    /// `convert@f32>bf16`, `mux4@w8`, `register@w8`, `accumulator.add@i32`, `split@w32n8`,
    /// `rescale@i32`.
    pub key: String,
    /// The Verilog module and file stem.
    pub id: String,
    pub kind: Kind,
    pub op: String,
    pub variant: Option<String>,
    /// `bool`, `sint`, `uint`, `float`, or `bits` for structural primitives.
    pub family: String,
    /// Operand type names (`i32`, `ui8`, `i1`, `f32`, …; `w<n>` for bit vectors); for `convert` the source
    /// then the target.
    pub types: Vec<String>,
    /// The width that trends are taken over: result bits (the source width for `convert`; the input
    /// count `n` of a mux is part of its op name).
    pub width: u32,
    /// The RTL is a stand-in until the soft-float library replaces it (XLS `apfloat`, which flushes
    /// subnormals and has IEEE special values for every format).
    pub provisional: bool,
    /// Why there is no RTL yet (no rows are measured).
    pub pending: Option<String>,
    /// Ids of primitives whose RTL this one instantiates (their `rtl/<id>.v` is read with it).
    pub deps: Vec<String>,
    /// The design has a `clk` port.
    pub clocked: bool,
    pub source: Option<Source>,
}

/// A type of the catalog with its DSLX spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct T(Ty);

impl T {
    fn name(self) -> String {
        self.0.name()
    }

    fn family(self) -> &'static str {
        match self.0 {
            Ty::Bool => "bool",
            Ty::Int { signed: true, .. } => "sint",
            Ty::Int { signed: false, .. } => "uint",
            Ty::Float(_) | Ty::Complex(_) => "float",
        }
    }

    fn bits(self) -> u32 {
        match self.0 {
            Ty::Bool => 1,
            Ty::Int { w, .. } => w,
            Ty::Float(f) | Ty::Complex(f) => f.bits(),
        }
    }

    fn dslx(self) -> String {
        match self.0 {
            Ty::Bool => "bool".into(),
            Ty::Int { signed: true, w } => format!("sN[{w}]"),
            Ty::Int { signed: false, w } => format!("uN[{w}]"),
            Ty::Float(f) | Ty::Complex(f) => format!("apfloat::APFloat<{}, {}>", f.e, f.m),
        }
    }

    fn is_float(self) -> bool {
        matches!(self.0, Ty::Float(_))
    }
}

/// A float format by its short name (`f32`, `f16`, `bf16`, `f8e4m3fn`, `f8e5m2`).
pub fn format(name: &str) -> Option<Format> {
    Some(match name {
        "f32" => Format::F32,
        "f16" => Format::F16,
        "bf16" => Format::BF16,
        "f8e4m3fn" => Format::E4M3FN,
        "f8e5m2" => Format::E5M2,
        _ => return None,
    })
}

/// The scalar types of the catalog, in a fixed order: bool, then signed and unsigned integers by width,
/// then the float formats as configured.
fn types(cfg: &CharacterizeConfig) -> Result<Vec<T>, String> {
    let mut v = vec![T(Ty::Bool)];
    for signed in [true, false] {
        for &w in &cfg.int_widths {
            if w < 2 {
                return Err(format!("int_widths: {w} (1 bit is the bool type)"));
            }
            v.push(T(Ty::Int { signed, w }));
        }
    }
    for f in &cfg.float_formats {
        let fmt = format(f).ok_or_else(|| format!("float_formats: unknown format {f}"))?;
        if !fmt.fits_f32() {
            return Err(format!("float_formats: {f} does not fit Float32"));
        }
        v.push(T(Ty::Float(fmt)));
    }
    Ok(v)
}

/// The comparison directions with their own circuits (`GT`/`GE` are `LT`/`LE` with exchanged operands).
pub const DIRECTIONS: [&str; 4] = ["EQ", "NE", "LT", "LE"];

/// Does `cc_smt::ops` give `op` a meaning on these operand types and result type?
fn defined(op: &str, attrs: &Attrs, args: &[T], ret: T) -> bool {
    let mut c = Ctx::new();
    let mut vals = Vec::new();
    for (i, t) in args.iter().enumerate() {
        match cc_smt::ops::input(&mut c, &format!("x{i}"), t.0) {
            Ok(v) => vals.push(v),
            Err(_) => return false,
        }
    }
    cc_smt::ops::apply(&mut c, op, attrs, &vals, ret.0).is_ok()
}

/// The whole catalog, in a deterministic order.
pub fn catalog(cfg: &CharacterizeConfig) -> Result<Vec<Primitive>, String> {
    let tys = types(cfg)?;
    let mut out: Vec<Primitive> = Vec::new();
    let bool_t = T(Ty::Bool);
    for spec in TABLE.iter().filter(|s| s.elementwise) {
        let op = spec.name.trim_start_matches("stablehlo.");
        if !cfg.ops.is_empty() && !cfg.ops.iter().any(|o| o == op) {
            continue;
        }
        let arity = match spec.operands {
            Arity::Exactly(k) => k as usize,
            Arity::AtLeast(_) => continue,
        };
        match op {
            // Complex numbers are pairs of float rows; `real`/`imag`/`complex` are wiring.
            "complex" | "real" | "imag" => continue,
            "convert" => {
                for &a in &tys {
                    for &b in &tys {
                        if a != b && defined(op, &Attrs::default(), &[a], b) {
                            out.push(op_primitive(op, None, &[a], a, b, b.bits().max(a.bits())));
                        }
                    }
                }
                continue;
            }
            "compare" => {
                for &t in &tys {
                    for d in DIRECTIONS {
                        let attrs = Attrs {
                            direction: Some(d.into()),
                            ..Attrs::default()
                        };
                        if defined(op, &attrs, &[t, t], bool_t) {
                            out.push(op_primitive(op, Some(d), &[t, t], t, bool_t, t.bits()));
                        }
                    }
                }
                continue;
            }
            _ => {}
        }
        for &t in &tys {
            let (args, ret): (Vec<T>, T) = match (op, spec.elem) {
                ("select", _) => (vec![bool_t, t, t], t),
                ("is_finite", _) => (vec![t], bool_t),
                (_, ElemRule::Bool) => (vec![t; arity], bool_t),
                _ => (vec![t; arity], t),
            };
            if op == "select" && t == bool_t {
                // A 1-bit mux: the `mux2@w1` row.
                continue;
            }
            let attrs = Attrs {
                precision: (op == "reduce_precision").then_some((5, 10)),
                ..Attrs::default()
            };
            if defined(op, &attrs, &args, ret) {
                out.push(op_primitive(op, None, &args, t, ret, t.bits()));
            }
        }
    }

    // Structural primitives.
    for &n in &cfg.mux_inputs {
        if n < 2 {
            return Err(format!("mux_inputs: {n} (at least 2)"));
        }
        for &w in &cfg.mux_widths {
            out.push(mux(n, w));
        }
    }
    for &w in &cfg.register_widths {
        out.push(register(w));
    }
    let combiners: Vec<&str> = TABLE
        .iter()
        .filter(|s| s.elementwise && s.commutative)
        .map(|s| s.name.trim_start_matches("stablehlo."))
        .collect();
    let ops: Vec<Primitive> = out.clone();
    for c in &combiners {
        for p in ops
            .iter()
            .filter(|p| p.kind == Kind::Op && p.op == *c && p.pending.is_none())
        {
            out.push(accumulator(p));
        }
    }
    for &(w, n) in &cfg.split_multipliers {
        if n == 0 || n >= w {
            return Err(format!("split_multipliers: ({w}, {n}) needs 0 < n < w"));
        }
        out.push(split(w, n));
    }
    // After every other primitive, so that adding them keeps the earlier job batches.
    let conv = "convert_i32_f32";
    if out.iter().any(|p| p.id == conv && p.pending.is_none()) {
        for &t in &tys {
            if let Ty::Int { signed: true, w } = t.0
                && w <= cc_smt::ops::RESCALE_MAX_BITS
            {
                out.push(rescale(w, t, conv));
            }
        }
    }
    let mut seen = BTreeSet::new();
    for p in &out {
        if !seen.insert(p.id.clone()) {
            return Err(format!("two primitives with id {}", p.id));
        }
    }
    Ok(out)
}

fn ident(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn op_primitive(op: &str, variant: Option<&str>, args: &[T], t: T, ret: T, width: u32) -> Primitive {
    let types: Vec<String> = if op == "convert" {
        vec![t.name(), ret.name()]
    } else {
        vec![t.name()]
    };
    let key = format!(
        "{op}{}@{}",
        variant.map(|v| format!(".{v}")).unwrap_or_default(),
        types.join(">")
    );
    let id = ident(&format!(
        "{op}{}_{}",
        variant.map(|v| format!("_{v}")).unwrap_or_default(),
        types.join("_")
    ));
    let float = args.iter().chain([&ret]).any(|x| x.is_float());
    let body = dslx_body(op, variant, args, ret);
    let (source, pending) = match body {
        Some(b) => (Some(Source::Dslx(dslx_module(&id, args, ret, &b))), None),
        None => (None, Some(PENDING_SOFTFLOAT.to_string())),
    };
    Primitive {
        key,
        id,
        kind: Kind::Op,
        op: op.into(),
        variant: variant.map(str::to_string),
        family: t.family().into(),
        types,
        width,
        provisional: float && pending.is_none(),
        pending,
        deps: Vec::new(),
        clocked: false,
        source,
    }
}

const ARGS: [&str; 3] = ["a", "b", "c"];

fn dslx_module(id: &str, args: &[T], ret: T, body: &str) -> String {
    let float = args.iter().chain([&ret]).any(|x| x.is_float());
    let mut s = String::new();
    if float {
        s.push_str("import apfloat;\n");
    }
    if body.contains("std::") {
        s.push_str("import std;\n");
    }
    if !s.is_empty() {
        s.push('\n');
    }
    let params: Vec<String> = args
        .iter()
        .zip(ARGS)
        .map(|(t, n)| format!("{n}: {}", t.dslx()))
        .collect();
    s.push_str(&format!(
        "pub fn {id}({}) -> {} {{\n{body}\n}}\n",
        params.join(", "),
        ret.dslx()
    ));
    s
}

/// The body of an op's DSLX function over parameters `a`, `b`, `c`, mirroring `cc_smt::ops`; `None` when
/// the op has no RTL source here (float ops without an `apfloat` function).
fn dslx_body(op: &str, variant: Option<&str>, args: &[T], ret: T) -> Option<String> {
    let t = args[if op == "select" { 1 } else { 0 }];
    let line = |s: String| Some(format!("    {s}"));
    if op == "convert" {
        return convert_body(t, ret).map(|s| format!("    {s}"));
    }
    if op == "select" {
        return line("if a { b } else { c }".into());
    }
    match t.0 {
        Ty::Bool => line(
            match (op, variant) {
                ("and" | "minimum" | "multiply", _) => "a & b",
                ("or" | "maximum" | "add", _) => "a | b",
                ("xor", _) => "a ^ b",
                ("not", _) => "!a",
                // `clamp(min, x, max)` = `minimum(maximum(x, min), max)`: OR, then AND.
                ("clamp", _) => "(b | a) & c",
                ("compare", Some("EQ")) => "a == b",
                ("compare", Some("NE")) => "a != b",
                ("compare", Some("LT")) => "!a & b",
                ("compare", Some("LE")) => "!a | b",
                _ => return None,
            }
            .into(),
        ),
        Ty::Int { signed, w } => {
            let ty = t.dslx();
            let zero = format!("{ty}:0");
            line(match (op, variant) {
                ("add", _) => "a + b".into(),
                ("subtract", _) => "a - b".into(),
                ("multiply", _) => "a * b".into(),
                ("and", _) => "a & b".into(),
                ("or", _) => "a | b".into(),
                ("xor", _) => "a ^ b".into(),
                ("not", _) => "!a".into(),
                ("negate", _) => "-a".into(),
                ("maximum", _) => "if a < b { b } else { a }".into(),
                ("minimum", _) => "if a < b { a } else { b }".into(),
                ("clamp", _) => "let lo = if b < a { a } else { b };\n    if lo < c { lo } else { c }".into(),
                ("abs", _) if signed => format!("if a < {zero} {{ -a }} else {{ a }}"),
                ("sign", _) if signed => {
                    format!("if a < {zero} {{ {ty}:-1 }} else if a == {zero} {{ {zero} }} else {{ {ty}:1 }}")
                }
                ("sign", _) => format!("if a == {zero} {{ {zero} }} else {{ {ty}:1 }}"),
                // XLA: x / 0 = -1, INT_SMIN / -1 = INT_SMIN (as `cc_smt::ops`).
                ("divide", _) if signed => format!(
                    "if b == {zero} {{ {ty}:-1 }} else if a == std::signed_min_value<u32:{w}>() && b == {ty}:-1 {{ a }} else {{ a / b }}"
                ),
                ("divide", _) => {
                    format!("if b == {zero} {{ std::unsigned_max_value<u32:{w}>() }} else {{ a / b }}")
                }
                // XLA: x % 0 = x, INT_SMIN % -1 = 0; the sign follows the dividend.
                ("remainder", _) if signed => format!(
                    "if b == {zero} {{ a }} else if a == std::signed_min_value<u32:{w}>() && b == {ty}:-1 {{ {zero} }} else {{ a % b }}"
                ),
                ("remainder", _) => format!("if b == {zero} {{ a }} else {{ a % b }}"),
                ("compare", Some("EQ")) => "a == b".into(),
                ("compare", Some("NE")) => "a != b".into(),
                ("compare", Some("LT")) => "a < b".into(),
                ("compare", Some("LE")) => "a <= b".into(),
                _ => return None,
            })
        }
        Ty::Float(f) => {
            let (e, m) = (f.e, f.m);
            let zero = "a.bexp == bits[E]:0 && a.fraction == bits[M]:0"
                .replace("[E]", &format!("[{e}]"))
                .replace("[M]", &format!("[{m}]"));
            line(match (op, variant) {
                ("add", _) => "apfloat::add(a, b)".into(),
                ("subtract", _) => "apfloat::sub(a, b)".into(),
                ("multiply", _) => "apfloat::mul(a, b)".into(),
                ("negate", _) => "apfloat::negate(a)".into(),
                ("abs", _) => "apfloat::abs(a)".into(),
                // IEEE 754 maximum/minimum (spec): NaN propagates, -0 < +0.
                ("maximum", _) => float_max("a", "b"),
                ("minimum", _) => float_min("a", "b"),
                ("clamp", _) => format!("let lo = {};\n    {}", float_max("b", "a"), float_min("lo", "c")),
                ("compare", Some("EQ")) => "apfloat::eq_2(a, b)".into(),
                ("compare", Some("NE")) => "!apfloat::eq_2(a, b)".into(),
                ("compare", Some("LT")) => "apfloat::lt_2(a, b)".into(),
                ("compare", Some("LE")) => "apfloat::lte_2(a, b)".into(),
                ("sign", _) => format!(
                    "if apfloat::is_nan(a) || ({zero}) {{ a }} else {{ apfloat::one<u32:{e}, u32:{m}>(a.sign) }}"
                ),
                ("floor", _) => "apfloat::floor_with_denorms(a)".into(),
                ("ceil", _) => "apfloat::ceil_with_denorms(a)".into(),
                ("round_nearest_even", _) => "apfloat::round(a)".into(),
                ("is_finite", _) => "!(apfloat::is_nan(a) || apfloat::is_inf(a))".into(),
                _ => return None,
            })
        }
        Ty::Complex(_) => None,
    }
}

fn float_max(a: &str, b: &str) -> String {
    format!(
        "if apfloat::is_nan({a}) {{ {a} }} else if apfloat::is_nan({b}) {{ {b} }} else if apfloat::lt_2({a}, {b}) || (apfloat::eq_2({a}, {b}) && {a}.sign == u1:1) {{ {b} }} else {{ {a} }}"
    )
}

fn float_min(a: &str, b: &str) -> String {
    format!(
        "if apfloat::is_nan({a}) {{ {a} }} else if apfloat::is_nan({b}) {{ {b} }} else if apfloat::lt_2({a}, {b}) || (apfloat::eq_2({a}, {b}) && {a}.sign == u1:1) {{ {a} }} else {{ {b} }}"
    )
}

/// `convert` from `a` to `b` (spec `convert`; integer conversions wrap, float conversions round to
/// nearest even, float to integer as `apfloat`'s saturating casts, bool is `x != 0`).
fn convert_body(a: T, b: T) -> Option<String> {
    Some(match (a.0, b.0) {
        (Ty::Bool, Ty::Int { .. }) => format!("a as {}", b.dslx()),
        (Ty::Int { .. }, Ty::Bool) => format!("a != {}:0", a.dslx()),
        (Ty::Int { .. }, Ty::Int { .. }) => format!("a as {}", b.dslx()),
        (Ty::Bool, Ty::Float(f)) => format!(
            "if a {{ apfloat::one<u32:{e}, u32:{m}>(u1:0) }} else {{ apfloat::zero<u32:{e}, u32:{m}>(u1:0) }}",
            e = f.e,
            m = f.m
        ),
        (Ty::Float(f), Ty::Bool) => {
            format!("!(a.bexp == bits[{}]:0 && a.fraction == bits[{}]:0)", f.e, f.m)
        }
        (Ty::Int { signed: true, .. }, Ty::Float(f)) => {
            format!("apfloat::cast_from_fixed_using_rne<u32:{}, u32:{}>(a)", f.e, f.m)
        }
        (Ty::Int { signed: false, w }, Ty::Float(f)) => format!(
            "apfloat::cast_from_fixed_using_rne<u32:{}, u32:{}>(a as sN[{}])",
            f.e,
            f.m,
            w + 1
        ),
        (Ty::Float(_), Ty::Int { signed: true, w }) => {
            format!("apfloat::to_int<{}, u32:{w}>(a)", float_params(a))
        }
        (Ty::Float(_), Ty::Int { signed: false, w }) => format!("apfloat::to_uint<u32:{w}>(a)"),
        (Ty::Float(f), Ty::Float(g)) => {
            if g.e >= f.e && g.m >= f.m {
                format!("apfloat::upcast_with_denorms<u32:{}, u32:{}>(a)", g.e, g.m)
            } else if g.e <= f.e && g.m <= f.m {
                format!("apfloat::downcast_rne<u32:{}, u32:{}>(a)", g.m, g.e)
            } else {
                // Neither contains the other: widen exactly to Float32, then round once.
                format!(
                    "apfloat::downcast_rne<u32:{}, u32:{}>(apfloat::upcast_with_denorms<u32:8, u32:23>(a))",
                    g.m, g.e
                )
            }
        }
        _ => return None,
    })
}

fn float_params(t: T) -> String {
    match t.0 {
        Ty::Float(f) => format!("u32:{}, u32:{}", f.e, f.m),
        _ => String::new(),
    }
}

fn mux(n: u32, w: u32) -> Primitive {
    let id = format!("mux{n}_w{w}");
    let sel = (32 - (n - 1).leading_zeros()).max(1);
    let dslx = format!("pub fn {id}(sel: uN[{sel}], a: uN[{w}][{n}]) -> uN[{w}] {{\n    a[sel]\n}}\n");
    Primitive {
        key: format!("mux{n}@w{w}"),
        id,
        kind: Kind::Mux,
        op: format!("mux{n}"),
        variant: None,
        family: "bits".into(),
        types: vec![format!("w{w}")],
        width: w,
        provisional: false,
        pending: None,
        deps: Vec::new(),
        clocked: false,
        source: Some(Source::Dslx(dslx)),
    }
}

fn register(w: u32) -> Primitive {
    let id = format!("register_w{w}");
    let v = format!(
        "module {id}(input wire clk, input wire [{h}:0] d, output reg [{h}:0] q);\n  always @(posedge clk) q <= d;\nendmodule\n",
        h = w - 1
    );
    Primitive {
        key: format!("register@w{w}"),
        id,
        kind: Kind::Register,
        op: "register".into(),
        variant: None,
        family: "bits".into(),
        types: vec![format!("w{w}")],
        width: w,
        provisional: false,
        pending: None,
        deps: Vec::new(),
        clocked: true,
        source: Some(Source::Verilog(v)),
    }
}

/// The ports of an XLS combinational module: inputs `a`, `b`, output `out`; float values flatten to their
/// bit patterns (sign, exponent, fraction).
/// The combiner with its result register in the loop. `start` loads the first element (a reduction starts
/// from a defined value): without it nothing initializes the register, and synthesis may assume a start
/// state and remove it (a multiply accumulator from 0 stays 0).
fn accumulator(c: &Primitive) -> Primitive {
    let ty = &c.types[0];
    let w = format_width(ty);
    let id = format!("accumulator_{}", c.id);
    let h = w - 1;
    let v = format!(
        "module {id}(input wire clk, input wire start, input wire [{h}:0] x, output reg [{h}:0] acc);\n  wire [{h}:0] next;\n  {comb} u_comb(.a(acc), .b(x), .out(next));\n  always @(posedge clk) acc <= start ? x : next;\nendmodule\n",
        comb = c.id
    );
    Primitive {
        key: format!("accumulator.{}@{ty}", c.op),
        id,
        kind: Kind::Accumulator,
        op: format!("accumulator.{}", c.op),
        variant: None,
        family: c.family.clone(),
        types: vec![ty.clone()],
        width: c.width,
        provisional: c.provisional,
        pending: None,
        deps: vec![c.id.clone()],
        clocked: true,
        source: Some(Source::Verilog(v)),
    }
}

fn format_width(t: &str) -> u32 {
    if t == "i1" {
        return 1;
    }
    if let Some(f) = format(t) {
        return f.bits();
    }
    t.trim_start_matches("ui")
        .trim_start_matches('i')
        .parse()
        .unwrap_or(0)
}

/// `split@w<w>n<n>`: the low `w` bits of `a * b` from `ceil(w/n)` pieces per operand, every partial
/// product `a_i * b_j` with `(i + j) * n < w` (the others only reach bits above `w`) on its own `n x n`
/// unsigned multiplier instance, shifted and summed. The low `w` bits of a product are the same for
/// signed and unsigned operands, so one split serves both families (docs/passes/characterize.md §4.1).
fn split(w: u32, n: u32) -> Primitive {
    let id = format!("split_w{w}n{n}");
    let piece = format!("split_piece_n{n}");
    let k = w.div_ceil(n);
    let mut v = format!(
        "module {piece}(input wire [{a}:0] a, input wire [{a}:0] b, output wire [{p}:0] p);\n  assign p = a * b;\nendmodule\n\n",
        a = n - 1,
        p = 2 * n - 1
    );
    let wide = k * n;
    v.push_str(&format!(
        "module {id}(input wire [{h}:0] a, input wire [{h}:0] b, output wire [{h}:0] out);\n  wire [{wh}:0] ax = a;\n  wire [{wh}:0] bx = b;\n",
        h = w - 1,
        wh = wide - 1
    ));
    let mut terms = Vec::new();
    for i in 0..k {
        for j in 0..k {
            if (i + j) * n >= w {
                continue;
            }
            v.push_str(&format!(
                "  wire [{p}:0] p{i}_{j};\n  {piece} u{i}_{j}(.a(ax[{ah}:{al}]), .b(bx[{bh}:{bl}]), .p(p{i}_{j}));\n",
                p = 2 * n - 1,
                ah = (i + 1) * n - 1,
                al = i * n,
                bh = (j + 1) * n - 1,
                bl = j * n
            ));
            terms.push(format!("({{{{{w}{{1'b0}}}}, p{i}_{j}}} << {})", (i + j) * n));
        }
    }
    v.push_str(&format!(
        "  wire [{s}:0] sum = {};\n  assign out = sum[{h}:0];\nendmodule\n",
        terms.join("\n    + "),
        s = w + 2 * n - 1,
        h = w - 1
    ));
    Primitive {
        key: format!("split@w{w}n{n}"),
        id,
        kind: Kind::Split,
        op: "split".into(),
        variant: Some(format!("n{n}")),
        family: "bits".into(),
        types: vec![format!("w{w}")],
        width: w,
        provisional: false,
        pending: None,
        deps: Vec::new(),
        clocked: false,
        source: Some(Source::Verilog(v)),
    }
}

/// `rescale@i<n>` (`cc_smt::ops::rescale`, docs/passes/characterize.md §4.1): `rescale_core_i<n>` decodes
/// the f32 scale `b` (significand `M`, shift `k = 150 - max(e, 1)`), forms the exact product `a · M` in
/// `n + 25` bits, shifts it right with rounding half up (at most `n + 24` places) or left (at most 7
/// places, saturating beyond) and saturates to i32; `rescale_i<n>` converts that to f32 with the
/// characterized `convert@i32>f32` design.
fn rescale(n: u32, t: T, conv: &str) -> Primitive {
    let id = format!("rescale_i{n}");
    let w = n + 25;
    let w2 = n + 33;
    let v = format!(
        "module rescale_core_i{n}(input wire [{nh}:0] a, input wire [31:0] b, output wire [31:0] q);
  wire sgn = b[31];
  wire [7:0] e = b[30:23];
  wire ez = (e == 8'd0);
  wire [23:0] mant = {{~ez, b[22:0]}};
  wire signed [9:0] k = 10'sd150 - $signed({{2'b00, (ez ? 8'd1 : e)}});
  wire signed [{wh}:0] xw = {{{{{xe}{{a[{nh}]}}}}, a}};
  wire signed [{wh}:0] mw = {{{{{me}{{1'b0}}}}, mant}};
  wire signed [{wh}:0] p0 = xw * mw;
  wire signed [{wh}:0] p = sgn ? -p0 : p0;
  wire right = k > 10'sd0;
  wire signed [9:0] kk = (k > 10'sd{cap}) ? 10'sd{cap} : k;
  wire signed [{wh}:0] one = {{{{{oe}{{1'b0}}}}, 1'b1}};
  wire signed [{wh}:0] half = one << (kk - 10'sd1);
  wire signed [{wh}:0] sum = p + half;
  wire signed [{wh}:0] qr = sum >>> kk;
  wire [9:0] l = -k;
  wire near = l <= 10'd7;
  wire signed [{w2h}:0] p2 = {{{{8{{p[{wh}]}}}}, p}};
  wire signed [{w2h}:0] shifted = p2 <<< l[2:0];
  wire signed [{w2h}:0] bigp = {{{{{be}{{1'b0}}}}, 1'b1, 32'b0}};
  wire signed [{w2h}:0] big = (p == 0) ? {{{w2}{{1'b0}}}} : (p[{wh}] ? -bigp : bigp);
  wire signed [{w2h}:0] qr2 = {{{{8{{qr[{wh}]}}}}, qr}};
  wire signed [{w2h}:0] qq = right ? qr2 : (near ? shifted : big);
  wire signed [{w2h}:0] maxv = {{{{{se}{{1'b0}}}}, 1'b0, {{31{{1'b1}}}}}};
  wire signed [{w2h}:0] minv = {{{{{se}{{1'b1}}}}, 1'b1, {{31{{1'b0}}}}}};
  assign q = (qq > maxv) ? 32'h7fffffff : ((qq < minv) ? 32'h80000000 : qq[31:0]);
endmodule

module {id}(input wire [{nh}:0] a, input wire [31:0] b, output wire [31:0] out);
  wire [31:0] q;
  rescale_core_i{n} u_core(.a(a), .b(b), .q(q));
  {conv} u_conv(.a(q), .out(out));
endmodule
",
        nh = n - 1,
        wh = w - 1,
        w2h = w2 - 1,
        xe = w - n,
        me = w - 24,
        oe = w - 1,
        cap = n + 24,
        be = w2 - 33,
        se = w2 - 32,
    );
    Primitive {
        key: format!("rescale@{}", t.name()),
        id,
        kind: Kind::Rescale,
        op: "rescale".into(),
        variant: None,
        family: "sint".into(),
        types: vec![t.name(), "f32".into()],
        width: n,
        provisional: true,
        pending: None,
        deps: vec![conv.to_string()],
        clocked: false,
        source: Some(Source::Verilog(v)),
    }
}

/// The pending float ops of a catalog, by op.
pub fn pending(cat: &[Primitive]) -> Vec<&Primitive> {
    cat.iter().filter(|p| p.pending.is_some()).collect()
}

/// The float flavour note for E4M3FN-like formats (no infinities) that `apfloat` does not model.
pub fn finite_nan(f: Format) -> bool {
    f.flavour == Flavour::FiniteNan
}
