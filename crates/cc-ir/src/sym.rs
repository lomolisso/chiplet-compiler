//! Symbolic dimension expressions: JAX's algebra (`jax/_src/export/shape_poly.py`, jax 0.12.0).
//!
//! An expression is a sum of terms `k * m` (integer `k`, monomial `m`); a monomial is a product of atoms
//! with positive powers; an atom is a dimension variable or one of `floordiv(e1, e2)`, `mod(e1, e2)`,
//! `max(e1, e2)`, `min(e1, e2)` with Python semantics (`floordiv` rounds toward −∞, `mod` has the sign
//! of the divisor). The representation is canonical: terms sorted by monomial, no zero coefficient, atoms
//! sorted, so structural equality is equality of the normal forms. It is not a decision procedure: two
//! equal functions may have different normal forms (then `SymCtx` decides with bounds, or the caller
//! fails).

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// An atom of a monomial.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Atom {
    Var(String),
    FloorDiv(Box<SymExpr>, Box<SymExpr>),
    Mod(Box<SymExpr>, Box<SymExpr>),
    Max(Box<SymExpr>, Box<SymExpr>),
    Min(Box<SymExpr>, Box<SymExpr>),
}

/// A product of atoms with powers (sorted by atom; the empty monomial is 1).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct Mono(pub Vec<(Atom, u32)>);

impl Mono {
    pub fn one() -> Self {
        Mono(Vec::new())
    }
    pub fn is_one(&self) -> bool {
        self.0.is_empty()
    }
    fn mul(&self, o: &Mono) -> Mono {
        let mut m: BTreeMap<Atom, u32> = BTreeMap::new();
        for (a, p) in self.0.iter().chain(o.0.iter()) {
            *m.entry(a.clone()).or_default() += p;
        }
        Mono(m.into_iter().collect())
    }
    fn degree(&self) -> u32 {
        self.0.iter().map(|(_, p)| p).sum()
    }
}

/// A canonical sum of terms `(monomial, coefficient)`, sorted by monomial, coefficients non-zero.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct SymExpr {
    terms: Vec<(Mono, i64)>,
}

impl PartialOrd for SymExpr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SymExpr {
    fn cmp(&self, other: &Self) -> Ordering {
        self.terms.cmp(&other.terms)
    }
}

/// Errors of parsing or arithmetic overflow.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SymError {
    #[error("cannot parse dimension expression `{text}`: {why}")]
    Parse { text: String, why: String },
    #[error("integer overflow in a dimension expression")]
    Overflow,
}

impl SymExpr {
    pub fn zero() -> Self {
        SymExpr { terms: Vec::new() }
    }
    pub fn constant(c: i64) -> Self {
        Self::from_terms([(Mono::one(), c)])
    }
    pub fn var(name: &str) -> Self {
        Self::atom(Atom::Var(name.to_string()))
    }
    fn atom(a: Atom) -> Self {
        Self::from_terms([(Mono(vec![(a, 1)]), 1)])
    }

    fn from_terms(it: impl IntoIterator<Item = (Mono, i64)>) -> Self {
        let mut m: BTreeMap<Mono, i64> = BTreeMap::new();
        for (mono, k) in it {
            let e = m.entry(mono).or_default();
            *e = e.wrapping_add(k);
        }
        SymExpr {
            terms: m.into_iter().filter(|(_, k)| *k != 0).collect(),
        }
    }

    pub fn terms(&self) -> &[(Mono, i64)] {
        &self.terms
    }

    /// The value if the expression is a constant.
    pub fn as_const(&self) -> Option<i64> {
        match self.terms.as_slice() {
            [] => Some(0),
            [(m, k)] if m.is_one() => Some(*k),
            _ => None,
        }
    }

    pub fn is_const(&self) -> bool {
        self.as_const().is_some()
    }

    /// The variable if the expression is exactly one variable.
    pub fn as_var(&self) -> Option<&str> {
        match self.terms.as_slice() {
            [(Mono(atoms), 1)] => match atoms.as_slice() {
                [(Atom::Var(v), 1)] => Some(v),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn add(&self, o: &SymExpr) -> SymExpr {
        Self::from_terms(self.terms.iter().chain(o.terms.iter()).cloned())
    }
    pub fn neg(&self) -> SymExpr {
        Self::from_terms(self.terms.iter().map(|(m, k)| (m.clone(), k.wrapping_neg())))
    }
    pub fn sub(&self, o: &SymExpr) -> SymExpr {
        self.add(&o.neg())
    }
    pub fn add_const(&self, c: i64) -> SymExpr {
        self.add(&SymExpr::constant(c))
    }
    pub fn scale(&self, c: i64) -> SymExpr {
        Self::from_terms(self.terms.iter().map(|(m, k)| (m.clone(), k.wrapping_mul(c))))
    }
    pub fn mul(&self, o: &SymExpr) -> SymExpr {
        let mut out = Vec::new();
        for (m1, k1) in &self.terms {
            for (m2, k2) in &o.terms {
                out.push((m1.mul(m2), k1.wrapping_mul(*k2)));
            }
        }
        Self::from_terms(out)
    }

    /// `floor(self / d)` (Python `//`). Exact identities only: constants fold, and for a constant
    /// divisor `c > 0`, `floor((c*q + r) / c) = q + floor(r / c)` with the coefficients split by
    /// Euclidean division.
    pub fn floordiv(&self, d: &SymExpr) -> SymExpr {
        if let (Some(a), Some(b)) = (self.as_const(), d.as_const())
            && b != 0
        {
            return py_floordiv(a, b).map_or_else(|| self.floordiv_atom(d), SymExpr::constant);
        }
        match d.as_const() {
            Some(1) => return self.clone(),
            Some(c) if c > 1 => {
                let (q, r) = self.split(c);
                if r.terms.is_empty() {
                    return q;
                }
                let tail = match r.as_const() {
                    Some(rc) => SymExpr::constant(rc.div_euclid(c)),
                    None => match r.nested_floordiv() {
                        // floor((floor(x / a) + k) / c) = floor((x + k*a) / (a*c)) for a, c > 0.
                        Some((x, a, k)) => x.add_const(k * a).floordiv(&SymExpr::constant(a * c)),
                        None => Self::atom(Atom::FloorDiv(Box::new(r), Box::new(d.clone()))),
                    },
                };
                return q.add(&tail);
            }
            _ => {}
        }
        if self.terms.is_empty() {
            return SymExpr::zero();
        }
        if let Some(k) = self.multiple_of(d) {
            return SymExpr::constant(k);
        }
        self.floordiv_atom(d)
    }

    fn floordiv_atom(&self, d: &SymExpr) -> SymExpr {
        Self::atom(Atom::FloorDiv(Box::new(self.clone()), Box::new(d.clone())))
    }

    /// `self mod d` (Python `%`): `self - d * floordiv(self, d)` when that is exact.
    pub fn modulo(&self, d: &SymExpr) -> SymExpr {
        if let (Some(a), Some(b)) = (self.as_const(), d.as_const())
            && let Some(m) = py_mod(a, b)
        {
            return SymExpr::constant(m);
        }
        match d.as_const() {
            Some(1) => return SymExpr::zero(),
            Some(c) if c > 1 => {
                let (_, r) = self.split(c);
                return match r.as_const() {
                    Some(rc) => SymExpr::constant(rc.rem_euclid(c)),
                    None => Self::atom(Atom::Mod(Box::new(r), Box::new(d.clone()))),
                };
            }
            _ => {}
        }
        if self.terms.is_empty() || self.multiple_of(d).is_some() {
            return SymExpr::zero();
        }
        Self::atom(Atom::Mod(Box::new(self.clone()), Box::new(d.clone())))
    }

    /// `max(self, o)` without bounds (see `SymCtx::max` for the bound-aware form).
    pub fn max_of(&self, o: &SymExpr) -> SymExpr {
        if self == o {
            return self.clone();
        }
        if let (Some(a), Some(b)) = (self.as_const(), o.as_const()) {
            return SymExpr::constant(a.max(b));
        }
        let (a, b) = if self <= o { (self, o) } else { (o, self) };
        Self::atom(Atom::Max(Box::new(a.clone()), Box::new(b.clone())))
    }

    pub fn min_of(&self, o: &SymExpr) -> SymExpr {
        if self == o {
            return self.clone();
        }
        if let (Some(a), Some(b)) = (self.as_const(), o.as_const()) {
            return SymExpr::constant(a.min(b));
        }
        let (a, b) = if self <= o { (self, o) } else { (o, self) };
        Self::atom(Atom::Min(Box::new(a.clone()), Box::new(b.clone())))
    }

    /// `Some((x, a, k))` when `self = floordiv(x, a) + k` with a constant `a > 0`.
    fn nested_floordiv(&self) -> Option<(SymExpr, i64, i64)> {
        let (consts, rest): (Vec<_>, Vec<_>) = self.terms.iter().partition(|(m, _)| m.is_one());
        let [(Mono(atoms), 1)] = rest.as_slice() else {
            return None;
        };
        let [(Atom::FloorDiv(x, a), 1)] = atoms.as_slice() else {
            return None;
        };
        let a = a.as_const().filter(|&a| a > 0)?;
        Some(((**x).clone(), a, consts.first().map_or(0, |(_, k)| *k)))
    }

    /// `(q, r)` with `self = c*q + r` and every coefficient of `r` in `[0, c)` (Euclidean split).
    fn split(&self, c: i64) -> (SymExpr, SymExpr) {
        let q = Self::from_terms(self.terms.iter().map(|(m, k)| (m.clone(), k.div_euclid(c))));
        let r = Self::from_terms(self.terms.iter().map(|(m, k)| (m.clone(), k.rem_euclid(c))));
        (q, r)
    }

    /// `Some(k)` when `self == k * d` for an integer `k`.
    fn multiple_of(&self, d: &SymExpr) -> Option<i64> {
        let (m0, k0) = d.terms.first()?;
        let (_, ks) = self.terms.iter().find(|(m, _)| m == m0)?;
        if *k0 == 0 || ks % k0 != 0 {
            return None;
        }
        let k = ks / k0;
        (d.scale(k) == *self).then_some(k)
    }

    /// The variables, sorted.
    pub fn vars(&self) -> Vec<String> {
        let mut out = std::collections::BTreeSet::new();
        self.collect_vars(&mut out);
        out.into_iter().collect()
    }
    fn collect_vars(&self, out: &mut std::collections::BTreeSet<String>) {
        for (m, _) in &self.terms {
            for (a, _) in &m.0 {
                match a {
                    Atom::Var(v) => {
                        out.insert(v.clone());
                    }
                    Atom::FloorDiv(x, y) | Atom::Mod(x, y) | Atom::Max(x, y) | Atom::Min(x, y) => {
                        x.collect_vars(out);
                        y.collect_vars(out);
                    }
                }
            }
        }
    }

    /// Evaluate with every variable bound (checked arithmetic; Python division semantics).
    pub fn eval(&self, env: &BTreeMap<String, i64>) -> Result<i64, EvalError> {
        let mut acc: i64 = 0;
        for (m, k) in &self.terms {
            let mut v: i64 = *k;
            for (a, p) in &m.0 {
                let x = a.eval(env)?;
                for _ in 0..*p {
                    v = v.checked_mul(x).ok_or(EvalError::Overflow)?;
                }
            }
            acc = acc.checked_add(v).ok_or(EvalError::Overflow)?;
        }
        Ok(acc)
    }

    /// Parse JAX's dimension-expression syntax: integers, identifiers, `+ - *`, `//`, `%`, parentheses,
    /// and the functions `floordiv`, `mod`, `max`, `min`.
    pub fn parse(text: &str) -> Result<SymExpr, SymError> {
        let mut p = Parser {
            toks: tokenize(text).map_err(|why| SymError::Parse {
                text: text.into(),
                why,
            })?,
            pos: 0,
        };
        let e = p.expr().map_err(|why| SymError::Parse {
            text: text.into(),
            why,
        })?;
        if p.pos != p.toks.len() {
            return Err(SymError::Parse {
                text: text.into(),
                why: format!("unexpected `{:?}`", p.toks[p.pos]),
            });
        }
        Ok(e)
    }
}

impl Atom {
    fn eval(&self, env: &BTreeMap<String, i64>) -> Result<i64, EvalError> {
        match self {
            Atom::Var(v) => env.get(v).copied().ok_or_else(|| EvalError::Unbound(v.clone())),
            Atom::FloorDiv(a, b) => {
                let (a, b) = (a.eval(env)?, b.eval(env)?);
                py_floordiv(a, b).ok_or(EvalError::DivByZero)
            }
            Atom::Mod(a, b) => {
                let (a, b) = (a.eval(env)?, b.eval(env)?);
                py_mod(a, b).ok_or(EvalError::DivByZero)
            }
            Atom::Max(a, b) => Ok(a.eval(env)?.max(b.eval(env)?)),
            Atom::Min(a, b) => Ok(a.eval(env)?.min(b.eval(env)?)),
        }
    }
}

/// Python `a // b`.
pub fn py_floordiv(a: i64, b: i64) -> Option<i64> {
    if b == 0 {
        return None;
    }
    let q = a.checked_div(b)?;
    Some(if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    })
}

/// Python `a % b`.
pub fn py_mod(a: i64, b: i64) -> Option<i64> {
    let q = py_floordiv(a, b)?;
    a.checked_sub(q.checked_mul(b)?)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvalError {
    #[error("unbound dimension variable `{0}`")]
    Unbound(String),
    #[error("division by zero")]
    DivByZero,
    #[error("integer overflow")]
    Overflow,
}

impl fmt::Display for SymExpr {
    /// JAX's printing order: higher-degree terms first, the constant last.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        let mut terms: Vec<&(Mono, i64)> = self.terms.iter().collect();
        terms.sort_by(|a, b| b.0.degree().cmp(&a.0.degree()).then_with(|| a.0.cmp(&b.0)));
        for (i, (m, k)) in terms.iter().enumerate() {
            let (sign, mag) = if *k < 0 {
                ("-", k.unsigned_abs())
            } else {
                ("+", k.unsigned_abs())
            };
            if i == 0 {
                if sign == "-" {
                    write!(f, "-")?;
                }
            } else {
                write!(f, " {sign} ")?;
            }
            if m.is_one() {
                write!(f, "{mag}")?;
                continue;
            }
            if mag != 1 {
                write!(f, "{mag}*")?;
            }
            for (j, (a, p)) in m.0.iter().enumerate() {
                if j > 0 {
                    write!(f, "*")?;
                }
                match a {
                    Atom::Var(v) => write!(f, "{v}")?,
                    Atom::FloorDiv(x, y) => write!(f, "floordiv({x}, {y})")?,
                    Atom::Mod(x, y) => write!(f, "mod({x}, {y})")?,
                    Atom::Max(x, y) => write!(f, "max({x}, {y})")?,
                    Atom::Min(x, y) => write!(f, "min({x}, {y})")?,
                }
                if *p > 1 {
                    write!(f, "^{p}")?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Int(i64),
    Ident(String),
    Op(&'static str),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() {
            let st = i;
            while i < b.len() && (b[i] as char).is_ascii_digit() {
                i += 1;
            }
            out.push(Tok::Int(s[st..i].parse().map_err(|e| format!("{e}"))?));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok::Ident(s[st..i].to_string()));
        } else {
            let two = s.get(i..i + 2).unwrap_or("");
            let op: &'static str = match two {
                "//" => "//",
                ">=" => ">=",
                "<=" => "<=",
                "==" => "==",
                _ => match c {
                    '+' => "+",
                    '-' => "-",
                    '*' => "*",
                    '%' => "%",
                    '(' => "(",
                    ')' => ")",
                    ',' => ",",
                    '>' => ">",
                    '<' => "<",
                    _ => return Err(format!("unexpected character `{c}`")),
                },
            };
            i += op.len();
            out.push(Tok::Op(op));
        }
    }
    Ok(out)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn eat(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Op(o)) if *o == op) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, op: &str) -> Result<(), String> {
        if self.eat(op) {
            Ok(())
        } else {
            Err(format!("expected `{op}`"))
        }
    }
    fn expr(&mut self) -> Result<SymExpr, String> {
        let mut e = self.term()?;
        loop {
            if self.eat("+") {
                e = e.add(&self.term()?);
            } else if self.eat("-") {
                e = e.sub(&self.term()?);
            } else {
                return Ok(e);
            }
        }
    }
    fn term(&mut self) -> Result<SymExpr, String> {
        let mut e = self.unary()?;
        loop {
            if self.eat("*") {
                e = e.mul(&self.unary()?);
            } else if self.eat("//") {
                e = e.floordiv(&self.unary()?);
            } else if self.eat("%") {
                e = e.modulo(&self.unary()?);
            } else {
                return Ok(e);
            }
        }
    }
    fn unary(&mut self) -> Result<SymExpr, String> {
        if self.eat("-") {
            return Ok(self.unary()?.neg());
        }
        self.atom()
    }
    fn atom(&mut self) -> Result<SymExpr, String> {
        match self.peek().cloned() {
            Some(Tok::Int(v)) => {
                self.pos += 1;
                Ok(SymExpr::constant(v))
            }
            Some(Tok::Ident(name)) => {
                self.pos += 1;
                if self.eat("(") {
                    let a = self.expr()?;
                    self.expect(",")?;
                    let b = self.expr()?;
                    self.expect(")")?;
                    return match name.as_str() {
                        "floordiv" => Ok(a.floordiv(&b)),
                        "mod" => Ok(a.modulo(&b)),
                        "max" => Ok(a.max_of(&b)),
                        "min" => Ok(a.min_of(&b)),
                        _ => Err(format!("unknown function `{name}`")),
                    };
                }
                Ok(SymExpr::var(&name))
            }
            Some(Tok::Op("(")) => {
                self.pos += 1;
                let e = self.expr()?;
                self.expect(")")?;
                Ok(e)
            }
            t => Err(format!("unexpected {t:?}")),
        }
    }
}

/// A comparison of the symbolic scope (`S <= 512`, `mod(BU, 2) == 0`, `S >= P`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Constraint {
    pub lhs: SymExpr,
    pub rel: Rel,
    pub rhs: SymExpr,
    /// The declared text.
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum Rel {
    Ge,
    Le,
    Eq,
}

impl Constraint {
    pub fn parse(text: &str) -> Result<Constraint, SymError> {
        let toks = tokenize(text).map_err(|why| SymError::Parse {
            text: text.into(),
            why,
        })?;
        let at = toks
            .iter()
            .position(|t| matches!(t, Tok::Op(">=" | "<=" | "==")))
            .ok_or_else(|| SymError::Parse {
                text: text.into(),
                why: "expected `>=`, `<=` or `==`".into(),
            })?;
        let rel = match toks[at] {
            Tok::Op(">=") => Rel::Ge,
            Tok::Op("<=") => Rel::Le,
            _ => Rel::Eq,
        };
        let side = |ts: Vec<Tok>| -> Result<SymExpr, SymError> {
            let mut p = Parser { toks: ts, pos: 0 };
            let e = p.expr().map_err(|why| SymError::Parse {
                text: text.into(),
                why,
            })?;
            if p.pos != p.toks.len() {
                return Err(SymError::Parse {
                    text: text.into(),
                    why: "trailing tokens".into(),
                });
            }
            Ok(e)
        };
        Ok(Constraint {
            lhs: side(toks[..at].to_vec())?,
            rel,
            rhs: side(toks[at + 1..].to_vec())?,
            text: text.to_string(),
        })
    }

    /// Whether the constraint holds at `env`.
    pub fn holds(&self, env: &BTreeMap<String, i64>) -> Result<bool, EvalError> {
        let (l, r) = (self.lhs.eval(env)?, self.rhs.eval(env)?);
        Ok(match self.rel {
            Rel::Ge => l >= r,
            Rel::Le => l <= r,
            Rel::Eq => l == r,
        })
    }

    /// The constraint as facts `f >= 0`.
    pub fn facts(&self) -> Vec<SymExpr> {
        let d = self.lhs.sub(&self.rhs);
        match self.rel {
            Rel::Ge => vec![d],
            Rel::Le => vec![d.neg()],
            Rel::Eq => vec![d.clone(), d.neg()],
        }
    }
}

/// An interval with optional infinite ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub lo: Option<i128>,
    pub hi: Option<i128>,
}

impl Interval {
    pub const ALL: Interval = Interval { lo: None, hi: None };
    fn point(v: i128) -> Self {
        Interval {
            lo: Some(v),
            hi: Some(v),
        }
    }
    fn add(self, o: Interval) -> Interval {
        Interval {
            lo: self.lo.zip(o.lo).and_then(|(a, b)| a.checked_add(b)),
            hi: self.hi.zip(o.hi).and_then(|(a, b)| a.checked_add(b)),
        }
    }
    fn mul(self, o: Interval) -> Interval {
        // Exact only for finite ends; an infinite end makes the product unbounded unless signs fix it.
        match (self.lo, self.hi, o.lo, o.hi) {
            (Some(a), Some(b), Some(c), Some(d)) => {
                let ps = [
                    a.checked_mul(c),
                    a.checked_mul(d),
                    b.checked_mul(c),
                    b.checked_mul(d),
                ];
                if ps.iter().any(Option::is_none) {
                    return Interval::ALL;
                }
                let ps: Vec<i128> = ps.into_iter().flatten().collect();
                Interval {
                    lo: ps.iter().min().copied(),
                    hi: ps.iter().max().copied(),
                }
            }
            (Some(a), _, Some(c), _) if a >= 0 && c >= 0 => Interval {
                lo: a.checked_mul(c),
                hi: None,
            },
            _ => Interval::ALL,
        }
    }
    fn scale(self, k: i128) -> Interval {
        self.mul(Interval::point(k))
    }
}

/// Bounds of the dimension variables (each ≥ 1, refined by the declared constraints) and the constraint
/// facts, used to decide comparisons and to simplify `max`/`min`.
#[derive(Debug, Clone, Default)]
pub struct SymCtx {
    pub bounds: BTreeMap<String, Interval>,
    pub facts: Vec<SymExpr>,
}

impl SymCtx {
    pub fn new(symbols: &[String], constraints: &[Constraint]) -> Self {
        let mut ctx = SymCtx {
            bounds: symbols
                .iter()
                .map(|s| {
                    (
                        s.clone(),
                        Interval {
                            lo: Some(1),
                            hi: None,
                        },
                    )
                })
                .collect(),
            facts: Vec::new(),
        };
        for c in constraints {
            for f in c.facts() {
                ctx.assume(f);
            }
        }
        ctx
    }

    /// Add the fact `f >= 0`. A fact `k*X + c >= 0` with one variable also bounds that variable.
    pub fn assume(&mut self, f: SymExpr) {
        let (consts, vars): (Vec<_>, Vec<_>) = f.terms.iter().partition(|(m, _)| m.is_one());
        if let [(m, k)] = vars.as_slice()
            && let [(Atom::Var(v), 1)] = m.0.as_slice()
        {
            let c0 = consts.first().map_or(0i128, |(_, c)| *c as i128);
            let k = *k as i128;
            let mut iv = self.bounds.get(v).copied().unwrap_or(Interval::ALL);
            if k > 0 {
                // X >= ceil(-c0 / k)
                let lo = (-c0).div_euclid(k) + i128::from((-c0).rem_euclid(k) != 0);
                iv.lo = Some(iv.lo.map_or(lo, |l| l.max(lo)));
            } else if k < 0 {
                // X <= floor(c0 / -k)
                let hi = c0.div_euclid(-k);
                iv.hi = Some(iv.hi.map_or(hi, |h| h.min(hi)));
            }
            self.bounds.insert(v.clone(), iv);
        }
        self.facts.push(f);
    }

    /// This context with the extra facts `f >= 0`.
    pub fn with(&self, facts: &[SymExpr]) -> SymCtx {
        let mut c = self.clone();
        for f in facts {
            c.assume(f.clone());
        }
        c
    }

    /// Whether the facts are provably contradictory (an empty variable range, or a fact proven negative).
    pub fn contradictory(&self) -> bool {
        self.bounds
            .values()
            .any(|iv| matches!((iv.lo, iv.hi), (Some(l), Some(h)) if l > h))
            || self.facts.iter().any(|f| self.nonneg(f) == Some(false))
    }

    /// Interval of `e` by interval arithmetic over the variable bounds.
    pub fn interval(&self, e: &SymExpr) -> Interval {
        let mut acc = Interval::point(0);
        for (m, k) in &e.terms {
            let mut iv = Interval::point(1);
            for (a, p) in &m.0 {
                let ai = self.atom_interval(a);
                for _ in 0..*p {
                    iv = iv.mul(ai);
                }
            }
            acc = acc.add(iv.scale(*k as i128));
        }
        acc
    }

    fn atom_interval(&self, a: &Atom) -> Interval {
        match a {
            Atom::Var(v) => self.bounds.get(v).copied().unwrap_or(Interval::ALL),
            Atom::FloorDiv(x, y) => match y.as_const() {
                Some(c) if c > 0 => {
                    let ix = self.interval(x);
                    let c = c as i128;
                    Interval {
                        lo: ix.lo.map(|l| l.div_euclid(c)),
                        hi: ix.hi.map(|h| h.div_euclid(c)),
                    }
                }
                _ => Interval::ALL,
            },
            Atom::Mod(x, y) => match y.as_const() {
                Some(c) if c > 0 => {
                    let ix = self.interval(x);
                    let c = c as i128;
                    match (ix.lo, ix.hi) {
                        // Within one period the residue is affine in x.
                        (Some(l), Some(h)) if l.div_euclid(c) == h.div_euclid(c) => Interval {
                            lo: Some(l.rem_euclid(c)),
                            hi: Some(h.rem_euclid(c)),
                        },
                        _ => Interval {
                            lo: Some(0),
                            hi: Some(c - 1),
                        },
                    }
                }
                _ => Interval::ALL,
            },
            Atom::Max(x, y) => {
                let (ix, iy) = (self.interval(x), self.interval(y));
                Interval {
                    lo: match (ix.lo, iy.lo) {
                        (Some(a), Some(b)) => Some(a.max(b)),
                        (a, b) => a.or(b),
                    },
                    hi: ix.hi.zip(iy.hi).map(|(a, b)| a.max(b)),
                }
            }
            Atom::Min(x, y) => {
                let (ix, iy) = (self.interval(x), self.interval(y));
                Interval {
                    lo: ix.lo.zip(iy.lo).map(|(a, b)| a.min(b)),
                    hi: match (ix.hi, iy.hi) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (a, b) => a.or(b),
                    },
                }
            }
        }
    }

    /// `Some(true)` if `e >= 0` is proven, `Some(false)` if `e < 0` is proven, else `None`.
    pub fn nonneg(&self, e: &SymExpr) -> Option<bool> {
        let iv = self.interval(e);
        if iv.lo.is_some_and(|l| l >= 0) {
            return Some(true);
        }
        if iv.hi.is_some_and(|h| h < 0) {
            return Some(false);
        }
        // e = (e - f) + f with a fact f >= 0.
        for f in &self.facts {
            if self.interval(&e.sub(f)).lo.is_some_and(|l| l >= 0) {
                return Some(true);
            }
            if self.interval(&e.add(f)).hi.is_some_and(|h| h < 0) {
                return Some(false);
            }
        }
        None
    }

    /// Decide `a >= b`.
    pub fn ge(&self, a: &SymExpr, b: &SymExpr) -> Option<bool> {
        self.nonneg(&a.sub(b))
    }

    /// Decide `a == b`.
    pub fn eq(&self, a: &SymExpr, b: &SymExpr) -> Option<bool> {
        if a == b {
            return Some(true);
        }
        let d = a.sub(b);
        if self.nonneg(&d.add_const(-1)) == Some(true) || self.nonneg(&d.neg().add_const(-1)) == Some(true) {
            return Some(false);
        }
        None
    }

    pub fn max(&self, a: &SymExpr, b: &SymExpr) -> SymExpr {
        match self.ge(a, b) {
            Some(true) => a.clone(),
            Some(false) => b.clone(),
            None => a.max_of(b),
        }
    }

    pub fn min(&self, a: &SymExpr, b: &SymExpr) -> SymExpr {
        match self.ge(a, b) {
            Some(true) => b.clone(),
            Some(false) => a.clone(),
            None => a.min_of(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(kv: &[(&str, i64)]) -> BTreeMap<String, i64> {
        kv.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn canonical_forms() {
        let a = SymExpr::parse("32*H + 2*B*S - B*S - 1 + 1").unwrap();
        let b = SymExpr::parse("S*B + H*32").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.to_string(), "B*S + 32*H");
        assert_eq!(
            SymExpr::parse("(S + 1) * (S - 1)").unwrap(),
            SymExpr::parse("S*S - 1").unwrap()
        );
    }

    #[test]
    fn floordiv_and_mod() {
        let s = SymExpr::var("S");
        assert_eq!(s.scale(4).floordiv(&SymExpr::constant(2)), s.scale(2));
        let e = s.scale(7).add_const(3).floordiv(&SymExpr::constant(2));
        assert_eq!(e.to_string(), "3*S + floordiv(S + 1, 2) + 1");
        for v in 1..20 {
            let x = 7 * v + 3;
            assert_eq!(e.eval(&env(&[("S", v)])).unwrap(), x.div_euclid(2));
        }
        assert_eq!(s.scale(6).modulo(&SymExpr::constant(3)), SymExpr::zero());
        assert_eq!(
            SymExpr::constant(-7).floordiv(&SymExpr::constant(2)),
            SymExpr::constant(-4)
        );
        assert_eq!(
            SymExpr::constant(-7).modulo(&SymExpr::constant(2)),
            SymExpr::constant(1)
        );
        assert_eq!(s.scale(3).floordiv(&s), SymExpr::constant(3));
        let h = SymExpr::parse("floordiv(floordiv(H + 1, 2) + 1, 2)").unwrap();
        assert_eq!(h, SymExpr::parse("floordiv(H + 3, 4)").unwrap());
        for v in 1..40 {
            assert_eq!(
                h.eval(&env(&[("H", v)])).unwrap(),
                ((v + 1).div_euclid(2) + 1).div_euclid(2)
            );
        }
    }

    #[test]
    fn parse_constraints_and_bounds() {
        let cs: Vec<Constraint> = [
            "S <= 512",
            "P >= 128",
            "S >= P",
            "mod(BU, 2) == 0",
            "21*H*W >= 300",
        ]
        .iter()
        .map(|c| Constraint::parse(c).unwrap())
        .collect();
        let ctx = SymCtx::new(
            &["S".into(), "P".into(), "BU".into(), "H".into(), "W".into()],
            &cs,
        );
        assert_eq!(ctx.bounds["S"].hi, Some(512));
        assert_eq!(ctx.bounds["P"].lo, Some(128));
        let s = SymExpr::var("S");
        let p = SymExpr::var("P");
        assert_eq!(ctx.ge(&s, &p), Some(true));
        assert_eq!(ctx.ge(&s.add_const(-1), &SymExpr::zero()), Some(true));
        assert_eq!(ctx.ge(&SymExpr::constant(600), &s), Some(true));
        assert_eq!(ctx.eq(&s, &SymExpr::constant(0)), Some(false));
        assert_eq!(ctx.max(&s, &SymExpr::constant(1)), s);
        assert!(cs[3].holds(&env(&[("BU", 4)])).unwrap());
    }

    #[test]
    fn printing_round_trips() {
        for t in [
            "B*S + 32*H",
            "floordiv(S + 1, 2)",
            "mod(BU, 2)",
            "max(H, W) - 3",
            "2*P",
            "-S + 4",
        ] {
            let e = SymExpr::parse(t).unwrap();
            assert_eq!(SymExpr::parse(&e.to_string()).unwrap(), e, "{t}");
        }
    }
}
