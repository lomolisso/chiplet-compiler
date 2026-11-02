//! A typed, hash-consed SMT-LIB term DAG (docs/framework/smt.md, "Terms").
//!
//! Every constructor checks the sorts of its arguments, so an ill-sorted query is a Rust error, not a
//! solver error. Equal applications are one node; the arguments of commutative operators are ordered by
//! node id, so two sides of an equivalence built from the same values in a different operand order are
//! the same node. `Ctx::render` writes the reachable nodes into a [`Script`]: variables and functions as
//! declarations, every application used more than once as a `define-fun`, the rest inline.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use crate::Script;

/// A sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Sort {
    Bool,
    Bv(u32),
    /// `(_ FloatingPoint e s)`: `s` counts the hidden bit (Float32 is `e = 8, s = 24`).
    Fp {
        e: u32,
        s: u32,
    },
}

impl Sort {
    pub fn smt(self) -> String {
        match self {
            Sort::Bool => "Bool".into(),
            Sort::Bv(w) => format!("(_ BitVec {w})"),
            Sort::Fp { e, s } => format!("(_ FloatingPoint {e} {s})"),
        }
    }

    pub fn width(self) -> u32 {
        match self {
            Sort::Bool => 1,
            Sort::Bv(w) => w,
            Sort::Fp { e, s } => e + s,
        }
    }
}

/// A rounding mode of the FP theory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rm {
    Rne,
    Rtz,
    Rtp,
    Rtn,
}

impl Rm {
    fn smt(self) -> &'static str {
        match self {
            Rm::Rne => "RNE",
            Rm::Rtz => "RTZ",
            Rm::Rtp => "RTP",
            Rm::Rtn => "RTN",
        }
    }
}

/// An operator of the theories in use (QF_BV, QF_FP, core, uninterpreted functions).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
    // Core.
    Not,
    And,
    Or,
    Xor,
    Implies,
    Ite,
    Eq,
    Distinct,
    // Bit-vectors.
    BvAdd,
    BvSub,
    BvMul,
    BvNeg,
    BvNot,
    BvAnd,
    BvOr,
    BvXor,
    BvUdiv,
    BvUrem,
    BvSdiv,
    BvSrem,
    BvShl,
    BvLshr,
    BvAshr,
    BvUlt,
    BvUle,
    BvSlt,
    BvSle,
    Concat,
    Extract {
        hi: u32,
        lo: u32,
    },
    ZeroExtend(u32),
    SignExtend(u32),
    // Floating point.
    FpAdd(Rm),
    FpSub(Rm),
    FpMul(Rm),
    FpDiv(Rm),
    FpSqrt(Rm),
    FpRem,
    FpRoundToIntegral(Rm),
    FpAbs,
    FpNeg,
    FpLt,
    FpLeq,
    FpEq,
    FpIsNaN,
    FpIsInfinite,
    FpIsZero,
    FpIsNegative,
    FpIsSubnormal,
    /// `((_ to_fp e s) bv)`: reinterpret an IEEE bit pattern.
    FpFromBits {
        e: u32,
        s: u32,
    },
    /// `((_ to_fp e s) rm fp)`.
    FpToFp {
        e: u32,
        s: u32,
        rm: Rm,
    },
    /// `((_ to_fp e s) rm bv)`: a signed integer.
    FpFromSbv {
        e: u32,
        s: u32,
        rm: Rm,
    },
    /// `((_ to_fp_unsigned e s) rm bv)`.
    FpFromUbv {
        e: u32,
        s: u32,
        rm: Rm,
    },
    /// `((_ fp.to_sbv w) rm fp)` (unspecified out of range: callers guard it).
    FpToSbv {
        w: u32,
        rm: Rm,
    },
    /// `((_ fp.to_ubv w) rm fp)`.
    FpToUbv {
        w: u32,
        rm: Rm,
    },
    /// An uninterpreted function declared in the context.
    Uf(String),
}

impl Op {
    /// Operators whose arguments may be reordered (commutative in their SMT-LIB semantics).
    fn commutative(&self) -> bool {
        matches!(
            self,
            Op::And
                | Op::Or
                | Op::Xor
                | Op::Eq
                | Op::Distinct
                | Op::BvAdd
                | Op::BvMul
                | Op::BvAnd
                | Op::BvOr
                | Op::BvXor
                | Op::FpAdd(_)
                | Op::FpMul(_)
                | Op::FpEq
        )
    }

    fn head(&self) -> String {
        match self {
            Op::Not => "not".into(),
            Op::And => "and".into(),
            Op::Or => "or".into(),
            Op::Xor => "xor".into(),
            Op::Implies => "=>".into(),
            Op::Ite => "ite".into(),
            Op::Eq => "=".into(),
            Op::Distinct => "distinct".into(),
            Op::BvAdd => "bvadd".into(),
            Op::BvSub => "bvsub".into(),
            Op::BvMul => "bvmul".into(),
            Op::BvNeg => "bvneg".into(),
            Op::BvNot => "bvnot".into(),
            Op::BvAnd => "bvand".into(),
            Op::BvOr => "bvor".into(),
            Op::BvXor => "bvxor".into(),
            Op::BvUdiv => "bvudiv".into(),
            Op::BvUrem => "bvurem".into(),
            Op::BvSdiv => "bvsdiv".into(),
            Op::BvSrem => "bvsrem".into(),
            Op::BvShl => "bvshl".into(),
            Op::BvLshr => "bvlshr".into(),
            Op::BvAshr => "bvashr".into(),
            Op::BvUlt => "bvult".into(),
            Op::BvUle => "bvule".into(),
            Op::BvSlt => "bvslt".into(),
            Op::BvSle => "bvsle".into(),
            Op::Concat => "concat".into(),
            Op::Extract { hi, lo } => format!("(_ extract {hi} {lo})"),
            Op::ZeroExtend(n) => format!("(_ zero_extend {n})"),
            Op::SignExtend(n) => format!("(_ sign_extend {n})"),
            Op::FpAdd(r) => format!("fp.add {}", r.smt()),
            Op::FpSub(r) => format!("fp.sub {}", r.smt()),
            Op::FpMul(r) => format!("fp.mul {}", r.smt()),
            Op::FpDiv(r) => format!("fp.div {}", r.smt()),
            Op::FpSqrt(r) => format!("fp.sqrt {}", r.smt()),
            Op::FpRem => "fp.rem".into(),
            Op::FpRoundToIntegral(r) => format!("fp.roundToIntegral {}", r.smt()),
            Op::FpAbs => "fp.abs".into(),
            Op::FpNeg => "fp.neg".into(),
            Op::FpLt => "fp.lt".into(),
            Op::FpLeq => "fp.leq".into(),
            Op::FpEq => "fp.eq".into(),
            Op::FpIsNaN => "fp.isNaN".into(),
            Op::FpIsInfinite => "fp.isInfinite".into(),
            Op::FpIsZero => "fp.isZero".into(),
            Op::FpIsNegative => "fp.isNegative".into(),
            Op::FpIsSubnormal => "fp.isSubnormal".into(),
            Op::FpFromBits { e, s } => format!("(_ to_fp {e} {s})"),
            Op::FpToFp { e, s, rm } | Op::FpFromSbv { e, s, rm } => format!("(_ to_fp {e} {s}) {}", rm.smt()),
            Op::FpFromUbv { e, s, rm } => format!("(_ to_fp_unsigned {e} {s}) {}", rm.smt()),
            Op::FpToSbv { w, rm } => format!("(_ fp.to_sbv {w}) {}", rm.smt()),
            Op::FpToUbv { w, rm } => format!("(_ fp.to_ubv {w}) {}", rm.smt()),
            Op::Uf(f) => f.clone(),
        }
    }
}

/// A node of the DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct T(u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Node {
    Var(String),
    True,
    False,
    /// A bit-vector literal (value below 2^width).
    Bv(u128, u32),
    App(Op, Vec<T>),
}

/// An ill-sorted application.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct SortError(pub String);

type R = Result<T, SortError>;

/// The term DAG of one query.
#[derive(Debug, Default)]
pub struct Ctx {
    nodes: Vec<(Node, Sort)>,
    index: HashMap<Node, T>,
    vars: BTreeMap<String, T>,
    funs: BTreeMap<String, (Vec<Sort>, Sort)>,
    /// Side conditions (definitions of auxiliary variables), asserted with the query.
    side: Vec<T>,
    fresh: u32,
    /// The IEEE bits of FP terms (auxiliary variables), by term.
    bits: HashMap<T, T>,
    /// Encodings used (`<format>.<op>`), for the encoding lemmas.
    used: std::collections::BTreeSet<String>,
}

fn err<X>(s: impl Into<String>) -> Result<X, SortError> {
    Err(SortError(s.into()))
}

impl Ctx {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sort(&self, t: T) -> Sort {
        self.nodes[t.0 as usize].1
    }

    fn intern(&mut self, n: Node, s: Sort) -> T {
        if let Some(&t) = self.index.get(&n) {
            return t;
        }
        let t = T(self.nodes.len() as u32);
        self.nodes.push((n.clone(), s));
        self.index.insert(n, t);
        t
    }

    /// A free variable (one per name; the sort must agree).
    pub fn var(&mut self, name: &str, s: Sort) -> R {
        if let Some(&t) = self.vars.get(name) {
            if self.sort(t) != s {
                return err(format!("variable {name} declared with two sorts"));
            }
            return Ok(t);
        }
        let t = self.intern(Node::Var(name.to_string()), s);
        self.vars.insert(name.to_string(), t);
        Ok(t)
    }

    /// A fresh auxiliary variable `<prefix><k>`.
    pub fn fresh(&mut self, prefix: &str, s: Sort) -> R {
        let name = format!("{prefix}{}", self.fresh);
        self.fresh += 1;
        self.var(&name, s)
    }

    /// Assert `c` with the query (the definition of an auxiliary variable).
    pub fn side(&mut self, c: T) -> Result<(), SortError> {
        if self.sort(c) != Sort::Bool {
            return err("side condition is not Bool");
        }
        if !self.side.contains(&c) {
            self.side.push(c);
        }
        Ok(())
    }

    /// The IEEE bit pattern of an FP term: an auxiliary bit-vector `b` with `to_fp(b) = x`, the canonical
    /// quiet NaN (sign 0, the top significand bit set) when `x` is NaN. `fp.to_ieee_bv` would do, but
    /// Bitwuzla 0.9.1 does not have it; the definition makes `b` a function of `x`.
    pub fn bits_of(&mut self, x: T) -> R {
        if let Some(&b) = self.bits.get(&x) {
            return Ok(b);
        }
        let Sort::Fp { e, s } = self.sort(x) else {
            return err("bits of a non-FP term");
        };
        let b = self.fresh("bits", Sort::Bv(e + s))?;
        let back = self.app(Op::FpFromBits { e, s }, &[b])?;
        let same = self.eq(back, x)?;
        let nan = self.app(Op::FpIsNaN, &[x])?;
        let canon = self.bv(((1u128 << e) - 1) << (s - 1) | 1u128 << (s - 2), e + s);
        let is_canon = self.eq(b, canon)?;
        let def = self.app(Op::Implies, &[nan, is_canon])?;
        let both = self.and(&[same, def])?;
        self.side(both)?;
        self.bits.insert(x, b);
        Ok(b)
    }

    /// Record that an encoding was used (`<format>.<op>`).
    pub fn note(&mut self, what: impl Into<String>) {
        self.used.insert(what.into());
    }

    pub fn used(&self) -> &std::collections::BTreeSet<String> {
        &self.used
    }

    /// The declared uninterpreted functions.
    pub fn functions(&self) -> impl Iterator<Item = &String> {
        self.funs.keys()
    }

    pub fn bool(&mut self, b: bool) -> T {
        self.intern(if b { Node::True } else { Node::False }, Sort::Bool)
    }

    /// A bit-vector literal of width `w` (the value is taken modulo 2^w).
    pub fn bv(&mut self, v: u128, w: u32) -> T {
        let v = if w >= 128 { v } else { v & ((1u128 << w) - 1) };
        self.intern(Node::Bv(v, w), Sort::Bv(w))
    }

    /// A signed bit-vector literal.
    pub fn bvs(&mut self, v: i128, w: u32) -> T {
        self.bv(v as u128, w)
    }

    /// Declare an uninterpreted function (once per name, with one signature).
    pub fn fun(&mut self, name: &str, args: &[Sort], ret: Sort) -> Result<(), SortError> {
        match self.funs.get(name) {
            Some((a, r)) if a.as_slice() == args && *r == ret => Ok(()),
            Some(_) => err(format!("function {name} declared with two signatures")),
            None => {
                self.funs.insert(name.to_string(), (args.to_vec(), ret));
                Ok(())
            }
        }
    }

    /// Apply an operator.
    pub fn app(&mut self, op: Op, args: &[T]) -> R {
        let ss: Vec<Sort> = args.iter().map(|&a| self.sort(a)).collect();
        let s = self.result_sort(&op, &ss)?;
        let mut args = args.to_vec();
        if op.commutative() {
            args.sort();
        }
        Ok(self.intern(Node::App(op, args), s))
    }

    fn result_sort(&self, op: &Op, a: &[Sort]) -> Result<Sort, SortError> {
        use Sort::*;
        let n = a.len();
        let all = |s: Sort| a.iter().all(|&x| x == s);
        let same = || a.windows(2).all(|w| w[0] == w[1]);
        let bin_bv = || match a {
            [Bv(x), Bv(y)] if x == y => Ok(Bv(*x)),
            _ => err(format!(
                "{op:?}: bit-vector operands of one width expected, got {a:?}"
            )),
        };
        let cmp_bv = || match a {
            [Bv(x), Bv(y)] if x == y => Ok(Bool),
            _ => err(format!(
                "{op:?}: bit-vector operands of one width expected, got {a:?}"
            )),
        };
        let bin_fp = || match a {
            [f @ Fp { .. }, g] if f == g => Ok(*f),
            _ => err(format!("{op:?}: two FP operands of one sort expected, got {a:?}")),
        };
        let un_fp = || match a {
            [f @ Fp { .. }] => Ok(*f),
            _ => err(format!("{op:?}: one FP operand expected, got {a:?}")),
        };
        let pred_fp = |k: usize| match a {
            [Fp { .. }, ..] if n == k && same() => Ok(Bool),
            _ => err(format!("{op:?}: {k} FP operands of one sort expected, got {a:?}")),
        };
        match op {
            Op::Not if n == 1 && all(Bool) => Ok(Bool),
            Op::And | Op::Or | Op::Xor if n >= 2 && all(Bool) => Ok(Bool),
            Op::Implies if n == 2 && all(Bool) => Ok(Bool),
            Op::Ite => match a {
                [Bool, x, y] if x == y => Ok(*x),
                _ => err(format!("ite: (Bool, s, s) expected, got {a:?}")),
            },
            Op::Eq | Op::Distinct if n >= 2 && same() => Ok(Bool),
            Op::BvAdd
            | Op::BvSub
            | Op::BvMul
            | Op::BvAnd
            | Op::BvOr
            | Op::BvXor
            | Op::BvUdiv
            | Op::BvUrem
            | Op::BvSdiv
            | Op::BvSrem
            | Op::BvShl
            | Op::BvLshr
            | Op::BvAshr => bin_bv(),
            Op::BvNeg | Op::BvNot => match a {
                [Bv(x)] => Ok(Bv(*x)),
                _ => err(format!("{op:?}: one bit-vector operand expected")),
            },
            Op::BvUlt | Op::BvUle | Op::BvSlt | Op::BvSle => cmp_bv(),
            Op::Concat => match a {
                [Bv(x), Bv(y)] => Ok(Bv(x + y)),
                _ => err("concat: two bit-vectors expected"),
            },
            Op::Extract { hi, lo } => match a {
                [Bv(x)] if hi >= lo && hi < x => Ok(Bv(hi - lo + 1)),
                _ => err(format!("extract {hi} {lo} of {a:?}")),
            },
            Op::ZeroExtend(k) | Op::SignExtend(k) => match a {
                [Bv(x)] => Ok(Bv(x + k)),
                _ => err("extend: one bit-vector expected"),
            },
            Op::FpAdd(_) | Op::FpSub(_) | Op::FpMul(_) | Op::FpDiv(_) | Op::FpRem => bin_fp(),
            Op::FpSqrt(_) | Op::FpRoundToIntegral(_) | Op::FpAbs | Op::FpNeg => un_fp(),
            Op::FpLt | Op::FpLeq | Op::FpEq => pred_fp(2),
            Op::FpIsNaN | Op::FpIsInfinite | Op::FpIsZero | Op::FpIsNegative | Op::FpIsSubnormal => {
                pred_fp(1)
            }
            Op::FpFromBits { e, s } => match a {
                [Bv(w)] if *w == e + s => Ok(Fp { e: *e, s: *s }),
                _ => err(format!("to_fp {e} {s} of {a:?}")),
            },
            Op::FpToFp { e, s, .. } => match a {
                [Fp { .. }] => Ok(Fp { e: *e, s: *s }),
                _ => err("to_fp from FP: one FP operand expected"),
            },
            Op::FpFromSbv { e, s, .. } | Op::FpFromUbv { e, s, .. } => match a {
                [Bv(_)] => Ok(Fp { e: *e, s: *s }),
                _ => err("to_fp from an integer: one bit-vector expected"),
            },
            Op::FpToSbv { w, .. } | Op::FpToUbv { w, .. } => match a {
                [Fp { .. }] => Ok(Bv(*w)),
                _ => err("fp.to_*bv: one FP operand expected"),
            },
            Op::Uf(f) => match self.funs.get(f) {
                Some((args, ret)) if args.as_slice() == a => Ok(*ret),
                Some(_) => err(format!("{f}: arguments {a:?} do not match its signature")),
                None => err(format!("{f}: undeclared function")),
            },
            _ => err(format!("{op:?}: bad operands {a:?}")),
        }
    }

    // Shorthands (each checks sorts through `app`).
    pub fn not(&mut self, a: T) -> R {
        self.app(Op::Not, &[a])
    }
    /// `(and …)` without `true` operands (`false` if one is `false`).
    pub fn and(&mut self, xs: &[T]) -> R {
        let (t, f) = (self.bool(true), self.bool(false));
        if xs.contains(&f) {
            return Ok(f);
        }
        let xs: Vec<T> = xs.iter().copied().filter(|&x| x != t).collect();
        match xs.len() {
            0 => Ok(t),
            1 => Ok(xs[0]),
            _ => self.app(Op::And, &xs),
        }
    }
    /// `(or …)` without `false` operands (`true` if one is `true`).
    pub fn or(&mut self, xs: &[T]) -> R {
        let (t, f) = (self.bool(true), self.bool(false));
        if xs.contains(&t) {
            return Ok(t);
        }
        let xs: Vec<T> = xs.iter().copied().filter(|&x| x != f).collect();
        match xs.len() {
            0 => Ok(f),
            1 => Ok(xs[0]),
            _ => self.app(Op::Or, &xs),
        }
    }
    pub fn ite(&mut self, c: T, a: T, b: T) -> R {
        if a == b {
            return Ok(a);
        }
        self.app(Op::Ite, &[c, a, b])
    }
    pub fn eq(&mut self, a: T, b: T) -> R {
        if a == b {
            return Ok(self.bool(true));
        }
        self.app(Op::Eq, &[a, b])
    }
    pub fn op2(&mut self, op: Op, a: T, b: T) -> R {
        self.app(op, &[a, b])
    }
    pub fn op1(&mut self, op: Op, a: T) -> R {
        self.app(op, &[a])
    }
    /// Bits `hi..=lo` of a bit-vector.
    pub fn extract(&mut self, a: T, hi: u32, lo: u32) -> R {
        self.app(Op::Extract { hi, lo }, &[a])
    }
    /// A bit-vector resized to `w`: truncated, or extended by sign (`signed`) or zeros.
    pub fn resize(&mut self, a: T, w: u32, signed: bool) -> R {
        let Sort::Bv(v) = self.sort(a) else {
            return err("resize of a non bit-vector");
        };
        match w.cmp(&v) {
            std::cmp::Ordering::Equal => Ok(a),
            std::cmp::Ordering::Less => self.extract(a, w - 1, 0),
            std::cmp::Ordering::Greater if signed => self.app(Op::SignExtend(w - v), &[a]),
            std::cmp::Ordering::Greater => self.app(Op::ZeroExtend(w - v), &[a]),
        }
    }

    /// The SMT-LIB text of `t` with the names of the shared nodes.
    fn text(&self, t: T, named: &BTreeMap<T, String>, out: &mut String) {
        if let Some(n) = named.get(&t) {
            out.push_str(n);
            return;
        }
        match &self.nodes[t.0 as usize].0 {
            Node::Var(v) => out.push_str(v),
            Node::True => out.push_str("true"),
            Node::False => out.push_str("false"),
            Node::Bv(v, w) => {
                let _ = write!(out, "(_ bv{v} {w})");
            }
            Node::App(op, args) => {
                let h = op.head();
                if args.is_empty() {
                    out.push_str(&h);
                    return;
                }
                if h.contains(' ') && !h.starts_with('(') {
                    // An operator with a rounding mode: `(fp.add RNE a b)`.
                    let _ = write!(out, "({h}");
                } else if h.starts_with("(_ to_fp") || h.starts_with("(_ fp.to") {
                    // An indexed operator with a rounding mode: `((_ to_fp 8 24) RNE x)`.
                    let (ix, rm) = h.split_at(h.find(')').map_or(h.len(), |i| i + 1));
                    let _ = write!(out, "({ix}{rm}");
                } else {
                    let _ = write!(out, "({h}");
                }
                for &a in args {
                    out.push(' ');
                    self.text(a, named, out);
                }
                out.push(')');
            }
        }
    }

    /// Add `assert`s of `roots` (with the side conditions) to `script`, declaring every variable and
    /// function they use and defining every application used more than once.
    pub fn render(&self, script: &mut Script, roots: &[T]) {
        let mut all = roots.to_vec();
        all.extend(self.side.iter().copied());
        // Reachable nodes and use counts.
        let mut uses: BTreeMap<T, u32> = BTreeMap::new();
        let mut stack: Vec<T> = all.clone();
        let mut seen = std::collections::BTreeSet::new();
        while let Some(t) = stack.pop() {
            if !seen.insert(t) {
                continue;
            }
            if let Node::App(_, args) = &self.nodes[t.0 as usize].0 {
                for &a in args {
                    *uses.entry(a).or_default() += 1;
                    stack.push(a);
                }
            }
        }
        let mut funs = std::collections::BTreeSet::new();
        for &t in &seen {
            match &self.nodes[t.0 as usize].0 {
                Node::Var(v) => {
                    script.constant(v, &self.sort(t).smt());
                }
                Node::App(Op::Uf(f), _) => {
                    funs.insert(f.clone());
                }
                _ => {}
            }
        }
        for f in funs {
            let (a, r) = &self.funs[&f];
            let args: Vec<String> = a.iter().map(|s| s.smt()).collect();
            script.fun(&f, &args, &r.smt());
        }
        // Shared applications in id order (arguments have smaller ids).
        let mut named: BTreeMap<T, String> = BTreeMap::new();
        for &t in &seen {
            let shared = uses.get(&t).copied().unwrap_or(0) > 1;
            if shared && matches!(self.nodes[t.0 as usize].0, Node::App(..)) {
                let mut body = String::new();
                self.text(t, &named, &mut body);
                let name = format!("t{}", t.0);
                script.define(&name, &self.sort(t).smt(), &body);
                named.insert(t, name);
            }
        }
        for &r in &all {
            let mut s = String::new();
            self.text(r, &named, &mut s);
            script.assert(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_sorts() {
        let mut c = Ctx::new();
        let a = c.var("a", Sort::Bv(8)).unwrap();
        let b = c.var("b", Sort::Bv(16)).unwrap();
        assert!(c.op2(Op::BvAdd, a, b).is_err());
        assert!(c.var("a", Sort::Bv(16)).is_err());
        let x = c.var("x", Sort::Fp { e: 8, s: 24 }).unwrap();
        assert!(c.op2(Op::FpAdd(Rm::Rne), x, a).is_err());
        assert!(c.app(Op::Uf("f".into()), &[a]).is_err());
    }

    #[test]
    fn hash_conses_and_orders_commutative_operands() {
        let mut c = Ctx::new();
        let a = c.var("a", Sort::Bv(8)).unwrap();
        let b = c.var("b", Sort::Bv(8)).unwrap();
        let x = c.op2(Op::BvAdd, a, b).unwrap();
        let y = c.op2(Op::BvAdd, b, a).unwrap();
        assert_eq!(x, y);
        let s = c.op2(Op::BvSub, a, b).unwrap();
        let t = c.op2(Op::BvSub, b, a).unwrap();
        assert_ne!(s, t);
        let e = c.eq(x, y).unwrap();
        assert_eq!(e, c.bool(true));
    }

    #[test]
    fn renders_shared_nodes_once() {
        let mut c = Ctx::new();
        let x = c.var("x", Sort::Fp { e: 8, s: 24 }).unwrap();
        let s = c.op2(Op::FpAdd(Rm::Rne), x, x).unwrap();
        let m = c.op2(Op::FpMul(Rm::Rne), s, s).unwrap();
        assert!(
            c.app(
                Op::FpFromSbv {
                    e: 8,
                    s: 24,
                    rm: Rm::Rne
                },
                &[x]
            )
            .is_err()
        );
        let n = c.app(Op::FpIsNaN, &[m]).unwrap();
        let mut sc = Script::new();
        c.render(&mut sc, &[n]);
        let t = sc.render();
        assert!(t.contains("(declare-const x (_ FloatingPoint 8 24))"));
        assert!(t.contains("(define-fun t1 () (_ FloatingPoint 8 24) (fp.add RNE x x))"));
        assert!(t.contains("(assert (fp.isNaN (fp.mul RNE t1 t1)))"));
    }
}
