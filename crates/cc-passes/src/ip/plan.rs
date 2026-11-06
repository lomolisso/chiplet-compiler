//! The wrapper of every float op of the catalog (design §4.12; docs/passes/characterize.md §4.1a): which
//! glue of `cc_float` (a wrapper whose proof is the glue proof), around which cores of the flow's
//! providers, sized how. [`plan`] tries the candidate wrappers of an op in order and takes the first
//! whose every core a provider offers (combinational, with a subnormal rule the wrapper allows) and whose
//! bound is within the `[numerics]` budget; without one within the budget, the first wrapper found is
//! accepted at its cores' documented guarantee and marked so. An op no candidate fits is a gap.
//!
//! - **Exact operators** (add, subtract, multiply, divide, sqrt) run on correctly rounded cores: at the
//!   format (f16, E5M2; f32 and bf16 sums), with one more exponent bit (f32 and bf16 products, quotients,
//!   roots) or on f16 cores (E4M3FN), else on an f32 core with one rounding back (innocuous double
//!   rounding: precision `q >= 2p + 2`, Figueroa 1995).
//! - **Approximate operators** run on a core at the format (with one more exponent bit for a flushing
//!   one) or at f32 with one rounding back: within `e` ulps at the format, `1/2 + e 2^(p-q)` through a
//!   wider core of precision `q`.
//! - **Compositions** (power, tanh, log_plus_one, rsqrt without a core; power from one power core when
//!   its logarithm cannot be made accurate enough) at the smallest extra precision `k` whose cores exist
//!   and whose bound `1/2 + C 2^-k` is within the budget (`cc_float::compose`).
//! - **Cores** must be combinational (later RTL instantiates them as combinational external modules): an
//!   iterative core is skipped. A core that flushes subnormals is used only where none can arise or
//!   matter: for a flushing format at its exponent width or wider, else when the format's subnormals are
//!   normal in the core's format and everything below the core's normal range rounds to zero in the
//!   format.

use cc_core::adapters::ip::ulp_bound;
use cc_core::adapters::{
    Accuracy, CoreInstance, CoreOffer, CoreOp, Encoding, IpProvider, Rounding, Subnormals, Timing,
};
use cc_core::config::NumericsConfig;
use cc_float::compose::{self, Budget, Recipe};
use cc_float::glue::{CoreNeed, Dir, Glue, Role, SignOp, int_via, wide};
use cc_smt::float::{Arith, Flavour, Format};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::characterize::catalog::format;

/// What a route is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RouteKind {
    /// One wrapper around one or two cores.
    Core,
    /// A composition of cores (`cc_float::recipe`; rounding to integral through two cores).
    Composition,
    /// Bit-level glue only.
    Glue,
    /// Deliberately without a primitive (float `remainder`: constant computation only).
    Excluded,
    /// No provider of the flow offers what the op needs.
    Gap,
}

/// One core of a wrapper, as a provider answered for it.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// What the wrapper needs (role and format).
    pub need: CoreNeed,
    pub provider: &'static str,
    /// The core operation asked for.
    pub op: CoreOp,
    /// The format the provider was asked at (the operands').
    pub asked: Format,
    /// The offer, its instance's widths set to the need's (`cc_float`'s convention: the format the core
    /// computes in, the result for a conversion).
    pub offer: CoreOffer,
}

/// The wrapper of one float op.
#[derive(Debug, Clone)]
pub struct Plan {
    pub kind: RouteKind,
    pub glue: Option<Glue>,
    pub cores: Vec<Resolved>,
    pub recipe: Option<String>,
    /// Extra significand bits of a composition (`q = p + k`).
    pub k: Option<u32>,
    pub budget: f64,
    /// The bound in ulps of the op's format, when the contracts give one.
    pub bound: Option<f64>,
    pub within: Option<bool>,
    pub note: String,
}

impl Plan {
    fn simple(kind: RouteKind, note: impl Into<String>, budget: f64) -> Plan {
        Plan {
            kind,
            glue: None,
            cores: Vec::new(),
            recipe: None,
            k: None,
            budget,
            bound: None,
            within: None,
            note: note.into(),
        }
    }
}

const F32: Format = Format::F32;

fn precision(f: Format) -> u32 {
    f.m + 1
}

/// A format's name: the catalog's (`f32`, `f16`, `bf16`, `f8e5m2`) or `e<E>m<M>`.
pub fn format_name(f: Format) -> String {
    for n in ["f32", "f16", "bf16", "f8e5m2", "f8e4m3fn"] {
        if let Some(g) = format(n)
            && g == f
        {
            return n.to_string();
        }
    }
    format!("e{}m{}", f.e, f.m)
}

/// The budget as `cc_float` takes it (op -> format -> ulps).
pub fn budget_of(n: &NumericsConfig) -> Budget {
    let mut m: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    for (fmt, ops) in &n.budget {
        for (op, u) in ops {
            m.entry(op.clone()).or_default().insert(fmt.clone(), *u);
        }
    }
    Budget(m)
}

/// May a core with `sub` subnormal handling compute in `c` for a wrapper of format `t`?
fn subnormals_ok(sub: Subnormals, t: Format, c: Format) -> bool {
    if sub == Subnormals::Ieee {
        return true;
    }
    if t.flushes() {
        return c.e >= t.e;
    }
    // Every subnormal of t is normal in c, and what c flushes rounds to zero in t.
    let lowest = t.emin() - i64::from(t.m);
    c.flavour == Flavour::Ieee && c.emin() <= lowest && c.emin() < lowest - 1
}

/// The core operation a need asks for, and the format it is asked at.
fn core_op(n: &CoreNeed) -> Result<(CoreOp, Format), String> {
    let f = Format::ieee(n.exp_width, n.sig_width);
    let by_name = |op: &str| -> Option<CoreOp> {
        Some(match op {
            "add" => CoreOp::Add,
            "sub" | "subtract" => CoreOp::Sub,
            "mul" | "multiply" => CoreOp::Mul,
            "div" | "divide" => CoreOp::Div,
            "sqrt" => CoreOp::Sqrt,
            "rsqrt" => CoreOp::Rsqrt,
            "exp" | "exponential" => CoreOp::Exp,
            "exp2" => CoreOp::Exp2,
            "log" | "ln" => CoreOp::Ln,
            "log2" => CoreOp::Log2,
            "sine" => CoreOp::Sin,
            "cosine" => CoreOp::Cos,
            "pow" | "power" => CoreOp::Pow,
            _ => return None,
        })
    };
    Ok(match &n.role {
        Role::Arith { op } | Role::Approx { op } => {
            (by_name(op).ok_or_else(|| format!("no core op for `{op}`"))?, f)
        }
        Role::Convert { from } => {
            let src = cc_float::format_by_name(from).ok_or_else(|| format!("unknown format {from}"))?;
            (
                CoreOp::Convert {
                    exp_width: n.exp_width,
                    sig_width: n.sig_width,
                },
                src,
            )
        }
        Role::Less => (CoreOp::Compare, f),
        Role::ToInt { signed, w } => (
            CoreOp::ToInt {
                rnd: Rounding::TowardZero,
                signed: *signed,
                width: *w,
            },
            f,
        ),
        Role::RoundToInt { rnd, signed, w } => (
            CoreOp::ToInt {
                rnd: Rounding::ALL
                    .into_iter()
                    .find(|r| r.name() == rnd)
                    .ok_or_else(|| format!("rounding {rnd}"))?,
                signed: *signed,
                width: *w,
            },
            f,
        ),
        Role::FromInt { signed, w } => (
            CoreOp::FromInt {
                signed: *signed,
                width: *w,
            },
            f,
        ),
    })
}

/// The providers of a flow, in order, and the budget.
pub struct Planner<'a> {
    pub providers: &'a [&'static dyn IpProvider],
    pub numerics: &'a NumericsConfig,
}

impl std::fmt::Debug for Planner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.providers.iter().map(|p| p.name()).collect();
        f.debug_struct("Planner").field("providers", &names).finish()
    }
}

impl Planner<'_> {
    /// The first provider's combinational core for `need` in a wrapper of format `t`.
    fn resolve(&self, need: &CoreNeed, t: Format) -> Result<Resolved, String> {
        let (op, asked) = core_op(need)?;
        let name = op.to_string();
        let mut skipped = Vec::new();
        for p in self.providers {
            let Some(mut offer) = p.core(&name, asked.e, asked.m) else {
                continue;
            };
            if let Timing::Iterative { .. } = offer.timing {
                skipped.push(format!("{} (iterative)", p.name()));
                continue;
            }
            let cf = Format::ieee(need.exp_width, need.sig_width);
            if !subnormals_ok(offer.contract.subnormals, t, cf) {
                skipped.push(format!("{} (flushes subnormals)", p.name()));
                continue;
            }
            offer.instance.exp_width = need.exp_width;
            offer.instance.sig_width = need.sig_width;
            return Ok(Resolved {
                need: need.clone(),
                provider: p.name(),
                op,
                asked,
                offer,
            });
        }
        let mut why = format!("no combinational core `{name}` at {}", format_name(asked));
        if !skipped.is_empty() {
            why.push_str(&format!(" (skipped: {})", skipped.join(", ")));
        }
        Err(why)
    }

    fn resolve_all(&self, g: &Glue) -> Result<Vec<Resolved>, String> {
        let needs = g.needs().map_err(|e| e.to_string())?;
        let t = g.format();
        needs.iter().map(|n| self.resolve(n, t)).collect()
    }

    /// The plan of a catalog op on its types (`None`: not a float op).
    pub fn plan(&self, op: &str, variant: Option<&str>, types: &[String]) -> Option<Plan> {
        let src = types.first()?;
        let dst = types.get(1).unwrap_or(src);
        let (fs, fd) = (format(src), format(dst));
        if fs.is_none() && fd.is_none() {
            return None;
        }
        let (t, tname) = if op == "convert" {
            (fd.or(fs)?, if fd.is_some() { dst.as_str() } else { src.as_str() })
        } else {
            (fs?, src.as_str())
        };
        let budget = self.numerics.budget(tname, op);
        let arith = |a: Arith| self.exact(op, a, t, budget);
        let one = |g: Glue, kind: RouteKind| self.fixed(g, kind, budget);
        Some(match op {
            "add" => arith(Arith::Add),
            "subtract" => arith(Arith::Sub),
            "multiply" => arith(Arith::Mul),
            "divide" => arith(Arith::Div),
            "sqrt" => arith(Arith::Sqrt),
            "exponential" | "log" | "sine" | "cosine" => self.approx(op, t, budget),
            "rsqrt" => {
                let single = self.approx(op, t, budget);
                if single.within == Some(true) {
                    single
                } else {
                    match self.compose(op, Recipe::Rsqrt, t, budget) {
                        Ok(p) => p,
                        Err(why) if single.kind == RouteKind::Gap => {
                            Plan::simple(RouteKind::Gap, format!("{}; {why}", single.note), budget)
                        }
                        Err(_) => single,
                    }
                }
            }
            "power" => match self.compose(op, Recipe::Power, t, budget) {
                Ok(p) => p,
                Err(a) => self
                    .compose(op, Recipe::PowerCore, t, budget)
                    .unwrap_or_else(|b| Plan::simple(RouteKind::Gap, format!("{a}; {b}"), budget)),
            },
            "tanh" => self
                .compose(op, Recipe::Tanh, t, budget)
                .unwrap_or_else(|e| Plan::simple(RouteKind::Gap, e, budget)),
            "log_plus_one" => self
                .compose(op, Recipe::LogPlusOne, t, budget)
                .unwrap_or_else(|e| Plan::simple(RouteKind::Gap, e, budget)),
            "maximum" | "minimum" => one(
                Glue::MaxMin {
                    max: op == "maximum",
                    fmt: t,
                },
                RouteKind::Core,
            ),
            "clamp" => one(Glue::Clamp { fmt: t }, RouteKind::Core),
            "compare" => {
                let dir = match variant? {
                    "EQ" => Dir::Eq,
                    "NE" => Dir::Ne,
                    "LT" => Dir::Lt,
                    "LE" => Dir::Le,
                    _ => return None,
                };
                one(Glue::Compare { dir, fmt: t }, RouteKind::Core)
            }
            "abs" | "negate" | "sign" => {
                let s = match op {
                    "abs" => SignOp::Abs,
                    "negate" => SignOp::Negate,
                    _ => SignOp::Sign,
                };
                one(Glue::Sign { op: s, fmt: t }, RouteKind::Glue)
            }
            "is_finite" => one(Glue::IsFinite { fmt: t }, RouteKind::Glue),
            "floor" | "ceil" | "round_nearest_even" => {
                let rnd = match op {
                    "floor" => Rounding::Floor,
                    "ceil" => Rounding::Ceil,
                    _ => Rounding::NearestEven,
                };
                one(Glue::Round { fmt: t, rnd }, RouteKind::Composition)
            }
            "reduce_precision" => one(Glue::ReducePrecision { fmt: t, e: 5, m: 10 }, RouteKind::Glue),
            "remainder" => Plan::simple(
                RouteKind::Excluded,
                "constant computation only (design §4.9)",
                budget,
            ),
            // A multiplexer on the encodings: the DSLX primitive.
            "select" => Plan::simple(RouteKind::Glue, "bit-level select", budget),
            "convert" => match (fs, fd) {
                (Some(a), Some(b)) => one(Glue::Convert { from: a, to: b }, RouteKind::Core),
                (Some(a), None) => match int_type(dst) {
                    Some((signed, w)) => one(Glue::ToInt { fmt: a, signed, w }, RouteKind::Core),
                    None => one(
                        Glue::Bool {
                            fmt: a,
                            to_float: false,
                        },
                        RouteKind::Glue,
                    ),
                },
                (None, Some(b)) => match int_type(src) {
                    Some((signed, w)) => one(
                        Glue::FromInt {
                            fmt: b,
                            signed,
                            w,
                            via: int_via(b, signed, w),
                        },
                        RouteKind::Core,
                    ),
                    None => one(
                        Glue::Bool {
                            fmt: b,
                            to_float: true,
                        },
                        RouteKind::Glue,
                    ),
                },
                (None, None) => return None,
            },
            _ => Plan::simple(RouteKind::Gap, format!("no wrapper for `{op}`"), budget),
        })
    }

    /// A wrapper whose cores are exact (or none): kind as given, correctly rounded.
    fn fixed(&self, g: Glue, kind: RouteKind, budget: f64) -> Plan {
        match self.resolve_all(&g) {
            Ok(cores) => {
                let kind = if cores.is_empty() { RouteKind::Glue } else { kind };
                Plan {
                    kind,
                    recipe: recipe_text(&g),
                    glue: Some(g),
                    cores,
                    k: None,
                    budget,
                    bound: None,
                    within: None,
                    note: String::new(),
                }
            }
            Err(why) => Plan::simple(RouteKind::Gap, why, budget),
        }
    }

    /// An exact operator: [`Glue::Arith`], else one f32 core and one rounding back.
    fn exact(&self, op: &str, a: Arith, t: Format, budget: f64) -> Plan {
        let mut why = Vec::new();
        let mut gs = vec![(Glue::Arith { op: a, fmt: t }, String::new())];
        if t.bits() < 32 && precision(F32) >= 2 * precision(t) + 2 {
            gs.push((
                Glue::Approx {
                    op: op.into(),
                    arity: a.arity(),
                    fmt: t,
                    via: F32,
                },
                "through f32 (exact widening, one rounding: innocuous double rounding)".to_string(),
            ));
        }
        for (g, note) in gs {
            match self.resolve_all(&g) {
                Ok(cores) => {
                    let cr = cores
                        .iter()
                        .all(|c| c.offer.contract.accuracy == Accuracy::CorrectlyRounded);
                    if !cr {
                        why.push(format!("{}: a core is not correctly rounded", g.name()));
                        continue;
                    }
                    return Plan {
                        kind: RouteKind::Core,
                        glue: Some(g),
                        cores,
                        recipe: None,
                        k: None,
                        budget,
                        bound: Some(0.5),
                        within: Some(true),
                        note,
                    };
                }
                Err(e) => why.push(e),
            }
        }
        Plan::simple(RouteKind::Gap, why.join("; "), budget)
    }

    /// An approximate operator on one core at the format (one more exponent bit for a flushing one) or
    /// at f32.
    fn approx(&self, op: &str, t: Format, budget: f64) -> Plan {
        let mut vias = Vec::new();
        if t.flushes() {
            vias.extend([wide(t), t]);
        } else if t.flavour == Flavour::Ieee {
            vias.push(t);
        }
        if t.bits() < 32 {
            vias.push(F32);
        }
        let mut fallback: Option<Plan> = None;
        let mut why = Vec::new();
        for via in vias {
            let g = Glue::Approx {
                op: op.into(),
                arity: 1,
                fmt: t,
                via,
            };
            let cores = match self.resolve_all(&g) {
                Ok(c) => c,
                Err(e) => {
                    why.push(e);
                    continue;
                }
            };
            // A flushing format on a core of its own width: only a core that flushes itself (the
            // wrapper adds no flush there).
            if via == t && t.flushes() && cores[0].offer.contract.subnormals == Subnormals::Ieee {
                why.push(format!("{}: the core keeps subnormals", g.name()));
                continue;
            }
            let e = ulp_bound(&cores[0].offer.contract.accuracy);
            let bound = if via.m == t.m {
                e
            } else {
                e.map(|e| 0.5 + e * 2f64.powi(precision(t) as i32 - precision(via) as i32))
            };
            let within = bound.map(|b| b <= budget.max(0.5));
            let note = match within {
                Some(true) if via.m == t.m => String::new(),
                Some(true) => format!("through {} (exact widening, one rounding)", format_name(via)),
                Some(false) => "accepted at the core's documented guarantee, above the budget".into(),
                None => "no ulp bound in the contract: accepted at the core's documented guarantee".into(),
            };
            let p = Plan {
                kind: RouteKind::Core,
                glue: Some(g),
                cores,
                recipe: None,
                k: None,
                budget,
                bound,
                within,
                note,
            };
            if within == Some(true) {
                return p;
            }
            if fallback.is_none() {
                fallback = Some(p);
            }
        }
        fallback.unwrap_or_else(|| Plan::simple(RouteKind::Gap, why.join("; "), budget))
    }

    /// A composition at the smallest extra precision whose cores exist and whose bound is within the
    /// budget.
    fn compose(&self, op: &str, r: Recipe, t: Format, budget: f64) -> Result<Plan, String> {
        let b = budget_of(self.numerics);
        let start = if r == Recipe::PowerCore { 0 } else { 1 };
        let mut last = String::from("no inner precision tried");
        for k in start..=40 {
            let g = Glue::Compose {
                recipe: r,
                fmt: t,
                q: precision(t) + k,
            };
            let cores = match self.resolve_all(&g) {
                Ok(c) => c,
                Err(e) => {
                    last = format!("{} at k = {k}: {e}", g.name());
                    continue;
                }
            };
            let mut acc: BTreeMap<String, Accuracy> = BTreeMap::new();
            for c in &cores {
                let role = match (&r, &c.need.role) {
                    (Recipe::Rsqrt, Role::Arith { op }) if op == "sqrt" => "sqrt",
                    (_, Role::Approx { op }) => match op.as_str() {
                        "log2" | "log" => "log",
                        "exp2" | "exp" => "exp",
                        "pow" => "pow",
                        _ => continue,
                    },
                    _ => continue,
                };
                acc.insert(role.to_string(), c.offer.contract.accuracy.clone());
            }
            let s = match compose::size(op, t, r, &b, &acc) {
                Ok(s) => s,
                Err(e) => return Err(format!("{}: {e}", g.name())),
            };
            if s.k > k {
                last = format!(
                    "{} at k = {k}: the cores need k = {} for {} ulp",
                    g.name(),
                    s.k,
                    s.budget_ulp
                );
                continue;
            }
            let bound = 0.5 + s.constant * 2f64.powi(-(k as i32));
            return Ok(Plan {
                kind: RouteKind::Composition,
                recipe: recipe_text(&g),
                glue: Some(g),
                cores,
                k: Some(k),
                budget,
                bound: Some(bound),
                within: Some(bound <= budget.max(0.5)),
                note: format!(
                    "q = p + {k}; first-order bound 1/2 + C 2^-k, C = {:.3}",
                    s.constant
                ),
            });
        }
        Err(last)
    }
}

fn recipe_text(g: &Glue) -> Option<String> {
    Some(
        match g {
            Glue::Compose { recipe, .. } => match recipe {
                Recipe::Power => "exp2(y log2 x) on |x|, C11 Annex F special values",
                Recipe::PowerCore => "one power core on |x|, C11 Annex F special values",
                Recipe::Tanh => "x - x^3/3 for small x, else (e^2x - 1)/(e^2x + 1)",
                Recipe::LogPlusOne => "Goldberg: x ln(w)/(w - 1), w = 1 + x",
                Recipe::Rsqrt => "sqrt(1/x)",
                Recipe::Wider => "a wider core, rounded once",
            },
            Glue::Round { .. } => "to integer and back below 2^(p-1), x itself above; sign of zero kept",
            Glue::MaxMin { .. } | Glue::Clamp { .. } => "less-than cores; NaN and signed-zero glue",
            Glue::Compare { .. } => "less-than cores; NaN glue",
            Glue::ToInt { .. } => "toward zero, saturating; NaN to 0 glue",
            _ => return None,
        }
        .to_string(),
    )
}

fn int_type(name: &str) -> Option<(bool, u32)> {
    if name == "i1" {
        return None;
    }
    if let Some(w) = name.strip_prefix("ui") {
        return w.parse().ok().map(|w| (false, w));
    }
    name.strip_prefix('i')
        .and_then(|w| w.parse().ok())
        .map(|w| (true, w))
}

/// The instance the wrapper uses for a resolved core: the provider's own, or for a core in FloPoCo's
/// encoding the IEEE adapter (`<module>_ieee`: `from_ieee` on every operand, the core, `to_ieee`).
pub fn wrapper_instance(r: &Resolved) -> CoreInstance {
    match r.offer.encoding {
        Encoding::Ieee => r.offer.instance.clone(),
        Encoding::Flopoco => {
            let mut ports = BTreeMap::new();
            for j in 0..r.op.arity() {
                ports.insert(format!("a{j}"), format!("a{j}"));
            }
            ports.insert("z".into(), "z".into());
            CoreInstance {
                module: format!("{}_ieee", r.offer.instance.module),
                params: BTreeMap::new(),
                ports,
                exp_width: r.offer.instance.exp_width,
                sig_width: r.offer.instance.sig_width,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ip::PROVIDERS;

    const F16: Format = Format::F16;

    fn planner_for(names: &[&str]) -> Vec<&'static dyn IpProvider> {
        names.iter().map(|n| PROVIDERS.by_name(n).unwrap()).collect()
    }

    #[test]
    fn subnormal_rule() {
        // f32 products on a flushing core with one more exponent bit; f16 not at its own width.
        assert!(subnormals_ok(Subnormals::Flush, F32, wide(F32)));
        assert!(!subnormals_ok(Subnormals::Flush, F16, F16));
        assert!(subnormals_ok(Subnormals::Flush, F16, F32));
        assert!(subnormals_ok(Subnormals::Flush, Format::E4M3FN, F16));
        assert!(subnormals_ok(Subnormals::Ieee, F16, F16));
    }

    #[test]
    fn open_flow_division_avoids_iterative_cores() {
        let ps = planner_for(&["hardfloat", "flopoco", "xls_math"]);
        let n = NumericsConfig::default();
        let pl = Planner {
            providers: &ps,
            numerics: &n,
        };
        let p = pl.plan("divide", None, &["f32".into()]).unwrap();
        assert_eq!(p.kind, RouteKind::Core);
        assert_eq!(p.cores[0].provider, "flopoco");
        assert_eq!((p.cores[0].asked.e, p.cores[0].asked.m), (9, 23));
        // f16: FloPoCo flushes at f16, so through f32.
        let p = pl.plan("divide", None, &["f16".into()]).unwrap();
        assert!(
            matches!(p.glue, Some(Glue::Approx { via, .. }) if via == F32),
            "{p:?}"
        );
    }
}
