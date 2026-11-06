//! Fit checks (design §3.7, §4.12): every float op of the catalog mapped to the IP cores of the configured
//! flow, in the order `[passes.characterize] providers` gives for it, or reported as a gap.
//!
//! - **Core.** One core, at the op's format ("native") or at a wider one through exact widening and one
//!   rounding back. An exact operator (design §6.4) takes a correctly rounded core, and a wider one only
//!   when that double rounding is innocuous (precision q >= 2p + 2, Figueroa 1995): FP8 through f16, f16
//!   and bf16 through f32. An approximate operator takes the first candidate (native, then f32) whose bound
//!   in ulps of the op's format is within its `[numerics]` budget: e for a core of e ulps at the op's
//!   width, 1/2 + e 2^(p - q) through a wider core of precision q. Without one, the first core found is
//!   accepted at its documented guarantee (design §6.4: a single core at its datasheet accuracy), recorded
//!   as not within the budget, or as unknown for a contract without an ulp bound (absolute, unbounded).
//!   E4M3FN has no core of its own (no core implements its encoding): only wider routes.
//! - **Composition.** An op without a core (or `power`, `rsqrt` without one) built from cores per design
//!   §4.12; the route lists the cores it needs at the op's format or the wider one, and the extra bits are
//!   sized to the budget where the composition is built.
//! - **Glue.** Bit-level logic only: sign, absolute value, negation, select, `is_finite`, exact widening,
//!   bool conversions, the E4M3FN encoding.
//! - **Excluded.** Float `remainder`: its only use is constant computation (design §4.9).
//! - **Gap.** No provider of the flow offers a core the route needs: listed, never a failure.

use cc_core::adapters::ip::ulp_bound;
use cc_core::adapters::{CoreOffer, CoreOp, IpProvider, Rounding};
use cc_core::config::NumericsConfig;
use cc_smt::float::{Flavour, Format};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::characterize::catalog::{Kind, Primitive, format};

pub const ROUTES_SCHEMA: &str = "cc.characterize.routes/1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RouteKind {
    Core,
    Composition,
    Glue,
    Excluded,
    Gap,
}

/// One core of a route.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RouteCore {
    pub provider: String,
    /// The core operation (`cc_core::adapters::CoreOp` syntax).
    pub op: String,
    /// The operand format: a format name (`f32`, `f16`, `bf16`, `f8e5m2`) or `e<E>m<M>`.
    pub format: String,
    pub module: String,
    /// The contract's accuracy: `correctly_rounded`, `faithful`, `<n> ulp`, `absolute <bound>`,
    /// `unbounded`; and its subnormal mode (`ieee`, `flush`).
    pub accuracy: String,
    pub subnormals: String,
}

/// The route of one catalog primitive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Route {
    pub key: String,
    pub op: String,
    pub types: Vec<String>,
    pub kind: RouteKind,
    /// What is built around or from the cores.
    pub recipe: Option<String>,
    pub cores: Vec<RouteCore>,
    /// The op's budget in ulps (`[numerics]`; 0: correctly rounded).
    pub budget_ulp: f64,
    /// The route's error bound in ulps of the op's format, when the contracts give one.
    pub bound_ulp: Option<f64>,
    /// Whether the bound is within the budget (absent: no bound to compare).
    pub within_budget: Option<bool>,
    pub note: String,
}

/// `out/library/routes.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Routes {
    pub schema: String,
    /// The synthesis flow and its providers, in order.
    pub flow: String,
    pub providers: Vec<String>,
    pub routes: Vec<Route>,
    /// Keys of the routes that are gaps.
    pub gaps: Vec<String>,
}

const F16: Format = Format::F16;
const F32: Format = Format::F32;

fn precision(f: Format) -> u32 {
    f.m + 1
}

fn format_name(e: u32, m: u32) -> String {
    for n in ["f32", "f16", "bf16", "f8e5m2"] {
        if let Some(f) = format(n)
            && (f.e, f.m) == (e, m)
        {
            return n.to_string();
        }
    }
    format!("e{e}m{m}")
}

fn accuracy_text(o: &CoreOffer) -> (String, String) {
    use cc_core::adapters::{Accuracy, Subnormals};
    let a = match &o.contract.accuracy {
        Accuracy::CorrectlyRounded => "correctly_rounded".to_string(),
        Accuracy::Faithful => "faithful".to_string(),
        Accuracy::Ulp { ulp } => format!("{ulp} ulp"),
        Accuracy::Absolute { bound } => format!("absolute {bound:e}"),
        Accuracy::Unbounded => "unbounded".to_string(),
    };
    let s = match o.contract.subnormals {
        Subnormals::Ieee => "ieee",
        Subnormals::Flush => "flush",
    };
    (a, s.to_string())
}

/// The providers of a flow, in order.
struct Finder<'a> {
    providers: &'a [&'static dyn IpProvider],
}

impl Finder<'_> {
    fn find(&self, op: &CoreOp, f: Format) -> Option<(RouteCore, CoreOffer)> {
        let name = op.to_string();
        self.providers.iter().find_map(|p| {
            p.core(&name, f.e, f.m).map(|o| {
                let (accuracy, subnormals) = accuracy_text(&o);
                (
                    RouteCore {
                        provider: p.name().to_string(),
                        op: name.clone(),
                        format: format_name(f.e, f.m),
                        module: o.instance.module.clone(),
                        accuracy,
                        subnormals,
                    },
                    o,
                )
            })
        })
    }
}

/// The formats a core for an op on `t` may have: `t` itself (not E4M3FN, which no core implements), then
/// the wider one (exact ops: the narrowest of f16, f32 with innocuous double rounding; approximate: f32).
fn candidates(t: Format, exact: bool) -> Vec<Format> {
    let mut v = Vec::new();
    if t.flavour != Flavour::FiniteNan {
        v.push(t);
    }
    let contains = |q: Format| q.e >= t.e && q.m >= t.m && (q.e, q.m) != (t.e, t.m);
    if exact {
        if let Some(q) = [F16, F32]
            .into_iter()
            .find(|&q| contains(q) && precision(q) >= 2 * precision(t) + 2)
        {
            v.push(q);
        }
    } else if contains(F32) {
        v.push(F32);
    }
    v
}

#[derive(Default)]
struct Draft {
    kind: Option<RouteKind>,
    recipe: Option<String>,
    cores: Vec<RouteCore>,
    bound: Option<f64>,
    within: Option<bool>,
    note: String,
}

impl Draft {
    fn glue(what: &str) -> Self {
        Self {
            kind: Some(RouteKind::Glue),
            recipe: Some(what.to_string()),
            ..Self::default()
        }
    }
}

/// One core for `op` on format `t` (`core_t`: the format whose widths the core takes, normally `t`).
fn single(find: &Finder, op: &CoreOp, t: Format, budget: f64) -> Draft {
    let mut fallback: Option<Draft> = None;
    for q in candidates(t, op.exact()) {
        let Some((core, offer)) = find.find(op, q) else {
            continue;
        };
        let native = (q.e, q.m) == (t.e, t.m);
        let e = ulp_bound(&offer.contract.accuracy);
        let cr = matches!(
            offer.contract.accuracy,
            cc_core::adapters::Accuracy::CorrectlyRounded
        );
        let bound = if op.exact() {
            // Innocuous double rounding (candidates() only offers such a q) keeps a CR result CR.
            cr.then_some(0.5)
        } else if native {
            e
        } else {
            e.map(|e| 0.5 + e * 2f64.powi(precision(t) as i32 - precision(q) as i32))
        };
        let within = bound.map(|b| b <= budget.max(0.5));
        let note = match (native, within) {
            (true, Some(true)) => String::new(),
            (false, Some(true)) => {
                format!("through {} (exact widening, one rounding)", format_name(q.e, q.m))
            }
            (_, Some(false)) => "accepted at the core's documented guarantee, above the budget".to_string(),
            (_, None) => {
                "no ulp bound in the contract: accepted at the core's documented guarantee".to_string()
            }
        };
        let d = Draft {
            kind: Some(RouteKind::Core),
            cores: vec![core],
            bound,
            within,
            note,
            ..Draft::default()
        };
        if within == Some(true) {
            return d;
        }
        if fallback.is_none() && !(op.exact() && !cr) {
            fallback = Some(d);
        }
    }
    fallback.unwrap_or_else(|| Draft {
        kind: Some(RouteKind::Gap),
        note: format!(
            "no provider of the flow offers `{op}` at {} or wider",
            format_name(t.e, t.m)
        ),
        ..Draft::default()
    })
}

/// A composition of `ops` on format `t`, each at `t` or the wider format.
fn composition(find: &Finder, recipe: &str, ops: &[CoreOp], t: Format) -> Draft {
    let mut cores = Vec::new();
    let mut missing = Vec::new();
    for op in ops {
        match candidates(t, op.exact())
            .into_iter()
            .chain(std::iter::once(F32))
            .find_map(|q| find.find(op, q))
        {
            Some((c, _)) => cores.push(c),
            None => missing.push(op.to_string()),
        }
    }
    if missing.is_empty() {
        Draft {
            kind: Some(RouteKind::Composition),
            recipe: Some(recipe.to_string()),
            cores,
            note: "extra bits sized to the budget where the composition is built (design §4.12)".into(),
            ..Draft::default()
        }
    } else {
        Draft {
            kind: Some(RouteKind::Gap),
            recipe: Some(recipe.to_string()),
            cores,
            note: format!("no provider of the flow offers {}", missing.join(", ")),
            ..Draft::default()
        }
    }
}

/// The first route that is not a gap, else the last gap.
fn first(drafts: impl IntoIterator<Item = Draft>) -> Draft {
    let mut last = Draft::default();
    for d in drafts {
        if d.kind != Some(RouteKind::Gap) {
            return d;
        }
        last = d;
    }
    last
}

fn float(name: &str) -> Option<Format> {
    format(name)
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

fn route_of(find: &Finder, p: &Primitive, numerics: &NumericsConfig) -> Option<Route> {
    if p.kind != Kind::Op {
        return None;
    }
    let src = p.types.first()?;
    let dst = p.types.get(1).unwrap_or(src);
    let (fs, fd) = (float(src), float(dst));
    if fs.is_none() && fd.is_none() {
        return None;
    }
    // The format the op's accuracy is measured in: the result's for `convert`, else the operand's.
    let (t, tname) = if p.op == "convert" {
        (fd.or(fs)?, if fd.is_some() { dst.as_str() } else { src.as_str() })
    } else {
        (fs?, src.as_str())
    };
    let budget = numerics.budget(tname, &p.op);
    let one = |op: CoreOp| single(find, &op, t, budget);
    let d = match p.op.as_str() {
        "add" => one(CoreOp::Add),
        "subtract" => one(CoreOp::Sub),
        "multiply" => one(CoreOp::Mul),
        "divide" => one(CoreOp::Div),
        "sqrt" => one(CoreOp::Sqrt),
        "exponential" => one(CoreOp::Exp),
        "log" => one(CoreOp::Ln),
        "sine" => one(CoreOp::Sin),
        "cosine" => one(CoreOp::Cos),
        "compare" => one(CoreOp::Compare),
        "maximum" | "minimum" | "clamp" => {
            let mut d = one(CoreOp::Compare);
            d.recipe = Some("select by the compare core; NaN and signed-zero glue".into());
            d
        }
        "rsqrt" => first([
            one(CoreOp::Rsqrt),
            composition(find, "sqrt(1/x)", &[CoreOp::Div, CoreOp::Sqrt], t),
        ]),
        "power" => first([
            one(CoreOp::Pow),
            composition(
                find,
                "exp2(y log2 x), wider log2",
                &[CoreOp::Log2, CoreOp::Mul, CoreOp::Exp2],
                t,
            ),
            composition(
                find,
                "exp(y ln x), wider ln",
                &[CoreOp::Ln, CoreOp::Mul, CoreOp::Exp],
                t,
            ),
        ]),
        "tanh" => composition(
            find,
            "x - x^3/3 for small x, else (e^2x - 1)/(e^2x + 1)",
            &[CoreOp::Exp, CoreOp::Add, CoreOp::Sub, CoreOp::Mul, CoreOp::Div],
            t,
        ),
        "log_plus_one" => composition(
            find,
            "Goldberg: x ln(w)/(w - 1), w = 1 + x",
            &[CoreOp::Add, CoreOp::Sub, CoreOp::Ln, CoreOp::Div, CoreOp::Mul],
            t,
        ),
        "round_nearest_even" | "floor" | "ceil" => {
            let rnd = match p.op.as_str() {
                "floor" => Rounding::Floor,
                "ceil" => Rounding::Ceil,
                _ => Rounding::NearestEven,
            };
            let width = precision(t) + 1;
            composition(
                find,
                "to integer and back for |x| < 2^(p-1), x itself above; sign of zero kept",
                &[
                    CoreOp::ToInt {
                        rnd,
                        signed: true,
                        width,
                    },
                    CoreOp::FromInt { signed: true, width },
                ],
                t,
            )
        }
        "reduce_precision" => {
            if t.e <= 5 && t.m <= 10 {
                Draft::glue("already within (5, 10)")
            } else {
                let mut d = one(CoreOp::Convert {
                    exp_width: 5,
                    sig_width: 10,
                });
                d.recipe = Some("narrow to (5, 10), widen exactly".into());
                d
            }
        }
        "remainder" => Draft {
            kind: Some(RouteKind::Excluded),
            note: "constant computation only (design §4.9)".into(),
            ..Draft::default()
        },
        "abs" | "negate" | "sign" | "is_finite" | "select" => Draft::glue("bit-level"),
        "convert" => match (fs, fd) {
            (Some(a), Some(b)) => {
                // E4M3FN decodes exactly to f16.
                let a = if a.flavour == Flavour::FiniteNan { F16 } else { a };
                if b.flavour == Flavour::FiniteNan {
                    Draft::glue("E4M3FN encode, round to nearest even")
                } else if b.e >= a.e && b.m >= a.m {
                    Draft::glue("exact widening")
                } else {
                    single(
                        find,
                        &CoreOp::Convert {
                            exp_width: b.e,
                            sig_width: b.m,
                        },
                        a,
                        budget,
                    )
                }
            }
            (Some(a), None) => match int_type(dst) {
                Some((signed, width)) => {
                    let mut d = single(
                        find,
                        &CoreOp::ToInt {
                            rnd: Rounding::TowardZero,
                            signed,
                            width,
                        },
                        a,
                        budget,
                    );
                    d.recipe = Some("NaN to 0 glue".into());
                    d
                }
                None => Draft::glue("x != 0"),
            },
            (None, Some(b)) => match int_type(src) {
                Some((signed, width)) => single(find, &CoreOp::FromInt { signed, width }, b, budget),
                None => Draft::glue("0 or 1"),
            },
            (None, None) => return None,
        },
        _ => Draft {
            kind: Some(RouteKind::Gap),
            note: format!("no route for `{}`", p.op),
            ..Draft::default()
        },
    };
    Some(Route {
        key: p.key.clone(),
        op: p.op.clone(),
        types: p.types.clone(),
        kind: d.kind.unwrap_or(RouteKind::Gap),
        recipe: d.recipe,
        cores: d.cores,
        budget_ulp: budget,
        bound_ulp: d.bound,
        within_budget: d.within,
        note: d.note,
    })
}

/// The routes of every float op of `cat` in `flow`, through `providers` in order.
pub fn routes(
    cat: &[Primitive],
    flow: &str,
    providers: &[&'static dyn IpProvider],
    numerics: &NumericsConfig,
) -> Routes {
    let find = Finder { providers };
    let routes: Vec<Route> = cat.iter().filter_map(|p| route_of(&find, p, numerics)).collect();
    let gaps = routes
        .iter()
        .filter(|r| r.kind == RouteKind::Gap)
        .map(|r| r.key.clone())
        .collect();
    Routes {
        schema: ROUTES_SCHEMA.into(),
        flow: flow.into(),
        providers: providers.iter().map(|p| p.name().to_string()).collect(),
        routes,
        gaps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::characterize::{CharacterizeConfig, catalog};
    use crate::ip::PROVIDERS;

    fn routes_for(flow: &str) -> Routes {
        let cfg = CharacterizeConfig {
            float_formats: vec![
                "f32".into(),
                "f16".into(),
                "bf16".into(),
                "f8e4m3fn".into(),
                "f8e5m2".into(),
            ],
            ..CharacterizeConfig::default()
        };
        let cat = catalog::catalog(&cfg).unwrap();
        let providers = cfg.providers_of(flow).unwrap();
        routes(&cat, flow, &providers, &NumericsConfig::default())
    }

    fn get<'a>(r: &'a Routes, key: &str) -> &'a Route {
        r.routes
            .iter()
            .find(|x| x.key == key)
            .unwrap_or_else(|| panic!("no route {key}"))
    }

    #[test]
    fn every_float_op_has_a_route_in_both_flows() {
        for flow in ["dc", "yosys"] {
            let r = routes_for(flow);
            assert!(r.gaps.is_empty(), "{flow}: gaps {:?}", r.gaps);
            assert!(r.routes.len() > 100, "{}", r.routes.len());
        }
    }

    #[test]
    fn routes_follow_the_budget_and_the_flow() {
        let dc = routes_for("dc");
        let open = routes_for("yosys");
        // f32 exp: DesignWare's 2 ulps meet the f32 budget of 2; f16 exp (budget 1) runs through f32.
        let e = get(&dc, "exponential@f32");
        assert_eq!(
            (e.cores[0].module.as_str(), e.within_budget),
            ("DW_fp_exp", Some(true))
        );
        let e = get(&dc, "exponential@f16");
        assert_eq!(e.cores[0].format, "f32");
        assert_eq!(e.bound_ulp, Some(0.5 + 2.0 * 2f64.powi(-13)));
        // The open flow: FloPoCo is faithful (1 ulp) at f16 already.
        let e = get(&open, "exponential@f16");
        assert_eq!(
            (e.cores[0].provider.as_str(), e.cores[0].format.as_str()),
            ("flopoco", "f16")
        );
        // FP8 arithmetic through f16 (innocuous double rounding), E4M3FN never native.
        let a = get(&dc, "add@f8e4m3fn");
        assert_eq!(
            (a.cores[0].module.as_str(), a.cores[0].format.as_str()),
            ("DW_fp_add", "f16")
        );
        let a = get(&open, "multiply@f8e5m2");
        assert_eq!(a.cores[0].format, "f8e5m2");
        // Narrowing float conversion: HardFloat in both flows.
        assert_eq!(get(&dc, "convert@f32>bf16").cores[0].provider, "hardfloat");
        assert_eq!(get(&dc, "convert@bf16>f32").kind, RouteKind::Glue);
        // f32 sine: DesignWare unbounded in DC, XLS's absolute contract in the open flow.
        let s = get(&dc, "sine@f32");
        assert_eq!(
            (s.cores[0].module.as_str(), s.within_budget),
            ("DW_fp_sincos", None)
        );
        let s = get(&open, "sine@bf16");
        assert_eq!(
            (s.cores[0].provider.as_str(), s.cores[0].format.as_str()),
            ("xls_math", "f32")
        );
        // Compositions where no core exists.
        assert_eq!(get(&dc, "rsqrt@f32").cores[0].module, "DW_fp_invsqrt");
        assert_eq!(get(&open, "rsqrt@f32").kind, RouteKind::Composition);
        assert_eq!(get(&open, "power@f32").cores[0].module, "flopoco_e8m23_pow");
        assert_eq!(get(&dc, "power@f32").kind, RouteKind::Composition);
        assert_eq!(get(&dc, "tanh@bf16").kind, RouteKind::Composition);
        assert_eq!(get(&dc, "remainder@f32").kind, RouteKind::Excluded);
        // Exact ops are correctly rounded wherever they run.
        for r in dc.routes.iter().chain(&open.routes) {
            if r.kind == RouteKind::Core && r.budget_ulp == 0.0 {
                assert_eq!(r.within_budget, Some(true), "{}", r.key);
            }
        }
    }

    #[test]
    fn a_flow_without_a_provider_reports_gaps() {
        let cfg = CharacterizeConfig {
            float_formats: vec!["f32".into()],
            ..CharacterizeConfig::default()
        };
        let cat = catalog::catalog(&cfg).unwrap();
        let hf = PROVIDERS.by_name("hardfloat").unwrap();
        let r = routes(&cat, "yosys", &[hf], &NumericsConfig::default());
        assert!(r.gaps.contains(&"exponential@f32".to_string()), "{:?}", r.gaps);
        assert!(r.gaps.contains(&"sine@f32".to_string()));
        assert!(!r.gaps.contains(&"add@f32".to_string()));
        assert!(
            get(&r, "tanh@f32").note.contains("exp"),
            "{:?}",
            get(&r, "tanh@f32")
        );
    }
}
