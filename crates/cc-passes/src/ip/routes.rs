//! Fit checks (design §3.7, §4.12): every float op of the catalog with the wrapper and cores
//! [`super::plan`] chose for it in the configured flow, or the reason it has none (a gap, listed, never a
//! failure). `out/library/routes.json` records, per op, the route's kind, the wrapper (its `cc_float`
//! glue), the recipe, the cores (provider, core op, format, module, parameters, contract, timing), the
//! extra precision `k` of a composition, the `[numerics]` budget, the bound and whether it is within it.

use cc_core::adapters::{Accuracy, CoreSource, IpProvider, Subnormals, Timing};
use cc_core::config::NumericsConfig;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::plan::{Plan, Planner, Resolved, format_name};
use crate::characterize::catalog::{Kind, Primitive};

pub use super::plan::RouteKind;

pub const ROUTES_SCHEMA: &str = "cc.characterize.routes/2";

/// One core of a route.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RouteCore {
    pub provider: String,
    /// The core operation (`cc_core::adapters::CoreOp` syntax).
    pub op: String,
    /// The format the provider was asked at (its operands'): a format name (`f32`, `f16`, `bf16`,
    /// `f8e5m2`) or `e<E>m<M>`.
    pub format: String,
    pub module: String,
    /// The instance's parameters (Verilog literals).
    pub params: BTreeMap<String, String>,
    /// The contract's accuracy: `correctly_rounded`, `faithful`, `<n> ulp`, `absolute <bound>`,
    /// `unbounded`; its subnormal mode (`ieee`, `flush`) and whether it has the IEEE special values.
    pub accuracy: String,
    pub subnormals: String,
    pub ieee_specials: bool,
    /// Where the RTL comes from: `designware` (the vendor library), `hardfloat`, `flopoco`, `xls_dslx`.
    pub source: String,
    /// `combinational` (every core of a route is).
    pub timing: String,
}

/// The route of one catalog primitive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Route {
    pub key: String,
    pub op: String,
    pub types: Vec<String>,
    pub kind: RouteKind,
    /// The `cc_float` glue module of the wrapper (`glue_<op>_<format>...`), when there is one.
    pub glue: Option<String>,
    /// What is built around or from the cores.
    pub recipe: Option<String>,
    pub cores: Vec<RouteCore>,
    /// Extra significand bits of a composition's inner cores (`q = p + k`).
    pub k: Option<u32>,
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

fn accuracy_text(a: &Accuracy) -> String {
    match a {
        Accuracy::CorrectlyRounded => "correctly_rounded".to_string(),
        Accuracy::Faithful => "faithful".to_string(),
        Accuracy::Ulp { ulp } => format!("{ulp} ulp"),
        Accuracy::Absolute { bound } => format!("absolute {bound:e}"),
        Accuracy::Unbounded => "unbounded".to_string(),
    }
}

pub fn route_core(r: &Resolved) -> RouteCore {
    let o = &r.offer;
    RouteCore {
        provider: r.provider.to_string(),
        op: r.op.to_string(),
        format: format_name(r.asked),
        module: o.instance.module.clone(),
        params: o.instance.params.clone(),
        accuracy: accuracy_text(&o.contract.accuracy),
        subnormals: match o.contract.subnormals {
            Subnormals::Ieee => "ieee",
            Subnormals::Flush => "flush",
        }
        .into(),
        ieee_specials: o.contract.ieee_specials,
        source: match &o.source {
            CoreSource::Library { name } => name.clone(),
            CoreSource::Generated { generator } => match generator {
                cc_core::adapters::Generator::Hardfloat { .. } => "hardfloat".into(),
                cc_core::adapters::Generator::Flopoco { .. } => "flopoco".into(),
                cc_core::adapters::Generator::XlsDslx { .. } => "xls_dslx".into(),
            },
        },
        timing: match o.timing {
            Timing::Combinational => "combinational".into(),
            Timing::Iterative { max_cycles } => format!("iterative {max_cycles}"),
        },
    }
}

/// The route of a plan.
pub fn route(p: &Primitive, plan: &Plan) -> Route {
    Route {
        key: p.key.clone(),
        op: p.op.clone(),
        types: p.types.clone(),
        kind: plan.kind,
        glue: plan.glue.as_ref().map(|g| g.name()),
        recipe: plan.recipe.clone(),
        cores: plan.cores.iter().map(route_core).collect(),
        k: plan.k,
        budget_ulp: plan.budget,
        bound_ulp: plan.bound,
        within_budget: plan.within,
        note: plan.note.clone(),
    }
}

/// The plan of a catalog primitive (`None`: not a float op).
pub fn plan_of(planner: &Planner, p: &Primitive) -> Option<Plan> {
    if p.kind != Kind::Op {
        return None;
    }
    planner.plan(&p.op, p.variant.as_deref(), &p.types)
}

/// The routes of every float op of `cat` in `flow`, through `providers` in order.
pub fn routes(
    cat: &[Primitive],
    flow: &str,
    providers: &[&'static dyn IpProvider],
    numerics: &NumericsConfig,
) -> Routes {
    let planner = Planner { providers, numerics };
    let routes: Vec<Route> = cat
        .iter()
        .filter_map(|p| plan_of(&planner, p).map(|plan| route(p, &plan)))
        .collect();
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
            tool: flow.into(),
            float_formats: vec![
                "f32".into(),
                "f16".into(),
                "bf16".into(),
                "f8e4m3fn".into(),
                "f8e5m2".into(),
            ],
            ..CharacterizeConfig::default()
        };
        let n = NumericsConfig::default();
        let cat = catalog::catalog(&cfg, &n).unwrap();
        let providers = cfg.providers_of(flow).unwrap();
        routes(&cat, flow, &providers, &n)
    }

    fn get<'a>(r: &'a Routes, key: &str) -> &'a Route {
        r.routes
            .iter()
            .find(|x| x.key == key)
            .unwrap_or_else(|| panic!("no route {key}"))
    }

    #[test]
    fn every_float_op_of_the_default_formats_has_a_route_in_both_flows() {
        for flow in ["dc", "yosys"] {
            let r = routes_for(flow);
            let gaps: Vec<&Route> = r
                .routes
                .iter()
                .filter(|x| x.kind == RouteKind::Gap && !x.types.iter().any(|t| t == "f8e5m2"))
                .collect();
            assert!(gaps.is_empty(), "{flow}: gaps {gaps:#?}");
            assert!(r.routes.len() > 100, "{}", r.routes.len());
            // Every core of a route is combinational.
            for x in &r.routes {
                for c in &x.cores {
                    assert_eq!(c.timing, "combinational", "{flow} {}", x.key);
                }
            }
        }
    }

    #[test]
    fn routes_follow_the_budget_and_the_flow() {
        let dc = routes_for("dc");
        let open = routes_for("yosys");
        // f32 exp: DesignWare's 2 ulps meet the f32 budget of 2, on a core with one more exponent bit.
        let e = get(&dc, "exponential@f32");
        assert_eq!(
            (
                e.cores[0].module.as_str(),
                e.cores[0].format.as_str(),
                e.within_budget
            ),
            ("DW_fp_exp", "e9m23", Some(true))
        );
        // f16 exp (budget 1) runs through f32 in both flows (FloPoCo flushes f16 subnormals).
        let e = get(&dc, "exponential@f16");
        assert_eq!(e.cores[0].format, "f32");
        assert_eq!(e.bound_ulp, Some(0.5 + 2.0 * 2f64.powi(-13)));
        let e = get(&open, "exponential@f16");
        assert_eq!(
            (e.cores[0].provider.as_str(), e.cores[0].format.as_str()),
            ("flopoco", "f32")
        );
        // FP8 arithmetic on f16 cores.
        let a = get(&dc, "add@f8e4m3fn");
        assert_eq!(
            (a.cores[0].module.as_str(), a.cores[0].format.as_str()),
            ("DW_fp_add", "f16")
        );
        // Narrowing float conversion: HardFloat in both flows.
        assert_eq!(get(&dc, "convert@f32>bf16").cores[0].provider, "hardfloat");
        assert_eq!(get(&dc, "convert@bf16>f32").kind, RouteKind::Glue);
        // An integer into E4M3FN through a format that holds it exactly.
        assert_eq!(get(&dc, "convert@i8>f8e4m3fn").cores[0].format, "f16");
        assert_eq!(get(&dc, "convert@i16>f8e4m3fn").cores[0].format, "f32");
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
        // Open-flow division and square root on FloPoCo (HardFloat's are iterative).
        assert_eq!(get(&open, "divide@f32").cores[0].provider, "flopoco");
        assert_eq!(get(&open, "sqrt@bf16").cores[0].provider, "flopoco");
        // Compositions where no core exists.
        assert_eq!(get(&dc, "rsqrt@f32").cores[0].module, "DW_fp_invsqrt");
        assert_eq!(get(&open, "rsqrt@f32").kind, RouteKind::Composition);
        assert_eq!(get(&dc, "tanh@bf16").kind, RouteKind::Composition);
        // f32 power: DesignWare's log2 stops at 29 significand bits, short of the k the budget needs,
        // so one power core (FloPoCo) with the special values as glue, within 4 ulps.
        let p = get(&dc, "power@f32");
        assert_eq!(
            (p.kind, p.cores[0].provider.as_str(), p.within_budget),
            (RouteKind::Composition, "flopoco", Some(true))
        );
        assert_eq!(get(&dc, "power@bf16").cores[0].module, "DW_fp_log2");
        assert_eq!(get(&dc, "remainder@f32").kind, RouteKind::Excluded);
        // Exact ops are correctly rounded wherever they run.
        for r in dc.routes.iter().chain(&open.routes) {
            if r.kind == RouteKind::Core && r.budget_ulp == 0.0 && r.bound_ulp.is_some() {
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
        let n = NumericsConfig::default();
        let cat = catalog::catalog(&cfg, &n).unwrap();
        let hf = PROVIDERS.by_name("hardfloat").unwrap();
        let r = routes(&cat, "yosys", &[hf], &n);
        assert!(r.gaps.contains(&"exponential@f32".to_string()), "{:?}", r.gaps);
        assert!(r.gaps.contains(&"sine@f32".to_string()));
        // HardFloat's divider is iterative: no combinational division either.
        assert!(r.gaps.contains(&"divide@f32".to_string()));
        assert!(get(&r, "divide@f32").note.contains("iterative"));
        assert!(!r.gaps.contains(&"add@f32".to_string()));
    }
}
