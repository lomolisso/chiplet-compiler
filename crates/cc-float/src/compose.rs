//! Compositions of cores sized to an error budget (design §4.12, §6.4; docs/framework/float.md,
//! "Compositions").
//!
//! An operator without a core of its own is computed from cores at an inner precision `q = p + k` (`k`
//! extra significand bits) and rounded once to the format. With the composition's error constant `C`
//! (in units of `2^-q`, first order), the result is within `1/2 + C · 2^-k` ulps, so a budget of `T` ulps
//! needs `k >= ceil(log2(C / (T - 1/2)))` (Higham, *Accuracy and Stability of Numerical Algorithms*,
//! 2nd ed., SIAM 2002, §2.2; the constants per recipe below). The core errors `e` are in ulps of
//! precision `q`: 1/2 for a correctly rounded core, 1 for a faithful one, `n` for an `n`-ulp core.

use std::collections::BTreeMap;

use cc_core::adapters::Accuracy;
use cc_smt::float::{Flavour, Format};
use serde::{Deserialize, Serialize};

use crate::FloatError;

type Res<X> = Result<X, FloatError>;

/// An error budget per operator (catalog name) and format (`cc_smt` name), in ulps of the format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Budget(pub BTreeMap<String, BTreeMap<String, f64>>);

impl Budget {
    /// The default: NVIDIA's CUDA math library accuracy table for f32 (CUDA C++ Programming Guide,
    /// "Mathematical Functions", single precision: `expf` 2, `exp2f` 2, `logf` 1, `log2f` 1, `log1pf`
    /// 1, `sinf` 2, `cosf` 2, `tanhf` 2, `rsqrtf` 2, `powf` 4 ulp), 1 ulp for f16, bf16 and the 8-bit
    /// formats (CUDA rounds its half-precision functions from f32 once).
    pub fn cuda() -> Budget {
        let f32_ulps = [
            ("exponential", 2.0),
            ("exp2", 2.0),
            ("log", 1.0),
            ("log2", 1.0),
            ("log_plus_one", 1.0),
            ("sine", 2.0),
            ("cosine", 2.0),
            ("tanh", 2.0),
            ("rsqrt", 2.0),
            ("power", 4.0),
        ];
        let mut m: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        for (op, u) in f32_ulps {
            let e = m.entry(op.to_string()).or_default();
            e.insert("f32".into(), u);
            for f in ["f16", "bf16", "f8e4m3fn", "f8e5m2"] {
                e.insert(f.into(), 1.0);
            }
        }
        Budget(m)
    }

    /// The budget of `op` at `fmt`.
    pub fn get(&self, op: &str, fmt: Format) -> Option<f64> {
        self.0.get(op).and_then(|m| m.get(&fmt.name())).copied()
    }
}

/// A core's maximal error in ulps of its own precision, from its contract (`None`: no relative bound).
pub fn ulps(a: &Accuracy) -> Option<f64> {
    match a {
        Accuracy::CorrectlyRounded => Some(0.5),
        Accuracy::Faithful => Some(1.0),
        Accuracy::Ulp { ulp } => Some(*ulp),
        Accuracy::Absolute { .. } | Accuracy::Unbounded => None,
    }
}

/// The recipes (theory: float_composition notes in docs/framework/float.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recipe {
    /// `x^y = exp2(y · log2 x)`: the log's error is multiplied by the condition number of exp at
    /// `t = y log2 x`, `|t ln 2| <= tau` for a normal result: `C = ln2 · tau · (2 e_log + 1) + 2 e_exp`.
    Power,
    /// `tanh x = (e^2x - 1) / (e^2x + 1)` above `x1`, `x - x^3/3` below it, odd symmetry, 1 above the
    /// clamp: the cancellation in `e^2x - 1` costs `C = (2 e_exp + 2) / (2 x1)`.
    Tanh,
    /// Goldberg's `log1p x = x ln(w) / (w - 1)`, `w = fl(1 + x)` (Goldberg 1991, Theorem 4): `C = 2
    /// e_log + 3`.
    LogPlusOne,
    /// `rsqrt x = sqrt(RN(1/x))`: the reciprocal's error is halved by the root, `C = 2 e_sqrt + 1/2`.
    Rsqrt,
    /// A core of a wider format, rounded once: within `1/2 + e · 2^-k` ulps.
    Wider,
    /// `x^y` from one power core at the inner format (no core for C11 Annex F's special values: they
    /// are the glue of [`Recipe::Power`]); rounded once: `C = e_pow`, as [`Recipe::Wider`].
    PowerCore,
}

/// Precision `p` (with the hidden bit) of a format.
pub fn precision(f: Format) -> u32 {
    f.m + 1
}

/// `tau`: the largest `|t|` with `2^t` a result of `f` worth computing: `2^(e-1)` for an IEEE format
/// (its largest exponent plus one), `max(log2 448, 9)` for E4M3FN (its subnormals reach `2^-9`).
pub fn tau(f: Format) -> f64 {
    match f.flavour {
        Flavour::Ieee => (1i64 << (f.e - 1)) as f64,
        Flavour::FiniteNan => {
            let max = (crate::rtl::max_finite(f) as f64).log2();
            let sub = -(f.emin() as f64) + f64::from(f.m);
            max.max(sub)
        }
    }
}

/// The small-argument threshold of tanh: below it `x - x^3/3` is within `2^-(p+2)` relative (the
/// truncation `2 x^5 / 15`).
pub fn tanh_cubic_limit(f: Format) -> f64 {
    (7.5 * 2f64.powi(-(precision(f) as i32 + 2))).powf(0.25)
}

/// Above it tanh rounds to 1 in `f`: `(p + 2) ln 2 / 2`.
pub fn tanh_clamp(f: Format) -> f64 {
    f64::from(precision(f) + 2) * std::f64::consts::LN_2 / 2.0
}

/// The error constant `C` of a recipe (units of `2^-q`) for core errors `errors` (by role: `log`,
/// `exp`, `sqrt`; ulps of precision `q`).
pub fn constant(r: Recipe, f: Format, errors: &BTreeMap<String, f64>) -> Res<f64> {
    let e = |n: &str| {
        errors
            .get(n)
            .copied()
            .ok_or_else(|| FloatError::Compose(format!("{r:?}: no {n} core error")))
    };
    Ok(match r {
        Recipe::Power => std::f64::consts::LN_2 * tau(f) * (2.0 * e("log")? + 1.0) + 2.0 * e("exp")?,
        Recipe::Tanh => (2.0 * e("exp")? + 2.0) / (2.0 * tanh_cubic_limit(f)),
        Recipe::LogPlusOne => 2.0 * e("log")? + 3.0,
        Recipe::Rsqrt => 2.0 * e("sqrt")? + 0.5,
        Recipe::Wider => e("core")?,
        Recipe::PowerCore => e("pow")?,
    })
}

/// The smallest `k >= 0` with `1/2 + C 2^-k <= budget`.
pub fn extra_bits(c: f64, budget: f64) -> Res<u32> {
    if budget <= 0.5 {
        return Err(FloatError::Compose(format!(
            "a budget of {budget} ulp leaves nothing for the cores (a composition is within 1/2 ulp \
             only in the limit)"
        )));
    }
    let k = (c / (budget - 0.5)).log2().ceil();
    Ok(if k.is_finite() && k > 0.0 { k as u32 } else { 0 })
}

/// A sized composition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sizing {
    pub op: String,
    pub format: String,
    pub recipe: Recipe,
    pub budget_ulp: f64,
    /// The core errors used (ulps of precision `q`).
    pub core_errors: BTreeMap<String, f64>,
    pub constant: f64,
    pub k: u32,
    /// `q = p + k`: the inner cores' precision (with the hidden bit).
    pub q: u32,
    /// The first-order bound `1/2 + C 2^-k` (ulps of the format).
    pub bound_ulp: f64,
}

/// Size a recipe for `op` at `f` with the budget and the inner cores' contracts (by role).
pub fn size(
    op: &str,
    f: Format,
    r: Recipe,
    budget: &Budget,
    cores: &BTreeMap<String, Accuracy>,
) -> Res<Sizing> {
    let t = budget
        .get(op, f)
        .ok_or_else(|| FloatError::Compose(format!("no budget for {op} at {}", f.name())))?;
    let mut errors = BTreeMap::new();
    for (role, a) in cores {
        let e = ulps(a).ok_or_else(|| {
            FloatError::Compose(format!(
                "{op}: the {role} core has no relative error bound ({a:?}); the recipe needs one"
            ))
        })?;
        errors.insert(role.clone(), e);
    }
    let c = constant(r, f, &errors)?;
    let k = extra_bits(c, t)?;
    Ok(Sizing {
        op: op.into(),
        format: f.name(),
        recipe: r,
        budget_ulp: t,
        core_errors: errors,
        constant: c,
        k,
        q: precision(f) + k,
        bound_ulp: 0.5 + c * 2f64.powi(-(k as i32)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cr(roles: &[&str]) -> BTreeMap<String, Accuracy> {
        roles
            .iter()
            .map(|r| (r.to_string(), Accuracy::CorrectlyRounded))
            .collect()
    }

    /// The predictions of the composition notes (k for a budget of 1 ulp with correctly rounded cores).
    #[test]
    fn k_matches_the_predicted_table() {
        let mut b = Budget::cuda();
        for m in b.0.values_mut() {
            m.insert("f32".into(), 1.0);
        }
        let k = |op: &str, f: Format, r: Recipe, roles: &[&str]| size(op, f, r, &b, &cr(roles)).unwrap().k;
        assert_eq!(k("power", Format::F32, Recipe::Power, &["log", "exp"]), 9);
        assert_eq!(k("power", Format::F16, Recipe::Power, &["log", "exp"]), 6);
        assert_eq!(k("tanh", Format::F32, Recipe::Tanh, &["exp"]), 8);
        assert_eq!(k("tanh", Format::BF16, Recipe::Tanh, &["exp"]), 4);
        assert_eq!(k("log_plus_one", Format::F32, Recipe::LogPlusOne, &["log"]), 3);
        assert_eq!(k("rsqrt", Format::F32, Recipe::Rsqrt, &["sqrt"]), 2);
    }

    #[test]
    fn the_default_budget_is_the_cuda_table() {
        let b = Budget::cuda();
        assert_eq!(b.get("power", Format::F32), Some(4.0));
        assert_eq!(b.get("log", Format::F32), Some(1.0));
        assert_eq!(b.get("tanh", Format::BF16), Some(1.0));
        assert_eq!(b.get("exponential", Format::E4M3FN), Some(1.0));
    }

    #[test]
    fn absolute_cores_and_tight_budgets_are_refused() {
        let b = Budget::cuda();
        let abs: BTreeMap<String, Accuracy> =
            [("log".to_string(), Accuracy::Absolute { bound: 1e-7 })].into();
        assert!(size("log_plus_one", Format::F32, Recipe::LogPlusOne, &b, &abs).is_err());
        assert!(extra_bits(4.0, 0.5).is_err());
    }
}
