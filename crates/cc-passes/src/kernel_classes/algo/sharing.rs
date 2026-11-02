//! The sharing report (docs/passes/kernel_classes.md §4.5): per model, the set of components and the set
//! of kernel classes it uses; Jaccard indices between models; and how much of each model's workload lies
//! in items that another model uses too.

use std::collections::{BTreeMap, BTreeSet};

use crate::kernel_classes::json::{Level, SHARING_SCHEMA, Sharing, Variant};

/// One kernel instance as the report sees it.
#[derive(Debug, Clone)]
pub struct Use<'a> {
    pub model: &'a str,
    pub component: &'a str,
    pub class: &'a str,
    pub weighted: u64,
    pub activation: bool,
}

fn level<'a>(models: &[String], uses: &[(&'a str, &'a str, u64)]) -> Level {
    let mut sets: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut users: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for &(m, item, _) in uses {
        sets.entry(m).or_default().insert(item);
        users.entry(item).or_default().insert(m);
    }
    let shared: BTreeSet<&str> = users
        .iter()
        .filter(|(_, m)| m.len() >= 2)
        .map(|(i, _)| *i)
        .collect();
    let empty = BTreeSet::new();
    let set = |m: &str| sets.get(m).unwrap_or(&empty);
    let mut jaccard = vec![vec![0.0; models.len()]; models.len()];
    let (mut sum, mut pairs) = (0.0, 0u64);
    for (i, a) in models.iter().enumerate() {
        for (j, b) in models.iter().enumerate() {
            let (x, y) = (set(a), set(b));
            let union = x.union(y).count();
            let v = if union == 0 {
                0.0
            } else {
                x.intersection(y).count() as f64 / union as f64
            };
            jaccard[i][j] = v;
            if i < j {
                sum += v;
                pairs += 1;
            }
        }
    }
    let mut tot: BTreeMap<&str, (u128, u128)> = BTreeMap::new();
    for &(m, item, w) in uses {
        let e = tot.entry(m).or_default();
        e.1 += u128::from(w);
        if shared.contains(item) {
            e.0 += u128::from(w);
        }
    }
    let frac = |(s, t): (u128, u128)| if t == 0 { 0.0 } else { s as f64 / t as f64 };
    let all = tot.values().fold((0u128, 0u128), |a, b| (a.0 + b.0, a.1 + b.1));
    let coverage: BTreeMap<String, f64> = models
        .iter()
        .map(|m| (m.clone(), frac(tot.get(m.as_str()).copied().unwrap_or_default())))
        .collect();
    let coverage_mean = if models.is_empty() {
        0.0
    } else {
        coverage.values().sum::<f64>() / models.len() as f64
    };
    Level {
        per_model: models.iter().map(|m| (m.clone(), set(m).len() as u64)).collect(),
        items: users.len() as u64,
        shared: shared.len() as u64,
        jaccard,
        mean_jaccard: if pairs == 0 { 0.0 } else { sum / pairs as f64 },
        coverage,
        coverage_total: frac(all),
        coverage_mean,
    }
}

fn variant(models: &[String], uses: &[Use<'_>], keep: impl Fn(&Use<'_>) -> bool) -> Variant {
    let kept: Vec<&Use<'_>> = uses.iter().filter(|u| keep(u)).collect();
    let comps: Vec<(&str, &str, u64)> = kept.iter().map(|u| (u.model, u.component, u.weighted)).collect();
    let classes: Vec<(&str, &str, u64)> = kept.iter().map(|u| (u.model, u.class, u.weighted)).collect();
    let mut workload: BTreeMap<String, u64> = models.iter().map(|m| (m.clone(), 0)).collect();
    for u in &kept {
        if let Some(w) = workload.get_mut(u.model) {
            *w = w.saturating_add(u.weighted);
        }
    }
    Variant {
        components: level(models, &comps),
        kernels: level(models, &classes),
        workload,
    }
}

/// The report over every kernel instance (`models`: every model of the run, sorted).
pub fn sharing(models: &[String], uses: &[Use<'_>]) -> Sharing {
    let all = variant(models, uses, |_| true);
    let activation = variant(models, uses, |u| u.activation);
    let exceeds = activation.kernels.mean_jaccard > activation.components.mean_jaccard
        && activation.kernels.coverage_total > activation.components.coverage_total;
    Sharing {
        schema: SHARING_SCHEMA.into(),
        models: models.to_vec(),
        all,
        activation,
        kernel_sharing_exceeds_component_sharing: exceeds,
    }
}
