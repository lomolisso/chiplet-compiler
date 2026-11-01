//! `views/normalize.md`: rewrites per rule and reductions per order, per component and weighted by
//! instances per model.

use std::collections::{BTreeMap, BTreeSet};

use super::algo::rules::RULES;
use super::json::NormalizedSummary;

pub fn table(comps: &[NormalizedSummary]) -> String {
    let mut s = String::from("# Normalize\n\n");
    let nodes_in: u64 = comps.iter().map(|c| c.nodes_in).sum();
    let nodes: u64 = comps.iter().map(|c| c.nodes).sum();
    s.push_str(&format!(
        "{} components: {nodes_in} nodes before, {nodes} after.\n\n",
        comps.len()
    ));
    s.push_str("| rule | components | rewrites | rewrites × instances |\n|---|---:|---:|---:|\n");
    for r in RULES {
        let (mut k, mut n, mut w) = (0u64, 0u64, 0u64);
        for c in comps {
            if let Some(&x) = c.rules.get(r) {
                k += 1;
                n += x;
                w += x * c.instances;
            }
        }
        s.push_str(&format!("| `{r}` | {k} | {n} | {w} |\n"));
    }
    let orders: BTreeSet<&String> = comps.iter().flat_map(|c| c.omega.keys()).collect();
    let mut models: BTreeMap<&String, (u64, u64, BTreeMap<&String, u64>)> = BTreeMap::new();
    for c in comps {
        for (m, &k) in &c.models {
            let e = models.entry(m).or_default();
            e.0 += k * c.nodes_in;
            e.1 += k * c.nodes;
            for (o, &x) in &c.omega {
                *e.2.entry(o).or_default() += k * x;
            }
        }
    }
    s.push_str("\n| model | nodes before × instances | after |");
    for o in &orders {
        s.push_str(&format!(" ω `{o}` |"));
    }
    s.push_str("\n|---|---:|---:|");
    for _ in &orders {
        s.push_str("---:|");
    }
    s.push('\n');
    for (m, (a, b, om)) in &models {
        s.push_str(&format!("| {m} | {a} | {b} |"));
        for o in &orders {
            s.push_str(&format!(" {} |", om.get(o).copied().unwrap_or(0)));
        }
        s.push('\n');
    }
    s
}
