//! `views/components.md`: components × models.

use std::collections::{BTreeMap, BTreeSet};

use super::json::ComponentSummary;

/// The sharing table: one row per component, sorted by the number of models using it (then instances,
/// then id); one column per model with its instance count.
pub fn table(models: &[String], comps: &[ComponentSummary]) -> String {
    let mut rows: Vec<&ComponentSummary> = comps.iter().collect();
    rows.sort_by(|a, b| {
        (b.models.len(), b.instances.len(), &a.id).cmp(&(a.models.len(), a.instances.len(), &b.id))
    });
    let shared = comps.iter().filter(|c| c.models.len() >= 2).count();
    let instances: usize = comps.iter().map(|c| c.instances.len()).sum();
    let mut by_models: BTreeMap<usize, usize> = BTreeMap::new();
    for c in comps {
        *by_models.entry(c.models.len()).or_default() += 1;
    }
    let mut s = String::new();
    s.push_str("# Components\n\n");
    s.push_str(&format!(
        "{} components, {} instances with compute, {} models; {} components are used by two or more models.\n\n",
        comps.len(),
        instances,
        models.len(),
        shared
    ));
    s.push_str("| models using a component | components |\n|---:|---:|\n");
    for (k, n) in by_models.iter().rev() {
        s.push_str(&format!("| {k} | {n} |\n"));
    }
    s.push_str("\n| component | names | nodes | models | instances |");
    for m in models {
        s.push_str(&format!(" {m} |"));
    }
    s.push_str("\n|---|---|---:|---:|---:|");
    for _ in models {
        s.push_str("---:|");
    }
    s.push('\n');
    for c in rows {
        let names: BTreeSet<&str> = c.names.keys().map(String::as_str).collect();
        let mut nl: Vec<&str> = names.into_iter().take(3).collect();
        if c.names.len() > 3 {
            nl.push("…");
        }
        s.push_str(&format!(
            "| `{}` | {} | {} | {} | {} |",
            c.id,
            nl.join(", "),
            c.nodes,
            c.models.len(),
            c.instances.len()
        ));
        for m in models {
            match c.models.get(m) {
                Some(n) => s.push_str(&format!(" {n} |")),
                None => s.push_str(" |"),
            }
        }
        s.push('\n');
    }
    s
}
