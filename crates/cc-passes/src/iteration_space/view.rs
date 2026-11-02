//! Human views: `views/routing.md` (every class, its route and reason) and `views/iteration_space.md`
//! (the summary).

use std::fmt::Write;

use crate::iteration_space::json::{Route, Routing, Summary};

pub fn routing(r: &Routing) -> String {
    let mut s = String::from(
        "# Routing\n\n| class | class | core | route | dataflow | weighted workload | reason |\n|---|---|---|---|---|---:|---|\n",
    );
    let mut rows: Vec<_> = r.classes.iter().collect();
    rows.sort_by(|a, b| b.workload.cmp(&a.workload).then(a.id.cmp(&b.id)));
    for c in rows {
        let _ = writeln!(
            s,
            "| `{}` | {} | `{}` | {} | {} | {} | {} |",
            c.id,
            c.class.name(),
            c.core,
            match c.route {
                Route::Template => "template",
                Route::Datapath => "datapath",
            },
            c.dataflow.as_deref().unwrap_or(""),
            c.workload,
            c.reason.replace('|', "\\|")
        );
    }
    s
}

pub fn summary(x: &Summary) -> String {
    let mut s = String::from("# Iteration space\n\n");
    let _ = writeln!(
        s,
        "{} classes: {} uniform recurrences; {} routed to templates ({} weighted workload), {} to datapaths ({}). {} embedding pairs ({} searches over budget); {} proof files.\n",
        x.classes,
        x.uniform,
        x.templates,
        x.template_workload,
        x.datapaths,
        x.datapath_workload,
        x.embedding_pairs,
        x.embedding_over_budget,
        x.proofs
    );
    let table = |s: &mut String, title: &str, m: &std::collections::BTreeMap<String, u64>| {
        let _ = writeln!(s, "| {title} | classes |\n|---|---:|");
        for (k, v) in m {
            let _ = writeln!(s, "| {k} | {v} |");
        }
        s.push('\n');
    };
    table(
        &mut s,
        "kernel_classes kind -> skeleton class",
        &x.classes_by_kind,
    );
    table(&mut s, "template core", &x.template_cores);
    table(&mut s, "selected dataflow", &x.dataflows);
    table(&mut s, "datapath reason", &x.datapath_reasons);
    s
}
