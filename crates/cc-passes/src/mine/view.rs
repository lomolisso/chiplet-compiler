//! `views/mine.md`: per family the sweep and the selection (MNI next to the MIS count), and the patterns
//! found in several families.

use std::collections::HashMap;
use std::fmt::Write;

use super::algo::sweep::FamilyOut;
use super::json::{Overlap, Summary};

pub fn view(outs: &[FamilyOut], overlap: &Overlap, s: &Summary) -> String {
    let mut x = String::from("# Mining\n\n");
    let _ = writeln!(
        x,
        "{} node labels, {} edge labels; MIS by `{}`.\n\n| family | units | vertices | edges | τ | stopped by | reported | kept | selected |\n|---|---:|---:|---:|---:|---|---:|---:|---:|",
        s.node_labels, s.edge_labels, s.mis_algorithm
    );
    for (f, v) in &s.families {
        let _ = writeln!(
            x,
            "| {f} | {} | {} | {} | {} | {:?} | {} | {} | {} |",
            v.units,
            v.vertices,
            v.edges,
            v.tau.map(|t| t.to_string()).unwrap_or_default(),
            v.stopped_by,
            v.reported,
            v.kept,
            v.selected
        );
    }
    for o in outs {
        let f = o.sweep.family.name();
        let _ = writeln!(x, "\n## {f}\n\nSweep:");
        for st in &o.sweep.steps {
            let _ = writeln!(
                x,
                "- τ = {}: {}",
                st.tau,
                if st.finished {
                    format!("{} reported, {} kept, selection {}", st.reported, st.kept, st.selection)
                } else {
                    "out of time".into()
                }
            );
        }
        if o.selection.patterns.is_empty() {
            continue;
        }
        let by: HashMap<&str, _> = o.patterns.iter().map(|p| (p.id.as_str(), p)).collect();
        let _ = writeln!(
            x,
            "\n| # | pattern | vertices | MNI | Σ MIS | occurrences | score |\n|---:|---|---:|---:|---:|---:|---:|"
        );
        for (k, sel) in o.selection.patterns.iter().enumerate() {
            let p = by.get(sel.id.as_str());
            let shape = p.map(|p| p.code.split(';').next().unwrap_or("").replace(',', ", ")).unwrap_or_default();
            let _ = writeln!(
                x,
                "| {} | `{}` {} | {} | {} | {} | {} | {} |",
                k + 1,
                sel.id,
                shape.replace('|', "\\|"),
                sel.vertices,
                sel.mni,
                sel.mis,
                p.map(|p| p.occurrences.len()).unwrap_or(0),
                sel.score
            );
        }
    }
    if !overlap.patterns.is_empty() {
        let _ = writeln!(x, "\n## In several families\n\n| pattern | families | selected in |\n|---|---|---|");
        for p in &overlap.patterns {
            let names = |v: &[super::json::Family]| v.iter().map(|f| f.name()).collect::<Vec<_>>().join(", ");
            let _ = writeln!(x, "| `{}` | {} | {} |", p.id, names(&p.families), names(&p.selected_in));
        }
    }
    x
}
