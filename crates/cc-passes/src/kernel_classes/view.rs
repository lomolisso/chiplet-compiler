//! `views/sharing.md` and `views/classes.md`.

use std::fmt::Write;

use super::json::{ClassFile, ClassesSummary, Level, Sharing, Variant};

fn pct(x: f64) -> String {
    format!("{:.1} %", 100.0 * x)
}

fn level_row(out: &mut String, name: &str, l: &Level) {
    let _ = writeln!(
        out,
        "| {name} | {} | {} | {:.3} | {} | {} |",
        l.items,
        l.shared,
        l.mean_jaccard,
        pct(l.coverage_total),
        pct(l.coverage_mean)
    );
}

fn variant(out: &mut String, title: &str, v: &Variant) {
    let _ = writeln!(out, "## {title}\n");
    let _ = writeln!(
        out,
        "| level | items | used by ≥ 2 models | mean pairwise Jaccard | workload in shared items | per-model mean |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|");
    level_row(out, "components", &v.components);
    level_row(out, "kernel classes", &v.kernels);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| model | components | kernel classes | shared workload (components) | shared workload (kernel classes) |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---:|");
    for (m, n) in &v.components.per_model {
        let _ = writeln!(
            out,
            "| {m} | {n} | {} | {} | {} |",
            v.kernels.per_model.get(m).copied().unwrap_or_default(),
            pct(v.components.coverage.get(m).copied().unwrap_or_default()),
            pct(v.kernels.coverage.get(m).copied().unwrap_or_default()),
        );
    }
    let _ = writeln!(out);
}

/// The sharing report.
pub fn sharing(s: &Sharing) -> String {
    let mut out = String::from("# Sharing: components vs kernel classes\n\n");
    let _ = writeln!(
        out,
        "Kernel-level sharing exceeds component-level sharing (activation kernels, mean Jaccard and \
         shared workload): **{}**.\n",
        if s.kernel_sharing_exceeds_component_sharing {
            "yes"
        } else {
            "no"
        }
    );
    variant(
        &mut out,
        "Activation kernels (size-independent classes left out)",
        &s.activation,
    );
    variant(&mut out, "All kernels", &s.all);
    out
}

/// Classes per kind and core, and the classes with the most workload.
pub fn classes(s: &ClassesSummary, cs: &[ClassFile]) -> String {
    let mut out = String::from("# Kernel classes\n\n");
    let _ = writeln!(
        out,
        "{} classes from {} kernels ({} member kernels), {} instances; {} size-independent, {} low-rank, \
         {} with call-fed members, {} positional.\n",
        s.classes, s.kernels, s.members, s.rows, s.size_independent, s.low_rank, s.call_fed, s.positional
    );
    let _ = writeln!(out, "| core | classes |\n|---|---:|");
    for (k, n) in &s.cores {
        let _ = writeln!(out, "| `{k}` | {n} |");
    }
    let _ = writeln!(out, "\n| models using a class | classes |\n|---:|---:|");
    for (k, n) in &s.models_per_class {
        let _ = writeln!(out, "| {k} | {n} |");
    }
    let mut top: Vec<&ClassFile> = cs.iter().collect();
    top.sort_by(|a, b| b.workload.cmp(&a.workload).then_with(|| a.id.cmp(&b.id)));
    let _ = writeln!(
        out,
        "\n## The 40 classes with the most workload\n\n| class | core | models | members | instances | workload |\n|---|---|---:|---:|---:|---:|"
    );
    for c in top.iter().take(40) {
        let _ = writeln!(
            out,
            "| `{}` | `{}` | {} | {} | {} | {:.3e} |",
            c.id,
            c.core,
            c.models.len(),
            c.members.len(),
            c.instances,
            c.workload as f64
        );
    }
    out
}
