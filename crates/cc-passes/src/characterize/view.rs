//! `views/library.md`: the library as tables (per kit and timing point, area and delay of every
//! primitive), the area-versus-width trends and their flags, and the pending primitives.

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::ip::routes::{RouteKind, Routes};

use super::json::{Coverage, Library, Status, Summary};
use super::sanity;

/// (kit, target) → (area, delay) of one type of one op.
type Cells = BTreeMap<(String, String), (Option<f64>, Option<f64>)>;

fn f(x: Option<f64>, digits: usize) -> String {
    x.map_or("-".into(), |v| format!("{v:.digits$}"))
}

pub fn view(lib: &Library, cov: &Coverage, s: &Summary, routes: &Routes) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "# Primitive library\n");
    let lec = match &lib.lec_tool {
        Some(t) => format!("{} proven equivalent ({t})", s.lec_equivalent),
        None => "equivalence check off".into(),
    };
    let _ = writeln!(
        o,
        "{} primitives ({} measured, {} provisional, {} pending), {} rows, {lec}. Tool: {}; {} sessions run, {} reused, {:.0} s in the tools.\n",
        s.primitives,
        s.measured,
        s.provisional,
        s.pending,
        s.rows,
        lib.tool,
        s.jobs_run,
        s.jobs_reused,
        s.tool_seconds
    );
    let _ = writeln!(o, "Units: area µm² (Liberty 1x scale), delay ps, leakage nW.\n");
    let _ = writeln!(
        o,
        "| kit | corner | library | time unit (ps) | driving cell | load |"
    );
    let _ = writeln!(o, "|---|---|---|---|---|---|");
    for k in &lib.kits {
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {:.4} |",
            k.name, k.corner, k.lib_name, k.time_unit_ps, k.driving_cell, k.load
        );
    }
    let _ = writeln!(o);

    // One table per op: rows by type, columns per (kit, target).
    let cols: Vec<(String, String)> = lib
        .targets
        .iter()
        .map(|t| (t.kit.clone(), t.target.clone()))
        .collect();
    let mut by_op: BTreeMap<String, BTreeMap<String, Cells>> = BTreeMap::new();
    for r in &lib.rows {
        let op = format!(
            "{}{}",
            r.op,
            r.variant.as_ref().map(|v| format!(".{v}")).unwrap_or_default()
        );
        by_op
            .entry(op)
            .or_default()
            .entry(r.types.join(">"))
            .or_default()
            .insert(
                (r.kit.clone(), r.target.clone()),
                (r.measured.area_um2, r.measured.delay_ps),
            );
    }
    let _ = writeln!(o, "## Area and delay\n");
    for (op, types) in &by_op {
        let _ = writeln!(o, "### {op}\n");
        let mut head = String::from("| type |");
        let mut rule = String::from("|---|");
        for (k, t) in &cols {
            let _ = write!(head, " {k} {t} area | delay |");
            rule.push_str("---|---|");
        }
        let _ = writeln!(o, "{head}\n{rule}");
        for (ty, cells) in types {
            let mut line = format!("| {ty} |");
            for c in &cols {
                let (a, d) = cells.get(c).copied().unwrap_or((None, None));
                let _ = write!(line, " {} | {} |", f(a, 2), f(d, 0));
            }
            let _ = writeln!(o, "{line}");
        }
        let _ = writeln!(o);
    }

    let _ = writeln!(o, "## Area against width\n");
    let _ = writeln!(o, "| series | kit | target | widths | area | exponent | flags |");
    let _ = writeln!(o, "|---|---|---|---|---|---|---|");
    let series = sanity::series(&lib.rows);
    let mut all_flags = Vec::new();
    for sr in &series {
        let widths: Vec<String> = sr.points.iter().map(|p| p.0.to_string()).collect();
        let areas: Vec<String> = sr.points.iter().map(|p| format!("{:.1}", p.1)).collect();
        let fl = sanity::flags(sr);
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} | {} |",
            sr.name,
            sr.kit,
            sr.target,
            widths.join(", "),
            areas.join(", "),
            f(sanity::exponent(&sr.points), 2),
            fl.len()
        );
        all_flags.extend(fl);
    }
    let _ = writeln!(o);
    if !all_flags.is_empty() {
        let _ = writeln!(o, "Flags (notes, not failures):\n");
        for fl in &all_flags {
            let _ = writeln!(o, "- {fl}");
        }
        let _ = writeln!(o);
    }

    let open: Vec<String> = lib
        .rows
        .iter()
        .filter(|r| r.inconclusive)
        .filter_map(|r| {
            let l = r.lec.as_ref()?;
            Some(format!(
                "- {} {} {}: {} passing, {} unverified, 0 failing",
                r.key, r.kit, r.target, l.passing, l.aborted
            ))
        })
        .collect();
    let _ = writeln!(
        o,
        "## Equivalence inconclusive within the time limit ({})\n",
        open.len()
    );
    for a in &open {
        let _ = writeln!(o, "{a}");
    }
    if !open.is_empty() {
        let _ = writeln!(o);
    }

    let pending: Vec<&str> = cov
        .primitives
        .iter()
        .filter(|p| p.status == Status::Pending)
        .map(|p| p.key.as_str())
        .collect();
    let _ = writeln!(o, "## Pending ({})\n", pending.len());
    if !pending.is_empty() {
        let _ = writeln!(o, "{}\n", pending.join(", "));
    }
    let _ = writeln!(
        o,
        "## Float IP routes ({}: {})\n\n{} float ops, {} gaps (`routes.json`).\n",
        routes.flow,
        routes.providers.join(", "),
        routes.routes.len(),
        routes.gaps.len()
    );
    for r in routes.routes.iter().filter(|r| r.kind == RouteKind::Gap) {
        let _ = writeln!(o, "- gap `{}`: {}", r.key, r.note);
    }
    if !routes.gaps.is_empty() {
        let _ = writeln!(o);
    }
    o
}
