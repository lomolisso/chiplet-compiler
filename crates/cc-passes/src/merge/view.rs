//! `views/merge.md`: units per family, the area-vs-units curve, arrays, coverage.

use std::fmt::Write;

use crate::merge::json::{ArrayFile, CoverageFile, FuFile, Summary, Trace};

pub fn view(
    s: &Summary,
    fus: &[FuFile],
    arrays: &[ArrayFile],
    traces: &[Trace],
    cov: &CoverageFile,
) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "# merge\n");
    let _ = writeln!(
        o,
        "Kit `{}`, clique `{}`, requantization `{}`. {} units, {} arrays, {} configurations ({} proofs, {} array proofs). Coverage: {}/{} classes.\n",
        s.kit,
        s.algorithm,
        s.requant_variant,
        s.fus,
        s.arrays,
        s.configs,
        s.proofs,
        s.array_proofs,
        cov.covered,
        cov.total
    );
    if s.requant_variant == "fixed" {
        let _ = writeln!(
            o,
            "Fixed requantization: {} classes rescale with `rescale` units, {} keep fp32 (a tolerance lemma not proven; `requant.json`).\n",
            s.requant_fixed, s.requant_kept
        );
    }
    let _ = writeln!(o, "## Families\n");
    let _ = writeln!(
        o,
        "| family | units in | units out | area in µm² | area out µm² | steps | evaluated | bound ps | stop |"
    );
    let _ = writeln!(o, "|---|---|---|---|---|---|---|---|---|");
    for (f, x) in &s.families {
        let _ = writeln!(
            o,
            "| {f} | {} | {} | {:.1} | {:.1} | {} | {} | {:.0} | {} |",
            x.units, x.fus, x.initial_area_um2, x.area_um2, x.steps, x.evaluated, x.bound_ps, x.stopped_by
        );
    }
    let _ = writeln!(o, "\n## Area vs units\n");
    for t in traces {
        let _ = writeln!(o, "### {}\n", t.family.name());
        let _ = writeln!(o, "| units | area µm² | step gain µm² | depth ps | clique |");
        let _ = writeln!(o, "|---|---|---|---|---|");
        let _ = writeln!(o, "| {} | {:.1} | | | |", t.initial_units, t.initial_area_um2);
        let n = t.steps.len();
        for (i, st) in t.steps.iter().enumerate() {
            // Every step for short traces; otherwise every tenth and the last.
            if n > 60 && i % 10 != 9 && i + 1 != n {
                continue;
            }
            let _ = writeln!(
                o,
                "| {} | {:.1} | {:.1} | {:.0} | {}{} |",
                st.units,
                st.area_um2,
                st.gain_um2,
                st.depth_ps,
                st.clique.algorithm,
                if st.clique.optimal { "" } else { " (not optimal)" }
            );
        }
        let _ = writeln!(o);
    }
    let _ = writeln!(o, "## Units\n");
    let _ = writeln!(
        o,
        "| unit | family | covers | blocks | muxes | area µm² | depth ps |"
    );
    let _ = writeln!(o, "|---|---|---|---|---|---|---|");
    for f in fus {
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {:.1}{} | {:.0} |",
            f.id,
            f.family.name(),
            f.covers.len(),
            f.blocks.len(),
            f.muxes.len(),
            f.area_um2,
            if f.estimated { " (est.)" } else { "" },
            f.depth_ps
        );
    }
    let _ = writeln!(o, "\n## Arrays\n");
    let _ = writeln!(o, "| array | dataflow | element | host | classes | modes |");
    let _ = writeln!(o, "|---|---|---|---|---|---|");
    for a in arrays {
        let mut modes = std::collections::BTreeMap::<String, u32>::new();
        for c in &a.covers {
            for t in &c.tuples {
                *modes.entry(format!("{:?}", t.mode).to_lowercase()).or_default() += 1;
            }
        }
        let m: Vec<String> = modes.iter().map(|(k, v)| format!("{k} {v}")).collect();
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} |",
            a.id,
            a.dataflow,
            a.pe.join(" × "),
            a.host,
            a.covers.len(),
            m.join(", ")
        );
    }
    o
}
