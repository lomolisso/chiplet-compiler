//! `views/softfloat.md`: evidence and accuracy per primitive, grouped by op.

use std::collections::BTreeMap;
use std::fmt::Write;

use cc_pipeline::{PassContext, PassError};

use super::VIEW;
use super::is_approximate;
use super::json::{CoverageFile, Evidence, Summary};

pub fn write(ctx: &PassContext, cov: &CoverageFile, s: &Summary) -> Result<(), PassError> {
    let mut t = String::new();
    let _ = writeln!(t, "# softfloat\n");
    let _ = writeln!(
        t,
        "{} primitives: {} proven, {} exhaustive, {} tested; {} exact, {} correctly rounded, {} faithful; {} approximate ops not built yet. {} vectors. Library {}.\n",
        s.primitives,
        s.proven,
        s.exhaustive,
        s.tested,
        s.exact,
        s.correctly_rounded,
        s.faithful,
        s.approximate,
        s.vectors,
        &s.library_hash[..s.library_hash.len().min(12)]
    );
    let mut by_op: BTreeMap<&str, Vec<&super::json::Entry>> = BTreeMap::new();
    for e in &cov.entries {
        by_op.entry(e.op.as_str()).or_default().push(e);
    }
    let _ = writeln!(
        t,
        "| op | primitives | proven | exhaustive | tested | max error (ulp) | IR nodes (max) |"
    );
    let _ = writeln!(t, "|---|---|---|---|---|---|---|");
    for (op, es) in &by_op {
        let n = |ev: Evidence| es.iter().filter(|e| e.evidence == ev).count();
        let err = es
            .iter()
            .filter_map(|e| e.max_ulp)
            .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))))
            .map_or_else(|| "exact".to_string(), |v| format!("{v:.4}"));
        let _ = writeln!(
            t,
            "| {op} | {} | {} | {} | {} | {err} | {} |",
            es.len(),
            n(Evidence::Proven),
            n(Evidence::Exhaustive),
            n(Evidence::Tested),
            es.iter().map(|e| e.ir_nodes).max().unwrap_or(0)
        );
    }
    let unproven: Vec<&str> = cov
        .entries
        .iter()
        .filter(|e| e.evidence != Evidence::Proven && !is_approximate(&e.op))
        .map(|e| e.key.as_str())
        .collect();
    if !unproven.is_empty() {
        let _ = writeln!(t, "\nNot proven (solver time limit): {}.", unproven.join(", "));
    }
    ctx.dir()
        .write_view(VIEW, &t)
        .map_err(|e| PassError::failed(e.to_string()))?;
    Ok(())
}
