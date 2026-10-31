//! The recipe audit (spec §2) over the census: the same rules as `verify/dtype_audit.py` `check`,
//! reimplemented so `verify` does not trust the exporter's own audit.

use std::collections::{BTreeMap, BTreeSet};

use super::json::{Audit, SigSummary};

/// Every violation, as a human-readable line (empty = the census meets the recipe).
pub fn check(
    audit: &Audit,
    per_phase: &BTreeMap<String, BTreeMap<String, SigSummary>>,
    config: &str,
    param_dtypes: &BTreeSet<String>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let mut seen = BTreeSet::new();
    for (ph, summ) in per_phase {
        let ph_audit = audit.phases.get(ph);
        let allow = ph_audit.and_then(|a| a.allow.as_ref()).unwrap_or(&audit.allow);
        for (sig, s) in summ {
            seen.insert(sig.clone());
            if !allow.contains(sig) {
                errors.push(format!("{ph}: {sig} x{} not allowed by the recipe", s.count));
            }
        }
        for sig in ph_audit.map(|a| a.require.as_slice()).unwrap_or_default() {
            if !summ.contains_key(sig) {
                errors.push(format!("{ph}: required {sig} not found"));
            }
        }
        if let Some(counts) = audit.counts.get(config).and_then(|c| c.get(ph)) {
            for (sig, n) in counts {
                let got = summ.get(sig).map_or(0, |s| s.count);
                if got != *n {
                    errors.push(format!("{ph}: {sig} x{got}, recipe expects x{n}"));
                }
            }
        }
    }
    for sig in &audit.require {
        if !seen.contains(sig) {
            errors.push(format!("required {sig} not found"));
        }
    }
    for (dt, why) in &audit.params {
        if !param_dtypes.contains(dt) {
            errors.push(format!("no parameter stored as {dt} ({why})"));
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::json::{Allow, PhaseAudit};

    fn s(n: u64) -> SigSummary {
        SigSummary {
            count: n,
            flops: None,
        }
    }

    fn audit() -> Audit {
        Audit {
            allow: Allow::Explained([("dot_general i8xi8->i32".to_string(), "fc".to_string())].into()),
            require: vec!["dot_general i8xi8->i32".into()],
            phases: [(
                "vae".to_string(),
                PhaseAudit {
                    allow: Some(Allow::List(vec!["convolution f32xf32->f32".to_string()])),
                    require: vec![],
                },
            )]
            .into(),
            counts: [(
                "reduced".to_string(),
                [(
                    "fwd".to_string(),
                    [("dot_general i8xi8->i32".to_string(), 2)].into(),
                )]
                .into(),
            )]
            .into(),
            params: [("int8".to_string(), "weights".to_string())].into(),
        }
    }

    #[test]
    fn passes_and_fails_like_dtype_audit() {
        let dt: BTreeSet<String> = ["int8".to_string()].into();
        let mut pp = BTreeMap::new();
        pp.insert(
            "fwd".to_string(),
            [("dot_general i8xi8->i32".to_string(), s(2))].into(),
        );
        pp.insert(
            "vae".to_string(),
            [("convolution f32xf32->f32".to_string(), s(1))].into(),
        );
        assert!(check(&audit(), &pp, "reduced", &dt).is_empty());
        // count mismatch, disallowed signature, missing param dtype
        pp.insert(
            "fwd".to_string(),
            [
                ("dot_general i8xi8->i32".to_string(), s(3)),
                ("dot_general f32xf32->f32".to_string(), s(1)),
            ]
            .into(),
        );
        let e = check(&audit(), &pp, "reduced", &BTreeSet::new());
        assert_eq!(e.len(), 3, "{e:?}");
    }
}
