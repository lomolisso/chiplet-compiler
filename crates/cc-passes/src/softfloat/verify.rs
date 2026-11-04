//! `verify` for softfloat (docs/passes/softfloat.md §6). It reads only `out/softfloat/` and the
//! configuration: the primitive list, the queries and the reference are recomputed, the proofs re-checked
//! by the other solver; nothing of the run's own bookkeeping is trusted.

use std::collections::BTreeMap;

use cc_core::{Check, VerifyReport};
use cc_pipeline::{PassContext, PassError};
use cc_smt::Verdict;
use rayon::prelude::*;

use super::json::{Accuracy, BuildResult, CoverageFile, Evidence, Summary};
use super::{
    OUT_IN_CONTAINER, SOFTFLOAT, SoftfloatConfig, is_approximate, primitives, python, python_runner, query_of,
};
use crate::characterize::catalog::{self, Primitive};
use crate::datapaths::json::ProofResult;

const FILES: &[&str] = &[
    "kernel.py",
    "kernel.mlir",
    "xls/test_block.cpp",
    "fu.ir",
    "fu.opt.ir",
    "rtl.v",
    "vectors.npz",
    "result.json",
];

/// The proof files of an exactly specified op.
const PROOF_FILES: &[&str] = &["proof.smt2", "proof.result"];

/// Slack on a measured error in ulps: the reference's own error allowance (2^-48 relative, at most 2^-24
/// ulp in these formats) and float64 rounding of the ratio.
const ULP_SLACK: f64 = 1e-6;

/// Does `text` name a float type (`f16`, `f32`, `f64`, `bf16` as a whole word, or as the element type of
/// a shape such as `4x8xbf16` or `?xf32`)? Allo's XLS backend rejects such a module; the library must be
/// integer-only.
pub fn has_float_type(text: &str) -> bool {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|w| {
            let mut parts: Vec<&str> = w.split('x').collect();
            let elem = parts.pop().unwrap_or_default();
            matches!(elem, "f16" | "f32" | "f64" | "bf16")
                && parts.iter().all(|d| d.chars().all(|c| c.is_ascii_digit()))
        })
}

fn sample<T>(v: &[T], n: usize) -> String
where
    T: std::fmt::Display,
{
    v.iter()
        .take(n)
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

pub fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: SoftfloatConfig = ctx.options()?;
    let out = ctx.output(SOFTFLOAT)?;
    let dir = out.dir();
    let mut rep = VerifyReport::new("softfloat");
    let cov: CoverageFile = out.read_json("coverage.json")?;
    let summary: Summary = out.read_json("summary.json")?;

    // The primitive list regenerates from the configuration.
    let (prims, approx) = match primitives(&cfg) {
        Ok(x) => x,
        Err(e) => {
            rep.push(Check::new("primitives", false).note(e));
            return Ok(rep);
        }
    };
    let keys: Vec<&str> = prims.iter().map(|p| p.key.as_str()).collect();
    let listed: Vec<&str> = cov.entries.iter().map(|x| x.key.as_str()).collect();
    let same = keys == listed && approx == cov.approximate;
    rep.push(
        Check::new("primitives", same)
            .count("primitives", prims.len() as u64)
            .count("approximate", approx.len() as u64)
            .path("out/softfloat/coverage.json")
            .note(if same {
                "every float op primitive of the catalog is built (approximate ops not built yet listed apart)"
            } else {
                "coverage.json does not list the catalog's float primitives"
            }),
    );
    if !same {
        return Ok(rep);
    }
    let by_id: BTreeMap<&str, &Primitive> = prims.iter().map(|p| (p.id.as_str(), p)).collect();

    // Files, build results, integer-only modules, RTL modules.
    let mut missing = Vec::new();
    let mut bad_build = Vec::new();
    let mut floats = Vec::new();
    let mut bad_rtl = Vec::new();
    let mut bad_exhaustive = Vec::new();
    for x in &cov.entries {
        let proof_files = if is_approximate(&x.op) {
            &[][..]
        } else {
            PROOF_FILES
        };
        for f in FILES.iter().chain(proof_files) {
            if !dir.join(format!("ops/{}/{f}", x.id)).is_file() {
                missing.push(format!("{}/{f}", x.id));
            }
        }
        match out.read_json::<BuildResult>(&format!("ops/{}/result.json", x.id)) {
            Ok(r) => {
                let mism = r.reference.as_ref().map_or(u64::MAX, |c| c.mismatches);
                if !r.ok || mism != 0 || r.ir_vs_llvm != 0 || r.vectors != x.vectors || r.vectors == 0 {
                    bad_build.push(x.key.clone());
                }
            }
            Err(_) => bad_build.push(x.key.clone()),
        }
        if let Ok(m) = std::fs::read_to_string(dir.join(format!("ops/{}/kernel.mlir", x.id)))
            && has_float_type(&m)
        {
            floats.push(x.key.clone());
        }
        let rtl = std::fs::read_to_string(dir.join(format!("ops/{}/rtl.v", x.id))).unwrap_or_default();
        if !rtl.contains(&format!("module {}(", x.id)) {
            bad_rtl.push(x.key.clone());
        }
        if x.exhaustive || x.evidence == Evidence::Exhaustive {
            let bits: u32 = catalog::signature(by_id[x.id.as_str()])
                .map(|(a, _, _)| a.iter().map(|t| super::proof::bits(*t)).sum())
                .unwrap_or(64);
            if !x.exhaustive || bits > 32 || x.vectors != 1u64 << bits {
                bad_exhaustive.push(x.key.clone());
            }
        }
    }
    rep.push(
        Check::new("files", missing.is_empty())
            .count("missing", missing.len() as u64)
            .note(sample(&missing, 8)),
    );
    rep.push(
        Check::new("builds", bad_build.is_empty())
            .count("failed", bad_build.len() as u64)
            .note("result.json: no reference mismatch, XLS IR equal to the llvm simulation on every vector")
            .note(sample(&bad_build, 8)),
    );
    rep.push(
        Check::new("integer_only", floats.is_empty())
            .count("with_float_types", floats.len() as u64)
            .note("kernel.mlir has no f16/f32/f64/bf16 type")
            .note(sample(&floats, 8)),
    );
    rep.push(
        Check::new("rtl", bad_rtl.is_empty())
            .count("bad", bad_rtl.len() as u64)
            .note("rtl.v defines the primitive's module")
            .note(sample(&bad_rtl, 8)),
    );
    rep.push(
        Check::new("exhaustive", bad_exhaustive.is_empty())
            .count("bad", bad_exhaustive.len() as u64)
            .note("an exhaustive entry has one vector per input combination")
            .note(sample(&bad_exhaustive, 8)),
    );

    // The reference, recomputed on a sample of every primitive's recorded vectors.
    let runner = python_runner(ctx);
    let threads = ctx.resources().threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| PassError::failed(e.to_string()))?;
    let batches: Vec<&[super::json::Entry]> = cov.entries.chunks(cfg.batch.max(1) * 4).collect();
    let lines: Vec<String> = pool
        .install(|| {
            batches
                .par_iter()
                .with_max_len(1)
                .map(|b| -> Result<Vec<String>, PassError> {
                    let mut args = vec!["-m".to_string(), "cc_softfloat".into(), "recheck".into()];
                    for x in b.iter() {
                        args.extend(["--key".into(), x.key.clone(), "--id".into(), x.id.clone()]);
                    }
                    args.extend([
                        "--out".into(),
                        format!("{OUT_IN_CONTAINER}/ops"),
                        "--sample".into(),
                        cfg.recheck_sample.to_string(),
                    ]);
                    let o = runner.run(&python(ctx, &cfg, dir, &args))?;
                    let text = o.stdout_string().unwrap_or_default();
                    Ok(text
                        .lines()
                        .filter_map(|l| l.strip_prefix("@@"))
                        .map(String::from)
                        .collect())
                })
                .collect::<Result<Vec<_>, _>>()
        })?
        .into_iter()
        .flatten()
        .collect();
    let mut checked = 0u64;
    let mut differ = Vec::new();
    let mut seen = 0usize;
    for l in &lines {
        let v: serde_json::Value = serde_json::from_str(l).unwrap_or_default();
        seen += 1;
        let n = |k: &str| v.get(k).and_then(serde_json::Value::as_u64);
        checked += n("checked").unwrap_or(0);
        let ok = n("checked").unwrap_or(0) > 0
            && n("want_differs") == Some(0)
            && n("llvm_differs") == Some(0)
            && n("ir_differs") == Some(0);
        if !ok {
            differ.push(v.get("key").and_then(|k| k.as_str()).unwrap_or("?").to_string());
        }
    }
    let ok = differ.is_empty() && seen == cov.entries.len();
    rep.push(
        Check::new("reference", ok)
            .count("vectors_checked", checked)
            .count("primitives_differing", differ.len() as u64)
            .note("the reference recomputed on a sample of the recorded vectors equals the recorded reference, llvm and XLS IR results")
            .note(sample(&differ, 8)),
    );

    // Approximate ops: within their accuracy limit, and the design's bound within it too.
    let mut bad_acc = Vec::new();
    let mut approx_n = 0u64;
    let mut worst = 0f64;
    for x in &cov.entries {
        let r: Option<BuildResult> = out.read_json(&format!("ops/{}/result.json", x.id)).ok();
        let recorded = r.as_ref().and_then(|r| r.accuracy).unwrap_or(Accuracy::Exact);
        let ok = if is_approximate(&x.op) {
            approx_n += 1;
            let limit = x.accuracy.limit();
            let within = |v: Option<f64>| match (v, limit) {
                (Some(v), Some(l)) => v.is_finite() && v <= l + ULP_SLACK,
                _ => false,
            };
            worst = worst.max(x.max_ulp.unwrap_or(f64::INFINITY));
            x.proof.is_none()
                && limit.is_some()
                && within(x.max_ulp)
                && within(x.bound_ulp)
                && r.as_ref()
                    .is_some_and(|r| r.max_ulp == x.max_ulp && r.bound_ulp == x.bound_ulp)
        } else {
            x.accuracy == Accuracy::Exact && x.max_ulp.is_none() && x.bound_ulp.is_none()
        };
        if !ok || recorded != x.accuracy {
            bad_acc.push(x.key.clone());
        }
    }
    rep.push(
        Check::new("accuracy", bad_acc.is_empty())
            .count("approximate", approx_n)
            .count("bad", bad_acc.len() as u64)
            .note(format!(
                "every approximate op's largest error and its design bound are within its limit (correctly rounded: 1/2 ulp, faithful: 1 ulp); worst measured {worst:.4} ulp; exact ops are exact"
            ))
            .note(sample(&bad_acc, 8)),
    );

    // Queries regenerate byte for byte; the recorded verdicts match the evidence.
    let mut bad_query = Vec::new();
    let mut bad_result = Vec::new();
    let mut proven: Vec<(String, String)> = Vec::new();
    for x in cov.entries.iter().filter(|x| !is_approximate(&x.op)) {
        let p = by_id[x.id.as_str()];
        let f = format!("ops/{}/proof.smt2", x.id);
        let stored = std::fs::read_to_string(dir.join(&f)).unwrap_or_default();
        match query_of(dir, p) {
            Ok(q)
                if q.text == stored
                    && Some(q.product_abstracted) == x.proof.as_ref().map(|r| r.product_abstracted) => {}
            _ => bad_query.push(x.key.clone()),
        }
        let res: Option<ProofResult> = out.read_json(&f.replace(".smt2", ".result")).ok();
        let consistent = res.as_ref().is_some_and(|r| {
            r.query_hash == cc_smt::query_hash(&stored)
                && Some(&r.verdict) == x.proof.as_ref().map(|p| &p.verdict)
                && match x.evidence {
                    Evidence::Proven => r.verdict == Verdict::Unsat,
                    Evidence::Exhaustive | Evidence::Tested => {
                        matches!(r.verdict, Verdict::Timeout | Verdict::Unknown)
                    }
                }
        });
        if !consistent {
            bad_result.push(x.key.clone());
        }
        if x.evidence == Evidence::Proven {
            proven.push((f, stored));
        }
    }
    rep.push(
        Check::new("queries", bad_query.is_empty())
            .count(
                "regenerated",
                cov.entries.iter().filter(|x| !is_approximate(&x.op)).count() as u64,
            )
            .count("differ", bad_query.len() as u64)
            .note("each proof.smt2 equals the query regenerated from fu.opt.ir")
            .note(sample(&bad_query, 8)),
    );
    rep.push(
        Check::new("verdicts", bad_result.is_empty())
            .count("bad", bad_result.len() as u64)
            .note("proof.result matches its query and the evidence (proven: unsat; otherwise a timeout)")
            .note(sample(&bad_result, 8)),
    );

    // The proven ones, re-checked by the other solver.
    let (solver, srunner, location) = crate::normalize::smt(ctx)?;
    let other = solver.other();
    let cache = cc_smt::Cache::new(ctx.dir().path().join("cache"));
    let answers = crate::datapaths::solve(
        &srunner,
        &location,
        other,
        dir,
        &proven,
        cfg.recheck_timeout_s,
        threads,
        &cache,
    )?;
    let not_unsat: Vec<String> = proven
        .iter()
        .filter(|(f, _)| !answers.get(f).is_some_and(|(_, v)| v.is_unsat()))
        .map(|(f, _)| format!("{f}: {:?}", answers.get(f).map(|(_, v)| v)))
        .collect();
    rep.push(
        Check::new("recheck", not_unsat.is_empty())
            .count("proven", proven.len() as u64)
            .count("not_unsat", not_unsat.len() as u64)
            .note(format!(
                "every proven primitive's query is unsat in {}",
                other.name()
            ))
            .note(sample(&not_unsat, 4)),
    );

    let count = |ev: Evidence| cov.entries.iter().filter(|x| x.evidence == ev).count() as u64;
    let by_acc = |a: Accuracy| cov.entries.iter().filter(|x| x.accuracy == a).count() as u64;
    let ok = summary.primitives == cov.entries.len() as u64
        && summary.proven == count(Evidence::Proven)
        && summary.exhaustive == count(Evidence::Exhaustive)
        && summary.tested == count(Evidence::Tested)
        && summary.exact == by_acc(Accuracy::Exact)
        && summary.correctly_rounded == by_acc(Accuracy::CorrectlyRounded)
        && summary.faithful == by_acc(Accuracy::Faithful)
        && summary.approximate == cov.approximate.len() as u64
        && summary.library_hash == super::library_hash(ctx.repo_root())?;
    rep.push(
        Check::new("summary", ok)
            .path("out/softfloat/summary.json")
            .note("counts and the library hash agree with coverage.json and python/cc_softfloat"),
    );
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::has_float_type;

    #[test]
    fn finds_float_types_as_whole_words() {
        assert!(has_float_type("%0 = arith.sitofp %a : i32 to f32"));
        assert!(has_float_type("memref<4xbf16>"));
        assert!(has_float_type("tensor<?x8xf16>"));
        assert!(!has_float_type("%max = arith.maxui %a, %b : i32"));
        assert!(!has_float_type("%f32x = arith.addi %f16_a, %b : i32"));
    }
}
