//! `softfloat` (design §4.12): the float primitives as integer-only Allo functions (`python/cc_softfloat`),
//! each simulated on Allo's `llvm` target against a NumPy/ml_dtypes reference, taken through Allo's XLS
//! backend, xlscc, `opt_main` and `codegen_main` to Verilog, evaluated on its optimized XLS IR, and either
//! proven equal to the op's semantics (`cc_smt::ops`) where the solver finishes, or, for an approximate op,
//! checked to be within its accuracy limit in ulps (docs/passes/softfloat.md).

pub mod config;
pub mod json;
pub mod proof;
mod verify;
mod view;

use std::collections::BTreeMap;
use std::path::Path;

use cc_core::runner::{BoundRunner, Caps, Invocation, Mount};
use cc_core::{ArtifactKind, CoreError, LoadedConfig, PassSection, VerifyReport, canon};
use cc_pipeline::{Pass, PassContext, PassError, RunnerKind, ToolSpec};
use cc_smt::Verdict;
use rayon::prelude::*;

use crate::characterize::catalog::{self, Kind, Primitive};
use crate::datapaths::json::{ProofResult, RESULT_SCHEMA};
pub use config::SoftfloatConfig;
use json::{
    Accuracy, BuildResult, COVERAGE_SCHEMA, CoverageFile, Entry, Evidence, ProofRef, SUMMARY_SCHEMA, Summary,
};

/// `out/softfloat/{coverage,summary}.json`, `ops/<id>/`.
pub const SOFTFLOAT: ArtifactKind = ArtifactKind("softfloat");
pub const VIEW: &str = "softfloat.md";

/// Ops whose float result the spec leaves to an approximate function (uninterpreted in `cc_smt::ops`):
/// their accuracy is a choice of this library (docs/passes/softfloat.md §4).
pub const APPROXIMATE: &[&str] = &[
    "exponential",
    "exponential_minus_one",
    "log",
    "log_plus_one",
    "logistic",
    "tanh",
    "rsqrt",
    "cbrt",
    "sine",
    "cosine",
    "tan",
    "atan2",
    "power",
];

/// The approximate ops the library builds (`python/cc_softfloat/approx.py`); the others are listed apart.
pub const APPROXIMATE_BUILT: &[&str] = &["exponential", "log", "log_plus_one", "tanh", "rsqrt"];

pub fn is_approximate(op: &str) -> bool {
    APPROXIMATE.contains(&op)
}

const REPO_IN_CONTAINER: &str = "/work/repo";
const OUT_IN_CONTAINER: &str = "/work/out";

const TOOLS: &[ToolSpec] = &[ToolSpec {
    name: "opt_main",
    runner: RunnerKind::Docker,
    program: "opt_main",
    version_args: &["--version"],
}];

#[derive(Debug)]
pub struct Softfloat;

/// The float op primitives of the catalog: those built here, and the keys of the approximate ones not
/// built yet.
pub fn primitives(cfg: &SoftfloatConfig) -> Result<(Vec<Primitive>, Vec<String>), String> {
    let cat = catalog::catalog(&cfg.catalog_config())?;
    let mut built = Vec::new();
    let mut approx = Vec::new();
    for p in cat {
        if p.kind != Kind::Op || !p.types.iter().any(|t| catalog::format(t).is_some()) {
            continue;
        }
        if is_approximate(&p.op) && !APPROXIMATE_BUILT.contains(&p.op.as_str()) {
            approx.push(p.key);
        } else {
            built.push(p);
        }
    }
    Ok((built, approx))
}

pub fn library_hash(root: &Path) -> Result<String, CoreError> {
    canon::hash_tree(&root.join("python/cc_softfloat"), &[".py", ".h"])
}

impl Pass for Softfloat {
    fn name(&self) -> &'static str {
        "softfloat"
    }

    fn version(&self) -> u32 {
        2
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[SOFTFLOAT]
    }

    fn tools(&self) -> &[ToolSpec] {
        TOOLS
    }

    fn model_dependent(&self) -> bool {
        false
    }

    /// The library sources and the image are inputs (they are not artifacts).
    fn fingerprints(
        &self,
        config: &LoadedConfig,
        _section: &PassSection,
    ) -> Result<BTreeMap<String, String>, CoreError> {
        let mut m = BTreeMap::new();
        m.insert("cc_softfloat".into(), library_hash(&config.root)?);
        m.insert("catalog".into(), catalog::CATALOG_VERSION.to_string());
        let d = &config.config.tools.docker;
        m.insert("image".into(), format!("{}:{}", d.image, d.tag));
        Ok(m)
    }

    fn check_config(&self, section: &PassSection) -> Result<(), CoreError> {
        let c: SoftfloatConfig = section.parse()?;
        c.check().map_err(|message| CoreError::PassConfig {
            pass: "softfloat".into(),
            message,
        })
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: SoftfloatConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let fail = |e: String| PassError::failed(e);
        cfg.check().map_err(fail)?;
        let (prims, approximate) = primitives(&cfg).map_err(fail)?;
        let out = ctx.output(SOFTFLOAT)?;
        let ops_dir = out.dir().join("ops");
        std::fs::create_dir_all(&ops_dir).map_err(|e| fail(format!("{}: {e}", ops_dir.display())))?;

        // Build and simulate, `batch` primitives per container, `threads` containers at a time.
        let runner = python_runner(ctx);
        let threads = ctx.resources().threads.max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| fail(e.to_string()))?;
        let batches: Vec<&[Primitive]> = prims.chunks(cfg.batch).collect();
        pool.install(|| {
            batches
                .par_iter()
                .with_max_len(1)
                .try_for_each(|b| -> Result<(), PassError> {
                    ctx.check_cancelled()?;
                    let mut args = vec!["-m".to_string(), "cc_softfloat".into(), "build".into()];
                    for p in b.iter() {
                        args.extend(["--key".into(), p.key.clone(), "--id".into(), p.id.clone()]);
                    }
                    args.extend([
                        "--out".into(),
                        format!("{OUT_IN_CONTAINER}/ops"),
                        "--exhaustive-bits".into(),
                        cfg.exhaustive_bits.to_string(),
                        "--random".into(),
                        cfg.random_vectors.to_string(),
                        "--precision".into(),
                        format!("{},{}", catalog::REDUCE_PRECISION.0, catalog::REDUCE_PRECISION.1),
                    ]);
                    let o = runner.run(&python(ctx, &cfg, out.dir(), &args))?;
                    // Exit 1 is a primitive that failed (reported in its result.json); anything else is fatal.
                    if o.killed.is_some() || !matches!(o.exit_code, Some(0 | 1)) {
                        o.check()?;
                    }
                    Ok(())
                })
        })?;

        // Results.
        let mut failures = Vec::new();
        let mut results: BTreeMap<String, BuildResult> = BTreeMap::new();
        for p in &prims {
            let f = format!("ops/{}/result.json", p.id);
            match out.read_json::<BuildResult>(&f) {
                Ok(r) => {
                    if !r.ok {
                        failures.push(describe(&r));
                    }
                    results.insert(p.id.clone(), r);
                }
                Err(_) => failures.push(format!("{}: no result.json", p.key)),
            }
        }
        if !failures.is_empty() {
            return Err(fail(format!(
                "{} of {} primitives failed: {}",
                failures.len(),
                prims.len(),
                failures.iter().take(8).cloned().collect::<Vec<_>>().join("; ")
            )));
        }

        // Proofs of the exactly specified ops.
        let mut files: Vec<(String, String)> = Vec::new();
        let mut abstracted: BTreeMap<String, bool> = BTreeMap::new();
        let exact: Vec<&Primitive> = prims.iter().filter(|p| !is_approximate(&p.op)).collect();
        for p in &exact {
            let q = query_of(out.dir(), p).map_err(fail)?;
            let f = format!("ops/{}/proof.smt2", p.id);
            write_file(&out.path(&f)?, &q.text)?;
            abstracted.insert(p.id.clone(), q.product_abstracted);
            files.push((f, q.text));
        }
        let (solver, srunner, location) = crate::normalize::smt(ctx)?;
        let cache = cc_smt::Cache::new(ctx.dir().path().join("cache"));
        let answers = crate::datapaths::solve(
            &srunner,
            &location,
            solver,
            out.dir(),
            &files,
            cfg.proof_timeout_s,
            threads,
            &cache,
        )?;
        let mut entries = Vec::new();
        let mut refuted = Vec::new();
        let mut version = String::new();
        let mut proofs: BTreeMap<String, ProofRef> = BTreeMap::new();
        let mut proven: BTreeMap<String, Evidence> = BTreeMap::new();
        for (p, (f, text)) in exact.iter().zip(&files) {
            let (v, verdict) = answers.get(f).cloned().unwrap_or((
                String::new(),
                Verdict::Error {
                    message: "no answer".into(),
                },
            ));
            version.clone_from(&v);
            out.write_json(
                &f.replace(".smt2", ".result"),
                &ProofResult {
                    schema: RESULT_SCHEMA.into(),
                    file: f.clone(),
                    query_hash: cc_smt::query_hash(text),
                    solver,
                    version: v,
                    verdict: verdict.clone(),
                    expect: "unsat".into(),
                    allowed: None,
                },
            )?;
            let r = &results[&p.id];
            let evidence = match &verdict {
                Verdict::Unsat => Evidence::Proven,
                Verdict::Sat { .. } | Verdict::Error { .. } => {
                    refuted.push(format!("{}: {verdict:?}", p.key));
                    Evidence::Tested
                }
                _ if r.exhaustive => Evidence::Exhaustive,
                _ => Evidence::Tested,
            };
            proven.insert(p.id.clone(), evidence);
            proofs.insert(
                p.id.clone(),
                ProofRef {
                    file: f.clone(),
                    verdict,
                    product_abstracted: abstracted[&p.id],
                },
            );
        }
        for p in &prims {
            let r = &results[&p.id];
            let accuracy = r.accuracy.unwrap_or(Accuracy::Exact);
            let evidence = match proven.get(&p.id) {
                Some(ev) => *ev,
                None if r.exhaustive => Evidence::Exhaustive,
                None => Evidence::Tested,
            };
            entries.push(Entry {
                key: p.key.clone(),
                id: p.id.clone(),
                op: p.op.clone(),
                types: p.types.clone(),
                evidence,
                vectors: r.vectors,
                exhaustive: r.exhaustive,
                ir_nodes: r.ir_nodes,
                accuracy,
                bound_ulp: r.bound_ulp,
                max_ulp: r.max_ulp,
                proof: proofs.remove(&p.id),
            });
        }
        let count = |ev: Evidence| entries.iter().filter(|x| x.evidence == ev).count() as u64;
        let by_acc = |a: Accuracy| entries.iter().filter(|x| x.accuracy == a).count() as u64;
        let d = &ctx.tools().docker;
        let summary = Summary {
            schema: SUMMARY_SCHEMA.into(),
            primitives: entries.len() as u64,
            proven: count(Evidence::Proven),
            exhaustive: count(Evidence::Exhaustive),
            tested: count(Evidence::Tested),
            exact: by_acc(Accuracy::Exact),
            correctly_rounded: by_acc(Accuracy::CorrectlyRounded),
            faithful: by_acc(Accuracy::Faithful),
            approximate: approximate.len() as u64,
            vectors: entries.iter().map(|x| x.vectors).sum(),
            library_hash: library_hash(ctx.repo_root())?,
            image: format!("{}:{}", d.image, d.tag),
            solver: format!("{} {version}", solver.name()),
        };
        let cov = CoverageFile {
            schema: COVERAGE_SCHEMA.into(),
            entries,
            approximate,
        };
        out.write_json("coverage.json", &cov)?;
        out.write_json("summary.json", &summary)?;
        if cfg.views {
            view::write(ctx, &cov, &summary)?;
        }
        if !refuted.is_empty() {
            return Err(fail(format!(
                "{} proofs failed (a counterexample or a solver error): {}",
                refuted.len(),
                refuted.iter().take(8).cloned().collect::<Vec<_>>().join("; ")
            )));
        }
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

fn describe(r: &BuildResult) -> String {
    if let Some(e) = &r.error {
        return format!("{}: {e}", r.key);
    }
    let m = r.reference.as_ref().map_or(0, |x| x.mismatches);
    format!(
        "{}: {m} reference mismatches, {} IR results differ from llvm",
        r.key, r.ir_vs_llvm
    )
}

/// The proof query of `p` from its `ops/<id>/fu.opt.ir`.
pub fn query_of(out: &Path, p: &Primitive) -> Result<proof::Query, String> {
    let (args, ret, attrs) = catalog::signature(p)?;
    let f = out.join(format!("ops/{}/fu.opt.ir", p.id));
    let ir = std::fs::read_to_string(&f).map_err(|e| format!("{}: {e}", f.display()))?;
    proof::query(&p.key, &p.op, &args, ret, &attrs, &ir)
}

/// The image's Python, one CPU and `memory_gb / threads` per container.
fn python_runner(ctx: &PassContext) -> BoundRunner {
    let r = ctx.docker();
    let res = ctx.resources();
    let per = Caps {
        memory_gb: res.memory_gb / res.threads.max(1) as f64,
        cpus: 1.0,
        ..*r.caps()
    };
    let caps = r.caps().min(&per);
    r.with_caps(caps)
}

/// `python <args>` in the image, the repository read-only and the output directory writable.
fn python(ctx: &PassContext, cfg: &SoftfloatConfig, out: &Path, args: &[String]) -> Invocation {
    Invocation::new("cc_softfloat", cfg.docker_python.clone())
        .env("PYTHONPATH", format!("{REPO_IN_CONTAINER}/python"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .args(args.iter().cloned())
        .cwd(OUT_IN_CONTAINER)
        .mount(Mount {
            src: ctx.repo_root().to_path_buf(),
            dst: REPO_IN_CONTAINER.into(),
            readonly: true,
        })
        .mount(Mount {
            src: out.to_path_buf(),
            dst: OUT_IN_CONTAINER.into(),
            readonly: false,
        })
}

fn write_file(path: &Path, text: &str) -> Result<(), PassError> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| PassError::failed(format!("{}: {e}", d.display())))?;
    }
    std::fs::write(path, text).map_err(|e| PassError::failed(format!("{}: {e}", path.display())))
}
