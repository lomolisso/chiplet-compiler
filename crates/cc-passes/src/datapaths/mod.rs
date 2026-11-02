//! `datapaths` (design §4.8): per kernel class the scalar datapath of one iteration (the stages of a
//! template class), proven equal to the StableHLO semantics of the kernel's ops for every width tuple of
//! its instance table, with the order freedom of its combiner (docs/passes/datapaths.md).

pub mod algo;
pub mod config;
pub mod eval;
pub mod json;
pub mod proofs;
pub mod spec;
mod verify;
mod view;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use cc_core::{ArtifactKind, CoreError, LoadedConfig, PassSection, VerifyReport};
use cc_pipeline::{Pass, PassContext, PassError};
use cc_smt::{Solver, Verdict};
use rayon::prelude::*;

pub use config::{Allowed, DatapathsConfig};

use crate::iteration_space::json::{Route, Routed, Routing, Skeleton};
use crate::iteration_space::{ITERATION_SPACE, Loaded, write_file};
use crate::kernel_classes::KERNEL_CLASSES;
use crate::kernels::KERNELS;
use crate::normalize::smt;
use json::{Datapath, OrderFree, ProofResult, RESULT_SCHEMA, SUMMARY_SCHEMA, Summary};
use proofs::Obligations;

/// `out/datapaths/{datapaths,proofs,lemmas}/…`, `summary.json`.
pub const DATAPATHS: ArtifactKind = ArtifactKind("datapaths");
pub const VIEW: &str = "datapaths.md";

#[derive(Debug)]
pub struct Datapaths;

impl Pass for Datapaths {
    fn name(&self) -> &'static str {
        "datapaths"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[ITERATION_SPACE, KERNEL_CLASSES, KERNELS]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[DATAPATHS]
    }

    fn fingerprints(
        &self,
        config: &LoadedConfig,
        _section: &PassSection,
    ) -> Result<BTreeMap<String, String>, CoreError> {
        let t = &config.config.tools;
        let mut m = BTreeMap::new();
        m.insert("solver".into(), t.smt.solver.clone());
        if t.smt.runner == "docker" {
            m.insert("image".into(), format!("{}:{}", t.docker.image, t.docker.tag));
        }
        Ok(m)
    }

    fn check_config(&self, section: &PassSection) -> Result<(), CoreError> {
        let _: DatapathsConfig = section.parse()?;
        Ok(())
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: DatapathsConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let fail = |e: String| PassError::failed(e);
        let inputs = load(ctx)?;
        let n = inputs.loaded.classes.len();

        // Extraction, tuples and obligations, per class.
        let results: Vec<Result<(Datapath, Obligations), String>> = (0..n)
            .into_par_iter()
            .map(|i| {
                ctx.check_cancelled()?;
                Ok(build(&inputs, i, cfg.max_tuples)
                    .map_err(|e| format!("{}: {e}", inputs.loaded.classes[i].id)))
            })
            .collect::<Result<_, PassError>>()?;
        let errors: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
        if !errors.is_empty() {
            let shown: Vec<&str> = errors.iter().take(40).map(|e| e.as_str()).collect();
            return Err(fail(format!(
                "{} classes failed:\n{}",
                errors.len(),
                shown.join("\n")
            )));
        }
        let mut built: Vec<(Datapath, Obligations)> = results.into_iter().filter_map(Result::ok).collect();

        let out = ctx.output(DATAPATHS)?;
        let mut used = BTreeSet::new();
        let mut functions = BTreeSet::new();
        let mut files: Vec<(String, String)> = Vec::new();
        for (_, ob) in &built {
            used.extend(ob.used.iter().cloned());
            functions.extend(ob.functions.iter().cloned());
            for q in &ob.queries {
                files.push((q.file.clone(), q.text.clone()));
            }
        }
        for (f, text) in &files {
            write_file(&out.path(&format!("proofs/{f}"))?, text)?;
        }
        let lemmas = cc_smt::float::lemmas(&used).map_err(|e| fail(e.0))?;
        for (f, text) in &lemmas {
            write_file(&out.path(&format!("lemmas/{f}"))?, text)?;
        }

        // Solvers: the configured one proves the obligations, Z3 the encoding lemmas.
        let (solver, runner, location) = smt(ctx)?;
        let threads = ctx.resources().threads.max(1);
        let cache = cc_smt::Cache::new(ctx.dir().path().join("cache"));
        let answers = solve(
            &runner,
            &location,
            solver,
            &out.path("proofs")?,
            &files,
            ctx.smt().timeout_s,
            threads,
            &cache,
        )?;
        let lemma_answers = solve(
            &runner,
            &location,
            Solver::Z3,
            &out.path("lemmas")?,
            &lemmas,
            cfg.lemma_timeout_s,
            threads,
            &cache,
        )?;

        let allowed: HashMap<&str, &str> = cfg
            .allow_unknown
            .iter()
            .map(|a| (a.class.as_str(), a.reason.as_str()))
            .collect();
        let mut bad = Vec::new();
        let mut allowed_unknown = 0u64;
        for (dp, ob) in &mut built {
            let mut order = Vec::new();
            for q in &ob.queries {
                let (version, verdict) = answers.get(&q.file).cloned().unwrap_or_else(none);
                let mut allow = None;
                if q.order {
                    order.push(OrderFree {
                        tuple: q.tuple.clone(),
                        free: verdict.is_unsat(),
                    });
                } else if !verdict.is_unsat() {
                    match (&verdict, allowed.get(dp.id.as_str())) {
                        (Verdict::Unknown | Verdict::Timeout, Some(r)) => {
                            allow = Some(r.to_string());
                            allowed_unknown += 1;
                        }
                        _ => bad.push(format!("{}: {}", q.file, brief(&verdict))),
                    }
                }
                out.write_json(
                    &format!("proofs/{}", q.file.replace(".smt2", ".result")),
                    &ProofResult {
                        schema: RESULT_SCHEMA.into(),
                        file: q.file.clone(),
                        query_hash: cc_smt::query_hash(&q.text),
                        solver,
                        version,
                        verdict,
                        expect: if q.order { "either".into() } else { "unsat".into() },
                        allowed: allow,
                    },
                )?;
            }
            if let Some(a) = &mut dp.accumulator {
                a.order = order;
            }
        }
        for (f, text) in &lemmas {
            let (version, verdict) = lemma_answers.get(f).cloned().unwrap_or_else(none);
            if !verdict.is_unsat() {
                bad.push(format!("lemma {f}: {}", brief(&verdict)));
            }
            out.write_json(
                &format!("lemmas/{}", f.replace(".smt2", ".result")),
                &ProofResult {
                    schema: RESULT_SCHEMA.into(),
                    file: f.clone(),
                    query_hash: cc_smt::query_hash(text),
                    solver: Solver::Z3,
                    version,
                    verdict,
                    expect: "unsat".into(),
                    allowed: None,
                },
            )?;
        }

        built
            .par_iter()
            .try_for_each(|(dp, _)| -> Result<(), PassError> {
                out.write_json(&format!("datapaths/{}.json", dp.id), dp)?;
                Ok(())
            })?;
        let summary = summarize(&built, lemmas.len() as u64, allowed_unknown, functions);
        out.write_json("summary.json", &summary)?;
        if cfg.views {
            ctx.dir().write_view(VIEW, &view::view(&built, &summary))?;
        }
        if !bad.is_empty() {
            bad.sort();
            return Err(fail(format!("{} proofs failed:\n{}", bad.len(), bad.join("\n"))));
        }
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

fn none() -> (String, Verdict) {
    (
        String::new(),
        Verdict::Error {
            message: "no answer".into(),
        },
    )
}

fn brief(v: &Verdict) -> String {
    match v {
        Verdict::Sat { model } => format!("sat ({})", model.chars().take(300).collect::<String>()),
        v => format!("{v:?}"),
    }
}

/// Everything the pass reads: the classes, instance tables and representative graphs, the routing and
/// the skeletons.
#[derive(Debug)]
pub struct Inputs {
    pub loaded: Loaded,
    pub routed: Vec<Routed>,
    pub skeletons: Vec<Skeleton>,
}

pub fn load(ctx: &PassContext) -> Result<Inputs, PassError> {
    let loaded = crate::iteration_space::load(ctx)?;
    let isp = ctx.input(ITERATION_SPACE)?;
    let routing: Routing = isp.read_json("routing.json")?;
    let by_id: HashMap<&str, &Routed> = routing.classes.iter().map(|r| (r.id.as_str(), r)).collect();
    let mut routed = Vec::new();
    for c in &loaded.classes {
        routed.push(
            (*by_id
                .get(c.id.as_str())
                .ok_or_else(|| PassError::failed(format!("{}: not in routing.json", c.id)))?)
            .clone(),
        );
    }
    let skeletons = loaded
        .classes
        .par_iter()
        .map(|c| isp.read_json::<Skeleton>(&format!("skeletons/{}.json", c.id)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Inputs {
        loaded,
        routed,
        skeletons,
    })
}

/// The datapath of class `i` with its tuples, and its obligations.
pub fn build(inputs: &Inputs, i: usize, max_tuples: usize) -> Result<(Datapath, Obligations), String> {
    let l = &inputs.loaded;
    let (class, table, g) = (&l.classes[i], &l.tables[i], &l.graphs[i]);
    let (skel, routed) = (&inputs.skeletons[i], &inputs.routed[i]);
    let mut dp = algo::extract::extract(class, skel, routed, g)?;
    dp.tuples = algo::tuples::tuples(class, table)?;
    if dp.tuples.len() > max_tuples {
        return Err(format!(
            "{} width tuples, over max_tuples = {max_tuples}",
            dp.tuples.len()
        ));
    }
    let ob = proofs::obligations(class, skel, routed, g, &dp)?;
    Ok((dp, ob))
}

/// Answers for `files` (relative to `dir`): cached ones, then one sharded run for the rest.
#[allow(clippy::too_many_arguments)]
fn solve(
    runner: &cc_core::runner::BoundRunner,
    location: &cc_smt::Location,
    solver: Solver,
    dir: &std::path::Path,
    files: &[(String, String)],
    timeout_s: u64,
    threads: usize,
    cache: &cc_smt::Cache,
) -> Result<HashMap<String, (String, Verdict)>, PassError> {
    let mut out = HashMap::new();
    if files.is_empty() {
        return Ok(out);
    }
    let version = cc_smt::solver_version(runner, location, solver)?;
    let mut todo = Vec::new();
    for (f, text) in files {
        match cache.get(solver, &version, &cc_smt::query_hash(text)) {
            Some(v) => {
                out.insert(f.clone(), (version.clone(), v));
            }
            None => todo.push(f.clone()),
        }
    }
    if !todo.is_empty() {
        let batch = cc_smt::run_sharded(runner, location, solver, dir, &todo, timeout_s, threads)?;
        if batch.version != version {
            return Err(PassError::failed(format!(
                "{} reported {} and then {}",
                solver.name(),
                version,
                batch.version
            )));
        }
        let text: HashMap<&str, &str> = files.iter().map(|(f, t)| (f.as_str(), t.as_str())).collect();
        for f in todo {
            let v = batch.verdicts.get(&f).cloned().unwrap_or(Verdict::Error {
                message: "no answer".into(),
            });
            if let Some(t) = text.get(f.as_str()) {
                let _ = cache.put(solver, &version, &cc_smt::query_hash(t), &v);
            }
            out.insert(f, (version.clone(), v));
        }
    }
    Ok(out)
}

fn summarize(
    built: &[(Datapath, Obligations)],
    lemmas: u64,
    allowed: u64,
    functions: BTreeSet<String>,
) -> Summary {
    let mut s = Summary {
        schema: SUMMARY_SCHEMA.into(),
        classes: built.len() as u64,
        by_route: BTreeMap::new(),
        vertices: 0,
        labels: BTreeMap::new(),
        tuples: 0,
        proofs: 0,
        order_queries: 0,
        order_free: 0,
        order_bound: 0,
        lemmas,
        allowed_unknown: allowed,
        uninterpreted: functions.into_iter().collect(),
    };
    for (dp, ob) in built {
        let r = match dp.route {
            Route::Template => "template",
            Route::Datapath => "datapath",
        };
        *s.by_route.entry(r.into()).or_default() += 1;
        s.vertices += dp.vertices.len() as u64;
        for v in &dp.vertices {
            *s.labels.entry(v.label.clone()).or_default() += 1;
        }
        s.tuples += dp.tuples.len() as u64;
        for q in &ob.queries {
            if q.order {
                s.order_queries += 1;
            } else {
                s.proofs += 1;
            }
        }
        if let Some(a) = &dp.accumulator {
            for o in &a.order {
                if o.free {
                    s.order_free += 1;
                } else {
                    s.order_bound += 1;
                }
            }
        }
    }
    s
}
