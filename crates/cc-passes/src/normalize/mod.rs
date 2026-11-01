//! `normalize` (design §4.4): exactness-preserving rewrites of every component (DCE, CSE, layout
//! collapses and identities, splat-constant folding), each rule proven once with SMT, and a reduction
//! order ω on every reduction (docs/passes/normalize.md).

pub mod algo;
pub mod config;
pub mod json;
pub mod proofs;
mod verify;
mod view;

use std::collections::BTreeMap;
use std::path::Path;

use cc_core::runner::BoundRunner;
use cc_core::{ArtifactKind, CoreError, LoadedConfig, PassSection, VerifyReport};
use cc_pipeline::{Pass, PassContext, PassError};
use cc_smt::{Location, Solver};
use rayon::prelude::*;

pub use config::NormalizeConfig;

use crate::components::COMPONENTS;
use crate::components::json::{ComponentSummary, StoredComponent};
use json::{
    COMPONENT_SCHEMA, Expect, LEDGER_SCHEMA, LedgerFile, NormalizedComponent, NormalizedSummary,
    RESULT_SCHEMA, RuleResult, SUMMARY_SCHEMA,
};

/// `out/normalize/{components,ledger,rules}/…`.
pub const NORMALIZE: ArtifactKind = ArtifactKind("normalize");
pub const VIEW_FILE: &str = "normalize.md";

#[derive(Debug)]
pub struct Normalize;

impl Pass for Normalize {
    fn name(&self) -> &'static str {
        "normalize"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[COMPONENTS]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[NORMALIZE]
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
        let c: NormalizeConfig = section.parse()?;
        c.order.check().map_err(|e| CoreError::PassConfig {
            pass: "normalize".into(),
            message: format!("order: {e}"),
        })?;
        Ok(())
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: NormalizeConfig = ctx.options()?;
        let orders = cfg.order.parse().map_err(PassError::failed)?;
        let ctx: &PassContext = ctx;
        let (solver, runner, location) = smt(ctx)?;
        // Proofs first: no rule is applied before it is proven.
        prove(ctx, solver, &runner, &location)?;
        let ids = component_ids(ctx)?;
        let out = ctx.output(NORMALIZE)?;
        let input = ctx.input(COMPONENTS)?;
        let summaries: Vec<NormalizedSummary> = ids
            .par_iter()
            .map(|id| -> Result<NormalizedSummary, PassError> {
                ctx.check_cancelled()?;
                let c: StoredComponent = input.read_postcard(&format!("components/{id}.postcard"))?;
                let cs: ComponentSummary =
                    input.read_json(&format!("components/{id}.postcard.summary.json"))?;
                let n = algo::normalize(&c.graph, &orders)
                    .map_err(|e| PassError::failed(format!("{id}: {e}")))?;
                let mut rules: BTreeMap<String, u64> = BTreeMap::new();
                for s in &n.steps {
                    *rules.entry(s.rule.clone()).or_default() += 1;
                }
                let mut omega: BTreeMap<String, u64> = BTreeMap::new();
                for x in &n.graph.nodes {
                    if let Some(o) = &x.omega {
                        *omega.entry(algo::omega::order_name(&o.order)).or_default() += 1;
                    }
                }
                let s = NormalizedSummary {
                    schema: SUMMARY_SCHEMA.into(),
                    id: id.clone(),
                    height: c.height,
                    nodes_in: c.graph.node_count() as u64,
                    nodes: n.graph.node_count() as u64,
                    edges: n.graph.end_count() as u64,
                    rules,
                    omega,
                    instances: cs.instances.len() as u64,
                    models: cs.models.clone(),
                };
                out.write_postcard(
                    &format!("components/{id}.postcard"),
                    &NormalizedComponent {
                        schema: COMPONENT_SCHEMA.into(),
                        id: id.clone(),
                        height: c.height,
                        graph: n.graph,
                        iface: c.iface.clone(),
                        dims: c.dims,
                        elems: c.elems,
                    },
                    &s,
                )?;
                out.write_json(
                    &format!("ledger/{id}.json"),
                    &LedgerFile {
                        schema: LEDGER_SCHEMA.into(),
                        id: id.clone(),
                        steps: n.steps,
                    },
                )?;
                Ok(s)
            })
            .collect::<Result<_, _>>()?;
        if cfg.views {
            ctx.dir().write_view(VIEW_FILE, &view::table(&summaries))?;
        }
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

/// The component ids of the input, sorted.
pub(crate) fn component_ids(ctx: &PassContext) -> Result<Vec<String>, PassError> {
    let dir = ctx.input(COMPONENTS)?.dir().join("components");
    let mut ids = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            if let Some(id) = e.file_name().to_str().and_then(|n| n.strip_suffix(".postcard")) {
                ids.push(id.to_string());
            }
        }
    }
    ids.sort();
    Ok(ids)
}

/// The configured solver, its runner and where its binary is.
pub(crate) fn smt(ctx: &PassContext) -> Result<(Solver, BoundRunner, Location), PassError> {
    let c = ctx.smt();
    let solver = Solver::parse(&c.solver)
        .ok_or_else(|| PassError::failed(format!("tools.smt.solver: unknown solver `{}`", c.solver)))?;
    match c.runner.as_str() {
        "docker" => Ok((solver, ctx.docker(), Location::Image)),
        "host" => Ok((solver, ctx.host(), cc_smt::host_paths(&c.z3, &c.bitwuzla))),
        r => Err(PassError::failed(format!(
            "tools.smt.runner: unknown runner `{r}`"
        ))),
    }
}

pub(crate) fn query_hash(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

/// Write every obligation, run the solver on them, write the results; fail unless each answer is the
/// expected one (`unsat` for rules and the positive control, `sat` for the negative control).
fn prove(
    ctx: &PassContext,
    solver: Solver,
    runner: &BoundRunner,
    location: &Location,
) -> Result<(), PassError> {
    let out = ctx.output(NORMALIZE)?;
    let obligations = proofs::obligations();
    let mut files = Vec::new();
    for (rule, _, text) in &obligations {
        let p = out.path(&format!("rules/{rule}.smt2"))?;
        write_file(&p, text)?;
        files.push(format!("{rule}.smt2"));
    }
    let dir = out.path("rules")?;
    let batch = cc_smt::run_batch(runner, location, solver, &dir, &files, ctx.smt().timeout_s)?;
    let mut bad = Vec::new();
    for (rule, expect, text) in &obligations {
        let verdict =
            batch
                .verdicts
                .get(&format!("{rule}.smt2"))
                .cloned()
                .unwrap_or(cc_smt::Verdict::Error {
                    message: "no answer".into(),
                });
        let ok = match expect {
            Expect::Unsat => verdict.is_unsat(),
            Expect::Sat => matches!(verdict, cc_smt::Verdict::Sat { .. }),
        };
        if !ok {
            bad.push(format!("{rule}: {verdict:?}"));
        }
        out.write_json(
            &format!("rules/{rule}.result"),
            &RuleResult {
                schema: RESULT_SCHEMA.into(),
                rule: (*rule).to_string(),
                expect: *expect,
                query_hash: query_hash(text),
                solver,
                version: batch.version.clone(),
                verdict,
            },
        )?;
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(PassError::failed(format!(
            "rule proofs failed:\n{}",
            bad.join("\n")
        )))
    }
}

fn write_file(p: &Path, text: &str) -> Result<(), PassError> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| PassError::failed(format!("{}: {e}", d.display())))?;
    }
    std::fs::write(p, text).map_err(|e| PassError::failed(format!("{}: {e}", p.display())))
}
