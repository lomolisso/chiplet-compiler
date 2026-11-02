//! `iteration_space` (design §4.7, §7): per kernel class its iteration domain, affine access functions
//! proven against the StableHLO spec, dependences under ω, the map / reduce / contraction class, the
//! embedding relation ⊑ between skeletons, and the routing of every class to a generated array template
//! (by a proven space-time mapping) or to datapath mining (docs/passes/iteration_space.md).

pub mod algo;
pub mod config;
pub mod json;
pub mod proofs;
pub mod spec;
mod verify;
mod view;

use std::collections::{BTreeMap, BTreeSet};

use cc_core::{ArtifactKind, CoreError, LoadedConfig, PassSection, VerifyReport};
use cc_ir::absgraph::Graph;
use cc_pipeline::{Pass, PassContext, PassError};
use rayon::prelude::*;

pub use config::{IterationSpaceConfig, TemplateConfig};

use crate::kernel_classes::KERNEL_CLASSES;
use crate::kernel_classes::json::{ClassFile, InstancesTable};
use crate::kernels::KERNELS;
use crate::kernels::json::{KNode, KernelGraphs};
use crate::normalize::{query_hash, smt};
use algo::eval::Table;
use json::{
    Dependences, EMBEDDING_SCHEMA, Embedding, Pair, ProofResult, RESULT_SCHEMA, ROUTING_SCHEMA, Route,
    Routed, Routing, SUMMARY_SCHEMA, Skeleton, Summary, TEMPLATE_SCHEMA, TemplateFile,
};

/// `out/iteration_space/{skeletons,dependences,templates,smt}/…`, `embedding.json`, `routing.json`,
/// `summary.json`.
pub const ITERATION_SPACE: ArtifactKind = ArtifactKind("iteration_space");
pub const ROUTING_VIEW: &str = "routing.md";
pub const SUMMARY_VIEW: &str = "iteration_space.md";
/// Steps one embedding search may take (a search over it counts as no embedding, and is reported).
pub const EMBED_BUDGET: u64 = 200_000;

#[derive(Debug)]
pub struct IterationSpace;

impl Pass for IterationSpace {
    fn name(&self) -> &'static str {
        "iteration_space"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[KERNEL_CLASSES, KERNELS]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[ITERATION_SPACE]
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
        let _: IterationSpaceConfig = section.parse()?;
        Ok(())
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: IterationSpaceConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let fail = |e: String| PassError::failed(e);
        let loaded = load(ctx)?;
        let n = loaded.classes.len();

        let analysed: Vec<Analysed> = (0..n)
            .into_par_iter()
            .map(|i| {
                ctx.check_cancelled()?;
                analyse(&loaded, i, &cfg.template).map_err(|e| fail(format!("{}: {e}", loaded.classes[i].id)))
            })
            .collect::<Result<_, _>>()?;

        // ⊑ over the skeletons of one class.
        let skels: Vec<&Skeleton> = analysed.iter().map(|a| &a.skeleton).collect();
        let found: Vec<(Vec<Pair>, u64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut pairs = Vec::new();
                let mut over = 0;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let (p, exhausted) = algo::embed::embed(skels[i], skels[j], EMBED_BUDGET);
                    over += u64::from(exhausted);
                    pairs.extend(p);
                }
                (pairs, over)
            })
            .collect();
        let over_budget: u64 = found.iter().map(|x| x.1).sum();
        let pairs: Vec<Pair> = found.into_iter().flat_map(|x| x.0).collect();

        // Proofs.
        let out = ctx.output(ITERATION_SPACE)?;
        let mut files: Vec<(String, String)> = Vec::new();
        for a in &analysed {
            files.push((format!("access.{}.smt2", a.skeleton.id), a.access.clone()));
            if let Some(m) = &a.mapping {
                files.push((format!("mapping.{}.smt2", a.skeleton.id), m.clone()));
            }
        }
        for (f, text) in &files {
            write_file(&out.path(&format!("smt/{f}"))?, text)?;
        }
        let (solver, runner, location) = smt(ctx)?;
        let names: Vec<String> = files.iter().map(|(f, _)| f.clone()).collect();
        let batch = cc_smt::run_batch(
            &runner,
            &location,
            solver,
            &out.path("smt")?,
            &names,
            ctx.smt().timeout_s,
        )?;
        let mut bad = Vec::new();
        for (f, text) in &files {
            let verdict = batch.verdicts.get(f).cloned().unwrap_or(cc_smt::Verdict::Error {
                message: "no answer".into(),
            });
            if !verdict.is_unsat() {
                bad.push(format!("{f}: {verdict:?}"));
            }
            out.write_json(
                &format!("smt/{}", f.replace(".smt2", ".result")),
                &ProofResult {
                    schema: RESULT_SCHEMA.into(),
                    file: f.clone(),
                    query_hash: query_hash(text),
                    solver,
                    version: batch.version.clone(),
                    verdict,
                },
            )?;
        }
        if !bad.is_empty() {
            return Err(fail(format!("proofs failed:\n{}", bad.join("\n"))));
        }

        analysed.par_iter().try_for_each(|a| -> Result<(), PassError> {
            let id = &a.skeleton.id;
            out.write_json(&format!("skeletons/{id}.json"), &a.skeleton)?;
            out.write_json(&format!("dependences/{id}.json"), &a.deps)?;
            if let Some(t) = &a.template {
                out.write_json(&format!("templates/{id}.json"), t)?;
            }
            Ok(())
        })?;
        let routing = Routing {
            schema: ROUTING_SCHEMA.into(),
            classes: analysed.iter().map(|a| a.routed.clone()).collect(),
        };
        out.write_json("routing.json", &routing)?;
        crate::kernel_classes::write_compact(
            out,
            "embedding.json",
            &Embedding {
                schema: EMBEDDING_SCHEMA.into(),
                pairs: pairs.clone(),
            },
        )?;
        let summary = summarize(&analysed, pairs.len() as u64, over_budget, files.len() as u64);
        out.write_json("summary.json", &summary)?;
        if cfg.views {
            ctx.dir().write_view(ROUTING_VIEW, &view::routing(&routing))?;
            ctx.dir().write_view(SUMMARY_VIEW, &view::summary(&summary))?;
        }
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

/// The inputs: every class file, its instance table and its representative's kernel graph.
#[derive(Debug)]
pub struct Loaded {
    pub classes: Vec<ClassFile>,
    pub tables: Vec<InstancesTable>,
    pub graphs: Vec<Graph<KNode>>,
}

pub fn load(ctx: &PassContext) -> Result<Loaded, PassError> {
    let kc = ctx.input(KERNEL_CLASSES)?;
    let kin = ctx.input(KERNELS)?;
    let mut ids = Vec::new();
    if let Ok(rd) = std::fs::read_dir(kc.dir().join("classes")) {
        for e in rd.flatten() {
            if let Some(id) = e.file_name().to_str().and_then(|n| n.strip_suffix(".json")) {
                ids.push(id.to_string());
            }
        }
    }
    ids.sort();
    let pairs: Vec<(ClassFile, InstancesTable)> = ids
        .par_iter()
        .map(|id| -> Result<_, PassError> {
            Ok((
                kc.read_json(&format!("classes/{id}.json"))?,
                kc.read_json(&format!("instances/{id}.json"))?,
            ))
        })
        .collect::<Result<_, _>>()?;
    let comps: BTreeSet<String> = pairs
        .iter()
        .map(|(c, _)| c.representative.component.clone())
        .collect();
    let graphs: BTreeMap<String, KernelGraphs> = comps
        .par_iter()
        .map(|c| -> Result<_, PassError> {
            Ok((c.clone(), kin.read_postcard(&format!("kernels/{c}.postcard"))?))
        })
        .collect::<Result<_, _>>()?;
    let mut classes = Vec::new();
    let mut tables = Vec::new();
    let mut gs = Vec::new();
    for (c, t) in pairs {
        let kg = graphs[&c.representative.component]
            .kernels
            .iter()
            .find(|k| k.id == c.representative.kernel)
            .ok_or_else(|| {
                PassError::failed(format!(
                    "{}: kernel {}/{} not found",
                    c.id, c.representative.component, c.representative.kernel
                ))
            })?;
        gs.push(kg.graph.clone());
        classes.push(c);
        tables.push(t);
    }
    Ok(Loaded {
        classes,
        tables,
        graphs: gs,
    })
}

/// Everything the pass derives for one class.
#[derive(Debug)]
pub struct Analysed {
    pub skeleton: Skeleton,
    pub deps: Dependences,
    pub template: Option<TemplateFile>,
    pub routed: Routed,
    pub access: String,
    pub mapping: Option<String>,
}

pub fn analyse(l: &Loaded, i: usize, cfg: &TemplateConfig) -> Result<Analysed, String> {
    let (class, table, g) = (&l.classes[i], &l.tables[i], &l.graphs[i]);
    let mut skeleton = algo::skeleton::build(class, g)?;
    skeleton.proof = format!("smt/access.{}.smt2", class.id);
    let deps = algo::deps::dependences(&skeleton);
    let t = Table::new(&table.rows, &class.params.values);
    let cands = if skeleton.class == json::SkelClass::Contraction && deps.uniform {
        algo::mapping::candidates(&skeleton, &deps, &t)
    } else {
        Vec::new()
    };
    let routed = algo::route::route(&skeleton, &deps, &cands, &t, cfg, class.workload);
    let claims = spec::claims(class, &skeleton, g)?;
    let access = proofs::access(&class.id, &claims)?;
    let (template, mapping) = if routed.route == Route::Template {
        let fam = |name: &str| {
            algo::deps::operand(&skeleton, name)
                .and_then(|o| class.params.elems.get(o.elem as usize))
                .map(|f| crate::normalize::algo::omega::family_key(f).to_string())
                .unwrap_or_default()
        };
        let tf = TemplateFile {
            schema: TEMPLATE_SCHEMA.into(),
            id: class.id.clone(),
            space: deps.space.clone(),
            groups: deps
                .space
                .iter()
                .map(|g| (*g, deps.groups.get(g).cloned().unwrap_or_default()))
                .collect(),
            candidates: cands,
            selected: 0,
            families: vec![fam("lhs"), fam("rhs"), fam("result")],
            proof: format!("smt/mapping.{}.smt2", class.id),
        };
        let vs: Vec<(String, Vec<i64>)> = deps
            .vectors
            .iter()
            .map(|v| (v.operand.clone(), v.v.clone()))
            .collect();
        let m = proofs::mapping(&tf, &vs);
        (Some(tf), Some(m))
    } else {
        (None, None)
    };
    Ok(Analysed {
        skeleton,
        deps,
        template,
        routed,
        access,
        mapping,
    })
}

pub(crate) fn write_file(p: &std::path::Path, text: &str) -> Result<(), PassError> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| PassError::failed(format!("{}: {e}", d.display())))?;
    }
    std::fs::write(p, text).map_err(|e| PassError::failed(format!("{}: {e}", p.display())))
}

/// The first words of a datapath reason, as a kind.
pub fn reason_kind(r: &str) -> String {
    let head = r.split([':', ';']).next().unwrap_or(r);
    if let Some(rest) = r.strip_prefix("contraction: ") {
        let w: Vec<&str> = rest.split_whitespace().take(3).collect();
        return format!("contraction: {}", w.join(" "));
    }
    if r.starts_with("contraction, not a uniform recurrence") {
        return "contraction, not uniform".into();
    }
    head.to_string()
}

fn summarize(a: &[Analysed], pairs: u64, over_budget: u64, proofs: u64) -> Summary {
    let mut s = Summary {
        schema: SUMMARY_SCHEMA.into(),
        classes: a.len() as u64,
        classes_by_kind: BTreeMap::new(),
        uniform: 0,
        templates: 0,
        datapaths: 0,
        template_cores: BTreeMap::new(),
        dataflows: BTreeMap::new(),
        datapath_reasons: BTreeMap::new(),
        template_workload: 0,
        datapath_workload: 0,
        embedding_pairs: pairs,
        embedding_over_budget: over_budget,
        proofs,
    };
    for x in a {
        *s.classes_by_kind
            .entry(format!("{} -> {}", x.skeleton.kind, x.skeleton.class.name()))
            .or_default() += 1;
        s.uniform += u64::from(x.deps.uniform);
        match x.routed.route {
            Route::Template => {
                s.templates += 1;
                s.template_workload = s.template_workload.saturating_add(x.routed.workload);
                *s.template_cores.entry(x.routed.core.clone()).or_default() += 1;
                *s.dataflows
                    .entry(x.routed.dataflow.clone().unwrap_or_default())
                    .or_default() += 1;
            }
            Route::Datapath => {
                s.datapaths += 1;
                s.datapath_workload = s.datapath_workload.saturating_add(x.routed.workload);
                *s.datapath_reasons
                    .entry(reason_kind(&x.routed.reason))
                    .or_default() += 1;
            }
        }
    }
    s
}
