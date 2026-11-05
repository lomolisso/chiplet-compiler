//! `merge` (design §4.11): functional units covering every class. Datapath classes and template stages
//! are merged into configurable units per family (Moreano's compatibility graph and a maximum-weight
//! clique, agglomerated under a depth bound); template classes share parameterized arrays. Every
//! configuration is read off the clique solution and proven equal to its datapath (docs/passes/merge.md).

pub mod algo;
pub mod config;
pub mod json;
pub mod proofs;
pub mod requant;
mod verify;
mod view;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use cc_core::{ArtifactKind, CoreError, LoadedConfig, PassSection, VerifyReport};
use cc_pipeline::{Pass, PassContext, PassError};
use cc_smt::Verdict;
use rayon::prelude::*;

pub use config::{CliqueAlgorithm, MergeConfig};

use crate::characterize::LIBRARY;
use crate::characterize::json::Library;
use crate::datapaths::DATAPATHS;
use crate::datapaths::json::{Datapath, OutputKind, ProofResult};
use crate::iteration_space::json::{Embedding, Route, Routing, TemplateFile};
use crate::iteration_space::{ITERATION_SPACE, write_file};
use crate::kernel_classes::KERNEL_CLASSES;
use crate::mine::MINE;
use crate::mine::json::{Family, PatternFile, Selection};
use algo::agglomerate::{Agglomerated, Ctx as AggCtx, agglomerate};
use algo::arrays::{TemplateIn, arrays};
use algo::fu::{Costed, Fu};
use algo::library::Lib;
use algo::unit::{Unit, contract, empty_stages, units_of};
use json::*;

/// `out/merge/{fus, configs, proofs, arrays, trace, cliques}/…`, `coverage.json`, `summary.json`.
pub const MERGE: ArtifactKind = ArtifactKind("merge");
pub const VIEW: &str = "merge.md";
pub const RESULT_SCHEMA: &str = "cc.merge.proof_result/1";

#[derive(Debug)]
pub struct Merge;

impl Pass for Merge {
    fn name(&self) -> &'static str {
        "merge"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[DATAPATHS, MINE, ITERATION_SPACE, KERNEL_CLASSES, LIBRARY]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[MERGE]
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
        let c: MergeConfig = section.parse()?;
        c.check().map_err(|message| CoreError::PassConfig {
            pass: "merge".into(),
            message,
        })
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: MergeConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let fail = PassError::failed;
        let mut requant_file: Option<RequantFile> = None;
        let inputs = load(ctx, &cfg, &mut |dps| {
            let (set, file) = decide_requant(ctx, &cfg, dps)?;
            requant_file = file;
            Ok(set)
        })?;
        let lib = Lib::new(&inputs.library, &cfg.kit).map_err(fail)?;
        let cx = AggCtx {
            units: &inputs.units,
            lib: &lib,
            cfg: &cfg,
            embeds: &inputs.embeds,
        };
        let fams: Vec<(Family, Vec<u32>)> = Family::ALL
            .iter()
            .map(|&f| {
                let idx: Vec<u32> = inputs
                    .units
                    .iter()
                    .enumerate()
                    .filter(|(_, u)| u.family == f)
                    .map(|(i, _)| i as u32)
                    .collect();
                (f, idx)
            })
            .collect();
        let results: Vec<Result<(Family, Agglomerated), String>> = fams
            .par_iter()
            .map(|(f, idx)| {
                Ok((
                    *f,
                    agglomerate(&cx, idx).map_err(|e| format!("{}: {e}", f.name()))?,
                ))
            })
            .collect();
        let mut fams_out = Vec::new();
        for r in results {
            fams_out.push(r.map_err(fail)?);
        }
        ctx.check_cancelled()?;

        let out = ctx.output(MERGE)?;
        let dps: HashMap<&str, &Datapath> = inputs.datapaths.iter().map(|d| (d.id.as_str(), d)).collect();
        let mut fu_files: Vec<FuFile> = Vec::new();
        let mut configs: Vec<ConfigFile> = Vec::new();
        let mut unit_fu: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut summary = Summary {
            schema: SUMMARY_SCHEMA.into(),
            kit: lib.kit.clone(),
            requant_variant: if requant_file.is_some() { "fixed" } else { "fp32" }.into(),
            algorithm: cfg.clique.algorithm.name().into(),
            families: BTreeMap::new(),
            fus: 0,
            arrays: 0,
            configs: 0,
            proofs: 0,
            array_proofs: 0,
            widenings: 0,
            cliques: BTreeMap::new(),
            not_optimal: 0,
            estimated_blocks: 0,
            classes: inputs.datapaths.len() as u64,
            covered: 0,
            requant_fixed: 0,
            requant_kept: 0,
        };
        if let Some(r) = &requant_file {
            for c in &r.classes {
                match c.status {
                    RequantStatus::Fixed => summary.requant_fixed += 1,
                    RequantStatus::Kept => summary.requant_kept += 1,
                }
            }
            ctx.output(MERGE)?.write_json("requant.json", r)?;
        }
        for (f, ag) in &fams_out {
            let mut order: Vec<usize> = (0..ag.fus.len()).collect();
            let first = |i: usize| {
                ag.fus[i]
                    .cfgs
                    .iter()
                    .map(|c| inputs.units[c.unit as usize].id.clone())
                    .min()
                    .unwrap_or_default()
            };
            order.sort_by_key(|&i| first(i));
            for (k, &i) in order.iter().enumerate() {
                let id = format!("{}{k:03}", f.name());
                let (file, cfgs) = files(
                    &id,
                    *f,
                    &ag.fus[i],
                    &ag.costs[i],
                    ag.bound_ps,
                    &inputs.units,
                    &dps,
                )
                .map_err(fail)?;
                for c in &file.covers {
                    unit_fu.entry(c.unit.clone()).or_default().insert(id.clone());
                }
                summary.estimated_blocks += file.blocks.iter().filter(|b| b.price.estimated).count() as u64;
                fu_files.push(file);
                configs.extend(cfgs);
            }
            let trace = Trace {
                schema: TRACE_SCHEMA.into(),
                family: *f,
                period_ps: lib.period_ps,
                bound_ps: ag.bound_ps,
                initial_units: ag.fus.iter().map(|x| x.cfgs.len() as u64).sum(),
                initial_area_um2: ag.initial_area,
                steps: ag.steps.clone(),
                stopped_by: ag.stopped_by.clone(),
                evaluated: ag.evaluated,
            };
            for s in &ag.steps {
                *summary.cliques.entry(s.clique.algorithm.clone()).or_default() += 1;
                if !s.clique.optimal {
                    summary.not_optimal += 1;
                }
            }
            out.write_json(&format!("trace/{}.json", f.name()), &trace)?;
            for (step, cert) in &ag.certs {
                out.write_json(&format!("cliques/{}/{step}.json", f.name()), cert)?;
            }
            summary.families.insert(
                f.name().into(),
                FamilySummary {
                    units: trace.initial_units,
                    fus: ag.fus.len() as u64,
                    initial_area_um2: ag.initial_area,
                    area_um2: ag.costs.iter().map(|c| c.area).sum(),
                    steps: ag.steps.len() as u64,
                    evaluated: ag.evaluated,
                    bound_ps: ag.bound_ps,
                    stopped_by: ag.stopped_by.clone(),
                    macro_groups: inputs
                        .units
                        .iter()
                        .filter(|u| u.family == *f)
                        .map(|u| u.groups.len() as u64)
                        .sum(),
                },
            );
        }

        // Arrays for the template classes.
        let arr = arrays(&inputs.templates, &inputs.embeds, &lib);

        // Proof obligations.
        let mut queries: Vec<proofs::Query> = Vec::new();
        let fu_by: HashMap<&str, &FuFile> = fu_files.iter().map(|f| (f.id.as_str(), f)).collect();
        let built: Vec<Result<proofs::Query, String>> = configs
            .par_iter()
            .map(|c| {
                let fu = fu_by[c.fu.as_str()];
                let dp = dps[c.class.as_str()];
                let t = dp
                    .tuples
                    .iter()
                    .find(|t| t.name == c.tuple)
                    .ok_or_else(|| format!("{}: no tuple {}", c.class, c.tuple))?;
                proofs::config_query(fu, c, dp, t)
            })
            .collect();
        for q in built {
            queries.push(q.map_err(fail)?);
        }
        let fu_map: BTreeMap<String, &FuFile> = fu_files.iter().map(|f| (f.id.clone(), f)).collect();
        let widen = proofs::widenings(&fu_map, &configs).map_err(fail)?;
        for w in &widen {
            queries.push(proofs::widen_query(w).map_err(fail)?);
        }
        summary.widenings = widen.len() as u64;
        let nfu = queries.len() - widen.len();
        // Composed array steps go to Z3 with a longer limit (`compose_timeout_s`).
        let mut composed: Vec<proofs::Query> = Vec::new();
        for a in &arr {
            for cv in &a.covers {
                for t in &cv.tuples {
                    let q = proofs::array_query(a, &cv.class, t).map_err(fail)?;
                    if t.mode == ArrayMode::Compose {
                        composed.push(q);
                    } else {
                        queries.push(q);
                    }
                }
            }
        }
        summary.proofs = nfu as u64;
        summary.array_proofs = (queries.len() - nfu - widen.len() + composed.len()) as u64;
        let files: Vec<(String, String)> = queries.iter().map(|q| (q.file.clone(), q.text.clone())).collect();
        let zfiles: Vec<(String, String)> = composed
            .iter()
            .map(|q| (q.file.clone(), q.text.clone()))
            .collect();
        for (f, text) in files.iter().chain(&zfiles) {
            write_file(&out.path(&format!("proofs/{f}"))?, text)?;
        }
        let (solver, runner, location) = crate::normalize::smt(ctx)?;
        let cache = cc_smt::Cache::new(ctx.smt_cache_dir());
        let threads = ctx.resources().threads.max(1);
        let answers = cc_smt::solve_cached(
            &runner,
            &location,
            solver,
            &out.path("proofs")?,
            &files,
            ctx.smt().timeout_s,
            threads,
            &cache,
        )?;
        let zanswers = cc_smt::solve_cached(
            &runner,
            &location,
            cc_smt::lemma_solver(),
            &out.path("proofs")?,
            &zfiles,
            cfg.compose_timeout_s,
            threads,
            &cache,
        )?;
        let mut bad = Vec::new();
        for (set, by, s) in [(&files, &answers, solver), (&zfiles, &zanswers, cc_smt::lemma_solver())] {
            for (f, text) in set {
                let (version, verdict) = by.get(f).cloned().unwrap_or((
                    String::new(),
                    Verdict::Error {
                        message: "no answer".into(),
                    },
                ));
                if !verdict.is_unsat() {
                    bad.push(format!("{f}: {verdict:?}"));
                }
                out.write_json(
                    &format!("proofs/{}", f.replace(".smt2", ".result")),
                    &ProofResult {
                        schema: RESULT_SCHEMA.into(),
                        file: f.clone(),
                        query_hash: cc_smt::query_hash(text),
                        solver: s,
                        version,
                        verdict,
                        expect: "unsat".into(),
                        allowed: None,
                    },
                )?;
            }
        }

        for f in &fu_files {
            out.write_json(&format!("fus/{}.json", f.id), f)?;
        }
        configs.par_iter().try_for_each(|c| -> Result<(), PassError> {
            out.write_json(&format!("configs/{}/{}/{}.json", c.fu, c.unit, c.tuple), c)?;
            Ok(())
        })?;
        for a in &arr {
            out.write_json(&format!("arrays/{}.json", a.id), a)?;
        }
        let coverage = coverage(&inputs.datapaths, &inputs.units, &unit_fu, &configs, &arr);
        summary.fus = fu_files.len() as u64;
        summary.arrays = arr.len() as u64;
        summary.configs = configs.len() as u64;
        summary.covered = coverage.covered;
        out.write_json("coverage.json", &coverage)?;
        out.write_json("summary.json", &summary)?;
        if cfg.views {
            let traces: Vec<Trace> = Family::ALL
                .iter()
                .filter_map(|f| out.read_json(&format!("trace/{}.json", f.name())).ok())
                .collect();
            ctx.dir()
                .write_view(VIEW, &view::view(&summary, &fu_files, &arr, &traces, &coverage))?;
        }
        if !bad.is_empty() {
            bad.sort();
            return Err(fail(format!(
                "{} proofs failed:\n{}",
                bad.len(),
                bad.iter().take(40).cloned().collect::<Vec<_>>().join("\n")
            )));
        }
        if coverage.covered != coverage.total {
            return Err(fail(format!(
                "{} of {} classes are not covered",
                coverage.total - coverage.covered,
                coverage.total
            )));
        }
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

/// The `fixed` decision (docs/passes/merge.md §4.11): the rescaling chains of every class, one tolerance
/// lemma per integer type they rescale (solved here, `requant_timeout_s`), and the classes whose every
/// chain has a proven lemma. `None` for the other variants.
fn decide_requant(
    ctx: &PassContext,
    cfg: &MergeConfig,
    dps: &[Datapath],
) -> Result<(BTreeSet<String>, Option<RequantFile>), PassError> {
    let fail = PassError::failed;
    if cfg.requant_variant != config::RequantVariant::Fixed {
        return Ok((BTreeSet::new(), None));
    }
    let found: Vec<(String, Vec<RequantSite>)> = dps
        .par_iter()
        .map(|dp| {
            let sites: Vec<RequantSite> = requant::sites(dp)
                .into_iter()
                .map(|x| RequantSite {
                    types: requant::site_types(dp, &x),
                    convert: x.convert,
                    multiply: x.multiply,
                    round: x.round,
                    x: x.x,
                    scale: x.scale,
                })
                .collect();
            (dp.id.clone(), sites)
        })
        .filter(|(_, s)| !s.is_empty())
        .collect();
    let types: BTreeSet<String> = found
        .iter()
        .flat_map(|(_, s)| s.iter().flat_map(|x| x.types.iter().cloned()))
        .collect();
    let out = ctx.output(MERGE)?;
    let mut files = Vec::new();
    let mut lemma_of = BTreeMap::new();
    for t in &types {
        let ty = proofs::ty(t).map_err(fail)?;
        let (f, text) = requant::lemma(ty).map_err(fail)?;
        write_file(&out.path(&format!("proofs/{f}"))?, &text)?;
        lemma_of.insert(t.clone(), f.clone());
        files.push((f, text));
    }
    let (solver, runner, location) = crate::normalize::smt(ctx)?;
    let cache = cc_smt::Cache::new(ctx.smt_cache_dir());
    let answers = cc_smt::solve_cached(
        &runner,
        &location,
        solver,
        &out.path("proofs")?,
        &files,
        cfg.requant_timeout_s,
        ctx.resources().threads.max(1),
        &cache,
    )?;
    let mut lemmas = Vec::new();
    let mut verdict_of: BTreeMap<String, Verdict> = BTreeMap::new();
    for (t, f) in &lemma_of {
        let text = &files
            .iter()
            .find(|(g, _)| g == f)
            .map(|x| x.1.clone())
            .unwrap_or_default();
        let (version, verdict) = answers.get(f).cloned().unwrap_or((
            String::new(),
            Verdict::Error {
                message: "no answer".into(),
            },
        ));
        out.write_json(
            &format!("proofs/{}", f.replace(".smt2", ".result")),
            &ProofResult {
                schema: RESULT_SCHEMA.into(),
                file: f.clone(),
                query_hash: cc_smt::query_hash(text),
                solver,
                version,
                verdict: verdict.clone(),
                expect: "unsat".into(),
                allowed: None,
            },
        )?;
        lemmas.push(RequantLemma {
            ty: t.clone(),
            file: f.clone(),
            proven: verdict.is_unsat(),
        });
        verdict_of.insert(t.clone(), verdict);
    }
    let mut set = BTreeSet::new();
    let mut classes = Vec::new();
    for (class, sites) in found {
        let open: Vec<String> = sites
            .iter()
            .flat_map(|x| x.types.iter())
            .filter(|t| !verdict_of.get(*t).is_some_and(Verdict::is_unsat))
            .map(|t| {
                let v = match verdict_of.get(t) {
                    Some(Verdict::Sat { .. }) => "sat".to_string(),
                    Some(v) => format!("{v:?}").to_lowercase(),
                    None => "no answer".into(),
                };
                format!("{}: {v}", lemma_of.get(t).cloned().unwrap_or_default())
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let status = if open.is_empty() {
            set.insert(class.clone());
            RequantStatus::Fixed
        } else {
            RequantStatus::Kept
        };
        classes.push(RequantClass {
            class,
            sites,
            status,
            reason: (!open.is_empty()).then(|| open.join("; ")),
        });
    }
    Ok((
        set,
        Some(RequantFile {
            schema: REQUANT_SCHEMA.into(),
            lemmas,
            classes,
        }),
    ))
}

/// Per pattern (in selection order), the datapath vertices of each chosen occurrence in one unit.
type Occurrences = Vec<(String, Vec<Vec<u32>>)>;

/// Everything the pass reads.
#[derive(Debug)]
pub struct Inputs {
    pub datapaths: Vec<Datapath>,
    pub units: Vec<Unit>,
    pub templates: Vec<TemplateIn>,
    pub embeds: HashSet<(String, String)>,
    pub library: Library,
}

/// Decides which classes take the `fixed` requantization (given the datapaths as `datapaths` wrote them).
pub type Decide<'a> = dyn FnMut(&[Datapath]) -> Result<BTreeSet<String>, PassError> + 'a;

pub fn load(ctx: &PassContext, cfg: &MergeConfig, decide: &mut Decide) -> Result<Inputs, PassError> {
    let fail = PassError::failed;
    let mi = crate::mine::load(ctx)?;
    let fixed = decide(&mi.datapaths)?;
    // Per rewritten class: the old -> new vertex ids and the rounds that became `rescale`.
    let mut moved: HashMap<String, (Vec<Option<u32>>, BTreeSet<u32>)> = HashMap::new();
    let mut datapaths: Vec<Datapath> = Vec::with_capacity(mi.datapaths.len());
    for dp in mi.datapaths {
        if !fixed.contains(&dp.id) {
            datapaths.push(dp);
            continue;
        }
        let sites = requant::sites(&dp);
        let (new, map) = requant::rewrite(&dp, &sites).map_err(fail)?;
        moved.insert(dp.id.clone(), (map, sites.iter().map(|x| x.round).collect()));
        datapaths.push(new);
    }
    let isp = ctx.input(ITERATION_SPACE)?;
    let routing: Routing = isp.read_json("routing.json")?;
    let embedding: Embedding = isp.read_json("embedding.json")?;
    let embeds: HashSet<(String, String)> = embedding
        .pairs
        .iter()
        .map(|p| (p.guest.clone(), p.host.clone()))
        .collect();
    let library: Library = ctx.input(LIBRARY)?.read_json("library.json")?;

    let mut units = Vec::new();
    for dp in &datapaths {
        units.extend(units_of(dp).map_err(|e| fail(format!("{}: {e}", dp.id)))?);
    }
    if cfg.contract {
        let mine = ctx.input(MINE)?;
        let mut occ: HashMap<String, Occurrences> = HashMap::new();
        for f in Family::ALL {
            let Ok(sel) = mine.read_json::<Selection>(&format!("selection/{}.json", f.name())) else {
                continue;
            };
            for s in &sel.patterns {
                let p: PatternFile = mine.read_json(&format!("patterns/{}/{}.json", f.name(), s.id))?;
                for um in &p.mis {
                    let vs: Vec<Vec<u32>> = um
                        .chosen
                        .iter()
                        .filter_map(|&k| p.occurrences.get(k as usize))
                        .map(|o| o.map.clone())
                        .collect();
                    occ.entry(um.unit.clone()).or_default().push((p.id.clone(), vs));
                }
            }
        }
        for u in &mut units {
            let Some(o) = occ.get(&u.id) else { continue };
            match moved.get(&u.class) {
                // Occurrences through the renumbering; one that holds a rewritten round or a removed
                // vertex no longer matches its pattern.
                Some((map, rounds)) => {
                    let o: Occurrences = o
                        .iter()
                        .map(|(pid, occs)| {
                            let kept = occs
                                .iter()
                                .filter_map(|vs| {
                                    vs.iter()
                                        .map(|v| {
                                            if rounds.contains(v) {
                                                None
                                            } else {
                                                map.get(*v as usize).copied().flatten()
                                            }
                                        })
                                        .collect::<Option<Vec<u32>>>()
                                })
                                .collect();
                            (pid.clone(), kept)
                        })
                        .collect();
                    contract(u, &o);
                }
                None => contract(u, o),
            }
        }
    }

    let mut templates = Vec::new();
    let by_id: HashMap<&str, &Datapath> = datapaths.iter().map(|d| (d.id.as_str(), d)).collect();
    for r in routing.classes.iter().filter(|r| r.route == Route::Template) {
        let dp = by_id
            .get(r.id.as_str())
            .ok_or_else(|| fail(format!("{}: a template class without a datapath", r.id)))?;
        let tf: TemplateFile = isp.read_json(&format!("templates/{}.json", r.id))?;
        let cand = tf
            .candidates
            .get(tf.selected as usize)
            .ok_or_else(|| fail(format!("{}: no selected mapping", r.id)))?;
        let ops: Vec<&crate::datapaths::json::Output> = dp
            .outputs
            .iter()
            .filter(|o| o.kind == OutputKind::CoreOperand)
            .collect();
        let res = dp
            .inputs
            .iter()
            .find(|i| i.name == "core_result0")
            .ok_or_else(|| fail(format!("{}: no core result", r.id)))?;
        if ops.len() != 2 {
            return Err(fail(format!(
                "{}: {} core operands (two expected)",
                r.id,
                ops.len()
            )));
        }
        let mut tuples = Vec::new();
        for t in &dp.tuples {
            let ty = |w: &str| crate::datapaths::spec::ty_of_width(t, w).map(|x| x.name());
            let types = vec![
                ty(&ops[0].width).map_err(fail)?,
                ty(&ops[1].width).map_err(fail)?,
                ty(&res.width).map_err(fail)?,
            ];
            tuples.push((t.name.clone(), types, t.rows));
        }
        templates.push(TemplateIn {
            class: r.id.clone(),
            dataflow: r.dataflow.clone().unwrap_or_else(|| cand.dataflow.clone()),
            workload: r.workload,
            candidate: tf.selected,
            dims: cand.array.clone(),
            tuples,
        });
    }
    Ok(Inputs {
        datapaths,
        units,
        templates,
        embeds,
        library,
    })
}

/// The JSON of a unit and of its configurations.
fn files(
    id: &str,
    family: Family,
    fu: &Fu,
    c: &Costed,
    bound_ps: f64,
    units: &[Unit],
    dps: &HashMap<&str, &Datapath>,
) -> Result<(FuFile, Vec<ConfigFile>), String> {
    let blocks: Vec<Block> = fu
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| Block {
            id: i as u32,
            kind: b.kind,
            op: b.op.clone(),
            attrs: b.attrs.clone(),
            types: b.types.clone(),
            sources: c.wiring[i].clone(),
            group: b.group,
            member: b.group.map(|_| b.member),
            price: c.blocks[i].clone(),
        })
        .collect();
    let muxes: Vec<Mux> = c
        .muxes
        .iter()
        .map(|(b, p, n, w, price)| Mux {
            block: *b,
            port: *p,
            inputs: *n,
            width: *w,
            price: price.clone(),
        })
        .collect();
    let mut covers = Vec::new();
    let mut cfgs = Vec::new();
    for uc in &fu.cfgs {
        let u = &units[uc.unit as usize];
        let dp = dps
            .get(u.class.as_str())
            .ok_or_else(|| format!("{}: no datapath", u.class))?;
        covers.push(Covered {
            unit: u.id.clone(),
            class: u.class.clone(),
            tuples: u.tuples.clone(),
        });
        let mut selects: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        for (n, node) in u.nodes.iter().enumerate() {
            let b = uc.map[n];
            for (i, &(s, res)) in node.args.iter().enumerate() {
                let p = Fu::port(uc, n, i, node) as u32;
                let srcs = &c.wiring[b as usize][p as usize];
                if srcs.len() >= 2 {
                    let want = Source {
                        block: uc.map[s as usize],
                        res,
                    };
                    let k = srcs
                        .iter()
                        .position(|x| *x == want)
                        .ok_or_else(|| format!("{id}: a source missing from the wiring"))?;
                    selects.insert((b, p), k as u32);
                }
            }
        }
        for (k, t) in u.tuples.iter().enumerate() {
            let _ = dp;
            let bindings = u
                .nodes
                .iter()
                .enumerate()
                .map(|(n, node)| {
                    let b = uc.map[n];
                    let types = node.sigs[k].clone();
                    let mode = if types == fu.blocks[b as usize].types {
                        Mode::Native
                    } else {
                        Mode::Widen
                    };
                    Binding {
                        block: b,
                        node: node.name.clone(),
                        var: node.var.clone(),
                        types,
                        mode,
                        ports: (0..node.args.len())
                            .map(|i| Fu::port(uc, n, i, node) as u32)
                            .filter(|_| node.kind == BlockKind::Op)
                            .collect(),
                    }
                })
                .collect();
            cfgs.push(ConfigFile {
                schema: CONFIG_SCHEMA.into(),
                fu: id.to_string(),
                unit: u.id.clone(),
                class: u.class.clone(),
                tuple: t.clone(),
                bindings,
                selects: selects
                    .iter()
                    .map(|(&(block, port), &select)| Select { block, port, select })
                    .collect(),
            });
        }
    }
    let estimated = blocks.iter().any(|b| b.price.estimated) || muxes.iter().any(|m| m.price.estimated);
    Ok((
        FuFile {
            schema: FU_SCHEMA.into(),
            id: id.to_string(),
            family,
            host: fu.host.clone(),
            blocks,
            groups: fu
                .groups
                .iter()
                .enumerate()
                .map(|(i, p)| Group {
                    id: i as u32,
                    pattern: p.clone(),
                })
                .collect(),
            muxes,
            covers,
            area_um2: c.area,
            depth_ps: c.depth,
            bound_ps,
            estimated,
        },
        cfgs,
    ))
}

/// Z_compute ⊆ ⋃ cover(F): every datapath class's unit has a configuration per tuple; every template
/// class has an array for every tuple and a unit for every non-empty stage.
pub fn coverage(
    dps: &[Datapath],
    units: &[Unit],
    unit_fu: &BTreeMap<String, BTreeSet<String>>,
    configs: &[ConfigFile],
    arr: &[ArrayFile],
) -> CoverageFile {
    let have: BTreeSet<(&str, &str)> = configs
        .iter()
        .map(|c| (c.unit.as_str(), c.tuple.as_str()))
        .collect();
    let mut by_class: BTreeMap<&str, Vec<&Unit>> = BTreeMap::new();
    for u in units {
        let e = by_class.entry(u.class.as_str()).or_default();
        if !e.iter().any(|x| x.id == u.id && x.tuples == u.tuples) {
            e.push(u);
        }
    }
    let mut arr_of: BTreeMap<&str, (&str, BTreeSet<&str>)> = BTreeMap::new();
    for a in arr {
        for c in &a.covers {
            let e = arr_of
                .entry(c.class.as_str())
                .or_insert((a.id.as_str(), BTreeSet::new()));
            e.1.extend(c.tuples.iter().map(|t| t.tuple.as_str()));
        }
    }
    let mut classes = Vec::new();
    let mut covered = 0;
    for dp in dps {
        let us = by_class.get(dp.id.as_str()).cloned().unwrap_or_default();
        let mut ok = true;
        let mut pairs = Vec::new();
        for u in &us {
            match unit_fu.get(&u.id) {
                Some(fs) => {
                    for f in fs {
                        if !pairs.contains(&(u.id.clone(), f.clone())) {
                            pairs.push((u.id.clone(), f.clone()));
                        }
                    }
                }
                None => ok = false,
            }
            ok &= u
                .tuples
                .iter()
                .all(|t| have.contains(&(u.id.as_str(), t.as_str())));
        }
        let array = if dp.route == Route::Template {
            match arr_of.get(dp.id.as_str()) {
                Some((a, ts)) => {
                    ok &= dp.tuples.iter().all(|t| ts.contains(t.name.as_str()));
                    Some(a.to_string())
                }
                None => {
                    ok = false;
                    None
                }
            }
        } else {
            ok &= !us.is_empty();
            None
        };
        if ok {
            covered += 1;
        }
        classes.push(ClassCover {
            class: dp.id.clone(),
            route: match dp.route {
                Route::Template => "template".into(),
                Route::Datapath => "datapath".into(),
            },
            units: pairs,
            array,
            empty_stages: empty_stages(dp),
        });
    }
    CoverageFile {
        schema: COVERAGE_SCHEMA.into(),
        total: classes.len() as u64,
        covered,
        classes,
    }
}
