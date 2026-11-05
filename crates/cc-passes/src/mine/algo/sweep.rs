//! The τ sweep of one family (docs/passes/mine.md §4.2, §4.5): GraMi at τ = `tau_start`, halving to
//! `tau_min`; every reported pattern within `max_vertices` is re-matched, its MIS per unit computed and
//! scored; the sweep stops when the top-k selection (non-empty) equals the previous τ's, at `tau_min`,
//! or when the time budget runs out (the last finished τ is kept).

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use cc_ir::absgraph::{Graph, hash_bytes};
use cc_pipeline::{PassContext, PassError};
use rayon::prelude::*;

use super::canon::canon;
use super::family::FamilyGraph;
use super::grami::{self, Run};
use super::lg::{LgGraph, parse_output};
use super::mis;
use super::rematch::{UnitCtx, rematch};
use crate::mine::config::{GramiRuns, MineConfig};
use crate::mine::json::{
    Family, GRAMI_SCHEMA, GramiFile, Labels, PATTERN_SCHEMA, PatternEdge, PatternFile, Reported,
    SELECTION_SCHEMA, SWEEP_SCHEMA, Selected, Selection, Step, Stop, Sweep, UnitMis,
};

/// One family's results.
#[derive(Debug, Clone)]
pub struct FamilyOut {
    pub grami: GramiFile,
    pub patterns: Vec<PatternFile>,
    pub selection: Selection,
    pub sweep: Sweep,
}

/// A GraMi pattern in canonical order, with label ids.
#[derive(Debug, Clone)]
pub struct Canonical {
    pub id: String,
    pub code: String,
    pub labels: Vec<String>,
    pub edges: Vec<PatternEdge>,
    pub graph: Graph<u32>,
}

/// Canonicalize a pattern of GraMi's output (labels are ids of `labels`).
pub fn canonical(p: &LgGraph, labels: &Labels) -> Result<Canonical, String> {
    let keys: Vec<String> = p
        .nodes
        .iter()
        .map(|&l| {
            labels
                .nodes
                .get(l as usize)
                .cloned()
                .ok_or_else(|| format!("node label {l} not in labels.json"))
        })
        .collect::<Result<_, _>>()?;
    let edges: Vec<(u32, u32, cc_ir::absgraph::EdgeLabel)> = p
        .edges
        .iter()
        .map(|&(a, b, l)| {
            labels
                .edges
                .get(l as usize)
                .cloned()
                .map(|e| (a, b, e))
                .ok_or_else(|| format!("edge label {l} not in labels.json"))
        })
        .collect::<Result<_, _>>()?;
    let c = canon(&keys, &edges);
    let mut pos = vec![0u32; keys.len()];
    for (i, &v) in c.order.iter().enumerate() {
        pos[v] = i as u32;
    }
    let mut graph = Graph::default();
    for &v in &c.order {
        graph.add_node(p.nodes[v]);
    }
    let mut pe = Vec::new();
    for (a, b, l) in &edges {
        let (a, b) = (pos[*a as usize], pos[*b as usize]);
        if graph.edge(a, b).is_some() {
            return Err(format!("parallel pattern edges {a} -> {b}"));
        }
        for e in l {
            graph.add_edge(a, b, *e);
        }
        pe.push(PatternEdge {
            src: a,
            dst: b,
            ends: l.clone(),
        });
    }
    pe.sort();
    Ok(Canonical {
        id: c.id,
        code: c.code,
        labels: c.order.iter().map(|&v| keys[v].clone()).collect(),
        edges: pe,
        graph,
    })
}

/// Re-match, MIS per unit and score of one pattern (its `tau` is set by the caller).
pub fn evaluate(
    family: Family,
    p: &Canonical,
    units: &[UnitCtx],
    cfg: &MineConfig,
) -> Result<PatternFile, String> {
    let m = rematch(&p.graph, units, cfg.vf2_budget).map_err(|e| format!("{}: {e}", p.id))?;
    let mut by_unit: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for (k, (u, _)) in m.at.iter().enumerate() {
        by_unit.entry(*u).or_default().push(k as u32);
    }
    let mut out_mis = Vec::new();
    let mut score = 0u128;
    for (u, occ) in by_unit {
        let imgs: Vec<&[u32]> = occ.iter().map(|&k| m.at[k as usize].1.as_slice()).collect();
        let g = mis::overlap(&imgs);
        let r =
            mis::solve(cfg.mis.algorithm, &g, cfg.mis.node_budget).map_err(|e| format!("{}: {e}", p.id))?;
        let size = r.chosen.len() as u64;
        let w = units[u].unit.workload;
        score = score.saturating_add(w.saturating_mul(size as u128));
        out_mis.push(UnitMis {
            unit: units[u].unit.id.clone(),
            workload: w,
            chosen: r.chosen.iter().map(|&i| occ[i as usize]).collect(),
            occurrences: occ,
            algorithm: cfg.mis.algorithm.name().into(),
            size,
            optimal: r.optimal,
            bound: r.bound,
        });
    }
    Ok(PatternFile {
        schema: PATTERN_SCHEMA.into(),
        id: p.id.clone(),
        family,
        code: p.code.clone(),
        labels: p.labels.clone(),
        edges: p.edges.clone(),
        tau: 0,
        mni: m.mni,
        embeddings: m.embeddings,
        non_convex: m.non_convex,
        occurrences: m.occurrences,
        mis: out_mis,
        score,
    })
}

/// The top-k with a positive score, by (score desc, vertices desc, id).
pub fn select<'a>(patterns: impl Iterator<Item = &'a PatternFile>, k: usize) -> Vec<Selected> {
    let mut v: Vec<Selected> = patterns
        .filter(|p| p.score > 0)
        .map(|p| Selected {
            id: p.id.clone(),
            vertices: p.labels.len() as u32,
            score: p.score,
            mni: p.mni,
            mis: p.mis.iter().map(|m| m.size).sum(),
        })
        .collect();
    v.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then(b.vertices.cmp(&a.vertices))
            .then(a.id.cmp(&b.id))
    });
    v.truncate(k);
    v
}

/// Hash of an ordered selection.
pub fn selection_hash(s: &[Selected]) -> String {
    let ids: Vec<&str> = s.iter().map(|x| x.id.as_str()).collect();
    format!("{:016x}", (hash_bytes(ids.join("\n").as_bytes()) >> 64) as u64)
}

/// Run the sweep of one family. `key` names the GraMi cache of its graph.
pub fn sweep(
    ctx: &PassContext,
    cfg: &MineConfig,
    fg: &FamilyGraph,
    units: &[UnitCtx],
    labels: &Labels,
    key: &str,
) -> Result<FamilyOut, PassError> {
    let family = fg.family;
    let taus = cfg.taus();
    let mut steps = Vec::new();
    let mut evaluated: HashMap<String, PatternFile> = HashMap::new();
    // (τ, GraMi count, dropped, reported, selection) of the last finished step.
    type Last = (u64, u64, u64, Vec<Reported>, Vec<Selected>);
    let mut last: Option<Last> = None;
    let mut stop = Stop::TauMin;
    let t0 = Instant::now();
    // `once`: GraMi at τ_min, every pattern canonicalized and evaluated; a step's set is then the
    // patterns whose MNI reaches its τ. GraMi enumerates every frequent pattern within the size bound
    // and MNI is anti-monotone, so these are the sets one run per step reports (equal counts at every
    // step of every family on the model set). A timeout or a pattern over the bound falls back to the
    // sweep.
    let mut once: Option<Vec<Canonical>> = None;
    if cfg.grami_runs == GramiRuns::Once
        && fg.map.edges != 0
        && let Some(&tau) = taus.last()
    {
        let limit = (cfg.budget_s as f64).min(cfg.run_timeout_s as f64);
        if let Run::Done(text) = grami::run(ctx, key, &fg.lg, tau, cfg.heap_gb, cfg.max_vertices, limit)? {
            let fail = |e: String| PassError::failed(format!("{} at τ = {tau}: {e}", family.name()));
            let raw = parse_output(&text).map_err(fail)?;
            if raw.iter().all(|p| p.nodes.len() <= cfg.max_vertices) {
                let canon: Vec<Canonical> = raw
                    .par_iter()
                    .map(|p| canonical(p, labels))
                    .collect::<Result<_, _>>()
                    .map_err(fail)?;
                let done: Vec<PatternFile> = canon
                    .par_iter()
                    .map(|c| evaluate(family, c, units, cfg))
                    .collect::<Result<_, _>>()
                    .map_err(fail)?;
                for p in done {
                    evaluated.insert(p.id.clone(), p);
                }
                once = Some(canon);
            }
        }
    }
    if fg.map.edges == 0 {
        stop = Stop::Empty;
    } else {
        for (i, &tau) in taus.iter().enumerate() {
            ctx.check_cancelled()?;
            let fail = |e: String| PassError::failed(format!("{} at τ = {tau}: {e}", family.name()));
            let (raw_len, dropped, canon) = if let Some(all) = &once {
                let canon: Vec<Canonical> = all
                    .iter()
                    .filter(|c| evaluated.get(&c.id).is_some_and(|p| p.mni >= tau))
                    .cloned()
                    .collect();
                (canon.len(), 0u64, canon)
            } else {
                let left = cfg.budget_s as f64 - t0.elapsed().as_secs_f64();
                let run = if left <= 0.0 {
                    Run::Timeout
                } else {
                    grami::run(
                        ctx,
                        key,
                        &fg.lg,
                        tau,
                        cfg.heap_gb,
                        cfg.max_vertices,
                        left.min(cfg.run_timeout_s as f64),
                    )?
                };
                let text = match run {
                    Run::Done(t) => t,
                    Run::Timeout => {
                        steps.push(Step {
                            tau,
                            finished: false,
                            reported: 0,
                            kept: 0,
                            selection: String::new(),
                        });
                        stop = Stop::Budget;
                        break;
                    }
                };
                let raw = parse_output(&text).map_err(fail)?;
                // Only patterns within `max_vertices` are canonicalized (the code search is factorial in
                // the size of a label class).
                let small: Vec<&LgGraph> = raw.iter().filter(|p| p.nodes.len() <= cfg.max_vertices).collect();
                let dropped = (raw.len() - small.len()) as u64;
                let canon: Vec<Canonical> = small
                    .par_iter()
                    .map(|p| canonical(p, labels))
                    .collect::<Result<_, _>>()
                    .map_err(fail)?;
                (raw.len(), dropped, canon)
            };
            let mut reported: Vec<Reported> = canon
                .iter()
                .map(|c| Reported {
                    id: c.id.clone(),
                    vertices: c.labels.len() as u32,
                })
                .collect();
            reported.sort_by(|a, b| a.id.cmp(&b.id));
            if reported.windows(2).any(|w| w[0].id == w[1].id) {
                return Err(fail("GraMi reported a pattern twice".into()));
            }
            let todo: Vec<&Canonical> = canon.iter().filter(|c| !evaluated.contains_key(&c.id)).collect();
            let done: Vec<PatternFile> = todo
                .par_iter()
                .map(|c| evaluate(family, c, units, cfg))
                .collect::<Result<_, _>>()
                .map_err(fail)?;
            for p in done {
                evaluated.insert(p.id.clone(), p);
            }
            let kept: Vec<&PatternFile> = reported.iter().filter_map(|r| evaluated.get(&r.id)).collect();
            let sel = select(kept.iter().copied(), cfg.top_k);
            steps.push(Step {
                tau,
                finished: true,
                reported: raw_len as u64,
                kept: kept.len() as u64,
                selection: selection_hash(&sel),
            });
            let stable = last
                .as_ref()
                .is_some_and(|(_, _, _, _, prev)| !sel.is_empty() && *prev == sel);
            last = Some((tau, raw_len as u64, dropped, reported, sel));
            if stable {
                stop = Stop::Stable;
                break;
            }
            if i + 1 == taus.len() {
                stop = Stop::TauMin;
            }
        }
    }
    let (tau, count, dropped, reported, sel) = match last {
        Some((t, c, d, r, s)) => (Some(t), c, d, r, s),
        None => (None, 0, 0, Vec::new(), Vec::new()),
    };
    let mut patterns: Vec<PatternFile> = reported
        .iter()
        .filter_map(|r| evaluated.get(&r.id).cloned())
        .collect();
    for p in &mut patterns {
        p.tau = tau.unwrap_or(0);
    }
    Ok(FamilyOut {
        grami: GramiFile {
            schema: GRAMI_SCHEMA.into(),
            family,
            tau,
            reported: count,
            dropped,
            patterns: reported,
        },
        patterns,
        selection: Selection {
            schema: SELECTION_SCHEMA.into(),
            family,
            tau,
            top_k: cfg.top_k as u64,
            patterns: sel,
        },
        sweep: Sweep {
            schema: SWEEP_SCHEMA.into(),
            family,
            steps,
            tau,
            stopped_by: stop,
        },
    })
}
