//! `mine` (design §4.10): frequent datapath patterns per family, mined by GraMi (MNI support), re-found
//! by VF2 (non-induced), kept when convex, counted by a maximum independent set of their occurrences
//! per unit and scored by the work they cover (docs/passes/mine.md).

pub mod algo;
pub mod config;
pub mod json;
mod verify;
mod view;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Mutex;

use cc_core::{ArtifactKind, CoreError, LoadedConfig, PassSection, VerifyReport};
use cc_pipeline::{Pass, PassContext, PassError, RunnerKind, ToolSpec};
use rayon::prelude::*;

pub use config::{MineConfig, MisAlgorithm};

use crate::datapaths::DATAPATHS;
use crate::datapaths::json::Datapath;
use crate::iteration_space::json::{Route, Skeleton};
use crate::iteration_space::{ITERATION_SPACE, write_file};
use crate::kernel_classes::KERNEL_CLASSES;
use crate::kernel_classes::json::InstancesTable;
use algo::family::{FamilyGraph, Interner, UnitGraph, family_graph, labels, units};
use algo::rematch::UnitCtx;
use algo::sweep::{FamilyOut, sweep};
use json::{Family, FamilySummary, OVERLAP_SCHEMA, Overlap, SUMMARY_SCHEMA, Shared, Summary};

/// `out/mine/{labels.json, lg, grami, patterns, selection, sweep, overlap.json, summary.json}`.
pub const MINE: ArtifactKind = ArtifactKind("mine");
pub const VIEW: &str = "mine.md";

const TOOLS: &[ToolSpec] = &[ToolSpec {
    name: "java",
    runner: RunnerKind::Docker,
    program: "java",
    version_args: &["-version"],
}];

#[derive(Debug)]
pub struct Mine;

impl Pass for Mine {
    fn name(&self) -> &'static str {
        "mine"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[DATAPATHS, ITERATION_SPACE, KERNEL_CLASSES]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[MINE]
    }

    fn tools(&self) -> &[ToolSpec] {
        TOOLS
    }

    fn fingerprints(
        &self,
        config: &LoadedConfig,
        _section: &PassSection,
    ) -> Result<BTreeMap<String, String>, CoreError> {
        let t = &config.config.tools;
        let mut m = BTreeMap::new();
        m.insert("image".into(), format!("{}:{}", t.docker.image, t.docker.tag));
        m.insert("grami".into(), t.grami.commit.clone());
        Ok(m)
    }

    fn check_config(&self, section: &PassSection) -> Result<(), CoreError> {
        let c: MineConfig = section.parse()?;
        c.check().map_err(|message| CoreError::PassConfig {
            pass: "mine".into(),
            message,
        })
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: MineConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let inputs = load(ctx)?;
        let all = units(&inputs.datapaths, &inputs.skeletons, &inputs.tables).map_err(PassError::failed)?;
        let labels = labels(&all);
        let it = Interner::new(&labels);
        let out = ctx.output(MINE)?;
        out.write_json("labels.json", &labels)?;
        let by_id: HashMap<&str, &Datapath> = inputs.datapaths.iter().map(|d| (d.id.as_str(), d)).collect();
        let t = &ctx.tools();
        let mut prepared = Vec::new();
        for f in Family::ALL {
            let fg = family_graph(f, &all, &it).map_err(PassError::failed)?;
            write_file(&out.path(&format!("lg/{}.lg", f.name()))?, &fg.lg)?;
            out.write_json(&format!("lg/{}.map.json", f.name()), &fg.map)?;
            let ctxs = unit_ctxs(&fg, &all, &by_id)?;
            let key = format!(
                "{:016x}",
                (cc_ir::absgraph::hash_bytes(
                    format!(
                        "{}\n{}:{}\nmaxNodes={}\n{}",
                        t.grami.commit, t.docker.image, t.docker.tag, cfg.max_vertices, fg.lg
                    )
                    .as_bytes()
                ) >> 64) as u64
            );
            prepared.push((fg, ctxs, key));
        }

        // Families in parallel, as many as the memory admits GraMi containers.
        let per = (cfg.heap_gb + 2) as f64;
        let slots = ((ctx.resources().memory_gb / per).floor() as usize).clamp(1, prepared.len());
        let queue = Mutex::new((0..prepared.len()).collect::<Vec<_>>());
        let results: Mutex<Vec<Option<Result<FamilyOut, PassError>>>> =
            Mutex::new((0..prepared.len()).map(|_| None).collect());
        rayon::scope(|s| {
            for _ in 0..slots {
                s.spawn(|_| {
                    loop {
                        let next = queue.lock().ok().and_then(|mut q| q.pop());
                        let Some(i) = next else { break };
                        let (fg, ctxs, key) = &prepared[i];
                        let r = sweep(ctx, &cfg, fg, ctxs, &labels, key);
                        if let Ok(mut v) = results.lock() {
                            v[i] = Some(r);
                        }
                    }
                });
            }
        });
        let results = results.into_inner().map_err(|_| PassError::failed("a family worker panicked"))?;
        let mut outs = Vec::new();
        for r in results {
            outs.push(r.ok_or_else(|| PassError::failed("a family was not mined"))??);
        }

        let mut summary = Summary {
            schema: SUMMARY_SCHEMA.into(),
            families: BTreeMap::new(),
            node_labels: labels.nodes.len() as u64,
            edge_labels: labels.edges.len() as u64,
            overlap: 0,
            mis_algorithm: cfg.mis.algorithm.name().into(),
        };
        for ((fg, _, _), o) in prepared.iter().zip(&outs) {
            let f = fg.family.name();
            o.patterns.par_iter().try_for_each(|p| -> Result<(), PassError> {
                out.write_json(&format!("patterns/{f}/{}.json", p.id), p)?;
                Ok(())
            })?;
            out.write_json(&format!("grami/{f}.json"), &o.grami)?;
            out.write_json(&format!("selection/{f}.json"), &o.selection)?;
            out.write_json(&format!("sweep/{f}.json"), &o.sweep)?;
            summary.families.insert(f.into(), family_summary(fg, o));
        }
        let overlap = overlap(&outs);
        summary.overlap = overlap.patterns.len() as u64;
        out.write_json("overlap.json", &overlap)?;
        out.write_json("summary.json", &summary)?;
        if cfg.views {
            ctx.dir().write_view(VIEW, &view::view(&outs, &overlap, &summary))?;
        }
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

/// Everything the pass reads: every datapath, the skeletons of template classes, every instance table.
#[derive(Debug)]
pub struct Inputs {
    pub datapaths: Vec<Datapath>,
    pub skeletons: HashMap<String, Skeleton>,
    pub tables: HashMap<String, InstancesTable>,
}

pub fn load(ctx: &PassContext) -> Result<Inputs, PassError> {
    let dps = ctx.input(DATAPATHS)?;
    let dir = dps.dir().join("datapaths");
    let mut ids: Vec<String> = std::fs::read_dir(&dir)
        .map_err(|e| PassError::failed(format!("{}: {e}", dir.display())))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".json")).map(str::to_string))
        .collect();
    ids.sort();
    let datapaths: Vec<Datapath> = ids
        .par_iter()
        .map(|id| dps.read_json::<Datapath>(&format!("datapaths/{id}.json")))
        .collect::<Result<_, _>>()?;
    let isp = ctx.input(ITERATION_SPACE)?;
    let skeletons = datapaths
        .par_iter()
        .filter(|d| d.route == Route::Template)
        .map(|d| Ok((d.id.clone(), isp.read_json::<Skeleton>(&format!("skeletons/{}.json", d.id))?)))
        .collect::<Result<HashMap<_, _>, CoreError>>()?;
    let kc = ctx.input(KERNEL_CLASSES)?;
    let tables = datapaths
        .par_iter()
        .map(|d| Ok((d.id.clone(), kc.read_json::<InstancesTable>(&format!("instances/{}.json", d.id))?)))
        .collect::<Result<HashMap<_, _>, CoreError>>()?;
    Ok(Inputs {
        datapaths,
        skeletons,
        tables,
    })
}

fn unit_ctxs(
    fg: &FamilyGraph,
    all: &[UnitGraph],
    by_id: &HashMap<&str, &Datapath>,
) -> Result<Vec<UnitCtx>, PassError> {
    let src: HashMap<&str, &UnitGraph> = all.iter().map(|u| (u.unit.id.as_str(), u)).collect();
    fg.map
        .units
        .iter()
        .zip(&fg.graphs)
        .map(|(u, g)| {
            let dp = by_id
                .get(u.class.as_str())
                .ok_or_else(|| PassError::failed(format!("{}: no datapath", u.class)))?;
            debug_assert!(src.contains_key(u.id.as_str()));
            UnitCtx::new(u.clone(), g.clone(), dp).map_err(PassError::failed)
        })
        .collect()
}

fn family_summary(fg: &FamilyGraph, o: &FamilyOut) -> FamilySummary {
    FamilySummary {
        units: fg.map.units.len() as u64,
        vertices: fg.map.vertices,
        edges: fg.map.edges,
        tau: o.sweep.tau,
        stopped_by: o.sweep.stopped_by,
        reported: o.grami.reported,
        kept: o.patterns.len() as u64,
        selected: o.selection.patterns.len() as u64,
        occurrences: o.patterns.iter().map(|p| p.occurrences.len() as u64).sum(),
        workload: fg.map.units.iter().map(|u| u.workload).sum(),
    }
}

/// Kept patterns found in several families.
pub fn overlap(outs: &[FamilyOut]) -> Overlap {
    let mut fams: BTreeMap<&str, BTreeSet<Family>> = BTreeMap::new();
    let mut sel: BTreeMap<&str, BTreeSet<Family>> = BTreeMap::new();
    for o in outs {
        for p in &o.patterns {
            fams.entry(&p.id).or_default().insert(p.family);
        }
        for s in &o.selection.patterns {
            sel.entry(&s.id).or_default().insert(o.selection.family);
        }
    }
    Overlap {
        schema: OVERLAP_SCHEMA.into(),
        patterns: fams
            .into_iter()
            .filter(|(_, f)| f.len() > 1)
            .map(|(id, f)| Shared {
                id: id.into(),
                families: f.into_iter().collect(),
                selected_in: sel.get(id).map(|s| s.iter().copied().collect()).unwrap_or_default(),
            })
            .collect(),
    }
}
