//! `components` (design §4.3): every executed `cc.*` composite call, and every maximal weakly connected
//! region of the entry's own ops, is an instance; each instance is abstracted into a size- and
//! width-free graph; equal graphs across all models are one component, certified by a stored vertex map
//! (docs/passes/components.md).

pub mod algo;
mod config;
pub mod json;
mod verify;
mod view;

use std::collections::{BTreeMap, BTreeSet};

use cc_core::{ArtifactKind, CoreError, PassSection, VerifyReport};
use cc_ir::Module;
use cc_pipeline::{Pass, PassContext, PassError};
use rayon::prelude::*;

pub use config::ComponentsConfig;

use crate::import::{IMPORT, IR_FILE};
use algo::PhaseOut;
use algo::dedup::{Comp, Member};
use json::{
    Bindings, CERTIFICATES_SCHEMA, COMPONENT_SCHEMA, CertificatesFile, ComponentSummary, Counts,
    INSTANCES_SCHEMA, InstanceKind, InstanceRecord, InstancesFile, Position, SUMMARY_SCHEMA, StoredComponent,
    ranges,
};

/// `out/components/{instances,components,certificates}/…`.
pub const COMPONENTS: ArtifactKind = ArtifactKind("components");
pub const VIEW_FILE: &str = "components.md";

#[derive(Debug)]
pub struct Components;

impl Pass for Components {
    fn name(&self) -> &'static str {
        "components"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[IMPORT]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[COMPONENTS]
    }

    fn check_config(&self, section: &PassSection) -> Result<(), CoreError> {
        section.parse::<ComponentsConfig>().map(drop)
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: ComponentsConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let mut units = phases(ctx)?;
        // Largest phases first, so that they do not all end up in flight at the end.
        units.sort_by_key(|u| std::cmp::Reverse(u.2));
        let results: Vec<Result<PhaseOut, String>> = units
            .par_iter()
            .with_max_len(1)
            .map(|(m, p, _)| -> Result<Result<PhaseOut, String>, PassError> {
                ctx.check_cancelled()?;
                let t = std::time::Instant::now();
                let module: Module = ctx.input(IMPORT)?.read_postcard(&format!("{m}/{p}/{IR_FILE}"))?;
                let r = algo::phase(&module, cfg.wl_max_rounds).map_err(|e| format!("{m}/{p}: {e}"));
                tracing::debug!(model = %m, phase = %p, ms = t.elapsed().as_millis() as u64, "phase");
                Ok(r)
            })
            .collect::<Result<_, _>>()?;
        let mut outs = Vec::new();
        let mut failed = Vec::new();
        for r in results {
            match r {
                Ok(o) => outs.push(o),
                Err(e) => failed.push(e),
            }
        }
        if !failed.is_empty() {
            failed.sort();
            return Err(PassError::failed(format!(
                "{} phase(s) failed:\n{}",
                failed.len(),
                failed.join("\n")
            )));
        }
        outs.sort_by(|a, b| (&a.model, &a.phase).cmp(&(&b.model, &b.phase)));
        ctx.check_cancelled()?;
        let t = std::time::Instant::now();
        let (comps, members) = algo::dedup::dedup(&outs, cfg.vf2_budget).map_err(PassError::failed)?;
        tracing::debug!(
            ms = t.elapsed().as_millis() as u64,
            components = comps.len(),
            "dedup"
        );
        write(ctx, &cfg, &outs, &comps, &members)
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

/// Every imported `(model, phase)` with the size of its module file, sorted.
pub(crate) fn phases(ctx: &PassContext) -> Result<Vec<(String, String, u64)>, PassError> {
    let input = ctx.input(IMPORT)?;
    let mut out = Vec::new();
    for m in ctx.models() {
        let dir = input.dir().join(m);
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path().join(IR_FILE);
            if let (Ok(meta), Some(name)) = (std::fs::metadata(&p), e.file_name().to_str()) {
                out.push((m.clone(), name.to_string(), meta.len()));
            }
        }
    }
    out.sort();
    Ok(out)
}

fn cc_path(attrs: &BTreeMap<String, cc_ir::Attr>) -> Option<String> {
    match attrs.get("composite_attributes") {
        Some(cc_ir::Attr::Dict(d)) => d.get("cc_path").and_then(cc_ir::Attr::as_str).map(str::to_string),
        _ => None,
    }
}

fn hex(h: u128) -> String {
    format!("{h:032x}")
}

fn write(
    ctx: &PassContext,
    cfg: &ComponentsConfig,
    outs: &[PhaseOut],
    comps: &[Comp],
    members: &[Vec<Member>],
) -> Result<(), PassError> {
    let out = ctx.output(COMPONENTS)?;
    let mut summaries: Vec<ComponentSummary> = comps
        .iter()
        .map(|c| {
            let mut ops = BTreeMap::new();
            for n in &c.graph.nodes {
                if let algo::graph::Kind::Op { name, .. } = &n.kind {
                    *ops.entry(name.clone()).or_insert(0u64) += 1;
                }
            }
            let children: BTreeSet<String> = c.graph.nodes.iter().filter_map(|n| n.child.clone()).collect();
            ComponentSummary {
                schema: SUMMARY_SCHEMA.into(),
                id: c.id.clone(),
                hash: hex(c.hash),
                height: c.height,
                nodes: c.graph.node_count() as u64,
                edges: c.graph.end_count() as u64,
                dims: c.dims,
                elems: c.elems,
                ops,
                children: children.into_iter().collect(),
                names: BTreeMap::new(),
                models: BTreeMap::new(),
                phases: 0,
                instances: Vec::new(),
            }
        })
        .collect();
    let mut comp_phases: Vec<BTreeSet<(String, String)>> = vec![BTreeSet::new(); comps.len()];
    for (pi, p) in outs.iter().enumerate() {
        let mut recs = Vec::with_capacity(p.instances.len());
        let mut certs = BTreeMap::new();
        for d in &p.instances {
            let mem = d.graph.map(|g| &members[pi][g as usize]);
            let component = mem.map(|x| comps[x.comp as usize].id.clone());
            if let Some(x) = mem {
                let s = &mut summaries[x.comp as usize];
                *s.names
                    .entry(d.name.clone().unwrap_or_else(|| "anon".into()))
                    .or_insert(0) += 1;
                *s.models.entry(p.model.clone()).or_insert(0) += 1;
                s.instances.push(d.id.clone());
                comp_phases[x.comp as usize].insert((p.model.clone(), p.phase.clone()));
                certs.insert(d.id.clone(), x.refs.clone());
            }
            recs.push(InstanceRecord {
                id: d.id.clone(),
                kind: if d.anon {
                    InstanceKind::Anon
                } else {
                    InstanceKind::Composite
                },
                name: d.name.clone(),
                position: Position {
                    path: d.path.clone(),
                    cc_path: cc_path(&d.attrs),
                    parent: d.parent.clone(),
                    children: d.children.clone(),
                },
                attributes: d.attrs.clone(),
                func: d.func,
                ops: ranges(&d.ops),
                counts: Counts {
                    ops: d.ops.len() as u64,
                    size: d.size_ops,
                    nodes: mem.map(|x| x.map.len() as u64).unwrap_or(0),
                },
                component,
                bindings: mem.map(|x| Bindings {
                    dims: x.dims.clone(),
                    elems: x.elems.iter().map(ToString::to_string).collect(),
                    params: x.params.clone(),
                }),
            });
        }
        out.write_json(
            &format!("instances/{}/{}.json", p.model, p.phase),
            &InstancesFile {
                schema: INSTANCES_SCHEMA.into(),
                model: p.model.clone(),
                phase: p.phase.clone(),
                ops: p.ops,
                root: p.root,
                instances: recs,
            },
        )?;
        out.write_json(
            &format!("certificates/{}/{}.json", p.model, p.phase),
            &CertificatesFile {
                schema: CERTIFICATES_SCHEMA.into(),
                model: p.model.clone(),
                phase: p.phase.clone(),
                certificates: certs,
            },
        )?;
    }
    for (s, ph) in summaries.iter_mut().zip(&comp_phases) {
        s.phases = ph.len() as u64;
        s.instances.sort();
    }
    comps
        .par_iter()
        .zip(summaries.par_iter())
        .try_for_each(|(c, s)| -> Result<(), PassError> {
            out.write_postcard(
                &format!("components/{}.postcard", c.id),
                &StoredComponent {
                    schema: COMPONENT_SCHEMA.into(),
                    id: c.id.clone(),
                    hash: hex(c.hash),
                    height: c.height,
                    graph: c.graph.clone(),
                    iface: c.iface.clone(),
                    dims: c.dims,
                    elems: c.elems,
                },
                s,
            )?;
            Ok(())
        })?;
    if cfg.views {
        ctx.dir()
            .write_view(VIEW_FILE, &view::table(ctx.models(), &summaries))?;
    }
    Ok(())
}
