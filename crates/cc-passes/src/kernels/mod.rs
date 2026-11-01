//! `kernels` (design §4.5): every normalized component partitioned into kernels (at most one core with
//! its fused element-wise prologue and epilogue, or a map kernel of element-wise ops) and boundary ops,
//! with every kernel also written as a graph of its own (docs/passes/kernels.md).

pub mod algo;
pub mod config;
pub mod json;
pub mod verify;
mod view;

use std::collections::BTreeMap;

use cc_core::ArtifactKind;
use cc_core::VerifyReport;
use cc_ir::absgraph::Graph;
use cc_pipeline::{Pass, PassContext, PassError};
use rayon::prelude::*;

pub use config::KernelsConfig;

use crate::normalize::NORMALIZE;
use crate::normalize::json::{NNode, NormalizedComponent, NormalizedSummary};
use algo::Partition;
use json::{
    BoundaryKind, GRAPHS_SCHEMA, KernelGraphs, KernelPartition, KernelsSummary, PARTITION_SCHEMA,
    PartitionSummary, SUMMARY_SCHEMA, TOTALS_SCHEMA,
};

/// `out/kernels/{components,kernels}/…`, `out/kernels/summary.json`.
pub const KERNELS: ArtifactKind = ArtifactKind("kernels");
pub const VIEW_FILE: &str = "kernels.md";

#[derive(Debug)]
pub struct Kernels;

impl Pass for Kernels {
    fn name(&self) -> &'static str {
        "kernels"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[NORMALIZE]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[KERNELS]
    }

    fn check_config(&self, section: &cc_core::PassSection) -> Result<(), cc_core::CoreError> {
        let _: KernelsConfig = section.parse()?;
        Ok(())
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: KernelsConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let ids = component_ids(ctx)?;
        let out = ctx.output(KERNELS)?;
        let input = ctx.input(NORMALIZE)?;
        let done: Vec<(PartitionSummary, Option<String>)> = ids
            .par_iter()
            .map(|id| -> Result<(PartitionSummary, Option<String>), PassError> {
                ctx.check_cancelled()?;
                let c: NormalizedComponent = input.read_postcard(&format!("components/{id}.postcard"))?;
                let ns: NormalizedSummary =
                    input.read_json(&format!("components/{id}.postcard.summary.json"))?;
                let p = algo::partition(&c.graph).map_err(|e| PassError::failed(format!("{id}: {e}")))?;
                let graphs = p
                    .kernels
                    .iter()
                    .map(|k| algo::extract::extract(&c.graph, k))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| PassError::failed(format!("{id}: {e}")))?;
                let s = summarize(id, &c.graph, &p, &ns);
                out.write_json(
                    &format!("components/{id}.json"),
                    &KernelPartition {
                        schema: PARTITION_SCHEMA.into(),
                        id: id.clone(),
                        kernels: p.kernels.clone(),
                        boundaries: p.boundaries.clone(),
                        order: p.order.clone(),
                        evictions: p.evictions,
                    },
                )?;
                out.write_postcard(
                    &format!("kernels/{id}.postcard"),
                    &KernelGraphs {
                        schema: GRAPHS_SCHEMA.into(),
                        id: id.clone(),
                        kernels: graphs,
                    },
                    &s,
                )?;
                let dot = cfg.views.then(|| view::dot(id, &c.graph, &p));
                Ok((s, dot))
            })
            .collect::<Result<_, _>>()?;
        let summaries: Vec<PartitionSummary> = done.iter().map(|d| d.0.clone()).collect();
        let total = totals(&summaries);
        out.write_json("summary.json", &total)?;
        if cfg.views {
            for (s, dot) in &done {
                if let Some(d) = dot {
                    ctx.dir().write_view(&format!("{}.dot", s.id), d)?;
                }
            }
            ctx.dir()
                .write_view(VIEW_FILE, &view::table(&total, &summaries))?;
        }
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

/// The component ids of the input, sorted.
pub(crate) fn component_ids(ctx: &PassContext) -> Result<Vec<String>, PassError> {
    let dir = ctx.input(NORMALIZE)?.dir().join("components");
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

/// `<op>/<family>` of a core node, e.g. `dot_general/sint`.
pub fn core_key(g: &Graph<NNode>, core: u32) -> String {
    let n = &g.nodes[core as usize].label;
    let op = match &n.kind {
        // A custom call is named by its target (`dynamic_conv`, `dynamic_reduce_window`).
        crate::components::algo::graph::Kind::Op { name, target } => {
            let n = target.as_deref().unwrap_or(name);
            n.strip_prefix("stablehlo.").unwrap_or(n).to_string()
        }
        k => format!("{k:?}"),
    };
    let fam = n
        .results
        .first()
        .map(|t| family_name(&t.family))
        .unwrap_or_else(|| "none".into());
    format!("{op}/{fam}")
}

fn family_name(f: &crate::components::algo::graph::Family) -> String {
    use crate::components::algo::graph::Family as F;
    match f {
        F::Bool => "bool".into(),
        F::Sint => "sint".into(),
        F::Uint => "uint".into(),
        F::Float => "float".into(),
        F::Complex(e) => format!("complex<{}>", family_name(e)),
    }
}

fn summarize(id: &str, g: &Graph<NNode>, p: &Partition, ns: &NormalizedSummary) -> PartitionSummary {
    let mut cores: BTreeMap<String, u64> = BTreeMap::new();
    let mut roles: BTreeMap<String, u64> = BTreeMap::new();
    let mut boundaries: BTreeMap<String, u64> = BTreeMap::new();
    let (mut call_fed, mut low_rank, mut core_kernels) = (0u64, 0u64, 0u64);
    let calls: std::collections::BTreeSet<u32> = p
        .boundaries
        .iter()
        .filter(|b| b.kind == BoundaryKind::Call)
        .map(|b| b.node)
        .collect();
    for k in &p.kernels {
        for m in &k.members {
            *roles.entry(m.role.name().into()).or_default() += 1;
        }
        match k.core {
            Some(c) => {
                core_kernels += 1;
                *cores.entry(core_key(g, c)).or_default() += 1;
            }
            None => {
                if k.inputs.iter().any(|c| calls.contains(&c.src)) {
                    call_fed += 1;
                }
                if k.domain.len() <= 1 {
                    low_rank += 1;
                }
            }
        }
    }
    for b in &p.boundaries {
        *boundaries.entry(b.kind.name().into()).or_default() += 1;
    }
    PartitionSummary {
        schema: SUMMARY_SCHEMA.into(),
        id: id.into(),
        nodes: g.node_count() as u64,
        kernels: p.kernels.len() as u64,
        core_kernels,
        map_kernels: p.kernels.len() as u64 - core_kernels,
        cores,
        roles,
        boundaries,
        call_fed_maps: call_fed,
        low_rank_maps: low_rank,
        evictions: u64::from(p.evictions),
        instances: ns.instances,
        models: ns.models.clone(),
    }
}

pub fn totals(ss: &[PartitionSummary]) -> KernelsSummary {
    let mut t = KernelsSummary {
        schema: TOTALS_SCHEMA.into(),
        components: ss.len() as u64,
        kernels: 0,
        core_kernels: 0,
        map_kernels: 0,
        kernels_per_component: BTreeMap::new(),
        cores: BTreeMap::new(),
        roles: BTreeMap::new(),
        boundaries: BTreeMap::new(),
        call_fed_maps: 0,
        low_rank_maps: 0,
        evictions: 0,
        models: BTreeMap::new(),
    };
    for s in ss {
        t.kernels += s.kernels;
        t.core_kernels += s.core_kernels;
        t.map_kernels += s.map_kernels;
        *t.kernels_per_component.entry(s.kernels).or_default() += 1;
        for (k, &v) in &s.cores {
            let e = t.cores.entry(k.clone()).or_default();
            e.0 += v;
            e.1 += v * s.instances;
        }
        for (k, &v) in &s.roles {
            *t.roles.entry(k.clone()).or_default() += v;
        }
        for (k, &v) in &s.boundaries {
            *t.boundaries.entry(k.clone()).or_default() += v;
        }
        t.call_fed_maps += s.call_fed_maps;
        t.low_rank_maps += s.low_rank_maps;
        t.evictions += s.evictions;
        for (m, &n) in &s.models {
            let e = t.models.entry(m.clone()).or_default();
            e.0 += n * s.kernels;
            e.1 += n * s.core_kernels;
        }
    }
    t
}
