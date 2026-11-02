//! `kernel_classes` (design §4.6): every kernel of every component grouped into canonical classes Z
//! (identity = structure and dtype family; extents, widths and values are parameters), with an instance
//! table per class, certificates, and the sharing report at component and kernel level
//! (docs/passes/kernel_classes.md).

pub mod algo;
pub mod config;
pub mod json;
pub mod verify;
pub mod view;

use std::collections::{BTreeMap, BTreeSet};

use cc_core::{ArtifactKind, VerifyReport};
use cc_ir::Elem;
use cc_pipeline::{Pass, PassContext, PassError};
use rayon::prelude::*;

pub use config::KernelClassesConfig;

use crate::components::COMPONENTS;
use crate::components::json::InstancesFile;
use crate::export::EXPORT;
use crate::export::json::SymbolsFile;
use crate::kernels::KERNELS;
use crate::kernels::json::{KernelGraphs, KernelPartition};
use crate::normalize::NORMALIZE;
use crate::normalize::json::NormalizedComponent;
use algo::encode::Encoded;
use algo::{Inst, KInfo, RawRow};
use json::{
    CERTIFICATES_SCHEMA, CLASS_SCHEMA, Certificate, CertificatesFile, ClassFile, ClassesSummary,
    INSTANCES_SCHEMA, InstancesTable, MemberKey, Params, Row, SUMMARY_SCHEMA, Tags, ZNode,
};

/// `out/kernel_classes/{classes,instances,certificates}/<z>.json`, `sharing.json`, `summary.json`.
pub const KERNEL_CLASSES: ArtifactKind = ArtifactKind("kernel_classes");
pub const SHARING_VIEW: &str = "sharing.md";
pub const CLASSES_VIEW: &str = "classes.md";

#[derive(Debug)]
pub struct KernelClasses;

impl Pass for KernelClasses {
    fn name(&self) -> &'static str {
        "kernel_classes"
    }

    fn version(&self) -> u32 {
        1
    }

    fn requires(&self) -> &[ArtifactKind] {
        &[KERNELS, NORMALIZE, COMPONENTS, EXPORT]
    }

    fn provides(&self) -> &[ArtifactKind] {
        &[KERNEL_CLASSES]
    }

    fn check_config(&self, section: &cc_core::PassSection) -> Result<(), cc_core::CoreError> {
        let _: KernelClassesConfig = section.parse()?;
        Ok(())
    }

    fn run(&self, ctx: &mut PassContext) -> Result<(), PassError> {
        let cfg: KernelClassesConfig = ctx.options()?;
        let ctx: &PassContext = ctx;
        let fail = |e: String| PassError::failed(e);
        let comps = load_kernels(ctx)?;
        let phases = phases(ctx)?;
        let models: Vec<String> = phases
            .iter()
            .map(|(m, _)| m.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        // Rows, in each kernel's own numbering.
        let per_phase: Vec<Vec<RawRow>> = phases
            .par_iter()
            .map(|(m, p)| {
                ctx.check_cancelled()?;
                phase_rows(ctx, &comps, m, p)
            })
            .collect::<Result<_, _>>()?;
        let raw: Vec<RawRow> = per_phase.into_iter().flatten().collect();

        // Member kernels: one per kernel and group kinds seen.
        let mut keys: BTreeSet<MemberKey> = raw.iter().map(|r| member_key(&comps, r)).collect();
        for (c, ks) in &comps {
            for k in ks {
                if !keys.iter().any(|x| &x.component == c && x.kernel == k.id) {
                    keys.insert(MemberKey {
                        component: c.clone(),
                        kernel: k.id.clone(),
                        groups: Default::default(),
                    });
                }
            }
        }
        let keys: Vec<MemberKey> = keys.into_iter().collect();
        let info = |k: &MemberKey| -> &KInfo {
            let ks = &comps[&k.component];
            &ks[ks.iter().position(|x| x.id == k.kernel).unwrap_or_default()]
        };
        let encoded: Vec<Encoded> = keys
            .par_iter()
            .map(|k| {
                let ki = info(k);
                algo::encode::encode(&ki.graph, &ki.local, &k.groups)
                    .map_err(|e| format!("{}/{}: {e}", k.component, k.kernel))
            })
            .collect::<Result<_, _>>()
            .map_err(fail)?;
        let refs: Vec<&Encoded> = encoded.iter().collect();
        let (found, split_buckets) =
            algo::classify::classify(&refs, cfg.wl_max_rounds, cfg.vf2_budget).map_err(fail)?;

        // Certificates and the member → class map.
        let mut member_of: BTreeMap<&MemberKey, (usize, u32)> = BTreeMap::new();
        let mut certs: Vec<Vec<Certificate>> = Vec::with_capacity(found.len());
        for (ci, f) in found.iter().enumerate() {
            let rep = &encoded[f.members[0].0];
            let (n, nd) = (rep.n as usize, rep.nd as usize);
            let mut cs = Vec::new();
            for (mi, (item, map)) in f.members.iter().enumerate() {
                member_of.insert(&keys[*item], (ci, mi as u32));
                let off = |x: usize, base: usize| (x - base) as u32;
                cs.push(Certificate {
                    member: keys[*item].clone(),
                    nodes: map[..n].iter().map(|&x| x as u32).collect(),
                    axes: map[n..n + nd].iter().map(|&x| off(x, n)).collect(),
                    elems: map[n + nd..n + nd + rep.ne as usize]
                        .iter()
                        .map(|&x| off(x, n + nd))
                        .collect(),
                    local: info(&keys[*item]).local.clone(),
                });
            }
            certs.push(cs);
        }

        // Rows per class, in the representative's numbering.
        let mut rows: Vec<Vec<Row>> = vec![Vec::new(); found.len()];
        let mut columns: Vec<BTreeSet<String>> = vec![BTreeSet::new(); found.len()];
        let mut named: Vec<Vec<BTreeMap<String, cc_ir::Attr>>> = vec![Vec::new(); found.len()];
        let mut symbolic: Vec<bool> = vec![false; found.len()];
        for r in &raw {
            let key = member_key(&comps, r);
            let (ci, mi) = member_of[&key];
            let cert = &certs[ci][mi as usize];
            let mut inv = vec![0u32; cert.nodes.len()];
            for (i, &j) in cert.nodes.iter().enumerate() {
                inv[j as usize] = i as u32;
            }
            let vals: BTreeMap<String, cc_ir::Attr> = r
                .values
                .iter()
                .map(|(n, k, v)| (format!("n{}.{k}", inv[*n as usize]), v.clone()))
                .collect();
            columns[ci].extend(vals.keys().cloned());
            symbolic[ci] |= r.symbolic;
            named[ci].push(vals);
            rows[ci].push(Row {
                model: r.model.clone(),
                phase: r.phase.clone(),
                instance: r.instance.clone(),
                component: r.component.clone(),
                kernel: key.kernel.clone(),
                member: mi,
                extents: cert.axes.iter().map(|&c| r.extents[c as usize].0).collect(),
                sizes: cert
                    .axes
                    .iter()
                    .map(|&c| r.extents[c as usize].1.clone())
                    .collect(),
                elems: cert.elems.iter().map(|&e| r.elems[e as usize].clone()).collect(),
                values: Vec::new(),
                workload: r.workload,
                trip_count: r.trip_count,
                weighted: r
                    .workload
                    .checked_mul(r.trip_count)
                    .ok_or_else(|| fail(format!("{}: weighted workload overflows", r.instance)))?,
            });
        }
        let columns: Vec<Vec<String>> = columns.into_iter().map(|c| c.into_iter().collect()).collect();
        for ci in 0..found.len() {
            for (row, vals) in rows[ci].iter_mut().zip(&named[ci]) {
                row.values = columns[ci].iter().map(|c| vals.get(c).cloned()).collect();
            }
            let mut idx: Vec<usize> = (0..rows[ci].len()).collect();
            idx.sort_by(|&a, &b| {
                let (x, y) = (&rows[ci][a], &rows[ci][b]);
                (&x.model, &x.phase, &x.instance, &x.kernel, x.member).cmp(&(
                    &y.model,
                    &y.phase,
                    &y.instance,
                    &y.kernel,
                    y.member,
                ))
            });
            rows[ci] = idx.into_iter().map(|i| rows[ci][i].clone()).collect();
        }

        // Class files.
        let classes: Vec<ClassFile> = found
            .iter()
            .enumerate()
            .map(|(ci, f)| {
                let rep_key = &keys[f.members[0].0];
                let ki = info(rep_key);
                let rep = &encoded[f.members[0].0];
                let first = (rep.n + rep.nd) as usize;
                let elems = rep.graph.nodes[first..first + rep.ne as usize]
                    .iter()
                    .filter_map(|z| match z {
                        ZNode::Elem { family } => Some(family.clone()),
                        _ => None,
                    })
                    .collect();
                let members: Vec<MemberKey> = f.members.iter().map(|(i, _)| keys[*i].clone()).collect();
                let model_set: BTreeSet<String> = rows[ci].iter().map(|r| r.model.clone()).collect();
                ClassFile {
                    schema: CLASS_SCHEMA.into(),
                    id: f.id.clone(),
                    hash: format!("{:032x}", f.hash),
                    kind: ki.kind,
                    core: ki.core.clone(),
                    tags: Tags {
                        size_independent: !rows[ci].is_empty() && !symbolic[ci],
                        low_rank: ki.low_rank,
                        call_fed: members.iter().filter(|m| info(m).call_fed).count() as u64,
                        positional: rep.positional,
                    },
                    representative: rep_key.clone(),
                    roles: ki.roles.clone(),
                    local: ki.local.clone(),
                    params: Params {
                        axes: ki.local.unit.clone(),
                        elems,
                        values: columns[ci].clone(),
                    },
                    iteration: ki.iteration.clone(),
                    kernel_nodes: rep.n,
                    identity: rep.graph.clone(),
                    members,
                    models: model_set.into_iter().collect(),
                    instances: rows[ci].len() as u64,
                    workload: rows[ci].iter().fold(0u64, |a, r| a.saturating_add(r.weighted)),
                }
            })
            .collect();

        let out = ctx.output(KERNEL_CLASSES)?;
        found
            .par_iter()
            .enumerate()
            .try_for_each(|(ci, f)| -> Result<(), PassError> {
                out.write_json(&format!("classes/{}.json", f.id), &classes[ci])?;
                out.write_json(
                    &format!("certificates/{}.json", f.id),
                    &CertificatesFile {
                        schema: CERTIFICATES_SCHEMA.into(),
                        id: f.id.clone(),
                        certificates: certs[ci].clone(),
                    },
                )?;
                let table = InstancesTable {
                    schema: INSTANCES_SCHEMA.into(),
                    id: f.id.clone(),
                    omega: info(&keys[f.members[0].0]).omega.clone(),
                    rows: rows[ci].clone(),
                };
                write_compact(out, &format!("instances/{}.json", f.id), &table)?;
                Ok(())
            })?;

        let uses: Vec<algo::sharing::Use<'_>> = classes
            .iter()
            .zip(&rows)
            .flat_map(|(c, rs)| {
                rs.iter().map(move |r| algo::sharing::Use {
                    model: &r.model,
                    component: &r.component,
                    class: &c.id,
                    weighted: r.weighted,
                    activation: !c.tags.size_independent,
                })
            })
            .collect();
        let sharing = algo::sharing::sharing(&models, &uses);
        out.write_json("sharing.json", &sharing)?;
        let summary = summarize(&classes, &comps, keys.len(), raw.len(), split_buckets);
        out.write_json("summary.json", &summary)?;
        if cfg.views {
            ctx.dir().write_view(SHARING_VIEW, &view::sharing(&sharing))?;
            ctx.dir()
                .write_view(CLASSES_VIEW, &view::classes(&summary, &classes))?;
        }
        Ok(())
    }

    fn verify(&self, ctx: &PassContext) -> Result<VerifyReport, PassError> {
        verify::verify(ctx)
    }
}

/// Compact canonical JSON (instance tables are large).
pub fn write_compact<T: serde::Serialize>(out: &cc_core::KindDir, rel: &str, v: &T) -> Result<(), PassError> {
    let p = out.path(rel)?;
    let bytes =
        cc_core::canon::to_canonical_compact(v).map_err(|e| PassError::failed(format!("{rel}: {e}")))?;
    cc_core::canon::write_atomic(&p, &bytes)?;
    Ok(())
}

fn member_key(comps: &BTreeMap<String, Vec<KInfo>>, r: &RawRow) -> MemberKey {
    MemberKey {
        component: r.component.clone(),
        kernel: comps[&r.component][r.kernel as usize].id.clone(),
        groups: r.groups.clone(),
    }
}

/// Every component's kernels, analysed (components without kernels are left out).
pub fn load_kernels(ctx: &PassContext) -> Result<BTreeMap<String, Vec<KInfo>>, PassError> {
    let kin = ctx.input(KERNELS)?;
    let nin = ctx.input(NORMALIZE)?;
    let mut ids = Vec::new();
    if let Ok(rd) = std::fs::read_dir(kin.dir().join("components")) {
        for e in rd.flatten() {
            if let Some(id) = e.file_name().to_str().and_then(|n| n.strip_suffix(".json")) {
                ids.push(id.to_string());
            }
        }
    }
    ids.sort();
    let out: Vec<(String, Vec<KInfo>)> = ids
        .par_iter()
        .map(|id| -> Result<(String, Vec<KInfo>), PassError> {
            let p: KernelPartition = kin.read_json(&format!("components/{id}.json"))?;
            let g: KernelGraphs = kin.read_postcard(&format!("kernels/{id}.postcard"))?;
            let nc: NormalizedComponent = nin.read_postcard(&format!("components/{id}.postcard"))?;
            let ks = p
                .kernels
                .iter()
                .zip(g.kernels)
                .map(|(k, kg)| {
                    algo::kernel_info(&nc.graph, k, kg.graph).map_err(|e| format!("{id}/{}: {e}", k.id))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(PassError::failed)?;
            Ok((id.clone(), ks))
        })
        .collect::<Result<_, _>>()?;
    Ok(out.into_iter().filter(|(_, ks)| !ks.is_empty()).collect())
}

/// Every `(model, phase)` with component instances, sorted.
pub fn phases(ctx: &PassContext) -> Result<Vec<(String, String)>, PassError> {
    let dir = ctx.input(COMPONENTS)?.dir().join("instances");
    let mut out = Vec::new();
    for m in std::fs::read_dir(&dir).map_err(|e| PassError::failed(format!("{}: {e}", dir.display())))? {
        let m = m.map_err(|e| PassError::failed(e.to_string()))?;
        let Some(model) = m.file_name().to_str().map(str::to_string) else {
            continue;
        };
        for p in std::fs::read_dir(m.path())
            .map_err(|e| PassError::failed(e.to_string()))?
            .flatten()
        {
            if let Some(ph) = p.file_name().to_str().and_then(|n| n.strip_suffix(".json")) {
                out.push((model.clone(), ph.to_string()));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// The symbol sizes and trip count of a phase.
pub fn scenario(
    ctx: &PassContext,
    model: &str,
    phase: &str,
) -> Result<(BTreeMap<String, i64>, u64), PassError> {
    let s: SymbolsFile = ctx
        .input(EXPORT)?
        .read_json(&format!("{model}/{phase}/symbols.json"))?;
    let mut env = BTreeMap::new();
    for (n, d) in &s.symbols {
        let v = d
            .size
            .and_then(|v| i64::try_from(v).ok())
            .ok_or_else(|| PassError::failed(format!("{model}/{phase}: symbol {n} has no size")))?;
        env.insert(n.clone(), v);
    }
    Ok((env, s.trip_count))
}

fn phase_rows(
    ctx: &PassContext,
    comps: &BTreeMap<String, Vec<KInfo>>,
    model: &str,
    phase: &str,
) -> Result<Vec<RawRow>, PassError> {
    let f: InstancesFile = ctx
        .input(COMPONENTS)?
        .read_json(&format!("instances/{model}/{phase}.json"))?;
    let (env, trip) = scenario(ctx, model, phase)?;
    let mut out = Vec::new();
    for i in &f.instances {
        let (Some(c), Some(b)) = (&i.component, &i.bindings) else {
            continue;
        };
        let Some(ks) = comps.get(c) else { continue };
        let elems = b
            .elems
            .iter()
            .map(|e| Elem::parse(e).ok_or_else(|| PassError::failed(format!("{}: element type {e}", i.id))))
            .collect::<Result<Vec<_>, _>>()?;
        let inst = Inst { b, elems, env: &env };
        for (ki, k) in ks.iter().enumerate() {
            let err = |e: String| PassError::failed(format!("{} {}: {e}", i.id, k.id));
            let groups = algo::groups(k, &inst).map_err(err)?;
            let (extents, symbolic, elems) = algo::raw_row(k, &inst).map_err(err)?;
            let values = algo::values(k, &inst).map_err(err)?;
            let workload = algo::workload(k, &extents, &values).map_err(err)?;
            out.push(RawRow {
                model: model.into(),
                phase: phase.into(),
                instance: i.id.clone(),
                component: c.clone(),
                kernel: ki as u32,
                groups,
                extents,
                symbolic,
                elems,
                values,
                workload,
                trip_count: trip,
            });
        }
    }
    Ok(out)
}

fn summarize(
    classes: &[ClassFile],
    comps: &BTreeMap<String, Vec<KInfo>>,
    members: usize,
    rows: usize,
    split_buckets: u64,
) -> ClassesSummary {
    let mut s = ClassesSummary {
        schema: SUMMARY_SCHEMA.into(),
        classes: classes.len() as u64,
        members: members as u64,
        kernels: comps.values().map(|k| k.len() as u64).sum(),
        split_by_group: 0,
        rows: rows as u64,
        kinds: BTreeMap::new(),
        cores: BTreeMap::new(),
        size_independent: 0,
        low_rank: 0,
        call_fed: 0,
        positional: 0,
        split_buckets,
        models_per_class: BTreeMap::new(),
    };
    s.split_by_group = s.members - s.kernels;
    for c in classes {
        let kind = serde_json::to_value(c.kind)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        *s.kinds.entry(kind).or_default() += 1;
        *s.cores.entry(c.core.clone()).or_default() += 1;
        s.size_independent += u64::from(c.tags.size_independent);
        s.low_rank += u64::from(c.tags.low_rank);
        s.call_fed += u64::from(c.tags.call_fed > 0);
        s.positional += u64::from(c.tags.positional);
        *s.models_per_class.entry(c.models.len() as u64).or_default() += 1;
    }
    s
}
