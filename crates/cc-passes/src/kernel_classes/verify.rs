//! Independent checks of kernel_classes (design §4.6, §5). They read the kernels, normalize, components
//! and export artifacts and the written files, and never call the encoder or the classifier: local
//! classes are recomputed here with their own union-find over the op table's rules, and every member is
//! compared with its representative through a position-free description of each node written here.
//!
//! - `complete`: every kernel of every partition is a member of one class, every member key is in
//!   exactly one class, and every component instance has one row per kernel of its component.
//! - `ids`: class ids are unique, carry their hash, and name their files.
//! - `<z>:certificate`: the stored local classes of the representative and of every member are the ones
//!   the rules give; the node, axis and element maps are bijections; and every node of the
//!   representative maps to a member node with the same description (op, identity attributes, families,
//!   classes of its results, of its contracted, reduced, batch, window and expanded axes, ω's order and
//!   axes, folded operands, group kinds) under the class maps, with every edge mapped to an equal edge
//!   (`check_isomorphism`, O(V + E)).
//! - `<z>:rows`: every row's extents, element types, trip count and workload recompute from its
//!   instance's bindings through the certificate and the phase's symbol sizes.
//! - `workload:<model>/<phase>`: Σ 2 × workload of the contraction rows, plus the GEMMs and convolutions
//!   that `normalize` removed (its ledger), equals the concrete census (`census_concrete.json`) exactly.
//! - `sharing`: the per-model sets, items, shared items, mean Jaccard and workload coverage recompute
//!   from the rows; the views equal the ones rendered from the written report and summary.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use cc_core::{Check, VerifyReport};
use cc_ir::Attr;
use cc_ir::absgraph::{Graph, PortLabel, UnionFind, check_isomorphism};
use cc_ir::ops::{DimsIn, Side, Slot, dims::relations};
use cc_pipeline::{PassContext, PassError};
use rayon::prelude::*;

use super::json::{
    CertificatesFile, ClassFile, ClassKind, ClassesSummary, InstancesTable, LocalClasses, Sharing,
};
use super::{CLASSES_VIEW, KERNEL_CLASSES, SHARING_VIEW, algo::encode, view};
use crate::components::COMPONENTS;
use crate::components::algo::graph::{AAttr, ADim, ANode, AType, Kind, instantiate, rule_attrs};
use crate::components::json::{Bindings, InstancesFile, StoredComponent};
use crate::export::EXPORT;
use crate::export::json::{CensusConcreteFile, SymbolsFile};
use crate::kernels::KERNELS;
use crate::kernels::json::{KNode, KernelGraphs, KernelPartition};
use crate::normalize::NORMALIZE;
use crate::normalize::json::{Edit, LedgerFile};

fn check(name: impl Into<String>, notes: &[String], counts: &[(&str, u64)]) -> Check {
    let mut c = Check::new(name, notes.is_empty());
    for (k, v) in counts {
        c = c.count(*k, *v);
    }
    for n in notes.iter().take(20) {
        c = c.note(n.clone());
    }
    c
}

fn fail(e: impl std::fmt::Display) -> PassError {
    PassError::failed(e.to_string())
}

// ---------------------------------------------------------------------------------------------------
// Kernel graphs as this check reads them.

fn res_types(n: &KNode) -> Vec<AType> {
    match n {
        KNode::Member { node, .. } => node.label.results.clone(),
        KNode::Input { ty, .. } => vec![ty.clone()],
        KNode::Output { .. } => Vec::new(),
    }
}

fn label(n: &KNode) -> Option<&ANode> {
    match n {
        KNode::Member { node, .. } => Some(&node.label),
        _ => None,
    }
}

fn op_name(n: &KNode) -> Option<(&str, Option<&str>)> {
    match &label(n)?.kind {
        Kind::Op { name, target } => Some((name, target.as_deref())),
        _ => None,
    }
}

fn lit_attr<'a>(n: &'a KNode, k: &str) -> Option<&'a Attr> {
    match label(n)?.attrs.get(k)? {
        AAttr::Lit(a) => Some(a),
        _ => None,
    }
}

/// Producers of every node's operand ports; commutative ports counted in edge order.
fn producers(g: &Graph<KNode>) -> Vec<BTreeMap<u32, (u32, u32)>> {
    let mut out = vec![BTreeMap::new(); g.node_count()];
    let mut comm = vec![0u32; g.node_count()];
    for (u, o) in g.out.iter().enumerate() {
        for (v, ends) in o {
            for e in ends {
                let p = match e.port {
                    PortLabel::Port(p) => p,
                    PortLabel::Commutative => {
                        comm[*v as usize] += 1;
                        comm[*v as usize] - 1
                    }
                    _ => continue,
                };
                out[*v as usize].insert(p, (u as u32, e.res));
            }
        }
    }
    out
}

fn is_swapped(n: &KNode) -> bool {
    op_name(n).is_some_and(|o| o.0 == "stablehlo.compare")
        && matches!(lit_attr(n, "comparison_direction"), Some(Attr::Enum { value, .. }) if value == "GT" || value == "GE")
}

/// The graph with the operands of `GT`/`GE` comparisons exchanged (they read as `LT`/`LE`).
fn canonical_edges(g: &Graph<KNode>) -> Graph<()> {
    let mut out: Graph<()> = Graph::default();
    for _ in &g.nodes {
        out.add_node(());
    }
    for (u, o) in g.out.iter().enumerate() {
        for (v, ends) in o {
            for e in ends {
                let mut e = *e;
                if is_swapped(&g.nodes[*v as usize]) {
                    e.port = match e.port {
                        PortLabel::Port(0) => PortLabel::Port(1),
                        PortLabel::Port(1) => PortLabel::Port(0),
                        p => p,
                    };
                }
                out.add_edge(u as u32, *v, e);
            }
        }
    }
    out
}

/// Operand axes a broadcast expands (operand literal 1, result axis not 1).
fn expands(g: &Graph<KNode>, prod: &[BTreeMap<u32, (u32, u32)>], n: usize) -> Vec<usize> {
    let x = &g.nodes[n];
    if !matches!(
        op_name(x),
        Some((
            "stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim",
            _
        ))
    ) {
        return Vec::new();
    }
    let Some(&(pn, pr)) = prod[n].get(&0) else {
        return Vec::new();
    };
    let src = res_types(&g.nodes[pn as usize]);
    let res = res_types(x);
    let (Some(s), Some(r)) = (src.get(pr as usize), res.first()) else {
        return Vec::new();
    };
    let bd = lit_attr(x, "broadcast_dimensions")
        .and_then(Attr::as_ints)
        .unwrap_or_default();
    (0..s.dims.len())
        .filter(|&a| {
            s.dims[a] == ADim::One
                && bd
                    .get(a)
                    .and_then(|&d| r.dims.get(d as usize))
                    .is_some_and(|d| *d != ADim::One)
        })
        .collect()
}

/// The kernel's local classes, recomputed (docs/passes/kernel_classes.md §4.1).
fn local_classes(g: &Graph<KNode>) -> Result<LocalClasses, String> {
    let prod = producers(g);
    let types: Vec<Vec<AType>> = g.nodes.iter().map(res_types).collect();
    let mut base: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut ebase: HashMap<(u32, u32), u32> = HashMap::new();
    for (n, ts) in types.iter().enumerate() {
        for (r, t) in ts.iter().enumerate() {
            for a in 0..t.dims.len() {
                let k = base.len() as u32;
                base.insert((n as u32, r as u32, a as u32), k);
            }
            let k = ebase.len() as u32;
            ebase.insert((n as u32, r as u32), k);
        }
    }
    let mut ud = UnionFind::new(base.len());
    let mut ue = UnionFind::new(ebase.len());
    let mut expanded: BTreeSet<u32> = BTreeSet::new();
    for (n, x) in g.nodes.iter().enumerate() {
        let Some((name, target)) = op_name(x) else {
            continue;
        };
        let spec = cc_ir::lookup(name, target).ok_or_else(|| format!("`{name}` not in the op table"))?;
        let exp = expands(g, &prod, n);
        let nops = prod[n].keys().next_back().map_or(0, |&p| p as usize + 1);
        let opnd: Vec<Option<Vec<bool>>> = (0..nops)
            .map(|p| {
                prod[n].get(&(p as u32)).and_then(|&(pn, pr)| {
                    types[pn as usize]
                        .get(pr as usize)
                        .map(|t| (0..t.dims.len()).map(|a| p == 0 && exp.contains(&a)).collect())
                })
            })
            .collect();
        if let Some(&(pn, pr)) = prod[n].get(&0) {
            for &a in &exp {
                expanded.insert(base[&(pn, pr, a as u32)]);
            }
        }
        let res: Vec<Vec<bool>> = types[n].iter().map(|t| vec![false; t.dims.len()]).collect();
        let attrs = rule_attrs(label(x).ok_or("member without a label")?);
        let fam = |i: usize| {
            prod[n]
                .get(&(i as u32))
                .and_then(|&(pn, pr)| types[pn as usize].get(pr as usize).map(|t| t.family.clone()))
        };
        let rel = relations(
            spec,
            &DimsIn {
                operands: &opnd,
                results: &res,
                attrs: &attrs,
            },
            |a, b| fam(a).is_some() && fam(a) == fam(b),
        );
        let side = |s: Side| match s {
            Side::Operand(i) => prod[n].get(&i).copied(),
            Side::Result(i) => Some((n as u32, i)),
        };
        let slot = |s: Slot| side(s.side).and_then(|(a, b)| base.get(&(a, b, s.axis)).copied());
        for (a, b) in rel.dims {
            if let (Some(x), Some(y)) = (slot(a), slot(b)) {
                ud.union(x, y);
            }
        }
        for (a, b) in rel.elems {
            if let (Some(x), Some(y)) = (side(a), side(b))
                && let (Some(&p), Some(&q)) = (ebase.get(&x), ebase.get(&y))
            {
                ue.union(p, q);
            }
        }
    }
    let mut out = LocalClasses::default();
    let (mut dn, mut en): (HashMap<u32, u32>, HashMap<u32, u32>) = Default::default();
    for (n, ts) in types.iter().enumerate() {
        let (mut rd, mut re) = (Vec::new(), Vec::new());
        for (r, t) in ts.iter().enumerate() {
            let mut axes = Vec::new();
            for a in 0..t.dims.len() {
                let s = base[&(n as u32, r as u32, a as u32)];
                let root = ud.find(s);
                let k = dn.len() as u32;
                let c = *dn.entry(root).or_insert(k);
                if c as usize == out.unit.len() {
                    out.unit.push(false);
                }
                out.unit[c as usize] |= expanded.contains(&s);
                axes.push(c);
            }
            rd.push(axes);
            let root = ue.find(ebase[&(n as u32, r as u32)]);
            let k = en.len() as u32;
            re.push(*en.entry(root).or_insert(k));
        }
        out.dims.push(rd);
        out.elems.push(re);
    }
    out.nd = dn.len() as u32;
    out.ne = en.len() as u32;
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------
// The position-free description of a node.

/// A description whose class numbers can be renamed: `Fixed` parts must be equal, class lists are
/// compared after mapping (sorted ones as multisets, ordered ones in order).
#[derive(Debug, Clone, PartialEq)]
struct Desc {
    fixed: String,
    /// `(name, classes, ordered)`.
    dims: Vec<(String, Vec<u32>, bool)>,
    elems: Vec<u32>,
}

fn describe(
    g: &Graph<KNode>,
    prod: &[BTreeMap<u32, (u32, u32)>],
    l: &LocalClasses,
    n: usize,
    groups: &BTreeMap<String, super::json::Group>,
) -> Result<Desc, String> {
    let x = &g.nodes[n];
    let mut dims: Vec<(String, Vec<u32>, bool)> = Vec::new();
    for (r, axes) in l.dims[n].iter().enumerate() {
        dims.push((format!("res{r}"), axes.clone(), false));
    }
    let oc = |p: u32, a: i64| -> Result<u32, String> {
        let &(pn, pr) = prod[n].get(&p).ok_or_else(|| format!("operand {p} missing"))?;
        l.dims[pn as usize]
            .get(pr as usize)
            .and_then(|v| v.get(a as usize))
            .copied()
            .ok_or_else(|| format!("operand {p} axis {a}"))
    };
    let rc = |a: i64| -> Result<u32, String> {
        l.dims[n]
            .first()
            .and_then(|v| v.get(a as usize))
            .copied()
            .ok_or_else(|| format!("result axis {a}"))
    };
    let fixed = match x {
        KNode::Input { ty, .. } => format!("input {:?}", ty.family),
        KNode::Output { .. } => "output".into(),
        KNode::Member { node, .. } => {
            let swapped = is_swapped(x);
            let name = op_name(x).map(|o| o.0).unwrap_or_default();
            let pos = encode::positional(name);
            let mut attrs = Vec::new();
            for (k, a) in &node.label.attrs {
                let AAttr::Lit(v) = a else { continue };
                if encode::DROPPED.contains(&k.as_str()) || pos.contains(&k.as_str()) {
                    continue;
                }
                let s = match v {
                    Attr::Enum { value, .. } if swapped && k == "comparison_direction" => {
                        if value == "GT" {
                            "LT".to_string()
                        } else {
                            "LE".to_string()
                        }
                    }
                    v => format!("{v:?}"),
                };
                attrs.push(format!("{k}={s}"));
            }
            let splats: Vec<String> = node
                .splats
                .iter()
                .map(|s| {
                    let p = match s.port {
                        PortLabel::Port(0) if swapped => PortLabel::Port(1),
                        PortLabel::Port(1) if swapped => PortLabel::Port(0),
                        p => p,
                    };
                    format!("{p:?}:{:?}", s.ty.family)
                })
                .collect();
            let fams: Vec<String> = node
                .label
                .results
                .iter()
                .map(|t| format!("{:?}", t.family))
                .collect();
            if let Some(om) = &node.omega {
                let port = match op_name(x) {
                    Some(("stablehlo.convolution" | "stablehlo.dynamic_conv", _)) => 1,
                    Some(("stablehlo.scatter", _)) => node.label.results.len() as u32 + 1,
                    _ => 0,
                };
                let cs = om
                    .axes
                    .iter()
                    .map(|&a| oc(port, i64::from(a)))
                    .collect::<Result<_, _>>()?;
                dims.push(("omega".into(), cs, true));
            }
            match op_name(x) {
                Some(("stablehlo.dot_general", _)) => {
                    if let Some(Attr::DotDims(d)) = lit_attr(x, "dot_dimension_numbers") {
                        dims.push((
                            "batch".into(),
                            d.lhs_batching
                                .iter()
                                .map(|&a| oc(0, a))
                                .collect::<Result<_, _>>()?,
                            false,
                        ));
                        dims.push((
                            "contract".into(),
                            d.lhs_contracting
                                .iter()
                                .map(|&a| oc(0, a))
                                .collect::<Result<_, _>>()?,
                            false,
                        ));
                    }
                }
                Some(("stablehlo.convolution" | "stablehlo.dynamic_conv", _)) => {
                    if let Some(Attr::ConvDims(d)) = lit_attr(x, "dimension_numbers") {
                        dims.push((
                            "conv".into(),
                            vec![
                                oc(0, d.input_batch)?,
                                oc(0, d.input_feature)?,
                                oc(1, d.kernel_input_feature)?,
                                oc(1, d.kernel_output_feature)?,
                                rc(d.output_batch)?,
                                rc(d.output_feature)?,
                            ],
                            true,
                        ));
                        for k in 0..d.input_spatial.len() {
                            dims.push((
                                "window".into(),
                                vec![
                                    oc(0, d.input_spatial[k])?,
                                    oc(1, d.kernel_spatial[k])?,
                                    rc(d.output_spatial[k])?,
                                ],
                                true,
                            ));
                        }
                    }
                }
                Some(("stablehlo.reduce", _)) => {
                    let ds = lit_attr(x, "dimensions")
                        .and_then(Attr::as_ints)
                        .unwrap_or_default();
                    dims.push((
                        "reduced".into(),
                        ds.iter().map(|&a| oc(0, a)).collect::<Result<_, _>>()?,
                        false,
                    ));
                }
                Some(("stablehlo.reduce_window", _))
                | Some(("stablehlo.custom_call", Some("stablehlo.dynamic_reduce_window"))) => {
                    let r = prod[n]
                        .get(&0)
                        .map_or(0, |&(pn, pr)| l.dims[pn as usize][pr as usize].len());
                    for a in 0..r as i64 {
                        dims.push(("window".into(), vec![oc(0, a)?, rc(a)?], true));
                    }
                }
                Some(("stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim", _)) => {
                    let exp = expands(g, prod, n);
                    let bd = lit_attr(x, "broadcast_dimensions")
                        .and_then(Attr::as_ints)
                        .unwrap_or_default();
                    let mapped: BTreeSet<i64> = bd
                        .iter()
                        .enumerate()
                        .filter(|(a, _)| !exp.contains(a))
                        .map(|(_, &d)| d)
                        .collect();
                    let rank = l.dims[n].first().map_or(0, Vec::len) as i64;
                    dims.push((
                        "expand".into(),
                        (0..rank)
                            .filter(|j| !mapped.contains(j))
                            .map(rc)
                            .collect::<Result<_, _>>()?,
                        false,
                    ));
                    for k in ["known_expanding_dimensions", "known_nonexpanding_dimensions"] {
                        let v = lit_attr(x, k).and_then(Attr::as_ints).unwrap_or_default();
                        dims.push((
                            k.into(),
                            v.iter().map(|&a| oc(0, a)).collect::<Result<_, _>>()?,
                            false,
                        ));
                    }
                }
                Some(("stablehlo.scatter", _)) => {
                    if let Some(Attr::ScatterDims(d)) = lit_attr(x, "scatter_dimension_numbers") {
                        let k = node.label.results.len() as u32;
                        for (name, port, list) in [
                            ("update_window", k + 1, &d.update_window_dims),
                            ("inserted_window", 0, &d.inserted_window_dims),
                            ("input_batching", 0, &d.input_batching_dims),
                            ("indices_batching", k, &d.scatter_indices_batching_dims),
                            ("to_operand", 0, &d.scattered_dims_to_operand_dims),
                        ] {
                            dims.push((
                                name.into(),
                                list.iter().map(|&a| oc(port, a)).collect::<Result<_, _>>()?,
                                true,
                            ));
                        }
                        let rank = prod[n]
                            .get(&k)
                            .map_or(0, |&(pn, pr)| l.dims[pn as usize][pr as usize].len());
                        if (d.index_vector_dim as usize) < rank {
                            dims.push(("index_vector".into(), vec![oc(k, d.index_vector_dim)?], true));
                        }
                    }
                }
                _ => {}
            }
            format!(
                "{:?} attrs {attrs:?} results {fams:?} order {:?} splats {splats:?} groups {groups:?}",
                node.label.kind,
                node.omega.as_ref().map(|o| &o.order)
            )
        }
    };
    let elems = l.elems[n].clone();
    Ok(Desc { fixed, dims, elems })
}

fn mapped(d: &Desc, axes: &[u32], elems: &[u32]) -> Desc {
    Desc {
        fixed: d.fixed.clone(),
        dims: d
            .dims
            .iter()
            .map(|(k, cs, ordered)| {
                let cs = cs
                    .iter()
                    .map(|&c| axes.get(c as usize).copied().unwrap_or(u32::MAX));
                (k.clone(), cs.collect(), *ordered)
            })
            .collect(),
        elems: d
            .elems
            .iter()
            .map(|&e| elems.get(e as usize).copied().unwrap_or(u32::MAX))
            .collect(),
    }
}

fn normalized(mut d: Desc) -> Desc {
    for (_, cs, ordered) in &mut d.dims {
        if !*ordered {
            cs.sort_unstable();
        }
    }
    d.dims.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    d
}

fn bijection(m: &[u32], n: usize) -> bool {
    let mut seen = vec![false; n];
    m.len() == n
        && m.iter()
            .all(|&x| (x as usize) < n && !std::mem::replace(&mut seen[x as usize], true))
}

// ---------------------------------------------------------------------------------------------------

struct Loaded {
    graphs: BTreeMap<String, Vec<(String, Graph<KNode>)>>,
    classes: Vec<ClassFile>,
    certs: Vec<CertificatesFile>,
    tables: Vec<InstancesTable>,
}

fn load(ctx: &PassContext) -> Result<Loaded, PassError> {
    let kin = ctx.input(KERNELS)?;
    let out = ctx.output(KERNEL_CLASSES)?;
    let mut ids = Vec::new();
    for e in std::fs::read_dir(kin.dir().join("components"))
        .map_err(fail)?
        .flatten()
    {
        if let Some(id) = e.file_name().to_str().and_then(|n| n.strip_suffix(".json")) {
            ids.push(id.to_string());
        }
    }
    ids.sort();
    let graphs: BTreeMap<String, Vec<(String, Graph<KNode>)>> = ids
        .par_iter()
        .map(|id| -> Result<_, PassError> {
            let p: KernelPartition = kin.read_json(&format!("components/{id}.json"))?;
            let g: KernelGraphs = kin.read_postcard(&format!("kernels/{id}.postcard"))?;
            Ok((
                id.clone(),
                p.kernels
                    .iter()
                    .map(|k| k.id.clone())
                    .zip(g.kernels.into_iter().map(|k| k.graph))
                    .collect(),
            ))
        })
        .collect::<Result<_, _>>()?;
    let mut zs = Vec::new();
    if let Ok(rd) = std::fs::read_dir(out.dir().join("classes")) {
        for e in rd.flatten() {
            if let Some(id) = e.file_name().to_str().and_then(|n| n.strip_suffix(".json")) {
                zs.push(id.to_string());
            }
        }
    }
    zs.sort();
    let files: Vec<(ClassFile, CertificatesFile, InstancesTable)> = zs
        .par_iter()
        .map(|z| -> Result<_, PassError> {
            Ok((
                out.read_json(&format!("classes/{z}.json"))?,
                out.read_json(&format!("certificates/{z}.json"))?,
                out.read_json(&format!("instances/{z}.json"))?,
            ))
        })
        .collect::<Result<_, _>>()?;
    let mut l = Loaded {
        graphs,
        classes: Vec::new(),
        certs: Vec::new(),
        tables: Vec::new(),
    };
    for (c, k, t) in files {
        l.classes.push(c);
        l.certs.push(k);
        l.tables.push(t);
    }
    Ok(l)
}

fn graph_of<'a>(l: &'a Loaded, component: &str, kernel: &str) -> Option<&'a Graph<KNode>> {
    l.graphs
        .get(component)?
        .iter()
        .find(|(k, _)| k == kernel)
        .map(|(_, g)| g)
}

/// The certificate check of one class.
fn check_class(l: &Loaded, ci: usize) -> Vec<String> {
    let c = &l.classes[ci];
    let mut notes = Vec::new();
    let rep = &c.representative;
    let Some(rg) = graph_of(l, &rep.component, &rep.kernel) else {
        return vec![format!(
            "representative {}/{} not found",
            rep.component, rep.kernel
        )];
    };
    match local_classes(rg) {
        Ok(x) if x == c.local => {}
        Ok(_) => return vec!["the representative's stored local classes differ from the rules'".into()],
        Err(e) => return vec![format!("representative: {e}")],
    }
    let rprod = producers(rg);
    let rcan = canonical_edges(rg);
    let rdesc: Result<Vec<Desc>, String> = (0..rg.node_count())
        .map(|i| {
            describe(
                rg,
                &rprod,
                &c.local,
                i,
                rep.groups.get(&(i as u32)).unwrap_or(&Default::default()),
            )
        })
        .collect();
    let rdesc = match rdesc {
        Ok(d) => d,
        Err(e) => return vec![format!("representative: {e}")],
    };
    if c.members.len() != l.certs[ci].certificates.len() {
        notes.push("members and certificates differ in number".into());
    }
    for (m, cert) in c.members.iter().zip(&l.certs[ci].certificates) {
        let who = format!("{}/{}", m.component, m.kernel);
        if &cert.member != m {
            notes.push(format!("{who}: certificate for another member"));
            continue;
        }
        let Some(mg) = graph_of(l, &m.component, &m.kernel) else {
            notes.push(format!("{who}: kernel not found"));
            continue;
        };
        let ml = match local_classes(mg) {
            Ok(x) if x == cert.local => x,
            Ok(_) => {
                notes.push(format!("{who}: stored local classes differ from the rules'"));
                continue;
            }
            Err(e) => {
                notes.push(format!("{who}: {e}"));
                continue;
            }
        };
        if !bijection(&cert.axes, ml.nd as usize) || ml.nd != c.local.nd {
            notes.push(format!("{who}: the axis map is not a bijection"));
            continue;
        }
        if !bijection(&cert.elems, ml.ne as usize) || ml.ne != c.local.ne {
            notes.push(format!("{who}: the element map is not a bijection"));
            continue;
        }
        let mprod = producers(mg);
        let mdesc: Result<Vec<Desc>, String> = (0..mg.node_count())
            .map(|i| {
                describe(
                    mg,
                    &mprod,
                    &ml,
                    i,
                    m.groups.get(&(i as u32)).unwrap_or(&Default::default()),
                )
            })
            .collect();
        let mdesc = match mdesc {
            Ok(d) => d,
            Err(e) => {
                notes.push(format!("{who}: {e}"));
                continue;
            }
        };
        let map: Vec<usize> = cert.nodes.iter().map(|&x| x as usize).collect();
        let r = check_isomorphism(&rcan, &canonical_edges(mg), &map, |i, j| {
            let a = normalized(mapped(&rdesc[i], &cert.axes, &cert.elems));
            let b = normalized(mdesc[j].clone());
            if a == b {
                Ok(())
            } else {
                Err(format!("{a:?} vs {b:?}"))
            }
        });
        if let Err(e) = r {
            notes.push(format!("{who}: {}", e.chars().take(400).collect::<String>()));
        }
    }
    notes
}

/// The rows of one phase, recomputed.
fn check_phase_rows(
    ctx: &PassContext,
    l: &Loaded,
    model: &str,
    phase: &str,
    rows: &[(usize, usize)],
) -> Result<Vec<(usize, String)>, PassError> {
    let f: InstancesFile = ctx
        .input(COMPONENTS)?
        .read_json(&format!("instances/{model}/{phase}.json"))?;
    let s: SymbolsFile = ctx
        .input(EXPORT)?
        .read_json(&format!("{model}/{phase}/symbols.json"))?;
    let env: BTreeMap<String, i64> = s
        .symbols
        .iter()
        .filter_map(|(k, d)| Some((k.clone(), i64::try_from(d.size?).ok()?)))
        .collect();
    let by_id: HashMap<&str, &Bindings> = f
        .instances
        .iter()
        .filter_map(|i| Some((i.id.as_str(), i.bindings.as_ref()?)))
        .collect();
    let mut notes: Vec<(usize, String)> = Vec::new();
    for &(ci, ri) in rows {
        let mut note = |s: String| notes.push((ci, s));
        let c = &l.classes[ci];
        let r = &l.tables[ci].rows[ri];
        let who = format!("{} {}", r.instance, r.kernel);
        let Some(b) = by_id.get(r.instance.as_str()) else {
            note(format!("{who}: no such instance"));
            continue;
        };
        let Some(cert) = l.certs[ci].certificates.get(r.member as usize) else {
            note(format!("{who}: no member {}", r.member));
            continue;
        };
        let Some(mg) = graph_of(l, &cert.member.component, &cert.member.kernel) else {
            note(format!("{who}: kernel not found"));
            continue;
        };
        let mut slot: HashMap<u32, (usize, usize, usize)> = HashMap::new();
        let mut eslot: HashMap<u32, (usize, usize)> = HashMap::new();
        for (n, rs) in cert.local.dims.iter().enumerate() {
            for (ri2, axes) in rs.iter().enumerate() {
                for (a, &k) in axes.iter().enumerate() {
                    slot.entry(k).or_insert((n, ri2, a));
                }
                eslot.entry(cert.local.elems[n][ri2]).or_insert((n, ri2));
            }
        }
        let extent = |k: u32| -> Option<u64> {
            let &(n, ri2, a) = slot.get(&k)?;
            match res_types(&mg.nodes[n]).get(ri2)?.dims.get(a)? {
                ADim::One => Some(1),
                ADim::Var(v) => match b.dims.get(*v as usize)? {
                    cc_ir::Dim::Const(x) => Some(*x),
                    cc_ir::Dim::Sym(e) => u64::try_from(e.eval(&env).ok()?).ok(),
                },
            }
        };
        let ext: Option<Vec<u64>> = cert.axes.iter().map(|&k| extent(k)).collect();
        if ext.as_deref() != Some(r.extents.as_slice()) {
            note(format!("{who}: extents {:?}, recomputed {ext:?}", r.extents));
            continue;
        }
        let elems: Option<Vec<String>> = cert
            .elems
            .iter()
            .map(|&e| {
                let &(n, ri2) = eslot.get(&e)?;
                b.elems
                    .get(res_types(&mg.nodes[n]).get(ri2)?.elem as usize)
                    .cloned()
            })
            .collect();
        if elems.as_deref() != Some(r.elems.as_slice()) {
            note(format!(
                "{who}: element types {:?}, recomputed {elems:?}",
                r.elems
            ));
        }
        let mut w: Option<u64> = Some(1);
        for &k in &c.iteration {
            w = w.and_then(|w| w.checked_mul(*r.extents.get(k as usize)?));
        }
        for (col, v) in c.params.values.iter().zip(&r.values) {
            if col.ends_with(".window_dimensions")
                && c.kind == ClassKind::Reduction
                && let Some(v) = v
            {
                for s in v.as_ints().unwrap_or_default() {
                    w = w.and_then(|w| w.checked_mul(u64::try_from(s).ok()?));
                }
            }
        }
        if w != Some(r.workload)
            || r.trip_count != s.trip_count
            || r.weighted != r.workload.saturating_mul(r.trip_count)
        {
            note(format!(
                "{who}: workload {} × {} = {}, recomputed {w:?} × {}",
                r.workload, r.trip_count, r.weighted, s.trip_count
            ));
        }
    }
    Ok(notes)
}

/// MACs of the GEMMs and convolutions `normalize` removed from a component, per instance bindings.
fn removed_macs(
    comp: &StoredComponent,
    ledger: &LedgerFile,
    b: &Bindings,
    env: &BTreeMap<String, i64>,
) -> Result<u64, String> {
    let elems: Vec<cc_ir::Elem> = b.elems.iter().filter_map(|e| cc_ir::Elem::parse(e)).collect();
    let mut total = 0u64;
    for step in &ledger.steps {
        for e in &step.edits {
            let Edit::Remove { node } = e else { continue };
            let Some(n) = comp.graph.nodes.get(*node as usize) else {
                continue;
            };
            let Kind::Op { name, .. } = &n.kind else { continue };
            if !matches!(
                name.as_str(),
                "stablehlo.dot_general" | "stablehlo.convolution" | "stablehlo.dynamic_conv"
            ) {
                continue;
            }
            let c = instantiate(
                n,
                &b.dims,
                &elems,
                b.params.get(*node as usize).map(Vec::as_slice).unwrap_or(&[]),
            )?;
            let ev = |d: &cc_ir::Dim| -> Result<u64, String> {
                match d {
                    cc_ir::Dim::Const(x) => Ok(*x),
                    cc_ir::Dim::Sym(s) => {
                        u64::try_from(s.eval(env).map_err(|e| format!("{e:?}"))?).map_err(|e| e.to_string())
                    }
                }
            };
            let mut m = 1u64;
            for d in &c.results[0].shape {
                m = m.saturating_mul(ev(d)?);
            }
            // The contracted extents, from the operands' types (their producers in the component).
            let ins: BTreeMap<u32, (u32, u32)> = comp
                .graph
                .out
                .iter()
                .enumerate()
                .flat_map(|(u, o)| {
                    o.iter().filter(|(v, _)| v == node).flat_map(move |(_, ends)| {
                        ends.iter().filter_map(move |x| match x.port {
                            PortLabel::Port(p) => Some((p, (u as u32, x.res))),
                            _ => None,
                        })
                    })
                })
                .collect();
            let dims_of = |p: u32| -> Result<Vec<u64>, String> {
                let &(pn, pr) = ins.get(&p).ok_or("operand missing")?;
                let pnode = &comp.graph.nodes[pn as usize];
                let pc = instantiate(
                    pnode,
                    &b.dims,
                    &elems,
                    b.params.get(pn as usize).map(Vec::as_slice).unwrap_or(&[]),
                )?;
                pc.results
                    .get(pr as usize)
                    .ok_or("result missing")?
                    .shape
                    .iter()
                    .map(ev)
                    .collect()
            };
            match c
                .attrs
                .get("dot_dimension_numbers")
                .or_else(|| c.attrs.get("dimension_numbers"))
            {
                Some(Attr::DotDims(d)) => {
                    let lhs = dims_of(0)?;
                    for &a in &d.lhs_contracting {
                        m = m.saturating_mul(lhs[a as usize]);
                    }
                }
                Some(Attr::ConvDims(d)) => {
                    let ker = dims_of(1)?;
                    for (a, &x) in ker.iter().enumerate() {
                        if a as i64 != d.kernel_output_feature {
                            m = m.saturating_mul(x);
                        }
                    }
                }
                _ => {}
            }
            total = total.saturating_add(m);
        }
    }
    Ok(total)
}

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let mut rep = VerifyReport::new(ctx.name());
    let l = load(ctx)?;

    // complete + ids
    let mut notes = Vec::new();
    let mut seen: BTreeMap<(String, String), u64> = BTreeMap::new();
    let mut keys = BTreeSet::new();
    for c in &l.classes {
        for m in &c.members {
            *seen.entry((m.component.clone(), m.kernel.clone())).or_default() += 1;
            if !keys.insert(m.clone()) {
                notes.push(format!("{}/{} is a member twice", m.component, m.kernel));
            }
        }
    }
    let mut kernels = 0u64;
    for (comp, ks) in &l.graphs {
        for (k, _) in ks {
            kernels += 1;
            if !seen.contains_key(&(comp.clone(), k.clone())) {
                notes.push(format!("{comp}/{k} is in no class"));
            }
        }
    }
    let phases = super::phases(ctx)?;
    let mut expected = 0u64;
    for (m, p) in &phases {
        let f: InstancesFile = ctx
            .input(COMPONENTS)?
            .read_json(&format!("instances/{m}/{p}.json"))?;
        for i in &f.instances {
            if let Some(c) = &i.component
                && i.bindings.is_some()
            {
                expected += l.graphs.get(c).map_or(0, |k| k.len() as u64);
            }
        }
    }
    let rows: u64 = l.tables.iter().map(|t| t.rows.len() as u64).sum();
    let mut uniq = BTreeSet::new();
    for t in &l.tables {
        for r in &t.rows {
            if !uniq.insert((r.instance.as_str(), r.kernel.as_str())) {
                notes.push(format!("{} {}: two rows", r.instance, r.kernel));
            }
        }
    }
    if rows != expected {
        notes.push(format!("{rows} rows, {expected} kernel instances"));
    }
    rep.push(check(
        "complete",
        &notes,
        &[
            ("classes", l.classes.len() as u64),
            ("kernels", kernels),
            ("rows", rows),
        ],
    ));

    let mut notes = Vec::new();
    let mut ids = BTreeSet::new();
    for (c, t) in l.classes.iter().zip(&l.tables) {
        if !ids.insert(c.id.clone()) || t.id != c.id {
            notes.push(format!("{}: duplicate or misfiled", c.id));
        }
        if !c
            .id
            .starts_with(&format!("z{}", c.hash.get(..16).unwrap_or_default()))
        {
            notes.push(format!("{}: id does not carry hash {}", c.id, c.hash));
        }
    }
    rep.push(check("ids", &notes, &[]));

    // certificates
    let per: Vec<Vec<String>> = (0..l.classes.len())
        .into_par_iter()
        .map(|ci| check_class(&l, ci))
        .collect();
    for (c, notes) in l.classes.iter().zip(per) {
        rep.push(check(
            format!("{}:certificate", c.id),
            &notes,
            &[("members", c.members.len() as u64)],
        ));
    }

    // rows, per phase
    let mut by_phase: BTreeMap<(String, String), Vec<(usize, usize)>> = BTreeMap::new();
    for (ci, t) in l.tables.iter().enumerate() {
        for (ri, r) in t.rows.iter().enumerate() {
            by_phase
                .entry((r.model.clone(), r.phase.clone()))
                .or_default()
                .push((ci, ri));
        }
    }
    let row_notes: Vec<Vec<(usize, String)>> = by_phase
        .par_iter()
        .map(|((m, p), rs)| check_phase_rows(ctx, &l, m, p, rs))
        .collect::<Result<_, _>>()?;
    let mut per_class: Vec<Vec<String>> = vec![Vec::new(); l.classes.len()];
    for (ci, n) in row_notes.into_iter().flatten() {
        per_class[ci].push(n);
    }
    for (c, notes) in l.classes.iter().zip(&per_class) {
        rep.push(check(format!("{}:rows", c.id), notes, &[("rows", c.instances)]));
    }

    // workload vs the concrete census
    let contraction: Vec<bool> = l
        .classes
        .iter()
        .map(|c| c.kind == ClassKind::Contraction)
        .collect();
    let mut macs: BTreeMap<(String, String), u64> = BTreeMap::new();
    for (ci, t) in l.tables.iter().enumerate() {
        if contraction[ci] {
            for r in &t.rows {
                let e = macs.entry((r.model.clone(), r.phase.clone())).or_default();
                *e = e.saturating_add(r.workload);
            }
        }
    }
    let nin = ctx.input(NORMALIZE)?;
    let cin = ctx.input(COMPONENTS)?;
    let census: Vec<Check> = phases
        .par_iter()
        .map(|(m, p)| -> Result<Check, PassError> {
            let cc: CensusConcreteFile = ctx
                .input(EXPORT)?
                .read_json(&format!("{m}/{p}/census_concrete.json"))?;
            let f: InstancesFile = cin.read_json(&format!("instances/{m}/{p}.json"))?;
            let s: SymbolsFile = ctx.input(EXPORT)?.read_json(&format!("{m}/{p}/symbols.json"))?;
            let env: BTreeMap<String, i64> = s
                .symbols
                .iter()
                .filter_map(|(k, d)| Some((k.clone(), i64::try_from(d.size?).ok()?)))
                .collect();
            let mut removed = 0u64;
            let mut cache: BTreeMap<String, Option<(StoredComponent, LedgerFile)>> = BTreeMap::new();
            for i in &f.instances {
                let (Some(c), Some(b)) = (&i.component, &i.bindings) else {
                    continue;
                };
                if !cache.contains_key(c) {
                    let ledger: LedgerFile = nin.read_json(&format!("ledger/{c}.json"))?;
                    let any = ledger
                        .steps
                        .iter()
                        .any(|s| s.edits.iter().any(|e| matches!(e, Edit::Remove { .. })));
                    let v = if any {
                        Some((cin.read_postcard(&format!("components/{c}.postcard"))?, ledger))
                    } else {
                        None
                    };
                    cache.insert(c.clone(), v);
                }
                if let Some(Some((comp, ledger))) = cache.get(c) {
                    removed = removed.saturating_add(
                        removed_macs(comp, ledger, b, &env).map_err(|e| fail(format!("{}: {e}", i.id)))?,
                    );
                }
            }
            let ours = macs.get(&(m.clone(), p.clone())).copied().unwrap_or(0);
            let total = ours.saturating_add(removed).saturating_mul(2);
            let notes = if cc.flops == Some(total) {
                Vec::new()
            } else {
                vec![format!(
                    "2 × ({ours} + {removed} removed) = {total}, census {:?}",
                    cc.flops
                )]
            };
            Ok(check(
                format!("workload:{m}/{p}"),
                &notes,
                &[("flops", total), ("removed_macs", removed)],
            ))
        })
        .collect::<Result<_, _>>()?;
    for c in census {
        rep.push(c);
    }

    // sharing, recomputed here
    let out = ctx.output(KERNEL_CLASSES)?;
    let sh: Sharing = out.read_json("sharing.json")?;
    let summary: ClassesSummary = out.read_json("summary.json")?;
    let mut notes = Vec::new();
    for (variant, keep_all) in [(&sh.all, true), (&sh.activation, false)] {
        for (lvl, by_class) in [(&variant.components, false), (&variant.kernels, true)] {
            let mut sets: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
            let mut users: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
            let mut work: BTreeMap<&str, Vec<(&str, u64)>> = BTreeMap::new();
            for (c, t) in l.classes.iter().zip(&l.tables) {
                if !keep_all && c.tags.size_independent {
                    continue;
                }
                for r in &t.rows {
                    let item = if by_class {
                        c.id.as_str()
                    } else {
                        r.component.as_str()
                    };
                    sets.entry(&r.model).or_default().insert(item);
                    users.entry(item).or_default().insert(&r.model);
                    work.entry(&r.model).or_default().push((item, r.weighted));
                }
            }
            let shared: BTreeSet<&str> = users
                .iter()
                .filter(|(_, u)| u.len() > 1)
                .map(|(i, _)| *i)
                .collect();
            let (mut num, mut den) = (0u128, 0u128);
            for ws in work.values() {
                for &(i, w) in ws {
                    den += u128::from(w);
                    if shared.contains(i) {
                        num += u128::from(w);
                    }
                }
            }
            let cov = if den == 0 { 0.0 } else { num as f64 / den as f64 };
            let ms = &sh.models;
            let (mut js, mut np) = (0.0, 0u64);
            for i in 0..ms.len() {
                for j in i + 1..ms.len() {
                    let e = BTreeSet::new();
                    let (a, b) = (
                        sets.get(ms[i].as_str()).unwrap_or(&e),
                        sets.get(ms[j].as_str()).unwrap_or(&e),
                    );
                    let u = a.union(b).count();
                    js += if u == 0 {
                        0.0
                    } else {
                        a.intersection(b).count() as f64 / u as f64
                    };
                    np += 1;
                }
            }
            let mj = if np == 0 { 0.0 } else { js / np as f64 };
            let mut covs = Vec::new();
            for m in ms {
                let (mut a, mut b) = (0u128, 0u128);
                for &(i, w) in work.get(m.as_str()).map(Vec::as_slice).unwrap_or(&[]) {
                    b += u128::from(w);
                    if shared.contains(i) {
                        a += u128::from(w);
                    }
                }
                covs.push(if b == 0 { 0.0 } else { a as f64 / b as f64 });
            }
            let cm = if covs.is_empty() {
                0.0
            } else {
                covs.iter().sum::<f64>() / covs.len() as f64
            };
            let per_model: BTreeMap<String, u64> = ms
                .iter()
                .map(|m| (m.clone(), sets.get(m.as_str()).map_or(0, |s| s.len() as u64)))
                .collect();
            if lvl.items != users.len() as u64
                || lvl.shared != shared.len() as u64
                || lvl.per_model != per_model
                || (lvl.mean_jaccard - mj).abs() > 1e-9
                || (lvl.coverage_total - cov).abs() > 1e-9
                || (lvl.coverage_mean - cm).abs() > 1e-9
            {
                notes.push(format!(
                    "{} {}: items {} shared {} jaccard {} coverage {}; recomputed {} {} {mj} {cov}",
                    if keep_all { "all" } else { "activation" },
                    if by_class { "kernels" } else { "components" },
                    lvl.items,
                    lvl.shared,
                    lvl.mean_jaccard,
                    lvl.coverage_total,
                    users.len(),
                    shared.len()
                ));
            }
        }
    }
    let views = [
        (SHARING_VIEW, view::sharing(&sh)),
        (CLASSES_VIEW, view::classes(&summary, &l.classes)),
    ];
    for (name, want) in views {
        let p = ctx.dir().views_dir().join(name);
        if p.exists() && std::fs::read_to_string(&p).ok().as_deref() != Some(want.as_str()) {
            notes.push(format!("views/{name} differs from the report"));
        }
    }
    rep.push(check("sharing", &notes, &[]));
    Ok(rep)
}
