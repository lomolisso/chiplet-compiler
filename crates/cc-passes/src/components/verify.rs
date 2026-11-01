//! Independent checks of the components (design §4.3, §5). They read the imported modules and the
//! written artifacts; they rebuild every instance's concrete graph from the module with their own code
//! and check the stored certificate edge by edge, without classes or VF2. Per phase
//! (`<model>:<phase>:<check>`):
//!
//! - `partition`: every op of the composite expansion (each op at each executed call path) is owned by
//!   exactly one instance, or is the entry's terminator; the count equals `Module::expanded_multiset`.
//! - `recoverability`: for every instance, the component instantiated with its bindings is its concrete
//!   graph under the certificate: a bijection of nodes, equal labels (op, attributes, types, child
//!   component, size value) and every edge with an equal label, with equal edge counts.
//!
//! Once: `complete` (every imported phase has its files), `classes` (each component's classes and
//! interface recompute from its graph), `wl_hash` (each component's WL hash recomputes), `dedup` (every
//! instance's component exists and lists it, and every listed instance points back) and `sharing_table`
//! (the view equals the table recomputed from the summaries). All members of a component are checked
//! against the one stored graph, so the map between any two members (one certificate composed with the
//! inverse of the other) is an isomorphism.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use cc_core::{Check, VerifyReport};
use cc_ir::absgraph::{EdgeEnd, EdgeLabel, Graph, PortLabel, hash_bytes, wl};
use cc_ir::{Def, FuncId, Module, OpId, Owner, SymCtx, ValueId};
use cc_pipeline::{PassContext, PassError};
use rayon::prelude::*;

use super::algo::classes::{assign, dim_eq, iface};
use super::algo::graph::{ANode, CALL_ATTRS, Concrete, Kind, NodeRef, SizeValue, instantiate};
use super::algo::instance_id;
use super::json::{
    CertificatesFile, ComponentSummary, InstanceKind, InstanceRecord, InstancesFile, StoredComponent, expand,
};
use super::{COMPONENTS, ComponentsConfig, VIEW_FILE, phases, view};
use crate::import::algo::size_tag::is_shape_port;
use crate::import::{IMPORT, IR_FILE};

struct Acc {
    name: String,
    notes: Vec<String>,
    counts: BTreeMap<String, u64>,
}

impl Acc {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            notes: Vec::new(),
            counts: BTreeMap::new(),
        }
    }
    fn fail(&mut self, n: impl Into<String>) {
        if self.notes.len() < 20 {
            self.notes.push(n.into());
        }
        *self.counts.entry("failures".into()).or_default() += 1;
    }
    fn count(&mut self, k: &str, v: u64) {
        *self.counts.entry(k.into()).or_default() += v;
    }
    fn done(self) -> Check {
        let ok = !self.counts.contains_key("failures");
        let mut c = Check::new(self.name, ok);
        for (k, v) in self.counts {
            c = c.count(k, v);
        }
        for n in self.notes {
            c = c.note(n);
        }
        c
    }
}

type Comps = HashMap<String, (StoredComponent, ComponentSummary)>;

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: ComponentsConfig = ctx.options()?;
    let units = phases(ctx)?;
    let out = ctx.output(COMPONENTS)?;
    let mut r = VerifyReport::new(ctx.name());
    // Components.
    let mut ids = Vec::new();
    if let Ok(rd) = std::fs::read_dir(out.dir().join("components")) {
        for e in rd.flatten() {
            if let Some(id) = e.file_name().to_str().and_then(|n| n.strip_suffix(".postcard")) {
                ids.push(id.to_string());
            }
        }
    }
    ids.sort();
    let comps: Comps = ids
        .par_iter()
        .map(
            |id| -> Result<(String, (StoredComponent, ComponentSummary)), PassError> {
                let c: StoredComponent = out.read_postcard(&format!("components/{id}.postcard"))?;
                let s: ComponentSummary = out.read_json(&format!("components/{id}.postcard.summary.json"))?;
                Ok((id.clone(), (c, s)))
            },
        )
        .collect::<Result<_, _>>()?;
    let mut complete = Acc::new("complete");
    for (m, p, _) in &units {
        for k in ["instances", "certificates"] {
            if !out.path(&format!("{k}/{m}/{p}.json"))?.is_file() {
                complete.fail(format!("{m}/{p}: no {k} file"));
            }
        }
    }
    complete.count("phases", units.len() as u64);
    r.push(complete.done());
    let per: Vec<PhaseResult> = units
        .par_iter()
        .with_max_len(1)
        .map(|(m, p, _)| verify_phase(ctx, &comps, m, p))
        .collect::<Result<_, _>>()?;
    let mut members: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut dedup = Acc::new("dedup");
    for (_, recs) in &per {
        for (inst, comp) in recs {
            if let Some(c) = comp {
                if !comps.contains_key(c) {
                    dedup.fail(format!("{inst}: component {c} does not exist"));
                }
                members.entry(c.clone()).or_default().insert(inst.clone());
            }
        }
    }
    for (id, (_, s)) in &comps {
        let listed: BTreeSet<String> = s.instances.iter().cloned().collect();
        let have = members.get(id).cloned().unwrap_or_default();
        if listed != have {
            dedup.fail(format!(
                "{id}: summary lists {} instances, {} point to it",
                listed.len(),
                have.len()
            ));
        }
    }
    dedup.count("components", comps.len() as u64);
    dedup.count("instances", members.values().map(|s| s.len() as u64).sum());
    r.push(dedup.done());
    r.push(check_classes(&comps));
    r.push(check_wl(&comps, &cfg));
    let mut table = Acc::new("sharing_table");
    if cfg.views {
        let mut sums: Vec<ComponentSummary> = comps.values().map(|(_, s)| s.clone()).collect();
        sums.sort_by(|a, b| a.id.cmp(&b.id));
        let want = view::table(ctx.models(), &sums);
        let have = std::fs::read_to_string(ctx.dir().views_dir().join(VIEW_FILE)).unwrap_or_default();
        if want != have {
            table.fail("views/components.md differs from the table recomputed from the summaries");
        }
        table.count("rows", sums.len() as u64);
    }
    r.push(table.done());
    for (checks, _) in per {
        for c in checks {
            r.push(c);
        }
    }
    Ok(r)
}

fn check_classes(comps: &Comps) -> Check {
    let mut acc = Acc::new("classes");
    for (id, (c, _)) in comps {
        let mut g = c.graph.clone();
        let child = |n: u32| -> Option<super::algo::classes::Iface> {
            let ch = g_child(&c.graph, n)?;
            comps.get(&ch).map(|(cc, _)| cc.iface.clone())
        };
        match assign(&mut g, &child) {
            Ok((nd, ne)) => {
                if g != c.graph || nd != c.dims || ne != c.elems {
                    acc.fail(format!("{id}: classes differ from the recomputed ones"));
                }
                if iface(&g) != c.iface {
                    acc.fail(format!("{id}: interface differs"));
                }
            }
            Err(e) => acc.fail(format!("{id}: {e}")),
        }
    }
    acc.count("components", comps.len() as u64);
    acc.done()
}

fn g_child(g: &Graph<ANode>, n: u32) -> Option<String> {
    g.nodes.get(n as usize).and_then(|x| x.child.clone())
}

fn check_wl(comps: &Comps, cfg: &ComponentsConfig) -> Check {
    let mut acc = Acc::new("wl_hash");
    for (id, (c, _)) in comps {
        let child: Vec<Option<u128>> = c
            .graph
            .nodes
            .iter()
            .map(|n| {
                n.child
                    .as_ref()
                    .and_then(|ch| comps.get(ch))
                    .and_then(|(cc, _)| u128::from_str_radix(&cc.hash, 16).ok())
            })
            .collect();
        let keys: Vec<Vec<u8>> = c.graph.nodes.iter().map(ANode::key).collect();
        let ig = Graph {
            nodes: (0..c.graph.node_count()).collect::<Vec<usize>>(),
            out: c.graph.out.clone(),
            inc: c.graph.inc.clone(),
        };
        let w = wl(
            &ig,
            |&i: &usize| {
                let mut b = keys[i].clone();
                if let Some(h) = child[i] {
                    b.extend_from_slice(&h.to_le_bytes());
                }
                hash_bytes(&b)
            },
            cfg.wl_max_rounds,
        );
        if format!("{:032x}", w.hash) != c.hash {
            acc.fail(format!("{id}: WL hash does not recompute"));
        }
        if !id.starts_with(&format!("c{:016x}", (w.hash >> 64) as u64)) {
            acc.fail(format!("{id}: id does not match its hash"));
        }
    }
    acc.count("components", comps.len() as u64);
    acc.done()
}

/// Ops left out of the graphs: the greatest set of ops that are size computation, or have a use and
/// every use is a shape port or another such op (iterated to the fixpoint).
fn shape_only(m: &Module) -> Vec<bool> {
    let users = m.users();
    let mut s: Vec<bool> = m
        .ops
        .iter()
        .map(|op| op.size || op.results.iter().any(|r| !users[r.index()].is_empty()))
        .collect();
    loop {
        let mut changed = false;
        for (i, op) in m.ops.iter().enumerate() {
            if !s[i] || op.size {
                continue;
            }
            let keep = op.results.iter().all(|r| {
                users[r.index()]
                    .iter()
                    .all(|&(u, p)| s[u.index()] || is_shape_port(m, m.op(u), p as usize))
            });
            if !keep {
                s[i] = false;
                changed = true;
            }
        }
        if !changed {
            return s;
        }
    }
}

struct Conc {
    nodes: HashMap<NodeRef, Concrete>,
    edges: HashMap<(NodeRef, NodeRef), EdgeLabel>,
}

impl Conc {
    fn edge(&mut self, a: NodeRef, b: NodeRef, e: EdgeEnd) {
        let l = self.edges.entry((a, b)).or_default();
        l.push(e);
        l.sort();
    }
}

struct PhaseFacts<'m> {
    m: &'m Module,
    shape_only: Vec<bool>,
    users: Vec<Vec<(OpId, u32)>>,
    sv: Vec<crate::import::algo::size_value::Sv>,
    func_ops: Vec<Vec<OpId>>,
}

fn func_of_op(m: &Module, o: OpId) -> FuncId {
    m.region_func(m.op(o).parent)
}

/// The concrete graph of an instance, built from the module.
fn concrete(
    f: &PhaseFacts<'_>,
    rec: &InstanceRecord,
    child: &dyn Fn(OpId) -> Option<String>,
) -> Result<Conc, String> {
    let m = f.m;
    let mut c = Conc {
        nodes: HashMap::new(),
        edges: HashMap::new(),
    };
    let ty = |v: ValueId| m.value(v).ty.tensor().cloned().ok_or("non-tensor value");
    let ops: Vec<OpId> = match rec.func {
        Some(fi) => f
            .func_ops
            .get(fi as usize)
            .cloned()
            .ok_or("function index out of range")?,
        None => expand(&rec.ops).map(OpId).collect(),
    };
    let set: BTreeSet<OpId> = ops.iter().copied().collect();
    if let Some(fi) = rec.func {
        let mut k = 0;
        for &a in &m.region(m.func(FuncId(fi)).body).args {
            if m.value(a).dim_var.is_none() {
                c.nodes.insert(
                    NodeRef::Value(a.0),
                    Concrete {
                        kind: Kind::Input { index: Some(k) },
                        attrs: Default::default(),
                        results: vec![ty(a)?],
                        child: None,
                        size: None,
                    },
                );
                k += 1;
            }
        }
    }
    let graph: Vec<OpId> = ops.iter().copied().filter(|o| !f.shape_only[o.index()]).collect();
    for &o in &graph {
        let op = m.op(o);
        let composite = op.name == "stablehlo.composite";
        let mut attrs = op.attrs.clone();
        if composite {
            for k in CALL_ATTRS {
                attrs.remove(k);
            }
        }
        c.nodes.insert(
            NodeRef::Op(o.0),
            Concrete {
                kind: Kind::Op {
                    name: op.name.clone(),
                    target: op.target.clone(),
                },
                attrs,
                results: op.results.iter().map(|&v| ty(v)).collect::<Result<_, _>>()?,
                child: if composite { child(o) } else { None },
                size: None,
            },
        );
        for (ri, &r) in op.regions.iter().enumerate() {
            for (ai, &a) in m.region(r).args.iter().enumerate() {
                c.nodes.insert(
                    NodeRef::Value(a.0),
                    Concrete {
                        kind: Kind::RegionArg {
                            region: ri as u32,
                            index: ai as u32,
                        },
                        attrs: Default::default(),
                        results: vec![ty(a)?],
                        child: None,
                        size: None,
                    },
                );
                c.edge(
                    NodeRef::Op(o.0),
                    NodeRef::Value(a.0),
                    EdgeEnd {
                        res: 0,
                        port: PortLabel::Region {
                            region: ri as u32,
                            index: ai as u32,
                        },
                    },
                );
            }
        }
    }
    let anon = rec.func.is_none();
    for &o in &graph {
        let op = m.op(o);
        let spec = cc_ir::lookup(&op.name, op.target.as_deref()).ok_or("op not in the table")?;
        let mut k = 0u32;
        for (p, &v) in op.operands.iter().enumerate() {
            if is_shape_port(m, op, p) {
                continue;
            }
            let port = if spec.commutative {
                PortLabel::Commutative
            } else if op.name == "stablehlo.composite" {
                PortLabel::Port(k)
            } else {
                PortLabel::Port(p as u32)
            };
            k += 1;
            let dst = NodeRef::Op(o.0);
            if op.dim_as_data.contains(&(p as u32)) {
                let s = NodeRef::SizeIn {
                    op: o.0,
                    port: p as u32,
                };
                c.nodes.insert(
                    s,
                    Concrete {
                        kind: Kind::SizeIn,
                        attrs: Default::default(),
                        results: vec![ty(v)?],
                        child: None,
                        size: Some(SizeValue::of(&f.sv[v.index()])),
                    },
                );
                c.edge(s, dst, EdgeEnd { res: 0, port });
                continue;
            }
            match m.value(v).def {
                Def::Result { op: u, index } if set.contains(&u) => {
                    if f.shape_only[u.index()] {
                        return Err(format!("op {} port {p}: operand from a shape-only op", o.0));
                    }
                    c.edge(NodeRef::Op(u.0), dst, EdgeEnd { res: index, port });
                }
                _ => {
                    let src = NodeRef::Value(v.0);
                    if let std::collections::hash_map::Entry::Vacant(e) = c.nodes.entry(src) {
                        if !anon {
                            return Err(format!("op {} port {p}: operand {} from outside", o.0, v.0));
                        }
                        e.insert(Concrete {
                            kind: Kind::Input { index: None },
                            attrs: Default::default(),
                            results: vec![ty(v)?],
                            child: None,
                            size: None,
                        });
                    }
                    c.edge(src, dst, EdgeEnd { res: 0, port });
                }
            }
        }
        let reg = m.region(op.parent);
        if let Owner::Op(owner) = reg.owner
            && reg.ops.last() == Some(&o)
            && set.contains(&owner)
        {
            let ri = m
                .op(owner)
                .regions
                .iter()
                .position(|&x| x == op.parent)
                .unwrap_or(0);
            c.edge(
                NodeRef::Op(o.0),
                NodeRef::Op(owner.0),
                EdgeEnd {
                    res: 0,
                    port: PortLabel::Yield { region: ri as u32 },
                },
            );
        }
        if anon {
            for (ri, &r) in op.results.iter().enumerate() {
                let outside = f.users[r.index()]
                    .iter()
                    .any(|&(u, p)| !set.contains(&u) && !is_shape_port(m, m.op(u), p as usize));
                if outside {
                    c.nodes.insert(
                        NodeRef::Out(r.0),
                        Concrete {
                            kind: Kind::Output,
                            attrs: Default::default(),
                            results: Vec::new(),
                            child: None,
                            size: None,
                        },
                    );
                    c.edge(
                        NodeRef::Op(o.0),
                        NodeRef::Out(r.0),
                        EdgeEnd {
                            res: ri as u32,
                            port: PortLabel::Port(0),
                        },
                    );
                }
            }
        }
    }
    Ok(c)
}

fn same(a: &Concrete, b: &Concrete, ctx: &SymCtx) -> Result<(), String> {
    if a.kind != b.kind {
        return Err(format!("kind {:?} vs {:?}", a.kind, b.kind));
    }
    if a.attrs != b.attrs {
        let keys: BTreeSet<&String> = a.attrs.keys().chain(b.attrs.keys()).collect();
        let k = keys.into_iter().find(|k| a.attrs.get(*k) != b.attrs.get(*k));
        let show = |x: Option<&cc_ir::Attr>| {
            let mut t = format!("{x:?}");
            t.truncate(200);
            t
        };
        return Err(match k {
            Some(k) => format!(
                "attribute {k}: {} vs {}",
                show(a.attrs.get(k)),
                show(b.attrs.get(k))
            ),
            None => "attributes differ".into(),
        });
    }
    if a.child != b.child {
        return Err(format!("child {:?} vs {:?}", a.child, b.child));
    }
    if a.size != b.size {
        return Err("size value differs".into());
    }
    if a.results.len() != b.results.len() {
        return Err("result count differs".into());
    }
    for (x, y) in a.results.iter().zip(&b.results) {
        if x.elem != y.elem || x.shape.len() != y.shape.len() {
            return Err(format!("type {x} vs {y}"));
        }
        if !x.shape.iter().zip(&y.shape).all(|(p, q)| dim_eq(p, q, ctx)) {
            return Err(format!("type {x} vs {y}"));
        }
    }
    Ok(())
}

fn recover(
    f: &PhaseFacts<'_>,
    ctx: &SymCtx,
    rec: &InstanceRecord,
    comp: &StoredComponent,
    cert: &[NodeRef],
    child: &dyn Fn(OpId) -> Option<String>,
) -> Result<(), String> {
    let b = rec.bindings.as_ref().ok_or("no bindings")?;
    let elems: Vec<cc_ir::Elem> = b
        .elems
        .iter()
        .map(|e| cc_ir::Elem::parse(e).ok_or_else(|| format!("element type `{e}` does not parse")))
        .collect::<Result<_, _>>()?;
    let conc = concrete(f, rec, child)?;
    let g = &comp.graph;
    if cert.len() != g.node_count() || conc.nodes.len() != g.node_count() {
        return Err(format!(
            "{} component nodes, {} certificate entries, {} instance nodes",
            g.node_count(),
            cert.len(),
            conc.nodes.len()
        ));
    }
    let distinct: BTreeSet<&NodeRef> = cert.iter().collect();
    if distinct.len() != cert.len() {
        return Err("the certificate is not injective".into());
    }
    if b.params.len() != g.node_count() {
        return Err("parameter lists do not match the nodes".into());
    }
    for (i, n) in g.nodes.iter().enumerate() {
        let want = instantiate(n, &b.dims, &elems, &b.params[i]).map_err(|e| format!("node {i}: {e}"))?;
        let have = conc
            .nodes
            .get(&cert[i])
            .ok_or_else(|| format!("node {i}: image {:?} is not an instance node", cert[i]))?;
        same(&want, have, ctx).map_err(|e| format!("node {i} ({:?}): {e}", cert[i]))?;
    }
    let mut ends = 0usize;
    for (u, out) in g.out.iter().enumerate() {
        for (v, l) in out {
            ends += l.len();
            match conc.edges.get(&(cert[u], cert[*v as usize])) {
                Some(x) if x == l => {}
                Some(x) => return Err(format!("edge {u}->{v}: {l:?} vs {x:?}")),
                None => return Err(format!("edge {u}->{v}: no image")),
            }
        }
    }
    let conc_ends: usize = conc.edges.values().map(Vec::len).sum();
    if conc.edges.len() != g.edge_count() || conc_ends != ends {
        return Err(format!(
            "edge counts differ: {}/{} vs {}/{}",
            g.edge_count(),
            ends,
            conc.edges.len(),
            conc_ends
        ));
    }
    Ok(())
}

type PhaseResult = (Vec<Check>, Vec<(String, Option<String>)>);

fn verify_phase(
    ctx: &PassContext,
    comps: &Comps,
    model: &str,
    phase: &str,
) -> Result<PhaseResult, PassError> {
    ctx.check_cancelled()?;
    let m: Module = ctx
        .input(IMPORT)?
        .read_postcard(&format!("{model}/{phase}/{IR_FILE}"))?;
    let out = ctx.output(COMPONENTS)?;
    let name = |c: &str| format!("{model}:{phase}:{c}");
    let (inst, certs): (InstancesFile, CertificatesFile) = match (
        out.read_json::<InstancesFile>(&format!("instances/{model}/{phase}.json")),
        out.read_json::<CertificatesFile>(&format!("certificates/{model}/{phase}.json")),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        _ => {
            return Ok((
                vec![Check::skip(
                    name("partition"),
                    "no instances or certificates file",
                )],
                Vec::new(),
            ));
        }
    };
    let func_of: Vec<FuncId> = (0..m.ops.len() as u32).map(|o| func_of_op(&m, OpId(o))).collect();
    let mut func_ops: Vec<Vec<OpId>> = vec![Vec::new(); m.funcs.len()];
    for (i, f) in func_of.iter().enumerate() {
        func_ops[f.index()].push(OpId(i as u32));
    }
    let facts = PhaseFacts {
        shape_only: shape_only(&m),
        users: m.users(),
        sv: crate::import::algo::size_value::interpret(&m, &super::algo::graph::dim_as_data_values(&m)),
        func_ops,
        m: &m,
    };
    let sctx = SymCtx::new(&m.symbols, &m.constraints);
    // Partition.
    let mut part = Acc::new(name("partition"));
    let mut owned: HashMap<(Vec<u32>, u32), u32> = HashMap::new();
    for rec in &inst.instances {
        for o in expand(&rec.ops) {
            *owned.entry((rec.position.path.clone(), o)).or_default() += 1;
        }
        if rec.kind == InstanceKind::Composite
            && let Some((&last, parent)) = rec.position.path.split_last()
        {
            *owned.entry((parent.to_vec(), last)).or_default() += 1;
        }
    }
    if let Some(root) = inst.root {
        *owned.entry((Vec::new(), root)).or_default() += 1;
    }
    let func_ops = &facts.func_ops;
    let mut occurrences = 0u64;
    let mut stack: Vec<(FuncId, Vec<u32>)> = vec![(m.entry, Vec::new())];
    while let Some((f, path)) = stack.pop() {
        for &o in &func_ops[f.index()] {
            occurrences += 1;
            match owned.remove(&(path.clone(), o.0)) {
                Some(1) => {}
                Some(n) => part.fail(format!("op {} at {path:?} owned {n} times", o.0)),
                None => part.fail(format!("op {} at {path:?} owned by no instance", o.0)),
            }
            let op = m.op(o);
            if op.name == "stablehlo.composite"
                && let Some(c) = op.callee
            {
                let mut p = path.clone();
                p.push(o.0);
                stack.push((c, p));
            }
        }
    }
    for (k, _) in owned.iter().take(5) {
        part.fail(format!("owned op {} at {:?} is not executed there", k.1, k.0));
    }
    let expanded: u64 = m.expanded_multiset().values().sum();
    if expanded != occurrences {
        part.fail(format!("{occurrences} occurrences, expanded multiset {expanded}"));
    }
    part.count("occurrences", occurrences);
    part.count("instances", inst.instances.len() as u64);
    // Recoverability.
    let mut recov = Acc::new(name("recoverability"));
    let comp_of: HashMap<&str, Option<&String>> = inst
        .instances
        .iter()
        .map(|r| (r.id.as_str(), r.component.as_ref()))
        .collect();
    let mut checked = 0u64;
    for rec in &inst.instances {
        let Some(cid) = &rec.component else {
            if rec.counts.nodes != 0 {
                recov.fail(format!("{}: nodes but no component", rec.id));
            }
            continue;
        };
        let Some((comp, _)) = comps.get(cid) else {
            recov.fail(format!("{}: component {cid} missing", rec.id));
            continue;
        };
        let Some(cert) = certs.certificates.get(&rec.id) else {
            recov.fail(format!("{}: no certificate", rec.id));
            continue;
        };
        let path = rec.position.path.clone();
        let child = |o: OpId| -> Option<String> {
            let mut p = path.clone();
            p.push(o.0);
            comp_of
                .get(instance_id(model, phase, &p).as_str())
                .copied()
                .flatten()
                .cloned()
        };
        if let Err(e) = recover(&facts, &sctx, rec, comp, cert, &child) {
            recov.fail(format!("{}: {e}", rec.id));
        }
        checked += 1;
    }
    recov.count("instances", checked);
    let recs = inst
        .instances
        .iter()
        .map(|r| (r.id.clone(), r.component.clone()))
        .collect();
    Ok((vec![part.done(), recov.done()], recs))
}
