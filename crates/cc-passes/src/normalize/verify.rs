//! Independent checks of normalize (design §4.4, §5). They read the input components and the written
//! artifacts and never call the pass's rewriter: each ledger step is re-derived from its matched nodes by
//! this module's own rule predicates and applied by its own applier.
//!
//! - `rules:<rule>`: the obligation regenerates byte for byte, its stored result is `unsat` from the
//!   configured solver for this query, and the other solver also answers `unsat`.
//! - `solver_sanity`: both solvers answer the controls as expected (integer reassociation `unsat`, float
//!   reassociation `sat`).
//! - `complete`: every input component has a normalized graph and a ledger.
//! - `<id>:replay`: the ledger, step by step, from the input graph: every step's match satisfies its
//!   rule, its edits are exactly the rule's edits for that match, and the result is the stored graph.
//! - `<id>:normal_form`: no rule matches the stored graph.
//! - `<id>:omega`: every reduction has the ω the config gives its family, with the op's reduced axes;
//!   no other node has one.
//! - `<id>:classes`: classes recomputed on the stored graph refine the stored class ids (so every instance
//!   binding stays valid), the interface is unchanged, and every interface and call node is kept.

use std::collections::{BTreeMap, HashMap};

use cc_core::{Check, VerifyReport};
use cc_ir::Attr;
use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};
use cc_pipeline::{PassContext, PassError};
use cc_smt::Verdict;
use rayon::prelude::*;

use super::json::{Edit, Expect, LedgerFile, NNode, NormalizedComponent, Omega, Order, RuleResult, Splat};
use super::{NORMALIZE, NormalizeConfig, component_ids, proofs, query_hash, smt};
use crate::components::COMPONENTS;
use crate::components::algo::classes::{Iface, assign, iface};
use crate::components::algo::graph::{AAttr, ADim, ANode, Family, Kind};
use crate::components::json::StoredComponent;

/// Label bytes and incoming ends: equal keys are CSE candidates.
type CseKey = (Vec<u8>, Vec<(u32, EdgeEnd)>);

const RANK: usize = 8;

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

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: NormalizeConfig = ctx.options()?;
    let orders = cfg.order.parse().map_err(PassError::failed)?;
    let out = ctx.output(NORMALIZE)?;
    let input = ctx.input(COMPONENTS)?;
    let mut r = VerifyReport::new(ctx.name());

    // Rule proofs.
    let (solver, runner, location) = smt(ctx)?;
    let obligations = proofs::obligations();
    let files: Vec<String> = obligations
        .iter()
        .map(|(rule, _, _)| format!("{rule}.smt2"))
        .collect();
    let other = cc_smt::run_batch(
        &runner,
        &location,
        solver.other(),
        &out.path("rules")?,
        &files,
        ctx.smt().timeout_s,
    );
    let mut sanity = Vec::new();
    for (rule, expect, text) in &obligations {
        let mut notes = Vec::new();
        let stored = std::fs::read_to_string(out.path(&format!("rules/{rule}.smt2"))?).unwrap_or_default();
        if stored != *text {
            notes.push("the stored query differs from the regenerated one".to_string());
        }
        let want = |v: &Verdict| match expect {
            Expect::Unsat => v.is_unsat(),
            Expect::Sat => matches!(v, Verdict::Sat { .. }),
        };
        match out.read_json::<RuleResult>(&format!("rules/{rule}.result")) {
            Ok(res) => {
                if res.expect != *expect || res.query_hash != query_hash(text) || res.solver != solver {
                    notes.push(format!(
                        "result is for another query or solver ({:?}, {})",
                        res.solver, res.query_hash
                    ));
                }
                if !want(&res.verdict) {
                    notes.push(format!("{}: {:?}", res.solver.name(), res.verdict));
                }
            }
            Err(e) => notes.push(format!("no result: {e}")),
        }
        match &other {
            Ok(b) => match b.verdicts.get(&format!("{rule}.smt2")) {
                Some(v) if want(v) => {}
                v => notes.push(format!("{} re-check: {v:?}", solver.other().name())),
            },
            Err(e) => notes.push(format!("{} re-check failed: {e}", solver.other().name())),
        }
        if rule.starts_with("control_") {
            sanity.extend(notes.into_iter().map(|n| format!("{rule}: {n}")));
        } else {
            r.push(check(format!("rules:{rule}"), &notes, &[]));
        }
    }
    r.push(check("solver_sanity", &sanity, &[]));

    // Components.
    let ids = component_ids(ctx)?;
    let inputs: HashMap<String, StoredComponent> = ids
        .par_iter()
        .map(|id| -> Result<(String, StoredComponent), PassError> {
            Ok((
                id.clone(),
                input.read_postcard(&format!("components/{id}.postcard"))?,
            ))
        })
        .collect::<Result<_, _>>()?;
    let mut missing = Vec::new();
    for id in &ids {
        for f in [format!("components/{id}.postcard"), format!("ledger/{id}.json")] {
            if !out.path(&f)?.is_file() {
                missing.push(format!("{id}: no {f}"));
            }
        }
    }
    r.push(check("complete", &missing, &[("components", ids.len() as u64)]));
    let per: Vec<Vec<Check>> = ids
        .par_iter()
        .filter(|id| !missing.iter().any(|m| m.starts_with(&format!("{id}:"))))
        .map(|id| -> Result<Vec<Check>, PassError> {
            let c = &inputs[id];
            let n: NormalizedComponent = out.read_postcard(&format!("components/{id}.postcard"))?;
            let l: LedgerFile = out.read_json(&format!("ledger/{id}.json"))?;
            Ok(vec![
                check(
                    format!("{id}:replay"),
                    &replay(c, &n, &l),
                    &[("steps", l.steps.len() as u64)],
                ),
                check(format!("{id}:normal_form"), &normal_form(&n.graph), &[]),
                check(format!("{id}:omega"), &omega_check(&n.graph, &orders), &[]),
                check(format!("{id}:classes"), &classes(c, &n, &inputs), &[]),
            ])
        })
        .collect::<Result<_, _>>()?;
    for cs in per {
        for c in cs {
            r.push(c);
        }
    }
    Ok(r)
}

// ---------------------------------------------------------------------------------------------------
// The graph state of the replay.

#[derive(Debug, Clone, Default)]
struct V {
    nodes: BTreeMap<u32, NNode>,
    ins: HashMap<u32, Vec<(u32, EdgeEnd)>>,
    outs: HashMap<u32, Vec<(u32, EdgeEnd)>>,
}

fn data(p: &PortLabel) -> bool {
    matches!(p, PortLabel::Port(_) | PortLabel::Commutative)
}

impl V {
    fn of(g: &Graph<NNode>, ids: impl Fn(usize) -> u32) -> V {
        let mut v = V::default();
        for (i, n) in g.nodes.iter().enumerate() {
            v.nodes.insert(ids(i), n.clone());
        }
        for (u, o) in g.out.iter().enumerate() {
            for (w, ends) in o {
                for e in ends {
                    v.add(ids(u), ids(*w as usize), *e);
                }
            }
        }
        v
    }

    fn add(&mut self, u: u32, w: u32, e: EdgeEnd) {
        let o = self.outs.entry(u).or_default();
        o.push((w, e));
        o.sort();
        let i = self.ins.entry(w).or_default();
        i.push((u, e));
        i.sort();
    }

    fn ins(&self, n: u32) -> Vec<(u32, EdgeEnd)> {
        self.ins.get(&n).cloned().unwrap_or_default()
    }

    fn outs(&self, n: u32) -> Vec<(u32, EdgeEnd)> {
        self.outs.get(&n).cloned().unwrap_or_default()
    }

    fn data_ins(&self, n: u32) -> Vec<(u32, EdgeEnd)> {
        self.ins(n).into_iter().filter(|(_, e)| data(&e.port)).collect()
    }

    fn apply(&mut self, e: &Edit) -> Result<(), String> {
        match e {
            Edit::Unlink { src, dst, end } => {
                let o = self.outs.entry(*src).or_default();
                let a = o
                    .iter()
                    .position(|x| *x == (*dst, *end))
                    .ok_or_else(|| format!("unlink {src} -> {dst}: no such edge"))?;
                o.remove(a);
                let i = self.ins.entry(*dst).or_default();
                if let Some(b) = i.iter().position(|x| *x == (*src, *end)) {
                    i.remove(b);
                }
                Ok(())
            }
            Edit::Link { src, dst, end } => {
                if !self.nodes.contains_key(src) || !self.nodes.contains_key(dst) {
                    return Err(format!("link {src} -> {dst}: missing node"));
                }
                self.add(*src, *dst, *end);
                Ok(())
            }
            Edit::Relabel { node, value } => {
                let n = self
                    .nodes
                    .get_mut(node)
                    .ok_or_else(|| format!("relabel {node}: missing"))?;
                if n.origin != value.origin {
                    return Err(format!("relabel {node}: origin changed"));
                }
                *n = (**value).clone();
                Ok(())
            }
            Edit::Remove { node } => {
                if !self.ins(*node).is_empty() || !self.outs(*node).is_empty() {
                    return Err(format!("remove {node}: node has edges"));
                }
                self.nodes
                    .remove(node)
                    .map(drop)
                    .ok_or_else(|| format!("remove {node}: missing"))
            }
        }
    }

    /// The graph with nodes in ascending id.
    fn graph(&self) -> Graph<NNode> {
        let idx: HashMap<u32, u32> = self
            .nodes
            .keys()
            .enumerate()
            .map(|(i, k)| (*k, i as u32))
            .collect();
        let mut g: Graph<NNode> = Graph::default();
        for n in self.nodes.values() {
            g.add_node(n.clone());
        }
        for (u, o) in &self.outs {
            for (w, e) in o {
                if let (Some(a), Some(b)) = (idx.get(u), idx.get(w)) {
                    g.add_edge(*a, *b, *e);
                }
            }
        }
        g
    }
}

// ---------------------------------------------------------------------------------------------------
// Rule predicates: the edits a rule makes for a match, or why the match is not one.

fn name(x: &NNode) -> &str {
    match &x.label.kind {
        Kind::Op { name, .. } => name,
        _ => "",
    }
}

fn kept(v: &V, n: u32) -> Result<&NNode, String> {
    let x = v
        .nodes
        .get(&n)
        .ok_or_else(|| format!("node {n} does not exist"))?;
    let structural = !matches!(x.label.kind, Kind::Op { .. })
        || matches!(
            name(x),
            "func.return" | "stablehlo.return" | "stablehlo.composite"
        )
        || x.label.results.is_empty()
        || v.outs(n)
            .iter()
            .chain(v.ins(n).iter())
            .any(|(_, e)| !data(&e.port));
    if structural {
        Err(format!("node {n} may not be rewritten"))
    } else {
        Ok(x)
    }
}

fn ints(x: &NNode, k: &str) -> Option<Vec<i64>> {
    match x.label.attrs.get(k)? {
        AAttr::Lit(Attr::Ints(v)) => Some(v.clone()),
        AAttr::Lit(Attr::Dense { values: Some(vs), .. }) => vs
            .iter()
            .map(|s| s.as_int().and_then(|i| i64::try_from(i).ok()))
            .collect(),
        _ => None,
    }
}

fn is_iota(v: &[i64]) -> bool {
    (0..v.len()).all(|i| v[i] == i as i64)
}

fn moved(v: &V, from: u32, res: u32, to: u32, to_res: u32, ed: &mut Vec<Edit>) {
    for (w, e) in v.outs(from) {
        if e.res == res && data(&e.port) {
            ed.push(Edit::Unlink {
                src: from,
                dst: w,
                end: e,
            });
            ed.push(Edit::Link {
                src: to,
                dst: w,
                end: EdgeEnd {
                    res: to_res,
                    port: e.port,
                },
            });
        }
    }
}

fn removed(v: &V, n: u32, ed: &mut Vec<Edit>) {
    for (u, e) in v.ins(n) {
        ed.push(Edit::Unlink {
            src: u,
            dst: n,
            end: e,
        });
    }
    ed.push(Edit::Remove { node: n });
}

fn one_in(v: &V, n: u32) -> Result<(u32, EdgeEnd), String> {
    match v.data_ins(n).as_slice() {
        [x] => Ok(*x),
        _ => Err(format!("node {n} does not have exactly one operand")),
    }
}

fn expected(v: &V, rule: &str, m: &[u32], regions: &dyn Fn(&V) -> Vec<u32>) -> Result<Vec<Edit>, String> {
    let at = |i: usize| {
        m.get(i)
            .copied()
            .ok_or_else(|| format!("{rule}: matched list too short"))
    };
    let mut ed = Vec::new();
    match rule {
        "dce" => {
            let n = at(0)?;
            kept(v, n)?;
            if !v.outs(n).is_empty() {
                return Err(format!("dce: node {n} has uses"));
            }
            removed(v, n, &mut ed);
        }
        "transpose_identity" | "broadcast_identity" | "convert_identity" => {
            let (n, u) = (at(0)?, at(1)?);
            let x = kept(v, n)?;
            let ok = match rule {
                "transpose_identity" => {
                    name(x) == "stablehlo.transpose" && ints(x, "permutation").is_some_and(|p| is_iota(&p))
                }
                "broadcast_identity" => {
                    matches!(
                        name(x),
                        "stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim"
                    ) && x.label.results.len() == 1
                        && ints(x, "broadcast_dimensions")
                            .is_some_and(|d| is_iota(&d) && d.len() == x.label.results[0].dims.len())
                }
                _ => name(x) == "stablehlo.convert",
            };
            if !ok || x.label.results.len() != 1 || !x.splats.is_empty() {
                return Err(format!("{rule}: node {n} is not that identity"));
            }
            let (src, e) = one_in(v, n)?;
            let t = v
                .nodes
                .get(&src)
                .and_then(|y| y.label.results.get(e.res as usize));
            if src != u
                || e.port != PortLabel::Port(0)
                || t != Some(&x.label.results[0])
                || x.label.results[0].dims.len() > RANK
            {
                return Err(format!(
                    "{rule}: node {n}: the operand's type differs or is not node {u}"
                ));
            }
            moved(v, n, 0, u, e.res, &mut ed);
            removed(v, n, &mut ed);
        }
        "transpose_collapse" => {
            let (n, k) = (at(0)?, at(1)?);
            let x = kept(v, n)?;
            let y = kept(v, k)?;
            if name(x) != "stablehlo.transpose" || name(y) != "stablehlo.transpose" {
                return Err("transpose_collapse: not two transposes".into());
            }
            let (s1, e1) = one_in(v, n)?;
            let (s0, e0) = one_in(v, k)?;
            let (q, p) = (
                ints(x, "permutation").unwrap_or_default(),
                ints(y, "permutation").unwrap_or_default(),
            );
            if s1 != k || q.len() != p.len() || q.len() > RANK {
                return Err("transpose_collapse: shapes of the match differ".into());
            }
            let c: Option<Vec<i64>> = q
                .iter()
                .map(|&d| p.get(usize::try_from(d).ok()?).copied())
                .collect();
            let mut nx = x.clone();
            match (nx.label.attrs.get_mut("permutation"), c) {
                (Some(AAttr::Lit(Attr::Ints(a))), Some(c)) => *a = c,
                _ => return Err("transpose_collapse: no permutation".into()),
            }
            ed.push(Edit::Unlink {
                src: k,
                dst: n,
                end: e1,
            });
            ed.push(Edit::Link {
                src: s0,
                dst: n,
                end: EdgeEnd {
                    res: e0.res,
                    port: e1.port,
                },
            });
            ed.push(Edit::Relabel {
                node: n,
                value: Box::new(nx),
            });
        }
        "reshape_collapse" => {
            let (n, k) = (at(0)?, at(1)?);
            let reshape = |x: &NNode| matches!(name(x), "stablehlo.reshape" | "stablehlo.dynamic_reshape");
            if !reshape(kept(v, n)?) || !reshape(kept(v, k)?) {
                return Err("reshape_collapse: not two reshapes".into());
            }
            let p0 = |a: u32| {
                v.data_ins(a)
                    .into_iter()
                    .find(|(_, e)| e.port == PortLabel::Port(0))
            };
            let (s1, e1) = p0(n).ok_or("reshape_collapse: no operand")?;
            let (s0, e0) = p0(k).ok_or("reshape_collapse: no operand")?;
            if s1 != k {
                return Err("reshape_collapse: the outer reshape does not read the inner one".into());
            }
            ed.push(Edit::Unlink {
                src: k,
                dst: n,
                end: e1,
            });
            ed.push(Edit::Link {
                src: s0,
                dst: n,
                end: EdgeEnd {
                    res: e0.res,
                    port: e1.port,
                },
            });
        }
        "splat_fold" => {
            let (w, c) = (at(0)?, at(1)?);
            let b = m.get(2).copied();
            let x = kept(v, w)?;
            let ew = match &x.label.kind {
                Kind::Op { name, target } => {
                    cc_ir::lookup(name, target.as_deref()).is_some_and(|s| s.elementwise)
                }
                _ => false,
            };
            if !ew {
                return Err(format!("splat_fold: node {w} is not element-wise"));
            }
            let u = b.unwrap_or(c);
            if uniform_of(v, u) != Some((c, b)) {
                return Err(format!("splat_fold: node {u} is not a uniform operand"));
            }
            let (_, e) = v
                .data_ins(w)
                .into_iter()
                .find(|(s, _)| *s == u)
                .ok_or("splat_fold: the uniform operand is not an operand")?;
            let ty = v.nodes[&u]
                .label
                .results
                .get(e.res as usize)
                .cloned()
                .ok_or("splat_fold: no type")?;
            if ty.dims.len() > RANK {
                return Err("splat_fold: rank above the proven bound".into());
            }
            let mut nx = x.clone();
            nx.splats.push(Splat {
                port: e.port,
                constant: c,
                broadcast: b,
                ty,
            });
            nx.splats.sort_by_key(|a| (a.port, a.constant));
            ed.push(Edit::Unlink {
                src: u,
                dst: w,
                end: e,
            });
            ed.push(Edit::Relabel {
                node: w,
                value: Box::new(nx),
            });
        }
        "cse" => {
            let (a, b) = (at(0)?, at(1)?);
            let reg = regions(v);
            let (x, y) = (kept(v, a)?, kept(v, b)?);
            let fresh = |z: &NNode| !params(z) && z.splats.is_empty() && z.omega.is_none();
            if !(a < b && fresh(x) && fresh(y) && x.label == y.label && v.ins(a) == v.ins(b))
                || reg.contains(&a)
                || reg.contains(&b)
            {
                return Err(format!("cse: nodes {a} and {b} are not equal computations"));
            }
            for r in 0..y.label.results.len() as u32 {
                moved(v, b, r, a, r, &mut ed);
            }
            removed(v, b, &mut ed);
        }
        _ => return Err(format!("unknown rule `{rule}`")),
    }
    Ok(ed)
}

fn params(x: &NNode) -> bool {
    matches!(x.label.kind, Kind::SizeIn)
        || x.label.attrs.values().any(|a| match a {
            AAttr::Value { .. } => true,
            AAttr::Sized { entries, .. } => entries.contains(&None),
            AAttr::Lit(_) => false,
        })
}

/// `(constant, broadcast)` when `u` is a splat constant, or a broadcast of a rank-0 constant.
fn uniform_of(v: &V, u: u32) -> Option<(u32, Option<u32>)> {
    let x = v.nodes.get(&u)?;
    if name(x) == "stablehlo.constant"
        && matches!(x.label.attrs.get("value"), Some(AAttr::Value { splat: true }))
    {
        return Some((u, None));
    }
    if matches!(
        name(x),
        "stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim"
    ) && kept(v, u).is_ok()
    {
        let (c, _) = one_in(v, u).ok()?;
        let y = v.nodes.get(&c)?;
        if name(y) == "stablehlo.constant"
            && y.label.results.first().is_some_and(|t| t.dims.is_empty())
            && matches!(y.label.attrs.get("value"), Some(AAttr::Value { .. }))
        {
            return Some((c, Some(u)));
        }
    }
    None
}

/// Nodes in or feeding a region: forward from region arguments, backward from region terminators.
fn region_nodes(v: &V) -> Vec<u32> {
    let mut seen: BTreeMap<u32, ()> = BTreeMap::new();
    let mut st: Vec<u32> = v
        .nodes
        .iter()
        .filter(|(_, x)| matches!(x.label.kind, Kind::RegionArg { .. }))
        .map(|(k, _)| *k)
        .collect();
    while let Some(n) = st.pop() {
        if seen.insert(n, ()).is_some() {
            continue;
        }
        st.extend(
            v.outs(n)
                .into_iter()
                .filter(|(_, e)| data(&e.port))
                .map(|(w, _)| w),
        );
    }
    let mut back: BTreeMap<u32, ()> = BTreeMap::new();
    let mut st: Vec<u32> = v
        .nodes
        .keys()
        .copied()
        .filter(|&n| {
            v.outs(n)
                .iter()
                .any(|(_, e)| matches!(e.port, PortLabel::Yield { .. }))
        })
        .collect();
    while let Some(n) = st.pop() {
        if back.insert(n, ()).is_some() {
            continue;
        }
        st.extend(
            v.ins(n)
                .into_iter()
                .filter(|(_, e)| data(&e.port))
                .map(|(u, _)| u),
        );
    }
    seen.keys().chain(back.keys()).copied().collect()
}

fn replay(c: &StoredComponent, n: &NormalizedComponent, l: &LedgerFile) -> Vec<String> {
    let mut notes = Vec::new();
    if l.id != c.id || n.id != c.id || n.height != c.height || n.dims != c.dims || n.elems != c.elems {
        notes.push("ids, height or class counts differ from the input".to_string());
    }
    let start: Graph<NNode> = c.graph.map(|a| NNode {
        label: a.clone(),
        omega: None,
        splats: Vec::new(),
        origin: 0,
    });
    let mut v = V::of(&start, |i| i as u32);
    for (k, x) in v.nodes.iter_mut() {
        x.origin = *k;
    }
    for (i, s) in l.steps.iter().enumerate() {
        match expected(&v, &s.rule, &s.matched, &region_nodes) {
            Ok(ed) if ed == s.edits => {}
            Ok(_) => {
                notes.push(format!("step {i} ({}): edits differ from the rule's", s.rule));
                return notes;
            }
            Err(e) => {
                notes.push(format!("step {i}: {e}"));
                return notes;
            }
        }
        for e in &s.edits {
            if let Err(m) = v.apply(e) {
                notes.push(format!("step {i}: {m}"));
                return notes;
            }
        }
    }
    let mut stored = n.graph.clone();
    for x in &mut stored.nodes {
        x.omega = None;
    }
    if v.graph() != stored {
        notes.push("the replayed graph differs from the stored one".into());
    }
    notes
}

fn normal_form(g: &Graph<NNode>) -> Vec<String> {
    let v = V::of(g, |i| g.nodes[i].origin);
    let mut notes = Vec::new();
    let reg = region_nodes(&v);
    for &n in v.nodes.keys() {
        let pre = |w: u32| v.data_ins(w).first().map(|(u, _)| *u);
        let mut tries: Vec<(&str, Vec<u32>)> = vec![("dce", vec![n])];
        if let Some(u) = pre(n) {
            for r in [
                "transpose_identity",
                "broadcast_identity",
                "convert_identity",
                "transpose_collapse",
                "reshape_collapse",
            ] {
                tries.push((r, vec![n, u]));
            }
        }
        for (u, _) in v.data_ins(n) {
            if let Some((c, b)) = uniform_of(&v, u) {
                let mut m = vec![n, c];
                m.extend(b);
                tries.push(("splat_fold", m));
            }
        }
        for (rule, m) in tries {
            if expected(&v, rule, &m, &region_nodes).is_ok() {
                notes.push(format!("{rule} still matches {m:?}"));
            }
        }
    }
    let mut seen: HashMap<CseKey, u32> = HashMap::new();
    for (&n, x) in &v.nodes {
        if kept(&v, n).is_err() || params(x) || !x.splats.is_empty() || reg.contains(&n) {
            continue;
        }
        let key = (postcard::to_stdvec(&x.label).unwrap_or_default(), v.ins(n));
        if let Some(a) = seen.insert(key, n) {
            notes.push(format!("cse still matches [{a}, {n}]"));
        }
    }
    notes
}

fn fam(f: &Family) -> &'static str {
    match f {
        Family::Bool => "bool",
        Family::Sint => "sint",
        Family::Uint => "uint",
        Family::Float => "float",
        Family::Complex(_) => "complex",
    }
}

/// The ω a node must carry (spec definitions, docs/passes/normalize.md §4), or `None`.
fn expected_omega(v: &V, n: u32, orders: &super::config::Orders) -> Result<Option<Omega>, String> {
    let x = &v.nodes[&n];
    let (op, tgt) = match &x.label.kind {
        Kind::Op { name, target } => (name.as_str(), target.as_deref()),
        _ => return Ok(None),
    };
    let rank_of = |port: u32| -> Option<u32> {
        let (u, e) = v
            .data_ins(n)
            .into_iter()
            .find(|(_, e)| e.port == PortLabel::Port(port))?;
        Some(v.nodes.get(&u)?.label.results.get(e.res as usize)?.dims.len() as u32)
    };
    let us = |a: &[i64]| a.iter().map(|&d| d as u32).collect::<Vec<u32>>();
    let axes: Vec<u32> = match (op, tgt) {
        ("stablehlo.reduce", _) => {
            let mut d = us(&ints(x, "dimensions").ok_or("reduce without dimensions")?);
            d.sort_unstable();
            d
        }
        ("stablehlo.reduce_window", _)
        | ("stablehlo.custom_call", Some("stablehlo.dynamic_reduce_window")) => {
            (0..rank_of(0).ok_or("reduce_window without operand")?).collect()
        }
        ("stablehlo.dot_general", _) => match x.label.attrs.get("dot_dimension_numbers") {
            Some(AAttr::Lit(Attr::DotDims(d))) => us(&d.lhs_contracting),
            _ => return Err("dot_general without dimension numbers".into()),
        },
        ("stablehlo.convolution" | "stablehlo.dynamic_conv", _) => {
            match x.label.attrs.get("dimension_numbers") {
                Some(AAttr::Lit(Attr::ConvDims(d))) => {
                    let mut a = us(&d.kernel_spatial);
                    a.push(d.kernel_input_feature as u32);
                    a
                }
                _ => return Err("convolution without dimension numbers".into()),
            }
        }
        ("stablehlo.scatter", _) => {
            let k = x.label.results.len() as u32;
            return Ok(Some(Omega {
                order: Order::Seq,
                axes: (0..rank_of(k + 1).ok_or("scatter without updates")?).collect(),
            }));
        }
        _ => return Ok(None),
    };
    let keys: Vec<&str> = x.label.results.iter().map(|t| fam(&t.family)).collect();
    let key = ["complex", "float", "sint", "uint", "bool"]
        .into_iter()
        .find(|k| keys.contains(k))
        .ok_or("reduction without results")?;
    Ok(Some(Omega {
        order: orders.get(key),
        axes,
    }))
}

fn omega_check(g: &Graph<NNode>, orders: &super::config::Orders) -> Vec<String> {
    let v = V::of(g, |i| g.nodes[i].origin);
    let mut notes = Vec::new();
    for (&n, x) in &v.nodes {
        match expected_omega(&v, n, orders) {
            Ok(want) if want == x.omega => {}
            Ok(want) => notes.push(format!("node {n}: ω {:?}, expected {want:?}", x.omega)),
            Err(e) => notes.push(format!("node {n}: {e}")),
        }
    }
    notes
}

fn classes(
    c: &StoredComponent,
    n: &NormalizedComponent,
    inputs: &HashMap<String, StoredComponent>,
) -> Vec<String> {
    let mut notes = Vec::new();
    let mut g: Graph<ANode> = n.graph.map(|x| x.label.clone());
    let child_iface = |k: u32| -> Option<Iface> {
        let id = g.nodes.get(k as usize)?.child.clone()?;
        inputs.get(&id).map(|s| s.iface.clone())
    };
    let ifaces: Vec<Option<Iface>> = (0..g.node_count() as u32).map(child_iface).collect();
    if iface(&g) != c.iface || n.iface != c.iface {
        notes.push("the interface changed".into());
    }
    let stored: Vec<Vec<(Vec<ADim>, u32)>> = g
        .nodes
        .iter()
        .map(|x| x.results.iter().map(|t| (t.dims.clone(), t.elem)).collect())
        .collect();
    if let Err(e) = assign(&mut g, &|k| ifaces.get(k as usize).cloned().flatten()) {
        notes.push(format!("classes: {e}"));
        return notes;
    }
    let (mut dmap, mut emap): (HashMap<u32, u32>, HashMap<u32, u32>) = (HashMap::new(), HashMap::new());
    for (x, s) in g.nodes.iter().zip(&stored) {
        for (t, (sd, se)) in x.results.iter().zip(s) {
            if *emap.entry(t.elem).or_insert(*se) != *se {
                notes.push(format!("element class {} spans two stored classes", t.elem));
            }
            for (a, b) in t.dims.iter().zip(sd) {
                match (a, b) {
                    (ADim::Var(p), ADim::Var(q)) => {
                        if *dmap.entry(*p).or_insert(*q) != *q {
                            notes.push(format!("dim class {p} spans two stored classes"));
                        }
                    }
                    (ADim::One, ADim::One) => {}
                    _ => notes.push("a literal 1 changed".into()),
                }
            }
        }
    }
    // Interface and call nodes are kept, with their labels.
    let by_origin: HashMap<u32, &NNode> = n.graph.nodes.iter().map(|x| (x.origin, x)).collect();
    for (i, a) in c.graph.nodes.iter().enumerate() {
        let must = !matches!(a.kind, Kind::Op { .. })
            || a.is_call()
            || matches!(&a.kind, Kind::Op { name, .. } if name == "func.return");
        if must && by_origin.get(&(i as u32)).map(|x| &x.label) != Some(a) {
            notes.push(format!("node {i} (interface or call) was not kept"));
        }
    }
    notes.truncate(20);
    notes
}
