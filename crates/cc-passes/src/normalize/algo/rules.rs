//! The rewrite rules (docs/passes/normalize.md §4, rule catalogue). Each rule is a matcher on the working
//! graph that returns the [`Step`] it would apply; nodes are never created, so every surviving node keeps
//! its input id (its origin) and every class id. Rules never touch the interface (inputs, outputs, region
//! arguments, `size_in`), composite calls, terminators or region owners.

use std::collections::HashMap;

use cc_ir::Attr;
use cc_ir::absgraph::{EdgeEnd, PortLabel};

use super::work::{Work, is_data};
use crate::components::algo::graph::{AAttr, Kind};
use crate::normalize::json::{Edit, NNode, Splat, Step};

/// Label bytes and incoming ends: equal keys are CSE candidates.
type CseKey = (Vec<u8>, Vec<(u32, EdgeEnd)>);

/// The rules, in application order.
pub const RULES: [&str; 8] = [
    "dce",
    "transpose_identity",
    "broadcast_identity",
    "convert_identity",
    "transpose_collapse",
    "reshape_collapse",
    "splat_fold",
    "cse",
];

/// Largest rank the index-map obligations are stated for; layout rules never match above it.
pub const MAX_RANK: usize = 8;

pub fn op_name(n: &NNode) -> Option<&str> {
    match &n.label.kind {
        Kind::Op { name, .. } => Some(name.as_str()),
        _ => None,
    }
}

fn target(n: &NNode) -> Option<&str> {
    match &n.label.kind {
        Kind::Op { target, .. } => target.as_deref(),
        _ => None,
    }
}

/// Nodes no rule removes, merges or relabels.
pub fn protected(w: &Work, n: u32) -> bool {
    let Some(x) = w.node(n) else {
        return true;
    };
    match op_name(x) {
        None => true,
        Some("func.return" | "stablehlo.return" | "stablehlo.composite") => true,
        Some(_) => x.label.results.is_empty() || w.has_regions(n),
    }
}

/// Whether the node carries an instance parameter (a constant's value or a size-attribute entry).
pub fn has_params(n: &NNode) -> bool {
    n.label.attrs.values().any(|a| match a {
        AAttr::Value { .. } => true,
        AAttr::Sized { entries, .. } => entries.iter().any(Option::is_none),
        AAttr::Lit(_) => false,
    }) || matches!(n.label.kind, Kind::SizeIn)
}

/// An integer-list attribute kept literally.
pub fn lit_ints(n: &NNode, key: &str) -> Option<Vec<i64>> {
    match n.label.attrs.get(key)? {
        AAttr::Lit(Attr::Ints(v)) => Some(v.clone()),
        AAttr::Lit(Attr::Dense {
            values: Some(vals), ..
        }) => vals
            .iter()
            .map(|s| s.as_int().and_then(|x| i64::try_from(x).ok()))
            .collect(),
        _ => None,
    }
}

fn iota(v: &[i64]) -> bool {
    v.iter().enumerate().all(|(i, &x)| x == i as i64)
}

/// Data uses of `n`'s result `res` moved to `(m, rm)`.
fn redirect(w: &Work, n: u32, res: u32, m: u32, rm: u32, ed: &mut Vec<Edit>) {
    for (v, e) in w.uses(n, res) {
        ed.push(Edit::Unlink {
            src: n,
            dst: v,
            end: e,
        });
        ed.push(Edit::Link {
            src: m,
            dst: v,
            end: EdgeEnd {
                res: rm,
                port: e.port,
            },
        });
    }
}

/// Unlink every incoming end of `n`, then remove it.
fn drop_node(w: &Work, n: u32, ed: &mut Vec<Edit>) {
    for (u, e) in &w.ins[n as usize] {
        ed.push(Edit::Unlink {
            src: *u,
            dst: n,
            end: *e,
        });
    }
    ed.push(Edit::Remove { node: n });
}

fn step(rule: &str, matched: Vec<u32>, edits: Vec<Edit>) -> Step {
    Step {
        rule: rule.into(),
        matched,
        edits,
    }
}

/// `dce`: an op without uses (and without regions) is removed.
fn dce(w: &Work, n: u32) -> Option<Step> {
    if protected(w, n) || !w.outs[n as usize].is_empty() {
        return None;
    }
    let mut ed = Vec::new();
    drop_node(w, n, &mut ed);
    Some(step("dce", vec![n], ed))
}

/// Shared check of the identity rules: one data operand at port 0, one result, the operand's type
/// equal to the result's (family, element class and every dim class).
fn identity_shape(w: &Work, n: u32) -> Option<(u32, EdgeEnd)> {
    let x = w.node(n)?;
    if protected(w, n) || x.label.results.len() != 1 || !x.splats.is_empty() {
        return None;
    }
    let (u, e) = w.operand0(n)?;
    if e.port != PortLabel::Port(0) {
        return None;
    }
    let t = w.node(u)?.label.results.get(e.res as usize)?;
    (t == &x.label.results[0] && t.dims.len() <= MAX_RANK).then_some((u, e))
}

fn identity(w: &Work, n: u32, rule: &str, ok: impl Fn(&NNode) -> bool) -> Option<Step> {
    let (u, e) = identity_shape(w, n)?;
    if !ok(w.node(n)?) {
        return None;
    }
    let mut ed = Vec::new();
    redirect(w, n, 0, u, e.res, &mut ed);
    drop_node(w, n, &mut ed);
    Some(step(rule, vec![n, u], ed))
}

/// `transpose_identity`: `transpose(x, iota)` is `x`.
fn transpose_identity(w: &Work, n: u32) -> Option<Step> {
    identity(w, n, "transpose_identity", |x| {
        op_name(x) == Some("stablehlo.transpose") && lit_ints(x, "permutation").is_some_and(|p| iota(&p))
    })
}

/// `broadcast_identity`: a broadcast of the same rank with `broadcast_dimensions = iota` and equal dim
/// classes is its operand.
fn broadcast_identity(w: &Work, n: u32) -> Option<Step> {
    identity(w, n, "broadcast_identity", |x| {
        matches!(
            op_name(x),
            Some("stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim")
        ) && lit_ints(x, "broadcast_dimensions")
            .is_some_and(|d| iota(&d) && d.len() == x.label.results[0].dims.len())
    })
}

/// `convert_identity`: a conversion whose operand has the result's element class is its operand.
fn convert_identity(w: &Work, n: u32) -> Option<Step> {
    identity(w, n, "convert_identity", |x| {
        op_name(x) == Some("stablehlo.convert")
    })
}

/// `transpose_collapse`: `transpose(transpose(x, p), q)` becomes `transpose(x, c)` with `c[d] = p[q[d]]`.
/// Matched: `[outer, inner]`.
fn transpose_collapse(w: &Work, n: u32) -> Option<Step> {
    let x = w.node(n)?;
    if protected(w, n) || op_name(x) != Some("stablehlo.transpose") {
        return None;
    }
    let (m, e) = w.operand0(n)?;
    let y = w.node(m)?;
    if protected(w, m) || op_name(y) != Some("stablehlo.transpose") {
        return None;
    }
    let q = lit_ints(x, "permutation")?;
    let p = lit_ints(y, "permutation")?;
    if q.len() != p.len() || q.len() > MAX_RANK {
        return None;
    }
    let (src, se) = w.operand0(m)?;
    let c: Vec<i64> = q
        .iter()
        .map(|&d| p.get(d as usize).copied().unwrap_or(-1))
        .collect();
    if c.iter().any(|&v| v < 0) {
        return None;
    }
    let mut nx = x.clone();
    let AAttr::Lit(a) = nx.label.attrs.get_mut("permutation")? else {
        return None;
    };
    *a = match a {
        Attr::Ints(_) => Attr::Ints(c),
        _ => return None,
    };
    let ed = vec![
        Edit::Unlink {
            src: m,
            dst: n,
            end: e,
        },
        Edit::Link {
            src,
            dst: n,
            end: EdgeEnd {
                res: se.res,
                port: e.port,
            },
        },
        Edit::Relabel {
            node: n,
            value: Box::new(nx),
        },
    ];
    Some(step("transpose_collapse", vec![n, m], ed))
}

fn is_reshape(x: &NNode) -> bool {
    matches!(
        op_name(x),
        Some("stablehlo.reshape" | "stablehlo.dynamic_reshape")
    )
}

/// `reshape_collapse`: a reshape of a reshape reads the inner reshape's operand. Matched: `[outer, inner]`.
fn reshape_collapse(w: &Work, n: u32) -> Option<Step> {
    let x = w.node(n)?;
    if protected(w, n) || !is_reshape(x) {
        return None;
    }
    let (m, e) = w
        .data_ins(n)
        .find(|(_, e)| e.port == PortLabel::Port(0))
        .copied()?;
    let y = w.node(m)?;
    if protected(w, m) || !is_reshape(y) {
        return None;
    }
    let (src, se) = w
        .data_ins(m)
        .find(|(_, e)| e.port == PortLabel::Port(0))
        .copied()?;
    let ed = vec![
        Edit::Unlink {
            src: m,
            dst: n,
            end: e,
        },
        Edit::Link {
            src,
            dst: n,
            end: EdgeEnd {
                res: se.res,
                port: e.port,
            },
        },
    ];
    Some(step("reshape_collapse", vec![n, m], ed))
}

/// A uniform operand: `(constant, broadcast)` when `u` is a splat constant, or a `broadcast_in_dim` of a
/// rank-0 constant.
pub fn uniform(w: &Work, u: u32) -> Option<(u32, Option<u32>)> {
    let x = w.node(u)?;
    let splat_const = |y: &NNode| {
        op_name(y) == Some("stablehlo.constant")
            && matches!(y.label.attrs.get("value"), Some(AAttr::Value { splat: true }))
    };
    if splat_const(x) {
        return Some((u, None));
    }
    if matches!(
        op_name(x),
        Some("stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim")
    ) && !protected(w, u)
    {
        let (c, _) = w.operand0(u)?;
        let y = w.node(c)?;
        if (op_name(y) == Some("stablehlo.constant"))
            && y.label.results.first().is_some_and(|t| t.dims.is_empty())
            && matches!(y.label.attrs.get("value"), Some(AAttr::Value { .. }))
        {
            return Some((c, Some(u)));
        }
    }
    None
}

fn elementwise(x: &NNode) -> bool {
    match &x.label.kind {
        Kind::Op { name, target } => {
            cc_ir::lookup(name, target.as_deref()).is_some_and(|s| s.elementwise && !x.label.is_call())
        }
        _ => false,
    }
}

/// `splat_fold`: a uniform operand of an element-wise op becomes a port annotation. Matched:
/// `[consumer, constant, broadcast?]`.
fn splat_fold(w: &Work, v: u32) -> Option<Step> {
    let x = w.node(v)?;
    if protected(w, v) || !elementwise(x) {
        return None;
    }
    let (u, e) = w.data_ins(v).find(|(u, _)| uniform(w, *u).is_some()).copied()?;
    let (c, b) = uniform(w, u)?;
    let ty = w.node(u)?.label.results.get(e.res as usize)?.clone();
    if ty.dims.len() > MAX_RANK {
        return None;
    }
    let mut nx = x.clone();
    nx.splats.push(Splat {
        port: e.port,
        constant: c,
        broadcast: b,
        ty,
    });
    nx.splats.sort_by_key(|a| (a.port, a.constant));
    let ed = vec![
        Edit::Unlink {
            src: u,
            dst: v,
            end: e,
        },
        Edit::Relabel {
            node: v,
            value: Box::new(nx),
        },
    ];
    let mut matched = vec![v, c];
    matched.extend(b);
    Some(step("splat_fold", matched, ed))
}

/// Nodes inside a region, or feeding one: reachable from a region argument, or reaching a region
/// terminator, along data edges. CSE only merges nodes outside this set.
pub fn in_regions(w: &Work) -> Vec<bool> {
    let n = w.nodes.len();
    let mut fwd = vec![false; n];
    let mut stack: Vec<u32> = w
        .live()
        .filter(|&v| matches!(w.node(v).map(|x| &x.label.kind), Some(Kind::RegionArg { .. })))
        .collect();
    while let Some(v) = stack.pop() {
        if std::mem::replace(&mut fwd[v as usize], true) {
            continue;
        }
        for (d, e) in &w.outs[v as usize] {
            if is_data(&e.port) {
                stack.push(*d);
            }
        }
    }
    let mut back = vec![false; n];
    let mut stack: Vec<u32> = w
        .live()
        .filter(|&v| {
            w.outs[v as usize]
                .iter()
                .any(|(_, e)| matches!(e.port, PortLabel::Yield { .. }))
        })
        .collect();
    while let Some(v) = stack.pop() {
        if std::mem::replace(&mut back[v as usize], true) {
            continue;
        }
        for (s, e) in &w.ins[v as usize] {
            if is_data(&e.port) {
                stack.push(*s);
            }
        }
    }
    let mark: Vec<bool> = fwd.iter().zip(&back).map(|(a, b)| *a || *b).collect();
    mark
}

/// Whether `a` and `b` may be merged (`a < b`): equal labels, equal result types (class ids), no
/// parameters, folded operands or regions, the same incoming ends, both outside every region.
pub fn cse_ok(w: &Work, a: u32, b: u32, regions: &[bool]) -> bool {
    let (Some(x), Some(y)) = (w.node(a), w.node(b)) else {
        return false;
    };
    a < b
        && !protected(w, a)
        && !protected(w, b)
        && !regions[a as usize]
        && !regions[b as usize]
        && !has_params(x)
        && !has_params(y)
        && x.splats.is_empty()
        && y.splats.is_empty()
        && x.omega.is_none()
        && y.omega.is_none()
        && x.label == y.label
        && w.ins[a as usize] == w.ins[b as usize]
}

/// `cse`: one sweep; each later duplicate merges into the first node with the same label and inputs.
/// Matched: `[kept, removed]`.
fn cse_sweep(w: &mut Work, ledger: &mut Vec<Step>) -> Result<bool, String> {
    let regions = in_regions(w);
    let mut first: HashMap<CseKey, u32> = HashMap::new();
    let mut pairs = Vec::new();
    for v in w.live().collect::<Vec<_>>() {
        let Some(x) = w.node(v) else { continue };
        if protected(w, v) || regions[v as usize] || has_params(x) || !x.splats.is_empty() {
            continue;
        }
        let key = (
            postcard::to_stdvec(&x.label).unwrap_or_default(),
            w.ins[v as usize].clone(),
        );
        match first.get(&key) {
            Some(&a) => pairs.push((a, v)),
            None => {
                first.insert(key, v);
            }
        }
    }
    let mut changed = false;
    for (a, b) in pairs {
        let regions = in_regions(w);
        if !cse_ok(w, a, b, &regions) {
            continue;
        }
        let nres = w.node(b).map(|x| x.label.results.len()).unwrap_or(0) as u32;
        let mut ed = Vec::new();
        for r in 0..nres {
            redirect(w, b, r, a, r, &mut ed);
        }
        drop_node(w, b, &mut ed);
        let s = step("cse", vec![a, b], ed);
        apply(w, &s)?;
        ledger.push(s);
        changed = true;
    }
    Ok(changed)
}

pub fn apply(w: &mut Work, s: &Step) -> Result<(), String> {
    for e in &s.edits {
        w.apply(e)
            .map_err(|m| format!("{} {:?}: {m}", s.rule, s.matched))?;
    }
    Ok(())
}

/// The matcher of a (non-CSE) rule at node `n`.
pub fn try_rule(rule: &str, w: &Work, n: u32) -> Option<Step> {
    match rule {
        "dce" => dce(w, n),
        "transpose_identity" => transpose_identity(w, n),
        "broadcast_identity" => broadcast_identity(w, n),
        "convert_identity" => convert_identity(w, n),
        "transpose_collapse" => transpose_collapse(w, n),
        "reshape_collapse" => reshape_collapse(w, n),
        "splat_fold" => splat_fold(w, n),
        _ => None,
    }
}

/// Rewrite to a fixpoint: sweeps over the rules in order, each sweep over the nodes in ascending id.
/// Every step strictly shrinks the graph (a node or an edge end fewer, or a transpose chain shorter), so
/// the loop ends.
pub fn rewrite(w: &mut Work) -> Result<Vec<Step>, String> {
    let mut ledger = Vec::new();
    loop {
        let mut changed = false;
        for rule in RULES {
            if rule == "cse" {
                changed |= cse_sweep(w, &mut ledger)?;
                continue;
            }
            let ids: Vec<u32> = w.live().collect();
            for n in ids {
                if let Some(s) = try_rule(rule, w, n) {
                    apply(w, &s)?;
                    ledger.push(s);
                    changed = true;
                }
            }
        }
        if !changed {
            return Ok(ledger);
        }
    }
}

/// `target` is re-exported for the ω assignment.
pub fn op_target(n: &NNode) -> Option<&str> {
    target(n)
}
