//! Independent checks of iteration_space (docs/passes/iteration_space.md §6). They read the inputs and
//! the written artifacts; they never call the op table's iteration hooks, the skeleton builder, the
//! grouping, the mapping search, the router or the embedding search.
//!
//! - `complete`: every class has a skeleton, dependences and one routing entry; a template file exists
//!   exactly for the classes routed to a template.
//! - `<z>:skeleton`: reuse lists, the class (map / reduce / contraction) and `affine` recompute from the
//!   stored accesses; the core operands' accesses are the core's.
//! - `<z>:access`: the obligation regenerates from the stored skeleton and the representative kernel
//!   (`spec.rs`, the spec's index formulas), its stored result is `unsat` from the configured solver for
//!   this query, and the other solver also answers `unsat`.
//! - `<z>:dependence`: the groups recompute from incidence; every propagation vector `u` of an operand X
//!   satisfies `F_X u = 0`; the accumulation vector is the unit vector of the reduction, in ω's order.
//! - `<z>:template`: per candidate, `λ·d ≥ 1`, `λ·u ≠ 0`, `S u = 0`, `[λ; S]` nonsingular, `S d` on a
//!   nearest-neighbour link, delays `λ·d`, the stationary operand and the dataflow name; every valid
//!   (λ, u) is present; utilization and link bits recompute from the rows; the ranking holds; the
//!   mapping obligation regenerates and both solvers answer `unsat`.
//! - `<z>:routing`: the reuse numbers recompute from the rows, and the route follows from them, the
//!   class, uniformity and the template.
//! - `embedding`: every certificate: φ injective and tag-preserving, the extent rule, π injective and
//!   direction-preserving over the embedding operands, the row maps, and every guest row equal to its
//!   host row under the stored substitution (other host rows vanish).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use cc_core::{Check, VerifyReport};
use cc_ir::ops::iter::Tag;
use cc_pipeline::{PassContext, PassError};
use rayon::prelude::*;

use super::json::{
    Access, Dependences, Direction, Embedding, Factor, Group, ProofResult, Route, Routing, Row, SkelClass,
    Skeleton, TemplateFile, Value, VectorKind,
};
use super::{ITERATION_SPACE, IterationSpaceConfig, Loaded, load, proofs, spec};
use crate::kernel_classes::json::Row as IRow;
use crate::normalize::json::Order;
use crate::normalize::{query_hash, smt};

/// A polynomial: (dim, sorted factors) → coefficient.
type Poly = BTreeMap<(Option<u32>, Vec<Factor>), i64>;

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

fn used(a: &Access) -> BTreeSet<u32> {
    a.iter()
        .flat_map(|r| r.terms.iter().filter_map(|t| t.dim))
        .collect()
}

fn core_input(name: &str) -> bool {
    name == "lhs" || name == "rhs" || name.starts_with("input") || name.starts_with("updates")
}

fn op<'a>(s: &'a Skeleton, name: &str) -> Option<&'a super::json::Operand> {
    s.operands.iter().find(|o| o.core && o.name == name)
}

fn skeleton_notes(s: &Skeleton) -> Vec<String> {
    let mut n = Vec::new();
    let nd = s.dims.len() as u32;
    for o in &s.operands {
        let u = used(&o.access);
        let reuse: Vec<u32> = (0..nd).filter(|d| !u.contains(d)).collect();
        if reuse != o.reuse {
            n.push(format!(
                "{}: reuse {:?}, the access gives {reuse:?}",
                o.name, o.reuse
            ));
        }
        if u.iter().any(|&d| d >= nd) {
            n.push(format!("{}: an index names no dim", o.name));
        }
    }
    let affine = !s
        .accesses
        .iter()
        .flat_map(|a| a.results.iter().chain(a.operands.iter().flatten()))
        .flatten()
        .any(|r| r.data);
    if affine != s.affine {
        n.push(format!("affine is {}, the accesses give {affine}", s.affine));
    }
    let red = s.dims.iter().any(|d| d.tag == Tag::Red);
    let reused = s.operands.iter().any(|o| {
        o.core
            && o.direction == Direction::In
            && core_input(&o.name)
            && (0..nd).any(|d| s.dims[d as usize].tag == Tag::Par && !used(&o.access).contains(&d))
    });
    let class = match (red, reused && affine) {
        (false, _) => SkelClass::Map,
        (true, true) => SkelClass::Contraction,
        (true, false) => SkelClass::Reduce,
    };
    if class != s.class {
        n.push(format!("class {:?}, the accesses give {class:?}", s.class));
    }
    if let Some(c) = s.core_node {
        match s.accesses.iter().find(|a| a.node == c) {
            None => n.push("the core has no access".into()),
            Some(a) => {
                for o in s.operands.iter().filter(|o| o.core) {
                    let ok = match o.direction {
                        Direction::In => a.operands.iter().flatten().any(|x| *x == o.access),
                        Direction::Out => a.results.get(o.res as usize) == Some(&o.access),
                    };
                    if !ok {
                        n.push(format!("{}: not the core's access", o.name));
                    }
                }
            }
        }
    }
    n
}

fn dependence_notes(s: &Skeleton, d: &Dependences) -> Vec<String> {
    let mut n = Vec::new();
    let tree = matches!(s.omega.as_ref().map(|o| &o.order), Some(Order::Tree { .. }));
    let par: Vec<u32> = (0..s.dims.len() as u32)
        .filter(|&i| s.dims[i as usize].tag == Tag::Par)
        .collect();
    let red: Vec<u32> = (0..s.dims.len() as u32)
        .filter(|&i| s.dims[i as usize].tag == Tag::Red)
        .collect();
    let omega: Vec<u32> = s.omega.as_ref().map(|o| o.dims.clone()).unwrap_or_default();
    let in_omega_order = |v: &[u32]| {
        let pos: Vec<usize> = v
            .iter()
            .map(|x| omega.iter().position(|y| y == x).unwrap_or(usize::MAX))
            .collect();
        pos.windows(2).all(|w| w[0] <= w[1])
    };
    let unit = |space: &[Group], g: Group| -> Vec<i64> { space.iter().map(|&x| i64::from(x == g)).collect() };
    match s.class {
        SkelClass::Map => {
            if !d.uniform || !d.vectors.is_empty() || !d.space.is_empty() {
                n.push("a map has no dependences".into());
            }
        }
        SkelClass::Reduce if !s.affine => {
            if d.uniform {
                n.push("a data-dependent target cannot be uniform".into());
            }
        }
        SkelClass::Reduce => {
            let r = d.groups.get(&Group::Red).cloned().unwrap_or_default();
            let mut rs = r.clone();
            rs.sort_unstable();
            if rs != red || !in_omega_order(&r) {
                n.push("reduced dims not in ω's order".into());
            }
            if d.groups.get(&Group::Par).cloned().unwrap_or_default() != par {
                n.push("kept dims".into());
            }
            if d.uniform == tree {
                n.push(format!("uniform is {} with ω tree = {tree}", d.uniform));
            }
            if d.uniform {
                let ok = d.vectors.len() == 1
                    && d.vectors[0].kind == VectorKind::Accumulate
                    && d.space.last() == Some(&Group::Red)
                    && d.vectors[0].v == unit(&d.space, Group::Red);
                if !ok {
                    n.push("the accumulation vector".into());
                }
            }
        }
        SkelClass::Contraction => {
            let (Some(a), Some(b), Some(c)) = (op(s, "lhs"), op(s, "rhs"), op(s, "result")) else {
                n.push("no lhs, rhs, result".into());
                return n;
            };
            let (ua, ub, uc) = (used(&a.access), used(&b.access), used(&c.access));
            let mut groups: BTreeMap<Group, Vec<u32>> = BTreeMap::new();
            let mut other = Vec::new();
            for i in 0..s.dims.len() as u32 {
                let t = s.dims[i as usize].tag;
                let g = match (ua.contains(&i), ub.contains(&i), uc.contains(&i), t) {
                    (true, true, true, Tag::Par) => Some(Group::Batch),
                    (true, false, true, Tag::Par) => Some(Group::M),
                    (false, true, true, Tag::Par) => Some(Group::N),
                    (true, true, false, Tag::Red) => Some(Group::K),
                    _ => None,
                };
                match g {
                    Some(g) => groups.entry(g).or_default().push(i),
                    None => other.push(i),
                }
            }
            let mut stored = d.groups.clone();
            if let Some(k) = stored.get_mut(&Group::K) {
                if !in_omega_order(k) {
                    n.push("k not in ω's order".into());
                }
                k.sort_unstable();
            }
            if stored != groups || d.other != other {
                n.push(format!(
                    "groups {:?} / {:?}, incidence gives {groups:?} / {other:?}",
                    d.groups, d.other
                ));
            }
            let want_uniform = other.is_empty() && groups.contains_key(&Group::K) && !tree;
            if d.uniform != want_uniform {
                n.push(format!("uniform is {}, expected {want_uniform}", d.uniform));
            }
            if d.uniform {
                let space: Vec<Group> = [Group::M, Group::N, Group::K]
                    .into_iter()
                    .filter(|g| groups.contains_key(g))
                    .collect();
                if space != d.space {
                    n.push("space".into());
                }
                let mut want = Vec::new();
                if groups.contains_key(&Group::N) {
                    want.push(("lhs", VectorKind::Propagate, Group::N));
                }
                if groups.contains_key(&Group::M) {
                    want.push(("rhs", VectorKind::Propagate, Group::M));
                }
                want.push(("result", VectorKind::Accumulate, Group::K));
                if want.len() != d.vectors.len() {
                    n.push("vector count".into());
                }
                for (v, (name, kind, g)) in d.vectors.iter().zip(&want) {
                    if v.operand != *name || v.kind != *kind || v.v != unit(&space, *g) {
                        n.push(format!("vector of {}", v.operand));
                    }
                    // F_X u = 0: the operand is indexed by no dim of the group u moves along.
                    let x = op(s, &v.operand).map(|o| used(&o.access)).unwrap_or_default();
                    if groups.get(g).into_iter().flatten().any(|i| x.contains(i)) {
                        n.push(format!("{}: F u != 0", v.operand));
                    }
                }
            }
        }
    }
    n
}

/// A dim's extent at a row (`None` when unknown); the checker's own evaluation.
fn extent(s: &Skeleton, d: u32, r: &IRow, cols: &HashMap<String, usize>) -> Option<u64> {
    use super::json::Extent as E;
    let col = |c: &str, i: u32| -> Option<i64> {
        let v = r.values.get(*cols.get(c)?)?.as_ref()?;
        v.as_ints()
            .and_then(|x| x.get(i as usize).copied())
            .or_else(|| v.as_i64())
    };
    match &s.dims.get(d as usize)?.extent {
        E::Class { class } => r.extents.get(*class as usize).copied(),
        E::Const { value } => u64::try_from(*value).ok(),
        E::Column { column, index } => u64::try_from(col(column, *index)?).ok(),
        E::Runtime { .. } => None,
        E::Quot { class, by } => {
            let b = match by {
                Factor::Column { column, index } => u64::try_from(col(column, *index)?).ok()?,
                _ => return None,
            };
            (b > 0).then(|| r.extents.get(*class as usize).copied().unwrap_or(0) / b)
        }
    }
}

fn det(m: &[Vec<i64>]) -> i64 {
    match m.len() {
        1 => m[0][0],
        2 => m[0][0] * m[1][1] - m[0][1] * m[1][0],
        3 => {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        }
        _ => 0,
    }
}

fn dotp(a: &[i64], b: &[i64]) -> i64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn template_notes(
    s: &Skeleton,
    d: &Dependences,
    t: &TemplateFile,
    rows: &[IRow],
    cols: &HashMap<String, usize>,
) -> Vec<String> {
    let mut n = Vec::new();
    let k = t.space.len();
    if t.space != d.space {
        n.push("space differs from the dependences".into());
    }
    let deps: Vec<(&str, &[i64])> = d
        .vectors
        .iter()
        .map(|v| (v.operand.as_str(), v.v.as_slice()))
        .collect();
    let weight = |name: &str| -> bool {
        op(s, name).is_some_and(|o| {
            rows.iter().all(|r| {
                o.axes
                    .iter()
                    .all(|&c| r.sizes.get(c as usize).is_some_and(|x| x.parse::<i64>().is_ok()))
            })
        })
    };
    // Per row: weight, group extents, widths.
    let groups: Vec<Vec<u32>> = t
        .space
        .iter()
        .map(|g| t.groups.get(g).cloned().unwrap_or_default())
        .collect();
    let mut data: Vec<(f64, Vec<u64>, &IRow)> = Vec::new();
    for r in rows.iter().filter(|r| r.weighted > 0) {
        let e: Option<Vec<u64>> = groups
            .iter()
            .map(|g| {
                g.iter()
                    .try_fold(1u64, |a, &x| a.checked_mul(extent(s, x, r, cols)?))
            })
            .collect();
        if let Some(e) = e {
            data.push((r.weighted as f64, e, r));
        }
    }
    let total: f64 = data.iter().map(|x| x.0).sum();
    let width = |r: &IRow, name: &str| -> u32 {
        op(s, name)
            .and_then(|o| r.elems.get(o.elem as usize))
            .and_then(|e| cc_ir::Elem::parse(e))
            .map_or(0, |e| e.bits())
    };
    let mut seen = BTreeSet::new();
    for c in &t.candidates {
        let tag = format!("candidate {}", c.rank);
        let ok3 = |v: &[i64]| v.len() == k && v.iter().all(|x| x.abs() <= 1);
        if !ok3(&c.lambda) || !ok3(&c.u) {
            n.push(format!("{tag}: entries"));
            continue;
        }
        seen.insert((c.lambda.clone(), c.u.clone()));
        if deps.iter().any(|(_, v)| dotp(&c.lambda, v) < 1) {
            n.push(format!("{tag}: λ·d < 1"));
        }
        if c.u.iter().find(|&&x| x != 0).is_none_or(|&x| x < 0) || dotp(&c.lambda, &c.u) == 0 {
            n.push(format!("{tag}: u"));
        }
        let mut m = vec![c.lambda.clone()];
        m.extend(c.s.iter().cloned());
        if c.s.len() + 1 != k || c.s.iter().any(|r| r.len() != k || dotp(r, &c.u) != 0) || det(&m) == 0 {
            n.push(format!("{tag}: allocation"));
        }
        if c.links.len() != deps.len() {
            n.push(format!("{tag}: links"));
        }
        for (l, (name, v)) in c.links.iter().zip(&deps) {
            let step: Vec<i64> = c.s.iter().map(|r| dotp(r, v)).collect();
            if l.operand != *name
                || l.step != step
                || step.iter().any(|x| x.abs() > 1)
                || l.delay != dotp(&c.lambda, v)
            {
                n.push(format!("{tag}: link of {name}"));
            }
        }
        let st = c
            .links
            .iter()
            .find(|l| l.step.iter().all(|&x| x == 0))
            .map(|l| l.operand.clone());
        let flow = match st.as_deref() {
            None => "systolic",
            Some("result") => "output_stationary",
            Some(x) if weight(x) => "weight_stationary",
            Some(_) => "input_stationary",
        };
        if c.stationary != st || c.dataflow != flow {
            n.push(format!("{tag}: stationarity"));
        }
        let (mut u, mut b) = (0.0, 0.0);
        for (w, e, r) in &data {
            let all: u128 = e.iter().map(|&x| u128::from(x)).product();
            let shifted: u128 = e
                .iter()
                .zip(&c.u)
                .map(|(&x, &y)| u128::from(x.saturating_sub(y.unsigned_abs())))
                .product();
            let pes = (all - shifted).max(1);
            let tt: u128 = e
                .iter()
                .zip(&c.lambda)
                .map(|(&x, &l)| u128::from(l.unsigned_abs()) * u128::from(x.saturating_sub(1)))
                .sum::<u128>()
                + 1;
            u += w * (all as f64) / ((pes as f64) * (tt as f64));
            let bits: u32 = c
                .links
                .iter()
                .filter(|l| l.step.iter().any(|&x| x != 0))
                .map(|l| width(r, &l.operand))
                .sum();
            b += w * f64::from(bits);
        }
        if total > 0.0 {
            u /= total;
            b /= total;
        }
        if (u - c.utilization).abs() > 1e-5 || (b - c.link_bits).abs() > 1e-5 {
            n.push(format!(
                "{tag}: metrics {u} {b} vs {} {}",
                c.utilization, c.link_bits
            ));
        }
    }
    for (i, w) in t.candidates.windows(2).enumerate() {
        let (a, b) = (&w[0], &w[1]);
        let ordered = a.utilization > b.utilization
            || (a.utilization == b.utilization
                && (a.link_bits < b.link_bits
                    || (a.link_bits == b.link_bits
                        && (a.lambda.clone(), a.u.clone()) < (b.lambda.clone(), b.u.clone()))));
        if !ordered || a.rank != i as u32 || b.rank != i as u32 + 1 {
            n.push(format!("ranking at {i}"));
        }
    }
    if t.selected != 0 {
        n.push("the selected candidate is not the first".into());
    }
    if seen.len() != t.candidates.len() {
        n.push("a candidate repeats".into());
    }
    if t.candidates.is_empty() {
        n.push("a template without a candidate".into());
    }
    n
}

/// The checker's own substitution of a host row into guest terms.
fn host_poly(r: &Row, phi_inv: &[Option<u32>], sigma: &BTreeMap<Factor, Value>) -> Option<Poly> {
    let mut p = Poly::new();
    'terms: for t in &r.terms {
        let dim = match t.dim {
            None => None,
            Some(h) => match phi_inv.get(h as usize).copied().flatten() {
                Some(g) => Some(g),
                None => continue,
            },
        };
        let mut f = Vec::new();
        for x in &t.factors {
            match x {
                Factor::Extent { dim } => {
                    if let Some(g) = phi_inv.get(*dim as usize).copied().flatten() {
                        f.push(Factor::Extent { dim: g });
                    }
                }
                _ => match sigma.get(x)? {
                    Value::Zero => continue 'terms,
                    Value::One => {}
                    Value::Guest(y) => f.push(y.clone()),
                },
            }
        }
        f.sort();
        *p.entry((dim, f)).or_default() += t.k;
    }
    p.retain(|_, k| *k != 0);
    Some(p)
}

fn guest_poly(r: &Row) -> Poly {
    let mut p = Poly::new();
    for t in &r.terms {
        let mut f = t.factors.clone();
        f.sort();
        *p.entry((t.dim, f)).or_default() += t.k;
    }
    p.retain(|_, k| *k != 0);
    p
}

fn emb_ops(s: &Skeleton) -> Vec<usize> {
    let core = s.class != SkelClass::Map;
    (0..s.operands.len())
        .filter(|&i| s.operands[i].core == core)
        .collect()
}

fn embedding_notes(e: &Embedding, skels: &HashMap<&str, &Skeleton>) -> Vec<String> {
    e.pairs
        .par_iter()
        .filter_map(|p| {
            let tag = format!("{} ⊑ {}", p.guest, p.host);
            let (Some(g), Some(h)) = (skels.get(p.guest.as_str()), skels.get(p.host.as_str())) else {
                return Some(format!("{tag}: unknown class"));
            };
            if g.class != h.class || p.guest == p.host {
                return Some(format!("{tag}: classes"));
            }
            if p.dims.len() != g.dims.len() {
                return Some(format!("{tag}: φ length"));
            }
            let mut inv = vec![None; h.dims.len()];
            for (gi, &hd) in p.dims.iter().enumerate() {
                let Some(slot) = inv.get_mut(hd as usize) else {
                    return Some(format!("{tag}: φ range"));
                };
                if slot.is_some() {
                    return Some(format!("{tag}: φ not injective"));
                }
                *slot = Some(gi as u32);
                if h.dims[hd as usize].tag != g.dims[gi].tag {
                    return Some(format!("{tag}: φ changes a tag"));
                }
            }
            for (a, &ha) in p.dims.iter().enumerate() {
                for (b, &hb) in p.dims.iter().enumerate() {
                    if h.dims[ha as usize].extent == h.dims[hb as usize].extent
                        && g.dims[a].extent != g.dims[b].extent
                    {
                        return Some(format!("{tag}: shared host extent"));
                    }
                }
            }
            for (hd, m) in inv.iter().enumerate() {
                if m.is_none()
                    && p.dims
                        .iter()
                        .any(|&x| h.dims[x as usize].extent == h.dims[hd].extent)
                {
                    return Some(format!("{tag}: a dim fixed to 1 shares an extent"));
                }
            }
            let (gops, hops) = (emb_ops(g), emb_ops(h));
            if p.operands.len() != gops.len() || p.rows.len() != gops.len() {
                return Some(format!("{tag}: π length"));
            }
            let sigma: BTreeMap<Factor, Value> = p.params.iter().cloned().collect();
            let mut taken = BTreeSet::new();
            for (i, &gi) in gops.iter().enumerate() {
                let ho = p.operands[i] as usize;
                if !hops.contains(&ho) || !taken.insert(ho) {
                    return Some(format!("{tag}: π"));
                }
                let (go, hop) = (&g.operands[gi], &h.operands[ho]);
                if go.direction != hop.direction || p.rows[i].len() != go.access.len() {
                    return Some(format!("{tag}: operand {}", go.name));
                }
                let mut used_rows = BTreeSet::new();
                for (gr, &hr) in go.access.iter().zip(&p.rows[i]) {
                    let Some(hrow) = hop.access.get(hr as usize) else {
                        return Some(format!("{tag}: row range"));
                    };
                    if !used_rows.insert(hr) || hrow.data != gr.data {
                        return Some(format!("{tag}: row map"));
                    }
                    if host_poly(hrow, &inv, &sigma) != Some(guest_poly(gr)) {
                        return Some(format!("{tag}: {} differs", go.name));
                    }
                    let hd = hrow.divisor.as_ref().map(|f| match sigma.get(f) {
                        Some(Value::Guest(x)) => Some(x.clone()),
                        _ => None,
                    });
                    if hd.flatten() != gr.divisor {
                        return Some(format!("{tag}: {} divisor", go.name));
                    }
                }
                for (j, r) in hop.access.iter().enumerate() {
                    if !used_rows.contains(&(j as u32))
                        && (r.data || host_poly(r, &inv, &sigma).is_none_or(|x| !x.is_empty()))
                    {
                        return Some(format!("{tag}: {} host row {j} does not vanish", go.name));
                    }
                }
            }
            None
        })
        .collect()
}

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: IterationSpaceConfig = ctx.options()?;
    let out = ctx.output(ITERATION_SPACE)?;
    let Loaded {
        classes,
        tables,
        graphs,
    } = load(ctx)?;
    let mut r = VerifyReport::new(ctx.name());

    let ids: Vec<String> = classes.iter().map(|c| c.id.clone()).collect();
    let routing: Routing = out.read_json("routing.json")?;
    let mut complete = Vec::new();
    let routed: BTreeMap<&str, &super::json::Routed> =
        routing.classes.iter().map(|x| (x.id.as_str(), x)).collect();
    if routing.classes.len() != ids.len() || routed.len() != ids.len() {
        complete.push("routing does not list every class once".to_string());
    }
    let skels: Vec<Option<Skeleton>> = ids
        .par_iter()
        .map(|id| out.read_json(&format!("skeletons/{id}.json")).ok())
        .collect();
    let deps: Vec<Option<Dependences>> = ids
        .par_iter()
        .map(|id| out.read_json(&format!("dependences/{id}.json")).ok())
        .collect();
    let temps: Vec<Option<TemplateFile>> = ids
        .par_iter()
        .map(|id| out.read_json(&format!("templates/{id}.json")).ok())
        .collect();
    for (i, id) in ids.iter().enumerate() {
        if skels[i].is_none() || deps[i].is_none() {
            complete.push(format!("{id}: no skeleton or dependences"));
        }
        let is_t = routed
            .get(id.as_str())
            .is_some_and(|x| x.route == Route::Template);
        if is_t != temps[i].is_some() {
            complete.push(format!("{id}: template file and route disagree"));
        }
    }
    r.push(check("complete", &complete, &[("classes", ids.len() as u64)]));

    // Obligations, regenerated.
    let regen: Vec<(Result<String, String>, Option<String>)> = (0..ids.len())
        .into_par_iter()
        .map(|i| {
            let Some(s) = &skels[i] else {
                return (Err("no skeleton".into()), None);
            };
            let a = spec::claims(&classes[i], s, &graphs[i]).and_then(|c| proofs::access(&ids[i], &c));
            let m = match (&temps[i], &deps[i]) {
                (Some(t), Some(d)) => {
                    let vs: Vec<(String, Vec<i64>)> = d
                        .vectors
                        .iter()
                        .map(|v| (v.operand.clone(), v.v.clone()))
                        .collect();
                    Some(proofs::mapping(t, &vs))
                }
                _ => None,
            };
            (a, m)
        })
        .collect();
    let mut files = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        files.push(format!("access.{id}.smt2"));
        if regen[i].1.is_some() {
            files.push(format!("mapping.{id}.smt2"));
        }
    }
    let (solver, runner, location) = smt(ctx)?;
    let other = cc_smt::run_batch(
        &runner,
        &location,
        solver.other(),
        &out.path("smt")?,
        &files,
        ctx.smt().timeout_s,
    );
    let proof_notes = |file: &str, text: &str| -> Vec<String> {
        let mut n = Vec::new();
        let stored =
            std::fs::read_to_string(out.path(&format!("smt/{file}")).unwrap_or_default()).unwrap_or_default();
        if stored != text {
            n.push(format!("{file}: differs from the regenerated obligation"));
        }
        match out.read_json::<ProofResult>(&format!("smt/{}", file.replace(".smt2", ".result"))) {
            Ok(p) => {
                if p.query_hash != query_hash(text) || p.solver != solver || !p.verdict.is_unsat() {
                    n.push(format!("{file}: stored result {:?} {:?}", p.solver, p.verdict));
                }
            }
            Err(e) => n.push(format!("{file}: no result: {e}")),
        }
        match &other {
            Ok(b) => match b.verdicts.get(file) {
                Some(v) if v.is_unsat() => {}
                v => n.push(format!("{file}: {} re-check {v:?}", solver.other().name())),
            },
            Err(e) => n.push(format!("{} re-check failed: {e}", solver.other().name())),
        }
        n
    };

    let cfg_t = &cfg.template;
    let per: Vec<Vec<Check>> = (0..ids.len())
        .into_par_iter()
        .map(|i| {
            let id = &ids[i];
            let mut cs = Vec::new();
            let (Some(s), Some(d)) = (&skels[i], &deps[i]) else {
                return cs;
            };
            cs.push(check(
                format!("{id}:skeleton"),
                &skeleton_notes(s),
                &[("dims", s.dims.len() as u64)],
            ));
            let mut an = Vec::new();
            match &regen[i].0 {
                Ok(text) => an.extend(proof_notes(&format!("access.{id}.smt2"), text)),
                Err(e) => an.push(format!("obligation: {e}")),
            }
            cs.push(
                check(format!("{id}:access"), &an, &[])
                    .path(format!("out/iteration_space/smt/access.{id}.smt2")),
            );
            cs.push(check(format!("{id}:dependence"), &dependence_notes(s, d), &[]));
            let cols: HashMap<String, usize> = classes[i]
                .params
                .values
                .iter()
                .enumerate()
                .map(|(j, c)| (c.clone(), j))
                .collect();
            if let Some(t) = &temps[i] {
                let mut tn = template_notes(s, d, t, &tables[i].rows, &cols);
                if let Some(m) = &regen[i].1 {
                    tn.extend(proof_notes(&format!("mapping.{id}.smt2"), m));
                }
                cs.push(check(
                    format!("{id}:template"),
                    &tn,
                    &[("candidates", t.candidates.len() as u64)],
                ));
            }
            // Routing, from the rows.
            let mut rn = Vec::new();
            if let Some(x) = routed.get(id.as_str()) {
                let mut want = Route::Datapath;
                if s.class == SkelClass::Contraction && op(s, "lhs").is_some() && op(s, "rhs").is_some() {
                    // Reuse in the recurrence: lhs along N, rhs along M (the groups are checked above).
                    let empty = Vec::new();
                    let gn = d.groups.get(&Group::N).unwrap_or(&empty);
                    let gm = d.groups.get(&Group::M).unwrap_or(&empty);
                    let (mut total, mut good) = (0.0f64, 0.0f64);
                    for row in tables[i].rows.iter().filter(|r| r.weighted > 0) {
                        let w = row.weighted as f64;
                        total += w;
                        let prod = |g: &[u32]| -> Option<u64> {
                            g.iter()
                                .try_fold(1u64, |acc, &dd| acc.checked_mul(extent(s, dd, row, &cols)?))
                        };
                        if let (Some(fa), Some(fb)) = (prod(gn), prod(gm))
                            && fa >= cfg_t.min_reuse
                            && fb >= cfg_t.min_reuse
                        {
                            good += w;
                        }
                    }
                    let share = if total > 0.0 { good / total } else { 0.0 };
                    if x.reuse.as_ref().is_none_or(|q| (q.share - share).abs() > 1e-5) {
                        rn.push(format!(
                            "share {share}, stored {:?}",
                            x.reuse.as_ref().map(|q| q.share)
                        ));
                    }
                    // A uniform contraction always has a valid mapping: λ = (1, …, 1), u = e_k (output
                    // stationary) is causal, conflict-free and local for the unit dependence vectors.
                    if cfg_t.enabled
                        && d.uniform
                        && d.space.len() <= cfg_t.max_dims as usize
                        && total > 0.0
                        && share >= cfg_t.min_share
                    {
                        want = Route::Template;
                    }
                }
                if x.route != want || x.class != s.class {
                    rn.push(format!("route {:?}, expected {want:?}", x.route));
                }
                if x.route == Route::Template {
                    let df = temps[i]
                        .as_ref()
                        .and_then(|t| t.candidates.get(t.selected as usize))
                        .map(|c| c.dataflow.clone());
                    if x.dataflow != df || x.stages.as_ref() != Some(&s.stages) {
                        rn.push("dataflow or stages".into());
                    }
                }
            } else {
                rn.push("not routed".into());
            }
            cs.push(check(format!("{id}:routing"), &rn, &[]));
            cs
        })
        .collect();
    for cs in per {
        for c in cs {
            r.push(c);
        }
    }

    // Embedding certificates.
    let e: Embedding = out.read_json("embedding.json")?;
    let map: HashMap<&str, &Skeleton> = skels.iter().flatten().map(|s| (s.id.as_str(), s)).collect();
    let en = embedding_notes(&e, &map);
    r.push(check(
        "embedding",
        &en,
        &[("pairs", e.pairs.len() as u64), ("failed", en.len() as u64)],
    ));
    Ok(r)
}
