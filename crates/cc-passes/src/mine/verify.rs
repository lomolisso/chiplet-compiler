//! Independent checks of the mining (design §5, docs/passes/mine.md §6). They never call the miner's
//! code (`algo`): units, labels, the `.lg` reading, canonical codes, matching, convexity and the
//! independent-set bound are written again here.
//!
//! - `labels`: the label tables are exactly the node keys and merged edge ends of the units;
//! - `<f>:lg`: the `.lg` graph and its map translate the units (every vertex and edge, nothing more);
//! - `<f>:grami`: GraMi's final list (re-read from the cache when present) is the stored one, and every
//!   pattern within `max_vertices` has a pattern file;
//! - `<f>:patterns`: every pattern's code is canonical and its id the code's hash; a backtracking matcher
//!   finds the same embeddings, the same MNI (≥ τ) and exactly the stored convex occurrences (by a
//!   forward/backward reachability test); operand feeds recompute; every MIS is independent (a mark
//!   array), its size and bound agree with an exact search, and the score recomputes;
//! - `<f>:selection`, `<f>:sweep`: the top-k and the stop rule recompute;
//! - `overlap`, `summary`: recount.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use cc_core::{Check, KindDir, VerifyReport};
use cc_ir::absgraph::{EdgeEnd, EdgeLabel, PortLabel, hash_bytes};
use cc_pipeline::{PassContext, PassError};
use rayon::prelude::*;

use super::json::{
    Family, Feed, GramiFile, Labels, LgMap, Overlap, PatternFile, Selection, Stop, Summary, Sweep,
};
use super::{Inputs, MINE, MineConfig, load};
use crate::datapaths::json::{Src, Stage};
use crate::iteration_space::json::{Direction, Route, SkelClass};

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

/// A unit as verify rebuilds it.
struct VUnit {
    id: String,
    class: String,
    stage: Option<Stage>,
    family: Family,
    workload: u128,
    /// Datapath vertex ids (non-constant), in datapath order.
    verts: Vec<u32>,
    keys: Vec<String>,
    /// (src, dst) local → sorted ends.
    edges: BTreeMap<(u32, u32), Vec<EdgeEnd>>,
    /// Per local vertex: per operand, its source.
    args: Vec<Vec<Src>>,
    consts: HashSet<u32>,
}

fn key(v: &crate::datapaths::json::Vertex) -> String {
    let mut extra = Vec::new();
    for k in ["comparison_direction", "compare_type"] {
        if let Some(x) = v.attrs.get(k) {
            extra.push(x.as_str());
        }
    }
    if extra.is_empty() {
        v.label.clone()
    } else {
        format!("{}[{}]", v.label, extra.join(","))
    }
}

fn end(res: u32, port: &str) -> Result<EdgeEnd, String> {
    let port = match port {
        "*" => PortLabel::Commutative,
        p => PortLabel::Port(
            p.strip_prefix('p')
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| format!("bad port {p:?}"))?,
        ),
    };
    Ok(EdgeEnd { res, port })
}

fn vunits(inp: &Inputs) -> Result<Vec<VUnit>, String> {
    let mut out = Vec::new();
    for dp in &inp.datapaths {
        let table = inp.tables.get(&dp.id).ok_or_else(|| format!("{}: no instances", dp.id))?;
        let consts: HashSet<u32> = dp.vertices.iter().filter(|v| v.op == "const").map(|v| v.id).collect();
        let mut groups: BTreeMap<Option<Stage>, Vec<u32>> = BTreeMap::new();
        for v in dp.vertices.iter().filter(|v| v.op != "const") {
            let g = match dp.route {
                Route::Datapath => None,
                Route::Template => Some(v.stage),
            };
            groups.entry(g).or_default().push(v.id);
        }
        for (stage, verts) in groups {
            let (id, family, workload) = match stage {
                None => (
                    dp.id.clone(),
                    match dp.class {
                        SkelClass::Map => Family::Map,
                        SkelClass::Reduce => Family::Reduce,
                        SkelClass::Contraction => Family::Contraction,
                    },
                    table.rows.iter().map(|r| r.weighted as u128).sum::<u128>(),
                ),
                Some(st) => {
                    let skel = inp.skeletons.get(&dp.id).ok_or_else(|| format!("{}: no skeleton", dp.id))?;
                    let (name, operand) = match st {
                        Stage::Prologue { port } => (
                            format!("{}/prologue{port}", dp.id),
                            skel.operands
                                .iter()
                                .filter(|o| o.core && o.direction == Direction::In)
                                .nth(port as usize),
                        ),
                        Stage::Epilogue => (
                            format!("{}/epilogue", dp.id),
                            skel.operands.iter().find(|o| o.core && o.direction == Direction::Out),
                        ),
                        s => return Err(format!("{}: template vertex in {s:?}", dp.id)),
                    };
                    let o = operand.ok_or_else(|| format!("{name}: no core operand"))?;
                    let mut w = 0u128;
                    for r in &table.rows {
                        let mut e = r.trip_count as u128;
                        for &a in &o.axes {
                            e = e.saturating_mul(*r.extents.get(a as usize).ok_or("axis not in row")? as u128);
                        }
                        w = w.saturating_add(e);
                    }
                    (name, Family::Stage, w)
                }
            };
            let local: HashMap<u32, u32> = verts.iter().enumerate().map(|(i, v)| (*v, i as u32)).collect();
            let by: HashMap<u32, &crate::datapaths::json::Vertex> = dp.vertices.iter().map(|v| (v.id, v)).collect();
            let mut keys = Vec::new();
            let mut args = Vec::new();
            let mut edges: BTreeMap<(u32, u32), Vec<EdgeEnd>> = BTreeMap::new();
            for (i, v) in verts.iter().enumerate() {
                let x = by[v];
                keys.push(key(x));
                args.push(x.args.iter().map(|a| a.src.clone()).collect());
                for a in &x.args {
                    if let Src::Vertex { v: u, res } = &a.src
                        && let Some(&j) = local.get(u)
                    {
                        edges.entry((j, i as u32)).or_default().push(end(*res, &a.port)?);
                    }
                }
            }
            for e in edges.values_mut() {
                e.sort();
            }
            out.push(VUnit {
                id,
                class: dp.id.clone(),
                stage,
                family,
                workload,
                verts,
                keys,
                edges,
                args,
                consts: consts.clone(),
            });
        }
    }
    Ok(out)
}

/// The canonical code by plain search over every order that sorts the labels.
fn code_of(labels: &[String], edges: &[(u32, u32, EdgeLabel)]) -> String {
    let n = labels.len();
    let text = |l: &EdgeLabel| -> String {
        l.iter()
            .map(|e| match e.port {
                PortLabel::Port(i) => format!("{}p{i}", e.res),
                PortLabel::Commutative => format!("{}*", e.res),
                p => format!("{}{p:?}", e.res),
            })
            .collect::<Vec<_>>()
            .join("+")
    };
    let mut sorted: Vec<&String> = labels.iter().collect();
    sorted.sort();
    let mut best: Option<Vec<(u32, u32, String)>> = None;
    // Every permutation, keeping those whose label sequence is sorted.
    let mut perm: Vec<usize> = (0..n).collect();
    let mut c = vec![0usize; n];
    let mut consider = |perm: &[usize]| {
        // perm[pos] = vertex
        if perm.iter().map(|&v| &labels[v]).ne(sorted.iter().copied()) {
            return;
        }
        let mut pos = vec![0u32; n];
        for (p, &v) in perm.iter().enumerate() {
            pos[v] = p as u32;
        }
        let mut e: Vec<(u32, u32, String)> = edges
            .iter()
            .map(|(a, b, l)| (pos[*a as usize], pos[*b as usize], text(l)))
            .collect();
        e.sort();
        if best.as_ref().is_none_or(|b| e < *b) {
            best = Some(e);
        }
    };
    consider(&perm);
    // Heap's algorithm.
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i % 2 == 0 {
                perm.swap(0, i);
            } else {
                perm.swap(c[i], i);
            }
            consider(&perm);
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    let e = best.unwrap_or_default();
    let ls: Vec<&str> = sorted.iter().map(|s| s.as_str()).collect();
    let es: Vec<String> = e.iter().map(|(a, b, l)| format!("{a}>{b}:{l}")).collect();
    format!("{};{}", ls.join(","), es.join(","))
}

fn hash16(s: &str) -> String {
    format!("{:016x}", (hash_bytes(s.as_bytes()) >> 64) as u64)
}

/// Every embedding of the pattern into the unit: (map, as local ids).
fn embeddings(p: &PatternFile, u: &VUnit) -> Vec<Vec<u32>> {
    let n = p.labels.len();
    let pe: HashMap<(u32, u32), &EdgeLabel> = p.edges.iter().map(|e| ((e.src, e.dst), &e.ends)).collect();
    let mut out = Vec::new();
    let mut map: Vec<u32> = Vec::with_capacity(n);
    let mut used = vec![false; u.verts.len()];
    fn rec(
        p: &PatternFile,
        u: &VUnit,
        pe: &HashMap<(u32, u32), &EdgeLabel>,
        map: &mut Vec<u32>,
        used: &mut [bool],
        out: &mut Vec<Vec<u32>>,
    ) {
        let i = map.len();
        if i == p.labels.len() {
            out.push(map.clone());
            return;
        }
        for h in 0..u.verts.len() as u32 {
            if used[h as usize] || u.keys[h as usize] != p.labels[i] {
                continue;
            }
            let ok = (0..i as u32).all(|j| {
                let hj = map[j as usize];
                let f = pe.get(&(j, i as u32)).is_none_or(|l| u.edges.get(&(hj, h)) == Some(*l));
                let b = pe.get(&(i as u32, j)).is_none_or(|l| u.edges.get(&(h, hj)) == Some(*l));
                f && b
            });
            if ok {
                used[h as usize] = true;
                map.push(h);
                rec(p, u, pe, map, used, out);
                map.pop();
                used[h as usize] = false;
            }
        }
    }
    rec(p, u, &pe, &mut map, &mut used, &mut out);
    out
}

/// Convex: no vertex outside `s` is both reachable from `s` and reaching `s`.
fn convex(u: &VUnit, s: &[u32]) -> bool {
    let n = u.verts.len();
    let mut fwd: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut bwd: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(a, b) in u.edges.keys() {
        fwd[a as usize].push(b);
        bwd[b as usize].push(a);
    }
    let reach = |adj: &Vec<Vec<u32>>| {
        let mut seen = vec![false; n];
        let mut stack: Vec<u32> = s.to_vec();
        while let Some(v) = stack.pop() {
            for &w in &adj[v as usize] {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    stack.push(w);
                }
            }
        }
        seen
    };
    let (f, b) = (reach(&fwd), reach(&bwd));
    let inside: HashSet<u32> = s.iter().copied().collect();
    (0..n).all(|w| inside.contains(&(w as u32)) || !(f[w] && b[w]))
}

/// α of the graph on `verts` (adjacency sets), by include/exclude on a vertex of maximum degree;
/// `None` over `budget` calls.
fn alpha(adj: &[HashSet<usize>], budget: &mut u64) -> Option<usize> {
    fn rec(adj: &[HashSet<usize>], live: &mut Vec<bool>, budget: &mut u64) -> Option<usize> {
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        let alive: Vec<usize> = (0..adj.len()).filter(|&v| live[v]).collect();
        if alive.is_empty() {
            return Some(0);
        }
        let deg = |v: usize, live: &Vec<bool>| adj[v].iter().filter(|&&w| live[w]).count();
        let v = alive.iter().copied().max_by_key(|&v| deg(v, live)).unwrap_or(0);
        live[v] = false;
        if deg(v, live) == 0 {
            let r = rec(adj, live, budget).map(|x| x + 1);
            live[v] = true;
            return r;
        }
        let without = rec(adj, live, budget);
        let nb: Vec<usize> = adj[v].iter().copied().filter(|&w| live[w]).collect();
        for &w in &nb {
            live[w] = false;
        }
        let with = rec(adj, live, budget).map(|x| x + 1);
        for &w in &nb {
            live[w] = true;
        }
        live[v] = true;
        Some(without?.max(with?))
    }
    rec(adj, &mut vec![true; adj.len()], budget)
}

struct Ctx<'a> {
    cfg: &'a MineConfig,
    out: &'a KindDir,
    units: Vec<VUnit>,
    labels: Option<Labels>,
}

pub fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: MineConfig = ctx.options()?;
    let inp = load(ctx)?;
    let out = ctx.output(MINE)?;
    let mut rep = VerifyReport::new("mine");
    let units = vunits(&inp).map_err(PassError::failed)?;
    let labels: Option<Labels> = out.read_json("labels.json").ok();
    let c = Ctx {
        cfg: &cfg,
        out,
        units,
        labels,
    };
    rep.push(labels_check(&c));
    let mut all_patterns: Vec<PatternFile> = Vec::new();
    let mut sels: BTreeMap<Family, Selection> = BTreeMap::new();
    let mut sweeps: BTreeMap<Family, Sweep> = BTreeMap::new();
    let mut maps: BTreeMap<Family, LgMap> = BTreeMap::new();
    for f in Family::ALL {
        let (lg, map) = lg_check(&c, ctx, f);
        rep.push(lg);
        let sweep: Option<Sweep> = out.read_json(&format!("sweep/{}.json", f.name())).ok();
        let grami: Option<GramiFile> = out.read_json(&format!("grami/{}.json", f.name())).ok();
        let pats = read_patterns(out, f);
        rep.push(grami_check(&c, ctx, f, grami.as_ref(), sweep.as_ref(), &pats, map.as_ref()));
        rep.push(patterns_check(&c, f, sweep.as_ref(), &pats));
        let sel: Option<Selection> = out.read_json(&format!("selection/{}.json", f.name())).ok();
        rep.push(selection_check(&c, f, sel.as_ref(), &pats));
        rep.push(sweep_check(&c, f, sweep.as_ref(), sel.as_ref(), map.as_ref()));
        all_patterns.extend(pats.into_iter().filter_map(Result::ok));
        if let Some(s) = sel {
            sels.insert(f, s);
        }
        if let Some(s) = sweep {
            sweeps.insert(f, s);
        }
        if let Some(m) = map {
            maps.insert(f, m);
        }
    }
    rep.push(overlap_check(out, &all_patterns, &sels));
    rep.push(summary_check(&c, &all_patterns, &sels, &sweeps, &maps));
    Ok(rep)
}

fn labels_check(c: &Ctx) -> Check {
    let mut notes = Vec::new();
    let mut nodes = BTreeSet::new();
    let mut edges: BTreeSet<EdgeLabel> = BTreeSet::new();
    for u in &c.units {
        nodes.extend(u.keys.iter().cloned());
        edges.extend(u.edges.values().cloned());
    }
    match &c.labels {
        None => notes.push("labels.json is missing or unreadable".into()),
        Some(l) => {
            if l.nodes != nodes.iter().cloned().collect::<Vec<_>>() {
                notes.push("node labels are not the units' keys".into());
            }
            if l.edges != edges.iter().cloned().collect::<Vec<_>>() {
                notes.push("edge labels are not the units' merged ends".into());
            }
        }
    }
    check(
        "labels",
        &notes,
        &[("node_labels", nodes.len() as u64), ("edge_labels", edges.len() as u64)],
    )
}

fn lg_check(c: &Ctx, ctx: &PassContext, f: Family) -> (Check, Option<LgMap>) {
    let name = format!("{}:lg", f.name());
    let mut notes = Vec::new();
    let map: Option<LgMap> = c.out.read_json(&format!("lg/{}.map.json", f.name())).ok();
    let text = std::fs::read_to_string(c.out.dir().join(format!("lg/{}.lg", f.name())));
    let (Some(m), Ok(text), Some(l)) = (&map, text, &c.labels) else {
        return (check(name, &["lg file, map or labels missing".to_string()], &[]), map);
    };
    let nid: HashMap<&str, usize> = l.nodes.iter().enumerate().map(|(i, k)| (k.as_str(), i)).collect();
    let eid: HashMap<&EdgeLabel, usize> = l.edges.iter().enumerate().map(|(i, k)| (k, i)).collect();
    let mine: Vec<&VUnit> = c.units.iter().filter(|u| u.family == f).collect();
    let mut want_v = Vec::new();
    let mut want_e = BTreeSet::new();
    if m.units.len() != mine.len() {
        notes.push(format!("{} units in the map, {} rebuilt", m.units.len(), mine.len()));
    }
    let mut ids: Vec<&str> = mine.iter().map(|u| u.id.as_str()).collect();
    ids.sort_unstable();
    let by: HashMap<&str, &VUnit> = mine.iter().map(|u| (u.id.as_str(), *u)).collect();
    for mu in &m.units {
        let Some(u) = by.get(mu.id.as_str()) else {
            notes.push(format!("{}: not a unit", mu.id));
            continue;
        };
        if mu.class != u.class || mu.stage != u.stage || mu.workload != u.workload || mu.vertices != u.verts {
            notes.push(format!("{}: class, stage, workload or vertices differ", mu.id));
        }
        if mu.first as usize != want_v.len() {
            notes.push(format!("{}: first vertex {} (expected {})", mu.id, mu.first, want_v.len()));
        }
        let base = want_v.len() as u32;
        want_v.extend(u.keys.iter().map(|k| nid.get(k.as_str()).copied().unwrap_or(usize::MAX)));
        for ((a, b), ends) in &u.edges {
            want_e.insert((base + a, base + b, eid.get(ends).copied().unwrap_or(usize::MAX)));
        }
    }
    let mut got_v = Vec::new();
    let mut got_e = BTreeSet::new();
    let mut lines = text.lines();
    if lines.next().is_none() {
        notes.push("empty .lg".into());
    }
    for l in lines {
        let t: Vec<&str> = l.split(' ').collect();
        match (t.first(), t.len()) {
            (Some(&"v"), 3) if got_e.is_empty() => {
                if t[1].parse::<usize>().ok() != Some(got_v.len()) {
                    notes.push(format!("vertex line {l:?} out of order"));
                }
                got_v.push(t[2].parse::<usize>().unwrap_or(usize::MAX - 1));
            }
            (Some(&"e"), 4) => {
                let x: Vec<usize> = t[1..].iter().map(|s| s.parse().unwrap_or(usize::MAX - 1)).collect();
                got_e.insert((x[0] as u32, x[1] as u32, x[2]));
            }
            _ => notes.push(format!("bad line {l:?}")),
        }
    }
    if got_v != want_v {
        notes.push("vertices differ from the units' labels".into());
    }
    if got_e != want_e {
        notes.push(format!("{} edges, {} expected, or labels differ", got_e.len(), want_e.len()));
    }
    if m.vertices != want_v.len() as u64 || m.edges != want_e.len() as u64 {
        notes.push("map counts differ".into());
    }
    let _ = ctx;
    (
        check(name, &notes, &[("vertices", want_v.len() as u64), ("edges", want_e.len() as u64)]),
        map,
    )
}

fn read_patterns(out: &KindDir, f: Family) -> Vec<Result<PatternFile, String>> {
    let dir = out.dir().join("patterns").join(f.name());
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = rd
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .collect();
    names.sort();
    names
        .par_iter()
        .map(|n| {
            out.read_json::<PatternFile>(&format!("patterns/{}/{n}", f.name()))
                .map_err(|e| format!("{n}: {e}"))
        })
        .collect()
}

fn grami_check(
    c: &Ctx,
    ctx: &PassContext,
    f: Family,
    g: Option<&GramiFile>,
    sweep: Option<&Sweep>,
    pats: &[Result<PatternFile, String>],
    map: Option<&LgMap>,
) -> Check {
    let name = format!("{}:grami", f.name());
    let mut notes = Vec::new();
    let (Some(g), Some(sweep)) = (g, sweep) else {
        return check(name, &["grami or sweep file missing".to_string()], &[]);
    };
    if g.tau != sweep.tau {
        notes.push("τ differs from the sweep's".into());
    }
    if g.patterns.windows(2).any(|w| w[0].id >= w[1].id) {
        notes.push("patterns not sorted by id or repeated".into());
    }
    for r in &g.patterns {
        if r.vertices as usize > c.cfg.max_vertices || r.vertices < 2 {
            notes.push(format!("{}: {} vertices", r.id, r.vertices));
        }
    }
    if g.reported != g.patterns.len() as u64 + g.dropped {
        notes.push(format!(
            "{} reported, {} listed and {} dropped",
            g.reported,
            g.patterns.len(),
            g.dropped
        ));
    }
    let kept: BTreeSet<&str> = g.patterns.iter().map(|r| r.id.as_str()).collect();
    let files: BTreeSet<String> = pats.iter().filter_map(|p| p.as_ref().ok()).map(|p| p.id.clone()).collect();
    if kept != files.iter().map(String::as_str).collect() {
        notes.push(format!("{} kept patterns, {} pattern files", kept.len(), files.len()));
    }
    // GraMi's own list, re-read from the cache when it is there.
    let mut reread = 0u64;
    if let (Some(tau), Some(_), Some(l)) = (g.tau, map, &c.labels) {
        let t = ctx.tools();
        let lg = std::fs::read_to_string(c.out.dir().join(format!("lg/{}.lg", f.name()))).unwrap_or_default();
        let key = hash16(&format!(
            "{}\n{}:{}\nmaxLabelAppearance={}\n{}",
            t.grami.commit, t.docker.image, t.docker.tag, c.cfg.max_vertices, lg
        ));
        let path = ctx
            .dir()
            .path()
            .join("cache/grami")
            .join(key)
            .join(format!("tau{tau}/Output.txt"));
        if let Ok(text) = std::fs::read_to_string(&path) {
            match grami_ids(&text, l, c.cfg.max_vertices) {
                Ok((ids, dropped)) => {
                    reread = ids.len() as u64 + dropped;
                    if ids.iter().map(String::as_str).collect::<BTreeSet<_>>() != kept || dropped != g.dropped {
                        notes.push("GraMi's Output.txt lists other patterns".into());
                    }
                }
                Err(e) => notes.push(format!("Output.txt: {e}")),
            }
        }
    }
    check(
        name,
        &notes,
        &[
            ("reported", g.reported),
            ("kept", kept.len() as u64),
            ("reread", reread),
        ],
    )
}

/// The canonical ids of the patterns within `max` vertices in GraMi's `Output.txt`, and the number of
/// larger ones.
fn grami_ids(text: &str, l: &Labels, max: usize) -> Result<(Vec<String>, u64), String> {
    let mut lines = text.lines().skip(1);
    let n: usize = lines
        .next()
        .and_then(|x| x.trim().parse().ok())
        .ok_or("no count")?;
    let mut ids = Vec::new();
    let mut dropped = 0u64;
    type Block = (Vec<String>, Vec<(u32, u32, EdgeLabel)>);
    let mut cur: Option<Block> = None;
    let mut flush = |cur: &mut Option<Block>| {
        if let Some((ls, es)) = cur.take() {
            if ls.len() > max {
                dropped += 1;
            } else {
                ids.push(format!("p{}", hash16(&code_of(&ls, &es))));
            }
        }
    };
    for line in lines {
        let t: Vec<&str> = line.split(' ').collect();
        if line.ends_with(':') {
            flush(&mut cur);
            cur = Some((Vec::new(), Vec::new()));
            continue;
        }
        let (ls, es) = cur.as_mut().ok_or("body before a header")?;
        let num = |s: &str| s.parse::<usize>().map_err(|_| format!("bad number {s:?}"));
        match t.as_slice() {
            ["v", _, lab] => ls.push(l.nodes.get(num(lab)?).cloned().ok_or("unknown node label")?),
            ["e", a, b, lab] => es.push((
                num(a)? as u32,
                num(b)? as u32,
                l.edges.get(num(lab)?).cloned().ok_or("unknown edge label")?,
            )),
            _ => return Err(format!("bad line {line:?}")),
        }
    }
    flush(&mut cur);
    if ids.len() + dropped as usize != n {
        return Err(format!("{} patterns, {n} announced", ids.len() + dropped as usize));
    }
    Ok((ids, dropped))
}

fn patterns_check(c: &Ctx, f: Family, sweep: Option<&Sweep>, pats: &[Result<PatternFile, String>]) -> Check {
    let name = format!("{}:patterns", f.name());
    let units: Vec<&VUnit> = c.units.iter().filter(|u| u.family == f).collect();
    let by: HashMap<&str, &VUnit> = units.iter().map(|u| (u.id.as_str(), *u)).collect();
    let tau = sweep.and_then(|s| s.tau);
    let results: Vec<(Vec<String>, u64, u64, u64)> = pats
        .par_iter()
        .map(|p| match p {
            Err(e) => (vec![e.clone()], 0, 0, 0),
            Ok(p) => one_pattern(c, p, tau, &units, &by),
        })
        .collect();
    let mut notes = Vec::new();
    let (mut occ, mut mis, mut exact) = (0, 0, 0);
    for (n, o, m, e) in results {
        notes.extend(n);
        occ += o;
        mis += m;
        exact += e;
    }
    check(
        name,
        &notes,
        &[
            ("patterns", pats.len() as u64),
            ("occurrences", occ),
            ("mis_units", mis),
            ("alpha_recomputed", exact),
        ],
    )
}

/// Checks of one pattern file: (notes, occurrences, MIS entries, MIS sizes re-derived exactly).
fn one_pattern(
    c: &Ctx,
    p: &PatternFile,
    tau: Option<u64>,
    units: &[&VUnit],
    by: &HashMap<&str, &VUnit>,
) -> (Vec<String>, u64, u64, u64) {
    let mut notes = Vec::new();
    let id = &p.id;
    let n = p.labels.len();
    let edges: Vec<(u32, u32, EdgeLabel)> = p.edges.iter().map(|e| (e.src, e.dst, e.ends.clone())).collect();
    if edges.iter().any(|(a, b, _)| *a as usize >= n || *b as usize >= n) {
        return (vec![format!("{id}: edge to an unknown vertex")], 0, 0, 0);
    }
    let code = code_of(&p.labels, &edges);
    if code != p.code || p.id != format!("p{}", hash16(&code)) {
        notes.push(format!("{id}: the code is not canonical or the id not its hash"));
    }
    let mut sorted = p.labels.clone();
    sorted.sort();
    if sorted != p.labels {
        notes.push(format!("{id}: labels not in canonical order"));
    }
    if Some(p.tau) != tau || p.family != units.first().map(|u| u.family).unwrap_or(p.family) {
        notes.push(format!("{id}: τ or family differs from the sweep"));
    }
    if n > c.cfg.max_vertices || n < 2 {
        notes.push(format!("{id}: {n} vertices"));
    }
    // Every embedding, independently.
    let mut images: Vec<HashSet<(usize, u32)>> = vec![HashSet::new(); n];
    let mut total = 0u64;
    let mut want: BTreeSet<(String, Vec<u32>)> = BTreeSet::new();
    let mut non_convex = 0u64;
    for (ui, u) in units.iter().enumerate() {
        let emb = embeddings(p, u);
        total += emb.len() as u64;
        let mut seen: BTreeSet<Vec<u32>> = BTreeSet::new();
        for m in &emb {
            for (i, &h) in m.iter().enumerate() {
                images[i].insert((ui, h));
            }
            let mut s = m.clone();
            s.sort_unstable();
            seen.insert(s);
        }
        for s in seen {
            if convex(u, &s) {
                want.insert((u.id.clone(), s.iter().map(|&h| u.verts[h as usize]).collect()));
            } else {
                non_convex += 1;
            }
        }
    }
    let mni = images.iter().map(|s| s.len() as u64).min().unwrap_or(0);
    if mni != p.mni || total != p.embeddings || non_convex != p.non_convex {
        notes.push(format!(
            "{id}: MNI {}/{mni}, embeddings {}/{total}, non-convex {}/{non_convex} (stored/recomputed)",
            p.mni, p.embeddings, p.non_convex
        ));
    }
    if tau.is_some_and(|t| p.mni < t) {
        notes.push(format!("{id}: MNI {} below τ", p.mni));
    }
    // Stored occurrences: valid embeddings, convex images, feeds.
    let mut got: BTreeSet<(String, Vec<u32>)> = BTreeSet::new();
    for o in &p.occurrences {
        let Some(u) = by.get(o.unit.as_str()) else {
            notes.push(format!("{id}: occurrence in unknown unit {}", o.unit));
            continue;
        };
        let local: HashMap<u32, u32> = u.verts.iter().enumerate().map(|(i, v)| (*v, i as u32)).collect();
        let Some(m): Option<Vec<u32>> = o.map.iter().map(|v| local.get(v).copied()).collect() else {
            notes.push(format!("{id}: occurrence maps outside {}", o.unit));
            continue;
        };
        let distinct: HashSet<u32> = m.iter().copied().collect();
        let labels_ok = m.len() == n && distinct.len() == n && m.iter().zip(&p.labels).all(|(h, l)| &u.keys[*h as usize] == l);
        let edges_ok = edges
            .iter()
            .all(|(a, b, l)| labels_ok && u.edges.get(&(m[*a as usize], m[*b as usize])) == Some(l));
        if !labels_ok || !edges_ok {
            notes.push(format!("{id}: occurrence in {} is not an embedding", o.unit));
            continue;
        }
        let mut img = o.map.clone();
        img.sort_unstable();
        got.insert((o.unit.clone(), img));
        // Feeds.
        let mut feeds = Vec::new();
        for (i, &h) in m.iter().enumerate() {
            for (k, s) in u.args[h as usize].iter().enumerate() {
                let feed = match s {
                    Src::Vertex { v, .. } => {
                        let inner = m
                            .iter()
                            .position(|&x| u.verts[x as usize] == *v)
                            .is_some_and(|j| edges.iter().any(|(a, b, _)| *a as usize == j && *b as usize == i));
                        if inner {
                            continue;
                        }
                        if u.consts.contains(v) { Feed::Const } else { Feed::Vertex }
                    }
                    Src::Input { .. } => Feed::Input,
                    Src::Acc { .. } => Feed::Acc,
                    Src::Result { .. } => Feed::Result,
                };
                feeds.push((i as u32, k as u32, feed));
            }
        }
        let stored: Vec<(u32, u32, Feed)> = o.inputs.iter().map(|x| (x.at, x.arg, x.feed)).collect();
        if stored != feeds {
            notes.push(format!("{id}: operand feeds of an occurrence in {} differ", o.unit));
        }
    }
    if got != want || got.len() != p.occurrences.len() {
        notes.push(format!(
            "{id}: {} stored occurrences, {} convex images recomputed",
            p.occurrences.len(),
            want.len()
        ));
    }
    // MIS per unit.
    let mut exact = 0u64;
    let mut score = 0u128;
    let mut covered = 0usize;
    let mut units_seen = BTreeSet::new();
    for m in &p.mis {
        let Some(u) = by.get(m.unit.as_str()) else {
            notes.push(format!("{id}: MIS of unknown unit {}", m.unit));
            continue;
        };
        units_seen.insert(m.unit.clone());
        if m.workload != u.workload || m.algorithm != c.cfg.mis.algorithm.name() {
            notes.push(format!("{id}: {} workload or algorithm differs", m.unit));
        }
        let mine: Vec<u32> = (0..p.occurrences.len() as u32)
            .filter(|&k| p.occurrences[k as usize].unit == m.unit)
            .collect();
        covered += mine.len();
        if m.occurrences != mine {
            notes.push(format!("{id}: {} occurrence list differs", m.unit));
            continue;
        }
        // Independence by a mark array over datapath vertices.
        let mut mark: HashSet<u32> = HashSet::new();
        let mut indep = m.chosen.iter().all(|k| mine.binary_search(k).is_ok());
        for k in &m.chosen {
            if let Some(o) = p.occurrences.get(*k as usize) {
                for v in &o.map {
                    indep &= mark.insert(*v);
                }
            }
        }
        if !indep || m.size != m.chosen.len() as u64 || m.bound < m.size || (m.optimal && m.bound != m.size) {
            notes.push(format!("{id}: {}: not independent, or size/bound inconsistent", m.unit));
        }
        // α by an exact search.
        let occs: Vec<HashSet<u32>> = mine
            .iter()
            .map(|&k| p.occurrences[k as usize].map.iter().copied().collect())
            .collect();
        let adj: Vec<HashSet<usize>> = (0..occs.len())
            .map(|a| {
                (0..occs.len())
                    .filter(|&b| b != a && !occs[a].is_disjoint(&occs[b]))
                    .collect()
            })
            .collect();
        let mut budget = 2_000_000u64;
        if let Some(a) = alpha(&adj, &mut budget) {
            exact += 1;
            let a = a as u64;
            if (m.optimal && m.size != a) || m.size > a || m.bound < a {
                notes.push(format!(
                    "{id}: {}: α = {a}, stored size {} bound {} optimal {}",
                    m.unit, m.size, m.bound, m.optimal
                ));
            }
        }
        score = score.saturating_add(m.workload.saturating_mul(m.size as u128));
    }
    let with_occ: BTreeSet<String> = p.occurrences.iter().map(|o| o.unit.clone()).collect();
    if with_occ != units_seen || covered != p.occurrences.len() {
        notes.push(format!("{id}: MIS entries do not cover the units with occurrences"));
    }
    if score != p.score {
        notes.push(format!("{id}: score {} recomputes to {score}", p.score));
    }
    (notes, p.occurrences.len() as u64, p.mis.len() as u64, exact)
}

fn top(pats: &[&PatternFile], k: usize) -> Vec<(String, u32, u128, u64, u64)> {
    let mut v: Vec<(String, u32, u128, u64, u64)> = pats
        .iter()
        .filter(|p| p.score > 0)
        .map(|p| {
            (
                p.id.clone(),
                p.labels.len() as u32,
                p.score,
                p.mni,
                p.mis.iter().map(|m| m.size).sum(),
            )
        })
        .collect();
    v.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)).then(a.0.cmp(&b.0)));
    v.truncate(k);
    v
}

fn selection_check(c: &Ctx, f: Family, sel: Option<&Selection>, pats: &[Result<PatternFile, String>]) -> Check {
    let name = format!("{}:selection", f.name());
    let Some(sel) = sel else {
        return check(name, &["selection file missing".to_string()], &[]);
    };
    let ps: Vec<&PatternFile> = pats.iter().filter_map(|p| p.as_ref().ok()).collect();
    let want = top(&ps, c.cfg.top_k);
    let got: Vec<(String, u32, u128, u64, u64)> = sel
        .patterns
        .iter()
        .map(|s| (s.id.clone(), s.vertices, s.score, s.mni, s.mis))
        .collect();
    let mut notes = Vec::new();
    if got != want || sel.top_k != c.cfg.top_k as u64 {
        notes.push("the selection does not recompute from the pattern files".into());
    }
    check(name, &notes, &[("selected", got.len() as u64)])
}

fn sweep_check(c: &Ctx, f: Family, s: Option<&Sweep>, sel: Option<&Selection>, map: Option<&LgMap>) -> Check {
    let name = format!("{}:sweep", f.name());
    let (Some(s), Some(sel), Some(map)) = (s, sel, map) else {
        return check(name, &["sweep, selection or map missing".to_string()], &[]);
    };
    let mut notes = Vec::new();
    let mut taus = vec![c.cfg.tau_start];
    while let Some(&t) = taus.last()
        && t > c.cfg.tau_min
    {
        taus.push((t / 2).max(c.cfg.tau_min));
    }
    let seq: Vec<u64> = s.steps.iter().map(|x| x.tau).collect();
    if !taus.starts_with(&seq) {
        notes.push(format!("τ sequence {seq:?} is not a prefix of {taus:?}"));
    }
    let finished: Vec<&super::json::Step> = s.steps.iter().filter(|x| x.finished).collect();
    if s.steps.iter().rev().skip(1).any(|x| !x.finished) {
        notes.push("an unfinished step before the last".into());
    }
    if s.tau != finished.last().map(|x| x.tau) || sel.tau != s.tau {
        notes.push("τ is not the last finished step's".into());
    }
    let ids: Vec<&str> = sel.patterns.iter().map(|p| p.id.as_str()).collect();
    if let Some(last) = finished.last()
        && last.selection != hash16(&ids.join("\n"))
    {
        notes.push("the last step's selection hash is not the selection's".into());
    }
    let ok = match s.stopped_by {
        Stop::Empty => map.edges == 0 && s.steps.is_empty(),
        Stop::Budget => s.steps.last().is_some_and(|x| !x.finished),
        Stop::TauMin => s.steps.last().is_some_and(|x| x.finished && x.tau == c.cfg.tau_min),
        Stop::Stable => {
            let n = finished.len();
            n >= 2
                && finished[n - 1].selection == finished[n - 2].selection
                && !sel.patterns.is_empty()
                && s.steps.last().is_some_and(|x| x.finished)
        }
    };
    if !ok {
        notes.push(format!("the stop reason {:?} does not hold", s.stopped_by));
    }
    // No earlier stop was missed.
    for w in finished.windows(2).take(finished.len().saturating_sub(2)) {
        if w[0].selection == w[1].selection && w[1].selection != hash16("") {
            notes.push(format!("the selection was already stable at τ = {}", w[1].tau));
        }
    }
    check(name, &notes, &[("steps", s.steps.len() as u64)])
}

fn overlap_check(out: &KindDir, pats: &[PatternFile], sels: &BTreeMap<Family, Selection>) -> Check {
    let mut fams: BTreeMap<&str, BTreeSet<Family>> = BTreeMap::new();
    for p in pats {
        fams.entry(&p.id).or_default().insert(p.family);
    }
    type Row = (String, Vec<Family>, Vec<Family>);
    let want: Vec<Row> = fams
        .into_iter()
        .filter(|(_, f)| f.len() > 1)
        .map(|(id, f)| {
            let s: Vec<Family> = sels
                .iter()
                .filter(|(_, s)| s.patterns.iter().any(|x| x.id == id))
                .map(|(f, _)| *f)
                .collect();
            (id.to_string(), f.into_iter().collect(), s)
        })
        .collect();
    let got: Option<Overlap> = out.read_json("overlap.json").ok();
    let got: Option<Vec<Row>> = got.map(|o| {
        o.patterns
            .into_iter()
            .map(|p| (p.id, p.families, p.selected_in))
            .collect()
    });
    let notes: Vec<String> = if got.as_ref() == Some(&want) {
        Vec::new()
    } else {
        vec!["overlap.json does not recompute".into()]
    };
    check("overlap", &notes, &[("shared", want.len() as u64)])
}

fn summary_check(
    c: &Ctx,
    pats: &[PatternFile],
    sels: &BTreeMap<Family, Selection>,
    sweeps: &BTreeMap<Family, Sweep>,
    maps: &BTreeMap<Family, LgMap>,
) -> Check {
    let mut notes = Vec::new();
    let s: Option<Summary> = c.out.read_json("summary.json").ok();
    let Some(s) = s else {
        return check("summary", &["summary.json missing".to_string()], &[]);
    };
    for f in Family::ALL {
        let Some(x) = s.families.get(f.name()) else {
            notes.push(format!("{}: missing", f.name()));
            continue;
        };
        let mine: Vec<&PatternFile> = pats.iter().filter(|p| p.family == f).collect();
        let ok = maps.get(&f).is_some_and(|m| {
            x.units == m.units.len() as u64
                && x.vertices == m.vertices
                && x.edges == m.edges
                && x.workload == m.units.iter().map(|u| u.workload).sum::<u128>()
        }) && sweeps
            .get(&f)
            .is_some_and(|w| x.tau == w.tau && x.stopped_by == w.stopped_by)
            && x.kept == mine.len() as u64
            && x.occurrences == mine.iter().map(|p| p.occurrences.len() as u64).sum::<u64>()
            && sels.get(&f).is_some_and(|q| x.selected == q.patterns.len() as u64);
        if !ok {
            notes.push(format!("{}: counts differ", f.name()));
        }
    }
    if let Some(l) = &c.labels
        && (s.node_labels != l.nodes.len() as u64 || s.edge_labels != l.edges.len() as u64)
    {
        notes.push("label counts differ".into());
    }
    if s.mis_algorithm != c.cfg.mis.algorithm.name() {
        notes.push("MIS algorithm differs".into());
    }
    check("summary", &notes, &[])
}
