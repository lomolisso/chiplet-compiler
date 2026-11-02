//! Independent checks of the datapaths (design §5, docs/passes/datapaths.md §6). They never call the
//! extraction (`algo::extract`) or the tuple builder; the obligations are regenerated from the stored
//! datapaths (the statement, not the algorithm) and re-checked by the second solver.
//!
//! - `complete`: every class has a datapath, its tuples are the distinct (element types, constant types,
//!   structural attributes) of its instance rows, every tuple has a proof and a result, every
//!   accumulator an order query per tuple, every encoding a lemma;
//! - `<z>:structure`: the vertex list is topological, every source is declared, every vertex is the op
//!   and family of the kernel node it names (or a core vertex of its kind), its width is that node's
//!   element class, constants are value columns every row has, inputs are skeleton operands, the
//!   outputs are the kernel's (and a template's core inputs), the stages agree with the routing, and
//!   no vertex is dead;
//! - `<z>:proof`: every stored query equals the regenerated one, its result matches its hash, the
//!   configured solver answered `unsat` (or the class is allowed with its reason), the second solver
//!   agrees, and the stored order flags are the order verdicts;
//! - `lemmas`: the lemma files are the regenerated ones, each `unsat` by Z3 (stored and re-run);
//! - `solver_sanity`: a mutated datapath (a non-commutative vertex's operands exchanged) gives `sat`
//!   in both solvers;
//! - `summary`: the summary recounts from the datapaths.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use cc_core::{Check, VerifyReport};
use cc_ir::Attr;
use cc_pipeline::{PassContext, PassError};
use cc_smt::{Solver, Verdict};
use rayon::prelude::*;

use super::json::{Datapath, ProofResult, Src, Stage, Summary};
use super::{DATAPATHS, DatapathsConfig, Inputs, load, proofs};
use crate::components::algo::graph::Kind;
use crate::iteration_space::json::{Direction, Route};
use crate::kernels::json::{KNode, Role};
use crate::normalize::algo::omega::family_key;
use crate::normalize::smt;

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

/// The distinct tuples of a class's rows, recomputed: (element types, constant types, attributes).
type TupleKey = (Vec<String>, BTreeMap<String, String>, BTreeMap<String, i64>);

fn row_tuples(inputs: &Inputs, i: usize) -> Result<BTreeMap<TupleKey, u64>, String> {
    let class = &inputs.loaded.classes[i];
    let mut out = BTreeMap::new();
    for row in &inputs.loaded.tables[i].rows {
        let mut consts = BTreeMap::new();
        let mut attrs = BTreeMap::new();
        for (col, v) in class.params.values.iter().zip(&row.values) {
            let (Some((_, field)), Some(v)) = (col.rsplit_once('.'), v) else {
                continue;
            };
            if field.starts_with("splat") {
                let t = match v {
                    Attr::Dense {
                        ty: cc_ir::Type::Tensor(t),
                        ..
                    } => t.elem.to_string(),
                    Attr::Int { ty, .. } | Attr::Float { ty, .. } => ty.clone(),
                    Attr::Bool(_) => "i1".into(),
                    _ => return Err(format!("{col}: a constant without a type")),
                };
                consts.insert(col.clone(), t);
            } else if (field == "exponent_bits" || field == "mantissa_bits")
                && let Attr::Int { value, .. } = v
            {
                attrs.insert(col.clone(), *value as i64);
            }
        }
        *out.entry((row.elems.clone(), consts, attrs)).or_default() += 1;
    }
    Ok(out)
}

fn short(name: &str) -> &str {
    name.strip_prefix("stablehlo.").unwrap_or(name)
}

/// Structural notes of one datapath against its kernel graph, skeleton and route.
fn structure(inputs: &Inputs, i: usize, dp: &Datapath) -> Vec<String> {
    let mut n = Vec::new();
    let class = &inputs.loaded.classes[i];
    let g = &inputs.loaded.graphs[i];
    let skel = &inputs.skeletons[i];
    let routed = &inputs.routed[i];
    let template = routed.route == Route::Template;
    let elem = |node: u32, r: u32| -> Option<u32> {
        class
            .local
            .elems
            .get(node as usize)
            .and_then(|x| x.get(r as usize))
            .copied()
    };
    let fam = |e: u32| {
        class
            .params
            .elems
            .get(e as usize)
            .map(|f| family_key(f).to_string())
    };
    let input_names: BTreeSet<&str> = dp.inputs.iter().map(|x| x.name.as_str()).collect();
    let core_results = skel
        .core_node
        .and_then(|c| match &g.nodes[c as usize] {
            KNode::Member { node, .. } => Some(node.label.results.len() as u32),
            _ => None,
        })
        .unwrap_or(0);
    let nacc = dp.accumulator.as_ref().map_or(0, |a| a.entries.len() as u32);
    let src_ok = |s: &Src, id: u32, n: &mut Vec<String>| match s {
        Src::Vertex { v, res } => {
            if *v >= id || *res != 0 {
                n.push(format!(
                    "vertex {id}: reads vertex {v} result {res}, not an earlier value"
                ));
            }
        }
        Src::Input { name } => {
            if !input_names.contains(name.as_str()) {
                n.push(format!("vertex {id}: reads undeclared input {name}"));
            }
        }
        Src::Acc { index } => {
            if *index >= nacc {
                n.push(format!(
                    "vertex {id}: reads accumulator {index}, which does not exist"
                ));
            }
        }
        Src::Result { index } => {
            if *index >= core_results || skel.core_node.is_none() {
                n.push(format!(
                    "vertex {id}: reads core result {index}, which does not exist"
                ));
            }
        }
    };
    for (k, v) in dp.vertices.iter().enumerate() {
        let id = k as u32;
        if v.id != id {
            n.push(format!("vertex {k} has id {}", v.id));
        }
        for a in &v.args {
            src_ok(&a.src, id, &mut n);
        }
        if v.label != format!("{}@{}", v.op, v.family) {
            n.push(format!("vertex {id}: label {} is not op@family", v.label));
        }
        let node = v.from.node;
        let Some(x) = g.nodes.get(node as usize) else {
            n.push(format!("vertex {id}: node {node} does not exist"));
            continue;
        };
        let (role, label) = match x {
            KNode::Member { role, node, .. } => (*role, &node.label),
            _ => {
                n.push(format!("vertex {id}: node {node} is not a member"));
                continue;
            }
        };
        // The stage agrees with the node's role (template stages: the routing's member lists).
        let stage_ok = match (v.stage, role) {
            (Stage::Prologue { port }, Role::Prologue) => {
                !template
                    || routed
                        .stages
                        .as_ref()
                        .and_then(|s| s.prologue.get(&port))
                        .is_some_and(|m| m.contains(&node))
            }
            (Stage::Epilogue, Role::Epilogue) => {
                !template || routed.stages.as_ref().is_some_and(|s| s.epilogue.contains(&node))
            }
            (Stage::Map, Role::Map) => true,
            (Stage::Core, Role::Body | Role::Core) => !template,
            _ => false,
        };
        if !stage_ok {
            n.push(format!(
                "vertex {id}: stage {:?} for a {} node {node}",
                v.stage,
                role.name()
            ));
        }
        if v.op == "const" {
            match (v.attrs.get("column"), v.attrs.get("value")) {
                (Some(col), None) => {
                    let ci = class.params.values.iter().position(|c| c == col);
                    match ci {
                        None => n.push(format!("vertex {id}: column {col} is not a value column")),
                        Some(ci) => {
                            if inputs.loaded.tables[i]
                                .rows
                                .iter()
                                .any(|r| r.values.get(ci).is_none_or(Option::is_none))
                            {
                                n.push(format!("vertex {id}: a row has no value for {col}"));
                            }
                        }
                    }
                    if !col.starts_with(&format!("n{node}.splat")) || v.width != *col {
                        n.push(format!(
                            "vertex {id}: constant {col} is not a folded operand of node {node}"
                        ));
                    }
                }
                (None, Some(z)) if z == "0" && role == Role::Core && !template => {}
                _ => n.push(format!("vertex {id}: a constant needs a column or the init 0")),
            }
            continue;
        }
        // Width: the element class of the node's result (core vertices: the core result's).
        let want_w = elem(node, 0).map(|e| format!("w{e}"));
        if Some(&v.width) != want_w.as_ref() {
            n.push(format!(
                "vertex {id}: width {} but node {node} has {want_w:?}",
                v.width
            ));
        }
        let want_f = elem(node, 0).and_then(fam);
        if Some(&v.family) != want_f.as_ref() {
            n.push(format!(
                "vertex {id}: family {} but node {node} has {want_f:?}",
                v.family
            ));
        }
        let Kind::Op { name, .. } = &label.kind else {
            n.push(format!("vertex {id}: node {node} is not an op"));
            continue;
        };
        if role == Role::Core {
            let ok = matches!(v.op.as_str(), "multiply" | "add" | "convert")
                && matches!(short(name), "dot_general" | "convolution" | "dynamic_conv");
            if !ok || v.from.access.is_some() {
                n.push(format!(
                    "vertex {id}: {} is not a vertex of core {}",
                    v.op,
                    short(name)
                ));
            }
        } else {
            if short(name) != v.op {
                n.push(format!(
                    "vertex {id}: op {} but node {node} is {}",
                    v.op,
                    short(name)
                ));
            }
            match (role, v.from.access) {
                (Role::Body, None) => {}
                (Role::Body, Some(_)) => n.push(format!("vertex {id}: a body vertex with an access")),
                (_, Some(k)) => {
                    if skel.accesses.get(k as usize).is_none_or(|a| a.node != node) {
                        n.push(format!("vertex {id}: access {k} is not node {node}'s"));
                    }
                }
                (_, None) => n.push(format!("vertex {id}: no access instance")),
            }
            // Comparison directions: GT/GE only as LT/LE with exchanged operands.
            if let Some(d) = v.attrs.get("comparison_direction") {
                let orig = crate::datapaths::algo::extract::enum_attr(x, "comparison_direction");
                let ok = match orig.as_deref() {
                    Some("GT") => d == "LT",
                    Some("GE") => d == "LE",
                    Some(o) => o == d,
                    None => false,
                };
                if !ok {
                    n.push(format!("vertex {id}: direction {d} for {orig:?}"));
                }
            }
        }
    }
    // Inputs: skeleton operands (and a template's core results).
    for x in &dp.inputs {
        let ok = match x.name.strip_prefix("core_result") {
            Some(k) => template && k.parse::<u32>().is_ok_and(|k| k < core_results),
            None => skel.operands.iter().any(|o| {
                o.name == x.name && o.direction == Direction::In && o.node == x.node && o.res == x.res
            }),
        };
        if !ok {
            n.push(format!("input {}: not a skeleton operand", x.name));
        }
    }
    // Outputs: the kernel's, and a template's core inputs.
    let mut want: BTreeSet<String> = skel
        .operands
        .iter()
        .filter(|o| o.direction == Direction::Out && !o.core)
        .map(|o| o.name.clone())
        .collect();
    if template {
        want.extend(
            skel.operands
                .iter()
                .filter(|o| {
                    o.core
                        && o.direction == Direction::In
                        && crate::iteration_space::algo::skeleton::is_core_input(&o.name)
                })
                .map(|o| o.name.clone()),
        );
    }
    let have: BTreeSet<String> = dp.outputs.iter().map(|o| o.name.clone()).collect();
    if want != have {
        n.push(format!("outputs {have:?}, the kernel has {want:?}"));
    }
    let end = dp.vertices.len() as u32;
    for o in &dp.outputs {
        src_ok(&o.src, end, &mut n);
    }
    // The accumulator: datapath-routed cores with a reduction only.
    let wants_acc =
        !template && skel.core_node.is_some() && skel.class != crate::iteration_space::json::SkelClass::Map;
    if wants_acc != dp.accumulator.is_some() {
        n.push(format!(
            "accumulator present: {}, expected {wants_acc}",
            dp.accumulator.is_some()
        ));
    }
    if let Some(a) = &dp.accumulator {
        for e in &a.entries {
            src_ok(&e.init, end, &mut n);
            src_ok(&e.next, end, &mut n);
        }
        if a.omega != skel.omega {
            n.push("ω differs from the skeleton's".into());
        }
    }
    // No dead vertex.
    let mut live = vec![false; dp.vertices.len()];
    let mut stack: Vec<u32> = Vec::new();
    let push = |s: &Src, stack: &mut Vec<u32>| {
        if let Src::Vertex { v, .. } = s {
            stack.push(*v);
        }
    };
    for o in &dp.outputs {
        push(&o.src, &mut stack);
    }
    if let Some(a) = &dp.accumulator {
        for e in &a.entries {
            push(&e.init, &mut stack);
            push(&e.next, &mut stack);
        }
    }
    while let Some(v) = stack.pop() {
        if let Some(l) = live.get_mut(v as usize) {
            if *l {
                continue;
            }
            *l = true;
            for a in &dp.vertices[v as usize].args {
                push(&a.src, &mut stack);
            }
        }
    }
    if let Some(d) = live.iter().position(|l| !l) {
        n.push(format!("vertex {d} is dead"));
    }
    n
}

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: DatapathsConfig = ctx.options()?;
    let mut r = VerifyReport::new(ctx.name());
    let out = ctx.output(DATAPATHS)?;
    let inputs = load(ctx)?;
    let ids: Vec<String> = inputs.loaded.classes.iter().map(|c| c.id.clone()).collect();
    let dps: Vec<Option<Datapath>> = ids
        .par_iter()
        .map(|id| out.read_json::<Datapath>(&format!("datapaths/{id}.json")).ok())
        .collect();

    // Completeness and tuples.
    let mut complete = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let Some(dp) = &dps[i] else {
            complete.push(format!("{id}: no datapath"));
            continue;
        };
        match row_tuples(&inputs, i) {
            Ok(want) => {
                let have: BTreeMap<TupleKey, u64> = dp
                    .tuples
                    .iter()
                    .map(|t| ((t.elems.clone(), t.consts.clone(), t.attrs.clone()), t.rows))
                    .collect();
                if have != want || have.len() != dp.tuples.len() {
                    complete.push(format!("{id}: tuples differ from the instance rows"));
                }
            }
            Err(e) => complete.push(format!("{id}: {e}")),
        }
        for t in &dp.tuples {
            let mut fs = vec![format!("{id}/{}.smt2", t.name)];
            if dp.accumulator.is_some() {
                fs.push(format!("{id}/{}.order.smt2", t.name));
            }
            for f in fs {
                for p in [f.clone(), f.replace(".smt2", ".result")] {
                    if !out.path(&format!("proofs/{p}")).is_ok_and(|x| x.exists()) {
                        complete.push(format!("proofs/{p} is missing"));
                    }
                }
            }
        }
    }
    r.push(check("complete", &complete, &[("classes", ids.len() as u64)]));

    // Structure.
    let notes: Vec<Vec<String>> = (0..ids.len())
        .into_par_iter()
        .map(|i| {
            dps[i]
                .as_ref()
                .map(|dp| structure(&inputs, i, dp))
                .unwrap_or_default()
        })
        .collect();
    for (i, id) in ids.iter().enumerate() {
        if dps[i].is_some() {
            r.push(check(format!("{id}:structure"), &notes[i], &[]));
        }
    }

    // Obligations, regenerated from the stored datapaths.
    let regen: Vec<Result<proofs::Obligations, String>> = (0..ids.len())
        .into_par_iter()
        .map(|i| match &dps[i] {
            Some(dp) => {
                let l = &inputs.loaded;
                proofs::obligations(
                    &l.classes[i],
                    &inputs.skeletons[i],
                    &inputs.routed[i],
                    &l.graphs[i],
                    dp,
                )
            }
            None => Err("no datapath".into()),
        })
        .collect();
    let (solver, runner, location) = smt(ctx)?;
    let threads = ctx.resources().threads.max(1);
    let mut files = Vec::new();
    let mut used = BTreeSet::new();
    for o in regen.iter().flatten() {
        files.extend(o.queries.iter().map(|q| q.file.clone()));
        used.extend(o.used.iter().cloned());
    }
    let other = cc_smt::run_sharded(
        &runner,
        &location,
        solver.other(),
        &out.path("proofs")?,
        &files,
        ctx.smt().timeout_s,
        threads,
    );
    let allowed: HashMap<&str, &str> = cfg
        .allow_unknown
        .iter()
        .map(|a| (a.class.as_str(), a.reason.as_str()))
        .collect();
    for (i, id) in ids.iter().enumerate() {
        let Some(dp) = &dps[i] else { continue };
        let mut n = Vec::new();
        let ob = match &regen[i] {
            Ok(o) => o,
            Err(e) => {
                r.push(check(format!("{id}:proof"), std::slice::from_ref(e), &[]));
                continue;
            }
        };
        let order_flags: BTreeMap<&str, bool> = dp
            .accumulator
            .iter()
            .flat_map(|a| a.order.iter().map(|o| (o.tuple.as_str(), o.free)))
            .collect();
        for q in &ob.queries {
            let stored = std::fs::read_to_string(out.path(&format!("proofs/{}", q.file)).unwrap_or_default())
                .unwrap_or_default();
            if stored != q.text {
                n.push(format!("{}: differs from the regenerated query", q.file));
            }
            let res = out.read_json::<ProofResult>(&format!("proofs/{}", q.file.replace(".smt2", ".result")));
            let res = match res {
                Ok(p) => p,
                Err(e) => {
                    n.push(format!("{}: no result: {e}", q.file));
                    continue;
                }
            };
            if res.query_hash != cc_smt::query_hash(&q.text) || res.solver != solver {
                n.push(format!("{}: the result is for another query or solver", q.file));
            }
            let second = other.as_ref().ok().and_then(|b| b.verdicts.get(&q.file).cloned());
            if q.order {
                let free = res.verdict.is_unsat();
                if order_flags.get(q.tuple.as_str()) != Some(&free) {
                    n.push(format!("{}: order flag disagrees with the verdict", q.file));
                }
                // A claimed free order must be re-proven; a counterexample must not be refuted.
                match (free, &second) {
                    (true, Some(v)) if v.is_unsat() => {}
                    (false, Some(v)) if !v.is_unsat() => {}
                    (_, v) => n.push(format!("{}: {} re-check {v:?}", q.file, solver.other().name())),
                }
            } else {
                let allow = allowed.get(id.as_str()).copied();
                let ok = match (&res.verdict, allow) {
                    (Verdict::Unsat, _) => res.allowed.is_none(),
                    (Verdict::Unknown | Verdict::Timeout, Some(reason)) => {
                        res.allowed.as_deref() == Some(reason)
                    }
                    _ => false,
                };
                if !ok {
                    n.push(format!("{}: stored verdict {:?}", q.file, res.verdict));
                }
                match (&second, allow) {
                    (Some(v), _) if v.is_unsat() => {}
                    (Some(Verdict::Unknown | Verdict::Timeout), Some(_)) => {}
                    (v, _) => n.push(format!("{}: {} re-check {v:?}", q.file, solver.other().name())),
                }
            }
        }
        if let Err(e) = &other {
            n.push(format!("{} re-check failed: {e}", solver.other().name()));
        }
        r.push(check(
            format!("{id}:proof"),
            &n,
            &[("queries", ob.queries.len() as u64)],
        ));
    }

    // Lemmas.
    let mut ln = Vec::new();
    let lemmas = cc_smt::float::lemmas(&used).unwrap_or_else(|e| {
        ln.push(e.0);
        Vec::new()
    });
    let stored: BTreeSet<String> = std::fs::read_dir(out.path("lemmas")?)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.file_name().to_str().map(str::to_string))
                .filter(|f| f.ends_with(".smt2"))
                .collect()
        })
        .unwrap_or_default();
    let want: BTreeSet<String> = lemmas.iter().map(|(f, _)| f.clone()).collect();
    if stored != want {
        ln.push(format!(
            "lemma files {stored:?}, the encodings used need {want:?}"
        ));
    }
    for (f, text) in &lemmas {
        let s =
            std::fs::read_to_string(out.path(&format!("lemmas/{f}")).unwrap_or_default()).unwrap_or_default();
        if &s != text {
            ln.push(format!("{f}: differs from the regenerated lemma"));
        }
        match out.read_json::<ProofResult>(&format!("lemmas/{}", f.replace(".smt2", ".result"))) {
            Ok(p)
                if p.verdict.is_unsat()
                    && p.query_hash == cc_smt::query_hash(text)
                    && p.solver == Solver::Z3 => {}
            Ok(p) => ln.push(format!("{f}: stored {:?} {:?}", p.solver, p.verdict)),
            Err(e) => ln.push(format!("{f}: no result: {e}")),
        }
    }
    let names: Vec<String> = lemmas.iter().map(|(f, _)| f.clone()).collect();
    match cc_smt::run_sharded(
        &runner,
        &location,
        Solver::Z3,
        &out.path("lemmas")?,
        &names,
        cfg.lemma_timeout_s,
        threads,
    ) {
        Ok(b) => {
            for f in &names {
                if !b.verdicts.get(f).is_some_and(Verdict::is_unsat) {
                    ln.push(format!("{f}: Z3 re-run {:?}", b.verdicts.get(f)));
                }
            }
        }
        Err(e) => ln.push(format!("Z3 re-run failed: {e}")),
    }
    r.push(check("lemmas", &ln, &[("lemmas", lemmas.len() as u64)]));

    // A mutated datapath must give a counterexample.
    r.push(sanity(ctx, &inputs, &dps, solver, &runner, &location)?);

    // Summary.
    let mut sn = Vec::new();
    match out.read_json::<Summary>("summary.json") {
        Ok(s) => {
            let all: Vec<&Datapath> = dps.iter().flatten().collect();
            let vertices: u64 = all.iter().map(|d| d.vertices.len() as u64).sum();
            let tuples: u64 = all.iter().map(|d| d.tuples.len() as u64).sum();
            let mut labels: BTreeMap<String, u64> = BTreeMap::new();
            for d in &all {
                for v in &d.vertices {
                    *labels.entry(v.label.clone()).or_default() += 1;
                }
            }
            if s.classes != all.len() as u64
                || s.vertices != vertices
                || s.tuples != tuples
                || s.labels != labels
            {
                sn.push("summary counts differ from the datapaths".into());
            }
            if s.proofs != tuples {
                sn.push(format!("{} proofs for {tuples} tuples", s.proofs));
            }
        }
        Err(e) => sn.push(format!("no summary: {e}")),
    }
    r.push(check("summary", &sn, &[]));
    Ok(r)
}

/// Exchange the operands of the first non-commutative binary vertex with distinct operands (in class
/// order) and check that both solvers find the difference.
fn sanity(
    ctx: &PassContext,
    inputs: &Inputs,
    dps: &[Option<Datapath>],
    solver: Solver,
    runner: &cc_core::runner::BoundRunner,
    location: &cc_smt::Location,
) -> Result<Check, PassError> {
    let pick = dps.iter().enumerate().find_map(|(i, d)| {
        let d = d.as_ref()?;
        let v = d.vertices.iter().position(|v| {
            matches!(v.op.as_str(), "subtract" | "divide" | "remainder" | "power")
                && v.args.len() == 2
                && v.args[0].src != v.args[1].src
        })?;
        Some((i, v))
    });
    let Some((i, v)) = pick else {
        return Ok(Check::skip(
            "solver_sanity",
            "no class has a non-commutative binary vertex",
        ));
    };
    let mut dp = dps[i]
        .clone()
        .ok_or_else(|| PassError::failed("sanity: no datapath"))?;
    dp.vertices[v].args.swap(0, 1);
    dp.tuples.truncate(1);
    dp.accumulator = dp.accumulator.map(|mut a| {
        a.order.clear();
        a
    });
    let l = &inputs.loaded;
    let ob = proofs::obligations(
        &l.classes[i],
        &inputs.skeletons[i],
        &inputs.routed[i],
        &l.graphs[i],
        &dp,
    )
    .map_err(PassError::failed)?;
    let q = ob
        .queries
        .iter()
        .find(|q| !q.order)
        .ok_or_else(|| PassError::failed("sanity: no query"))?;
    let dir = ctx.dir().path().join("cache").join("sanity");
    std::fs::create_dir_all(&dir).map_err(|e| PassError::failed(format!("{}: {e}", dir.display())))?;
    let file = "sanity.smt2".to_string();
    std::fs::write(dir.join(&file), &q.text).map_err(|e| PassError::failed(format!("sanity: {e}")))?;
    let mut n = Vec::new();
    for s in [solver, solver.other()] {
        let b = cc_smt::run_batch(
            runner,
            location,
            s,
            &dir,
            std::slice::from_ref(&file),
            ctx.smt().timeout_s,
        )?;
        match b.verdicts.get(&file) {
            Some(Verdict::Sat { .. }) => {}
            v => n.push(format!("{}: {v:?} for the mutated datapath", s.name())),
        }
    }
    Ok(check("solver_sanity", &n, &[("vertex", v as u64)]).note(format!(
        "{}: vertex {v} ({}) operands exchanged",
        dp.id, dp.vertices[v].label
    )))
}
