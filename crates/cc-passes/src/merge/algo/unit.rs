//! Units and their dataflow graphs (docs/passes/merge.md §4.1).
//!
//! A unit is one datapath class (route `datapath`) or one non-empty stage of a template class, as in
//! `mine`. Its graph has a node per datapath vertex (operators and constants), per source it reads
//! (an input, a template's core result), per accumulator (a register: ports `next`, `init`; result 0 its
//! current value, result 1 its final value) and per output. Every node has its types at every width
//! tuple: operand types then the result type for an operator, the value type otherwise.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::datapaths::json::{Datapath, OutputKind, Src, Stage};
use crate::datapaths::spec::ty_of_width;
use crate::iteration_space::json::{Route, SkelClass};
use crate::merge::json::BlockKind;
use crate::mine::algo::family::{SEMANTIC_ATTRS, stage_unit_id};
use crate::mine::json::Family;

type R<X> = Result<X, String>;

/// One node of a unit's graph.
#[derive(Debug, Clone, PartialEq)]
pub struct UNode {
    /// `v<id>`, `in:<var>`, `acc<i>`, `out:<name>`.
    pub name: String,
    pub kind: BlockKind,
    pub op: String,
    /// Semantic attributes (`comparison_direction`, `compare_type`), a literal constant's `value`.
    pub attrs: BTreeMap<String, String>,
    /// The SMT variable of a source (input name, constant column, `acc<i>`), an output's name.
    pub var: Option<String>,
    /// Operand sources `(node, result)` in operand order; an accumulator's `[next, init]`.
    pub args: Vec<(u32, u32)>,
    /// The operands are a commutative pair (`*` ports).
    pub commutative: bool,
    /// Per tuple: operand types then result type (operators), else the value type; `reduce_precision`
    /// adds its `e<E>m<M>` token.
    pub sigs: Vec<Vec<String>>,
}

impl UNode {
    /// What must be equal for two nodes to share a block.
    pub fn label(&self) -> String {
        let mut s = match self.kind {
            BlockKind::Op => self.op.clone(),
            BlockKind::Const => "const".into(),
            BlockKind::Input => "input".into(),
            BlockKind::Acc => "acc".into(),
            BlockKind::Output => "output".into(),
        };
        let vals: Vec<&str> = SEMANTIC_ATTRS
            .iter()
            .filter_map(|k| self.attrs.get(*k).map(String::as_str))
            .collect();
        if !vals.is_empty() {
            s.push_str(&format!("[{}]", vals.join(",")));
        }
        s
    }
}

/// A unit: its graph, its tuples and the mined-pattern occurrences it is covered with.
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    pub id: String,
    pub class: String,
    pub family: Family,
    /// Tuple names, in the datapath's order.
    pub tuples: Vec<String>,
    pub nodes: Vec<UNode>,
    /// `(pattern, nodes in pattern-vertex order)`.
    pub groups: Vec<(String, Vec<u32>)>,
}

pub fn family_of(dp: &Datapath) -> Family {
    match (dp.route, dp.class) {
        (Route::Template, _) => Family::Stage,
        (_, SkelClass::Map) => Family::Map,
        (_, SkelClass::Reduce) => Family::Reduce,
        (_, SkelClass::Contraction) => Family::Contraction,
    }
}

/// The units of one datapath: the class, or each stage of a template class that has vertices; a unit
/// whose tuples cannot share block types (two float formats) is split into parts with the same id, one
/// per group of tuples that can.
pub fn units_of(dp: &Datapath) -> R<Vec<Unit>> {
    let whole = match dp.route {
        Route::Datapath => vec![unit(dp, None)?],
        Route::Template => {
            let stages: BTreeSet<Stage> = dp.vertices.iter().map(|v| v.stage).collect();
            stages
                .into_iter()
                .map(|st| unit(dp, Some(st)))
                .collect::<R<Vec<_>>>()?
        }
    };
    Ok(whole.into_iter().flat_map(split).collect())
}

/// The tuples of `u` in groups whose node types join (first fit, in tuple order).
fn split(u: Unit) -> Vec<Unit> {
    let joins = |ks: &[usize]| {
        u.nodes.iter().all(|n| {
            let widen = super::fu::widen_ok(n.kind, &n.op, &n.attrs, &n.sigs[ks[0]]);
            let mut acc = n.sigs[ks[0]].clone();
            ks[1..]
                .iter()
                .all(|&k| match super::fu::join_sig(&acc, &n.sigs[k], widen) {
                    Some(j) => {
                        acc = j;
                        true
                    }
                    None => false,
                })
        })
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for k in 0..u.tuples.len() {
        let mut placed = false;
        for g in &mut groups {
            g.push(k);
            if joins(g) {
                placed = true;
                break;
            }
            g.pop();
        }
        if !placed {
            groups.push(vec![k]);
        }
    }
    if groups.len() <= 1 {
        return vec![u];
    }
    groups
        .into_iter()
        .map(|g| {
            let mut p = u.clone();
            p.tuples = g.iter().map(|&k| u.tuples[k].clone()).collect();
            for (n, src) in p.nodes.iter_mut().zip(&u.nodes) {
                n.sigs = g.iter().map(|&k| src.sigs[k].clone()).collect();
            }
            p
        })
        .collect()
}

/// The stages of a template class without vertices (`prologue0`, `prologue1` for the core's two
/// operands, `epilogue`).
pub fn empty_stages(dp: &Datapath) -> Vec<String> {
    if dp.route != Route::Template {
        return Vec::new();
    }
    let has: BTreeSet<Stage> = dp.vertices.iter().map(|v| v.stage).collect();
    let mut out = Vec::new();
    for p in 0..2 {
        if !has.contains(&Stage::Prologue { port: p }) {
            out.push(format!("prologue{p}"));
        }
    }
    if !has.contains(&Stage::Epilogue) {
        out.push("epilogue".into());
    }
    out
}

fn unit(dp: &Datapath, stage: Option<Stage>) -> R<Unit> {
    let id = match stage {
        None => dp.id.clone(),
        Some(st) => {
            stage_unit_id(&dp.id, st).ok_or_else(|| format!("{}: stage {st:?} in a template", dp.id))?
        }
    };
    let in_stage = |s: Stage| stage.is_none_or(|st| st == s);
    let members: BTreeSet<u32> = dp
        .vertices
        .iter()
        .filter(|v| in_stage(v.stage))
        .map(|v| v.id)
        .collect();
    // Constants read by the members belong to the unit too.
    let mut keep = members.clone();
    for v in dp.vertices.iter().filter(|v| members.contains(&v.id)) {
        for a in &v.args {
            if let Src::Vertex { v: u, .. } = a.src
                && dp.vertices.get(u as usize).is_some_and(|x| x.op == "const")
            {
                keep.insert(u);
            }
        }
    }
    let outputs: Vec<&crate::datapaths::json::Output> = dp
        .outputs
        .iter()
        .filter(|o| match stage {
            None => true,
            Some(Stage::Epilogue) => o.kind == OutputKind::Output,
            Some(_) => {
                o.kind == OutputKind::CoreOperand
                    && matches!(o.src, Src::Vertex { v, .. } if members.contains(&v))
            }
        })
        .collect();
    let acc = dp.accumulator.as_ref().filter(|_| stage.is_none());

    let mut b = Builder {
        dp,
        nodes: Vec::new(),
        at: HashMap::new(),
    };
    // Sources first, in a fixed order: inputs, results, accumulators.
    for v in dp.vertices.iter().filter(|v| keep.contains(&v.id)) {
        for a in &v.args {
            b.source(&a.src, acc.is_some())?;
        }
    }
    for o in &outputs {
        b.source(&o.src, acc.is_some())?;
    }
    if let Some(a) = acc {
        for i in 0..a.entries.len() {
            b.acc(i as u32)?;
        }
        for e in &a.entries {
            b.source(&e.next, true)?;
            b.source(&e.init, true)?;
        }
    }
    for v in dp.vertices.iter().filter(|v| keep.contains(&v.id)) {
        b.vertex(v.id)?;
    }
    if let Some(a) = acc {
        for (i, e) in a.entries.iter().enumerate() {
            let next = b.src_node(&e.next, true)?;
            let init = b.src_node(&e.init, true)?;
            let n = b.at[&format!("acc{i}")];
            b.nodes[n as usize].args = vec![next, init];
        }
    }
    for o in &outputs {
        let s = b.src_node(&o.src, acc.is_some())?;
        let sigs = dp
            .tuples
            .iter()
            .map(|t| Ok(vec![ty_of_width(t, &o.width)?.name()]))
            .collect::<R<Vec<_>>>()?;
        b.push(UNode {
            name: format!("out:{}", o.name),
            kind: BlockKind::Output,
            op: "output".into(),
            attrs: BTreeMap::new(),
            var: Some(o.name.clone()),
            args: vec![s],
            commutative: false,
            sigs,
        });
    }
    Ok(Unit {
        id,
        class: dp.id.clone(),
        family: family_of(dp),
        tuples: dp.tuples.iter().map(|t| t.name.clone()).collect(),
        nodes: b.nodes,
        groups: Vec::new(),
    })
}

struct Builder<'a> {
    dp: &'a Datapath,
    nodes: Vec<UNode>,
    at: HashMap<String, u32>,
}

impl Builder<'_> {
    fn push(&mut self, n: UNode) -> u32 {
        let i = self.nodes.len() as u32;
        self.at.insert(n.name.clone(), i);
        self.nodes.push(n);
        i
    }

    fn typed(&self, w: &str) -> R<Vec<Vec<String>>> {
        self.dp
            .tuples
            .iter()
            .map(|t| Ok(vec![ty_of_width(t, w)?.name()]))
            .collect()
    }

    fn res_width(&self, i: u32) -> R<String> {
        if let Some(a) = &self.dp.accumulator {
            return a
                .entries
                .get(i as usize)
                .map(|e| e.width.clone())
                .ok_or_else(|| format!("result {i}: no accumulator"));
        }
        self.dp
            .inputs
            .iter()
            .find(|x| x.name == format!("core_result{i}"))
            .map(|x| x.width.clone())
            .ok_or_else(|| format!("result {i}: no core result input"))
    }

    /// Make the node of a source that is not a vertex.
    fn source(&mut self, s: &Src, has_acc: bool) -> R<()> {
        match s {
            Src::Vertex { .. } | Src::Acc { .. } => Ok(()),
            Src::Result { .. } if has_acc => Ok(()),
            Src::Input { name } => {
                let key = format!("in:{name}");
                if self.at.contains_key(&key) {
                    return Ok(());
                }
                let i = self
                    .dp
                    .inputs
                    .iter()
                    .find(|x| &x.name == name)
                    .ok_or_else(|| format!("input {name} is not declared"))?;
                let sigs = self.typed(&i.width)?;
                self.push(UNode {
                    name: key,
                    kind: BlockKind::Input,
                    op: "input".into(),
                    attrs: BTreeMap::new(),
                    var: Some(name.clone()),
                    args: Vec::new(),
                    commutative: false,
                    sigs,
                });
                Ok(())
            }
            Src::Result { index } => {
                let var = format!("res{index}");
                let key = format!("in:{var}");
                if self.at.contains_key(&key) {
                    return Ok(());
                }
                let w = self.res_width(*index)?;
                let sigs = self.typed(&w)?;
                self.push(UNode {
                    name: key,
                    kind: BlockKind::Input,
                    op: "input".into(),
                    attrs: BTreeMap::new(),
                    var: Some(var),
                    args: Vec::new(),
                    commutative: false,
                    sigs,
                });
                Ok(())
            }
        }
    }

    fn acc(&mut self, i: u32) -> R<()> {
        let a = self.dp.accumulator.as_ref().ok_or("no accumulator")?;
        let e = a.entries.get(i as usize).ok_or("accumulator index")?;
        let sigs = self.typed(&e.width)?;
        self.push(UNode {
            name: format!("acc{i}"),
            kind: BlockKind::Acc,
            op: "acc".into(),
            attrs: BTreeMap::new(),
            var: Some(format!("acc{i}")),
            args: Vec::new(),
            commutative: false,
            sigs,
        });
        Ok(())
    }

    fn src_node(&self, s: &Src, has_acc: bool) -> R<(u32, u32)> {
        let get = |k: &str| {
            self.at
                .get(k)
                .copied()
                .ok_or_else(|| format!("{}: {k} has no node", self.dp.id))
        };
        Ok(match s {
            Src::Vertex { v, res } => (get(&format!("v{v}"))?, *res),
            Src::Input { name } => (get(&format!("in:{name}"))?, 0),
            Src::Acc { index } => (get(&format!("acc{index}"))?, 0),
            Src::Result { index } if has_acc => (get(&format!("acc{index}"))?, 1),
            Src::Result { index } => (get(&format!("in:res{index}"))?, 0),
        })
    }

    fn vertex(&mut self, id: u32) -> R<()> {
        let dp = self.dp;
        let v = dp
            .vertices
            .get(id as usize)
            .filter(|v| v.id == id)
            .ok_or_else(|| format!("{}: vertex {id}", dp.id))?;
        let has_acc = dp.accumulator.is_some();
        let mut attrs: BTreeMap<String, String> = SEMANTIC_ATTRS
            .iter()
            .filter_map(|k| v.attrs.get(*k).map(|x| (k.to_string(), x.clone())))
            .collect();
        if v.op == "const" {
            let (var, sigs) = match (v.attrs.get("column"), v.attrs.get("value")) {
                (Some(col), _) => (Some(col.clone()), self.typed(&v.width)?),
                (None, Some(val)) => {
                    attrs.insert("value".into(), val.clone());
                    (None, self.typed(&v.width)?)
                }
                _ => return Err(format!("{}: constant {id} without a column or value", dp.id)),
            };
            self.push(UNode {
                name: format!("v{id}"),
                kind: BlockKind::Const,
                op: "const".into(),
                attrs,
                var,
                args: Vec::new(),
                commutative: false,
                sigs,
            });
            return Ok(());
        }
        let mut args = Vec::new();
        for a in &v.args {
            args.push(self.src_node(&a.src, has_acc)?);
        }
        let commutative = v.args.len() == 2 && v.args.iter().all(|a| a.port == "*");
        let mut sigs = Vec::new();
        for (k, t) in dp.tuples.iter().enumerate() {
            let mut s = Vec::new();
            for &(n, _) in &args {
                let node = &self.nodes[n as usize];
                let ty = match node.kind {
                    BlockKind::Op => result_type(&node.sigs[k]).cloned(),
                    _ => node.sigs[k].first().cloned(),
                };
                s.push(ty.ok_or_else(|| format!("{}: vertex {id}: operand type", dp.id))?);
            }
            s.push(ty_of_width(t, &v.width)?.name());
            let col = |k: &str| v.attrs.get(k).and_then(|c| t.attrs.get(c)).copied();
            if let (Some(e), Some(m)) = (col("exponent_bits"), col("mantissa_bits")) {
                s.push(format!("rp:e{e}m{m}"));
            }
            sigs.push(s);
        }
        self.push(UNode {
            name: format!("v{id}"),
            kind: BlockKind::Op,
            op: v.op.clone(),
            attrs,
            var: None,
            args,
            commutative,
            sigs,
        });
        Ok(())
    }
}

/// The result type of a node's signature (operators: the last type before any token).
pub fn result_type(sig: &[String]) -> Option<&String> {
    sig.iter().rev().find(|x| !x.starts_with("rp:"))
}

/// Cover a unit with occurrences of the selected patterns: patterns in selection (score) order, each
/// pattern's independent occurrences in this unit, kept when they share no vertex with one already
/// taken. `occ` lists, per pattern id in order, the datapath vertex ids of each chosen occurrence.
pub fn contract(u: &mut Unit, occ: &[(String, Vec<Vec<u32>>)]) {
    let at: HashMap<&str, u32> = u
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.as_str(), i as u32))
        .collect();
    let mut taken: BTreeSet<u32> = BTreeSet::new();
    let mut groups = Vec::new();
    for (pid, occs) in occ {
        for vs in occs {
            let nodes: Option<Vec<u32>> = vs
                .iter()
                .map(|v| at.get(format!("v{v}").as_str()).copied())
                .collect();
            let Some(nodes) = nodes else { continue };
            if nodes.len() < 2 || nodes.iter().any(|n| taken.contains(n)) {
                continue;
            }
            taken.extend(nodes.iter().copied());
            groups.push((pid.clone(), nodes));
        }
    }
    u.groups = groups;
}
