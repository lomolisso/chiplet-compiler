//! Extraction (docs/passes/datapaths.md §4.1): the scalar datapath of one iteration of a class, from its
//! representative kernel graph, its skeleton and its route. The rules come from the op table only:
//!
//! - an element-wise member becomes one vertex per access instance (a member read with two accesses is
//!   two values), labelled `<op>@<family>`, with its result's element class as width variable;
//!   `convert` stays explicit; GT/GE are written LT/LE with the operands exchanged (as class identity);
//! - a broadcast is the identity on one element: it has no vertex, its consumers read its operand;
//! - a folded constant is a `const` vertex whose value is the instance table's column;
//! - a contraction core becomes `multiply` (after converting both operands to the result's element type,
//!   the spec's `dot_general` multiplies elements of one type) and the accumulator's `add`, init 0;
//! - a reduction or scatter core becomes its body's ops on the accumulators; a scatter's indices are an
//!   address input that no vertex reads;
//! - a template class keeps only its stages: each prologue ends in a core-operand output, and the
//!   epilogue reads the core's results.

use std::collections::{BTreeMap, HashMap};

use cc_ir::Attr;
use cc_ir::absgraph::{Graph, PortLabel};

use crate::components::algo::graph::{AAttr, Kind};
use crate::datapaths::json::{
    AccEntry, Accumulator, Arg, DATAPATH_SCHEMA, Datapath, From, Input, InputKind, Output, OutputKind, Src,
    Stage, Vertex,
};
use crate::iteration_space::json::{Access, Direction, Route, Routed, SkelClass, Skeleton};
use crate::kernel_classes::algo::local::{op_of, operands};
use crate::kernel_classes::json::ClassFile;
use crate::kernels::json::{KNode, Role};
use crate::normalize::algo::omega::family_key;

type Ops = Vec<Vec<Option<(u32, u32)>>>;

/// The core's kind, from its op (the op table's iteration semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreKind {
    /// `dot_general`, `convolution`, `dynamic_conv`: multiply and add.
    Contraction,
    /// `reduce`, `reduce_window`, `dynamic_reduce_window`: the body on (accumulators, inputs).
    Reduce,
    /// `scatter`: the body on (current results, updates); the indices are addresses.
    Scatter,
}

pub fn core_kind(name: &str, target: Option<&str>) -> Option<CoreKind> {
    match (name, target) {
        ("stablehlo.dot_general" | "stablehlo.convolution" | "stablehlo.dynamic_conv", _) => {
            Some(CoreKind::Contraction)
        }
        ("stablehlo.reduce" | "stablehlo.reduce_window", _)
        | ("stablehlo.custom_call", Some("stablehlo.dynamic_reduce_window")) => Some(CoreKind::Reduce),
        ("stablehlo.scatter", _) => Some(CoreKind::Scatter),
        _ => None,
    }
}

/// The spec name of an op (without `stablehlo.`).
pub fn short(name: &str) -> &str {
    name.strip_prefix("stablehlo.").unwrap_or(name)
}

/// The operands of node `n` by port, folded constants included (`Err(column)` for a folded port).
pub fn ports(g: &Graph<KNode>, ops: &Ops, n: u32) -> Vec<Result<(u32, u32), String>> {
    let KNode::Member { node, .. } = &g.nodes[n as usize] else {
        return Vec::new();
    };
    let edges = &ops[n as usize];
    let mut out: Vec<Option<Result<(u32, u32), String>>> = edges.iter().map(|e| e.map(Ok)).collect();
    for (j, s) in node.splats.iter().enumerate() {
        let col = format!("n{n}.splat{j}");
        match s.port {
            PortLabel::Port(p) => {
                if out.len() <= p as usize {
                    out.resize(p as usize + 1, None);
                }
                out[p as usize] = Some(Err(col));
            }
            // A commutative operand: the first free port.
            _ => match out.iter().position(Option::is_none) {
                Some(i) => out[i] = Some(Err(col)),
                None => out.push(Some(Err(col))),
            },
        }
    }
    out.into_iter()
        .map(|x| x.unwrap_or_else(|| Err(String::new())))
        .collect()
}

fn role(x: &KNode) -> Option<Role> {
    match x {
        KNode::Member { role, .. } => Some(*role),
        _ => None,
    }
}

/// A literal enum attribute of a member (`comparison_direction`, `compare_type`).
pub fn enum_attr(x: &KNode, key: &str) -> Option<String> {
    let KNode::Member { node, .. } = x else {
        return None;
    };
    match node.label.attrs.get(key)? {
        AAttr::Lit(Attr::Enum { value, .. }) => Some(value.clone()),
        _ => None,
    }
}

struct Ex<'a> {
    class: &'a ClassFile,
    skel: &'a Skeleton,
    g: &'a Graph<KNode>,
    ops: Ops,
    vertices: Vec<Vertex>,
    memo: HashMap<(u32, usize), Src>,
    consts: HashMap<String, Src>,
    body: HashMap<u32, Src>,
    inputs: BTreeMap<String, Input>,
    widths: BTreeMap<String, String>,
    /// Region argument `i` of the core's body reads this.
    bind: Vec<Src>,
    /// Core result `i` is this (a scatter applied once: its body's output).
    results: Option<Vec<Src>>,
    /// The prologue port of each prologue member.
    port_of: HashMap<u32, u32>,
}

impl Ex<'_> {
    fn width(&mut self, n: u32, r: u32) -> Result<String, String> {
        let e = *self
            .class
            .local
            .elems
            .get(n as usize)
            .and_then(|x| x.get(r as usize))
            .ok_or_else(|| format!("node {n} result {r}: no element class"))?;
        let fam = self
            .class
            .params
            .elems
            .get(e as usize)
            .map(|f| family_key(f).to_string())
            .ok_or_else(|| format!("element class {e}: no family"))?;
        let w = format!("w{e}");
        self.widths.insert(w.clone(), fam);
        Ok(w)
    }

    fn family_of(&self, w: &str) -> String {
        self.widths.get(w).cloned().unwrap_or_default()
    }

    fn stage(&self, n: u32) -> Stage {
        match role(&self.g.nodes[n as usize]) {
            Some(Role::Prologue) => Stage::Prologue {
                port: self.port_of.get(&n).copied().unwrap_or(0),
            },
            Some(Role::Epilogue) => Stage::Epilogue,
            Some(Role::Body | Role::Core) => Stage::Core,
            _ => Stage::Map,
        }
    }

    fn instance(&self, n: u32, r: u32, a: &Access) -> Result<usize, String> {
        self.skel
            .accesses
            .iter()
            .position(|x| x.node == n && x.results.get(r as usize) == Some(a))
            .ok_or_else(|| format!("node {n} result {r}: no access instance for the requested access"))
    }

    fn constant(&mut self, col: &str, n: u32, stage: Stage) -> Result<Src, String> {
        if let Some(s) = self.consts.get(col) {
            return Ok(s.clone());
        }
        let KNode::Member { node, .. } = &self.g.nodes[n as usize] else {
            return Err(format!("{col}: not a member"));
        };
        let j: usize = col
            .rsplit("splat")
            .next()
            .and_then(|x| x.parse().ok())
            .ok_or_else(|| format!("{col}: bad column"))?;
        let fam = node
            .splats
            .get(j)
            .map(|s| family_key(&s.ty.family).to_string())
            .ok_or_else(|| format!("{col}: no splat"))?;
        self.widths.insert(col.to_string(), fam.clone());
        let id = self.vertices.len() as u32;
        self.vertices.push(Vertex {
            id,
            label: format!("const@{fam}"),
            op: "const".into(),
            family: fam,
            width: col.to_string(),
            attrs: BTreeMap::from([("column".into(), col.to_string())]),
            stage,
            args: Vec::new(),
            from: From {
                node: n,
                access: None,
            },
        });
        let s = Src::Vertex { v: id, res: 0 };
        self.consts.insert(col.to_string(), s.clone());
        Ok(s)
    }

    fn input(&mut self, n: u32, a: &Access, kind: InputKind) -> Result<Src, String> {
        let o = self
            .skel
            .operands
            .iter()
            .find(|o| o.direction == Direction::In && !o.core && o.node == n && &o.access == a)
            .ok_or_else(|| format!("input {n}: no skeleton operand with this access"))?;
        let name = o.name.clone();
        let incidence = (0..self.skel.dims.len() as u32)
            .filter(|d| !o.reuse.contains(d))
            .collect();
        let width = self.width(n, 0)?;
        self.inputs.entry(name.clone()).or_insert(Input {
            name: name.clone(),
            kind,
            node: n,
            res: 0,
            incidence,
            width,
        });
        Ok(Src::Input { name })
    }

    /// The value of result `r` of node `n` at access `a`, read by a vertex of stage `by`.
    fn value(&mut self, n: u32, r: u32, a: &Access, by: Stage) -> Result<Src, String> {
        match &self.g.nodes[n as usize] {
            KNode::Input { .. } => {
                let kind = match by {
                    Stage::Prologue { .. } => InputKind::Prologue,
                    Stage::Core => InputKind::CoreOperand,
                    Stage::Epilogue => InputKind::EpilogueParam,
                    Stage::Map => InputKind::Map,
                };
                self.input(n, a, kind)
            }
            KNode::Output { .. } => Err(format!("node {n}: an output has no value")),
            KNode::Member { role, .. } => match role {
                Role::Core => match &self.results {
                    Some(rs) => rs
                        .get(r as usize)
                        .cloned()
                        .ok_or_else(|| format!("core result {r}")),
                    None => Ok(Src::Result { index: r }),
                },
                Role::Body => self.body_value(n),
                Role::Broadcast => {
                    let k = self.instance(n, r, a)?;
                    let a0 = self.skel.accesses[k]
                        .operands
                        .first()
                        .cloned()
                        .flatten()
                        .unwrap_or_default();
                    match ports(self.g, &self.ops, n).first().cloned() {
                        Some(Ok((pn, pr))) => self.value(pn, pr, &a0, by),
                        Some(Err(col)) if !col.is_empty() => self.constant(&col, n, by),
                        _ => Err(format!("broadcast {n} without an operand")),
                    }
                }
                _ => {
                    let k = self.instance(n, r, a)?;
                    self.vertex(n, k)
                }
            },
        }
    }

    /// The vertex of member `n` at access instance `k`.
    fn vertex(&mut self, n: u32, k: usize) -> Result<Src, String> {
        if let Some(s) = self.memo.get(&(n, k)) {
            return Ok(s.clone());
        }
        let stage = self.stage(n);
        let accs = self.skel.accesses[k].operands.clone();
        let mut args = Vec::new();
        for (p, port) in ports(self.g, &self.ops, n).into_iter().enumerate() {
            args.push(match port {
                Ok((pn, pr)) => {
                    let a = accs
                        .get(p)
                        .cloned()
                        .flatten()
                        .ok_or_else(|| format!("node {n} port {p}: no access"))?;
                    self.value(pn, pr, &a, stage)?
                }
                Err(col) if !col.is_empty() => self.constant(&col, n, stage)?,
                Err(_) => return Err(format!("node {n} port {p}: no operand")),
            });
        }
        let s = self.op_vertex(n, Some(k as u32), stage, args)?;
        self.memo.insert((n, k), s.clone());
        Ok(s)
    }

    /// A vertex for the op of node `n` on `args` (ports in the op's order).
    fn op_vertex(
        &mut self,
        n: u32,
        access: Option<u32>,
        stage: Stage,
        mut args: Vec<Src>,
    ) -> Result<Src, String> {
        let x = &self.g.nodes[n as usize];
        let (name, _) = op_of(x).ok_or_else(|| format!("node {n} is not an op"))?;
        let op = short(name).to_string();
        let spec = cc_ir::lookup(name, None).ok_or_else(|| format!("`{name}` is not in the op table"))?;
        let mut attrs = BTreeMap::new();
        if let Some(d) = enum_attr(x, "comparison_direction") {
            // a > b ⇔ b < a, a ≥ b ⇔ b ≤ a (NaN included), as kernel class identity writes them.
            let d = match d.as_str() {
                "GT" => {
                    args.swap(0, 1);
                    "LT".to_string()
                }
                "GE" => {
                    args.swap(0, 1);
                    "LE".to_string()
                }
                _ => d,
            };
            attrs.insert("comparison_direction".into(), d);
        }
        if let Some(t) = enum_attr(x, "compare_type") {
            attrs.insert("compare_type".into(), t);
        }
        for key in ["exponent_bits", "mantissa_bits"] {
            if let KNode::Member { node, .. } = x
                && node.label.attrs.contains_key(key)
            {
                attrs.insert(key.into(), format!("n{n}.{key}"));
            }
        }
        let width = self.width(n, 0)?;
        let family = self.family_of(&width);
        let id = self.vertices.len() as u32;
        self.vertices.push(Vertex {
            id,
            label: format!("{op}@{family}"),
            op,
            family,
            width,
            attrs,
            stage,
            args: args
                .into_iter()
                .enumerate()
                .map(|(p, src)| Arg {
                    src,
                    port: if spec.commutative {
                        "*".into()
                    } else {
                        format!("p{p}")
                    },
                })
                .collect(),
            from: From { node: n, access },
        });
        Ok(Src::Vertex { v: id, res: 0 })
    }

    /// A body node's value (region arguments bound by `bind`).
    fn body_value(&mut self, n: u32) -> Result<Src, String> {
        if let Some(s) = self.body.get(&n) {
            return Ok(s.clone());
        }
        let s = match &self.g.nodes[n as usize] {
            KNode::Member { node, .. } => match node.label.kind {
                Kind::RegionArg { index, .. } => self
                    .bind
                    .get(index as usize)
                    .cloned()
                    .ok_or_else(|| format!("region argument {index} unbound"))?,
                _ => {
                    let mut args = Vec::new();
                    for (p, port) in ports(self.g, &self.ops, n).into_iter().enumerate() {
                        args.push(match port {
                            Ok((pn, _)) => self.body_value(pn)?,
                            Err(col) if !col.is_empty() => self.constant(&col, n, Stage::Core)?,
                            Err(_) => return Err(format!("body node {n} port {p}: no operand")),
                        });
                    }
                    self.op_vertex(n, None, Stage::Core, args)?
                }
            },
            _ => return Err(format!("node {n} is not in the body")),
        };
        self.body.insert(n, s.clone());
        Ok(s)
    }

    /// The terminator of the core's body: its operands by port.
    fn returns(&self, core: u32) -> Result<Vec<u32>, String> {
        for (u, o) in self.g.out.iter().enumerate() {
            for (v, ends) in o {
                if *v == core && ends.iter().any(|e| matches!(e.port, PortLabel::Yield { .. })) {
                    return Ok(self.ops[u]
                        .iter()
                        .map(|p| p.map_or(u32::MAX, |(q, _)| q))
                        .collect());
                }
            }
        }
        Err(format!("core {core}: no body terminator"))
    }

    /// The value of core operand port `p`, read at the core's operand access.
    fn core_operand(&mut self, core: u32, p: usize) -> Result<Src, String> {
        let k = self
            .skel
            .accesses
            .iter()
            .position(|x| x.node == core)
            .ok_or("the core has no access")?;
        match ports(self.g, &self.ops, core).get(p).cloned() {
            Some(Ok((pn, pr))) => {
                let a = self.skel.accesses[k]
                    .operands
                    .get(p)
                    .cloned()
                    .flatten()
                    .ok_or_else(|| format!("core port {p}: no access"))?;
                self.value(pn, pr, &a, Stage::Core)
            }
            Some(Err(col)) if !col.is_empty() => self.constant(&col, core, Stage::Core),
            _ => Err(format!("core port {p}: no operand")),
        }
    }
}

/// The datapath of class `class` (the tuples are filled by the caller).
pub fn extract(
    class: &ClassFile,
    skel: &Skeleton,
    routed: &Routed,
    g: &Graph<KNode>,
) -> Result<Datapath, String> {
    let ops = operands(g);
    let mut port_of = HashMap::new();
    for (p, ms) in &skel.stages.prologue {
        for m in ms {
            port_of.insert(*m, *p);
        }
    }
    let mut ex = Ex {
        class,
        skel,
        g,
        ops,
        vertices: Vec::new(),
        memo: HashMap::new(),
        consts: HashMap::new(),
        body: HashMap::new(),
        inputs: BTreeMap::new(),
        widths: BTreeMap::new(),
        bind: Vec::new(),
        results: None,
        port_of,
    };
    let mut outputs = Vec::new();
    let mut accumulator = None;
    let template = routed.route == Route::Template;

    if let Some(core) = skel.core_node {
        let x = &g.nodes[core as usize];
        let (name, target) = op_of(x).ok_or("the core is not an op")?;
        let kind = core_kind(name, target).ok_or_else(|| format!("core `{name}`: no core semantics"))?;
        let core_ops: Vec<&crate::iteration_space::json::Operand> = skel
            .operands
            .iter()
            .filter(|o| o.core && o.direction == Direction::In)
            .collect();
        let nres = crate::kernel_classes::algo::local::results(x).len();
        if template {
            // Stages only: the prologue of every core input ends in a core-operand output; the epilogue
            // reads the core results.
            for o in &core_ops {
                if !crate::iteration_space::algo::skeleton::is_core_input(&o.name) {
                    continue;
                }
                let p = port_index(&ex, core, o.node, o.res)?;
                let src = ex.core_operand(core, p)?;
                let width = ex.width(o.node, o.res)?;
                outputs.push(Output {
                    name: o.name.clone(),
                    kind: OutputKind::CoreOperand,
                    src,
                    width,
                });
            }
            for i in 0..nres as u32 {
                let width = ex.width(core, i)?;
                ex.inputs.insert(
                    format!("core_result{i}"),
                    Input {
                        name: format!("core_result{i}"),
                        kind: InputKind::CoreResult,
                        node: core,
                        res: i,
                        incidence: skel
                            .operands
                            .iter()
                            .find(|o| o.core && o.direction == Direction::Out && o.res == i)
                            .map(|o| {
                                (0..skel.dims.len() as u32)
                                    .filter(|d| !o.reuse.contains(d))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        width,
                    },
                );
            }
        } else {
            match kind {
                CoreKind::Contraction => {
                    let w = ex.width(core, 0)?;
                    let lhs = ex.core_operand(core, 0)?;
                    let rhs = ex.core_operand(core, 1)?;
                    let conv = |ex: &mut Ex, src: Src, p: usize| -> Result<Src, String> {
                        let (pn, pr) = match ports(g, &ex.ops, core).get(p).cloned() {
                            Some(Ok(x)) => x,
                            _ => return Ok(src),
                        };
                        let we = ex.width(pn, pr)?;
                        if we == w {
                            return Ok(src);
                        }
                        // dot_general's spec multiplies elements of the result's type.
                        let fam = ex.family_of(&w);
                        let id = ex.vertices.len() as u32;
                        ex.vertices.push(Vertex {
                            id,
                            label: format!("convert@{fam}"),
                            op: "convert".into(),
                            family: fam,
                            width: w.clone(),
                            attrs: BTreeMap::new(),
                            stage: Stage::Core,
                            args: vec![Arg {
                                src,
                                port: "p0".into(),
                            }],
                            from: From {
                                node: core,
                                access: None,
                            },
                        });
                        Ok(Src::Vertex { v: id, res: 0 })
                    };
                    let l = conv(&mut ex, lhs, 0)?;
                    let r = conv(&mut ex, rhs, 1)?;
                    let fam = ex.family_of(&w);
                    let mk = |ex: &mut Ex, op: &str, a: Src, b: Src, attrs: BTreeMap<String, String>| {
                        let id = ex.vertices.len() as u32;
                        ex.vertices.push(Vertex {
                            id,
                            label: format!("{op}@{fam}"),
                            op: op.into(),
                            family: fam.clone(),
                            width: w.clone(),
                            attrs,
                            stage: Stage::Core,
                            args: [a, b]
                                .into_iter()
                                .map(|src| Arg {
                                    src,
                                    port: "*".into(),
                                })
                                .collect(),
                            from: From {
                                node: core,
                                access: None,
                            },
                        });
                        id
                    };
                    let m = mk(&mut ex, "multiply", l, r, BTreeMap::new());
                    let add = mk(
                        &mut ex,
                        "add",
                        Src::Acc { index: 0 },
                        Src::Vertex { v: m, res: 0 },
                        BTreeMap::new(),
                    );
                    let zid = ex.vertices.len() as u32;
                    ex.vertices.push(Vertex {
                        id: zid,
                        label: format!("const@{fam}"),
                        op: "const".into(),
                        family: fam.clone(),
                        width: w.clone(),
                        attrs: BTreeMap::from([("value".into(), "0".into())]),
                        stage: Stage::Core,
                        args: Vec::new(),
                        from: From {
                            node: core,
                            access: None,
                        },
                    });
                    accumulator = Some(Accumulator {
                        entries: vec![AccEntry {
                            width: w.clone(),
                            init: Src::Vertex { v: zid, res: 0 },
                            next: Src::Vertex { v: add, res: 0 },
                        }],
                        combiner: vec![add],
                        omega: skel.omega.clone(),
                        order: Vec::new(),
                    });
                }
                CoreKind::Reduce | CoreKind::Scatter => {
                    let n = nres;
                    // Spec operand layout: reduce (inputs, inits); scatter (inputs, indices, updates).
                    let (xs, inits): (Vec<usize>, Vec<usize>) = match kind {
                        CoreKind::Reduce => ((0..n).collect(), (n..2 * n).collect()),
                        _ => ((n + 1..2 * n + 1).collect(), (0..n).collect()),
                    };
                    if kind == CoreKind::Scatter {
                        // The indices: an address input, read by no vertex (their computation is
                        // address generation and is not extracted).
                        if let Some(o) = skel.operands.iter().find(|o| o.core && o.name == "indices") {
                            let width = ex.width(o.node, o.res)?;
                            ex.inputs.insert(
                                "indices".into(),
                                Input {
                                    name: "indices".into(),
                                    kind: InputKind::Address,
                                    node: o.node,
                                    res: o.res,
                                    incidence: (0..skel.dims.len() as u32)
                                        .filter(|d| !o.reuse.contains(d))
                                        .collect(),
                                    width,
                                },
                            );
                        }
                    }
                    let mut xv = Vec::new();
                    for &p in &xs {
                        xv.push(ex.core_operand(core, p)?);
                    }
                    let mut iv = Vec::new();
                    for &p in &inits {
                        iv.push(ex.core_operand(core, p)?);
                    }
                    let once = skel.class == SkelClass::Map;
                    ex.bind = if once {
                        // A scatter without a reduction: the body applied once to (operand, update).
                        iv.iter().cloned().chain(xv.iter().cloned()).collect()
                    } else {
                        (0..n as u32)
                            .map(|i| Src::Acc { index: i })
                            .chain(xv.iter().cloned())
                            .collect()
                    };
                    let rets = ex.returns(core)?;
                    let first = ex.vertices.len() as u32;
                    let mut next = Vec::new();
                    for &r in rets.iter().take(n) {
                        next.push(ex.body_value(r)?);
                    }
                    let combiner: Vec<u32> = (first..ex.vertices.len() as u32)
                        .filter(|&v| ex.vertices[v as usize].op != "const")
                        .collect();
                    if once {
                        ex.results = Some(next);
                    } else {
                        let mut entries = Vec::new();
                        for i in 0..n {
                            entries.push(AccEntry {
                                width: ex.width(core, i as u32)?,
                                init: iv[i].clone(),
                                next: next[i].clone(),
                            });
                        }
                        accumulator = Some(Accumulator {
                            entries,
                            combiner,
                            omega: skel.omega.clone(),
                            order: Vec::new(),
                        });
                    }
                }
            }
        }
    }
    // The kernel's outputs.
    for o in skel
        .operands
        .iter()
        .filter(|o| o.direction == Direction::Out && !o.core)
    {
        let by = match role(&g.nodes[o.node as usize]) {
            Some(Role::Epilogue | Role::Core) => Stage::Epilogue,
            _ => Stage::Map,
        };
        let src = ex.value(o.node, o.res, &o.access, by)?;
        let width = ex.width(o.node, o.res)?;
        outputs.push(Output {
            name: o.name.clone(),
            kind: OutputKind::Output,
            src,
            width,
        });
    }
    Ok(Datapath {
        schema: DATAPATH_SCHEMA.into(),
        id: class.id.clone(),
        route: routed.route,
        class: skel.class,
        core: skel.core.clone(),
        widths: ex.widths,
        vertices: ex.vertices,
        inputs: ex.inputs.into_values().collect(),
        outputs,
        accumulator,
        tuples: Vec::new(),
    })
}

/// The core's port that reads (node, res).
fn port_index(ex: &Ex, core: u32, n: u32, r: u32) -> Result<usize, String> {
    ports(ex.g, &ex.ops, core)
        .iter()
        .position(|p| p.as_ref().is_ok_and(|x| *x == (n, r)))
        .ok_or_else(|| format!("core {core} does not read node {n}"))
}
