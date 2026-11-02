//! The kernel's semantics on symbolic scalars (docs/passes/datapaths.md §6): one iteration of the
//! representative kernel graph, evaluated with the StableHLO op semantics of `cc_smt::ops`, read directly
//! from the graph's members (original operand order and attributes), independently of the extraction.
//!
//! Values are named as the datapath names its inputs: a kernel input read with access `a` is the
//! skeleton operand of that node and access (`in<j>`, `in<j>.<u>`); a folded constant is its value column
//! (`n<i>.splat<j>`); accumulator `i` is `acc<i>`; the final value of core result `i` is `res<i>`. The
//! core follows the spec for one ω-`seq` step:
//!
//! - `dot_general`, `convolution`: `reduce(multiply(lhs, rhs), 0, add)` on the result's element type,
//!   so a step is `acc + convert(lhs) · convert(rhs)` and the init value is 0;
//! - `reduce`, `reduce_window`: `body(acc…, x…)` (the accumulators are the body's first arguments: the
//!   left-deep schedule with the init value first), the init values are the `init` operands;
//! - `scatter`: `update_computation(current…, update…)`, the init values are the operands' elements; a
//!   scatter without a reduction applies it once.

use std::collections::{BTreeMap, HashMap};

use cc_ir::absgraph::{Graph, PortLabel};
use cc_ir::{Attr, Elem};
use cc_smt::ops::{self, Attrs, Ty, Val};
use cc_smt::term::Ctx;

use crate::components::algo::graph::{AAttr, Kind};
use crate::datapaths::json::Tuple;
use crate::iteration_space::json::{Access, Direction, Route, Routed, SkelClass, Skeleton};
use crate::kernel_classes::json::ClassFile;
use crate::kernels::json::{KNode, Role};

type R<X> = Result<X, String>;

/// The scalar type of a type name of the instance table.
pub fn ty(name: &str) -> R<Ty> {
    let e = Elem::parse(name).ok_or_else(|| format!("element type {name}"))?;
    Ty::of(&e).map_err(|e| e.0)
}

/// The type of an element class or a constant column under a tuple.
pub fn ty_of_width(t: &Tuple, w: &str) -> R<Ty> {
    if let Some(e) = w.strip_prefix('w').and_then(|x| x.parse::<usize>().ok()) {
        return ty(t
            .elems
            .get(e)
            .ok_or_else(|| format!("{}: element class {e} out of range", t.name))?);
    }
    ty(t.consts
        .get(w)
        .ok_or_else(|| format!("{}: no type for {w}", t.name))?)
}

/// The attributes the semantics reads, with `reduce_precision`'s bits from the tuple.
pub fn attrs(direction: Option<String>, compare_type: Option<String>, bits: Option<(i64, i64)>) -> R<Attrs> {
    Ok(Attrs {
        direction,
        compare_type,
        precision: match bits {
            Some((e, m)) => Some((
                u32::try_from(e).map_err(|_| "exponent_bits".to_string())?,
                u32::try_from(m).map_err(|_| "mantissa_bits".to_string())?,
            )),
            None => None,
        },
    })
}

struct Sem<'a> {
    class: &'a ClassFile,
    skel: &'a Skeleton,
    g: &'a Graph<KNode>,
    t: &'a Tuple,
    /// Per node, its operands by port: an edge `(node, res)` or a folded constant's column.
    ports: Vec<Vec<Option<Port>>>,
    memo: HashMap<(u32, u32, String), Val>,
    /// Region argument values while evaluating the core's body.
    region: Vec<Val>,
    /// Core result values when the core's result is computed (a scatter applied once).
    results: Option<Vec<Val>>,
}

#[derive(Debug, Clone)]
enum Port {
    Edge(u32, u32),
    Splat(String),
}

/// Operands by port, written independently of the extraction's helper: edges in port order (commutative
/// operands in edge order), folded constants in their port or the first free one.
fn port_table(g: &Graph<KNode>) -> Vec<Vec<Option<Port>>> {
    let n = g.node_count();
    let mut out: Vec<Vec<Option<Port>>> = vec![Vec::new(); n];
    let mut comm = vec![0usize; n];
    for (u, o) in g.out.iter().enumerate() {
        for (v, ends) in o {
            for e in ends {
                let p = match e.port {
                    PortLabel::Port(p) => p as usize,
                    PortLabel::Commutative => {
                        comm[*v as usize] += 1;
                        comm[*v as usize] - 1
                    }
                    _ => continue,
                };
                let ps = &mut out[*v as usize];
                if ps.len() <= p {
                    ps.resize(p + 1, None);
                }
                ps[p] = Some(Port::Edge(u as u32, e.res));
            }
        }
    }
    for (i, x) in g.nodes.iter().enumerate() {
        let KNode::Member { node, .. } = x else { continue };
        for (j, s) in node.splats.iter().enumerate() {
            let col = Port::Splat(format!("n{i}.splat{j}"));
            let ps = &mut out[i];
            match s.port {
                PortLabel::Port(p) => {
                    let p = p as usize;
                    if ps.len() <= p {
                        ps.resize(p + 1, None);
                    }
                    ps[p] = Some(col);
                }
                _ => match ps.iter().position(Option::is_none) {
                    Some(k) => ps[k] = Some(col),
                    None => ps.push(Some(col)),
                },
            }
        }
    }
    out
}

fn key(a: &Access) -> String {
    serde_json::to_string(a).unwrap_or_default()
}

impl Sem<'_> {
    fn elem_ty(&self, n: u32, r: u32) -> R<Ty> {
        let e = self
            .class
            .local
            .elems
            .get(n as usize)
            .and_then(|x| x.get(r as usize))
            .ok_or_else(|| format!("node {n}: no element class"))?;
        ty_of_width(self.t, &format!("w{e}"))
    }

    fn splat(&self, c: &mut Ctx, col: &str) -> R<Val> {
        let t = ty_of_width(self.t, col)?;
        ops::input(c, col, t).map_err(|e| e.0)
    }

    fn port(&mut self, c: &mut Ctx, n: u32, p: usize, a: Option<&Access>) -> R<Val> {
        match self.ports[n as usize].get(p).cloned().flatten() {
            Some(Port::Splat(col)) => self.splat(c, &col),
            Some(Port::Edge(pn, pr)) => match a {
                Some(a) => self.value(c, pn, pr, a),
                None => self.body(c, pn, pr),
            },
            None => Err(format!("node {n} port {p}: no operand")),
        }
    }

    fn apply(&mut self, c: &mut Ctx, n: u32, args: &[Val]) -> R<Val> {
        let KNode::Member { node, .. } = &self.g.nodes[n as usize] else {
            return Err(format!("node {n} is not a member"));
        };
        let Kind::Op { name, .. } = &node.label.kind else {
            return Err(format!("node {n} is not an op"));
        };
        let en = |k: &str| match node.label.attrs.get(k) {
            Some(AAttr::Lit(Attr::Enum { value, .. })) => Some(value.clone()),
            _ => None,
        };
        let bits = match (
            self.t.attrs.get(&format!("n{n}.exponent_bits")),
            self.t.attrs.get(&format!("n{n}.mantissa_bits")),
        ) {
            (Some(e), Some(m)) => Some((*e, *m)),
            _ => None,
        };
        let a = attrs(en("comparison_direction"), en("compare_type"), bits)?;
        let ret = self.elem_ty(n, 0)?;
        let op = name.strip_prefix("stablehlo.").unwrap_or(name);
        ops::apply(c, op, &a, args, ret).map_err(|e| format!("node {n}: {}", e.0))
    }

    /// The value of result `r` of node `n` read at access `a`.
    fn value(&mut self, c: &mut Ctx, n: u32, r: u32, a: &Access) -> R<Val> {
        let k = (n, r, key(a));
        if let Some(v) = self.memo.get(&k) {
            return Ok(*v);
        }
        let v = match &self.g.nodes[n as usize] {
            KNode::Input { .. } => {
                let o = self
                    .skel
                    .operands
                    .iter()
                    .find(|o| o.direction == Direction::In && !o.core && o.node == n && &o.access == a)
                    .ok_or_else(|| format!("input {n}: not a skeleton operand at this access"))?;
                let t = self.elem_ty(n, 0)?;
                ops::input(c, &o.name, t).map_err(|e| e.0)?
            }
            KNode::Output { .. } => return Err(format!("node {n} is an output")),
            KNode::Member { role, .. } => {
                let inst = self
                    .skel
                    .accesses
                    .iter()
                    .find(|x| x.node == n && x.results.get(r as usize) == Some(a))
                    .cloned();
                match role {
                    Role::Core => match &self.results {
                        Some(rs) => *rs.get(r as usize).ok_or("core result")?,
                        None => {
                            let t = self.elem_ty(n, r)?;
                            ops::input(c, &format!("res{r}"), t).map_err(|e| e.0)?
                        }
                    },
                    Role::Body => self.body(c, n, r)?,
                    _ => {
                        let inst = inst.ok_or_else(|| format!("node {n}: no instance at this access"))?;
                        let nports = self.ports[n as usize].len();
                        // A broadcast is the identity on the element it reads (spec `broadcast_in_dim`).
                        if *role == Role::Broadcast {
                            let a0 = inst.operands.first().cloned().flatten();
                            self.port(c, n, 0, a0.as_ref())?
                        } else {
                            let mut args = Vec::new();
                            for p in 0..nports {
                                let a = inst.operands.get(p).cloned().flatten();
                                args.push(self.port(c, n, p, a.as_ref())?);
                            }
                            self.apply(c, n, &args)?
                        }
                    }
                }
            }
        };
        self.memo.insert(k, v);
        Ok(v)
    }

    /// A node of the core's body (region arguments from `region`).
    fn body(&mut self, c: &mut Ctx, n: u32, _r: u32) -> R<Val> {
        let k = (n, 0, "body".to_string());
        if let Some(v) = self.memo.get(&k) {
            return Ok(*v);
        }
        let KNode::Member { node, .. } = &self.g.nodes[n as usize] else {
            return Err(format!("node {n} is not in a body"));
        };
        let v = match &node.label.kind {
            Kind::RegionArg { index, .. } => *self
                .region
                .get(*index as usize)
                .ok_or_else(|| format!("region argument {index}"))?,
            _ => {
                let mut args = Vec::new();
                for p in 0..self.ports[n as usize].len() {
                    args.push(self.port(c, n, p, None)?);
                }
                self.apply(c, n, &args)?
            }
        };
        self.memo.insert(k, v);
        Ok(v)
    }

    /// The body's results for region arguments `region`.
    fn run_body(&mut self, c: &mut Ctx, core: u32, region: Vec<Val>, n: usize) -> R<Vec<Val>> {
        self.memo.retain(|k, _| k.2 != "body");
        self.region = region;
        let ret = self
            .g
            .out
            .iter()
            .enumerate()
            .find(|(_, o)| {
                o.iter().any(|(v, ends)| {
                    *v == core && ends.iter().any(|e| matches!(e.port, PortLabel::Yield { .. }))
                })
            })
            .map(|(u, _)| u as u32)
            .ok_or("no body terminator")?;
        let mut out = Vec::new();
        for p in 0..n {
            out.push(self.port(c, ret, p, None)?);
        }
        Ok(out)
    }

    /// The value of core port `p` at the core's operand access.
    fn core_port(&mut self, c: &mut Ctx, core: u32, p: usize) -> R<Val> {
        let inst = self
            .skel
            .accesses
            .iter()
            .find(|x| x.node == core)
            .cloned()
            .ok_or("core without an access")?;
        let a = inst.operands.get(p).cloned().flatten();
        match (&a, self.ports[core as usize].get(p).cloned().flatten()) {
            (_, Some(Port::Splat(col))) => self.splat(c, &col),
            (Some(a), Some(Port::Edge(pn, pr))) => self.value(c, pn, pr, a),
            _ => Err(format!("core port {p}: no operand or access")),
        }
    }
}

/// The core op's spec kind (independent of the extraction's table).
fn core_op(g: &Graph<KNode>, core: u32) -> R<(String, usize)> {
    let KNode::Member { node, .. } = &g.nodes[core as usize] else {
        return Err("the core is not a member".into());
    };
    let Kind::Op { name, target } = &node.label.kind else {
        return Err("the core is not an op".into());
    };
    let k = match (name.as_str(), target.as_deref()) {
        ("stablehlo.dot_general" | "stablehlo.convolution" | "stablehlo.dynamic_conv", _) => "dot",
        ("stablehlo.reduce" | "stablehlo.reduce_window", _)
        | ("stablehlo.custom_call", Some("stablehlo.dynamic_reduce_window")) => "reduce",
        ("stablehlo.scatter", _) => "scatter",
        (n, _) => return Err(format!("core `{n}`: no semantics")),
    };
    Ok((k.to_string(), node.label.results.len()))
}

/// The values the datapath must reproduce for tuple `t`, by name: every kernel output (`out<j>`); for a
/// template, every core input (by its port name); for an accumulating core, every accumulator's next
/// value (`next<i>`) and init value (`init<i>`).
pub fn expected(
    c: &mut Ctx,
    class: &ClassFile,
    skel: &Skeleton,
    routed: &Routed,
    g: &Graph<KNode>,
    t: &Tuple,
) -> R<BTreeMap<String, Val>> {
    let mut s = Sem {
        class,
        skel,
        g,
        t,
        ports: port_table(g),
        memo: HashMap::new(),
        region: Vec::new(),
        results: None,
    };
    let mut out = BTreeMap::new();
    if let Some(core) = skel.core_node {
        let (kind, n) = core_op(g, core)?;
        if routed.route == Route::Template {
            for o in skel
                .operands
                .iter()
                .filter(|o| o.core && o.direction == Direction::In)
            {
                if !crate::iteration_space::algo::skeleton::is_core_input(&o.name) {
                    continue;
                }
                let p = s.ports[core as usize]
                    .iter()
                    .position(|x| matches!(x, Some(Port::Edge(a, b)) if (*a, *b) == (o.node, o.res)))
                    .ok_or("core input without a port")?;
                out.insert(o.name.clone(), s.core_port(c, core, p)?);
            }
        } else {
            match kind.as_str() {
                "dot" => {
                    let rt = s.elem_ty(core, 0)?;
                    let lhs = s.core_port(c, core, 0)?;
                    let rhs = s.core_port(c, core, 1)?;
                    let l = ops::convert(c, lhs, rt).map_err(|e| e.0)?;
                    let r = ops::convert(c, rhs, rt).map_err(|e| e.0)?;
                    let none = Attrs::default();
                    let m = ops::apply(c, "multiply", &none, &[l, r], rt).map_err(|e| e.0)?;
                    let acc = ops::input(c, "acc0", rt).map_err(|e| e.0)?;
                    out.insert(
                        "next0".into(),
                        ops::apply(c, "add", &none, &[acc, m], rt).map_err(|e| e.0)?,
                    );
                    out.insert("init0".into(), ops::zero(c, rt).map_err(|e| e.0)?);
                }
                _ => {
                    // reduce: (inputs, inits); scatter: (inputs, indices, updates).
                    let (xs, inits): (Vec<usize>, Vec<usize>) = if kind == "reduce" {
                        ((0..n).collect(), (n..2 * n).collect())
                    } else {
                        ((n + 1..2 * n + 1).collect(), (0..n).collect())
                    };
                    let mut xv = Vec::new();
                    for p in xs {
                        xv.push(s.core_port(c, core, p)?);
                    }
                    let mut iv = Vec::new();
                    for p in inits {
                        iv.push(s.core_port(c, core, p)?);
                    }
                    if kind == "scatter" && skel.class == SkelClass::Map {
                        let rs = s.run_body(c, core, iv.into_iter().chain(xv).collect(), n)?;
                        s.results = Some(rs);
                        s.memo.clear();
                    } else {
                        let mut acc = Vec::new();
                        for i in 0..n {
                            let t = s.elem_ty(core, i as u32)?;
                            acc.push(ops::input(c, &format!("acc{i}"), t).map_err(|e| e.0)?);
                        }
                        let next = s.run_body(c, core, acc.into_iter().chain(xv).collect(), n)?;
                        for (i, (nv, init)) in next.into_iter().zip(iv).enumerate() {
                            out.insert(format!("next{i}"), nv);
                            out.insert(format!("init{i}"), init);
                        }
                    }
                }
            }
        }
    }
    for o in skel
        .operands
        .iter()
        .filter(|o| o.direction == Direction::Out && !o.core)
    {
        out.insert(o.name.clone(), s.value(c, o.node, o.res, &o.access)?);
    }
    Ok(out)
}

/// The combiner as a function of two accumulator tuples (for the order queries), from the kernel graph.
pub fn combiner(
    c: &mut Ctx,
    class: &ClassFile,
    skel: &Skeleton,
    g: &Graph<KNode>,
    t: &Tuple,
    a: &[Val],
    b: &[Val],
) -> R<Vec<Val>> {
    let core = skel.core_node.ok_or("no core")?;
    let (kind, n) = core_op(g, core)?;
    let mut s = Sem {
        class,
        skel,
        g,
        t,
        ports: port_table(g),
        memo: HashMap::new(),
        region: Vec::new(),
        results: None,
    };
    if kind == "dot" {
        let rt = s.elem_ty(core, 0)?;
        return Ok(vec![
            ops::apply(c, "add", &Attrs::default(), &[a[0], b[0]], rt).map_err(|e| e.0)?,
        ]);
    }
    s.run_body(c, core, a.iter().chain(b).copied().collect(), n)
}

/// The accumulator types of a class's core under a tuple.
pub fn acc_types(class: &ClassFile, skel: &Skeleton, g: &Graph<KNode>, t: &Tuple) -> R<Vec<Ty>> {
    let core = skel.core_node.ok_or("no core")?;
    let (_, n) = core_op(g, core)?;
    let s = Sem {
        class,
        skel,
        g,
        t,
        ports: Vec::new(),
        memo: HashMap::new(),
        region: Vec::new(),
        results: None,
    };
    (0..n as u32).map(|i| s.elem_ty(core, i)).collect()
}
