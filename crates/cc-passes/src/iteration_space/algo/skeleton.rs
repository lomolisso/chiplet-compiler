//! Skeletons (docs/passes/iteration_space.md §4.1): the iteration domain of a class's representative
//! kernel and the affine access of every value in it, composed through the op table's iteration hooks
//! from the core (or, for a map kernel, from its domain) outward. A prologue member reads through the
//! core operand it feeds, an epilogue member sits on the result's domain, and a broadcast folds into a
//! projection (design §4.5, K4).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use cc_ir::absgraph::Graph;
use cc_ir::ops::iter::{self as it, IterIn, OpIteration, Tag};

use crate::components::algo::graph::{AAttr, ANode};
use crate::iteration_space::json::{
    Access, Dim, DimOrder, Direction, Extent, Factor, NodeAccess, Operand, Row, SKELETON_SCHEMA, SkelClass,
    Skeleton, Stages, Term,
};
use crate::kernel_classes::algo::local::{expanding_axes, op_of, operands, results};
use crate::kernel_classes::json::{ClassFile, ClassKind, Group as GroupKind, LocalClasses};
use crate::kernels::json::{KNode, Role};

/// A factor after binding: a literal or a symbol.
enum Bound {
    Lit(i64),
    Sym(Factor),
}

fn label(n: &KNode) -> Option<&ANode> {
    match n {
        KNode::Member { node, .. } => Some(&node.label),
        _ => None,
    }
}

fn role(n: &KNode) -> Option<Role> {
    match n {
        KNode::Member { role, .. } => Some(*role),
        _ => None,
    }
}

fn bind_factor(node: &ANode, n: u32, f: &it::Factor) -> Result<Bound, String> {
    match f {
        it::Factor::Attr { name, index } => match node.attrs.get(name) {
            Some(AAttr::Sized { entries, .. }) => match entries.get(*index as usize) {
                Some(Some(v)) => Ok(Bound::Lit(*v)),
                Some(None) => Ok(Bound::Sym(Factor::Column {
                    column: format!("n{n}.{name}"),
                    index: *index,
                })),
                None => Err(format!("node {n}: `{name}` has no entry {index}")),
            },
            Some(AAttr::Lit(a)) => a
                .as_ints()
                .and_then(|v| v.get(*index as usize).copied())
                .map(Bound::Lit)
                .ok_or_else(|| format!("node {n}: `{name}` entry {index} is not an integer")),
            _ => Err(format!("node {n}: `{name}` is not a size attribute")),
        },
        it::Factor::Runtime { port, index } => Ok(Bound::Sym(Factor::Runtime {
            node: n,
            port: *port,
            index: *index,
        })),
        it::Factor::Extent { dim } => Ok(Bound::Sym(Factor::Extent { dim: *dim })),
    }
}

/// Combine like terms, drop zero terms, sort: the canonical form every comparison and proof uses.
pub fn normal(mut r: Row) -> Row {
    let mut acc: BTreeMap<(Option<u32>, Vec<Factor>), i64> = BTreeMap::new();
    for mut t in r.terms.drain(..) {
        t.factors.sort();
        *acc.entry((t.dim, t.factors)).or_default() += t.k;
    }
    r.terms = acc
        .into_iter()
        .filter(|(_, k)| *k != 0)
        .map(|((dim, factors), k)| Term { k, factors, dim })
        .collect();
    r
}

fn bind_row(node: &ANode, n: u32, r: &it::Row) -> Result<Row, String> {
    let mut terms = Vec::new();
    for t in &r.terms {
        let mut k = t.k;
        let mut factors = Vec::new();
        for f in &t.factors {
            match bind_factor(node, n, f)? {
                Bound::Lit(v) => k *= v,
                Bound::Sym(s) => factors.push(s),
            }
        }
        terms.push(Term {
            k,
            factors,
            dim: t.dim,
        });
    }
    let divisor = match &r.divisor {
        None => None,
        Some(f) => match bind_factor(node, n, f)? {
            Bound::Lit(1) => None,
            Bound::Lit(v) => return Err(format!("node {n}: literal divisor {v}")),
            Bound::Sym(s) => Some(s),
        },
    };
    Ok(normal(Row {
        terms,
        divisor,
        padded: r.padded,
        data: r.data,
    }))
}

/// `op_row` over the op's own dims, with op dim `j` equal to the domain row `sub[j]`.
fn compose(op_row: &Row, sub: &[Row]) -> Result<Row, String> {
    // A plain index `i_j` is the domain row itself (with its divisor and guards).
    if let [t] = op_row.terms.as_slice()
        && t.k == 1
        && t.factors.is_empty()
        && op_row.divisor.is_none()
        && let Some(j) = t.dim
    {
        let mut r = sub
            .get(j as usize)
            .ok_or_else(|| format!("op dim {j} has no row"))?
            .clone();
        r.padded |= op_row.padded;
        r.data |= op_row.data;
        return Ok(r);
    }
    let mut terms = Vec::new();
    let (mut padded, mut data) = (op_row.padded, op_row.data);
    for t in &op_row.terms {
        match t.dim {
            None => terms.push(t.clone()),
            Some(j) => {
                let s = sub
                    .get(j as usize)
                    .ok_or_else(|| format!("op dim {j} has no row"))?;
                if s.divisor.is_some() {
                    return Err("a divided index feeds another access".into());
                }
                padded |= s.padded;
                data |= s.data;
                for u in &s.terms {
                    let mut factors = t.factors.clone();
                    factors.extend(u.factors.iter().cloned());
                    terms.push(Term {
                        k: t.k * u.k,
                        factors,
                        dim: u.dim,
                    });
                }
            }
        }
    }
    Ok(normal(Row {
        terms,
        divisor: op_row.divisor.clone(),
        padded,
        data,
    }))
}

/// The hook's view of kernel node `n`.
fn iterate(
    class: &ClassFile,
    g: &Graph<KNode>,
    ops: &[Vec<Option<(u32, u32)>>],
    n: u32,
) -> Result<OpIteration, String> {
    let x = &g.nodes[n as usize];
    let (name, target) = op_of(x).ok_or_else(|| format!("node {n} is not an op"))?;
    let spec =
        cc_ir::lookup(name, target).ok_or_else(|| format!("node {n}: `{name}` is not in the op table"))?;
    let hook = spec
        .iteration
        .ok_or_else(|| format!("node {n}: `{}` has no iteration semantics", spec.key()))?;
    let node = label(x).ok_or("not a member")?;
    let ranks: Vec<Option<usize>> = ops[n as usize]
        .iter()
        .map(|p| {
            p.and_then(|(pn, pr)| {
                results(&g.nodes[pn as usize])
                    .get(pr as usize)
                    .map(|t| t.dims.len())
            })
        })
        .collect();
    let res: Vec<usize> = node.results.iter().map(|t| t.dims.len()).collect();
    let lit = |k: &str| match node.attrs.get(k) {
        Some(AAttr::Lit(a)) => Some(a),
        _ => None,
    };
    let has = |k: &str| node.attrs.contains_key(k);
    let expanded = expanding_axes(g, ops, n);
    let grouped = |k: &str| {
        class
            .representative
            .groups
            .get(&n)
            .and_then(|m| m.get(k))
            .is_some_and(|g| *g != GroupKind::One)
    };
    hook(&IterIn {
        operands: &ranks,
        results: &res,
        lit: &lit,
        has: &has,
        expanded: &expanded,
        feature_grouped: grouped("feature_group_count"),
        batch_grouped: grouped("batch_group_count"),
    })
    .map_err(|e| format!("node {n} ({name}): {e}"))
}

fn bind_extent(
    local: &LocalClasses,
    node: &ANode,
    n: u32,
    ops: &[Option<(u32, u32)>],
    e: &it::Extent,
) -> Result<Extent, String> {
    let class = |pn: u32, pr: u32, a: u32| -> Result<u32, String> {
        local
            .dims
            .get(pn as usize)
            .and_then(|r| r.get(pr as usize))
            .and_then(|ax| ax.get(a as usize))
            .copied()
            .ok_or_else(|| format!("node {pn}: no axis {a} of result {pr}"))
    };
    let factor = |f: &it::Factor| -> Result<Bound, String> { bind_factor(node, n, f) };
    match e {
        it::Extent::Result { res, axis } => Ok(Extent::Class {
            class: class(n, *res, *axis)?,
        }),
        it::Extent::Operand { port, axis } => {
            let (pn, pr) = ops
                .get(*port as usize)
                .copied()
                .flatten()
                .ok_or_else(|| format!("node {n}: no operand {port}"))?;
            Ok(Extent::Class {
                class: class(pn, pr, *axis)?,
            })
        }
        it::Extent::Factor(f) => Ok(match factor(f)? {
            Bound::Lit(v) => Extent::Const { value: v },
            Bound::Sym(Factor::Column { column, index }) => Extent::Column { column, index },
            Bound::Sym(Factor::Runtime { node, port, index }) => Extent::Runtime { node, port, index },
            Bound::Sym(Factor::Extent { .. }) => return Err("an extent of an extent".into()),
        }),
        it::Extent::Quot { res, axis, by } => {
            let c = class(n, *res, *axis)?;
            Ok(match factor(by)? {
                Bound::Lit(1) => Extent::Class { class: c },
                Bound::Lit(v) => return Err(format!("node {n}: literal group count {v}")),
                Bound::Sym(by) => Extent::Quot { class: c, by },
            })
        }
    }
}

/// Dims not indexed by an access.
pub fn reuse(dims: usize, access: &Access) -> Vec<u32> {
    let used: BTreeSet<u32> = access
        .iter()
        .flat_map(|r| r.terms.iter().filter_map(|t| t.dim))
        .collect();
    (0..dims as u32).filter(|d| !used.contains(d)).collect()
}

/// Names of a core's operand ports and results.
fn core_names(kind: ClassKind, name: &str, nops: usize, nres: usize) -> (Vec<String>, Vec<String>) {
    let res: Vec<String> = if nres == 1 && kind == ClassKind::Contraction {
        vec!["result".into()]
    } else {
        (0..nres).map(|i| format!("result{i}")).collect()
    };
    let ops = match name {
        "stablehlo.dot_general" | "stablehlo.convolution" | "stablehlo.dynamic_conv" => (0..nops)
            .map(|p| {
                ["lhs", "rhs", "padding"]
                    .get(p)
                    .map_or(format!("port{p}"), |s| s.to_string())
            })
            .collect(),
        "stablehlo.scatter" => (0..nops)
            .map(|p| {
                if p < nres {
                    format!("input{p}")
                } else if p == nres {
                    "indices".into()
                } else {
                    format!("updates{}", p - nres - 1)
                }
            })
            .collect(),
        _ => (0..nops)
            .map(|p| {
                if p < nres {
                    format!("input{p}")
                } else {
                    format!("init{}", p - nres)
                }
            })
            .collect(),
    };
    (ops, res)
}

/// Whether a core operand is an input of the computation (not an init value or an index).
pub fn is_core_input(name: &str) -> bool {
    name == "lhs" || name == "rhs" || name.starts_with("input") || name.starts_with("updates")
}

/// The skeleton of a class from its representative kernel graph (`proof` is filled by the caller).
pub fn build(class: &ClassFile, g: &Graph<KNode>) -> Result<Skeleton, String> {
    let local = &class.local;
    let ops = operands(g);
    let core = g
        .nodes
        .iter()
        .position(|x| role(x) == Some(Role::Core))
        .map(|c| c as u32);
    // A value may be read with several accesses: a broadcast absorbed into the kernel can feed the
    // prologues of both core operands. Every (member, access) pair is one `NodeAccess`.
    let mut val: HashMap<(u32, u32), Vec<Access>> = HashMap::new();
    let mut uses: HashMap<(u32, u32), Vec<Access>> = HashMap::new();
    let mut accesses: Vec<NodeAccess> = Vec::new();
    let mut dims: Vec<Dim> = Vec::new();
    let mut omega = None;

    fn add(m: &mut HashMap<(u32, u32), Vec<Access>>, k: (u32, u32), a: &Access) {
        let v = m.entry(k).or_default();
        if !v.contains(a) {
            v.push(a.clone());
        }
    }
    let record = |n: u32,
                  results: Vec<Access>,
                  operands: Vec<Option<Access>>,
                  val: &mut HashMap<(u32, u32), Vec<Access>>,
                  uses: &mut HashMap<(u32, u32), Vec<Access>>,
                  accesses: &mut Vec<NodeAccess>| {
        for (r, a) in results.iter().enumerate() {
            add(val, (n, r as u32), a);
        }
        for (p, a) in operands.iter().enumerate() {
            let Some(a) = a else { continue };
            add(uses, (n, p as u32), a);
            if let Some(Some((pn, pr))) = ops[n as usize].get(p).copied() {
                add(val, (pn, pr), a);
            }
        }
        accesses.push(NodeAccess {
            node: n,
            results,
            operands,
        });
    };

    match core {
        Some(c) => {
            let x = &g.nodes[c as usize];
            let node = label(x).ok_or("core is not a member")?;
            let oi = iterate(class, g, &ops, c)?;
            for d in &oi.dims {
                dims.push(Dim {
                    tag: d.tag,
                    extent: bind_extent(local, node, c, &ops[c as usize], &d.extent)?,
                    role: d.role.clone(),
                });
            }
            if let KNode::Member { node: nn, .. } = x
                && let Some(o) = &nn.omega
            {
                let mut od = Vec::new();
                for &a in &o.axes {
                    let d = oi
                        .dims
                        .iter()
                        .position(|d| d.omega == Some(a))
                        .ok_or_else(|| format!("ω axis {a} is no iteration dim"))?;
                    od.push(d as u32);
                }
                omega = Some(DimOrder {
                    order: o.order.clone(),
                    dims: od,
                });
            }
            let results = oi
                .results
                .iter()
                .map(|a| {
                    a.iter()
                        .map(|r| bind_row(node, c, r))
                        .collect::<Result<Access, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let operands = oi
                .operands
                .iter()
                .enumerate()
                .map(|(p, a)| match (a, ops[c as usize].get(p).copied().flatten()) {
                    (Some(a), Some(_)) => a
                        .iter()
                        .map(|r| bind_row(node, c, r))
                        .collect::<Result<Access, _>>()
                        .map(Some),
                    _ => Ok(None),
                })
                .collect::<Result<Vec<_>, _>>()?;
            record(c, results, operands, &mut val, &mut uses, &mut accesses);
        }
        None => {
            // A map kernel: the domain is its widest element-wise member's result (K4).
            let m0 = g
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, x)| role(x) == Some(Role::Map))
                .max_by_key(|(i, x)| {
                    (
                        results(x).first().map_or(0, |t| t.dims.len()),
                        std::cmp::Reverse(*i),
                    )
                })
                .map(|(i, _)| i as u32)
                .ok_or("a map kernel without members")?;
            let r = results(&g.nodes[m0 as usize]).first().map_or(0, |t| t.dims.len());
            for a in 0..r {
                dims.push(Dim {
                    tag: Tag::Par,
                    extent: Extent::Class {
                        class: local.dims[m0 as usize][0][a],
                    },
                    role: "elem".into(),
                });
            }
            val.insert((m0, 0), vec![(0..r as u32).map(|d| normal(row_dim(d))).collect()]);
        }
    }

    // Propagate to a fixpoint: backward through operands, forward through element-wise results.
    let members: Vec<u32> = (0..g.node_count() as u32)
        .filter(|&n| matches!(role(&g.nodes[n as usize]), Some(r) if r != Role::Body && r != Role::Core))
        .collect();
    let mut hooks: HashMap<u32, OpIteration> = HashMap::new();
    for &n in &members {
        hooks.insert(n, iterate(class, g, &ops, n)?);
    }
    let mut done: HashMap<u32, Vec<Access>> = HashMap::new();
    loop {
        let mut progress = false;
        for &n in &members {
            let oi = &hooks[&n];
            let node = label(&g.nodes[n as usize]).ok_or("not a member")?;
            let rank = node.results.first().map_or(0, |t| t.dims.len());
            let mut rs: Vec<Access> = val.get(&(n, 0)).cloned().unwrap_or_default();
            if rs.is_empty() && spec_elementwise(&g.nodes[n as usize]) {
                for (pn, pr) in ops[n as usize].iter().flatten() {
                    let same = results(&g.nodes[*pn as usize])
                        .get(*pr as usize)
                        .is_some_and(|t| t.dims.len() == rank);
                    if same {
                        for a in val.get(&(*pn, *pr)).into_iter().flatten() {
                            if !rs.contains(a) {
                                rs.push(a.clone());
                            }
                        }
                    }
                }
            }
            for r in rs {
                if done.get(&n).is_some_and(|d| d.contains(&r)) {
                    continue;
                }
                let operands = oi
                    .operands
                    .iter()
                    .enumerate()
                    .map(|(p, a)| match (a, ops[n as usize].get(p).copied().flatten()) {
                        (Some(a), Some(_)) => a
                            .iter()
                            .map(|row| bind_row(node, n, row).and_then(|row| compose(&row, &r)))
                            .collect::<Result<Access, _>>()
                            .map(Some),
                        _ => Ok(None),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                done.entry(n).or_default().push(r.clone());
                record(n, vec![r], operands, &mut val, &mut uses, &mut accesses);
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }
    if let Some(n) = members.iter().find(|n| !done.contains_key(n)) {
        return Err(format!("member {n} has no access from the domain"));
    }

    // Operands: the core's, then the kernel interface.
    let mut operands_out: Vec<Operand> = Vec::new();
    let classes_of = |n: u32, r: u32| -> (Vec<u32>, u32) {
        (
            local
                .dims
                .get(n as usize)
                .and_then(|x| x.get(r as usize))
                .cloned()
                .unwrap_or_default(),
            local
                .elems
                .get(n as usize)
                .and_then(|x| x.get(r as usize))
                .copied()
                .unwrap_or_default(),
        )
    };
    let nd = dims.len();
    if let Some(c) = core {
        let x = &g.nodes[c as usize];
        let (name, _) = op_of(x).ok_or("core is not an op")?;
        let nres = results(x).len();
        let (pn_names, rn_names) = core_names(class.kind, name, ops[c as usize].len(), nres);
        for (p, prod) in ops[c as usize].iter().enumerate() {
            let (Some((pn, pr)), Some(a)) = (prod, uses.get(&(c, p as u32)).and_then(|v| v.first())) else {
                continue;
            };
            let (axes, elem) = classes_of(*pn, *pr);
            operands_out.push(Operand {
                name: pn_names[p].clone(),
                direction: Direction::In,
                core: true,
                node: *pn,
                res: *pr,
                axes,
                elem,
                reuse: reuse(nd, a),
                access: a.clone(),
            });
        }
        for (r, rn) in rn_names.iter().enumerate() {
            let a = val
                .get(&(c, r as u32))
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or_default();
            let (axes, elem) = classes_of(c, r as u32);
            operands_out.push(Operand {
                name: rn.clone(),
                direction: Direction::Out,
                core: true,
                node: c,
                res: r as u32,
                axes,
                elem,
                reuse: reuse(nd, &a),
                access: a,
            });
        }
    }
    for (i, x) in g.nodes.iter().enumerate() {
        let i = i as u32;
        match x {
            KNode::Input { .. } => {
                let (axes, elem) = classes_of(i, 0);
                let mut seen: Vec<Access> = Vec::new();
                for (v, ends) in &g.out[i as usize] {
                    if role(&g.nodes[*v as usize]) == Some(Role::Body) {
                        continue;
                    }
                    for e in ends {
                        let p = match e.port {
                            cc_ir::absgraph::PortLabel::Port(p) => Some(p),
                            cc_ir::absgraph::PortLabel::Commutative => ops[*v as usize]
                                .iter()
                                .position(|q| *q == Some((i, e.res)))
                                .map(|p| p as u32),
                            _ => None,
                        };
                        for a in p.and_then(|p| uses.get(&(*v, p))).into_iter().flatten() {
                            if !seen.contains(a) {
                                seen.push(a.clone());
                            }
                        }
                    }
                }
                for (u, a) in seen.into_iter().enumerate() {
                    operands_out.push(Operand {
                        name: if u == 0 {
                            format!("in{i}")
                        } else {
                            format!("in{i}.{u}")
                        },
                        direction: Direction::In,
                        core: false,
                        node: i,
                        res: 0,
                        axes: axes.clone(),
                        elem,
                        reuse: reuse(nd, &a),
                        access: a,
                    });
                }
            }
            KNode::Output { .. } => {
                // The output's producer is its one in-edge (port 0), in kernel numbering.
                let (node, res) = ops[i as usize]
                    .first()
                    .copied()
                    .flatten()
                    .ok_or_else(|| format!("output {i} has no producer"))?;
                let a = val
                    .get(&(node, res))
                    .and_then(|v| v.first())
                    .cloned()
                    .ok_or_else(|| format!("output {i}: its value has no access"))?;
                let (axes, elem) = classes_of(node, res);
                operands_out.push(Operand {
                    name: format!("out{i}"),
                    direction: Direction::Out,
                    core: false,
                    node,
                    res,
                    axes,
                    elem,
                    reuse: reuse(nd, &a),
                    access: a,
                });
            }
            KNode::Member { .. } => {}
        }
    }

    let affine = !accesses
        .iter()
        .flat_map(|a| a.results.iter().chain(a.operands.iter().flatten()))
        .flatten()
        .any(|r| r.data);
    let has_red = dims.iter().any(|d| d.tag == Tag::Red);
    let reused_input = operands_out.iter().any(|o| {
        o.core
            && o.direction == Direction::In
            && is_core_input(&o.name)
            && o.reuse.iter().any(|&d| dims[d as usize].tag == Tag::Par)
    });
    let skel_class = if !has_red {
        SkelClass::Map
    } else if reused_input && affine {
        SkelClass::Contraction
    } else {
        SkelClass::Reduce
    };

    // Stages: prologue members per core input port, then the epilogue.
    let mut stages = Stages::default();
    if let Some(c) = core {
        let mut in_prologue: BTreeSet<u32> = BTreeSet::new();
        for (p, prod) in ops[c as usize].iter().enumerate() {
            let Some((pn, _)) = prod else { continue };
            let mut stack = vec![*pn];
            let mut found = BTreeSet::new();
            while let Some(v) = stack.pop() {
                if !matches!(role(&g.nodes[v as usize]), Some(Role::Prologue | Role::Broadcast))
                    || !found.insert(v)
                {
                    continue;
                }
                stack.extend(ops[v as usize].iter().flatten().map(|(q, _)| *q));
            }
            if !found.is_empty() {
                in_prologue.extend(found.iter().copied());
                stages.prologue.insert(p as u32, found.into_iter().collect());
            }
        }
        stages.epilogue = members
            .iter()
            .copied()
            .filter(|&n| {
                let r = role(&g.nodes[n as usize]);
                r == Some(Role::Epilogue) || (r == Some(Role::Broadcast) && !in_prologue.contains(&n))
            })
            .collect();
    }

    Ok(Skeleton {
        schema: SKELETON_SCHEMA.into(),
        id: class.id.clone(),
        class: skel_class,
        kind: serde_json::to_value(class.kind)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default(),
        core: class.core.clone(),
        core_node: core,
        affine,
        dims,
        omega,
        operands: operands_out,
        accesses,
        stages,
        proof: String::new(),
    })
}

fn row_dim(d: u32) -> Row {
    Row {
        terms: vec![Term {
            k: 1,
            factors: Vec::new(),
            dim: Some(d),
        }],
        divisor: None,
        padded: false,
        data: false,
    }
}

fn spec_elementwise(x: &KNode) -> bool {
    op_of(x)
        .and_then(|(n, t)| cc_ir::lookup(n, t))
        .is_some_and(|s| s.elementwise)
}
