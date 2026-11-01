//! Abstract graphs of instances (docs/passes/components.md §4).
//!
//! An instance graph has one node per decomposition argument (`input k`, dimension arguments excluded),
//! per compute op of the body (nested regions included, size computation excluded), per region argument,
//! per dimension value entering compute (`size_in`) and, for an anon region, per external input and per
//! exported value. Edges carry (result index, port); shape operands are not edges. In the abstract form
//! every dim except the literal 1 is a class variable, every element type is (family, class), the
//! integers of size attributes other than 0 and 1 are parameters and constant values are parameters.

use std::collections::BTreeMap;

use cc_ir::absgraph::{EdgeEnd, Graph, PortLabel};
use cc_ir::{
    Attr, Def, Dim, Elem, FuncId, Module, Op, OpId, Owner, RegionId, Scalar, SymExpr, Type, ValueId,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::import::algo::size_tag::is_shape_port;
use crate::import::algo::size_value::Sv;

/// Where a node of an instance graph comes from in the module (a certificate entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NodeRef {
    /// An op.
    Op(u32),
    /// A value used as an input: a decomposition argument, a region argument, or (anon regions) a value
    /// defined outside the region.
    Value(u32),
    /// The size value at operand `port` of op `op` (a `dim_as_data` port).
    SizeIn { op: u32, port: u32 },
    /// An exported value of an anon region.
    Out(u32),
}

/// What a node is.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Argument `k` (dimension arguments not counted) of a decomposition; `None` for an external value
    /// of an anon region (unordered).
    Input {
        index: Option<u32>,
    },
    /// A value of an anon region used outside it.
    Output,
    /// Argument `index` of region `region` of the owner op.
    RegionArg {
        region: u32,
        index: u32,
    },
    /// A dimension value entering compute.
    SizeIn,
    Op {
        name: String,
        target: Option<String>,
    },
}

/// An element-type family (design §6.1): the part of a type in a component's identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Bool,
    Sint,
    Uint,
    Float,
    Complex(Box<Family>),
}

impl Family {
    pub fn of(e: &Elem) -> Family {
        match e {
            Elem::Bool => Family::Bool,
            Elem::Int { signed: true, .. } => Family::Sint,
            Elem::Int { signed: false, .. } => Family::Uint,
            Elem::Float { .. } => Family::Float,
            Elem::Complex(e) => Family::Complex(Box::new(Family::of(e))),
        }
    }
}

/// An abstract dim: the literal 1, or a class variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ADim {
    One,
    Var(u32),
}

/// An abstract tensor type: family, element class, dims.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct AType {
    pub family: Family,
    pub elem: u32,
    pub dims: Vec<ADim>,
}

/// How a size attribute is spelled, so that it can be rebuilt from its entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SizedForm {
    Ints,
    Int { ty: String },
    Dense { ty: Type, splat: bool },
}

/// An abstract attribute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AAttr {
    /// Kept as is (axis numbers, dimension numbers, enums, …).
    Lit(Attr),
    /// A size attribute: `Some(v)` entries are the literals 0 and 1, `None` entries are parameters.
    Sized {
        form: SizedForm,
        entries: Vec<Option<i64>>,
    },
    /// A constant's value: a parameter (`splat` stays).
    Value { splat: bool },
}

/// A node of an abstract graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ANode {
    pub kind: Kind,
    pub attrs: BTreeMap<String, AAttr>,
    pub results: Vec<AType>,
    /// The child component of a composite call (filled by deduplication).
    pub child: Option<String>,
}

/// One instance parameter of a node, in attribute order (then the `size_in` value).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Param {
    Int(i64),
    Value(Attr),
    Size(SizeValue),
}

/// A size value: per element, its guarded cases (`guard` facts `f >= 0` → `value`), as printed
/// expressions; `None` when nothing is known.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SizeValue {
    pub elems: Option<Vec<Vec<Case>>>,
}

/// One case of a size element: its guard facts (`f >= 0`) and its value.
pub type Case = (Vec<String>, String);

impl SizeValue {
    pub fn of(sv: &Sv) -> SizeValue {
        SizeValue {
            elems: sv.as_ref().map(|els| {
                els.iter()
                    .map(|g| {
                        g.iter()
                            .map(|a| (a.guard.iter().map(SymExpr::to_string).collect(), a.v.to_string()))
                            .collect()
                    })
                    .collect()
            }),
        }
    }
}

/// The node label that enters matching: everything structural, class numbers erased (they are a function
/// of the labelled graph), the child left out (compared separately).
#[derive(Serialize)]
struct Key<'a> {
    kind: &'a Kind,
    attrs: &'a BTreeMap<String, AAttr>,
    results: Vec<(&'a Family, Vec<bool>)>,
}

impl ANode {
    pub fn key(&self) -> Vec<u8> {
        let k = Key {
            kind: &self.kind,
            attrs: &self.attrs,
            results: self
                .results
                .iter()
                .map(|t| (&t.family, t.dims.iter().map(|d| *d == ADim::One).collect()))
                .collect(),
        };
        postcard::to_stdvec(&k).unwrap_or_default()
    }

    /// Equal matching keys (`key` bytes equal), compared on the structure.
    pub fn same_key(&self, o: &ANode) -> bool {
        self.kind == o.kind
            && self.attrs == o.attrs
            && self.results.len() == o.results.len()
            && self.results.iter().zip(&o.results).all(|(a, b)| {
                a.family == b.family
                    && a.dims.len() == b.dims.len()
                    && a.dims
                        .iter()
                        .zip(&b.dims)
                        .all(|(x, y)| (*x == ADim::One) == (*y == ADim::One))
            })
    }

    pub fn is_call(&self) -> bool {
        matches!(&self.kind, Kind::Op { name, .. } if name == "stablehlo.composite")
    }
}

/// Marks a parameter entry of a size attribute in `rule_attrs` (never 0 or 1).
pub const PARAM_MARK: i64 = i64::MIN;

/// The attributes a dims rule may read: `Lit` attributes, and size attributes with every parameter entry
/// replaced by `PARAM_MARK` (a rule sees their length and which entries are 0 or 1, never a value).
pub fn rule_attrs(n: &ANode) -> cc_ir::Attrs {
    n.attrs
        .iter()
        .filter_map(|(k, a)| match a {
            AAttr::Lit(x) => Some((k.clone(), x.clone())),
            AAttr::Sized { form, entries } => {
                let v: Vec<i64> = entries.iter().map(|e| e.unwrap_or(PARAM_MARK)).collect();
                Some((k.clone(), rebuild(form, &v)))
            }
            AAttr::Value { .. } => None,
        })
        .collect()
}

/// A size attribute from its form and entries.
pub fn rebuild(form: &SizedForm, v: &[i64]) -> Attr {
    match form {
        SizedForm::Ints => Attr::Ints(v.to_vec()),
        SizedForm::Int { ty } => Attr::Int {
            value: i128::from(v.first().copied().unwrap_or_default()),
            ty: ty.clone(),
        },
        SizedForm::Dense { ty, splat } => Attr::Dense {
            ty: ty.clone(),
            splat: *splat,
            values: Some(v.iter().map(|&x| Scalar::Int(i128::from(x))).collect()),
            digest: None,
        },
    }
}

/// Split a size attribute into its form and entries, or `None` when it has no integer form.
pub fn split_sized(a: &Attr) -> Option<(SizedForm, Vec<i64>)> {
    match a {
        Attr::Ints(v) => Some((SizedForm::Ints, v.clone())),
        Attr::Int { value, ty } => Some((
            SizedForm::Int { ty: ty.clone() },
            vec![i64::try_from(*value).ok()?],
        )),
        Attr::Dense {
            ty,
            splat,
            values: Some(vals),
            digest: None,
        } => {
            let v: Option<Vec<i64>> = vals
                .iter()
                .map(|s| s.as_int().and_then(|x| i64::try_from(x).ok()))
                .collect();
            Some((
                SizedForm::Dense {
                    ty: ty.clone(),
                    splat: *splat,
                },
                v?,
            ))
        }
        _ => None,
    }
}

/// Attributes of a composite call that describe the instance, not the graph (kept in its record).
pub const CALL_ATTRS: [&str; 4] = ["composite_attributes", "decomposition", "name", "version"];

/// Abstract attributes and parameters of an op (the attribute map in key order).
pub fn abstract_attrs(op: &Op) -> (BTreeMap<String, AAttr>, Vec<Param>) {
    let mut attrs = BTreeMap::new();
    let mut params = Vec::new();
    if op.name == "stablehlo.composite" {
        return (attrs, params);
    }
    let spec = cc_ir::lookup(&op.name, op.target.as_deref());
    let sized = spec.map(|s| s.size_attrs).unwrap_or(&[]);
    for (k, a) in &op.attrs {
        let x = if op.name == "stablehlo.constant" && k == "value" {
            match a {
                Attr::Dense { splat, .. } => {
                    params.push(Param::Value(a.clone()));
                    AAttr::Value { splat: *splat }
                }
                _ => AAttr::Lit(a.clone()),
            }
        } else if sized.contains(&k.as_str())
            && let Some((form, v)) = split_sized(a)
        {
            let entries = v
                .iter()
                .map(|&e| {
                    if e == 0 || e == 1 {
                        Some(e)
                    } else {
                        params.push(Param::Int(e));
                        None
                    }
                })
                .collect();
            AAttr::Sized { form, entries }
        } else {
            AAttr::Lit(a.clone())
        };
        attrs.insert(k.clone(), x);
    }
    (attrs, params)
}

/// Per-module facts the builders share.
#[derive(Debug)]
pub struct Facts<'m> {
    pub m: &'m Module,
    /// The function each region belongs to.
    pub region_func: Vec<FuncId>,
    pub users: Vec<Vec<(OpId, u32)>>,
    /// Ops left out of every graph: size computation, and ops whose every use is a shape operand or such
    /// an op (constant shape vectors).
    pub shape_only: Vec<bool>,
    /// Size values (for `size_in` parameters).
    pub sv: Vec<Sv>,
}

impl<'m> Facts<'m> {
    pub fn new(m: &'m Module) -> Self {
        let region_func = (0..m.regions.len())
            .map(|r| m.region_func(RegionId(r as u32)))
            .collect();
        let users = m.users();
        let mut shape_only = vec![false; m.ops.len()];
        for i in (0..m.ops.len()).rev() {
            let op = &m.ops[i];
            let uses: Vec<(OpId, u32)> = op
                .results
                .iter()
                .flat_map(|r| users[r.index()].iter().copied())
                .collect();
            shape_only[i] = op.size
                || (!uses.is_empty()
                    && uses
                        .iter()
                        .all(|&(u, p)| shape_only[u.index()] || is_shape_port(m, m.op(u), p as usize)));
        }
        let sv = crate::import::algo::size_value::interpret(m, &dim_as_data_values(m));
        Self {
            m,
            region_func,
            users,
            shape_only,
            sv,
        }
    }

    pub fn op_func(&self, o: OpId) -> FuncId {
        self.region_func[self.m.op(o).parent.index()]
    }

    /// The op of the function's body region that holds `o` (`o` itself at top level).
    pub fn top(&self, mut o: OpId) -> OpId {
        loop {
            let op = self.m.op(o);
            match self.m.region(op.parent).owner {
                Owner::Op(p) => o = p,
                Owner::Func(_) => return o,
            }
        }
    }

    pub fn is_call(&self, o: OpId) -> bool {
        self.m.op(o).name == "stablehlo.composite"
    }

    /// Whether op `o` is a node of its function's graph.
    pub fn in_graph(&self, o: OpId) -> bool {
        !self.shape_only[o.index()]
    }
}

/// The values at the `dim_as_data` ports of a module.
pub fn dim_as_data_values(m: &Module) -> Vec<ValueId> {
    m.ops
        .iter()
        .flat_map(|op| op.dim_as_data.iter().map(|&p| op.operands[p as usize]))
        .collect()
}

/// The type of a value as a tensor (every value of an imported module is one).
pub fn tensor(m: &Module, v: ValueId) -> Result<&cc_ir::TensorType, String> {
    m.value(v)
        .ty
        .tensor()
        .ok_or_else(|| format!("value {} is not a tensor", v.0))
}

/// The port label of operand `p` of `op` in a graph: `Commutative` for a commutative op; for a composite,
/// the index among its non-shape operands; else the port.
pub fn port_label(m: &Module, op: &Op, p: usize) -> PortLabel {
    let spec = cc_ir::lookup(&op.name, op.target.as_deref());
    if spec.is_some_and(|s| s.commutative) {
        return PortLabel::Commutative;
    }
    if op.name == "stablehlo.composite" {
        let k = (0..p).filter(|&q| !is_shape_port(m, op, q)).count();
        return PortLabel::Port(k as u32);
    }
    PortLabel::Port(p as u32)
}

/// A graph under construction: nodes with their module identity and concrete result types.
#[derive(Debug, Default)]
pub struct Built {
    /// Nodes with structural labels; result classes are filled by `classes`.
    pub graph: Graph<ANode>,
    pub refs: Vec<NodeRef>,
    /// Concrete result types per node.
    pub types: Vec<Vec<cc_ir::TensorType>>,
    pub params: Vec<Vec<Param>>,
    /// For call nodes: the callee.
    pub callee: Vec<Option<FuncId>>,
}

fn atype(t: &cc_ir::TensorType) -> AType {
    AType {
        family: Family::of(&t.elem),
        elem: 0,
        dims: t
            .shape
            .iter()
            .map(|d| {
                if *d == Dim::Const(1) {
                    ADim::One
                } else {
                    ADim::Var(0)
                }
            })
            .collect(),
    }
}

impl Built {
    fn node(
        &mut self,
        kind: Kind,
        attrs: BTreeMap<String, AAttr>,
        types: Vec<cc_ir::TensorType>,
        params: Vec<Param>,
        r: NodeRef,
        callee: Option<FuncId>,
    ) -> u32 {
        let n = self.graph.add_node(ANode {
            kind,
            attrs,
            results: types.iter().map(atype).collect(),
            child: None,
        });
        self.refs.push(r);
        self.types.push(types);
        self.params.push(params);
        self.callee.push(callee);
        n
    }
}

/// Build the graph of a set of ops of one function. `inputs` gives the nodes of the decomposition's
/// arguments (`None` for an anon region: values from outside become unordered inputs, values used
/// outside become outputs).
pub fn build(f: &Facts<'_>, ops: &[OpId], decomposition: Option<FuncId>) -> Result<Built, String> {
    let m = f.m;
    let mut b = Built::default();
    let mut node_of_op: BTreeMap<OpId, u32> = BTreeMap::new();
    let mut node_of_val: BTreeMap<ValueId, u32> = BTreeMap::new();
    if let Some(func) = decomposition {
        let mut k = 0u32;
        for &a in &m.region(m.func(func).body).args {
            if m.value(a).dim_var.is_some() {
                continue;
            }
            let t = tensor(m, a)?.clone();
            let n = b.node(
                Kind::Input { index: Some(k) },
                BTreeMap::new(),
                vec![t],
                Vec::new(),
                NodeRef::Value(a.0),
                None,
            );
            node_of_val.insert(a, n);
            k += 1;
        }
    }
    let members: std::collections::BTreeSet<OpId> = ops.iter().copied().collect();
    let graph_ops: Vec<OpId> = ops.iter().copied().filter(|&o| f.in_graph(o)).collect();
    for &o in &graph_ops {
        let op = m.op(o);
        let (attrs, params) = abstract_attrs(op);
        let types = op
            .results
            .iter()
            .map(|&v| tensor(m, v).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let n = b.node(
            Kind::Op {
                name: op.name.clone(),
                target: op.target.clone(),
            },
            attrs,
            types,
            params,
            NodeRef::Op(o.0),
            (op.name == "stablehlo.composite").then_some(op.callee).flatten(),
        );
        node_of_op.insert(o, n);
        for (ri, &r) in op.regions.iter().enumerate() {
            for (ai, &a) in m.region(r).args.iter().enumerate() {
                let t = tensor(m, a)?.clone();
                let an = b.node(
                    Kind::RegionArg {
                        region: ri as u32,
                        index: ai as u32,
                    },
                    BTreeMap::new(),
                    vec![t],
                    Vec::new(),
                    NodeRef::Value(a.0),
                    None,
                );
                node_of_val.insert(a, an);
                b.graph.add_edge(
                    n,
                    an,
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
    // Edges, `size_in` and anon inputs.
    for &o in &graph_ops {
        let op = m.op(o);
        let dst = node_of_op[&o];
        for (p, &v) in op.operands.iter().enumerate() {
            if is_shape_port(m, op, p) {
                continue;
            }
            let port = port_label(m, op, p);
            if op.dim_as_data.contains(&(p as u32)) {
                let t = tensor(m, v)?.clone();
                let s = b.node(
                    Kind::SizeIn,
                    BTreeMap::new(),
                    vec![t],
                    vec![Param::Size(SizeValue::of(&f.sv[v.index()]))],
                    NodeRef::SizeIn {
                        op: o.0,
                        port: p as u32,
                    },
                    None,
                );
                b.graph.add_edge(s, dst, EdgeEnd { res: 0, port });
                continue;
            }
            let (src, res) = match m.value(v).def {
                Def::Result { op: u, index } if members.contains(&u) => match node_of_op.get(&u) {
                    Some(&n) => (n, index),
                    None => {
                        return Err(format!(
                            "op {} port {p}: operand from op {} outside the graph",
                            o.0, u.0
                        ));
                    }
                },
                _ => match node_of_val.get(&v) {
                    Some(&n) => (n, 0),
                    None if decomposition.is_none() => {
                        let t = tensor(m, v)?.clone();
                        let n = b.node(
                            Kind::Input { index: None },
                            BTreeMap::new(),
                            vec![t],
                            Vec::new(),
                            NodeRef::Value(v.0),
                            None,
                        );
                        node_of_val.insert(v, n);
                        (n, 0)
                    }
                    None => return Err(format!("op {} port {p}: operand {} has no node", o.0, v.0)),
                },
            };
            b.graph.add_edge(src, dst, EdgeEnd { res, port });
        }
        // A region terminator yields to its owner.
        if let Owner::Op(owner) = m.region(op.parent).owner
            && m.region(op.parent).ops.last() == Some(&o)
            && let Some(&on) = node_of_op.get(&owner)
        {
            let ri = m
                .op(owner)
                .regions
                .iter()
                .position(|&r| r == op.parent)
                .unwrap_or(0);
            b.graph.add_edge(
                dst,
                on,
                EdgeEnd {
                    res: 0,
                    port: PortLabel::Yield { region: ri as u32 },
                },
            );
        }
    }
    // Anon outputs: values used by ops outside the region (at non-shape ports).
    if decomposition.is_none() {
        for &o in &graph_ops {
            for (ri, &r) in m.op(o).results.iter().enumerate() {
                let outside = f.users[r.index()]
                    .iter()
                    .any(|&(u, p)| !members.contains(&u) && !is_shape_port(m, m.op(u), p as usize));
                if outside {
                    let n = b.node(
                        Kind::Output,
                        BTreeMap::new(),
                        Vec::new(),
                        Vec::new(),
                        NodeRef::Out(r.0),
                        None,
                    );
                    b.graph.add_edge(
                        node_of_op[&o],
                        n,
                        EdgeEnd {
                            res: ri as u32,
                            port: PortLabel::Port(0),
                        },
                    );
                }
            }
        }
    }
    Ok(b)
}

/// A node instantiated with its bindings: what the instance's node must be.
#[derive(Debug, Clone, PartialEq)]
pub struct Concrete {
    pub kind: Kind,
    pub attrs: cc_ir::Attrs,
    pub results: Vec<cc_ir::TensorType>,
    pub child: Option<String>,
    pub size: Option<SizeValue>,
}

/// Substitute bindings into an abstract node.
pub fn instantiate(n: &ANode, dims: &[Dim], elems: &[Elem], params: &[Param]) -> Result<Concrete, String> {
    let mut it = params.iter();
    let mut attrs = cc_ir::Attrs::new();
    for (k, a) in &n.attrs {
        let x = match a {
            AAttr::Lit(x) => x.clone(),
            AAttr::Value { .. } => match it.next() {
                Some(Param::Value(v)) => v.clone(),
                p => return Err(format!("attribute {k}: expected a value parameter, got {p:?}")),
            },
            AAttr::Sized { form, entries } => {
                let mut v = Vec::with_capacity(entries.len());
                for e in entries {
                    v.push(match e {
                        Some(x) => *x,
                        None => match it.next() {
                            Some(Param::Int(x)) => *x,
                            p => {
                                return Err(format!(
                                    "attribute {k}: expected an integer parameter, got {p:?}"
                                ));
                            }
                        },
                    });
                }
                rebuild(form, &v)
            }
        };
        attrs.insert(k.clone(), x);
    }
    let size = match n.kind {
        Kind::SizeIn => match it.next() {
            Some(Param::Size(s)) => Some(s.clone()),
            p => return Err(format!("size_in: expected a size parameter, got {p:?}")),
        },
        _ => None,
    };
    if it.next().is_some() {
        return Err("parameters left over".into());
    }
    let results = n
        .results
        .iter()
        .map(|t| {
            let elem = elems
                .get(t.elem as usize)
                .cloned()
                .ok_or_else(|| format!("element class {} unbound", t.elem))?;
            if Family::of(&elem) != t.family {
                return Err(format!(
                    "element class {} bound to {elem}, family {:?}",
                    t.elem, t.family
                ));
            }
            let shape = t
                .dims
                .iter()
                .map(|d| match d {
                    ADim::One => Ok(Dim::Const(1)),
                    ADim::Var(k) => dims
                        .get(*k as usize)
                        .cloned()
                        .ok_or_else(|| format!("dim class {k} unbound")),
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(cc_ir::TensorType { shape, elem })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Concrete {
        kind: n.kind.clone(),
        attrs,
        results,
        child: n.child.clone(),
        size,
    })
}
