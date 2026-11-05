//! Constant-computation tagging (design §4.2).
//!
//! A value is *reached* when it depends on a data argument of the phase: an entry argument that is not a
//! dimension variable (activations, weights, run-time inputs), or the result of an operand-less op that
//! does not take its value from attributes (none in the op table today). Reachability runs forward over
//! the inlined graph:
//!
//! - an op is reached when a reached value enters it at a port that is not a shape operand, or when an op
//!   of one of its regions is reached (a value captured from the enclosing scope);
//! - a reached op's results and its regions' arguments are reached, and so is every argument of a custom
//!   call's computation (its reducer);
//! - a decomposition argument is reached when the operand bound to it at some call is reached, so an op
//!   of a shared decomposition is unreached only when it is constant at every call;
//! - size computation is never reached: its values are dimension values.
//!
//! An unreached op computes a constant from constants, `iota` and size values. It is tagged `constant`
//! unless it only materializes a literal (a `constant`, an `iota`, or a layout op whose data operands are
//! such literals: `normalize` folds splat and broadcast constants into ports) or is structure (returns,
//! composite calls). The ops in the regions of a tagged op are tagged with it. Size computation is
//! unreached and tagged: it is a part of constant computation.

use cc_ir::ops::OpClass;
use cc_ir::{Def, FuncKind, Module, Op, OpId, Owner, ValueId};

use super::size_tag::is_shape_port;

fn class(op: &Op) -> Option<OpClass> {
    cc_ir::lookup(&op.name, op.target.as_deref()).map(|s| s.class)
}

/// Per op: whether it is reached from a data argument (module doc).
pub fn reached(m: &Module) -> Vec<bool> {
    let mut r = Reach {
        m,
        users: m.users(),
        val: vec![false; m.values.len()],
        op: vec![false; m.ops.len()],
        values: Vec::new(),
        ops: Vec::new(),
    };
    for &a in &m.region(m.func(m.entry).body).args {
        if m.value(a).dim_var.is_none() {
            r.value(a);
        }
    }
    for (i, op) in m.ops.iter().enumerate() {
        if op.operands.is_empty() && !matches!(class(op), Some(OpClass::Constant | OpClass::Control)) {
            r.op(OpId(i as u32));
        }
    }
    r.run();
    r.op
}

struct Reach<'m> {
    m: &'m Module,
    users: Vec<Vec<(OpId, u32)>>,
    val: Vec<bool>,
    op: Vec<bool>,
    values: Vec<ValueId>,
    ops: Vec<OpId>,
}

impl Reach<'_> {
    fn value(&mut self, v: ValueId) {
        if !self.val[v.index()] {
            self.val[v.index()] = true;
            self.values.push(v);
        }
    }

    fn op(&mut self, o: OpId) {
        if !self.op[o.index()] && !self.m.op(o).size {
            self.op[o.index()] = true;
            self.ops.push(o);
        }
    }

    fn run(&mut self) {
        let m = self.m;
        loop {
            if let Some(v) = self.values.pop() {
                for i in 0..self.users[v.index()].len() {
                    let (u, p) = self.users[v.index()][i];
                    let op = m.op(u);
                    if is_shape_port(m, op, p as usize) {
                        continue;
                    }
                    self.op(u);
                    if op.name == "stablehlo.composite"
                        && let Some(f) = op.callee
                        && let Some(&a) = m.region(m.func(f).body).args.get(p as usize)
                    {
                        self.value(a);
                    }
                }
            } else if let Some(o) = self.ops.pop() {
                let op = m.op(o);
                for &r in &op.results {
                    self.value(r);
                }
                for &r in &op.regions {
                    for &a in &m.region(r).args {
                        self.value(a);
                    }
                }
                if let Some(f) = op.callee
                    && m.func(f).kind == FuncKind::Computation
                {
                    for &a in &m.region(m.func(f).body).args {
                        self.value(a);
                    }
                }
                if let Owner::Op(p) = m.region(op.parent).owner {
                    self.op(p);
                }
            } else {
                return;
            }
        }
    }
}

/// The constant tags from the reached ops (module doc).
pub fn tags(m: &Module, reached: &[bool]) -> Vec<bool> {
    // Op ids order every definition before its uses, so one forward pass finds the literals.
    let mut literal = vec![false; m.ops.len()];
    for (i, op) in m.ops.iter().enumerate() {
        literal[i] = !reached[i]
            && match class(op) {
                Some(OpClass::Constant) => true,
                Some(OpClass::Layout) => op.operands.iter().enumerate().all(|(p, &v)| {
                    is_shape_port(m, op, p)
                        || matches!(m.value(v).def, Def::Result { op: d, .. } if literal[d.index()])
                }),
                _ => false,
            };
    }
    let own: Vec<bool> = m
        .ops
        .iter()
        .enumerate()
        .map(|(i, op)| !reached[i] && !literal[i] && class(op) != Some(OpClass::Control))
        .collect();
    // The regions of a tagged op go with it.
    (0..m.ops.len())
        .map(|i| {
            let mut o = OpId(i as u32);
            loop {
                if own[o.index()] {
                    return true;
                }
                match m.region(m.op(o).parent).owner {
                    Owner::Op(p) => o = p,
                    Owner::Func(_) => return false,
                }
            }
        })
        .collect()
}

/// Set `constant` on every op. Runs after size tagging (`size_tag::tag`), whose tags it reads.
pub fn tag(m: &mut Module) {
    let t = tags(m, &reached(m));
    for (op, c) in m.ops.iter_mut().zip(t) {
        op.constant = c;
    }
}

/// Whether value `v` is a constant entering compute: the result of a constant-tagged op that is not
/// size computation (a size value entering compute is a `dim_as_data` port instead).
pub fn const_value(m: &Module, v: ValueId) -> bool {
    match m.value(v).def {
        Def::Result { op, .. } => {
            let d = m.op(op);
            d.constant && !d.size
        }
        Def::Arg { .. } => false,
    }
}

/// The operand ports of op `o` where a constant enters compute (`const_value` at a non-shape port of an
/// op that is not constant-tagged), ascending.
pub fn const_in_ports(m: &Module, o: OpId) -> Vec<u32> {
    let op = m.op(o);
    if op.constant {
        return Vec::new();
    }
    op.operands
        .iter()
        .enumerate()
        .filter(|&(p, &v)| const_value(m, v) && !is_shape_port(m, op, p))
        .map(|(p, _)| p as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cc_ir::{Dim, Elem, Func, Provenance, Region, RegionId, TensorType, Type, Value};

    fn ty(shape: &[u64], elem: &str) -> Type {
        Type::Tensor(TensorType {
            shape: shape.iter().map(|&d| Dim::Const(d)).collect(),
            elem: Elem::parse(elem).unwrap(),
        })
    }

    /// A module under construction: function bodies and op regions are added as regions.
    struct B {
        m: Module,
    }

    impl B {
        fn new() -> Self {
            let mut b = B {
                m: Module {
                    schema: String::new(),
                    model: "t".into(),
                    phase: "p".into(),
                    config: "reduced".into(),
                    symbols: Vec::new(),
                    constraints: Vec::new(),
                    entry: cc_ir::FuncId(0),
                    funcs: Vec::new(),
                    regions: Vec::new(),
                    ops: Vec::new(),
                    values: Vec::new(),
                    locs: vec![String::new()],
                },
            };
            b.func("main", FuncKind::Entry);
            b
        }

        fn func(&mut self, name: &str, kind: FuncKind) -> RegionId {
            let f = cc_ir::FuncId(self.m.funcs.len() as u32);
            let r = RegionId(self.m.regions.len() as u32);
            self.m.regions.push(Region {
                owner: Owner::Func(f),
                args: Vec::new(),
                ops: Vec::new(),
            });
            self.m.funcs.push(Func {
                name: name.into(),
                kind,
                body: r,
                results: Vec::new(),
            });
            r
        }

        fn arg(&mut self, r: RegionId, t: Type, dim_var: Option<&str>) -> ValueId {
            let v = ValueId(self.m.values.len() as u32);
            let index = self.m.regions[r.index()].args.len() as u32;
            self.m.values.push(Value {
                ty: t,
                def: Def::Arg { region: r, index },
                dim_var: dim_var.map(str::to_string),
                src: 0,
            });
            self.m.regions[r.index()].args.push(v);
            v
        }

        fn op(&mut self, r: RegionId, name: &str, operands: &[ValueId], res: Option<Type>) -> (OpId, ValueId) {
            let o = OpId(self.m.ops.len() as u32);
            let mut results = Vec::new();
            if let Some(t) = res {
                results.push(ValueId(self.m.values.len() as u32));
                self.m.values.push(Value {
                    ty: t,
                    def: Def::Result { op: o, index: 0 },
                    dim_var: None,
                    src: 0,
                });
            }
            self.m.ops.push(Op {
                name: name.into(),
                target: None,
                operands: operands.to_vec(),
                results: results.clone(),
                attrs: Default::default(),
                regions: Vec::new(),
                callee: None,
                parent: r,
                size: false,
                constant: false,
                dim_as_data: Vec::new(),
                prov: Provenance {
                    func: String::new(),
                    op: 0,
                    loc: 0,
                    inlined_from: Vec::new(),
                },
            });
            self.m.regions[r.index()].ops.push(o);
            (o, results.first().copied().unwrap_or(ValueId(u32::MAX)))
        }

        /// A region of op `o`.
        fn region(&mut self, o: OpId) -> RegionId {
            let r = RegionId(self.m.regions.len() as u32);
            self.m.regions.push(Region {
                owner: Owner::Op(o),
                args: Vec::new(),
                ops: Vec::new(),
            });
            self.m.ops[o.index()].regions.push(r);
            r
        }

        fn tagged(mut self) -> Module {
            super::super::size_tag::tag(&mut self.m);
            tag(&mut self.m);
            self.m
        }
    }

    #[test]
    fn tables_from_iota_and_sizes_are_constant_and_literals_are_not() {
        let mut b = B::new();
        let main = RegionId(0);
        let x = b.arg(main, ty(&[8], "f32"), None);
        let s = b.arg(main, ty(&[], "i32"), Some("S"));
        let (c0, zero) = b.op(main, "stablehlo.constant", &[], Some(ty(&[], "f32")));
        let (bc, zeros) = b.op(main, "stablehlo.broadcast_in_dim", &[zero], Some(ty(&[8], "f32")));
        let (io, idx) = b.op(main, "stablehlo.iota", &[], Some(ty(&[8], "i32")));
        let (cv, pos) = b.op(main, "stablehlo.convert", &[idx], Some(ty(&[8], "f32")));
        let (sn, table) = b.op(main, "stablehlo.sine", &[pos], Some(ty(&[8], "f32")));
        let (cs, _) = b.op(main, "stablehlo.convert", &[s], Some(ty(&[], "f32")));
        let (gd, _) = b.op(main, "stablehlo.get_dimension_size", &[x], Some(ty(&[], "i32")));
        let (ad, y) = b.op(main, "stablehlo.add", &[x, table], Some(ty(&[8], "f32")));
        let (mx, z) = b.op(main, "stablehlo.maximum", &[y, zeros], Some(ty(&[8], "f32")));
        let (rt, _) = b.op(main, "func.return", &[z], None);
        let m = b.tagged();
        let t = |o: OpId| m.op(o).constant;
        assert!(!t(c0) && !t(bc) && !t(io), "literals stay");
        assert!(t(cv) && t(sn) && t(cs), "computed from iota and sizes");
        assert!(m.op(gd).size && t(gd), "size computation is constant computation");
        assert!(!t(ad) && !t(mx) && !t(rt), "reached from x");
        assert_eq!(const_in_ports(&m, ad), vec![1]);
        assert!(const_in_ports(&m, mx).is_empty());
    }

    #[test]
    fn a_shared_decomposition_is_constant_only_when_every_call_is() {
        for data_call in [false, true] {
            let mut b = B::new();
            let main = RegionId(0);
            let x = b.arg(main, ty(&[8], "f32"), None);
            let body = b.func("dec", FuncKind::Decomposition);
            let a = b.arg(body, ty(&[8], "f32"), None);
            let (ng, n) = b.op(body, "stablehlo.negate", &[a], Some(ty(&[8], "f32")));
            b.op(body, "func.return", &[n], None);
            let (_, idx) = b.op(main, "stablehlo.iota", &[], Some(ty(&[8], "f32")));
            let (c1, r1) = b.op(main, "stablehlo.composite", &[idx], Some(ty(&[8], "f32")));
            let (ex, e) = b.op(main, "stablehlo.exponential", &[r1], Some(ty(&[8], "f32")));
            let arg = if data_call { x } else { idx };
            let (c2, r2) = b.op(main, "stablehlo.composite", &[arg], Some(ty(&[8], "f32")));
            let (ad, s) = b.op(main, "stablehlo.add", &[e, r2], Some(ty(&[8], "f32")));
            b.op(main, "func.return", &[s], None);
            b.m.ops[c1.index()].callee = Some(cc_ir::FuncId(1));
            b.m.ops[c2.index()].callee = Some(cc_ir::FuncId(1));
            let m = b.tagged();
            assert_eq!(m.op(ng).constant, !data_call);
            assert!(!m.op(c1).constant && !m.op(c2).constant, "calls are structure");
            assert!(m.op(ex).constant, "the result of a constant call is a constant");
            assert_eq!(m.op(ad).constant, !data_call);
        }
    }

    #[test]
    fn regions_go_with_their_owner_and_captures_reach_it() {
        let mut b = B::new();
        let main = RegionId(0);
        let x = b.arg(main, ty(&[8], "f32"), None);
        let (_, idx) = b.op(main, "stablehlo.iota", &[], Some(ty(&[8], "f32")));
        let (_, init) = b.op(main, "stablehlo.constant", &[], Some(ty(&[], "f32")));
        let reduce = |b: &mut B, input: ValueId, capture: Option<ValueId>| {
            let (rd, out) = b.op(main, "stablehlo.reduce", &[input, init], Some(ty(&[], "f32")));
            let r = b.region(rd);
            let p = b.arg(r, ty(&[], "f32"), None);
            let q = b.arg(r, ty(&[], "f32"), None);
            let (ad, s) = b.op(r, "stablehlo.add", &[p, capture.unwrap_or(q)], Some(ty(&[], "f32")));
            let (rt, _) = b.op(r, "stablehlo.return", &[s], None);
            (rd, ad, rt, out)
        };
        let (r1, a1, t1, _) = reduce(&mut b, idx, None);
        let (r2, a2, t2, _) = reduce(&mut b, x, None);
        let (_, xs) = b.op(main, "stablehlo.reshape", &[x], Some(ty(&[], "f32")));
        let (r3, a3, _, _) = reduce(&mut b, idx, Some(xs));
        let m = b.tagged();
        let t = |o: OpId| m.op(o).constant;
        assert!(t(r1) && t(a1) && t(t1), "a reduction of a table, with its body");
        assert!(!t(r2) && !t(a2) && !t(t2));
        assert!(!t(r3) && !t(a3), "a captured data value reaches the owner");
    }
}
