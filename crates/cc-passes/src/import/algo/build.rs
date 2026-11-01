//! `graph.json` → `cc_ir::Module` in one walk from the entry: `func.call`s (and composites configured as
//! inlined) are inlined with provenance, component composites become ops whose decomposition is built at
//! its call site, and every value gets its type with symbolic dims.
//!
//! Symbolic shapes: the entry's inputs carry their declared expressions (`graph.json` `inputs[].shape`),
//! dimension-variable arguments (`jax.global_constant = "<sym>"`) are the variable itself, size
//! computation is evaluated over `SymExpr` (`size_value`), and each op's result shape comes from the
//! op table's shape rule. A declared constant dim must agree with the derived one; a declared `?`
//! takes the derived expression.

use std::collections::HashMap;

use cc_ir::ops::{self, ShapeIn};
use cc_ir::{
    CallSite, Constraint, Def, Dim, Func, FuncId, FuncKind, Module, Op, OpId, Owner, Provenance, Region,
    RegionId, SymCtx, SymExpr, TensorType, Type, Value, ValueId,
};

use super::convert;
use super::size_value::{self, Sv, known, opaque, single, tracked};
use crate::export::json;
use crate::import::config::ImportConfig;
use crate::import::json::Ledger;

/// Why a phase could not be imported: the function and op of `graph.json` it happened at.
#[derive(Debug, Clone, thiserror::Error)]
#[error("function `{func}`, op {op} `{name}`: {msg}")]
pub struct BuildError {
    pub func: String,
    pub op: u64,
    pub name: String,
    pub msg: String,
}

#[derive(Debug)]
pub struct Built {
    pub module: Module,
    pub ledger: Ledger,
}

type Env = HashMap<u64, ValueId>;

const GLOBAL_CONSTANT: &str = "jax.global_constant";
const MAX_DEPTH: usize = 2_000;

struct Builder<'g> {
    cfg: &'g ImportConfig,
    funcs: HashMap<&'g str, &'g json::Func>,
    m: Module,
    sv: Vec<Sv>,
    built: HashMap<String, FuncId>,
    locs: HashMap<String, u32>,
    ctx: SymCtx,
    /// Inlining done while building the current function.
    ledger: Ledger,
    /// Per built function: its inlining, nested composites included (added again at every call site).
    func_ledgers: HashMap<FuncId, Ledger>,
    depth: usize,
}

fn add_ledger(into: &mut Ledger, l: &Ledger) {
    into.calls += l.calls;
    into.returns += l.returns;
    for (k, v) in &l.composites {
        *into.composites.entry(k.clone()).or_default() += v;
    }
}

fn err(func: &str, op: &json::Op, msg: impl Into<String>) -> BuildError {
    BuildError {
        func: func.to_string(),
        op: op.id,
        name: op.name.clone(),
        msg: msg.into(),
    }
}

fn func_err(func: &str, msg: impl Into<String>) -> BuildError {
    BuildError {
        func: func.to_string(),
        op: 0,
        name: "func.func".into(),
        msg: msg.into(),
    }
}

/// Declared `?` takes the derived dim; a declared constant must equal it.
fn merge(decl: &[Option<u64>], inf: &[Dim], ctx: &SymCtx) -> Result<Vec<Dim>, String> {
    if decl.len() != inf.len() {
        return Err(format!(
            "declared rank {} but derived rank {}",
            decl.len(),
            inf.len()
        ));
    }
    decl.iter()
        .zip(inf)
        .map(|(d, i)| match (d, i) {
            (None, i) => Ok(i.clone()),
            (Some(c), Dim::Const(x)) if c == x => Ok(Dim::Const(*c)),
            (Some(c), Dim::Sym(e)) if ctx.eq(e, &SymExpr::constant(*c as i64)) == Some(true) => {
                Ok(Dim::Const(*c))
            }
            (Some(c), i) => Err(format!("declared dimension {c} but derived {i}")),
        })
        .collect()
}

/// A value bound across a call boundary must have the callee's declared type up to `?`.
fn compatible(t: &TensorType, decl: &(Vec<Option<u64>>, cc_ir::Elem)) -> bool {
    t.elem == decl.1
        && t.rank() == decl.0.len()
        && t.shape
            .iter()
            .zip(&decl.0)
            .all(|(d, e)| e.is_none_or(|c| d.as_const() == Some(c)))
}

impl<'g> Builder<'g> {
    fn loc(&mut self, l: &str) -> u32 {
        if let Some(&i) = self.locs.get(l) {
            return i;
        }
        let i = self.m.locs.len() as u32;
        self.m.locs.push(l.to_string());
        self.locs.insert(l.to_string(), i);
        i
    }

    fn value(&mut self, ty: TensorType, def: Def, dim_var: Option<String>, src: u64) -> ValueId {
        let id = ValueId(self.m.values.len() as u32);
        self.m.values.push(Value {
            ty: Type::Tensor(ty),
            def,
            dim_var,
            src,
        });
        self.sv.push(None);
        id
    }

    fn ty(&self, v: ValueId) -> &TensorType {
        match &self.m.values[v.index()].ty {
            Type::Tensor(t) => t,
            Type::Other(_) => unreachable!("only tensors are created"),
        }
    }

    fn region(&mut self, owner: Owner) -> RegionId {
        let id = RegionId(self.m.regions.len() as u32);
        self.m.regions.push(Region {
            owner,
            args: Vec::new(),
            ops: Vec::new(),
        });
        id
    }

    /// Build a function to keep (entry, decomposition, computation) with the given argument types.
    fn func(&mut self, name: &str, kind: FuncKind, args: Vec<TensorType>) -> Result<FuncId, BuildError> {
        let f = *self
            .funcs
            .get(name)
            .ok_or_else(|| func_err(name, "no such function"))?;
        let [block] = f.body.blocks.as_slice() else {
            return Err(func_err(name, "expected one block"));
        };
        if block.args.len() != args.len() {
            return Err(func_err(
                name,
                format!("{} arguments bound, {} declared", args.len(), block.args.len()),
            ));
        }
        let outer = std::mem::take(&mut self.ledger);
        let fid = FuncId(self.m.funcs.len() as u32);
        self.m.funcs.push(Func {
            name: name.to_string(),
            kind,
            body: RegionId(0),
            results: Vec::new(),
        });
        let rid = self.region(Owner::Func(fid));
        self.m.funcs[fid.index()].body = rid;
        let mut env = Env::new();
        for (i, (a, ty)) in block.args.iter().zip(args).enumerate() {
            let dim_var = f
                .arg_attrs
                .get(i)
                .and_then(|aa| aa.get(GLOBAL_CONSTANT))
                .and_then(json::Attr::as_str)
                .map(str::to_string);
            let v = self.value(
                ty,
                Def::Arg {
                    region: rid,
                    index: i as u32,
                },
                dim_var.clone(),
                a.v,
            );
            if let Some(s) = dim_var {
                if !self.m.symbols.contains(&s) {
                    return Err(func_err(
                        name,
                        format!("dimension argument `{s}` is not a declared symbol"),
                    ));
                }
                self.sv[v.index()] = Some(vec![single(SymExpr::var(&s))]);
            } else if kind != FuncKind::Entry
                && let Some(n) = tracked(self.ty(v))
            {
                // An integer argument that is not a dimension: a run-time value.
                self.sv[v.index()] = opaque(&v.0.to_string(), n);
            }
            self.m.regions[rid.index()].args.push(v);
            env.insert(a.v, v);
        }
        self.ops(&block.ops, rid, &mut env, &mut Vec::new(), name, false)?;
        // The results: the operands of the terminator.
        let term = self.m.regions[rid.index()].ops.last().copied();
        let results: Vec<Type> = match term {
            Some(t) if self.m.ops[t.index()].name == "func.return" => self.m.ops[t.index()]
                .operands
                .iter()
                .map(|&v| self.m.values[v.index()].ty.clone())
                .collect(),
            _ => return Err(func_err(name, "no func.return")),
        };
        if results.len() != f.ty.results.len() {
            return Err(func_err(name, "result count differs from the declared type"));
        }
        for (r, d) in results.iter().zip(&f.ty.results) {
            let d = convert::declared(d).map_err(|e| func_err(name, e))?;
            if !r.tensor().is_some_and(|t| compatible(t, &d)) {
                return Err(func_err(
                    name,
                    format!("result {r} does not match the declared type"),
                ));
            }
        }
        self.m.funcs[fid.index()].results = results;
        let own = std::mem::replace(&mut self.ledger, outer);
        self.func_ledgers.insert(fid, own);
        Ok(fid)
    }

    /// Emit `ops` into `region`. For an inlined body (`inlined`), the `func.return` is not emitted and
    /// its operands are returned.
    fn ops(
        &mut self,
        ops: &'g [json::Op],
        region: RegionId,
        env: &mut Env,
        inl: &mut Vec<CallSite>,
        src: &str,
        inlined: bool,
    ) -> Result<Option<Vec<ValueId>>, BuildError> {
        for op in ops {
            if inlined && op.name == "func.return" {
                let rets = self.operands(op, env, src)?;
                return Ok(Some(rets));
            }
            let comp_name = op.attributes.get("name").and_then(json::Attr::as_str);
            let callee = op.attributes.get(match op.name.as_str() {
                "func.call" => "callee",
                _ => "decomposition",
            });
            let inline_target = match op.name.as_str() {
                "func.call" => callee.and_then(json::Attr::as_str),
                "stablehlo.composite" if comp_name.is_some_and(|n| self.cfg.inlines(n)) => {
                    callee.and_then(json::Attr::as_str)
                }
                _ => None,
            };
            match inline_target {
                Some(target) => self.inline(op, target, region, env, inl, src)?,
                None => self.op(op, region, env, inl, src)?,
            }
        }
        Ok(None)
    }

    fn operands(&self, op: &json::Op, env: &Env, src: &str) -> Result<Vec<ValueId>, BuildError> {
        op.operands
            .iter()
            .map(|v| {
                env.get(v)
                    .copied()
                    .ok_or_else(|| err(src, op, format!("operand %{v} is not defined before use")))
            })
            .collect()
    }

    fn inline(
        &mut self,
        op: &'g json::Op,
        target: &str,
        region: RegionId,
        env: &mut Env,
        inl: &mut Vec<CallSite>,
        src: &str,
    ) -> Result<(), BuildError> {
        let f = *self
            .funcs
            .get(target)
            .ok_or_else(|| err(src, op, format!("no function `{target}`")))?;
        let [block] = f.body.blocks.as_slice() else {
            return Err(err(src, op, "callee has more than one block"));
        };
        let operands = self.operands(op, env, src)?;
        if operands.len() != block.args.len() {
            return Err(err(src, op, "operand count differs from the callee's arguments"));
        }
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(err(src, op, "call depth over the limit"));
        }
        let mut inner = Env::new();
        for (a, &v) in block.args.iter().zip(&operands) {
            let d = convert::declared(&a.ty).map_err(|e| err(src, op, e))?;
            if !compatible(self.ty(v), &d) {
                return Err(err(
                    src,
                    op,
                    format!("operand {} does not match the callee argument", self.ty(v)),
                ));
            }
            inner.insert(a.v, v);
        }
        inl.push(CallSite {
            op: op.id,
            callee: target.to_string(),
        });
        let rets = self.ops(&block.ops, region, &mut inner, inl, target, true)?;
        inl.pop();
        self.depth -= 1;
        let rets = rets.ok_or_else(|| err(src, op, "callee has no func.return"))?;
        if rets.len() != op.results.len() {
            return Err(err(src, op, "result count differs from the callee's"));
        }
        for (r, &v) in op.results.iter().zip(&rets) {
            let d = convert::declared(&r.ty).map_err(|e| err(src, op, e))?;
            if !compatible(self.ty(v), &d) {
                return Err(err(
                    src,
                    op,
                    format!("result {} does not match the declared type", self.ty(v)),
                ));
            }
            env.insert(r.v, v);
        }
        if op.name == "func.call" {
            self.ledger.calls += 1;
        } else if let Some(n) = op.attributes.get("name").and_then(json::Attr::as_str) {
            *self.ledger.composites.entry(n.to_string()).or_default() += 1;
        }
        self.ledger.returns += 1;
        Ok(())
    }

    fn op(
        &mut self,
        op: &'g json::Op,
        region: RegionId,
        env: &mut Env,
        inl: &mut Vec<CallSite>,
        src: &str,
    ) -> Result<(), BuildError> {
        let target = (op.name == "stablehlo.custom_call")
            .then(|| op.attributes.get("call_target_name").and_then(json::Attr::as_str))
            .flatten();
        let spec = ops::lookup(&op.name, target).ok_or_else(|| {
            err(
                src,
                op,
                match target {
                    Some(t) => format!("custom_call target `{t}` is not in the op-semantics table"),
                    None => "op is not in the op-semantics table".to_string(),
                },
            )
        })?;
        if !spec.operands.admits(op.operands.len()) || !spec.results.admits(op.results.len()) {
            return Err(err(src, op, "operand or result count outside the table's arity"));
        }
        if op.regions.len() != spec.regions as usize {
            return Err(err(src, op, "region count differs from the table"));
        }
        let operands = self.operands(op, env, src)?;
        let types: Vec<TensorType> = operands.iter().map(|&v| self.ty(v).clone()).collect();
        // The callee: a component's decomposition, or a custom call's computation.
        let callee = match op.name.as_str() {
            "stablehlo.composite" => {
                let d = op
                    .attributes
                    .get("decomposition")
                    .and_then(json::Attr::as_str)
                    .ok_or_else(|| err(src, op, "composite without decomposition"))?;
                Some(self.callee(op, d, FuncKind::Decomposition, types.clone(), src)?)
            }
            "stablehlo.custom_call" => match op.attributes.get("called_computations") {
                Some(json::Attr::Array { value }) => match value.first().and_then(json::Attr::as_str) {
                    Some(c) => {
                        let f = *self
                            .funcs
                            .get(c)
                            .ok_or_else(|| err(src, op, format!("no function `{c}`")))?;
                        let args =
                            f.ty.inputs
                                .iter()
                                .map(|t| {
                                    static_tensor(t)
                                        .ok_or_else(|| err(src, op, "computation with a dynamic argument"))
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                        Some(self.callee(op, c, FuncKind::Computation, args, src)?)
                    }
                    None => None,
                },
                _ => None,
            },
            _ => None,
        };
        // Dimension arguments of a composite must be bound to their own symbol.
        if let Some(f) = callee
            && op.name == "stablehlo.composite"
        {
            let body = self.m.funcs[f.index()].body;
            for (i, &a) in self.m.regions[body.index()].args.clone().iter().enumerate() {
                if let Some(s) = self.m.values[a.index()].dim_var.clone() {
                    let bound = known(&self.sv[operands[i].index()]);
                    if bound != Some(vec![SymExpr::var(&s)]) {
                        return Err(err(
                            src,
                            op,
                            format!("dimension argument `{s}` bound to {bound:?}"),
                        ));
                    }
                }
            }
        }
        let oid = OpId(self.m.ops.len() as u32);
        let loc = self.loc(&op.loc);
        self.m.ops.push(Op {
            name: op.name.clone(),
            target: target.map(str::to_string),
            operands: operands.clone(),
            results: Vec::new(),
            attrs: convert::attrs(&op.attributes),
            regions: Vec::new(),
            callee,
            parent: region,
            size: false,
            dim_as_data: Vec::new(),
            prov: Provenance {
                func: src.to_string(),
                op: op.id,
                loc,
                inlined_from: inl.clone(),
            },
        });
        self.m.regions[region.index()].ops.push(oid);
        for r in &op.regions {
            let [block] = r.blocks.as_slice() else {
                return Err(err(src, op, "region with more than one block"));
            };
            let rid = self.region(Owner::Op(oid));
            for (i, a) in block.args.iter().enumerate() {
                let t = static_tensor(&a.ty)
                    .ok_or_else(|| err(src, op, "region argument with a dynamic type"))?;
                let v = self.value(
                    t,
                    Def::Arg {
                        region: rid,
                        index: i as u32,
                    },
                    None,
                    a.v,
                );
                self.m.regions[rid.index()].args.push(v);
                env.insert(a.v, v);
            }
            self.m.ops[oid.index()].regions.push(rid);
            self.ops(&block.ops, rid, env, inl, src, false)?;
        }
        // Result types: declared element types, dims from the shape rule.
        let decl: Vec<(Vec<Option<u64>>, cc_ir::Elem)> = op
            .results
            .iter()
            .map(|r| convert::declared(&r.ty))
            .collect::<Result<_, _>>()
            .map_err(|e| err(src, op, e))?;
        let callee_results: Option<Vec<TensorType>> = callee.map(|f| {
            self.m.funcs[f.index()]
                .results
                .iter()
                .filter_map(|t| t.tensor().cloned())
                .collect()
        });
        let refs: Vec<&TensorType> = types.iter().collect();
        let vals: Vec<Sv> = operands.iter().map(|v| self.sv[v.index()].clone()).collect();
        let shape_vals: Vec<Option<Vec<SymExpr>>> = vals.iter().map(known).collect();
        let declared: Vec<Vec<Option<u64>>> = decl.iter().map(|d| d.0.clone()).collect();
        let attrs = &self.m.ops[oid.index()].attrs;
        let shapes = (spec.shape)(&ShapeIn {
            operands: &refs,
            attrs,
            values: &shape_vals,
            declared: &declared,
            callee: callee_results.as_deref(),
            ctx: &self.ctx,
        })
        .map_err(|e| err(src, op, e))?;
        if shapes.len() != decl.len() {
            return Err(err(
                src,
                op,
                format!("{} results derived, {} declared", shapes.len(), decl.len()),
            ));
        }
        let mut res_types = Vec::with_capacity(shapes.len());
        for ((d, e), s) in decl.into_iter().zip(shapes) {
            let shape = merge(&d, &s, &self.ctx).map_err(|m| err(src, op, m))?;
            for dim in &shape {
                if let Dim::Sym(e) = dim
                    && let Some(x) = e.vars().into_iter().find(|x| !self.m.symbols.contains(x))
                {
                    return Err(err(
                        src,
                        op,
                        format!("dimension {e} depends on the run-time value {x}"),
                    ));
                }
            }
            res_types.push(TensorType { shape, elem: e });
        }
        let results: Vec<ValueId> = res_types
            .iter()
            .zip(&op.results)
            .enumerate()
            .map(|(i, (t, r))| {
                self.value(
                    t.clone(),
                    Def::Result {
                        op: oid,
                        index: i as u32,
                    },
                    None,
                    r.v,
                )
            })
            .collect();
        for (r, &v) in op.results.iter().zip(&results) {
            env.insert(r.v, v);
        }
        let attrs = &self.m.ops[oid.index()].attrs;
        let mut out = size_value::eval(&op.name, attrs, &refs, &vals, &res_types, &self.ctx);
        if op.name == "stablehlo.select" && out.first().is_some_and(|o| known(o).is_none()) {
            let m = &self.m;
            let def = |v: ValueId| match m.values[v.index()].def {
                Def::Result { op, .. } => {
                    let o = &m.ops[op.index()];
                    Some((o.name.clone(), o.operands.clone(), o.attrs.clone()))
                }
                Def::Arg { .. } => None,
            };
            let val =
                |v: ValueId| known(&self.sv[v.index()]).and_then(|x| (x.len() == 1).then(|| x[0].clone()));
            if let Some(e) = size_value::floor_identity(&operands, &def, &val, &self.ctx) {
                out[0] = Some(vec![single(e)]);
            }
        }
        for (&v, s) in results.iter().zip(out) {
            self.sv[v.index()] = match s {
                Some(s) => Some(s),
                None => tracked(self.ty(v)).and_then(|n| opaque(&v.0.to_string(), n)),
            };
        }
        self.m.ops[oid.index()].results = results;
        Ok(())
    }

    fn callee(
        &mut self,
        op: &json::Op,
        name: &str,
        kind: FuncKind,
        args: Vec<TensorType>,
        src: &str,
    ) -> Result<FuncId, BuildError> {
        if let Some(&f) = self.built.get(name) {
            let body = self.m.funcs[f.index()].body;
            let have: Vec<&Type> = self.m.regions[body.index()]
                .args
                .iter()
                .map(|&a| &self.m.values[a.index()].ty)
                .collect();
            let same = have.len() == args.len() && have.iter().zip(&args).all(|(h, a)| h.tensor() == Some(a));
            if !same {
                return Err(err(
                    src,
                    op,
                    format!("`{name}` is called with different argument types"),
                ));
            }
            if let Some(l) = self.func_ledgers.get(&f) {
                add_ledger(&mut self.ledger, &l.clone());
            }
            return Ok(f);
        }
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(err(src, op, "call depth over the limit"));
        }
        let f = self.func(name, kind, args)?;
        self.depth -= 1;
        self.built.insert(name.to_string(), f);
        // Computations are not executed through calls (the export census does not expand them).
        if kind == FuncKind::Decomposition
            && let Some(l) = self.func_ledgers.get(&f)
        {
            add_ledger(&mut self.ledger, &l.clone());
        }
        Ok(f)
    }
}

fn static_tensor(t: &json::Type) -> Option<TensorType> {
    match convert::static_type(t) {
        Type::Tensor(t) => Some(t),
        Type::Other(_) => None,
    }
}

/// The entry's argument types: the declared input shapes (symbolic where the port declared a symbol).
fn entry_args(g: &json::GraphFile, f: &json::Func) -> Result<Vec<TensorType>, BuildError> {
    let by_arg: HashMap<u64, &json::Input> = g.inputs.iter().filter_map(|i| Some((i.main_arg?, i))).collect();
    f.ty.inputs
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let (decl, elem) = convert::declared(t).map_err(|e| func_err(&f.name, e))?;
            let shape = match by_arg.get(&(i as u64)) {
                Some(inp) => {
                    if inp.shape.len() != decl.len() {
                        return Err(func_err(
                            &f.name,
                            format!("input {} rank differs from main's argument", inp.path),
                        ));
                    }
                    inp.shape
                        .iter()
                        .zip(&decl)
                        .map(|(d, c)| {
                            let dim = match d {
                                json::Dim::Const(c) => Dim::Const(*c),
                                json::Dim::Sym(s) => {
                                    let e =
                                        SymExpr::parse(s).map_err(|e| func_err(&f.name, e.to_string()))?;
                                    Dim::from_expr(e)
                                        .ok_or_else(|| func_err(&f.name, "negative input dim"))?
                                }
                            };
                            match (c, &dim) {
                                (Some(c), Dim::Const(x)) if c != x => Err(func_err(
                                    &f.name,
                                    format!("input {} declares {x}, main has {c}", inp.path),
                                )),
                                _ => Ok(dim),
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?
                }
                None => decl
                    .iter()
                    .map(|c| {
                        c.map(Dim::Const).ok_or_else(|| {
                            func_err(&f.name, format!("argument {i} is dynamic but not an input"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            };
            Ok(TensorType { shape, elem })
        })
        .collect()
}

/// Build the module of one phase.
pub fn build(g: &json::GraphFile, syms: &json::SymbolsFile, cfg: &ImportConfig) -> Result<Built, BuildError> {
    let constraints: Vec<Constraint> = syms
        .constraints
        .iter()
        .map(|c| Constraint::parse(c).map_err(|e| func_err(&g.entry, e.to_string())))
        .collect::<Result<_, _>>()?;
    let symbols: Vec<String> = syms.symbols.keys().cloned().collect();
    let ctx = SymCtx::new(&symbols, &constraints);
    let funcs: HashMap<&str, &json::Func> = g.module.funcs.iter().map(|f| (f.name.as_str(), f)).collect();
    let entry = *funcs
        .get(g.entry.as_str())
        .ok_or_else(|| func_err(&g.entry, "no entry function"))?;
    let mut b = Builder {
        cfg,
        funcs,
        m: Module {
            schema: cc_ir::ir::IR_SCHEMA.into(),
            model: g.model.clone(),
            phase: g.phase.clone(),
            config: g.config.clone(),
            symbols,
            constraints,
            entry: FuncId(0),
            funcs: Vec::new(),
            regions: Vec::new(),
            ops: Vec::new(),
            values: Vec::new(),
            locs: Vec::new(),
        },
        sv: Vec::new(),
        built: HashMap::new(),
        locs: HashMap::new(),
        ctx,
        ledger: Ledger::default(),
        func_ledgers: HashMap::new(),
        depth: 0,
    };
    let args = entry_args(g, entry)?;
    let fid = b.func(&g.entry, FuncKind::Entry, args)?;
    b.m.entry = fid;
    let ledger = b.func_ledgers.remove(&fid).unwrap_or_default();
    Ok(Built { module: b.m, ledger })
}
