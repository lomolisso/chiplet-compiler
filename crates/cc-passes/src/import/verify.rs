//! Independent checks of the import (design §4.2, §5). They read the written module and the export's
//! files, and never call the builder. Per phase (check names `<model>:<phase>:<check>`):
//!
//! - `ssa`: every value is defined once, by the op or region its `def` names, and every use is dominated
//!   by its definition (earlier in the same region or in an enclosing one).
//! - `types`: every op's result types follow its table entry (element rule, and the shape rule
//!   re-derived symbolically from the stored operand types for every op whose shape does not depend on
//!   operand values); every value agrees with its declared type in `graph.json` (rank, element type,
//!   constant dims; a `?` may be any dim).
//! - `sym_eval`: at sample sizes satisfying the constraints, a concrete integer interpreter of the
//!   module (StableHLO's truncating division) recomputes every size value; at each point every op's
//!   stored result dims equal its table rule on the concrete operands, every shape assertion holds, and
//!   every dimension argument of a composite is bound to its symbol's value.
//! - `census`: the module's op multiset (composites expanded) equals the export census minus what
//!   inlining removed, recounted here from `graph.json`; the inlining ledger matches; the GEMM and
//!   convolution signatures and declared shapes equal `census.json`'s.
//! - `coverage`: every op has a table entry and fits its arity.
//! - `size_tag`: the size tags and `dim_as_data` ports equal an independent recomputation (work lists
//!   over users, and an iterated greatest fixpoint instead of the builder's ordered passes).
//! - `summary`: `ir.postcard.summary.json` equals the summary recomputed from the module (its `ir_hash`
//!   included).
//!
//! Once per run: `inventory` (`op_inventory.json` lists every op of every phase, none unknown) and
//! `imported` (every exported phase has a module).

use std::collections::{BTreeMap, HashMap, HashSet};

use cc_core::{Check, VerifyReport};
use cc_ir::ops::{ElemRule, OpClass, ShapeIn, ShapePorts};
use cc_ir::{
    Attr, AttrsExt, Def, Dim, Elem, FuncKind, Module, Op, OpId, Owner, RegionId, Scalar, SymCtx, SymExpr,
    TensorType, ValueId,
};
use cc_pipeline::{PassContext, PassError};
use rayon::prelude::*;

use super::json::{Inventory, Ledger, Summary};
use super::{IMPORT, INVENTORY_FILE, IR_FILE, ImportConfig, op_key, phases, summarize};
use crate::export::EXPORT;
use crate::export::json::{self as ej, CensusFile, GraphFile};

pub(super) fn verify(ctx: &PassContext) -> Result<VerifyReport, PassError> {
    let cfg: ImportConfig = ctx.options()?;
    let units = phases(ctx)?;
    let per: Vec<Vec<Check>> = units
        .par_iter()
        .with_max_len(1)
        .map(|(m, p)| verify_phase(ctx, &cfg, m, p))
        .collect::<Result<_, _>>()?;
    let mut r = VerifyReport::new(ctx.name());
    let out = ctx.output(IMPORT)?;
    let mut imported = Acc::new("imported");
    for (m, p) in &units {
        if !out.path(&format!("{m}/{p}/{IR_FILE}"))?.is_file() {
            imported.fail(format!("{m}/{p}: no {IR_FILE}"));
        }
    }
    imported.count("phases", units.len() as u64);
    r.push(imported.done());
    r.push(check_inventory(ctx, &units)?);
    for c in per.into_iter().flatten() {
        r.push(c);
    }
    Ok(r)
}

/// Failing items of one check, with its evidence.
struct Acc {
    name: String,
    notes: Vec<String>,
    counts: BTreeMap<String, u64>,
}

impl Acc {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            notes: Vec::new(),
            counts: BTreeMap::new(),
        }
    }
    fn fail(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }
    fn count(&mut self, k: &str, v: u64) {
        *self.counts.entry(k.into()).or_default() += v;
    }
    fn done(self) -> Check {
        let mut c = Check::new(self.name, self.notes.is_empty());
        let n = self.notes.len();
        for note in self.notes.into_iter().take(50) {
            c = c.note(note);
        }
        if n > 50 {
            c = c.note(format!("… {} more", n - 50));
        }
        for (k, v) in self.counts {
            c = c.count(k, v);
        }
        c
    }
}

struct Phase {
    m: Module,
    summary: Summary,
    graph: GraphFile,
    census: CensusFile,
}

fn load(ctx: &PassContext, model: &str, phase: &str) -> Result<Phase, String> {
    let out = ctx.output(IMPORT).map_err(|e| e.to_string())?;
    let input = ctx.input(EXPORT).map_err(|e| e.to_string())?;
    let rel = format!("{model}/{phase}");
    Ok(Phase {
        m: out
            .read_postcard(&format!("{rel}/{IR_FILE}"))
            .map_err(|e| e.to_string())?,
        summary: out
            .read_json(&format!("{rel}/{IR_FILE}.summary.json"))
            .map_err(|e| e.to_string())?,
        graph: input
            .read_json(&format!("{rel}/graph.json"))
            .map_err(|e| e.to_string())?,
        census: input
            .read_json(&format!("{rel}/census.json"))
            .map_err(|e| e.to_string())?,
    })
}

const CHECKS: [&str; 7] = [
    "ssa", "types", "sym_eval", "census", "coverage", "size_tag", "summary",
];

fn verify_phase(
    ctx: &PassContext,
    cfg: &ImportConfig,
    model: &str,
    phase: &str,
) -> Result<Vec<Check>, PassError> {
    ctx.check_cancelled()?;
    let name = |c: &str| format!("{model}:{phase}:{c}");
    let p = match load(ctx, model, phase) {
        Ok(p) => p,
        Err(e) => {
            return Ok(CHECKS
                .iter()
                .map(|c| Check::new(name(c), false).note(format!("cannot load: {e}")))
                .collect());
        }
    };
    let ctxs = SymCtx::new(&p.m.symbols, &p.m.constraints);
    let mut out = Vec::new();
    let mut a = Acc::new(name("ssa"));
    check_ssa(&p.m, &mut a);
    let ssa_ok = a.notes.is_empty();
    out.push(a.done());
    // The other checks walk the module; a broken one is reported once, under `ssa`.
    let guarded = |c: &str, f: &dyn Fn(&mut Acc)| {
        let mut a = Acc::new(name(c));
        if ssa_ok {
            f(&mut a);
        } else {
            a.fail("skipped: the module is not well formed (see ssa)");
        }
        a.done()
    };
    out.push(guarded("types", &|a| check_types(&p, &ctxs, a)));
    out.push(guarded("sym_eval", &|a| {
        check_sym_eval(&p, u64::from(ctx.testing().tiny_dim), a)
    }));
    out.push(guarded("census", &|a| check_census(&p, cfg, a)));
    out.push(guarded("coverage", &|a| check_coverage(&p.m, a)));
    out.push(guarded("size_tag", &|a| check_size_tag(&p.m, a)));
    out.push(guarded("summary", &|a| {
        let mut s = summarize(&p.m, p.summary.ledger.clone());
        // JSON floats read back within an ulp.
        if (s.size_share - p.summary.size_share).abs() < 1e-9 {
            s.size_share = p.summary.size_share;
        }
        if s != p.summary {
            let (x, y) = (
                serde_json::to_value(&s).unwrap_or_default(),
                serde_json::to_value(&p.summary).unwrap_or_default(),
            );
            let fields: Vec<String> = match (x, y) {
                (serde_json::Value::Object(x), serde_json::Value::Object(y)) => x
                    .iter()
                    .filter(|(k, v)| y.get(*k) != Some(*v))
                    .map(|(k, _)| k.clone())
                    .collect(),
                _ => Vec::new(),
            };
            a.fail(format!(
                "the summary differs from the one recomputed from the module in {fields:?}"
            ));
        }
    }));
    Ok(out)
}

// ---------------------------------------------------------------- ssa

fn check_ssa(m: &Module, a: &mut Acc) {
    let nv = m.values.len();
    let mut defs = vec![0u32; nv];
    let in_range = |v: ValueId| v.index() < nv;
    for (ri, r) in m.regions.iter().enumerate() {
        for (i, &v) in r.args.iter().enumerate() {
            if !in_range(v) {
                a.fail(format!("region {ri}: argument out of range"));
                continue;
            }
            defs[v.index()] += 1;
            if m.value(v).def
                != (Def::Arg {
                    region: RegionId(ri as u32),
                    index: i as u32,
                })
            {
                a.fail(format!(
                    "value {}: def does not name region {ri} argument {i}",
                    v.0
                ));
            }
        }
    }
    for (oi, op) in m.ops.iter().enumerate() {
        for (i, &v) in op.results.iter().enumerate() {
            if !in_range(v) {
                a.fail(format!("op {oi}: result out of range"));
                continue;
            }
            defs[v.index()] += 1;
            if m.value(v).def
                != (Def::Result {
                    op: OpId(oi as u32),
                    index: i as u32,
                })
            {
                a.fail(format!("value {}: def does not name op {oi} result {i}", v.0));
            }
        }
        if op.callee.is_some_and(|f| f.index() >= m.funcs.len()) {
            a.fail(format!("op {oi}: callee out of range"));
        }
    }
    for (i, d) in defs.iter().enumerate() {
        if *d != 1 {
            a.fail(format!("value {i} is defined {d} times"));
        }
    }
    if !a.notes.is_empty() {
        return;
    }
    // Dominance: walk every function body with the set of visible values.
    let mut placed = vec![0u32; m.ops.len()];
    fn walk(m: &Module, r: RegionId, visible: &mut HashSet<ValueId>, placed: &mut [u32], a: &mut Acc) {
        let reg = m.region(r);
        let mut added = Vec::new();
        for &v in &reg.args {
            visible.insert(v);
            added.push(v);
        }
        for (k, &o) in reg.ops.iter().enumerate() {
            let op = m.op(o);
            placed[o.index()] += 1;
            if op.parent != r {
                a.fail(format!(
                    "op {} is listed in region {} but its parent is {}",
                    o.0, r.0, op.parent.0
                ));
            }
            for (p, v) in op.operands.iter().enumerate() {
                if !visible.contains(v) {
                    a.fail(format!(
                        "op {} `{}` port {p}: value {} is not dominated by its definition",
                        o.0, op.name, v.0
                    ));
                }
            }
            for &rr in &op.regions {
                if m.region(rr).owner != Owner::Op(o) {
                    a.fail(format!("region {} is not owned by op {}", rr.0, o.0));
                }
                walk(m, rr, visible, placed, a);
            }
            let term = matches!(op.name.as_str(), "func.return" | "stablehlo.return");
            if term != (k + 1 == reg.ops.len()) {
                a.fail(format!("region {}: the terminator is not the last op", r.0));
            }
            for &v in &op.results {
                visible.insert(v);
                added.push(v);
            }
        }
        for v in added {
            visible.remove(&v);
        }
    }
    for (fi, f) in m.funcs.iter().enumerate() {
        if m.region(f.body).owner != Owner::Func(cc_ir::FuncId(fi as u32)) {
            a.fail(format!("function {} body is not owned by it", f.name));
        }
        walk(m, f.body, &mut HashSet::new(), &mut placed, a);
    }
    for (i, n) in placed.iter().enumerate() {
        if *n != 1 {
            a.fail(format!("op {i} appears {n} times in the regions"));
        }
    }
    a.count("values", nv as u64);
    a.count("ops", m.ops.len() as u64);
}

// ---------------------------------------------------------------- types

fn tensor(m: &Module, v: ValueId) -> Option<&TensorType> {
    m.value(v).ty.tensor()
}

fn component(e: &Elem) -> Option<Elem> {
    match e {
        Elem::Complex(c) => Some((**c).clone()),
        _ => None,
    }
}

/// Declared types of `graph.json` values, by value id.
fn declared_types(g: &GraphFile) -> HashMap<u64, &ej::Type> {
    fn region<'a>(r: &'a ej::Region, t: &mut HashMap<u64, &'a ej::Type>) {
        for b in &r.blocks {
            for a in &b.args {
                t.insert(a.v, &a.ty);
            }
            for op in &b.ops {
                for res in &op.results {
                    t.insert(res.v, &res.ty);
                }
                for rr in &op.regions {
                    region(rr, t);
                }
            }
        }
    }
    let mut t = HashMap::new();
    for f in &g.module.funcs {
        region(&f.body, &mut t);
    }
    t
}

fn check_types(p: &Phase, ctx: &SymCtx, a: &mut Acc) {
    let m = &p.m;
    let decl = declared_types(&p.graph);
    for (vi, v) in m.values.iter().enumerate() {
        let Some(t) = v.ty.tensor() else {
            a.fail(format!("value {vi} is not a tensor"));
            continue;
        };
        let Some(ej::Type::Tensor { shape, elem }) = decl.get(&v.src) else {
            a.fail(format!(
                "value {vi}: source %{} has no tensor type in graph.json",
                v.src
            ));
            continue;
        };
        if Elem::parse(elem).as_ref() != Some(&t.elem) || shape.len() != t.rank() {
            a.fail(format!("value {vi}: {t} vs declared %{}", v.src));
            continue;
        }
        for (d, s) in t.shape.iter().zip(shape) {
            if let Some(c) = s
                && d.as_const() != Some(*c)
            {
                a.fail(format!("value {vi}: dim {d} where graph.json declares {c}"));
            }
        }
    }
    for (oi, op) in m.ops.iter().enumerate() {
        let Some(spec) = cc_ir::lookup(&op.name, op.target.as_deref()) else {
            continue; // coverage
        };
        let ins: Option<Vec<&TensorType>> = op.operands.iter().map(|&v| tensor(m, v)).collect();
        let outs: Option<Vec<&TensorType>> = op.results.iter().map(|&v| tensor(m, v)).collect();
        let (Some(ins), Some(outs)) = (ins, outs) else {
            continue;
        };
        let want = match spec.elem {
            ElemRule::SameAs(i) => ins.get(i as usize).map(|t| t.elem.clone()),
            ElemRule::Bool => Some(Elem::Bool),
            ElemRule::I32 => Some(Elem::Int {
                signed: true,
                width: 32,
            }),
            ElemRule::ComponentOf0 => ins.first().and_then(|t| component(&t.elem)),
            ElemRule::ComplexOf0 => ins.first().map(|t| Elem::Complex(Box::new(t.elem.clone()))),
            ElemRule::Free => None,
        };
        if let Some(w) = want
            && outs.iter().any(|t| t.elem != w)
        {
            a.fail(format!("op {oi} `{}`: result element type is not {w}", op.name));
        }
        // Shapes that follow from operand types alone.
        let value_dependent = !matches!(spec.shape_operands, ShapePorts::None | ShapePorts::DimVarArgs);
        if value_dependent {
            a.count("value_dependent_shapes", 1);
            continue;
        }
        let callee: Option<Vec<TensorType>> = op.callee.map(|f| {
            m.func(f)
                .results
                .iter()
                .filter_map(|t| t.tensor().cloned())
                .collect()
        });
        let declared: Vec<Vec<Option<u64>>> = outs
            .iter()
            .map(|t| t.shape.iter().map(Dim::as_const).collect())
            .collect();
        let vals = vec![None; ins.len()];
        match (spec.shape)(&ShapeIn {
            operands: &ins,
            attrs: &op.attrs,
            values: &vals,
            declared: &declared,
            callee: callee.as_deref(),
            ctx,
        }) {
            Err(e) => a.fail(format!("op {oi} `{}`: {e}", op.name)),
            Ok(shapes) => {
                if shapes.len() != outs.len() {
                    a.fail(format!("op {oi} `{}`: result count", op.name));
                }
                for (s, t) in shapes.iter().zip(&outs) {
                    let same = s.len() == t.rank()
                        && s.iter()
                            .zip(&t.shape)
                            .all(|(x, y)| x == y || ctx.eq(&x.expr(), &y.expr()) == Some(true));
                    if !same {
                        a.fail(format!("op {oi} `{}`: stored {t}, rule gives {s:?}", op.name));
                    }
                }
            }
        }
        a.count("rechecked_shapes", 1);
    }
}

// ---------------------------------------------------------------- sym_eval

/// Sample assignments of the symbols that satisfy the constraints: every symbol at a base size, then
/// repaired constraint by constraint (deterministic).
pub(crate) fn sample_points(m: &Module, tiny: u64) -> Vec<BTreeMap<String, i64>> {
    let mut out: Vec<BTreeMap<String, i64>> = Vec::new();
    for base in [tiny as i64, tiny as i64 + 1, 2 * tiny as i64 + 1] {
        let mut env: BTreeMap<String, i64> = m.symbols.iter().map(|s| (s.clone(), base.max(1))).collect();
        for _ in 0..64 {
            let Some(c) = m.constraints.iter().find(|c| c.holds(&env) != Ok(true)) else {
                break;
            };
            // Set a lone variable side to the other side's value, else grow the first variable.
            let (lv, rv) = (c.lhs.as_var(), c.rhs.as_var());
            let (l, r) = (c.lhs.eval(&env).unwrap_or(0), c.rhs.eval(&env).unwrap_or(0));
            match (lv, rv) {
                (Some(x), _) if r >= 1 => {
                    env.insert(x.to_string(), r);
                }
                (_, Some(y)) if l >= 1 => {
                    env.insert(y.to_string(), l);
                }
                _ => {
                    let vars = c.lhs.vars().into_iter().chain(c.rhs.vars()).collect::<Vec<_>>();
                    if let Some(x) = vars.first() {
                        *env.entry(x.clone()).or_insert(1) += 1;
                    }
                }
            }
        }
        if m.constraints.iter().all(|c| c.holds(&env) == Ok(true)) && !out.contains(&env) {
            out.push(env);
        }
    }
    out
}

fn concrete_dims(t: &TensorType, env: &BTreeMap<String, i64>) -> Option<Vec<i64>> {
    t.shape.iter().map(|d| d.expr().eval(env).ok()).collect()
}

/// Concrete integer values (row-major) of small integer/boolean tensors.
type Cv = Option<Vec<i64>>;

fn rowmajor(ix: &[i64], shape: &[i64]) -> i64 {
    ix.iter().zip(shape).fold(0, |acc, (i, n)| acc * n + i)
}

fn index_of(mut i: i64, shape: &[i64]) -> Vec<i64> {
    let mut ix = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        let n = shape[d].max(1);
        ix[d] = i % n;
        i /= n;
    }
    ix
}

struct Interp<'a> {
    m: &'a Module,
    env: &'a BTreeMap<String, i64>,
    vals: Vec<Cv>,
}

impl Interp<'_> {
    fn data_at_zero(&self, v: ValueId, counts: &mut BTreeMap<String, u64>) -> Cv {
        let t = tensor(self.m, v)?;
        if !matches!(t.elem, Elem::Int { .. } | Elem::Bool) {
            return None;
        }
        let n: i64 = concrete_dims(t, self.env)?.iter().product();
        (0..=64).contains(&n).then(|| {
            *counts.entry("data_values_at_zero".into()).or_default() += 1;
            vec![0; n as usize]
        })
    }

    fn dims(&self, v: ValueId) -> Option<Vec<i64>> {
        concrete_dims(tensor(self.m, v)?, self.env)
    }

    fn eval_op(&self, op: &Op) -> Cv {
        let res = *op.results.first()?;
        let rt = tensor(self.m, res)?;
        if !matches!(rt.elem, Elem::Int { .. } | Elem::Bool) {
            return None;
        }
        let rs = concrete_dims(rt, self.env)?;
        let n: i64 = rs.iter().product();
        if !(0..=64).contains(&n) {
            return None;
        }
        let get = |p: usize| -> Option<&Vec<i64>> { self.vals[op.operands.get(p)?.index()].as_ref() };
        let elem = |p: usize, i: i64| -> Option<i64> {
            let v = get(p)?;
            if v.len() == 1 {
                v.first().copied()
            } else {
                v.get(i as usize).copied()
            }
        };
        let map = |f: &dyn Fn(i64) -> Option<i64>| -> Cv { (0..n).map(f).collect() };
        let is_bool = rt.elem == Elem::Bool;
        match op.name.as_str() {
            "stablehlo.constant" => {
                let Some(Attr::Dense {
                    splat,
                    values: Some(vs),
                    ..
                }) = op.attrs.get("value")
                else {
                    return None;
                };
                let xs: Option<Vec<i64>> = vs
                    .iter()
                    .map(|s| match s {
                        Scalar::Int(v) => i64::try_from(*v).ok(),
                        Scalar::Bool(b) => Some(i64::from(*b)),
                        _ => None,
                    })
                    .collect();
                let xs = xs?;
                if *splat {
                    Some(vec![*xs.first()?; n as usize])
                } else {
                    Some(xs)
                }
            }
            "stablehlo.get_dimension_size" => {
                let d = usize::try_from(op.attrs.int("dimension")?).ok()?;
                Some(vec![*self.dims(op.operands[0])?.get(d)?])
            }
            "stablehlo.iota" => {
                let d = usize::try_from(op.attrs.int("iota_dimension")?).ok()?;
                map(&|i| index_of(i, &rs).get(d).copied())
            }
            "stablehlo.convert" => {
                let from = &tensor(self.m, op.operands[0])?.elem;
                if !matches!(from, Elem::Int { .. } | Elem::Bool) {
                    return None;
                }
                map(&|i| elem(0, i).map(|x| if is_bool { i64::from(x != 0) } else { x }))
            }
            "stablehlo.reshape" => get(0).cloned(),
            "stablehlo.broadcast_in_dim" => {
                let os = self.dims(op.operands[0])?;
                let bd = op.attrs.ints("broadcast_dimensions")?;
                map(&|i| {
                    let ri = index_of(i, &rs);
                    let oi: Vec<i64> = bd
                        .iter()
                        .enumerate()
                        .map(|(d, &r)| if os[d] == 1 { 0 } else { ri[r as usize] })
                        .collect();
                    get(0)?.get(rowmajor(&oi, &os) as usize).copied()
                })
            }
            "stablehlo.concatenate" => {
                let d = usize::try_from(op.attrs.int("dimension")?).ok()?;
                let shapes: Option<Vec<Vec<i64>>> = op.operands.iter().map(|&v| self.dims(v)).collect();
                let shapes = shapes?;
                map(&|i| {
                    let mut ix = index_of(i, &rs);
                    let mut k = 0;
                    while ix[d] >= shapes.get(k)?[d] {
                        ix[d] -= shapes[k][d];
                        k += 1;
                    }
                    get(k)?.get(rowmajor(&ix, &shapes[k]) as usize).copied()
                })
            }
            "stablehlo.slice" => {
                let os = self.dims(op.operands[0])?;
                let st = op.attrs.ints("start_indices")?;
                let sd = op.attrs.ints("strides").unwrap_or_else(|| vec![1; os.len()]);
                map(&|i| {
                    let oi: Vec<i64> = index_of(i, &rs)
                        .iter()
                        .enumerate()
                        .map(|(d, x)| st[d] + x * sd[d])
                        .collect();
                    get(0)?.get(rowmajor(&oi, &os) as usize).copied()
                })
            }
            "stablehlo.transpose" => {
                let os = self.dims(op.operands[0])?;
                let perm = op.attrs.ints("permutation")?;
                map(&|i| {
                    let ri = index_of(i, &rs);
                    let mut oi = vec![0; os.len()];
                    for (r, &s) in perm.iter().enumerate() {
                        oi[s as usize] = ri[r];
                    }
                    get(0)?.get(rowmajor(&oi, &os) as usize).copied()
                })
            }
            "stablehlo.add" => map(&|i| elem(0, i)?.checked_add(elem(1, i)?)),
            "stablehlo.subtract" => map(&|i| elem(0, i)?.checked_sub(elem(1, i)?)),
            "stablehlo.multiply" => map(&|i| elem(0, i)?.checked_mul(elem(1, i)?)),
            // Spec `#divide`, `#remainder`: truncation toward zero.
            "stablehlo.divide" => map(&|i| elem(0, i)?.checked_div(elem(1, i)?)),
            "stablehlo.remainder" => map(&|i| elem(0, i)?.checked_rem(elem(1, i)?)),
            "stablehlo.maximum" => map(&|i| Some(elem(0, i)?.max(elem(1, i)?))),
            "stablehlo.minimum" => map(&|i| Some(elem(0, i)?.min(elem(1, i)?))),
            "stablehlo.negate" => map(&|i| elem(0, i)?.checked_neg()),
            "stablehlo.abs" => map(&|i| elem(0, i)?.checked_abs()),
            "stablehlo.sign" => map(&|i| Some(elem(0, i)?.signum())),
            "stablehlo.clamp" => map(&|i| Some(elem(1, i)?.max(elem(0, i)?).min(elem(2, i)?))),
            "stablehlo.select" => map(&|i| if elem(0, i)? != 0 { elem(1, i) } else { elem(2, i) }),
            "stablehlo.and" => map(&|i| Some(elem(0, i)? & elem(1, i)?)),
            "stablehlo.or" => map(&|i| Some(elem(0, i)? | elem(1, i)?)),
            "stablehlo.xor" => map(&|i| Some(elem(0, i)? ^ elem(1, i)?)),
            "stablehlo.not" => map(&|i| {
                Some(if is_bool {
                    i64::from(elem(0, i)? == 0)
                } else {
                    !elem(0, i)?
                })
            }),
            "stablehlo.compare" => {
                let dir = op.attrs.enum_value("comparison_direction")?.to_string();
                map(&|i| {
                    let (x, y) = (elem(0, i)?, elem(1, i)?);
                    let r = match dir.as_str() {
                        "EQ" => x == y,
                        "NE" => x != y,
                        "GE" => x >= y,
                        "GT" => x > y,
                        "LE" => x <= y,
                        "LT" => x < y,
                        _ => return None,
                    };
                    Some(i64::from(r))
                })
            }
            _ => None,
        }
    }
}

fn check_sym_eval(p: &Phase, tiny: u64, a: &mut Acc) {
    let m = &p.m;
    let points = sample_points(m, tiny);
    if points.is_empty() && !m.symbols.is_empty() {
        a.fail("no sample sizes satisfy the constraints");
        return;
    }
    let points = if m.symbols.is_empty() {
        vec![BTreeMap::new()]
    } else {
        points
    };
    a.count("points", points.len() as u64);
    let shape_ctx = SymCtx::default();
    for env in &points {
        let mut it = Interp {
            m,
            env,
            vals: vec![None; m.values.len()],
        };
        for (i, v) in m.values.iter().enumerate() {
            if let Some(s) = &v.dim_var {
                it.vals[i] = env.get(s).map(|x| vec![*x]);
            } else if let Def::Arg { region, .. } = v.def
                && matches!(m.region(region).owner, Owner::Func(f) if m.func(f).kind != FuncKind::Entry)
            {
                it.vals[i] = it.data_at_zero(ValueId(i as u32), &mut a.counts);
            }
        }
        for (oi, op) in m.ops.iter().enumerate() {
            let Some(spec) = cc_ir::lookup(&op.name, op.target.as_deref()) else {
                continue;
            };
            // Concrete operand types and values.
            let ins: Option<Vec<TensorType>> = op
                .operands
                .iter()
                .map(|&v| {
                    let t = tensor(m, v)?;
                    let d = concrete_dims(t, env)?;
                    Some(TensorType {
                        shape: d.into_iter().map(|x| Dim::Const(x.max(0) as u64)).collect(),
                        elem: t.elem.clone(),
                    })
                })
                .collect();
            let Some(ins) = ins else {
                a.fail(format!("op {oi} `{}`: an operand dim does not evaluate", op.name));
                continue;
            };
            let vals: Vec<Option<Vec<SymExpr>>> = op
                .operands
                .iter()
                .map(|&v| {
                    it.vals[v.index()]
                        .as_ref()
                        .map(|xs| xs.iter().map(|&x| SymExpr::constant(x)).collect())
                })
                .collect();
            // Shape operands must be computable.
            for (pi, v) in vals.iter().enumerate() {
                let shape_port = match spec.shape_operands {
                    ShapePorts::DimVarArgs => op.callee.is_some_and(|f| {
                        m.region(m.func(f).body)
                            .args
                            .get(pi)
                            .is_some_and(|&x| m.value(x).dim_var.is_some())
                    }),
                    sp => sp.contains(pi, op.operands.len()),
                };
                if shape_port && v.is_none() {
                    a.fail(format!(
                        "op {oi} `{}` port {pi}: shape operand not computable at {env:?}",
                        op.name
                    ));
                }
            }
            if op.target.as_deref() == Some("shape_assertion")
                && it.vals[op.operands[0].index()].as_deref() != Some(&[1])
            {
                a.fail(format!("op {oi}: shape assertion does not hold at {env:?}"));
            }
            if op.name == "stablehlo.composite"
                && let Some(f) = op.callee
            {
                for (pi, &x) in m.region(m.func(f).body).args.iter().enumerate() {
                    if let Some(s) = &m.value(x).dim_var {
                        let got = it.vals[op.operands[pi].index()].as_deref();
                        if got != env.get(s).map(std::slice::from_ref) {
                            a.fail(format!(
                                "op {oi}: dimension argument `{s}` bound to {got:?} at {env:?}"
                            ));
                        }
                    }
                }
            }
            let outs: Option<Vec<Vec<i64>>> = op
                .results
                .iter()
                .map(|&v| concrete_dims(tensor(m, v)?, env))
                .collect();
            let Some(outs) = outs else {
                a.fail(format!("op {oi} `{}`: a result dim does not evaluate", op.name));
                continue;
            };
            if outs.iter().flatten().any(|&d| d < 0) {
                a.fail(format!("op {oi} `{}`: negative dim at {env:?}", op.name));
            }
            let callee: Option<Vec<TensorType>> = op.callee.map(|f| {
                m.func(f)
                    .results
                    .iter()
                    .filter_map(|t| {
                        let t = t.tensor()?;
                        Some(TensorType {
                            shape: concrete_dims(t, env)?
                                .into_iter()
                                .map(|x| Dim::Const(x.max(0) as u64))
                                .collect(),
                            elem: t.elem.clone(),
                        })
                    })
                    .collect()
            });
            let refs: Vec<&TensorType> = ins.iter().collect();
            let declared: Vec<Vec<Option<u64>>> = outs
                .iter()
                .map(|d| d.iter().map(|&x| Some(x.max(0) as u64)).collect())
                .collect();
            match (spec.shape)(&ShapeIn {
                operands: &refs,
                attrs: &op.attrs,
                values: &vals,
                declared: &declared,
                callee: callee.as_deref(),
                ctx: &shape_ctx,
            }) {
                Ok(shapes) => {
                    let got: Vec<Vec<i64>> = shapes
                        .iter()
                        .map(|s| s.iter().map(|d| d.as_const().map_or(-1, |c| c as i64)).collect())
                        .collect();
                    if got != outs {
                        a.fail(format!(
                            "op {oi} `{}`: at {env:?} the rule gives {got:?}, the module {outs:?}",
                            op.name
                        ));
                    }
                }
                Err(e) => a.fail(format!("op {oi} `{}`: at {env:?}: {e}", op.name)),
            }
            let r = it.eval_op(op);
            if let Some(&v) = op.results.first() {
                // Integers computed from run-time data are taken at 0: a derived dim never depends on
                // them (the builder rejects that), so any value must reproduce the stored dims.
                it.vals[v.index()] = r.or_else(|| it.data_at_zero(v, &mut a.counts));
            }
        }
    }
}

// ---------------------------------------------------------------- census

/// Op names of the expansion of `graph.json` from the entry: `(all, kept)` where `kept` leaves out what
/// inlining removes (calls, inlined composites, the returns of inlined bodies), and the ledger.
fn graph_expansion(
    g: &GraphFile,
    cfg: &ImportConfig,
) -> (BTreeMap<String, u64>, BTreeMap<String, u64>, Ledger) {
    let funcs: HashMap<&str, &ej::Func> = g.module.funcs.iter().map(|f| (f.name.as_str(), f)).collect();
    let mut all = BTreeMap::new();
    let mut kept = BTreeMap::new();
    let mut ledger = Ledger::default();
    let mut stack: Vec<(&ej::Region, bool)> = funcs
        .get(g.entry.as_str())
        .map(|f| (&f.body, false))
        .into_iter()
        .collect();
    while let Some((r, inlined)) = stack.pop() {
        for b in &r.blocks {
            let mut ops: Vec<(&ej::Op, bool)> = b.ops.iter().map(|o| (o, true)).collect();
            while let Some((op, top)) = ops.pop() {
                *all.entry(op.name.clone()).or_insert(0u64) += 1;
                for rr in &op.regions {
                    for bb in &rr.blocks {
                        ops.extend(bb.ops.iter().map(|o| (o, false)));
                    }
                }
                let name = op.attributes.get("name").and_then(ej::Attr::as_str);
                let (callee, removed) = match op.name.as_str() {
                    "func.call" => (op.attributes.get("callee").and_then(ej::Attr::as_str), true),
                    "stablehlo.composite" => (
                        op.attributes.get("decomposition").and_then(ej::Attr::as_str),
                        name.is_some_and(|n| cfg.inlines(n)),
                    ),
                    _ => (None, false),
                };
                let is_inlined_return = top && inlined && op.name == "func.return";
                if !removed && !is_inlined_return {
                    *kept.entry(op.name.clone()).or_insert(0u64) += 1;
                }
                if is_inlined_return {
                    ledger.returns += 1;
                }
                if removed {
                    if op.name == "func.call" {
                        ledger.calls += 1;
                    } else if let Some(n) = name {
                        *ledger.composites.entry(n.to_string()).or_default() += 1;
                    }
                }
                if let Some(c) = callee.and_then(|c| funcs.get(c)) {
                    stack.push((&c.body, removed));
                }
            }
        }
    }
    (all, kept, ledger)
}

fn check_census(p: &Phase, cfg: &ImportConfig, a: &mut Acc) {
    let (all, kept, ledger) = graph_expansion(&p.graph, cfg);
    if all != p.census.executed {
        a.fail("the expansion of graph.json differs from census.json `executed`");
    }
    let got = p.m.expanded_multiset();
    let names: std::collections::BTreeSet<&String> = got.keys().chain(kept.keys()).collect();
    for n in names {
        let (x, y) = (
            got.get(n).copied().unwrap_or(0),
            kept.get(n).copied().unwrap_or(0),
        );
        if x != y {
            a.fail(format!("`{n}`: module {x}, export census minus inlining {y}"));
        }
    }
    if ledger != p.summary.ledger {
        a.fail(format!(
            "inlining ledger {:?}, recounted {ledger:?}",
            p.summary.ledger
        ));
    }
    a.count("calls_inlined", ledger.calls);
    // GEMM / convolution signatures with the declared shapes, composites expanded.
    let decl = declared_types(&p.graph);
    let shape_of = |v: ValueId| -> Option<(Vec<Option<u64>>, String)> {
        match decl.get(&p.m.value(v).src)? {
            ej::Type::Tensor { shape, elem } => Some((shape.clone(), elem.clone())),
            _ => None,
        }
    };
    let mut ours: Vec<String> = Vec::new();
    let mut stack = vec![p.m.func(p.m.entry).body];
    while let Some(r) = stack.pop() {
        for &o in &p.m.region(r).ops {
            let op = p.m.op(o);
            stack.extend(op.regions.iter().copied());
            if op.name == "stablehlo.composite"
                && let Some(f) = op.callee
            {
                stack.push(p.m.func(f).body);
            }
            let kind = match op.name.as_str() {
                "stablehlo.dot_general" => "dot_general",
                "stablehlo.convolution" | "stablehlo.dynamic_conv" => "convolution",
                _ => continue,
            };
            let (Some(l), Some(rr), Some(out)) = (
                shape_of(op.operands[0]),
                shape_of(op.operands[1]),
                shape_of(op.results[0]),
            ) else {
                a.fail(format!("op {} has no declared operand types", o.0));
                continue;
            };
            ours.push(format!(
                "{kind} {}x{}->{} {:?} {:?} {:?}",
                l.1, rr.1, out.1, l.0, rr.0, out.0
            ));
        }
    }
    let mut theirs: Vec<String> = p
        .census
        .ops
        .iter()
        .map(|c| {
            format!(
                "{} {} {:?} {:?} {:?}",
                c.kind, c.sig, c.lhs_shape, c.rhs_shape, c.out_shape
            )
        })
        .collect();
    ours.sort();
    theirs.sort();
    if ours != theirs {
        a.fail(format!(
            "GEMM/conv census: {} in the module, {} in census.json, or their signatures differ",
            ours.len(),
            theirs.len()
        ));
    }
    a.count("cores", ours.len() as u64);
}

// ---------------------------------------------------------------- coverage

fn check_coverage(m: &Module, a: &mut Acc) {
    for (oi, op) in m.ops.iter().enumerate() {
        match cc_ir::lookup(&op.name, op.target.as_deref()) {
            None => a.fail(format!(
                "op {oi} `{}` ({:?}) has no table entry",
                op.name, op.target
            )),
            Some(s) => {
                if !s.operands.admits(op.operands.len()) || !s.results.admits(op.results.len()) {
                    a.fail(format!("op {oi} `{}`: arity outside the table entry", op.name));
                }
                if op.name == "func.call" {
                    a.fail(format!("op {oi}: a call survived inlining"));
                }
            }
        }
    }
    for f in &m.funcs {
        if f.kind == FuncKind::Decomposition
            && !m.ops.iter().any(|o| {
                o.name == "stablehlo.composite" && o.callee.is_some_and(|c| m.func(c).name == f.name)
            })
        {
            a.fail(format!("decomposition `{}` has no composite", f.name));
        }
    }
    a.count("ops", m.ops.len() as u64);
}

// ---------------------------------------------------------------- size_tag

fn check_size_tag(m: &Module, a: &mut Acc) {
    let users = m.users();
    let shape_port = |op: &Op, p: usize| -> bool {
        match cc_ir::lookup(&op.name, op.target.as_deref()).map(|s| s.shape_operands) {
            Some(ShapePorts::DimVarArgs) => op.callee.is_some_and(|f| {
                m.region(m.func(f).body)
                    .args
                    .get(p)
                    .is_some_and(|&x| m.value(x).dim_var.is_some())
            }),
            Some(sp) => sp.contains(p, op.operands.len()),
            None => false,
        }
    };
    let size_class =
        |op: &Op| cc_ir::lookup(&op.name, op.target.as_deref()).is_some_and(|s| s.class == OpClass::Size);
    let small = |op: &Op| {
        !op.results.is_empty()
            && op.results.iter().all(|&r| {
                m.value(r).ty.tensor().is_some_and(|t| {
                    matches!(t.elem, Elem::Int { .. } | Elem::Bool)
                        && t.num_elements().is_some_and(|n| n <= 64)
                })
            })
    };
    // Candidates: a work list forward from the dimension sources through small integer ops.
    let mut cand = vec![false; m.ops.len()];
    let mut work: Vec<ValueId> = Vec::new();
    for (i, v) in m.values.iter().enumerate() {
        if v.dim_var.is_some() {
            work.push(ValueId(i as u32));
        }
    }
    for (i, op) in m.ops.iter().enumerate() {
        if size_class(op) || op.name == "stablehlo.get_dimension_size" {
            cand[i] = true;
            work.extend(op.results.iter().copied());
        }
    }
    while let Some(v) = work.pop() {
        for &(o, _) in &users[v.index()] {
            if !cand[o.index()] && small(m.op(o)) {
                cand[o.index()] = true;
                work.extend(m.op(o).results.iter().copied());
            }
        }
    }
    // Greatest fixpoint: drop candidates with a use outside size computation and shape ports.
    let mut want = cand;
    loop {
        let mut changed = false;
        for (i, op) in m.ops.iter().enumerate() {
            if !want[i] || size_class(op) {
                continue;
            }
            let escapes = op.results.iter().any(|r| {
                users[r.index()]
                    .iter()
                    .any(|&(u, p)| !want[u.index()] && !shape_port(m.op(u), p as usize))
            });
            if escapes {
                want[i] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut tagged = 0u64;
    let mut crossings = 0u64;
    for (oi, op) in m.ops.iter().enumerate() {
        if op.size != want[oi] {
            a.fail(format!(
                "op {oi} `{}`: size tag {}, recomputed {}",
                op.name, op.size, want[oi]
            ));
        }
        tagged += u64::from(want[oi]);
        let cross: Vec<u32> = if want[oi] {
            Vec::new()
        } else {
            op.operands
                .iter()
                .enumerate()
                .filter(|&(p, &v)| {
                    let val = m.value(v);
                    let size_value = val.dim_var.is_some()
                        || matches!(val.def, Def::Result { op, .. } if want[op.index()]);
                    size_value && !shape_port(op, p)
                })
                .map(|(p, _)| p as u32)
                .collect()
        };
        if cross != op.dim_as_data {
            a.fail(format!(
                "op {oi} `{}`: dim_as_data {:?}, recomputed {cross:?}",
                op.name, op.dim_as_data
            ));
        }
        crossings += cross.len() as u64;
    }
    a.count("size_ops", tagged);
    a.count("dim_as_data", crossings);
}

// ---------------------------------------------------------------- inventory

fn check_inventory(ctx: &PassContext, units: &[(String, String)]) -> Result<Check, PassError> {
    let mut a = Acc::new("inventory");
    let out = ctx.output(IMPORT)?;
    let inv: Inventory = match out.read_json(INVENTORY_FILE) {
        Ok(i) => i,
        Err(e) => {
            a.fail(format!("cannot read {INVENTORY_FILE}: {e}"));
            return Ok(a.done());
        }
    };
    for u in &inv.unknown {
        a.fail(format!("`{u}` has no table entry"));
    }
    for (k, row) in &inv.ops {
        let (name, target) = match k.split_once('@') {
            Some((n, t)) => (n, Some(t)),
            None => (k.as_str(), None),
        };
        if row.entry.is_some() != cc_ir::lookup(name, target).is_some() {
            a.fail(format!("`{k}`: entry presence differs from the table"));
        }
    }
    let input = ctx.input(EXPORT)?;
    for (m, p) in units {
        let g: GraphFile = match input.read_json(&format!("{m}/{p}/graph.json")) {
            Ok(g) => g,
            Err(e) => {
                a.fail(format!("{m}/{p}: {e}"));
                continue;
            }
        };
        let mut missing = Vec::new();
        let mut stack: Vec<&ej::Region> = g.module.funcs.iter().map(|f| &f.body).collect();
        while let Some(r) = stack.pop() {
            for b in &r.blocks {
                for op in &b.ops {
                    let k = op_key(op);
                    if !inv.ops.get(&k).is_some_and(|row| row.models.contains(m)) {
                        missing.push(k);
                    }
                    stack.extend(op.regions.iter());
                }
            }
        }
        missing.sort();
        missing.dedup();
        for k in missing {
            a.fail(format!("{m}/{p}: `{k}` is not in the inventory"));
        }
    }
    a.count("ops", inv.ops.len() as u64);
    Ok(a.done())
}
