//! Configuration proofs (docs/passes/merge.md §6), one SMT-LIB query each:
//!
//! - `<F>/<unit>/<t>.smt2`: the unit `F` with the stored configuration differs from the datapath of
//!   `unit` at tuple `t` on some output (or accumulator next/init value). The unit side follows only
//!   the unit's wiring and the configuration: each multiplexer takes its selected source, each block
//!   computes at its own types (extending narrower operands, truncating the result back in `widen`
//!   mode). The datapath side is `datapaths::eval::values`; inputs, constants and accumulators are the
//!   same variables on both sides. `unsat` proves the configuration correct for every input and every
//!   constant value;
//! - `arrays/<A>/<z>/<t>.smt2`: one step of the array's processing element in the class's mode
//!   (native, widened operands, or a composed operand: one product per piece, shifted and summed)
//!   differs from the class's core step `acc + convert(lhs) · convert(rhs)` at the accumulator type.

use std::collections::{BTreeMap, HashMap};

use cc_smt::Script;
use cc_smt::ops::{self, Attrs, Ty, Val};
use cc_smt::term::{Ctx, Op};

use crate::datapaths::eval;
use crate::datapaths::json::{Datapath, Tuple};
use crate::merge::json::{ArrayFile, ArrayMode, ArrayTuple, BlockKind, ConfigFile, FuFile, Mode};

type R<X> = Result<X, String>;

/// The scalar type of a type name as `Ty::name` writes it (`i8`, `ui16`, `i1`, `bf16`, `f8e4m3fn`, `cf32`).
pub fn ty(name: &str) -> R<Ty> {
    use cc_smt::float::Format;
    let float = |n: &str| -> Option<Format> {
        Some(match n {
            "f16" => Format::F16,
            "bf16" => Format::BF16,
            "f32" => Format::F32,
            "f64" => Format::F64,
            "f8e4m3fn" => Format::E4M3FN,
            "f8e5m2" => Format::E5M2,
            _ => return None,
        })
    };
    if name == "i1" {
        return Ok(Ty::Bool);
    }
    if let Some(f) = float(name) {
        return Ok(Ty::Float(f));
    }
    if let Some(f) = name.strip_prefix('c').and_then(float) {
        return Ok(Ty::Complex(f));
    }
    let int = |w: &str, signed: bool| w.parse::<u32>().ok().map(|w| Ty::Int { signed, w });
    name.strip_prefix("ui")
        .and_then(|w| int(w, false))
        .or_else(|| name.strip_prefix('i').and_then(|w| int(w, true)))
        .ok_or_else(|| format!("element type {name}"))
}

/// One query (`file` relative to `proofs/`).
#[derive(Debug, Clone)]
pub struct Query {
    pub file: String,
    pub text: String,
}

fn smt(e: cc_smt::term::SortError) -> String {
    e.0
}

/// The values of `fu` under `cfg`: every output (by name), `next<i>`/`init<i>` of every accumulator.
pub fn fu_values(c: &mut Ctx, fu: &FuFile, cfg: &ConfigFile) -> R<BTreeMap<String, Val>> {
    let bind: HashMap<u32, &crate::merge::json::Binding> =
        cfg.bindings.iter().map(|b| (b.block, b)).collect();
    let sel: HashMap<(u32, u32), u32> = cfg
        .selects
        .iter()
        .map(|s| ((s.block, s.port), s.select))
        .collect();
    let mut memo: HashMap<(u32, u32), Val> = HashMap::new();

    fn port_src(
        fu: &FuFile,
        sel: &HashMap<(u32, u32), u32>,
        b: u32,
        p: u32,
    ) -> R<crate::merge::json::Source> {
        let blk = fu.blocks.get(b as usize).ok_or_else(|| format!("block {b}"))?;
        let srcs = blk
            .sources
            .get(p as usize)
            .ok_or_else(|| format!("block {b} port {p}: no sources"))?;
        let k = match srcs.len() {
            0 => return Err(format!("block {b} port {p}: no sources")),
            1 => 0,
            _ => *sel
                .get(&(b, p))
                .ok_or_else(|| format!("block {b} port {p}: a multiplexer without a select"))?
                as usize,
        };
        srcs.get(k)
            .copied()
            .ok_or_else(|| format!("block {b} port {p}: select {k} out of range"))
    }

    #[allow(clippy::too_many_arguments)]
    fn val(
        c: &mut Ctx,
        fu: &FuFile,
        bind: &HashMap<u32, &crate::merge::json::Binding>,
        sel: &HashMap<(u32, u32), u32>,
        memo: &mut HashMap<(u32, u32), Val>,
        b: u32,
        res: u32,
        depth: usize,
    ) -> R<Val> {
        if let Some(v) = memo.get(&(b, res)) {
            return Ok(*v);
        }
        if depth > 100_000 {
            return Err("the configuration has a cycle".into());
        }
        let blk = fu.blocks.get(b as usize).ok_or_else(|| format!("block {b}"))?;
        let bd = bind
            .get(&b)
            .ok_or_else(|| format!("block {b} is read but not configured"))?;
        let tys: Vec<String> = bd
            .types
            .iter()
            .filter(|t| !t.starts_with("rp:"))
            .cloned()
            .collect();
        let out_ty = ty(tys.last().ok_or("a binding without types")?)?;
        let v = match blk.kind {
            BlockKind::Input => {
                ops::input(c, bd.var.as_deref().ok_or("an input without a variable")?, out_ty).map_err(smt)?
            }
            BlockKind::Const => match (&bd.var, blk.attrs.get("value").map(String::as_str)) {
                (Some(col), _) => ops::input(c, col, out_ty).map_err(smt)?,
                (None, _) => ops::zero(c, out_ty).map_err(smt)?,
            },
            BlockKind::Acc => {
                let var = bd.var.as_deref().ok_or("an accumulator without a variable")?;
                let name = if res == 0 {
                    var.to_string()
                } else {
                    var.replacen("acc", "res", 1)
                };
                ops::input(c, &name, out_ty).map_err(smt)?
            }
            BlockKind::Output => {
                let s = port_src(fu, sel, b, 0)?;
                val(c, fu, bind, sel, memo, s.block, s.res, depth + 1)?
            }
            BlockKind::Op => {
                let mut args = Vec::new();
                for (i, &p) in bd.ports.iter().enumerate() {
                    let s = port_src(fu, sel, b, p)?;
                    let x = val(c, fu, bind, sel, memo, s.block, s.res, depth + 1)?;
                    let want = tys.get(i).ok_or("an operand without a type")?;
                    if ty(want)? != x.ty {
                        return Err(format!(
                            "block {b} operand {i}: {} where the configuration says {want}",
                            x.ty.name()
                        ));
                    }
                    args.push((p, x));
                }
                let bits = bd.types.iter().find_map(|t| {
                    let r = t.strip_prefix("rp:e")?;
                    let (e, m) = r.split_once('m')?;
                    Some((e.parse::<u32>().ok()?, m.parse::<u32>().ok()?))
                });
                let a = Attrs {
                    direction: blk.attrs.get("comparison_direction").cloned(),
                    compare_type: blk.attrs.get("compare_type").cloned(),
                    precision: bits,
                };
                // A widened block computes at its own types; its widening lemma (`widen_query`) proves
                // that equal to the op at the configuration's types, which is what is evaluated here.
                let xs: Vec<Val> = args.iter().map(|x| x.1).collect();
                ops::apply(c, &blk.op, &a, &xs, out_ty).map_err(smt)?
            }
        };
        memo.insert((b, res), v);
        Ok(v)
    }

    let mut out = BTreeMap::new();
    for bd in &cfg.bindings {
        let blk = fu
            .blocks
            .get(bd.block as usize)
            .ok_or_else(|| format!("block {}", bd.block))?;
        match blk.kind {
            BlockKind::Output => {
                let v = val(c, fu, &bind, &sel, &mut memo, bd.block, 0, 0)?;
                out.insert(bd.var.clone().ok_or("an output without a name")?, v);
            }
            BlockKind::Acc => {
                let i = bd
                    .var
                    .as_deref()
                    .and_then(|v| v.strip_prefix("acc"))
                    .ok_or("an accumulator without an index")?;
                for (p, what) in [(0u32, "next"), (1, "init")] {
                    let s = port_src(fu, &sel, bd.block, p)?;
                    let v = val(c, fu, &bind, &sel, &mut memo, s.block, s.res, 0)?;
                    out.insert(format!("{what}{i}"), v);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// A widening: an op at narrow types run on a block at wide types.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Widening {
    pub op: String,
    pub direction: Option<String>,
    pub compare_type: Option<String>,
    pub narrow: Vec<String>,
    pub wide: Vec<String>,
}

impl Widening {
    pub fn file(&self) -> String {
        let mut s = format!("widen/{}", self.op);
        for x in [&self.direction, &self.compare_type].into_iter().flatten() {
            s.push('.');
            s.push_str(x);
        }
        s.push_str(&format!("_{}_{}.smt2", self.narrow.join("-"), self.wide.join("-")));
        s.replace(':', "")
    }
}

/// Every widening used by a configuration.
pub fn widenings(fus: &BTreeMap<String, &FuFile>, cfgs: &[ConfigFile]) -> R<Vec<Widening>> {
    let mut set = std::collections::BTreeSet::new();
    for c in cfgs {
        let fu = fus.get(&c.fu).ok_or_else(|| format!("{}: no unit", c.fu))?;
        for b in c.bindings.iter().filter(|b| b.mode == Mode::Widen) {
            let blk = fu.blocks.get(b.block as usize).ok_or_else(|| format!("block {}", b.block))?;
            if blk.kind != BlockKind::Op {
                continue;
            }
            set.insert(Widening {
                op: blk.op.clone(),
                direction: blk.attrs.get("comparison_direction").cloned(),
                compare_type: blk.attrs.get("compare_type").cloned(),
                narrow: b.types.clone(),
                wide: blk.types.clone(),
            });
        }
    }
    Ok(set.into_iter().collect())
}

/// The widening lemma: the op at the wide types on operands extended by their signedness, truncated to
/// the narrow result, differs from the op at the narrow types (expect unsat).
pub fn widen_query(w: &Widening) -> R<Query> {
    let mut c = Ctx::new();
    let narrow: Vec<Ty> = w.narrow.iter().filter(|t| !t.starts_with("rp:")).map(|t| ty(t)).collect::<R<_>>()?;
    let wide: Vec<Ty> = w.wide.iter().filter(|t| !t.starts_with("rp:")).map(|t| ty(t)).collect::<R<_>>()?;
    if narrow.len() != wide.len() || narrow.is_empty() {
        return Err(format!("{}: types {:?} and {:?}", w.op, w.narrow, w.wide));
    }
    let a = Attrs {
        direction: w.direction.clone(),
        compare_type: w.compare_type.clone(),
        precision: None,
    };
    let k = narrow.len() - 1;
    let mut xs = Vec::new();
    let mut ws = Vec::new();
    for i in 0..k {
        let x = ops::input(&mut c, &format!("x{i}"), narrow[i]).map_err(smt)?;
        ws.push(ops::convert(&mut c, x, wide[i]).map_err(smt)?);
        xs.push(x);
    }
    let want = ops::apply(&mut c, &w.op, &a, &xs, narrow[k]).map_err(smt)?;
    let r = ops::apply(&mut c, &w.op, &a, &ws, wide[k]).map_err(smt)?;
    let got = ops::convert(&mut c, r, narrow[k]).map_err(smt)?;
    let same = ops::same(&mut c, got, want).map_err(smt)?;
    let goal = c.not(same).map_err(smt)?;
    let mut s = Script::new();
    s.comment(format!(
        "merge widening {}: ({}) on a ({}) block, truncated, differs from the narrow op (expect unsat)",
        w.op,
        w.narrow.join(", "),
        w.wide.join(", ")
    ));
    c.render(&mut s, &[goal]);
    Ok(Query {
        file: w.file(),
        text: s.render(),
    })
}

/// The query of one configuration.
pub fn config_query(fu: &FuFile, cfg: &ConfigFile, dp: &Datapath, t: &Tuple) -> R<Query> {
    let mut c = Ctx::new();
    let got = fu_values(&mut c, fu, cfg)?;
    let all = eval::values(&mut c, dp, t)?;
    let mut eqs = Vec::new();
    for (k, v) in &got {
        let w = all
            .get(k)
            .ok_or_else(|| format!("{}: the datapath has no value {k}", cfg.unit))?;
        eqs.push(ops::same(&mut c, *v, *w).map_err(|e| format!("{k}: {}", e.0))?);
    }
    if eqs.is_empty() {
        return Err(format!("{}: nothing to compare", cfg.unit));
    }
    let a = c.and(&eqs).map_err(smt)?;
    let goal = c.not(a).map_err(smt)?;
    let mut s = Script::new();
    s.comment(format!(
        "merge {} {} {}: the configured unit differs from the datapath (expect unsat)",
        fu.id, cfg.unit, cfg.tuple
    ));
    s.comment(format!(
        "values: {}",
        got.keys().cloned().collect::<Vec<_>>().join(" ")
    ));
    c.render(&mut s, &[goal]);
    Ok(Query {
        file: format!("{}/{}/{}.smt2", fu.id, cfg.unit, cfg.tuple),
        text: s.render(),
    })
}

fn int_of(t: Ty) -> Option<(bool, u32)> {
    match t {
        Ty::Int { signed, w } => Some((signed, w)),
        _ => None,
    }
}

/// The class's core step at its own types.
fn core_step(c: &mut Ctx, l: Val, r: Val, acc: Val) -> R<Val> {
    let a = Attrs::default();
    let lc = ops::convert(c, l, acc.ty).map_err(smt)?;
    let rc = ops::convert(c, r, acc.ty).map_err(smt)?;
    let p = ops::apply(c, "multiply", &a, &[lc, rc], acc.ty).map_err(smt)?;
    ops::apply(c, "add", &a, &[acc, p], acc.ty).map_err(smt)
}

/// One step of the processing element for a class tuple in its mode.
pub fn pe_step(c: &mut Ctx, arr: &ArrayFile, t: &ArrayTuple, l: Val, r: Val, acc: Val) -> R<Val> {
    let pe: Vec<Ty> = arr.pe.iter().map(|x| ty(x)).collect::<R<_>>()?;
    if pe.len() != 3 {
        return Err(format!("{}: a processing element needs three types", arr.id));
    }
    if pe[2] != acc.ty {
        return Err(format!(
            "{}: accumulator {} on a {} array",
            arr.id,
            acc.ty.name(),
            pe[2].name()
        ));
    }
    match t.mode {
        ArrayMode::Native => {
            if l.ty != pe[0] || r.ty != pe[1] {
                return Err(format!("{} {}: not native", arr.id, t.tuple));
            }
            core_step(c, l, r, acc)
        }
        ArrayMode::Widen => {
            let lw = ops::convert(c, l, pe[0]).map_err(smt)?;
            let rw = ops::convert(c, r, pe[1]).map_err(smt)?;
            core_step(c, lw, rw, acc)
        }
        ArrayMode::Compose => {
            let left = t.split.as_deref() == Some("lhs");
            let (x, y, px, py) = if left {
                (l, r, pe[0], pe[1])
            } else {
                (r, l, pe[1], pe[0])
            };
            let (xs, xw) = int_of(x.ty).ok_or("only integers are composed")?;
            let (ps, pw) = int_of(px).ok_or("only integer arrays compose")?;
            let (_, aw) = int_of(acc.ty).ok_or("an integer accumulator")?;
            if t.pieces.iter().sum::<u32>() != xw {
                return Err(format!(
                    "{} {}: pieces {:?} do not make {xw} bits",
                    arr.id, t.tuple, t.pieces
                ));
            }
            let yv = ops::convert(c, y, py).map_err(smt)?;
            let mut sum = acc;
            let mut lo = 0u32;
            for (k, &len) in t.pieces.iter().enumerate() {
                let top = k + 1 == t.pieces.len();
                let piece = c.extract(x.t, lo + len - 1, lo).map_err(smt)?;
                let signed_piece = top && xs;
                if len > pw || (!ps && signed_piece) || (ps && !signed_piece && len >= pw) {
                    return Err(format!(
                        "{} {}: a {len}-bit piece does not fit the element",
                        arr.id, t.tuple
                    ));
                }
                let pv = c.resize(piece, pw, signed_piece).map_err(smt)?;
                let pv = Val::new(px, pv);
                let (xp, yp) = if left { (pv, yv) } else { (yv, pv) };
                let zero = ops::zero(c, acc.ty).map_err(smt)?;
                let prod = core_step(c, xp, yp, zero)?;
                let sh = c.bv(u128::from(lo), aw);
                let shifted = c.op2(Op::BvShl, prod.t, sh).map_err(smt)?;
                let s = c.op2(Op::BvAdd, sum.t, shifted).map_err(smt)?;
                sum = Val::new(acc.ty, s);
                lo += len;
            }
            Ok(sum)
        }
    }
}

/// The query of one class tuple on an array.
pub fn array_query(arr: &ArrayFile, class: &str, t: &ArrayTuple) -> R<Query> {
    let mut c = Ctx::new();
    let tys: Vec<Ty> = t.types.iter().map(|x| ty(x)).collect::<R<_>>()?;
    if tys.len() != 3 {
        return Err(format!("{class} {}: three types", t.tuple));
    }
    let l = ops::input(&mut c, "lhs", tys[0]).map_err(smt)?;
    let r = ops::input(&mut c, "rhs", tys[1]).map_err(smt)?;
    let acc = ops::input(&mut c, "acc", tys[2]).map_err(smt)?;
    let want = core_step(&mut c, l, r, acc)?;
    let got = pe_step(&mut c, arr, t, l, r, acc)?;
    let same = ops::same(&mut c, got, want).map_err(smt)?;
    let goal = c.not(same).map_err(smt)?;
    let mut s = Script::new();
    s.comment(format!(
        "merge array {} {class} {}: the processing element ({:?}) differs from the core step (expect unsat)",
        arr.id, t.tuple, t.mode
    ));
    c.render(&mut s, &[goal]);
    Ok(Query {
        file: format!("arrays/{}/{class}/{}.smt2", arr.id, t.tuple),
        text: s.render(),
    })
}
