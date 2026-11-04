//! XLS IR functions as SMT terms: the `top fn` of an IR package (the text `opt_main` writes) is parsed and
//! each node translated into the context, so a query can state what the circuit computes
//! (docs/framework/smt.md, "XLS IR").
//!
//! Every value is a bit-vector (`bits[1]` too). The node semantics follow the XLS IR reference
//! (https://google.github.io/xls/ir_semantics/): shifts by at least the width give 0 (`shra`: the sign
//! fill), `sel` takes `default` for a selector past its cases, `priority_sel` the lowest set selector bit,
//! `udiv` by 0 gives all ones and `umod` by 0 gives 0, `umul`/`smul` keep the low result-width bits of the
//! extended operands. A node kind outside this list is an error, never a silent approximation.

use std::collections::BTreeMap;

use crate::term::{Ctx, Op, Sort, SortError, T};

/// One parsed node: `name: bits[width] = op(operands, attrs)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub name: String,
    pub width: u32,
    pub op: String,
    pub operands: Vec<String>,
    pub attrs: BTreeMap<String, String>,
}

/// The `top fn` of a package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    /// `(name, width)` in order.
    pub params: Vec<(String, u32)>,
    pub ret_width: u32,
    pub nodes: Vec<Node>,
    pub ret: String,
}

/// How multiplications are translated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Multiply {
    /// `bvmul` of the extended operands.
    #[default]
    Exact,
    /// Each `umul` is an application of the uninterpreted function [`umul_name`] of its operands as they
    /// are (a query that proves a claim for every such function proves it for the product).
    Uninterpreted,
}

/// The uninterpreted function standing for a `umul` of a `wa`- and `wb`-bit operand into `wr` bits.
pub fn umul_name(wa: u32, wb: u32, wr: u32) -> String {
    format!("umul_{wa}_{wb}_{wr}")
}

fn bits_width(t: &str) -> Result<u32, String> {
    let t = t.trim();
    let inner = t
        .strip_prefix("bits[")
        .and_then(|r| r.strip_suffix(']'))
        .ok_or_else(|| format!("unsupported type {t:?} (bits only)"))?;
    inner.parse().map_err(|_| format!("bad type {t:?}"))
}

/// Split at top-level commas (not inside `[...]` or `(...)`).
fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            _ => {}
        }
        if ch == ',' && depth == 0 {
            out.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Parse the `top fn` of an XLS IR package.
pub fn parse(text: &str) -> Result<Function, String> {
    let mut lines = text.lines();
    let header = lines
        .by_ref()
        .find(|l| l.trim_start().starts_with("top fn "))
        .ok_or("no `top fn` in the package")?;
    let h = header.trim_start().trim_start_matches("top fn ");
    let open = h.find('(').ok_or("bad fn header")?;
    let name = h[..open].trim().to_string();
    let close = h.rfind(") ->").ok_or("bad fn header")?;
    let mut params = Vec::new();
    for p in split_top(&h[open + 1..close]) {
        let (n, rest) = p.split_once(':').ok_or_else(|| format!("bad param {p:?}"))?;
        let ty = rest.trim().split(" id=").next().unwrap_or("");
        params.push((n.trim().to_string(), bits_width(ty)?));
    }
    let ret_ty = h[close + 4..].trim().trim_end_matches('{').trim();
    let ret_width = bits_width(ret_ty)?;
    let mut nodes = Vec::new();
    let mut ret = None;
    for l in lines {
        let l = l.trim();
        if l == "}" {
            break;
        }
        if l.is_empty() {
            continue;
        }
        let (is_ret, l) = match l.strip_prefix("ret ") {
            Some(r) => (true, r),
            None => (false, l),
        };
        let (lhs, rhs) = l.split_once(" = ").ok_or_else(|| format!("bad node {l:?}"))?;
        let (n, ty) = lhs.split_once(':').ok_or_else(|| format!("bad node {l:?}"))?;
        let width = bits_width(ty)?;
        let op_end = rhs.find('(').ok_or_else(|| format!("bad node {l:?}"))?;
        let op = rhs[..op_end].trim().to_string();
        let body = rhs[op_end + 1..]
            .strip_suffix(')')
            .ok_or_else(|| format!("bad node {l:?}"))?;
        let mut operands = Vec::new();
        let mut attrs = BTreeMap::new();
        for a in split_top(body) {
            match a.split_once('=') {
                Some((k, v)) if !k.contains('[') => {
                    attrs.insert(k.trim().to_string(), v.trim().to_string());
                }
                _ => operands.push(a),
            }
        }
        let n = n.trim().to_string();
        if is_ret {
            ret = Some(n.clone());
        }
        nodes.push(Node {
            name: n,
            width,
            op,
            operands,
            attrs,
        });
    }
    let ret = ret.ok_or("no `ret` node")?;
    Ok(Function {
        name,
        params,
        ret_width,
        nodes,
        ret,
    })
}

fn literal(v: &str, w: u32) -> Result<u128, String> {
    let v = v.replace('_', "");
    let (neg, v) = match v.strip_prefix('-') {
        Some(r) => (true, r.to_string()),
        None => (false, v),
    };
    let x = if let Some(h) = v.strip_prefix("0x") {
        u128::from_str_radix(h, 16)
    } else if let Some(b) = v.strip_prefix("0b") {
        u128::from_str_radix(b, 2)
    } else {
        v.parse::<u128>()
    }
    .map_err(|_| format!("bad literal {v:?}"))?;
    let mask = if w >= 128 { u128::MAX } else { (1u128 << w) - 1 };
    Ok(if neg { x.wrapping_neg() & mask } else { x & mask })
}

fn list(v: &str) -> Vec<String> {
    split_top(v.trim().trim_start_matches('[').trim_end_matches(']'))
}

impl Function {
    /// The value of the function on `args` (one term per parameter, of its width).
    pub fn translate(&self, c: &mut Ctx, args: &[T], mul: Multiply) -> Result<T, String> {
        let e = |x: SortError| x.0;
        if args.len() != self.params.len() {
            return Err(format!(
                "{}: {} arguments for {} parameters",
                self.name,
                args.len(),
                self.params.len()
            ));
        }
        let mut val: BTreeMap<&str, T> = BTreeMap::new();
        for ((n, w), &a) in self.params.iter().zip(args) {
            if c.sort(a) != Sort::Bv(*w) {
                return Err(format!("{}: parameter {n} is bits[{w}]", self.name));
            }
            val.insert(n.as_str(), a);
        }
        for nd in &self.nodes {
            let get = |n: &str| -> Result<T, String> {
                val.get(n)
                    .copied()
                    .ok_or_else(|| format!("{}: unknown operand {n:?}", nd.name))
            };
            let ops: Vec<T> = nd.operands.iter().map(|o| get(o)).collect::<Result<_, _>>()?;
            let w = nd.width;
            let width_of = |c: &Ctx, t: T| c.sort(t).width();
            let attr = |k: &str| -> Result<&String, String> {
                nd.attrs
                    .get(k)
                    .ok_or_else(|| format!("{}: {} without {k}", nd.name, nd.op))
            };
            let num = |k: &str| -> Result<u32, String> {
                attr(k)?.parse().map_err(|_| format!("{}: bad {k}", nd.name))
            };
            let bit = |c: &mut Ctx, b: T| -> Result<T, String> {
                // A Bool as bits[1].
                let one = c.bv(1, 1);
                let zero = c.bv(0, 1);
                c.ite(b, one, zero).map_err(e)
            };
            let fold = |c: &mut Ctx, op: Op, xs: &[T]| -> Result<T, String> {
                let mut acc = *xs.first().ok_or("no operands")?;
                for &x in &xs[1..] {
                    acc = c.op2(op.clone(), acc, x).map_err(e)?;
                }
                Ok(acc)
            };
            let t: T = match nd.op.as_str() {
                "literal" => c.bv(literal(attr("value")?, w)?, w),
                "identity" => ops[0],
                "bit_slice" => {
                    let s = num("start")?;
                    c.extract(ops[0], s + w - 1, s).map_err(e)?
                }
                "dynamic_bit_slice" => {
                    let xw = width_of(c, ops[0]);
                    let total = xw + w;
                    let x = c.resize(ops[0], total, false).map_err(e)?;
                    let sw = width_of(c, ops[1]);
                    let st = if sw <= total {
                        c.resize(ops[1], total, false).map_err(e)?
                    } else {
                        let lim = c.bv(u128::from(total), sw);
                        let big = c.op2(Op::BvUle, lim, ops[1]).map_err(e)?;
                        let lo = c.extract(ops[1], total - 1, 0).map_err(e)?;
                        let cap = c.bv(u128::from(total), total);
                        c.ite(big, cap, lo).map_err(e)?
                    };
                    let sh = c.op2(Op::BvLshr, x, st).map_err(e)?;
                    c.extract(sh, w - 1, 0).map_err(e)?
                }
                "concat" => fold(c, Op::Concat, &ops)?,
                "zero_ext" | "sign_ext" => c.resize(ops[0], w, nd.op == "sign_ext").map_err(e)?,
                "and" => fold(c, Op::BvAnd, &ops)?,
                "or" => fold(c, Op::BvOr, &ops)?,
                "xor" => fold(c, Op::BvXor, &ops)?,
                "nand" => {
                    let a = fold(c, Op::BvAnd, &ops)?;
                    c.op1(Op::BvNot, a).map_err(e)?
                }
                "nor" => {
                    let a = fold(c, Op::BvOr, &ops)?;
                    c.op1(Op::BvNot, a).map_err(e)?
                }
                "not" => c.op1(Op::BvNot, ops[0]).map_err(e)?,
                "neg" => c.op1(Op::BvNeg, ops[0]).map_err(e)?,
                "add" => c.op2(Op::BvAdd, ops[0], ops[1]).map_err(e)?,
                "sub" => c.op2(Op::BvSub, ops[0], ops[1]).map_err(e)?,
                "umul" | "smul" => {
                    let signed = nd.op == "smul";
                    let (wa, wb) = (width_of(c, ops[0]), width_of(c, ops[1]));
                    if !signed && mul == Multiply::Uninterpreted {
                        let f = umul_name(wa, wb, w);
                        c.fun(&f, &[Sort::Bv(wa), Sort::Bv(wb)], Sort::Bv(w)).map_err(e)?;
                        c.app(Op::Uf(f), &[ops[0], ops[1]]).map_err(e)?
                    } else {
                        let wm = w.max(wa).max(wb);
                        let a = c.resize(ops[0], wm, signed).map_err(e)?;
                        let b = c.resize(ops[1], wm, signed).map_err(e)?;
                        let p = c.op2(Op::BvMul, a, b).map_err(e)?;
                        c.extract(p, w - 1, 0).map_err(e)?
                    }
                }
                "udiv" | "umod" => {
                    let z = c.bv(0, w);
                    let isz = c.eq(ops[1], z).map_err(e)?;
                    if nd.op == "udiv" {
                        c.op2(Op::BvUdiv, ops[0], ops[1]).map_err(e)?
                    } else {
                        let r = c.op2(Op::BvUrem, ops[0], ops[1]).map_err(e)?;
                        c.ite(isz, z, r).map_err(e)?
                    }
                }
                "shll" | "shrl" | "shra" => {
                    let op = match nd.op.as_str() {
                        "shll" => Op::BvShl,
                        "shrl" => Op::BvLshr,
                        _ => Op::BvAshr,
                    };
                    let aw = width_of(c, ops[1]);
                    let amt = if aw <= w {
                        c.resize(ops[1], w, false).map_err(e)?
                    } else {
                        // An amount past the width saturates to the width (same result).
                        let lim = c.bv(u128::from(w), aw);
                        let big = c.op2(Op::BvUle, lim, ops[1]).map_err(e)?;
                        let lo = c.extract(ops[1], w - 1, 0).map_err(e)?;
                        let cap = c.bv(u128::from(w), w);
                        c.ite(big, cap, lo).map_err(e)?
                    };
                    c.op2(op, ops[0], amt).map_err(e)?
                }
                "eq" | "ne" | "ult" | "ule" | "ugt" | "uge" | "slt" | "sle" | "sgt" | "sge" => {
                    let (a, b) = (ops[0], ops[1]);
                    let r = match nd.op.as_str() {
                        "eq" => c.eq(a, b),
                        "ne" => {
                            let q = c.eq(a, b).map_err(e)?;
                            c.not(q)
                        }
                        "ult" => c.op2(Op::BvUlt, a, b),
                        "ule" => c.op2(Op::BvUle, a, b),
                        "ugt" => c.op2(Op::BvUlt, b, a),
                        "uge" => c.op2(Op::BvUle, b, a),
                        "slt" => c.op2(Op::BvSlt, a, b),
                        "sle" => c.op2(Op::BvSle, a, b),
                        "sgt" => c.op2(Op::BvSlt, b, a),
                        _ => c.op2(Op::BvSle, b, a),
                    }
                    .map_err(e)?;
                    bit(c, r)?
                }
                "or_reduce" | "and_reduce" => {
                    let xw = width_of(c, ops[0]);
                    let r = if nd.op == "or_reduce" {
                        let z = c.bv(0, xw);
                        let q = c.eq(ops[0], z).map_err(e)?;
                        c.not(q).map_err(e)?
                    } else {
                        let ones = c.bv(if xw >= 128 { u128::MAX } else { (1u128 << xw) - 1 }, xw);
                        c.eq(ops[0], ones).map_err(e)?
                    };
                    bit(c, r)?
                }
                "xor_reduce" => {
                    let xw = width_of(c, ops[0]);
                    let mut acc = c.extract(ops[0], 0, 0).map_err(e)?;
                    for i in 1..xw {
                        let b = c.extract(ops[0], i, i).map_err(e)?;
                        acc = c.op2(Op::BvXor, acc, b).map_err(e)?;
                    }
                    acc
                }
                "reverse" => {
                    let xw = width_of(c, ops[0]);
                    let bits: Vec<T> = (0..xw)
                        .map(|i| c.extract(ops[0], i, i).map_err(e))
                        .collect::<Result<_, _>>()?;
                    fold(c, Op::Concat, &bits)?
                }
                "sel" => {
                    let cases: Vec<T> = list(attr("cases")?)
                        .iter()
                        .map(|n| get(n))
                        .collect::<Result<_, _>>()?;
                    let default = nd.attrs.get("default").map(|d| get(d)).transpose()?;
                    let sw = width_of(c, ops[0]);
                    let mut acc = match default {
                        Some(d) => d,
                        None => *cases.last().ok_or("sel without cases")?,
                    };
                    let n = cases.len() - usize::from(default.is_none());
                    for i in (0..n).rev() {
                        let k = c.bv(i as u128, sw);
                        let is = c.eq(ops[0], k).map_err(e)?;
                        acc = c.ite(is, cases[i], acc).map_err(e)?;
                    }
                    acc
                }
                "priority_sel" => {
                    let cases: Vec<T> = list(attr("cases")?)
                        .iter()
                        .map(|n| get(n))
                        .collect::<Result<_, _>>()?;
                    let mut acc = get(attr("default")?)?;
                    for i in (0..cases.len()).rev() {
                        let b = c.extract(ops[0], i as u32, i as u32).map_err(e)?;
                        let one = c.bv(1, 1);
                        let is = c.eq(b, one).map_err(e)?;
                        acc = c.ite(is, cases[i], acc).map_err(e)?;
                    }
                    acc
                }
                "one_hot_sel" => {
                    let cases: Vec<T> = list(attr("cases")?)
                        .iter()
                        .map(|n| get(n))
                        .collect::<Result<_, _>>()?;
                    let mut acc = c.bv(0, w);
                    for (i, &k) in cases.iter().enumerate() {
                        let b = c.extract(ops[0], i as u32, i as u32).map_err(e)?;
                        let mask = c.resize(b, w, true).map_err(e)?;
                        let part = c.op2(Op::BvAnd, mask, k).map_err(e)?;
                        acc = c.op2(Op::BvOr, acc, part).map_err(e)?;
                    }
                    acc
                }
                "one_hot" => {
                    let xw = width_of(c, ops[0]);
                    let lsb = attr("lsb_prio")? == "true";
                    let mut acc = c.bv(1u128 << xw, w);
                    let order: Vec<u32> = if lsb {
                        (0..xw).rev().collect()
                    } else {
                        (0..xw).collect()
                    };
                    for i in order {
                        let b = c.extract(ops[0], i, i).map_err(e)?;
                        let one = c.bv(1, 1);
                        let is = c.eq(b, one).map_err(e)?;
                        let v = c.bv(1u128 << i, w);
                        acc = c.ite(is, v, acc).map_err(e)?;
                    }
                    acc
                }
                "encode" => {
                    let xw = width_of(c, ops[0]);
                    let mut acc = c.bv(0, w);
                    for i in 0..xw {
                        let b = c.extract(ops[0], i, i).map_err(e)?;
                        let one = c.bv(1, 1);
                        let is = c.eq(b, one).map_err(e)?;
                        let v = c.bv(u128::from(i) & ((1u128 << w) - 1), w);
                        let z = c.bv(0, w);
                        let part = c.ite(is, v, z).map_err(e)?;
                        acc = c.op2(Op::BvOr, acc, part).map_err(e)?;
                    }
                    acc
                }
                "decode" => {
                    let xw = width_of(c, ops[0]);
                    let mut acc = c.bv(0, w);
                    for i in 0..w {
                        if xw < 128 && u128::from(i) >= (1u128 << xw) {
                            break;
                        }
                        let k = c.bv(u128::from(i), xw);
                        let is = c.eq(ops[0], k).map_err(e)?;
                        let v = c.bv(1u128 << i, w);
                        acc = c.ite(is, v, acc).map_err(e)?;
                    }
                    acc
                }
                other => return Err(format!("{}: XLS op {other:?} is not translated", nd.name)),
            };
            if c.sort(t) != Sort::Bv(w) {
                return Err(format!(
                    "{}: {} gave {:?}, declared bits[{w}]",
                    nd.name,
                    nd.op,
                    c.sort(t)
                ));
            }
            val.insert(nd.name.as_str(), t);
        }
        val.get(self.ret.as_str())
            .copied()
            .ok_or_else(|| format!("{}: no value for ret {}", self.name, self.ret))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Script;

    const IR: &str = "package p\n\ntop fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {\n  \
        literal.3: bits[8] = literal(value=0x0f, id=3)\n  \
        and.4: bits[8] = and(a, literal.3, id=4, pos=[(1,2,3)])\n  \
        bit_slice.5: bits[1] = bit_slice(b, start=7, width=1, id=5)\n  \
        shrl.6: bits[8] = shrl(b, bit_slice.5, id=6)\n  \
        ret sel.7: bits[8] = sel(bit_slice.5, cases=[and.4, shrl.6], id=7)\n}\n";

    #[test]
    fn parses_a_function() {
        let f = parse(IR).unwrap();
        assert_eq!(f.name, "f");
        assert_eq!(f.params, vec![("a".to_string(), 8), ("b".to_string(), 8)]);
        assert_eq!(f.ret, "sel.7");
        assert_eq!(f.nodes.len(), 5);
        assert_eq!(f.nodes[4].attrs["cases"], "[and.4, shrl.6]");
        assert_eq!(f.nodes[0].attrs["value"], "0x0f");
    }

    #[test]
    fn translates_and_renders() {
        let f = parse(IR).unwrap();
        let mut c = Ctx::new();
        let a = c.var("a", Sort::Bv(8)).unwrap();
        let b = c.var("b", Sort::Bv(8)).unwrap();
        let r = f.translate(&mut c, &[a, b], Multiply::Exact).unwrap();
        let goal = c.eq(r, a).unwrap();
        let mut sc = Script::new();
        c.render(&mut sc, &[goal]);
        let text = sc.render();
        assert!(text.contains("bvlshr"), "{text}");
        assert!(text.contains("(_ bv15 8)"), "{text}");
    }

    #[test]
    fn rejects_unknown_ops_and_bad_widths() {
        let bad = IR.replace("shrl(b, bit_slice.5, id=6)", "frob(b, id=6)");
        let f = parse(&bad).unwrap();
        let mut c = Ctx::new();
        let a = c.var("a", Sort::Bv(8)).unwrap();
        let b = c.var("b", Sort::Bv(8)).unwrap();
        assert!(
            f.translate(&mut c, &[a, b], Multiply::Exact)
                .unwrap_err()
                .contains("frob")
        );
        let narrow = c.var("n", Sort::Bv(4)).unwrap();
        assert!(f.translate(&mut c, &[a, narrow], Multiply::Exact).is_err());
    }

    #[test]
    fn literals_wrap_to_their_width() {
        assert_eq!(literal("0xff_ff", 8).unwrap(), 0xff);
        assert_eq!(literal("-1", 4).unwrap(), 0xf);
        assert_eq!(literal("300", 16).unwrap(), 300);
    }
}
