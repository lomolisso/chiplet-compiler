//! The proof obligation of every rewrite rule, as SMT-LIB 2 text (docs/passes/normalize.md §4). Each
//! script asserts the negation of the rule's equivalence for every case it covers (ranks up to
//! `MAX_RANK`, arities, formats) as one disjunction: `unsat` proves every case. Tensors are uninterpreted
//! functions from bit-vector indices to bit-vector elements, so a proof holds for every element type;
//! element-wise ops are uninterpreted functions too, so it holds for every op. Two controls check the
//! solver: integer reassociation (`unsat`) and float reassociation (`sat`, with a counterexample).

use cc_smt::{Script, and, app, bv, bv_lit, or};

use super::algo::rules::MAX_RANK;
use super::json::Expect;

const IDX: u32 = 32;
const AXIS: u32 = 4;
const ELEM: u32 = 32;

/// `(rule, expected verdict, script)`, in a fixed order.
pub fn obligations() -> Vec<(&'static str, Expect, String)> {
    vec![
        ("dce", Expect::Unsat, dce()),
        ("transpose_identity", Expect::Unsat, transpose_identity()),
        ("broadcast_identity", Expect::Unsat, broadcast_identity()),
        ("convert_identity", Expect::Unsat, convert_identity()),
        ("transpose_collapse", Expect::Unsat, transpose_collapse()),
        ("reshape_collapse", Expect::Unsat, reshape_collapse()),
        ("splat_fold", Expect::Unsat, splat_fold()),
        ("cse", Expect::Unsat, cse()),
        ("control_int_reassociation", Expect::Unsat, int_reassociation()),
        ("control_float_reassociation", Expect::Sat, float_reassociation()),
    ]
}

fn idx_args(r: usize) -> Vec<String> {
    vec![bv(IDX); r]
}

fn eq(a: &str, b: &str) -> String {
    format!("(= {a} {b})")
}

fn not(a: &str) -> String {
    format!("(not {a})")
}

/// `v[pos]` for a symbolic position `pos` (an `ite` chain; the last entry is the default, so the caller
/// constrains `pos` to the rank).
fn at_pos(v: &[String], pos: &str) -> String {
    let r = v.len();
    let mut t = v[r - 1].clone();
    for k in (0..r - 1).rev() {
        t = format!("(ite (= {pos} {}) {} {t})", bv_lit(k as u64, AXIS), v[k]);
    }
    t
}

/// The result index that `transpose(x, perm)` gives the element at operand index `oi` (spec `#transpose`:
/// `result_index[d] = operand_index[permutation[d]]`).
fn transpose_fwd(perm: &[String], oi: &[String]) -> Vec<String> {
    perm.iter().map(|p| at_pos(oi, p)).collect()
}

fn in_rank(p: &[String]) -> String {
    let r = p.len();
    let c: Vec<String> = p
        .iter()
        .map(|x| format!("(bvult {x} {})", bv_lit(r as u64, AXIS)))
        .collect();
    and(&c)
}

fn all_eq(a: &[String], b: &[String]) -> String {
    and(&a.iter().zip(b).map(|(x, y)| eq(x, y)).collect::<Vec<_>>())
}

/// `dce`: the outputs are a function of the nodes they reach; removing a node that reaches no output leaves
/// them unchanged. `g(x)` before (with `h(x)` computed and unused) and after.
fn dce() -> String {
    let mut s = Script::new();
    s.comment("dce: an unused pure op does not change the outputs");
    s.fun("g", &[bv(ELEM)], &bv(ELEM))
        .fun("h", &[bv(ELEM)], &bv(ELEM));
    s.constant("x", &bv(ELEM)).constant("d", &bv(ELEM));
    s.assert(eq("d", "(h x)"));
    s.assert(not(&eq("(g x)", "(g x)")));
    s.render()
}

/// `cse`: two applications of one pure op to equal operands are equal (congruence), for arities 1 to 3.
fn cse() -> String {
    let mut s = Script::new();
    s.comment("cse: f(a...) = f(b...) when a = b, for arities 1..3");
    let mut cases = Vec::new();
    for k in 1..=3usize {
        let f = format!("f{k}");
        s.fun(&f, &vec![bv(ELEM); k], &bv(ELEM));
        let a: Vec<String> = (0..k).map(|i| format!("a{k}_{i}")).collect();
        let b: Vec<String> = (0..k).map(|i| format!("b{k}_{i}")).collect();
        for (x, y) in a.iter().zip(&b) {
            s.constant(x, &bv(ELEM)).constant(y, &bv(ELEM));
        }
        cases.push(and(&[all_eq(&a, &b), not(&eq(&app(&f, &a), &app(&f, &b)))]));
    }
    s.assert(or(&cases));
    s.render()
}

/// `transpose_identity`: `transpose(x, [0, …, r-1])` sends every element to its own index, ranks 1 to
/// `MAX_RANK`.
fn transpose_identity() -> String {
    let mut s = Script::new();
    s.comment("transpose_identity: transpose(x, iota) maps every operand index to itself, ranks 1..MAX_RANK");
    let mut cases = Vec::new();
    for r in 1..=MAX_RANK {
        let i: Vec<String> = (0..r).map(|d| format!("i{r}_{d}")).collect();
        for v in &i {
            s.constant(v, &bv(IDX));
        }
        let p: Vec<String> = (0..r).map(|d| bv_lit(d as u64, AXIS)).collect();
        cases.push(not(&all_eq(&transpose_fwd(&p, &i), &i)));
    }
    s.assert(or(&cases));
    s.render()
}

/// `transpose_collapse`: `transpose(transpose(x, p), q)` and `transpose(x, c)` with `c[d] = p[q[d]]` send
/// every operand index to the same result index. Result axis `d` of the outer transpose reads position
/// `j = q[d]` of the inner one's result index, so both sides agree on axis `d` exactly when
/// `fwd(p, xi)[j] = xi[p[j]]`; the query proves that for a symbolic position `j` (every `q[d]` is one), a
/// symbolic `p` and a symbolic operand index, ranks 1 to `MAX_RANK`. Both sides are bijections of the
/// index space (`#transpose` C2), so equal index maps give equal tensors. Index entries are compared only
/// for equality and at most `MAX_RANK` of them occur, so `AXIS` bits suffice for them (any
/// counterexample renames into that width).
fn transpose_collapse() -> String {
    let mut s = Script::new();
    s.comment("transpose_collapse: T(T(x,p),q) and T(x,c), c[d] = p[q[d]], map indices alike: fwd(p,xi)[j] = xi[p[j]], symbolic j, p, xi; ranks 1..MAX_RANK");
    let mut cases = Vec::new();
    for r in 1..=MAX_RANK {
        let xi: Vec<String> = (0..r).map(|d| format!("x{r}_{d}")).collect();
        let p: Vec<String> = (0..r).map(|d| format!("p{r}_{d}")).collect();
        let j = format!("j{r}");
        for v in xi.iter().chain(&p) {
            s.constant(v, &bv(AXIS));
        }
        s.constant(&j, &bv(AXIS));
        let before = at_pos(&transpose_fwd(&p, &xi), &j);
        let after = at_pos(&xi, &at_pos(&p, &j));
        cases.push(and(&[
            in_rank(&p),
            in_rank_of(&j, r),
            not(&eq(&before, &after)),
        ]));
    }
    s.assert(or(&cases));
    s.render()
}

fn in_rank_of(x: &str, r: usize) -> String {
    format!("(bvult {x} {})", bv_lit(r as u64, AXIS))
}

/// `broadcast_identity`: `broadcast_in_dim(x, iota)` into the operand's own shape reads `x` at the same
/// index (spec `#broadcast_in_dim`: `operand_index[d] = 0` if `dim(operand, d) = 1`, else
/// `result_index[broadcast_dimensions[d]]`), for in-bounds indices, ranks 0 to `MAX_RANK`.
fn broadcast_identity() -> String {
    let mut s = Script::new();
    s.comment("broadcast_identity: broadcast_in_dim(x, iota) to x's shape = x, ranks 1..MAX_RANK");
    let mut cases = Vec::new();
    for r in 1..=MAX_RANK {
        let x = format!("x{r}");
        s.fun(&x, &idx_args(r), &bv(ELEM));
        let i: Vec<String> = (0..r).map(|d| format!("i{r}_{d}")).collect();
        let n: Vec<String> = (0..r).map(|d| format!("n{r}_{d}")).collect();
        for v in i.iter().chain(&n) {
            s.constant(v, &bv(IDX));
        }
        let bounds: Vec<String> = i
            .iter()
            .zip(&n)
            .map(|(a, b)| format!("(bvult {a} {b})"))
            .collect();
        let oi: Vec<String> = i
            .iter()
            .zip(&n)
            .map(|(a, b)| format!("(ite (= {b} {}) {} {a})", bv_lit(1, IDX), bv_lit(0, IDX)))
            .collect();
        cases.push(and(&[and(&bounds), not(&eq(&app(&x, &oi), &app(&x, &i)))]));
    }
    s.assert(or(&cases));
    s.render()
}

/// `convert_identity`: converting to the operand's own type is the identity: for the IEEE binary formats
/// f64, f32 and f16 (`to_fp` with RNE from the same format), and for integers (the value is kept: zero
/// extension by 0). Bitwuzla supports only the IEEE interchange formats, so bf16 and the FP8 formats are
/// stated on their encodings, which a conversion to the same type keeps (the integer cases of 8 and 16
/// bits).
fn convert_identity() -> String {
    let mut s = Script::new();
    s.comment("convert_identity: convert(x : T) : T = x for f64, f32, f16, and on the encoding of every type up to 64 bits");
    let mut cases = Vec::new();
    for (k, (e, m)) in [(11u32, 53u32), (8, 24), (5, 11)].into_iter().enumerate() {
        let v = format!("f{k}");
        s.constant(&v, &format!("(_ FloatingPoint {e} {m})"));
        cases.push(not(&eq(&format!("((_ to_fp {e} {m}) RNE {v})"), &v)));
    }
    for w in [1u32, 2, 4, 8, 16, 32, 64] {
        let v = format!("i{w}");
        s.constant(&v, &bv(w));
        cases.push(not(&eq(&format!("((_ zero_extend 0) {v})"), &v)));
    }
    s.assert(or(&cases));
    s.render()
}

/// `reshape_collapse`: a reshape keeps the row-major (lexicographic) order of the elements (spec
/// `#reshape`), so `reshape_s(x) = unflatten_s(flatten(x))` and `flatten(unflatten_s(v)) = v`; then
/// `reshape_s2(reshape_s1(x)) = reshape_s2(x)`. The axiom is instantiated at the term it is needed for.
fn reshape_collapse() -> String {
    let mut s = Script::new();
    s.comment("reshape_collapse: reshape_s2(reshape_s1(x)) = reshape_s2(x) under the row-major definition");
    s.fun("flatten", &[bv(ELEM)], &bv(ELEM))
        .fun("unflatten1", &[bv(ELEM)], &bv(ELEM))
        .fun("unflatten2", &[bv(ELEM)], &bv(ELEM))
        .constant("x", &bv(ELEM));
    s.assert(eq("(flatten (unflatten1 (flatten x)))", "(flatten x)"));
    s.assert(not(&eq(
        "(unflatten2 (flatten (unflatten1 (flatten x))))",
        "(unflatten2 (flatten x))",
    )));
    s.render()
}

/// `splat_fold`: an element-wise op reading a uniform operand (a splat constant, or a broadcast of a
/// rank-0 constant: `operand_index` is empty, so every result element is the scalar) computes
/// `f(…, c, …)` at every index; ranks 0 to `MAX_RANK`, arities 1 to 3, every port.
fn splat_fold() -> String {
    let mut s = Script::new();
    s.comment("splat_fold: f(x, broadcast(c))[i] = f(x[i], c), ranks 0..MAX_RANK, arities 1..3, every port");
    s.constant("c", &bv(ELEM));
    s.fun("bc", &[], &bv(ELEM));
    s.assert(eq("bc", "c"));
    let mut cases = Vec::new();
    for k in 1..=3usize {
        let f = format!("f{k}");
        s.fun(&f, &vec![bv(ELEM); k], &bv(ELEM));
        for r in 0..=MAX_RANK {
            let i: Vec<String> = (0..r).map(|d| format!("i{k}_{r}_{d}")).collect();
            for v in &i {
                s.constant(v, &bv(IDX));
            }
            let xs: Vec<String> = (0..k).map(|j| format!("x{k}_{r}_{j}")).collect();
            for x in &xs {
                s.fun(x, &idx_args(r), &bv(ELEM));
            }
            for port in 0..k {
                // Before: the uniform operand read at `i` is the broadcast's rank-0 operand `bc`.
                let before: Vec<String> = (0..k)
                    .map(|j| {
                        if j == port {
                            "bc".to_string()
                        } else {
                            app(&xs[j], &i)
                        }
                    })
                    .collect();
                let after: Vec<String> = (0..k)
                    .map(|j| {
                        if j == port {
                            "c".to_string()
                        } else {
                            app(&xs[j], &i)
                        }
                    })
                    .collect();
                cases.push(not(&eq(&app(&f, &before), &app(&f, &after))));
            }
        }
    }
    s.assert(or(&cases));
    s.render()
}

/// Positive control: two's-complement addition is associative (32 bits).
fn int_reassociation() -> String {
    let mut s = Script::new();
    s.comment("control: (a + b) + c = a + (b + c) in 32-bit two's complement (expected unsat)");
    for v in ["a", "b", "c"] {
        s.constant(v, &bv(32));
    }
    s.assert(not(&eq("(bvadd (bvadd a b) c)", "(bvadd a (bvadd b c))")));
    s.render()
}

/// Negative control: an unsound rule (float reassociation, binary16, RNE) must fail its proof.
fn float_reassociation() -> String {
    let mut s = Script::new();
    s.comment("control: (a + b) + c = a + (b + c) in binary16 with RNE (expected sat: not a valid rule)");
    for v in ["a", "b", "c"] {
        s.constant(v, "(_ FloatingPoint 5 11)");
    }
    s.assert(not(&eq(
        "(fp.add RNE (fp.add RNE a b) c)",
        "(fp.add RNE a (fp.add RNE b c))",
    )));
    s.render()
}
