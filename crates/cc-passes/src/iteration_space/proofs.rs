//! SMT-LIB obligations (docs/passes/iteration_space.md §6).
//!
//! - **Access** (`smt/access.<z>.smt2`): every claim of `spec.rs`, stored row = specified row (numerator
//!   and divisor). Both sides are polynomials in the iteration indices, extents and size parameters of
//!   degree at most 2 in each variable. A polynomial of degree at most `t_i` in variable `x_i` that
//!   vanishes on a grid `S_1 × … × S_n` with `|S_i| > t_i` is zero (Alon, "Combinatorial
//!   Nullstellensatz", Combinatorics, Probability and Computing 8:7-29, 1999, Lemma 2.1), so it is enough
//!   to prove the equalities for every variable in `{0, …, 7}`. In 32-bit vectors no term of such values
//!   wraps, so the bit-vector proof is the integer proof and therefore the proof for all values.
//! - **Mapping** (`smt/mapping.<z>.smt2`): for every candidate of a template, `λ·d ≥ 1` for every
//!   dependence, and no two distinct points of a box `[0, E)` with `E < 2^20` share a PE (`S z1 = S z2`)
//!   and a time step (`λ·z1 = λ·z2`): stated for their difference `x = z1 − z2` (`|x_i| < 2^20`),
//!   `x ≠ 0 ∧ S x = 0 ∧ λ·x = 0` is unsatisfiable. Every term is a sum of at most three such values
//!   times ±1, so 32-bit arithmetic is exact here as well.
//!
//! Each file asserts the disjunction of the negated claims; `unsat` proves them all.

use std::collections::BTreeSet;

use cc_smt::Script;

use crate::iteration_space::algo::embed::{Poly, poly};
use crate::iteration_space::json::{Factor, Row, TemplateFile};
use crate::iteration_space::spec::Claim;

const W: u32 = 32;
const GRID: u64 = 8;

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn factor_name(f: &Factor) -> String {
    match f {
        Factor::Column { column, index } => format!("c_{}_{index}", sanitize(column)),
        Factor::Runtime { node, port, index } => format!("r_{node}_{port}_{index}"),
        Factor::Extent { dim } => format!("e{dim}"),
    }
}

fn lit(v: i64) -> String {
    cc_smt::bv_lit((v as u64) & ((1u64 << W) - 1), W)
}

fn render(p: &Poly) -> String {
    let mut parts = Vec::new();
    for ((dim, factors), k) in p {
        let mut xs: Vec<String> = factors.iter().map(factor_name).collect();
        if let Some(d) = dim {
            xs.push(format!("i{d}"));
        }
        let prod = match xs.len() {
            0 => None,
            1 => Some(xs.remove(0)),
            _ => Some(cc_smt::app("bvmul", &xs)),
        };
        parts.push(match prod {
            None => lit(*k),
            Some(p) => scale(*k, p),
        });
    }
    match parts.len() {
        0 => lit(0),
        1 => parts.remove(0),
        _ => cc_smt::app("bvadd", &parts),
    }
}

fn divisor(r: &Row) -> String {
    r.divisor.as_ref().map_or(lit(1), factor_name)
}

fn symbols(p: &Poly, out: &mut BTreeSet<String>) {
    for (dim, factors) in p.keys() {
        if let Some(d) = dim {
            out.insert(format!("i{d}"));
        }
        out.extend(factors.iter().map(factor_name));
    }
}

/// The largest degree of one variable in a polynomial (the grid argument needs it below 8).
pub fn max_degree(p: &Poly) -> usize {
    let mut m = 0;
    for (dim, factors) in p.keys() {
        let mut names: Vec<String> = factors.iter().map(factor_name).collect();
        if let Some(d) = dim {
            names.push(format!("i{d}"));
        }
        names.sort();
        let mut run = 1;
        for w in names.windows(2) {
            run = if w[0] == w[1] { run + 1 } else { 1 };
            m = m.max(run);
        }
        m = m.max(usize::from(!names.is_empty()));
    }
    m
}

/// The access obligation of a class.
pub fn access(id: &str, claims: &[Claim]) -> Result<String, String> {
    let mut s = Script::new();
    s.comment(format!("iteration_space access obligation of class {id}"));
    s.comment("stored index = StableHLO spec index, for every variable in [0, 8) (Alon 1999, Lemma 2.1)");
    let mut syms = BTreeSet::new();
    let mut bad = Vec::new();
    for c in claims {
        let (a, b) = (poly(&c.stored), poly(&c.spec));
        if max_degree(&a).max(max_degree(&b)) >= GRID as usize {
            return Err(format!("{}: degree too high for the grid argument", c.what));
        }
        symbols(&a, &mut syms);
        symbols(&b, &mut syms);
        for f in c.stored.divisor.iter().chain(c.spec.divisor.iter()) {
            syms.insert(factor_name(f));
        }
        bad.push(cc_smt::distinct(&[render(&a), render(&b)]));
        bad.push(cc_smt::distinct(&[divisor(&c.stored), divisor(&c.spec)]));
    }
    for v in &syms {
        s.constant(v, &cc_smt::bv(W));
        s.assert(format!("(bvult {v} {})", lit(GRID as i64)));
    }
    s.assert(if bad.is_empty() {
        "false".to_string()
    } else {
        cc_smt::or(&bad)
    });
    Ok(s.render())
}

/// `c·v` without a multiplier for the coefficients ±1 (a bit-blasted multiplier makes the query slow).
fn scale(c: i64, v: String) -> String {
    match c {
        1 => v,
        -1 => format!("(bvneg {v})"),
        _ => cc_smt::app("bvmul", &[lit(c), v]),
    }
}

fn lin(coef: &[i64], vars: &[String]) -> String {
    let parts: Vec<String> = coef
        .iter()
        .zip(vars)
        .filter(|(c, _)| **c != 0)
        .map(|(c, v)| scale(*c, v.clone()))
        .collect();
    match parts.len() {
        0 => lit(0),
        1 => parts[0].clone(),
        _ => cc_smt::app("bvadd", &parts),
    }
}

/// The mapping obligation of a template class (`deps`: the dependence vectors). Two points `z1 ≠ z2`
/// of the box `[0, E)` with `E < 2^20` meet on a PE at one time iff their difference `x = z1 − z2`
/// (each `|x_i| < 2^20`) is nonzero with `S x = 0` and `λ·x = 0`, so the obligation states that for
/// `x` directly.
pub fn mapping(t: &TemplateFile, deps: &[(String, Vec<i64>)]) -> String {
    let mut s = Script::new();
    s.comment(format!("iteration_space mapping obligation of class {}", t.id));
    s.comment("every candidate: causal (λ·d >= 1) and conflict-free (S x = 0 and λ·x = 0 imply x = 0 for |x_i| < 2^20)");
    let n = t.space.len();
    let x: Vec<String> = (0..n).map(|i| format!("x{i}")).collect();
    for v in &x {
        s.constant(v, &cc_smt::bv(W));
        s.assert(format!("(bvslt {v} {})", lit(1 << 20)));
        s.assert(format!("(bvsgt {v} {})", lit(-(1 << 20))));
    }
    let nonzero = cc_smt::or(
        &x.iter()
            .map(|v| cc_smt::distinct(&[v.clone(), lit(0)]))
            .collect::<Vec<_>>(),
    );
    let mut bad = Vec::new();
    for c in &t.candidates {
        let mut viol = Vec::new();
        for (_, d) in deps {
            let ld: Vec<String> = c.lambda.iter().map(|v| lit(*v)).collect();
            viol.push(format!("(bvslt {} {})", lin(d, &ld), lit(1)));
        }
        let mut same = vec![nonzero.clone()];
        for r in &c.s {
            same.push(format!("(= {} {})", lin(r, &x), lit(0)));
        }
        same.push(format!("(= {} {})", lin(&c.lambda, &x), lit(0)));
        viol.push(cc_smt::and(&same));
        bad.push(cc_smt::or(&viol));
    }
    s.assert(if bad.is_empty() {
        "false".to_string()
    } else {
        cc_smt::or(&bad)
    });
    s.render()
}
