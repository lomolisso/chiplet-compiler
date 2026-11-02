//! The proof obligations of a class (docs/passes/datapaths.md §6), one SMT-LIB query each:
//!
//! - `<z>/<t>.smt2`: for width tuple `t`, some input makes a datapath value differ from the kernel's
//!   (every output; for a template every core input; every accumulator's next and init value). `unsat`
//!   proves datapath = kernel semantics for all inputs, constant values included;
//! - `<z>/<t>.order.smt2`: the combiner is not both commutative and associative. `unsat` proves every
//!   order of the reduction gives the same bits (ω is free); `sat` gives the counterexample (ω binds).

use std::collections::BTreeSet;

use cc_ir::absgraph::Graph;
use cc_smt::Script;
use cc_smt::ops::{self, Val};
use cc_smt::term::Ctx;

use crate::datapaths::json::{Datapath, Tuple};
use crate::datapaths::{eval, spec};
use crate::iteration_space::json::{Routed, Skeleton};
use crate::kernel_classes::json::ClassFile;
use crate::kernels::json::KNode;

type R<X> = Result<X, String>;

/// One query file (`file` relative to `proofs/`).
#[derive(Debug, Clone)]
pub struct Query {
    pub file: String,
    pub text: String,
    pub tuple: String,
    pub order: bool,
}

/// The queries of one class, the encodings they use and their uninterpreted functions.
#[derive(Debug, Clone, Default)]
pub struct Obligations {
    pub queries: Vec<Query>,
    pub used: BTreeSet<String>,
    pub functions: BTreeSet<String>,
}

fn names(t: &Tuple) -> String {
    let mut s = t.elems.join(" ");
    for (k, v) in &t.consts {
        s.push_str(&format!(" {k}:{v}"));
    }
    for (k, v) in &t.attrs {
        s.push_str(&format!(" {k}={v}"));
    }
    s
}

pub fn obligations(
    class: &ClassFile,
    skel: &Skeleton,
    routed: &Routed,
    g: &Graph<KNode>,
    dp: &Datapath,
) -> R<Obligations> {
    let mut out = Obligations::default();
    for t in &dp.tuples {
        let mut c = Ctx::new();
        let want = spec::expected(&mut c, class, skel, routed, g, t)?;
        let got = eval::values(&mut c, dp, t)?;
        let (wk, gk): (BTreeSet<&String>, BTreeSet<&String>) = (want.keys().collect(), got.keys().collect());
        if wk != gk {
            return Err(format!(
                "{}: the datapath has values {gk:?}, the kernel {wk:?}",
                t.name
            ));
        }
        let mut eqs = Vec::new();
        for (k, w) in &want {
            eqs.push(ops::same(&mut c, got[k], *w).map_err(|e| format!("{k}: {}", e.0))?);
        }
        let all = c.and(&eqs).map_err(|e| e.0)?;
        let goal = c.not(all).map_err(|e| e.0)?;
        let mut s = Script::new();
        s.comment(format!(
            "datapaths {} {}: the datapath differs from the kernel's semantics (expect unsat)",
            dp.id, t.name
        ));
        s.comment(format!("tuple: {}", names(t)));
        s.comment(format!(
            "values: {}",
            want.keys().cloned().collect::<Vec<_>>().join(" ")
        ));
        c.render(&mut s, &[goal]);
        out.used.extend(c.used().iter().cloned());
        out.functions.extend(c.functions().cloned());
        out.queries.push(Query {
            file: format!("{}/{}.smt2", dp.id, t.name),
            text: s.render(),
            tuple: t.name.clone(),
            order: false,
        });
        if dp.accumulator.is_some() {
            out.queries.push(order(class, skel, g, dp, t, &mut out.used)?);
        }
    }
    Ok(out)
}

fn order(
    class: &ClassFile,
    skel: &Skeleton,
    g: &Graph<KNode>,
    dp: &Datapath,
    t: &Tuple,
    used: &mut BTreeSet<String>,
) -> R<Query> {
    let mut c = Ctx::new();
    let tys = spec::acc_types(class, skel, g, t)?;
    let mk = |c: &mut Ctx, p: &str| -> R<Vec<Val>> {
        tys.iter()
            .enumerate()
            .map(|(i, ty)| ops::input(c, &format!("{p}{i}"), *ty).map_err(|e| e.0))
            .collect()
    };
    let (a, b, d) = (mk(&mut c, "a")?, mk(&mut c, "b")?, mk(&mut c, "c")?);
    let f = |c: &mut Ctx, x: &[Val], y: &[Val]| spec::combiner(c, class, skel, g, t, x, y);
    let ab = f(&mut c, &a, &b)?;
    let ba = f(&mut c, &b, &a)?;
    let bc = f(&mut c, &b, &d)?;
    let left = f(&mut c, &ab, &d)?;
    let right = f(&mut c, &a, &bc)?;
    let mut eqs = Vec::new();
    for (x, y) in ab.iter().zip(&ba).chain(left.iter().zip(&right)) {
        eqs.push(ops::same(&mut c, *x, *y).map_err(|e| e.0)?);
    }
    let all = c.and(&eqs).map_err(|e| e.0)?;
    let goal = c.not(all).map_err(|e| e.0)?;
    let mut s = Script::new();
    s.comment(format!(
        "datapaths {} {}: the combiner is not commutative and associative (unsat: every order is exact; sat: omega binds)",
        dp.id, t.name
    ));
    c.render(&mut s, &[goal]);
    used.extend(c.used().iter().cloned());
    Ok(Query {
        file: format!("{}/{}.order.smt2", dp.id, t.name),
        text: s.render(),
        tuple: t.name.clone(),
        order: true,
    })
}
