//! The embedding relation ⊑ (docs/passes/iteration_space.md §4.5): `guest ⊑ host` when an injective,
//! tag-preserving map φ of iteration dims and an injective, direction-preserving map π of operands make
//! every guest access equal to its host access once the host dims outside φ's image are fixed to extent 1
//! (index 0) and the host's strides, dilations and window sizes are set to 1, its offsets to 0, or to the
//! guest's own factors. Rows of a host access that vanish are axes of extent 1. Searched by
//! backtracking; the witness (φ, π, the row maps, the host factor values) is the certificate.

use std::collections::BTreeMap;

use cc_ir::ops::iter::Tag;

use crate::iteration_space::json::{Direction, Factor, Pair, Row, SkelClass, Skeleton, Value};

/// The operands that take part: a core's operands and results, or a map kernel's interface.
pub fn embedding_operands(s: &Skeleton) -> Vec<usize> {
    let core = s.class != SkelClass::Map;
    (0..s.operands.len())
        .filter(|&i| s.operands[i].core == core)
        .collect()
}

pub type Poly = BTreeMap<(Option<u32>, Vec<Factor>), i64>;

pub fn poly(r: &Row) -> Poly {
    let mut p = Poly::new();
    for t in &r.terms {
        let mut f = t.factors.clone();
        f.sort();
        *p.entry((t.dim, f)).or_default() += t.k;
    }
    p.retain(|_, k| *k != 0);
    p
}

type Sigma = BTreeMap<Factor, Value>;

/// The host's dim `h` as a guest dim (`None`: outside φ's image).
struct View<'a> {
    inv: &'a [Option<u32>],
}

impl View<'_> {
    /// A host factor's value in guest terms: `Ok(None)` is 1, `Err(())` is 0 or unassigned.
    fn factor(&self, f: &Factor, sigma: &Sigma) -> Result<Option<Factor>, Option<()>> {
        match f {
            Factor::Extent { dim } => match self.inv.get(*dim as usize).copied().flatten() {
                Some(g) => Ok(Some(Factor::Extent { dim: g })),
                None => Ok(None),
            },
            _ => match sigma.get(f) {
                Some(Value::One) => Ok(None),
                Some(Value::Guest(x)) => Ok(Some(x.clone())),
                Some(Value::Zero) => Err(Some(())),
                None => Err(None),
            },
        }
    }
}

/// Unify one host row with one guest row, extending σ; `None` when they cannot be equal.
fn unify(host: &Row, guest: &Row, view: &View<'_>, sigma: &Sigma) -> Option<Sigma> {
    if host.data != guest.data {
        return None;
    }
    let mut sigma = sigma.clone();
    let gp = poly(guest);
    // Terms on mapped dims fix the host's coefficient factors.
    for t in &host.terms {
        let Some(hd) = t.dim else { continue };
        let Some(gd) = view.inv.get(hd as usize).copied().flatten() else {
            continue;
        };
        let want: Vec<Factor> = gp
            .iter()
            .find(|((d, _), _)| *d == Some(gd))
            .map(|((_, f), _)| f.clone())?;
        let mut have: Vec<Factor> = Vec::new();
        let mut open: Vec<&Factor> = Vec::new();
        for f in &t.factors {
            match view.factor(f, &sigma) {
                Ok(Some(x)) => have.push(x),
                Ok(None) => {}
                Err(Some(())) => return None,
                Err(None) => open.push(f),
            }
        }
        let mut rest = want.clone();
        for h in &have {
            let i = rest.iter().position(|x| x == h)?;
            rest.remove(i);
        }
        match (open.as_slice(), rest.as_slice()) {
            ([], []) => {}
            ([o], []) => {
                sigma.insert((*o).clone(), Value::One);
            }
            ([o], [g]) => {
                sigma.insert((*o).clone(), Value::Guest(g.clone()));
            }
            _ => return None,
        }
    }
    // Parameters left only in constant terms are offsets: 0.
    for t in host.terms.iter().filter(|t| t.dim.is_none()) {
        for f in &t.factors {
            if !matches!(f, Factor::Extent { .. }) && !sigma.contains_key(f) {
                sigma.insert(f.clone(), Value::Zero);
            }
        }
    }
    match (&host.divisor, &guest.divisor) {
        (None, None) => {}
        (Some(h), g) => match (view.factor(h, &sigma), g) {
            (Ok(None), None) => {}
            (Ok(Some(x)), Some(y)) if &x == y => {}
            (Err(None), None) => {
                sigma.insert(h.clone(), Value::One);
            }
            (Err(None), Some(y)) => {
                sigma.insert(h.clone(), Value::Guest(y.clone()));
            }
            _ => return None,
        },
        (None, Some(_)) => return None,
    }
    (substitute(host, view.inv, &sigma)? == gp).then_some(sigma)
}

/// The host row as a guest polynomial (`None`: a factor is unassigned).
pub fn substitute(host: &Row, inv: &[Option<u32>], sigma: &Sigma) -> Option<Poly> {
    let view = View { inv };
    let mut p = Poly::new();
    'terms: for t in &host.terms {
        let dim = match t.dim {
            None => None,
            Some(hd) => match inv.get(hd as usize).copied().flatten() {
                Some(g) => Some(g),
                None => continue,
            },
        };
        let mut f = Vec::new();
        for x in &t.factors {
            match view.factor(x, sigma) {
                Ok(Some(y)) => f.push(y),
                Ok(None) => {}
                Err(Some(())) => continue 'terms,
                Err(None) => return None,
            }
        }
        f.sort();
        *p.entry((dim, f)).or_default() += t.k;
    }
    p.retain(|_, k| *k != 0);
    Some(p)
}

struct Search<'a> {
    g: &'a Skeleton,
    h: &'a Skeleton,
    gops: Vec<usize>,
    hops: Vec<usize>,
    budget: u64,
}

impl Search<'_> {
    fn tick(&mut self) -> bool {
        if self.budget == 0 {
            return false;
        }
        self.budget -= 1;
        true
    }

    fn dims(&mut self, phi: &mut Vec<u32>, used: &mut Vec<bool>) -> Option<(Vec<u32>, Vec<Vec<u32>>, Sigma)> {
        if !self.tick() {
            return None;
        }
        let i = phi.len();
        if i == self.g.dims.len() {
            return self.close(phi);
        }
        for hd in 0..self.h.dims.len() {
            if used[hd] || self.h.dims[hd].tag != self.g.dims[i].tag {
                continue;
            }
            // Host dims that share an extent need guest dims that share one.
            let ok = phi.iter().enumerate().all(|(gj, &hj)| {
                self.h.dims[hj as usize].extent != self.h.dims[hd].extent
                    || self.g.dims[gj].extent == self.g.dims[i].extent
            });
            if !ok {
                continue;
            }
            used[hd] = true;
            phi.push(hd as u32);
            if let Some(r) = self.dims(phi, used) {
                return Some(r);
            }
            phi.pop();
            used[hd] = false;
        }
        None
    }

    fn close(&mut self, phi: &[u32]) -> Option<(Vec<u32>, Vec<Vec<u32>>, Sigma)> {
        let mut inv = vec![None; self.h.dims.len()];
        for (g, &h) in phi.iter().enumerate() {
            inv[h as usize] = Some(g as u32);
        }
        // A host dim fixed to 1 must not share its extent with a mapped one.
        for (hd, m) in inv.iter().enumerate() {
            if m.is_none()
                && phi
                    .iter()
                    .any(|&hj| self.h.dims[hj as usize].extent == self.h.dims[hd].extent)
            {
                return None;
            }
        }
        let mut pi = Vec::new();
        let mut rows = Vec::new();
        let mut used = vec![false; self.hops.len()];
        self.ops(&inv, 0, &mut pi, &mut rows, &mut used, Sigma::new())
            .map(|s| (pi, rows, s))
    }

    #[allow(clippy::too_many_arguments)]
    fn ops(
        &mut self,
        inv: &[Option<u32>],
        i: usize,
        pi: &mut Vec<u32>,
        rows: &mut Vec<Vec<u32>>,
        used: &mut [bool],
        sigma: Sigma,
    ) -> Option<Sigma> {
        if !self.tick() {
            return None;
        }
        if i == self.gops.len() {
            return Some(sigma);
        }
        let go = &self.g.operands[self.gops[i]];
        for j in 0..self.hops.len() {
            let ho = &self.h.operands[self.hops[j]];
            if used[j] || ho.direction != go.direction || ho.access.len() < go.access.len() {
                continue;
            }
            let view = View { inv };
            let mut rho = Vec::new();
            let mut taken = vec![false; ho.access.len()];
            let Some(s2) = self.rows(
                &view,
                &ho.access,
                &go.access,
                0,
                &mut rho,
                &mut taken,
                sigma.clone(),
            ) else {
                continue;
            };
            used[j] = true;
            pi.push(j as u32);
            rows.push(rho);
            if let Some(s3) = self.ops(inv, i + 1, pi, rows, used, s2) {
                return Some(s3);
            }
            rows.pop();
            pi.pop();
            used[j] = false;
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn rows(
        &mut self,
        view: &View<'_>,
        host: &[Row],
        guest: &[Row],
        i: usize,
        rho: &mut Vec<u32>,
        taken: &mut [bool],
        sigma: Sigma,
    ) -> Option<Sigma> {
        if !self.tick() {
            return None;
        }
        if i == guest.len() {
            // Every other host row must vanish.
            let mut sigma = sigma;
            for (j, r) in host.iter().enumerate() {
                if taken[j] {
                    continue;
                }
                for t in r.terms.iter().filter(|t| t.dim.is_none()) {
                    for f in &t.factors {
                        if !matches!(f, Factor::Extent { .. }) && !sigma.contains_key(f) {
                            sigma.insert(f.clone(), Value::Zero);
                        }
                    }
                }
                if r.data || !substitute(r, view.inv, &sigma)?.is_empty() {
                    return None;
                }
            }
            return Some(sigma);
        }
        for j in 0..host.len() {
            if taken[j] {
                continue;
            }
            let Some(s2) = unify(&host[j], &guest[i], view, &sigma) else {
                continue;
            };
            taken[j] = true;
            rho.push(j as u32);
            if let Some(s3) = self.rows(view, host, guest, i + 1, rho, taken, s2) {
                return Some(s3);
            }
            rho.pop();
            taken[j] = false;
        }
        None
    }
}

fn tags(s: &Skeleton) -> (usize, usize) {
    let p = s.dims.iter().filter(|d| d.tag == Tag::Par).count();
    (p, s.dims.len() - p)
}

/// Necessary conditions, cheap: tag counts, operand counts and ranks per direction.
fn plausible(g: &Skeleton, h: &Skeleton, gops: &[usize], hops: &[usize]) -> bool {
    if g.class != h.class {
        return false;
    }
    let (gp, gr) = tags(g);
    let (hp, hr) = tags(h);
    if gp > hp || gr > hr {
        return false;
    }
    for dir in [Direction::In, Direction::Out] {
        let mut gr: Vec<usize> = gops
            .iter()
            .map(|&i| &g.operands[i])
            .filter(|o| o.direction == dir)
            .map(|o| o.access.len())
            .collect();
        let mut hr: Vec<usize> = hops
            .iter()
            .map(|&i| &h.operands[i])
            .filter(|o| o.direction == dir)
            .map(|o| o.access.len())
            .collect();
        if gr.len() > hr.len() {
            return false;
        }
        gr.sort_unstable_by(|a, b| b.cmp(a));
        hr.sort_unstable_by(|a, b| b.cmp(a));
        if gr.iter().zip(&hr).any(|(a, b)| a > b) {
            return false;
        }
    }
    true
}

/// `g ⊑ h` with its certificate, or `None`; the flag says the search ran out of `budget` steps.
pub fn embed(g: &Skeleton, h: &Skeleton, budget: u64) -> (Option<Pair>, bool) {
    let gops = embedding_operands(g);
    let hops = embedding_operands(h);
    if !plausible(g, h, &gops, &hops) {
        return (None, false);
    }
    let mut s = Search {
        g,
        h,
        gops: gops.clone(),
        hops: hops.clone(),
        budget,
    };
    let mut phi = Vec::new();
    let mut used = vec![false; h.dims.len()];
    let Some((pi, rows, sigma)) = s.dims(&mut phi, &mut used) else {
        return (None, s.budget == 0);
    };
    let params: Vec<(Factor, Value)> = sigma.into_iter().collect();
    (
        Some(Pair {
            guest: g.id.clone(),
            host: h.id.clone(),
            dims: phi,
            operands: pi.iter().map(|&j| hops[j as usize] as u32).collect(),
            rows,
            params,
        }),
        false,
    )
}
