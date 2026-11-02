//! Routing (docs/passes/iteration_space.md §4.4): a class goes to an array template when it is a
//! contraction whose recurrence is uniform, has a valid space-time mapping, and the rows where both
//! inputs are reused at least `min_reuse` times in the recurrence (Kung 1982, "multiple use of each input
//! data item") hold at least `min_share` of its weighted workload. Every other class goes to datapath
//! mining.

use crate::iteration_space::TemplateConfig;
use crate::iteration_space::algo::deps::operand;
use crate::iteration_space::algo::eval::Table;
use crate::iteration_space::json::{
    Candidate, Dependences, Group, Reuse, Route, Routed, SkelClass, Skeleton,
};

/// The reuse numbers of a contraction: per row, each input is used once per point of the group it is
/// passed along in the recurrence (lhs along N, rhs along M; 1 when that group is empty), which is the
/// reuse an array realizes. Sliding-window reuse inside the reduction (a convolution's input along
/// `e_y − e_r`) is not realized by the grouped recurrence and does not count.
pub fn reuse(s: &Skeleton, d: &Dependences, t: &Table<'_>, min_reuse: u64) -> Option<Reuse> {
    operand(s, "lhs")?;
    operand(s, "rhs")?;
    let empty = Vec::new();
    let n = d.groups.get(&Group::N).unwrap_or(&empty);
    let m = d.groups.get(&Group::M).unwrap_or(&empty);
    let (mut total, mut good, mut ra, mut rb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for r in t.rows {
        let w = r.weighted as f64;
        if w <= 0.0 {
            continue;
        }
        total += w;
        let (Some(fa), Some(fb)) = (t.group_extent(s, n, r), t.group_extent(s, m, r)) else {
            continue;
        };
        ra += w * fa as f64;
        rb += w * fb as f64;
        if fa >= min_reuse && fb >= min_reuse {
            good += w;
        }
    }
    let q = |x: f64| {
        if total > 0.0 {
            super::mapping::round6(x / total)
        } else {
            0.0
        }
    };
    Some(Reuse {
        share: q(good),
        lhs: q(ra),
        rhs: q(rb),
    })
}

pub fn route(
    s: &Skeleton,
    d: &Dependences,
    cands: &[Candidate],
    t: &Table<'_>,
    cfg: &TemplateConfig,
    workload: u64,
) -> Routed {
    let mut r = Routed {
        id: s.id.clone(),
        route: Route::Datapath,
        class: s.class,
        core: s.core.clone(),
        reason: String::new(),
        reuse: None,
        dataflow: None,
        stages: None,
        workload,
    };
    match s.class {
        SkelClass::Map if !s.affine => {
            r.reason = "map with a data-dependent index (scatter without scatter dims): not affine".into()
        }
        SkelClass::Map => r.reason = "map: no reduction dims".into(),
        SkelClass::Reduce if !s.affine => {
            r.reason = "scatter-reduce: a data-dependent target index is not affine".into()
        }
        SkelClass::Reduce => r.reason = "reduction: no core input is reused along a parallel dim".into(),
        SkelClass::Contraction => {
            r.reuse = reuse(s, d, t, cfg.min_reuse);
            let share = r.reuse.as_ref().map_or(0.0, |x| x.share);
            r.reason = if !cfg.enabled {
                "contraction: templates are disabled".into()
            } else if !d.uniform {
                format!("contraction, not a uniform recurrence: {}", d.reason)
            } else if d.space.len() > cfg.max_dims as usize {
                format!(
                    "contraction: a {}-dim space (> max_dims = {})",
                    d.space.len(),
                    cfg.max_dims
                )
            } else if cands.is_empty() {
                "contraction: no valid space-time mapping".into()
            } else if t.rows.iter().all(|x| x.weighted == 0) {
                "contraction: no instance with workload".into()
            } else if share < cfg.min_share {
                format!(
                    "contraction: rows where both inputs are reused >= {} times hold {:.1}% of the workload (< {:.1}%)",
                    cfg.min_reuse,
                    100.0 * share,
                    100.0 * cfg.min_share
                )
            } else {
                let c = &cands[0];
                r.route = Route::Template;
                r.dataflow = Some(c.dataflow.clone());
                r.stages = Some(s.stages.clone());
                let space: Vec<&str> = d.space.iter().map(|g| g.name()).collect();
                format!(
                    "contraction, uniform over ({}); {} valid mappings, selected {} (λ = {:?}, u = {:?}, utilization {:.3}); \
                     both inputs reused >= {} times in rows holding {:.1}% of the workload",
                    space.join(", "),
                    cands.len(),
                    c.dataflow,
                    c.lambda,
                    c.u,
                    c.utilization,
                    cfg.min_reuse,
                    100.0 * share
                )
            };
        }
    }
    r
}
