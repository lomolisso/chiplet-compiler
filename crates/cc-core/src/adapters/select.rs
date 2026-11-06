//! Selections that are NP-hard (design §3.7, §4.10, §4.11): a maximum-weight clique on `merge`'s
//! compatibility graphs and a maximum independent set on `mine`'s overlap graphs. Each problem type is
//! the caller's (`G`, with its solution `S`); implementations are exact or heuristic, chosen by
//! `algorithm` in the pass's section.

use std::time::Duration;

use super::Named;

/// What a selection algorithm guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectCapabilities {
    /// Returns an optimum when it finishes within its limits.
    pub exact: bool,
    /// Honours `Limits::node_budget` (then returns its incumbent, not claimed optimal).
    pub budgeted: bool,
    /// Honours `Limits::timeout`.
    pub timed: bool,
}

/// Bounds of one solve; `0` / `None` is no bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Limits {
    /// Search nodes (branch and bound); deterministic, unlike the timeout.
    pub node_budget: u64,
    pub timeout: Option<Duration>,
}

/// A maximum-weight clique solver on graphs `G`, with `nogoods` (vertex sets that may not all be chosen).
/// `Err` says why it gave no answer (`timeout`, `budget`); the caller then falls back.
pub trait CliqueSolver<G, S>: Named {
    fn capabilities(&self) -> SelectCapabilities;
    fn solve(&self, graph: &G, nogoods: &[Vec<usize>], limits: &Limits) -> Result<S, String>;
}

/// A maximum independent set solver on graphs `G`.
pub trait MisSolver<G, S>: Named {
    fn capabilities(&self) -> SelectCapabilities;
    fn solve(&self, graph: &G, limits: &Limits) -> Result<S, String>;
}
