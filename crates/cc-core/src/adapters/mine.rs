//! Frequent subgraph miners (design §3.7, §4.10; docs/tools/grami.md). Implementation: GraMi
//! (`cc_passes::mine`). Graphs and patterns are exchanged in the `.lg` text format: `t # <id>`, then
//! `v <id> <label>` and `e <from> <to> <label>` lines.

use std::path::Path;

use super::Named;
use crate::error::Result;
use crate::runner::BoundRunner;

/// What a miner supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinerCapabilities {
    /// Directed edges (an operand order is part of a pattern).
    pub directed: bool,
    /// Prunes patterns over `MineJob::max_vertices` during the search (not after it).
    pub vertex_bound: bool,
    /// Support is the minimum image count (MNI, Bringmann and Nijssen 2008), exact, not sampled.
    pub mni_support: bool,
}

/// One mining run.
#[derive(Debug, Clone, Copy)]
pub struct MineJob<'a> {
    /// The run's own directory: inputs, outputs and a cached result live there.
    pub dir: &'a Path,
    /// The data graph, `.lg` text.
    pub graph: &'a str,
    /// Minimum support.
    pub support: u64,
    /// Largest pattern, in vertices.
    pub max_vertices: usize,
    /// The miner's working memory, GB (GraMi: its Java heap).
    pub memory_gb: u64,
    pub timeout_s: f64,
}

/// What a run gave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MineRun {
    /// The frequent patterns, in the miner's output text (`.lg` blocks with their support).
    Done(String),
    /// Stopped at `timeout_s`.
    Timeout,
}

/// A frequent subgraph miner. A finished run is cached in `job.dir` and reused.
pub trait Miner: Named {
    fn capabilities(&self) -> MinerCapabilities;
    fn mine(&self, runner: &BoundRunner, job: &MineJob) -> Result<MineRun>;
}
