use serde::Deserialize;

/// `[passes.mine]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MineConfig {
    /// First support threshold of the sweep (MNI, an absolute count of distinct images).
    pub tau_start: u64,
    /// Last support threshold; the sweep halves τ from `tau_start` down to it.
    pub tau_min: u64,
    /// Patterns with more vertices are dropped after GraMi (GraMi has no size bound of its own).
    pub max_vertices: usize,
    /// Patterns kept per family, by score.
    pub top_k: usize,
    /// Wall-clock budget of all GraMi runs of one family (safety net, not the stop rule).
    pub budget_s: u64,
    /// Wall-clock limit of one GraMi run.
    pub run_timeout_s: u64,
    /// Java heap of one GraMi run; the container gets 2 GB more.
    pub heap_gb: u64,
    /// Candidate pairs one VF2 search (one pattern in one datapath) may try.
    pub vf2_budget: u64,
    /// Write `views/mine.md`.
    pub views: bool,
    pub mis: MisConfig,
}

/// `[passes.mine.mis]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MisConfig {
    pub algorithm: MisAlgorithm,
    /// Search nodes `branch_and_bound` may expand per overlap-graph component before it reports its
    /// incumbent as a lower bound.
    pub node_budget: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MisAlgorithm {
    ExactIlp,
    BranchAndBound,
    GreedyMinDegree,
}

impl MisAlgorithm {
    pub fn name(self) -> &'static str {
        match self {
            MisAlgorithm::ExactIlp => "exact_ilp",
            MisAlgorithm::BranchAndBound => "branch_and_bound",
            MisAlgorithm::GreedyMinDegree => "greedy_min_degree",
        }
    }
}

impl Default for MineConfig {
    fn default() -> Self {
        Self {
            tau_start: 40,
            tau_min: 2,
            max_vertices: 8,
            top_k: 32,
            budget_s: 3600,
            run_timeout_s: 600,
            heap_gb: 6,
            vf2_budget: 100_000_000,
            views: true,
            mis: MisConfig::default(),
        }
    }
}

impl Default for MisConfig {
    fn default() -> Self {
        Self {
            algorithm: MisAlgorithm::ExactIlp,
            node_budget: 1_000_000,
        }
    }
}

impl MineConfig {
    pub fn check(&self) -> Result<(), String> {
        if self.tau_min < 1 || self.tau_min > self.tau_start {
            return Err(format!(
                "need 1 <= tau_min <= tau_start (tau_min = {}, tau_start = {})",
                self.tau_min, self.tau_start
            ));
        }
        if self.max_vertices < 2 {
            return Err("max_vertices must be at least 2 (GraMi patterns have an edge)".into());
        }
        if self.top_k == 0 || self.heap_gb == 0 || self.run_timeout_s == 0 || self.budget_s == 0 {
            return Err("top_k, heap_gb, run_timeout_s and budget_s must be positive".into());
        }
        Ok(())
    }

    /// The thresholds of the sweep: `tau_start`, halving, ending exactly at `tau_min`.
    pub fn taus(&self) -> Vec<u64> {
        let mut v = vec![self.tau_start];
        let mut t = self.tau_start;
        while t > self.tau_min {
            t = (t / 2).max(self.tau_min);
            v.push(t);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sweep_halves_down_to_tau_min() {
        let c = MineConfig::default();
        assert_eq!(c.taus(), vec![40, 20, 10, 5, 2]);
        let c = MineConfig {
            tau_start: 3,
            tau_min: 3,
            ..MineConfig::default()
        };
        assert_eq!(c.taus(), vec![3]);
        assert!(
            MineConfig {
                tau_min: 0,
                ..MineConfig::default()
            }
            .check()
            .is_err()
        );
    }
}
