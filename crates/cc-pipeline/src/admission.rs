//! Admission control (spec §3.3, §8). A ready pass starts when the passes already running plus this one
//! stay within `pipeline.max_parallel_passes`, `resources.memory_budget_gb` and
//! `resources.threads_total`.
//!
//! Oversize rule: a pass whose own declaration exceeds a limit can never satisfy it, so it is admitted
//! only when nothing else is running, and then runs alone.

use cc_core::PassResources;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    pub max_parallel: usize,
    pub memory_budget_gb: f64,
    pub threads_total: usize,
}

/// What is currently running.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Load {
    pub count: usize,
    pub memory_gb: f64,
    pub threads: usize,
}

impl Load {
    pub fn add(&mut self, r: &PassResources) {
        self.count += 1;
        self.memory_gb += r.memory_gb;
        self.threads += r.threads;
    }
}

const EPS: f64 = 1e-9;

pub fn is_oversize(limits: &Limits, req: &PassResources) -> bool {
    req.memory_gb > limits.memory_budget_gb + EPS || req.threads > limits.threads_total
}

pub fn admissible(load: &Load, limits: &Limits, req: &PassResources) -> bool {
    if load.count == 0 {
        return true;
    }
    load.count < limits.max_parallel
        && load.memory_gb + req.memory_gb <= limits.memory_budget_gb + EPS
        && load.threads + req.threads <= limits.threads_total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(memory_gb: f64, threads: usize) -> PassResources {
        PassResources {
            threads,
            memory_gb,
            timeout_min: 1.0,
        }
    }

    const L: Limits = Limits {
        max_parallel: 2,
        memory_budget_gb: 4.0,
        threads_total: 8,
    };

    #[test]
    fn memory_budget_blocks_second_pass() {
        let mut load = Load::default();
        assert!(admissible(&load, &L, &res(3.0, 1)));
        load.add(&res(3.0, 1));
        assert!(!admissible(&load, &L, &res(3.0, 1)));
        assert!(admissible(&load, &L, &res(1.0, 1)));
    }

    #[test]
    fn count_and_threads_limits() {
        let mut load = Load::default();
        load.add(&res(1.0, 6));
        assert!(!admissible(&load, &L, &res(1.0, 3)));
        load.add(&res(1.0, 1));
        assert!(!admissible(&load, &L, &res(0.5, 1)));
    }

    #[test]
    fn oversize_runs_only_alone() {
        let big = res(8.0, 1);
        assert!(is_oversize(&L, &big));
        assert!(admissible(&Load::default(), &L, &big));
        let mut load = Load::default();
        load.add(&res(0.5, 1));
        assert!(!admissible(&load, &L, &big));
    }
}
