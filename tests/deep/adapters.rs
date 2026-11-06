//! Conformance of the miners (design §3.7; docs/framework/adapters.md): every registered `Miner`, on
//! GraMi's two-path fixture at support 2, finds exactly the frequent patterns (the edges 1->2 and 2->3 and
//! the path 1->2->3, each with two disjoint images), and a second call reuses the finished run.

use cc_core::adapters::{MineJob, MineRun};
use cc_core::runner::{BoundRunner, Caps, Runners, ToolLog};
use cc_core::{CancelToken, LoadedConfig};
use cc_passes::adapters::MINERS;
use cc_passes::mine::algo::lg::parse_output;
use tests::repo_root;

/// A pattern: its vertex labels and its edges `(from, to, label)`.
type Pattern = (Vec<u32>, Vec<(u32, u32, u32)>);

#[test]
fn miners_conform() {
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &[]).unwrap();
    let graph = std::fs::read_to_string(repo_root().join("tests/fixtures/grami/two_paths.lg")).unwrap();
    for miner in MINERS.iter() {
        let d = tempfile::tempdir().unwrap();
        let runner = BoundRunner::new(
            Runners::from_config(&cfg).docker,
            Caps {
                memory_gb: 4.0,
                cpus: 1.0,
                timeout_s: 300.0,
                pids_limit: 512,
            },
            ToolLog::new("adapters", d.path().join("logs")),
            CancelToken::new(),
        );
        let job = MineJob {
            dir: &d.path().join("run"),
            graph: &graph,
            support: 2,
            max_vertices: 6,
            memory_gb: 1,
            timeout_s: 300.0,
        };
        let MineRun::Done(text) = miner.mine(&runner, &job).unwrap() else {
            panic!("{}: timeout", miner.name());
        };
        let mut found: Vec<Pattern> = parse_output(&text)
            .unwrap()
            .into_iter()
            .map(|p| (p.nodes, p.edges))
            .collect();
        found.sort();
        assert_eq!(
            found,
            vec![
                (vec![1, 2], vec![(0, 1, 0)]),
                (vec![1, 2, 3], vec![(0, 1, 0), (1, 2, 0)]),
                (vec![2, 3], vec![(0, 1, 0)]),
            ],
            "{}: {text}",
            miner.name()
        );
        // A finished run is cached in its directory.
        assert_eq!(
            miner.mine(&runner, &job).unwrap(),
            MineRun::Done(text),
            "{}",
            miner.name()
        );
    }
}
