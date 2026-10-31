use cc_core::PassStatus;
use cc_pipeline::{RunOptions, RunStatus, RunSummary};
use tests::{Env, Fixture, TOY_CONFIG, resources};

fn status(s: &RunSummary, p: &str) -> RunStatus {
    s.status_of(p)
        .unwrap_or_else(|| panic!("{p} missing from {s:#?}"))
}

#[test]
fn run_then_all_hits_then_from_reruns_only_dependents() {
    let env = Env::new();
    let p = env.toy_pipeline(TOY_CONFIG, &[]);
    let s = p.run(&RunOptions::default()).unwrap();
    assert!(s.ok);
    assert_eq!((s.totals.ran, s.totals.cache_hits), (3, 0));
    let s = p.run(&RunOptions::default()).unwrap();
    assert_eq!((s.totals.ran, s.totals.cache_hits), (0, 3));
    let s = p
        .run(&RunOptions {
            from: Some("hello_b".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(status(&s, "hello_b"), RunStatus::Ok);
    assert_eq!(status(&s, "hello_a"), RunStatus::CacheHit);
    assert_eq!(status(&s, "hello_c"), RunStatus::CacheHit);
    let s = p
        .run(&RunOptions {
            from: Some("hello_a".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!((s.totals.ran, s.totals.cache_hits), (2, 1));
}

#[test]
fn run_summary_is_written_and_parses() {
    let env = Env::new();
    let s = env
        .toy_pipeline(TOY_CONFIG, &[])
        .run(&RunOptions::default())
        .unwrap();
    let path = s.path.clone().unwrap();
    assert!(path.starts_with(env.build_dir().join(".pipeline")));
    let back: RunSummary = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(back.passes, s.passes);
}

#[test]
fn only_runs_one_pass_and_needs_its_inputs() {
    let env = Env::new();
    let p = env.toy_pipeline(TOY_CONFIG, &[]);
    let s = p
        .run(&RunOptions {
            only: Some("hello_b".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(status(&s, "hello_b"), RunStatus::Failed);
    assert!(
        s.pass("hello_b")
            .unwrap()
            .message
            .as_deref()
            .unwrap()
            .contains("not available")
    );
    assert_eq!(s.passes.len(), 1);
    p.run(&RunOptions::default()).unwrap();
    let s = p
        .run(&RunOptions {
            only: Some("hello_b".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(s.passes.len(), 1);
    assert_eq!(status(&s, "hello_b"), RunStatus::Ok, "--only forces the pass");
}

#[test]
fn resume_after_induced_failure() {
    let env = Env::new();
    let s = env
        .toy_pipeline(TOY_CONFIG, &["passes.hello_b.fail=true"])
        .run(&RunOptions::default())
        .unwrap();
    assert!(!s.ok);
    assert_eq!(status(&s, "hello_a"), RunStatus::Ok);
    assert_eq!(status(&s, "hello_b"), RunStatus::Failed);
    assert_eq!(
        status(&s, "hello_c"),
        RunStatus::Ok,
        "independent branch finishes"
    );
    let m = env
        .toy_pipeline(TOY_CONFIG, &[])
        .store()
        .pass_dir("hello_b")
        .read_manifest()
        .unwrap()
        .unwrap();
    assert_eq!(m.status, PassStatus::Failed);
    assert!(m.error.unwrap().contains("induced failure"));
    let s = env
        .toy_pipeline(TOY_CONFIG, &[])
        .run(&RunOptions::default())
        .unwrap();
    assert!(s.ok);
    assert_eq!(status(&s, "hello_a"), RunStatus::CacheHit);
    assert_eq!(status(&s, "hello_b"), RunStatus::Ok);
    assert_eq!(status(&s, "hello_c"), RunStatus::CacheHit);
}

#[test]
fn failure_skips_dependents_transitively() {
    let env = Env::new();
    let passes = vec![
        Fixture::new("a").provides("k.a").fail().boxed(),
        Fixture::new("b").requires("k.a").provides("k.b").boxed(),
        Fixture::new("c").requires("k.b").boxed(),
        Fixture::new("d").boxed(),
    ];
    let s = env.pipeline(passes, "").run(&RunOptions::default()).unwrap();
    assert_eq!(status(&s, "a"), RunStatus::Failed);
    assert_eq!(status(&s, "b"), RunStatus::Skipped);
    assert_eq!(status(&s, "c"), RunStatus::Skipped);
    assert_eq!(status(&s, "d"), RunStatus::Ok);
}

#[test]
fn parallel_branch_overlaps_the_chain() {
    let env = Env::new();
    let p = env.toy_pipeline(
        TOY_CONFIG,
        &[
            "passes.hello_a.sleep_ms=300",
            "passes.hello_b.sleep_ms=300",
            "passes.hello_c.sleep_ms=400",
        ],
    );
    let s = p.run(&RunOptions::default()).unwrap();
    assert!(s.ok);
    let (a, b, c) = (
        s.pass("hello_a").unwrap(),
        s.pass("hello_b").unwrap(),
        s.pass("hello_c").unwrap(),
    );
    assert!(c.overlaps(a), "hello_c {c:?} must overlap hello_a {a:?}");
    assert!(!a.overlaps(b), "the chain is sequential");
    assert!(b.start_offset_ms >= a.end_offset_ms);
}

#[test]
fn memory_admission_serialises_and_oversize_runs_alone() {
    let env = Env::new();
    let cfg = format!(
        "[pipeline]\nmax_parallel_passes = 4\n[resources]\nmemory_budget_gb = 4\nthreads_total = 16\n{}{}{}{}",
        resources("x", 3.0, 1),
        resources("y", 3.0, 1),
        resources("z", 8.0, 1),
        resources("w", 0.5, 1),
    );
    let passes = vec![
        Fixture::new("w").sleep(400).boxed(),
        Fixture::new("x").sleep(200).boxed(),
        Fixture::new("y").sleep(200).boxed(),
        Fixture::new("z").sleep(200).boxed(),
    ];
    let s = env.pipeline(passes, &cfg).run(&RunOptions::default()).unwrap();
    assert!(s.ok, "{s:#?}");
    let get = |n| s.pass(n).unwrap();
    assert!(
        get("w").overlaps(get("x")),
        "w (0.5) and x (3) fit the budget together"
    );
    assert!(
        !get("x").overlaps(get("y")),
        "3 + 3 > 4: x and y must not overlap"
    );
    for other in ["w", "x", "y"] {
        assert!(
            !get("z").overlaps(get(other)),
            "z (8 > budget) must run alone, overlapped {other}"
        );
    }
}

#[test]
fn thread_budget_is_admission_too() {
    let env = Env::new();
    let cfg = format!(
        "[pipeline]\nmax_parallel_passes = 4\n[resources]\nmemory_budget_gb = 64\nthreads_total = 4\n{}{}",
        resources("x", 1.0, 3),
        resources("y", 1.0, 3),
    );
    let passes = vec![
        Fixture::new("x").sleep(200).boxed(),
        Fixture::new("y").sleep(200).boxed(),
    ];
    let s = env.pipeline(passes, &cfg).run(&RunOptions::default()).unwrap();
    assert!(!s.pass("x").unwrap().overlaps(s.pass("y").unwrap()));
}

#[test]
fn stop_on_failure_cancels_running_and_skips_pending() {
    let env = Env::new();
    let cfg = "[pipeline]\nmax_parallel_passes = 2\nstop_on_failure = true\n";
    let passes = vec![
        Fixture::new("a_fails").sleep(50).fail().boxed(),
        Fixture::new("b_sleeper").sleep(10_000).boxed(),
        Fixture::new("c_later").sleep(10).boxed(),
    ];
    let t = std::time::Instant::now();
    let s = env.pipeline(passes, cfg).run(&RunOptions::default()).unwrap();
    assert!(t.elapsed().as_secs() < 5, "the sleeper must be cancelled");
    assert_eq!(status(&s, "a_fails"), RunStatus::Failed);
    assert_eq!(status(&s, "b_sleeper"), RunStatus::Cancelled);
    assert_eq!(status(&s, "c_later"), RunStatus::Skipped);
    let m = env
        .pipeline(vec![], "")
        .store()
        .pass_dir("b_sleeper")
        .read_manifest()
        .unwrap()
        .unwrap();
    assert_eq!(m.status, PassStatus::Cancelled);
}

#[test]
fn without_stop_on_failure_independent_passes_finish() {
    let env = Env::new();
    let passes = vec![
        Fixture::new("a_fails").fail().boxed(),
        Fixture::new("b").sleep(100).boxed(),
    ];
    let s = env.pipeline(passes, "").run(&RunOptions::default()).unwrap();
    assert_eq!(status(&s, "b"), RunStatus::Ok);
}

#[test]
fn timeout_cancels_and_fails_the_pass() {
    let env = Env::new();
    let cfg = "[passes.slow.resources]\nthreads = 1\nmemory_gb = 1\ntimeout_min = 0.005\n"; // 0.3 s
    let s = env
        .pipeline(vec![Fixture::new("slow").sleep(10_000).boxed()], cfg)
        .run(&RunOptions::default())
        .unwrap();
    let r = s.pass("slow").unwrap();
    assert_eq!(r.status, RunStatus::Failed);
    assert!(r.message.as_deref().unwrap().contains("timeout"), "{r:?}");
    assert!(r.duration_ms.unwrap() < 5000);
}

#[test]
fn panic_and_verify_failure_are_failures() {
    let env = Env::new();
    let passes = vec![
        Fixture::new("p").panics().boxed(),
        Fixture::new("v").fail_verify().boxed(),
    ];
    let s = env.pipeline(passes, "").run(&RunOptions::default()).unwrap();
    assert_eq!(status(&s, "p"), RunStatus::Failed);
    assert!(s.pass("p").unwrap().message.as_deref().unwrap().contains("panic"));
    assert_eq!(status(&s, "v"), RunStatus::Failed);
    assert!(
        s.pass("v")
            .unwrap()
            .message
            .as_deref()
            .unwrap()
            .contains("verify failed: fixture")
    );
    let report = env
        .pipeline(vec![], "")
        .store()
        .pass_dir("v")
        .read_verify()
        .unwrap()
        .unwrap();
    assert!(!report.passed());
}

#[test]
fn undeclared_input_is_refused() {
    struct Sneaky;
    impl cc_pipeline::Pass for Sneaky {
        fn name(&self) -> &'static str {
            "sneaky"
        }
        fn version(&self) -> u32 {
            1
        }
        fn requires(&self) -> &[cc_core::ArtifactKind] {
            &[]
        }
        fn provides(&self) -> &[cc_core::ArtifactKind] {
            &[]
        }
        fn check_config(&self, _: &cc_core::PassSection) -> Result<(), cc_core::CoreError> {
            Ok(())
        }
        fn run(&self, ctx: &mut cc_pipeline::PassContext) -> Result<(), cc_pipeline::PassError> {
            ctx.input(cc_core::ArtifactKind("hello.a"))?;
            Ok(())
        }
        fn verify(
            &self,
            ctx: &cc_pipeline::PassContext,
        ) -> Result<cc_core::VerifyReport, cc_pipeline::PassError> {
            Ok(cc_core::VerifyReport::new(ctx.name()))
        }
    }
    let env = Env::new();
    let s = env
        .pipeline(vec![Box::new(Sneaky)], "")
        .run(&RunOptions::default())
        .unwrap();
    assert_eq!(status(&s, "sneaky"), RunStatus::Failed);
    assert!(
        s.pass("sneaky")
            .unwrap()
            .message
            .as_deref()
            .unwrap()
            .contains("did not declare")
    );
}

#[test]
fn clean_removes_only_pass_dirs() {
    let env = Env::new();
    let p = env.toy_pipeline(TOY_CONFIG, &[]);
    p.run(&RunOptions::default()).unwrap();
    std::fs::write(env.build_dir().join("keep.txt"), "x").unwrap();
    assert_eq!(p.clean(Some("hello_c")).unwrap().len(), 1);
    assert!(!env.build_dir().join(".hello_c").exists());
    p.clean(None).unwrap();
    assert!(!env.build_dir().join(".hello_a").exists() && !env.build_dir().join(".pipeline").exists());
    assert!(env.build_dir().join("keep.txt").exists());
}
