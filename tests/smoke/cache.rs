use cc_pipeline::{RunOptions, RunStatus};
use tests::{Env, TOY_CONFIG};

fn statuses(s: &cc_pipeline::RunSummary) -> Vec<(String, RunStatus)> {
    let mut v: Vec<_> = s.passes.iter().map(|p| (p.pass.clone(), p.status)).collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

fn st(a: RunStatus, b: RunStatus, c: RunStatus) -> Vec<(String, RunStatus)> {
    vec![
        ("hello_a".into(), a),
        ("hello_b".into(), b),
        ("hello_c".into(), c),
    ]
}

use RunStatus::{CacheHit as H, Ok as R};

#[test]
fn key_stable_under_toml_key_reordering() {
    let env = Env::new();
    let a = format!("{TOY_CONFIG}\n[passes.hello_a]\ngreeting = \"x\"\nsleep_ms = 0\n");
    let b = format!("[passes.hello_a]\nsleep_ms = 0\ngreeting = \"x\"\n{TOY_CONFIG}");
    let s1 = env.toy_pipeline(&a, &[]).run(&RunOptions::default()).unwrap();
    assert_eq!(statuses(&s1), st(R, R, R));
    let s2 = env.toy_pipeline(&b, &[]).run(&RunOptions::default()).unwrap();
    assert_eq!(statuses(&s2), st(H, H, H));
    assert_eq!(
        s1.pass("hello_a").unwrap().cache_key,
        s2.pass("hello_a").unwrap().cache_key
    );
}

#[test]
fn value_change_reruns_pass_and_dependents_only() {
    let env = Env::new();
    env.toy_pipeline(TOY_CONFIG, &[])
        .run(&RunOptions::default())
        .unwrap();
    let s = env
        .toy_pipeline(TOY_CONFIG, &["passes.hello_a.greeting=hi"])
        .run(&RunOptions::default())
        .unwrap();
    assert_eq!(statuses(&s), st(R, R, H));
}

#[test]
fn resources_do_not_change_the_key() {
    let env = Env::new();
    env.toy_pipeline(TOY_CONFIG, &[])
        .run(&RunOptions::default())
        .unwrap();
    let s = env
        .toy_pipeline(
            TOY_CONFIG,
            &[
                "passes.hello_a.resources.threads=2",
                "passes.hello_a.resources.memory_gb=2",
            ],
        )
        .run(&RunOptions::default())
        .unwrap();
    assert_eq!(statuses(&s), st(H, H, H));
}

#[test]
fn tampered_output_is_not_a_cache_hit() {
    let env = Env::new();
    env.toy_pipeline(TOY_CONFIG, &[])
        .run(&RunOptions::default())
        .unwrap();
    std::fs::write(env.build_dir().join(".hello_c/out/hello.c/echo.json"), "{}").unwrap();
    let s = env
        .toy_pipeline(TOY_CONFIG, &[])
        .run(&RunOptions::default())
        .unwrap();
    assert_eq!(statuses(&s), st(H, H, R));
}
