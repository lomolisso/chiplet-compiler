//! Negative tests: every check of the toy passes' `verify.rs` fails on a mutated artifact
//! (conventions: tests), and a failed `chipc verify` invalidates the cache.

use cc_core::{CheckStatus, PassStatus};
use cc_pipeline::{RunOptions, RunStatus};
use tests::{Env, TOY_CONFIG};

fn failing(env: &Env, pass: &str) -> Vec<String> {
    let r = env.toy_pipeline(TOY_CONFIG, &[]).verify_pass(pass).unwrap();
    r.checks
        .into_iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .map(|c| c.name)
        .collect()
}

fn fresh() -> Env {
    let env = Env::new();
    assert!(
        env.toy_pipeline(TOY_CONFIG, &[])
            .run(&RunOptions::default())
            .unwrap()
            .ok
    );
    env
}

#[test]
fn intact_artifacts_verify() {
    let env = fresh();
    for p in ["hello_a", "hello_b", "hello_c"] {
        assert!(failing(&env, p).is_empty(), "{p}");
    }
}

#[test]
fn hello_a_char_count_mutation() {
    let env = fresh();
    let f = env.build_dir().join(".hello_a/out/hello.a/greeting.json");
    std::fs::write(&f, r#"{"chars": 9, "greeting": "hello"}"#).unwrap();
    let fails = failing(&env, "hello_a");
    assert!(fails.contains(&"char_count".to_string()), "{fails:?}");
    assert!(fails.contains(&"framework.outputs_match_manifest".to_string()));
    let m = env
        .toy_pipeline(TOY_CONFIG, &[])
        .store()
        .pass_dir("hello_a")
        .read_manifest()
        .unwrap()
        .unwrap();
    assert_eq!(
        m.status,
        PassStatus::Failed,
        "failed verify invalidates the cache"
    );
    let s = env
        .toy_pipeline(TOY_CONFIG, &[])
        .run(&RunOptions::default())
        .unwrap();
    assert_eq!(s.status_of("hello_a"), Some(RunStatus::Ok));
}

#[test]
fn hello_a_greeting_mutation() {
    let env = fresh();
    let f = env.build_dir().join(".hello_a/out/hello.a/greeting.json");
    std::fs::write(&f, r#"{"chars": 5, "greeting": "HELLO"}"#).unwrap();
    assert!(failing(&env, "hello_a").contains(&"greeting_matches_config".to_string()));
}

#[test]
fn hello_a_unparsable() {
    let env = fresh();
    std::fs::write(
        env.build_dir().join(".hello_a/out/hello.a/greeting.json"),
        "not json",
    )
    .unwrap();
    assert!(failing(&env, "hello_a").contains(&"greeting_parses".to_string()));
}

#[test]
fn hello_b_message_and_certificate_mutations() {
    let env = fresh();
    let f = env.build_dir().join(".hello_b/out/hello.b/message.json");
    let orig = std::fs::read_to_string(&f).unwrap();
    std::fs::write(&f, orig.replace("hello, world", "hello, there")).unwrap();
    assert!(failing(&env, "hello_b").contains(&"message_recomputed".to_string()));

    let env = fresh();
    let f = env.build_dir().join(".hello_b/out/hello.b/message.json");
    let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
    v["input_hash"] = "0000".into();
    std::fs::write(&f, v.to_string()).unwrap();
    assert!(failing(&env, "hello_b").contains(&"certificate_input_hash".to_string()));

    let env = fresh();
    std::fs::write(env.build_dir().join(".hello_b/out/hello.b/message.json"), "[]").unwrap();
    assert!(failing(&env, "hello_b").contains(&"message_parses".to_string()));
}

#[test]
fn hello_c_mutations() {
    let env = fresh();
    let f = env.build_dir().join(".hello_c/out/hello.c/echo.json");
    let orig = std::fs::read_to_string(&f).unwrap();
    std::fs::write(
        &f,
        orig.replace("\"stdout\": \"independent\\n\"", "\"stdout\": \"other\\n\""),
    )
    .unwrap();
    assert!(failing(&env, "hello_c").contains(&"stdout_is_text".to_string()));

    let env = fresh();
    std::fs::remove_file(env.build_dir().join(".hello_c/logs/tools.jsonl")).unwrap();
    assert!(failing(&env, "hello_c").contains(&"tool_call_logged".to_string()));

    let env = fresh();
    std::fs::write(env.build_dir().join(".hello_c/out/hello.c/echo.json"), "1").unwrap();
    assert!(failing(&env, "hello_c").contains(&"echo_parses".to_string()));
}
