use cc_cli::{Cli, execute};
use clap::Parser;
use tests::{Env, TOY_CONFIG};

fn parse(args: &[&str]) -> Cli {
    Cli::parse_from(args)
}

fn chipc(env: &Env, args: &[&str]) -> (i32, String) {
    let cfg = env.config_path();
    let mut full = vec!["chipc", "--config", cfg.to_str().unwrap()];
    full.extend_from_slice(args);
    let cli: Cli = parse(&full);
    let mut out = Vec::new();
    let code = execute(&cli, cc_passes::registry(), &mut out).unwrap();
    (code, String::from_utf8(out).unwrap())
}

fn env() -> Env {
    let env = Env::new();
    std::fs::write(env.config_path(), TOY_CONFIG).unwrap();
    env
}

#[test]
fn graph_prints_dot() {
    let (code, out) = chipc(&env(), &["graph"]);
    assert_eq!(code, 0);
    assert!(
        out.starts_with("digraph passes {") && out.contains("\"hello_a\" -> \"hello_b\""),
        "{out}"
    );
}

#[test]
fn run_status_verify_clean() {
    let env = env();
    let (code, out) = chipc(&env, &["status"]);
    assert_eq!(code, 0);
    // the toy targets: hello_a, hello_b, hello_c; the model passes are registered but not targets here
    assert_eq!(out.matches("not run").count(), 9, "{out}");

    let (code, out) = chipc(&env, &["run"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("ran 3, cache hits 0"), "{out}");
    let (_, out) = chipc(&env, &["run"]);
    assert!(out.contains("ran 0, cache hits 3"), "{out}");
    let (_, out) = chipc(&env, &["run", "--from", "hello_b"]);
    assert!(out.contains("ran 1, cache hits 2"), "{out}");

    let (_, out) = chipc(&env, &["status"]);
    let rows: Vec<&str> = out.lines().skip(1).filter(|r| r.starts_with("hello_")).collect();
    assert_eq!(rows.len(), 3);
    assert!(
        rows.iter()
            .all(|r| r.split_whitespace().nth(1) == Some("ok") && r.contains(" yes ")),
        "{out}"
    );

    let (code, out) = chipc(&env, &["verify", "hello_b"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("hello_b: verified"));

    let (code, out) = chipc(&env, &["run", "--set", "passes.hello_b.fail=true"]);
    assert_eq!(code, 1, "{out}");

    let cli = parse(&["chipc", "--config", env.config_path().to_str().unwrap(), "clean"]);
    assert!(
        execute(&cli, cc_passes::registry(), &mut Vec::new()).is_err(),
        "full clean needs --yes"
    );
    let (code, _) = chipc(&env, &["clean", "--yes"]);
    assert_eq!(code, 0);
    assert!(!env.build_dir().join(".hello_a").exists());
}
