use cc_core::LoadedConfig;
use cc_pipeline::Pipeline;
use tests::{Env, TOY_CONFIG, repo_root};

#[test]
fn unknown_key_in_fixed_section_names_file_and_key() {
    let env = Env::new();
    let err = env
        .try_config("[resources]\nmemory_budget = 4\n", &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("config.toml"), "{err}");
    assert!(err.contains("memory_budget"), "{err}");
}

#[test]
fn unknown_key_in_local_file_names_the_local_file() {
    let env = Env::new();
    std::fs::write(
        env.root().join("config.local.toml"),
        "[paths]\nscratch = \"/x\"\n",
    )
    .unwrap();
    let err = env.try_config("", &[]).unwrap_err().to_string();
    assert!(
        err.contains("config.local.toml") && err.contains("scratch"),
        "{err}"
    );
}

#[test]
fn bad_type_is_rejected() {
    let env = Env::new();
    let err = env
        .try_config("[pipeline]\nmax_parallel_passes = \"two\"\n", &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("max_parallel_passes"), "{err}");
}

#[test]
fn unknown_key_in_pass_section_fails_before_running() {
    let env = Env::new();
    let cfg = env.config(
        &format!("{TOY_CONFIG}\n[passes.hello_a]\ngretting = \"hi\"\n"),
        &[],
    );
    let err = Pipeline::new(cc_passes::registry(), cfg).unwrap_err().to_string();
    assert!(err.contains("hello_a") && err.contains("gretting"), "{err}");
    assert!(!env.build_dir().exists(), "nothing may run");
}

#[test]
fn unknown_key_in_pass_resources_is_rejected() {
    let env = Env::new();
    let cfg = env.config("[passes.hello_a.resources]\nmemory = 2\n", &[]);
    let err = Pipeline::new(cc_passes::registry(), cfg).unwrap_err().to_string();
    assert!(err.contains("hello_a") && err.contains("memory"), "{err}");
}

#[test]
fn unregistered_pass_section_is_allowed() {
    let env = Env::new();
    let p = env.toy_pipeline(
        &format!("{TOY_CONFIG}\n[passes.future_pass]\nanything = 1\n"),
        &[],
    );
    assert_eq!(p.unregistered_sections(), ["future_pass"]);
}

#[test]
fn merge_order_defaults_file_local_set() {
    let env = Env::new();
    std::fs::write(
        env.root().join("config.local.toml"),
        "[resources]\nmemory_budget_gb = 16\nthreads_total = 4\n",
    )
    .unwrap();
    let c = env.config(
        "[resources]\nmemory_budget_gb = 32\n",
        &["resources.threads_total=2"],
    );
    assert_eq!(c.config.resources.memory_budget_gb, 16.0);
    assert_eq!(c.config.resources.threads_total, 2);
    assert_eq!(c.config.pipeline.max_parallel_passes, 2, "built-in default");
    assert_eq!(c.layers.len(), 3);
}

#[test]
fn tracked_config_is_valid_for_the_registry() {
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &[]).unwrap();
    assert_eq!(cfg.config.pipeline.targets, ["hello_b", "hello_c"]);
    assert_eq!(cfg.model_catalog().unwrap().len(), 25, "models/<key>/recipe.py");
    Pipeline::new(cc_passes::registry(), cfg).unwrap();
}

#[test]
fn unknown_model_key_is_rejected() {
    let cfg = LoadedConfig::load(&repo_root().join("config.toml"), &[]).unwrap();
    assert!(cfg.resolve_models(Some(&["not_a_model".to_string()])).is_err());
    assert_eq!(
        cfg.resolve_models(Some(&["resnet50".to_string()])).unwrap(),
        ["resnet50"]
    );
}
