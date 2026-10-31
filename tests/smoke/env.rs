//! smoke checks: `chipc doctor --dry` builds the right commands, the image build inputs come from
//! `config.toml`, the submodules are shallow and pinned, the Dockerfile lints. No Docker, no EDA.

use std::process::Command;

use cc_cli::{Cli, execute};
use cc_pipeline::env::doctor::gitmodules_shallow;
use clap::Parser;
use tests::{Env, TOY_CONFIG, repo_root};

fn chipc(config: &std::path::Path, args: &[&str]) -> (i32, String) {
    let mut full = vec!["chipc", "--config", config.to_str().unwrap()];
    full.extend_from_slice(args);
    let mut out = Vec::new();
    let code = execute(&Cli::parse_from(&full), cc_passes::registry(), &mut out).unwrap();
    (code, String::from_utf8(out).unwrap())
}

/// A config whose machine paths point at the header-only fixtures in tests/fixtures/liberty/.
fn doctor_env() -> Env {
    let env = Env::new();
    let fx = repo_root().join("tests/fixtures/liberty");
    let toggle = env.root().join("eda-toggle.sh");
    std::fs::write(&toggle, "eda-run() { \"$@\"; }\n").unwrap();
    let text = format!(
        "{TOY_CONFIG}\n[paths]\neda_toggle = \"{}\"\nscratch_dir = \"{}\"\nnangate45_lib = \"{}\"\n\
         nangate45_lef = \"{}\"\nasap7_nldm_dir = \"{}\"\n",
        toggle.display(),
        env.root().join("scratch").display(),
        fx.join("nangate45/stdcells.lib").display(),
        fx.join("nangate45/stdcells.lef").display(),
        fx.join("asap7_nldm").display(),
    );
    std::fs::write(env.config_path(), text).unwrap();
    env
}

#[test]
fn doctor_dry_prints_capped_commands_and_writes_nothing() {
    let env = doctor_env();
    let (code, out) = chipc(&env.config_path(), &["doctor", "--dry"]);
    assert_eq!(code, 0, "{out}");
    // EDA calls use the verified eda-run form (eda_environment.md §2).
    assert!(
        out.contains(r#"[eda] bash --norc -i -c '. "$EDA_TOGGLE"; eda-run "$@"' eda-run dc_shell -version"#),
        "{out}"
    );
    assert!(out.contains("eda-run lc_shell -f make_db.tcl"), "{out}");
    assert!(out.contains("eda-run dc_shell -no_home_init -f syn.tcl"), "{out}");
    assert!(
        out.contains("eda-run lec -XL -nogui -nolicwait -dofile exit.do"),
        "{out}"
    );
    // Container calls carry hard caps and no network.
    let selftest = out
        .lines()
        .find(|l| l.contains("cc-selftest /work/out"))
        .expect("selftest call");
    for flag in [
        "--memory 4096m",
        "--memory-swap 4096m",
        "--cpus 2",
        "--pids-limit 1024",
        "--network none",
    ] {
        assert!(selftest.contains(flag), "{flag} missing: {selftest}");
    }
    // The fixture Liberty set is found (LVT and .7z files are ignored).
    assert!(
        out.contains("pass  pdk.asap7") && out.contains("RVT TT: 5 libraries"),
        "{out}"
    );
    assert!(out.contains("dry run:"), "{out}");
    assert!(!env.build_dir().join(".env").exists(), "--dry writes nothing");
}

#[test]
fn build_args_come_from_config_and_match_dockerfile_args() {
    let (code, out) = chipc(&repo_root().join("config.toml"), &["doctor", "--build-args"]);
    assert_eq!(code, 0, "{out}");
    let dockerfile = std::fs::read_to_string(repo_root().join("docker/Dockerfile")).unwrap();
    let mut n = 0;
    for line in out.lines() {
        let (k, v) = line.split_once('=').unwrap();
        assert!(!v.is_empty(), "{k} is empty");
        if !k.starts_with("CAP_") {
            n += 1;
            assert!(
                dockerfile
                    .lines()
                    .any(|l| l.trim() == format!("ARG {k}") || l.trim().starts_with(&format!("ARG {k}="))),
                "{k} is not an ARG of docker/Dockerfile"
            );
        }
    }
    assert!(n >= 20, "{out}");
}

#[test]
fn submodules_are_shallow_and_match_the_config_pins() {
    let root = repo_root();
    let gm = std::fs::read_to_string(root.join(".gitmodules")).unwrap();
    let cfg = cc_core::LoadedConfig::load(&root.join("config.toml"), &[]).unwrap();
    let t = &cfg.config.tools;
    for (name, url, pin) in [
        (
            "grami",
            "https://github.com/ehab-abdelhamid/GraMi.git",
            &t.grami.commit,
        ),
        (
            "allo",
            "https://github.com/cornell-zhang/allo.git",
            &t.allo.commit,
        ),
        ("xls", "https://github.com/google/xls.git", &t.xls.commit),
    ] {
        let path = format!("third_party/{name}");
        assert!(gitmodules_shallow(&gm, &path), "{path} not shallow");
        assert!(gm.contains(&format!("url = {url}")), "{path} url");
        assert!(
            pin.len() == 40 && pin.chars().all(|c| c.is_ascii_hexdigit()),
            "{name} pin {pin}"
        );
        // The commit the superproject records for the submodule (works without checking it out).
        let out = Command::new("git")
            .args(["-C", root.to_str().unwrap(), "ls-tree", "HEAD", &path])
            .output()
            .unwrap();
        let rec = String::from_utf8_lossy(&out.stdout);
        if rec.trim().is_empty() {
            continue; // not committed yet (working tree only)
        }
        assert_eq!(rec.split_whitespace().nth(2), Some(pin.as_str()), "{path}: {rec}");
    }
    assert!(
        t.xls.tag.ends_with(&format!("-g{}", &t.xls.commit[..9])),
        "XLS tag names its commit"
    );
}

#[test]
fn dockerfile_lints_clean_when_hadolint_is_installed() {
    let Ok(out) = Command::new("hadolint")
        .args(["--config", "docker/.hadolint.yaml", "docker/Dockerfile"])
        .current_dir(repo_root())
        .output()
    else {
        eprintln!("hadolint not on PATH: skipped (CI installs it)");
        return;
    };
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
