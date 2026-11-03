//! Deep check: `chipc doctor` (the `.db` files and the merged ASAP7 Liberty) into the build area's
//! ephemeral `runs/deep-characterize/`, then `characterize` on a reduced catalog (`add` and `multiply` at
//! 8 bits and f32, their accumulators, a 2-input mux, a register, a 16-from-8 split multiplier) on both
//! kits at both timing points, through each flow: Design Compiler + Formality, Genus + Conformal, Yosys +
//! ABC (without the f32 and split rows, which its equivalence flow cannot prove). Every check of `verify` passes for each, a golden snapshot keeps the keys, and the directory is
//! removed when the test passes.

use cc_core::runner::Runners;
use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::characterize::LIBRARY;
use cc_passes::characterize::json::Library;
use cc_pipeline::env::{DoctorOptions, doctor};
use cc_pipeline::{Pipeline, RunOptions, RunStatus};
use tests::repo_root;

const REDUCED: &[&str] = &[
    "passes.characterize.ops=[\"add\", \"multiply\"]",
    "passes.characterize.int_widths=[8]",
    "passes.characterize.float_formats=[\"f32\"]",
    "passes.characterize.mux_inputs=[2]",
    "passes.characterize.mux_widths=[8]",
    "passes.characterize.register_widths=[8]",
    "passes.characterize.split_multipliers=[[16, 8]]",
];

#[test]
fn characterize_reduced_catalog_on_every_flow() {
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-characterize");
    let base = vec![format!("paths.build_dir={}", build.display())];
    let load = |extra: &[String]| {
        let mut sets = base.clone();
        sets.extend(REDUCED.iter().map(|s| s.to_string()));
        sets.extend(extra.iter().cloned());
        LoadedConfig::load(&repo_root().join("config.toml"), &sets).unwrap()
    };
    let cfg = load(&[]);
    assert!(
        cfg.config.paths.eda_toggle.is_some(),
        "deep tests need config.local.toml"
    );

    let mut text = Vec::new();
    let env = doctor(
        &cfg,
        &Runners::from_config(&cfg),
        DoctorOptions::default(),
        &mut text,
    )
    .unwrap();
    for name in ["pdk.asap7_merged", "pdk.db", "dc.nangate45", "dc.asap7"] {
        let c = env.checks.iter().find(|c| c.name == name).unwrap();
        assert_eq!(
            c.status,
            CheckStatus::Pass,
            "{name}: {}",
            String::from_utf8_lossy(&text)
        );
    }

    let mut digest = Vec::new();
    for tool in ["dc", "genus", "yosys"] {
        // Yosys' equivalence flow proves integer and structural primitives but not the float ones or a
        // split multiplier within its 600 s (docs/passes/characterize.md §8): it runs without them.
        let mut extra = vec![format!("passes.characterize.tool=\"{tool}\"")];
        if tool == "yosys" {
            extra.push("passes.characterize.float_formats=[]".into());
            extra.push("passes.characterize.split_multipliers=[]".into());
        }
        let cfg = load(&extra);
        let p = Pipeline::new(cc_passes::registry(), cfg).unwrap();
        let s = p
            .run(&RunOptions {
                targets: vec!["characterize".into()],
                ..RunOptions::default()
            })
            .unwrap();
        let dir = p.store().pass_dir("characterize");
        let report = dir.read_verify().unwrap().unwrap();
        let failed: Vec<_> = report
            .checks
            .iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| (c.name.clone(), c.evidence.notes.clone()))
            .collect();
        assert!(failed.is_empty(), "{tool}: {failed:#?}");
        assert!(
            matches!(
                s.status_of("characterize"),
                Some(RunStatus::Ok | RunStatus::CacheHit)
            ),
            "{tool}: {s:?}"
        );
        let lib: Library = dir.kind(LIBRARY).read_json("library.json").unwrap();
        // Proven, or out of time with no failing point (the row says so); never a failing point.
        let open: Vec<String> = lib
            .rows
            .iter()
            .filter(|r| !r.lec.equivalent)
            .map(|r| format!("{} {} {}", r.key, r.kit, r.target))
            .collect();
        assert!(
            lib.rows
                .iter()
                .all(|r| r.lec.failing == 0 && (r.lec.equivalent || r.inconclusive)),
            "{tool}: {open:?}"
        );
        // A wider multiply costs more than a narrower one, and a split 16-bit multiply has its pieces.
        for kit in ["nangate45", "asap7"] {
            let area = |key: &str| {
                lib.rows
                    .iter()
                    .find(|r| r.key == key && r.kit == kit && r.target == "clock")
                    .and_then(|r| r.measured.area_um2)
                    .unwrap()
            };
            assert!(area("multiply@i8") > area("add@i8"), "{tool} {kit}");
            if tool != "yosys" {
                assert!(area("multiply@f32") > area("add@f32"), "{tool} {kit}");
            }
            if tool == "dc" {
                let split = lib
                    .rows
                    .iter()
                    .find(|r| r.key == "split@w16n8" && r.kit == kit && r.target == "clock")
                    .unwrap();
                // The three piece instances (u<i>_<j>), besides any DesignWare block of the sum.
                let pieces = split
                    .measured
                    .children
                    .keys()
                    .filter(|k| k.starts_with('u'))
                    .count();
                assert_eq!(pieces, 3, "{kit}: {:?}", split.measured.children);
            }
            let fast = lib
                .rows
                .iter()
                .find(|r| r.key == "multiply@i8" && r.kit == kit && r.target == "fast")
                .unwrap();
            let clock = lib
                .rows
                .iter()
                .find(|r| r.key == "multiply@i8" && r.kit == kit && r.target == "clock")
                .unwrap();
            assert!(
                fast.measured.delay_ps.unwrap() <= clock.measured.delay_ps.unwrap(),
                "{tool} {kit}"
            );
        }
        let mut keys: Vec<String> = lib
            .rows
            .iter()
            .map(|r| format!("{} {} {}", r.key, r.kit, r.target))
            .collect();
        keys.sort();
        digest.push(serde_json::json!({
            "tool": tool,
            "rows": keys.len(),
            "inconclusive": open,
            "keys": keys,
        }));
    }
    insta::assert_json_snapshot!("characterize_reduced_keys", digest);
    std::fs::remove_dir_all(&build).unwrap();
}
