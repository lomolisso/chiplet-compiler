//! Deep checks of `characterize`.
//!
//! - `chipc doctor` (the `.db` files and the merged ASAP7 Liberty) into the build area's ephemeral
//!   `runs/deep-characterize/`, then `characterize` on a reduced catalog (`add` and `multiply` at 8 bits and
//!   f32, their accumulators, a 2-input mux, a register, a 16-from-8 split multiplier) on both kits at both
//!   timing points, through every synthesis flow (the `SynthFlow` conformance suite: the same catalog,
//!   every check of `verify` green): Design Compiler without and then with Formality (the second run
//!   reuses every synthesis and runs only the check), and Yosys + ABC. A golden snapshot keeps the keys,
//!   and the directory is removed when the test passes.
//! - The job directories this code computes for the default configuration are the ones of the standing
//!   Design Compiler library in the build area's `build/.characterize/` (read only), so a library
//!   characterized by an earlier version is reused.

use cc_core::runner::Runners;
use cc_core::{CheckStatus, LoadedConfig};
use cc_passes::characterize::LIBRARY;
use cc_passes::characterize::json::{Library, Summary};
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
    for (tool, lec) in [("dc", false), ("dc", true), ("yosys", false)] {
        let extra = vec![
            format!("passes.characterize.tool=\"{tool}\""),
            format!("passes.characterize.lec={lec}"),
        ];
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
        let summary: Summary = dir.kind(LIBRARY).read_json("summary.json").unwrap();
        // Proven, or out of time with no failing point (the row says so); never a failing point.
        let open: Vec<String> = lib
            .rows
            .iter()
            .filter(|r| r.lec.as_ref().is_some_and(|l| !l.equivalent))
            .map(|r| format!("{} {} {}", r.key, r.kit, r.target))
            .collect();
        assert!(
            lib.rows.iter().all(|r| match &r.lec {
                Some(l) => lec && l.failing == 0 && (l.equivalent || r.inconclusive),
                None => !lec,
            }),
            "{tool}: {open:?}"
        );
        if tool == "dc" && lec {
            // Turning the check on reran only Formality: every synthesis was reused.
            let log = std::fs::read_to_string(dir.logs_dir().join("tools.jsonl")).unwrap_or_default();
            assert!(!log.contains("\"dc_shell\""), "{tool}: synthesis ran again");
            assert!(summary.jobs_run > 0, "{tool}: {summary:?}");
        }
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
            assert!(area("multiply@f32") > area("add@f32"), "{tool} {kit}");
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
            "lec": lec,
            "rows": keys.len(),
            "inconclusive": open,
            "keys": keys,
        }));
    }
    insta::assert_json_snapshot!("characterize_reduced_keys", digest);
    std::fs::remove_dir_all(&build).unwrap();
}

/// The standing Design Compiler library (`<build area>/build/.characterize`, or `CC_STANDING_BUILD`'s): the
/// RTL and synthesis job directories that this code computes for the default configuration (Design
/// Compiler, the equivalence check off) are exactly the ones there. Nothing is written; without the
/// standing library the test has nothing to compare and passes with a note.
#[test]
fn characterize_job_keys_match_the_standing_library() {
    use cc_passes::characterize::{CharacterizeConfig, catalog, flow};
    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::env::var("CC_STANDING_BUILD")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::Path::new(&area).join("build"));
    let eda = build.join(".characterize/eda");
    if !eda.is_dir() {
        eprintln!("no standing library at {}: nothing to compare", eda.display());
        return;
    }
    let lc = LoadedConfig::load(
        &repo_root().join("config.toml"),
        &[format!("paths.build_dir={}", build.display())],
    )
    .unwrap();
    let section = lc.pass_section("characterize").unwrap();
    let cfg: CharacterizeConfig = section.parse().unwrap();
    assert!(!cfg.lec && cfg.tool == "dc", "the default configuration");
    let d = &lc.config.tools.docker;
    let image = format!("{}:{}", d.image, d.tag);
    let cat = catalog::catalog(&cfg).unwrap();
    let rtl_dir = eda.join(flow::rtl_dir_name(&image, &cat));
    assert!(
        rtl_dir.join("rtl.done").is_file(),
        "{} is not in the standing library",
        rtl_dir.display()
    );
    let pdks: Vec<_> = cfg
        .pdks
        .iter()
        .map(|k| cc_core::pdk::load(&lc, k).unwrap())
        .collect();
    let env = cc_core::ArtifactStore::new(build.clone()).pass_dir(cc_pipeline::env::doctor::ENV_DIR);
    let kits = flow::kits_of(&env, &pdks, &cfg).unwrap();
    let designs = flow::designs(&cat);
    let jobs = flow::job_list(
        &cfg,
        &kits,
        &designs,
        &rtl_dir,
        &build.join(".characterize"),
        &image,
        flow::session_caps(&section.resources, &cfg),
    )
    .unwrap();
    let want: std::collections::BTreeSet<String> = jobs
        .iter()
        .map(|j| j.dir.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let have: std::collections::BTreeSet<String> = std::fs::read_dir(&eda)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with("rtl-"))
        .collect();
    let missing: Vec<&String> = want.difference(&have).collect();
    assert!(
        missing.is_empty(),
        "{} job directories are not in the standing library: {missing:?}",
        missing.len()
    );
    assert!(
        jobs.iter().all(|j| j.dir.join("synth.done").is_file()),
        "every job is finished"
    );
    eprintln!("{} job directories found in {}", want.len(), eda.display());
}

/// Deterministic test vectors for `rescale`: edge cases and pseudo-random operands with scales over the
/// exponent range that matters (right shifts, ties, left shifts, saturation, subnormal and zero scales).
fn rescale_vectors(n: u32) -> Vec<(i64, u32)> {
    let lo = -(1i64 << (n - 1));
    let hi = (1i64 << (n - 1)) - 1;
    let mut v = Vec::new();
    for x in [0, 1, -1, 2, -2, 3, -3, 5, -5, lo, hi, lo + 1, hi - 1] {
        for s in [
            0.5f32,
            0.25,
            0.75,
            1.5,
            0.001,
            2f32.powi(-40),
            2f32.powi(23),
            2f32.powi(24),
            2f32.powi(30),
            2f32.powi(31),
            0.0,
            -0.0,
            -0.5,
            -0.75,
            f32::INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
        ] {
            v.push((x, s.to_bits()));
        }
        v.push((x, 1));
    }
    let mut st: u64 = 0x9e37_79b9_7f4a_7c15 ^ u64::from(n);
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    for _ in 0..3000 {
        let r = next();
        let x = ((r as i64) >> (64 - n)).clamp(lo, hi);
        let e = 90 + (next() % 70) as u32;
        let s = ((next() & 1) as u32) << 31 | e << 23 | (next() as u32 & 0x7f_ffff);
        v.push((x, s));
    }
    v
}

#[test]
fn rescale_rtl_matches_its_reference() {
    let env = tests::Env::new();
    let cfg: toml::Value =
        toml::from_str(&std::fs::read_to_string(repo_root().join("config.toml")).unwrap()).unwrap();
    let tag = cfg["tools"]["docker"]["tag"].as_str().unwrap().to_string();
    let lc = env.config("[models]\nkeys = []\n", &[&format!("tools.docker.tag={tag}")]);
    let dir = env.root().join("rescale");
    std::fs::create_dir_all(&dir).unwrap();
    let cat =
        cc_passes::characterize::catalog::catalog(&cc_passes::characterize::CharacterizeConfig::default())
            .unwrap();
    let runner = cc_core::runner::BoundRunner::new(
        Runners::from_config(&lc).docker,
        cc_core::runner::Caps {
            memory_gb: 2.0,
            cpus: 1.0,
            timeout_s: 300.0,
            pids_limit: 64,
        },
        cc_core::runner::ToolLog::new("rescale-rtl", dir.join("logs")),
        cc_core::CancelToken::new(),
    );
    for n in [8u32, 16, 32] {
        let p = cat.iter().find(|p| p.key == format!("rescale@i{n}")).unwrap();
        let Some(cc_passes::characterize::catalog::Source::Verilog(v)) = &p.source else {
            panic!("rescale@i{n}: no Verilog");
        };
        std::fs::write(dir.join(format!("rescale_i{n}.v")), v).unwrap();
        let vs = rescale_vectors(n);
        let mut hex = String::new();
        for &(x, s) in &vs {
            let want = cc_smt::ops::rescale_reference_int(x, s) as u32;
            hex.push_str(&format!("{:08x}{s:08x}{want:08x}\n", x as u32));
        }
        std::fs::write(dir.join(format!("vec_i{n}.hex")), hex).unwrap();
        let tb = format!(
            "module tb;\n  reg [95:0] mem [0:{last}];\n  reg [31:0] x, s, want;\n  wire [31:0] q;\n  integer i, bad;\n  rescale_core_i{n} dut(.a(x[{nh}:0]), .b(s), .q(q));\n  initial begin\n    $readmemh(\"/w/vec_i{n}.hex\", mem);\n    bad = 0;\n    for (i = 0; i <= {last}; i = i + 1) begin\n      {{x, s, want}} = mem[i];\n      #1;\n      if (q !== want) begin\n        bad = bad + 1;\n        if (bad < 10) $display(\"MISMATCH x=%h s=%h q=%h want=%h\", x, s, q, want);\n      end\n    end\n    $display(\"DONE %0d %0d\", {count}, bad);\n  end\nendmodule\n",
            last = vs.len() - 1,
            count = vs.len(),
            nh = n - 1
        );
        std::fs::write(dir.join(format!("tb_i{n}.v")), tb).unwrap();
        let inv = cc_core::runner::Invocation::new("iverilog", "sh")
            .args([
                "-c".to_string(),
                format!(
                    "iverilog -g2012 -s tb -o /tmp/tb.vvp /w/rescale_i{n}.v /w/tb_i{n}.v && vvp -n /tmp/tb.vvp"
                ),
            ])
            .mount(cc_core::runner::Mount {
                src: dir.clone(),
                dst: "/w".into(),
                readonly: true,
            });
        let o = runner.run(&inv).unwrap();
        let out = std::fs::read_to_string(&o.stdout).unwrap();
        let err = std::fs::read_to_string(&o.stderr).unwrap_or_default();
        assert!(
            out.contains(&format!("DONE {} 0", vs.len())),
            "rescale_i{n}: {out}\n{err}"
        );
    }
}

/// IP cores through both flows: the RTL step generates HardFloat (Verilog), FloPoCo (VHDL) and XLS DSLX
/// library cores in the image; Verilog wrappers instantiate them (dependencies followed transitively) and
/// DesignWare components (the synthetic library linked only for those jobs); Yosys (GHDL for the VHDL)
/// and Design Compiler synthesize the wrappers on Nangate45 at its clock, and every row passes the
/// cheap checks. Writes `<build area>/runs/deep-ipcores/`, removed when the test passes.
#[test]
fn ip_cores_generate_and_synthesize_in_their_flows() {
    use std::collections::BTreeMap;

    use cc_core::adapters::{CoreOffer, CoreSource, FlowRunners, SynthJob};
    use cc_passes::adapters::IP_PROVIDERS;
    use cc_passes::characterize::catalog::{Kind, Primitive, Source};
    use cc_passes::characterize::{CharacterizeConfig, flow};
    use cc_passes::ip::verilog_instance;

    let area = std::env::var("CC_BUILD_AREA").unwrap_or_else(|_| {
        format!(
            "/scratch/{}/chiplet-compiler",
            std::env::var("USER").unwrap_or_default()
        )
    });
    let build = std::path::Path::new(&area).join("runs").join("deep-ipcores");
    let _ = std::fs::remove_dir_all(&build);
    // Nangate45 only: a manifests directory with its manifest alone.
    let pdks = build.join("pdks");
    std::fs::create_dir_all(&pdks).unwrap();
    std::fs::copy(
        repo_root().join("pdks/nangate45.toml"),
        pdks.join("nangate45.toml"),
    )
    .unwrap();
    let lc = LoadedConfig::load(
        &repo_root().join("config.toml"),
        &[
            format!("paths.build_dir={}", build.display()),
            format!("paths.pdk_manifests={}", pdks.display()),
        ],
    )
    .unwrap();
    assert!(
        lc.config.paths.eda_toggle.is_some(),
        "deep tests need config.local.toml"
    );
    let runners = Runners::from_config(&lc);
    let mut text = Vec::new();
    let env = doctor(&lc, &runners, DoctorOptions::default(), &mut text).unwrap();
    for name in [
        "pdk.db",
        "dc.nangate45",
        "image.selftest.hardfloat",
        "image.selftest.flopoco_ghdl",
    ] {
        let c = env.checks.iter().find(|c| c.name == name).unwrap();
        assert_eq!(
            c.status,
            CheckStatus::Pass,
            "{name}: {}",
            String::from_utf8_lossy(&text)
        );
    }

    // The catalog: cores (generated or DesignWare) and wrappers around them.
    let offer = |p: &str, op: &str, e: u32, s: u32| -> CoreOffer {
        IP_PROVIDERS
            .by_name(p)
            .unwrap()
            .core(op, e, s)
            .unwrap_or_else(|| panic!("{p} {op} ({e}, {s})"))
    };
    let prim = |id: &str, deps: Vec<String>, source: Source, libraries: Vec<String>| Primitive {
        key: id.into(),
        id: id.into(),
        kind: Kind::Op,
        op: "core".into(),
        variant: None,
        family: "float".into(),
        types: vec!["f16".into()],
        width: 16,
        provisional: false,
        pending: None,
        deps,
        libraries,
        clocked: false,
        source: Some(source),
    };
    let mut cat: Vec<Primitive> = Vec::new();
    let core = |o: &CoreOffer, cat: &mut Vec<Primitive>| -> (String, Vec<String>) {
        match &o.source {
            CoreSource::Generated { generator } => {
                if !cat.iter().any(|p| p.id == o.instance.module) {
                    cat.push(prim(
                        &o.instance.module,
                        Vec::new(),
                        Source::Generated(generator.clone()),
                        Vec::new(),
                    ));
                }
                (o.instance.module.clone(), Vec::new())
            }
            CoreSource::Library { name } => (String::new(), vec![name.clone()]),
        }
    };
    // A wrapper with IEEE operands `a0`, `a1` and result `z` (or the compare outputs) around a chain of
    // single-operand cores (FloPoCo's from_ieee, op, to_ieee) or one core.
    let wrapper = |id: &str, chain: &[CoreOffer], width_in: u32, width_out: u32, cat: &mut Vec<Primitive>| {
        let mut deps = Vec::new();
        let mut libs = Vec::new();
        let mut body = String::new();
        let arity = chain[0]
            .instance
            .ports
            .values()
            .filter(|v| v.starts_with('a'))
            .count();
        let mut ports: Vec<String> = (0..arity)
            .map(|i| format!("input wire [{}:0] a{i}", width_in - 1))
            .collect();
        let compare = chain[0].instance.ports.values().any(|v| v == "lt");
        if compare {
            ports.extend(["output wire lt", "output wire eq", "output wire gt"].map(String::from));
        } else {
            ports.push(format!("output wire [{}:0] z", width_out - 1));
        }
        for (i, o) in chain.iter().enumerate() {
            let (dep, lib) = core(o, cat);
            if !dep.is_empty() && !deps.contains(&dep) {
                deps.push(dep);
            }
            libs.extend(lib);
            let last = i + 1 == chain.len();
            if !last {
                // FloPoCo's format between its operators: exception bits, sign, exponent, significand.
                let w = o.instance.exp_width + o.instance.sig_width + 3;
                body.push_str(&format!("  wire [{}:0] t{i};\n", w - 1));
            }
            let src = |x: &str| -> String {
                match (x, i) {
                    ("z", _) if last => "z".into(),
                    ("z", _) => format!("t{i}"),
                    (a, 0) => a.to_string(),
                    ("a0", _) => format!("t{}", i - 1),
                    (other, _) => other.to_string(),
                }
            };
            body.push_str(&format!(
                "  {}\n",
                verilog_instance(&o.instance, &format!("u{i}"), src)
            ));
        }
        let v = format!("module {id}({});\n{body}endmodule\n", ports.join(", "));
        cat.push(prim(id, deps, Source::Verilog(v), libs));
    };
    // Open flow.
    wrapper(
        "w_hf_add_f16",
        &[offer("hardfloat", "add", 5, 10)],
        16,
        16,
        &mut cat,
    );
    wrapper(
        "w_hf_cmp_bf16",
        &[offer("hardfloat", "compare", 8, 7)],
        16,
        1,
        &mut cat,
    );
    wrapper(
        "w_fp_exp_f16",
        &[
            offer("flopoco", "from_ieee", 5, 10),
            offer("flopoco", "exp", 5, 10),
            offer("flopoco", "to_ieee", 5, 10),
        ],
        16,
        16,
        &mut cat,
    );
    // Industry flow: DesignWare, and HardFloat where DesignWare has no component.
    wrapper(
        "w_dw_add_f16",
        &[offer("designware", "add", 5, 10)],
        16,
        16,
        &mut cat,
    );
    wrapper(
        "w_dw_exp_f16",
        &[offer("designware", "exp", 5, 10)],
        16,
        16,
        &mut cat,
    );
    wrapper(
        "w_hf_cvt_f32_bf16",
        &[offer("hardfloat", "convert.e8m7", 8, 23)],
        32,
        16,
        &mut cat,
    );
    // Generated only (no synthesis here): an iterative HardFloat divider and XLS's f32 sine.
    for o in [offer("hardfloat", "div", 5, 10), offer("xls_math", "sin", 8, 23)] {
        core(&o, &mut cat);
    }

    let d = &lc.config.tools.docker;
    let image = format!("{}:{}", d.image, d.tag);
    let caps = cc_core::runner::Caps {
        memory_gb: 4.0,
        cpus: 2.0,
        timeout_s: 3600.0,
        pids_limit: 1024,
    };
    let bound = |r: std::sync::Arc<dyn cc_core::runner::ToolRunner>, what: &str| {
        cc_core::runner::BoundRunner::new(
            r,
            caps,
            cc_core::runner::ToolLog::new(what, build.join("logs")),
            cc_core::CancelToken::new(),
        )
    };
    let fr = FlowRunners {
        eda: bound(runners.eda.clone(), "ipcores-eda"),
        docker: bound(runners.docker.clone(), "ipcores-docker"),
    };
    let pass_dir = build.join(".characterize");
    let rtl_dir = pass_dir.join("eda").join(flow::rtl_dir_name(&image, &cat));
    flow::rtl_in(&fr.docker, &rtl_dir, &cat).unwrap();
    let read = |f: &str| std::fs::read_to_string(rtl_dir.join(f)).unwrap();
    let div = read("hardfloat_e5m10_div.v");
    for port in ["clock", "start", "done", "ready"] {
        assert!(
            div.contains(&format!(" {port},")) || div.contains(&format!(" {port}\n")),
            "{port}"
        );
    }
    assert!(read("xls_e8m23_sin.v").contains("module xls_e8m23_sin("));
    assert!(read("flopoco_e5m10_exp.vhdl").contains("entity flopoco_e5m10_exp is"));

    let by_id: BTreeMap<&str, &Primitive> = cat.iter().map(|p| (p.id.as_str(), p)).collect();
    let all = flow::designs(&cat);
    let pdk = cc_core::pdk::load(&lc, "nangate45").unwrap();
    let env_dir = cc_core::ArtifactStore::new(build.clone()).pass_dir(cc_pipeline::env::doctor::ENV_DIR);
    for (tool, tops) in [
        ("yosys", &["w_hf_add_f16", "w_hf_cmp_bf16", "w_fp_exp_f16"][..]),
        ("dc", &["w_dw_add_f16", "w_dw_exp_f16", "w_hf_cvt_f32_bf16"][..]),
    ] {
        let cfg = CharacterizeConfig {
            tool: tool.into(),
            pdks: vec!["nangate45".into()],
            targets: vec!["clock".into()],
            ..CharacterizeConfig::default()
        };
        let kits = flow::kits_of(&env_dir, std::slice::from_ref(&pdk), &cfg).unwrap();
        let designs: Vec<_> = all
            .iter()
            .filter(|d| tops.contains(&d.id.as_str()))
            .cloned()
            .collect();
        let fp = designs.iter().find(|d| d.id == "w_fp_exp_f16");
        if let Some(fp) = fp {
            assert_eq!(
                fp.files.iter().filter(|f| f.ends_with(".vhdl")).count(),
                3,
                "{:?}",
                fp.files
            );
        }
        let jobs = flow::job_list(&cfg, &kits, &designs, &rtl_dir, &pass_dir, &image, caps).unwrap();
        let out_root = build.join(format!("out-{tool}"));
        for j in &jobs {
            if tool == "dc" {
                let dw = j
                    .synth
                    .iter()
                    .filter(|(_, t)| t.contains("dw_foundation.sldb"))
                    .count();
                let with_dw = j.designs.iter().filter(|d| !d.libraries.is_empty()).count();
                assert_eq!(dw, with_dw, "only DesignWare jobs link its library");
            }
            let job = SynthJob {
                kit: &kits[j.kit].0,
                designs: &j.designs,
                period_ps: j.period_ps,
                caps,
                cache_dir: &pass_dir.join("alib"),
                lec: None,
            };
            flow::run_job_with(&fr, &cfg, &job, j, &rtl_dir).unwrap();
            for (row, problems) in flow::rows(&cfg, &kits[j.kit].0, j, &by_id, &out_root).unwrap() {
                assert!(problems.is_empty(), "{tool} {}: {problems:?}", row.id);
                let a = row.measured.area_um2.unwrap_or(0.0);
                assert!(a > 50.0, "{tool} {}: area {a}", row.id);
                eprintln!("{tool} {}: {a:.1} um2, {:?} ps", row.id, row.measured.delay_ps);
            }
        }
    }
    std::fs::remove_dir_all(&build).unwrap();
}
