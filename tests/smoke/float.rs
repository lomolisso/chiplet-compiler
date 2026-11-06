//! `cc-float` without tools: every wrapper of the standard set and every composition builds around
//! stand-in cores, each core wired as its need says, the Verilog deterministic and free of zero-width
//! literals; wrong core widths and impossible wrappers are refused. The proofs need Yosys and the
//! solvers (deep suite, `tests/deep/float.rs`).

use cc_core::adapters::{Accuracy, Contract, Subnormals};
use cc_float::compose::{self, Budget, Recipe};
use cc_float::glue::{self, Glue, stand_in};
use cc_float::rtl::Mutation;
use cc_smt::float::{Arith, Format};

fn every_wrapper() -> Vec<Glue> {
    let mut gs = glue::standard(&[Format::F32, Format::BF16, Format::F16, Format::E4M3FN]);
    for from in [Format::F32, Format::F16, Format::BF16] {
        gs.push(Glue::Convert {
            from,
            to: Format::E5M2,
        });
    }
    for fmt in [Format::F32, Format::BF16, Format::F16, Format::E4M3FN] {
        for recipe in [Recipe::Power, Recipe::Tanh, Recipe::LogPlusOne, Recipe::Rsqrt] {
            gs.push(Glue::Compose {
                recipe,
                fmt,
                q: fmt.m + 1 + 6,
            });
        }
    }
    gs
}

#[test]
fn every_wrapper_builds_around_its_cores() {
    let gs = every_wrapper();
    assert!(gs.len() > 90, "{}", gs.len());
    let mut names = std::collections::BTreeSet::new();
    for g in &gs {
        let needs = g.needs().unwrap();
        let cores: Vec<_> = needs.iter().map(stand_in).collect();
        let b = g.build(&cores, Mutation::None).unwrap();
        let v = b.rtl.verilog();
        assert!(names.insert(b.rtl.name().to_string()), "{}", b.rtl.name());
        assert!(v.starts_with(&format!("module {} (", g.name())), "{v}");
        assert!(v.trim_end().ends_with("endmodule"));
        let zero_width = v
            .match_indices("0'")
            .any(|(i, _)| i == 0 || !v.as_bytes()[i - 1].is_ascii_digit());
        assert!(!zero_width, "{}: zero-width literal", g.name());
        assert!(!v.contains('\u{2014}'));
        assert_eq!(b.rtl.core_wires().len(), needs.len(), "{}", g.name());
        for (w, n) in b.rtl.core_wires().iter().zip(&needs) {
            assert_eq!(w.operands.len(), n.arity, "{}", g.name());
        }
        let stubs = b.rtl.stub_verilog();
        assert_eq!(stubs.matches("(* blackbox *)").count(), b.rtl.stubs().len());
        // Deterministic.
        assert_eq!(g.build(&cores, Mutation::None).unwrap().rtl.verilog(), v);
    }
}

#[test]
fn wrong_cores_and_impossible_wrappers_are_refused() {
    let g = Glue::Arith {
        op: Arith::Mul,
        fmt: Format::F32,
    };
    let needs = g.needs().unwrap();
    // The multiply needs one more exponent bit: an f32 core does not fit.
    assert_eq!((needs[0].exp_width, needs[0].sig_width), (9, 23));
    let mut core = stand_in(&needs[0]);
    core.exp_width = 8;
    assert!(g.build(&[core], Mutation::None).is_err());
    assert!(g.build(&[], Mutation::None).is_err());
    // Sums of a flushing format run at its own width.
    let add = Glue::Arith {
        op: Arith::Add,
        fmt: Format::BF16,
    };
    assert_eq!(add.needs().unwrap()[0].exp_width, 8);
    // No exact widening from f16 to bf16, no convert from E4M3FN to E5M2.
    assert!(!glue::exact_widening(Format::F16, Format::BF16));
    assert!(
        Glue::Convert {
            from: Format::E4M3FN,
            to: Format::E5M2
        }
        .needs()
        .is_err()
    );
    // A composition needs more inner bits than the format has.
    assert!(
        Glue::Compose {
            recipe: Recipe::Power,
            fmt: Format::F32,
            q: 24
        }
        .needs()
        .is_err()
    );
}

#[test]
fn compositions_are_sized_from_the_budget_and_the_contracts() {
    let b = Budget::cuda();
    let c = Contract {
        accuracy: Accuracy::Ulp { ulp: 2.0 },
        ieee_specials: true,
        subnormals: Subnormals::Ieee,
    };
    let cores = [
        ("log".to_string(), c.accuracy.clone()),
        ("exp".to_string(), c.accuracy.clone()),
    ]
    .into();
    let s = compose::size("power", Format::F32, Recipe::Power, &b, &cores).unwrap();
    // 4 ulp for powf: k = 7, as the sampler confirms (3.2 ulp measured).
    assert_eq!((s.k, s.q), (7, 31));
    assert!(s.bound_ulp <= 4.0);
    let json = serde_json::to_value(&c).unwrap();
    assert_eq!(json["accuracy"]["class"], "ulp");
}
