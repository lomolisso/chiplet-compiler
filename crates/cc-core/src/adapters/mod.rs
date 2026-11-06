//! External resources behind adapters (design §3.7; docs/framework/adapters.md): one trait per family,
//! implementations chosen by name in `config.toml`, each declaring what it supports. The traits and the
//! data they exchange live here; the implementations stay next to the code that uses them and are listed
//! in one [`Registry`] per family (`cc_passes::adapters`, `cc_smt::SOLVERS`).

pub mod hls;
pub mod mine;
pub mod pnr;
pub mod select;
pub mod smt;
pub mod synth;

pub use hls::{HlsCapabilities, HlsFrontend};
pub use mine::{MineJob, MineRun, Miner, MinerCapabilities};
pub use pnr::{PnrCapabilities, PnrFlow};
pub use select::{CliqueSolver, Limits, MisSolver, SelectCapabilities};
pub use smt::{SmtCapabilities, SmtSolver};
pub use synth::{
    Design, FlowRunners, Kit, Lec, LecOptions, LibraryFormat, Measured, ReportTexts, SynthCapabilities,
    SynthFlow, SynthJob,
};

use crate::error::CoreError;

/// What every implementation has: the name `config.toml` selects it by.
pub trait Named: Send + Sync {
    fn name(&self) -> &'static str;
}

/// The implementations of one family, in a fixed order.
#[derive(Debug)]
pub struct Registry<T: ?Sized + 'static> {
    /// What the family is called in messages (`synthesis flow`, `SMT solver`).
    pub family: &'static str,
    pub items: &'static [&'static T],
}

impl<T: ?Sized + Named + 'static> Registry<T> {
    pub const fn new(family: &'static str, items: &'static [&'static T]) -> Self {
        Self { family, items }
    }

    /// The implementation called `name`; an error lists the available ones.
    pub fn by_name(&self, name: &str) -> Result<&'static T, CoreError> {
        self.items
            .iter()
            .copied()
            .find(|i| i.name() == name)
            .ok_or_else(|| CoreError::UnknownAdapter {
                family: self.family.to_string(),
                name: name.to_string(),
                available: if self.items.is_empty() {
                    "none yet".into()
                } else {
                    self.names().join(", ")
                },
            })
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.items.iter().map(|i| i.name()).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &'static T> + '_ {
        self.items.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    trait Thing: Named {
        fn value(&self) -> u32;
    }
    struct A;
    struct B;
    impl Named for A {
        fn name(&self) -> &'static str {
            "a"
        }
    }
    impl Named for B {
        fn name(&self) -> &'static str {
            "b"
        }
    }
    impl Thing for A {
        fn value(&self) -> u32 {
            1
        }
    }
    impl Thing for B {
        fn value(&self) -> u32 {
            2
        }
    }
    static THINGS: Registry<dyn Thing> = Registry::new("thing", &[&A, &B]);
    static NONE: Registry<dyn Thing> = Registry::new("thing", &[]);

    #[test]
    fn selects_by_name_and_lists_the_others() {
        assert_eq!(THINGS.by_name("b").unwrap().value(), 2);
        assert_eq!(THINGS.names(), ["a", "b"]);
        let e = THINGS.by_name("c").err().unwrap().to_string();
        assert!(e.contains("thing `c`") && e.contains("a, b"), "{e}");
        assert!(NONE.by_name("a").err().unwrap().to_string().contains("none yet"));
    }
}
