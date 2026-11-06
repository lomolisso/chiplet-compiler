//! The float IP providers (design §3.7, §4.12; `cc_core::adapters::ip`), as data: DesignWare for the
//! industry flow, Berkeley HardFloat, FloPoCo and XLS's float library for the open flow (HardFloat also
//! fills what DesignWare lacks in the industry flow). `[passes.characterize] providers` orders them per
//! flow; [`routes`] maps every float op of the catalog to the cores of the flow, or reports a gap.
//! Facts (ranges, parameters, accuracy) are the vendors' documentation, listed in docs/third-party.md and
//! docs/framework/adapters.md.

pub mod designware;
pub mod flopoco;
pub mod hardfloat;
pub mod routes;
pub mod xls_math;

use std::collections::BTreeMap;

use cc_core::adapters::{Accuracy, Contract, CoreInstance, IpProvider, Registry, Subnormals};

/// Every IP provider, by name (`[passes.characterize] providers`).
pub static PROVIDERS: Registry<dyn IpProvider> = Registry::new(
    "IP provider",
    &[
        &designware::DesignWare,
        &hardfloat::HardFloat,
        &flopoco::FloPoCo,
        &xls_math::XlsMath,
    ],
);

/// An instance: `module`, parameters and port connections as given.
fn instance(
    module: &str,
    params: &[(&str, String)],
    ports: &[(&str, &str)],
    exp_width: u32,
    sig_width: u32,
) -> CoreInstance {
    CoreInstance {
        module: module.to_string(),
        params: params
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect::<BTreeMap<_, _>>(),
        ports: ports
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<BTreeMap<_, _>>(),
        exp_width,
        sig_width,
    }
}

fn contract(accuracy: Accuracy, ieee_specials: bool, subnormals: Subnormals) -> Contract {
    Contract {
        accuracy,
        ieee_specials,
        subnormals,
    }
}

/// The name of a generated core: `<provider>_e<E>m<M>_<op>` (the op's `.` as `_`), a Verilog identifier.
fn generated_name(provider: &str, op: &str, exp_width: u32, sig_width: u32) -> String {
    format!("{provider}_e{exp_width}m{sig_width}_{}", op.replace('.', "_"))
}

/// The Python model of an op: the IEEE operation for an exact core, the correctly rounded function for an
/// approximate one (`CoreOffer::model`).
fn model(op: &cc_core::adapters::CoreOp) -> String {
    if op.exact() {
        format!("ieee.{}", op.family())
    } else {
        format!("reference.{}", op.family())
    }
}
