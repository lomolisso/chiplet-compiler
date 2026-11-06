//! The float IP providers (design §3.7, §4.12; `cc_core::adapters::ip`), as data: DesignWare for the
//! industry flow, Berkeley HardFloat, FloPoCo and XLS's float library for the open flow (HardFloat also
//! fills what DesignWare lacks in the industry flow). `[passes.characterize] providers` orders them per
//! flow; [`routes`] maps every float op of the catalog to the cores of the flow, or reports a gap.
//! Facts (ranges, parameters, accuracy) are the vendors' documentation, listed in docs/third-party.md and
//! docs/framework/adapters.md.

pub mod designware;
pub mod flopoco;
pub mod hardfloat;
pub mod plan;
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

/// A Verilog instantiation of `inst` named `name`: each port connected to `connect(<what the instance
/// ties it to>)` (an operand `a0`, a result `z`, ...), a constant as is, an empty connection left open.
pub fn verilog_instance(inst: &CoreInstance, name: &str, connect: impl Fn(&str) -> String) -> String {
    let params = if inst.params.is_empty() {
        String::new()
    } else {
        let p: Vec<String> = inst.params.iter().map(|(k, v)| format!(".{k}({v})")).collect();
        format!(" #({})", p.join(", "))
    };
    let ports: Vec<String> = inst
        .ports
        .iter()
        .map(|(port, to)| {
            let e = if to.is_empty() || to.contains('\'') {
                to.clone()
            } else {
                connect(to)
            };
            format!(".{port}({e})")
        })
        .collect();
    format!("{}{params} {name}({});", inst.module, ports.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instances_render_as_verilog() {
        let o = PROVIDERS
            .by_name("designware")
            .unwrap()
            .core("add", 5, 10)
            .unwrap();
        let v = verilog_instance(&o.instance, "u0", |x| format!("w_{x}"));
        assert_eq!(
            v,
            "DW_fp_add #(.exp_width(5), .ieee_compliance(1), .sig_width(10)) u0(.a(w_a0), .b(w_a1), .rnd(3'b000), .status(), .z(w_z));"
        );
        let o = PROVIDERS.by_name("hardfloat").unwrap().core("mul", 8, 7).unwrap();
        assert!(verilog_instance(&o.instance, "u", |x| x.to_string()).starts_with("hardfloat_e8m7_mul u("));
    }
}
