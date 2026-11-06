//! XLS's float function library (`third_party/xls_go_math`, from the Go `math` package and Cephes; in the
//! toolchain image under `$XLS_DSLX_PATH`): f32 sine and cosine, `fp_sin_32` and `fp_cos_32` of
//! `fp_sincos_32.x` (Payne-Hanek reduction, full range, radians). No accuracy is documented; the project
//! takes them under an absolute contract, |error| <= 2^-22 (design §6.4). Subnormal inputs read as zero and
//! results flush (XLS `apfloat` arithmetic). Ports of the generated module: `x`, `out`.

use cc_core::adapters::{
    Accuracy, CoreOffer, CoreOp, CoreSource, Encoding, Generator, IpCapabilities, IpProvider, Named,
    Subnormals, Timing,
};

use super::{contract, generated_name, instance, model};

/// The absolute error bound of the f32 sine and cosine (design §6.4).
pub const SINCOS_ABS_BOUND: f64 = 1.0 / 4_194_304.0; // 2^-22

/// XLS's float library (open flow).
#[derive(Debug, Clone, Copy)]
pub struct XlsMath;

impl Named for XlsMath {
    fn name(&self) -> &'static str {
        "xls_math"
    }
}

impl IpProvider for XlsMath {
    fn capabilities(&self) -> IpCapabilities {
        IpCapabilities {
            flows: &["yosys"],
            licence: "Apache-2.0 (XLS) with the Go BSD-3-Clause and Cephes notices",
            ops: &["sin", "cos"],
        }
    }

    fn core(&self, op: &str, e: u32, s: u32) -> Option<CoreOffer> {
        let op: CoreOp = op.parse().ok()?;
        let top = match (op, e, s) {
            (CoreOp::Sin, 8, 23) => "fp_sin_32",
            (CoreOp::Cos, 8, 23) => "fp_cos_32",
            _ => return None,
        };
        Some(CoreOffer {
            instance: instance(
                &generated_name("xls", &op.to_string(), e, s),
                &[],
                &[("x", "a0"), ("out", "z")],
                e,
                s,
            ),
            contract: contract(
                Accuracy::Absolute {
                    bound: SINCOS_ABS_BOUND,
                },
                true,
                Subnormals::Flush,
            ),
            delay_ps: None,
            model: model(&op),
            source: CoreSource::Generated {
                generator: Generator::XlsDslx {
                    file: "third_party/xls_go_math/fp_sincos_32.x".into(),
                    top: top.into(),
                },
            },
            encoding: Encoding::Ieee,
            timing: Timing::Combinational,
        })
    }
}
