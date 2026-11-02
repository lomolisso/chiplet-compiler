//! All passes, one module each (conventions: pass anatomy), and the registry (design §3.2).
//!
//! `hello_a`, `hello_b`, `hello_c` are the framework's self-test passes: they exercise the DAG,
//! caching, parallel branches, tool runners and verification without touching any model.

pub mod components;
pub mod export;
pub mod hello_a;
pub mod hello_b;
pub mod hello_c;
pub mod import;
pub mod iteration_space;
pub mod kernel_classes;
pub mod kernels;
pub mod normalize;
pub mod registry;

pub use registry::registry;

/// `(file name, schema)` pairs of the passes' artifact kinds, committed under `docs/schemas/`.
pub fn pass_schemas() -> Vec<(&'static str, serde_json::Value)> {
    let mut v = export::json::schemas();
    v.extend(import::json::schemas());
    v.extend(components::json::schemas());
    v.extend(normalize::json::schemas());
    v.extend(kernels::json::schemas());
    v.extend(kernel_classes::json::schemas());
    v.extend(iteration_space::json::schemas());
    v
}
