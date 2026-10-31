//! All passes, one module each (conventions: pass anatomy), and the registry (design §3.2).
//!
//! `hello_a`, `hello_b`, `hello_c` are the framework's self-test passes: they exercise the DAG,
//! caching, parallel branches, tool runners and verification without touching any model.

pub mod export;
pub mod hello_a;
pub mod hello_b;
pub mod hello_c;
pub mod registry;

pub use registry::registry;

/// `(file name, schema)` pairs of the passes' artifact kinds, committed under `docs/schemas/`.
pub fn pass_schemas() -> Vec<(&'static str, serde_json::Value)> {
    export::json::schemas()
}
