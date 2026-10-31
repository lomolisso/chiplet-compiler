//! The pipeline manager: the `Pass` trait, the pass DAG, the scheduler with memory admission, caching
//! and resume (spec §3.1–§3.3). See `README.md` for the admission rule and how to add a pass.

pub mod admission;
pub mod context;
pub mod dag;
pub mod env;
pub mod error;
pub mod pass;
pub mod pipeline;
pub mod summary;
pub mod versions;

pub use context::PassContext;
pub use dag::{DagError, PassGraph};
pub use error::{PassError, PipelineError};
pub use pass::{Pass, RunnerKind, ToolSpec};
pub use pipeline::{Pipeline, RunOptions, StatusRow};
pub use summary::{PassRun, RunStatus, RunSummary};

/// JSON Schemas owned by `cc-pipeline` (see `cc_core::schema`).
pub fn pipeline_schemas() -> Vec<(&'static str, serde_json::Value)> {
    vec![(
        "run_summary.schema.json",
        serde_json::to_value(schemars::schema_for!(RunSummary)).unwrap_or_default(),
    )]
}
