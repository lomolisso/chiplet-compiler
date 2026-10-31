use cc_core::CoreError;

use crate::dag::DagError;

/// Errors of the pipeline manager itself (configuration, DAG, I/O of manifests).
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Core(#[from] CoreError),

    #[error(transparent)]
    Dag(#[from] DagError),

    #[error("pass `{pass}`: {message}")]
    Pass { pass: String, message: String },
}

/// What a pass's `run` or `verify` returns on failure. Passes keep their own `thiserror` enums and
/// convert with `PassError::other` (conventions: errors).
#[derive(Debug, thiserror::Error)]
pub enum PassError {
    #[error(transparent)]
    Core(#[from] CoreError),

    #[error("{0}")]
    Failed(String),

    #[error(transparent)]
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl PassError {
    pub fn failed(msg: impl Into<String>) -> Self {
        Self::Failed(msg.into())
    }

    pub fn other<E: std::error::Error + Send + Sync + 'static>(e: E) -> Self {
        Self::Other(Box::new(e))
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Core(CoreError::Cancelled))
    }
}
