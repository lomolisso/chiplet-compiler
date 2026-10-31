use std::path::{Path, PathBuf};

/// Errors of `cc-core`. Every variant names what failed (file, pass, artifact, tool).
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("config {layer}: {message}")]
    Config { layer: String, message: String },

    #[error("config [passes.{pass}]: {message}")]
    PassConfig { pass: String, message: String },

    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{path}: json: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("{path}: postcard: {source}")]
    Postcard {
        path: PathBuf,
        #[source]
        source: postcard::Error,
    },

    #[error("artifact {kind}: {message}")]
    Artifact { kind: String, message: String },

    #[error("tool {tool}: {message}")]
    Tool { tool: String, message: String },

    #[error("{runner} runner is not implemented yet (see {task})")]
    Unimplemented {
        runner: &'static str,
        task: &'static str,
    },

    #[error("cancelled")]
    Cancelled,
}

pub type Result<T, E = CoreError> = std::result::Result<T, E>;

/// Attach the offending path to an `io::Error`.
pub trait IoContext<T> {
    fn at(self, path: &Path) -> Result<T>;
}

impl<T> IoContext<T> for std::io::Result<T> {
    fn at(self, path: &Path) -> Result<T> {
        self.map_err(|source| CoreError::Io {
            path: path.to_path_buf(),
            source,
        })
    }
}
