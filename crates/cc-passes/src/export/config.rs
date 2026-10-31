use serde::{Deserialize, Serialize};

/// Where the exporter runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportRunner {
    /// The toolchain image (`[tools.docker]`), the locked JAX stack in `docker_python`'s venv.
    Docker,
    /// The host, through `host_python` (uv-managed, from `uv.lock`); CI uses this.
    Host,
}

/// `[passes.export]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ExportConfig {
    pub runner: ExportRunner,
    /// Module class names never wrapped as composites (their children still are).
    pub exclude_classes: Vec<String>,
    /// The interpreter inside the image.
    pub docker_python: String,
    /// The host command prefix that runs Python in the project environment.
    pub host_python: Vec<String>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            runner: ExportRunner::Docker,
            exclude_classes: Vec::new(),
            docker_python: "/opt/cc/jax-venv/bin/python".into(),
            host_python: vec!["uv".into(), "run".into(), "--frozen".into(), "python".into()],
        }
    }
}
