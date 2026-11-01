use serde::Deserialize;

/// `[passes.kernels]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct KernelsConfig {
    /// Write `views/kernels.md` and `views/<id>.dot`.
    pub views: bool,
}

impl Default for KernelsConfig {
    fn default() -> Self {
        Self { views: true }
    }
}
