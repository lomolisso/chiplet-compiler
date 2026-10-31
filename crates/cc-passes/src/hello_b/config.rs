use serde::Deserialize;

/// `[passes.hello_b]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct HelloBConfig {
    pub suffix: String,
    pub sleep_ms: u64,
    /// Framework self-test: make `run` fail, to exercise failure propagation and resume.
    pub fail: bool,
}

impl Default for HelloBConfig {
    fn default() -> Self {
        Self {
            suffix: ", world".into(),
            sleep_ms: 0,
            fail: false,
        }
    }
}
