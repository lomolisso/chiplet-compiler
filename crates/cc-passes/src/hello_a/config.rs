use serde::Deserialize;

/// `[passes.hello_a]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct HelloAConfig {
    pub greeting: String,
    /// Artificial work time; the overlap test uses it to make branches run concurrently.
    pub sleep_ms: u64,
}

impl Default for HelloAConfig {
    fn default() -> Self {
        Self {
            greeting: "hello".into(),
            sleep_ms: 0,
        }
    }
}
