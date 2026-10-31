use serde::Deserialize;

/// `[passes.hello_c]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct HelloCConfig {
    pub text: String,
    pub sleep_ms: u64,
}

impl Default for HelloCConfig {
    fn default() -> Self {
        Self {
            text: "independent".into(),
            sleep_ms: 0,
        }
    }
}
