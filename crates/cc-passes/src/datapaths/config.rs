use serde::Deserialize;

/// `[passes.datapaths]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DatapathsConfig {
    /// Classes allowed to end with a solver `unknown` or timeout, each with its reason.
    pub allow_unknown: Vec<Allowed>,
    /// Upper bound on the width tuples of one class (more fails the pass).
    pub max_tuples: usize,
    /// Per-query timeout of the encoding lemmas (Z3), which bit-blast a divider or multiplier.
    pub lemma_timeout_s: u64,
    /// Write `views/datapaths.md`.
    pub views: bool,
}

/// One `allow_unknown` entry.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Allowed {
    pub class: String,
    pub reason: String,
}

impl Default for DatapathsConfig {
    fn default() -> Self {
        Self {
            allow_unknown: Vec::new(),
            max_tuples: 64,
            lemma_timeout_s: 600,
            views: true,
        }
    }
}
