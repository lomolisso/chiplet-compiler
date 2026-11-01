use serde::Deserialize;

/// `[passes.import]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ImportConfig {
    /// Composite names that are inlined like calls instead of kept as component boundaries. A trailing
    /// `*` matches a prefix. JAX's own composites (`chlo.top_k`, `chlo.erf`, …) are not module boundaries.
    pub inline_composites: Vec<String>,
    /// Write `views/<model>/<phase>.dot`.
    pub views: bool,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            inline_composites: vec!["chlo.*".into()],
            views: true,
        }
    }
}

impl ImportConfig {
    pub fn inlines(&self, composite: &str) -> bool {
        self.inline_composites.iter().any(|p| match p.strip_suffix('*') {
            Some(prefix) => composite.starts_with(prefix),
            None => composite == p,
        })
    }
}
