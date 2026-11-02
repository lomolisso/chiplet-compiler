use serde::Deserialize;

/// `[passes.iteration_space]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct IterationSpaceConfig {
    /// Write `views/routing.md` and `views/iteration_space.md`.
    pub views: bool,
    pub template: TemplateConfig,
}

/// `[passes.iteration_space.template]`: when a contraction with a valid space-time mapping is routed to
/// an array template (docs/passes/iteration_space.md §4).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TemplateConfig {
    /// Route classes to templates at all (`false`: every class goes to datapath mining).
    pub enabled: bool,
    /// Minimum reuse factor (MACs per element) of each core input in a row that counts for the template.
    pub min_reuse: u64,
    /// Minimum share of the class's weighted workload in rows where both inputs reach `min_reuse`.
    pub min_share: f64,
    /// Upper bound on the dims of the space the mapping search enumerates (after grouping; a class with
    /// more is routed to datapaths).
    pub max_dims: u32,
}

impl Default for IterationSpaceConfig {
    fn default() -> Self {
        Self {
            views: true,
            template: TemplateConfig::default(),
        }
    }
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_reuse: 8,
            min_share: 0.5,
            max_dims: 7,
        }
    }
}
