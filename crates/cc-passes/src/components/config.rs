use serde::Deserialize;

/// `[passes.components]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ComponentsConfig {
    /// Upper bound on WL refinement rounds (refinement stops earlier when the partition is stable).
    pub wl_max_rounds: u32,
    /// Upper bound on the candidate pairs one VF2 search may test; a search over it fails the pass.
    pub vf2_budget: u64,
    /// Write `views/components.md`.
    pub views: bool,
}

impl Default for ComponentsConfig {
    fn default() -> Self {
        Self {
            wl_max_rounds: 64,
            vf2_budget: 200_000_000,
            views: true,
        }
    }
}
