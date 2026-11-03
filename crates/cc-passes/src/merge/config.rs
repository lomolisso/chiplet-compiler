use serde::Deserialize;

/// `[passes.merge]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MergeConfig {
    /// Requantization hardware (design §6.3): `recipe` follows each model's recipe (every recipe
    /// declares fp32 rescaling today), `fp32`, or `fixed` (rescaling chains become `rescale` vertices
    /// where its tolerance lemma is proven).
    pub requant_variant: RequantVariant,
    /// Limit of one `rescale` tolerance lemma.
    pub requant_timeout_s: u64,
    /// Stop agglomerating a family at this many units (0: stop when no merge saves area).
    pub target_units: usize,
    /// Critical-path bound of a merged unit in ps (0: per family, the deepest single-class datapath
    /// rounded up to whole clock periods).
    pub max_logic_depth: f64,
    /// The library kit whose areas (`clock` target) and delays (`fast` target) weigh merges.
    pub kit: String,
    /// Cover each datapath with the selected patterns of `mine` first; an occurrence then merges only
    /// with an occurrence of the same pattern.
    pub contract: bool,
    /// Compatibility graphs with more nodes than this are solved by `greedy`.
    pub max_compat_nodes: usize,
    /// Merge steps whose compatibility graph has at most this many nodes store it, so `verify`
    /// re-solves them with the other exact algorithm.
    pub certify_max_nodes: usize,
    /// Limit of one composed-step proof (Z3; a multiplier against its decomposition is hard for
    /// bit-blasting, so these run like encoding lemmas).
    pub compose_timeout_s: u64,
    /// Write `views/merge.md`.
    pub views: bool,
    pub clique: CliqueConfig,
}

/// `[passes.merge.clique]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CliqueConfig {
    pub algorithm: CliqueAlgorithm,
    /// How candidate pairs are ranked: by `greedy` cliques (the accepted merge is then solved by
    /// `algorithm`), or by `algorithm` itself.
    pub rank: Rank,
    /// Branch-and-bound nodes one solve may expand (both exact algorithms); the incumbent is then
    /// reported as not optimal. Deterministic, unlike `timeout_s`.
    pub node_budget: u64,
    /// Wall-clock limit of one solve (safety net); on expiry `greedy` answers and the step records it.
    pub timeout_s: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CliqueAlgorithm {
    ExactIlp,
    Ostergard,
    Greedy,
}

impl CliqueAlgorithm {
    pub fn name(self) -> &'static str {
        match self {
            CliqueAlgorithm::ExactIlp => "exact_ilp",
            CliqueAlgorithm::Ostergard => "ostergard",
            CliqueAlgorithm::Greedy => "greedy",
        }
    }

    pub fn parse(s: &str) -> Option<CliqueAlgorithm> {
        match s {
            "exact_ilp" => Some(CliqueAlgorithm::ExactIlp),
            "ostergard" => Some(CliqueAlgorithm::Ostergard),
            "greedy" => Some(CliqueAlgorithm::Greedy),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rank {
    Greedy,
    Algorithm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequantVariant {
    Recipe,
    Fp32,
    Fixed,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            requant_variant: RequantVariant::Recipe,
            requant_timeout_s: 1800,
            target_units: 0,
            max_logic_depth: 0.0,
            kit: "asap7".into(),
            contract: true,
            max_compat_nodes: 4000,
            certify_max_nodes: 60,
            compose_timeout_s: 600,
            views: true,
            clique: CliqueConfig::default(),
        }
    }
}

impl Default for CliqueConfig {
    fn default() -> Self {
        Self {
            algorithm: CliqueAlgorithm::ExactIlp,
            rank: Rank::Greedy,
            node_budget: 2_000,
            timeout_s: 30,
        }
    }
}

impl MergeConfig {
    pub fn check(&self) -> Result<(), String> {
        if self.requant_timeout_s == 0 {
            return Err("requant_timeout_s must be positive".into());
        }
        if !(self.max_logic_depth >= 0.0 && self.max_logic_depth.is_finite()) {
            return Err(format!(
                "max_logic_depth = {}: a non-negative number of ps",
                self.max_logic_depth
            ));
        }
        if self.kit.is_empty() {
            return Err("kit: name a kit of the characterize library".into());
        }
        if self.max_compat_nodes == 0 || self.clique.node_budget == 0 || self.clique.timeout_s == 0 {
            return Err("max_compat_nodes, clique.node_budget and clique.timeout_s must be positive".into());
        }
        Ok(())
    }
}
