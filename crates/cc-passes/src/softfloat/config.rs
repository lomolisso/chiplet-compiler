use serde::Deserialize;

use crate::characterize::CharacterizeConfig;

/// `[passes.softfloat]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SoftfloatConfig {
    /// Float formats: `f32`, `f16`, `bf16`, `f8e4m3fn`, `f8e5m2` (as `characterize` names them).
    pub float_formats: Vec<String>,
    /// Widths of the signed and unsigned integers that conversions reach (bool is the 1-bit type).
    pub int_widths: Vec<u32>,
    /// Element-wise ops to build (op names without `stablehlo.`); empty: every float op of the catalog.
    pub ops: Vec<String>,
    /// Every input combination is simulated when the operands have at most this many bits in total.
    pub exhaustive_bits: u32,
    /// Seeded random operand sets per primitive otherwise (next to the edge cases).
    pub random_vectors: u32,
    /// Primitives per container.
    pub batch: usize,
    /// Bitwuzla's limit per proof; a proof that runs out of time leaves the primitive `exhaustive` or
    /// `tested`.
    pub proof_timeout_s: u64,
    /// Z3's limit per proof when `verify` re-checks the proven ones.
    pub recheck_timeout_s: u64,
    /// Vectors per primitive whose reference `verify` recomputes.
    pub recheck_sample: u32,
    /// The image's Python (with Allo).
    pub docker_python: String,
    /// Write `views/softfloat.md`.
    pub views: bool,
}

impl Default for SoftfloatConfig {
    fn default() -> Self {
        let c = CharacterizeConfig::default();
        Self {
            float_formats: c.float_formats,
            int_widths: c.int_widths,
            ops: Vec::new(),
            exhaustive_bits: 16,
            random_vectors: 100_000,
            batch: 8,
            proof_timeout_s: 600,
            recheck_timeout_s: 7200,
            recheck_sample: 2000,
            docker_python: "python".into(),
            views: true,
        }
    }
}

impl SoftfloatConfig {
    pub fn check(&self) -> Result<(), String> {
        if self.float_formats.is_empty() {
            return Err("float_formats is empty".into());
        }
        if self.batch == 0 {
            return Err("batch must be at least 1".into());
        }
        if self.exhaustive_bits > 32 {
            return Err(format!("exhaustive_bits = {} (at most 32)", self.exhaustive_bits));
        }
        if self.proof_timeout_s == 0 || self.recheck_timeout_s == 0 {
            return Err("timeouts must be positive".into());
        }
        Ok(())
    }

    /// The `characterize` configuration whose catalog lists this library's primitives.
    pub fn catalog_config(&self) -> CharacterizeConfig {
        CharacterizeConfig {
            float_formats: self.float_formats.clone(),
            int_widths: self.int_widths.clone(),
            ops: self.ops.clone(),
            ..CharacterizeConfig::default()
        }
    }
}
