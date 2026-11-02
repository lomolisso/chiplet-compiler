use std::collections::BTreeMap;

use serde::Deserialize;

/// `[passes.characterize]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CharacterizeConfig {
    /// `dc` (Design Compiler + Formality), `genus` (Genus + Conformal) or `yosys` (Yosys + ABC, its
    /// equivalence flow; no license).
    pub tool: Tool,
    /// Kits: `nangate45`, `asap7`.
    pub pdks: Vec<String>,
    /// Clock target per kit, in ps.
    pub clock_ps: BTreeMap<String, f64>,
    /// Timing points: `clock` (the clock target, the area row) and `fast` (a tight constraint: the
    /// minimum delay and its area).
    pub targets: Vec<String>,
    /// The `fast` period as a fraction of the clock target.
    pub fast_fraction: f64,
    /// Element-wise ops of the catalog (op names without `stablehlo.`); empty: every op of the table.
    pub ops: Vec<String>,
    /// Widths of the signed and unsigned integer types (bool is the 1-bit type).
    pub int_widths: Vec<u32>,
    /// Float formats: `f32`, `f16`, `bf16`, `f8e4m3fn`, `f8e5m2`.
    pub float_formats: Vec<String>,
    /// Input counts of the multiplexers.
    pub mux_inputs: Vec<u32>,
    pub mux_widths: Vec<u32>,
    pub register_widths: Vec<u32>,
    /// `(w, n)`: a w-bit multiply from n-bit piece multipliers.
    pub split_multipliers: Vec<(u32, u32)>,
    /// Cell driving every input, per kit.
    pub driving_cell: BTreeMap<String, String>,
    /// Every output drives this many inputs of the driving cell.
    pub load_fanout: f64,
    /// Designs per tool session.
    pub batch: usize,
    /// Cores per DC / Genus session.
    pub cores: u32,
    /// Write `views/library.md`.
    pub views: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tool {
    Dc,
    Genus,
    Yosys,
}

impl Tool {
    pub fn name(self) -> &'static str {
        match self {
            Tool::Dc => "dc",
            Tool::Genus => "genus",
            Tool::Yosys => "yosys",
        }
    }

    /// The equivalence checker that goes with the synthesis tool.
    pub fn lec(self) -> &'static str {
        match self {
            Tool::Dc => "formality",
            Tool::Genus => "conformal",
            Tool::Yosys => "yosys-equiv",
        }
    }
}

impl Default for CharacterizeConfig {
    fn default() -> Self {
        Self {
            tool: Tool::Dc,
            pdks: vec!["nangate45".into(), "asap7".into()],
            clock_ps: [("nangate45".into(), 2000.0), ("asap7".into(), 1000.0)].into(),
            targets: vec!["clock".into(), "fast".into()],
            fast_fraction: 0.05,
            ops: Vec::new(),
            int_widths: vec![2, 4, 8, 16, 32],
            float_formats: vec!["f32".into(), "f16".into(), "bf16".into(), "f8e4m3fn".into()],
            mux_inputs: vec![2, 3, 4, 5, 6, 7, 8],
            mux_widths: vec![1, 2, 4, 8, 16, 32],
            register_widths: vec![1, 2, 4, 8, 16, 32],
            split_multipliers: vec![(8, 4), (16, 8), (32, 8), (32, 16)],
            driving_cell: [
                ("nangate45".into(), "BUF_X1".into()),
                ("asap7".into(), "BUFx2_ASAP7_75t_R".into()),
            ]
            .into(),
            load_fanout: 4.0,
            batch: 24,
            cores: 2,
            views: true,
        }
    }
}

impl CharacterizeConfig {
    /// The period of `target` on `kit`, in ps.
    pub fn period_ps(&self, kit: &str, target: &str) -> Result<f64, String> {
        let clock = *self
            .clock_ps
            .get(kit)
            .ok_or_else(|| format!("clock_ps has no entry for {kit}"))?;
        match target {
            "clock" => Ok(clock),
            "fast" => Ok(clock * self.fast_fraction),
            t => Err(format!("unknown target {t} (clock | fast)")),
        }
    }

    pub fn check(&self) -> Result<(), String> {
        for k in &self.pdks {
            if !matches!(k.as_str(), "nangate45" | "asap7") {
                return Err(format!("pdks: unknown kit {k} (nangate45 | asap7)"));
            }
            for t in &self.targets {
                self.period_ps(k, t)?;
            }
            if !self.driving_cell.contains_key(k) {
                return Err(format!("driving_cell has no entry for {k}"));
            }
        }
        if !(self.fast_fraction > 0.0 && self.fast_fraction < 1.0) {
            return Err("fast_fraction must be in (0, 1)".into());
        }
        if self.batch == 0 || self.cores == 0 {
            return Err("batch and cores must be positive".into());
        }
        Ok(())
    }
}
