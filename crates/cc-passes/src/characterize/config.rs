use std::collections::BTreeMap;

use cc_core::adapters::{IpProvider, LecOptions, SynthFlow};
use serde::Deserialize;

/// `[passes.characterize]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CharacterizeConfig {
    /// The synthesis flow (`eda::FLOWS`): `dc` (Design Compiler) or `yosys` (Yosys + ABC, no license).
    pub tool: String,
    /// Check every netlist against its RTL with the flow's equivalence checker (Formality, Yosys'
    /// equivalence flow). Off by default: the synthesis tool is trusted (design §5).
    pub lec: bool,
    /// Kits, by PDK manifest (`pdks/<kit>.toml`); empty: every manifest.
    pub pdks: Vec<String>,
    /// Clock target per kit, in ps, overriding its manifest's `clock_ps`.
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
    /// Cell driving every input, per kit, overriding its manifest's `driving_cell`.
    pub driving_cell: BTreeMap<String, String>,
    /// Every output drives this many inputs of the driving cell.
    pub load_fanout: f64,
    /// Wall-clock limit of one design's equivalence check (Formality `verification_timeout_limit`), per
    /// strategy, in minutes.
    pub lec_timeout_min: u32,
    /// Formality's alternate strategies, tried in order while a design is not proven
    /// (`verification_alternate_strategy_names` lists them in the recommended order).
    pub lec_strategies: Vec<String>,
    /// Designs per tool session.
    pub batch: usize,
    /// Cores per licensed tool session.
    pub cores: u32,
    /// The float IP providers (`ip::PROVIDERS`) per synthesis flow, in the order a core is looked for: the
    /// first that offers it wins (docs/framework/adapters.md, `IpProvider`).
    pub providers: BTreeMap<String, Vec<String>>,
    /// Write `views/library.md`.
    pub views: bool,
}

impl Default for CharacterizeConfig {
    fn default() -> Self {
        Self {
            tool: "dc".into(),
            lec: false,
            pdks: Vec::new(),
            clock_ps: BTreeMap::new(),
            targets: vec!["clock".into(), "fast".into()],
            fast_fraction: 0.05,
            ops: Vec::new(),
            int_widths: vec![2, 4, 8, 16, 32],
            float_formats: vec!["f32".into(), "f16".into(), "bf16".into(), "f8e4m3fn".into()],
            mux_inputs: vec![2, 3, 4, 5, 6, 7, 8],
            mux_widths: vec![1, 2, 4, 8, 16, 32],
            register_widths: vec![1, 2, 4, 8, 16, 32],
            split_multipliers: vec![(8, 4), (16, 8), (32, 8), (32, 16)],
            driving_cell: BTreeMap::new(),
            load_fanout: 4.0,
            lec_timeout_min: 2,
            lec_strategies: Vec::new(),
            batch: 24,
            cores: 2,
            providers: BTreeMap::from([
                ("dc".to_string(), vec!["designware".to_string(), "hardfloat".to_string()]),
                (
                    "yosys".to_string(),
                    vec!["hardfloat".to_string(), "flopoco".to_string(), "xls_math".to_string()],
                ),
            ]),
            views: true,
        }
    }
}

impl CharacterizeConfig {
    /// The period of `target` for a kit whose clock target is `clock_ps`, in ps.
    pub fn period_ps(&self, clock_ps: f64, target: &str) -> Result<f64, String> {
        match target {
            "clock" => Ok(clock_ps),
            "fast" => Ok(clock_ps * self.fast_fraction),
            t => Err(format!("unknown target {t} (clock | fast)")),
        }
    }

    /// The selected flow.
    pub fn flow(&self) -> Result<&'static dyn SynthFlow, String> {
        crate::eda::FLOWS
            .by_name(&self.tool)
            .map_err(|e| format!("tool: {e}"))
    }

    /// The equivalence check's options, when it is on.
    pub fn lec_options(&self) -> Option<LecOptions> {
        self.lec.then(|| LecOptions {
            timeout_min: self.lec_timeout_min,
            strategies: self.lec_strategies.clone(),
        })
    }

    /// The IP providers of `flow`, in order; a provider that does not serve the flow is an error (an IP
    /// the flow cannot read).
    pub fn providers_of(&self, flow: &str) -> Result<Vec<&'static dyn IpProvider>, String> {
        let names = self.providers.get(flow).map(Vec::as_slice).unwrap_or_default();
        names
            .iter()
            .map(|n| {
                let p = crate::ip::PROVIDERS
                    .by_name(n)
                    .map_err(|e| format!("providers.{flow}: {e}"))?;
                if !p.capabilities().flows.contains(&flow) {
                    return Err(format!(
                        "providers.{flow}: `{n}` does not serve the {flow} flow (it serves {})",
                        p.capabilities().flows.join(", ")
                    ));
                }
                Ok(p)
            })
            .collect()
    }

    pub fn check(&self) -> Result<(), String> {
        self.flow()?;
        for flow in self.providers.keys() {
            crate::eda::FLOWS
                .by_name(flow)
                .map_err(|e| format!("providers: {e}"))?;
            self.providers_of(flow)?;
        }
        for t in &self.targets {
            self.period_ps(1.0, t)?;
        }
        if !(self.fast_fraction > 0.0 && self.fast_fraction < 1.0) {
            return Err("fast_fraction must be in (0, 1)".into());
        }
        if self.batch == 0 || self.cores == 0 {
            return Err("batch and cores must be positive".into());
        }
        if let Some((k, _)) = self.clock_ps.iter().find(|(_, c)| c.is_nan() || **c <= 0.0) {
            return Err(format!("clock_ps.{k} must be positive"));
        }
        Ok(())
    }
}
