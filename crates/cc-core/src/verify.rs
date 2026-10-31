//! `build/.<pass>/verify.json` (design §5): every check with its status and evidence.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const VERIFY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Fail,
    Skip,
}

/// What a check looked at: certificate and log paths, counts, free-form notes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct Evidence {
    pub paths: Vec<String>,
    pub counts: BTreeMap<String, u64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub name: String,
    pub status: CheckStatus,
    pub evidence: Evidence,
}

impl Check {
    pub fn new(name: impl Into<String>, ok: bool) -> Self {
        let status = if ok { CheckStatus::Pass } else { CheckStatus::Fail };
        Self {
            name: name.into(),
            status,
            evidence: Evidence::default(),
        }
    }

    pub fn skip(name: impl Into<String>, why: impl Into<String>) -> Self {
        Self::new(name, true).with_status(CheckStatus::Skip).note(why)
    }

    pub fn with_status(mut self, s: CheckStatus) -> Self {
        self.status = s;
        self
    }

    pub fn path(mut self, p: impl Into<String>) -> Self {
        self.evidence.paths.push(p.into());
        self
    }

    pub fn count(mut self, k: impl Into<String>, v: u64) -> Self {
        self.evidence.counts.insert(k.into(), v);
        self
    }

    pub fn note(mut self, n: impl Into<String>) -> Self {
        self.evidence.notes.push(n.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VerifyReport {
    pub schema_version: u32,
    pub pass: String,
    pub checks: Vec<Check>,
}

impl VerifyReport {
    pub fn new(pass: impl Into<String>) -> Self {
        Self {
            schema_version: VERIFY_SCHEMA_VERSION,
            pass: pass.into(),
            checks: Vec::new(),
        }
    }

    pub fn push(&mut self, c: Check) -> &mut Self {
        self.checks.push(c);
        self
    }

    /// No check failed (skips are allowed).
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.status != CheckStatus::Fail)
    }

    pub fn failed_checks(&self) -> Vec<&str> {
        self.checks
            .iter()
            .filter(|c| c.status == CheckStatus::Fail)
            .map(|c| c.name.as_str())
            .collect()
    }
}
