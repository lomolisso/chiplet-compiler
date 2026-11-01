use serde::Deserialize;

use super::json::Order;

/// `[passes.normalize]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct NormalizeConfig {
    /// Write `views/normalize.md`.
    pub views: bool,
    /// `[passes.normalize.order]`: ω per dtype family.
    pub order: OrderConfig,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            views: true,
            order: OrderConfig::default(),
        }
    }
}

/// `[passes.normalize.order]`: `"seq"` or `"tree:<fan-in>"` per family. The integer families default to
/// `"seq"` (every order gives the same bits); `float` and `complex` have no default: the pass refuses to
/// run until they are set.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderConfig {
    pub bool: Option<String>,
    pub sint: Option<String>,
    pub uint: Option<String>,
    pub float: Option<String>,
    pub complex: Option<String>,
}

/// The parsed orders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Orders {
    pub bool: Order,
    pub sint: Order,
    pub uint: Order,
    pub float: Order,
    pub complex: Order,
}

impl Orders {
    pub fn get(&self, family: &str) -> Order {
        match family {
            "bool" => self.bool.clone(),
            "sint" => self.sint.clone(),
            "uint" => self.uint.clone(),
            "complex" => self.complex.clone(),
            _ => self.float.clone(),
        }
    }
}

pub fn parse_order(s: &str) -> Result<Order, String> {
    if s == "seq" {
        return Ok(Order::Seq);
    }
    match s.strip_prefix("tree:").map(str::parse::<u32>) {
        Some(Ok(b)) if b >= 2 => Ok(Order::Tree { fan_in: b }),
        _ => Err(format!(
            "order `{s}`: expected \"seq\" or \"tree:<fan-in>\" with fan-in >= 2"
        )),
    }
}

impl OrderConfig {
    pub fn parse(&self) -> Result<Orders, String> {
        let int = |o: &Option<String>| o.as_deref().map(parse_order).unwrap_or(Ok(Order::Seq));
        let must = |o: &Option<String>, k: &str| {
            o.as_deref()
                .ok_or_else(|| {
                    format!("`{k}` must be set: a float reduction has no meaning until its order is fixed")
                })
                .and_then(parse_order)
        };
        Ok(Orders {
            bool: int(&self.bool)?,
            sint: int(&self.sint)?,
            uint: int(&self.uint)?,
            float: must(&self.float, "float")?,
            complex: must(&self.complex, "complex")?,
        })
    }
}

impl OrderConfig {
    /// Check the values that are set (the pass checks completeness when it runs).
    pub fn check(&self) -> Result<(), String> {
        for o in [&self.bool, &self.sint, &self.uint, &self.float, &self.complex]
            .into_iter()
            .flatten()
        {
            parse_order(o)?;
        }
        Ok(())
    }
}
