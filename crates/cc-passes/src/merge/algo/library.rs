//! Prices from the characterize library (docs/passes/merge.md §4.2): areas at the kit's `clock` target,
//! delays at its `fast` target. Keys follow the catalog (`<op>[.<variant>]@<type>[><type>]`); a key
//! without a row (a pending soft-float op) is priced as the largest operator row of the same type and
//! flagged `estimated`.

use std::collections::HashMap;

use crate::characterize::json::Library;
use crate::merge::json::{BlockKind, Price};

#[derive(Debug, Clone, Copy, Default)]
struct Cell {
    area: f64,
    delay: f64,
}

#[derive(Debug, Clone)]
pub struct Lib {
    pub kit: String,
    /// The kit's clock period (the `clock` target).
    pub period_ps: f64,
    rows: HashMap<String, Cell>,
    /// Per type, the largest operator row (area, delay): the estimate of a missing operator.
    by_type: HashMap<String, Cell>,
    mux_widths: Vec<u32>,
    mux_inputs: Vec<u32>,
    reg_widths: Vec<u32>,
}

/// Bits of a type name (`i8`, `ui16`, `i1`, `f32`, `bf16`, `f8e4m3fn`, `cf32`).
pub fn bits(t: &str) -> u32 {
    if let Some(r) = t.strip_prefix("cf") {
        return 2 * bits(&format!("f{r}"));
    }
    match t {
        "i1" => 1,
        "f32" => 32,
        "f16" | "bf16" => 16,
        "f64" => 64,
        _ if t.starts_with("f8") => 8,
        _ => t
            .trim_start_matches("ui")
            .trim_start_matches('i')
            .parse()
            .unwrap_or(32),
    }
}

impl Lib {
    pub fn new(lib: &Library, kit: &str) -> Result<Lib, String> {
        let period_ps = lib
            .targets
            .iter()
            .find(|t| t.kit == kit && t.target == "clock")
            .map(|t| t.period_ps)
            .ok_or_else(|| format!("the library has no `clock` target for kit {kit}"))?;
        let mut rows: HashMap<String, Cell> = HashMap::new();
        let mut fast: HashMap<&str, f64> = HashMap::new();
        for r in lib.rows.iter().filter(|r| r.kit == kit) {
            match r.target.as_str() {
                "clock" => {
                    let c = rows.entry(r.key.clone()).or_default();
                    c.area = r.measured.area_um2.unwrap_or(0.0);
                    if c.delay == 0.0 {
                        c.delay = r.measured.delay_ps.unwrap_or(0.0);
                    }
                }
                "fast" => {
                    fast.insert(&r.key, r.measured.delay_ps.unwrap_or(0.0));
                }
                _ => {}
            }
        }
        for (k, d) in fast {
            if let Some(c) = rows.get_mut(k) {
                c.delay = d;
            }
        }
        let mut by_type: HashMap<String, Cell> = HashMap::new();
        let (mut mux_widths, mut mux_inputs, mut reg_widths) = (Vec::new(), Vec::new(), Vec::new());
        for (k, c) in &rows {
            if let Some(rest) = k.strip_prefix("mux")
                && let Some((n, w)) = rest.split_once("@w")
            {
                if let (Ok(n), Ok(w)) = (n.parse::<u32>(), w.parse::<u32>()) {
                    mux_inputs.push(n);
                    mux_widths.push(w);
                }
                continue;
            }
            if let Some(w) = k.strip_prefix("register@w").and_then(|w| w.parse::<u32>().ok()) {
                reg_widths.push(w);
                continue;
            }
            if k.starts_with("accumulator.") || k.starts_with("split@") {
                continue;
            }
            if let Some((_, ty)) = k.split_once('@') {
                let ty = ty.split('>').next_back().unwrap_or(ty).to_string();
                let e = by_type.entry(ty).or_default();
                e.area = e.area.max(c.area);
                e.delay = e.delay.max(c.delay);
            }
        }
        for v in [&mut mux_widths, &mut mux_inputs, &mut reg_widths] {
            v.sort_unstable();
            v.dedup();
        }
        Ok(Lib {
            kit: kit.into(),
            period_ps,
            rows,
            by_type,
            mux_widths,
            mux_inputs,
            reg_widths,
        })
    }

    fn key(&self, k: &str, ty: &str, p: &mut Price) {
        p.keys.push(k.to_string());
        match self.rows.get(k) {
            Some(c) => {
                p.area_um2 += c.area;
                p.delay_ps = p.delay_ps.max(c.delay);
            }
            None => {
                p.estimated = true;
                if let Some(c) = self.by_type.get(ty) {
                    p.area_um2 += c.area;
                    p.delay_ps = p.delay_ps.max(c.delay);
                }
            }
        }
    }

    /// A register of `w` bits (the smallest characterized width that holds it, several when wider).
    pub fn register(&self, w: u32) -> Price {
        let mut p = empty();
        let (n, rw) = fit(&self.reg_widths, w);
        if rw == 0 {
            p.estimated = true;
            return p;
        }
        for _ in 0..n {
            self.key(&format!("register@w{rw}"), "", &mut p);
        }
        p.delay_ps = 0.0;
        p
    }

    /// An `n`-input multiplexer of `w` bits (`n` above the library: the last input's increment per
    /// input; wider than the library: several side by side).
    pub fn mux(&self, n: u32, w: u32) -> Price {
        let mut p = empty();
        if n < 2 {
            return p;
        }
        let (copies, mw) = fit(&self.mux_widths, w);
        let top = self.mux_inputs.last().copied().unwrap_or(0);
        if mw == 0 || top < 2 {
            p.estimated = true;
            return p;
        }
        let one = |k: u32| self.rows.get(&format!("mux{k}@w{mw}")).copied();
        let cell = if n <= top {
            match self.mux_inputs.iter().find(|&&k| k >= n) {
                Some(&k) => {
                    p.keys.push(format!("mux{k}@w{mw}"));
                    if k != n {
                        p.estimated = true;
                    }
                    one(k)
                }
                None => None,
            }
        } else {
            p.estimated = true;
            p.keys.push(format!("mux{top}@w{mw}"));
            let below = self.mux_inputs.iter().rev().nth(1).copied().unwrap_or(top);
            match (one(top), one(below)) {
                (Some(a), Some(b)) => {
                    let step = (a.area - b.area).max(0.0) / f64::from((top - below).max(1));
                    Some(Cell {
                        area: a.area + step * f64::from(n - top),
                        delay: a.delay * (f64::from(n).log2() / f64::from(top).log2()),
                    })
                }
                (Some(a), None) => Some(a),
                _ => None,
            }
        };
        match cell {
            Some(c) => {
                p.area_um2 = c.area * f64::from(copies);
                p.delay_ps = c.delay;
            }
            None => p.estimated = true,
        }
        p
    }

    /// The price of a block of kind `kind`, op `op`, attributes `attrs` and types `types` (operands then
    /// result for an operator).
    pub fn block(
        &self,
        kind: BlockKind,
        op: &str,
        attrs: &std::collections::BTreeMap<String, String>,
        types: &[String],
        literal: bool,
    ) -> Price {
        let ty = |i: usize| types.get(i).cloned().unwrap_or_default();
        match kind {
            BlockKind::Input | BlockKind::Output => empty(),
            BlockKind::Const if literal => empty(),
            BlockKind::Const | BlockKind::Acc => self.register(bits(&ty(0))),
            BlockKind::Op => {
                let mut p = empty();
                let tys: Vec<&String> = types.iter().filter(|t| !t.starts_with("rp:")).collect();
                let rt = tys.last().map(|s| s.as_str()).unwrap_or("");
                let a0 = tys.first().map(|s| s.as_str()).unwrap_or("");
                if let Some(inner) = rt.strip_prefix('c').filter(|r| r.starts_with('f')) {
                    let parts: &[(&str, u32)] = match op {
                        "multiply" => &[("multiply", 4), ("add", 1), ("subtract", 1)],
                        "add" | "subtract" | "negate" => &[(op, 2)],
                        _ => &[],
                    };
                    for (o, k) in parts {
                        for _ in 0..*k {
                            self.key(&format!("{o}@{inner}"), inner, &mut p);
                        }
                    }
                    return p;
                }
                match op {
                    "complex" | "real" | "imag" => {}
                    "convert" if a0 == rt => {}
                    "convert" => self.key(&format!("convert@{a0}>{rt}"), rt, &mut p),
                    "compare" => {
                        let dir = attrs.get("comparison_direction").cloned().unwrap_or_default();
                        let ot = match attrs.get("compare_type").map(String::as_str) {
                            Some("UNSIGNED") if a0.starts_with('i') && a0 != "i1" => format!("u{a0}"),
                            Some("SIGNED") if a0.starts_with("ui") => a0[1..].to_string(),
                            _ => a0.to_string(),
                        };
                        self.key(&format!("compare.{dir}@{ot}"), &ot, &mut p);
                    }
                    "select" if rt == "i1" => self.key("mux2@w1", "i1", &mut p),
                    "select" => self.key(&format!("select@{rt}"), rt, &mut p),
                    "is_finite" => self.key(&format!("is_finite@{a0}"), a0, &mut p),
                    _ if rt == "i1" && !a0.is_empty() && a0 != "i1" => {
                        self.key(&format!("{op}@{a0}"), a0, &mut p)
                    }
                    _ => self.key(&format!("{op}@{rt}"), rt, &mut p),
                }
                p
            }
        }
    }
}

fn empty() -> Price {
    Price {
        keys: Vec::new(),
        area_um2: 0.0,
        delay_ps: 0.0,
        estimated: false,
    }
}

/// `(copies, width)`: the smallest width in `ws` that holds `w`, or copies of the widest.
fn fit(ws: &[u32], w: u32) -> (u32, u32) {
    match ws.iter().find(|&&x| x >= w) {
        Some(&x) => (1, x),
        None => match ws.last() {
            Some(&x) if x > 0 => (w.div_ceil(x), x),
            _ => (0, 0),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_of_type_names() {
        assert_eq!(bits("i8"), 8);
        assert_eq!(bits("ui16"), 16);
        assert_eq!(bits("i1"), 1);
        assert_eq!(bits("bf16"), 16);
        assert_eq!(bits("f8e4m3fn"), 8);
        assert_eq!(bits("cf32"), 64);
        assert_eq!(fit(&[1, 2, 4, 8, 16, 32], 3), (1, 4));
        assert_eq!(fit(&[1, 2, 4, 8, 16, 32], 64), (2, 32));
    }
}
