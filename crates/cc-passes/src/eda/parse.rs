//! Report and log parsers, pure functions of the tool's text (docs/tools/synopsys.md §4–5, cadence.md
//! §2–3). Every value comes back in the flow's units: area µm², time ps, leakage nW.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The log of a multi-design session split at `@@begin <id>` / `@@end <id>`. A design whose end marker
/// is missing gets the text up to the end of the log.
pub fn segments(log: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut cur: Option<(String, String)> = None;
    for line in log.lines() {
        let t = line.trim();
        if let Some(id) = t.strip_prefix("@@begin ") {
            if let Some((k, v)) = cur.take() {
                out.insert(k, v);
            }
            cur = Some((id.trim().to_string(), String::new()));
            continue;
        }
        if let Some(id) = t.strip_prefix("@@end ")
            && let Some((k, v)) = cur.take()
        {
            if k == id.trim() {
                out.insert(k, v);
            } else {
                cur = Some((k, v));
            }
            continue;
        }
        if let Some((_, v)) = cur.as_mut() {
            v.push_str(line);
            v.push('\n');
        }
    }
    if let Some((k, v)) = cur {
        out.insert(k, v);
    }
    out
}

/// Lines that start with `Error` (every tool here prints errors so; Genus as `Error   : …`). The echo
/// of a script command that mentions errors does not count: those lines start with the command.
pub fn errors(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("Error") || l.starts_with("ERROR:"))
        .map(str::to_string)
        .collect()
}

/// Unresolved references reported by `link` / `check_design` (DC `UID-341`, `LINK-5`; Genus's check
/// report lists them by name).
pub fn unresolved(log: &str, check_rpt: &str) -> u64 {
    let mut n = 0u64;
    for l in log.lines().chain(check_rpt.lines()) {
        let l = l.trim();
        if l.contains("unresolved references")
            && l.starts_with("Warning")
            && let Some(k) = l.split('\'').nth(3).and_then(|x| x.parse::<u64>().ok())
        {
            n = n.max(k);
        }
        if l.contains("Unable to resolve reference") {
            n = n.max(1);
        }
        if l.starts_with("Unresolved reference") || l.contains("(LINK-5)") {
            n = n.max(1);
        }
    }
    n
}

/// The first number after `label` on a line that starts with it (after whitespace).
fn after(text: &str, label: &str) -> Option<f64> {
    text.lines().find_map(|l| {
        let rest = l.trim_start().strip_prefix(label)?;
        rest.split_whitespace()
            .find_map(|w| w.trim_matches(|c| c == ':' || c == '=').parse::<f64>().ok())
    })
}

fn all_after(text: &str, label: &str) -> Vec<f64> {
    text.lines()
        .filter_map(|l| {
            let rest = l.trim_start().strip_prefix(label)?;
            rest.split_whitespace()
                .find_map(|w| w.trim_matches(|c| c == ':' || c == '=').parse::<f64>().ok())
        })
        .collect()
}

/// A power value with its unit (`119.7374 nW`, `1.2e-03 mW`) in nW.
pub fn power_nw(value: f64, unit: &str) -> Option<f64> {
    let k = match unit.trim() {
        "W" => 1e9,
        "mW" => 1e6,
        "uW" => 1e3,
        "nW" => 1.0,
        "pW" => 1e-3,
        "fW" => 1e-6,
        _ => return None,
    };
    Some(value * k)
}

/// Area, delay and leakage of one design as the synthesis reports state them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Measured {
    /// Total cell area, µm².
    pub area_um2: Option<f64>,
    /// Sequential (non-combinational) cell area, µm².
    pub seq_area_um2: Option<f64>,
    /// Longest path: arrival at the endpoint, ps.
    pub delay_ps: Option<f64>,
    /// Worst slack against the period, ps.
    pub slack_ps: Option<f64>,
    /// Cell leakage, nW.
    pub leakage_nw: Option<f64>,
    /// Area per child instance (hierarchy kept), µm².
    pub children: BTreeMap<String, f64>,
}

/// Design Compiler: `report_area -hierarchy`, `report_qor` and `report_power` (the time unit is the
/// library's, given in ps).
pub fn dc(area: &str, qor: &str, power: &str, time_unit_ps: f64) -> Measured {
    let lengths = all_after(qor, "Critical Path Length:");
    let slacks = all_after(qor, "Critical Path Slack:");
    let leakage = power.lines().find_map(|l| {
        let rest = l.trim_start().strip_prefix("Cell Leakage Power")?;
        let rest = rest.trim_start().strip_prefix('=')?;
        let mut w = rest.split_whitespace();
        let v: f64 = w.next()?.parse().ok()?;
        power_nw(v, w.next()?)
    });
    Measured {
        area_um2: after(area, "Total cell area:"),
        seq_area_um2: after(area, "Noncombinational area:"),
        delay_ps: lengths.iter().copied().reduce(f64::max).map(|x| x * time_unit_ps),
        slack_ps: slacks.iter().copied().reduce(f64::min).map(|x| x * time_unit_ps),
        leakage_nw: leakage,
        children: dc_children(area),
    }
}

/// The child rows of `report_area -hierarchy -nosplit`: `<instance> <absolute> <percent> …` under the
/// `Hierarchical cell` header, without the top row.
fn dc_children(area: &str) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    // 0: before the header, 1: in the header, 2: in the rows, 3: done.
    let mut state = 0u8;
    let mut rows = 0usize;
    for l in area.lines() {
        let t = l.trim();
        match state {
            0 if t.starts_with("Hierarchical cell") => state = 1,
            1 if t.starts_with("---") => state = 2,
            2 if t.starts_with("---") => state = 3,
            2 => {
                let w: Vec<&str> = t.split_whitespace().collect();
                if w.len() >= 2
                    && let Ok(a) = w[1].parse::<f64>()
                {
                    rows += 1;
                    if rows > 1 {
                        out.insert(w[0].to_string(), a);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Genus: `report_area` (the row of `top`), `report_timing` (`Data Path` and `Slack`, in ps) and
/// `report_power` (`Power Unit`, leakage of the `Subtotal` row).
pub fn genus(area: &str, timing: &str, power: &str, top: &str) -> Measured {
    let area_um2 = area.lines().find_map(|l| {
        let w: Vec<&str> = l.split_whitespace().collect();
        (w.first() == Some(&top) && w.len() >= 5)
            .then(|| w[2].parse::<f64>().ok())
            .flatten()
    });
    let data = all_after(timing, "Data Path:-");
    let slack = all_after(timing, "Slack:=");
    let unit = power
        .lines()
        .find_map(|l| l.trim().strip_prefix("Power Unit:").map(|u| u.trim().to_string()));
    let leakage = power.lines().find_map(|l| {
        let w: Vec<&str> = l.split_whitespace().collect();
        if w.first() != Some(&"Subtotal") {
            return None;
        }
        let v: f64 = w.get(1)?.parse().ok()?;
        power_nw(v, unit.as_deref()?)
    });
    Measured {
        area_um2,
        seq_area_um2: None,
        delay_ps: data.into_iter().reduce(f64::max),
        slack_ps: slack.into_iter().reduce(f64::min),
        leakage_nw: leakage,
        children: BTreeMap::new(),
    }
}

/// Yosys `stat -liberty` (`Chip area for module '\<top>': <a>`) and ABC's `stime -p` (`Delay = <d> ps`;
/// the last mapping's).
pub fn yosys(area: &str, log: &str, top: &str, period_ps: f64) -> Measured {
    let key = format!("Chip area for module '\\{top}':");
    let area_um2 = area
        .lines()
        .chain(log.lines())
        .find_map(|l| l.trim().strip_prefix(&key)?.trim().parse::<f64>().ok());
    let delay = log
        .lines()
        .filter_map(|l| {
            let rest = &l[l.find("Delay =")? + "Delay =".len()..];
            rest.split_whitespace().next()?.parse::<f64>().ok()
        })
        .next_back();
    Measured {
        area_um2,
        seq_area_um2: None,
        delay_ps: delay,
        slack_ps: delay.map(|d| period_ps - d),
        leakage_nw: None,
        children: BTreeMap::new(),
    }
}

/// An equivalence check's outcome.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Lec {
    /// Every compare point proven equivalent and none failing, aborted or unmatched.
    pub equivalent: bool,
    pub passing: u64,
    pub failing: u64,
    /// Aborted or inconclusive points (never a pass).
    pub aborted: u64,
    pub unmatched: u64,
    pub verdict: String,
}

fn row_total(text: &str, label: &str) -> Option<u64> {
    text.lines().find_map(|l| {
        let rest = l.trim_start().strip_prefix(label)?;
        rest.split_whitespace().last()?.parse().ok()
    })
}

/// Formality `report_status`: the verdict line and the matched compare point table.
pub fn formality(status: &str) -> Lec {
    let verdict = status
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("Verification "))
        .unwrap_or("no verdict")
        .to_string();
    let passing = row_total(status, "Passing (equivalent)").unwrap_or(0);
    let failing = row_total(status, "Failing (not equivalent)").unwrap_or(0);
    let aborted = row_total(status, "Aborted").unwrap_or(0) + row_total(status, "Unverified").unwrap_or(0);
    let unmatched = row_total(status, "Unmatched").unwrap_or(0);
    Lec {
        equivalent: verdict == "Verification SUCCEEDED" && failing == 0 && aborted == 0 && passing > 0,
        passing,
        failing,
        aborted,
        unmatched,
        verdict,
    }
}

/// Conformal LEC log: the last `compare` table (`Equivalent`, `Non-equivalent`, `Abort` rows).
pub fn conformal(log: &str) -> Lec {
    let last = log.rfind("Compared points").map_or("", |k| &log[k..]);
    let table = last.split("CPU time").next().unwrap_or("");
    let get = |label: &str| row_total(table, label).unwrap_or(0);
    let passing = get("Equivalent");
    let failing = get("Non-equivalent");
    let aborted = get("Abort") + get("Not-compared");
    let unmapped = log
        .lines()
        .filter_map(|l| {
            let t = l.trim();
            t.strip_suffix("Unmapped point(s) reported")
                .or_else(|| t.strip_suffix("unmapped point(s) reported"))
                .and_then(|n| n.trim().parse::<u64>().ok())
        })
        .sum::<u64>();
    let equivalent = passing > 0 && failing == 0 && aborted == 0;
    Lec {
        equivalent,
        passing,
        failing,
        aborted,
        unmatched: unmapped,
        verdict: if table.is_empty() {
            "no compare".into()
        } else if equivalent {
            "equivalent".into()
        } else {
            "not proven".into()
        },
    }
}

/// ABC `cec` / `dsec`: `Networks are equivalent` or not.
pub fn abc_cec(out: &str) -> Lec {
    let eq = out.lines().any(|l| l.contains("Networks are equivalent"));
    let ne = out
        .lines()
        .any(|l| l.contains("NOT EQUIVALENT") || l.contains("are not equivalent"));
    Lec {
        equivalent: eq && !ne,
        passing: u64::from(eq && !ne),
        failing: u64::from(ne),
        aborted: u64::from(!eq && !ne),
        unmatched: 0,
        verdict: if eq && !ne {
            "equivalent".into()
        } else if ne {
            "not equivalent".into()
        } else {
            "undecided".into()
        },
    }
}

/// Did `compile_ultra` use DesignWare (it adds `dw_foundation.sldb` when licensed; `compile_ultra(2)`)?
pub fn designware(segment: &str) -> bool {
    segment.contains("Added key list 'DesignWare'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_split_at_markers() {
        let s = segments("head\n@@begin a\nx\nError: bad\n@@end a\n@@begin b\ny\n");
        assert_eq!(s["a"], "x\nError: bad\n");
        assert_eq!(s["b"], "y\n");
        assert_eq!(errors(&s["a"]), ["Error: bad"]);
        assert!(errors(&s["b"]).is_empty());
    }

    #[test]
    fn dc_reports() {
        let area = "Number of cells:  1347\nCombinational area:   137.47\nNoncombinational area:  0.000000\nTotal cell area:                   137.474818\n";
        let qor = "  Timing Path Group 'vclk'\n  Critical Path Length:        998.31\n  Critical Path Slack:           1.69\n";
        let power = "Cell Leakage Power     = 119.7374 nW\n";
        let m = dc(area, qor, power, 1.0);
        assert_eq!(m.area_um2, Some(137.474818));
        assert_eq!(m.delay_ps, Some(998.31));
        assert_eq!(m.slack_ps, Some(1.69));
        assert_eq!(m.leakage_nw, Some(119.7374));
        let m = dc(
            area,
            "  Critical Path Length:  2.00\n  Critical Path Slack:  -0.37\n",
            "Cell Leakage Power = 1.5 uW\n",
            1000.0,
        );
        assert_eq!(
            (m.delay_ps, m.slack_ps, m.leakage_nw),
            (Some(2000.0), Some(-370.0), Some(1500.0))
        );
    }

    #[test]
    fn dc_hierarchy_rows() {
        let area = "Hierarchical area distribution\n------------------------------\n                                  Global cell area          Local cell area\n                                  -------------------  ---------------------------\nHierarchical cell                 Absolute    Percent  Combi-    Noncombi-  Black-\n                                  Total       Total    national  national   boxes   Design\n--------------------------------  ----------  -------  --------  ---------  ------  ---------\nsplit_w16n8                         500.0     100.0     20.0       0.0     0.0  split_w16n8\nu0_0                                 160.0      32.0    160.0       0.0     0.0  split_piece_n8\nu0_1                                 160.0      32.0    160.0       0.0     0.0  split_piece_n8\n--------------------------------  ----------  -------  --------  ---------  ------  ---------\nTotal                                                    500.0       0.0     0.0\n";
        let c = dc_children(area);
        assert_eq!(c.len(), 2);
        assert_eq!(c["u0_0"], 160.0);
    }

    #[test]
    fn unresolved_references() {
        assert_eq!(
            unresolved(
                "Warning: Design 'unres_1' has '1' unresolved references. For more detailed information, use the \"link\" command. (UID-341)\n",
                ""
            ),
            1
        );
        assert_eq!(unresolved("", "No unresolved references in design 'x'\n"), 0);
    }

    #[test]
    fn genus_reports() {
        let area = "Instance Module  Cell Count  Cell Area  Net Area   Total Area       Wireload\n------\nmul_s32                1006   1858.542     0.000     1858.542 5K_hvratio_1_1 (D)\n";
        let timing = "         Data Path:-     3395\n             Slack:=  1996549\n";
        let power =
            "Power Unit: W\n    Subtotal     4.01583e-05  1.69663e-06  1.15963e-06  4.30146e-05 100.00%\n";
        let m = genus(area, timing, power, "mul_s32");
        assert_eq!(m.area_um2, Some(1858.542));
        assert_eq!(m.delay_ps, Some(3395.0));
        assert!((m.leakage_nw.unwrap() - 40158.3).abs() < 1e-6);
    }

    #[test]
    fn formality_status() {
        let ok = "Verification SUCCEEDED\n----\nPassing (equivalent)           0       0       0       0      32       0       0      32\nFailing (not equivalent)       0       0       0       0       0       0       0       0\n";
        assert!(formality(ok).equivalent);
        let bad = ok.replace("SUCCEEDED", "FAILED").replace(
            "Failing (not equivalent)       0       0       0       0       0       0       0       0",
            "Failing (not equivalent)       0       0       0       0       1       0       0       1",
        );
        let l = formality(&bad);
        assert!(!l.equivalent);
        assert_eq!(l.failing, 1);
        assert!(!formality("Verification INCONCLUSIVE\n").equivalent);
    }

    #[test]
    fn conformal_and_abc() {
        let log = "Compared points      PO        Total\n------\nEquivalent           32        32\n======\nCPU time     : 1.32    seconds\n";
        assert!(conformal(log).equivalent);
        let ab = "Compared points      PO        Total\nEquivalent           31        31\nAbort                 1         1\nCPU time\n";
        assert!(!conformal(ab).equivalent);
        assert!(abc_cec("Networks are equivalent.  Time =     0.01 sec\n").equivalent);
        assert!(!abc_cec("Networks are NOT EQUIVALENT.\n").equivalent);
        assert!(!abc_cec("").equivalent);
    }

    #[test]
    fn yosys_reports() {
        let log = "ABC: WireLoad = \"none\"  Gates =    155 ( 11.6 %)   Cap =  3.3 ff (  0.0 %)   Area =      163.86 (100.0 %)   Delay =   576.81 ps  ( 21.9 %)\n   Chip area for module '\\mul_s8': 163.856000\n";
        let m = yosys("", log, "mul_s8", 2000.0);
        assert_eq!(m.area_um2, Some(163.856));
        assert_eq!(m.delay_ps, Some(576.81));
    }
}
