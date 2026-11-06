//! Area against width, per primitive series (docs/passes/characterize.md §6): area should not decrease
//! with width, an integer multiplier should grow about quadratically and an adder about linearly
//! (`n x n` array and tree multipliers have Θ(n²) cells, Wallace 1964; ripple and prefix adders Θ(n) to
//! Θ(n log n)). Points off these trends are notes: synthesis picks different structures at different
//! widths. A clear violation fails: a wider design with less than [`CLEAR_DROP`] of a narrower one's area
//! has lost logic, which no choice of structure explains.

use std::collections::BTreeMap;

use super::catalog::Kind;
use super::json::Row;

/// One series: a primitive at every width, on one kit at one timing point.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    pub name: String,
    pub op: String,
    pub kit: String,
    pub target: String,
    /// (width, area µm², delay ps), by width.
    pub points: Vec<(u32, f64, Option<f64>)>,
}

/// The series of the rows whose width is a parameter: integer and bit-vector primitives (float formats
/// are not one family of widths).
pub fn series(rows: &[Row]) -> Vec<Series> {
    let mut m: BTreeMap<(String, String, String), Series> = BTreeMap::new();
    for r in rows {
        if !matches!(r.family.as_str(), "sint" | "uint" | "bits") || r.op == "convert" {
            continue;
        }
        let Some(a) = r.measured.area_um2 else {
            continue;
        };
        let name = format!(
            "{}{}@{}",
            r.op,
            r.variant.as_ref().map(|v| format!(".{v}")).unwrap_or_default(),
            match r.kind {
                Kind::Split => "w".to_string(),
                _ => r.family.clone(),
            }
        );
        let s = m
            .entry((name.clone(), r.kit.clone(), r.target.clone()))
            .or_insert_with(|| Series {
                name,
                op: r.op.clone(),
                kit: r.kit.clone(),
                target: r.target.clone(),
                points: Vec::new(),
            });
        s.points.push((r.width, a, r.measured.delay_ps));
    }
    let mut v: Vec<Series> = m.into_values().collect();
    for s in &mut v {
        s.points.sort_by_key(|p| p.0);
    }
    v
}

/// Least-squares slope of log(area) against log(width) over the points with positive area.
pub fn exponent(points: &[(u32, f64, Option<f64>)]) -> Option<f64> {
    let pts: Vec<(f64, f64)> = points
        .iter()
        .filter(|p| p.1 > 0.0 && p.0 > 0)
        .map(|p| (f64::from(p.0).ln(), p.1.ln()))
        .collect();
    if pts.len() < 3 {
        return None;
    }
    let n = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (mx, my) = (sx / n, sy / n);
    let (num, den) = pts.iter().fold((0.0, 0.0), |a, p| {
        (a.0 + (p.0 - mx) * (p.1 - my), a.1 + (p.0 - mx).powi(2))
    });
    (den > 0.0).then(|| num / den)
}

/// The growth exponent expected for an op, with the band that is not flagged.
pub fn expected(op: &str) -> Option<(f64, f64, f64)> {
    match op {
        "multiply" => Some((2.0, 1.5, 2.5)),
        "add" | "subtract" => Some((1.0, 0.7, 1.5)),
        _ => None,
    }
}

/// A wider point below this fraction of a narrower point's area is a clear violation.
pub const CLEAR_DROP: f64 = 0.5;

/// Clear violations of one series: a width whose area is below `CLEAR_DROP` of a narrower width's.
pub fn violations(s: &Series) -> Vec<String> {
    let mut v = Vec::new();
    for (k, hi) in s.points.iter().enumerate() {
        if let Some(lo) = s.points[..k]
            .iter()
            .filter(|lo| lo.0 < hi.0)
            .find(|lo| hi.1 < CLEAR_DROP * lo.1)
        {
            v.push(format!(
                "{} {} {}: area {:.3} at {} bits is below half of {:.3} at {} bits",
                s.name, s.kit, s.target, hi.1, hi.0, lo.1, lo.0
            ));
        }
    }
    v
}

/// Flags of one series: widths where the area decreases, and an exponent outside the expected band.
pub fn flags(s: &Series) -> Vec<String> {
    let mut f = Vec::new();
    for w in s.points.windows(2) {
        if w[1].1 < w[0].1 {
            f.push(format!(
                "{} {} {}: area falls from {:.3} at {} bits to {:.3} at {} bits",
                s.name, s.kit, s.target, w[0].1, w[0].0, w[1].1, w[1].0
            ));
        }
    }
    if let (Some((e, lo, hi)), Some(x)) = (expected(&s.op), exponent(&s.points))
        && (x < lo || x > hi)
    {
        f.push(format!(
            "{} {} {}: area grows as width^{x:.2}, expected about width^{e}",
            s.name, s.kit, s.target
        ));
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(op: &str, pts: &[(u32, f64)]) -> Series {
        Series {
            name: format!("{op}@sint"),
            op: op.into(),
            kit: "k".into(),
            target: "clock".into(),
            points: pts.iter().map(|&(w, a)| (w, a, None)).collect(),
        }
    }

    #[test]
    fn trends() {
        let quad = s("multiply", &[(2, 4.0), (4, 16.0), (8, 64.0), (16, 256.0)]);
        assert!((exponent(&quad.points).unwrap() - 2.0).abs() < 1e-9);
        assert!(flags(&quad).is_empty());
        let lin_as_mul = s("multiply", &[(2, 2.0), (4, 4.0), (8, 8.0)]);
        assert_eq!(flags(&lin_as_mul).len(), 1);
        let dip = s("add", &[(2, 2.0), (4, 3.9), (8, 3.8), (16, 16.0)]);
        assert!(flags(&dip).iter().any(|f| f.contains("area falls")));
        assert!(violations(&dip).is_empty());
        let lost = s("add", &[(2, 2.0), (4, 4.0), (8, 1.9), (16, 16.0)]);
        assert_eq!(violations(&lost).len(), 1);
        assert!(violations(&lost)[0].contains("at 8 bits"));
    }
}
