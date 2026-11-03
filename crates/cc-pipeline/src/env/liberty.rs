//! Merging Liberty libraries of one corner into one (docs/tools/pdks.md §4).
//!
//! ASAP7 splits one VT/corner into five libraries by cell group. Design Compiler's library analysis
//! (`alib`) runs per library and leaves placeholders for a group without inverters or flops, so every
//! compile repeats it; Yosys' `abc -liberty` reads one file. Both get one library: the first file's header
//! and groups, then every cell of the others, then their other top-level groups that the first lacks (the
//! same practice as OpenROAD's `mergeLib.pl` for ASAP7).

use std::collections::{BTreeMap, BTreeSet};

/// One top-level group of a `library`: `kind (name) { … }`, with its byte range.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Group {
    kind: String,
    name: String,
    start: usize,
    end: usize,
}

/// The top-level groups of the first `library` group in `text`, its body start and its closing brace.
fn groups(text: &str) -> Result<(Vec<Group>, usize, usize), String> {
    let b = text.as_bytes();
    let lib = find_library(text).ok_or("no `library (…) {` group")?;
    let open = text[lib..].find('{').ok_or("no `{` after `library`")? + lib;
    let mut out = Vec::new();
    let (mut depth, mut i, mut head) = (0usize, open + 1, open + 1);
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                i = text[i + 2..]
                    .find("*/")
                    .map(|k| i + 2 + k + 1)
                    .ok_or("unterminated comment")?;
                if depth == 0 {
                    head = i + 1;
                }
            }
            b';' if depth == 0 => head = i + 1,
            b'{' => {
                if depth == 0 {
                    let (kind, name) = group_head(&text[head..i])
                        .ok_or_else(|| format!("malformed group head {:?}", text[head..i].trim()))?;
                    out.push(Group {
                        kind,
                        name,
                        start: head,
                        end: 0,
                    });
                }
                depth += 1;
            }
            b'}' => {
                if depth == 0 {
                    return Ok((out, open + 1, i));
                }
                depth -= 1;
                if depth == 0 {
                    let g = out.last_mut().ok_or("unbalanced braces")?;
                    g.end = i + 1;
                    head = i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    Err("unbalanced braces: the library group does not close".into())
}

fn find_library(text: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(k) = text[from..].find("library") {
        let at = from + k;
        let before_ok = at == 0 || !text.as_bytes()[at - 1].is_ascii_alphanumeric();
        let rest = text[at + "library".len()..].trim_start();
        if before_ok && rest.starts_with('(') {
            return Some(at);
        }
        from = at + 1;
    }
    None
}

/// `kind (name)` with comments and quotes stripped.
fn group_head(s: &str) -> Option<(String, String)> {
    let s = strip_comments(s);
    let s = s.trim();
    let open = s.find('(')?;
    let close = s.rfind(')')?;
    let kind = s[..open].trim();
    if kind.is_empty() || !kind.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let name = s[open + 1..close].trim().trim_matches('"').trim();
    Some((kind.to_string(), name.to_string()))
}

fn strip_comments(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(k) = rest.find("/*") {
        out.push_str(&rest[..k]);
        match rest[k + 2..].find("*/") {
            Some(e) => rest = &rest[k + 2 + e + 2..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Simple top-level attributes (`name : value ;`) of the library body (outside every group).
fn attributes(text: &str, body: usize, end: usize, gs: &[Group]) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    let mut at = body;
    let mut chunks = Vec::new();
    for g in gs {
        chunks.push(&text[at..g.start]);
        at = g.end;
    }
    chunks.push(&text[at..end]);
    for c in chunks {
        for stmt in strip_comments(c).split(';') {
            if let Some((k, v)) = stmt.split_once(':') {
                let k = k.trim();
                if !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    m.insert(k.to_string(), v.trim().trim_matches('"').to_string());
                }
            }
        }
    }
    m
}

/// Attributes every merged library must agree on: the units and the nominal operating point.
pub const AGREE: &[&str] = &[
    "delay_model",
    "time_unit",
    "voltage_unit",
    "current_unit",
    "leakage_power_unit",
    "pulling_resistance_unit",
    "nom_process",
    "nom_temperature",
    "nom_voltage",
];

/// The merged library and what the merge did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Merged {
    pub text: String,
    pub cells: usize,
    /// Non-cell groups present in several files with different text (the first file's is kept).
    pub differing: Vec<String>,
}

/// Merge `libs` (file contents, in order) into one library named `name`. Fails on a cell defined twice
/// or on a disagreement in [`AGREE`].
pub fn merge(name: &str, libs: &[&str]) -> Result<Merged, String> {
    let (first, rest) = libs.split_first().ok_or("no libraries to merge")?;
    let (bg, body, bend) = groups(first).map_err(|e| format!("library 1: {e}"))?;
    let base_attrs = attributes(first, body, bend, &bg);
    let mut seen: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut cells = BTreeSet::new();
    for g in &bg {
        let t = norm(&first[g.start..g.end]);
        if g.kind == "cell" && !cells.insert(g.name.clone()) {
            return Err(format!("cell {} defined twice in library 1", g.name));
        }
        seen.entry((g.kind.clone(), g.name.clone())).or_insert(t);
    }
    let mut extra = String::new();
    let mut differing = BTreeSet::new();
    for (n, text) in rest.iter().enumerate() {
        let (gs, b, e) = groups(text).map_err(|e| format!("library {}: {e}", n + 2))?;
        let attrs = attributes(text, b, e, &gs);
        for k in AGREE {
            if base_attrs.get(*k) != attrs.get(*k) {
                return Err(format!(
                    "library {} disagrees on `{k}`: {:?} vs {:?}",
                    n + 2,
                    base_attrs.get(*k),
                    attrs.get(*k)
                ));
            }
        }
        for g in &gs {
            let raw = &text[g.start..g.end];
            if g.kind == "cell" {
                if !cells.insert(g.name.clone()) {
                    return Err(format!("cell {} defined twice", g.name));
                }
            } else if let Some(t) = seen.get(&(g.kind.clone(), g.name.clone())) {
                if *t != norm(raw) {
                    differing.insert(format!("{} ({})", g.kind, g.name));
                }
                continue;
            } else {
                seen.insert((g.kind.clone(), g.name.clone()), norm(raw));
            }
            extra.push('\n');
            extra.push_str(raw.trim_start_matches(['\n', '\r']));
            extra.push('\n');
        }
    }
    let lib = find_library(first).ok_or("no library group")?;
    let open = first[lib..].find('{').ok_or("no `{`")? + lib;
    let mut out = String::with_capacity(first.len() + extra.len());
    out.push_str(&first[..lib]);
    out.push_str(&format!("library ({name}) "));
    out.push_str(&first[open..bend]);
    out.push_str(&extra);
    out.push_str("}\n");
    Ok(Merged {
        text: out,
        cells: cells.len(),
        differing: differing.into_iter().collect(),
    })
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The cells of a library with a `latch` group and with an `ff` group (sequential cells).
pub fn sequential_cells(text: &str) -> Result<(BTreeSet<String>, BTreeSet<String>), String> {
    let (gs, _, _) = groups(text)?;
    let (mut latches, mut flops) = (BTreeSet::new(), BTreeSet::new());
    for g in gs.iter().filter(|g| g.kind == "cell") {
        let inner = children(&text[g.start..g.end]);
        if inner.iter().any(|(k, _, _)| k == "latch") {
            latches.insert(g.name.clone());
        }
        if inner.iter().any(|(k, _, _)| k == "ff" || k == "ff_bank") {
            flops.insert(g.name.clone());
        }
    }
    Ok((latches, flops))
}

/// Every cell name of a library.
pub fn cell_names(text: &str) -> Result<BTreeSet<String>, String> {
    let (gs, _, _) = groups(text)?;
    Ok(gs
        .into_iter()
        .filter(|g| g.kind == "cell")
        .map(|g| g.name)
        .collect())
}

/// The simple attributes of the library group itself (units, operating point, defaults).
pub fn header(text: &str) -> Result<BTreeMap<String, String>, String> {
    let (gs, body, end) = groups(text)?;
    Ok(attributes(text, body, end, &gs))
}

/// The `capacitance` of the first input pin of `cell`, in the library's capacitance unit.
pub fn input_capacitance(text: &str, cell: &str) -> Result<Option<f64>, String> {
    let (gs, _, _) = groups(text)?;
    let Some(g) = gs.iter().find(|g| g.kind == "cell" && g.name == cell) else {
        return Ok(None);
    };
    let ct = &text[g.start..g.end];
    for (k, _, pin) in children(ct) {
        if k != "pin" {
            continue;
        }
        let attrs = simple_attributes(pin);
        if attrs.get("direction").map(String::as_str) == Some("input")
            && let Some(c) = attrs.get("capacitance")
        {
            return c
                .parse()
                .map(Some)
                .map_err(|_| format!("{cell}: capacitance {c:?}"));
        }
    }
    Ok(None)
}

/// The simple attributes directly inside one group's text (not those of nested groups).
fn simple_attributes(group: &str) -> BTreeMap<String, String> {
    let Some(open) = group.find('{') else {
        return BTreeMap::new();
    };
    let body = &group[open + 1..group.rfind('}').unwrap_or(group.len())];
    let mut flat = String::new();
    let mut depth = 0usize;
    for ch in body.chars() {
        match ch {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            c if depth == 0 => flat.push(c),
            _ => {}
        }
    }
    let mut m = BTreeMap::new();
    for stmt in strip_comments(&flat).split(';') {
        if let Some((k, v)) = stmt.split_once(':') {
            let k = k.trim();
            if !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                m.insert(k.to_string(), v.trim().trim_matches('"').to_string());
            }
        }
    }
    m
}

/// The groups directly inside one group's text (`cell (X) { … }`): kind, name and text.
fn children(group: &str) -> Vec<(String, String, &str)> {
    let b = group.as_bytes();
    let Some(open) = group.find('{') else {
        return Vec::new();
    };
    let (mut depth, mut i, mut head, mut start) = (0usize, open + 1, open + 1, 0usize);
    let mut out = Vec::new();
    let mut cur: Option<(String, String)> = None;
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    i += 1;
                }
            }
            b';' if depth == 0 => head = i + 1,
            b'{' => {
                if depth == 0 {
                    cur = group_head(&group[head..i]);
                    start = head;
                }
                depth += 1;
            }
            b'}' => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
                if depth == 0 {
                    if let Some((k, n)) = cur.take() {
                        out.push((k, n, &group[start..i + 1]));
                    }
                    head = i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = r#"/* kit A */
library (lib_a) {
  delay_model : table_lookup;
  time_unit : "1ps";
  capacitive_load_unit (1,ff);
  lu_table_template (t7) { variable_1 : x; index_1 ("1, 2"); }
  cell (INVx1) { area : 0.04; pin (A) { direction : input; capacitance : 0.62; } }
}
"#;
    const B: &str = r#"library (lib_b) {
  delay_model : table_lookup;
  time_unit : "1ps";
  lu_table_template (t7) { variable_1 : x; index_1 ("1, 2"); }
  lu_table_template (t9) { variable_1 : y; }
  cell (DFFx1) { area : 0.29; ff (IQ, IQN) { next_state : "D"; } }
  cell (DHLx1) { area : 0.2; latch (IQ, IQN) { data_in : "D"; } }
}
"#;

    #[test]
    fn merges_cells_and_new_templates() {
        let m = merge("lib_ab", &[A, B]).unwrap();
        assert_eq!(m.cells, 3);
        assert!(m.differing.is_empty());
        assert!(m.text.contains("library (lib_ab) {"));
        assert_eq!(m.text.matches("lu_table_template (t7)").count(), 1);
        assert!(m.text.contains("lu_table_template (t9)"));
        let names = cell_names(&m.text).unwrap();
        assert_eq!(names.into_iter().collect::<Vec<_>>(), ["DFFx1", "DHLx1", "INVx1"]);
        let (latches, flops) = sequential_cells(&m.text).unwrap();
        assert_eq!(latches.into_iter().collect::<Vec<_>>(), ["DHLx1"]);
        assert_eq!(flops.into_iter().collect::<Vec<_>>(), ["DFFx1"]);
        assert_eq!(input_capacitance(&m.text, "INVx1").unwrap(), Some(0.62));
        assert_eq!(input_capacitance(&m.text, "DFFx1").unwrap(), None);
        assert_eq!(
            header(&m.text).unwrap().get("time_unit").map(String::as_str),
            Some("1ps")
        );
    }

    #[test]
    fn rejects_duplicate_cells_and_unit_mismatch() {
        assert!(merge("x", &[A, A]).unwrap_err().contains("defined twice"));
        let ns = B.replace("1ps", "1ns");
        assert!(merge("x", &[A, &ns]).unwrap_err().contains("time_unit"));
    }

    #[test]
    fn reports_differing_shared_groups() {
        let b = B.replace("index_1 (\"1, 2\")", "index_1 (\"1, 3\")");
        let m = merge("x", &[A, &b]).unwrap();
        assert_eq!(m.differing, ["lu_table_template (t7)"]);
    }
}
