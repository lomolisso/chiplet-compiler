//! A gate-level Verilog netlist as written by the synthesis tools: its modules and the cell types it
//! instantiates. Only structural Verilog (`module … endmodule` with declarations, `assign` and
//! instances) is expected; anything else is reported as an instance of an unknown type.

use std::collections::{BTreeMap, BTreeSet};

/// Modules defined in the netlist and the count of instances per type (cells and submodules).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Netlist {
    pub modules: BTreeSet<String>,
    pub instances: BTreeMap<String, u64>,
}

impl Netlist {
    /// Instance types that are neither a module of the netlist nor a cell of `cells`.
    pub fn foreign<'a>(&'a self, cells: &'a BTreeSet<String>) -> impl Iterator<Item = &'a String> + 'a {
        self.instances
            .keys()
            .filter(move |t| !self.modules.contains(*t) && !cells.contains(*t))
    }

    /// Instances of the given cell types.
    pub fn count_of(&self, cells: &BTreeSet<String>) -> u64 {
        self.instances
            .iter()
            .filter(|(t, _)| cells.contains(*t))
            .map(|(_, n)| n)
            .sum()
    }
}

/// The direction of a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dir {
    Input,
    Output,
    Inout,
}

/// The ports of `module` (name → direction and width), from its header (ANSI style: `input wire [7:0]
/// a, …`) and its body declarations (`input [7:0] a;`, as synthesis tools write them). `None` when the
/// module is not in the text.
pub fn ports(text: &str, module: &str) -> Option<BTreeMap<String, (Dir, u32)>> {
    let text = strip(text);
    let mut stmts = text.split(';').map(str::trim);
    let header = loop {
        let st = stmts.next()?;
        let st = st.trim_start_matches("endmodule").trim();
        if let Some(rest) = st.strip_prefix("module") {
            let rest = rest.trim_start();
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '\\')
                .collect();
            if name.trim_start_matches('\\') == module {
                break rest[name.len()..].to_string();
            }
        }
    };
    let mut out = BTreeMap::new();
    let mut order = Vec::new();
    // Header: `( item, item, … )`; an ANSI item carries a direction that applies to the names after it.
    if let (Some(o), Some(c)) = (header.find('('), header.rfind(')')) {
        let mut cur: Option<(Dir, u32)> = None;
        for item in header[o + 1..c].split(',') {
            let (decl, names) = declaration(item);
            if let Some(d) = decl {
                cur = Some(d);
            }
            for n in names {
                order.push(n.clone());
                if let Some(d) = cur.filter(|_| decl.is_some() || cur.is_some()) {
                    out.insert(n, d);
                }
            }
        }
    }
    // Body declarations up to `endmodule`.
    for st in stmts {
        if st.starts_with("endmodule") || st.contains("\nendmodule") {
            let head = st.split("endmodule").next().unwrap_or("");
            if let (Some(d), names) = declaration(head) {
                for n in names {
                    out.insert(n, d);
                }
            }
            break;
        }
        if let (Some(d), names) = declaration(st) {
            for n in names {
                out.insert(n, d);
            }
        }
    }
    out.retain(|k, _| order.is_empty() || order.contains(k));
    Some(out)
}

/// `input|output|inout [wire|reg|signed] [[msb:lsb]] a, b` → direction and width, and the names; a bare
/// list of names has no declaration.
fn declaration(s: &str) -> (Option<(Dir, u32)>, Vec<String>) {
    let s = s.trim();
    let (dir, rest) = if let Some(r) = s.strip_prefix("input") {
        (Some(Dir::Input), r)
    } else if let Some(r) = s.strip_prefix("output") {
        (Some(Dir::Output), r)
    } else if let Some(r) = s.strip_prefix("inout") {
        (Some(Dir::Inout), r)
    } else {
        (None, s)
    };
    if dir.is_some() && !rest.starts_with(|c: char| c.is_whitespace() || c == '[') {
        return (None, Vec::new());
    }
    let mut rest = rest.trim();
    for kw in ["wire", "reg", "logic", "signed"] {
        if let Some(r) = rest.strip_prefix(kw)
            && r.starts_with(|c: char| c.is_whitespace() || c == '[')
        {
            rest = r.trim();
        }
    }
    let mut width = 1u32;
    if let Some(r) = rest.strip_prefix('[')
        && let Some(close) = r.find(']')
    {
        let range = &r[..close];
        if let Some((h, l)) = range.split_once(':')
            && let (Ok(h), Ok(l)) = (h.trim().parse::<i64>(), l.trim().parse::<i64>())
        {
            width = (h - l).unsigned_abs() as u32 + 1;
        }
        rest = r[close + 1..].trim();
    }
    let names: Vec<String> = rest
        .split(',')
        .map(|n| n.trim().trim_start_matches('\\').to_string())
        .filter(|n| {
            !n.is_empty()
                && n.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
        })
        .collect();
    let decl = if dir.is_none() && names.len() == 1 && s.contains(' ') {
        None
    } else {
        dir.map(|d| (d, width))
    };
    (decl, names)
}

const KEYWORDS: &[&str] = &[
    "input",
    "output",
    "inout",
    "wire",
    "reg",
    "assign",
    "supply0",
    "supply1",
    "tri",
    "parameter",
    "localparam",
    "genvar",
    "integer",
];

pub fn parse(text: &str) -> Netlist {
    let text = strip(text);
    let mut n = Netlist::default();
    let mut in_module = false;
    for stmt in text.split(';') {
        let mut s = stmt.trim();
        loop {
            if let Some(rest) = s.strip_prefix("endmodule") {
                in_module = false;
                s = rest.trim();
                continue;
            }
            break;
        }
        if s.is_empty() {
            continue;
        }
        if let Some(rest) = s.strip_prefix("module") {
            if let Some(name) = rest
                .trim()
                .split(|c: char| c.is_whitespace() || c == '(' || c == '#')
                .next()
                .filter(|x| !x.is_empty())
            {
                n.modules.insert(name.trim_start_matches('\\').to_string());
            }
            in_module = true;
            continue;
        }
        if !in_module {
            continue;
        }
        let first = s.split_whitespace().next().unwrap_or("");
        if KEYWORDS.contains(&first) {
            continue;
        }
        // `TYPE [#(...)] NAME ( ports )`
        if s.contains('(') && !first.is_empty() {
            *n.instances
                .entry(first.trim_start_matches('\\').to_string())
                .or_default() += 1;
        }
    }
    n
}

/// The output bits of `module` that nothing in its body connects: a bit is driven when the output (or that
/// bit, or a range holding it) appears in an `assign` or an instance's port connection. Empty when the
/// module is not in the text (the port check reports that).
pub fn undriven_outputs(text: &str, module: &str) -> Vec<String> {
    let Some(ports) = ports(text, module) else {
        return Vec::new();
    };
    let text = strip(text);
    let mut stmts = text.split(';').map(str::trim);
    // Skip to the module's header.
    let found = stmts.any(|st| {
        let st = st.trim_start_matches("endmodule").trim();
        st.strip_prefix("module").is_some_and(|r| {
            let name: String = r
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '\\')
                .collect();
            name.trim_start_matches('\\') == module
        })
    });
    if !found {
        return Vec::new();
    }
    let mut whole: BTreeSet<String> = BTreeSet::new();
    let mut bits: BTreeMap<String, BTreeSet<i64>> = BTreeMap::new();
    for st in stmts {
        let (body, last) = match st.find("endmodule") {
            Some(k) => (&st[..k], true),
            None => (st, false),
        };
        let first = body.split_whitespace().next().unwrap_or("");
        if !KEYWORDS.contains(&first) || first == "assign" {
            references(body, &mut whole, &mut bits);
        }
        if last {
            break;
        }
    }
    let mut out = Vec::new();
    for (name, (dir, width)) in &ports {
        if *dir != Dir::Output || whole.contains(name) {
            continue;
        }
        let have = bits.get(name).map_or(0, BTreeSet::len) as u32;
        if have < *width {
            out.push(if *width == 1 {
                name.clone()
            } else {
                format!("{name} ({have} of {width} bits)")
            });
        }
    }
    out
}

/// The identifiers a statement references: whole (`y`) or by bit (`y[3]`, `y[3:1]`).
fn references(s: &str, whole: &mut BTreeSet<String>, bits: &mut BTreeMap<String, BTreeSet<i64>>) {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        let start = i;
        let name = if c == '\\' {
            while i < b.len() && !b[i].is_ascii_whitespace() {
                i += 1;
            }
            s[start + 1..i].to_string()
        } else if c.is_ascii_alphabetic() || c == '_' {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'$') {
                i += 1;
            }
            s[start..i].to_string()
        } else {
            // Skip numbers with their base and digits (`8'b0`, `3'h7`) whole.
            if c.is_ascii_digit() || c == '\'' {
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'\'' || b[i] == b'_') {
                    i += 1;
                }
            } else {
                i += 1;
            }
            continue;
        };
        let mut j = i;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if j < b.len()
            && b[j] == b'['
            && let Some(close) = s[j..].find(']')
        {
            let range = &s[j + 1..j + close];
            let set = bits.entry(name).or_default();
            match range.split_once(':') {
                Some((h, l)) => {
                    if let (Ok(h), Ok(l)) = (h.trim().parse::<i64>(), l.trim().parse::<i64>()) {
                        set.extend(h.min(l)..=h.max(l));
                    }
                }
                None => {
                    if let Ok(k) = range.trim().parse::<i64>() {
                        set.insert(k);
                    }
                }
            }
            i = j + close + 1;
        } else {
            whole.insert(name);
        }
    }
}

/// The text without comments. An escaped identifier (`\\u_div/*Logic0*`, from `\\` to the next
/// whitespace) is kept whole: comment markers inside it are part of the name.
fn strip(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut from = 0;
    while i < b.len() {
        match b[i] {
            b'\\' => {
                while i < b.len() && !b[i].is_ascii_whitespace() {
                    i += 1;
                }
                continue;
            }
            b'/' if b.get(i + 1) == Some(&b'/') => {
                out.push_str(&text[from..i]);
                i = text[i..].find('\n').map_or(b.len(), |e| i + e);
                from = i;
                continue;
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                out.push_str(&text[from..i]);
                i = text[i + 2..].find("*/").map_or(b.len(), |e| i + 2 + e + 2);
                from = i;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    out.push_str(&text[from..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const NET: &str = "// generated\nmodule piece ( a, p );\n  input [1:0] a;\n  output [3:0] p;\n  wire n1;\n  NAND2_X1 U1 ( .A1(a[0]), .A2(a[1]), .ZN(n1) );\n  INV_X1 U2 ( .A(n1), .ZN(p[0]) );\n  assign p[3:1] = 3'b0;\nendmodule\n\nmodule top ( a, p );\n  input [1:0] a;\n  output [3:0] p;\n  piece u0 ( .a(a), .p(p) );\n  DHLx1_ASAP7_75t_R l0 ( .D(a[0]), .CLK(a[1]), .Q() );\n  GTECH_AND2 g0 ( .A(a[0]), .B(a[1]), .Z() );\nendmodule\n";

    #[test]
    fn escaped_names_with_comment_markers_are_names() {
        let net = "// header\nmodule p ( a, y );\n  input a;\n  output y;\n  wire \\u_div/*Logic0* ;\n  TIELOx1 U1 ( .L(\\u_div/*Logic0* ) );\n  INVx1 U2 ( .A(a), .Y(y) ); /* a comment */\nendmodule\nmodule top ( a, y );\n  input a;\n  output y;\n  p u0 ( .a(a), .y(y) );\nendmodule\n";
        let n = parse(net);
        assert_eq!(n.modules.iter().collect::<Vec<_>>(), ["p", "top"]);
        assert_eq!(n.instances.get("INVx1"), Some(&1));
        assert_eq!(n.instances.get("TIELOx1"), Some(&1));
        assert!(ports(net, "top").is_some());
    }

    #[test]
    fn ports_in_both_styles() {
        let ansi = "module add_i8(\n  input wire [7:0] a,\n  input wire [7:0] b,\n  output wire [7:0] out\n);\n  assign out = a + b;\nendmodule\n";
        let body = "module add_i8 ( a, b, out );\n  input [7:0] a;\n  input [7:0] b;\n  output [7:0] out;\n  wire n1;\nendmodule\n";
        let pa = ports(ansi, "add_i8").unwrap();
        assert_eq!(pa, ports(body, "add_i8").unwrap());
        assert_eq!(pa["a"], (Dir::Input, 8));
        assert_eq!(pa["out"], (Dir::Output, 8));
        let reg = "module register_w8(input wire clk, input wire [7:0] d, output reg [7:0] q);\n  always @(posedge clk) q <= d;\nendmodule\n";
        let pr = ports(reg, "register_w8").unwrap();
        assert_eq!(pr.len(), 3);
        assert_eq!(pr["clk"], (Dir::Input, 1));
        assert_eq!(pr["q"], (Dir::Output, 8));
        assert!(ports(body, "other").is_none());
        let wrong = body.replace("output [7:0] out", "output [6:0] out");
        assert_ne!(ports(&wrong, "add_i8").unwrap(), pa);
    }

    #[test]
    fn outputs_driven_whole_by_bit_or_by_range() {
        // `p` of `piece`: bit 0 by a cell, bits 3..1 by an assign; `p` of `top` whole by a submodule.
        assert!(undriven_outputs(NET, "piece").is_empty());
        assert!(undriven_outputs(NET, "top").is_empty());
        let open = NET.replace("  assign p[3:1] = 3'b0;\n", "  assign p[2:1] = 2'b0;\n");
        assert_eq!(undriven_outputs(&open, "piece"), ["p (3 of 4 bits)"]);
        let one = "module m ( a, y, z );\n  input a;\n  output y;\n  output z;\n  INV_X1 U1 ( .A(a), .ZN(y) );\nendmodule\n";
        assert_eq!(undriven_outputs(one, "m"), ["z"]);
        let esc =
            "module m ( a, y );\n  input a;\n  output y;\n  INV_X1 \\U1/x ( .A(a), .ZN(y) );\nendmodule\n";
        assert!(undriven_outputs(esc, "m").is_empty());
        assert!(undriven_outputs(one, "other").is_empty());
    }

    #[test]
    fn modules_instances_and_foreign_types() {
        let n = parse(NET);
        assert_eq!(n.modules.iter().collect::<Vec<_>>(), ["piece", "top"]);
        assert_eq!(n.instances.get("NAND2_X1"), Some(&1));
        assert_eq!(n.instances.get("piece"), Some(&1));
        let cells: BTreeSet<String> = ["NAND2_X1", "INV_X1", "DHLx1_ASAP7_75t_R"]
            .map(String::from)
            .into();
        assert_eq!(n.foreign(&cells).collect::<Vec<_>>(), ["GTECH_AND2"]);
        let latches: BTreeSet<String> = ["DHLx1_ASAP7_75t_R".to_string()].into();
        assert_eq!(n.count_of(&latches), 1);
    }
}
