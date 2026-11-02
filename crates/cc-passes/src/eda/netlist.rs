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

fn strip(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let line = rest.find("//");
        let block = rest.find("/*");
        match (line, block) {
            (None, None) => {
                out.push_str(rest);
                break;
            }
            (Some(l), b) if b.is_none_or(|b| l < b) => {
                out.push_str(&rest[..l]);
                rest = rest[l..].find('\n').map_or("", |e| &rest[l + e..]);
            }
            (_, Some(b)) => {
                out.push_str(&rest[..b]);
                rest = rest[b + 2..].find("*/").map_or("", |e| &rest[b + 2 + e + 2..]);
            }
            _ => unreachable!(),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const NET: &str = "// generated\nmodule piece ( a, p );\n  input [1:0] a;\n  output [3:0] p;\n  wire n1;\n  NAND2_X1 U1 ( .A1(a[0]), .A2(a[1]), .ZN(n1) );\n  INV_X1 U2 ( .A(n1), .ZN(p[0]) );\n  assign p[3:1] = 3'b0;\nendmodule\n\nmodule top ( a, p );\n  input [1:0] a;\n  output [3:0] p;\n  piece u0 ( .a(a), .p(p) );\n  DHLx1_ASAP7_75t_R l0 ( .D(a[0]), .CLK(a[1]), .Q() );\n  GTECH_AND2 g0 ( .A(a[0]), .B(a[1]), .Z() );\nendmodule\n";

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
