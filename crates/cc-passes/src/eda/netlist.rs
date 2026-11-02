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
