//! GraMi's text formats: the `.lg` input graph and the `Output.txt` pattern list (docs/tools/grami.md
//! §2, §5). Both parsers are strict: anything GraMi's own loader would misread is an error.

/// A labelled directed graph as integer lists: node labels, then `(src, dst, label)` edges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LgGraph {
    pub nodes: Vec<u32>,
    pub edges: Vec<(u32, u32, u32)>,
}

impl LgGraph {
    /// The `.lg` text: a header, the vertices in id order, then the edges sorted.
    pub fn to_text(&self) -> String {
        let mut s = String::from("# t 1\n");
        for (i, l) in self.nodes.iter().enumerate() {
            s.push_str(&format!("v {i} {l}\n"));
        }
        let mut e = self.edges.clone();
        e.sort_unstable();
        for (a, b, l) in e {
            s.push_str(&format!("e {a} {b} {l}\n"));
        }
        s
    }
}

fn num(tok: Option<&str>, line: usize, what: &str) -> Result<u32, String> {
    tok.ok_or_else(|| format!("line {line}: missing {what}"))?
        .parse()
        .map_err(|_| format!("line {line}: {what} is not a non-negative integer"))
}

/// Parse vertex and edge lines (`v`/`e`) into `g`; `line0` is the number of the first line.
fn parse_body<'a>(
    lines: impl Iterator<Item = &'a str>,
    line0: usize,
    g: &mut LgGraph,
) -> Result<(), String> {
    let mut edges_started = false;
    for (k, l) in lines.enumerate() {
        let line = line0 + k;
        let mut t = l.split(' ');
        match t.next() {
            Some("v") if !edges_started => {
                let id = num(t.next(), line, "vertex id")?;
                if id as usize != g.nodes.len() {
                    return Err(format!("line {line}: vertex {id} out of order"));
                }
                g.nodes.push(num(t.next(), line, "vertex label")?);
            }
            Some("v") => return Err(format!("line {line}: vertex after an edge")),
            Some("e") => {
                edges_started = true;
                let a = num(t.next(), line, "edge source")?;
                let b = num(t.next(), line, "edge target")?;
                let lab = num(t.next(), line, "edge label")?;
                if a as usize >= g.nodes.len() || b as usize >= g.nodes.len() {
                    return Err(format!("line {line}: edge {a} -> {b} names an unknown vertex"));
                }
                g.edges.push((a, b, lab));
            }
            _ => return Err(format!("line {line}: expected a `v` or `e` line, got {l:?}")),
        }
        if t.next().is_some() {
            return Err(format!("line {line}: trailing fields"));
        }
    }
    Ok(())
}

/// Parse a `.lg` file: one header line, then vertices and edges; no blank line.
pub fn parse_lg(text: &str) -> Result<LgGraph, String> {
    let mut lines = text.lines();
    lines.next().ok_or("empty file")?;
    let mut g = LgGraph::default();
    parse_body(lines, 2, &mut g)?;
    Ok(g)
}

/// GraMi's `Output.txt`: elapsed seconds, the pattern count, then per pattern `<i>:` and its `v`/`e`
/// lines (pattern-local ids). Returns the patterns, checked against the count.
pub fn parse_output(text: &str) -> Result<Vec<LgGraph>, String> {
    let mut lines = text.lines();
    let secs = lines.next().ok_or("empty output")?;
    secs.trim()
        .parse::<f64>()
        .map_err(|_| format!("line 1: elapsed time {secs:?} is not a number"))?;
    let count: usize = lines
        .next()
        .ok_or("no pattern count")?
        .trim()
        .parse()
        .map_err(|_| "line 2: the pattern count is not an integer".to_string())?;
    let mut out: Vec<LgGraph> = Vec::new();
    let mut block: Vec<&str> = Vec::new();
    let mut start = 3;
    let flush = |block: &mut Vec<&str>, start: usize, out: &mut Vec<LgGraph>| -> Result<(), String> {
        if let Some(h) = block.first() {
            let i: usize = h
                .strip_suffix(':')
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| format!("line {start}: expected `<i>:`, got {h:?}"))?;
            if i != out.len() {
                return Err(format!("line {start}: pattern {i} out of order"));
            }
            let mut g = LgGraph::default();
            parse_body(block[1..].iter().copied(), start + 1, &mut g)?;
            if g.nodes.is_empty() {
                return Err(format!("line {start}: pattern {i} has no vertex"));
            }
            out.push(g);
        }
        block.clear();
        Ok(())
    };
    for (k, l) in lines.enumerate() {
        if l.ends_with(':') {
            flush(&mut block, start, &mut out)?;
            start = k + 3;
        }
        if block.is_empty() && !l.ends_with(':') {
            return Err(format!("line {}: pattern body without a header", k + 3));
        }
        block.push(l);
    }
    flush(&mut block, start, &mut out)?;
    if out.len() != count {
        return Err(format!("{} patterns listed, {count} announced", out.len()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lg_round_trips_and_rejects_what_grami_misreads() {
        let g = LgGraph {
            nodes: vec![3, 7, 3],
            edges: vec![(1, 2, 0), (0, 1, 2)],
        };
        let t = g.to_text();
        assert_eq!(t, "# t 1\nv 0 3\nv 1 7\nv 2 3\ne 0 1 2\ne 1 2 0\n");
        let mut back = parse_lg(&t).unwrap();
        back.edges.sort_unstable();
        let mut want = g.clone();
        want.edges.sort_unstable();
        assert_eq!(back, want);
        for bad in [
            "# t\nv 0 1\n\ne 0 0 1\n",
            "# t\nv 1 1\n",
            "# t\nv 0 1\ne 0 0 1\nv 1 2\n",
            "# t\nv 0 x\n",
            "# t\nv 0 1.5\n",
            "# t\nv 0 1\ne 0 3 1\n",
            "# t\nv 0 1 2\n",
        ] {
            assert!(parse_lg(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn output_parses_and_checks_the_count() {
        let t = "1.388\n2\n0:\nv 0 2\nv 1 2\ne 0 1 1\n1:\nv 0 1\nv 1 2\nv 2 2\ne 0 1 0\ne 1 2 1\n";
        let p = parse_output(t).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[1].nodes, vec![1, 2, 2]);
        assert_eq!(p[1].edges, vec![(0, 1, 0), (1, 2, 1)]);
        assert!(parse_output("0.1\n0\n").unwrap().is_empty());
        assert!(parse_output("0.1\n3\n0:\nv 0 1\n").is_err());
        assert!(parse_output("0.1\n1\n1:\nv 0 1\n").is_err());
        assert!(parse_output("0.1\n1\nv 0 1\n").is_err());
    }
}
