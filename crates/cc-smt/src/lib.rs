//! SMT-LIB 2 scripts and an out-of-process solver runner (docs/framework/smt.md).
//!
//! A query is a plain SMT-LIB 2 file (the certificate: any solver can re-check it). [`Script`] builds one;
//! [`run_batch`] runs a solver on a set of files through a capped [`BoundRunner`], either inside the
//! toolchain image or as a host binary, and parses each answer into a [`Verdict`]. `unknown`, a timeout
//! or an error is never a proof.

pub mod float;
pub mod ops;
pub mod run;
pub mod term;

pub use run::{Cache, query_hash, run_sharded, solver_version};

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use cc_core::CoreError;
use cc_core::runner::{BoundRunner, Invocation, Mount};
use serde::{Deserialize, Serialize};

/// A solver binary.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Solver {
    Bitwuzla,
    Z3,
}

impl Solver {
    pub const ALL: [Solver; 2] = [Solver::Bitwuzla, Solver::Z3];

    pub fn parse(s: &str) -> Option<Solver> {
        match s {
            "bitwuzla" => Some(Solver::Bitwuzla),
            "z3" => Some(Solver::Z3),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Solver::Bitwuzla => "bitwuzla",
            Solver::Z3 => "z3",
        }
    }

    /// The other solver (for re-checking a proof).
    pub fn other(self) -> Solver {
        match self {
            Solver::Bitwuzla => Solver::Z3,
            Solver::Z3 => Solver::Bitwuzla,
        }
    }
}

/// A solver's answer to one `(check-sat)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Unsat,
    /// With the model the solver printed (a counterexample of the negated equivalence).
    Sat {
        model: String,
    },
    Unknown,
    Timeout,
    Error {
        message: String,
    },
}

impl Verdict {
    /// Parse a solver's output for a script with one `(check-sat)` followed by `(get-model)`.
    pub fn parse(output: &str, exit: Option<i32>) -> Verdict {
        if exit == Some(124) {
            return Verdict::Timeout;
        }
        let mut lines = output.lines().map(str::trim).filter(|l| !l.is_empty());
        match lines.next() {
            Some("unsat") => Verdict::Unsat,
            Some("sat") => Verdict::Sat {
                model: lines.collect::<Vec<_>>().join("\n"),
            },
            Some("unknown") => Verdict::Unknown,
            other => Verdict::Error {
                message: other.unwrap_or("no output").chars().take(400).collect(),
            },
        }
    }

    pub fn is_unsat(&self) -> bool {
        *self == Verdict::Unsat
    }
}

/// Where the solver binaries are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    /// In the toolchain image (on its `PATH`), through a Docker runner.
    Image,
    /// Host binaries, per solver.
    Host(BTreeMap<Solver, String>),
}

/// The answers of one solver to a batch of files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Batch {
    pub solver: Solver,
    /// The first line of `<solver> --version`.
    pub version: String,
    /// Per file name, the verdict.
    pub verdicts: BTreeMap<String, Verdict>,
}

const MARK: &str = "@@cc-smt";
const DIR_IN_CONTAINER: &str = "/work/smt";

/// Run `solver` on every `files` entry (names inside `dir`), each with `timeout_s`, in one process. The
/// runner must be a Docker runner for [`Location::Image`] and a host runner otherwise.
pub fn run_batch(
    runner: &BoundRunner,
    location: &Location,
    solver: Solver,
    dir: &Path,
    files: &[String],
    timeout_s: u64,
) -> Result<Batch, CoreError> {
    let (program, base) = match location {
        Location::Image => (solver.name().to_string(), DIR_IN_CONTAINER.to_string()),
        Location::Host(p) => (
            p.get(&solver)
                .cloned()
                .unwrap_or_else(|| solver.name().to_string()),
            dir.display().to_string(),
        ),
    };
    // One shell per batch: a container start per query would dominate the run time.
    let script = format!(
        "p=\"$1\"; t=\"$2\"; shift 2; for f in \"$@\"; do echo '{MARK} file' \"$f\"; timeout \"$t\" \"$p\" \"$f\" 2>&1; \
         echo '{MARK} exit' $?; done; echo '{MARK} version'; \"$p\" --version 2>&1 | head -n 1"
    );
    let mut inv = Invocation::new(format!("smt-{}", solver.name()), "sh")
        .arg("-c")
        .arg(script)
        .arg("sh")
        .arg(program)
        .arg(timeout_s.to_string())
        .args(files.iter().map(|f| format!("{base}/{f}")));
    if *location == Location::Image {
        inv = inv.mount(Mount {
            src: dir.to_path_buf(),
            dst: DIR_IN_CONTAINER.into(),
            readonly: true,
        });
    }
    let out = runner.run(&inv)?;
    let text = out.stdout_string()?;
    let out = out.check()?;
    drop(out);
    parse_batch(solver, &text, &base)
}

fn parse_batch(solver: Solver, text: &str, base: &str) -> Result<Batch, CoreError> {
    let mut verdicts = BTreeMap::new();
    let mut version = String::new();
    let mut cur: Option<String> = None;
    let mut buf = String::new();
    let mut in_version = false;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(MARK) {
            let rest = rest.trim();
            if let Some(f) = rest.strip_prefix("file ") {
                let name = f.strip_prefix(base).unwrap_or(f).trim_start_matches('/');
                cur = Some(name.to_string());
                buf.clear();
            } else if let Some(code) = rest.strip_prefix("exit ") {
                if let Some(f) = cur.take() {
                    verdicts.insert(f, Verdict::parse(&buf, code.trim().parse().ok()));
                }
            } else if rest == "version" {
                in_version = true;
            }
            continue;
        }
        if in_version {
            if version.is_empty() {
                version = line.trim().to_string();
            }
        } else if cur.is_some() {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    Ok(Batch {
        solver,
        version,
        verdicts,
    })
}

/// An SMT-LIB 2 script with one `(check-sat)`: declarations, then assertions.
#[derive(Debug, Clone, Default)]
pub struct Script {
    comments: Vec<String>,
    decls: Vec<String>,
    asserts: Vec<String>,
}

impl Script {
    pub fn new() -> Self {
        Self::default()
    }

    /// A `;` comment line in the header.
    pub fn comment(&mut self, c: impl Into<String>) -> &mut Self {
        self.comments.push(c.into());
        self
    }

    /// `(declare-const name sort)`.
    pub fn constant(&mut self, name: &str, sort: &str) -> &mut Self {
        self.decls.push(format!("(declare-const {name} {sort})"));
        self
    }

    /// `(declare-fun name (args) sort)`.
    pub fn fun(&mut self, name: &str, args: &[String], sort: &str) -> &mut Self {
        self.decls
            .push(format!("(declare-fun {name} ({}) {sort})", args.join(" ")));
        self
    }

    /// `(define-fun name () sort body)`.
    pub fn define(&mut self, name: &str, sort: &str, body: &str) -> &mut Self {
        self.decls.push(format!("(define-fun {name} () {sort} {body})"));
        self
    }

    pub fn assert(&mut self, t: impl Into<String>) -> &mut Self {
        self.asserts.push(format!("(assert {})", t.into()));
        self
    }

    /// The text: logic `ALL` (both solvers accept it for any mix of UF, BV, FP and arrays), models on.
    pub fn render(&self) -> String {
        let mut s = String::new();
        for c in &self.comments {
            let _ = writeln!(s, "; {c}");
        }
        s.push_str("(set-logic ALL)\n(set-option :produce-models true)\n");
        for d in &self.decls {
            let _ = writeln!(s, "{d}");
        }
        for a in &self.asserts {
            let _ = writeln!(s, "{a}");
        }
        s.push_str("(check-sat)\n(get-model)\n");
        s
    }
}

/// A bit-vector sort.
pub fn bv(w: u32) -> String {
    format!("(_ BitVec {w})")
}

/// A bit-vector literal of width `w`.
pub fn bv_lit(v: u64, w: u32) -> String {
    format!("(_ bv{v} {w})")
}

/// `(and …)`, `true` when empty.
pub fn and(ts: &[String]) -> String {
    match ts.len() {
        0 => "true".into(),
        1 => ts[0].clone(),
        _ => format!("(and {})", ts.join(" ")),
    }
}

/// `(or …)`, `false` when empty.
pub fn or(ts: &[String]) -> String {
    match ts.len() {
        0 => "false".into(),
        1 => ts[0].clone(),
        _ => format!("(or {})", ts.join(" ")),
    }
}

/// `(distinct …)`, `true` for fewer than two terms.
pub fn distinct(ts: &[String]) -> String {
    if ts.len() < 2 {
        "true".into()
    } else {
        format!("(distinct {})", ts.join(" "))
    }
}

/// An application `(f a b …)`, or `f` without arguments.
pub fn app(f: &str, args: &[String]) -> String {
    if args.is_empty() {
        f.to_string()
    } else {
        format!("({f} {})", args.join(" "))
    }
}

/// A file path relative to a directory, for [`run_batch`]'s `files`.
pub fn file_name(p: &Path) -> Option<String> {
    p.file_name().and_then(|n| n.to_str()).map(str::to_string)
}

/// Default host paths: the solver names (resolved on `PATH`).
pub fn host_paths(z3: &str, bitwuzla: &str) -> Location {
    Location::Host(BTreeMap::from([
        (Solver::Z3, z3.to_string()),
        (Solver::Bitwuzla, bitwuzla.to_string()),
    ]))
}

/// The directory containing a file (for callers that hold full paths).
pub fn parent(p: &Path) -> PathBuf {
    p.parent().map(Path::to_path_buf).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_batches() {
        let t = "@@cc-smt file /d/a.smt2\nunsat\n(error \"model is not available\")\n@@cc-smt exit 1\n\
                 @@cc-smt file /d/b.smt2\nsat\n(\n (define-fun x () Bool true)\n)\n@@cc-smt exit 0\n\
                 @@cc-smt file /d/c.smt2\n@@cc-smt exit 124\n@@cc-smt version\nZ3 version 5.3.3 - 64 bit\n";
        let b = parse_batch(Solver::Z3, t, "/d").unwrap();
        assert_eq!(b.version, "Z3 version 5.3.3 - 64 bit");
        assert_eq!(b.verdicts["a.smt2"], Verdict::Unsat);
        assert!(matches!(&b.verdicts["b.smt2"], Verdict::Sat { model } if model.contains("define-fun x")));
        assert_eq!(b.verdicts["c.smt2"], Verdict::Timeout);
    }

    #[test]
    fn errors_are_not_proofs() {
        assert!(!Verdict::parse("(error \"bad\")", Some(1)).is_unsat());
        assert!(!Verdict::parse("", Some(0)).is_unsat());
        assert!(!Verdict::parse("unknown", Some(0)).is_unsat());
    }

    #[test]
    fn renders_a_script() {
        let mut s = Script::new();
        s.comment("x").constant("a", &bv(8)).assert("(not (= a a))");
        let t = s.render();
        assert!(t.starts_with("; x\n(set-logic ALL)"));
        assert!(t.ends_with("(check-sat)\n(get-model)\n"));
    }
}
