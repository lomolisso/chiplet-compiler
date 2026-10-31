# GraMi (frequent subgraph mining in a single graph)

GraMi mines frequent connected subgraphs of **one large labelled graph** using the **MNI** (minimum image based)
support, formulated as a constraint-satisfaction problem (Elseidy, Abdelhamid, Skiadopoulos, Kalnis, "GRAMI:
Frequent Subgraph and Pattern Mining in a Single Large Graph", PVLDB 7(7):517–528, 2014). In chiplet-compiler it
mines recurring datapath fragments across the kernel dataflow graphs (all kernels merged into one disconnected
graph) as candidates for shared functional units. Repo: https://github.com/ehab-abdelhamid/GraMi (last commit
`d2875ae`, 2018-06-03; README "Last updated … 11 March, 2014"). License: LGPL (`LICENCE.txt`; source headers say "version 2 … or (at your option) any later version").

## 1. What it computes

- Input: a single graph, integer node labels, integer edge labels.
- Output: all connected subgraphs (patterns) whose MNI support ≥ `freq`. MNI support of a pattern = min over
  pattern nodes `u` of the number of distinct graph nodes `u` can map to in some embedding; in the source,
  `DFSSearch.getFrequencyOfPattern()` returns the minimum domain size over pattern variables.
- Variants (separate source trees, separate `bin/` dirs):

| Directory | Mines |
|---|---|
| `GRAMI_UNDIRECTED_SUBGRAPHS` | subgraphs, undirected |
| `GRAMI_DIRECTED_SUBGRAPHS` | subgraphs, directed (**the one we use**: dataflow is directed) |
| `GRAMI_UNDIRECTED_PATTERNS`, `GRAMI_DIRECTED_PATTERNS` | "patterns" (edges may map to paths ≤ distance); not used |

Directedness is decided **only by which tree you run**: the `grami` script's `-t` picks the directory, and the Java
`CommandLineParser` has no `type` key (it parses `freq`, `filename`, `datasetFolder`, `maxLabelAppearance`,
`approximate`, `approxConst`, `automorphism`, `distinct`, `caching`, `partial`, `limitedtime`; unknown keys such as
`type=`, `distance=`, `mlabels=` are silently ignored).

## 2. Input format (`.lg`, parsed by `Graph.loadFromFile_Ehab`)

```
# t 1          <- first line is skipped (any header)
v 0 3          <- v <id> <label>; ids must be 0,1,2,… in order ("The node list is not sorted" otherwise)
v 1 7
e 0 1 2        <- e <src> <dst> <label>; all e lines after all v lines
```
- Node labels: `Integer.parseInt` → must be integers. Edge labels: parsed as `double`, then cast `(int)` when
  building initial edges (`Searcher.java`) and parsed back with `Integer.parseInt` on output → **use integers**.
- No blank lines and no comments after the header: the loader does `line.charAt(0)` on every line (empty line →
  exception) and stops reading edges at the first non-`e` line.
- Our encoding: node label = interned op class (e.g. `mul.i8xi8->i32`, with operand-position info folded into the
  edge label); edge label = operand index. Keep the label dictionary in our JSON next to the `.lg`.
- Infrequent edge labels (count < `freq`) are pruned at load time.

## 3. Build

`build` script: `javac -sourcepath ./<TREE>/src -d ./<TREE>/bin/ ./<TREE>/src/**/*.java` per tree (works without
`globstar` because all sources sit one level below `src/`). Main class: `Dijkstra.main`.

Java version: README says "Java JRE v1.6.0 or later", but `search/RightMostExtension.java` (both SUBGRAPHS trees)
has `import javax.jws.Oneway;` and `import javax.rmi.CORBA.Util;`, both unused (the code calls
`utilities.Util.clone` fully qualified). The `java.xml.ws` and `java.corba` modules were removed in Java 11
(JEP 320), so **`javac` ≥ 11 fails** on these imports. Options:
1. Patch: delete those two import lines (our Docker build applies a checked-in patch; Java 17 then works).
2. Build/run with Java 8 (host has `openjdk version "1.8.0_504"`; would also work outside Docker, untested).
`org.w3c.dom.ls.LSInput` (also unused, in `myNode.java`) still exists in `java.xml` (harmless).

## 4. Invocation (call Java directly, not the `grami` script)

```bash
cd <job_dir>                                   # Output.txt is written to the CWD
java -Xmx8g -XX:ActiveProcessorCount=2 -XX:+UseSerialGC \
     -cp /opt/grami/GRAMI_DIRECTED_SUBGRAPHS/bin Dijkstra.main \
     filename=kernels.lg datasetFolder=/work/out/ freq=3 \
     limitedtime=false maxLabelAppearance=-1 approximate=1
```
- The graph path is `datasetFolder + filename` (plain string concatenation; default `datasetFolder=../Datasets/`)
  → `datasetFolder` must end with `/`.
- `freq` = minimum MNI support (default 1000 if omitted).
- `maxLabelAppearance=k` limits repeats of a label in results (-1 = no limit).
- `approximate=<α<1>` enables approximate mining (`approxConst` = β); `approximate=1` = exact.
- `automorphism`, `distinct`, `caching`, `partial` default to `true` (`Settings.java`); leave as is.
- **`limitedtime` defaults to `true`**: each per-candidate existence check runs under a 5 s timer
  (`DFSSearch.java`: `timer.schedule(new StopTask(), 5*1000)`, 7 s in `searchParticularExistance`); when it fires
  the search returns `-3` and the candidate node is **added to the domain as if valid** (`tmp.add(firstNode)`). So
  with the default, supports can be over-estimated and infrequent patterns reported. Always pass
  `limitedtime=false` for exact results, and bound total runtime externally.
- The JVM is the only resource control: `-Xmx` (heap cap, enforced by the JVM even on cgroups v1),
  `-XX:ActiveProcessorCount`, serial GC; plus Docker `--memory`/`--cpus` and a wall-clock timeout in `ToolRunner`.

## 5. Output (`Output.txt`)

```
1.388          <- elapsed seconds
21             <- number of frequent subgraphs
0:             <- pattern index
v 0 2          <- pattern-local node ids, labels
v 1 2
e 0 1 1        <- direction preserved for directed tree
1:
…
```
(sample `Output.txt` in the repo; writer in `Dijkstra/main.java`, `DFScodeSerializer.serialize`).

Limits: **no per-pattern support value and no embeddings** are written; the `sr.result` list carries them only in
memory. Also no maximality/closedness filter: all frequent patterns (exponentially many for low `freq`).
Our pass therefore: (1) parses `Output.txt`, (2) recomputes embeddings and MNI support in Rust with `vf2`
(non-induced subgraph isomorphism; `rust_stack.md`) and checks support ≥ `freq` (this is the certificate),
(3) optionally keeps maximal/closed patterns only.

## 6. Pitfalls

| Pitfall | Mitigation |
|---|---|
| `limitedtime=true` default over-counts | always `limitedtime=false`; verify supports in Rust |
| Java ≥ 11 compile error | patch the two `javax` imports or use Java 8 |
| `type=` ignored | choose the DIRECTED tree via classpath |
| Output in CWD, fixed name | one job dir per run |
| Blank line / unsorted ids in `.lg` | generate with a strict writer and a round-trip test |
| Exponential blow-up at low `freq` | start high, lower stepwise; cap with `maxLabelAppearance`; wall-clock timeout |
| Edge labels truncated to int | integer labels only |

## Sources

- GraMi source @ `d2875ae`: `README.txt`, `grami`, `build`, `Output.txt`, `Datasets/test1.lg`,
  `GRAMI_DIRECTED_SUBGRAPHS/src/{Dijkstra/main.java, utilities/CommandLineParser.java, utilities/Settings.java,
  dataStructures/Graph.java, dataStructures/DFScodeSerializer.java, CSP/DFSSearch.java, search/Searcher.java,
  search/RightMostExtension.java, dataStructures/myNode.java}`.
- M. Elseidy, E. Abdelhamid, S. Skiadopoulos, P. Kalnis, PVLDB 7(7):517–528, 2014, https://www.vldb.org/pvldb/vol7/p517-elseidy.pdf
- JEP 320 (Remove the Java EE and CORBA Modules, Java 11): https://openjdk.org/jeps/320
