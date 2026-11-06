# Third-party submodules (`third_party/`)

Git submodules (shallow). Each pin is recorded in `config.toml` `[tools.*]`. `chipc doctor` checks that the checkout, the commit the superproject records and the Docker image all agree. Tools are built only inside the image (`docker/`), never on the host.

| Submodule | Upstream | Pin (commit) | `config.toml` key | Used by |
|---|---|---|---|---|
| `grami/` | https://github.com/ehab-abdelhamid/GraMi | `d2875ae80afa2f141323a3f6b89a387787fcda38` (2018-06-03, last upstream commit) | `tools.grami.commit` | `mine` (directed build) |
| `allo/` | https://github.com/cornell-zhang/allo | `ac21bae49ca5f912ecebea00908b56976d2361a2` (`main`, 2026-10-12) | `tools.allo.commit` | `allo_emit`, `xls` |
| `xls/` | https://github.com/google/xls | `8ddae8bf466742285aa2d405470512cc73a9aeec` = release tag `v0.0.0-10898-g8ddae8bf4` (2026-10-24) | `tools.xls.commit`, `tools.xls.tag` | `xls`, `characterize` |
| `hardfloat/` | https://github.com/ucb-bar/berkeley-hardfloat | `c1105e6ac6a0dd90fc80893efc4830ab609005d3` (2026-08-18, `master`) | `tools.hardfloat.commit` | `characterize` (float IP cores through `hardfloat-gen`, `docs/toolchain-image.md`) |

The Allo pin also fixes LLVM. Allo's own submodule `externals/llvm-project` records `6b09f739c4d085dc39eb9ff220c786bc3aa8c7fb` (`tools.allo.llvm_commit`). Nested submodules are **not** checked out on the host: `/home` is NFS and nearly full, and LLVM is large. The image build fetches LLVM at that commit itself. Allo's other nested submodule, `past-python-bindings`, is not needed: its `requirements.txt` installs the pinned `past` wheel.

## Patches

`patches/<submodule>/*.patch` are applied inside the image build only, in file-name order (`patch -p1 --forward`).
A submodule is never edited in place. The patches are pinned with the submodule and kept local: they are ours,
written against the pinned commits, and an upstream change is picked up only when a pin moves (step 3 below).

| Patch | Why |
|---|---|
| `patches/grami/0001-drop-unused-javax-imports.patch` | Removes `import javax.jws.Oneway;` and `import javax.rmi.CORBA.Util;` (both unused) from the two `*_SUBGRAPHS` trees. Java 11 removed those modules (JEP 320), so GraMi does not compile with the image's Java 17 otherwise (`docs/tools/grami.md` §3). |
| `patches/grami/0002-add-a-node-bound.patch` | Adds `maxNodes=k` to the directed subgraph tree: a pattern with more than `k` nodes counts as infrequent at the pruner that implements `maxLabelAppearance`, so it is neither reported nor extended (the authors left the same check commented out there). GraMi has no size bound otherwise, and on the `map` family it enumerated every frequent pattern of any size: over 16 minutes at τ = 10 for patterns `mine` drops (docs/passes/mine.md §7). Every pattern within the bound is still found with its support (deep test `grami_equals_brute_force_on_random_graphs`). |

| `patches/allo/0001-xls-calls-return-their-value.patch` | The XLS emitter defines a called function as returning its value but inherited the Vitis emitter's call, which passes the result through a pointer (`fmul(v5, v6, &v8)`), so `xlscc` rejected every kernel with a call. Adds an `emitCall` for the XLS emitter (`T v = f(args);`; array arguments or several results are an emitter error) and drops the include of `xls_builtin.h`, which XLS does not ship. |
| `patches/allo/0002-external-verilog-modules.patch` | `allo.extern_verilog`: a function implemented by an external Verilog module (instantiation template or port map, delay, NumPy model). The XLS emitter writes `[[xlscc::hls_extern_verilog(...)]]` on the callee with the placeholders renamed to the C++ argument names; the `llvm` target binds the callee to the model. Allo's `IPModule` covers only C++ for the simulator and Vitis HLS (`docs/tools/allo.md` §5a). |
| `patches/allo/0003-xls-loop-directives.patch` | The XLS emitter fully unrolled every loop without a pipeline directive, dropped the requested initiation interval and emitted `unroll` factors as `hls_unroll yes factor=N`. Now each loop gets one directive that follows the schedule: no full unrolling unless asked (sequential pipelined loops in blocks), intervals and factors passed through (`docs/tools/allo.md` §5a). |
| `patches/allo/0004-xls-partition.patch` | `partition` had no effect on the XLS emitter (it emitted Vitis pragmas that `xlscc` ignores). A completely partitioned array becomes `hls_array_as_tuple` and stays in registers in memory mode; a partial partition is recorded as a comment. |
| `patches/xls/0001-xlscc-extern-verilog.patch` | `xlscc` had no way to mark a C++ function as an external Verilog module, which XLS IR and codegen support (DSLX's `extern_verilog`). Adds the attribute `[[xlscc::hls_extern_verilog("<template>", <delay_ps>)]]`: the function's IR function gets the template as foreign function data, so `codegen_main` instantiates the module. Unit tests in `xls/contrib/xlscc/unit_tests/translator_extern_verilog_test.cc` (`docs/tools/xls.md` §7a). |

Each patch adds its tests: the Allo ones in `tests/test_xls_extensions.py`, run in the image's final stage; the
`xlscc` one as a Bazel test, run in the image's Bazel stage. `chipc doctor` checks that every patch set still
applies to its submodule, each patch on top of the ones before it (`git apply` on a copy of the touched files).

## Image-only tools

Built or installed only in the toolchain image (`docker/Dockerfile`), never vendored here. What they generate
(VHDL, Verilog netlists) stays in build directories and never enters the repository.

| Tool | Source and pin | Licence | Used for |
|---|---|---|---|
| FloPoCo | https://gitlab.com/flopoco/flopoco, tag `flopoco-5.1` = `8aa634812d2c9959ed33718d16648d7afcf7fa23` (`tools.docker.build.flopoco_tag`, `flopoco_commit`); built against the apt snapshot's Sollya, GMP, MPFR, MPFI, Boost and LAPACK, at `/opt/flopoco/bin/flopoco` | FloPoCo licence (GNU AGPLv3-based) for the generator; the code it generates is conveyed under GNU LGPLv3 (`/opt/flopoco/COPYING`) | open-flow float operators without another open core (exponential, logarithm, power; design §4.12) |
| GHDL 4.1.0 and the ghdl-yosys plugin | Ubuntu 24.04 packages `ghdl`, `ghdl-gcc` (the library tree the plugin reads), `yosys-plugin-ghdl` (git `5b64ccf`), frozen by `apt_snapshot` | GPL-2.0-or-later (GHDL), GPL-3.0-or-later (plugin) | reading FloPoCo's VHDL into Yosys (`yosys -m ghdl`) |
| Berkeley HardFloat generator (`hardfloat-gen`) | HardFloat's Chisel sources from `hardfloat/`, our `docker/hardfloat/` (`Generate.scala`), Chisel `5.0.0` and Scala `2.13.10` from Maven Central, built with sbt `1.9.9` (`tools.docker.build.chisel_version`, `scala_version`, `sbt_version`, `sbt_sha256`); at `/opt/hardfloat` | BSD-3-Clause (HardFloat, notice below); Apache-2.0 (Chisel) | open-flow float arithmetic, compare and conversions, and the narrowing conversions DesignWare lacks (design §4.12) |
| firtool (CIRCT) 1.40.0 | https://github.com/llvm/circt release `firtool-1.40.0`, `firrtl-bin-ubuntu-20.04.tar.gz` (`tools.docker.build.firtool_version`, `firtool_sha256`), the release Chisel 5.0.0 is tested with; at `/opt/firtool/bin/firtool` | Apache-2.0 with LLVM exceptions | lowering the HardFloat generator's circuits to Verilog |
| XLS DSLX float libraries | `third_party/xls_go_math/*.x` (`fp_sincos_32`, `fp_trig_reduce`, `fpexp_32`, `fpsqrt_32`) and `third_party/xls_berkeley_softfloat/fpdiv_2x32.x` from the pinned XLS submodule, at `/opt/xls/share/third_party/` (`XLS_DSLX_PATH=/opt/xls/share`) | BSD-3-Clause (Go authors; Berkeley SoftFloat), with the Cephes notice in the sources | open-flow f32 sine and cosine (`fp_sincos_32`; design §6.4) |

FloPoCo 5.1 includes PAGSuite's header in its constant multiplier unconditionally, although its build treats
PAGSuite (and ScaLP) as optional. PAGSuite is research-only ("for commercial use, please ask for a license") and
not packaged, so the image builds FloPoCo without it: `docker/patches/flopoco/0001-build-without-pagsuite.patch`
guards that include and defines PAGSuite's don't-care marker. The shift-and-add constant multipliers are then
unavailable; FPExp, FPLog and FPPow use table-based constant multipliers (`FixRealKCM`) and are unaffected.

## Berkeley HardFloat licence

HardFloat is used under its BSD-3-Clause licence, reproduced here from `third_party/hardfloat/LICENSE` (the
image keeps a copy at `/opt/hardfloat/LICENSE`). The Verilog the generator writes carries HardFloat's
circuits and stays in build directories.

> Copyright (c) 2010-2015, The Regents of the University of California
> (Regents).  All Rights Reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
> 1. Redistributions of source code must retain the above copyright
>    notice, this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright
>    notice, this list of conditions and the following disclaimer in the
>    documentation and/or other materials provided with the distribution.
> 3. Neither the name of the Regents nor the
>    names of its contributors may be used to endorse or promote products
>    derived from this software without specific prior written permission.
>
> IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
> SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING
> OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS
> BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
> THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
> PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
> HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
> MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

DesignWare components are referenced by name and parameters only (the industry flow links them from the
licensed installation); nothing of them is in the repository or the image.

## Fresh clone

```bash
git submodule update --init --depth 1        # non-recursive: no LLVM on the host
```

## Updating a pin

1. Fetch the new commit into the submodule and check it out:
   ```bash
   git -C third_party/<name> fetch --depth 1 origin <sha>
   git -C third_party/<name> checkout <sha>
   ```
   XLS: take the commit of a release tag, so the release tarball and the Bazel-built tools come from the same source.
2. Update `config.toml`: the commit. XLS also needs `tag` and `release_sha256`, which is the release asset's digest from the GitHub releases API. For Allo, set `llvm_commit` to `git -C third_party/allo ls-tree HEAD externals/llvm-project`.
3. Check that the patches still apply, as a series: `chipc doctor` (`submodule.<name>.patches`). Regenerate a
   patch that does not apply against the new commit.
4. Bump `[tools.docker] tag` and rebuild the image (`docker/build.sh`, see `docs/toolchain-image.md`).
5. `git add third_party/<name> config.toml`, then run `chipc doctor` until it is green.
