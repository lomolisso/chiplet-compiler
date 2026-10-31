#!/bin/bash
# Image self-test: every tool does one small real job, so a broken image fails at build
# time instead of in a pass. Runs in seconds; only tiny generated inputs (spec §8).
# Usage: cc-selftest [KEEP_DIR]   With KEEP_DIR, inputs and outputs stay there (chipc doctor keeps
# them as evidence). Each finished step prints `cc-selftest: pass <step>`; the first failure stops.
set -euo pipefail
if [[ -n "${1:-}" ]]; then
  d="$1"
  mkdir -p "$d"
else
  d="$(mktemp -d)"
  trap 'rm -rf "$d"' EXIT
fi
cd "$d"
step() { printf 'cc-selftest: %s\n' "$*"; }
pass() { printf 'cc-selftest: pass %s\n' "$1"; }

step "xlscc -> opt_main -> codegen_main"
cat > add8.cc <<'EOF'
#include "xls_int.h"
#pragma hls_top
XlsInt<9, false> add8(XlsInt<8, false> a, XlsInt<8, false> b) {
  return XlsInt<9, false>(a) + XlsInt<9, false>(b);
}
EOF
xlscc add8.cc --top=add8 --include_dirs="${XLSCC_INCLUDE_DIRS}" --defines="${XLSCC_DEFINES}" --out=add8.ir
top="$(sed -n 's/^top fn \([^(]*\)(.*/\1/p' add8.ir)"
test -n "${top}"
opt_main add8.ir --top="${top}" > add8.opt.ir
codegen_main add8.opt.ir --top="${top}" --generator=combinational --use_system_verilog=false \
  --module_name=add8 --output_verilog_path=add8.v --output_signature_path=add8.sig.textproto
test -s add8.v
pass xls

step "check_ir_equivalence_main (unoptimised == optimised)"
check_ir_equivalence_main add8.ir add8.opt.ir --top="${top}" --timeout=60s
pass xls_equiv

step "yosys / iverilog / verilator read the generated RTL"
yosys -q -p 'read_verilog add8.v; synth -top add8'
iverilog -o add8.vvp add8.v
verilator --lint-only -Wno-fatal add8.v
pass rtl_tools

step "z3 / bitwuzla"
printf '%s\n' '(set-logic QF_BV)' '(declare-const x (_ BitVec 8))' \
  '(assert (distinct (bvadd x x) (bvshl x #x01)))' '(check-sat)' > q.smt2
test "$(z3 q.smt2)" = unsat
test "$(bitwuzla q.smt2)" = unsat
pass smt

step "GraMi (directed)"
printf '%s\n' '# t 1' 'v 0 1' 'v 1 2' 'v 2 3' 'v 3 1' 'v 4 2' 'v 5 3' \
  'e 0 1 0' 'e 1 2 0' 'e 3 4 0' 'e 4 5 0' > two_paths.lg
java -Xmx512m -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -cp "${GRAMI_DIRECTED_CP}" Dijkstra.main \
  filename=two_paths.lg datasetFolder=./ freq=2 limitedtime=false maxLabelAppearance=-1 approximate=1 >/dev/null
test "$(sed -n 2p Output.txt)" -ge 1
pass grami

step "Allo (llvm target)"
cat > allo_add.py <<'EOF'
import numpy as np
import allo
from allo.ir.types import int32


def add(A: int32[4], B: int32[4]) -> int32[4]:
    C: int32[4] = 0
    for i in range(4):
        C[i] = A[i] + B[i]
    return C


mod = allo.customize(add).build()
a = np.arange(4, dtype=np.int32)
b = 3 * np.arange(4, dtype=np.int32)
assert (mod(a, b) == a + b).all()
EOF
python3 allo_add.py
pass allo

step "ok"
