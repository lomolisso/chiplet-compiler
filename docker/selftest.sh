#!/bin/bash
# Image self-test: every tool does one small real job, so a broken image fails at build
# time instead of in a pass. Runs in seconds; only tiny generated inputs (design §8).
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

step "Allo external module -> xlscc -> codegen instantiates it"
cat > allo_extern.py <<'EOF2'
import numpy as np
import allo
from allo.ir.types import uint32


def f32_mul(a, b):
    x = np.array([a, b], dtype=np.uint32).view(np.float32)
    return int(np.array([x[0] * x[1]], dtype=np.float32).view(np.uint32)[0])


@allo.extern_verilog(
    "DW_fp_mult #(.sig_width(23), .exp_width(8), .ieee_compliance(0)) "
    "{fn} (.a({a}), .b({b}), .rnd(3'b000), .z({return}), .status())",
    delay_ps=800,
    model=f32_mul,
)
def fmul(a: uint32, b: uint32) -> uint32:
    r: uint32 = a ^ b
    return r


def unit(a: uint32, b: uint32, c: uint32) -> uint32:
    p: uint32 = fmul(a, b)
    y: uint32 = p ^ c
    return y


s = allo.customize(unit)
a, b = int(np.float32(1.5).view(np.uint32)), int(np.float32(-2.25).view(np.uint32))
assert np.uint32(s.build()(a, b, 0)).view(np.float32) == np.float32(-3.375)
s.build(target="xls", project="allo_extern")
EOF2
python3 allo_extern.py
xlscc allo_extern/test_block.cpp --top=unit --include_dirs="${XLSCC_INCLUDE_DIRS}" --defines="${XLSCC_DEFINES}" \
  --out=allo_extern.ir
grep -q 'ffi_proto' allo_extern.ir
top="$(sed -n 's/^top fn \([^(]*\)(.*/\1/p' allo_extern.ir)"
opt_main allo_extern.ir --top="${top}" > allo_extern.opt.ir
codegen_main allo_extern.opt.ir --top="${top}" --generator=pipeline --delay_model=asap7 --clock_period_ps=1000 \
  --use_system_verilog=false --module_name=unit --output_verilog_path=allo_extern.v \
  --output_signature_path=allo_extern.sig.textproto
grep -q '^ *DW_fp_mult #(' allo_extern.v
pass xls_extern

step "XLS float library (fp_sincos_32) -> Verilog -> yosys"
ir_converter_main --dslx_stdlib_path="${XLS_STDLIB}" --dslx_path="${XLS_DSLX_PATH}" --top=fp_sincos_32 \
  "${XLS_DSLX_PATH}/third_party/xls_go_math/fp_sincos_32.x" > sincos.ir
opt_main sincos.ir > sincos.opt.ir
codegen_main sincos.opt.ir --generator=combinational --use_system_verilog=false --module_name=fp_sincos_32 \
  --output_verilog_path=sincos.v
yosys -q -p 'read_verilog sincos.v; hierarchy -check -top fp_sincos_32; tee -q -o sincos.stat stat'
grep -q 'cells' sincos.stat
pass xls_float_lib

step "FloPoCo -> GHDL -> yosys"
flopoco FPExp wE=5 wF=10 frequency=0 outputFile=fpexp.vhdl > fpexp.log 2>&1
yosys -q -m ghdl -p 'ghdl --std=08 -fsynopsys fpexp.vhdl -e; hierarchy -check; tee -q -o fpexp.stat stat'
grep -q 'cells' fpexp.stat
pass flopoco_ghdl

step "HardFloat generator -> Verilog -> yosys, iverilog"
hardfloat-gen add --exp 5 --sig 10 --module hf_add_f16 --out hf_add_f16.v
yosys -q -p 'read_verilog hf_add_f16.v; hierarchy -check -top hf_add_f16; synth -top hf_add_f16 -flatten; tee -q -o hf_add_f16.stat stat'
grep -q 'cells' hf_add_f16.stat
cat > hf_add_f16_tb.v <<'EOF3'
module tb;
  reg [15:0] a, b;
  wire [15:0] z;
  wire [4:0] flags;
  hf_add_f16 dut(.a(a), .b(b), .rnd(3'd0), .z(z), .flags(flags));
  initial begin
    a = 16'h3c00; b = 16'h4000; #1;  // 1 + 2 = 3
    if (z !== 16'h4200) begin $display("FAIL 1+2 = %h", z); $finish; end
    a = 16'h0001; b = 16'h0001; #1;  // two smallest subnormals: exact
    if (z !== 16'h0002) begin $display("FAIL subnormal sum = %h", z); $finish; end
    $display("PASS");
  end
endmodule
EOF3
iverilog -o hf_add_f16.vvp hf_add_f16_tb.v hf_add_f16.v
vvp -n hf_add_f16.vvp | grep -q '^PASS'
pass hardfloat

step "ok"
