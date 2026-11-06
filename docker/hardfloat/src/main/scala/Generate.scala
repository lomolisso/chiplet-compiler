// The HardFloat generator: an IEEE-interface module around one Berkeley HardFloat core (J. R. Hauser,
// github.com/ucb-bar/berkeley-hardfloat, BSD-3-Clause). HardFloat's cores work on its recoded format (one
// more exponent bit); the module converts its IEEE operands with `recFNFromFN` and its result back with
// `fNFromRecFN`, both HardFloat's own. Chisel elaborates it and firtool lowers it to Verilog; every module
// but the top is renamed `<top>__<name>`, so that cores of different widths can share one design.
//
// Widths follow DesignWare's convention: `--exp` exponent bits, `--sig` stored significand bits (no hidden
// bit); HardFloat's own `sigWidth` is `--sig` + 1. Ports: operands `a`, `b`, rounding mode `rnd` (HardFloat's
// encoding: 0 nearest even, 1 toward zero, 2 down, 3 up, 4 nearest away), result `z` (compare: `lt`, `eq`,
// `gt`), exception flags `flags`; the iterative divider and square root add `clock`, `reset`, `start`,
// `ready` and `done`.
package chipc.hardfloat

import chisel3._
import circt.stage.ChiselStage
import hardfloat._

final case class Spec(
    op: String,
    exp: Int,
    sig: Int,
    intWidth: Int,
    signed: Boolean,
    outExp: Int,
    outSig: Int,
    module: String,
    out: String
)

object Ports {
  def tininess: UInt = consts.tininess_afterRounding
}

/** a + b or a - b (`AddRecFN`). */
class Add(s: Spec, sub: Boolean) extends RawModule {
  override def desiredName = s.module
  private val p = s.sig + 1
  val a = IO(Input(UInt((s.exp + p).W)))
  val b = IO(Input(UInt((s.exp + p).W)))
  val rnd = IO(Input(UInt(3.W)))
  val z = IO(Output(UInt((s.exp + p).W)))
  val flags = IO(Output(UInt(5.W)))
  private val core = Module(new AddRecFN(s.exp, p))
  core.io.subOp := sub.B
  core.io.a := recFNFromFN(s.exp, p, a)
  core.io.b := recFNFromFN(s.exp, p, b)
  core.io.roundingMode := rnd
  core.io.detectTininess := Ports.tininess
  z := fNFromRecFN(s.exp, p, core.io.out)
  flags := core.io.exceptionFlags
}

/** a * b (`MulRecFN`). */
class Mul(s: Spec) extends RawModule {
  override def desiredName = s.module
  private val p = s.sig + 1
  val a = IO(Input(UInt((s.exp + p).W)))
  val b = IO(Input(UInt((s.exp + p).W)))
  val rnd = IO(Input(UInt(3.W)))
  val z = IO(Output(UInt((s.exp + p).W)))
  val flags = IO(Output(UInt(5.W)))
  private val core = Module(new MulRecFN(s.exp, p))
  core.io.a := recFNFromFN(s.exp, p, a)
  core.io.b := recFNFromFN(s.exp, p, b)
  core.io.roundingMode := rnd
  core.io.detectTininess := Ports.tininess
  z := fNFromRecFN(s.exp, p, core.io.out)
  flags := core.io.exceptionFlags
}

/** a / b or sqrt(a), iterative (`DivSqrtRecFN_small`): `start` with the operands while `ready`, the
  * result in `z` while `done`.
  */
class DivSqrt(s: Spec, sqrt: Boolean) extends Module {
  override def desiredName = s.module
  private val p = s.sig + 1
  val start = IO(Input(Bool()))
  val ready = IO(Output(Bool()))
  val a = IO(Input(UInt((s.exp + p).W)))
  val b = if (sqrt) None else Some(IO(Input(UInt((s.exp + p).W))))
  val rnd = IO(Input(UInt(3.W)))
  val done = IO(Output(Bool()))
  val z = IO(Output(UInt((s.exp + p).W)))
  val flags = IO(Output(UInt(5.W)))
  private val core = Module(new DivSqrtRecFN_small(s.exp, p, 0))
  core.io.inValid := start
  core.io.sqrtOp := sqrt.B
  core.io.a := recFNFromFN(s.exp, p, a)
  core.io.b := b.map(x => recFNFromFN(s.exp, p, x)).getOrElse(recFNFromFN(s.exp, p, a))
  core.io.roundingMode := rnd
  core.io.detectTininess := Ports.tininess
  ready := core.io.inReady
  done := (if (sqrt) core.io.outValid_sqrt else core.io.outValid_div)
  z := fNFromRecFN(s.exp, p, core.io.out)
  flags := core.io.exceptionFlags
}

/** Quiet comparison (`CompareRecFN`): exactly one of `lt`, `eq`, `gt` unless an operand is NaN. */
class Compare(s: Spec) extends RawModule {
  override def desiredName = s.module
  private val p = s.sig + 1
  val a = IO(Input(UInt((s.exp + p).W)))
  val b = IO(Input(UInt((s.exp + p).W)))
  val lt = IO(Output(Bool()))
  val eq = IO(Output(Bool()))
  val gt = IO(Output(Bool()))
  val flags = IO(Output(UInt(5.W)))
  private val core = Module(new CompareRecFN(s.exp, p))
  core.io.a := recFNFromFN(s.exp, p, a)
  core.io.b := recFNFromFN(s.exp, p, b)
  core.io.signaling := false.B
  lt := core.io.lt
  eq := core.io.eq
  gt := core.io.gt
  flags := core.io.exceptionFlags
}

/** Float to an integer of `intWidth` bits, rounded per `rnd` (`RecFNToIN`, saturating). */
class ToInt(s: Spec) extends RawModule {
  override def desiredName = s.module
  private val p = s.sig + 1
  val a = IO(Input(UInt((s.exp + p).W)))
  val rnd = IO(Input(UInt(3.W)))
  val z = IO(Output(UInt(s.intWidth.W)))
  val flags = IO(Output(UInt(3.W)))
  // RecFNToIN is a Module without state: tie its implicit clock and reset.
  private val core = withClockAndReset(false.B.asClock, false.B) { Module(new RecFNToIN(s.exp, p, s.intWidth)) }
  core.io.in := recFNFromFN(s.exp, p, a)
  core.io.roundingMode := rnd
  core.io.signedOut := s.signed.B
  z := core.io.out
  flags := core.io.intExceptionFlags
}

/** An integer of `intWidth` bits to float, rounded per `rnd` (`INToRecFN`). */
class FromInt(s: Spec) extends RawModule {
  override def desiredName = s.module
  private val p = s.sig + 1
  val a = IO(Input(UInt(s.intWidth.W)))
  val rnd = IO(Input(UInt(3.W)))
  val z = IO(Output(UInt((s.exp + p).W)))
  val flags = IO(Output(UInt(5.W)))
  private val core = Module(new INToRecFN(s.intWidth, s.exp, p))
  core.io.signedIn := s.signed.B
  core.io.in := a
  core.io.roundingMode := rnd
  core.io.detectTininess := Ports.tininess
  z := fNFromRecFN(s.exp, p, core.io.out)
  flags := core.io.exceptionFlags
}

/** Float to float, rounded per `rnd` (`RecFNToRecFN`). */
class Convert(s: Spec) extends RawModule {
  override def desiredName = s.module
  private val p = s.sig + 1
  private val q = s.outSig + 1
  val a = IO(Input(UInt((s.exp + p).W)))
  val rnd = IO(Input(UInt(3.W)))
  val z = IO(Output(UInt((s.outExp + q).W)))
  val flags = IO(Output(UInt(5.W)))
  private val core = Module(new RecFNToRecFN(s.exp, p, s.outExp, q))
  core.io.in := recFNFromFN(s.exp, p, a)
  core.io.roundingMode := rnd
  core.io.detectTininess := Ports.tininess
  z := fNFromRecFN(s.outExp, q, core.io.out)
  flags := core.io.exceptionFlags
}

object Generate {
  val ops = Seq("add", "sub", "mul", "div", "sqrt", "compare", "to_int", "from_int", "convert")

  val usage: String =
    s"""usage: hardfloat-gen <op> --exp E --sig S [--int W --signed 0|1] [--out-exp E --out-sig S] --module NAME --out FILE
       |  op: ${ops.mkString(", ")}
       |  E: exponent bits, S: stored significand bits (no hidden bit); HardFloat needs E >= 3, S >= 2 and
       |  S + 1 <= 2^(E-2) + 3. --int and --signed for to_int and from_int; --out-exp and --out-sig for convert.""".stripMargin

  def fail(msg: String): Nothing = {
    System.err.println(s"hardfloat-gen: $msg\n$usage")
    sys.exit(2)
  }

  def parse(args: Array[String]): Spec = {
    if (args.isEmpty || args.contains("--help")) { println(usage); sys.exit(if (args.isEmpty) 2 else 0) }
    val op = args.head
    if (!ops.contains(op)) fail(s"unknown op $op")
    val kv = args.tail.grouped(2).map {
      case Array(k, v) if k.startsWith("--") => k.drop(2) -> v
      case other => fail(s"bad arguments ${other.mkString(" ")}")
    }.toMap
    val known = Set("exp", "sig", "int", "signed", "out-exp", "out-sig", "module", "out")
    kv.keys.find(k => !known(k)).foreach(k => fail(s"unknown option --$k"))
    def int(k: String, default: Option[Int] = None): Int =
      kv.get(k).map(v => v.toIntOption.getOrElse(fail(s"--$k: not a number: $v"))).orElse(default)
        .getOrElse(fail(s"--$k is required"))
    def str(k: String): String = kv.getOrElse(k, fail(s"--$k is required"))
    val needsInt = op == "to_int" || op == "from_int"
    val s = Spec(
      op = op,
      exp = int("exp"),
      sig = int("sig"),
      intWidth = if (needsInt) int("int") else 0,
      signed = if (needsInt) int("signed") == 1 else false,
      outExp = if (op == "convert") int("out-exp") else 0,
      outSig = if (op == "convert") int("out-sig") else 0,
      module = str("module"),
      out = str("out")
    )
    def check(e: Int, sg: Int, what: String): Unit = {
      val p = sg + 1
      if (e < 3 || p < 3 || BigInt(p) > (BigInt(1) << (e - 2)) + 3)
        fail(s"$what: (exp $e, sig $sg) outside HardFloat's formats")
    }
    check(s.exp, s.sig, "operand")
    if (op == "convert") check(s.outExp, s.outSig, "result")
    if (needsInt && s.intWidth < 1) fail("--int must be positive")
    if (!s.module.matches("[A-Za-z_][A-Za-z0-9_]*")) fail(s"--module: not a Verilog identifier: ${s.module}")
    s
  }

  def module(s: Spec): RawModule = s.op match {
    case "add"      => new Add(s, sub = false)
    case "sub"      => new Add(s, sub = true)
    case "mul"      => new Mul(s)
    case "div"      => new DivSqrt(s, sqrt = false)
    case "sqrt"     => new DivSqrt(s, sqrt = true)
    case "compare"  => new Compare(s)
    case "to_int"   => new ToInt(s)
    case "from_int" => new FromInt(s)
    case "convert"  => new Convert(s)
  }

  /** Every module the file defines except `top` becomes `<top>__<name>`. */
  def prefix(verilog: String, top: String): String = {
    val defined = """(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)""".r
      .findAllMatchIn(verilog)
      .map(_.group(1))
      .filter(_ != top)
      .toSet
    defined.toSeq.sortBy(-_.length).foldLeft(verilog) { (v, m) =>
      v.replaceAll(s"(?<![A-Za-z0-9_$$])${java.util.regex.Pattern.quote(m)}(?![A-Za-z0-9_$$])", s"${top}__$m")
    }
  }

  def main(args: Array[String]): Unit = {
    val s = parse(args)
    // Chisel's side files (annotations) go to a scratch directory, never next to the output.
    val scratch = java.nio.file.Files.createTempDirectory("hardfloat-gen")
    val v = ChiselStage.emitSystemVerilog(
      module(s),
      Array("--target-dir", scratch.toString),
      Array(
        "-disable-all-randomization",
        "-strip-debug-info",
        "--lowering-options=disallowLocalVariables,disallowPackedArrays,locationInfoStyle=none"
      )
    )
    val text = prefix(v, s.module)
    java.nio.file.Files.write(java.nio.file.Paths.get(s.out), text.getBytes("UTF-8"))
    scala.reflect.io.Directory(scratch.toFile).deleteRecursively()
  }
}
