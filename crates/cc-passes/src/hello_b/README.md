# `hello_b` (framework self-test)

1. **Purpose.** Second link of `hello_a → hello_b`: exercises a dependency edge, reading a declared input, and a certificate.
2. **Background.** `crates/cc-pipeline/README.md`.
3. **Inputs / outputs.** Reads `hello.a` (`greeting.json`). Writes `out/hello.b/message.json`: `{ "message": string, "input_hash": blake3 hex }`, where `input_hash` certifies which `hello.a` content the message came from.
4. **Algorithm.** `message = greeting + suffix`.
5. **Config** (`[passes.hello_b]`): `suffix = ", world"`, `sleep_ms = 0`, `fail = false` (true makes `run` fail: used by the resume test).
6. **Verification.** `message_parses`; `message_recomputed` (from the input and config); `certificate_input_hash` (the recorded hash equals the input's current content hash).
7. **Resources.** Constant.
8. **Limitations.** None; it is a fixture.
