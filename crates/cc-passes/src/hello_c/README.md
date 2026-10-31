# `hello_c` (framework self-test)

1. **Purpose.** A branch independent of `hello_a → hello_b`, so the scheduler can run it in parallel; also the one toy pass that calls an external tool.
2. **Background.** `crates/cc-pipeline/README.md`, `crates/cc-core/README.md` (tool runners).
3. **Outputs.** `out/hello.c/echo.json`: `{ "text": string, "stdout": string, "log": path of the call's stdout log }`.
4. **Algorithm.** Sleep `sleep_ms`, run `/bin/echo <text>` through the `HostRunner` (capped, logged to `logs/tools.jsonl`).
5. **Config** (`[passes.hello_c]`): `text = "independent"`, `sleep_ms = 0`.
6. **Verification.** `echo_parses`; `stdout_is_text`; `tool_call_logged` (the call appears in `logs/tools.jsonl`).
7. **Resources.** Constant. Tool: `echo` (its `--version` output enters the cache key).
8. **Limitations.** Needs GNU coreutils `/bin/echo` (for `--version`).
