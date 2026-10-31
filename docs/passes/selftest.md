# Self-test passes

`hello_a`, `hello_b` and `hello_c` exercise the framework (DAG, inputs, certificates, failure and resume). They never touch a model.

## `hello_a` (framework self-test)

1. **Purpose.** Root of the toy chain `hello_a → hello_b`; exists only to exercise the framework. It never touches a model.
2. **Background.** `docs/framework/pipeline.md`.
3. **Outputs.** `out/hello.a/greeting.json`: `{ "greeting": string, "chars": u64 }`.
4. **Algorithm.** Sleep `sleep_ms` (cancellable), write the greeting and its character count.
5. **Config** (`[passes.hello_a]`): `greeting = "hello"`, `sleep_ms = 0`.
6. **Verification.** `greeting_parses`, `greeting_matches_config` (against the config, not `run`), `char_count` (recounted).
7. **Resources.** Constant.
8. **Limitations.** None; it is a fixture.

## `hello_b` (framework self-test)

1. **Purpose.** Second link of `hello_a → hello_b`: exercises a dependency edge, reading a declared input, and a certificate.
2. **Background.** `docs/framework/pipeline.md`.
3. **Inputs / outputs.** Reads `hello.a` (`greeting.json`). Writes `out/hello.b/message.json`: `{ "message": string, "input_hash": blake3 hex }`, where `input_hash` certifies which `hello.a` content the message came from.
4. **Algorithm.** `message = greeting + suffix`.
5. **Config** (`[passes.hello_b]`): `suffix = ", world"`, `sleep_ms = 0`, `fail = false` (true makes `run` fail: used by the resume test).
6. **Verification.** `message_parses`; `message_recomputed` (from the input and config); `certificate_input_hash` (the recorded hash equals the input's current content hash).
7. **Resources.** Constant.
8. **Limitations.** None; it is a fixture.

## `hello_c` (framework self-test)

1. **Purpose.** A branch independent of `hello_a → hello_b`, so the scheduler can run it in parallel; also the one toy pass that calls an external tool.
2. **Background.** `docs/framework/pipeline.md`, `docs/framework/core.md` (tool runners).
3. **Outputs.** `out/hello.c/echo.json`: `{ "text": string, "stdout": string, "log": path of the call's stdout log }`.
4. **Algorithm.** Sleep `sleep_ms`, run `/bin/echo <text>` through the `HostRunner` (capped, logged to `logs/tools.jsonl`).
5. **Config** (`[passes.hello_c]`): `text = "independent"`, `sleep_ms = 0`.
6. **Verification.** `echo_parses`; `stdout_is_text`; `tool_call_logged` (the call appears in `logs/tools.jsonl`).
7. **Resources.** Constant. Tool: `echo` (its `--version` output enters the cache key).
8. **Limitations.** Needs GNU coreutils `/bin/echo` (for `--version`).
