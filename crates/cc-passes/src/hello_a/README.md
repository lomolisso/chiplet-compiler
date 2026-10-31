# `hello_a` (framework self-test)

1. **Purpose.** Root of the toy chain `hello_a → hello_b`; exists only to exercise the framework. It never touches a model.
2. **Background.** `crates/cc-pipeline/README.md`.
3. **Outputs.** `out/hello.a/greeting.json`: `{ "greeting": string, "chars": u64 }`.
4. **Algorithm.** Sleep `sleep_ms` (cancellable), write the greeting and its character count.
5. **Config** (`[passes.hello_a]`): `greeting = "hello"`, `sleep_ms = 0`.
6. **Verification.** `greeting_parses`, `greeting_matches_config` (against the config, not `run`), `char_count` (recounted).
7. **Resources.** Constant.
8. **Limitations.** None; it is a fixture.
