# Rolling Memory

Updated after iteration 16.

- Read `./policy.json` first and use this file only for recent distilled lessons.
- Keep only recurring failures and verified repair heuristics here.

Recent distilled patterns:
- [D6] Missing or malformed artifacts are still the top live failure mode. Enforce a hard per-edit gate: rerun the exact failing producer/test command, assert required files with `test -s`, then run one predicate-level probe tied to the failing assertion.
  Evidence: dna-assembly, dna-insert, video-processing, model-extraction-relu-logits
- [D4] Repeated process/runtime signatures plateau when logic edits happen too early. After two identical signatures, freeze logic changes and patch runner/entrypoint/args/env/build-target first, then require one bounded smoke exit 0.
  Evidence: torch-pipeline-parallelism, torch-tensor-parallelism
- [D4] One-test-left plateaus need forced hypothesis pivots. If the same final failing test repeats twice, rotate hypothesis class (contract/path, runtime/env, algorithm/perf) and keep checks focused on that single failing test.
  Evidence: adaptive-rejection-sampler, query-optimize

Guardrail:
- Never treat completion text as evidence; only verifier-facing command outputs count.
