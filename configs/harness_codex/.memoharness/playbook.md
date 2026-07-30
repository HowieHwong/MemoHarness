# Repo Playbook

Use `./policy.json` as the authoritative D1-D6 summary.
Use this file for stable MemoHarness heuristics that should survive iteration-to-iteration.

Stable loop:
- Contract first: extract required files/paths, output format/order, numeric bounds, and exact stdout/text constraints.
- Reproduce first: run the exact failing producer/test command and capture the error signature.
- Edit narrowly: one hypothesis class, one targeted edit, one proof check.
- Gate before done: run verifier-facing checks again before final response.

Iteration-16 priorities:
- D6 hard artifact gate (dna-assembly, dna-insert, video-processing, model-extraction-relu-logits): exact failing command + `test -s` required paths + one predicate probe.
- D4 repeated-process runtime lane (torch-pipeline-parallelism, torch-tensor-parallelism): after two identical process signatures, patch runner/entrypoint/args/env/build-target and prove smoke exit 0 before logic edits.
- D4 one-test-left pivot (adaptive-rejection-sampler, query-optimize): after two unchanged signatures, switch hypothesis class and rerun only the failing test to confirm delta.

Guardrails:
- Keep tool reads/searches path-scoped and <=200 lines.
- Avoid speculative rewrites on never-solved capability-limited tasks unless verifier evidence changes.
- Do not rely on completion claims; rely on command outputs only.
