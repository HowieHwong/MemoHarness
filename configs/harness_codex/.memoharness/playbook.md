# Repo Playbook

Use `./policy.json` as the authoritative D1-D6 summary.
Use this file for stable AdaHarness execution heuristics that should survive iteration-to-iteration.

Current priorities:
- Read `./policy.json` first, then use this playbook for stable execution heuristics.
- D1 strategy: Capture the exact livecodebench contract before editing: parsing, indexing, ordering, tie-breaks, blank-output legality, impossible outputs, numeric bounds, and exact stdout formatting.
- D2 strategy: Within the first three tool calls inspect the current solution and one bounded contract or verifier clue; then keep reads path-scoped, cap windows at 200 lines, avoid raw log dumps, and prefer one focused proof over broad exploration.
- D4 strategy: Use a contract -> invariant -> adversarial case -> edit -> proof loop; after one disproved hypothesis or two low-signal reads, pivot toward impossible or default branches, blank-output traps, parity bugs, boundary math, duplicates, and winner-selection logic before rewriting.
- D5 strategy: Keep only verifier-backed patterns: `calls=0` runs are unacceptable, TPM blowups come from oversized or repeated reads, and wrong outputs like `-1`, `0`, placeholder strings, or close numeric drift usually point to branch, boundary, or formatting bugs.
- D6 strategy: Before finishing, leave a real inspect/edit/check trace, run the smallest proof that exercises the repaired invariant, verify exact stdout formatting, and return a terse file-plus-proof summary.

Distilled emphasis:
- Recent distilled pressure is highest on D2 (2 pattern(s)); keep that dimension under active scrutiny.
- Recent distilled pressure is highest on D4 (2 pattern(s)); keep that dimension under active scrutiny.
- Recent distilled pressure is highest on D3 (1 pattern(s)); keep that dimension under active scrutiny.