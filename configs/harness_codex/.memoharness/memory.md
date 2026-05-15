# Rolling Memory

Updated after iteration 4.

- Read `./policy.json` first and use this file only for recent distilled lessons.
- Keep only recurring failures and verified repair heuristics here.

Recent distilled patterns:
- [D2] Context throttling likely became too aggressive after iteration 2 (`top_k` 4→3, history 6→4, tokens 4096→2048), while tasks need exact contract details. This can amplify wrong-branch fixes and force heuristic rewrites, especially when combined with noisy verifier snippets.
  Effect: Correlates with no recovery after config shrink in 5+ cases. Raise D2 recall for failing cases: `top_k` back to 5 with one required problem-statement source + one verifier clue; pair with D3 max_tokens 3072 to preserve contract and proof context without large read dumps.
  Evidence: arc188_c, arc195_c, abc397_d
- [D4] Before infrastructure failures, outputs show default/sentinel or extreme placeholder values (`11111`, `500000000`, `-1`, wrong winner/string), suggesting weak contract-to-invariant grounding and insufficient targeted counterexample checks. The current D4 pivot policy is broad but not forcing minimal falsification against the observed mismatch class.
  Effect: At least 6 concrete wrong-answer signatures across cases. Tighten D4 with mandatory mismatch-driven microtests: reproduce failing line pattern, add 2 boundary tests + 1 formatting test before finalization; block completion unless new tests pass and differ from prior placeholder/default branch behavior.
  Evidence: arc188_c, arc195_c, abc397_d
- [D4] The dominant failure is execution-path collapse in the agentic loop: iterations 3–4 repeatedly die with shell-level `codex exec` exit 1 before meaningful inspect/edit/check work. This indicates D4 workflow/stop behavior is not resilient to runner faults and keeps re-entering a broken path, producing zero-reward plateaus across tasks.
  Effect: Impacts 6/6 cases (all D4-primary, 0 reward in last 2 iters). Change D4 to fail-open: on first runner error, switch to direct bash+local edit mode, require at least one successful check command before stop, and add retry budget (e.g., 2) with alternate execution template.
  Evidence: arc188_c, arc195_c, abc397_d
- [D3] Token/model downgrade reduced reasoning headroom on hard hidden-test bugs; moving from gpt-5.3-codex/4096 to gpt-5.4/2048 coincides with no recovery and repeated D4 failures. Complex game/parity/default-branch cases likely need deeper search and longer chains.
  Effect: Across 4+ cases, post-downgrade iterations remain at reward 0. In D3, restore max_tokens to 4096–6144 for terminal domains and enable candidate_count=2 only after first failed proof run; keep high reasoning_effort to improve hidden-branch diagnosis.
  Evidence: arc192_b, arc188_c, arc195_c
- [D2] Early retrieval budget is too tight (top_k 3, first-three-call constraints), leading to under-specification and persistent contract mistakes (sentinel outputs, wrong branch defaults, formatting/line mismatches). The loop edits before collecting decisive checker/statement clues.
  Effect: Multiple cases show stable wrong outputs despite iterations. In D2, raise top_k to 5 and require first calls include: solution file + official checker/validator or tests + one failing trace artifact; keep 200-line caps but mandate one targeted grep over output-format tokens.
  Evidence: arc195_c, abc397_d, 3308