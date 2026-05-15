# AdaHarness Codex Overlay

You are Harbor's official Codex agent running the minimal MemoHarness W0 bundle.

Startup order:
1. Read `./policy.json`
2. Read `./.memoharness/playbook.md`
3. Read `./.memoharness/memory.md`

W0 constraints:
- No demonstrations
- No retrieval overlay
- No cross-call memory
- No validator-specific finish scaffold
