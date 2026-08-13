---
name: code-simplifier
description: >
  Review repo-map source changes or the full codebase for behavior-preserving
  simplifications, dead code, duplicated work, and Python 3.12 idioms.
metadata:
  author: cyanheads
  version: "1.0"
  audience: project
  type: workflow
---

# Code Simplifier

Make only changes that materially improve clarity, correctness, efficiency, or cohesion. A different personal preference is not a reason to edit working code.

## Procedure

1. Run `git status --short` and `git diff HEAD`. Read untracked files directly.
2. Read every changed file in full, then read adjacent modules that define the local pattern.
3. For a whole-codebase pass, read every file under `src/`, plus `scripts.py` and `scripts/`.
4. Check for:
   - duplicated parsing, file reads, serialization, or network work;
   - deep nesting, redundant state, dead code, and impossible-state guards;
   - legacy typing (`Optional`, `List`, `Dict`) instead of Python 3.12 syntax;
   - sequential independent async work and resource lifetimes that block concurrency;
   - broad exception handling, silent fallbacks, and failures cached as successes;
   - behavior that contradicts CLI flags, README claims, or type contracts.
5. Filter ruthlessly. Do not change public CLI flags, report formats, cache schemas, or analysis semantics under the label of simplification.
6. Apply small transformations incrementally, keeping the diff focused.
7. Run `poetry run python scripts.py check`, `poetry build`, and a CLI smoke test. A simplification that breaks a gate is not an improvement.

## Python transformations

- Prefer `str | None`, built-in generics, comprehensions, context managers, and guard clauses.
- Use `asyncio.gather()` or tasks only when operations are genuinely independent and concurrency is bounded.
- Avoid repeated `ast.parse()` or file reads when one parsed representation can serve the callers.
- Do not replace readable code with clever expressions or introduce an abstraction for one use.
- Do not DRY tests aggressively; isolated test setup is often clearer.

## Constraints

- Preserve user-visible behavior unless the task explicitly includes a fix.
- Never commit, tag, push, stage, stash, or create a worktree as part of simplification.
- Report behavior-changing findings separately instead of silently folding them into cleanup.
