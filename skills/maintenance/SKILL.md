---
name: maintenance
description: >
  Investigate, adopt, and verify dependency updates for repo-map. Captures what
  changed, understands why, cross-references against the codebase, applies
  fixes, and runs final checks. Supports two entry modes: run the full flow
  end-to-end, or review updates already applied.
metadata:
  author: cyanheads
  version: "1.0"
  audience: project
  type: workflow
---

## When to Use

- After running `poetry update` (or manually bumping a `pyproject.toml` constraint) and wanting to review the impact (**Mode B** — typical)
- To run the whole flow end-to-end — outdated check → update → investigate → adopt → verify (**Mode A**)
- Before tagging a release, to make sure the lockfile and code are coherent

## Entry Modes

| Mode | Starting Point | First Step |
|:-----|:---------------|:-----------|
| **A — Full flow** | Lockfile is current; want to update | Step 1 |
| **B — Post-update review** | User already ran `poetry update` (and possibly edited `pyproject.toml`) | Skip to Step 3 with the update output or `git diff poetry.lock pyproject.toml` |

Both modes converge at Step 3 and end at Step 8.

## Steps

### 1. Survey what's outdated (Mode A only)

```bash
poetry show --outdated
```

Columns are `name | current | latest | description`. Note which jumps cross majors — Poetry's caret constraints (`^1.2.3`) cap at the next major and **`poetry update` will not cross them**. For pre-1.0 packages a minor bump is effectively a major (e.g. `tree-sitter ^0.21 → 0.25` requires editing the constraint).

### 2. Apply the update (Mode A only)

For in-range bumps:

```bash
poetry update
```

For out-of-range jumps (caret-locked majors, pre-1.0 minor crossings), edit the constraint in `pyproject.toml` first, then `poetry update <package>`. Alternatively `poetry add <package>@latest` rewrites the constraint and updates in one step.

Capture the version deltas from stdout (or `git diff poetry.lock pyproject.toml`) — these feed Step 3.

### 3. Investigate changelogs

If the `changelog` skill is available, invoke it with the captured list of updated packages — it resolves repos, fetches release notes between old and new versions, and cross-references against actual imports. Skip the per-package research below if so.

Otherwise, for each updated package:

1. Find the source repo (`poetry show <package>` shows the homepage; PyPI page links to it)
2. Read CHANGELOG / release notes between old and new version tags
3. Grep `src/` for imports of the package and cross-reference against the changes
4. Note: what changed, impact on repo-map, action items

Focus only on packages we actually import from. Transitive bumps (`attrs`, `idna`, `platformdirs`, etc.) usually need no investigation unless they fail at runtime.

The packages that matter for repo-map:

| Package | Used In | Why It Matters |
|:--------|:--------|:---------------|
| `aiohttp` | `llm_service.py` | OpenRouter HTTP client; session/timeout/retry semantics |
| `pydantic` / `pydantic-settings` | `config.py`, `models.py` | Settings + data models; v2.x is iterating fast |
| `tree-sitter` / `tree-sitter-languages` | `code_parser.py` | Parser API has had breaking changes pre-1.0 |
| `pathspec` | `file_scanner.py` | `.gitignore` matching |
| `tqdm` | `main.py`, `logging_utils.py` | Progress bars + logging handler |
| `certifi` | `llm_service.py` | TLS bundle; new releases just refresh certs |
| `ruff` / `black` / `pytest` | dev tooling | New rules / format diffs / test runner changes |

### 4. Adopt changes in the codebase

Apply findings from Step 3. Defaults are cost/benefit — there's no upstream framework here that's authoritative.

- **Breaking changes** — fix call sites. Not optional.
- **Deprecation warnings** — migrate now, while context is fresh. Don't silence.
- **New `ruff` rules** that flag existing code — fix the code; don't add ignores.
- **`black` format drift** — run `poetry run black src` and commit the reformat as a separate concern from code changes if the diff is large.
- **New library APIs that supersede local helper code** — swap them in if it's a clear win (smaller surface, removed dep, fewer bugs). Skip if marginal.
- **New features that don't match existing call sites** — note for later, don't refactor speculatively.

### 5. Verify

```bash
poetry run ruff check src
poetry run black --check src
poetry run pytest
```

`pytest` will pass trivially if no tests exist yet — that's expected for now. Smoke test the actual tool against a small sample repo to catch runtime regressions the linters won't:

```bash
poetry run repo-map /path/to/small/repo -y
```

Watch for new exceptions, broken parsing, or LLM call failures. Fix anything that fails. Re-run until clean.

### 6. Sync skills to agent mirrors

`skills/` is the canonical source; `.agents/skills/` and `.claude/skills/` are read-only mirrors that local agent toolchains consume. They drift silently when `skills/` is edited.

A Claude Code `PostToolUse` hook (in `.claude/settings.json`) runs the sync automatically after any Write/Edit/MultiEdit under `skills/`. If you authored skills outside Claude Code, or want to be sure, run it manually:

```bash
poetry run sync-skills
# or, no Poetry:
python3 scripts/sync_skills.py
```

The script is idempotent — it only copies missing or content-drifted files, and never deletes mirror-only files (those may be general-purpose skills sourced elsewhere).

### 7. Wrap-up artifacts

Update files that frequently fall out of sync:

| File | When |
|:-----|:-----|
| `CHANGELOG.md` | Always — add an entry under a new version with the dependency bumps and any code adoptions |
| `pyproject.toml` `version` | Bump (patch for dep-only updates, minor if code behavior changed) |
| `AGENTS.md` / `CLAUDE.md` | If a workflow command, env var, or module responsibility changed |
| `README.md` | Only if user-facing behavior, defaults, or commands changed (badges are dynamic from PyPI — skip those) |

Use a concrete version + date in `CHANGELOG.md` (e.g. `## [0.7.1] - 2026-04-25`) — never `[Unreleased]`.

### 8. Summary

Present a concise numbered summary to the user:

1. **Updated packages** — short list with version deltas (N total)
2. **Breaking changes handled** — call sites fixed
3. **Adoptions made** — new APIs swapped in, helpers removed
4. **Wrap-up artifacts** — files updated (CHANGELOG entry, version bump, doc edits)
5. **Open decisions** — ambiguous items: out-of-range majors that weren't taken, adoptions where the tradeoff is close, deprecation warnings deferred
6. **Status** — ruff / black / pytest / smoke-test results

## Checklist

- [ ] Update applied (`poetry update` or constraint edit + `poetry update <pkg>`) — Mode A, or already done by user — Mode B
- [ ] Per-package changelog investigation done (via `changelog` skill or manually)
- [ ] Adoption opportunities identified and applied
- [ ] `poetry run ruff check src` clean
- [ ] `poetry run black --check src` clean
- [ ] `poetry run pytest` passes
- [ ] Smoke test against a sample repo passes
- [ ] `skills/` mirrors in sync (auto via hook, or `poetry run sync-skills`)
- [ ] `CHANGELOG.md` entry added with concrete version + date
- [ ] `pyproject.toml` version bumped
- [ ] `AGENTS.md` / `CLAUDE.md` / `README.md` updated if behavior or commands changed
- [ ] Numbered summary presented to user
