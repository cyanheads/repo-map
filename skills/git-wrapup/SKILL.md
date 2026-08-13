---
name: git-wrapup
description: >
  Verify and land a repo-map release locally as logical commits plus an
  annotated tag. Stops before push, PyPI publication, or GitHub release creation.
metadata:
  author: cyanheads
  version: "1.0"
  audience: project
  type: workflow
---

# Git Wrap-up

Use only when the user explicitly requests a commit or release wrap-up. Committing means the work is complete and ready to release.

## Preflight

- Read `git status`, the complete `git diff`, and every changed file in full.
- Run the local `code-simplifier` skill for substantive source changes.
- Confirm addressed GitHub issues contain a concise implementation update.
- Update user-facing documentation when behavior, flags, output, setup, or defaults changed.
- Never use `git stash`, worktrees, destructive git commands, or partial staging within one file.

## Version and changelog

Read the current version from `pyproject.toml`. Use the version explicitly selected by the user; if none was selected and the workflow requires a release, stop and ask rather than inventing one.

Update `pyproject.toml`, add `## [X.Y.Z] - YYYY-MM-DD` to `CHANGELOG.md`, and revise `README.md`, `CLAUDE.md`, or `AGENTS.md` only where they pin changed behavior or commands. Keep the changelog consumer-facing and concrete.

## Gates

Run from the repository root:

```bash
poetry run python scripts.py check
poetry build
poetry run repo-map --help
```

When a fix affects live OpenRouter behavior, also run the smallest redacted end-to-end fixture that verifies it. Do not send a private repository to an LLM as a smoke test.

## Commits and tag

Group files by logical concern. The file is the atomic boundary: never split one file across commits. Put version and changelog updates in the final release commit unless the release is genuinely a single tiny concern.

Use Conventional Commit subjects. Every commit gets a one- or two-line body explaining the load-bearing reason; no closing keywords, chat references, generated-by trailers, or marketing adjectives.

Create an annotated `vX.Y.Z` tag only after every gate passes. If the tag exists, stop and report the existing tag SHA and current HEAD SHA. Verify the tag points at HEAD and the working tree is clean.

## Stop condition

Stop after local commits and the annotated tag. Do not push, publish to PyPI, create a GitHub release, or close issues without separate authorization or an explicitly authorized release workflow.
