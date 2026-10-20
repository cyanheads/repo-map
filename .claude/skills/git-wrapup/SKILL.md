---
name: git-wrapup
description: >
  Land repo-map working-tree changes on main as a stack of Conventional Commits,
  then a chore(release) commit and an annotated vX.Y.Z tag. Use only when the
  maintainer explicitly asks to commit, wrap up, or cut a release. Stops at
  local commits and the local tag; release-and-publish pushes, publishes to
  PyPI, creates the GitHub release, and closes issues.
metadata:
  author: cyanheads
  version: "2.0"
  audience: project
  type: workflow
---

# Git Wrap-up

repo-map releases straight from `main`: no release branches, no pull requests. This skill builds the commit stack and the tag locally. Pushing and publishing belong to `release-and-publish`.

## Was a commit requested?

Only an explicit request in the maintainer's own words counts: "commit this", "wrap up", "cut 0.15.0". "Fix this", "make the changes", and "ship it" ask for the work, not the commit. Without a request, stop at a reviewed working tree.

## Pre-commit gate

Every item holds before anything is staged:

- [ ] **Work is complete**: no half-finished unit, no TODO placeholder.
- [ ] **Nothing that shouldn't ship**: no `.env`, API keys, `.repo-map-cache.db`, `*_repo_map.md`, `.repo_map_structure.json`, `dist/`, or scratch files.
- [ ] **`make check` passes**, run raw and read in full: ruff check, ruff format, pyright, pytest, `uv lock --check`, and the changelog check. A pyright warning or a new pytest warning counts as red. Fix a failure even if it predates this session.
- [ ] **`uv run repo-map --help` runs**, which proves the entry point still imports.
- [ ] **Simplified**: run the `code-simplifier` skill first when the diff touches more than a line or two of `src/repo_map/`.
- [ ] **Analysis contract**: if the diff changes `SYSTEM_PROMPT`, the message content `_build_messages()` assembles, the request payload in `rate_limited_api_call()`, or the fields `_first_invalid_field()` validates, then `ANALYSIS_CONTRACT_VERSION` in `src/repo_map/llm_service.py` was bumped. Cached analyses are keyed on it, so a missed bump silently reuses answers to the old prompt.
- [ ] **Live check**: a prompt or request change was run once against a small public fixture (a scratch directory of a few files written for the check, or a small public repository). Never send a private repository to OpenRouter as a test.
- [ ] **Example output**: if the report format or the prompt changed, `examples/example.md` was regenerated. From the repo root, `uv run repo-map "$PWD" -y` writes `repo-map_repo_map.md`; replace the body of the outer four-backtick fence in `examples/example.md` with it, keeping the heading and fence lines. This spends real OpenRouter credit and is only ever run on this public repository.
- [ ] **Docs current**: `README.md`, `CLAUDE.md`, and `AGENTS.md` updated where the change made them wrong, and nowhere else.

## Steps

### 1. Review

```bash
git branch --show-current          # must print main
git status
git diff HEAD                      # staged and unstaged together
git ls-files --others --exclude-standard
git log "$(git describe --tags --abbrev=0)"..HEAD --oneline
```

On any branch other than `main`, or a detached HEAD, stop and report. Read every untracked file in full; no diff shows its content.

Uncommitted changes outside the work you were asked to land are candidates, not automatic exclusions. A complete, safe change idle for hours folds in as its own commit. A change you didn't make that was modified in the last few minutes (check mtimes), or that looks half-done, belongs to a writer who may still be working: leave it out and report it.

### 2. Group by concern

A fix and its test are one commit; two unrelated fixes are two commits. Order foundations first (a refactor before the feature that uses it) so every commit passes on its own.

**The file is the atomic boundary.** Never split one file's changes across commits, whether by `git add -p` or by editing the file between commits. A file serving two concerns ships whole with its dominant one.

Commit each group by pathspec:

```bash
git add <paths>
git commit --only <paths> -m "<subject>" -m "<body>"
```

`--only` commits exactly those paths, so anything staged earlier by a hook or another session can't ride along.

### 3. Messages

- Conventional Commits: `feat|fix|refactor|test|docs|build|chore(scope): subject`. Scopes follow the module: `cli`, `scanner`, `parser`, `llm`, `cache`, `report`, `config`, `deps`.
- Subject around 50 characters, with `(#N)` when it addresses an issue.
- **Every commit gets a body of one or two lines**: the why or the constraint, not a list of files or a walkthrough of the code.
- No closing keywords (`Fixes #N`); they close the issue on push, before the release exists. `release-and-publish` closes issues.
- No `Co-authored-by` or `Generated with` trailers, no marketing adjectives ("comprehensive", "robust", "enhanced"), no conversation references.
- Plain `-m` strings. A backtick inside a double-quoted `-m` runs as command substitution and vanishes from the message.

### 4. Version and changelog

The maintainer sets the version. When the request didn't name one, stop and ask, with a suggestion: patch for fixes and dependency updates, minor for a new flag, a changed output format, a cache change, or an analysis-contract bump.

1. `uv version X.Y.Z` sets `pyproject.toml` and updates the project's own entry in `uv.lock`.
2. Set the `placeholder` under `id: version` in `.github/ISSUE_TEMPLATE/bug_report.yml` to `"X.Y.Z"`.
3. Copy `changelog/template.md` to `changelog/<major.minor>.x/X.Y.Z.md` (for 0.15.2, `changelog/0.15.x/0.15.2.md`):
   - H1 `# X.Y.Z — YYYY-MM-DD` with today's date; the changelog builder reads the release date from it.
   - Frontmatter `summary` is required, double-quoted, one line, at most 350 characters: the headline, not a list. Set `breaking: true` when users must change how they invoke or configure repo-map. Set `security: true` only for a fix in repo-map's own code; a dependency CVE bump is recorded under Dependencies.
   - Sections in order Added, Changed, Deprecated, Removed, Fixed, Security, only those with entries. One sentence per bullet, leading with the symbol or flag. Issues linked with full URLs (`[#12](https://github.com/cyanheads/repo-map/issues/12)`).
   - A contract bump gets its own bullet: `` `ANALYSIS_CONTRACT_VERSION` <old> → <new>, so every cached analysis is reanalyzed once. ``
   - Dependency changes go in `## Dependencies`, split into `### Runtime` and `### Dev`, one row per package: `` `aiohttp` `3.14.3` → `3.14.5` ``.
   - Delete the template's comment block and every unused section.
4. `make changelog` regenerates `CHANGELOG.md`. Never edit `CHANGELOG.md` by hand.
5. `rg -nF '<old version>'` and update anything outside historical changelog entries that still names it.
6. Re-run `make check`.

The release commit goes on top of the work stack and carries only `pyproject.toml`, `uv.lock`, the new changelog file, `CHANGELOG.md`, and `bug_report.yml`:

```bash
git add <those paths>
git commit --only <those paths> -m "chore(release): X.Y.Z — <theme>" -m "<one-line body>"
```

### 5. Annotated tag

If `vX.Y.Z` already exists, stop and report its SHA and the current HEAD. Never move or delete a tag without the maintainer's say-so.

Write the message with the file-writing tool to a file outside the repository, then:

```bash
git tag -a vX.Y.Z -F <message-file> --cleanup=whitespace
```

Delete the message file afterwards. Never pass markdown as a shell argument: backticks inside double quotes run as command substitution, and an apostrophe ends a single-quoted string. `--cleanup=whitespace` keeps `#`-leading lines that the default cleanup would strip as comments.

The tag body becomes the GitHub release notes. It is a headline digest of the changelog entry, never a copy:

```
<one theme line, at most ~60 characters, no version, no semicolons>

- <notable user-facing change> (#N)
- <notable user-facing change>
- <one grouped line for minor or internal changes>
- deps: <one line naming only a major bump or a security fix; the arrows live in the changelog>

[CHANGELOG vX.Y.Z](https://github.com/cyanheads/repo-map/blob/main/changelog/<major.minor>.x/X.Y.Z.md)
```

Flat bullets only: no section headers, no gates line, no preamble. Every bullet must trace to the changelog entry. A small patch can be a theme line, two bullets, and the link.

### 6. Verify and stop

```bash
git log --oneline -8
git status                                        # clean apart from reported exclusions
test "$(git rev-parse 'vX.Y.Z^{commit}')" = "$(git rev-parse HEAD)" && echo tag-at-head
git config --get tag.gpgSign                      # true = signing enabled
git tag -l vX.Y.Z --format='%(if)%(contents:signature)%(then)signed%(else)unsigned%(end)'
```

With signing enabled, the last line must print `signed`. If it prints `unsigned`, or the tag message breaks the format above, delete the tag with `git tag -d vX.Y.Z` and recreate it. This is the one tag deletion that needs no sign-off: the tag is local, seconds old, and yours.

Then stop. Report the commit stack (short SHA and subject), the tag, the issues the release fixes (`release-and-publish` closes them), and any changes left out. Don't comment on issues here: the SHAs exist only locally until the push.

## Constraints

- Never `git stash`, never worktrees, never `--no-verify`.
- No destructive commands (`reset --hard`, `restore .`, `checkout -- .`, `clean -f`) without an explicit request.
- No force-push, ever. No push at all from this skill.
- When a hook fails, the commit didn't happen: fix the cause and commit again rather than `--amend`.
- Never edit a file to shape how its diff looks; rethink the grouping instead.

## Checklist

- [ ] Commit explicitly requested; on `main`
- [ ] Diff and every untracked file read; other writers' fresh changes left out and reported
- [ ] `make check` and `uv run repo-map --help` green; contract, live-check, example, and docs items satisfied
- [ ] Grouped by concern, foundations first, no file split; every commit has a one- or two-line body
- [ ] Version set with `uv version`; `bug_report.yml` placeholder updated; changelog file authored; `make changelog` run
- [ ] `chore(release): X.Y.Z — <theme>` on top
- [ ] Annotated `vX.Y.Z` at HEAD with a digest message, signed when signing is on
- [ ] Nothing pushed; report delivered
