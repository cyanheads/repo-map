---
name: maintenance
description: >
  Update and verify repo-map's dependencies with uv: survey what is outdated,
  upgrade the lockfile, raise lower bounds, read what changed upstream, adopt
  what the code needs, verify, and record the arrows in the changelog. Use for
  "update dependencies", "run maintenance", or after a manual uv lock --upgrade
  to review its impact. Hands off to git-wrapup for the commit and release.
metadata:
  author: cyanheads
  version: "2.0"
  audience: project
  type: workflow
---

# Maintenance

## Entry modes

| Mode | Starting point | First step |
|:--|:--|:--|
| **A, full flow** | `uv.lock` is current; an update is wanted | Step 1 |
| **B, post-update review** | Someone already ran `uv lock --upgrade` or edited constraints in `pyproject.toml` | The lower-bound raise at the end of Step 2 if it was skipped, then Step 3 with `git diff uv.lock pyproject.toml` as input |

## Steps

### 1. Survey (Mode A)

```bash
uv tree --outdated --depth 1
```

Rows marked `(latest: vX)` have a newer release. `pyproject.toml` declares lower bounds only (`>=`), so an upgrade can cross a major; note which ones do, because Step 3 has to find their breaking changes. For a pre-1.0 package, treat a minor bump as a major.

`[tool.uv] exclude-newer = "3 days"` in `pyproject.toml` is a supply-chain cooldown: the survey and the resolver both ignore releases younger than three days, so a just-published version is held back, not missing. Never lower or remove it. To take an urgent security fix sooner, add only that package to the same table, `exclude-newer-package = { <pkg> = "<RFC 3339 timestamp>" }`, re-lock, and drop the entry once the release has aged past the window. Don't use the `--exclude-newer-package` CLI flag: it records the override in `uv.lock`, and the next `uv lock --check` fails.

### 2. Upgrade (Mode A)

```bash
uv lock --upgrade
uv sync
```

Then raise each direct dependency's lower bound to its newly locked version, so the published metadata never admits a version the code wasn't tested against: `uv add '<pkg>>=<version>'` for runtime, `uv add --dev '<pkg>>=<version>'` for the dev group. A bare `uv add <pkg>` leaves an existing constraint untouched.

Capture the version deltas from `git diff uv.lock pyproject.toml`; they feed Steps 3 and 6.

### 3. Read what changed

Investigate only packages repo-map imports or runs. Transitive bumps need no reading unless something fails.

| Package | Used in | Watch for |
|:--|:--|:--|
| `aiohttp` | `llm_service.py` | Session, timeout, and retry semantics on the OpenRouter client |
| `certifi` | `llm_service.py` | Nothing in practice; releases refresh the CA bundle |
| `pydantic`, `pydantic-settings` | `config.py` | Validation errors, `.env` loading, `SettingsConfigDict` options |
| `pathspec` | `file_scanner.py` | `.gitignore` pattern matching |
| `tqdm` | `cli_handler.py`, `logging_utils.py` | Progress bar and logging handler behavior |
| `ruff` | lint and format gate | New rules or formatter changes |
| `pyright` | type gate | New diagnostics |
| `pytest` | test gate | Deprecated hooks, changed defaults |

For each one: find the source repository from `https://pypi.org/project/<pkg>/`, read the release notes between the old and new versions, and search `src/repo_map/` for the APIs it uses. Record what changed, whether it affects repo-map, and the action item.

### 4. Adopt

- **Breaking changes**: fix the call sites.
- **Deprecations**: migrate now, while the context is fresh. A new `DeprecationWarning` in pytest's warnings summary is an adoption item, not noise.
- **New ruff rules** that flag existing code: fix the code; don't add an ignore.
- **Formatter drift**: `make format`. A large reformat becomes its own commit, separate from code changes.
- **New pyright diagnostics**: fix the types. A `# type: ignore` is acceptable only for a real third-party typing gap, with the reason on the same line.
- **A library API that supersedes local code**: adopt it when it is a clear win (less code, a dependency removed, a bug class gone). Skip when marginal.
- **New features with no matching call site**: note them; don't refactor speculatively.

An upgrade that changes request behavior (`aiohttp`) or settings loading (`pydantic-settings`) gets the live check in Step 5. One that changes what is sent to the model is an analysis-contract change: bump `ANALYSIS_CONTRACT_VERSION` per the `git-wrapup` skill.

### 5. Verify

```bash
make check
uv run repo-map --help
```

`make check` runs ruff check, ruff format, pyright, pytest, `uv lock --check`, and the changelog check. Read the output in full; a warning is red.

When a runtime dependency changed, run the CLI once against a small public fixture, a scratch directory of a few files written for the check:

```bash
uv run repo-map <fixture-dir> -y
```

This spends OpenRouter credit, so keep the fixture tiny. Never point it at a private repository. Watch for new exceptions, broken parsing, and failed requests; fix and re-run until clean. Dev-only updates skip the live run.

### 6. Record

Dependency arrows belong in the release's changelog entry under `## Dependencies`, split into `### Runtime` and `### Dev`, one row per package:

```markdown
- `aiohttp` `3.14.3` → `3.14.5`
- `multidict` `6.7.1` → `7.1.0` (transitive via aiohttp)
```

If the upcoming release's file, `changelog/<major.minor>.x/X.Y.Z.md`, already exists, add the rows there and run `make changelog`. Otherwise carry them in the Step 7 summary; `git-wrapup` creates the file once the version is set. A dependency CVE fix gets its advisory ID on the row and leaves `security: false`, which is reserved for fixes to repo-map's own code.

Update `CLAUDE.md`, `AGENTS.md`, or `README.md` only where a command, setting, or default changed. The version bump, commits, and tag belong to the `git-wrapup` skill; a dependency-only update suggests a patch release.

### 7. Summary

Report, numbered:

1. **Updated packages**: version arrows, majors marked
2. **Breaking changes handled**: call sites fixed
3. **Adoptions**: APIs swapped in, helpers removed
4. **Records**: changelog rows (written, or listed here for `git-wrapup`) and doc edits
5. **Open decisions**: majors held back, close-call adoptions, anything blocked by the cooldown
6. **Status**: `make check`, `repo-map --help`, live-run result

## Checklist

- [ ] Upgrade applied and lower bounds raised (Mode A), or the existing upgrade diff read (Mode B)
- [ ] `exclude-newer` cooldown untouched; any per-package override lives in `pyproject.toml`
- [ ] Release notes read for every directly used package that moved
- [ ] Breaking changes and deprecations handled; no new ignores
- [ ] `make check` and `uv run repo-map --help` green
- [ ] Live run on a public fixture when a runtime dependency changed
- [ ] `## Dependencies` rows written into the release's changelog file, or carried in the summary
- [ ] Numbered summary delivered
