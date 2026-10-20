## Example Generated Output For repo-map

Generated:
````markdown
# Repository Map

```markdown
/ (repo-map)
├── .agents/
├── .claude/
│   └── skills/
│       ├── code-simplifier/
│       │   └── SKILL.md (Markdown)
│       │       ├── Description: Agent skill definition for a behavior-preserving cleanup pass over repo-map. It sets scope (uncommitted diff or named modules), a review checklist, Python 3.12 rewrites, the contracts that must never change, and the report format.
│       │       ├── Developer Consideration: The 'Never a simplification' list hard-codes repo-map contracts (ANALYSIS_CONTRACT_VERSION, SYSTEM_PROMPT, the .repo-map-cache.db schema, CLI flags, settings names). If those identifiers are renamed or the contracts change, this skill must be updated or it will mislead the agent.
│       │       ├── Maintenance Flag: Volatile
│       │       ├── Architectural Role: Tooling
│       │       ├── Critical Dependencies:
│       │       ├──   - make check: Quality gate run before and after edits to confirm behavior is preserved.
│       │       ├──   - CLAUDE.md: Source of repo conventions the agent must read before editing.
│       │       └──   - src/repo_map/llm_service.py: Holds the analysis contract and failure semantics the skill protects.
│       ├── git-wrapup/
│       │   └── SKILL.md (Markdown)
│       │       ├── Description: Agent skill defining the git-wrapup workflow: a pre-commit gate, grouping changes into Conventional Commits on main, setting the version with `uv version`, authoring a changelog entry, a chore(release) commit, and a local annotated vX.Y.Z tag. It never pushes.
│       │       ├── Developer Consideration: The skill hardcodes repo-specific names (`ANALYSIS_CONTRACT_VERSION` in llm_service.py, `make check`, `make changelog`, the bug_report.yml placeholder, the changelog path scheme). Renaming any of them without editing this file makes the instructions wrong.
│       │       ├── Maintenance Flag: Volatile
│       │       ├── Architectural Role: Tooling
│       │       ├── Critical Dependencies:
│       │       ├──   - release-and-publish: Companion skill that pushes, publishes to PyPI, creates the GitHub release and closes issues.
│       │       ├──   - code-simplifier: Skill run before committing when the diff touches src/repo_map/ beyond a line or two.
│       │       ├──   - make check: Pre-commit gate: ruff, pyright, pytest, uv lock check and changelog check.
│       │       └──   - uv: `uv version` sets the release version; `uv run repo-map` is used for the smoke check and example regeneration.
│       ├── maintenance/
│       │   └── SKILL.md (Markdown)
│       │       ├── Description: Claude agent skill defining the dependency-maintenance workflow for repo-map using uv: survey outdated packages, run `uv lock --upgrade`, raise lower bounds, read upstream release notes, adopt changes, verify with `make check`, and record version arrows in the changelog.
│       │       ├── Developer Consideration: The `[tool.uv] exclude-newer = "3 days"` cooldown must stay; urgent overrides go in `exclude-newer-package` in pyproject.toml, never via the `--exclude-newer-package` CLI flag, which writes to uv.lock and breaks `uv lock --check`.
│       │       ├── Maintenance Flag: Volatile
│       │       ├── Architectural Role: Tooling
│       │       ├── Critical Dependencies:
│       │       ├──   - uv: Drives survey, lock upgrade, sync, and lower-bound raising via `uv add`.
│       │       ├──   - make check: Verification gate: ruff, pyright, pytest, `uv lock --check`, changelog check.
│       │       ├──   - git-wrapup: Companion skill that owns version bump, commits, tags, and ANALYSIS_CONTRACT_VERSION bumps.
│       │       └──   - pyproject.toml: Holds lower-bound constraints and the exclude-newer supply-chain cooldown.
│       ├── release-and-publish/
│       │   └── SKILL.md (Markdown)
│       │       ├── Description: Agent skill runbook that publishes a repo-map release already committed and tagged by git-wrapup. It re-runs `make check`, builds into dist/X.Y.Z, pushes main and the vX.Y.Z tag, runs `uv publish`, verifies the install, creates the GitHub release, and closes fixed issues.
│       │       ├── Developer Consideration: Step order and idempotency matter: build into dist/X.Y.Z (never publish dist/*), reuse the existing build if PyPI already returns 200, and push main before the tag. Re-running must skip steps that already succeeded, and tags and PyPI releases are never force-pushed, deleted, or yanked.
│       │       ├── Maintenance Flag: Volatile
│       │       ├── Architectural Role: Tooling
│       │       ├── Critical Dependencies:
│       │       ├──   - uv: Builds distributions, publishes to PyPI, and runs uvx smoke tests of the wheel and the published version.
│       │       ├──   - gh: Creates the GitHub release from the tag and comments on and closes fixed issues.
│       │       ├──   - make check: Release gate that must pass cleanly before anything is built or pushed.
│       │       ├──   - git-wrapup: Upstream skill that creates the release commit, changelog entry, and annotated tag this skill requires.
│       │       └──   - security: macOS Keychain CLI that supplies the PyPI token inline to uv publish.
│       └── report-issue-local/
│           └── SKILL.md (Markdown)
│               ├── Description: Agent skill workflow for filing bug, feature, and documentation issues against cyanheads/repo-map with gh: duplicate search, public-fixture reproduction, redaction of private analyzed-repo content, title/label/assignee conventions, body templates, and follow-up comments.
│               ├── Developer Consideration: Bodies must go through `--body-file` (written outside the repo, then deleted), since a shell-string body loses backticked spans to command substitution. Headings must stay in sync with `.github/ISSUE_TEMPLATE/bug_report.yml` and `feature_request.yml`.
│               ├── Maintenance Flag: Volatile
│               ├── Architectural Role: Tooling
│               ├── Critical Dependencies:
│               ├──   - gh: GitHub CLI used to search, view, create, and comment on issues in cyanheads/repo-map.
│               ├──   - .github/ISSUE_TEMPLATE/bug_report.yml: Defines the bug-report headings and order the skill's body template mirrors.
│               ├──   - .github/ISSUE_TEMPLATE/feature_request.yml: Defines required feature-request headings the skill's body must follow.
│               └──   - release-and-publish: Sibling skill that closes issues after the fix is published to PyPI.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml (YAML)
│   │   │   ├── Description: GitHub issue form for bug reports on repo-map. It labels issues "bug", auto-assigns cyanheads, and requires version, Python version, OS, description, reproduction, actual and expected behavior, with optional additional context.
│   │   │   ├── Developer Consideration: The version input placeholder ("0.14.0") is hardcoded and will go stale as releases ship; update it during release.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Tooling
│   │   ├── config.yml (YAML)
│   │   │   ├── Description: GitHub issue-template chooser configuration that keeps blank issues enabled and adds a contact link to OpenRouter support for account, billing, and provider-availability problems that do not reproduce in repo-map.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Tooling
│   │   └── feature_request.yml (YAML)
│   │       ├── Description: GitHub issue form for feature requests against repo-map. It auto-applies the "enhancement" label, assigns "cyanheads", and collects required Use case and Proposal textareas plus optional Alternatives considered and Additional context.
│   │       ├── Developer Consideration: The hardcoded assignee "cyanheads" must be a user with repository access, or GitHub skips the assignment. The field ids (use-case, proposal, alternatives, context) key the submitted issue body, so renaming them breaks any automation that parses it.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Tooling
│   ├── workflows/
│   │   └── codeql.yml (YAML)
│   │       ├── Description: GitHub Actions workflow that runs CodeQL static analysis on the actions and python languages. It triggers on pushes and pull requests to main and on a weekly Monday cron, and uploads results under the /language:<language> category.
│   │       ├── Developer Consideration: GitHub's repo-level CodeQL "default setup" must be disabled or this workflow's uploads conflict. The `category: /language:${{ matrix.language }}` value must match default setup's category so old alerts close once the code is fixed.
│   │       ├── Maintenance Flag: Stable
│   │       ├── Architectural Role: Tooling
│   │       ├── Critical Dependencies:
│   │       ├──   - github/codeql-action/init@v4: Initializes CodeQL for each matrix language with build-mode none.
│   │       ├──   - github/codeql-action/analyze@v4: Runs the analysis and uploads SARIF results under the language category.
│   │       └──   - actions/checkout@v7: Checks out the repository source for analysis.
│   ├── CODE_OF_CONDUCT.md (Markdown)
│   │   ├── Description: Community code of conduct for the repository. It lists expected and unacceptable behavior, sets scope across issues, PRs, and commits, makes agent operators accountable for what their agents post, and gives a private email for reports and enforcement.
│   │   ├── Developer Consideration: The Scope section links to ./CONTRIBUTING.md as the rule for agents that file issues; if that file is moved or renamed, the link breaks. The reporting email is hardcoded and must be updated here if maintainers change.
│   │   ├── Maintenance Flag: Stable
│   │   └── Architectural Role: Other
│   ├── CONTRIBUTING.md (Markdown)
│   │   ├── Description: Contributor guide for repo-map that routes bugs and feature requests through the GitHub issue chooser forms. It lists pre-filing checks (latest release, duplicate search, redaction), the details an actionable report needs, agent triage guidance, and the private security reporting route.
│   │   ├── Developer Consideration: The relative links to ../.claude/skills/report-issue-local/SKILL.md and SECURITY.md break if those files or this document move, and the issue-form names (Bug Report, Feature Request) must match .github/ISSUE_TEMPLATE.
│   │   ├── Maintenance Flag: Stable
│   │   ├── Architectural Role: Other
│   │   ├── Critical Dependencies:
│   │   ├──   - .claude/skills/report-issue-local/SKILL.md: Agent workflow that this guide tells agents to read before filing issues.
│   │   └──   - .github/SECURITY.md: Destination for private vulnerability reports instead of public issues.
│   └── SECURITY.md (Markdown)
│       ├── Description: Security policy for repo-map: only the latest release gets fixes, and vulnerabilities go through GitHub private reporting or email rather than public issues. It defines in-scope reports as leaks of files the README says stay local, OpenRouter API key leaks or redirection, and analyzed-repo content steering repo-map.
│       ├── Developer Consideration: The in-scope definition depends on the README's "What gets sent" anchor (../README.md#what-gets-sent); renaming that heading or changing the upload rules silently changes what counts as a vulnerability.
│       ├── Maintenance Flag: Stable
│       └── Architectural Role: Tooling
├── changelog/
│   ├── 0.0.x/
│   │   └── 0.0.1.md (Markdown)
│   │       ├── Description: Changelog entry for release 0.0.1 (2025-09-28). Front matter carries a summary of "Initial commit", and the body lists a single "Added" item. It is one of the per-version files under changelog/.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.1.x/
│   │   └── 0.1.0.md (Markdown)
│   │       ├── Description: Changelog entry for release 0.1.0 (2025-09-29). It records the move from setuptools to Poetry, the split of a monolithic script into modules, SQLite caching, and multi-language code parsing. The front-matter summary feeds the changelog build.
│   │       ├── Developer Consideration: The YAML front-matter `summary` field is likely read by scripts/build-changelog.ts when it assembles CHANGELOG.md, so keep the front-matter format and the `# version — date` heading consistent with changelog/template.md.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.10.x/
│   │   ├── 0.10.0.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.10.0, dated 2026-08-13. They cover a directory-only ignore pattern fix and two security changes: the OpenRouter URL and headers become fixed constants, and default exclusions grow to include .envrc, *.tfvars, *.tfstate, *.ini, *.conf and *.cfg.
│   │   │   ├── Developer Consideration: The front matter (summary, breaking: true, security: true) is likely read by scripts/build-changelog.ts, so keep those keys and their format intact. Users who relied on the removed OPENROUTER_API_URL, HTTP_REFERER or APP_NAME settings need the breaking note to stay accurate.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   ├── 0.10.1.md (Markdown)
│   │   │   ├── Description: Changelog entry for release 0.10.1 (2026-08-13). It records two tree-rendering fixes: indentation prefixes now track open branches by depth (#18), and model-authored text is flattened to one line so it cannot split tree lines or close the Markdown report's code fence (#22).
│   │   │   ├── Developer Consideration: The YAML front matter `summary` is probably read by scripts/build-changelog.ts when it builds CHANGELOG.md, so keep the front matter and the `# version — date` heading format intact.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.10.2.md (Markdown)
│   │       ├── Description: Release notes for repo-map 0.10.2 (2026-08-13), listing parser fixes: async Python functions, JavaScript class scope closing, leading-header module docstring extraction for comment-based languages, and broader import and method pattern matching. Cites issues #12, #19 and #20.
│   │       ├── Developer Consideration: The front-matter `summary` line is likely consumed by scripts/build-changelog.ts to assemble CHANGELOG.md, so keep the YAML front matter and the `# version — date` heading format intact.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.11.x/
│   │   ├── 0.11.0.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.11.0 (2026-08-13). They cover two fixes: invalid configuration and corrupt or unwritable cache files now exit 1 with one line via ConfigurationError and CacheError, and the upload prompt discloses what is sent and requires an explicit y/yes.
│   │   │   ├── Developer Consideration: The YAML front-matter `summary` is probably consumed by scripts/build-changelog.ts to assemble CHANGELOG.md, so keep the front-matter keys and the `# version — date` heading format intact when editing.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.11.1.md (Markdown)
│   │       ├── Description: Release notes for repo-map 0.11.1 (2026-08-13). They cover three fixes: class regexes no longer match prose in comments and strings, Java and C# class scope now closes at the closing brace, and a malformed OpenRouter response fails only its own file's enhancement.
│   │       ├── Developer Consideration: The front-matter `summary` and the `# version — date` heading are likely parsed by scripts/build-changelog.ts to build CHANGELOG.md, so keep that format and the template.md structure when editing.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.12.x/
│   │   └── 0.12.0.md (Markdown)
│   │       ├── Description: Release notes for repo-map 0.12.0. They cover the new `rel_path` field on scanned files and in `.repo_map_structure.json`, the cache move from absolute to repository-relative keys, and the end-of-run `prune_cache` pass that deletes stale rows.
│   │       ├── Developer Consideration: The front-matter `summary` and the `# 0.12.0 — date` heading are likely read by scripts/build-changelog.ts when it assembles CHANGELOG.md, so keep that format unchanged when editing.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.13.x/
│   │   ├── 0.13.0.md (Markdown)
│   │   │   ├── Description: Release notes for 0.13.0 describing cache identity fingerprinting: cached analyses are reused only when file hash, model, and ANALYSIS_CONTRACT_VERSION all match. Also documents that legacy rows with NULL model/contract_version columns count as misses after the ALTER TABLE upgrade.
│   │   │   ├── Developer Consideration: ANALYSIS_CONTRACT_VERSION in llm_service.py is bumped by hand; changing SYSTEM_PROMPT or the validated field set without incrementing it leaves stale cached analyses being served as hits.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.13.1.md (Markdown)
│   │       ├── Description: Release note for 0.13.1 (2026-08-13). It records that the `check` command now runs `scripts/sync_skills.py --check`, a read-only content-hash comparison of skill mirrors that fails on a missing or drifted file and reports orphans without failing.
│   │       ├── Developer Consideration: The note names `poetry run python scripts.py`, `scripts/sync_skills.py` and a `skills/` directory, none of which appear in the repository layout (which lists `.claude/skills/` and `.agents/`). Verify those paths against the code before relying on this entry.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.14.x/
│   │   └── 0.14.0.md (Markdown)
│   │       ├── Description: Release notes for repo-map 0.14.0 (2026-10-20): rewritten analysis prompt with ANALYSIS_CONTRACT_VERSION 2, a default model change to anthropic/claude-sonnet-5.5, prompt-cached requests, the Poetry-to-uv migration, new community and CodeQL files, a retry backoff fix, and dependency bumps.
│   │       ├── Developer Consideration: The front matter (`summary`, `security: true`) is likely read by scripts/build-changelog.ts to generate CHANGELOG.md, so keep its keys and the `# version — date` heading format intact. Edit this file and rerun `make changelog` rather than editing the index.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.2.x/
│   │   └── 0.2.0.md (Markdown)
│   │       ├── Description: Release notes for version 0.2.0 (2025-09-29). They record the RepoMapApp refactor of main.py, TqdmLoggingHandler logging, more robust ignore-pattern handling in file_scanner.py, the APIRateLimiter in llm_service.py, and the new AGENTS.md.
│   │       ├── Developer Consideration: The front-matter `summary` is probably read by scripts/build-changelog.ts to build CHANGELOG.md, so keep the YAML keys and the `# 0.2.0 — date` heading format consistent with changelog/template.md.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.3.x/
│   │   └── 0.3.0.md (Markdown)
│   │       ├── Description: Changelog entry for release 0.3.0 (2025-09-29). Records one change: README.md was updated for the current default model name, .gitignore handling, and the description of main.py. The front-matter summary feeds the changelog build.
│   │       ├── Developer Consideration: The front-matter `summary` field and the `# 0.3.0 — date` heading follow the changelog/template.md format that scripts/build-changelog.ts likely consumes to assemble CHANGELOG.md; changing their shape could break that aggregation.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.4.x/
│   │   └── 0.4.0.md (Markdown)
│   │       ├── Description: Release notes for version 0.4.0 (2025-09-29). They record the per-file LLM analysis fields, the extended cache schema, and the changes to file_scanner.py, llm_service.py, main.py and cache_manager.py. A front-matter summary is included.
│   │       ├── Developer Consideration: The front-matter `summary` field is likely read by scripts/build-changelog.ts when it assembles CHANGELOG.md, so keep the YAML header and the `# version — date` heading format consistent with changelog/template.md.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.5.x/
│   │   └── 0.5.0.md (Markdown)
│   │       ├── Description: Changelog entry for release 0.5.0 (2025-09-29). Records the split of main.py into cli_handler.py, logging_utils.py, and report_generator.py, the new examples/example.md, a README refresh, and the version bump.
│   │       ├── Developer Consideration: The front-matter `summary` field and the dated heading follow changelog/template.md and are likely consumed by scripts/build-changelog.ts when it assembles CHANGELOG.md; keep that structure intact when editing.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.7.x/
│   │   ├── 0.7.0.md (Markdown)
│   │   │   ├── Description: Release notes for version 0.7.0 (2026-03-08). They record the default LLM model change to google/gemini-3-flash-preview, the added CLAUDE.md, the ruff lint config fix, Black formatting, `X | None` type hints, a .gitignore entry and dependency bumps.
│   │   │   ├── Developer Consideration: The front-matter `summary` line is probably read by scripts/build-changelog.ts to build CHANGELOG.md, so keep the `---` block and `summary` key intact when editing.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.7.1.md (Markdown)
│   │       ├── Description: Release notes for version 0.7.1 (2026-04-25). They cover the pathspec GitIgnoreSpec migration in file_scanner.py, the new maintenance skill with the sync-skills script, the deferred tree-sitter upgrade, and runtime and dev dependency bumps.
│   │       ├── Developer Consideration: The front-matter `summary` is likely consumed by scripts/build-changelog.ts to assemble CHANGELOG.md, so keep that key and the heading format intact. The notes also mention Poetry and scripts/sync_skills.py, which do not appear in the current layout (pyproject.toml and uv.lock suggest uv).
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.8.x/
│   │   ├── 0.8.0.md (Markdown)
│   │   │   ├── Description: Changelog entry for release 0.8.0 (2026-04-25), a breaking release. It records the switch of the LLM contract to a JSON object with a closed architectural_role enum, the new default model anthropic/claude-sonnet-4.6, and the removal of code_quality_score.
│   │   │   ├── Developer Consideration: The entry is marked breaking: true and has front matter (summary, breaking) that scripts/build-changelog.ts likely reads, so keep that front matter format intact. Existing caches keep the dropped code_quality_score column, so code must not depend on it.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   ├── 0.8.1.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.8.1 (2026-08-13). They cover the 25-test baseline suite with eight strict xfail regressions for issues #5–#12, the repaired scripts.py dispatcher, project-local skill workflows, removal of tree-sitter dependencies, and runtime and dev dependency bumps.
│   │   │   ├── Developer Consideration: The YAML front-matter `summary` is probably read by scripts/build-changelog.ts when it assembles CHANGELOG.md, so keep the front-matter keys and the `# version — date` heading format consistent with changelog/template.md.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   ├── 0.8.2.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.8.2 (2026-08-13) with two scanner fixes: symlinks are skipped before sorting or classification (#9), and language detection matches exact filenames, then longest suffix (#5). Includes a front-matter summary used by the changelog build.
│   │   │   ├── Developer Consideration: The front-matter `summary` field and the `# 0.8.2 — date` heading follow the changelog/template.md format that scripts/build-changelog.ts likely aggregates into CHANGELOG.md; changing their shape can break the build.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.8.3.md (Markdown)
│   │       ├── Description: Release notes for version 0.8.3 (2026-08-13). They record two fixes: file analyses are scheduled together and consumed in completion order within the request bound, and HTTP 429 backoff now runs after the concurrency permit is released.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.9.x/
│   │   ├── 0.9.0.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.9.0 (2026-08-13). They record source-grounded LLM analysis (full file source sent as delimited untrusted data), validation in parse_llm_response, scanner-driven enrichment eligibility, a Ruff line-length change to 88, and removal of get_constants.
│   │   │   ├── Developer Consideration: The summary front matter and the Added/Changed/Removed headings are likely consumed by scripts/build-changelog.ts to build CHANGELOG.md, so keep that format and the template.md structure when editing.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.9.1.md (Markdown)
│   │       ├── Description: Changelog entry for release 0.9.1 (2026-08-13). It records one transitive dev dependency bump, platformdirs 4.11.2 to 4.11.3, and states that repo-map does not import platformdirs, so no code changed.
│   │       ├── Developer Consideration: The front-matter `summary` field is likely read by scripts/build-changelog.ts when it builds CHANGELOG.md, so keep the `summary:` key and the `# version — date` heading format consistent with changelog/template.md.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   └── template.md (Markdown)
│       ├── Description: Format reference for per-release changelog entries. Defines the YAML front matter (summary, breaking, security, optional agent-notes), the H1 `# <version> — YYYY-MM-DD` heading, Keep a Changelog sections, and an authoring guide. It is copied to `changelog/<major.minor>.x/<version>.md`.
│       ├── Developer Consideration: Copy the file, never edit it in place. The `summary` value must stay quoted (unquoted `: ` breaks GitHub's strict YAML parser), and `security: true` is only for vulnerabilities in this project's own code. Dependency CVE bumps go under `## Dependencies`.
│       ├── Maintenance Flag: Stable
│       ├── Architectural Role: Tooling
│       ├── Critical Dependencies:
│       ├──   - scripts/build-changelog.ts: Presumed consumer that rolls per-version files into CHANGELOG.md using the breaking/security/summary front matter fields.
│       └──   - git-wrapup: Skill whose step 5 defines the annotated tag body format referenced by the guide.
├── examples/
│   └── example.md (Markdown)
│       ├── Description: Static sample of repo-map's Markdown report: a nested fenced ASCII tree of the repo-map repository in which each file lists Description, Developer Consideration, Maintenance Flag, Architectural Role, Critical Dependencies and optional Refactoring and Security notes. It shows readers the output format.
│       ├── Developer Consideration: This is a hand-kept snapshot inside a four-backtick fence wrapping a three-backtick fence, not live tool output. Its tree already differs from the real repository and report format, so regenerate it with `uv run repo-map` when the format or layout changes.
│       ├── Maintenance Flag: Volatile
│       └── Architectural Role: Other
├── scripts/
│   └── build-changelog.ts (TypeScript)
│       ├── Description: Generates CHANGELOG.md as an index from per-version files in changelog/<major.minor>.x/. Each file's YAML frontmatter (summary, breaking, security) and H1 date become a header link plus one-line summary, sorted by semver descending. A --check mode exits 1 when CHANGELOG.md has drifted.
│       ├── Developer Consideration: Frontmatter is parsed with regexes, not a YAML library. Summaries containing ': ' must be quoted and capped at 350 chars, and each file needs an H1 like '# X.Y.Z — YYYY-MM-DD', or buildRollup() throws. Paths resolve against process.cwd(), so run it from the repo root.
│       ├── Maintenance Flag: Stable
│       ├── Architectural Role: Tooling
│       ├── Refactoring Suggestions: compareSemverDesc() compares prerelease strings lexicographically, so rc.10 sorts below rc.2. Compare dot-separated prerelease identifiers numerically, as semver specifies, to fix the ordering of releases.
│       ├── Critical Dependencies:
│       ├──   - node:fs: Reads the changelog tree and per-version files, and writes CHANGELOG.md.
│       └──   - node:path: resolve() builds absolute paths for the changelog directory and output file.
├── src/
│   └── repo_map/
│       ├── __init__.py (Python)
│       │   ├── Description: Package marker for repo_map containing only a module docstring that describes the tool as producing repository trees with LLM-authored file documentation. It defines no code and re-exports nothing.
│       │   ├── Maintenance Flag: Stable
│       │   └── Architectural Role: Other
│       ├── cache_manager.py (Python)
│       │   ├── Description: SQLite cache for per-file LLM analyses. load_cache() opens or creates .repo-map-cache.db in the repo root and migrates legacy schemas. relative_cache_key() builds portable keys, and prune_cache() deletes rows for files no longer scanned.
│       │   ├── Developer Consideration: Cache keys must come from relative_cache_key() (forward-slash relative paths). Legacy absolute-path rows never match and are only removed by prune_cache(), never rewritten. A NULL model or contract_version in a migrated row makes it a miss.
│       │   ├── Maintenance Flag: Stable
│       │   ├── Architectural Role: Persistence
│       │   ├── Refactoring Suggestions: Fold the new_columns dict in _apply_schema into the CREATE TABLE definition, so the column list is declared once. The migration loop would then need only the columns added after the original schema, which removes the risk of the two lists drifting apart.
│       │   ├── Critical Dependencies:
│       │   └──   - sqlite3: Provides the on-disk cache database, schema creation, and row deletion.
│       ├── cli_handler.py (Python)
│       │   ├── Description: CLI orchestration for repo-map via RepoMapApp: parses arguments, checks the API key, asks for upload consent, opens the SQLite cache, scans the repo, runs concurrent LLM analysis, updates and prunes cache rows, and writes JSON and Markdown maps.
│       │   ├── Developer Consideration: Cache reuse in _get_files_to_process() requires hash, model and ANALYSIS_CONTRACT_VERSION to all match the row, and _update_cache_for_file() reads file_data['model'] and ['contract_version'] without a fallback. Changing the cache columns or what get_llm_descriptions() stamps onto a file breaks both.
│       │   ├── Maintenance Flag: Volatile
│       │   ├── Architectural Role: Entrypoint
│       │   ├── Refactoring Suggestions: The cache schema is spread across the raw SQL in _get_files_to_process() and _update_cache_for_file() here, while load_cache() lives in cache_manager. Moving these queries into cache_manager would keep the column list and the staleness rule next to the schema.
│       │   ├── Security Assessment: DISCLOSURE_PROMPT says the full text of each eligible file is sent to OpenRouter, and `-y/--yes` skips the prompt. Automation using `-y` can upload private source without review. Ignore rules come from the root .gitignore only, so secret files not listed there may be uploaded. Mitigation: keep the consent prompt by default and document that `-y` bypasses it.
│       │   ├── Critical Dependencies:
│       │   ├──   - repo_map.llm_service: Provides get_llm_descriptions, AnalysisOutcome, ANALYSIS_CONTRACT_VERSION and the concurrency limiter used for analysis.
│       │   ├──   - repo_map.cache_manager: Provides load_cache, prune_cache and CacheError for the SQLite analysis cache.
│       │   ├──   - repo_map.file_scanner: summarize_repo builds the file structure, including source_eligible, hash and rel_path.
│       │   ├──   - repo_map.report_generator: print_tree, save_json_map and save_markdown_map produce the outputs.
│       │   ├──   - repo_map.config: settings supplies the API key check, default model name and default concurrency.
│       │   └──   - tqdm: Shows the progress bar while LLM tasks complete.
│       ├── code_parser.py (Python)
│       │   ├── Description: Extracts classes, methods, functions, constants, imports, and module docstrings or header comments from source files. Python is parsed with ast; Java, C#, JavaScript, and TypeScript use regex plus brace-depth tracking. Entry points are get_structure(), get_imports(), and get_module_docstring().
│       │   ├── Developer Consideration: Non-Python parsing is regex and brace-counting: _get_braced_structure() and get_javascript_structure() close class scope when depth returns to class_depth. get_imports() runs its patterns on raw lines, not noise-stripped ones, so imports inside comments or strings can match. Parse failures are logged and return empty results.
│       │   ├── Maintenance Flag: Volatile
│       │   ├── Architectural Role: Utility
│       │   ├── Refactoring Suggestions: get_javascript_structure() and _get_braced_structure() duplicate the class-scope and brace-depth tracking loop, differing only in the method and constant patterns. Extracting a shared scanner that takes the member patterns would remove the duplication and keep scope fixes in one place.
│       │   ├── Critical Dependencies:
│       │   ├──   - ast: Parses Python source for structure, imports, and module docstrings.
│       │   └──   - re: Compiles the regex patterns that drive all non-Python extraction.
│       ├── config.py (Python)
│       │   ├── Description: Defines the pydantic-settings `Settings` class (OpenRouter API key, model name, `api_semaphore_limit`) loaded from the environment or `.env`, plus fixed OpenRouter request constants. Builds the module-level `settings` singleton through `load_settings()`, which converts validation failures into `ConfigurationError`.
│       │   ├── Developer Consideration: `settings = load_settings()` runs at import time, so any module importing `repo_map.config` raises `ConfigurationError` on a bad env value; callers like `run_main()` must catch it around the deferred import. Do not move `OPENROUTER_API_URL` or the headers into `Settings`: that would let a `.env` redirect requests.
│       │   ├── Maintenance Flag: Stable
│       │   ├── Architectural Role: Configuration
│       │   ├── Critical Dependencies:
│       │   ├──   - pydantic_settings: Provides BaseSettings and SettingsConfigDict for loading env and .env values into Settings.
│       │   └──   - pydantic: Supplies Field constraints (ge=1) and the ValidationError that load_settings() translates.
│       ├── file_scanner.py (Python)
│       │   ├── Description: Walks a repository with summarize_repo(), applying DEFAULT_IGNORE_PATTERNS plus the root .gitignore, and returns an ordered list of directory and file dicts. Each file dict is detected by language, hashed via load_source_snapshot(), and hydrated from the SQLite cache or from code_parser structure extraction.
│       │   ├── Developer Consideration: _process_file() hydrates cached fields on a hash match alone, ignoring model and contract_version; staleness is decided by RepoMapApp._get_files_to_process(). Cache lookups use rel_path (relative_cache_key) while disk reads use the absolute path, so the two must not be swapped.
│       │   ├── Maintenance Flag: Volatile
│       │   ├── Architectural Role: Service
│       │   ├── Refactoring Suggestions: _process_file() returns an untyped dict[str, Any] whose keys vary between the cached and freshly parsed branches (for example, classes and constants appear only on the fresh path). A TypedDict or dataclass would make that contract explicit and catch missing-key bugs in consumers.
│       │   ├── Critical Dependencies:
│       │   ├──   - pathspec: GitIgnoreSpec matches ignore patterns the way Git does, pruning directories during the walk.
│       │   ├──   - repo_map.cache_manager: relative_cache_key builds the portable cache key for each file path.
│       │   ├──   - repo_map.code_parser: get_structure, get_module_docstring and get_imports extract structure for uncached files.
│       │   └──   - repo_map.models: SUPPORTED_LANGUAGES, SUPPORTED_TEXT_LANGUAGES and MAX_SOURCE_BYTES decide eligibility and size limits.
│       ├── llm_service.py (Python)
│       │   ├── Description: OpenRouter client and prompt builder for per-file analysis. get_llm_descriptions() re-verifies the file hash, builds cached-prefix messages, retries with backoff, and validates the JSON reply via parse_llm_response() into file metadata. A global rate_limiter bounds concurrency.
│       │   ├── Developer Consideration: ANALYSIS_CONTRACT_VERSION must be bumped by hand whenever SYSTEM_PROMPT or the fields checked in _first_invalid_field() change, or stale cached analyses are reused. The system message must stay identical across a run, because _session_id() hashes it for sticky routing and prompt caching.
│       │   ├── Maintenance Flag: Volatile
│       │   ├── Architectural Role: Service
│       │   ├── Refactoring Suggestions: The 429 retry path is duplicated: rate_limited_api_call() converts a 429 into a fake {'error': {'code': 429}} dict, which get_llm_descriptions() then inspects. Raising a dedicated RateLimited exception carrying retry_after would remove the sentinel-dict contract and simplify the retry loop.
│       │   ├── Security Assessment: rate_limited_api_call() logs the full non-200 response body at error level, which could expose provider-echoed content in logs. Prompt injection from repository content is mitigated with digest-tagged source markers and _printable() escaping, so no further risk is visible.
│       │   ├── Critical Dependencies:
│       │   ├──   - aiohttp: Performs the async HTTP POST to OpenRouter and supplies ClientError handling for retries.
│       │   ├──   - certifi: Provides the CA bundle for the SSL context used in rate_limited_api_call().
│       │   ├──   - repo_map.config: Supplies settings (API key, semaphore limit), OPENROUTER_API_URL, APP_NAME, and HTTP_REFERER.
│       │   └──   - repo_map.file_scanner: load_source_snapshot() re-reads the file and checks its hash before analysis.
│       ├── logging_utils.py (Python)
│       │   ├── Description: Sets up root logging so messages go through tqdm.write and do not corrupt active progress bars. Provides TqdmLoggingHandler, a logging.Handler subclass, and setup_logging(level), which installs it with a timestamp/level/message format.
│       │   ├── Developer Consideration: setup_logging() clears every existing root handler before calling basicConfig, so handlers added earlier (including pytest's caplog or any file handler) are removed. Call it once at startup, before other handlers are attached.
│       │   ├── Maintenance Flag: Stable
│       │   ├── Architectural Role: Utility
│       │   ├── Critical Dependencies:
│       │   └──   - tqdm: tqdm.write prints log lines without breaking active progress bars.
│       ├── main.py (Python)
│       │   ├── Description: Console-script entrypoint for repo-map. run_main() configures logging, imports RepoMapApp lazily, runs RepoMapApp().run() under asyncio.run, and maps failures to exit codes: 1 for configuration errors and unexpected exceptions, 130 for Ctrl-C.
│       │   ├── Developer Consideration: The RepoMapApp import inside run_main() is deliberate: importing cli_handler builds the settings singleton and rate limiter, and a ValueError there (e.g. bad API_SEMAPHORE_LIMIT) must be caught after setup_logging(). Moving it to module scope breaks that handling.
│       │   ├── Maintenance Flag: Stable
│       │   ├── Architectural Role: Entrypoint
│       │   ├── Critical Dependencies:
│       │   ├──   - repo_map.logging_utils: setup_logging() configures logging before any application code or errors run.
│       │   └──   - repo_map.cli_handler: Supplies RepoMapApp, the application whose async run() does all the work.
│       ├── models.py (Python)
│       │   ├── Description: Constants for language detection: SUPPORTED_LANGUAGES maps file extensions and "dockerfile" to language names, MAX_SOURCE_BYTES caps source size at 65,536 bytes, and NON_TEXT_LANGUAGES and SUPPORTED_TEXT_LANGUAGES separate binary media from analyzable text languages.
│       │   ├── Developer Consideration: SUPPORTED_TEXT_LANGUAGES is computed at import time from SUPPORTED_LANGUAGES values minus NON_TEXT_LANGUAGES. A new non-text category added to the dict must also be added to NON_TEXT_LANGUAGES, or binary files will be treated as text.
│       │   ├── Maintenance Flag: Stable
│       │   └── Architectural Role: Data Model
│       ├── py.typed (None)
│       └── report_generator.py (Python)
│           ├── Description: Renders the scanned repository structure as an ASCII tree with per-file LLM metadata (format_tree_lines), logs it (print_tree), and writes it to a fenced Markdown report (save_markdown_map) or raw JSON (save_json_map).
│           ├── Developer Consideration: format_tree_lines expects a flat, depth-ordered list with 'level' values and json.loads()es each file's 'critical_dependencies' as a string, not a dict. Reordering the list or passing a dict breaks the branch prefixes or raises TypeError, which the JSONDecodeError handler does not catch.
│           ├── Maintenance Flag: Volatile
│           ├── Architectural Role: Service
│           ├── Refactoring Suggestions: The is_last scan in format_tree_lines rescans following siblings for every item, which is quadratic on large repositories. Compute it in one reverse pass, or with a stack, to avoid that cost.
│           └── Security Assessment: _render_text spaces out backtick runs so model output cannot close the Markdown fence, but 'name' and 'language' are written unsanitized, and so are 'maintenance_flag' and 'architectural_role'. A filename containing a newline or backticks, or an unvalidated model value for those two fields, could break the report. Pass them through _render_text too.
├── tests/
│   ├── test_cache_manager.py (Python)
│   │   ├── Description: Pytest suite for the repo-local SQLite cache in repo_map.cache_manager. It checks the schema created by load_cache, migration of legacy tables, actionable CacheError messages for unwritable or corrupt cache files, relative_cache_key normalization, and prune_cache deletions.
│   │   ├── Developer Consideration: The unwritable-directory test relies on chmod 0o555 to force a sqlite3.connect failure. It does not trigger when run as root or on Windows, so it can pass or fail depending on the environment. The schema test hardcodes the full column set, so any new cache column must be added there.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.cache_manager: Module under test: load_cache, prune_cache, relative_cache_key, CacheError, CACHE_FILE_NAME.
│   │   └──   - pytest: Provides the tmp_path fixture and pytest.raises for the CacheError assertions.
│   ├── test_cli_handler.py (Python)
│   │   ├── Description: Tests for RepoMapApp in repo_map.cli_handler. They cover output path naming, the disclosure prompt, clean exit messages for bad environment, cache or directory errors, concurrent LLM enhancement with per-file failure isolation, and cache keying by relative path, model and contract version.
│   │   ├── Developer Consideration: Several tests run the real repo-map console script in a subprocess with a scrubbed environment and scratch cwd, so a developer's .env or exported OPENROUTER_* variables cannot leak in. Tests also stub llm_module.rate_limited_api_call and monkeypatch ANALYSIS_CONTRACT_VERSION on both modules, so a rename of those attributes breaks them silently.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Refactoring Suggestions: The Progress stub class and the near-identical structure setup are copy-pasted across three test_enhancement_* tests. Extracting one shared fixture or helper would remove the duplication and keep the fake tqdm interface consistent.
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.cli_handler: Provides RepoMapApp and summarize_repo, the orchestration code under test.
│   │   ├──   - repo_map.llm_service: Supplies ANALYSIS_CONTRACT_VERSION, AnalysisOutcome and the rate_limited_api_call stub target.
│   │   ├──   - repo_map.cache_manager: Provides load_cache, CacheError and CACHE_FILE_NAME for the SQLite cache fixtures.
│   │   ├──   - repo_map.file_scanner: load_source_snapshot supplies the file hash that seeds the legacy cache row.
│   │   └──   - pytest: Test runner, supplying monkeypatch, tmp_path, caplog, capsys and parametrization.
│   ├── test_code_parser.py (Python)
│   │   ├── Description: Pytest suite for repo_map.code_parser covering get_structure, get_imports and get_module_docstring across Python, JavaScript, TypeScript, Java and C#. It checks class/method/function/constant extraction, import forms, module docstrings, and robustness to comments, string literals, braces, and missing or empty files.
│   │   ├── Developer Consideration: Expectations encode regex-style parser quirks, such as a member declared after a class body counting as a top-level function and an anonymous JS class being skipped. Changing the parser's scope tracking will break many of these cases.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Refactoring Suggestions: Many Java/C# and JS/TS tests repeat the same _write-then-get_structure pattern per language. Converting the paired cases to pytest.mark.parametrize over (language, filename, source, expected) would cut duplication and make coverage gaps between languages visible.
│   │   ├── Critical Dependencies:
│   │   └──   - repo_map.code_parser: Provides get_structure, get_imports and get_module_docstring, the functions under test.
│   ├── test_config.py (Python)
│   │   ├── Description: Tests Settings and load_settings() in repo_map.config: .env loading, shell-over-.env precedence, defaults, blank API key, and API_SEMAPHORE_LIMIT validation. Also checks that a planted .env cannot change the OpenRouter URL, Referer, or X-Title headers sent by rate_limited_api_call.
│   │   ├── Developer Consideration: The transport tests monkeypatch llm_module.aiohttp.ClientSession and llm_module.settings, so they depend on llm_service referencing those module-level names and on rate_limited_api_call([], model, session) keeping its signature. The scratch_cwd fixture must unset every variable in REPO_MAP_ENV_VARS and chdir, or the real .env leaks in.
│   │   ├── Maintenance Flag: Stable
│   │   ├── Architectural Role: Test
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.config: Provides Settings, load_settings, and ConfigurationError, which these tests exercise.
│   │   ├──   - repo_map.llm_service: Its rate_limited_api_call, aiohttp session, and settings are patched to capture the outgoing request.
│   │   └──   - pytest: Supplies fixtures, monkeypatch, parametrize, and raises.
│   ├── test_file_scanner.py (Python)
│   │   ├── Description: Pytest suite for repo_map.file_scanner. It covers load_source_snapshot (hashing, size limit, UTF-8/NUL/binary rejection) and summarize_repo (ordering, .gitignore and default pruning, symlink skipping, language detection, credential-file exclusion, relative-path cache keys and cache hydration).
│   │   ├── Developer Consideration: Tests seed the real SQLite cache schema via a raw INSERT in _seed_cache_row and depend on the cache's column names, including model and contract_version. Cache-key semantics (POSIX relative path versus absolute legacy path) are asserted, so schema or key changes break them.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.file_scanner: Module under test: summarize_repo, load_source_snapshot, MAX_SOURCE_BYTES; os.scandir is monkeypatched.
│   │   └──   - repo_map.cache_manager: load_cache supplies the SQLite connection that summarize_repo reads and tests seed.
│   ├── test_known_issues.py (Python)
│   │   ├── Description: Regression tests for four tracked repo-map defects: the LLM user prompt must include the target source and mark it untrusted, failed analyses must stay pending, class-only and JSON files must be eligible for enrichment, and the parser must keep async functions and JavaScript class/function scope.
│   │   ├── Developer Consideration: These tests call private APIs (llm_module._build_user_prompt, RepoMapApp._enhance_summary_with_llm, _get_files_to_process) and monkeypatch cli_module.get_llm_descriptions. Renaming or changing the signatures of these breaks them, and the monkeypatch only works while cli_handler imports that name at module level.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.cli_handler: Provides RepoMapApp, whose _enhance_summary_with_llm and _get_files_to_process are exercised, and get_llm_descriptions, which is patched.
│   │   ├──   - repo_map.llm_service: Provides _build_user_prompt and the AnalysisOutcome.FAILURE value used to simulate a failed analysis.
│   │   ├──   - repo_map.cache_manager: load_cache creates the SQLite connection that determines which files are pending.
│   │   ├──   - repo_map.file_scanner: summarize_repo and load_source_snapshot build the structures and source snapshots under test.
│   │   └──   - repo_map.code_parser: get_structure is checked for async function and JavaScript class/method scoping.
│   ├── test_llm_service.py (Python)
│   │   ├── Description: Pytest suite for repo_map.llm_service. It covers parse_llm_response normalization and rejection, _build_user_prompt and _build_repository_layout (untrusted-source markers, related-file notes, control-character escaping), a cacheable shared system prefix, 429 and transport retry limits, stale-source guards, and model/contract_version stamping.
│   │   ├── Developer Consideration: Tests monkeypatch module-level names in repo_map.llm_service (rate_limited_api_call, rate_limiter, asyncio.sleep, aiohttp.ClientSession, MAX_LAYOUT_ENTRIES), so renaming or re-importing them silently breaks the patches. Exact backoff values (5, 10, 5.0 fallback) and the 'Target file:' / 'BEGIN SOURCE' prompt wording are asserted.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Refactoring Suggestions: The fake Response/Session aiohttp classes are defined twice (in the sampling-parameter test and in the 429 retry test), and the file_data dict literal is repeated in several tests despite the _analysis_target helper. Shared fixtures or helper reuse would remove the duplication and keep the fakes aligned.
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.llm_service: Module under test: prompt builders, parse_llm_response, get_llm_descriptions, rate limiter, API call.
│   │   ├──   - repo_map.file_scanner: load_source_snapshot supplies the SHA-256 hash that the staleness guard checks.
│   │   └──   - pytest: Provides parametrization, tmp_path, monkeypatch, and raises assertions.
│   └── test_report_generator.py (Python)
│       ├── Description: Tests for repo_map.report_generator: exact tree rendering by format_tree_lines (connectors, ancestor bars, metadata lines), whitespace collapsing and fence defusing, plus save_markdown_map and save_json_map output and a timing check for linear scaling.
│       ├── Developer Consideration: Most tests pin exact box-drawing strings, so any change to connector or indent formatting in format_tree_lines breaks them. The timing test (test_format_tree_lines_scales_linearly) compares wall-clock runs and can flake on loaded CI machines.
│       ├── Maintenance Flag: Volatile
│       ├── Architectural Role: Test
│       ├── Refactoring Suggestions: test_format_tree_lines_renders_metadata duplicates the fixture of test_format_tree_lines_renders_metadata_exact_lines, and the exact-lines test already covers its substring assertions. Remove the first test, or share the fixture.
│       ├── Critical Dependencies:
│       └──   - repo_map.report_generator: Provides format_tree_lines, save_markdown_map, and save_json_map, the functions under test.
├── .env.example (None)
├── .gitignore (Git)
│   ├── Description: Git ignore rules for the repo-map project. Excludes OS and editor files, Python caches and build output, node_modules, .env secrets (except .env.example), local agent config (except tracked skills), and repo-map's own cache DB and generated report files.
│   ├── Developer Consideration: `.claude/*` with `!.claude/skills/` tracks skills, but `.agents/*` pairs with `!.agents/skills` (no trailing slash), so the two patterns differ. The `.env.*` pattern ignores every variant except `.env.example`.
│   ├── Maintenance Flag: Stable
│   └── Architectural Role: Tooling
├── .python-version (None)
├── CHANGELOG.md (Markdown)
│   ├── Description: Top-level release index for the project, listing versions 0.0.1 through 0.14.0 newest first. Each entry has a date, Breaking/Security badges, a one-line summary, and a link to its per-version file under changelog/.
│   ├── Developer Consideration: Entries link to per-version files in changelog/<minor>.x/, and scripts/build-changelog.ts likely regenerates this index from them. Edit the per-version file or its generator, not this index, or the next build overwrites the change.
│   ├── Maintenance Flag: Generated
│   └── Architectural Role: Other
├── CLAUDE.md (Markdown)
│   ├── Description: Agent-facing guide for the repo-map CLI: stack, the `make check` gate, architecture by module, project rules (cache versioning, upload eligibility, response contract, uv cooldown), skills, and task triggers. AGENTS.md is a symlink to it.
│   ├── Developer Consideration: Changing SYSTEM_PROMPT, the request shape, or the fields _first_invalid_field() validates requires bumping ANALYSIS_CONTRACT_VERSION in llm_service.py. Otherwise stale cached analyses keep being served. This guide has to be updated whenever those rules or module roles change.
│   ├── Maintenance Flag: Volatile
│   ├── Architectural Role: Other
│   ├── Critical Dependencies:
│   ├──   - .claude/skills/: Holds the project workflow skills that the Triggers table routes agent requests to.
│   ├──   - src/repo_map/llm_service.py: Defines SYSTEM_PROMPT, ANALYSIS_CONTRACT_VERSION and the response validation these rules refer to.
│   └──   - src/repo_map/config.py: Holds Settings and the transport constants that must not become configurable.
├── LICENSE (None)
├── Makefile (None)
├── README.md (Markdown)
│   ├── Description: Project README for repo-map, a CLI that walks a repository, sends eligible files to an OpenRouter LLM, and writes a Markdown tree of per-file notes to <repository>_repo_map.md. Covers install via uv, environment settings, CLI options, cache files, upload rules, and development workflow.
│   ├── Developer Consideration: The README restates behavior defined in code: default model, API_SEMAPHORE_LIMIT default, the 64 KiB limit, the default exclusion table, and ANALYSIS_CONTRACT_VERSION in src/repo_map/llm_service.py. Changing those values without editing the matching README sections leaves the docs wrong.
│   ├── Maintenance Flag: Volatile
│   └── Architectural Role: Other
├── pyproject.toml (None)
└── uv.lock (None)
└────────────── 
```
````
