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
│       │       ├── Description: Agent skill definition for a behavior-preserving cleanup pass over repo-map code. It sets scope (uncommitted diff, named module, or all of src/repo_map/ and tests/), a review checklist, Python 3.12 rewrites, protected contracts, and a report format.
│       │       ├── Developer Consideration: The "Never a simplification" list hardcodes repo-map contracts (ANALYSIS_CONTRACT_VERSION, SYSTEM_PROMPT, _build_messages(), _first_invalid_field(), cache schema, CLI flags). If those names or rules change in the code, this file must be updated or the agent will protect or break the wrong things.
│       │       ├── Maintenance Flag: Volatile
│       │       ├── Architectural Role: Tooling
│       │       ├── Critical Dependencies:
│       │       ├──   - make check: Baseline and post-edit gate the skill runs after every category of change.
│       │       ├──   - CLAUDE.md: Source of repo conventions the agent must read before editing.
│       │       └──   - src/repo_map/llm_service.py: Holds the analysis contract and failure-handling design the skill must preserve.
│       ├── git-wrapup/
│       │   └── SKILL.md (Markdown)
│       │       ├── Description: Agent skill defining the git-wrapup workflow: a pre-commit gate, commits grouped by concern as Conventional Commits, a version bump with changelog entry, a chore(release) commit, and a local annotated vX.Y.Z tag. It stops before any push or publish.
│       │       ├── Developer Consideration: The gate requires bumping ANALYSIS_CONTRACT_VERSION in src/repo_map/llm_service.py whenever the prompt, _build_messages(), the request payload or _first_invalid_field() changes, since cache keys depend on it. Commits must use `git commit --only` by pathspec with no closing keywords, and a file is never split across commits.
│       │       ├── Maintenance Flag: Volatile
│       │       ├── Architectural Role: Tooling
│       │       ├── Critical Dependencies:
│       │       ├──   - release-and-publish: Downstream skill that pushes, publishes to PyPI, creates the GitHub release and closes issues.
│       │       ├──   - code-simplifier: Skill to run first when the diff touches more than a line or two of src/repo_map/.
│       │       ├──   - make check: Gate running ruff, pyright, pytest, uv lock --check and the changelog check.
│       │       ├──   - changelog/template.md: Template copied to author each release's changelog entry.
│       │       └──   - uv: Sets the version via `uv version` and keeps uv.lock in sync.
│       ├── maintenance/
│       │   └── SKILL.md (Markdown)
│       │       ├── Description: Claude skill defining a seven-step uv dependency maintenance workflow for repo-map: survey outdated packages, run uv lock --upgrade, raise lower bounds, read upstream release notes, adopt changes, verify with make check, and record changelog rows.
│       │       ├── Developer Consideration: The `exclude-newer = "3 days"` cooldown in pyproject.toml must never be lowered; urgent overrides go in `exclude-newer-package` in pyproject.toml, not via the CLI flag, which writes to uv.lock and breaks `uv lock --check`.
│       │       ├── Maintenance Flag: Volatile
│       │       ├── Architectural Role: Tooling
│       │       ├── Refactoring Suggestions: The Step 3 package table duplicates the dependency list in pyproject.toml and the module mapping in src/repo_map; it will go stale when dependencies change. Point to pyproject.toml and grep for imports instead of hardcoding it.
│       │       ├── Critical Dependencies:
│       │       ├──   - git-wrapup: Companion skill that handles the version bump, commits, tag, and ANALYSIS_CONTRACT_VERSION.
│       │       ├──   - uv: Drives survey, lock upgrade, sync, and lower-bound raising via uv add.
│       │       └──   - make check: Verification gate running ruff, pyright, pytest, uv lock --check, and the changelog check.
│       ├── release-and-publish/
│       │   └── SKILL.md (Markdown)
│       │       ├── Description: Agent skill runbook for publishing a repo-map release already committed and tagged by git-wrapup: sanity checks, make check gate, per-version build in dist/X.Y.Z, push, uv publish to PyPI, install verification, GitHub release, and closing fixed issues.
│       │       ├── Developer Consideration: The steps depend on a strict order and idempotent re-runs: build into dist/X.Y.Z/ and reuse it after a partial upload, since PyPI rejects non-identical files under the same name. Never force-push, move a pushed tag, or yank a release.
│       │       ├── Maintenance Flag: Volatile
│       │       ├── Architectural Role: Tooling
│       │       ├── Critical Dependencies:
│       │       ├──   - git-wrapup: Upstream skill that must already have created the release commit, changelog entry, and annotated tag.
│       │       ├──   - uv: Builds distributions, publishes to PyPI, and runs the wheel via uvx for smoke tests.
│       │       ├──   - gh: Creates the GitHub release and comments on and closes fixed issues.
│       │       ├──   - make check: Quality gate that must pass cleanly before building or pushing.
│       │       └──   - security: macOS Keychain CLI supplying the PyPI token inline for uv publish.
│       └── report-issue-local/
│           └── SKILL.md (Markdown)
│               ├── Description: Agent skill workflow for filing bug, feature, and documentation issues against cyanheads/repo-map with the gh CLI. Covers duplicate search, public-fixture reproduction, redaction of private data, title/label/assignee conventions, body templates matching the repo's issue forms, and follow-up comments.
│               ├── Developer Consideration: Bug and feature body headings must stay in sync with .github/ISSUE_TEMPLATE/bug_report.yml and feature_request.yml, and the scope list must match the modules in src/repo_map. The body must go through --body-file, because a shell-string body loses backticked spans to command substitution.
│               ├── Maintenance Flag: Volatile
│               ├── Architectural Role: Tooling
│               ├── Critical Dependencies:
│               ├──   - gh: GitHub CLI used to search, view, create, and comment on issues.
│               ├──   - .github/ISSUE_TEMPLATE/bug_report.yml: Defines the bug body headings and order the skill mirrors.
│               ├──   - .github/ISSUE_TEMPLATE/feature_request.yml: Defines the required feature headings the skill reproduces.
│               └──   - release-and-publish: Sibling skill that closes issues after the fix is on PyPI.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml (YAML)
│   │   │   ├── Description: GitHub issue form for bug reports on repo-map. It applies the "bug" label, assigns cyanheads, and collects repo-map version, Python version, OS, description, reproduction, actual and expected behavior, and optional context, with a warning not to post secrets or private source.
│   │   │   ├── Developer Consideration: The version field placeholder "0.14.0" is hardcoded and will go stale after releases. Field ids (version, python-version, os, reproduction, etc.) are what GitHub uses to key submitted issue content, so renaming them changes the issue layout.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Tooling
│   │   ├── config.yml (YAML)
│   │   │   ├── Description: GitHub issue chooser configuration that keeps blank issues enabled and adds a contact link to OpenRouter support for account, billing, and provider availability problems that do not reproduce in repo-map.
│   │   │   ├── Developer Consideration: blank_issues_enabled: true lets reporters bypass bug_report.yml and feature_request.yml; setting it to false would force every issue through those templates.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Tooling
│   │   └── feature_request.yml (YAML)
│   │       ├── Description: GitHub issue form for feature requests against repo-map. Applies the "enhancement" label, assigns "cyanheads", and collects four textareas: use-case and proposal (required), alternatives and context (optional).
│   │       ├── Developer Consideration: The assignee "cyanheads" is hardcoded, so the maintainer must be changed here if ownership moves. Each textarea `id` (use-case, proposal, alternatives, context) is part of the form schema, so renaming one alters how new issues are structured.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Tooling
│   ├── workflows/
│   │   └── codeql.yml (YAML)
│   │       ├── Description: GitHub Actions workflow running CodeQL static analysis for the `actions` and `python` languages on pushes and PRs to main and a weekly schedule, uploading results under `/language:<language>` categories.
│   │       ├── Developer Consideration: GitHub's repo-level CodeQL "default setup" must stay off or it conflicts with this workflow. The `category: /language:${{ matrix.language }}` value must match default setup's category so alerts raised earlier close once the code is fixed.
│   │       ├── Maintenance Flag: Stable
│   │       ├── Architectural Role: Tooling
│   │       ├── Critical Dependencies:
│   │       ├──   - github/codeql-action/init@v4: Initializes CodeQL for each matrix language with build-mode none.
│   │       ├──   - github/codeql-action/analyze@v4: Runs the analysis and uploads results under the per-language category.
│   │       └──   - actions/checkout@v7: Checks out the repository source for analysis.
│   ├── CODE_OF_CONDUCT.md (Markdown)
│   │   ├── Description: Project code of conduct listing expected behavior, unacceptable conduct, scope across issues, PRs and commits, and a private email reporting channel. It also makes agent operators accountable for what their agents post.
│   │   ├── Developer Consideration: The relative link to ./CONTRIBUTING.md depends on both files staying together in .github/, and the reporting address is hardcoded to one maintainer, so update it here if ownership changes.
│   │   ├── Maintenance Flag: Stable
│   │   ├── Architectural Role: Other
│   │   ├── Critical Dependencies:
│   │   └──   - ./CONTRIBUTING.md: Linked for the rules agent operators must read before filing issues.
│   ├── CONTRIBUTING.md (Markdown)
│   │   ├── Description: Contributor guide for repo-map that routes bugs and feature requests to the GitHub issue forms, lists pre-filing checks (latest release, duplicate search, redaction), and states what an actionable report contains. Includes a section for agents and a security-reporting pointer.
│   │   ├── Developer Consideration: Links to .claude/skills/report-issue-local/SKILL.md and SECURITY.md by relative path, and describes the Bug Report and Feature Request forms; renaming or moving those files or the forms in .github/ISSUE_TEMPLATE breaks the guidance.
│   │   ├── Maintenance Flag: Stable
│   │   ├── Architectural Role: Other
│   │   ├── Critical Dependencies:
│   │   ├──   - .claude/skills/report-issue-local/SKILL.md: Agent workflow the guide tells agents to read before filing issues.
│   │   └──   - SECURITY.md: Destination for vulnerability reports instead of public issues.
│   └── SECURITY.md (Markdown)
│       ├── Description: Security policy for repo-map: only the latest release gets fixes, vulnerabilities go through GitHub private reporting or a security email, and the in-scope list covers wrongly uploaded files, OpenRouter API key leaks or redirects, and analyzed repository content steering the tool.
│       ├── Developer Consideration: The scope links to the README's "What gets sent" section (../README.md#what-gets-sent) as the definition of which files stay local; renaming that heading or changing the upload rules there breaks the anchor and changes what counts as in scope.
│       ├── Maintenance Flag: Stable
│       └── Architectural Role: Other
├── changelog/
│   ├── 0.0.x/
│   │   └── 0.0.1.md (Markdown)
│   │       ├── Description: Changelog entry for release 0.0.1 dated 2025-09-28, with YAML front matter summary "Initial commit" and a single Added item recording the initial commit.
│   │       ├── Developer Consideration: Entries follow changelog/template.md, and scripts/build-changelog.ts likely reads the front matter `summary` and the version heading; changing their format could break CHANGELOG.md generation.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.1.x/
│   │   └── 0.1.0.md (Markdown)
│   │       ├── Description: Release notes for version 0.1.0 (2025-09-29), with a front-matter summary line. It records the move from setuptools to Poetry, the split of the monolithic script into modules, SQLite caching, and multi-language code parsing.
│   │       ├── Developer Consideration: The front-matter `summary` field and the `# 0.1.0 — date` heading follow the format of changelog/template.md. scripts/build-changelog.ts presumably reads them to build CHANGELOG.md, so changing their shape could break that build.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.10.x/
│   │   ├── 0.10.0.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.10.0, a breaking security release. They record that the OpenRouter endpoint and headers became fixed constants, that default exclusions now cover more credential-bearing file types, and that directory-only ignore patterns now prune the walk.
│   │   │   ├── Developer Consideration: Front matter keys (summary, breaking, security) are likely consumed by scripts/build-changelog.ts to assemble CHANGELOG.md, so keep them and the heading format matching changelog/template.md when editing.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   ├── Architectural Role: Other
│   │   │   └── Security Assessment: No risk in the file itself. It documents fixes for a .env-planted endpoint redirect that could leak the API key and source (#17), and for credential-bearing files such as .envrc and *.tfstate being uploaded to the LLM (#15).
│   │   ├── 0.10.1.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.10.1 (2026-08-13) with front-matter summary. Records two tree-rendering fixes: indentation prefixes now track open branches by depth (#18), and model-authored text is flattened to one line so it cannot break tree lines or the Markdown report fence (#22).
│   │   │   ├── Developer Consideration: The front-matter `summary` field is likely read by scripts/build-changelog.ts when assembling CHANGELOG.md, so keep the `---` block and key intact and follow changelog/template.md. The release note says the cache and .repo_map_structure.json keep raw text, so flattening belongs only at render time.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.10.2.md (Markdown)
│   │       ├── Description: Release notes for repo-map 0.10.2, with a front-matter summary. Lists parser extraction fixes in code_parser: async Python functions, JavaScript class scope, leading-header module docstrings, and import and method patterns for scoped packages, re-exports, and Java/C# methods (issues #12, #19, #20).
│   │       ├── Developer Consideration: The YAML front-matter `summary` is likely read by scripts/build-changelog.ts to assemble CHANGELOG.md, so keep the front-matter keys and the `# version — date` heading format consistent with changelog/template.md.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.11.x/
│   │   ├── 0.11.0.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.11.0 (2026-08-13). They record two fixes: configuration and cache errors now exit with one actionable line (ConfigurationError, CacheError), and the upload prompt discloses full-source upload and requires an explicit y/yes.
│   │   │   ├── Developer Consideration: The YAML front matter `summary` field is likely consumed by scripts/build-changelog.ts to assemble CHANGELOG.md, so keep its keys and the `# version — date` heading format consistent with changelog/template.md.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.11.1.md (Markdown)
│   │       ├── Description: Release notes for repo-map 0.11.1 (2026-08-13). They record three fixes: class regexes no longer match prose in comments or strings, Java and C# class scope now closes at the closing brace, and a malformed OpenRouter response now fails only its own file.
│   │       ├── Developer Consideration: The YAML front matter `summary` is probably read by scripts/build-changelog.ts to build CHANGELOG.md, so keep its key and quoting intact and follow changelog/template.md when editing.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.12.x/
│   │   └── 0.12.0.md (Markdown)
│   │       ├── Description: Changelog entry for release 0.12.0 (2026-08-13). It records the new `rel_path` field on scanned files and in `.repo_map_structure.json`, the switch of cache keys to repository-relative paths, and the end-of-run `prune_cache` pass that removes stale rows.
│   │       ├── Developer Consideration: The front-matter `summary` line is likely read by the changelog build (scripts/build-changelog.ts) to populate CHANGELOG.md, so keep its format intact; the entry describes cache behavior, and the code is the source of truth if the two diverge.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.13.x/
│   │   ├── 0.13.0.md (Markdown)
│   │   │   ├── Description: Release notes for 0.13.0: the cache now stores `model` and `contract_version` beside `hash`, and reuses a stored analysis only when all three match the current run. Documents the `ANALYSIS_CONTRACT_VERSION` constant and treats old NULL rows as misses.
│   │   │   ├── Developer Consideration: `ANALYSIS_CONTRACT_VERSION` in `llm_service.py` is bumped by hand; forgetting to raise it after editing `SYSTEM_PROMPT` or the validated field set keeps serving stale cached analyses.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.13.1.md (Markdown)
│   │       ├── Description: Release note for version 0.13.1 (2026-08-13). It records that the `check` command now runs `scripts/sync_skills.py --check`, which compares skill mirrors by content hash without writing, and fails on a missing or drifted mirror file.
│   │       ├── Developer Consideration: The note refers to `scripts.py`, `scripts/sync_skills.py`, `skills/` and `poetry` commands that do not appear in the repository layout (which uses uv.lock and has no top-level skills/ directory), so verify those paths before relying on this text. The file's front-matter `summary` is probably read by scripts/build-changelog.ts, so keep the format in line with changelog/template.md.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.14.x/
│   │   └── 0.14.0.md (Markdown)
│   │       ├── Description: Release notes for repo-map 0.14.0 (2026-10-20): rewritten per-file analysis prompt with a smaller, cacheable request, new default model anthropic/claude-sonnet-5.5, Poetry-to-uv migration, `make check` gate, security markers, and dependency bumps. YAML front matter carries `summary` and `security: true`.
│   │       ├── Developer Consideration: The front matter (`summary`, `security: true`) is likely consumed by the changelog index generator (`scripts/build-changelog.ts`, `make changelog`), so keep it valid and follow `changelog/template.md`. Edit this file, then regenerate `CHANGELOG.md` rather than editing the index by hand.
│   │       ├── Maintenance Flag: Stable
│   │       ├── Architectural Role: Other
│   │       ├── Critical Dependencies:
│   │       └──   - scripts/build-changelog.ts: Generates the CHANGELOG.md index from per-version files and their front matter.
│   ├── 0.2.x/
│   │   └── 0.2.0.md (Markdown)
│   │       ├── Description: Release notes for version 0.2.0 (2025-09-29). They record the RepoMapApp refactor of main.py, TqdmLoggingHandler logging, stricter ignore-pattern handling in file_scanner.py, the APIRateLimiter in llm_service.py, and the new AGENTS.md.
│   │       ├── Developer Consideration: The front-matter `summary` field is likely read by scripts/build-changelog.ts to assemble CHANGELOG.md; keep the front-matter keys and the `# version — date` heading format consistent with changelog/template.md.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.3.x/
│   │   └── 0.3.0.md (Markdown)
│   │       ├── Description: Changelog entry for release 0.3.0 (2025-09-29). Records one change: README.md was updated for the default model name, .gitignore handling, and the description of main.py. Starts with YAML front matter holding a one-line summary.
│   │       ├── Developer Consideration: The front matter `summary` field and the `## Changed` heading follow the layout of changelog/template.md. scripts/build-changelog.ts likely reads them to build CHANGELOG.md, so keep that structure intact.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.4.x/
│   │   └── 0.4.0.md (Markdown)
│   │       ├── Description: Release notes for version 0.4.0 (2025-09-29), with a front-matter summary. They record the added per-file LLM analysis fields, the extended cache schema, and the changes to file_scanner.py, llm_service.py, main.py, cache_manager.py and AGENTS.md.
│   │       ├── Developer Consideration: The entry names AGENTS.md, which is not in the repository layout (only CLAUDE.md is), and lists a code quality score that the current output schema lacks; check these against the code before relying on them. The front-matter `summary` field is likely read by scripts/build-changelog.ts, so keep its format.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.5.x/
│   │   └── 0.5.0.md (Markdown)
│   │       ├── Description: Release notes for version 0.5.0 (2025-09-29). Records the split of main.py into cli_handler.py, logging_utils.py and report_generator.py, the added examples/example.md, a README refresh, and the version bump.
│   │       ├── Developer Consideration: The YAML front-matter `summary` field is likely consumed by scripts/build-changelog.ts to build CHANGELOG.md, so keep the front-matter and heading format aligned with changelog/template.md.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.7.x/
│   │   ├── 0.7.0.md (Markdown)
│   │   │   ├── Description: Release notes for version 0.7.0 (2026-03-08). Records the default LLM model change to google/gemini-3-flash-preview, the added CLAUDE.md, the ruff lint config fix, Black formatting, X | None type hints, and dependency bumps.
│   │   │   ├── Developer Consideration: The YAML front-matter `summary` field is presumably read by scripts/build-changelog.ts to assemble CHANGELOG.md, so keep the front-matter keys and the `# version — date` heading format consistent with changelog/template.md.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.7.1.md (Markdown)
│   │       ├── Description: Release notes for version 0.7.1 (2026-04-25). They cover the pathspec GitIgnoreSpec migration in file_scanner.py, the new maintenance skill with a sync-skills script, the deferred tree-sitter upgrade, and in-range runtime and dev dependency bumps.
│   │       ├── Developer Consideration: The notes reference paths and tooling (skills/maintenance/SKILL.md, scripts/sync_skills.py, .claude/settings.json, Poetry scripts) that do not match the current layout, which has .claude/skills/ and uv.lock. Check the actual repo before relying on them.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.8.x/
│   │   ├── 0.8.0.md (Markdown)
│   │   │   ├── Description: Release notes for version 0.8.0 (2026-04-25), flagged breaking: LLM output moves to a single JSON object with a closed architectural_role enum, the default model becomes anthropic/claude-sonnet-4.6, and code_quality_score is removed across the schema, cache, and report.
│   │   │   ├── Developer Consideration: The front matter carries `breaking: true` and a `summary` line that scripts/build-changelog.ts likely consumes to build CHANGELOG.md, so keep that front-matter shape when editing. The notes also say old caches keep the dropped code_quality_score column.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   ├── 0.8.1.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.8.1 (2026-08-13): a 25-test baseline suite with eight strict xfail regressions for issues #5–#12, a repaired scripts.py dispatcher, project-local workflows, removal of tree-sitter dependencies, and runtime and dev dependency bumps.
│   │   │   ├── Developer Consideration: The front-matter `summary` line is presumably consumed by scripts/build-changelog.ts when assembling CHANGELOG.md, so keep it single-line and in step with the body. The file follows changelog/template.md, and the section headings (Added, Changed, Removed, Dependencies) should be kept in that shape.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   ├── 0.8.2.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.8.2, dated 2026-08-13. They record two fixes: the scanner now skips file and directory symlinks (issue #9), and language detection matches exact filenames, then the longest suffix (issue #5).
│   │   │   ├── Developer Consideration: The front-matter `summary` line is likely consumed by scripts/build-changelog.ts when it builds CHANGELOG.md, so keep the `---` block and the `# 0.8.2 — date` heading format consistent with changelog/template.md.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.8.3.md (Markdown)
│   │       ├── Description: Release notes for repo-map 0.8.3 (2026-08-13). Records two fixes: file analyses are scheduled together and consumed in completion order under the request bound (#7), and HTTP 429 backoff now runs after the concurrency permit is released (#8).
│   │       ├── Developer Consideration: The front-matter `summary` line is probably what scripts/build-changelog.ts reads to build CHANGELOG.md, so keep the key and format consistent with changelog/template.md.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   ├── 0.9.x/
│   │   ├── 0.9.0.md (Markdown)
│   │   │   ├── Description: Release notes for repo-map 0.9.0 (2026-08-13): sends each eligible file's full source to the LLM as delimited untrusted data, validates responses in parse_llm_response before caching, and bases enrichment eligibility on the scanner's source policy.
│   │   │   ├── Developer Consideration: The front-matter `summary` field is likely consumed by scripts/build-changelog.ts when it assembles CHANGELOG.md, so keep the front-matter format and the `# version — date` heading intact and consistent with changelog/template.md.
│   │   │   ├── Maintenance Flag: Stable
│   │   │   └── Architectural Role: Other
│   │   └── 0.9.1.md (Markdown)
│   │       ├── Description: Changelog entry for release 0.9.1 (2026-08-13) recording a single dev-only transitive dependency bump, platformdirs 4.11.2 to 4.11.3, with no code changes. Front matter carries a one-line summary field.
│   │       ├── Developer Consideration: The YAML front-matter `summary` is likely consumed by scripts/build-changelog.ts to build CHANGELOG.md, so keep the front-matter keys and the heading format matching changelog/template.md.
│   │       ├── Maintenance Flag: Stable
│   │       └── Architectural Role: Other
│   └── template.md (Markdown)
│       ├── Description: Format reference for per-release changelog files in changelog/<major.minor>.x/<version>.md. Defines the YAML front matter (summary, breaking, security, agent-notes), the H1 date heading, Keep a Changelog sections, and an authoring guide for tone, links and dependency rows.
│       ├── Developer Consideration: The summary value must stay quoted, since an unquoted `: ` breaks GitHub's strict YAML parser, and `security: true` is only for vulnerabilities in this project's own code, never dependency CVE bumps. The file says not to edit it; copy it per release and delete unused sections and comments.
│       ├── Maintenance Flag: Stable
│       ├── Architectural Role: Tooling
│       ├── Critical Dependencies:
│       └──   - scripts/build-changelog.ts: Presumably reads the per-version files and front matter (summary, breaking, security) to build the CHANGELOG.md rollup.
├── examples/
│   └── example.md (Markdown)
│       ├── Description: Static sample of repo-map's Markdown report. A fenced ASCII tree of the repo-map repository gives each file a Description, Developer Consideration, Maintenance Flag, Architectural Role, Critical Dependencies and optional Refactoring and Security entries. It shows readers the output format.
│       ├── Developer Consideration: This is a hand-kept snapshot inside a four-backtick fence wrapping a three-backtick fence, not live tool output. Its entries can drift from the real repository layout and report format, so regenerate it with `uv run repo-map` when either changes.
│       ├── Maintenance Flag: Volatile
│       └── Architectural Role: Other
├── scripts/
│   └── build-changelog.ts (TypeScript)
│       ├── Description: Builds CHANGELOG.md as an index from per-version files in changelog/<major.minor>.x/, reading frontmatter (summary, breaking, security) and the H1 date. Sorts versions by semver descending; `--check` mode exits 1 if the file has drifted.
│       ├── Developer Consideration: CHANGELOG_DIR and CHANGELOG_PATH come from `resolve()` against the current working directory, so the script must run from the repo root. The --check comparison is byte-exact against the generated string, so any change to HEADER or renderEntry() whitespace makes every checkout report drift.
│       ├── Maintenance Flag: Stable
│       ├── Architectural Role: Tooling
│       ├── Refactoring Suggestions: Replace the hand-rolled regex frontmatter parser in parseFrontmatter() with a YAML library. It currently ignores multiline or folded summaries, and its ': ' check only approximates strict YAML.
│       ├── Critical Dependencies:
│       ├──   - node:fs: Reads the changelog tree and per-version files; writes CHANGELOG.md.
│       ├──   - node:path: Resolves changelog and CHANGELOG.md paths from the working directory.
│       └──   - node:process: Reads the --check flag and sets exit codes.
├── src/
│   └── repo_map/
│       ├── __init__.py (Python)
│       │   ├── Description: Package marker for repo_map containing only a module docstring that describes the tool as producing repository trees with LLM-authored file documentation. It defines no code and re-exports nothing.
│       │   ├── Maintenance Flag: Stable
│       │   └── Architectural Role: Other
│       ├── cache_manager.py (Python)
│       │   ├── Description: SQLite cache for per-file LLM analysis: load_cache() opens or migrates .repo-map-cache.db in the repo, load_cache_snapshot() copies it read-only into memory, relative_cache_key() builds keys, and prune_cache() deletes stale rows. Failures raise CacheError.
│       │   ├── Developer Consideration: Keys must come from relative_cache_key() (repo-relative, forward slashes). Legacy absolute-path rows never match and are only removed by prune_cache(). Rows from added columns have NULL model/contract_version, so they read as misses. Writes to a load_cache_snapshot() connection never reach disk.
│       │   ├── Maintenance Flag: Stable
│       │   ├── Architectural Role: Persistence
│       │   ├── Critical Dependencies:
│       │   └──   - sqlite3: Stores the cache table, runs schema migration, and copies the database via backup().
│       ├── cli_handler.py (Python)
│       │   ├── Description: CLI lifecycle for repo-map: RepoMapApp.run() parses arguments, scans the repo against an in-memory cache snapshot, builds an UploadPlan, asks consent (or honors -y/--dry-run), analyzes pending files concurrently via the LLM, caches results, and writes the JSON and Markdown maps.
│       │   ├── Developer Consideration: Order is load-bearing: scanning and plan_upload() run on an in-memory snapshot before consent, and the real cache opens only after it. The prompt's claims must stay in step with llm_service._build_messages(), file_scanner._IgnoreRules and README. _update_cache_for_file() reads model and contract_version without fallback.
│       │   ├── Maintenance Flag: Volatile
│       │   ├── Architectural Role: Service
│       │   ├── Refactoring Suggestions: _update_cache_for_file() writes SQL directly into the cache table from the CLI layer, and the schema columns are duplicated from cache_manager. Moving it to cache_manager as a save-analysis function would keep the schema in one place.
│       │   ├── Security Assessment: Source is uploaded to a third party, but the consent gate handles this: an explicit y/N prompt where empty input declines, plus -y and --dry-run. The cache and report use the repository path in SQL parameters and os.path joins only. No injection or secret leakage is visible.
│       │   ├── Critical Dependencies:
│       │   ├──   - repo_map.llm_service: Supplies get_llm_descriptions, resolve_response_format, the contract version, AnalysisOutcome and the fatal error type.
│       │   ├──   - repo_map.cache_manager: Provides load_cache, load_cache_snapshot, prune_cache and CacheError for the SQLite cache.
│       │   ├──   - repo_map.file_scanner: summarize_repo() builds the structure list whose entries plan_upload() filters and the analysis fills in.
│       │   ├──   - repo_map.report_generator: print_tree, save_json_map and save_markdown_map produce the output.
│       │   ├──   - repo_map.config: settings provides the API key check, default model and default concurrency.
│       │   └──   - tqdm: Shows progress while files are analyzed concurrently.
│       ├── code_parser.py (Python)
│       │   ├── Description: Extracts per-file structure (classes with methods, top-level functions, constants), module docstrings or header comments, and import lists from source files. Uses ast for Python and noise-stripping regex plus brace counting for Java, C#, JavaScript and TypeScript. Entry points are get_structure(), get_module_docstring() and get_imports().
│       │   ├── Developer Consideration: Non-Python parsing is heuristic: class scope is tracked by brace depth after _strip_noise() blanks strings and comments, so a missed brace or a one-line declaration can misattribute members. Python import names are built as "module.name", and relative imports with no module yield ".name".
│       │   ├── Maintenance Flag: Volatile
│       │   ├── Architectural Role: Utility
│       │   ├── Refactoring Suggestions: get_javascript_structure() duplicates the class-scope and brace-depth tracking loop in _get_braced_structure(). Extract that loop into one helper parameterised by method matcher so scope fixes land in one place.
│       │   ├── Critical Dependencies:
│       │   ├──   - ast: Parses Python files to get classes, functions, constants, docstrings and imports.
│       │   └──   - re: Compiled patterns detect classes, members, constants and imports in non-Python languages.
│       ├── config.py (Python)
│       │   ├── Description: Defines the pydantic-settings `Settings` class (OpenRouter API key, model name, `api_semaphore_limit`) and a module-level `settings` singleton built by `load_settings()`. Also holds the fixed OpenRouter URL, referer and app-name constants, and `ConfigurationError`.
│       │   ├── Developer Consideration: `settings = load_settings()` runs at import time, so any module importing `repo_map.config` fails with `ConfigurationError` on bad env values. Also, `.env` is resolved against the current working directory, and the endpoint constants are deliberately kept off `Settings`; do not move them there.
│       │   ├── Maintenance Flag: Stable
│       │   ├── Architectural Role: Configuration
│       │   ├── Critical Dependencies:
│       │   ├──   - pydantic_settings: BaseSettings and SettingsConfigDict load fields from environment variables and the `.env` file.
│       │   └──   - pydantic: Field enforces the ge=1 bound on api_semaphore_limit; ValidationError is translated into ConfigurationError.
│       ├── file_scanner.py (Python)
│       │   ├── Description: Walks a repository with Git-style ignore rules and builds the flat file/directory summary list via summarize_repo(). load_source_snapshot() gates which files may be uploaded to the LLM: size, UTF-8, NUL and private-key checks plus a SHA-256 hash. Cached analysis rows are hydrated by hash.
│       │   ├── Developer Consideration: _process_file() trusts a cache row on hash match alone, ignoring model and contract_version; cli_handler.plan_upload() decides if the analysis is still valid. Also, the 'path' field is absolute while 'rel_path' is the only cache key, so don't swap them.
│       │   ├── Maintenance Flag: Volatile
│       │   ├── Architectural Role: Service
│       │   ├── Refactoring Suggestions: Cached-row hydration in _process_file() fills defaults by hand ('{}' for critical_dependencies, 'None' for refactoring_suggestions) in an untyped dict. A typed record shared with the cache layer would prevent key and default drift between the cached and fresh paths.
│       │   ├── Critical Dependencies:
│       │   ├──   - pathspec: Compiles and matches gitignore patterns (GitIgnoreSpec, GitIgnoreSpecPattern) for default and per-directory ignore rules.
│       │   ├──   - repo_map.models: Supplies MAX_SOURCE_BYTES, SUPPORTED_LANGUAGES and SUPPORTED_TEXT_LANGUAGES for eligibility and language detection.
│       │   ├──   - repo_map.code_parser: get_structure, get_module_docstring and get_imports extract metadata for files missing from the cache.
│       │   ├──   - repo_map.cache_manager: relative_cache_key builds the portable cache key; the sqlite cache table is queried.
│       │   └──   - sqlite3: Connection type for the cache table queried by _process_file.
│       ├── llm_service.py (Python)
│       │   ├── Description: OpenRouter client for per-file analysis: builds cached chat prompts with repository layout and related-file notes, sends requests with retry and error classification, validates and normalizes the JSON reply into file metadata via get_llm_descriptions(), parse_llm_response() and resolve_response_format().
│       │   ├── Developer Consideration: Changing SYSTEM_PROMPT, ANALYSIS_SCHEMA, or the fields checked in _first_invalid_field() without bumping ANALYSIS_CONTRACT_VERSION leaves stale cached analyses accepted. The system message must stay identical across a run, since it carries the cache_control breakpoint and the _session_id() key.
│       │   ├── Maintenance Flag: Volatile
│       │   ├── Architectural Role: Service
│       │   ├── Refactoring Suggestions: Move the prompt and layout builders (_build_messages, _build_user_prompt, _related_notes, _resolve_import and helpers) into their own module, so the HTTP and retry client is separated from prompt construction.
│       │   ├── Critical Dependencies:
│       │   ├──   - aiohttp: Performs the chat-completion POST and the model endpoints lookup.
│       │   ├──   - certifi: Supplies the CA bundle for the shared TLS context in _ssl_context().
│       │   ├──   - repo_map.config: Provides settings (API key, concurrency limit), API URLs, APP_NAME and HTTP_REFERER.
│       │   └──   - repo_map.file_scanner: load_source_snapshot() re-reads the file and checks its hash before analysis.
│       ├── logging_utils.py (Python)
│       │   ├── Description: Provides TqdmLoggingHandler, which routes log records through tqdm.write() so messages don't corrupt active progress bars, and setup_logging(), which resets the root logger to use that handler with a timestamped format.
│       │   ├── Developer Consideration: setup_logging() clears every existing root logger handler before installing TqdmLoggingHandler, which removes handlers added by other libraries or tests, and it should run once at startup. Output goes through tqdm.write (stdout by default), not stderr.
│       │   ├── Maintenance Flag: Stable
│       │   ├── Architectural Role: Utility
│       │   ├── Critical Dependencies:
│       │   └──   - tqdm: tqdm.write() prints log lines without breaking active progress bars.
│       ├── main.py (Python)
│       │   ├── Description: Console-script entrypoint exposing run_main(), which sets up logging, lazily imports RepoMapApp, runs it under asyncio.run, and maps outcomes to exit codes: the app's status, 1 for config or unexpected errors, 130 for Ctrl-C.
│       │   ├── Developer Consideration: The RepoMapApp import inside run_main() is deliberately deferred: importing repo_map.cli_handler builds the settings singleton and rate limiter, so a bad API_SEMAPHORE_LIMIT raises ValueError there. Moving it to module scope puts that failure outside the handler.
│       │   ├── Maintenance Flag: Stable
│       │   ├── Architectural Role: Entrypoint
│       │   ├── Critical Dependencies:
│       │   ├──   - repo_map.cli_handler: Provides RepoMapApp, whose async run() does the work and returns the exit status.
│       │   ├──   - repo_map.logging_utils: setup_logging() must run before anything else so errors are logged.
│       │   └──   - asyncio: asyncio.run drives the async RepoMapApp.run() coroutine.
│       ├── models.py (Python)
│       │   ├── Description: Constants for language detection: SUPPORTED_LANGUAGES maps file extensions and special names (dockerfile, makefile) to language labels, NON_TEXT_LANGUAGES lists binary categories, SUPPORTED_TEXT_LANGUAGES is the derived text-only set, and MAX_SOURCE_BYTES caps source size at 65,536.
│       │   ├── Developer Consideration: SUPPORTED_TEXT_LANGUAGES is derived from SUPPORTED_LANGUAGES values minus NON_TEXT_LANGUAGES, so a new label for a binary type must also be added to NON_TEXT_LANGUAGES or it will be treated as text. Keys mix lowercase extensions and bare names like "dockerfile", so lookups must normalize case.
│       │   ├── Maintenance Flag: Stable
│       │   └── Architectural Role: Data Model
│       ├── py.typed (None)
│       └── report_generator.py (Python)
│           ├── Description: Renders the repository structure as an ASCII tree via format_tree_lines(), with per-file metadata lines for description, flags, and dependencies. Also provides print_tree() for logging, save_markdown_map() for a fenced Markdown report, and save_json_map() for the raw JSON dump.
│           ├── Developer Consideration: format_tree_lines() assumes `structure` is a depth-first flat list with a correct `level` per item, and expects `critical_dependencies` as a JSON string, not a dict. A dict raises TypeError, which the JSONDecodeError handler does not catch. The two save functions also handle OSError differently: save_markdown_map() raises it, save_json_map() logs it.
│           ├── Maintenance Flag: Volatile
│           ├── Architectural Role: Service
│           └── Refactoring Suggestions: The last-sibling check in format_tree_lines() rescans forward from every item, which is quadratic on large repos. A single reverse pass could precompute is_last. The `critical_dependencies` handling should also catch TypeError, or parse the value once upstream.
├── tests/
│   ├── test_cache_manager.py (Python)
│   │   ├── Description: Tests for the repository-local SQLite cache: schema creation and migration in load_cache(), CacheError messages for unwritable or corrupt files, relative_cache_key() normalization, prune_cache() row removal, and load_cache_snapshot() as an in-memory, read-only copy.
│   │   ├── Developer Consideration: The unwritable-directory tests rely on chmod(0o555), which does not block writes when run as root or on Windows, so they can fail or pass wrongly there. The snapshot tests assert the cache file's bytes and directory listing stay unchanged.
│   │   ├── Maintenance Flag: Stable
│   │   ├── Architectural Role: Test
│   │   ├── Refactoring Suggestions: Skip the two chmod-based tests (unwritable directory, read-only snapshot) with pytest.mark.skipif when running as root or on Windows. Otherwise they give false results where permissions are not enforced.
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.cache_manager: Module under test: load_cache, load_cache_snapshot, prune_cache, relative_cache_key, CacheError, CACHE_FILE_NAME.
│   │   ├──   - pytest: Provides the tmp_path and caplog fixtures and pytest.raises for the CacheError assertions.
│   │   └──   - sqlite3: Seeds legacy-schema databases and inspects columns via PRAGMA table_info.
│   ├── test_cli_handler.py (Python)
│   │   ├── Description: Tests for the repo-map CLI orchestration in RepoMapApp: disclosure-prompt consent, dry-run, cache keying by relative path, model and contract-version invalidation, concurrent analysis with failure isolation, fatal OpenRouter errors, report writing, and console exit codes. Runs in-process and as a subprocess.
│   │   ├── Developer Consideration: Tests monkeypatch module-level names in cli_module and llm_module (rate_limited_api_call, resolve_response_format, get_llm_descriptions, tqdm, ANALYSIS_CONTRACT_VERSION), so renaming or re-importing those names in the source silently breaks the stubs. The subprocess tests rely on cwd and env isolation in _run_cli.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Refactoring Suggestions: The repeated setup (load_cache, RepoMapApp(), set cache_conn, close) recurs across many tests. A fixture or context manager yielding an app with an open cache would remove the leaked-connection risk when an assertion fails before close().
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.cli_handler: Provides RepoMapApp, plan_upload, summarize_repo and the disclosure prompt under test.
│   │   ├──   - repo_map.llm_service: Source of get_llm_descriptions, AnalysisOutcome, FatalOpenRouterError and the contract version, which tests stub.
│   │   ├──   - repo_map.cache_manager: Supplies load_cache, CACHE_FILE_NAME and CacheError for inspecting and seeding the SQLite cache.
│   │   ├──   - repo_map.main: run_main is invoked to verify console-script exit statuses 0, 1 and 130.
│   │   ├──   - aiohttp: ClientSession._request is patched to refuse so no test reaches the network.
│   │   └──   - pytest: Test runner supplying fixtures, parametrize and raises.
│   ├── test_code_parser.py (Python)
│   │   ├── Description: Pytest suite for repo_map.code_parser. It checks get_structure, get_imports and get_module_docstring across Python, JavaScript, TypeScript, Java and C#, including comment and string stripping, brace-scope tracking, import forms, and empty or missing files.
│   │   ├── Developer Consideration: Tests pin the exact output of regex/brace-scanning heuristics, including list order and cases like a Java member after a class body being reported as a top-level function. Parser changes can break them even when the new output is arguably better.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Refactoring Suggestions: Many near-identical Java and C# tests (sequential classes, prose in comments, string literals, nested blocks) could be merged with pytest.mark.parametrize over (language, extension, source, expected). That would remove duplication and keep the two language parsers' coverage aligned.
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.code_parser: Provides get_structure, get_imports and get_module_docstring, the functions under test.
│   │   └──   - pathlib.Path: Used by the _write helper and tmp_path fixtures to create source fixtures.
│   ├── test_config.py (Python)
│   │   ├── Description: Tests Settings and load_settings() precedence (shell over .env, defaults, blank key, bad API_SEMAPHORE_LIMIT) and verify that a planted .env cannot change the OpenRouter chat or models URLs, HTTP-Referer or X-Title headers, using a fake aiohttp session.
│   │   ├── Developer Consideration: Request tests monkeypatch llm_module.aiohttp.ClientSession and llm_module.settings, so they depend on llm_service importing aiohttp as a module attribute and reading a module-level `settings`; changing either breaks them silently. Fixture scratch_cwd must clear all REPO_MAP_ENV_VARS, so new settings need adding to that tuple.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.config: Provides Settings, load_settings and ConfigurationError, the code under test.
│   │   ├──   - repo_map.llm_service: rate_limited_api_call and resolve_response_format are run against a fake session to inspect URLs and headers.
│   │   ├──   - pydantic: ValidationError is the expected failure for malformed or out-of-range settings.
│   │   └──   - pytest: Fixtures, monkeypatch, tmp_path and parametrization drive the tests.
│   ├── test_file_scanner.py (Python)
│   │   ├── Description: Tests for repo_map.file_scanner: summarize_repo traversal order, gitignore and nested .gitignore handling, .git/info/exclude, default credential-file exclusions, language detection, cache hydration with relative keys, and load_source_snapshot rules (size limit, UTF-8, NUL bytes, private-key header rejection).
│   │   ├── Developer Consideration: PEM header strings are built from fragments (_PEM_BEGIN, _pem_lines) so this file never contains a literal key header; the repo scans itself and a literal would make it ineligible for upload. Keep that when adding fixtures.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.file_scanner: Module under test: summarize_repo, load_source_snapshot, MAX_SOURCE_BYTES, _detect_language; os.scandir is monkeypatched.
│   │   ├──   - repo_map.cache_manager: load_cache supplies the SQLite connection that summarize_repo uses for cache lookups.
│   │   ├──   - repo_map.models: SUPPORTED_TEXT_LANGUAGES is checked against detected languages for credential suffixes.
│   │   └──   - pytest: Test runner supplying tmp_path, monkeypatch, caplog and parametrize.
│   ├── test_known_issues.py (Python)
│   │   ├── Description: Regression tests for four tracked repo-map defects: the LLM prompt must include the target file's source marked untrusted, failed analyses must stay pending, class-only and data files must be eligible for enrichment, and the parser must keep async-function and JavaScript class/function scope.
│   │   ├── Developer Consideration: The pending-after-failure test monkeypatches cli_module.get_llm_descriptions and resolve_response_format, and calls the private llm_module._build_user_prompt. Renaming those, changing their signatures, or importing them differently inside cli_handler breaks the tests silently or noisily.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.cli_handler: Supplies RepoMapApp and plan_upload, whose pending-file planning and LLM enrichment are tested.
│   │   ├──   - repo_map.llm_service: Provides _build_user_prompt and the AnalysisOutcome enum, whose FAILURE value is returned by the patched analysis.
│   │   ├──   - repo_map.file_scanner: load_source_snapshot and summarize_repo build the snapshots and structures the tests feed into planning.
│   │   ├──   - repo_map.cache_manager: load_cache opens the per-repository SQLite connection used to plan uploads.
│   │   └──   - repo_map.code_parser: get_structure is checked for Python async functions and JavaScript class/function scoping.
│   ├── test_llm_service.py (Python)
│   │   ├── Description: Tests for repo_map.llm_service: prompt and layout construction, structured response parsing and normalization, the strict JSON-schema response format and endpoint lookup, and OpenRouter error classification, retry, backoff, and cancellation, run against a fake aiohttp session.
│   │   ├── Developer Consideration: Tests monkeypatch llm_module.aiohttp.ClientSession, llm_module.asyncio.sleep, rate_limited_api_call, and module globals like rate_limiter and MAX_LAYOUT_ENTRIES, so renaming or restructuring those names in llm_service breaks them. Backoff expectations (5, 10, Retry-After) are hard-coded.
│   │   ├── Maintenance Flag: Volatile
│   │   ├── Architectural Role: Test
│   │   ├── Refactoring Suggestions: Several tests (stale snapshot, unfamiliar payload shape, invalid results) rebuild the file_data dict inline instead of calling the existing _analysis_target helper, so the dict shape is duplicated and drifts if the scanner's fields change.
│   │   ├── Critical Dependencies:
│   │   ├──   - repo_map.llm_service: Module under test: parsing, prompts, retry logic, error classes, response formats.
│   │   ├──   - repo_map.file_scanner: load_source_snapshot supplies the hashes that the staleness guard checks.
│   │   ├──   - aiohttp: Its ClientSession is replaced by a fake; its error types are raised.
│   │   └──   - pytest: Test runner with parametrization, monkeypatch, caplog, and tmp_path fixtures.
│   └── test_report_generator.py (Python)
│       ├── Description: Tests for repo_map.report_generator: exact tree-line output from format_tree_lines (connectors, ancestor bars, metadata lines, whitespace collapsing, fence defusing), save_markdown_map and save_json_map file output, plus a timing check that rendering scales linearly.
│       ├── Developer Consideration: Tests pin exact box-drawing strings and the ` ` ` fence-defusing format, so any rendering change breaks many assertions at once. test_format_tree_lines_scales_linearly is wall-clock based (large < small * 8) and can flake on a loaded CI machine.
│       ├── Maintenance Flag: Volatile
│       ├── Architectural Role: Test
│       ├── Refactoring Suggestions: The `structure` fixture in test_format_tree_lines_renders_metadata and test_format_tree_lines_renders_metadata_exact_lines is duplicated verbatim; build it once in a helper or pytest fixture so the two tests cannot drift apart.
│       ├── Critical Dependencies:
│       ├──   - repo_map.report_generator: Provides format_tree_lines, save_json_map and save_markdown_map, the functions under test.
│       └──   - pytest: Test runner; supplies tmp_path, monkeypatch and pytest.raises.
├── .env.example (None)
├── .gitignore (Git)
│   ├── Description: Git ignore rules for the repo-map project: OS/editor files, Python build and tool caches, node_modules for the Bun changelog script, .env secrets, local agent tooling, and repo-map's own cache and report output.
│   ├── Developer Consideration: `.claude/*` is ignored with only `!.claude/skills/` re-included, and `.agents/*` re-includes `!.agents/skills`, so only the skills directories are tracked; other files under .claude (e.g. settings) stay local. `.env.*` is ignored except `.env.example`.
│   ├── Maintenance Flag: Stable
│   └── Architectural Role: Tooling
├── .python-version (None)
├── CHANGELOG.md (Markdown)
│   ├── Description: Top-level release index for the repo-map project, listing versions 0.0.1 through 0.14.0 newest first. Each entry has a date, a one-line summary, Breaking/Security tags, and a link to its per-version file under changelog/.
│   ├── Developer Consideration: Entries look generated from changelog/<minor>.x/<version>.md by scripts/build-changelog.ts, so hand edits here are likely overwritten; add or change the per-version file and rerun the build instead.
│   ├── Maintenance Flag: Generated
│   ├── Architectural Role: Other
│   ├── Critical Dependencies:
│   ├──   - scripts/build-changelog.ts: Presumed generator that builds this index from the per-version files under changelog/.
│   └──   - changelog/template.md: Template for new per-version entries that this index links to.
├── CLAUDE.md (Markdown)
│   ├── Description: Agent-facing guide for the repo-map CLI: stack, the `make check` gate, the pipeline from `main.run_main()` to the report, module roles, and the rules for prompts, upload eligibility, cache keys, dependency cooldown, and the generated changelog.
│   ├── Developer Consideration: Changing `SYSTEM_PROMPT`, `ANALYSIS_SCHEMA`, or the fields `_first_invalid_field()` validates requires bumping `ANALYSIS_CONTRACT_VERSION` in `llm_service.py`; otherwise cached analyses made under the old prompt keep being reused.
│   ├── Maintenance Flag: Volatile
│   └── Architectural Role: Other
├── LICENSE (None)
├── Makefile (Make)
│   ├── Description: Developer task runner defining `check` (lint, typecheck, test, lockfile check, changelog index check), `lint`, `format`, `typecheck`, `test`, `changelog`, and `list-skills`, all through uv, ruff, pyright, pytest and bun.
│   ├── Developer Consideration: `check` is the release gate and also runs `uv lock --check` and `bun scripts/build-changelog.ts --check`, so a stale uv.lock or an out-of-date CHANGELOG.md fails it; run `make changelog` after editing changelog/*.md. `unexport PYTHONOPTIMIZE` keeps pytest asserts alive.
│   ├── Maintenance Flag: Stable
│   ├── Architectural Role: Tooling
│   ├── Critical Dependencies:
│   ├──   - uv: Runs ruff, pyright and pytest in the project environment and verifies uv.lock.
│   ├──   - bun: Executes scripts/build-changelog.ts to build or check CHANGELOG.md.
│   ├──   - scripts/build-changelog.ts: Generates and validates CHANGELOG.md from changelog/<major.minor>.x files.
│   ├──   - ruff: Provides linting and format checking in lint and format targets.
│   └──   - pyright: Provides the static type checking in the typecheck target.
├── README.md (Markdown)
│   ├── Description: Project README for repo-map, a CLI that sends eligible files to an OpenRouter model and writes a Markdown tree of per-file notes. Covers install, env settings, CLI options, exit codes, SQLite caching, default ignore rules, upload privacy, and development workflow.
│   ├── Developer Consideration: The README states exact behavior (defaults, exit codes, ignore patterns, 64 KiB limit, ANALYSIS_CONTRACT_VERSION bump rule) that duplicates code in config.py, file_scanner.py, and llm_service.py; changing those without editing this file leaves the docs wrong.
│   ├── Maintenance Flag: Volatile
│   └── Architectural Role: Other
├── pyproject.toml (TOML)
│   ├── Description: Packaging and tooling manifest for the repo-map CLI (v0.14.0). It declares runtime dependencies, the `repo-map` console script pointing to repo_map.main:run_main, the uv_build backend, and ruff, pyright and pytest settings.
│   ├── Developer Consideration: The version lives here and must be bumped in step with the changelog/ and CHANGELOG.md release flow. `[tool.uv] exclude-newer = "3 days"` makes dependency resolution depend on the current date, so a new lock or `uv add` can refuse very recent releases.
│   ├── Maintenance Flag: Volatile
│   ├── Architectural Role: Tooling
│   ├── Critical Dependencies:
│   ├──   - uv_build: Build backend that packages src/repo_map; pinned to >=0.13.7,<0.14.0.
│   ├──   - aiohttp: Async HTTP client, used for LLM API calls.
│   ├──   - pydantic-settings: Loads application settings from environment and .env files.
│   ├──   - pathspec: Gitignore-style pattern matching when scanning repository files.
│   └──   - pydantic: Data models and validation for the repo_map package.
└── uv.lock (None)
└────────────── 
```
````
