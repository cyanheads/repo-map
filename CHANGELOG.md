# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.2] - 2026-08-13

### Fixed
- Skip file and directory symlinks before scanner sorting or classification, preventing outside-root source from entering a map and stopping cycle links from repeating traversal ([#9](https://github.com/cyanheads/repo-map/issues/9)).
- Detect languages by exact canonical filename followed by longest matching suffix, restoring `.gitignore`, `.envrc`, `Dockerfile`, and compound `.tfstate.backup` mappings without changing ordinary extension handling ([#5](https://github.com/cyanheads/repo-map/issues/5)).

## [0.8.1] - 2026-08-13

### Added
- A 25-test baseline suite covering the cache, CLI orchestration, parsers, scanner, LLM service, report generation, and development commands. Eight strict expected-failure regressions track open issues #5–#12 and make unexpected passes release blockers.
- Project-local maintenance, simplification, issue-reporting, and release workflows with synchronized Claude and Codex mirrors.
- Structured GitHub bug and feature request forms.

### Changed
- Repaired `scripts.py` as the development-command dispatcher for checks, formatting, tests, skill synchronization, and skill discovery; `poetry run python scripts.py check` is now the complete local gate.
- Updated direct dependencies: aiohttp 3.13.5 → 3.14.3, certifi 2026.4.22 → 2026.7.22, pathspec 1.1.0 → 1.1.1, pydantic 2.13.3 → 2.13.4, pydantic-settings 2.14.0 → 2.15.0, tqdm 4.67.3 → 4.70.0, black 26.3.1 → 26.5.1, pytest 9.0.3 → 9.1.1, and ruff 0.15.12 → 0.16.3.
- Simplified parser, scanner, retry-loop, and report-tree internals without changing behavior.
- Refreshed contributor guidance, command documentation, and examples against the verified project layout and release process.

### Removed
- Unused `tree-sitter`, `tree-sitter-languages`, and `types-tqdm` dependencies. repo-map's parsers use Python's AST and language-specific regex extraction, so the former tree-sitter ABI blocker does not require a migration.

## [0.8.0] - 2026-04-25

### Changed
- `llm_service.py`: reworked the LLM contract from a flat text key-value block to a single JSON object. The OpenRouter request now sets `response_format: {"type": "json_object"}` for structured output, and `parse_llm_response` ingests JSON (tolerating ```` ```json ```` fences from non-compliant models). The user prompt is now built by `_build_user_prompt`, which emits a cleaner indented tree using paths relative to the repo root.
- `llm_service.py`: closed-enum `architectural_role` with `_normalize_architectural_role`, snapping unknown roles to `Other`. `Service Layer` is now `Service`; added `Persistence`, `API Route`, `Test`, `Tooling`, `Other`.
- `config.py`: default model changed from `google/gemini-3-flash-preview` to `anthropic/claude-sonnet-4.6`.
- `report_generator.py`: dropped surrounding quotes from the `Developer Consideration:` tree line for cleaner output.
- `AGENTS.md`, `CLAUDE.md`, `README.md`, `examples/example.md`: documentation refreshed to match the new JSON schema, default model, and tightened architectural role values.

### Removed
- `code_quality_score` field across the LLM response schema, cache schema (`cache_manager.py`), insert/select paths (`cli_handler.py`, `file_scanner.py`), and tree rendering (`report_generator.py`). Existing caches retain the column harmlessly; new caches omit it.

## [0.7.1] - 2026-04-25

### Added
- `skills/maintenance/SKILL.md` — workflow for dependency updates, with auto-sync to `.agents/skills/` and `.claude/skills/` mirrors via `scripts/sync_skills.py` and a Claude Code `PostToolUse` hook in `.claude/settings.json`.
- `sync-skills` Poetry script (`poetry run sync-skills`).

### Changed
- `file_scanner.py`: migrated `pathspec.PathSpec.from_lines("gitwildmatch", …)` → `pathspec.GitIgnoreSpec.from_lines(…)` for closer fidelity to Git's actual ignore semantics (especially negation patterns inside otherwise-ignored directories). The `gitwildmatch` alias has been deprecated since pathspec 1.0.
- Updated dependencies (in-range): aiohttp 3.13.3→3.13.5, pathspec 1.0.4→1.1.0, pydantic 2.12.5→2.13.3, pydantic-settings 2.13.1→2.14.0, certifi 2026.2.25→2026.4.22, ruff 0.15.5→0.15.12, black 26.3.0→26.3.1, pytest 9.0.2→9.0.3, types-tqdm patch, plus transitive bumps.

### Deferred
- `tree-sitter` 0.21.3 → 0.25.2 not taken: `tree-sitter-languages` 1.10.2 (last released 2024-02) bundles binary wheels built against the old tree-sitter ABI and would break at runtime. Proper fix is migrating off `tree-sitter-languages` to per-language packages (`tree-sitter-python`, `tree-sitter-javascript`, etc.) — separate work.

## [0.7.0] - 2026-03-08

### Added
- `CLAUDE.md` for Claude Code project guidance.

### Changed
- Updated default LLM model from `google/gemini-2.5-flash-preview-09-2025` to `google/gemini-3-flash-preview`.
- Updated dependency versions: aiohttp, certifi, pathspec, tqdm, pydantic, pydantic-settings, types-tqdm.
- Fixed ruff config to use `[tool.ruff.lint]` section for lint rules.
- Applied Black formatting across all source modules.
- Modernized type hints: `Optional[X]` → `X | None`.
- Added `.python-version` to `.gitignore`.

## [0.5.0] - 2025-09-29

### Changed
- Refactored `main.py` by extracting logic into `cli_handler.py`, `logging_utils.py`, and `report_generator.py` for improved modularity.
- Updated `README.md` with improved formatting, badges, and clearer instructions.
- Bumped project version to `0.5.0`.

### Added
- `examples/example.md` to showcase a sample of the generated output.

## [0.4.0] - 2025-09-29

### Added
- Comprehensive LLM analysis including developer considerations, maintenance flags, critical dependencies, architectural roles, code quality scores, refactoring suggestions, and security assessments.
- Extended caching to store all new LLM-generated metadata.

### Changed
- `file_scanner.py`: Improved file processing to handle the new cached fields.
- `llm_service.py`: Updated system prompt to generate a more detailed and structured analysis for each file.
- `main.py`: Refactored to integrate the new LLM analysis and caching mechanisms, and enhanced the tree output to display the new metadata.
- `cache_manager.py`: Expanded the cache schema to include the new analysis fields.
- `AGENTS.md`: Updated documentation to reflect the new execution flow and module responsibilities.

## [0.3.0] - 2025-09-29

### Changed
- Updated `README.md` to reflect recent changes, including the default model name, `.gitignore` handling, and the description of `main.py`.

## [0.2.0] - 2025-09-29

### Changed
- Refactored `main.py` into a `RepoMapApp` class for better structure.
- Improved logging with a `TqdmLoggingHandler`.
- Enhanced `.gitignore` to include repo-map specific files.
- Updated `file_scanner.py` for more robust ignore pattern handling.
- Refactored `llm_service.py` with an `APIRateLimiter` class.

### Added
- `AGENTS.md` for developer onboarding.

## [0.1.0] - 2025-09-29

### Added
- Converted project from setuptools to Poetry for dependency management.
- Refactored monolithic script into a modular structure.
- Added support for caching with SQLite.
- Introduced multi-language support for code parsing.

## [0.0.1] - 2025-09-28

### Added
- Initial commit.
