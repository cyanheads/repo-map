# Changelog

All notable changes to this project. Each entry links to its full per-version file in [changelog/](changelog/).

## [0.14.0](changelog/0.14.x/0.14.0.md) — 2026-10-20 · 🛡️ Security

Rewritten per-file analysis prompt with a smaller, cacheable request, on anthropic/claude-sonnet-5.5 by default

## [0.13.1](changelog/0.13.x/0.13.1.md) — 2026-08-13

Verified skill-mirror gate: the check command compares skill mirrors by content hash without writing, and fails on a missing or drifted mirror file

## [0.13.0](changelog/0.13.x/0.13.0.md) — 2026-08-13

Cache identity fingerprinting: a stored analysis is reused only when the file hash, model, and analysis contract version all match the current run

## [0.12.0](changelog/0.12.x/0.12.0.md) — 2026-08-13

Portable, self-pruning cache keyed by repository-relative path, with a rel_path field on scanned files and a pruning pass that drops stale rows

## [0.11.1](changelog/0.11.x/0.11.1.md) — 2026-08-13

Parser scope fixes and enrichment isolation: class patterns no longer match prose, Java and C# class scope closes correctly, and one malformed LLM response no longer aborts the enhancement pass

## [0.11.0](changelog/0.11.x/0.11.0.md) — 2026-08-13

Startup and consent hardening: invalid configuration and corrupt cache files exit with one actionable line, and the upload prompt discloses what is sent and requires an explicit yes

## [0.10.2](changelog/0.10.x/0.10.2.md) — 2026-08-13

Parser extraction fixes: async Python functions, JavaScript class scope, leading-header docstrings, and common import and method declarations

## [0.10.1](changelog/0.10.x/0.10.1.md) — 2026-08-13

Tree rendering fixes: correct indentation prefixes for nested entries and single-line model text so descriptions cannot break the tree or the Markdown report

## [0.10.0](changelog/0.10.x/0.10.0.md) — 2026-08-13 · ⚠️ Breaking · 🛡️ Security

OpenRouter credential and upload-exclusion hardening: the endpoint and request headers are fixed constants instead of environment settings, and default exclusions cover more credential-bearing formats

## [0.9.1](changelog/0.9.x/0.9.1.md) — 2026-08-13

Transitive dependency bump: platformdirs 4.11.2 to 4.11.3 via the dev toolchain, with no call-site changes

## [0.9.0](changelog/0.9.x/0.9.0.md) — 2026-08-13

Source-grounded analysis: each eligible file's complete source is sent to the LLM as delimited untrusted data, responses are validated before caching, and enrichment eligibility follows the scanner's source policy

## [0.8.3](changelog/0.8.x/0.8.3.md) — 2026-08-13

Bounded concurrent analysis: file analyses are scheduled together and consumed in completion order, and HTTP 429 backoff happens outside the concurrency permit

## [0.8.2](changelog/0.8.x/0.8.2.md) — 2026-08-13

Symlinks are skipped during scanning, and language detection restores exact-filename and compound-suffix mappings

## [0.8.1](changelog/0.8.x/0.8.1.md) — 2026-08-13

Baseline test suite, repaired development-command dispatcher, project-local workflows, dependency updates, and removal of unused tree-sitter dependencies

## [0.8.0](changelog/0.8.x/0.8.0.md) — 2026-04-25 · ⚠️ Breaking

LLM contract reworked to a single JSON object with a closed architectural_role enum, default model changed to anthropic/claude-sonnet-4.6, and code_quality_score removed

## [0.7.1](changelog/0.7.x/0.7.1.md) — 2026-04-25

pathspec GitIgnoreSpec migration for closer Git ignore fidelity, a maintenance skill with automatic mirror sync, and in-range dependency updates

## [0.7.0](changelog/0.7.x/0.7.0.md) — 2026-03-08

Default model updated to google/gemini-3-flash-preview, CLAUDE.md added, dependency updates, Black formatting across all modules, and modernized type hints

## [0.5.0](changelog/0.5.x/0.5.0.md) — 2025-09-29

main.py split into cli_handler, logging_utils, and report_generator modules, plus a refreshed README and an example output

## [0.4.0](changelog/0.4.x/0.4.0.md) — 2025-09-29

Comprehensive per-file LLM analysis covering developer considerations, architectural roles, and security assessments, with an extended cache schema and richer tree output

## [0.3.0](changelog/0.3.x/0.3.0.md) — 2025-09-29

README updated to match the current default model, .gitignore handling, and main.py description

## [0.2.0](changelog/0.2.x/0.2.0.md) — 2025-09-29

RepoMapApp class refactor of main.py, progress-aware logging, more robust ignore-pattern handling, an APIRateLimiter, and AGENTS.md

## [0.1.0](changelog/0.1.x/0.1.0.md) — 2025-09-29

Poetry migration, modular project structure, SQLite caching, and multi-language code parsing

## [0.0.1](changelog/0.0.x/0.0.1.md) — 2025-09-28

Initial commit
