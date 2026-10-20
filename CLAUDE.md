# repo-map

Python CLI, published on PyPI as `repo-map`, that walks a repository, sends each eligible file to an OpenRouter model, and writes the analyses as a Markdown tree to `<repository>_repo_map.md`. Analyses are cached in SQLite inside the analyzed repository, so a rerun only sends files that changed.

**Orientation:** this file is the behavioral layer for agents. `README.md` has install, configuration, the upload rules, and the default exclusions. `AGENTS.md` is a symlink to this file. Project workflows are skills in `.claude/skills/`, and `.agents/skills` links to the same tree.

## Stack & gate

Python ≥3.12 (classifiers 3.12–3.14), uv for everything, `uv_build` backend; aiohttp, pydantic-settings, and pathspec at runtime. Ruff lints and formats, pyright checks types, pytest runs the suite.

The gate is `make check`: `ruff check`, `ruff format --check`, pyright, pytest, `uv lock --check`, and `bun scripts/build-changelog.ts --check`. Bun is needed only for the changelog builder. The Makefile unsets `PYTHONOPTIMIZE`, because `-O` strips the asserts pytest reports through.

## Running it

| Mode | Command | What it does |
|:---|:---|:---|
| Setup | `uv sync` | Creates `.venv` with runtime and dev dependencies from `uv.lock` |
| Run | `uv run repo-map <path> [-y] [--model M] [--concurrency N]` | Analyzes `<path>`. Needs `OPENROUTER_API_KEY` from the environment or a `.env` in the working directory (`.env.example`) |
| Gate | `make check` | Everything that must pass before a commit |
| Format | `make format` | `ruff check --fix`, then `ruff format` |
| One stage | `make lint` · `make typecheck` · `make test` | A single part of the gate |
| Changelog | `make changelog` | Regenerates `CHANGELOG.md` from `changelog/` |
| Skills | `make list-skills` | Each project skill and its trigger |

A real run spends OpenRouter credit and uploads source. Live checks use a small public fixture, never a private repository.

## Architecture

`main.run_main()` → `cli_handler.RepoMapApp` loads settings and the cache → `file_scanner.summarize_repo()` walks the tree and extracts structure → `llm_service.get_llm_descriptions()` analyzes each pending file → `report_generator` prints the tree and writes the report.

| Path | Role |
|:---|:---|
| `src/repo_map/main.py` | Entry point. Imports `RepoMapApp` lazily so a settings error is caught; exit 1 on errors, 130 on Ctrl-C |
| `src/repo_map/cli_handler.py` | Arguments, upload consent prompt, cache reads and writes, the analysis pass |
| `src/repo_map/config.py` | `Settings` from env and `.env`; the OpenRouter URL and headers are module constants |
| `src/repo_map/file_scanner.py` | Walks the tree, applies `DEFAULT_IGNORE_PATTERNS` and the root `.gitignore`, hashes files, decides upload eligibility (`load_source_snapshot()`) |
| `src/repo_map/models.py` | Extension → language map, `NON_TEXT_LANGUAGES`, `MAX_SOURCE_BYTES` |
| `src/repo_map/code_parser.py` | Structure, imports, and docstrings: `ast` for Python, regular expressions for Java, JavaScript, TypeScript, and C# |
| `src/repo_map/llm_service.py` | `SYSTEM_PROMPT`, request assembly (`_build_messages()`), the OpenRouter client with retries and a concurrency semaphore, response validation |
| `src/repo_map/cache_manager.py` | SQLite cache: schema migration, relative-path keys, pruning |
| `src/repo_map/report_generator.py` | Console tree, `<repository>_repo_map.md`, `.repo_map_structure.json` |
| `tests/` | One module per source module, plus `test_known_issues.py` |

## The rules that matter

- **Bump `ANALYSIS_CONTRACT_VERSION`** in `src/repo_map/llm_service.py` whenever the meaning of `SYSTEM_PROMPT`, the request shape, or the field set `_first_invalid_field()` validates changes. A cached analysis is reused only while the file's SHA-256 hash, the model, and this revision all match, so a missed bump keeps serving analyses made under the old prompt. A refactor that changes none of those keeps the revision.
- **The response contract is all-or-nothing.** All seven fields (`description`, `developer_consideration`, `maintenance_flag`, `critical_dependencies`, `architectural_role`, `refactoring_suggestions`, `security_assessment`) must be present and well-typed. Anything less is a failed analysis: nothing is cached and the file stays pending for the next run.
- **Everything from the analyzed repository is untrusted.** File names, related-file notes, and source reach the model declared as data, and the source sits between marker lines tagged with a digest of that source so its content cannot close the block early. Any prompt change keeps both properties.
- **Upload eligibility is a security boundary.** Source is sent only for a regular file (never a symlink) in `SUPPORTED_TEXT_LANGUAGES`, at most `MAX_SOURCE_BYTES`, valid UTF-8, free of NUL bytes, and not excluded by `DEFAULT_IGNORE_PATTERNS` or the root `.gitignore`. A new binary or media format goes in `NON_TEXT_LANGUAGES` as well as `SUPPORTED_LANGUAGES`. A change to the default exclusions also updates the README's exclusions table.
- **Transport values are not settings.** `OPENROUTER_API_URL`, `HTTP_REFERER`, and `APP_NAME` stay module constants in `config.py`. `.env` loads from the working directory, so an analyzed repository's own `.env` must not be able to redirect requests, and the API key with them.
- **Cache keys are repository-relative, forward-slash paths**, so the database survives renames and clones. Rows under the older absolute-path key read as misses and are pruned at the end of the run. A new persisted field touches the schema in `cache_manager.py`, the read and write in `cli_handler.py`, and the tree in `report_generator.py`.
- **`[tool.uv] exclude-newer = "3 days"` is a supply-chain cooldown.** Never lower or remove it. An urgent fix gets a per-package `exclude-newer-package` entry in `pyproject.toml`, removed once the release ages past the window; the `--exclude-newer-package` CLI flag writes the override into `uv.lock` and breaks `uv lock --check`.
- **`CHANGELOG.md` is generated.** Author `changelog/<major.minor>.x/<version>.md` from `changelog/template.md`, then run `make changelog`.

## Where things live

| Path | What |
|:---|:---|
| `.claude/skills/` | Project workflows, listed below |
| `changelog/`, `scripts/build-changelog.ts` | Per-version release notes and the `CHANGELOG.md` index builder |
| `examples/example.md` | Real repo-map output on this repository; regenerate it when the prompt or the report format changes |
| `.env.example` | Every setting the CLI reads. `.env` is gitignored |
| `.github/` | Issue forms, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `workflows/codeql.yml`, the only workflow: verification runs locally |
| `dist/` | Build output, gitignored. Each release builds into `dist/<version>/` |

| Skill | Use it for |
|:---|:---|
| `code-simplifier` | Behavior-preserving cleanup of a change set or the codebase |
| `git-wrapup` | Landing work as logical commits, the version bump, the changelog entry, and the annotated tag |
| `release-and-publish` | Pushing, publishing to PyPI, and the GitHub release, after `git-wrapup` |
| `maintenance` | Dependency updates and their fallout |
| `report-issue-local` | Filing a bug or feature request against `cyanheads/repo-map` |

## Triggers

| When the ask is | Do this |
|:---|:---|
| "update deps", "run maintenance" | `.claude/skills/maintenance/SKILL.md` |
| "commit", "wrap up" | `.claude/skills/git-wrapup/SKILL.md` |
| "release", "publish", "ship it to PyPI" | `git-wrapup`, then `.claude/skills/release-and-publish/SKILL.md` |
| "file an issue", "report this" | `.claude/skills/report-issue-local/SKILL.md` |
| "clean this up", "simplify" | `.claude/skills/code-simplifier/SKILL.md` |
| "support <language>" | Map its extensions in `SUPPORTED_LANGUAGES` (`models.py`), add a branch in `code_parser.py`, cover it in `tests/test_code_parser.py` and `tests/test_file_scanner.py` |
| "change the prompt" | Edit `SYSTEM_PROMPT` or `_build_messages()`, bump `ANALYSIS_CONTRACT_VERSION`, verify on a public fixture, regenerate `examples/example.md` |

## Related

- PyPI: https://pypi.org/project/repo-map/
- OpenRouter API reference: https://openrouter.ai/docs/api_reference/overview

## Commit stance

Commit, tag, push, and publish only when the maintainer asks. Commits land directly on `main`, and `make check` passes before each one.
