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
| Run | `uv run repo-map <path> [-y] [--dry-run] [--model M] [--concurrency N]` | Analyzes `<path>`. Needs `OPENROUTER_API_KEY` from the environment or a `.env` in the working directory (`.env.example`), except under `--dry-run`, which lists what would be sent and touches nothing |
| Gate | `make check` | Everything that must pass before a commit |
| Format | `make format` | `ruff check --fix`, then `ruff format` |
| One stage | `make lint` · `make typecheck` · `make test` | A single part of the gate |
| Changelog | `make changelog` | Regenerates `CHANGELOG.md` from `changelog/` |
| Skills | `make list-skills` | Each project skill and its trigger |

A real run spends OpenRouter credit and uploads source. Live checks use a small public fixture, never a private repository.

## Architecture

`main.run_main()` → `cli_handler.RepoMapApp.run()` → `file_scanner.summarize_repo()` walks the tree against an in-memory copy of the cache (`cache_manager.load_cache_snapshot()`) → `plan_upload()` splits pending files from those with a reusable analysis → the consent prompt states the counts (`--dry-run` lists the files and stops) → `load_cache()` opens the real cache → `llm_service.resolve_response_format()` once, then `get_llm_descriptions()` per pending file → `report_generator` prints the tree and writes the report → `run()` returns the exit status. Nothing is written to the target or sent before consent.

| Path | Role |
|:---|:---|
| `src/repo_map/main.py` | Entry point. Imports `RepoMapApp` lazily so a settings error is caught; `run_main()` returns the exit status the console script exits with: `run()`'s status, 1 for a settings or unexpected error, 130 on Ctrl-C |
| `src/repo_map/cli_handler.py` | Arguments, the pre-consent scan and `UploadPlan` (`plan_upload()`), the counted consent prompt and `--dry-run`, cache writes, the analysis pass (a `FatalOpenRouterError` cancels the analyses in flight and stops the run), exit statuses: 0 only when every pending file was analyzed and the report written |
| `src/repo_map/config.py` | `Settings` from env and `.env`; the OpenRouter chat and model-endpoints URLs and the request headers are module constants |
| `src/repo_map/file_scanner.py` | Walks the tree, applies `DEFAULT_IGNORE_PATTERNS` plus the root `.gitignore`, and every `.gitignore` and `.git/info/exclude` from the enclosing work tree's top down (`_IgnoreRules`, parsed in-process, never by running `git`), hashes files, decides upload eligibility (`load_source_snapshot()`) |
| `src/repo_map/models.py` | Extension → language map, `NON_TEXT_LANGUAGES`, `MAX_SOURCE_BYTES` |
| `src/repo_map/code_parser.py` | Structure, imports, and docstrings: `ast` for Python, regular expressions for Java, JavaScript, TypeScript, and C# |
| `src/repo_map/llm_service.py` | `SYSTEM_PROMPT` and `ANALYSIS_SCHEMA`, request assembly (`_build_messages()`), the per-run `response_format` choice (`resolve_response_format()`: strict JSON schema when one of the model's endpoints supports structured outputs, `json_object` otherwise), the OpenRouter client with a concurrency semaphore, error classification (`_classify_error()`: retry, fail the file, or raise `FatalOpenRouterError` to end the run), response validation |
| `src/repo_map/cache_manager.py` | SQLite cache: schema migration, relative-path keys, pruning, and the read-only in-memory snapshot read before consent |
| `src/repo_map/report_generator.py` | Console tree, `<repository>_repo_map.md`, `.repo_map_structure.json` |
| `tests/` | One module per source module, plus `test_known_issues.py` |

## The rules that matter

- **Bump `ANALYSIS_CONTRACT_VERSION`** in `src/repo_map/llm_service.py` whenever the meaning of `SYSTEM_PROMPT`, the request shape (`ANALYSIS_SCHEMA` included), or the field set `_first_invalid_field()` validates changes. A cached analysis is reused only while the file's SHA-256 hash, the model, and this revision all match, so a missed bump keeps serving analyses made under the old prompt. A refactor that changes none of those keeps the revision.
- **The response contract is all-or-nothing.** All seven fields (`description`, `developer_consideration`, `maintenance_flag`, `critical_dependencies`, `architectural_role`, `refactoring_suggestions`, `security_assessment`) must be present and well-typed. Anything less is a failed analysis: nothing is cached and the file stays pending for the next run. `critical_dependencies` travels as `[{"name", "reason"}]`, because strict schemas allow no open-ended keys, and is stored as `{name: reason}` JSON, so the cache and report formats don't see the wire shape.
- **Everything from the analyzed repository is untrusted.** File names, related-file notes, and source reach the model declared as data, and the source sits between marker lines tagged with a digest of that source so its content cannot close the block early. Any prompt change keeps both properties.
- **Upload eligibility is a security boundary.** Source is sent only for a regular file (never a symlink) in `SUPPORTED_TEXT_LANGUAGES`, at most `MAX_SOURCE_BYTES`, valid UTF-8, free of NUL bytes and of a PEM private-key header, and not excluded by `DEFAULT_IGNORE_PATTERNS`, the root `.gitignore`, or any nested `.gitignore` or `.git/info/exclude` the walk reaches. Only the root `.gitignore` can re-include a default exclusion; nested rules only narrow. The scanner never runs `git` against the analyzed repository, whose config could execute commands. A new binary or media format goes in `NON_TEXT_LANGUAGES` as well as `SUPPORTED_LANGUAGES`. A change to the default exclusions also updates the README's exclusions table.
- **Transport values are not settings.** `OPENROUTER_API_URL`, `OPENROUTER_MODELS_URL`, `HTTP_REFERER`, and `APP_NAME` stay module constants in `config.py`. `.env` loads from the working directory, so an analyzed repository's own `.env` must not be able to redirect requests, and the API key with them.
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
| "support <language>" | Map its extensions in `SUPPORTED_LANGUAGES` (`models.py`), add a branch in `code_parser.py` when structure or imports are worth extracting (Go, Ruby, Rust, C, TOML, and Make are analyzed from source alone), cover it in `tests/test_code_parser.py` and `tests/test_file_scanner.py`. A text format that carries credentials also gets a `DEFAULT_IGNORE_PATTERNS` entry and a README exclusions row |
| "change the prompt" | Edit `SYSTEM_PROMPT` or `_build_messages()`, bump `ANALYSIS_CONTRACT_VERSION`, verify on a public fixture, regenerate `examples/example.md` |

## Related

- PyPI: https://pypi.org/project/repo-map/
- OpenRouter API reference: https://openrouter.ai/docs/api_reference/overview

## Commit stance

Commit, tag, push, and publish only when the maintainer asks. Commits land directly on `main`, and `make check` passes before each one.
