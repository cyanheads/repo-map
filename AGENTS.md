# AGENTS Cheat Sheet for repo-map

## Project TL;DR
- CLI tool that scans a target repository, extracts structural metadata, then asks an OpenRouter LLM to summarize interesting files.
- Caching layer (`.repo-map-cache.db`) avoids re-sending unchanged files; pre-enhanced results land in `.repo_map_structure.json`.
- Output is both a console tree and `<repo_name>_repo_map.md` written at the root of the analyzed repo.

## Local Setup
- Requirements: Python 3.12+, Poetry, OpenRouter API key.
- Install dependencies:
  ```bash
  poetry install
  ```
- Provide credentials before running (shell rc or `.env`):
  ```bash
  export OPENROUTER_API_KEY=<token>
  ```
- Optional overrides via `.env` or env vars:
  - `OPENROUTER_MODEL_NAME` (defaults to `anthropic/claude-sonnet-4.6`)
  - `API_SEMAPHORE_LIMIT` to tune concurrent calls.

## Running repo-map
- Basic command (from project root):
  ```bash
  poetry run repo-map /path/to/repository
  ```
- Flags:
  - `-y/--yes` skips the interactive disclaimer prompt.
  - `--model` overrides the default OpenRouter model.
  - `--concurrency` throttles simultaneous LLM calls (default matches `settings.api_semaphore_limit`).
- First run creates `.repo-map-cache.db` and `.repo_map_structure.json` inside the inspected repo; add to ignore lists as needed.

## Execution Flow
1. `run_main()` launches the async entrypoint (`src/repo_map/main.py`).
2. `summarize_repo()` walks the tree, merges the repo root `.gitignore` with built-in defaults, records directories, and captures docstrings, imports, classes, functions, and constants for supported extensions (`src/repo_map/file_scanner.py`).
3. Cached hashes short-circuit unchanged files; fresh files have structure extracted via `code_parser.py` helpers before LLM enhancement.
4. `_enhance_summary_with_llm()` selects files the scanner marked `source_eligible`, queues them, and streams prompts through `llm_service.get_llm_descriptions()` to populate descriptions, developer considerations, maintenance flags, and key dependencies. Only a validated success is written to the cache; a failed analysis leaves the file pending for the next run.
5. Responses update in-memory structures + cache, `_print_tree()` logs an ASCII map, and `_save_markdown_map()` writes `<repo>_repo_map.md` to disk.

## LLM Expectations
The tool sends each file to the LLM with the repository tree as background and the target file's complete source as delimited, explicitly untrusted data, and expects a JSON object with these fields (defined in `SYSTEM_PROMPT` in `src/repo_map/llm_service.py`):

1.  **`description`** (string) — 1-2 sentences on the file's role and responsibility.
2.  **`developer_consideration`** (string | null) — The single most important thing a contributor needs to know: a non-obvious dependency, performance pitfall, security concern, or usage pattern.
3.  **`maintenance_flag`** — One of:
    - `Stable`: foundational; rarely changes.
    - `Volatile`: under active iteration (business logic, UI, configuration).
    - `Generated`: machine-produced; do not edit.
    - `Unknown`: insufficient context.
4.  **`critical_dependencies`** (object) — Map of import name → one-line justification. Empty object if none notable.
5.  **`architectural_role`** — One of: `Entrypoint`, `Configuration`, `Service`, `Data Model`, `Persistence`, `UI Component`, `API Route`, `Utility`, `Test`, `Tooling`, `Other`.
6.  **`refactoring_suggestions`** (string | null) — Concrete and actionable, or null if the file is sound.
7.  **`security_assessment`** (string | null) — Specific risk and mitigation, or null if no concern.

The OpenRouter request sets `response_format: {"type": "json_object"}` to enforce structured output. The parser tolerates ```json fences from models that add them despite the instruction.

## Key Modules
- `src/repo_map/main.py` – CLI orchestration, disclaimer handling, persistence of results.
- `src/repo_map/file_scanner.py` – repository walker, root `.gitignore` + default ignore aggregation, hash computation, and cache hydration.
- `src/repo_map/code_parser.py` – language-aware extraction for Python/Java/JS/TS/C# plus import and docstring helpers.
- `src/repo_map/llm_service.py` – OpenRouter client, concurrency semaphore, exponential backoff, response parsing, and generation of descriptions, developer considerations, maintenance flags, and key dependencies.
- `src/repo_map/cache_manager.py` – SQLite lifecycle and schema migrations.
- `src/repo_map/config.py` – Pydantic settings bound to env / `.env`; instantiates a singleton `settings` object.
- `src/repo_map/models.py` – extension-to-language lookup, plus the text/non-text split (`NON_TEXT_LANGUAGES`, `SUPPORTED_TEXT_LANGUAGES`) and `MAX_SOURCE_BYTES` that decide which files may have their source uploaded.

## Working With the Cache
- Located inside the target repo (`.repo-map-cache.db`) so analyses travel with the project.
- `path` holds the repository-relative key in forward-slash form; `file_info["path"]` stays absolute for disk reads and `file_info["rel_path"]` carries the key. A rename, clone, or checkout at another prefix keeps every entry valid.
- A row under the older absolute-path key matches no relative key, so it reads as a miss; nothing rewrites keys in place, because two prefixes reduce to one key and would violate the primary key.
- Each run ends with a prune (`prune_cache`) that deletes rows no scanned file claims — deleted files and leftover absolute-path keys alike.
- Schema upgrades run automatically on load; delete the file to force a clean regeneration.
- Descriptions, developer considerations, maintenance flags, key dependencies, imports, and functions are cached alongside hashes for reuse.
- `hash` column stores SHA-256 of each processed file; updating source without deleting cache still reprocesses because hashes change.

## Developer Workflows
- Install/update the Poetry environment: `poetry install`.
- Full local gate: `poetry run python scripts.py check` (skill sync, Ruff, Black, pytest with assertions enabled, Poetry metadata).
- Formatting: `poetry run python scripts.py format`.
- Focused lint/test commands: `poetry run python scripts.py lint` and `poetry run python scripts.py test`.
- Build distributions: `poetry build`.
- Discover project workflows: `poetry run python scripts.py list-skills`.
- Type hints: project exposes `py.typed`; ensure new modules remain typed to support consumers.

## Extending repo-map
- Add new file types via `SUPPORTED_LANGUAGES` in `src/repo_map/models.py`, and add the language to `NON_TEXT_LANGUAGES` when the format is binary or media, so its bytes are never sent to OpenRouter.
- Provide language-specific structure/import extraction by expanding switch logic in `src/repo_map/code_parser.py`.
- Adjust concurrency by modifying `Settings` defaults in `src/repo_map/config.py`. The OpenRouter endpoint and request headers are module constants there (`OPENROUTER_API_URL`, `HTTP_REFERER`, `APP_NAME`), deliberately not settings — `env_file` resolves against the working directory, so a `.env` in an analyzed repository must not be able to reach them.
- To persist additional metadata, alter both the cache schema (`cache_manager.py`) and tree serialization in `main.py`.

## Troubleshooting
- Missing API key: tool exits early; confirm `OPENROUTER_API_KEY` is exported before invoking.
- Rate limits: `llm_service.py` retries with exponential backoff; increase `--concurrency` cautiously.
- Incorrect tree contents: inspect generated `.repo_map_structure.json` before LLM enhancement to confirm scanner output.
- SSL issues: certifi bundle is enforced; stale cert stores on the host can still cause `SSLCertVerificationError`.
