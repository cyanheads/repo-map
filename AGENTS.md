# AGENTS Cheat Sheet for repo-map

## Project TL;DR
- CLI tool that scans a target repository, extracts structural metadata, then asks an OpenRouter LLM to summarize interesting files.
- Caching layer (`.repo-map-cache.db`) avoids re-sending unchanged files; pre-enhanced results land in `.repo_map_structure.json`.
- Output is both a console tree and `<repo_name>_repo_map.md` stored beside the analyzed repo.

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
  - `OPENROUTER_MODEL_NAME` (defaults to `google/gemini-2.5-flash-preview-09-2025`)
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
4. `_enhance_summary_with_llm()` selects files with structural info, queues them, and streams prompts through `llm_service.get_llm_descriptions()` to populate descriptions, developer considerations, maintenance flags, and key dependencies.
5. Responses update in-memory structures + cache, `_print_tree()` logs an ASCII map, and `_save_markdown_map()` writes `<repo>_repo_map.md` to disk.

## Key Modules
- `src/repo_map/main.py` – CLI orchestration, disclaimer handling, persistence of results.
- `src/repo_map/file_scanner.py` – repository walker, root `.gitignore` + default ignore aggregation, hash computation, and cache hydration.
- `src/repo_map/code_parser.py` – language-aware extraction for Python/Java/JS/TS/C# plus import and docstring helpers.
- `src/repo_map/llm_service.py` – OpenRouter client, concurrency semaphore, exponential backoff, response parsing, and generation of descriptions, developer considerations, maintenance flags, and key dependencies.
- `src/repo_map/cache_manager.py` – SQLite lifecycle and schema migrations.
- `src/repo_map/config.py` – Pydantic settings bound to env / `.env`; instantiates a singleton `settings` object.
- `src/repo_map/models.py` – extension-to-language lookup used by the scanner.

## Working With the Cache
- Located inside the target repo (`.repo-map-cache.db`) so analyses travel with the project.
- Schema upgrades run automatically on load; delete the file to force a clean regeneration.
- Descriptions, developer considerations, maintenance flags, key dependencies, imports, and functions are cached alongside hashes for reuse.
- `hash` column stores SHA-256 of each processed file; updating source without deleting cache still reprocesses because hashes change.

## Developer Workflows
- Activate virtualenv (if Poetry manages one): `poetry shell`.
- Formatting: `poetry run black src` (also exposed via `poetry run format`).
- Linting: `poetry run ruff check src`.
- Tests: add under `tests/` and run `poetry run pytest` (pytest already bundled).
- Type hints: project exposes `py.typed`; ensure new modules remain typed to support consumers.

## Extending repo-map
- Add new file types via `SUPPORTED_LANGUAGES` in `src/repo_map/models.py`.
- Provide language-specific structure/import extraction by expanding switch logic in `src/repo_map/code_parser.py`.
- Adjust concurrency or headers by modifying `Settings` defaults in `src/repo_map/config.py`.
- To persist additional metadata, alter both the cache schema (`cache_manager.py`) and tree serialization in `main.py`.

## Troubleshooting
- Missing API key: tool exits early; confirm `OPENROUTER_API_KEY` is exported before invoking.
- Rate limits: `llm_service.py` retries with exponential backoff; increase `--concurrency` cautiously.
- Incorrect tree contents: inspect generated `.repo_map_structure.json` before LLM enhancement to confirm scanner output.
- SSL issues: certifi bundle is enforced; stale cert stores on the host can still cause `SSLCertVerificationError`.
