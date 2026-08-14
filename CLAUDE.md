# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`repo-map` is a Python CLI tool that generates AI-enhanced repository summaries. It scans a target repository, extracts structural metadata, and uses OpenRouter LLMs to provide comprehensive analysis including descriptions, developer considerations, maintenance flags, and architectural insights. The tool uses SQLite caching to only process changed files.

## Essential Commands

### Setup
```bash
# Install dependencies
poetry install

# Set required environment variable
export OPENROUTER_API_KEY=your_key_here
```

### Running
```bash
# Run repo-map on a target repository
poetry run repo-map /path/to/repository

# With options
poetry run repo-map /path/to/repo --model "anthropic/claude-sonnet-4.6" --concurrency 3 -y
```

### Development
```bash
# Full local gate: Ruff, Black, pytest, and Poetry metadata
poetry run python scripts.py check

# Format all project Python
poetry run python scripts.py format

# Run tests with assertions enabled
poetry run python scripts.py test

# List project workflows
poetry run python scripts.py list-skills

# Build both distribution artifacts
poetry build
```

### Testing During Development
The test suite covers the cache, CLI helpers, parsers, scanner, LLM response handling, report generation, and development commands. For changes to live OpenRouter behavior, also run the tool against a small public fixture:
```bash
poetry run repo-map /path/to/test/repo -y
```

## Architecture

### Core Pipeline
1. **CLI Entry** ([cli_handler.py](src/repo_map/cli_handler.py)) → Parses arguments, validates API key, initializes cache
2. **File Scanning** ([file_scanner.py](src/repo_map/file_scanner.py)) → Walks directory tree, respects `.gitignore`, computes file hashes
3. **Code Parsing** ([code_parser.py](src/repo_map/code_parser.py)) → Extracts Python structure with `ast` and Java, JavaScript, TypeScript, and C# structure with targeted regular expressions
4. **LLM Enhancement** ([llm_service.py](src/repo_map/llm_service.py)) → Sends file metadata plus the target file's complete source to the OpenRouter API with concurrency control
5. **Report Generation** ([report_generator.py](src/repo_map/report_generator.py)) → Produces tree visualization and Markdown output

### Key Modules

- **[main.py](src/repo_map/main.py)** - Entry point (`run_main()` function)
- **[config.py](src/repo_map/config.py)** - Pydantic settings from env vars/`.env` file
- **[cache_manager.py](src/repo_map/cache_manager.py)** - SQLite persistence, schema migrations
- **[models.py](src/repo_map/models.py)** - `SUPPORTED_LANGUAGES` dict mapping file extensions to language names, plus `NON_TEXT_LANGUAGES` / `SUPPORTED_TEXT_LANGUAGES` and `MAX_SOURCE_BYTES`, which together decide whether a file's source may be uploaded
- **[llm_service.py](src/repo_map/llm_service.py)** - OpenRouter API client with semaphore-based concurrency, exponential backoff, and response parsing

### Caching System

Cache files are created **inside the target repository**:
- `.repo-map-cache.db` - SQLite database with SHA-256 hashes and LLM results
- `.repo_map_structure.json` - Pre-enhancement structural data

When files change, their hash changes and they're reprocessed. Delete `.repo-map-cache.db` to force full regeneration.

Cache rows are keyed by repository-relative path in forward-slash form, so the database stays valid when the repository is renamed, cloned, or checked out at another prefix. A row written under the older absolute-path key matches nothing and reads as a miss. Each run ends by deleting rows whose key no file in that scan claims, which removes deleted files and any leftover absolute-path keys.

## Configuration

Settings are loaded from environment variables or `.env` file via Pydantic:
- `OPENROUTER_API_KEY` - **Required**
- `OPENROUTER_MODEL_NAME` - Default: `anthropic/claude-sonnet-4.6`
- `API_SEMAPHORE_LIMIT` - Concurrent API calls (default: 3)

All settings defined in [config.py](src/repo_map/config.py) as a singleton `settings` object.

## LLM Expectations

The tool sends each file to the LLM with the repository tree as background and the target file's complete source as delimited, explicitly untrusted data, and expects a JSON object with these fields. Every field must be present and well-typed; a partial or malformed response is a failed analysis and is not cached.

1. **`description`** (string) - 1-2 sentences on the file's role and responsibility
2. **`developer_consideration`** (string | null) - Single most important thing a contributor needs to know
3. **`maintenance_flag`** - One of: `Stable`, `Volatile`, `Generated`, `Unknown`
4. **`critical_dependencies`** (object) - Map of import name → one-line justification
5. **`architectural_role`** - One of: `Entrypoint`, `Configuration`, `Service`, `Data Model`, `Persistence`, `UI Component`, `API Route`, `Utility`, `Test`, `Tooling`, `Other`
6. **`refactoring_suggestions`** (string | null) - Concrete and actionable, or null
7. **`security_assessment`** (string | null) - Specific risk and mitigation, or null

The request sets `response_format: {"type": "json_object"}` to enforce structured output. Prompt is defined as `SYSTEM_PROMPT` in [llm_service.py](src/repo_map/llm_service.py).

## Adding Language Support

1. Add extension mappings to `SUPPORTED_LANGUAGES` in [models.py](src/repo_map/models.py); binary and media formats also go in `NON_TEXT_LANGUAGES` so their bytes are never uploaded
2. Implement parsing logic in [code_parser.py](src/repo_map/code_parser.py)
3. Add parser and scanner tests for filenames, imports, symbols, and lexical scope

## Output

- Console: ASCII tree visualization
- File: `<repo_name>_repo_map.md` saved at the root of the analyzed repo
- Structure: `.repo_map_structure.json` in analyzed repo root

## Important Notes

- Python 3.12+ required
- Python uses the standard-library AST; other structured languages currently use lightweight regular-expression extractors
- Async architecture with `aiohttp` for API calls
- Type hints throughout - `py.typed` marker present for downstream consumers
- Exponential backoff handles rate limits automatically
