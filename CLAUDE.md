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
# Format code (Black)
poetry run format
# or
poetry run black src

# Lint code (Ruff)
poetry run ruff check src

# Run tests (if/when added)
poetry run pytest
```

### Testing During Development
Since there are no tests yet, test changes by running the tool on a sample repository:
```bash
poetry run repo-map /path/to/test/repo -y
```

## Architecture

### Core Pipeline
1. **CLI Entry** ([cli_handler.py](src/repo_map/cli_handler.py)) → Parses arguments, validates API key, initializes cache
2. **File Scanning** ([file_scanner.py](src/repo_map/file_scanner.py)) → Walks directory tree, respects `.gitignore`, computes file hashes
3. **Code Parsing** ([code_parser.py](src/repo_map/code_parser.py)) → Extracts imports, classes, functions, and docstrings using tree-sitter
4. **LLM Enhancement** ([llm_service.py](src/repo_map/llm_service.py)) → Sends file metadata to OpenRouter API with concurrency control
5. **Report Generation** ([report_generator.py](src/repo_map/report_generator.py)) → Produces tree visualization and Markdown output

### Key Modules

- **[main.py](src/repo_map/main.py)** - Entry point (`run_main()` function)
- **[config.py](src/repo_map/config.py)** - Pydantic settings from env vars/`.env` file
- **[cache_manager.py](src/repo_map/cache_manager.py)** - SQLite persistence, schema migrations
- **[models.py](src/repo_map/models.py)** - `SUPPORTED_LANGUAGES` dict mapping file extensions to language names
- **[llm_service.py](src/repo_map/llm_service.py)** - OpenRouter API client with semaphore-based concurrency, exponential backoff, and response parsing

### Caching System

Cache files are created **inside the target repository**:
- `.repo-map-cache.db` - SQLite database with SHA-256 hashes and LLM results
- `.repo_map_structure.json` - Pre-enhancement structural data

When files change, their hash changes and they're reprocessed. Delete `.repo-map-cache.db` to force full regeneration.

## Configuration

Settings are loaded from environment variables or `.env` file via Pydantic:
- `OPENROUTER_API_KEY` - **Required**
- `OPENROUTER_MODEL_NAME` - Default: `anthropic/claude-sonnet-4.6`
- `API_SEMAPHORE_LIMIT` - Concurrent API calls (default: 3)

All settings defined in [config.py](src/repo_map/config.py) as a singleton `settings` object.

## LLM Expectations

The tool sends each file to the LLM with the repository tree as background and expects a JSON object with these fields:

1. **`description`** (string) - 1-2 sentences on the file's role and responsibility
2. **`developer_consideration`** (string | null) - Single most important thing a contributor needs to know
3. **`maintenance_flag`** - One of: `Stable`, `Volatile`, `Generated`, `Unknown`
4. **`critical_dependencies`** (object) - Map of import name → one-line justification
5. **`architectural_role`** - One of: `Entrypoint`, `Configuration`, `Service`, `Data Model`, `Persistence`, `UI Component`, `API Route`, `Utility`, `Test`, `Tooling`, `Other`
6. **`refactoring_suggestions`** (string | null) - Concrete and actionable, or null
7. **`security_assessment`** (string | null) - Specific risk and mitigation, or null

The request sets `response_format: {"type": "json_object"}` to enforce structured output. Prompt is defined as `SYSTEM_PROMPT` in [llm_service.py](src/repo_map/llm_service.py).

## Adding Language Support

1. Add extension mappings to `SUPPORTED_LANGUAGES` in [models.py](src/repo_map/models.py)
2. Implement parsing logic in [code_parser.py](src/repo_map/code_parser.py)
3. tree-sitter grammars are available via `tree-sitter-languages` package

## Output

- Console: ASCII tree visualization
- File: `<repo_name>_repo_map.md` saved in parent directory of analyzed repo
- Structure: `.repo_map_structure.json` in analyzed repo root

## Important Notes

- Python 3.12+ required
- Uses `tree-sitter` for parsing (not AST), enabling multi-language support
- Async architecture with `aiohttp` for API calls
- Type hints throughout - `py.typed` marker present for downstream consumers
- Exponential backoff handles rate limits automatically