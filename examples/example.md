## Example Generated Output For repo-map

Generated:
```md
# Repository Map

```markdown
/ (.)
├── .cache/
├── .git/
├── docs/
    └── poetry_cheatsheet.md (Markdown)
├── examples/
├── src/
    └── repo_map/
    │   ├── __pycache__/
    │   ├── cache_manager.py (Python)
    │   │   ├── Description: Manages the persistent SQLite cache for storing and retrieving expensive Language Model (LLM) response data.
    │   │   ├── Developer Consideration: The cache uses a simple key-value structure; ensure the cache key (prompt/input) is deterministic and unique for reliable lookups.
    │   │   ├── Maintenance Flag: Stable
    │   │   ├── Architectural Role: Persistence
    │   │   ├── Refactoring Suggestions: Implement a CacheManager class to encapsulate connection handling and ensure the database connection is properly closed.
    │   │   ├── Critical Dependencies:
    │   │   └──   - sqlite3: Provides the persistent, file-based database for storing cached LLM responses.
    │   ├── cli_handler.py (Python)
    │   │   ├── Description: Handles command-line argument parsing using argparse and orchestrates the main workflow, including scanning, LLM processing, and report generation.
    │   │   ├── Developer Consideration: This file manages the global API semaphore limit; ensure any changes to LLM concurrency are reflected in the `update_api_semaphore_limit` call.
    │   │   ├── Maintenance Flag: Volatile
    │   │   ├── Architectural Role: Entrypoint
    │   │   ├── Refactoring Suggestions: Extract the core business logic (scanning, processing, saving) into a dedicated `Controller` class to decouple it from argument parsing.
    │   │   ├── Critical Dependencies:
    │   │   ├──   - argparse: Defines and parses all command-line arguments, forming the user interface.
    │   │   ├──   - tqdm: Provides progress bars for long-running operations like file scanning and LLM calls.
    │   │   └──   - repo_map.llm_service: Crucial for fetching AI-generated descriptions, which is the core value proposition.
    │   ├── code_parser.py (Python)
    │   │   └── Description: Provides language-specific functions to parse source code files, extracting structural elements like imports, docstrings, and functions.
    │   ├── config.py (Python)
    │   │   ├── Description: Defines the application's core configuration settings using Pydantic, managing environment variables, API keys, and default parameters.
    │   │   ├── Developer Consideration: All configuration must inherit from `BaseSettings` and use `SettingsConfigDict` to ensure proper environment variable loading and validation.
    │   │   ├── Maintenance Flag: Stable
    │   │   ├── Architectural Role: Configuration
    │   │   ├── Security Assessment: Handles API keys; ensure sensitive fields are loaded securely from environment variables and never hardcoded.
    │   │   ├── Critical Dependencies:
    │   │   └──   - pydantic_settings: Provides robust, validated configuration management, automatically handling environment variable injection.
    │   ├── file_scanner.py (Python)
    │   │   ├── Description: Scans the repository recursively, applies ignore rules, and generates a structured summary of files, including content hashes and basic metadata.
    │   │   ├── Developer Consideration: The `summarize_repo` function is the primary entry point and handles all file filtering using `.gitignore` specifications via `pathspec`.
    │   │   ├── Maintenance Flag: Stable
    │   │   ├── Architectural Role: Utility
    │   │   ├── Refactoring Suggestions: The `_process_file` function is quite large; consider breaking out the language-specific parsing logic into separate, smaller helper functions.
    │   │   ├── Critical Dependencies:
    │   │   └──   - pathspec: Crucial for parsing and applying `.gitignore` rules to accurately filter files during the repository scan.
    │   ├── llm_service.py (Python)
    │   │   ├── Description: Manages asynchronous communication with external Language Model APIs, handling request throttling, response parsing, and error handling.
    │   │   ├── Developer Consideration: All LLM calls must be wrapped in the global rate limiter to respect rate limits; failure to do so will lead to API throttling.
    │   │   ├── Maintenance Flag: Stable
    │   │   ├── Architectural Role: Service
    │   │   ├── Refactoring Suggestions: Abstract the LLM provider logic into a factory pattern or separate classes for easier future expansion.
    │   │   ├── Security Assessment: Handles API keys via `repo_map.config.settings`; ensure these keys are never logged or exposed in exceptions.
    │   │   ├── Critical Dependencies:
    │   │   ├──   - aiohttp: Provides asynchronous HTTP client capabilities essential for non-blocking API calls.
    │   │   └──   - repo_map.config.settings: Accesses API keys and configuration parameters necessary for authentication and endpoint selection.
    │   ├── logging_utils.py (Python)
    │   │   ├── Description: Configures the application's logging system, integrating standard Python logging with `tqdm` to ensure clean output during progress bar updates.
    │   │   ├── Developer Consideration: The custom `TqdmHandler` is essential for preventing log messages from corrupting the visual integrity of the `tqdm` progress bars.
    │   │   ├── Maintenance Flag: Stable
    │   │   ├── Architectural Role: Utility
    │   │   ├── Refactoring Suggestions: Consider using `structlog` or a similar library for structured logging, which improves machine readability and analysis.
    │   │   ├── Critical Dependencies:
    │   │   ├──   - logging: Core Python module for standard logging functionality.
    │   │   └──   - tqdm: Used to integrate logging output cleanly with progress bar displays.
    │   ├── main.py (Python)
    │   │   ├── Description: The primary entrypoint for the `repo-map` CLI tool, responsible for initializing the application and executing the asynchronous main workflow.
    │   │   ├── Developer Consideration: The `run_main` function must be executed using `asyncio.run()` because the core application logic in `RepoMapApp.run()` is asynchronous.
    │   │   ├── Maintenance Flag: Stable
    │   │   ├── Architectural Role: Entrypoint
    │   │   ├── Critical Dependencies:
    │   │   └──   - asyncio: Required to execute the asynchronous application workflow defined in `RepoMapApp`.
    │   ├── models.py (Python)
    │   │   └── Description: Data models for the application, including supported languages.
    │   ├── py.typed (None)
    │   └── report_generator.py (Python)
    │       ├── Description: Functions responsible for formatting and saving the final repository map output in various formats, including Markdown and JSON.
    │       ├── Developer Consideration: The `format_tree_lines` function is critical for visual output; ensure any changes maintain correct indentation and path structure.
    │       ├── Maintenance Flag: Stable
    │       ├── Architectural Role: Utility
    │       ├── Refactoring Suggestions: Consider using a dedicated templating engine (like Jinja2) for the Markdown output to separate presentation logic from data processing.
    │       ├── Critical Dependencies:
    │       └──   - json: Standard library for serializing the structured repository map data into the JSON output format.
├── .gitignore (None)
├── AGENTS.md (Markdown)
├── CHANGELOG.md (Markdown)
├── LICENSE (None)
├── README.md (Markdown)
├── poetry.lock (None)
├── pyproject.toml (None)
└── scripts.py (Python)
    ├── Description: Utility module containing helper scripts, primarily for running code formatting and linting tools via subprocess calls.
    ├── Developer Consideration: The `run_format` function executes external tools (Black, isort) directly; ensure these tools are installed and accessible in the execution environment.
    ├── Maintenance Flag: Stable
    ├── Architectural Role: Tooling
    ├── Refactoring Suggestions: Consider using a dedicated task runner (like Invoke or Poetry scripts) instead of raw Python subprocess calls for better cross-platform compatibility.
    ├── Critical Dependencies:
    └──   - subprocess: Essential for executing external shell commands like Black and isort for code formatting.
└────────────── 
```
```
