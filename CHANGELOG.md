# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
