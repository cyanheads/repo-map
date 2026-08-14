<div align="center">
  <h1>repo-map</h1>
  <p>
    <a href="https://pypi.org/project/repo-map/"><img src="https://img.shields.io/pypi/v/repo-map.svg" alt="PyPI version"></a>
    <a href="https://pypi.org/project/repo-map/"><img src="https://img.shields.io/pypi/pyversions/repo-map.svg" alt="Python versions"></a>
    <a href="https://pypi.org/project/repo-map/"><img src="https://img.shields.io/pypi/l/repo-map.svg" alt="PyPI license"></a>
    <a href="https://pypi.org/project/repo-map/"><img src="https://img.shields.io/pypi/dm/repo-map.svg" alt="PyPI downloads"></a>
  </p>
  <p>Generate a repository tree with AI-authored file summaries and maintenance notes.</p>
</div>

repo-map scans a repository, extracts structural metadata, and asks an OpenRouter model to document eligible files. It writes a Markdown tree at the repository root and stores results in SQLite so unchanged files do not need another API call.

## Features

- Markdown and console repository trees
- Python AST extraction plus lightweight Java, JavaScript, TypeScript, and C# structure parsing
- Structured descriptions, developer considerations, maintenance flags, dependency notes, architectural roles, refactoring suggestions, and security assessments, grounded in each file's own source
- Root `.gitignore` support plus built-in exclusions for caches, dependencies, build output, credential-bearing config formats, and repo-map artifacts
- Symlink-safe traversal that does not follow linked files or directories
- SHA-256-based SQLite cache for unchanged files, keyed by repository-relative path so it survives a rename, clone, or CI checkout at another prefix

See [`examples/example.md`](examples/example.md) for a full sample report.

## Install

Install the published CLI with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install repo-map
```

For development, clone the repository and install its Poetry environment:

```bash
git clone https://github.com/cyanheads/repo-map.git
cd repo-map
poetry install
```

repo-map requires Python 3.12 or newer.

## Configure

Set an [OpenRouter](https://openrouter.ai/) API key in the environment or a project-root `.env` file:

```bash
export OPENROUTER_API_KEY=your_api_key_here
```

Optional environment variables:

| Variable | Default | Purpose |
|:---|:---|:---|
| `OPENROUTER_MODEL_NAME` | `anthropic/claude-sonnet-4.6` | OpenRouter model |
| `API_SEMAPHORE_LIMIT` | `3` | Maximum concurrent API calls |

repo-map sends repository paths, languages, imports, symbols, existing descriptions, and **the full text of each eligible file** to OpenRouter. Review the target repository and its ignore rules before approving a run. Do not analyze secrets or source you are not authorized to disclose.

A file's source is sent only when it is a regular (non-symlinked) file in a supported text format, at most 64 KiB, free of NUL bytes, and valid UTF-8. Recognized binary and media formats, oversized files, and unreadable files are never sent and never cached.

## Use

With the published tool installed:

```bash
repo-map /path/to/repository
```

From a source checkout:

```bash
poetry run repo-map /path/to/repository
```

Options:

| Option | Purpose |
|:---|:---|
| `-y`, `--yes` | Skip the disclosure confirmation |
| `--model MODEL` | Override `OPENROUTER_MODEL_NAME` |
| `--concurrency INT` | Override `API_SEMAPHORE_LIMIT` |

Examples:

```bash
poetry run repo-map /path/to/repository --model anthropic/claude-sonnet-4.6
poetry run repo-map /path/to/repository --concurrency 3 -y
```

Each run creates these files inside the target repository:

- `<repository>_repo_map.md`: generated Markdown report
- `.repo-map-cache.db`: source hashes and cached LLM metadata
- `.repo_map_structure.json`: pre-enhancement structural data

Add them to the target repository's ignore rules if needed. Delete `.repo-map-cache.db` to force a complete reprocessing pass; the cache is otherwise self-maintaining, since entries are keyed by repository-relative path and every run drops the entries whose files it no longer finds.

## How it works

1. Load built-in exclusions and the target repository's root `.gitignore`.
2. Walk the directory tree and extract supported structural metadata.
3. Compare file hashes with the SQLite cache.
4. Request structured JSON metadata from OpenRouter for eligible changed files, sending each file's source alongside the tree.
5. Cache validated results only, drop cache entries with no matching file, then write the console, JSON, and Markdown outputs.

A directory matching an exclusion is pruned rather than traversed, so nothing inside it is read, hashed, or sent.

### Default exclusions

| Group | Patterns |
|:---|:---|
| Version control | `.git/`, `.hg/`, `.svn/`, `CVS/` |
| Caches and bytecode | `__pycache__/`, `*.pyc`, `*.pyo`, `*.pyd`, `.pytest_cache/`, `.mypy_cache/` |
| Environments and dependencies | `.venv/`, `venv/`, `env/`, `node_modules/` |
| Build output | `build/`, `dist/`, `*.egg-info/` |
| Credential-bearing config | `.env`, `.envrc`, `*.tfvars`, `*.tfstate`, `*.ini`, `*.conf`, `*.cfg` |
| Databases and logs | `*.db`, `*.sqlite3`, `*.log` |
| Local noise and repo-map artifacts | `.DS_Store`, `.repo-map-cache.db`, `.repo_map_structure.json`, `*_repo_map.md` |

These load before the target repository's root `.gitignore`, and matching follows gitignore last-match-wins semantics. A repository that wants an excluded file documented re-includes it with a negation in its own `.gitignore`, such as `!app.conf`.

## Development

Run the local gate and build both distribution artifacts before submitting a change:

```bash
poetry run python scripts.py check
poetry build
```

Focused commands:

```bash
poetry run python scripts.py format
poetry run python scripts.py lint
poetry run python scripts.py test
poetry run python scripts.py list-skills
```

Project workflows live under [`skills/`](skills/). Bugs and feature requests use the forms on the [issues page](https://github.com/cyanheads/repo-map/issues).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
