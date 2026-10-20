<div align="center">
  <h1>repo-map</h1>
  <p><b>Map a repository as a Markdown tree with LLM-written notes on each file.</b></p>
</div>

<div align="center">

[![Version](https://img.shields.io/pypi/v/repo-map?style=flat-square&label=version)](./CHANGELOG.md) [![License](https://img.shields.io/pypi/l/repo-map?style=flat-square)](./LICENSE) [![Python](https://img.shields.io/pypi/pyversions/repo-map?style=flat-square)](https://pypi.org/project/repo-map/) [![Downloads](https://img.shields.io/pypi/dm/repo-map?style=flat-square)](https://pypi.org/project/repo-map/)

</div>

---

repo-map walks a repository, sends each eligible file to an [OpenRouter](https://openrouter.ai/) model, and writes the results as a tree to `<repository>_repo_map.md`. Analyses are cached in SQLite, so the next run only re-sends files that changed.

Each file gets a description, the one thing a contributor is most likely to get wrong, a maintenance flag, an architectural role, its critical dependencies, and a refactoring suggestion or security note when one applies. An entry from [`examples/example.md`](examples/example.md), which is repo-map run on itself:

```text
├── main.py (Python)
│   ├── Description: Console-script entrypoint for repo-map. run_main() configures logging, imports RepoMapApp lazily, runs RepoMapApp().run() under asyncio.run, and maps failures to exit codes: 1 for configuration errors and unexpected exceptions, 130 for Ctrl-C.
│   ├── Developer Consideration: The RepoMapApp import inside run_main() is deliberate: importing cli_handler builds the settings singleton and rate limiter, and a ValueError there (e.g. bad API_SEMAPHORE_LIMIT) must be caught after setup_logging(). Moving it to module scope breaks that handling.
│   ├── Maintenance Flag: Stable
│   ├── Architectural Role: Entrypoint
│   ├── Critical Dependencies:
│   ├──   - repo_map.logging_utils: setup_logging() configures logging before any application code or errors run.
│   └──   - repo_map.cli_handler: Supplies RepoMapApp, the application whose async run() does all the work.
```

## Getting started

Set an OpenRouter API key, then run repo-map with [uv](https://docs.astral.sh/uv/), no install step needed:

```bash
export OPENROUTER_API_KEY=sk-or-...
uvx repo-map /path/to/repository
```

To keep it installed instead:

```bash
uv tool install repo-map
repo-map /path/to/repository
```

Requires Python 3.12 or newer. repo-map asks for confirmation before it sends anything; `-y` skips the prompt.

## Configuration

Settings come from the environment, or from a `.env` file in the directory you run repo-map from. [`.env.example`](.env.example) lists them all.

| Variable | Required | Default | Purpose |
|:---|:---|:---|:---|
| `OPENROUTER_API_KEY` | Yes | | OpenRouter API key |
| `OPENROUTER_MODEL_NAME` | No | `anthropic/claude-sonnet-5.5` | Any OpenRouter model ID |
| `API_SEMAPHORE_LIMIT` | No | `3` | Concurrent API calls |

## Usage

```bash
repo-map /path/to/repository [-y] [--model MODEL] [--concurrency N]
```

| Option | Purpose |
|:---|:---|
| `-y`, `--yes` | Skip the upload confirmation |
| `--model MODEL` | Override `OPENROUTER_MODEL_NAME` for this run |
| `--concurrency N` | Override `API_SEMAPHORE_LIMIT` for this run |

A run writes three files into the target repository. Add them to its ignore rules if you don't want them committed.

| File | Contents |
|:---|:---|
| `<repository>_repo_map.md` | The report |
| `.repo-map-cache.db` | Cached analyses, keyed by repository-relative path |
| `.repo_map_structure.json` | The scanned structure, before analysis |

A cached analysis is reused while the file's SHA-256 hash, the model, and repo-map's analysis contract all still match. Editing a file, switching `--model`, or upgrading to a release with a revised prompt triggers re-analysis. Each run prunes entries for files it no longer finds, and the cache stays valid when the repository is renamed or cloned elsewhere.

## What gets sent

Each request carries the repository's file and directory names, short descriptions of the files the target imports or shares a directory with, and **the full text of the target file**. Review the repository and its ignore rules before approving a run, and don't analyze secrets or source you aren't authorized to share with OpenRouter.

A file's source is sent only if it is a regular file (symlinks are never followed) in a supported text format, at most 64 KiB, valid UTF-8, and free of NUL bytes. Binary and media formats are never sent.

repo-map skips these paths by default, then applies the target's root `.gitignore`. An excluded directory is not traversed, so nothing inside it is read or sent.

| Group | Patterns |
|:---|:---|
| Version control | `.git/`, `.hg/`, `.svn/`, `CVS/` |
| Caches and bytecode | `__pycache__/`, `*.pyc`, `*.pyo`, `*.pyd`, `.pytest_cache/`, `.mypy_cache/` |
| Environments and dependencies | `.venv/`, `venv/`, `env/`, `node_modules/` |
| Build output | `build/`, `dist/`, `*.egg-info/` |
| Credential-bearing config | `.env`, `.envrc`, `*.tfvars`, `*.tfstate`, `*.ini`, `*.conf`, `*.cfg` |
| Databases and logs | `*.db`, `*.sqlite3`, `*.log` |
| Local noise and repo-map output | `.DS_Store`, `.repo-map-cache.db`, `.repo_map_structure.json`, `*_repo_map.md` |

Matching follows gitignore's last-match-wins rule, so a negation in the target's `.gitignore`, such as `!app.conf`, brings an excluded file back. Only the root `.gitignore` is read.

## Project structure

| Path | Purpose |
|:---|:---|
| `src/repo_map/` | The package: CLI, scanner, parsers, OpenRouter client, cache, and report writer |
| `tests/` | pytest suite, one module per source module |
| `examples/example.md` | repo-map's report on this repository |
| `changelog/` | Per-version release notes; `CHANGELOG.md` is the generated index |
| `scripts/build-changelog.ts` | Builds `CHANGELOG.md` from `changelog/` |
| `.claude/skills/` | Agent workflows for maintaining this repository |

## Development

```bash
git clone https://github.com/cyanheads/repo-map.git
cd repo-map
uv sync
make check
```

- `make check` is the gate: Ruff lint and format check, pyright, pytest, `uv lock --check`, and the changelog index check. The changelog step runs under [Bun](https://bun.sh/).
- A change to the prompt or the response fields bumps `ANALYSIS_CONTRACT_VERSION` in `src/repo_map/llm_service.py`, so cached analyses made under the old contract are redone.
- `CHANGELOG.md` is generated: add `changelog/<major.minor>.x/<version>.md`, then run `make changelog`.
- [`CLAUDE.md`](CLAUDE.md) has the full set of working rules.

## Contributing

Bugs and feature requests go through the [issue forms](https://github.com/cyanheads/repo-map/issues/new/choose); [CONTRIBUTING.md](.github/CONTRIBUTING.md) covers what makes one actionable. Report vulnerabilities privately as described in [SECURITY.md](.github/SECURITY.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
