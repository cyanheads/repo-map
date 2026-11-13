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
│   ├── Description: Console-script entrypoint exposing run_main(), which sets up logging, lazily imports RepoMapApp, runs it under asyncio.run, and maps outcomes to exit codes: the app's status, 1 for config or unexpected errors, 130 for Ctrl-C.
│   ├── Developer Consideration: The RepoMapApp import inside run_main() is deliberately deferred: importing repo_map.cli_handler builds the settings singleton and rate limiter, so a bad API_SEMAPHORE_LIMIT raises ValueError there. Moving it to module scope puts that failure outside the handler.
│   ├── Maintenance Flag: Stable
│   ├── Architectural Role: Entrypoint
│   ├── Critical Dependencies:
│   ├──   - repo_map.cli_handler: Provides RepoMapApp, whose async run() does the work and returns the exit status.
│   ├──   - repo_map.logging_utils: setup_logging() must run before anything else so errors are logged.
│   └──   - asyncio: asyncio.run drives the async RepoMapApp.run() coroutine.
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

Requires Python 3.12 or newer. repo-map scans the repository first, then asks for confirmation, stating how many files and how much source it will send. `-y` skips the prompt, and `--dry-run` lists the files without sending anything.

## Configuration

Settings come from the environment, or from a `.env` file in the directory you run repo-map from. [`.env.example`](.env.example) lists them all.

| Variable | Required | Default | Purpose |
|:---|:---|:---|:---|
| `OPENROUTER_API_KEY` | Yes, except for `--dry-run` | | OpenRouter API key |
| `OPENROUTER_MODEL_NAME` | No | `anthropic/claude-sonnet-5.5` | Any OpenRouter model ID |
| `API_SEMAPHORE_LIMIT` | No | `3` | Concurrent API calls |

## Usage

```bash
repo-map /path/to/repository [-y] [--dry-run] [--model MODEL] [--concurrency N]
```

| Option | Purpose |
|:---|:---|
| `-y`, `--yes` | Skip the upload confirmation; the upload summary is still logged |
| `--dry-run` | Print the upload summary and one `<bytes>\t<path>` line per file the run would send, then exit. Sends nothing, writes nothing, and needs no API key |
| `--model MODEL` | Override `OPENROUTER_MODEL_NAME` for this run |
| `--concurrency N` | Override `API_SEMAPHORE_LIMIT` for this run |

The exit status is 0 when every pending file was analyzed and the report was written. It is 1 when a file stays unanalyzed (the report is still written, and a rerun retries only those files), when OpenRouter rejects the key, the account's credit, or the model (the run stops at the first such error), or on any other error, and 130 on Ctrl-C.

Once you approve a run, it writes three files into the target repository; a declined prompt or `--dry-run` writes nothing. Add them to its ignore rules if you don't want them committed.

| File | Contents |
|:---|:---|
| `<repository>_repo_map.md` | The report |
| `.repo-map-cache.db` | Cached analyses, keyed by repository-relative path |
| `.repo_map_structure.json` | The scanned structure, before analysis |

A cached analysis is reused while the file's SHA-256 hash, the model, and repo-map's analysis contract all still match. Editing a file, switching `--model`, or upgrading to a release with a revised prompt triggers re-analysis. Each run prunes entries for files it no longer finds, and the cache stays valid when the repository is renamed or cloned elsewhere.

## What gets sent

Each request carries the repository's file and directory names, short descriptions of the files the target imports or shares a directory with, and **the full text of the target file**. The confirmation prompt states how many files and how many KiB of source this run sends, and `--dry-run` lists them. Review them before approving a run, and don't analyze secrets or source you aren't authorized to share with OpenRouter. When the model supports structured outputs, repo-map asks for a strict JSON schema; that capability check is one keyless request to OpenRouter's model list, made after you approve the run.

A file's source is sent only if it is a regular file (symlinks are never followed) in a supported text format, at most 64 KiB, valid UTF-8, free of NUL bytes, and free of a PEM private-key header. Binary and media formats are never sent. The private-key check reads the content, so no ignore rule can bring such a file back.

repo-map skips these paths by default, then applies the target's Git ignore rules: its root `.gitignore`, the `.gitignore` in every directory it walks, and `.git/info/exclude`. When the target sits inside a larger Git work tree, the `.gitignore` files from that work tree's top down to the target apply as well. Your global excludes file (`core.excludesFile`) is not read, so the result depends on the repository alone. An excluded directory is not traversed, so nothing inside it is read or sent.

| Group | Patterns |
|:---|:---|
| Version control | `.git/`, `.hg/`, `.svn/`, `CVS/` |
| Caches and bytecode | `__pycache__/`, `*.pyc`, `*.pyo`, `*.pyd`, `.pytest_cache/`, `.mypy_cache/` |
| Environments and dependencies | `.venv/`, `venv/`, `env/`, `node_modules/` |
| Build output | `build/`, `dist/`, `*.egg-info/` |
| Credential-bearing config | `*.env`, `*.envrc`, `*.tfvars`, `*.tfvars.json`, `*.tfstate`, `*.tfstate.*`, `*.ini`, `*.conf`, `*.cfg`, `*secrets.toml`, `credentials.toml` |
| Databases and logs | `*.db`, `*.sqlite3`, `*.log` |
| Local noise and repo-map output | `.DS_Store`, `.repo-map-cache.db`, `.repo_map_structure.json`, `*_repo_map.md` |

Matching follows gitignore's last-match-wins rule, so a negation in the target's root `.gitignore`, such as `!app.conf`, brings a default-excluded file back. Rules below the root can only exclude more: a negation in a nested `.gitignore` or in `.git/info/exclude` overrides only other nested rules, never the defaults or the root `.gitignore`. A rules line repo-map can't parse is skipped with a warning.

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
