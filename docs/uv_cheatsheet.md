# repo-map uv commands

## Environment

Create `.venv` and install the locked runtime and dev dependencies:

```bash
uv sync
```

Inspect the active environment:

```bash
uv run python --version
uv tree --depth 1
```

## Development gates

Run the complete local gate:

```bash
uv run python scripts.py check
```

Run individual lanes:

```bash
uv run python scripts.py lint
uv run python scripts.py test
uv run python scripts.py format
```

Build the wheel and source distribution:

```bash
uv build
```

## CLI

```bash
uv run repo-map /path/to/repository
uv run repo-map /path/to/repository --model anthropic/claude-sonnet-5.5 --concurrency 3 -y
```

## Dependencies

Add or remove a dependency:

```bash
uv add <package>
uv add --dev <package>
uv remove <package>
```

Review and apply updates:

```bash
uv tree --outdated --depth 1
uv lock --upgrade
uv sync
```

`[tool.uv] exclude-newer = "3 days"` in `pyproject.toml` keeps releases younger than three days out of the lock. Follow [`skills/maintenance/SKILL.md`](../skills/maintenance/SKILL.md) for dependency review and verification.

## Project workflows

```bash
uv run python scripts.py list-skills
uv run python scripts.py sync-skills
```
