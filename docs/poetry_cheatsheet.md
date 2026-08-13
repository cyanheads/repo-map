# repo-map Poetry commands

## Environment

Install the locked dependencies and local CLI entry points:

```bash
poetry install
```

Inspect the active environment:

```bash
poetry env info
poetry run python --version
```

## Development gates

Run the complete local gate:

```bash
poetry run python scripts.py check
```

Run individual lanes:

```bash
poetry run python scripts.py lint
poetry run python scripts.py test
poetry run python scripts.py format
```

Build the wheel and source distribution:

```bash
poetry build
```

## CLI

```bash
poetry run repo-map /path/to/repository
poetry run repo-map /path/to/repository --model anthropic/claude-sonnet-4.6 --concurrency 3 -y
```

## Dependencies

Add or remove a dependency:

```bash
poetry add <package>
poetry remove <package>
```

Review and apply compatible updates:

```bash
poetry show --outdated
poetry update
```

Follow [`skills/maintenance/SKILL.md`](../skills/maintenance/SKILL.md) for dependency review and verification.

## Project workflows

```bash
poetry run python scripts.py list-skills
poetry run python scripts.py sync-skills
```
