.PHONY: check lint format typecheck test changelog list-skills

# pytest reports failures through assert statements, which -O strips.
unexport PYTHONOPTIMIZE

# The release gate: lint, types, tests, lockfile, and changelog index.
check: lint typecheck test
	uv lock --check
	bun scripts/build-changelog.ts --check

lint:
	uv run ruff check
	uv run ruff format --check

format:
	uv run ruff check --fix
	uv run ruff format

typecheck:
	uv run pyright

test:
	uv run pytest

# Regenerate CHANGELOG.md from changelog/<major.minor>.x/<version>.md.
changelog:
	bun scripts/build-changelog.ts

list-skills:
	@for f in .claude/skills/*/SKILL.md; do \
		awk '/^---$$/ { if (++fm == 2) exit; next } \
			/^name:/ { name = $$2 } \
			/^description:/ { d = 1; sub(/^description: *>? */, ""); desc = $$0; next } \
			d && /^  / { sub(/^ +/, ""); desc = desc (desc ? " " : "") $$0; next } \
			{ d = 0 } \
			END { printf "%-20s %s\n", name, desc }' "$$f"; \
	done
