# Skills

Project workflows for `repo-map`. Each subdirectory contains a `SKILL.md` following the [Agent Skills specification](https://agentskills.io/specification).

`skills/` is the committed source of truth. `python3 scripts/sync_skills.py` copies missing or changed skills into `.agents/skills/` and `.claude/skills/` for local toolchains; mirror-only skills are left untouched.

List the available project workflows with:

```bash
poetry run python scripts.py list-skills
```

When adding or changing a skill, edit `skills/<name>/SKILL.md`, run `poetry run python scripts.py sync-skills`, and verify the full project with `poetry run python scripts.py check`.
