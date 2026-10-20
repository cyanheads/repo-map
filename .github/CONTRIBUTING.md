# Contributing

Thanks for using `repo-map`. Bugs, feature requests, and documentation gaps all belong in an issue; that's where they get read and picked up.

Open one from the [issue chooser](https://github.com/cyanheads/repo-map/issues/new/choose) and pick the **Bug Report** or **Feature Request** form. Both are structured, and filling in the fields is what makes an issue actionable. Anything that fits neither can be a plain issue. A half-formed idea in your own words is fine.

## Before filing

1. **Check you're on the latest release.** Fixes land on the current version: `uv tool upgrade repo-map`.
2. **Search existing issues** before opening a new one. Add to the matching thread instead of filing a duplicate.
3. **Redact anything sensitive.** Issues are public and permanent: no API keys, private source, private repository names, or internal paths in code, logs, or reports.

## What makes an issue actionable

- repo-map version, Python version, operating system, and the model you ran with.
- The exact command, and a minimal public repository layout that reproduces the problem.
- Actual vs expected behavior, verbatim: error messages, tracebacks, and the relevant lines of `<repository>_repo_map.md`.
- For features: the use case first, then the command or output you'd want.

## For agents

Do the triage first. An unverified report costs more to read than it saves to file. [`.claude/skills/report-issue-local/SKILL.md`](../.claude/skills/report-issue-local/SKILL.md) is the workflow for filing against this repo; read it before filing on a user's behalf.

## Security

Don't open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).
