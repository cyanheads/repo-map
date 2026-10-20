---
name: report-issue-local
description: >
  File a bug, feature request, or documentation issue against cyanheads/repo-map:
  search for duplicates, reproduce on a public fixture, redact anything from an
  analyzed repository, and file through the repo's issue forms with the
  maintainer assigned. Use when work on repo-map turns up a defect or gap in its
  own code (scanner, parser, LLM request or response handling, cache, report,
  CLI, packaging, docs) that should be tracked.
metadata:
  author: cyanheads
  version: "2.0"
  audience: project
  type: workflow
---

# Report a repo-map Issue

## Is it repo-map's?

File here when repo-map itself misbehaves: a file skipped or misparsed, a malformed request, a valid response rejected, a stale or wrong cache hit, a broken report, a CLI or packaging fault.

Not here:
- OpenRouter account, billing, or provider availability problems that don't reproduce as a repo-map fault. The issue form's contact link points to OpenRouter support.
- One weak note in one generated report. A note the model got wrong is model variance unless the prompt or contract causes it across files; then it is a bug against the prompt.

## Before filing

1. **Search for duplicates**, open and closed, with the exact symptom, the affected symbol, and a broader subsystem term:

   ```bash
   gh issue list --repo cyanheads/repo-map --state all --search "<terms>"
   gh issue view <N> --repo cyanheads/repo-map --json title,state,body,comments
   ```

   Read the body and the comments of every close match. When one covers the same symptom, comment there with the new evidence instead of filing a duplicate. File separately only when the symptom or scope is distinct.

2. **Reproduce on a public fixture**: a scratch directory holding only the files the bug needs, written for the report. Parser and scanner bugs usually reproduce offline as a short pytest case or Python snippet against `repo_map.code_parser` or `repo_map.file_scanner`, which costs no OpenRouter credit. Record:
   - repo-map version: `uv version --short` in a checkout; for an installed copy, `uv tool list` or `pip show repo-map`
   - Python version and operating system
   - the exact command or snippet, the actual output, the expected output

## Redact

Issues are public. Never publish:

- anything from an analyzed repository that isn't public: source, file and directory names, generated notes (they paraphrase the source), or raw prompts and model responses built from it
- private repository names, hostnames, and absolute paths that include a home directory or user name (write `<repo>/src/app.py` instead)
- `OPENROUTER_API_KEY` or any token, in any partial form

Reduce a failure seen on a private repository to the public fixture that reproduces it. If it won't reproduce publicly, describe the shape ("a TypeScript file with a default-exported class") and file without the content.

## Write it

- **Title**: `type(scope): description`. Types `bug`, `feat`, `docs`, `chore`. Scopes follow the module: `cli`, `scanner`, `parser`, `llm`, `cache`, `report`, `config`, `packaging`, `docs`. Example: `bug(parser): async methods lose their class scope`.
- **Labels**: exactly one primary, `bug`, `enhancement`, or `documentation`. Add `performance`, `security`, or `dependencies` only when it helps triage. Don't create new labels.
- **Assignee**: every issue carries `--assignee cyanheads`.
- **Budget**: a bug reads in about 150 words, a feature in about 250, code and logs excluded. Lead with the specific symptom and symbol. Every section past the form's required fields has to change what the implementer does; cut restated context, mechanism walkthroughs, and sign-offs like "happy to open a PR".
- **Links**: `Related: #N` near the top when the issue grows out of another one, cited once. Name the module and function (`_first_invalid_field()` in `llm_service.py`) rather than paraphrasing it.

Write the body with the file-writing tool to a file outside the repository and pass it with `--body-file`; delete the file afterwards. A body passed as a shell string loses every backticked span to command substitution.

### Bug

Headings follow `.github/ISSUE_TEMPLATE/bug_report.yml`, in its order:

````markdown
### repo-map version

<X.Y.Z>

### Python version

<output of python --version>

### Operating system

<OS and version>

### Description

Two or three sentences: what happened, where, what was expected.

### Reproduction

```shell
mkdir fixture && printf 'class A:\n    async def run(self): ...\n' > fixture/a.py
uv run python -c "from repo_map.code_parser import ...; print(...)"
```

### Actual behavior

```shell
<redacted output or traceback>
```

### Expected behavior

What should have happened.
````

Add `### Additional context` only when it changes the fix: model name, `--concurrency`, cache state, a related issue.

```bash
gh issue create --repo cyanheads/repo-map \
  --title "bug(scope): description" \
  --label bug --assignee cyanheads \
  --body-file <body-file>
```

### Feature

Headings follow `.github/ISSUE_TEMPLATE/feature_request.yml`: `### Use case` (who hits the gap, in one or two sentences) and `### Proposal` (the behavior or flag, with an example) are required; `### Alternatives considered` comes third when there were any. Add `### Out of scope` in one or two lines when the change spans several modules. File with `--label enhancement`.

### Documentation

No form. Name the file and section, quote the wrong statement, and give the correct one. File with `--label documentation`. A typo or one stale sentence in a file you are already editing is a direct fix, not an issue.

## Follow up

```bash
gh issue comment <N> --repo cyanheads/repo-map --body-file <body-file>
```

Use a comment for new evidence on an existing issue. Closing is the release's job: `release-and-publish` closes issues once the fix is on PyPI.

## Checklist

- [ ] It is repo-map's fault, not OpenRouter's or one model answer's
- [ ] Open and closed issues searched; close matches read with comments; no duplicate filed
- [ ] Reproduced on a public fixture; version, Python, OS, command, actual, and expected recorded
- [ ] Nothing from an analyzed private repository, no user paths, no keys
- [ ] `type(scope): description` title, one primary label, `--assignee cyanheads`
- [ ] Body follows the form's headings and fits the budget; passed with `--body-file`
