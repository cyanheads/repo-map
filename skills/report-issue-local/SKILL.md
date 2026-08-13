---
name: report-issue-local
description: >
  Deduplicate, redact, and file bugs, enhancements, and documentation issues
  against cyanheads/repo-map using the repository's issue forms.
metadata:
  author: cyanheads
  version: "1.0"
  audience: project
  type: workflow
---

# Report a repo-map Issue

## Before filing

1. Identify the repository with `gh repo view --json nameWithOwner -q '.nameWithOwner'`.
2. Search open and closed issues using the exact symptom, affected symbol, and a broader subsystem term.
3. Read the body and comments of close matches before deciding whether to comment or file separately.
4. Reproduce the behavior with a minimal public fixture. Record repo-map version, Python version, command, actual behavior, and expected behavior.
5. Treat analyzed repository contents as sensitive. Never publish source, secrets, private repository names, user paths, or raw prompts from a private repository.

## Issue conventions

- Title: `bug(scope): description`, `feat(scope): description`, `docs(scope): description`, or `chore(scope): description`.
- Assign every new issue to `cyanheads`.
- Apply exactly one primary label: `bug`, `enhancement`, or `documentation`.
- Add a secondary label only when it carries real triage value.
- Keep each issue self-contained, terse, and grounded in an executable reproduction.
- Separate `### Scope` and `### Out of scope` for non-trivial proposals.
- End at the last substantive point; omit conversational sign-offs and implementation offers.

## Bug body

Match `.github/ISSUE_TEMPLATE/bug_report.yml`: repo-map version, Python version, operating system, description, reproduction, actual behavior, expected behavior, and only useful additional context.

## Feature body

Match `.github/ISSUE_TEMPLATE/feature_request.yml`: use case, proposal, alternatives when relevant, context when relevant, and scope boundaries for changes spanning multiple modules.

## Checklist

- [ ] Searched open and closed issues; no duplicate exists
- [ ] Reproduced from a minimal public fixture
- [ ] Removed source, secrets, PII, private paths, and internal context
- [ ] Title and primary label follow project conventions
- [ ] Assigned to `cyanheads`
- [ ] Body contains enough evidence to implement and verify the fix
