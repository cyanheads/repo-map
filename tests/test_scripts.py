"""Tests for project development commands."""

import sys

import scripts


def _record_commands(monkeypatch) -> list[list[str]]:
    """Capture the commands a workflow would run instead of running them."""
    commands: list[list[str]] = []
    monkeypatch.setattr(
        scripts, "_run", lambda command, **kwargs: commands.append(command)
    )
    return commands


def test_list_skills_reports_canonical_workflows(capsys) -> None:
    scripts.run_list_skills()

    assert capsys.readouterr().out.splitlines() == [
        "code-simplifier",
        "git-wrapup",
        "maintenance",
        "report-issue-local",
    ]


def test_check_verifies_skill_mirrors_instead_of_rewriting_them(monkeypatch) -> None:
    commands = _record_commands(monkeypatch)

    scripts.run_check()

    script = str(scripts.SYNC_SKILLS_SCRIPT)
    assert commands[0] == [sys.executable, script, "--check"]
    assert [sys.executable, script] not in commands


def test_sync_skills_still_rewrites_the_mirrors(monkeypatch) -> None:
    commands = _record_commands(monkeypatch)

    scripts.run_sync_skills()

    assert commands == [[sys.executable, str(scripts.SYNC_SKILLS_SCRIPT)]]
    assert scripts.COMMANDS["sync-skills"] is scripts.run_sync_skills
