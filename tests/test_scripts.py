"""Tests for project development commands."""

import scripts


def test_list_skills_reports_canonical_workflows(capsys) -> None:
    scripts.run_list_skills()

    assert capsys.readouterr().out.splitlines() == [
        "code-simplifier",
        "git-wrapup",
        "maintenance",
        "report-issue-local",
    ]
