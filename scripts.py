"""Development commands for the repo-map project."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

CHECK_PATHS = ("src", "tests", "scripts.py", "scripts")
SYNC_SKILLS_SCRIPT = Path(__file__).resolve().parent / "scripts" / "sync_skills.py"


def _run(command: list[str], *, unoptimized: bool = False) -> None:
    """Run a development command and fail immediately when it fails."""
    env = os.environ.copy()
    if unoptimized:
        env.pop("PYTHONOPTIMIZE", None)
    subprocess.run(command, check=True, env=env)


def run_format() -> None:
    """Apply Ruff's safe fixes and format all project Python."""
    _run([sys.executable, "-m", "ruff", "check", "--fix", *CHECK_PATHS])
    _run([sys.executable, "-m", "black", *CHECK_PATHS])


def run_lint() -> None:
    """Check lint and formatting for all project Python."""
    _run([sys.executable, "-m", "ruff", "check", *CHECK_PATHS])
    _run([sys.executable, "-m", "black", "--check", *CHECK_PATHS])


def run_tests() -> None:
    """Run pytest with assertions enabled, regardless of the parent shell."""
    _run([sys.executable, "-m", "pytest"], unoptimized=True)


def run_check() -> None:
    """Run the complete local verification gate."""
    run_check_skills()
    run_lint()
    run_tests()
    _run(["poetry", "check", "--lock"])


def run_list_skills() -> None:
    """List project workflows available under ``skills/``."""
    skills_dir = Path(__file__).resolve().parent / "skills"
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        print(skill_file.parent.name)


def run_check_skills() -> None:
    """Verify the ``skills/`` mirrors, failing on drift without rewriting them."""
    _run([sys.executable, str(SYNC_SKILLS_SCRIPT), "--check"])


def run_sync_skills() -> None:
    """Propagate ``skills/`` to local agent-tool mirrors.

    Delegates to ``scripts/sync_skills.py`` so the same logic is reachable
    without Poetry (e.g. from a Claude Code hook).
    """
    _run([sys.executable, str(SYNC_SKILLS_SCRIPT)])


COMMANDS = {
    "check": run_check,
    "format": run_format,
    "lint": run_lint,
    "list-skills": run_list_skills,
    "sync-skills": run_sync_skills,
    "test": run_tests,
}


def main(args: list[str] | None = None) -> None:
    """Run one project development command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parsed = parser.parse_args(args)
    COMMANDS[parsed.command]()


if __name__ == "__main__":
    main()
