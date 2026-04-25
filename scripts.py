"""This module provides scripts for the repo-map tool."""
import subprocess
import sys
from pathlib import Path


def run_format():
    """
    Run black to format the source code.
    """
    subprocess.run(["black", "src"], check=True)


def run_sync_skills():
    """
    Propagate ``skills/`` to ``.agents/skills/`` and ``.claude/skills/``.
    Delegates to ``scripts/sync_skills.py`` so the same logic is reachable
    without Poetry (e.g. from a Claude Code hook).
    """
    script = Path(__file__).resolve().parent / "scripts" / "sync_skills.py"
    sys.exit(subprocess.run([sys.executable, str(script)]).returncode)
