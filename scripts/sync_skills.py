#!/usr/bin/env python3
"""Propagate or verify ``skills`` against ``.agents/skills`` and ``.claude/skills``.

The canonical source lives at ``skills/`` in the project root. Local agent
toolchains (Claude Code, generic agent runners) read from their own mirror
directories — those drift silently unless re-synced. This script compares by
content hash in one of two modes.

Sync (default) — copies anything missing or changed:
  * Missing mirror dir → created
  * Missing file in mirror → copied
  * Content drift → overwritten
  * File only in mirror → left alone (probably a general-purpose skill)
  * Exit 0 when every mirror is synchronized; file errors fail loudly

Check (``--check``) — reports and never writes, so the project gate can verify
mirrors instead of rewriting them:
  * Missing file or content drift → reported with its mirror, exit 1
  * File only in mirror → reported as an orphan, left in place, exit unaffected
  * Nothing is created, copied, or deleted, including absent mirror directories

Run via ``poetry run python scripts.py sync-skills`` (sync),
``poetry run python scripts.py check`` (check), or
``python3 scripts/sync_skills.py [--check]``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
MIRRORS = [ROOT / ".agents" / "skills", ROOT / ".claude" / "skills"]
IGNORE_NAMES = {".DS_Store", "Thumbs.db"}
SYNC_COMMAND = "poetry run python scripts.py sync-skills"

FindingKind = Literal["missing", "drifted", "orphan"]


@dataclass(frozen=True)
class MirrorFinding:
    """One mirror path that does not match the canonical ``skills/`` tree."""

    kind: FindingKind
    mirror: Path
    relative_path: Path

    def describe(self) -> str:
        """Render the finding as one report line, relative to the project root."""
        location = (self.mirror / self.relative_path).relative_to(ROOT)
        suffix = (
            "  (no counterpart in skills/; left in place)"
            if self.kind == "orphan"
            else ""
        )
        return f"  {self.kind:<8} {location}{suffix}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.name not in IGNORE_NAMES]


def sync_to(mirror: Path, files: list[Path], skills_dir: Path) -> tuple[int, int]:
    added = changed = 0
    for src in files:
        rel = src.relative_to(skills_dir)
        dst = mirror / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            added += 1
        elif sha256(src) != sha256(dst):
            shutil.copy2(src, dst)
            changed += 1
    return added, changed


def check_mirrors(
    skills_dir: Path, mirrors: Sequence[Path], files: list[Path]
) -> list[MirrorFinding]:
    """Compare every mirror against the canonical tree without touching disk."""
    canonical = {src.relative_to(skills_dir): src for src in files}
    findings: list[MirrorFinding] = []
    for mirror in mirrors:
        for rel, src in sorted(canonical.items()):
            dst = mirror / rel
            if not dst.exists():
                findings.append(MirrorFinding("missing", mirror, rel))
            elif sha256(src) != sha256(dst):
                findings.append(MirrorFinding("drifted", mirror, rel))
        if not mirror.is_dir():
            continue
        for dst in sorted(walk_files(mirror)):
            rel = dst.relative_to(mirror)
            if rel not in canonical:
                findings.append(MirrorFinding("orphan", mirror, rel))
    return findings


def report(findings: list[MirrorFinding], file_count: int, mirror_count: int) -> int:
    """Print the check-mode report and return the exit code it implies."""
    drift = [finding for finding in findings if finding.kind != "orphan"]
    orphans = [finding for finding in findings if finding.kind == "orphan"]
    if drift:
        lines = [
            f"sync-skills: {len(drift)} mirror file(s) out of sync with skills/:",
            *(finding.describe() for finding in drift + orphans),
            f"sync-skills: run `{SYNC_COMMAND}` to regenerate the mirrors.",
        ]
        print("\n".join(lines), file=sys.stderr)
        return 1

    print(
        f"sync-skills: {file_count} file(s) verified across "
        f"{mirror_count} mirror(s); no drift."
    )
    for finding in orphans:
        print(finding.describe())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Propagate or verify the skills/ mirrors."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report mirror drift and exit non-zero instead of copying",
    )
    args = parser.parse_args(argv)

    if not SKILLS_DIR.is_dir():
        print(
            f"sync-skills: no {SKILLS_DIR.relative_to(ROOT)}/ directory; nothing to do."
        )
        return 0

    files = walk_files(SKILLS_DIR)
    if not files:
        print(f"sync-skills: {SKILLS_DIR.relative_to(ROOT)}/ is empty; nothing to do.")
        return 0

    if args.check:
        return report(
            check_mirrors(SKILLS_DIR, MIRRORS, files), len(files), len(MIRRORS)
        )

    total_added = total_changed = 0
    for mirror in MIRRORS:
        mirror.mkdir(parents=True, exist_ok=True)
        added, changed = sync_to(mirror, files, SKILLS_DIR)
        total_added += added
        total_changed += changed
        if added or changed:
            print(
                f"sync-skills: {mirror.relative_to(ROOT)} → +{added} added, ~{changed} changed"
            )

    if total_added == 0 and total_changed == 0:
        print(
            f"sync-skills: {len(files)} file(s) already in sync across {len(MIRRORS)} mirror(s)."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
