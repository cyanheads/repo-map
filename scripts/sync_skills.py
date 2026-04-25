#!/usr/bin/env python3
"""Propagate ``skills/`` (canonical) to ``.agents/skills/`` and ``.claude/skills/``.

The canonical source lives at ``skills/`` in the project root. Local agent
toolchains (Claude Code, generic agent runners) read from their own mirror
directories — those drift silently unless re-synced. This script does a
content-hash compare and copies anything missing or changed. Files that exist
only in a mirror are left alone (they may be globally-installed skills).

Behavior:
  * Missing mirror dir → created
  * Missing file in mirror → copied
  * Content drift → overwritten
  * File only in mirror → left alone (probably a general-purpose skill)
  * Exit 0 always; this is a best-effort sync, not a gate

Run via ``poetry run sync-skills`` or ``python3 scripts/sync_skills.py``.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
MIRRORS = [ROOT / ".agents" / "skills", ROOT / ".claude" / "skills"]
IGNORE_NAMES = {".DS_Store", "Thumbs.db"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.name not in IGNORE_NAMES]


def sync_to(mirror: Path, files: list[Path]) -> tuple[int, int]:
    added = changed = 0
    for src in files:
        rel = src.relative_to(SKILLS_DIR)
        dst = mirror / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            added += 1
        elif sha256(src) != sha256(dst):
            shutil.copy2(src, dst)
            changed += 1
    return added, changed


def main() -> int:
    if not SKILLS_DIR.is_dir():
        print(f"sync-skills: no {SKILLS_DIR.relative_to(ROOT)}/ directory; nothing to do.")
        return 0

    files = walk_files(SKILLS_DIR)
    if not files:
        print(f"sync-skills: {SKILLS_DIR.relative_to(ROOT)}/ is empty; nothing to do.")
        return 0

    total_added = total_changed = 0
    for mirror in MIRRORS:
        mirror.mkdir(parents=True, exist_ok=True)
        added, changed = sync_to(mirror, files)
        total_added += added
        total_changed += changed
        if added or changed:
            print(f"sync-skills: {mirror.relative_to(ROOT)} → +{added} added, ~{changed} changed")

    if total_added == 0 and total_changed == 0:
        print(f"sync-skills: {len(files)} file(s) already in sync across {len(MIRRORS)} mirror(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
