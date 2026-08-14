"""Tests for the skills mirror sync and verification script."""

import importlib.util
import sys
from pathlib import Path

# ``scripts/sync_skills.py`` is a standalone script, not an importable package:
# ``scripts.py`` at the project root claims the ``scripts`` module name. Load it
# from its path, registered in ``sys.modules`` so its dataclass resolves.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sync_skills.py"
_spec = importlib.util.spec_from_file_location("sync_skills", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
sync_skills = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sync_skills
_spec.loader.exec_module(sync_skills)


def _build_project(tmp_path: Path, monkeypatch) -> tuple[Path, list[Path]]:
    """Point the script at a temporary project tree and return its directories."""
    skills = tmp_path / "skills"
    (skills / "alpha" / "reference").mkdir(parents=True)
    (skills / "alpha" / "SKILL.md").write_text("alpha v1\n", encoding="utf-8")
    (skills / "alpha" / "reference" / "notes.md").write_text(
        "notes v1\n", encoding="utf-8"
    )
    (skills / "README.md").write_text("index\n", encoding="utf-8")
    mirrors = [tmp_path / ".agents" / "skills", tmp_path / ".claude" / "skills"]
    monkeypatch.setattr(sync_skills, "ROOT", tmp_path)
    monkeypatch.setattr(sync_skills, "SKILLS_DIR", skills)
    monkeypatch.setattr(sync_skills, "MIRRORS", mirrors)
    return skills, mirrors


def _snapshot(root: Path) -> dict[str, str]:
    """Map every file under ``root`` to its content, for change detection."""
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_sync_creates_missing_mirror_files(tmp_path, monkeypatch) -> None:
    _, mirrors = _build_project(tmp_path, monkeypatch)

    assert sync_skills.main([]) == 0

    for mirror in mirrors:
        assert (mirror / "README.md").read_text(encoding="utf-8") == "index\n"
        assert (mirror / "alpha" / "SKILL.md").read_text(
            encoding="utf-8"
        ) == "alpha v1\n"
        assert (mirror / "alpha" / "reference" / "notes.md").read_text(
            encoding="utf-8"
        ) == "notes v1\n"


def test_sync_overwrites_drifted_mirror_file(tmp_path, monkeypatch) -> None:
    _, mirrors = _build_project(tmp_path, monkeypatch)
    assert sync_skills.main([]) == 0
    drifted = mirrors[1] / "alpha" / "reference" / "notes.md"
    drifted.write_text("hand edit\n", encoding="utf-8")

    assert sync_skills.main([]) == 0

    assert drifted.read_text(encoding="utf-8") == "notes v1\n"


def test_sync_leaves_mirror_only_file_in_place(tmp_path, monkeypatch) -> None:
    _, mirrors = _build_project(tmp_path, monkeypatch)
    assert sync_skills.main([]) == 0
    orphan = mirrors[0] / "global-only" / "SKILL.md"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("installed elsewhere\n", encoding="utf-8")

    assert sync_skills.main([]) == 0

    assert orphan.read_text(encoding="utf-8") == "installed elsewhere\n"


def test_check_passes_when_mirrors_match(tmp_path, monkeypatch, capsys) -> None:
    _build_project(tmp_path, monkeypatch)
    assert sync_skills.main([]) == 0
    before = _snapshot(tmp_path)
    capsys.readouterr()

    assert sync_skills.main(["--check"]) == 0

    assert _snapshot(tmp_path) == before
    captured = capsys.readouterr()
    assert "no drift" in captured.out
    assert captured.err == ""


def test_check_fails_on_a_drifted_mirror_file(tmp_path, monkeypatch, capsys) -> None:
    _, mirrors = _build_project(tmp_path, monkeypatch)
    assert sync_skills.main([]) == 0
    drifted = mirrors[1] / "alpha" / "reference" / "notes.md"
    drifted.write_text("hand edit\n", encoding="utf-8")
    before = _snapshot(tmp_path)
    capsys.readouterr()

    assert sync_skills.main(["--check"]) == 1

    assert _snapshot(tmp_path) == before
    assert drifted.read_text(encoding="utf-8") == "hand edit\n"
    report = capsys.readouterr().err
    assert "drifted" in report
    assert ".claude/skills/alpha/reference/notes.md" in report
    assert "poetry run python scripts.py sync-skills" in report


def test_check_fails_on_a_missing_mirror_file(tmp_path, monkeypatch, capsys) -> None:
    _, mirrors = _build_project(tmp_path, monkeypatch)
    assert sync_skills.main([]) == 0
    (mirrors[0] / "alpha" / "SKILL.md").unlink()
    before = _snapshot(tmp_path)
    capsys.readouterr()

    assert sync_skills.main(["--check"]) == 1

    assert _snapshot(tmp_path) == before
    report = capsys.readouterr().err
    assert "missing" in report
    assert ".agents/skills/alpha/SKILL.md" in report
    assert "poetry run python scripts.py sync-skills" in report


def test_check_fails_when_a_mirror_directory_is_absent(
    tmp_path, monkeypatch, capsys
) -> None:
    _, mirrors = _build_project(tmp_path, monkeypatch)
    before = _snapshot(tmp_path)
    capsys.readouterr()

    assert sync_skills.main(["--check"]) == 1

    assert not mirrors[0].exists()
    assert not mirrors[1].exists()
    assert _snapshot(tmp_path) == before
    report = capsys.readouterr().err
    assert ".agents/skills/README.md" in report
    assert ".claude/skills/README.md" in report


def test_check_reports_an_orphan_without_failing(tmp_path, monkeypatch, capsys) -> None:
    _, mirrors = _build_project(tmp_path, monkeypatch)
    assert sync_skills.main([]) == 0
    orphan = mirrors[0] / "global-only" / "SKILL.md"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("installed elsewhere\n", encoding="utf-8")
    before = _snapshot(tmp_path)
    capsys.readouterr()

    assert sync_skills.main(["--check"]) == 0

    assert _snapshot(tmp_path) == before
    assert orphan.read_text(encoding="utf-8") == "installed elsewhere\n"
    captured = capsys.readouterr()
    assert "orphan" in captured.out
    assert ".agents/skills/global-only/SKILL.md" in captured.out
    assert captured.err == ""


def test_check_reports_an_orphan_alongside_drift(tmp_path, monkeypatch, capsys) -> None:
    _, mirrors = _build_project(tmp_path, monkeypatch)
    assert sync_skills.main([]) == 0
    (mirrors[1] / "README.md").write_text("hand edit\n", encoding="utf-8")
    orphan = mirrors[0] / "global-only" / "SKILL.md"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("installed elsewhere\n", encoding="utf-8")
    capsys.readouterr()

    assert sync_skills.main(["--check"]) == 1

    report = capsys.readouterr().err
    assert "drifted  .claude/skills/README.md" in report
    assert "orphan   .agents/skills/global-only/SKILL.md" in report


def test_check_is_silent_about_an_empty_canonical_tree(
    tmp_path, monkeypatch, capsys
) -> None:
    skills, _ = _build_project(tmp_path, monkeypatch)
    for path in sorted(skills.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()

    assert sync_skills.main(["--check"]) == 0

    assert "nothing to do" in capsys.readouterr().out
