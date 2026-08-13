"""Tests for CLI orchestration helpers."""

import argparse

from repo_map.cache_manager import load_cache
from repo_map.cli_handler import RepoMapApp


def test_get_output_path_uses_repository_name(tmp_path) -> None:
    repository = tmp_path / "sample-repo"
    repository.mkdir()
    app = RepoMapApp()
    app.args = argparse.Namespace(repository_path=str(repository))

    assert app._get_output_path() == str(repository / "sample-repo_repo_map.md")


def test_get_files_to_process_uses_hash_cache(tmp_path) -> None:
    source = tmp_path / "module.py"
    connection = load_cache(str(tmp_path))
    connection.execute(
        "INSERT INTO cache (path, hash) VALUES (?, ?)",
        (str(source), "current"),
    )
    connection.commit()
    app = RepoMapApp()
    app.cache_conn = connection
    structure = [
        {
            "path": str(source),
            "type": "file",
            "imports": ["pathlib"],
            "functions": ["run"],
            "hash": "current",
        },
        {
            "path": str(tmp_path / "changed.py"),
            "type": "file",
            "imports": ["os"],
            "functions": [],
            "hash": "changed",
        },
        {
            "path": str(tmp_path / "docs"),
            "type": "directory",
        },
    ]

    pending = app._get_files_to_process(structure)
    connection.close()

    assert pending == {str(tmp_path / "changed.py")}


def test_confirm_disclaimer_accepts_default(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "")

    assert RepoMapApp()._confirm_disclaimer() is True
