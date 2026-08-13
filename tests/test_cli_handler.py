"""Tests for CLI orchestration helpers."""

import argparse
import asyncio

import repo_map.cli_handler as cli_module
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


def test_enhancement_runs_concurrently_and_caches_completed_file(monkeypatch) -> None:
    active = 0
    peak = 0
    cache_updates = []
    progress_steps = 0
    delays = [0.04, 0.01, 0.03, 0.02]
    structure = [
        {
            "path": f"/tmp/{index}.py",
            "name": f"{index}.py",
            "type": "file",
            "imports": ["os"],
            "functions": [],
        }
        for index in range(4)
    ]

    async def fake_description(received_structure, file_data, model) -> None:
        nonlocal active, peak
        assert received_structure is structure
        active += 1
        peak = max(peak, active)
        index = int(file_data["name"].removesuffix(".py"))
        await asyncio.sleep(delays[index])
        file_data["description"] = f"description for {file_data['path']}"
        active -= 1

    class Progress:
        def __call__(self, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def update(self) -> None:
            nonlocal progress_steps
            progress_steps += 1

        @staticmethod
        def write(message: str) -> None:
            return None

    app = RepoMapApp()
    monkeypatch.setattr(
        app,
        "_get_files_to_process",
        lambda received: {item["path"] for item in received},
    )
    monkeypatch.setattr(app, "_update_cache_for_file", cache_updates.append)
    monkeypatch.setattr(cli_module, "get_llm_descriptions", fake_description)
    monkeypatch.setattr(cli_module, "tqdm", Progress())

    asyncio.run(app._enhance_summary_with_llm(structure, "test-model"))

    assert peak == len(structure)
    assert progress_steps == len(structure)
    assert [file_data["path"] for file_data in cache_updates] == [
        "/tmp/1.py",
        "/tmp/3.py",
        "/tmp/2.py",
        "/tmp/0.py",
    ]
    assert {id(file_data) for file_data in cache_updates} == {
        id(file_data) for file_data in structure
    }
    assert all(
        file_data["description"] == f"description for {file_data['path']}"
        for file_data in cache_updates
    )
