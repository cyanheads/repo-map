"""Executable reproductions for accepted, tracked repo-map defects."""

import asyncio

import pytest

import repo_map.cli_handler as cli_module
import repo_map.llm_service as llm_module
from repo_map.cache_manager import load_cache
from repo_map.cli_handler import RepoMapApp
from repo_map.code_parser import get_structure
from repo_map.file_scanner import summarize_repo


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/cyanheads/repo-map/issues/10",
)
def test_analysis_prompt_includes_target_source(tmp_path) -> None:
    source = "UNIQUE_SOURCE_SENTINEL_7E1C"
    source_path = tmp_path / "module.py"
    source_path.write_text(source, encoding="utf-8")
    file_data = {
        "path": str(source_path),
        "name": "module.py",
        "level": 0,
        "type": "file",
        "language": "Python",
        "imports": ["os"],
        "functions": ["run"],
        "source": source,
    }

    assert source in llm_module._build_user_prompt([file_data], file_data)


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/cyanheads/repo-map/issues/6",
)
def test_unsuccessful_analysis_remains_pending(tmp_path, monkeypatch) -> None:
    source = tmp_path / "module.py"
    source.write_text("import os\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))
    structure = summarize_repo(str(tmp_path), connection)
    app = RepoMapApp()
    app.cache_conn = connection

    async def failed_analysis(structure, file_data, model) -> None:
        return None

    monkeypatch.setattr(cli_module, "get_llm_descriptions", failed_analysis)
    asyncio.run(app._enhance_summary_with_llm(structure, "test-model"))

    pending = app._get_files_to_process(summarize_repo(str(tmp_path), connection))
    connection.close()

    assert str(source) in pending


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/cyanheads/repo-map/issues/11",
)
def test_class_only_and_data_files_are_eligible_for_enrichment(tmp_path) -> None:
    connection = load_cache(str(tmp_path))
    app = RepoMapApp()
    app.cache_conn = connection
    structure = [
        {
            "path": str(tmp_path / "class_only.py"),
            "type": "file",
            "classes": {"Only": []},
            "imports": [],
            "functions": [],
            "hash": "x",
        },
        {
            "path": str(tmp_path / "config.json"),
            "type": "file",
            "imports": [],
            "functions": [],
            "hash": "y",
        },
    ]

    pending = app._get_files_to_process(structure)
    connection.close()

    assert pending == {item["path"] for item in structure}


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/cyanheads/repo-map/issues/12",
)
def test_parser_preserves_async_and_javascript_scope(tmp_path) -> None:
    python_source = tmp_path / "sample.py"
    python_source.write_text("async def run():\n    pass\n", encoding="utf-8")
    javascript_source = tmp_path / "sample.js"
    javascript_source.write_text(
        "class A {\n  method() {}\n}\nfunction outside() {}\n",
        encoding="utf-8",
    )

    _, python_functions, _ = get_structure(str(python_source), "Python")
    javascript_classes, javascript_functions, _ = get_structure(
        str(javascript_source), "JavaScript"
    )

    assert python_functions == ["run"]
    assert javascript_classes == {"A": ["method"]}
    assert javascript_functions == ["outside"]
