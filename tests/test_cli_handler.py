"""Tests for CLI orchestration helpers."""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import repo_map.cli_handler as cli_module
import repo_map.llm_service as llm_module
from repo_map.cache_manager import CACHE_FILE_NAME, CacheError, load_cache
from repo_map.cli_handler import RepoMapApp
from repo_map.llm_service import AnalysisOutcome

#: Variables that must not leak from the developer's shell into a CLI subprocess.
INHERITED_SETTINGS_VARS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL_NAME",
    "API_SEMAPHORE_LIMIT",
)

#: Accepted by the CLI's key check without ever reaching the network, because
#: every failure exercised here aborts before the first OpenRouter request.
PLACEHOLDER_API_KEY = "not-a-real-key"

#: A response body that satisfies every field of the analysis contract.
WELL_FORMED_ANALYSIS = {
    "description": "A module.",
    "developer_consideration": None,
    "maintenance_flag": "Stable",
    "critical_dependencies": {},
    "architectural_role": "Utility",
    "refactoring_suggestions": None,
    "security_assessment": None,
}


def _console_script() -> list[str]:
    """The installed ``repo-map`` entry point, or its top-level import chain."""
    script = Path(sys.executable).parent / "repo-map"
    if script.is_file():
        return [str(script)]
    return [
        sys.executable,
        "-c",
        "import sys; from repo_map.main import run_main; sys.exit(run_main())",
    ]


def _run_cli(
    arguments: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the console script the way a user does, in an isolated environment.

    ``cwd`` is a scratch directory so the project's own ``.env`` cannot supply
    settings, and ``PYTHONOPTIMIZE`` is cleared for the same reason
    ``scripts.py`` clears it.
    """
    child_env = {
        name: value
        for name, value in os.environ.items()
        if name not in INHERITED_SETTINGS_VARS and name != "PYTHONOPTIMIZE"
    }
    child_env.update(env or {})
    return subprocess.run(
        [*_console_script(), *arguments],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=child_env,
    )


def _assert_single_line_failure(
    result: subprocess.CompletedProcess[str], *, expected: str
) -> None:
    """Assert the run failed with one actionable line and no traceback."""
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "Traceback (most recent call last)" not in output, output
    assert expected in output, output


def _answers(monkeypatch, *responses: str) -> list[str]:
    """Queue disclosure-prompt responses, returning the prompts actually shown."""
    shown: list[str] = []
    remaining = list(responses)

    def fake_input(prompt: str) -> str:
        shown.append(prompt)
        return remaining.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return shown


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
            "source_eligible": True,
            "hash": "current",
        },
        {
            "path": str(tmp_path / "changed.py"),
            "type": "file",
            "source_eligible": True,
            "hash": "changed",
        },
        {
            "path": str(tmp_path / "binary.png"),
            "type": "file",
            "source_eligible": False,
        },
        {
            "path": str(tmp_path / "docs"),
            "type": "directory",
        },
    ]

    pending = app._get_files_to_process(structure)
    connection.close()

    assert pending == {str(tmp_path / "changed.py")}


def test_disclosure_states_full_source_upload_and_root_gitignore_scope(
    monkeypatch,
) -> None:
    """The prompt names what leaves the machine and how narrow the ignore rules are."""
    shown = _answers(monkeypatch, "n")

    RepoMapApp()._confirm_disclaimer()

    prompt = shown[0]
    assert "full text of each eligible file" in prompt
    assert "OpenRouter" in prompt
    assert "root .gitignore" in prompt
    assert "subdirectories are not applied" in prompt


def test_confirm_disclaimer_declines_bare_enter(monkeypatch) -> None:
    """A stray newline is not consent; the advertised default declines."""
    shown = _answers(monkeypatch, "")

    assert RepoMapApp()._confirm_disclaimer() is False
    assert "[y/N]" in shown[0]


def test_confirm_disclaimer_declines_whitespace_only_input(monkeypatch) -> None:
    _answers(monkeypatch, "   \t ")

    assert RepoMapApp()._confirm_disclaimer() is False


@pytest.mark.parametrize("response", ["y", "yes", "Y", "YES", " yes "])
def test_confirm_disclaimer_accepts_explicit_yes(monkeypatch, response: str) -> None:
    _answers(monkeypatch, response)

    assert RepoMapApp()._confirm_disclaimer() is True


@pytest.mark.parametrize("response", ["n", "no", "N", "NO"])
def test_confirm_disclaimer_declines_explicit_no(monkeypatch, response: str) -> None:
    _answers(monkeypatch, response)

    assert RepoMapApp()._confirm_disclaimer() is False


def test_confirm_disclaimer_reprompts_unrecognized_input(monkeypatch, capsys) -> None:
    shown = _answers(monkeypatch, "maybe", "y")

    assert RepoMapApp()._confirm_disclaimer() is True
    assert len(shown) == 2
    assert "Invalid input" in capsys.readouterr().out


def test_confirm_disclaimer_cancels_on_closed_stdin(monkeypatch, capsys) -> None:
    """EOF declines and reports cancellation -- unchanged behavior."""

    def raise_eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    assert RepoMapApp()._confirm_disclaimer() is False
    assert "Operation cancelled." in capsys.readouterr().out


def test_confirm_disclaimer_cancels_on_keyboard_interrupt(monkeypatch, capsys) -> None:
    def raise_interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_interrupt)

    assert RepoMapApp()._confirm_disclaimer() is False
    assert "Operation cancelled." in capsys.readouterr().out


def test_yes_flag_bypasses_disclosure_prompt(monkeypatch, tmp_path, capsys) -> None:
    """``-y`` never prompts, and a cache failure still exits 1 with one line."""
    prompted: list[str] = []
    app = RepoMapApp()
    monkeypatch.setattr(
        app,
        "_parse_args",
        lambda: argparse.Namespace(
            repository_path=str(tmp_path), yes=True, model=None, concurrency=3
        ),
    )
    monkeypatch.setattr(app, "_confirm_disclaimer", lambda: prompted.append("asked"))
    monkeypatch.setattr(cli_module.settings, "openrouter_api_key", PLACEHOLDER_API_KEY)

    def explode(_repository_path: str):
        raise CacheError(f"cache is unusable at {tmp_path / CACHE_FILE_NAME}")

    monkeypatch.setattr(cli_module, "load_cache", explode)

    with pytest.raises(SystemExit) as failure:
        asyncio.run(app.run())

    assert failure.value.code == 1
    assert prompted == []
    assert CACHE_FILE_NAME in capsys.readouterr().out


def test_environment_concurrency_of_zero_exits_without_traceback(tmp_path) -> None:
    """`API_SEMAPHORE_LIMIT=0` fails during import today -- issue #21."""
    repository = tmp_path / "repo"
    repository.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_cli(
        [str(repository), "-y"],
        cwd=scratch,
        env={"API_SEMAPHORE_LIMIT": "0", "OPENROUTER_API_KEY": PLACEHOLDER_API_KEY},
    )

    _assert_single_line_failure(result, expected="API_SEMAPHORE_LIMIT")
    assert "1 or greater" in result.stdout + result.stderr


def test_environment_concurrency_non_integer_exits_without_traceback(tmp_path) -> None:
    """`API_SEMAPHORE_LIMIT=abc` fails during import today -- issue #21."""
    repository = tmp_path / "repo"
    repository.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_cli(
        [str(repository), "-y"],
        cwd=scratch,
        env={"API_SEMAPHORE_LIMIT": "abc", "OPENROUTER_API_KEY": PLACEHOLDER_API_KEY},
    )

    _assert_single_line_failure(result, expected="API_SEMAPHORE_LIMIT")
    assert "abc" in result.stdout + result.stderr


def test_unwritable_cache_directory_exits_without_traceback(tmp_path) -> None:
    """An unwritable target directory reports the cache path and the remedy."""
    repository = tmp_path / "readonly"
    repository.mkdir()
    repository.chmod(0o555)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    try:
        result = _run_cli(
            [str(repository), "-y"],
            cwd=scratch,
            env={"OPENROUTER_API_KEY": PLACEHOLDER_API_KEY},
        )
    finally:
        repository.chmod(0o755)

    _assert_single_line_failure(result, expected=str(repository / CACHE_FILE_NAME))
    assert "clean rebuild" in result.stdout + result.stderr


def test_corrupt_cache_file_exits_without_traceback(tmp_path) -> None:
    """A corrupt `.repo-map-cache.db` reports the cache path and the remedy."""
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / CACHE_FILE_NAME).write_text("not a database", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_cli(
        [str(repository), "-y"],
        cwd=scratch,
        env={"OPENROUTER_API_KEY": PLACEHOLDER_API_KEY},
    )

    _assert_single_line_failure(result, expected=str(repository / CACHE_FILE_NAME))
    assert "clean rebuild" in result.stdout + result.stderr


def test_concurrency_flag_of_zero_keeps_its_clean_error(tmp_path) -> None:
    """Regression: the CLI flag path already degrades cleanly and must stay that way."""
    repository = tmp_path / "repo"
    repository.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_cli(
        [str(repository), "--concurrency", "0"],
        cwd=scratch,
        env={"OPENROUTER_API_KEY": PLACEHOLDER_API_KEY},
    )

    _assert_single_line_failure(result, expected="Invalid concurrency value")


def test_missing_target_directory_keeps_its_clean_error(tmp_path) -> None:
    """Regression: a non-directory target already exits cleanly and must stay that way."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_cli(
        [str(tmp_path / "absent"), "-y"],
        cwd=scratch,
        env={"OPENROUTER_API_KEY": PLACEHOLDER_API_KEY},
    )

    _assert_single_line_failure(result, expected="is not a valid directory")


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

    async def fake_description(received_structure, file_data, model):
        nonlocal active, peak
        assert received_structure is structure
        active += 1
        peak = max(peak, active)
        index = int(file_data["name"].removesuffix(".py"))
        await asyncio.sleep(delays[index])
        file_data["description"] = f"description for {file_data['path']}"
        active -= 1
        return AnalysisOutcome.SUCCESS

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


def test_enhancement_isolates_failures_and_caches_each_success_once(
    monkeypatch,
) -> None:
    cache_updates = []
    progress_steps = 0
    structure = [
        {
            "path": f"/tmp/{name}.py",
            "name": f"{name}.py",
            "type": "file",
            "source_eligible": True,
        }
        for name in ("slow-success", "failure", "fast-success")
    ]

    async def fake_description(received_structure, file_data, model):
        assert received_structure is structure
        if file_data["name"] == "slow-success.py":
            await asyncio.sleep(0.02)
            return AnalysisOutcome.SUCCESS
        if file_data["name"] == "fast-success.py":
            await asyncio.sleep(0.005)
            return AnalysisOutcome.SUCCESS
        await asyncio.sleep(0.01)
        return AnalysisOutcome.FAILURE

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

    assert progress_steps == len(structure)
    assert [item["name"] for item in cache_updates] == [
        "fast-success.py",
        "slow-success.py",
    ]


@pytest.mark.parametrize("failure", [TypeError("bad payload"), KeyError("content")])
def test_enhancement_isolates_a_raising_task_from_its_peers(
    monkeypatch, caplog, failure
) -> None:
    """An exception the retry loop does not guard fails one file, not the pass."""
    cache_updates = []
    progress_steps = 0
    structure = [
        {
            "path": f"/tmp/{name}.py",
            "name": f"{name}.py",
            "type": "file",
            "source_eligible": True,
        }
        for name in ("fast-success", "raiser", "slow-success")
    ]

    async def fake_description(received_structure, file_data, model):
        assert received_structure is structure
        if file_data["name"] == "fast-success.py":
            await asyncio.sleep(0.005)
            return AnalysisOutcome.SUCCESS
        if file_data["name"] == "raiser.py":
            await asyncio.sleep(0.01)
            raise failure
        # Still awaiting its response when the raiser resolves.
        await asyncio.sleep(0.03)
        return AnalysisOutcome.SUCCESS

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

    with caplog.at_level(logging.ERROR, logger=cli_module.__name__):
        asyncio.run(app._enhance_summary_with_llm(structure, "test-model"))

    assert progress_steps == len(structure)
    assert [item["name"] for item in cache_updates] == [
        "fast-success.py",
        "slow-success.py",
    ]
    assert "/tmp/raiser.py" in caplog.text


def test_malformed_response_shapes_leave_the_rest_of_the_pass_intact(
    tmp_path, monkeypatch
) -> None:
    """A payload shape the parser never reaches fails one file, not the report."""
    names = ("early", "late", "no-content", "no-message")
    for index, name in enumerate(names):
        (tmp_path / f"{name}.py").write_text(f"VALUE = {index}\n", encoding="utf-8")

    async def shaped_response(messages, model, temperature):
        prompt = messages[1]["content"]
        if "Target file: `no-content.py`" in prompt:
            return {"choices": [{"message": {"role": "assistant"}}]}
        if "Target file: `no-message.py`" in prompt:
            return {"choices": [{"finish_reason": "stop"}]}
        if "Target file: `late.py`" in prompt:
            # Still awaiting its response when the malformed peers resolve.
            await asyncio.sleep(0.05)
        return {"choices": [{"message": {"content": json.dumps(WELL_FORMED_ANALYSIS)}}]}

    monkeypatch.setattr(llm_module, "rate_limited_api_call", shaped_response)

    app = RepoMapApp()
    app.args = argparse.Namespace(
        repository_path=str(tmp_path), yes=True, model="test-model", concurrency=3
    )
    app.cache_conn = load_cache(str(tmp_path))

    asyncio.run(app._process_repository())

    cached = {
        Path(row[0]).name
        for row in app.cache_conn.execute("SELECT path FROM cache").fetchall()
    }
    pending = {
        Path(path).name
        for path in app._get_files_to_process(
            cli_module.summarize_repo(str(tmp_path), app.cache_conn)
        )
    }
    app.cache_conn.close()

    assert (tmp_path / f"{tmp_path.name}_repo_map.md").is_file()
    assert cached == {"early.py", "late.py"}
    assert pending == {"no-content.py", "no-message.py"}
