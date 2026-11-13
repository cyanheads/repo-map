"""Tests for CLI orchestration helpers."""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import aiohttp
import pytest

import repo_map.cli_handler as cli_module
import repo_map.llm_service as llm_module
import repo_map.main as main_module
from repo_map.cache_manager import CACHE_FILE_NAME, CacheError, load_cache
from repo_map.cli_handler import RepoMapApp
from repo_map.file_scanner import load_source_snapshot
from repo_map.llm_service import (
    ANALYSIS_CONTRACT_VERSION,
    AnalysisOutcome,
    FatalOpenRouterError,
)

#: Variables that must not leak from the developer's shell into a CLI subprocess.
INHERITED_SETTINGS_VARS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL_NAME",
    "API_SEMAPHORE_LIMIT",
)

#: Accepted by the CLI's key check without ever reaching the network, because
#: every failure exercised here aborts before the first OpenRouter request.
PLACEHOLDER_API_KEY = "not-a-real-key"

#: The model these cases analyze with, standing in for `--model`.
ANALYZING_MODEL = "test-model"

#: The cache table as it stood before analyses recorded the model and contract
#: revision that produced them, for exercising the upgrade path.
PRE_IDENTITY_SCHEMA = """
    CREATE TABLE cache (
        path TEXT PRIMARY KEY,
        hash TEXT,
        description TEXT,
        developer_consideration TEXT,
        imports TEXT,
        functions TEXT,
        maintenance_flag TEXT,
        critical_dependencies TEXT,
        architectural_role TEXT,
        refactoring_suggestions TEXT,
        security_assessment TEXT
    )
"""

#: A response body that satisfies every field of the analysis contract.
WELL_FORMED_ANALYSIS = {
    "description": "A module.",
    "developer_consideration": None,
    "maintenance_flag": "Stable",
    "critical_dependencies": [],
    "architectural_role": "Utility",
    "refactoring_suggestions": None,
    "security_assessment": None,
}

#: The analysis request's line naming its target, by repository-relative path.
TARGET_FILE = re.compile(r"Target file: `([^`]+)`")


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
    arguments: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdin: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run the console script the way a user does, in an isolated environment.

    ``cwd`` is a scratch directory so the project's own ``.env`` cannot supply
    settings, and ``PYTHONOPTIMIZE`` is cleared for the same reason the
    ``Makefile`` clears it. ``stdin`` is everything the process can read, then
    end of file, so a prompt never waits on the terminal running the tests.
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
        input=stdin,
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


def _cache_every_scanned_file(
    app: RepoMapApp, repository: Path, model: str = ANALYZING_MODEL
) -> None:
    """Record a validated analysis for every file the scanner reports.

    The identity fields are the ones `get_llm_descriptions()` stamps onto a
    file it validated, without which the row cannot be written.
    """
    assert app.cache_conn is not None
    for entry in cli_module.summarize_repo(str(repository), app.cache_conn):
        if entry["type"] == "file":
            entry["description"] = "cached"
            entry["model"] = model
            entry["contract_version"] = ANALYSIS_CONTRACT_VERSION
            app._update_cache_for_file(entry)


def _pending_names(
    app: RepoMapApp, repository: Path, model: str = ANALYZING_MODEL
) -> set[str]:
    """Basenames of the files a fresh scan of ``repository`` would send to the LLM."""
    assert app.cache_conn is not None
    structure = cli_module.summarize_repo(str(repository), app.cache_conn)
    plan = cli_module.plan_upload(app.cache_conn, structure, model)
    return {item["name"] for item in plan.pending}


def _cache_keys(connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT path FROM cache")}


def _cache_rows(repository: Path) -> list[tuple]:
    """Every cache row as (key, model, contract revision, description)."""
    connection = load_cache(str(repository))
    rows = connection.execute(
        "SELECT path, model, contract_version, description FROM cache ORDER BY path"
    ).fetchall()
    connection.close()
    return rows


def _seed_pre_identity_cache(repository: Path, key: str, file_hash: str) -> None:
    """Write a cache database in the schema that predates the identity columns."""
    connection = sqlite3.connect(repository / CACHE_FILE_NAME)
    connection.execute(PRE_IDENTITY_SCHEMA)
    connection.execute(
        "INSERT INTO cache (path, hash, description) VALUES (?, ?, ?)",
        (key, file_hash, "written by an older repo-map"),
    )
    connection.commit()
    connection.close()


def _stub_response_format(monkeypatch) -> None:
    """Answer the per-run response-format choice locally, never over the network."""

    async def response_format(model):
        return {"type": "json_object"}

    monkeypatch.setattr(cli_module, "resolve_response_format", response_format)


def _stub_successful_analysis(monkeypatch) -> list[str]:
    """Answer every OpenRouter request with a contract-complete analysis.

    Returns the list of models actually requested.
    """
    _stub_response_format(monkeypatch)
    requested: list[str] = []

    async def analysis(messages, model, session_id, response_format):
        requested.append(model)
        return {"choices": [{"message": {"content": json.dumps(WELL_FORMED_ANALYSIS)}}]}

    monkeypatch.setattr(llm_module, "rate_limited_api_call", analysis)
    return requested


def _run_against(monkeypatch, repository: Path, model: str) -> int:
    """Run one unattended pass over ``repository`` with ``model``, as the CLI would."""
    return _run_app(monkeypatch, repository, "-y", "--model", model)


def _run_app(
    monkeypatch,
    repository: Path,
    *arguments: str,
    api_key: str | None = PLACEHOLDER_API_KEY,
):
    """Run the CLI flow in-process as ``repo-map <repository> <arguments>``.

    Goes through the real argument parser and ``RepoMapApp.run()``, so the
    order of the key check, scan, consent, and cache opening is the shipped
    one. Logging is reconfigured by ``run()`` onto stdout; read it with
    ``capsys``.
    """
    monkeypatch.setattr(sys, "argv", ["repo-map", str(repository), *arguments])
    monkeypatch.setattr(cli_module.settings, "openrouter_api_key", api_key)
    return asyncio.run(RepoMapApp().run())


def _record_requests(monkeypatch) -> list[str]:
    """Record every OpenRouter contact a run makes, answering analyses with success.

    Each entry is ``"response_format"`` for the per-run setup call or the
    target's repository-relative path for an analysis request. Anything that
    still reaches aiohttp is recorded as ``"http"`` and refused, so no test
    run touches the network whatever the client grows.
    """
    requests: list[str] = []

    async def response_format(model):
        requests.append("response_format")
        return {"type": "json_object"}

    async def analysis(messages, model, session_id, response_format):
        target = TARGET_FILE.search(messages[1]["content"])
        assert target is not None
        requests.append(target.group(1))
        return {"choices": [{"message": {"content": json.dumps(WELL_FORMED_ANALYSIS)}}]}

    async def refuse(self, *args, **kwargs):
        requests.append("http")
        raise AssertionError("a test run reached the network")

    monkeypatch.setattr(cli_module, "resolve_response_format", response_format)
    monkeypatch.setattr(llm_module, "rate_limited_api_call", analysis)
    monkeypatch.setattr(aiohttp.ClientSession, "_request", refuse)
    return requests


def _stub_analysis_except(
    monkeypatch, failures: dict[str, Exception | AnalysisOutcome]
) -> None:
    """Analyze every file for real against a stubbed success, except the named ones.

    A file whose name is in ``failures`` raises that exception, or returns that
    outcome, in place of its analysis.
    """
    _record_requests(monkeypatch)

    async def analysis(structure, file_data, model, response_format):
        failure = failures.get(file_data["name"])
        if isinstance(failure, Exception):
            raise failure
        if failure is not None:
            return failure
        return await llm_module.get_llm_descriptions(
            structure, file_data, model=model, response_format=response_format
        )

    monkeypatch.setattr(cli_module, "get_llm_descriptions", analysis)


class _Progress:
    """Stands in for ``tqdm``: counts progress updates and swallows writes."""

    def __init__(self) -> None:
        self.steps = 0

    def __call__(self, **kwargs) -> "_Progress":
        return self

    def __enter__(self) -> "_Progress":
        return self

    def __exit__(self, *args) -> None:
        return None

    def update(self) -> None:
        self.steps += 1

    @staticmethod
    def write(message: str) -> None:
        return None


def _refuse_input(monkeypatch) -> None:
    """Fail the test if the run asks the user anything."""

    def refuse(prompt: str) -> str:
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", refuse)


def _write(path: Path, size: int) -> None:
    """Write a valid Python source file of exactly ``size`` bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"#" * (size - 1) + b"\n")


def _tree(root: Path) -> dict[str, bytes | None]:
    """Every path under ``root`` with its bytes, or None for a directory."""
    return {
        path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


def _report(repository: Path) -> Path:
    return repository / f"{repository.name}_repo_map.md"


def _cache_all(repository: Path, model: str = ANALYZING_MODEL) -> None:
    """Give every file now in ``repository`` a current cached analysis under ``model``."""
    app = RepoMapApp()
    app.cache_conn = load_cache(str(repository))
    try:
        _cache_every_scanned_file(app, repository, model)
    finally:
        app.cache_conn.close()


def _console_exit_status() -> int:
    """The status the console script's ``sys.exit(run_main())`` ends the process with."""
    with pytest.raises(SystemExit) as exit_info:
        sys.exit(main_module.run_main())
    code = exit_info.value.code
    assert code is None or isinstance(code, int)
    return code or 0


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


def test_get_output_path_names_a_relative_path_after_its_directory(
    tmp_path, monkeypatch
) -> None:
    repository = tmp_path / "sample-repo"
    repository.mkdir()
    monkeypatch.chdir(repository)
    app = RepoMapApp()
    app.args = argparse.Namespace(repository_path=".")

    assert os.path.basename(app._get_output_path()) == "sample-repo_repo_map.md"


def test_plan_upload_uses_hash_cache(tmp_path) -> None:
    """Cache lookups key on the relative path; a current row keeps its file."""
    source = tmp_path / "module.py"
    connection = load_cache(str(tmp_path))
    connection.execute(
        "INSERT INTO cache (path, hash, model, contract_version) VALUES (?, ?, ?, ?)",
        ("module.py", "current", ANALYZING_MODEL, ANALYSIS_CONTRACT_VERSION),
    )
    connection.commit()
    structure = [
        {
            "path": str(source),
            "rel_path": "module.py",
            "type": "file",
            "source_eligible": True,
            "hash": "current",
        },
        {
            "path": str(tmp_path / "changed.py"),
            "rel_path": "changed.py",
            "type": "file",
            "source_eligible": True,
            "hash": "changed",
        },
        {
            "path": str(tmp_path / "binary.png"),
            "rel_path": "binary.png",
            "type": "file",
            "source_eligible": False,
        },
        {
            "path": str(tmp_path / "docs"),
            "type": "directory",
        },
    ]

    plan = cli_module.plan_upload(connection, structure, ANALYZING_MODEL)
    connection.close()

    assert [item["path"] for item in plan.pending] == [str(tmp_path / "changed.py")]
    assert plan.kept == 1


def test_plan_upload_ignores_a_legacy_absolute_path_row(tmp_path) -> None:
    """A pre-upgrade row never matches a relative key, so its file is pending."""
    source = tmp_path / "module.py"
    connection = load_cache(str(tmp_path))
    connection.execute(
        "INSERT INTO cache (path, hash) VALUES (?, ?)",
        (str(source), "current"),
    )
    connection.commit()

    plan = cli_module.plan_upload(
        connection,
        [
            {
                "path": str(source),
                "rel_path": "module.py",
                "type": "file",
                "source_eligible": True,
                "hash": "current",
            }
        ],
        ANALYZING_MODEL,
    )
    connection.close()

    assert [item["path"] for item in plan.pending] == [str(source)]
    assert plan.kept == 0


#: A disclosure as `run()` renders it, for the consent-grammar cases.
RENDERED_PROMPT = cli_module.DISCLOSURE_PROMPT.format(summary="Summary of this run.")


def test_confirm_disclaimer_declines_bare_enter(monkeypatch) -> None:
    """A stray newline is not consent; the advertised default declines."""
    shown = _answers(monkeypatch, "")

    assert RepoMapApp()._confirm_disclaimer(RENDERED_PROMPT) is False
    assert shown == [RENDERED_PROMPT]
    assert shown[0].endswith("Proceed? [y/N]: ")


def test_confirm_disclaimer_declines_whitespace_only_input(monkeypatch) -> None:
    _answers(monkeypatch, "   \t ")

    assert RepoMapApp()._confirm_disclaimer(RENDERED_PROMPT) is False


@pytest.mark.parametrize("response", ["y", "yes", "Y", "YES", " yes "])
def test_confirm_disclaimer_accepts_explicit_yes(monkeypatch, response: str) -> None:
    _answers(monkeypatch, response)

    assert RepoMapApp()._confirm_disclaimer(RENDERED_PROMPT) is True


@pytest.mark.parametrize("response", ["n", "no", "N", "NO"])
def test_confirm_disclaimer_declines_explicit_no(monkeypatch, response: str) -> None:
    _answers(monkeypatch, response)

    assert RepoMapApp()._confirm_disclaimer(RENDERED_PROMPT) is False


def test_confirm_disclaimer_reprompts_unrecognized_input(monkeypatch, capsys) -> None:
    shown = _answers(monkeypatch, "maybe", "y")

    assert RepoMapApp()._confirm_disclaimer(RENDERED_PROMPT) is True
    assert shown == [RENDERED_PROMPT, RENDERED_PROMPT]
    assert "Invalid input" in capsys.readouterr().out


def test_confirm_disclaimer_cancels_on_closed_stdin(monkeypatch, capsys) -> None:
    """EOF declines and reports cancellation -- unchanged behavior."""

    def raise_eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    assert RepoMapApp()._confirm_disclaimer(RENDERED_PROMPT) is False
    assert "Operation cancelled." in capsys.readouterr().out


def test_confirm_disclaimer_cancels_on_keyboard_interrupt(monkeypatch, capsys) -> None:
    def raise_interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_interrupt)

    assert RepoMapApp()._confirm_disclaimer(RENDERED_PROMPT) is False
    assert "Operation cancelled." in capsys.readouterr().out


def test_yes_flag_bypasses_disclosure_prompt(monkeypatch, tmp_path, capsys) -> None:
    """``-y`` never prompts, and a cache failure still exits 1 with one line."""
    _write(tmp_path / "mod.py", 10)
    _refuse_input(monkeypatch)

    def explode(_repository_path: str):
        raise CacheError(f"cache is unusable at {tmp_path / CACHE_FILE_NAME}")

    monkeypatch.setattr(cli_module, "load_cache", explode)

    status = _run_app(monkeypatch, tmp_path, "-y")

    assert status == 1
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


@pytest.mark.parametrize(
    ("arguments", "env"),
    [
        pytest.param(["-y"], {"OPENROUTER_API_KEY": PLACEHOLDER_API_KEY}, id="run"),
        pytest.param(["--dry-run"], {}, id="dry-run"),
    ],
)
def test_corrupt_cache_file_exits_without_traceback(
    tmp_path, arguments: list[str], env: dict[str, str]
) -> None:
    """A corrupt `.repo-map-cache.db` reports the cache path and the remedy."""
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / CACHE_FILE_NAME).write_text("not a database", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_cli([str(repository), *arguments], cwd=scratch, env=env)

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

    async def fake_description(received_structure, file_data, model, response_format):
        nonlocal active, peak
        assert received_structure is structure
        active += 1
        peak = max(peak, active)
        index = int(file_data["name"].removesuffix(".py"))
        await asyncio.sleep(delays[index])
        file_data["description"] = f"description for {file_data['path']}"
        active -= 1
        return AnalysisOutcome.SUCCESS

    app = RepoMapApp()
    progress = _Progress()
    monkeypatch.setattr(app, "_update_cache_for_file", cache_updates.append)
    monkeypatch.setattr(cli_module, "get_llm_descriptions", fake_description)
    monkeypatch.setattr(cli_module, "tqdm", progress)
    _stub_response_format(monkeypatch)

    unanalyzed = asyncio.run(
        app._enhance_summary_with_llm(structure, structure, "test-model")
    )

    assert unanalyzed == 0
    assert peak == len(structure)
    assert progress.steps == len(structure)
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
    structure = [
        {
            "path": f"/tmp/{name}.py",
            "name": f"{name}.py",
            "type": "file",
            "source_eligible": True,
        }
        for name in ("slow-success", "failure", "fast-success")
    ]

    async def fake_description(received_structure, file_data, model, response_format):
        assert received_structure is structure
        if file_data["name"] == "slow-success.py":
            await asyncio.sleep(0.02)
            return AnalysisOutcome.SUCCESS
        if file_data["name"] == "fast-success.py":
            await asyncio.sleep(0.005)
            return AnalysisOutcome.SUCCESS
        await asyncio.sleep(0.01)
        return AnalysisOutcome.FAILURE

    app = RepoMapApp()
    progress = _Progress()
    monkeypatch.setattr(app, "_update_cache_for_file", cache_updates.append)
    monkeypatch.setattr(cli_module, "get_llm_descriptions", fake_description)
    monkeypatch.setattr(cli_module, "tqdm", progress)
    _stub_response_format(monkeypatch)

    unanalyzed = asyncio.run(
        app._enhance_summary_with_llm(structure, structure, "test-model")
    )

    assert unanalyzed == 1
    assert progress.steps == len(structure)
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
    structure = [
        {
            "path": f"/tmp/{name}.py",
            "name": f"{name}.py",
            "type": "file",
            "source_eligible": True,
        }
        for name in ("fast-success", "raiser", "slow-success")
    ]

    async def fake_description(received_structure, file_data, model, response_format):
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

    app = RepoMapApp()
    progress = _Progress()
    monkeypatch.setattr(app, "_update_cache_for_file", cache_updates.append)
    monkeypatch.setattr(cli_module, "get_llm_descriptions", fake_description)
    monkeypatch.setattr(cli_module, "tqdm", progress)
    _stub_response_format(monkeypatch)

    with caplog.at_level(logging.ERROR, logger=cli_module.__name__):
        unanalyzed = asyncio.run(
            app._enhance_summary_with_llm(structure, structure, "test-model")
        )

    assert unanalyzed == 1
    assert progress.steps == len(structure)
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

    async def shaped_response(messages, model, session_id, response_format):
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
    _stub_response_format(monkeypatch)

    status = _run_against(monkeypatch, tmp_path, ANALYZING_MODEL)

    app = RepoMapApp()
    app.cache_conn = load_cache(str(tmp_path))
    pending = _pending_names(app, tmp_path)
    app.cache_conn.close()

    assert status == 1
    assert _report(tmp_path).is_file()
    assert {row[0] for row in _cache_rows(tmp_path)} == {"early.py", "late.py"}
    assert pending == {"no-content.py", "no-message.py"}


def test_cache_entries_survive_a_repository_rename(tmp_path) -> None:
    """Moving the repository invalidates nothing -- issue #24."""
    original = tmp_path / "project"
    (original / "src").mkdir(parents=True)
    (original / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (original / "src" / "nested.py").write_text("VALUE = 2\n", encoding="utf-8")
    connection = load_cache(str(original))
    app = RepoMapApp()
    app.cache_conn = connection
    _cache_every_scanned_file(app, original)

    assert _pending_names(app, original) == set()
    connection.close()

    moved = tmp_path / "project-renamed"
    shutil.move(str(original), str(moved))
    connection = load_cache(str(moved))
    app = RepoMapApp()
    app.cache_conn = connection
    pending = _pending_names(app, moved)
    keys = _cache_keys(connection)
    connection.close()

    assert pending == set()
    assert keys == {"mod.py", "src/nested.py"}


def test_cache_copied_into_a_fresh_checkout_is_reused(tmp_path) -> None:
    """A cache database travels into another checkout at a different prefix."""
    origin = tmp_path / "origin"
    (origin / "src").mkdir(parents=True)
    (origin / "src" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    connection = load_cache(str(origin))
    app = RepoMapApp()
    app.cache_conn = connection
    _cache_every_scanned_file(app, origin)
    connection.close()

    checkout = tmp_path / "ci" / "workspace" / "checkout"
    shutil.copytree(origin, checkout)
    connection = load_cache(str(checkout))
    structure = cli_module.summarize_repo(str(checkout), connection)
    plan = cli_module.plan_upload(connection, structure, ANALYZING_MODEL)
    connection.close()

    cached_file = next(item for item in structure if item["name"] == "mod.py")
    assert plan.pending == ()
    assert plan.kept == 1
    assert cached_file["description"] == "cached"


def test_changed_file_is_still_detected_as_pending(tmp_path) -> None:
    """Regression: hash mismatch on a relative key still marks a file pending."""
    repository = tmp_path / "project"
    (repository / "src").mkdir(parents=True)
    (repository / "stable.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "src" / "edited.py").write_text("VALUE = 2\n", encoding="utf-8")
    connection = load_cache(str(repository))
    app = RepoMapApp()
    app.cache_conn = connection
    _cache_every_scanned_file(app, repository)
    (repository / "src" / "edited.py").write_text("VALUE = 3\n", encoding="utf-8")

    pending = _pending_names(app, repository)
    connection.close()

    assert pending == {"edited.py"}


def test_run_prunes_rows_for_deleted_files_and_legacy_keys(
    tmp_path, monkeypatch
) -> None:
    """A run ends with one row per scanned file and nothing else -- issue #24."""
    _stub_successful_analysis(monkeypatch)
    (tmp_path / "kept.py").write_text("VALUE = 1\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))
    for key in ("deleted.py", "src/removed.py", str(tmp_path / "kept.py")):
        connection.execute(
            "INSERT INTO cache (path, hash) VALUES (?, ?)", (key, "stale")
        )
    connection.commit()

    status = _run_against(monkeypatch, tmp_path, ANALYZING_MODEL)
    app = RepoMapApp()
    app.cache_conn = connection
    keys = _cache_keys(connection)
    pending = _pending_names(app, tmp_path)
    connection.close()

    assert status == 0
    assert keys == {"kept.py"}
    assert pending == set()


def test_two_legacy_rows_for_one_file_upgrade_without_a_unique_violation(
    tmp_path, monkeypatch
) -> None:
    """Two absolute-path rows reducing to one relative key must not collide."""
    _stub_successful_analysis(monkeypatch)
    (tmp_path / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))
    for prefix in ("/checkout-a/project", "/checkout-b/project"):
        connection.execute(
            "INSERT INTO cache (path, hash) VALUES (?, ?)",
            (f"{prefix}/mod.py", "legacy"),
        )
    connection.commit()

    status = _run_against(monkeypatch, tmp_path, ANALYZING_MODEL)
    keys = _cache_keys(connection)
    connection.close()

    assert status == 0
    assert keys == {"mod.py"}


def test_an_unchanged_file_is_requeued_under_a_different_model(tmp_path) -> None:
    """A stored analysis belongs to the model that produced it -- issue #13."""
    repository = tmp_path / "project"
    (repository / "src").mkdir(parents=True)
    (repository / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "src" / "nested.py").write_text("VALUE = 2\n", encoding="utf-8")
    connection = load_cache(str(repository))
    app = RepoMapApp()
    app.cache_conn = connection
    _cache_every_scanned_file(app, repository, model="vendor/model-a")

    same_model = _pending_names(app, repository, model="vendor/model-a")
    other_model = _pending_names(app, repository, model="vendor/model-b")
    connection.close()

    assert same_model == set()
    assert other_model == {"mod.py", "nested.py"}


@pytest.mark.parametrize(
    "rerun_model",
    [
        pytest.param("Vendor/Model-A", id="case-variant"),
        pytest.param("vendor/model-a-20260101", id="dated-pin"),
        pytest.param("vendor/model", id="prefix"),
        pytest.param(" vendor/model-a", id="leading-space"),
        pytest.param("", id="empty"),
    ],
)
def test_a_model_string_is_matched_verbatim(tmp_path, rerun_model: str) -> None:
    """Anything but the exact recorded model is another model -- issue #13."""
    (tmp_path / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))
    app = RepoMapApp()
    app.cache_conn = connection
    _cache_every_scanned_file(app, tmp_path, model="vendor/model-a")

    pending = _pending_names(app, tmp_path, model=rerun_model)
    connection.close()

    assert pending == {"mod.py"}


def test_an_unchanged_file_is_requeued_after_a_contract_revision_bump(
    tmp_path, monkeypatch
) -> None:
    """Bumping the contract invalidates analyses produced under the old one -- issue #13."""
    (tmp_path / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))
    app = RepoMapApp()
    app.cache_conn = connection
    _cache_every_scanned_file(app, tmp_path)

    before_bump = _pending_names(app, tmp_path)
    monkeypatch.setattr(
        cli_module, "ANALYSIS_CONTRACT_VERSION", ANALYSIS_CONTRACT_VERSION + 1
    )
    after_bump = _pending_names(app, tmp_path)
    connection.close()

    assert before_bump == set()
    assert after_bump == {"mod.py"}


def test_a_model_change_reanalyzes_and_replaces_the_cached_row(
    tmp_path, monkeypatch
) -> None:
    """The row is superseded in place, not duplicated and not pruned -- issue #13."""
    requested = _stub_successful_analysis(monkeypatch)
    (tmp_path / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")

    _run_against(monkeypatch, tmp_path, "vendor/model-a")
    first_run = list(requested)
    requested.clear()
    _run_against(monkeypatch, tmp_path, "vendor/model-a")
    repeat_run = list(requested)
    requested.clear()
    _run_against(monkeypatch, tmp_path, "vendor/model-b")
    changed_run = list(requested)

    assert first_run == ["vendor/model-a"]
    assert repeat_run == []
    assert changed_run == ["vendor/model-b"]
    assert _cache_rows(tmp_path) == [
        ("mod.py", "vendor/model-b", ANALYSIS_CONTRACT_VERSION, "A module.")
    ]


def test_a_contract_revision_bump_reanalyzes_and_restamps_the_cached_row(
    tmp_path, monkeypatch
) -> None:
    """The same model under a new contract revision is a miss -- issue #13."""
    requested = _stub_successful_analysis(monkeypatch)
    (tmp_path / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    _run_against(monkeypatch, tmp_path, "vendor/model-a")
    requested.clear()

    bumped = ANALYSIS_CONTRACT_VERSION + 1
    monkeypatch.setattr(llm_module, "ANALYSIS_CONTRACT_VERSION", bumped)
    monkeypatch.setattr(cli_module, "ANALYSIS_CONTRACT_VERSION", bumped)
    _run_against(monkeypatch, tmp_path, "vendor/model-a")

    assert requested == ["vendor/model-a"]
    assert _cache_rows(tmp_path) == [("mod.py", "vendor/model-a", bumped, "A module.")]


def test_a_cache_without_identity_columns_upgrades_and_reanalyzes_once(
    tmp_path, monkeypatch
) -> None:
    """A pre-upgrade database loads, reads as a miss, then settles -- issue #13."""
    requested = _stub_successful_analysis(monkeypatch)
    source = tmp_path / "mod.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = load_source_snapshot(str(source), "Python")
    assert snapshot is not None
    _seed_pre_identity_cache(tmp_path, "mod.py", snapshot.sha256)

    _run_against(monkeypatch, tmp_path, "vendor/model-a")
    upgrade_run = list(requested)
    requested.clear()
    _run_against(monkeypatch, tmp_path, "vendor/model-a")

    assert upgrade_run == ["vendor/model-a"]
    assert requested == []
    assert _cache_rows(tmp_path) == [
        ("mod.py", "vendor/model-a", ANALYSIS_CONTRACT_VERSION, "A module.")
    ]


def test_a_failed_reanalysis_writes_no_row_and_leaves_the_stale_one_in_place(
    tmp_path, monkeypatch
) -> None:
    """A miss the analysis cannot replace survives the prune and stays pending -- issue #13."""
    (tmp_path / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))
    app = RepoMapApp()
    app.cache_conn = connection
    _cache_every_scanned_file(app, tmp_path, model="vendor/model-a")

    async def failed_analysis(messages, model, session_id, response_format):
        return {"error": {"code": 422, "message": "failed"}}

    monkeypatch.setattr(llm_module, "rate_limited_api_call", failed_analysis)
    _stub_response_format(monkeypatch)

    status = _run_against(monkeypatch, tmp_path, "vendor/model-b")
    pending = _pending_names(app, tmp_path, model="vendor/model-b")
    connection.close()

    assert status == 1
    assert pending == {"mod.py"}
    assert _cache_rows(tmp_path) == [
        ("mod.py", "vendor/model-a", ANALYSIS_CONTRACT_VERSION, "cached")
    ]


def test_update_cache_for_file_persists_the_analysis_identity(tmp_path) -> None:
    """Both identity values land on every successful write -- issue #13."""
    connection = load_cache(str(tmp_path))
    app = RepoMapApp()
    app.cache_conn = connection

    app._update_cache_for_file(
        {
            "rel_path": "src/mod.py",
            "hash": "abc123",
            "description": "A module.",
            "model": "vendor/model-a",
            "contract_version": ANALYSIS_CONTRACT_VERSION,
        }
    )
    row = connection.execute(
        "SELECT hash, model, contract_version FROM cache WHERE path = ?",
        ("src/mod.py",),
    ).fetchone()
    connection.close()

    assert row == ("abc123", "vendor/model-a", ANALYSIS_CONTRACT_VERSION)


def test_update_cache_for_file_refuses_a_file_with_no_analysis_identity(
    tmp_path,
) -> None:
    """No identity is invented for a file no analysis stamped -- issue #13."""
    connection = load_cache(str(tmp_path))
    app = RepoMapApp()
    app.cache_conn = connection

    with pytest.raises(KeyError):
        app._update_cache_for_file(
            {"rel_path": "src/mod.py", "hash": "abc123", "description": "A module."}
        )
    remaining = connection.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    connection.close()

    assert remaining == 0


def test_console_script_exits_130_on_ctrl_c(monkeypatch, capsys) -> None:
    async def interrupted(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(RepoMapApp, "run", interrupted)

    assert _console_exit_status() == 130
    assert "interrupted" in capsys.readouterr().out


def test_console_script_exits_1_on_an_unexpected_error(monkeypatch, capsys) -> None:
    async def crashed(self):
        raise RuntimeError("unforeseen")

    monkeypatch.setattr(RepoMapApp, "run", crashed)

    assert _console_exit_status() == 1
    assert "unforeseen" in capsys.readouterr().out


def test_a_successful_run_writes_the_report_and_says_so(
    tmp_path, monkeypatch, capsys
) -> None:
    _stub_successful_analysis(monkeypatch)
    (tmp_path / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")

    status = _run_app(monkeypatch, tmp_path, "-y", "--model", ANALYZING_MODEL)

    output = capsys.readouterr().out
    report = tmp_path / f"{tmp_path.name}_repo_map.md"
    assert status == 0
    assert "Description: A module." in report.read_text(encoding="utf-8")
    assert f"Your repo-map has been saved to '{report.name}'." in output
    assert _cache_rows(tmp_path) == [
        ("mod.py", ANALYZING_MODEL, ANALYSIS_CONTRACT_VERSION, "A module.")
    ]


@pytest.mark.parametrize("status", [0, 1])
def test_console_script_exits_with_the_status_run_reports(monkeypatch, status) -> None:
    async def finished(self):
        return status

    monkeypatch.setattr(RepoMapApp, "run", finished)

    assert _console_exit_status() == status


def test_prompt_states_this_runs_counts_after_the_scan(tmp_path, monkeypatch) -> None:
    """A fixture with known sizes pins the whole prompt -- issue #34."""
    _write(tmp_path / "cached.py", 100)
    _cache_all(tmp_path)
    _write(tmp_path / "alpha.py", 1500)
    _write(tmp_path / "src" / "beta.py", 650)
    requests = _record_requests(monkeypatch)
    shown = _answers(monkeypatch, "n")

    status = _run_app(monkeypatch, tmp_path, "--model", ANALYZING_MODEL)

    assert status == 0
    assert shown == [
        "repo-map will send the full text of 2 files (2.1 KiB) to OpenRouter "
        "(test-model), plus the repository's file and directory names and short "
        "notes on related files. 1 file with a current cached analysis keeps it; "
        "its source is not sent.\n"
        "Ignore rules come from repo-map's defaults, the repository's .gitignore "
        "files at every level, and .git/info/exclude; git's global excludes file "
        "is not read.\n"
        "Run with --dry-run to list the files. Proceed? [y/N]: "
    ]
    assert requests == []


def test_disclosure_names_every_ignore_source_and_the_unread_global_excludes(
    tmp_path, monkeypatch
) -> None:
    """The ignore-rule sentence tracks the scanner's rule set -- issue #35."""
    _write(tmp_path / "mod.py", 10)
    _record_requests(monkeypatch)
    shown = _answers(monkeypatch, "n")

    _run_app(monkeypatch, tmp_path)

    prompt = shown[0]
    assert "the full text of 1 file " in prompt
    assert "OpenRouter" in prompt
    assert ".gitignore files at every level" in prompt
    assert ".git/info/exclude" in prompt
    assert "global excludes file is not read" in prompt
    assert "not applied" not in prompt


def test_dry_run_without_a_key_lists_each_pending_file_and_contacts_nobody(
    tmp_path, monkeypatch, capsys
) -> None:
    _write(tmp_path / "src" / "b.py", 20)
    _write(tmp_path / "a.py", 10)
    _write(tmp_path / "src" / "a.py", 30)
    requests = _record_requests(monkeypatch)
    _refuse_input(monkeypatch)

    status = _run_app(
        monkeypatch, tmp_path, "--dry-run", "--model", ANALYZING_MODEL, api_key=None
    )

    assert status == 0
    assert capsys.readouterr().out.splitlines() == [
        "repo-map will send the full text of 3 files (0.1 KiB) to OpenRouter "
        "(test-model), plus the repository's file and directory names and short "
        "notes on related files. 0 files with a current cached analysis keep it; "
        "their source is not sent.",
        "10\ta.py",
        "30\tsrc/a.py",
        "20\tsrc/b.py",
    ]
    assert requests == []


def test_dry_run_counts_a_current_analysis_as_kept_and_another_model_as_pending(
    tmp_path, monkeypatch, capsys
) -> None:
    _write(tmp_path / "mod.py", 10)
    _cache_all(tmp_path, model="vendor/model-a")
    _record_requests(monkeypatch)

    same = _run_app(
        monkeypatch, tmp_path, "--dry-run", "--model", "vendor/model-a", api_key=None
    )
    same_lines = capsys.readouterr().out.splitlines()
    other = _run_app(
        monkeypatch, tmp_path, "--dry-run", "--model", "vendor/model-b", api_key=None
    )
    other_lines = capsys.readouterr().out.splitlines()

    assert (same, other) == (0, 0)
    assert same_lines == [
        "repo-map will send the full text of 0 files (0.0 KiB) to OpenRouter "
        "(vendor/model-a), plus the repository's file and directory names and "
        "short notes on related files. 1 file with a current cached analysis "
        "keeps it; its source is not sent."
    ]
    assert other_lines == [
        "repo-map will send the full text of 1 file (0.0 KiB) to OpenRouter "
        "(vendor/model-b), plus the repository's file and directory names and "
        "short notes on related files. 0 files with a current cached analysis "
        "keep it; their source is not sent.",
        "10\tmod.py",
    ]


@pytest.mark.parametrize(
    ("arguments", "answers", "api_key"),
    [
        pytest.param(("--dry-run",), (), None, id="dry-run"),
        pytest.param((), ("",), PLACEHOLDER_API_KEY, id="declined-with-bare-enter"),
    ],
)
def test_dry_run_and_a_declined_prompt_leave_the_target_byte_identical(
    tmp_path, monkeypatch, arguments, answers, api_key
) -> None:
    _write(tmp_path / "mod.py", 10)
    _write(tmp_path / "docs" / "guide.py", 20)
    requests = _record_requests(monkeypatch)
    _answers(monkeypatch, *answers)
    before = _tree(tmp_path)

    status = _run_app(monkeypatch, tmp_path, *arguments, api_key=api_key)

    assert status == 0
    assert _tree(tmp_path) == before
    assert requests == []


def test_dry_run_reads_a_pre_identity_cache_unmigrated_in_a_read_only_target(
    tmp_path, monkeypatch, capsys
) -> None:
    repository = tmp_path / "repo"
    source = repository / "mod.py"
    _write(source, 10)
    source_snapshot = load_source_snapshot(str(source), "Python")
    assert source_snapshot is not None
    _seed_pre_identity_cache(repository, "mod.py", source_snapshot.sha256)
    _record_requests(monkeypatch)
    before = _tree(repository)
    repository.chmod(0o555)

    try:
        status = _run_app(
            monkeypatch,
            repository,
            "--dry-run",
            "--model",
            ANALYZING_MODEL,
            api_key=None,
        )
    finally:
        repository.chmod(0o755)

    assert status == 0
    assert capsys.readouterr().out.splitlines()[1:] == ["10\tmod.py"]
    assert _tree(repository) == before


def test_dry_run_wins_over_yes_and_never_prompts(tmp_path, monkeypatch, capsys) -> None:
    _write(tmp_path / "mod.py", 10)
    requests = _record_requests(monkeypatch)
    _refuse_input(monkeypatch)

    status = _run_app(monkeypatch, tmp_path, "--dry-run", "-y", api_key=None)

    assert status == 0
    assert capsys.readouterr().out.splitlines()[1:] == ["10\tmod.py"]
    assert requests == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["mod.py"]


def test_a_file_created_while_the_prompt_waits_is_not_sent(
    tmp_path, monkeypatch
) -> None:
    """Every file sent was counted, and nothing is sent before consent -- issue #34."""
    _write(tmp_path / "a.py", 10)
    _write(tmp_path / "b.py", 10)
    requests = _record_requests(monkeypatch)
    prompts: list[str] = []
    before_consent: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        before_consent.extend(requests)
        _write(tmp_path / "late.py", 10)
        return "y"

    monkeypatch.setattr("builtins.input", answer)

    status = _run_app(monkeypatch, tmp_path, "--model", ANALYZING_MODEL)

    assert status == 0
    assert "the full text of 2 files " in prompts[0]
    assert before_consent == []
    assert sorted(requests) == ["a.py", "b.py", "response_format"]


@pytest.mark.parametrize("contents", ["empty", "all-cached"])
def test_with_nothing_to_send_the_prompt_is_skipped_and_the_report_written(
    tmp_path, monkeypatch, contents
) -> None:
    if contents == "all-cached":
        _write(tmp_path / "mod.py", 10)
        _cache_all(tmp_path)
    requests = _record_requests(monkeypatch)
    _refuse_input(monkeypatch)

    status = _run_app(monkeypatch, tmp_path, "--model", ANALYZING_MODEL)

    assert status == 0
    assert _report(tmp_path).is_file()
    assert requests == []


def test_yes_logs_the_summary_and_sends_what_the_prompt_run_sends(
    tmp_path, monkeypatch, capsys
) -> None:
    prompted = tmp_path / "prompted"
    unattended = tmp_path / "unattended"
    for repository in (prompted, unattended):
        _write(repository / "a.py", 10)
        _write(repository / "src" / "b.py", 20)
    requests = _record_requests(monkeypatch)
    shown = _answers(monkeypatch, "y")
    summary = (
        "repo-map will send the full text of 2 files (0.0 KiB) to OpenRouter "
        "(test-model), plus the repository's file and directory names and short "
        "notes on related files. 0 files with a current cached analysis keep it; "
        "their source is not sent."
    )

    prompted_status = _run_app(monkeypatch, prompted, "--model", ANALYZING_MODEL)
    prompted_requests = sorted(requests)
    requests.clear()
    capsys.readouterr()
    unattended_status = _run_app(
        monkeypatch, unattended, "-y", "--model", ANALYZING_MODEL
    )
    output = capsys.readouterr().out

    assert (prompted_status, unattended_status) == (0, 0)
    assert shown[0].startswith(f"{summary}\n")
    assert f" - INFO - {summary}" in output
    assert "--dry-run" not in output
    assert prompted_requests == ["a.py", "response_format", "src/b.py"]
    assert sorted(requests) == prompted_requests


def test_an_unwritable_report_path_logs_one_error_and_exits_1(
    tmp_path, monkeypatch, capsys
) -> None:
    _write(tmp_path / "mod.py", 10)
    _record_requests(monkeypatch)
    _report(tmp_path).mkdir()

    status = _run_app(monkeypatch, tmp_path, "-y", "--model", ANALYZING_MODEL)

    output = capsys.readouterr().out
    errors = [line for line in output.splitlines() if " - ERROR - " in line]
    assert status == 1
    assert len(errors) == 1
    assert _report(tmp_path).name in errors[0]
    assert "has been saved" not in output


@pytest.mark.parametrize(
    ("failing", "last_line"),
    [
        ({"b.py"}, "1 file was not analyzed; rerun repo-map to retry it."),
        ({"b.py", "c.py"}, "2 files were not analyzed; rerun repo-map to retry them."),
    ],
)
def test_a_partial_run_writes_the_report_then_says_what_a_rerun_retries(
    tmp_path, monkeypatch, capsys, failing, last_line
) -> None:
    for name in ("a.py", "b.py", "c.py"):
        _write(tmp_path / name, 10)
    _stub_response_format(monkeypatch)

    async def analysis(messages, model, session_id, response_format):
        target = TARGET_FILE.search(messages[1]["content"])
        assert target is not None
        content = "not json" if target.group(1) in failing else WELL_FORMED_ANALYSIS
        return {"choices": [{"message": {"content": json.dumps(content)}}]}

    monkeypatch.setattr(llm_module, "rate_limited_api_call", analysis)

    status = _run_app(monkeypatch, tmp_path, "-y", "--model", ANALYZING_MODEL)

    output = capsys.readouterr().out
    assert status == 1
    assert _report(tmp_path).is_file()
    assert "has been saved" in output
    assert output.splitlines()[-1].endswith(last_line)
    assert {row[0] for row in _cache_rows(tmp_path)} == {"a.py", "b.py", "c.py"} - (
        failing
    )


def test_a_fatal_openrouter_error_ends_the_run_without_a_tree_or_report(
    tmp_path, monkeypatch, capsys
) -> None:
    """The issue's reproduction with the 401 stubbed: one line, no report -- issue #33."""
    _write(tmp_path / "a.py", 10)
    _write(tmp_path / "b.py", 10)
    _stub_analysis_except(
        monkeypatch,
        {
            "a.py": FatalOpenRouterError(401, "User not found."),
            "b.py": FatalOpenRouterError(401, "User not found."),
        },
    )

    status = _run_app(monkeypatch, tmp_path, "-y", "--model", ANALYZING_MODEL)

    output = capsys.readouterr().out
    assert status == 1
    assert output.count("User not found.") == 1
    assert "Updated Repository Map" not in output
    assert "has been saved" not in output
    assert not _report(tmp_path).exists()


def test_analyses_finished_before_a_fatal_error_stay_cached(
    tmp_path, monkeypatch
) -> None:
    for name in ("a.py", "b.py", "z.py"):
        _write(tmp_path / name, 10)
    _record_requests(monkeypatch)

    async def analysis(structure, file_data, model, response_format):
        if file_data["name"] == "z.py":
            # Yield until the peers have finished and been recorded.
            for _ in range(20):
                await asyncio.sleep(0)
            raise FatalOpenRouterError(402, "Insufficient credits.")
        return await llm_module.get_llm_descriptions(
            structure, file_data, model=model, response_format=response_format
        )

    monkeypatch.setattr(cli_module, "get_llm_descriptions", analysis)

    status = _run_app(monkeypatch, tmp_path, "-y", "--model", ANALYZING_MODEL)

    assert status == 1
    assert [row[0] for row in _cache_rows(tmp_path)] == ["a.py", "b.py"]
    assert not _report(tmp_path).exists()


def test_a_fatal_error_cancels_the_analyses_still_in_flight(monkeypatch) -> None:
    """The first task's fatal error leaves no task running or pending behind it."""
    structure = [
        {"path": f"/tmp/{name}.py", "name": f"{name}.py", "type": "file"}
        for name in ("first", "slow-1", "slow-2")
    ]
    cancelled: list[str] = []
    finished: list[str] = []

    async def fake_description(received_structure, file_data, model, response_format):
        if file_data["name"] == "first.py":
            raise FatalOpenRouterError(401, "User not found.")
        try:
            for _ in range(50):
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            cancelled.append(file_data["name"])
            raise
        finished.append(file_data["name"])
        return AnalysisOutcome.SUCCESS

    app = RepoMapApp()
    monkeypatch.setattr(app, "_update_cache_for_file", lambda file_data: None)
    monkeypatch.setattr(cli_module, "get_llm_descriptions", fake_description)
    monkeypatch.setattr(cli_module, "tqdm", _Progress())
    _stub_response_format(monkeypatch)

    async def scenario() -> set[asyncio.Task]:
        with pytest.raises(FatalOpenRouterError):
            await app._enhance_summary_with_llm(structure, structure, "test-model")
        return asyncio.all_tasks() - {asyncio.current_task()}

    assert asyncio.run(scenario()) == set()
    assert sorted(cancelled) == ["slow-1.py", "slow-2.py"]
    assert finished == []


def test_an_unexpected_exception_fails_only_its_file_and_the_run(
    tmp_path, monkeypatch, capsys
) -> None:
    for name in ("a.py", "b.py", "c.py"):
        _write(tmp_path / name, 10)
    _stub_analysis_except(monkeypatch, {"b.py": TypeError("bad payload")})

    status = _run_app(monkeypatch, tmp_path, "-y", "--model", ANALYZING_MODEL)

    output = capsys.readouterr().out
    assert status == 1
    assert _report(tmp_path).is_file()
    assert [row[0] for row in _cache_rows(tmp_path)] == ["a.py", "c.py"]
    assert output.splitlines()[-1].endswith(
        "1 file was not analyzed; rerun repo-map to retry it."
    )


def test_console_script_exits_1_when_the_report_cannot_be_written(tmp_path) -> None:
    """Nothing to send, so the run reaches the write without a request."""
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "repo_repo_map.md").mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_cli(
        [str(repository), "-y"],
        cwd=scratch,
        env={"OPENROUTER_API_KEY": PLACEHOLDER_API_KEY},
    )

    _assert_single_line_failure(result, expected="repo_repo_map.md")
    assert "has been saved" not in result.stdout + result.stderr


def test_closed_stdin_at_the_prompt_cancels_and_exits_0(tmp_path) -> None:
    """Regression: end of file at the prompt is a cancellation -- issue #23."""
    repository = tmp_path / "repo"
    _write(repository / "mod.py", 10)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = _run_cli(
        [str(repository)],
        cwd=scratch,
        env={"OPENROUTER_API_KEY": PLACEHOLDER_API_KEY},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Operation cancelled." in result.stdout
    assert sorted(path.name for path in repository.iterdir()) == ["mod.py"]
