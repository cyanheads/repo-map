"""Tests for prompt construction and structured LLM-response parsing."""

import asyncio
import json

import pytest

import repo_map.llm_service as llm_module
from repo_map.file_scanner import load_source_snapshot
from repo_map.llm_service import (
    AnalysisOutcome,
    APIRateLimiter,
    _build_user_prompt,
    get_llm_descriptions,
    parse_llm_response,
)


def _valid_payload(**overrides) -> dict:
    payload = {
        "description": "Handles work.",
        "developer_consideration": None,
        "maintenance_flag": "Unknown",
        "critical_dependencies": {},
        "architectural_role": "Other",
        "refactoring_suggestions": None,
        "security_assessment": None,
    }
    payload.update(overrides)
    return payload


def test_parse_llm_response_normalizes_closed_fields() -> None:
    file_data = {"path": "/repo/module.py"}
    outcome = parse_llm_response(
        json.dumps(
            _valid_payload(
                maintenance_flag="auto-generated",
                critical_dependencies={"aiohttp": "HTTP client"},
                architectural_role="service",
            )
        ),
        file_data,
    )

    assert outcome is AnalysisOutcome.SUCCESS
    assert file_data["description"] == "Handles work."
    assert file_data["developer_consideration"] == ""
    assert file_data["maintenance_flag"] == "Generated"
    assert file_data["critical_dependencies"] == '{"aiohttp": "HTTP client"}'
    assert file_data["architectural_role"] == "Service"
    assert file_data["refactoring_suggestions"] == "None"
    assert file_data["security_assessment"] == "None"


def test_parse_llm_response_accepts_json_fence() -> None:
    file_data = {"path": "/repo/module.py"}
    outcome = parse_llm_response(
        f"```json\n{json.dumps(_valid_payload(description='Summary', maintenance_flag='stable'))}\n```",
        file_data,
    )

    assert outcome is AnalysisOutcome.SUCCESS
    assert file_data["description"] == "Summary"
    assert file_data["maintenance_flag"] == "Stable"


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"description": "Incomplete"}),
        json.dumps(_valid_payload(critical_dependencies=[])),
        json.dumps(_valid_payload(security_assessment=3)),
    ],
)
def test_parse_llm_response_rejects_malformed_or_incomplete_payload(content) -> None:
    file_data = {"path": "/repo/module.py", "description": "Existing"}

    outcome = parse_llm_response(content, file_data)

    assert outcome is AnalysisOutcome.FAILURE
    assert file_data == {"path": "/repo/module.py", "description": "Existing"}


def test_parse_llm_response_accepts_valid_trivial_result() -> None:
    file_data = {"path": "/repo/__init__.py"}

    outcome = parse_llm_response(
        json.dumps(_valid_payload(description="", maintenance_flag="Unknown")),
        file_data,
    )

    assert outcome is AnalysisOutcome.SUCCESS
    assert file_data["description"] == ""
    assert file_data["developer_consideration"] == ""
    assert file_data["maintenance_flag"] == "Unknown"


def test_build_user_prompt_marks_target_and_uses_relative_paths() -> None:
    structure = [
        {"path": "/repo/src", "name": "src", "level": 0, "type": "directory"},
        {
            "path": "/repo/src/module.py",
            "name": "module.py",
            "level": 1,
            "type": "file",
            "language": "Python",
            "imports": ["pathlib"],
            "functions": ["run"],
        },
    ]

    prompt = _build_user_prompt(structure, structure[1], "SOURCE_SENTINEL")

    assert "src/module.py (Python)  [CURRENT FILE]" in prompt
    assert "Imports: pathlib" in prompt
    assert "Symbols: run" in prompt
    assert "SOURCE_SENTINEL" in prompt
    assert "untrusted" in prompt


def test_rate_limiter_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        APIRateLimiter(0)


def test_http_429_retries_are_bounded_and_release_semaphore(
    tmp_path, monkeypatch
) -> None:
    responses = []
    request_count = 0
    sleep_delays = []

    class Response:
        def __init__(self, status: int, retry_after: str | None = None) -> None:
            self.status = status
            self.headers = (
                {"Retry-After": retry_after} if retry_after is not None else {}
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                _valid_payload(description="completed")
                            ),
                        }
                    }
                ]
            }

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        def post(self, *args, **kwargs) -> Response:
            nonlocal request_count
            request_count += 1
            return responses.pop(0)

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr(llm_module.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(llm_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm_module, "rate_limiter", APIRateLimiter(1))

    source = tmp_path / "module.py"
    source.write_text("def run():\n    pass\n", encoding="utf-8")
    snapshot = load_source_snapshot(str(source), "Python")
    assert snapshot is not None

    async def exercise_retry_contract() -> None:
        nonlocal request_count
        file_data = {
            "path": str(source),
            "name": "module.py",
            "level": 0,
            "type": "file",
            "language": "Python",
            "imports": [],
            "functions": ["run"],
            "hash": snapshot.sha256,
        }
        responses.extend([Response(429, "0"), Response(200)])
        outcome = await asyncio.wait_for(
            llm_module.get_llm_descriptions(
                [file_data], file_data, "test-model", max_retries=3
            ),
            timeout=0.05,
        )
        assert outcome is AnalysisOutcome.SUCCESS
        assert request_count == 2
        assert sleep_delays == [0.0]
        assert file_data["description"] == "completed"

        request_count = 0
        sleep_delays.clear()
        responses.extend(Response(429, "1.5") for _ in range(3))
        outcome = await llm_module.get_llm_descriptions(
            [file_data], file_data, "test-model", max_retries=3
        )
        assert outcome is AnalysisOutcome.FAILURE
        assert request_count == 3
        assert sleep_delays == [1.5, 3.0]

        for header, expected_delay in (
            ("0", 0.0),
            ("2.5", 2.5),
            (None, 5.0),
            ("invalid", 5.0),
            ("-1", 5.0),
            ("inf", 5.0),
            ("nan", 5.0),
        ):
            request_count = 0
            responses.append(Response(429, header))
            result = await llm_module.rate_limited_api_call([], "test-model", 0.0)
            assert request_count == 1
            assert result == {"error": {"code": 429, "retry_after": expected_delay}}

    asyncio.run(exercise_retry_contract())


def test_get_llm_descriptions_rejects_stale_snapshot_without_request(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = load_source_snapshot(str(source), "Python")
    assert snapshot is not None
    file_data = {
        "path": str(source),
        "name": "module.py",
        "level": 0,
        "type": "file",
        "language": "Python",
        "imports": [],
        "functions": [],
        "hash": snapshot.sha256,
    }
    source.write_text("VALUE = 2\n", encoding="utf-8")

    async def unexpected_request(*args, **kwargs):
        raise AssertionError("stale source must not reach OpenRouter")

    monkeypatch.setattr(llm_module, "rate_limited_api_call", unexpected_request)

    outcome = asyncio.run(get_llm_descriptions([file_data], file_data, "test-model"))

    assert outcome is AnalysisOutcome.FAILURE


@pytest.mark.parametrize(
    "response",
    [
        {"error": {"code": 500, "message": "failed"}},
        {"choices": []},
        {"choices": [{"message": {"content": "not json"}}]},
        {"choices": [{"message": {"content": json.dumps({"description": "partial"})}}]},
    ],
)
def test_get_llm_descriptions_returns_failure_for_invalid_results(
    tmp_path, monkeypatch, response
) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = load_source_snapshot(str(source), "Python")
    assert snapshot is not None
    file_data = {
        "path": str(source),
        "name": "module.py",
        "level": 0,
        "type": "file",
        "language": "Python",
        "imports": [],
        "functions": [],
        "hash": snapshot.sha256,
    }

    async def fake_call(*args, **kwargs):
        return response

    monkeypatch.setattr(llm_module, "rate_limited_api_call", fake_call)

    outcome = asyncio.run(get_llm_descriptions([file_data], file_data, "test-model"))

    assert outcome is AnalysisOutcome.FAILURE
    assert "description" not in file_data
