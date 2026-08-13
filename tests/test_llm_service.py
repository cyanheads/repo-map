"""Tests for prompt construction and structured LLM-response parsing."""

import asyncio
import json

import pytest

import repo_map.llm_service as llm_module
from repo_map.llm_service import (
    APIRateLimiter,
    _build_user_prompt,
    parse_llm_response,
)


def test_parse_llm_response_normalizes_closed_fields() -> None:
    file_data = {"path": "/repo/module.py"}
    parse_llm_response(
        json.dumps(
            {
                "description": "Handles work.",
                "developer_consideration": None,
                "maintenance_flag": "auto-generated",
                "critical_dependencies": {"aiohttp": "HTTP client"},
                "architectural_role": "service",
                "refactoring_suggestions": None,
                "security_assessment": None,
            }
        ),
        file_data,
    )

    assert file_data["description"] == "Handles work."
    assert file_data["developer_consideration"] == ""
    assert file_data["maintenance_flag"] == "Generated"
    assert file_data["critical_dependencies"] == '{"aiohttp": "HTTP client"}'
    assert file_data["architectural_role"] == "Service"
    assert file_data["refactoring_suggestions"] == "None"
    assert file_data["security_assessment"] == "None"


def test_parse_llm_response_accepts_json_fence() -> None:
    file_data = {"path": "/repo/module.py"}
    parse_llm_response(
        '```json\n{"description":"Summary","maintenance_flag":"stable"}\n```',
        file_data,
    )

    assert file_data["description"] == "Summary"
    assert file_data["maintenance_flag"] == "Stable"


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

    prompt = _build_user_prompt(structure, structure[1])

    assert "src/module.py (Python)  [CURRENT FILE]" in prompt
    assert "Imports: pathlib" in prompt
    assert "Symbols: run" in prompt


def test_rate_limiter_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        APIRateLimiter(0)


def test_http_429_retries_are_bounded_and_release_semaphore(monkeypatch) -> None:
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
                            "content": '{"description":"completed"}',
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

    async def exercise_retry_contract() -> None:
        nonlocal request_count
        file_data = {
            "path": "/repo/module.py",
            "name": "module.py",
            "level": 0,
            "type": "file",
            "language": "Python",
            "imports": [],
            "functions": ["run"],
        }
        responses.extend([Response(429, "0"), Response(200)])
        await asyncio.wait_for(
            llm_module.get_llm_descriptions(
                [file_data], file_data, "test-model", max_retries=3
            ),
            timeout=0.05,
        )
        assert request_count == 2
        assert sleep_delays == [0.0]
        assert file_data["description"] == "completed"

        request_count = 0
        sleep_delays.clear()
        responses.extend(Response(429, "1.5") for _ in range(3))
        await llm_module.get_llm_descriptions(
            [file_data], file_data, "test-model", max_retries=3
        )
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
