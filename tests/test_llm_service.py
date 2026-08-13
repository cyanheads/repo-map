"""Tests for prompt construction and structured LLM-response parsing."""

import json

import pytest

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
