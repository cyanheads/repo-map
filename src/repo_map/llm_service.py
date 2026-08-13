"""Handles interactions with the Language Learning Model (LLM)."""

import asyncio
import json
import logging
import os
import ssl
from typing import Any

import aiohttp
import certifi

from repo_map.config import settings

logger = logging.getLogger(__name__)


class APIRateLimiter:
    """Manages the rate of API calls using a semaphore."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("Concurrency limit must be greater than zero")
        self._semaphore = asyncio.Semaphore(limit)

    def update_limit(self, limit: int) -> None:
        """Updates the concurrency limit by creating a new semaphore."""
        if limit < 1:
            raise ValueError("Concurrency limit must be greater than zero")
        self._semaphore = asyncio.Semaphore(limit)

    async def __aenter__(self) -> None:
        """Acquire the semaphore."""
        await self._semaphore.acquire()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release the semaphore."""
        self._semaphore.release()


# Global instance of the rate limiter
rate_limiter = APIRateLimiter(settings.api_semaphore_limit)


def update_api_semaphore_limit(limit: int) -> None:
    """Updates the concurrency limit used for OpenRouter requests."""
    if limit < 1:
        raise ValueError("concurrency must be greater than zero")

    settings.api_semaphore_limit = limit
    rate_limiter.update_limit(limit)


ARCHITECTURAL_ROLES = (
    "Entrypoint",
    "Configuration",
    "Service",
    "Data Model",
    "Persistence",
    "UI Component",
    "API Route",
    "Utility",
    "Test",
    "Tooling",
    "Other",
)

SYSTEM_PROMPT = """
You analyze a single source file and return structured JSON documentation. The user supplies the repository tree (with prior documentation where available) for context, plus the target file's path, language, imports, and top-level symbols.

Return one JSON object with exactly these fields:

{
  "description": string,                          // 1-2 sentences on the file's role and responsibility
  "developer_consideration": string | null,       // The single most important thing a contributor needs to know — non-obvious dependency, performance pitfall, security concern, or usage pattern. null if nothing stands out.
  "maintenance_flag": "Stable" | "Volatile" | "Generated" | "Unknown",
  "critical_dependencies": { [import_name: string]: string },  // import name -> one-line justification; empty object if none notable
  "architectural_role": "Entrypoint" | "Configuration" | "Service" | "Data Model" | "Persistence" | "UI Component" | "API Route" | "Utility" | "Test" | "Tooling" | "Other",
  "refactoring_suggestions": string | null,       // concrete, actionable; null if the file is sound
  "security_assessment": string | null            // specific risk and mitigation; null if no concern
}

Maintenance flag values:
- Stable: foundational; rarely changes
- Volatile: under active iteration (business logic, UI, configuration)
- Generated: machine-produced; do not edit
- Unknown: insufficient context

Guidelines:
- Ground every claim in the file's actual content and symbols. Use the repo tree to disambiguate role, not as a substitute for the file's evidence.
- Evaluate against the conventions of the file's language, not generic best practices.
- Test files are judged as tests (clarity, isolation, coverage), not as production code.
- Trivial files (empty `__init__.py`, single-line config, simple re-exports): provide description only; null/Unknown the rest.
- Prefer null and "Unknown" over guessing. Speculation is worse than absence.
- Reference dependencies by import name (e.g. "aiohttp", "@tanstack/react-query"), not by description.
- When refining existing documentation, preserve accurate prior content; only revise where you have stronger evidence.
- Output the JSON object only — no prose before or after, no markdown fences.

Example output:
{
  "description": "Async OpenRouter client with semaphore-based concurrency limits and exponential-backoff retry on rate-limit responses.",
  "developer_consideration": "Rate limiter is a module-level singleton; tests must reset or override it to avoid state bleed across cases.",
  "maintenance_flag": "Volatile",
  "critical_dependencies": {
    "aiohttp": "Async HTTP transport for non-blocking concurrent requests.",
    "certifi": "Root CA bundle for SSL verification on platforms with stale system stores."
  },
  "architectural_role": "Service",
  "refactoring_suggestions": "Extract the response-handling branches into a typed result object to clarify success vs error paths.",
  "security_assessment": null
}
""".strip()


async def get_llm_descriptions(
    structure: list[dict[str, Any]],
    file: dict[str, Any],
    model: str,
    max_retries: int = 3,
) -> None:
    """Generate file-level documentation via the configured LLM."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(structure, file)},
    ]

    for retry_count in range(max_retries):
        try:
            response = await rate_limited_api_call(messages, model, 0.0)
        except aiohttp.ClientError as exc:
            retry_after = 5 * (2**retry_count)
            logger.warning(
                "Error communicating with OpenRouter LLM: %s. Retrying after %s seconds...",
                exc,
                retry_after,
            )
            await asyncio.sleep(retry_after)
            continue

        if "error" in response:
            if response["error"].get("code") == 429:
                retry_after = response["error"].get("retry_after", 5) * (2**retry_count)
                logger.warning(
                    "Rate limit exceeded. Retrying after %s seconds...",
                    retry_after,
                )
                await asyncio.sleep(retry_after)
                continue
            logger.error("Error from OpenRouter LLM: %s", response["error"])
            return

        if response.get("choices"):
            content = response["choices"][0]["message"]["content"].strip()
            parse_llm_response(content, file)
            return

        logger.error("Unexpected response structure from LLM.")
        return

    logger.error(
        "Failed to get descriptions for %s after %s retries.", file["name"], max_retries
    )


def _build_user_prompt(structure: list[dict[str, Any]], file: dict[str, Any]) -> str:
    """Build the user message: repository tree as background, target file as the task."""
    repo_root = os.path.dirname(structure[0]["path"]) if structure else ""

    def rel(path: str) -> str:
        return os.path.relpath(path, repo_root) if repo_root else path

    lines = ["Repository tree (background context):", ""]
    for itm in structure:
        indent = "  " * itm["level"]
        if itm["type"] == "directory":
            lines.append(f"{indent}- {itm['name']}/")
            continue

        language = itm.get("language") or "—"
        is_current = itm["path"] == file["path"]
        marker = "  [CURRENT FILE]" if is_current else ""
        lines.append(f"{indent}- {rel(itm['path'])} ({language}){marker}")

        detail_indent = indent + "    "
        if itm.get("description"):
            lines.append(f"{detail_indent}description: {itm['description']}")
        if itm.get("developer_consideration"):
            lines.append(
                f"{detail_indent}consideration: {itm['developer_consideration']}"
            )

    lines.extend(
        [
            "",
            f"Target file: `{rel(file['path'])}` ({file.get('language') or 'unknown'})",
        ]
    )
    if file.get("imports"):
        lines.append(f"Imports: {', '.join(file['imports'])}")
    if file.get("functions"):
        lines.append(f"Symbols: {', '.join(file['functions'])}")
    if not file.get("imports") and not file.get("functions"):
        lines.append("(No symbols extracted — likely config, data, or trivial.)")

    lines.append("")
    lines.append("Return the JSON object specified in the system prompt.")
    return "\n".join(lines)


def parse_llm_response(content: str, file: dict[str, Any]) -> None:
    """Parse the LLM's JSON response into the file metadata dictionary."""
    data = _load_json(content)
    if not isinstance(data, dict):
        logger.error(
            "LLM response was not a JSON object for %s",
            file.get("path", "<unknown>"),
        )
        return

    file["description"] = _coerce_str(
        data.get("description"), file.get("description", "")
    )
    file["developer_consideration"] = _coerce_str(
        data.get("developer_consideration"), file.get("developer_consideration", "")
    )
    file["maintenance_flag"] = _normalize_maintenance_flag(
        _coerce_str(data.get("maintenance_flag"), "Unknown")
    )

    deps = data.get("critical_dependencies")
    file["critical_dependencies"] = json.dumps(deps if isinstance(deps, dict) else {})

    file["architectural_role"] = _normalize_architectural_role(
        _coerce_str(data.get("architectural_role"), "Unknown")
    )
    file["refactoring_suggestions"] = (
        _coerce_str(
            data.get("refactoring_suggestions"),
            file.get("refactoring_suggestions", "None"),
        )
        or "None"
    )
    file["security_assessment"] = (
        _coerce_str(
            data.get("security_assessment"), file.get("security_assessment", "None")
        )
        or "None"
    )


def _load_json(content: str) -> Any:
    """Parse JSON, tolerating ```json fences some models add despite instructions."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        stripped = content.strip()
        for fence in ("```json", "```JSON", "```"):
            if stripped.startswith(fence):
                stripped = stripped[len(fence) :].lstrip()
                break
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.error("Could not parse LLM response as JSON: %s", exc)
            return None


def _coerce_str(value: Any, default: str) -> str:
    """Return value as a stripped string, falling back to default for null/empty."""
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _normalize_maintenance_flag(raw_flag: str) -> str:
    """Normalize maintenance flag values to the supported vocabulary."""
    allowed = {"stable", "volatile", "generated", "unknown"}
    candidate = raw_flag.strip().lower()
    if candidate in allowed:
        return candidate.capitalize()

    mappings = {
        "gen": "Generated",
        "auto-generated": "Generated",
        "autogenerated": "Generated",
        "unstable": "Volatile",
    }
    return mappings.get(candidate, "Unknown")


def _normalize_architectural_role(raw_role: str) -> str:
    """Snap architectural role to the closed enum, defaulting to 'Other'."""
    candidate = raw_role.strip()
    if candidate in ARCHITECTURAL_ROLES:
        return candidate
    lookup = {role.lower(): role for role in ARCHITECTURAL_ROLES}
    return lookup.get(candidate.lower(), "Other")


async def rate_limited_api_call(
    messages: list[dict[str, str]], model: str, temperature: float
) -> Any:
    """Perform a rate-limited API call to the OpenRouter LLM using aiohttp."""
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    async with rate_limiter:
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": settings.http_referer,
            "X-Title": settings.app_name,
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    settings.openrouter_api_url,
                    headers=headers,
                    json=data,
                    ssl=ssl_context,
                ) as response:
                    if response.status != 200:
                        rate_limited, retry_after = await handle_rate_limiting_async(
                            response
                        )
                        if rate_limited:
                            logger.warning(
                                "Rate limited by API. Retrying after %s seconds...",
                                retry_after,
                            )
                            await asyncio.sleep(retry_after)
                            return await rate_limited_api_call(
                                messages, model, temperature
                            )
                        response_text = await response.text()
                        logger.error(
                            "API request failed with status %s: %s",
                            response.status,
                            response_text,
                        )
                        response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as exc:
            logger.error("API request failed: %s", exc)
            raise
        except ssl.SSLCertVerificationError as exc:
            logger.error("SSL Certificate Verification Error: %s", exc)
            raise


async def handle_rate_limiting_async(response) -> tuple[bool, int]:
    """Determine whether a response indicates rate limiting and the retry delay."""
    if response.status == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                retry_after = int(retry_after)
            except ValueError:
                retry_after = 5
        else:
            retry_after = 5
        return True, retry_after
    return False, 0
