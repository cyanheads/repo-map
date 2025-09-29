"""Handles interactions with the Language Learning Model (LLM)."""

import asyncio
import logging
import re
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


SYSTEM_PROMPT = """
**Objective:** You are a principal-level software architect. Your task is to generate a comprehensive, machine-readable analysis for a given source file, grounded in its content and its position within the full repository structure.

**Analytical Directives:**

1.  **Description:** A concise, 20-30 word summary of the file's primary role and responsibility.
2.  **Developer Consideration:** The single most critical insight for a developer. This could be a non-obvious dependency, a performance pitfall, a security vulnerability, or a crucial usage pattern.
3.  **Maintenance Flag:** Classify the file's expected change frequency:
    - `Stable`: Core logic or foundational code that rarely changes.
    - `Volatile`: Business logic, UI, or configurations subject to frequent iteration.
    - `Generated`: Machine-produced code; do not edit directly.
    - `Unknown`: Insufficient context.
4.  **Critical Dependencies:** Identify the most critical imported modules/packages. For each, provide a brief justification of its importance. Format as a JSON string.
5.  **Architectural Role:** Classify the file's primary role in the system's architecture (e.g., `UI Component`, `Data Model`, `Service Layer`, `Configuration`, `Utility`, `Entrypoint`).
6.  **Code Quality Score:** A 1-10 rating of the file's maintainability, readability, and adherence to best practices. 1 is poor, 10 is excellent.
7.  **Refactoring Suggestions:** A concrete, actionable suggestion for improving the file's structure, performance, or readability. If none, state "None".
8.  **Security Assessment:** A high-level analysis of potential security risks or vulnerabilities (e.g., data handling, auth, input validation). If none, state "None".

**Output Specification:**
- Your response must be a flat text block containing exactly eight lines, strictly adhering to the key-value format below.
- `Critical Dependencies` must be a valid JSON string.

```
Description: <20-30 word summary of the file's purpose>
Developer Consideration: "<Actionable insight for developers>"
Maintenance Flag: <Stable|Volatile|Generated|Unknown>
Critical Dependencies: <JSON string: {"dependency": "justification", ...}>
Architectural Role: <Primary architectural role>
Code Quality Score: <1-10>
Refactoring Suggestions: <Concrete suggestion or "None">
Security Assessment: <Security analysis or "None">
```

**Example Output for a Modern TypeScript Project (2025):**
```
Description: Provides a reactive hook for fetching, caching, and globally managing authenticated user session data across the application.
Developer Consideration: "Leverages stale-while-revalidate caching; initial renders may show stale data for a moment before the background refetch completes."
Maintenance Flag: Stable
Critical Dependencies: {"@tanstack/react-query": "Manages asynchronous state, caching, and server-state synchronization."}
Architectural Role: State Management Hook
Code Quality Score: 9
Refactoring Suggestions: "Consider abstracting the query key into a shared constant to prevent inconsistencies across the codebase."
Security Assessment: "Ensure that sensitive user data returned by this hook is not exposed in client-side logs or error messages."
```
""".strip()


async def get_llm_descriptions(
    structure: list[dict[str, Any]],
    file: dict[str, Any],
    model: str,
    max_retries: int = 3,
) -> None:
    """Generate file-level documentation via the configured LLM."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    prompt = "**Background Context: Full Repository Map**\n"
    prompt += "Your task is to provide detailed documentation for the file marked with `-> CURRENT FILE <-`.\n"
    prompt += "The full repository map is provided for complete context. Existing documentation should be used to inform your response.\n\n"

    for itm in structure:
        is_current_file = itm["path"] == file["path"]
        marker = " -> CURRENT FILE <-" if is_current_file else ""
        indent = "│   " * itm["level"]

        if itm["type"] == "directory":
            prompt += f"{indent}├── {itm['name']}/\n"
        elif itm["type"] == "file":
            language = itm.get("language", "None")
            prompt += f"{indent}├── {itm['path']} ({language}){marker}\n"

            details_indent = indent + "│   "
            if itm.get("description"):
                prompt += f"{details_indent}└── Description: {itm['description']}\n"
            if itm.get("developer_consideration"):
                prompt += f'{details_indent}└── Developer Consideration: "{itm["developer_consideration"]}"\n'

    prompt += "\n---\n\n"
    prompt += (
        f"**Task: Generate documentation for the file marked above: `{file['path']}`**\n\n"
    )
    prompt += "**File Content Summary:**\n"

    if file.get("imports"):
        prompt += f"- Imports: {', '.join(file['imports'])}\n"
    if file.get("functions"):
        prompt += f"- Functions/Classes: {', '.join(file['functions'])}\n"
    if not file.get("imports") and not file.get("functions"):
        prompt += "- (No symbols extracted)\n"

    prompt += "\nBased on the full repository context and the file's content summary, provide the eight required documentation fields in the specified format."

    messages.append({"role": "user", "content": prompt})

    retries = 0
    while retries < max_retries:
        try:
            response = await rate_limited_api_call(messages, model, 0.0)
        except aiohttp.ClientError as exc:
            retry_after = 5 * (2**retries)
            logger.warning(
                "Error communicating with OpenRouter LLM: %s. Retrying after %s seconds...",
                exc,
                retry_after,
            )
            await asyncio.sleep(retry_after)
            retries += 1
            continue

        if "error" in response:
            if response["error"].get("code") == 429:
                retry_after = response["error"].get("retry_after", 5) * (2**retries)
                logger.warning(
                    "Rate limit exceeded. Retrying after %s seconds...",
                    retry_after,
                )
                await asyncio.sleep(retry_after)
                retries += 1
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


def parse_llm_response(content: str, file: dict[str, Any]) -> None:
    """Parse the LLM response into the file metadata dictionary."""
    def extract(pattern: str, default: str = "") -> str:
        match = re.search(pattern, content, re.IGNORECASE)
        return match.group(1).strip() if match else default

    file["description"] = extract(r"Description:\s*(.*)", file.get("description", ""))
    file["developer_consideration"] = extract(r"Developer Consideration:\s*\"(.*?)\"", file.get("developer_consideration", ""))
    file["maintenance_flag"] = _normalize_maintenance_flag(extract(r"Maintenance Flag:\s*(.*)", file.get("maintenance_flag", "Unknown")))
    file["critical_dependencies"] = extract(r"Critical Dependencies:\s*(.*)", file.get("critical_dependencies", "{}"))
    file["architectural_role"] = extract(r"Architectural Role:\s*(.*)", file.get("architectural_role", "Unknown"))

    quality_score_str = extract(r"Code Quality Score:\s*(\d+)", file.get("code_quality_score", "0"))
    try:
        file["code_quality_score"] = int(quality_score_str)
    except (ValueError, TypeError):
        file["code_quality_score"] = 0

    file["refactoring_suggestions"] = extract(r"Refactoring Suggestions:\s*(.*)", file.get("refactoring_suggestions", "None"))
    file["security_assessment"] = extract(r"Security Assessment:\s*(.*)", file.get("security_assessment", "None"))


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
    if candidate in mappings:
        return mappings[candidate]

    return "Unknown"


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
