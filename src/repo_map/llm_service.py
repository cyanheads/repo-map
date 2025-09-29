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


async def get_llm_descriptions(
    structure: list[dict[str, Any]],
    file_index: int,
    file: dict[str, Any],
    model: str,
    max_retries: int = 3,
) -> None:
    """
    Sends prompts to the LLM to generate descriptions for files.
    Updates the structure in-place with the received descriptions.
    Implements exponential backoff for retries.

    Args:
        structure (list[dict[str, Any]]): The repository structure summary.
        file_index (int): The index of the current file in the structure list.
        file (dict[str, Any]): The current file dictionary to be updated.
        model (str): The LLM model name to use for generating descriptions.
        max_retries (int): Maximum number of retries for API calls.
    """
    system_prompt = """
You are a principal software engineer documenting a codebase for other senior engineers. Your goal is to provide clear, concise, and insightful documentation.

**Primary Objectives:**
1.  **File Description:** A 5-15 word summary of the file's primary responsibility and role.
2.  **Developer Consideration:** A single, critical insight a developer needs to work with this file effectively.

**Guidelines for File Description:**
-   **Be Specific & Technical:** Instead of "handles logic," say "implements user authentication and session management."
-   **Focus on the 'What', not the 'How':** Describe its purpose, not its implementation details.
-   **Use Active Voice:** "Manages database connections" is better than "Database connections are managed."

**Guidelines for Developer Consideration:**
-   **Focus on Actionable Insights:** What should a developer *watch out for*, *be aware of*, or *leverage*?
-   **Highlight Non-Obvious Aspects:** Point out subtle complexities, performance bottlenecks, non-standard patterns, or critical dependencies.
-   **Be Concrete:** Instead of "has complex logic," say "Uses a recursive algorithm for tree traversal which can be stack-intensive."
-   **Potential Topics:**
    -   **Critical Dependencies:** "Tightly coupled with the `billing-service` API; changes here will likely require downstream updates."
    -   **Non-Standard Patterns:** "Implements a custom event bus instead of the standard library's observer pattern."
    -   **Performance/Security:** "Contains raw SQL queries; sanitize all inputs carefully to prevent injection attacks."
    -   **Hidden State/Side-Effects:** "Modifies a global configuration object, leading to potential side effects in other modules."
    -   **Critical Errors:** "Contains a potential race condition in the `update_cache` function."

**Output Format:**
You *MUST* follow this format exactly. Do not add any extra commentary.

Description: [Your concise, 5-15 word description here]
Developer Consideration: "[Your single, critical insight here]"
"""
    messages = [{"role": "system", "content": system_prompt.strip()}]

    # Construct prompt for the current file
    prompt = "**System Context: Repository Map**\n"
    prompt += "I am providing the structural map of the repository so far. Use this for context.\n\n"
    partial_map = structure[: file_index + 1]

    for itm in partial_map:
        indent = "│   " * itm["level"]
        if itm["type"] == "directory":
            prompt += f"{indent}├── {itm['name']}/\n"
        elif itm["type"] == "file":
            language = itm.get("language", "None")
            prompt += f"{indent}├── {itm['name']} ({language})\n"
            if "description" in itm and itm["description"]:
                prompt += f"{indent}│   └── Description: {itm['description']}\n"
            if "developer_consideration" in itm and itm["developer_consideration"]:
                prompt += f"{indent}│   └── Developer Consideration: \"{itm['developer_consideration']}\"\n"
            if "imports" in itm and itm["imports"]:
                prompt += f"{indent}│   ├── Imports: {', '.join(itm['imports'])}\n"
            if "functions" in itm and itm["functions"]:
                prompt += f"{indent}│   ├── Functions: {', '.join(itm['functions'])}\n"

    prompt += "\n---\n\n"
    prompt += "**Task: Document the following file**\n\n"
    language = file.get("language", "None")
    prompt += f"**File Path:** {file['name']} ({language})\n"
    if "imports" in file and file["imports"]:
        prompt += f"**Imports:** {', '.join(file['imports'])}\n"
    if "functions" in file and file["functions"]:
        prompt += f"**Functions/Classes:** {', '.join(file['functions'])}\n"

    prompt += "\nBased on the file's code and its place in the repository, generate the documentation following the system prompt's format (Description and Developer Consideration)."

    messages.append({"role": "user", "content": prompt})

    retries = 0
    while retries < max_retries:
        try:
            response = await rate_limited_api_call(messages, model, 0.0)
        except aiohttp.ClientError as e:
            retry_after = 5 * (2**retries)  # Exponential backoff
            logger.warning(
                "Error communicating with OpenRouter LLM: %s. Retrying after %s seconds...",
                e,
                retry_after,
            )
            await asyncio.sleep(retry_after)
            retries += 1
            continue

        if "error" in response:
            if response["error"].get("code") == 429:
                retry_after = response["error"].get("retry_after", 5) * (
                    2**retries
                )  # Exponential backoff
                logger.warning(
                    "Rate limit exceeded. Retrying after %s seconds...", retry_after
                )
                await asyncio.sleep(retry_after)
                retries += 1
                continue
            logger.error("Error from OpenRouter LLM: %s", response["error"])
            return

        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"].strip()
            parse_llm_response(content, file)
            return
        logger.error("Unexpected response structure from LLM.")
        return

    logger.error(
        "Failed to get descriptions for %s after %s retries.",
        file["name"],
        max_retries,
    )


def parse_llm_response(content: str, file: dict[str, Any]) -> None:
    """
    Parses the LLM response content and updates the file dictionary.
    Extracts the file-level Description and Developer Consideration.
    """
    desc_pattern = r"Description:\s*(.*)"
    consideration_pattern = r'Developer Consideration:\s*"(.*?)"'

    desc_match = re.search(desc_pattern, content, re.DOTALL)
    if desc_match:
        file["description"] = desc_match.group(1).strip()

    cons_match = re.search(consideration_pattern, content, re.DOTALL)
    if cons_match:
        file["developer_consideration"] = cons_match.group(1).strip()


async def rate_limited_api_call(
    messages: list[dict[str, str]], model: str, temperature: float
) -> Any:
    """
    Performs a rate-limited API call to the OpenRouter LLM using aiohttp.
    Utilizes certifi's CA bundle for SSL verification.
    """
    # Create an SSL context using certifi's CA bundle
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
                    ssl=ssl_context,  # Apply the SSL context here
                ) as response:
                    if response.status != 200:
                        # Handle rate limiting
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
        except aiohttp.ClientError as e:
            logger.error("API request failed: %s", e)
            raise
        except ssl.SSLCertVerificationError as e:
            logger.error("SSL Certificate Verification Error: %s", e)
            raise


async def handle_rate_limiting_async(response) -> tuple[bool, int]:
    """
    Asynchronously checks if the response status code indicates rate limiting.
    If so, returns True and the retry-after duration.
    """
    if response.status == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                retry_after = int(retry_after)
            except ValueError:
                retry_after = 5  # Default retry after 5 seconds if parsing fails
        else:
            retry_after = 5
        return True, retry_after
    return False, 0
