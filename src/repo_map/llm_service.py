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

# Add a semaphore to control the rate of API calls
api_semaphore = asyncio.Semaphore(settings.api_semaphore_limit)


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
    messages = [
        {
            "role": "system",
            "content": """You are an expert software documentation assistant specializing in generating precise, informative descriptions for code structures. Your task is to create concise yet comprehensive descriptions for files in various programming languages.

Key guidelines:
NOTE: This is designed to be an informative guide for a software engineer developer to better understand the codebase.

1. Provide a description between 5-15 words for each file.
2. Capture the core functionality, purpose, or key features of each file.
3. Use clear, technical language appropriate for experienced developers.
4. Highlight unique aspects or important roles of each file within the larger system.
6. Provide a single 'Developer Consideration' that highlights an unconventional, unusual, or potentially confusing aspect of the file. This consideration should focus on the file as a whole and not individual functions or classes, but it can encompass multiple aspects of the file's design or implementation. The goal is to help developers understand and work effectively with the file. If you identify any potential pitfalls, complexities, or challenges in the file, please mention them here. If you identify a crtitical issue or error in the file, please describe it here.
""",
        }
    ]

    # Construct prompt for the current file
    prompt = "Here is the current repository map:\n\n"
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
                prompt += f"{indent}│   ├── Imports: {itm['imports']}\n"
            if "functions" in itm and itm["functions"]:
                prompt += f"{indent}│   ├── Functions: {itm['functions']}\n"

    prompt += "\nNow, here is the new file to describe:\n\n"
    language = file.get("language", "None")
    prompt += f"File: {file['name']} ({language})\n"
    if "description" in file and file["description"]:
        prompt += f"Module Description: {file['description']}\n"
    if "imports" in file and file["imports"]:
        prompt += f"Imports: {file['imports']}\n"
    if "functions" in file and file["functions"]:
        prompt += f"Functions: {file['functions']}\n"

    prompt += "\nGenerate a concise description (5-15 words) for the file and provide a single 'Developer Consideration' focusing on the entire file. Follow this format:\n"
    prompt += """
Example:
├── .gitignore (Git)
│   └── Developer Consideration: "Uses complex regex patterns for selective ignores, which may lead to unexpected file inclusions/exclusions."
├── README.md (Markdown)
│   └── Developer Consideration: "Contains executable code snippets that auto-generate parts of the documentation, requiring careful management of code and doc synchronization."
├── __init__.py (Python)
│   └── Developer Consideration: "Implements dynamic importing that can make dependency tracking challenging. Pay attention to potential circular imports."
├── assistant_cli.py (Python)
│   └── Description: Orchestrates CLI operations, manages user interactions, and ensures robust application flow.
│   └── Developer Consideration: "Uses a custom event loop implementation that diverges from standard async patterns, potentially complicating integration with async libraries."
"""

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
    Extracts only the file-level Description and Developer Consideration.
    """
    file_desc_pattern = r"Description:\s*(.*)"
    considerations_pattern = r'Developer Consideration:\s*"(.*?)"'

    # Extract Description
    file_desc_match = re.search(file_desc_pattern, content)
    if file_desc_match:
        file["description"] = file_desc_match.group(1).strip()

    # Extract Developer Consideration for File
    considerations_match = re.search(considerations_pattern, content)
    if considerations_match:
        file["developer_consideration"] = considerations_match.group(1).strip()


async def rate_limited_api_call(
    messages: list[dict[str, str]], model: str, temperature: float
) -> Any:
    """
    Performs a rate-limited API call to the OpenRouter LLM using aiohttp.
    Utilizes certifi's CA bundle for SSL verification.
    """
    # Create an SSL context using certifi's CA bundle
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    async with api_semaphore:
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
