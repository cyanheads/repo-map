"""
Main script for the repo-map tool.

This script provides a CLI for generating a structured summary of a software
repository, enhanced with AI-generated descriptions.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

from tqdm import tqdm

from repo_map.cache_manager import load_cache
from repo_map.config import settings
from repo_map.file_scanner import summarize_repo
from repo_map.llm_service import get_llm_descriptions

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def enhance_repo_with_llm(
    structure: list[dict[str, Any]], cache_conn, model_name: str
) -> None:
    """
    Enhances the repository structure with descriptions using LLM.
    Utilizes cache to skip unchanged files and updates cache accordingly.
    Only processes file-level descriptions and developer considerations.
    Directories are not processed by the LLM.
    """
    cursor = cache_conn.cursor()
    tasks = []
    for index, item in enumerate(structure):
        if item["type"] == "file" and (item.get("imports") or item.get("functions")):
            # Check if the file needs processing
            cursor.execute("SELECT hash FROM cache WHERE path = ?", (item["path"],))
            row = cursor.fetchone()
            if not row or row[0] != item.get("hash", ""):
                tasks.append((index, item))

    for index, file in tqdm(tasks, desc="Enhancing files", ncols=100):
        tqdm.write(f"Processing: {file['name']}")
        await get_llm_descriptions(structure, index, file, model=model_name)
        # Update cache with new descriptions
        cursor.execute(
            """
            INSERT OR REPLACE INTO cache (
                path, hash, description, developer_consideration, imports, functions
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                file["path"],
                file.get("hash", ""),
                file.get("description", ""),
                file.get("developer_consideration", ""),
                json.dumps(file.get("imports", [])),
                json.dumps(file.get("functions", [])),
            ),
        )
        cache_conn.commit()

    logger.info("\nUpdated Repository Map:")
    print_tree(structure)
    logger.info("\n%s\n", "=" * 80)


def print_tree(structure: list[dict[str, Any]]):
    """
    Prints the repository structure in a tree format.
    Displays 'Developer Consideration' only at the file level.
    Includes directories with appropriate titles.
    """

    def print_item(item: dict[str, Any], prefix: str, is_last: bool):
        """Prints a single item in the tree."""
        connector = "└── " if is_last else "├── "
        if item["type"] == "directory":
            logger.info("%s%s%s/", prefix, connector, item["name"])
            return  # No further details for directories

        language = item.get("language", "None")
        logger.info("%s%s%s (%s)", prefix, connector, item["name"], language)

        new_prefix = prefix + ("    " if is_last else "│   ")

        if "description" in item and item["description"]:
            logger.info("%s├── Description: %s", new_prefix, item["description"])

        if "developer_consideration" in item and item["developer_consideration"]:
            logger.info(
                '%s├── Developer Consideration: "%s"',
                new_prefix,
                item["developer_consideration"],
            )
        if "imports" in item and item["imports"]:
            logger.info("%s├── Imports: %s", new_prefix, item["imports"])

        if "functions" in item and item["functions"]:
            logger.info("%s├── Functions: %s", new_prefix, item["functions"])

    logger.info("/ (Root Directory)")
    for i, item in enumerate(structure):
        prefix = "│   " * item["level"]
        is_last = i == len(structure) - 1
        print_item(item, prefix, is_last)
    logger.info("└────────────── ")


def save_tree_map(structure: list[dict[str, Any]], repo_root: str, output_path: str):
    """
    Saves the repository map to a Markdown file.
    Includes 'Developer Consideration' only at the file level.
    Includes directories with appropriate titles.
    """

    def write_item(item: dict[str, Any], prefix: str, is_last: bool, file_handle):
        """Writes a single item to the markdown file."""
        connector = "└── " if is_last else "├── "
        if item["type"] == "directory":
            file_handle.write(f"{prefix}{connector}{item['name']}/\n")
            return  # No further details for directories

        language = item.get("language", "None")
        file_handle.write(f"{prefix}{connector}{item['name']} ({language})\n")

        new_prefix = prefix + ("    " if is_last else "│   ")

        if "description" in item and item["description"]:
            file_handle.write(f"{new_prefix}├── Description: {item['description']}\n")

        if "developer_consideration" in item and item["developer_consideration"]:
            file_handle.write(
                f"{new_prefix}├── Developer Consideration: \"{item['developer_consideration']}\"\n"
            )
        if "imports" in item and item["imports"]:
            file_handle.write(f"{new_prefix}├── Imports: {item['imports']}\n")

        if "functions" in item and item["functions"]:
            file_handle.write(f"{new_prefix}├── Functions: {item['functions']}\n")

    repo_name = os.path.basename(os.path.normpath(repo_root))
    try:
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write("# Repository Map\n\n")
            file_handle.write("```markdown\n")
            file_handle.write(f"/ ({repo_name})\n")
            for i, item in enumerate(structure):
                prefix = "│   " * item["level"]
                is_last = i == len(structure) - 1
                write_item(item, prefix, is_last, file_handle)
            file_handle.write("└────────────── \n")
            file_handle.write("```\n")
        logger.info("Repository map saved to '%s'.", output_path)
    except OSError as e:
        logger.error("Error saving repository map: %s", e)


def save_pre_enhanced_map(
    structure: list[dict[str, Any]],
    output_path: str = ".repo_map_structure.json",
):
    """Saves the pre-enhancement repository summary to a JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=4)
        logger.info("repo-map structure saved to '%s'.", output_path)
    except OSError as e:
        logger.error("Error saving .repo_map_structure.json: %s", e)


def confirm_disclaimer() -> bool:
    """
    Prompts the user to acknowledge the disclaimer before proceeding.
    Returns True if the user agrees, False otherwise.
    """
    disclaimer_message = (
        "repo-map: A tool to generate a structured summary of a software repository, enhanced with AI.\n"
        "This tool uses the .gitignore in the target directory for files to not include in the repo map.\n"
        "DISCLAIMER: By using this script, you acknowledge that the files will be sent to the OpenRouter LLM for processing.\n"
        "Do you want to proceed? [y/n]: "
    )
    while True:
        user_input = input(disclaimer_message).strip().lower()
        if user_input in ("y", "yes", ""):
            return True
        if user_input in ("n", "no"):
            return False
        print("Invalid input. Please enter 'y' or 'n'.")


async def main():
    """Main function to run the repo-map script."""
    if not settings.openrouter_api_key:
        logger.error("Error: OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description=(
            "repo-map: Generates a structured summary of a software repository, enhanced with AI.\n"
            "Note: Primarily tested with Python. Other languages are parsed but may have varying results."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "repository_path", type=str, help="Path to the repository to be summarized."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically accept the disclaimer and proceed without prompting.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenRouter LLM model name. Overrides the .env file.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=settings.api_semaphore_limit,
        help="Number of concurrent API calls to the LLM.",
    )
    args = parser.parse_args()

    # Update settings from CLI arguments
    settings.api_semaphore_limit = args.concurrency
    model_name = args.model or settings.openrouter_model_name

    repo_path = args.repository_path
    if not os.path.isdir(repo_path):
        logger.error("Error: %s is not a valid directory", repo_path)
        sys.exit(1)

    if not args.yes and not confirm_disclaimer():
        logger.warning("Operation cancelled by the user.")
        sys.exit(0)

    cache_conn = load_cache(repo_path)
    logger.info("Generating repository summary...")
    summary = summarize_repo(repo_path, cache_conn)

    save_pre_enhanced_map(summary, os.path.join(repo_path, ".repo_map_structure.json"))

    logger.info(
        "Enhancing repository summary with descriptions using OpenRouter LLM..."
    )
    await enhance_repo_with_llm(summary, cache_conn, model_name=model_name)

    cache_conn.close()

    directory_name = os.path.basename(os.path.normpath(repo_path))
    output_file_name = f"{directory_name}_repo_map.md"
    output_path = os.path.join(repo_path, output_file_name)
    save_tree_map(summary, repo_path, output_path)
    logger.info("Your repo-map has been saved to '%s'.", output_file_name)


def run_main():
    """Runs the main async function."""
    asyncio.run(main())
