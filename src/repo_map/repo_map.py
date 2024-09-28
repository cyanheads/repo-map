# File: repo_map.py
# Description: Generate a comprehensive summary and visualization of a repository structure, enhanced with AI-powered descriptions and developer considerations.
# Uploading to PyPi for easy installation and usage.
# pip install repo-map

import os
import sys
import ast
import re
import ssl
import certifi
import json
import hashlib
import sqlite3
import requests
import logging
import argparse
from typing import Dict, List, Tuple, Any
import asyncio
import aiohttp
import pathspec
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
YOUR_SITE_URL = "https://github.com/cyanheads/repo-map"  # Replace with your actual site URL
YOUR_APP_NAME = "repo-map"  # Updated app name

# Add a semaphore to control the rate of API calls
api_semaphore = asyncio.Semaphore(3)

# Supported file extensions and their corresponding programming languages
SUPPORTED_LANGUAGES = {
    '.py': 'Python',
    '.java': 'Java',
    '.js': 'JavaScript',
    '.jsx': 'JavaScript',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript',
    '.cpp': 'C++',
    '.hpp': 'C++',
    '.h': 'C++',
    '.cs': 'C#',
    '.rb': 'Ruby',
    '.go': 'Go',
    '.php': 'PHP',
    '.txt': 'Text',
    '.md': 'Markdown',
    '.sh': 'Shell',
    '.yml': 'YAML',
    '.yaml': 'YAML',
    '.json': 'JSON',
    '.html': 'HTML',
    '.css': 'CSS',
    '.scss': 'SCSS',
    '.less': 'LESS',
    '.sql': 'SQL',
    '.r': 'R',
    '.kt': 'Kotlin',
    '.swift': 'Swift',
    '.pl': 'Perl',
    '.asm': 'Assembly',
    '.clj': 'Clojure',
    '.groovy': 'Groovy',
    '.lua': 'Lua',
    '.pas': 'Pascal',
    '.scala': 'Scala',
    '.tsv': 'TSV',
    '.csv': 'CSV',
    '.xml': 'XML',
    '.ini': 'INI',
    '.cfg': 'Config',
    '.conf': 'Config',
    '.env': 'Config',
    '.envrc': 'Config',
    '.tf': 'Terraform',
    '.tfvars': 'Terraform',
    '.tfstate': 'Terraform',
    '.tfstate.backup': 'Terraform',
    '.hcl': 'Terraform',
    '.dockerfile': 'Docker',
    '.tfignore': 'Terraform',
    '.gitignore': 'Git',
    '.gitattributes': 'Git',
    '.db': 'Database',
    '.sqlite': 'Database',
    '.db3': 'Database',
    '.dbf': 'Database',
    '.dbx': 'Database',
    '.mdb': 'Database',
    '.accdb': 'Database',
    '.frm': 'Database',
    '.sqlitedb': 'Database',
    '.png': 'Image',
    '.jpg': 'Image',
    '.jpeg': 'Image',
    '.gif': 'Image',
    '.svg': 'Image',
    '.bmp': 'Image',
    '.ico': 'Image',
    '.tif': 'Image',
    '.tiff': 'Image',
    '.webp': 'Image',
    '.heic': 'Image',
    '.heif': 'Image',
    '.pdf': 'PDF',
    '.doc': 'Document',
    '.docx': 'Document',
    '.ppt': 'PowerPointPresentation',
    '.wav': 'Audio',
    '.mp3': 'Audio',
    '.mp4': 'Video',
    '.mov': 'Video',
    '.avi': 'Video',
    '.mkv': 'Video',
    '.webm': 'Video',
    '.flv': 'Video',
    '.wmv': 'Video',
    '.m4a': 'Audio',
    '.flac': 'Audio',
    '.ogg': 'Audio',
    '.opus': 'Audio',
    '.wma': 'Audio',
    '.aac': 'Audio',
    '.aiff': 'Audio',
    '.ape': 'Audio',
    '.alac': 'Audio',
    # Add more languages and extensions as needed
}

def parse_gitignore(root_dir: str) -> List[str]:
    """
    Parses all .gitignore files in the repository to extract ignore patterns.
    This considers .gitignore files in nested directories as well.
    """
    ignore_patterns = []
    for dirpath, _, filenames in os.walk(root_dir):
        if '.gitignore' in filenames:
            gitignore_path = os.path.join(dirpath, '.gitignore')
            try:
                with open(gitignore_path, 'r') as f:
                    patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    # Prepend the relative path to patterns to handle nested .gitignore
                    rel_path = os.path.relpath(dirpath, root_dir)
                    if rel_path != '.':
                        patterns = [os.path.join(rel_path, pattern) for pattern in patterns]
                    ignore_patterns.extend(patterns)
            except IOError as e:
                logger.error(f"Error reading .gitignore file at {gitignore_path}: {e}")
    return ignore_patterns

def should_ignore(path: str, ignore_spec: pathspec.PathSpec) -> bool:
    """
    Determines if a given path should be ignored based on the PathSpec.

    Args:
        path (str): The file or directory path to check.
        ignore_spec (pathspec.PathSpec): The compiled PathSpec object containing ignore patterns.

    Returns:
        bool: True if the path should be ignored, False otherwise.
    """
    return ignore_spec.match_file(path)

def compute_file_hash(file_path: str) -> str:
    """
    Computes the SHA-256 hash of the given file.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except IOError as e:
        logger.error(f"Error reading file {file_path} for hashing: {e}")
        return ""

def get_python_structure(file_path: str) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Extracts classes, functions, and constants from a Python file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read())
    except (SyntaxError, IOError) as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return {}, [], []
    
    classes = {}
    functions = []
    constants = []
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append(target.id)
    
    return classes, functions, constants

def get_java_structure(file_path: str) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Extracts classes, methods, and constants from a Java file.
    """
    classes = {}
    functions = []
    constants = []
    class_pattern = re.compile(r'class\s+(\w+)')
    method_pattern = re.compile(r'(public|protected|private)\s+\w+\s+(\w+)\s*\(')
    constant_pattern = re.compile(r'public\s+static\s+final\s+\w+\s+(\w+)\s*=')
    
    current_class = None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                class_match = class_pattern.search(line)
                if class_match:
                    current_class = class_match.group(1)
                    classes[current_class] = []
                    continue
                method_match = method_pattern.search(line)
                if method_match and current_class:
                    method_name = method_match.group(2)
                    classes[current_class].append(method_name)
                elif method_match and not current_class:
                    functions.append(method_match.group(2))
                constant_match = constant_pattern.search(line)
                if constant_match:
                    constants.append(constant_match.group(1))
    except IOError as e:
        logger.error(f"Error reading Java file {file_path}: {e}")
    
    return classes, functions, constants

def get_javascript_structure(file_path: str) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Extracts classes, functions, and constants from a JavaScript file.
    """
    classes = {}
    functions = []
    constants = []
    class_pattern = re.compile(r'class\s+(\w+)')
    method_pattern = re.compile(r'(\w+)\s*\(')
    function_pattern = re.compile(r'function\s+(\w+)\s*\(')
    constant_pattern = re.compile(r'const\s+(\w+)\s*=')
    
    current_class = None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                class_match = class_pattern.search(line)
                if class_match:
                    current_class = class_match.group(1)
                    classes[current_class] = []
                    continue
                method_match = method_pattern.search(line)
                if method_match and current_class:
                    method_name = method_match.group(1)
                    classes[current_class].append(method_name)
                else:
                    func_match = function_pattern.search(line)
                    if func_match:
                        functions.append(func_match.group(1))
                constant_match = constant_pattern.search(line)
                if constant_match:
                    constants.append(constant_match.group(1))
    except IOError as e:
        logger.error(f"Error reading JavaScript file {file_path}: {e}")
    
    return classes, functions, constants

def get_csharp_structure(file_path: str) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Extracts classes, methods, and constants from a C# file.
    """
    classes = {}
    functions = []
    constants = []
    class_pattern = re.compile(r'class\s+(\w+)')
    method_pattern = re.compile(r'(public|protected|private)\s+\w+\s+(\w+)\s*\(')
    constant_pattern = re.compile(r'public\s+const\s+\w+\s+(\w+)\s*=')
    
    current_class = None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                class_match = class_pattern.search(line)
                if class_match:
                    current_class = class_match.group(1)
                    classes[current_class] = []
                    continue
                method_match = method_pattern.search(line)
                if method_match and current_class:
                    method_name = method_match.group(2)
                    classes[current_class].append(method_name)
                elif method_match and not current_class:
                    functions.append(method_match.group(2))
                constant_match = constant_pattern.search(line)
                if constant_match:
                    constants.append(constant_match.group(1))
    except IOError as e:
        logger.error(f"Error reading C# file {file_path}: {e}")
    
    return classes, functions, constants

def get_module_docstring(file_path: str, language: str) -> str:
    """
    Extracts the module-level docstring or comments from a file based on its language.
    """
    if language == 'Python':
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read())
                docstring = ast.get_docstring(tree)
                return docstring if docstring else ""
        except (SyntaxError, IOError) as e:
            logger.error(f"Error getting docstring from {file_path}: {e}")
            return ""
    elif language in ['Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Ruby', 'Go', 'PHP']:
        # Attempt to extract top-level comments
        comment_pattern = re.compile(r'^\s*//\s*(.*)|^\s*/\*\*\s*(.*?)\s*\*/', re.MULTILINE)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                matches = comment_pattern.findall(content)
                comments = [m[0] or m[1] for m in matches if m[0] or m[1]]
                return ' '.join(comments).strip()
        except IOError as e:
            logger.error(f"Error reading comments from {file_path}: {e}")
            return ""
    else:
        return ""

def get_imports(file_path: str, language: str) -> List[str]:
    """
    Extracts import statements from a file based on its language.
    """
    imports = []
    if language == 'Python':
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read())
        except (SyntaxError, IOError) as e:
            logger.error(f"Error parsing imports from {file_path}: {e}")
            return []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
    elif language in ['Java', 'JavaScript', 'TypeScript', 'C#', 'PHP']:
        # Simple regex-based import extraction
        import_patterns = {
            'Java': re.compile(r'import\s+([\w\.]+);'),
            'JavaScript': re.compile(r'import\s+.*?\s+from\s+[\'"]([\w\.\/]+)[\'"];'),
            'TypeScript': re.compile(r'import\s+.*?\s+from\s+[\'"]([\w\.\/]+)[\'"];'),
            'C#': re.compile(r'using\s+([\w\.]+);'),
            'PHP': re.compile(r'use\s+([\w\\]+);'),
        }
        pattern = import_patterns.get(language)
        if pattern:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        match = pattern.search(line)
                        if match:
                            imports.append(match.group(1))
            except IOError as e:
                logger.error(f"Error reading imports from {file_path}: {e}")
    # Add more languages and their import extraction as needed
    return imports

def get_constants(file_path: str, language: str) -> List[str]:
    """
    Extracts constants from a file based on its language.
    """
    constants = []
    if language == 'Python':
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id.isupper():
                                constants.append(target.id)
        except (SyntaxError, IOError) as e:
            logger.error(f"Error parsing constants from {file_path}: {e}")
    elif language == 'Java':
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                constants = re.findall(r'public\s+static\s+final\s+\w+\s+(\w+)\s*=', content)
        except IOError as e:
            logger.error(f"Error reading constants from {file_path}: {e}")
    # Add more languages and their constant extraction as needed
    return constants

def get_structure(file_path: str, language: str) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Extracts classes, functions/methods, and constants from a file based on its language.
    """
    if language == 'Python':
        return get_python_structure(file_path)
    elif language == 'Java':
        return get_java_structure(file_path)
    elif language in ['JavaScript', 'TypeScript']:
        return get_javascript_structure(file_path)
    elif language == 'C#':
        return get_csharp_structure(file_path)
    # Add more languages and their structure extraction as needed
    else:
        return {}, [], []

def summarize_repo(root_dir: str, cache_conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Summarizes the repository by walking through directories and files.
    Includes directories in the summary with appropriate levels.
    Utilizes cache to skip processing unchanged files.
    """
    summary = []
    ignore_patterns = parse_gitignore(root_dir)
    
    # Add a manual ignore list for generated files
    manual_ignore_patterns = ['.repo_map_structure.json', '.repo-map-cache.db']
    additional_patterns = ['*.pkl'] + manual_ignore_patterns
    combined_patterns = ignore_patterns + additional_patterns
    ignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', combined_patterns)
    
    cursor = cache_conn.cursor()
    
    for root, dirs, files in os.walk(root_dir):
        relative_root = os.path.relpath(root, root_dir)
        if relative_root == '.':
            relative_root = ''
        
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(relative_root, d), ignore_spec)]
        
        # Add the current directory to the summary
        if relative_root != '':
            dir_info = {
                'name': os.path.basename(root),
                'path': root,
                'level': relative_root.count(os.sep),
                'type': 'directory',
                'language': None
            }
            summary.append(dir_info)

        for file in sorted(files):
            full_path = os.path.join(root, file)
            relative_file_path = os.path.relpath(full_path, root_dir)
            if should_ignore(relative_file_path, ignore_spec):
                continue

            _, ext = os.path.splitext(file)
            language = SUPPORTED_LANGUAGES.get(ext.lower())
            file_info = {
                'name': file,
                'path': full_path,
                'level': relative_file_path.count(os.sep),
                'type': 'file',
                'language': language
            }

            if language:
                file_hash = compute_file_hash(full_path)
                cursor.execute("SELECT hash, description, developer_consideration, imports, functions FROM cache WHERE path = ?", (full_path,))
                row = cursor.fetchone()

                if row and row[0] == file_hash:
                    # Use cached descriptions
                    file_info.update({
                        'description': row[1],
                        'developer_consideration': row[2],
                        'imports': json.loads(row[3]) if row[3] else [],
                        'functions': json.loads(row[4]) if row[4] else [],
                        'hash': file_hash
                    })
                else:
                    # Need to process this file
                    classes, functions_extracted, constants = get_structure(full_path, language)
                    module_doc = get_module_docstring(full_path, language)
                    imports = get_imports(full_path, language)
                    file_info.update({
                        'classes': classes,
                        'functions': functions_extracted,
                        'constants': constants,
                        'imports': imports,
                        'description': module_doc,
                        'hash': file_hash
                    })

            summary.append(file_info)

    return summary

async def get_llm_descriptions(structure: List[Dict[str, Any]], file_index: int, file: Dict[str, Any], model: str, max_retries: int = 3) -> None:
    """
    Sends prompts to the LLM to generate descriptions for files.
    Updates the structure in-place with the received descriptions.
    Implements exponential backoff for retries.
    
    Args:
        structure (List[Dict[str, Any]]): The repository structure summary.
        file_index (int): The index of the current file in the structure list.
        file (Dict[str, Any]): The current file dictionary to be updated.
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
"""
        }
    ]

    # Construct prompt for the current file
    prompt = "Here is the current repository map:\n\n"
    partial_map = structure[:file_index + 1]

    for itm in partial_map:
        indent = '│   ' * itm['level']
        if itm['type'] == 'directory':
            prompt += f"{indent}├── {itm['name']}/\n"
        elif itm['type'] == 'file':
            language = itm.get('language', 'None')
            prompt += f"{indent}├── {itm['name']} ({language})\n"
            if 'description' in itm and itm['description']:
                prompt += f"{indent}│   └── Description: {itm['description']}\n"
            if 'developer_consideration' in itm and itm['developer_consideration']:
                prompt += f"{indent}│   └── Developer Consideration: \"{itm['developer_consideration']}\"\n"
            if 'imports' in itm and itm['imports']:
                prompt += f"{indent}│   ├── Imports: {itm['imports']}\n"
            if 'functions' in itm and itm['functions']:
                prompt += f"{indent}│   ├── Functions: {itm['functions']}\n"

    prompt += "\nNow, here is the new file to describe:\n\n"
    language = file.get('language', 'None')
    prompt += f"File: {file['name']} ({language})\n"
    if 'description' in file and file['description']:
        prompt += f"Module Description: {file['description']}\n"
    if 'imports' in file and file['imports']:
        prompt += f"Imports: {file['imports']}\n"
    if 'functions' in file and file['functions']:
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

    messages.append({
        "role": "user",
        "content": prompt
    })

    retries = 0
    while retries < max_retries:
        try:
            response = await rate_limited_api_call(messages, model, 0.0)  
            # Models: qwen/qwen-2.5-72b-instruct, anthropic/claude-3.5-sonnet, openai/gpt-4o
            # Any OpenRouter model can be used here
        except requests.exceptions.RequestException as e:
            retry_after = 5 * (2 ** retries)  # Exponential backoff
            logger.warning(f"Error communicating with OpenRouter LLM: {e}. Retrying after {retry_after} seconds...")
            await asyncio.sleep(retry_after)
            retries += 1
            continue

        if 'error' in response:
            if response['error'].get('code') == 429:
                retry_after = response['error'].get('retry_after', 5) * (2 ** retries)  # Exponential backoff
                logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                await asyncio.sleep(retry_after)
                retries += 1
                continue
            else:
                logger.error(f"Error from OpenRouter LLM: {response['error']}")
                return

        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content'].strip()
            parse_llm_response(content, file)
            return
        else:
            logger.error("Unexpected response structure from LLM.")
            return

    logger.error(f"Failed to get descriptions for {file['name']} after {max_retries} retries.")

def parse_llm_response(content: str, file: Dict[str, Any]) -> None:
    """
    Parses the LLM response content and updates the file dictionary with descriptions.
    Extracts only the file-level Description and Developer Consideration.
    """
    file_desc_pattern = r"Description:\s*(.*)"
    considerations_pattern = r"Developer Consideration:\s*\"(.*?)\""

    # Extract Description
    file_desc_match = re.search(file_desc_pattern, content)
    if file_desc_match:
        file['description'] = file_desc_match.group(1).strip()

    # Extract Developer Consideration for File
    considerations_match = re.search(considerations_pattern, content)
    if considerations_match:
        file['developer_consideration'] = considerations_match.group(1).strip()

async def rate_limited_api_call(messages: List[Dict[str, str]], model: str, temperature: float) -> Any:
    """
    Performs a rate-limited API call to the OpenRouter LLM using aiohttp.
    Utilizes certifi's CA bundle for SSL verification.
    """
    # Create an SSL context using certifi's CA bundle
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    async with api_semaphore:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": YOUR_SITE_URL,
            "X-Title": YOUR_APP_NAME,
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=data,
                    ssl=ssl_context  # Apply the SSL context here
                ) as response:
                    if response.status != 200:
                        # Handle rate limiting
                        rate_limited, retry_after = await handle_rate_limiting_async(response)
                        if rate_limited:
                            logger.warning(f"Rate limited by API. Retrying after {retry_after} seconds...")
                            await asyncio.sleep(retry_after)
                            return await rate_limited_api_call(messages, model, temperature)
                        else:
                            response_text = await response.text()
                            logger.error(f"API request failed with status {response.status}: {response_text}")
                            response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            raise
        except ssl.SSLCertVerificationError as e:
            logger.error(f"SSL Certificate Verification Error: {e}")
            raise

def handle_rate_limiting(response) -> Tuple[bool, int]:
    """
    Checks if the response status code indicates rate limiting.
    If so, returns True and the retry-after duration.
    Otherwise, returns False and 0.
    """
    if response.status_code == 429:
        retry_after = response.headers.get('retry-after')
        if retry_after:
            try:
                retry_after = int(retry_after)
            except ValueError:
                retry_after = 5  # Default retry after 5 seconds if parsing fails
        else:
            retry_after = 5
        return True, retry_after
    return False, 0

async def handle_rate_limiting_async(response) -> Tuple[bool, int]:
    """
    Asynchronously checks if the response status code indicates rate limiting.
    If so, returns True and the retry-after duration.
    Otherwise, returns False and 0.
    """
    if response.status == 429:
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                retry_after = int(retry_after)
            except ValueError:
                retry_after = 5  # Default retry after 5 seconds if parsing fails
        else:
            retry_after = 5
        return True, retry_after
    return False, 0

async def enhance_repo_with_llm(structure: List[Dict[str, Any]], cache_conn: sqlite3.Connection, model_name: str) -> None:
    """
    Enhances the repository structure with descriptions using LLM.
    Utilizes cache to skip unchanged files and updates cache accordingly.
    Only processes file-level descriptions and developer considerations.
    Directories are not processed by the LLM.
    """
    cursor = cache_conn.cursor()
    tasks = []
    for index, item in enumerate(structure):
        if item['type'] == 'file' and (item.get('imports') or item.get('functions')):
            # Check if the file needs processing
            cursor.execute("SELECT hash FROM cache WHERE path = ?", (item['path'],))
            row = cursor.fetchone()
            if not row or row[0] != item.get('hash', ''):
                tasks.append((index, item))

    for index, file in tqdm(tasks, desc="Enhancing files", ncols=100):
        tqdm.write(f"Processing: {file['name']}")
        await get_llm_descriptions(structure, index, file, model=model_name)
        # Update cache with new descriptions
        cursor.execute("""
            INSERT OR REPLACE INTO cache (
                path, hash, description, developer_consideration, imports, functions
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            file['path'],
            file.get('hash', ''),
            file.get('description', ''),
            file.get('developer_consideration', ''),
            json.dumps(file.get('imports', [])),
            json.dumps(file.get('functions', []))
        ))
        cache_conn.commit()

    logger.warning("\nUpdated Repository Map:")
    print_tree(structure)
    logger.warning("\n" + "="*80 + "\n")

def print_tree(structure: List[Dict[str, Any]]):
    """
    Prints the repository structure in a tree format.
    Displays 'Developer Consideration' only at the file level.
    Includes directories with appropriate titles.
    """
    def print_item(item: Dict[str, Any], prefix: str, is_last: bool):
        connector = '└── ' if is_last else '├── '
        if item['type'] == 'directory':
            logger.warning(f"{prefix}{connector}{item['name']}/")
            new_prefix = prefix + ('    ' if is_last else '│   ')
            return new_prefix  # Directories do not have further nested details in this representation
        else:
            language = item.get('language', 'None')
            logger.warning(f"{prefix}{connector}{item['name']} ({language})")

            new_prefix = prefix + ('    ' if is_last else '│   ')

            # Description
            if 'description' in item and item['description']:
                logger.warning(f"{new_prefix}├── Description: {item['description']}")
            
            # Developer Consideration
            if 'developer_consideration' in item and item['developer_consideration']:
                logger.warning(f"{new_prefix}├── Developer Consideration: \"{item['developer_consideration']}\"")

            # Imports
            if 'imports' in item and item['imports']:
                logger.warning(f"{new_prefix}├── Imports: {item['imports']}")

            # Functions
            if 'functions' in item and item['functions']:
                logger.warning(f"{new_prefix}├── Functions: {item['functions']}")

    logger.warning("/ (Root Directory)")
    for i, item in enumerate(structure):
        prefix = '│   ' * item['level']
        is_last = i == len(structure) - 1
        print_item(item, prefix, is_last)
    # Add bottom line
    logger.warning("└────────────── ")

def save_tree_map(structure: List[Dict[str, Any]], repo_root: str, output_path: str):
    """
    Saves the repository map to a Markdown file.
    Includes 'Developer Consideration' only at the file level.
    Includes directories with appropriate titles.
    """
    def write_item(item: Dict[str, Any], prefix: str, is_last: bool, file_handle):
        connector = '└── ' if is_last else '├── '
        if item['type'] == 'directory':
            file_handle.write(f"{prefix}{connector}{item['name']}/\n")
            return  # Directories do not have further nested details in this representation
        else:
            language = item.get('language', 'None')
            file_handle.write(f"{prefix}{connector}{item['name']} ({language})\n")

            new_prefix = prefix + ('    ' if is_last else '│   ')

            # Description
            if 'description' in item and item['description']:
                file_handle.write(f"{new_prefix}├── Description: {item['description']}\n")

            # Developer Consideration
            if 'developer_consideration' in item and item['developer_consideration']:
                file_handle.write(f"{new_prefix}├── Developer Consideration: \"{item['developer_consideration']}\"\n")

            # Imports
            if 'imports' in item and item['imports']:
                file_handle.write(f"{new_prefix}├── Imports: {item['imports']}\n")

            # Functions
            if 'functions' in item and item['functions']:
                file_handle.write(f"{new_prefix}├── Functions: {item['functions']}\n")

    repo_name = os.path.basename(os.path.normpath(repo_root))
    try:
        with open(output_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write("# Repository Map\n\n")
            file_handle.write("```markdown\n")
            file_handle.write(f"/ ({repo_name})\n")
            for i, item in enumerate(structure):
                prefix = '│   ' * item['level']
                is_last = i == len(structure) - 1
                write_item(item, prefix, is_last, file_handle)
            # Add bottom line
            file_handle.write("└────────────── \n")
            file_handle.write("```\n")
        logger.warning(f"Repository map saved to '{output_path}'.")
    except IOError as e:
        logger.error(f"Error saving repository map: {e}")

def load_cache(repo_root: str) -> sqlite3.Connection:
    """
    Loads the LLM response cache from a SQLite3 database.
    Creates the database and table if they don't exist.
    Also, adds new columns if they are missing.
    """
    cache_file_path = os.path.join(repo_root, '.repo-map-cache.db')
    conn = sqlite3.connect(cache_file_path)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            path TEXT PRIMARY KEY,
            hash TEXT,
            description TEXT,
            developer_consideration TEXT,
            imports TEXT,
            functions TEXT
        )
    """)
    
    # Add new columns if they don't exist
    cursor.execute("PRAGMA table_info(cache)")
    existing_columns = [info[1] for info in cursor.fetchall()]
    
    new_columns = {
        'developer_consideration',
    }
    
    for column in new_columns:
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE cache ADD COLUMN {column} TEXT")
    
    conn.commit()
    return conn

def save_pre_enhanced_map(structure: List[Dict[str, Any]], output_path: str = '.repo_map_structure.json'):
    """
    Saves the pre-enhancement repository summary to a JSON file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=4)
        logger.warning(f"repo-map structure saved to '{output_path}'.")
    except IOError as e:
        logger.error(f"Error saving .repo_map_structure.json: {e}")

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
        if user_input in ('y', 'yes', ''):
            return True
        elif user_input in ('n', 'no'):
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

async def main():
    """
    Main function to run the repo-map script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "repo-map: A tool to generate a structured summary of a software repository, enhanced with AI.\n"
            "Note: Python has been tested to work. repo-map can parse various languages but \n"
            "has not been extensively tested. Please submit an issue on GitHub for any issues."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'repository_path',
        type=str,
        help='Path to the repository to be summarized.'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Automatically accept the disclaimer and proceed without prompting.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='anthropic/claude-3.5-sonnet',
        help='OpenRouter LLM model name to use for generating descriptions (default: anthropic/claude-3.5-sonnet).'
    )
    args = parser.parse_args()

    repo_path = args.repository_path
    if not os.path.isdir(repo_path):
        logger.error(f"Error: {repo_path} is not a valid directory")
        sys.exit(1)

    # Disclaimer Confirmation
    if args.yes:
        logger.info(f"DISCLAIMER: By using this script, you acknowledge that the files will be \nsent to the OpenRouter LLM for processing.")
    else:
        if not confirm_disclaimer():
            logger.warning("Operation cancelled by the user.")
            sys.exit(0)

    # Load cache
    cache_conn = load_cache(repo_path)

    logger.info("Generating repository summary...")
    summary = summarize_repo(repo_path, cache_conn)
    
    # Save pre-enhanced repo map
    save_pre_enhanced_map(summary, os.path.join(repo_path, '.repo_map_structure.json'))
    
    logger.info("Enhancing repository summary with descriptions using OpenRouter LLM...")
    await enhance_repo_with_llm(summary, cache_conn, model_name=args.model)
    
    # Close the cache connection
    cache_conn.close()
    
    # Save the enhanced repository map
    directory_name = os.path.basename(os.path.normpath(repo_path))
    output_file_name = f"{directory_name}_repo_map.md"
    output_path = os.path.join(repo_path, output_file_name)
    save_tree_map(summary, repo_path, output_path)
    logger.info(f"Your repo-map has been saved to '{output_file_name}'.")

def run_main():
    asyncio.run(main())

if __name__ == "__main__":
    run_main()