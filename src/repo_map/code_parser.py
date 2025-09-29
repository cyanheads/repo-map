"""Parses code files to extract structure, docstrings, and imports."""

import ast
import logging
import re

logger = logging.getLogger(__name__)


def get_python_structure(
    file_path: str,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Extracts classes, functions, and constants from a Python file."""
    try:
        with open(file_path, encoding="utf-8") as file:
            tree = ast.parse(file.read())
    except (SyntaxError, OSError) as e:
        logger.error("Error parsing %s: %s", file_path, e)
        return {}, [], []

    classes = {}
    functions = []
    constants = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = [
                n.name for n in node.body if isinstance(n, ast.FunctionDef)
            ]
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append(target.id)

    return classes, functions, constants


def get_java_structure(
    file_path: str,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Extracts classes, methods, and constants from a Java file."""
    classes: dict[str, list[str]] = {}
    functions = []
    constants = []
    class_pattern = re.compile(r"class\s+(\w+)")
    method_pattern = re.compile(r'(public|protected|private)\s+\w+\s+(\w+)\s*\(')
    constant_pattern = re.compile(r'public\s+static\s+final\s+\w+\s+(\w+)\s*=')

    current_class = None
    try:
        with open(file_path, encoding="utf-8") as file:
            for line in file:
                class_match = class_pattern.search(line)
                if class_match:
                    current_class = class_match.group(1)
                    classes[current_class] = []
                    continue
                method_match = method_pattern.search(line)
                if method_match and current_class:
                    classes[current_class].append(method_match.group(2))
                elif method_match:
                    functions.append(method_match.group(2))
                constant_match = constant_pattern.search(line)
                if constant_match:
                    constants.append(constant_match.group(1))
    except OSError as e:
        logger.error("Error reading Java file %s: %s", file_path, e)

    return classes, functions, constants


def get_javascript_structure(
    file_path: str,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Extracts classes, functions, and constants from a JavaScript file."""
    classes: dict[str, list[str]] = {}
    functions = []
    constants = []
    class_pattern = re.compile(r"class\s+(\w+)")
    method_pattern = re.compile(r'(\w+)\s*\(')
    function_pattern = re.compile(r'function\s+(\w+)\s*\(')
    constant_pattern = re.compile(r'const\s+(\w+)\s*=')

    current_class = None
    try:
        with open(file_path, encoding="utf-8") as file:
            for line in file:
                class_match = class_pattern.search(line)
                if class_match:
                    current_class = class_match.group(1)
                    classes[current_class] = []
                    continue
                method_match = method_pattern.search(line)
                if method_match and current_class:
                    classes[current_class].append(method_match.group(1))
                else:
                    func_match = function_pattern.search(line)
                    if func_match:
                        functions.append(func_match.group(1))
                constant_match = constant_pattern.search(line)
                if constant_match:
                    constants.append(constant_match.group(1))
    except OSError as e:
        logger.error("Error reading JavaScript file %s: %s", file_path, e)

    return classes, functions, constants


def get_csharp_structure(
    file_path: str,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Extracts classes, methods, and constants from a C# file."""
    classes: dict[str, list[str]] = {}
    functions = []
    constants = []
    class_pattern = re.compile(r"class\s+(\w+)")
    method_pattern = re.compile(r'(public|protected|private)\s+\w+\s+(\w+)\s*\(')
    constant_pattern = re.compile(r'public\s+const\s+\w+\s+(\w+)\s*=')

    current_class = None
    try:
        with open(file_path, encoding="utf-8") as file:
            for line in file:
                class_match = class_pattern.search(line)
                if class_match:
                    current_class = class_match.group(1)
                    classes[current_class] = []
                    continue
                method_match = method_pattern.search(line)
                if method_match and current_class:
                    classes[current_class].append(method_match.group(2))
                elif method_match:
                    functions.append(method_match.group(2))
                constant_match = constant_pattern.search(line)
                if constant_match:
                    constants.append(constant_match.group(1))
    except OSError as e:
        logger.error("Error reading C# file %s: %s", file_path, e)

    return classes, functions, constants


def get_module_docstring(file_path: str, language: str) -> str:
    """
    Extracts the module-level docstring or comments from a file.
    """
    if language == "Python":
        try:
            with open(file_path, encoding="utf-8") as file:
                tree = ast.parse(file.read())
            return ast.get_docstring(tree) or ""
        except (SyntaxError, OSError) as e:
            logger.error("Error getting docstring from %s: %s", file_path, e)
            return ""
    if language in (
        "Java",
        "JavaScript",
        "TypeScript",
        "C++",
        "C#",
        "Ruby",
        "Go",
        "PHP",
    ):
        comment_pattern = re.compile(
            r"^\s*//\s*(.*)|^\s*/\*\*\s*(.*?)\s*\*/", re.MULTILINE
        )
        try:
            with open(file_path, encoding="utf-8") as file:
                content = file.read()
            matches = comment_pattern.findall(content)
            comments = [m[0] or m[1] for m in matches if m[0] or m[1]]
            return " ".join(comments).strip()
        except OSError as e:
            logger.error("Error reading comments from %s: %s", file_path, e)
            return ""
    return ""


def get_imports(file_path: str, language: str) -> list[str]:
    """Extracts import statements from a file based on its language."""
    if language == "Python":
        try:
            with open(file_path, encoding="utf-8") as file:
                tree = ast.parse(file.read())
        except (SyntaxError, OSError) as e:
            logger.error("Error parsing imports from %s: %s", file_path, e)
            return []

        imports: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend(f"{module}.{alias.name}" for alias in node.names)
        return imports

    imports = []
    import_patterns = {
        "Java": re.compile(r"import\s+([\w\.]+);"),
        "JavaScript": re.compile(r"import\s+.*?\s+from\s+['\"]([\w./]+)['\"];"),
        "TypeScript": re.compile(r"import\s+.*?\s+from\s+['\"]([\w./]+)['\"];"),
        "C#": re.compile(r"using\s+([\w\.]+);"),
        "PHP": re.compile(r"use\s+([\w\\]+);"),
    }
    pattern = import_patterns.get(language)
    if not pattern:
        return []

    imports = []
    try:
        with open(file_path, encoding="utf-8") as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    imports.append(match.group(1))
    except OSError as e:
        logger.error("Error reading imports from %s: %s", file_path, e)
    return imports


def get_constants(file_path: str, language: str) -> list[str]:
    """Extracts constants from a file based on its language."""
    if language == "Python":
        try:
            with open(file_path, encoding="utf-8") as file:
                tree = ast.parse(file.read())
            return [
                target.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name) and target.id.isupper()
            ]
        except (SyntaxError, OSError) as e:
            logger.error("Error parsing constants from %s: %s", file_path, e)
    elif language == "Java":
        try:
            with open(file_path, encoding="utf-8") as file:
                content = file.read()
            return re.findall(r"public\s+static\s+final\s+\w+\s+(\w+)\s*=", content)
        except OSError as e:
            logger.error("Error reading constants from %s: %s", file_path, e)
    return []


def get_structure(
    file_path: str, language: str
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """
    Extracts structure from a file based on its language.
    """
    if language == "Python":
        return get_python_structure(file_path)
    if language == "Java":
        return get_java_structure(file_path)
    if language in ("JavaScript", "TypeScript"):
        return get_javascript_structure(file_path)
    if language == "C#":
        return get_csharp_structure(file_path)
    return {}, [], []
