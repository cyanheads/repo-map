"""Parses code files to extract structure, docstrings, and imports."""

import ast
import logging
import re
from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)

# Languages whose module docstring is a leading comment header. Ruby is listed
# for the branch it has always taken; it uses "#" comments, so it yields "".
_COMMENT_LANGUAGES = (
    "Java",
    "JavaScript",
    "TypeScript",
    "C++",
    "C#",
    "Ruby",
    "Go",
    "PHP",
)

# Modifiers that may precede a Java or C# member's return type.
_MEMBER_MODIFIER = (
    r"(?:public|protected|private|internal|static|final|abstract|sealed|override"
    r"|virtual|synchronized|native|async|unsafe|extern|partial|readonly|new)"
)

# "modifiers [return type] name(" — the return type is absent on constructors.
_MEMBER_PATTERN = re.compile(
    rf"^\s*(?:{_MEMBER_MODIFIER}\s+)+(?:[\w.<>\[\],?\s]+\s+)?(\w+)\s*\("
)

# Modifiers that may precede a class declaration in Java, C#, or ECMAScript.
_CLASS_MODIFIER = (
    r"(?:public|protected|private|internal|static|final|abstract|sealed|partial"
    r"|export|default|declare)"
)

# A Java/TypeScript annotation or a C# attribute placed on the declaration line.
_CLASS_ANNOTATION = r"(?:@\w+(?:\([^)]*\))?|\[[^\]]*\])\s*"

# "annotations modifiers class Name" anchored to the start of a line, so the
# word "class" in prose is not read as a declaration. The lookahead keeps an
# anonymous class from being named after its clause; generic parameters fall
# outside the captured name.
_CLASS_PATTERN = re.compile(
    rf"^\s*(?:{_CLASS_ANNOTATION})*(?:{_CLASS_MODIFIER}\s+)*"
    r"class\s+(?!extends\b|implements\b)(\w+)"
)

# "modifiers name(params): Type {" — the body brace separates a declaration from
# a call site, and the optional annotation covers TypeScript return types.
_JS_METHOD_PATTERN = re.compile(
    r"^\s*(?:(?:public|private|protected|static|readonly|override|abstract|async"
    r"|get|set|\*)\s+)*([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\(.*\)\s*(?::[^{;]+)?\{"
)

# Keywords that open a braced block and would otherwise read as a method.
_JS_BLOCK_KEYWORDS = frozenset(
    {"catch", "do", "else", "for", "function", "if", "switch", "while", "with"}
)

_JS_FUNCTION_PATTERN = re.compile(r"function\s+(\w+)\s*\(")
_JS_CONSTANT_PATTERN = re.compile(r"const\s+(\w+)\s*=")

_JAVA_CONSTANT_PATTERN = re.compile(r"public\s+static\s+final\s+\w+\s+(\w+)\s*=")
_CSHARP_CONSTANT_PATTERN = re.compile(r"public\s+const\s+\w+\s+(\w+)\s*=")

# A string literal, a line comment, or a block comment. Matching left to right
# gives whichever opens first precedence, so a quote inside a comment and a
# comment marker inside a string are both inert. The captured alternative is a
# block comment the line leaves open.
_NOISE_PATTERN = re.compile(
    r"""(?:'(?:\\.|[^'\\])*'          # single-quoted string
        |"(?:\\.|[^"\\])*"            # double-quoted string
        |`(?:\\.|[^`\\])*`            # template literal
        |//.*                         # line comment
        |/\*(?:[^*]|\*(?!/))*\*/)     # block comment closed on this line
        |(/\*.*)                      # block comment left open
    """,
    re.VERBOSE,
)


def _strip_noise(line: str, in_block_comment: bool) -> tuple[str, bool]:
    """Blanks a line's strings and comments, carrying block-comment state."""
    if in_block_comment:
        end = line.find("*/")
        if end == -1:
            return "", True
        line = line[end + len("*/") :]

    opened = False

    def blank(match: re.Match[str]) -> str:
        nonlocal opened
        opened = match.group(1) is not None
        return ""

    return _NOISE_PATTERN.sub(blank, line), opened


def _iter_code_lines(lines: Iterable[str]) -> Iterator[str]:
    """Yields each line with its strings and comments blanked out."""
    in_block_comment = False
    for line in lines:
        code, in_block_comment = _strip_noise(line, in_block_comment)
        yield code


def _net_brace_change(code: str) -> int:
    """Counts the brace depth change of a line already stripped of noise."""
    return code.count("{") - code.count("}")


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
                n.name
                for n in node.body
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.Assign):
            constants.extend(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and target.id.isupper()
            )

    return classes, functions, constants


def _get_braced_structure(
    file_path: str, language: str, constant_pattern: re.Pattern[str]
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Extracts classes, methods, and constants from a Java or C# file.

    Class scope closes on the brace that ends the class body, so a member
    declared past it is reported as a top-level function.
    """
    classes: dict[str, list[str]] = {}
    functions: list[str] = []
    constants: list[str] = []

    current_class: str | None = None
    class_depth = 0
    class_body_opened = False
    depth = 0
    try:
        with open(file_path, encoding="utf-8") as file:
            for code in _iter_code_lines(file):
                class_match = _CLASS_PATTERN.search(code)
                if class_match:
                    current_class = class_match.group(1)
                    classes[current_class] = []
                    class_depth = depth
                    depth += _net_brace_change(code)
                    class_body_opened = depth > class_depth
                    continue

                method_match = _MEMBER_PATTERN.search(code)
                if method_match and current_class:
                    classes[current_class].append(method_match.group(1))
                elif method_match:
                    functions.append(method_match.group(1))
                constant_match = constant_pattern.search(code)
                if constant_match:
                    constants.append(constant_match.group(1))

                depth += _net_brace_change(code)
                if current_class:
                    if depth > class_depth:
                        class_body_opened = True
                    elif class_body_opened:
                        current_class = None
    except OSError as e:
        logger.error("Error reading %s file %s: %s", language, file_path, e)

    return classes, functions, constants


def get_java_structure(
    file_path: str,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Extracts classes, methods, and constants from a Java file."""
    return _get_braced_structure(file_path, "Java", _JAVA_CONSTANT_PATTERN)


def get_csharp_structure(
    file_path: str,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Extracts classes, methods, and constants from a C# file."""
    return _get_braced_structure(file_path, "C#", _CSHARP_CONSTANT_PATTERN)


def get_javascript_structure(
    file_path: str,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Extracts classes, functions, and constants from a JavaScript file."""
    classes: dict[str, list[str]] = {}
    functions: list[str] = []
    constants: list[str] = []

    current_class: str | None = None
    class_depth = 0
    class_body_opened = False
    depth = 0
    try:
        with open(file_path, encoding="utf-8") as file:
            for code in _iter_code_lines(file):
                class_match = _CLASS_PATTERN.search(code)
                if class_match:
                    current_class = class_match.group(1)
                    classes[current_class] = []
                    class_depth = depth
                    depth += _net_brace_change(code)
                    class_body_opened = depth > class_depth
                    continue

                method_match = _JS_METHOD_PATTERN.search(code)
                if (
                    current_class
                    and method_match
                    and method_match.group(1) not in _JS_BLOCK_KEYWORDS
                ):
                    classes[current_class].append(method_match.group(1))
                else:
                    func_match = _JS_FUNCTION_PATTERN.search(code)
                    if func_match:
                        functions.append(func_match.group(1))
                constant_match = _JS_CONSTANT_PATTERN.search(code)
                if constant_match:
                    constants.append(constant_match.group(1))

                depth += _net_brace_change(code)
                if current_class:
                    if depth > class_depth:
                        class_body_opened = True
                    elif class_body_opened:
                        current_class = None
    except OSError as e:
        logger.error("Error reading JavaScript file %s: %s", file_path, e)

    return classes, functions, constants


def _extract_block_comment(lines: list[str]) -> str:
    """Reads a leading ``/** … */`` block, stripping its gutter characters."""
    body: list[str] = []
    for offset, line in enumerate(lines):
        text = line.strip()
        if offset == 0:
            text = text[len("/**") :]
        end = text.find("*/")
        if end != -1:
            text = text[:end]
        text = text.lstrip("*").strip()
        if text:
            body.append(text)
        if end != -1:
            break
    return " ".join(body)


def _extract_header_comment(content: str) -> str:
    """Extracts a file's leading comment header, stopping where it ends."""
    lines = content.splitlines()
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines):
        return ""

    header = lines[start].strip()
    if header.startswith("/**"):
        return _extract_block_comment(lines[start:])
    if not header.startswith("//"):
        return ""

    body: list[str] = []
    for line in lines[start:]:
        text = line.strip()
        if not text.startswith("//"):
            break
        text = text.removeprefix("//").strip()
        if text:
            body.append(text)
    return " ".join(body)


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
    if language in _COMMENT_LANGUAGES:
        try:
            with open(file_path, encoding="utf-8") as file:
                return _extract_header_comment(file.read())
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

    # Module specifiers reach the quoted string directly on a side-effect import
    # ("import './polyfills'") and through a "from" clause otherwise, including
    # on an "export … from" re-export. The trailing semicolon is optional.
    module_specifier = re.compile(
        r"^\s*(?:import|export)\s+(?:[^'\"]*?\s+from\s+)?['\"]([^'\"]+)['\"]"
    )
    import_patterns = {
        "Java": re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+\*?)\s*;?"),
        "JavaScript": module_specifier,
        "TypeScript": module_specifier,
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
