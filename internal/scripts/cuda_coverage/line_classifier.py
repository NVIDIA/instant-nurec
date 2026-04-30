"""
Shared line classification for coverage tools.

Classifies source lines as executable vs non-executable (blank, comment,
preprocessor, brace-only, etc.) for C++/CUDA (.cpp, .cu, .cuh),
Slang (.slang), and Python (.py) files.

Used by: infer_ncu_coverage.py, postprocess_coverage.py, generate_baseline_coverage.py
"""

import re

from enum import Enum, auto
from typing import Set, Tuple


# =============================================================================
# Line Classification Types
# =============================================================================


class LineType(Enum):
    """Classification of source code lines."""

    EXECUTABLE = auto()  # Code that can be executed
    BLANK = auto()  # Empty or whitespace-only
    COMMENT = auto()  # Single-line or block comment
    PREPROCESSOR = auto()  # #include, #define, #pragma, etc.
    BRACE_ONLY = auto()  # Lines with only { or } or };
    DECLARATION = auto()  # Forward declarations, using statements
    ATTRIBUTE = auto()  # Slang attributes like [shader(...)]
    NAMESPACE = auto()  # namespace declarations
    TEMPLATE = auto()  # Template declarations (C++)
    INTERFACE = auto()  # Slang interface declarations


# =============================================================================
# Compiled Regex Patterns (module-level for reuse)
# =============================================================================

BLANK_PATTERN = re.compile(r"^\s*$")
SINGLE_COMMENT_PATTERN = re.compile(r"^\s*//")
PREPROCESSOR_PATTERN = re.compile(r"^\s*#")
BRACE_ONLY_PATTERN = re.compile(r"^\s*[{};\s]+\s*$")
NAMESPACE_PATTERN = re.compile(r"^\s*namespace\s+\w+\s*\{?\s*$")
USING_PATTERN = re.compile(r"^\s*using\s+(namespace\s+)?\w+")

# Slang-specific
SLANG_ATTRIBUTE_PATTERN = re.compile(r'^\s*\[[\w\(\)"]+\]\s*$')
SLANG_INTERFACE_PATTERN = re.compile(r"^\s*interface\s+\w+")

# Python-specific
PY_COMMENT_PATTERN = re.compile(r"^\s*#")
PY_DOCSTRING_ONLY_PATTERN = re.compile(r'^\s*("""|\'\'\')\s*$')

# File extensions by language family
CPP_CUDA_EXTENSIONS = frozenset((".cpp", ".cu", ".cuh", ".h", ".hpp", ".cc", ".c"))
SLANG_EXTENSIONS = frozenset((".slang",))
PYTHON_EXTENSIONS = frozenset((".py",))


# =============================================================================
# Core Classifier
# =============================================================================


def classify_line(line: str, in_block_comment: bool, is_slang: bool = False) -> Tuple[LineType, bool]:
    """
    Classify a single line of C++/CUDA/Slang source code.

    Args:
        line: The raw source line (with original whitespace).
        in_block_comment: Whether we are inside a block comment from a previous line.
        is_slang: Whether the file is a .slang file (enables Slang-specific patterns).

    Returns:
        (LineType, new_in_block_comment_state)
    """
    stripped = line.strip()

    # --- Block comment state machine ---
    has_start = "/*" in line
    has_end = "*/" in line

    if in_block_comment:
        if has_end:
            after_comment = line[line.find("*/") + 2 :].strip()
            if after_comment and not after_comment.startswith("//"):
                return LineType.EXECUTABLE, False
            return LineType.COMMENT, False
        return LineType.COMMENT, True

    if has_start:
        if has_end:
            before = line[: line.find("/*")].strip()
            after = line[line.find("*/") + 2 :].strip()
            if before or (after and not after.startswith("//")):
                return LineType.EXECUTABLE, False
            return LineType.COMMENT, False
        else:
            before = line[: line.find("/*")].strip()
            if before:
                return LineType.EXECUTABLE, True
            return LineType.COMMENT, True

    # --- Single-line classifications ---
    if BLANK_PATTERN.match(line):
        return LineType.BLANK, False

    if SINGLE_COMMENT_PATTERN.match(line):
        return LineType.COMMENT, False

    # Block comment continuation (lines starting with *)
    if stripped.startswith("*") and not stripped.startswith("*="):
        if stripped in ("*", "*/") or stripped.startswith("* "):
            return LineType.COMMENT, False

    if PREPROCESSOR_PATTERN.match(line):
        return LineType.PREPROCESSOR, False

    if BRACE_ONLY_PATTERN.match(line):
        return LineType.BRACE_ONLY, False

    if NAMESPACE_PATTERN.match(line):
        return LineType.NAMESPACE, False

    if USING_PATTERN.match(line):
        return LineType.DECLARATION, False

    # Slang-specific
    if is_slang:
        if SLANG_ATTRIBUTE_PATTERN.match(line):
            return LineType.ATTRIBUTE, False
        if SLANG_INTERFACE_PATTERN.match(line) and "{" not in line:
            return LineType.INTERFACE, False

    # Label (e.g., "public:" or "case 1:")
    if stripped.endswith(":") and " " not in stripped:
        return LineType.DECLARATION, False

    return LineType.EXECUTABLE, False


def classify_line_python(line: str, in_docstring: bool) -> Tuple[LineType, bool]:
    """
    Classify a single line of Python source code.

    Args:
        line: The raw source line.
        in_docstring: Whether we are inside a triple-quoted docstring.

    Returns:
        (LineType, new_in_docstring_state)
    """
    stripped = line.strip()

    # Count triple-quote occurrences to track docstring state
    dq_count = stripped.count('"""') + stripped.count("'''")

    if in_docstring:
        if dq_count % 2 == 1:
            # Odd count = docstring ends on this line
            return LineType.COMMENT, False
        return LineType.COMMENT, True

    if dq_count % 2 == 1:
        # Odd count = docstring starts (and doesn't end) on this line
        # Check if there's code before the docstring
        for marker in ('"""', "'''"):
            idx = stripped.find(marker)
            if idx > 0:
                # Code before docstring — executable
                return LineType.EXECUTABLE, True
        return LineType.COMMENT, True

    if BLANK_PATTERN.match(line):
        return LineType.BLANK, False

    if PY_COMMENT_PATTERN.match(line):
        return LineType.COMMENT, False

    return LineType.EXECUTABLE, False


# =============================================================================
# High-level Helpers
# =============================================================================


def get_executable_line_numbers(source_lines: list, file_ext: str = ".cpp") -> Set[int]:
    """
    Return the set of 1-indexed executable line numbers in the given source.

    Args:
        source_lines: List of source lines (as returned by file.readlines()).
        file_ext: File extension to select the appropriate classifier.

    Returns:
        Set of 1-indexed line numbers that are executable.
    """
    if not source_lines:
        return set()

    executable = set()

    if file_ext in PYTHON_EXTENSIONS:
        in_docstring = False
        for line_num, line in enumerate(source_lines, start=1):
            line_type, in_docstring = classify_line_python(line, in_docstring)
            if line_type == LineType.EXECUTABLE:
                executable.add(line_num)
    else:
        is_slang = file_ext in SLANG_EXTENSIONS
        in_block_comment = False
        for line_num, line in enumerate(source_lines, start=1):
            line_type, in_block_comment = classify_line(line, in_block_comment, is_slang)
            if line_type == LineType.EXECUTABLE:
                executable.add(line_num)

    return executable
