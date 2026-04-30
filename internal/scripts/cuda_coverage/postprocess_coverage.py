#!/usr/bin/env python3
"""
LCOV Coverage Post-Processor for CUDA/Slang Files

Cleans, infers, and processes LCOV coverage data for .cu, .cuh, and .slang files.
Uses streaming I/O for memory efficiency and source file analysis for intelligent
line classification and coverage propagation.

Features:
- Streaming LCOV parser (memory efficient)
- Source-aware line classification (executable vs non-executable)
- Coverage inference within code blocks
- Function boundary detection
- Path canonicalization for bazel/sandbox environments

Usage:
    python postprocess_coverage.py input.dat -o output.info -s /path/to/sources
    python postprocess_coverage.py input.dat --infer-blocks --detect-functions
"""

import argparse
import os
import re
import sys

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple

from line_classifier import LineType
from line_classifier import classify_line as _classify_line_shared


@dataclass
class LineInfo:
    """Information about a single source line."""

    line_num: int
    line_type: LineType
    content: str
    in_block_comment: bool = False
    in_function: bool = False
    function_name: Optional[str] = None
    brace_depth: int = 0
    zone_id: int = 0  # Control flow zone ID
    continuation_of: Optional[int] = None  # Primary line if this is a continuation


@dataclass
class FunctionInfo:
    """Information about a function/kernel."""

    name: str
    start_line: int
    end_line: int = 0
    is_kernel: bool = False
    is_shader: bool = False
    hit_count: int = 0


@dataclass
class CodeZone:
    """A control flow zone - a contiguous block of code in the same execution path."""

    zone_id: int
    start_line: int
    end_line: int = 0
    parent_zone: Optional[int] = None  # For nested branches
    zone_type: str = "default"  # "if", "else", "case", "loop", "function", "default"
    brace_depth: int = 0


@dataclass
class FileRecord:
    """Parsed LCOV record for a single source file."""

    source_file: str
    line_coverage: Dict[int, int] = field(default_factory=dict)  # line -> hit count
    functions: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # name -> (line, hits)
    lines_found: int = 0
    lines_hit: int = 0
    functions_found: int = 0
    functions_hit: int = 0


# =============================================================================
# LRU Cache for Source Files
# =============================================================================


class LRUCache:
    """Simple LRU cache for source file contents."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.cache: OrderedDict[str, List[str]] = OrderedDict()

    def get(self, key: str) -> Optional[List[str]]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: List[str]) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove oldest
                self.cache.popitem(last=False)
            self.cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.cache


# =============================================================================
# Zone Tracker - Control Flow Zone Detection (Brace-Centric)
# =============================================================================


class ZoneTracker:
    """
    Tracks control flow zones in source code using brace-centric analysis.

    Key principles:
    1. Each `{` opens a new zone
    2. Each `}` closes the current zone
    3. Lines with `} else {` span TWO zone transitions
    4. A line belongs to the zone that is active AFTER processing all braces

    This ensures coverage never bleeds across if/else/switch boundaries.
    """

    # Branch keywords that create semantically different zones
    BRANCH_KEYWORDS = re.compile(r"\b(if|else\s+if|else|switch|case|default|for|while|do|try|catch)\b")

    # Patterns to detect branch type from line content before a brace
    IF_CONSTEXPR_PATTERN = re.compile(r"\bif\s+constexpr\s*\(")
    IF_PATTERN = re.compile(r"\bif\s*\(")
    ELSE_IF_PATTERN = re.compile(r"\belse\s+if\s*\(")
    ELSE_PATTERN = re.compile(r"\belse\b")
    SWITCH_PATTERN = re.compile(r"\bswitch\s*\(")
    CASE_PATTERN = re.compile(r"\bcase\b|\bdefault\s*:")
    FOR_PATTERN = re.compile(r"\bfor\s*\(")
    WHILE_PATTERN = re.compile(r"\bwhile\s*\(")
    DO_PATTERN = re.compile(r"\bdo\b")
    TRY_PATTERN = re.compile(r"\btry\b")
    CATCH_PATTERN = re.compile(r"\bcatch\s*\(")

    # CUDA/Slang specific
    SYNCTHREADS_PATTERN = re.compile(r"__syncthreads\s*\(\s*\)")

    def __init__(self):
        self.zones: List[CodeZone] = []
        self.current_zone_id = 0

    def _remove_strings_and_comments(self, line: str) -> str:
        """Remove string literals and comments from a line for accurate brace counting."""
        result = []
        i = 0
        in_string = False
        string_char = None

        while i < len(line):
            char = line[i]

            # Handle string literals
            if not in_string:
                if char in ('"', "'"):
                    in_string = True
                    string_char = char
                    i += 1
                    continue
                # Handle // comments
                if char == "/" and i + 1 < len(line) and line[i + 1] == "/":
                    break  # Rest of line is comment
                result.append(char)
            else:
                # In string - look for end
                if char == string_char and (i == 0 or line[i - 1] != "\\"):
                    in_string = False
            i += 1

        return "".join(result)

    def _get_branch_type_before_brace(self, line: str, brace_pos: int) -> str:
        """Determine what type of branch opens at a given brace position."""
        # Look at the content before the brace
        before_brace = line[:brace_pos]

        # Check patterns in order of specificity
        if self.ELSE_IF_PATTERN.search(before_brace):
            return "else_if"
        if self.ELSE_PATTERN.search(before_brace):
            return "else"
        if self.IF_CONSTEXPR_PATTERN.search(before_brace):
            return "if_constexpr"
        if self.IF_PATTERN.search(before_brace):
            return "if"
        if self.CATCH_PATTERN.search(before_brace):
            return "catch"
        if self.TRY_PATTERN.search(before_brace):
            return "try"
        if self.SWITCH_PATTERN.search(before_brace):
            return "switch"
        if self.FOR_PATTERN.search(before_brace):
            return "for"
        if self.WHILE_PATTERN.search(before_brace):
            return "while"
        if self.DO_PATTERN.search(before_brace):
            return "do"

        return "block"  # Generic block

    def _is_sibling_branch(self, branch_type: str) -> bool:
        """Check if this branch type is a sibling to a previous branch (else after if)."""
        return branch_type in ("else", "else_if", "catch")

    def analyze_zones(self, lines: List[str], is_slang: bool = False) -> Dict[int, int]:
        """
        Analyze source lines and return mapping of line_num -> zone_id.

        Uses brace-centric processing:
        - Process each `{` and `}` in order within each line
        - A `}` closes the current zone
        - A `{` opens a new zone
        - For `} else {`, the line belongs to the NEW zone (else-zone)

        Args:
            lines: List of source code lines
            is_slang: Whether this is a Slang file

        Returns:
            Dictionary mapping line numbers (1-indexed) to zone IDs
        """
        line_zones: Dict[int, int] = {}
        self.zones = []
        self.current_zone_id = 0

        in_block_comment = False

        # Stack of (zone_id, zone_type) for nested zones
        # Start with root zone
        zone_stack: List[Tuple[int, str]] = [(0, "root")]

        # Track pending braceless branch (if without {)
        pending_branch: Optional[str] = None
        pending_branch_consumed = False

        for line_num, line in enumerate(lines, start=1):
            # Track block comments
            if "/*" in line:
                if "*/" not in line or line.index("/*") < line.index("*/"):
                    in_block_comment = True
            if "*/" in line:
                in_block_comment = False

            stripped = line.strip()

            # Skip empty lines and full comment lines
            if not stripped or stripped.startswith("//") or in_block_comment:
                line_zones[line_num] = zone_stack[-1][0] if zone_stack else 0
                continue

            # Remove strings and comments for brace analysis
            clean_line = self._remove_strings_and_comments(line)

            # Find all brace positions
            brace_positions: List[Tuple[int, str]] = []
            for i, char in enumerate(clean_line):
                if char in "{}":
                    brace_positions.append((i, char))

            # Track if this line has any zone changes
            zone_at_start = zone_stack[-1][0] if zone_stack else 0

            # Handle pending braceless branch
            if pending_branch and not pending_branch_consumed:
                # This statement is in the pending branch's zone
                # If it's a single statement (no braces), consume the pending branch
                if "{" not in clean_line:
                    # Single statement braceless branch - create zone for just this line
                    self.current_zone_id += 1
                    parent_zone = zone_stack[-1][0] if zone_stack else 0
                    new_zone = CodeZone(
                        zone_id=self.current_zone_id,
                        start_line=line_num,
                        end_line=line_num,
                        parent_zone=parent_zone,
                        zone_type=pending_branch,
                        brace_depth=len(zone_stack),
                    )
                    self.zones.append(new_zone)
                    line_zones[line_num] = self.current_zone_id
                    pending_branch = None
                    continue
                else:
                    pending_branch_consumed = True

            # Process braces left-to-right
            for brace_pos, brace_char in brace_positions:
                if brace_char == "}":
                    # Close current zone
                    if len(zone_stack) > 1:
                        closed = zone_stack.pop()
                        # Update end_line for the closed zone
                        for z in self.zones:
                            if z.zone_id == closed[0]:
                                z.end_line = line_num
                                break

                elif brace_char == "{":
                    # Determine the branch type for this brace
                    branch_type = self._get_branch_type_before_brace(clean_line, brace_pos)

                    # If this is a sibling branch (else/catch), we've already popped
                    # the previous sibling when we saw the `}`

                    # Create new zone
                    self.current_zone_id += 1
                    parent_zone = zone_stack[-1][0] if zone_stack else 0

                    new_zone = CodeZone(
                        zone_id=self.current_zone_id,
                        start_line=line_num,
                        parent_zone=parent_zone,
                        zone_type=branch_type,
                        brace_depth=len(zone_stack),
                    )
                    self.zones.append(new_zone)
                    zone_stack.append((self.current_zone_id, branch_type))

            # Check for braceless branches that start on this line
            # (if/else/for/while without braces)
            if "{" not in clean_line:
                # Check if this line starts a braceless branch
                braceless_type = self._detect_braceless_branch(stripped)
                if braceless_type:
                    pending_branch = braceless_type
                    pending_branch_consumed = False
            else:
                pending_branch = None

            # Assign line to the CURRENT zone (after all braces processed)
            # This is crucial: `} else {` belongs to the else-zone, not the if-zone
            line_zones[line_num] = zone_stack[-1][0] if zone_stack else 0

            # Handle syncthreads as a zone boundary
            if self.SYNCTHREADS_PATTERN.search(line):
                self.current_zone_id += 1
                parent_zone = zone_stack[-1][0] if zone_stack else 0
                new_zone = CodeZone(
                    zone_id=self.current_zone_id,
                    start_line=line_num + 1,
                    parent_zone=parent_zone,
                    zone_type="post_sync",
                    brace_depth=len(zone_stack),
                )
                self.zones.append(new_zone)

        return line_zones

    def _detect_braceless_branch(self, line: str) -> Optional[str]:
        """Detect if a line starts a braceless branch (if/else/for/while without {)."""
        # Must have a branch keyword but no opening brace
        if "{" in line:
            return None

        # Check patterns
        if self.ELSE_IF_PATTERN.search(line) and ")" in line:
            return "else_if"
        if self.ELSE_PATTERN.search(line):
            return "else"
        if self.IF_PATTERN.search(line) and ")" in line:
            return "if"
        if self.FOR_PATTERN.search(line) and ")" in line:
            return "for"
        if self.WHILE_PATTERN.search(line) and ")" in line:
            return "while"

        return None


# =============================================================================
# Continuation Tracker - Multi-line Construct Detection
# =============================================================================


class ContinuationTracker:
    """
    Tracks multi-line constructs where continuation lines should inherit
    coverage from their primary (starting) line.

    Examples:
    - Function signatures spanning multiple lines
    - Function calls with multi-line arguments
    - Template declarations (but NOT comparison operators)

    IMPORTANT: Only tracks PARENTHESES for continuations.
    Angle brackets (<>) are NOT tracked because they conflict with
    comparison operators in C/CUDA code (e.g., `if (a < b)` would be
    incorrectly seen as opening a template).
    """

    # Patterns for template declarations (where < > ARE angle brackets)
    TEMPLATE_PATTERN = re.compile(r"\btemplate\s*<")
    GENERIC_PATTERN = re.compile(r"<\s*(?:typename|class|let|each)\s+\w+")  # Slang generics

    def find_continuations(self, lines: List[str], is_slang: bool = False) -> Dict[int, int]:
        """
        Find continuation lines and map them to their primary line.

        Only tracks PARENTHESES depth to avoid false positives with
        comparison operators (< >) which look like angle brackets.

        Args:
            lines: List of source code lines
            is_slang: Whether this is a Slang file

        Returns:
            Dictionary mapping continuation_line -> primary_line
        """
        continuations: Dict[int, int] = {}

        paren_depth = 0
        primary_line: Optional[int] = None
        in_block_comment = False

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Track block comments
            if "/*" in line and "*/" not in line:
                in_block_comment = True
            if "*/" in line:
                in_block_comment = False

            if in_block_comment or stripped.startswith("//"):
                continue

            # Skip empty lines and preprocessor directives
            if not stripped or stripped.startswith("#"):
                continue

            # Remove strings and comments for accurate counting
            code = self._remove_strings_and_comments(line)

            # Count only parentheses (NOT angle brackets - too many false positives)
            open_parens = code.count("(")
            close_parens = code.count(")")

            # Check if this line starts a multi-line construct
            if paren_depth == 0:
                # Check if this opens a multi-line construct
                if open_parens > close_parens:
                    # This is a primary line - unclosed parenthesis
                    primary_line = line_num
            else:
                # We're in a continuation (unclosed paren from earlier line)
                if primary_line and primary_line != line_num:
                    continuations[line_num] = primary_line

            # Update paren depth
            paren_depth += open_parens - close_parens
            paren_depth = max(0, paren_depth)

            # If we've closed all parens, reset
            if paren_depth == 0:
                primary_line = None

        # Also handle Slang attributes that precede functions
        if is_slang:
            self._group_slang_attributes(lines, continuations)

        return continuations

    def _remove_strings_and_comments(self, line: str) -> str:
        """Remove string literals and trailing comments from a line."""
        result = []
        in_string = False
        string_char = None
        prev_char = ""
        i = 0

        while i < len(line):
            char = line[i]

            if not in_string:
                # Check for // comment start
                if char == "/" and i + 1 < len(line) and line[i + 1] == "/":
                    break  # Rest of line is comment

                if char in ('"', "'"):
                    in_string = True
                    string_char = char
                else:
                    result.append(char)
            else:
                if char == string_char and prev_char != "\\":
                    in_string = False

            prev_char = char
            i += 1

        return "".join(result)

    def _group_slang_attributes(self, lines: List[str], continuations: Dict[int, int]) -> None:
        """
        Group Slang attributes with the function they precede.
        Attributes like [Differentiable], [ForceInline] should inherit
        coverage from the function they decorate.
        """
        ATTR_PATTERN = re.compile(r'^\s*\[[\w\(\)"\']+\]\s*$')
        FUNC_PATTERN = re.compile(r"^\s*(?:public\s+|private\s+)?(?:static\s+)?[\w<>,\s]+\s+\w+\s*\(")

        attr_lines: List[int] = []

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            if ATTR_PATTERN.match(stripped):
                attr_lines.append(line_num)
            elif attr_lines and FUNC_PATTERN.match(stripped):
                # This is the function following attributes
                # Map all attribute lines to this function line
                for attr_line in attr_lines:
                    continuations[attr_line] = line_num
                attr_lines = []
            elif attr_lines and stripped and not stripped.startswith("//"):
                # Non-attribute, non-empty line - reset
                attr_lines = []


# =============================================================================
# Streaming LCOV Parser
# =============================================================================


class LCOVParser:
    """
    Streaming parser for LCOV format files.
    Yields one FileRecord at a time, keeping memory usage low.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def parse(self) -> Generator[FileRecord, None, None]:
        """
        Parse LCOV file and yield FileRecord objects one at a time.

        LCOV format reference:
            SF:<source file path>
            FN:<line>,<function name>
            FNDA:<hits>,<function name>
            FNF:<functions found>
            FNH:<functions hit>
            DA:<line>,<hits>
            LF:<lines found>
            LH:<lines hit>
            end_of_record
        """
        current_record: Optional[FileRecord] = None

        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith("SF:"):
                    # Start new record
                    if current_record is not None:
                        yield current_record
                    current_record = FileRecord(source_file=line[3:])

                elif line.startswith("DA:"):
                    # Line coverage data: DA:line,hits
                    if current_record:
                        parts = line[3:].split(",")
                        if len(parts) >= 2:
                            try:
                                line_num = int(parts[0])
                                hits = int(parts[1])
                                current_record.line_coverage[line_num] = hits
                            except ValueError:
                                pass

                elif line.startswith("FN:"):
                    # Function definition: FN:line,name
                    if current_record:
                        parts = line[3:].split(",", 1)
                        if len(parts) >= 2:
                            try:
                                line_num = int(parts[0])
                                func_name = parts[1]
                                current_record.functions[func_name] = (line_num, 0)
                            except ValueError:
                                pass

                elif line.startswith("FNDA:"):
                    # Function hit count: FNDA:hits,name
                    if current_record:
                        parts = line[5:].split(",", 1)
                        if len(parts) >= 2:
                            try:
                                hits = int(parts[0])
                                func_name = parts[1]
                                if func_name in current_record.functions:
                                    line_num = current_record.functions[func_name][0]
                                    current_record.functions[func_name] = (line_num, hits)
                            except ValueError:
                                pass

                elif line.startswith("FNF:"):
                    if current_record:
                        try:
                            current_record.functions_found = int(line[4:])
                        except ValueError:
                            pass

                elif line.startswith("FNH:"):
                    if current_record:
                        try:
                            current_record.functions_hit = int(line[4:])
                        except ValueError:
                            pass

                elif line.startswith("LF:"):
                    if current_record:
                        try:
                            current_record.lines_found = int(line[3:])
                        except ValueError:
                            pass

                elif line.startswith("LH:"):
                    if current_record:
                        try:
                            current_record.lines_hit = int(line[3:])
                        except ValueError:
                            pass

                elif line == "end_of_record":
                    if current_record is not None:
                        yield current_record
                        current_record = None

        # Yield final record if file doesn't end with end_of_record
        if current_record is not None:
            yield current_record


# =============================================================================
# Source File Analyzer
# =============================================================================


class SourceAnalyzer:
    """
    Analyzes source files (.cu, .cuh, .slang) to classify lines
    and detect function boundaries.
    """

    # Slang patterns needed for function/struct detection (not in shared classifier)
    SLANG_STRUCT_PATTERN = re.compile(r"^\s*struct\s+\w+")
    SLANG_PROPERTY_PATTERN = re.compile(r"^\s*property\s+")

    # Function detection patterns
    CUDA_KERNEL_PATTERN = re.compile(r"__global__\s+\w+")
    CUDA_DEVICE_PATTERN = re.compile(r"__device__\s+\w+")
    SLANG_SHADER_PATTERN = re.compile(r'\[shader\s*\(\s*["\'](\w+)["\']\s*\)\]')
    SLANG_DIFFERENTIABLE_PATTERN = re.compile(r"\[Differentiable\]")

    # General function signature pattern
    FUNCTION_PATTERN = re.compile(
        r"^\s*(?:static\s+)?"  # Optional static
        r"(?:inline\s+)?"  # Optional inline
        r"(?:__global__\s+|__device__\s+|__host__\s+)*"  # CUDA qualifiers
        r"(?:[\w:<>]+\s+)+?"  # Return type
        r"(\w+)\s*\("  # Function name and opening paren
    )

    def __init__(self, source_roots: List[str], cache_size: int = 20):
        self.source_roots = source_roots
        self.cache = LRUCache(max_size=cache_size)
        self._source_index: Optional[Dict[str, List[str]]] = None

    def _build_source_index(self) -> Dict[str, List[str]]:
        """Build index of source files for path resolution."""
        index: Dict[str, List[str]] = {}
        extensions = {".cu", ".cuh", ".slang", ".h", ".hpp"}

        for root in self.source_roots:
            root_path = Path(root)
            if not root_path.exists():
                continue

            for ext in extensions:
                for file_path in root_path.rglob(f"*{ext}"):
                    filename = file_path.name
                    if filename not in index:
                        index[filename] = []
                    index[filename].append(str(file_path))

        return index

    def _get_source_index(self) -> Dict[str, List[str]]:
        if self._source_index is None:
            self._source_index = self._build_source_index()
        return self._source_index

    def resolve_path(self, file_path: str) -> Optional[str]:
        """
        Resolve a file path to an actual source file.
        Handles bazel sandbox paths, symlinks, and relative paths.
        """
        # Direct path check
        if os.path.exists(file_path):
            return os.path.realpath(file_path)

        # Extract from common bazel path patterns
        patterns = [
            r".*/execroot/_main/(.*)",
            r".*/runfiles/_main/(.*)",
            r".*/\.?slangtorch_cache/[^/]+/[^/]+/\d+/((?:libs|nre|internal)/.*)",
            r"/proc/self/cwd/(.*)",
        ]

        relative_path = file_path
        for pattern in patterns:
            match = re.search(pattern, file_path)
            if match:
                relative_path = match.group(1)
                break

        # Try source roots with relative path
        for root in self.source_roots:
            candidate = os.path.join(root, relative_path)
            if os.path.exists(candidate):
                return os.path.realpath(candidate)

        # Fall back to filename lookup
        filename = os.path.basename(file_path)
        index = self._get_source_index()

        if filename in index:
            candidates = index[filename]
            if len(candidates) == 1:
                return candidates[0]

            # Find best match by path suffix
            path_parts = relative_path.replace("\\", "/").split("/")
            best_match = None
            best_score = 0

            for candidate in candidates:
                candidate_parts = candidate.replace("\\", "/").split("/")
                score = sum(
                    1
                    for i in range(1, min(len(path_parts), len(candidate_parts)) + 1)
                    if path_parts[-i] == candidate_parts[-i]
                )
                if score > best_score:
                    best_score = score
                    best_match = candidate

            if best_match:
                return best_match

        return None

    def load_source(self, file_path: str) -> Optional[List[str]]:
        """Load source file contents with caching."""
        resolved = self.resolve_path(file_path)
        if not resolved:
            return None

        # Check cache
        cached = self.cache.get(resolved)
        if cached is not None:
            return cached

        # Load from disk
        try:
            with open(resolved, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            self.cache.put(resolved, lines)
            return lines
        except (IOError, OSError):
            return None

    def classify_line(self, line: str, in_block_comment: bool, is_slang: bool) -> Tuple[LineType, bool]:
        """
        Classify a single line of source code.
        Delegates to shared line_classifier module.
        Returns (LineType, new_in_block_comment_state).
        """
        return _classify_line_shared(line, in_block_comment, is_slang)

    def analyze_file(
        self, file_path: str
    ) -> Tuple[Dict[int, LineInfo], List[FunctionInfo], Dict[int, int], Dict[int, int]]:
        """
        Analyze a source file and return line classifications, function info,
        zone mappings, and continuation mappings.

        Returns:
            Tuple of (line_infos, functions, zones, continuations)
            - line_infos: Dict[int, LineInfo] - line number to line info
            - functions: List[FunctionInfo] - detected functions
            - zones: Dict[int, int] - line number to zone ID
            - continuations: Dict[int, int] - continuation line to primary line
        """
        lines = self.load_source(file_path)
        if not lines:
            return {}, [], {}, {}

        is_slang = file_path.endswith(".slang")
        line_infos: Dict[int, LineInfo] = {}
        functions: List[FunctionInfo] = []

        # Analyze zones and continuations
        zone_tracker = ZoneTracker()
        zones = zone_tracker.analyze_zones(lines, is_slang)

        continuation_tracker = ContinuationTracker()
        continuations = continuation_tracker.find_continuations(lines, is_slang)

        in_block_comment = False
        brace_depth = 0
        current_function: Optional[FunctionInfo] = None
        pending_attributes: List[str] = []

        for line_num, line in enumerate(lines, start=1):
            # Classify line
            line_type, in_block_comment = self.classify_line(line, in_block_comment, is_slang)

            # Track brace depth
            if line_type == LineType.EXECUTABLE or line_type == LineType.BRACE_ONLY:
                brace_depth += line.count("{") - line.count("}")
                brace_depth = max(0, brace_depth)

            # Detect function boundaries
            func_name = None

            # Check for Slang shader attribute
            shader_match = self.SLANG_SHADER_PATTERN.search(line)
            if shader_match:
                pending_attributes.append(f"shader:{shader_match.group(1)}")

            if self.SLANG_DIFFERENTIABLE_PATTERN.search(line):
                pending_attributes.append("differentiable")

            # Check for CUDA kernel
            if self.CUDA_KERNEL_PATTERN.search(line):
                func_match = self.FUNCTION_PATTERN.search(line)
                if func_match:
                    func_name = func_match.group(1)
                    func = FunctionInfo(name=func_name, start_line=line_num, is_kernel=True)
                    functions.append(func)
                    current_function = func
                    pending_attributes = []

            # Check for regular function
            elif line_type == LineType.EXECUTABLE and "(" in line:
                func_match = self.FUNCTION_PATTERN.search(line)
                if func_match:
                    func_name = func_match.group(1)
                    # Skip common non-function patterns
                    if func_name not in ("if", "while", "for", "switch", "return", "sizeof"):
                        is_shader = any("shader:" in attr for attr in pending_attributes)
                        func = FunctionInfo(name=func_name, start_line=line_num, is_shader=is_shader)
                        functions.append(func)
                        current_function = func
                        pending_attributes = []

            # Track function end
            if current_function and brace_depth == 0 and line_type in (LineType.EXECUTABLE, LineType.BRACE_ONLY):
                if "}" in line:
                    current_function.end_line = line_num
                    current_function = None

            # Get zone and continuation info
            zone_id = zones.get(line_num, 0)
            continuation_of = continuations.get(line_num)

            # Store line info
            line_infos[line_num] = LineInfo(
                line_num=line_num,
                line_type=line_type,
                content=line.rstrip(),
                in_block_comment=in_block_comment,
                in_function=current_function is not None,
                function_name=current_function.name if current_function else None,
                brace_depth=brace_depth,
                zone_id=zone_id,
                continuation_of=continuation_of,
            )

        return line_infos, functions, zones, continuations

    def get_executable_lines(self, file_path: str) -> Set[int]:
        """Get set of executable line numbers for a file."""
        line_infos, _, _, _ = self.analyze_file(file_path)
        return {num for num, info in line_infos.items() if info.line_type == LineType.EXECUTABLE}


# =============================================================================
# Coverage Inference Engine
# =============================================================================


class CoverageInferrer:
    """
    Infers coverage for lines based on surrounding context.

    Key principles:
    1. Coverage only propagates within the same control flow ZONE
    2. Multi-line constructs (function args) inherit from primary line
    3. Never infer coverage across branch boundaries (if/else/switch/case)
    """

    def __init__(self, analyzer: SourceAnalyzer):
        self.analyzer = analyzer

    def apply_continuations(
        self, coverage: Dict[int, int], continuations: Dict[int, int], line_infos: Dict[int, LineInfo]
    ) -> Dict[int, int]:
        """
        Apply continuation coverage: lines that are continuations of a primary
        line inherit coverage from that primary line.

        This handles:
        - Multi-line function signatures
        - Multi-line function calls
        - Slang attributes preceding functions

        Args:
            coverage: Current coverage dictionary
            continuations: Mapping of continuation_line -> primary_line
            line_infos: Line classification data

        Returns:
            Updated coverage dictionary
        """
        updated = dict(coverage)

        for cont_line, primary_line in continuations.items():
            primary_info = line_infos.get(primary_line)
            cont_info = line_infos.get(cont_line)

            # Only propagate if both are valid lines
            if not primary_info or not cont_info:
                continue

            # Get primary line's coverage
            primary_hits = coverage.get(primary_line, 0)

            # If primary is covered, continuation inherits coverage
            if primary_hits > 0:
                if cont_line not in updated or updated[cont_line] == 0:
                    updated[cont_line] = primary_hits
            # If continuation is covered but primary isn't, propagate back
            elif cont_line in coverage and coverage[cont_line] > 0:
                if primary_line not in updated or updated[primary_line] == 0:
                    updated[primary_line] = coverage[cont_line]

        return updated

    def infer_block_coverage(
        self,
        record: FileRecord,
        line_infos: Dict[int, LineInfo],
        zones: Dict[int, int],
        continuations: Dict[int, int],
        max_gap: int = 5,
    ) -> Dict[int, int]:
        """
        Infer coverage for lines within code blocks, respecting zone boundaries.

        Coverage is ONLY propagated within the same zone. This ensures:
        - if-branch coverage doesn't bleed into else-branch
        - switch cases remain separate
        - loops and their bodies are tracked properly

        Args:
            record: Original coverage record
            line_infos: Line classification data
            zones: Mapping of line_num -> zone_id
            continuations: Mapping of continuation_line -> primary_line
            max_gap: Maximum gap between covered lines to infer within a zone

        Returns:
            Updated line coverage dictionary
        """
        coverage = dict(record.line_coverage)

        # First, apply continuation inheritance
        coverage = self.apply_continuations(coverage, continuations, line_infos)

        # Get executable lines grouped by zone
        zone_executable: Dict[int, List[int]] = {}
        for num, info in line_infos.items():
            if info.line_type == LineType.EXECUTABLE:
                zone_id = zones.get(num, 0)
                if zone_id not in zone_executable:
                    zone_executable[zone_id] = []
                zone_executable[zone_id].append(num)

        # Sort lines within each zone
        for zone_id in zone_executable:
            zone_executable[zone_id].sort()

        # Propagate coverage ONLY within each zone
        for zone_id, zone_lines in zone_executable.items():
            if len(zone_lines) < 2:
                continue

            # Find covered lines in this zone
            covered_in_zone = [line for line in zone_lines if coverage.get(line, 0) > 0]

            if len(covered_in_zone) < 2:
                continue

            # Propagate between consecutive covered lines in the same zone
            for i in range(len(covered_in_zone) - 1):
                start = covered_in_zone[i]
                end = covered_in_zone[i + 1]

                # Find executable lines in this gap (same zone only)
                gap_lines = [num for num in zone_lines if start < num < end]

                # Only infer if gap is small and all lines are in same zone
                if len(gap_lines) <= max_gap:
                    # Verify all gap lines are in the same zone
                    all_same_zone = all(zones.get(line, -1) == zone_id for line in gap_lines)

                    if all_same_zone:
                        # Also check same function
                        start_info = line_infos.get(start)
                        end_info = line_infos.get(end)

                        if start_info and end_info and start_info.function_name == end_info.function_name:
                            # Check that no gap line is a branch entry
                            has_branch = any(self._is_branch_entry(line_infos.get(line)) for line in gap_lines)

                            if not has_branch:
                                # Infer coverage with minimum hit count
                                min_hits = min(coverage[start], coverage[end])
                                for gap_line in gap_lines:
                                    if coverage.get(gap_line, 0) == 0:
                                        coverage[gap_line] = min_hits

        return coverage

    def _is_branch_entry(self, info: Optional[LineInfo]) -> bool:
        """Check if a line is a branch entry point (if/else/case)."""
        if not info:
            return False

        content = info.content.strip()

        # Check for branch keywords
        branch_patterns = [
            r"^\s*if\s*\(",
            r"^\s*else\s*\{?",
            r"^\s*else\s+if\s*\(",
            r"^\s*case\s+",
            r"^\s*default\s*:",
            r"^\s*catch\s*\(",
        ]

        for pattern in branch_patterns:
            if re.match(pattern, content):
                return True

        return False

    def detect_uncovered_functions(self, record: FileRecord, functions: List[FunctionInfo]) -> List[FunctionInfo]:
        """
        Detect functions that have zero coverage.

        Updates function hit counts based on whether any lines
        within the function are covered.
        """
        for func in functions:
            if func.end_line == 0:
                continue

            # Check if any line in function range is covered
            for line_num in range(func.start_line, func.end_line + 1):
                hits = record.line_coverage.get(line_num, 0)
                if hits > 0:
                    func.hit_count += hits

        return functions


# =============================================================================
# Coverage Processor
# =============================================================================


class CoverageProcessor:
    """
    Main processor that orchestrates parsing, analysis, and inference.
    """

    def __init__(
        self, source_roots: List[str], infer_blocks: bool = True, detect_functions: bool = True, verbose: bool = False
    ):
        self.source_roots = source_roots
        self.infer_blocks = infer_blocks
        self.detect_functions = detect_functions
        self.verbose = verbose

        self.analyzer = SourceAnalyzer(source_roots)
        self.inferrer = CoverageInferrer(self.analyzer)

        # Statistics
        self.total_files = 0
        self.total_lines_found = 0
        self.total_lines_hit = 0
        self.total_functions_found = 0
        self.total_functions_hit = 0
        self.files_processed: List[Tuple[str, float]] = []  # (file, coverage%)

    def canonicalize_path(self, file_path: str) -> str:
        """
        Canonicalize path to a repo-relative form.
        """
        path = file_path.replace("\\", "/")

        # Extract from common patterns
        patterns = [
            (r".*/\.?slangtorch_cache/[^/]+/[^/]+/\d+/((?:libs|nre|internal)/.*)", 1),
            (r".*/runfiles/_main/(.*)", 1),
            (r".*/execroot/_main/(.*)", 1),
            (r"/proc/self/cwd/(.*)", 1),
        ]

        for pattern, group in patterns:
            match = re.search(pattern, path)
            if match:
                result = match.group(group)
                # Handle nested patterns
                if result.startswith("bazel-out/") or result.startswith("bazel-bin/"):
                    nested = re.search(r"runfiles/_main/(.*)", result)
                    if nested:
                        return nested.group(1)
                return result

        # If already relative or no pattern matched
        if not path.startswith("/"):
            return path

        # Try to make path relative to source roots
        for root in self.source_roots:
            root_normalized = root.rstrip("/") + "/"
            if path.startswith(root_normalized):
                return path[len(root_normalized) :]

        return os.path.basename(path)

    def process_record(self, record: FileRecord) -> FileRecord:
        """
        Process a single file record with analysis and inference.

        Uses zone-aware inference to ensure coverage only propagates
        within the same control flow path, never across branch boundaries.
        """
        # Analyze source file - now returns zones and continuations too
        line_infos, functions, zones, continuations = self.analyzer.analyze_file(record.source_file)

        # Get executable lines
        executable_lines = {num for num, info in line_infos.items() if info.line_type == LineType.EXECUTABLE}

        # Apply zone-aware block inference if enabled
        if self.infer_blocks and line_infos:
            record.line_coverage = self.inferrer.infer_block_coverage(record, line_infos, zones, continuations)
        elif continuations:
            # Even if block inference is disabled, still apply continuations
            record.line_coverage = self.inferrer.apply_continuations(record.line_coverage, continuations, line_infos)

        # Detect function coverage if enabled
        if self.detect_functions and functions:
            functions = self.inferrer.detect_uncovered_functions(record, functions)

            # Update function info in record
            for func in functions:
                if func.name not in record.functions:
                    record.functions[func.name] = (func.start_line, func.hit_count)
                else:
                    old_line, old_hits = record.functions[func.name]
                    record.functions[func.name] = (old_line, max(old_hits, func.hit_count))

        # Filter coverage to only executable lines (and their continuations)
        if executable_lines:
            filtered_coverage = {}

            # Include executable lines
            for line_num, hits in record.line_coverage.items():
                if line_num in executable_lines:
                    filtered_coverage[line_num] = hits

            # Ensure all executable lines are in coverage (as 0 if not present)
            for line_num in executable_lines:
                if line_num not in filtered_coverage:
                    filtered_coverage[line_num] = 0

            record.line_coverage = filtered_coverage

        # Recalculate statistics
        record.lines_found = len(record.line_coverage)
        record.lines_hit = sum(1 for hits in record.line_coverage.values() if hits > 0)
        record.functions_found = len(record.functions)
        record.functions_hit = sum(1 for _, hits in record.functions.values() if hits > 0)

        return record

    # Supported file extensions for CUDA/Slang coverage
    SUPPORTED_EXTENSIONS = {".cu", ".cuh", ".slang", ".h", ".hpp"}

    def _is_supported_file(self, file_path: str) -> bool:
        """Check if a file has a supported extension (.cu, .cuh, .slang)."""
        lower_path = file_path.lower()
        return any(lower_path.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)

    def process_file(self, input_path: str, output_path: str) -> None:
        """
        Process an entire LCOV file and write cleaned output.

        Only processes .cu, .cuh, and .slang files. Other files are skipped.
        """
        parser = LCOVParser(input_path)

        print(f"Processing: {input_path}")
        print(f"Output: {output_path}")
        print(f"Source roots: {self.source_roots}")
        print(f"Supported extensions: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}")
        print("-" * 60)

        passthrough_files = 0

        with open(output_path, "w", encoding="utf-8") as out:
            for record in parser.parse():
                # Only process .cu, .cuh, .slang files; pass through others unchanged
                if not self._is_supported_file(record.source_file):
                    passthrough_files += 1
                    if self.verbose:
                        print(f"Passthrough (unchanged): {os.path.basename(record.source_file)}")
                    # Write the record as-is without processing
                    self._write_record(out, record)
                    continue

                self.total_files += 1

                if self.verbose:
                    print(f"Processing: {os.path.basename(record.source_file)}...")

                # Process the record
                processed = self.process_record(record)

                # Calculate coverage percentage
                if processed.lines_found > 0:
                    coverage_pct = 100.0 * processed.lines_hit / processed.lines_found
                else:
                    coverage_pct = 0.0

                self.files_processed.append((self.canonicalize_path(record.source_file), coverage_pct))

                # Update totals
                self.total_lines_found += processed.lines_found
                self.total_lines_hit += processed.lines_hit
                self.total_functions_found += processed.functions_found
                self.total_functions_hit += processed.functions_hit

                # Write to output
                self._write_record(out, processed)

        if passthrough_files > 0:
            print(f"\nPassed through {passthrough_files} files unchanged (non-CUDA/Slang)")

        # Print summary
        self._print_summary()

    def _write_record(self, out, record: FileRecord) -> None:
        """Write a processed record in LCOV format."""
        canonical_path = self.canonicalize_path(record.source_file)
        out.write(f"SF:{canonical_path}\n")

        # Write function data
        for func_name, (line, hits) in sorted(record.functions.items(), key=lambda x: x[1][0]):
            out.write(f"FN:{line},{func_name}\n")

        for func_name, (line, hits) in sorted(record.functions.items(), key=lambda x: x[1][0]):
            out.write(f"FNDA:{hits},{func_name}\n")

        out.write(f"FNF:{record.functions_found}\n")
        out.write(f"FNH:{record.functions_hit}\n")

        # Write line coverage data
        for line_num in sorted(record.line_coverage.keys()):
            hits = record.line_coverage[line_num]
            out.write(f"DA:{line_num},{hits}\n")

        out.write(f"LF:{record.lines_found}\n")
        out.write(f"LH:{record.lines_hit}\n")
        out.write("end_of_record\n")

    def _print_summary(self) -> None:
        """Print coverage summary to terminal."""
        print("\n" + "=" * 60)
        print("COVERAGE SUMMARY")
        print("=" * 60)

        # Sort files by coverage (ascending)
        sorted_files = sorted(self.files_processed, key=lambda x: x[1])

        # Show files with lowest coverage first
        print("\nFiles by coverage (lowest first):")
        print("-" * 60)
        for file_path, coverage in sorted_files[:20]:
            bar_len = int(coverage / 5)  # 20 chars = 100%
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {coverage:5.1f}% {bar} {os.path.basename(file_path)}")

        if len(sorted_files) > 20:
            print(f"  ... and {len(sorted_files) - 20} more files")

        # Overall statistics
        print("\n" + "-" * 60)
        print("TOTALS:")
        print("-" * 60)

        if self.total_lines_found > 0:
            overall_coverage = 100.0 * self.total_lines_hit / self.total_lines_found
        else:
            overall_coverage = 0.0

        if self.total_functions_found > 0:
            func_coverage = 100.0 * self.total_functions_hit / self.total_functions_found
        else:
            func_coverage = 0.0

        print(f"  Files processed:    {self.total_files}")
        print(f"  Lines found:        {self.total_lines_found}")
        print(f"  Lines hit:          {self.total_lines_hit}")
        print(f"  Line coverage:      {overall_coverage:.1f}%")
        print(f"  Functions found:    {self.total_functions_found}")
        print(f"  Functions hit:      {self.total_functions_hit}")
        print(f"  Function coverage:  {func_coverage:.1f}%")
        print("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Post-process LCOV coverage data for CUDA/Slang files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python postprocess_coverage.py coverage.dat -o cleaned.info

    # With source roots for path resolution
    python postprocess_coverage.py coverage.dat -o cleaned.info -s ./nre/libs -s ./libs

    # Disable inference
    python postprocess_coverage.py coverage.dat -o cleaned.info --no-infer-blocks

Output:
    Generates a cleaned LCOV .info file with:
    - Only executable lines in DA records
    - Accurate LF/LH counts based on source analysis
    - Inferred coverage for code blocks
    - Function coverage from source analysis
        """,
    )

    parser.add_argument("input", help="Input LCOV coverage file (.dat or .info)")
    parser.add_argument("-o", "--output", required=True, help="Output cleaned LCOV file")
    parser.add_argument(
        "-s", "--source", action="append", default=[], help="Source directory root (can specify multiple)"
    )
    parser.add_argument("--no-infer-blocks", action="store_true", help="Disable coverage inference within code blocks")
    parser.add_argument("--no-detect-functions", action="store_true", help="Disable function boundary detection")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Default source roots
    source_roots = args.source if args.source else ["."]

    # Also try to find common source directories
    for candidate in ["libs", "nre/libs", "internal"]:
        if os.path.exists(candidate) and candidate not in source_roots:
            source_roots.append(candidate)

    # Create processor
    processor = CoverageProcessor(
        source_roots=source_roots,
        infer_blocks=not args.no_infer_blocks,
        detect_functions=not args.no_detect_functions,
        verbose=args.verbose,
    )

    # Process file
    try:
        processor.process_file(args.input, args.output)
        print(f"\nSuccess: Cleaned coverage written to {args.output}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
