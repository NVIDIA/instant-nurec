#!/usr/bin/env python3
"""
CUDA Code Coverage and Metrics Report Generator
Extracts per-line metrics from NCU reports using correlation IDs
Filters to only source repository files (excludes CUDA system headers)
"""

import argparse
import gc
import os
import re

# NCU Python path is set by the shell script from Bazel's CUDA detection
# Falls back to common path if not set
import shutil
import sys
import tempfile

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional, Set

from line_classifier import get_executable_line_numbers


# =============================================================================
# Memory Management Configuration
# =============================================================================

# Process this many kernels before flushing accumulated data to disk
KERNEL_BATCH_SIZE = 50
# Maximum source files to keep in memory cache (LRU-style eviction)
MAX_SOURCE_CACHE_SIZE = 10
# Force garbage collection every N kernels
GC_INTERVAL = 25
# Print memory status every N kernels
MEMORY_STATUS_INTERVAL = 100

# =============================================================================
# Metric Auto-Detection Configuration
# =============================================================================

# Metric preference order for auto-detection
# SASS-level metrics are preferred (more accurate per-line coverage)
METRIC_PREFERENCES = [
    "smsp__sass_thread_inst_executed_op",  # SASS-level (most accurate)
    "thread_inst_executed_true",  # Hardware counter (faster)
    "smsp__thread_inst_executed",  # Hardware counter (faster)
    "inst_executed",  # Fallback
]


def find_ncu_python_path() -> Optional[str]:
    """Find NCU Python extras path for ncu_report module.

    Prefers NCU_PYTHON_PATH env var (set by wrapper for consistency).
    Fallback searches for installations with ncu_report.py.

    Returns:
        Path to NCU Python extras directory, or None if not found.
    """
    # Env var set by wrapper ensures consistency with ncu binary
    if "NCU_PYTHON_PATH" in os.environ:
        env_path = os.environ["NCU_PYTHON_PATH"]
        if os.path.exists(env_path) and os.path.exists(os.path.join(env_path, "ncu_report.py")):
            return env_path
        # Env var set but invalid - fall through to search

    # Fallback: search known installation locations
    search_dirs = [
        Path("/opt/nvidia/nsight-compute"),  # contains version subdirs
        Path("/opt/nvidia"),  # contains nsight-compute-* dirs
        Path("/usr/local/cuda"),  # contains nsight-compute-* dirs
    ]

    for base in search_dirs:
        if not base.exists():
            continue
        for child in sorted(base.iterdir(), reverse=True):
            if not child.is_dir() or not child.name.startswith("nsight-compute"):
                continue
            # Check direct child (e.g., /opt/nvidia/nsight-compute-2025.1.0/)
            candidate = child / "extras" / "python"
            if (candidate / "ncu_report.py").exists():
                return str(candidate)
            # Check nested version (e.g., /opt/nvidia/nsight-compute/2025.1.0/)
            for subdir in sorted(child.iterdir(), reverse=True):
                if subdir.is_dir():
                    candidate = subdir / "extras" / "python"
                    if (candidate / "ncu_report.py").exists():
                        return str(candidate)

    return None


NCU_PYTHON_PATH = find_ncu_python_path()
if NCU_PYTHON_PATH is None:
    print("Error: NCU Python module (ncu_report) not found.", file=sys.stderr)
    print(
        "  Searched: /opt/nvidia/nsight-compute/*/, /opt/nvidia/nsight-compute-*/, /usr/local/cuda/nsight-compute-*/",
        file=sys.stderr,
    )
    print("  Set NCU_PYTHON_PATH environment variable or install Nsight Compute with Python extras.", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, NCU_PYTHON_PATH)

try:
    import ncu_report
except ImportError:
    print("Error: Cannot import ncu_report module.", file=sys.stderr)
    print(f"  NCU_PYTHON_PATH={NCU_PYTHON_PATH}", file=sys.stderr)
    print("  The path exists but import failed. Check Python compatibility.", file=sys.stderr)
    sys.exit(1)


def _get_file_ext(file_path):
    """Get file extension from a path."""
    return os.path.splitext(file_path)[1].lower()


# =============================================================================
# Repository Root Detection
# =============================================================================

# Cached repo detection results
_REPO_ROOT: Optional[Path] = None
_REPO_DIRECTORIES: Optional[Set[str]] = None


def get_repo_root() -> Path:
    """
    Detect repository root directory.

    Uses BUILD_WORKSPACE_DIRECTORY env var (set by Bazel during bazel run),
    or walks up from script location looking for MODULE.bazel.
    """
    # Bazel sets BUILD_WORKSPACE_DIRECTORY during bazel run
    if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        return Path(os.environ["BUILD_WORKSPACE_DIRECTORY"])

    # Fallback: walk up from script location looking for MODULE.bazel
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "MODULE.bazel").exists():
            return current
        current = current.parent

    raise RuntimeError("Could not detect repository root")


def get_repo_directories(repo_root: Path) -> Set[str]:
    """
    Scan repo root for directories containing BUILD.bazel files.
    These are the valid bazel packages that can contain source files.
    """
    repo_dirs = set()
    for item in repo_root.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Check if it's a bazel package (has BUILD.bazel or BUILD)
            if (item / "BUILD.bazel").exists() or (item / "BUILD").exists():
                repo_dirs.add(item.name)
    return repo_dirs


def _ensure_repo_dirs_loaded():
    """Lazily load and cache repo directories on first use."""
    global _REPO_ROOT, _REPO_DIRECTORIES
    if _REPO_DIRECTORIES is None:
        _REPO_ROOT = get_repo_root()
        _REPO_DIRECTORIES = get_repo_directories(_REPO_ROOT)


# =============================================================================
# Source File Resolution
# =============================================================================

# Cached index of source files for fallback resolution
_SOURCE_FILE_INDEX: Optional[dict] = None


def _build_source_file_index(repo_root: Path) -> dict:
    """
    Build an index mapping filenames to their full paths in the repo.
    Used as fallback when symlinks don't exist.
    """
    index = defaultdict(list)
    search_dirs = ["libs", "nre", "internal"]

    for search_dir in search_dirs:
        search_path = repo_root / search_dir
        if not search_path.exists():
            continue
        for root, _, files in os.walk(search_path):
            for filename in files:
                if any(filename.endswith(ext) for ext in [".cu", ".cuh", ".slang", ".h", ".hpp"]):
                    full_path = os.path.join(root, filename)
                    index[filename].append(full_path)

    return dict(index)


def _get_source_file_index() -> dict:
    """Get or build the source file index."""
    global _SOURCE_FILE_INDEX
    if _SOURCE_FILE_INDEX is None:
        _ensure_repo_dirs_loaded()
        _SOURCE_FILE_INDEX = _build_source_file_index(_REPO_ROOT)
    return _SOURCE_FILE_INDEX


def resolve_runfiles_path(file_path: str) -> Optional[str]:
    """
    Resolve a bazel runfiles path to the actual source file when symlinks don't exist.

    Handles paths like:
        .../runfiles/_main/nrend/kernels/cuda/renderers/gutKBufferRenderer.cuh

    Maps to:
        /repo/libs/nrend/include/nrend/kernels/cuda/renderers/gutKBufferRenderer.cuh
    """
    # Extract path after runfiles/_main/
    match = re.search(r"runfiles/_main/(.*)", file_path)
    if not match:
        return None

    relative_path = match.group(1)  # e.g., "nrend/kernels/cuda/renderers/gutKBufferRenderer.cuh"
    filename = os.path.basename(relative_path)

    # Get the source file index
    index = _get_source_file_index()

    if filename not in index:
        return None

    candidates = index[filename]

    # If only one match, use it
    if len(candidates) == 1:
        return candidates[0]

    # Multiple matches - find best match by path suffix
    # e.g., for "nrend/kernels/cuda/renderers/gutKBufferRenderer.cuh"
    # prefer path ending with that suffix
    path_parts = relative_path.split("/")

    best_match = None
    best_score = 0

    for candidate in candidates:
        candidate_parts = candidate.replace("\\", "/").split("/")
        # Count matching path components from the end
        score = 0
        for i in range(1, min(len(path_parts), len(candidate_parts)) + 1):
            if path_parts[-i] == candidate_parts[-i]:
                score += 1
            else:
                break
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match


# =============================================================================
# File Filtering
# =============================================================================

# Only include CUDA/GPU files in coverage report
INCLUDED_EXTENSIONS = {".cu", ".cuh", ".slang", ".h", ".hpp"}

# Pre-compiled regex patterns for path canonicalization (compiled once at module load)
_RE_SLANGTORCH = re.compile(r".*/\.?slangtorch_cache/[^/]+/[^/]+/\d+/((?:libs|nre|internal)/.*)")
_RE_RUNFILES = re.compile(r".*/runfiles/_main/(.*)")
_RE_EXECROOT = re.compile(r".*/execroot/_main/(.*)")
_RE_RUNFILES_NESTED = re.compile(r"runfiles/_main/(.*)")
_RE_SLANGTORCH_NESTED = re.compile(r"\.?slangtorch_cache/[^/]+/[^/]+/\d+/((?:libs|nre|internal)/.*)")

# Cache for canonicalize_path results to avoid repeated regex operations
_CANONICAL_PATH_CACHE: dict = {}


def canonicalize_path(file_path: str) -> str:
    """
    Canonicalize a file path to a repo-relative form for deduplication.

    Converts paths like:
      /home/user/.cache/bazel/.../sandbox/.../execroot/_main/libs/foo/bar.cu
      /proc/self/cwd/libs/foo/bar.cu
      .../bazel-out/.../runfiles/_main/libs/foo/bar.slang
      .../slangtorch_cache/.../0/libs/foo/bar.slang
      /tmp/.slangtorch_cache/.../0/libs/foo/bar.slang

    To:
      libs/foo/bar.cu

    Results are cached to avoid repeated expensive regex operations.
    Uses pre-compiled regex patterns for speed.
    """
    if not file_path:
        return file_path

    if file_path in _CANONICAL_PATH_CACHE:
        return _CANONICAL_PATH_CACHE[file_path]

    # Normalize slashes
    path = file_path.replace("\\", "/")
    result = None

    # Extract from slangtorch cache paths (using pre-compiled pattern)
    # Pattern: .../slangtorch_cache/.../0/(libs/...) or .../slangtorch_cache/.../0/(nre/...)
    slangtorch_match = _RE_SLANGTORCH.search(path)
    if slangtorch_match:
        result = slangtorch_match.group(1)

    # Extract from runfiles paths (e.g., .../runfiles/_main/libs/...)
    if result is None:
        runfiles_match = _RE_RUNFILES.search(path)
        if runfiles_match:
            result = runfiles_match.group(1)

    # Extract repo-relative path from Bazel sandbox/execroot absolute paths
    if result is None:
        execroot_match = _RE_EXECROOT.search(path)
        if execroot_match:
            result = execroot_match.group(1)
            # If result still has bazel-out prefix, try to extract source path from it
            if result.startswith("bazel-out/") or result.startswith("bazel-bin/"):
                # Try runfiles path: runfiles/_main/(libs/...)
                runfiles_nested = _RE_RUNFILES_NESTED.search(result)
                if runfiles_nested:
                    result = runfiles_nested.group(1)
                else:
                    # Try slangtorch_cache path
                    nested_match = _RE_SLANGTORCH_NESTED.search(result)
                    if nested_match:
                        result = nested_match.group(1)
                    else:
                        result = os.path.basename(path)

    # Extract from /proc/self/cwd/... (bazel test working directory)
    if result is None and path.startswith("/proc/self/cwd/"):
        result = path[len("/proc/self/cwd/") :]

    # If path is already relative (doesn't start with /), preserve it as-is
    if result is None and not path.startswith("/"):
        result = path

    # Check if absolute path is under repo root (handles resolved symlinks)
    # e.g., /home/.../nre/libs/nrend/include/foo.cuh -> libs/nrend/include/foo.cuh
    if result is None:
        _ensure_repo_dirs_loaded()
        if _REPO_ROOT:
            repo_root_str = str(_REPO_ROOT).replace("\\", "/")
            if not repo_root_str.endswith("/"):
                repo_root_str += "/"
            if path.startswith(repo_root_str):
                result = path[len(repo_root_str) :]

    # Fallback: return basename if no pattern matches
    if result is None:
        result = os.path.basename(path)

    _CANONICAL_PATH_CACHE[file_path] = result
    return result


def is_repo_file(file_path: str) -> bool:
    """
    Determine if a file is a repository file that should be included.

    Uses allowlist approach: only include files that canonicalize to
    paths starting with auto-detected repo directories (those with BUILD.bazel).
    """
    _ensure_repo_dirs_loaded()

    if not file_path:
        return False

    normalized = os.path.normpath(file_path).replace("\\", "/")
    normalized_lower = normalized.lower()

    # Filter by extension - only include CUDA/GPU files
    ext = os.path.splitext(normalized_lower)[1]
    if ext not in INCLUDED_EXTENSIONS:
        return False

    # Exclude slangtorch-generated intermediate CUDA files (boilerplate)
    # These are in slangtorch_cache and end with _cuda.cu
    if "slangtorch_cache" in normalized_lower and normalized_lower.endswith("_cuda.cu"):
        return False

    # Exclude Slang-generated PTX cache files (JIT compilation cache)
    # These are in ptx_cache directories with hash-based names like render.<hash>.cu
    if "/ptx_cache/" in normalized or "\\ptx_cache\\" in normalized:
        return False

    # Canonicalize and check if first component is a known repo directory
    canonical = canonicalize_path(file_path)
    parts = canonical.split("/")
    if parts and parts[0] in _REPO_DIRECTORIES:
        return True

    return False


# Cache for is_system_file results to avoid repeated regex/path operations
_SYSTEM_FILE_CACHE: dict = {}


def is_system_file(file_path: str) -> bool:
    """
    Determine if a file is a CUDA system header/library file or external package.
    Returns True if file should be excluded from coverage.

    Results are cached to avoid repeated expensive path operations.
    """
    if file_path in _SYSTEM_FILE_CACHE:
        return _SYSTEM_FILE_CACHE[file_path]

    result = not is_repo_file(file_path)
    _SYSTEM_FILE_CACHE[file_path] = result
    return result


def find_matching_metric(action, explicit_metric=None, preferences=None):
    """Find metric with correlation IDs, optionally trying explicit metric first.

    Uses EXACT string matching for metric names.

    Args:
        action: NCU action object from which to extract metrics
        explicit_metric: User-specified metric name to try first (exact match)
        preferences: List of metric names to search for (default: METRIC_PREFERENCES)

    Returns:
        Tuple of (metric object, metric name) or (None, None) if not found
    """
    if preferences is None:
        preferences = METRIC_PREFERENCES

    try:
        available = list(action.metric_names())
    except (SystemError, RuntimeError, AttributeError) as e:
        print(f"Error: Failed to get metric names from NCU action: {e}", file=sys.stderr)
        return None, None

    # If explicit metric requested, try it first (exact match)
    if explicit_metric:
        if explicit_metric in available:
            metric = action.metric_by_name(explicit_metric)
            if metric and metric.has_correlation_ids():
                return metric, explicit_metric
            else:
                print(f"Warning: Metric '{explicit_metric}' has no correlation IDs, falling back to auto-detect")

    # Auto-detect from preference order (exact match)
    for preferred_metric in preferences:
        if preferred_metric not in available:
            continue
        metric = action.metric_by_name(preferred_metric)
        if not metric or not metric.has_correlation_ids():
            print(f"Warning: Metric '{preferred_metric}' has no correlation IDs, trying next")
            continue
        print(f"Found metric (with correlation IDs): {preferred_metric}")
        return metric, preferred_metric

    return None, None


class NCUCoverageAnalyzer:
    """Extract per-line metrics and code coverage from NCU reports

    Memory-optimized version that uses disk-backed storage for large reports.
    """

    def __init__(self, report_path, source_roots=None, target_metric=None, verbose=False):
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Report file not found: {report_path}")

        self.report = ncu_report.load_report(report_path)
        self.source_roots = source_roots or ["."]
        self.target_metric = target_metric
        self.verbose = verbose

        # Memory-optimized: use temp directory for intermediate data
        self.temp_dir = tempfile.mkdtemp(prefix="ncu_coverage_")
        self.line_data_file = os.path.join(self.temp_dir, "line_data.jsonl")

        # In-memory data structures - kept minimal
        # line_data is loaded from disk when needed, not accumulated in memory
        self.line_data = None  # Will be populated from disk when generating report

        # LRU-style source cache with size limit (OrderedDict for O(1) operations)
        self.source_cache = OrderedDict()

        # Lightweight tracking - just counts and sets of unique values
        self.total_kernel_launches = 0
        self.kernel_name_counts = defaultdict(int)
        self.file_coverage = {}
        self.corr_id_lines = defaultdict(set)  # file -> set of line numbers

        # Track excluded/included files
        self.excluded_files = set()
        self.included_files = set()

        # Batch accumulator for disk writes
        self._batch_buffer = []
        self._kernels_since_flush = 0

        self._kernels_since_gc = 0

        # Kernel source mapping cache: kernel_name -> list of (corr_idx, file, line)
        # Multiple instances of the same kernel have identical source correlations,
        # so we cache and reuse them to avoid redundant NCU API calls
        self._kernel_source_cache = {}

        # Path resolution cache (shared across kernels)
        self._path_cache = {}

    def _flush_batch_to_disk(self):
        """Write accumulated batch data to disk and clear buffer.

        Uses TSV format for faster serialization than JSON.
        """
        if not self._batch_buffer:
            return

        # Use writelines for single syscall - batch buffer contains pre-formatted TSV lines
        with open(self.line_data_file, "a") as f:
            f.writelines(self._batch_buffer)

        entries_flushed = len(self._batch_buffer)
        self._batch_buffer = []
        self._kernels_since_flush = 0

        if self.verbose:
            print(f"    [Memory] Flushed {entries_flushed} entries to disk")

    def _add_line_entry(self, file_name, line_num, metric_value, kernel_name, instance):
        """Add a line entry to the batch buffer as pre-formatted TSV line."""
        # TSV format: file\tline\tmetric\tkernel#instance\n
        self._batch_buffer.append(f"{file_name}\t{line_num}\t{metric_value}\t{kernel_name}#{instance}\n")

    def _maybe_gc(self):
        """Run garbage collection if needed"""
        self._kernels_since_gc += 1
        if self._kernels_since_gc >= GC_INTERVAL:
            gc.collect()
            self._kernels_since_gc = 0

    def _evict_source_cache(self):
        """Evict oldest entries from source cache if over limit (O(1) with OrderedDict)"""
        while len(self.source_cache) > MAX_SOURCE_CACHE_SIZE:
            self.source_cache.popitem(last=False)  # Remove oldest entry

    def _load_line_data_from_disk(self):
        """Load and aggregate line data from disk into memory for report generation.

        Parses TSV format: file\\tline\\tmetric\\tkernel#instance
        """
        print("Loading line data from disk...")

        line_data = defaultdict(
            lambda: defaultdict(lambda: {"metric_value": 0.0, "line_exec_count": 0, "kernels": set()})
        )

        if not os.path.exists(self.line_data_file):
            return line_data

        entry_count = 0
        with open(self.line_data_file, "r") as f:
            for line in f:
                try:
                    # TSV format: file\tline\tmetric\tkernel#instance
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 4:
                        continue
                    file_name, line_num_str, metric_str, kernel = parts
                    line_num = int(line_num_str)
                    metric_value = float(metric_str)

                    line_data[file_name][line_num]["metric_value"] += metric_value
                    line_data[file_name][line_num]["line_exec_count"] += 1
                    line_data[file_name][line_num]["kernels"].add(kernel)
                    entry_count += 1
                except (ValueError, IndexError):
                    continue

        print(f"  Loaded {entry_count} entries for {len(line_data)} files")
        return line_data

    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def analyze_all_kernels(self):
        """Scan all kernel launches - memory optimized: just counts, no storage"""

        print(f"\n{'=' * 80}")
        print(f"SCANNING KERNEL LAUNCHES")
        print(f"{'=' * 80}\n")

        total_ranges = self.report.num_ranges()
        total_launches = 0

        for range_idx in range(total_ranges):
            current_range = self.report.range_by_idx(range_idx)
            num_actions = current_range.num_actions()

            for action_idx in range(num_actions):
                action = current_range.action_by_idx(action_idx)
                kernel_name = action.name()

                self.kernel_name_counts[kernel_name] += 1
                total_launches += 1

                if self.verbose:
                    print(
                        f"  Launch {total_launches}: {kernel_name[:50]} (instance {self.kernel_name_counts[kernel_name]})"
                    )

        self.total_kernel_launches = total_launches

        print(f"Total kernel launches: {total_launches}")
        print(f"Unique kernel names: {len(self.kernel_name_counts)}")

        if self.kernel_name_counts:
            print(f"\nKernel launch counts:")
            for name, count in sorted(self.kernel_name_counts.items()):
                print(f"  {name[:60]}: {count} launches")

        return total_launches

    def extract_per_line_metrics(self):
        """Extract per-line metrics using correlation IDs - memory optimized with batch processing"""

        print(f"\n{'=' * 80}")
        print(f"EXTRACTING PER-LINE METRICS")
        print(f"Metric preference: {METRIC_PREFERENCES}")
        print(f"Filtering: Source repository files only")
        print(f"Batch size: {KERNEL_BATCH_SIZE} kernels, GC interval: {GC_INTERVAL}")
        print(f"{'=' * 80}\n")

        # Kernel patterns known to cause issues with source correlation
        SKIP_KERNEL_PATTERNS = [
            "optixLaunch",  # OptiX kernels don't have proper source correlation
            "cudaLaunchKernel",  # CUDA runtime internal
        ]

        skipped_kernels = 0
        failed_kernels = 0
        processed_kernels = 0

        # Reset kernel name counts for instance tracking during processing
        kernel_instance_counts = defaultdict(int)

        total_ranges = self.report.num_ranges()

        for range_idx in range(total_ranges):
            current_range = self.report.range_by_idx(range_idx)
            num_actions = current_range.num_actions()

            for action_idx in range(num_actions):
                action = current_range.action_by_idx(action_idx)
                kernel_name = action.name()

                kernel_instance_counts[kernel_name] += 1
                instance = kernel_instance_counts[kernel_name]
                launch_id = processed_kernels + skipped_kernels + failed_kernels

                # Skip known problematic kernels
                should_skip = any(pattern in kernel_name for pattern in SKIP_KERNEL_PATTERNS)
                if should_skip:
                    if self.verbose:
                        print(f"Skipping kernel {launch_id}: {kernel_name[:50]} (known problematic pattern)")
                    skipped_kernels += 1
                    continue

                try:
                    # Progress reporting
                    if launch_id % MEMORY_STATUS_INTERVAL == 0:
                        print(
                            f"[Memory] Processed {processed_kernels} kernels, "
                            f"buffer size: {len(self._batch_buffer)}, "
                            f"files found: {len(self.included_files)}"
                        )

                    # Extract metrics for this kernel
                    lines_mapped = self._extract_kernel_metrics(action, kernel_name, instance)

                    if self.verbose and lines_mapped > 0:
                        print(f"  Kernel {kernel_name[:50]}: {lines_mapped} lines")

                    processed_kernels += 1
                    self._kernels_since_flush += 1

                    # Batch flush to disk
                    if self._kernels_since_flush >= KERNEL_BATCH_SIZE:
                        self._flush_batch_to_disk()
                        self._evict_source_cache()

                    # Periodic garbage collection
                    self._maybe_gc()

                except (SystemError, MemoryError, OSError) as e:
                    # Critical error from NCU C extension
                    print(f"  ERROR: Critical failure processing kernel {kernel_name[:50]}: {e}")
                    failed_kernels += 1
                    # Try to recover memory
                    gc.collect()
                    continue
                except Exception as e:
                    print(f"  ERROR: Unexpected error processing kernel {kernel_name[:50]}: {e}")
                    failed_kernels += 1
                    continue

        # Final flush of any remaining data
        self._flush_batch_to_disk()
        gc.collect()

        if skipped_kernels > 0:
            print(f"\nSkipped {skipped_kernels} kernels (known problematic patterns)")
        if failed_kernels > 0:
            print(f"Failed to process {failed_kernels} kernels (errors encountered)")

        # Report filtering results
        print(f"\n{'=' * 80}")
        print(f"FILE FILTERING RESULTS")
        print(f"{'=' * 80}")
        print(f"Included files (source repository): {len(self.included_files)}")
        for f in sorted(self.included_files):
            print(f"  [INCLUDED] {f}")

        if self.excluded_files:
            print(f"\nExcluded files (CUDA system headers): {len(self.excluded_files)}")
            for f in sorted(self.excluded_files):
                print(f"  [EXCLUDED] {f}")

        # Load data from disk and calculate coverage
        print(f"\n{'=' * 80}")
        print(f"LOADING DATA FROM DISK")
        print(f"{'=' * 80}")
        self.line_data = self._load_line_data_from_disk()

        # Calculate coverage statistics
        self._calculate_coverage()

        total_lines = sum(len(lines) for lines in self.line_data.values()) if self.line_data else 0

        print(f"\n{'=' * 80}")
        print(f"EXTRACTION COMPLETE")
        print(f"Files with data: {len(self.line_data) if self.line_data else 0}")
        print(f"{'=' * 80}\n")

        return self.line_data

    def _extract_kernel_metrics(self, action, kernel_name, instance):
        """Extract metrics for a single kernel using correlation IDs

        Memory-optimized: writes to batch buffer instead of accumulating in memory.
        Uses kernel source cache to skip redundant source_info() calls for repeated kernels.
        """

        # Safety limit to prevent segfaults on massive kernels
        MAX_CORRELATIONS = 5_000_000

        try:
            # Use explicit metric if specified, otherwise auto-detect from preference list
            metric, metric_name = find_matching_metric(action, explicit_metric=self.target_metric)
            if not metric:
                return 0

            # Track what metric we're actually using (for reporting)
            if not hasattr(self, "_detected_metric"):
                self._detected_metric = metric_name
                if self.verbose:
                    print(f"  Auto-detected metric: {metric_name}")

            # Get correlation IDs metric
            corr_metric = metric.correlation_ids()
            if not hasattr(corr_metric, "num_instances"):
                return 0

            num_corr = corr_metric.num_instances()

            # Safety check: skip kernels with too many correlations to prevent segfaults
            if num_corr > MAX_CORRELATIONS:
                print(f"  Warning: Skipping kernel with {num_corr:,} correlations (limit: {MAX_CORRELATIONS:,})")
                return 0

            # Safety check: skip if num_corr is invalid
            if num_corr < 0:
                print(f"  Warning: Invalid correlation count {num_corr}, skipping kernel")
                return 0

            # Check if we have cached source mappings for this kernel type
            # Multiple instances of the same kernel have identical source correlations
            if kernel_name in self._kernel_source_cache:
                return self._process_cached_kernel(metric, kernel_name, instance)

            # First time seeing this kernel - process and cache source mappings
            lines_mapped = 0
            source_mappings = []  # List of (corr_idx, resolved_file, line_num) for repo files

            for i in range(num_corr):
                try:
                    # Get correlation ID
                    try:
                        corr_id = corr_metric.as_uint64(i)
                    except (SystemError, MemoryError, OSError) as e:
                        if self.verbose:
                            print(f"  Error: NCU API failed at correlation {i}: {e}")
                        break

                    # Map to source
                    try:
                        source_info = action.source_info(corr_id)
                    except (SystemError, MemoryError, OSError):
                        continue

                    if not source_info or not source_info.file_name():
                        continue

                    file_name = source_info.file_name()
                    line_num = source_info.line()

                    # Early extension filter - skip non-CUDA files before expensive path resolution
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext not in INCLUDED_EXTENSIONS:
                        continue

                    # Resolve symlinks/runfiles to get real source path (cached)
                    if file_name in self._path_cache:
                        file_name = self._path_cache[file_name]
                    else:
                        original_name = file_name
                        if os.path.exists(file_name):
                            try:
                                real_path = os.path.realpath(file_name)
                                _ensure_repo_dirs_loaded()
                                repo_root_str = str(_REPO_ROOT) + os.sep
                                if real_path.startswith(repo_root_str):
                                    file_name = real_path
                            except OSError:
                                pass
                        elif "runfiles/_main/" in file_name:
                            resolved = resolve_runfiles_path(file_name)
                            if resolved:
                                file_name = resolved
                        self._path_cache[original_name] = file_name

                    # Filter out CUDA system files (cached)
                    if is_system_file(file_name):
                        self.excluded_files.add(file_name)
                        continue

                    # This is a source repository file
                    self.included_files.add(file_name)

                    if line_num > 0:
                        # Cache this mapping for future kernel instances
                        source_mappings.append((i, file_name, line_num))

                        # Track this line has correlation ID (regardless of metric value)
                        self.corr_id_lines[file_name].add(line_num)

                        # Get metric value
                        try:
                            value = metric.as_uint64(i)
                        except (SystemError, MemoryError, OSError):
                            continue

                        # Only record metric data if value > 0
                        if value > 0:
                            self._add_line_entry(file_name, line_num, value, kernel_name, instance)
                            lines_mapped += 1

                except Exception:
                    continue

            # Cache source mappings for future instances of this kernel
            self._kernel_source_cache[kernel_name] = source_mappings

            return lines_mapped

        except (SystemError, MemoryError, OSError) as e:
            # Critical errors from C extension - always print
            print(f"  Error: NCU API critical failure for {kernel_name}: {e}")
            return 0
        except Exception as e:
            if self.verbose:
                print(f"  Error extracting metrics: {e}")
            return 0

    def _process_cached_kernel(self, metric, kernel_name, instance):
        """Process a kernel using cached source mappings - much faster than full extraction.

        Only fetches metric values for pre-mapped correlation indices.
        """
        source_mappings = self._kernel_source_cache[kernel_name]
        lines_mapped = 0

        for corr_idx, file_name, line_num in source_mappings:
            try:
                # Get metric value for this correlation index
                value = metric.as_uint64(corr_idx)

                # Only record metric data if value > 0
                if value > 0:
                    self._add_line_entry(file_name, line_num, value, kernel_name, instance)
                    lines_mapped += 1

            except (SystemError, MemoryError, OSError):
                continue

        return lines_mapped

    def _calculate_coverage(self):
        """Calculate code coverage for each file based on all executable lines"""

        print(f"\n{'=' * 80}")
        print(f"CALCULATING CODE COVERAGE (ALL EXECUTABLE LINES)")
        print(f"{'=' * 80}\n")

        # Get all unique files from both corr_id_lines and line_data
        all_files = set(self.corr_id_lines.keys()) | set(self.line_data.keys())

        for file_name in all_files:
            source_lines, _ = self._load_source_file(file_name)

            if not source_lines:
                continue

            # Count ALL executable lines in the file
            executable_line_nums = get_executable_line_numbers(source_lines, file_ext=_get_file_ext(file_name))
            num_executable = len(executable_line_nums)

            # Get lines with correlation IDs for this file
            corr_line_nums = self.corr_id_lines.get(file_name, set())

            # Get covered lines (lines with metrics)
            covered_lines = set(self.line_data[file_name].keys()) if file_name in self.line_data else set()

            # Calculate coverage: covered lines / all executable lines
            coverage_pct = (len(covered_lines) / num_executable * 100) if num_executable > 0 else 0

            self.file_coverage[file_name] = {
                "total_lines": len(source_lines),
                "executable_lines": num_executable,
                "corr_id_lines": len(corr_line_nums),
                "covered_lines": len(covered_lines),
                "coverage_pct": coverage_pct,
                "executable_line_nums": executable_line_nums,
            }

            print(
                f"  {os.path.basename(file_name)}: {len(covered_lines)}/{num_executable} executable lines ({coverage_pct:.1f}%)"
            )

    def _load_source_file(self, file_path):
        """Load source file content with LRU cache management (O(1) with OrderedDict)"""

        if file_path in self.source_cache:
            # Move to end of LRU order (most recently used) - O(1)
            self.source_cache.move_to_end(file_path)
            return self.source_cache[file_path]

        found_path = None

        # Canonicalize to repo-relative path (handles sandbox/slangtorch_cache/runfiles paths)
        canonical = canonicalize_path(file_path)

        # Try original path first (for cases where it's already resolved)
        if os.path.exists(file_path):
            found_path = file_path
        else:
            for root in self.source_roots:
                # Try canonical path directly under source root
                candidate = os.path.join(root, canonical)
                if os.path.exists(candidate):
                    found_path = candidate
                    break

                # Try by basename as fallback
                candidate = os.path.join(root, os.path.basename(file_path))
                if os.path.exists(candidate):
                    found_path = candidate
                    break

                # Try canonical path parts (for partial matches)
                parts = Path(canonical).parts
                for i in range(len(parts)):
                    candidate = os.path.join(root, *parts[i:])
                    if os.path.exists(candidate):
                        found_path = candidate
                        break
                if found_path:
                    break

        if found_path:
            try:
                with open(found_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.readlines()

                    # LRU cache management: evict old entries before adding
                    self._evict_source_cache()

                    # Add to OrderedDict (automatically at end = most recently used)
                    self.source_cache[file_path] = (content, found_path)
                    return content, found_path
            except (IOError, OSError, PermissionError) as e:
                if self.verbose:
                    print(f"  Warning: Could not read source file {found_path}: {e}")

        return None, None

    def export_lcov(self, output_path: str):
        """Export coverage in LCOV format for bazel coverage integration.

        LCOV format specification:
        - TN:<test_name>    - Test name
        - SF:<source_file>  - Source file path (absolute or relative)
        - DA:<line>,<count> - Line data: line number, execution count
        - LF:<count>        - Lines found (total executable lines)
        - LH:<count>        - Lines hit (covered lines)
        - end_of_record     - End of file record

        Args:
            output_path: Path to write the LCOV format file
        """
        print(f"Exporting LCOV format to {output_path}...")

        # Ensure data is loaded
        if self.line_data is None:
            self.line_data = self._load_line_data_from_disk()

        with open(output_path, "w") as f:
            f.write("TN:cuda_device_coverage\n")

            for file_name in sorted(self.file_coverage.keys()):
                cov = self.file_coverage[file_name]
                line_data = self.line_data.get(file_name, {})

                # Use canonical repo-relative path
                canonical_path = canonicalize_path(file_name)
                f.write(f"SF:{canonical_path}\n")

                # Write line data for all executable lines
                for line_num in sorted(cov["executable_line_nums"]):
                    # 1 if executed, 0 if not
                    hit_count = 1 if line_num in line_data else 0
                    f.write(f"DA:{line_num},{hit_count}\n")

                f.write(f"LF:{cov['executable_lines']}\n")
                f.write(f"LH:{cov['covered_lines']}\n")
                f.write("end_of_record\n")

        print(f"Success: LCOV export complete ({len(self.file_coverage)} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract CUDA code coverage from NCU reports (source files only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_ncu_coverage.py report.ncu-rep

  python infer_ncu_coverage.py report.ncu-rep --source /path/to/sources

Coverage Filtering:
  This tool analyzes ONLY your source repository files, automatically excluding:
  - CUDA runtime headers (/usr/local/cuda/*)
  - System libraries (/usr/include/, /usr/lib/)
  - CUDA library headers (thrust, cub, cooperative_groups, sm_* intrinsics, etc.)
  
  Coverage statistics reflect only YOUR code, not CUDA system code.
  
Coverage Measurement:
  Coverage is calculated based on ALL executable lines (CPU + GPU code).
  Output is in LCOV format - use 'genhtml' to generate HTML reports:
    genhtml coverage.lcov -o html_output/

Memory Optimization:
  This tool uses disk-backed storage to handle large NCU reports without OOM.
  - Batch processing: flushes data to disk every {BATCH_SIZE} kernels
  - LRU source cache: limits memory for source file caching
        """.format(BATCH_SIZE=KERNEL_BATCH_SIZE),
    )

    parser.add_argument("report", help="Path to .ncu-rep file")
    parser.add_argument(
        "-m", "--metric", default=None, help="Metric to extract (default: auto-detect from preference order)"
    )
    parser.add_argument("-s", "--source", action="append", help="Source directory (can specify multiple)")
    parser.add_argument("--lcov", required=True, help="Output LCOV format file (use genhtml to generate HTML)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    source_roots = args.source if args.source else ["."]
    analyzer = None

    try:
        print(f"CUDA Code Coverage Analyzer")
        print(f"Report: {args.report}")

        analyzer = NCUCoverageAnalyzer(args.report, source_roots, args.metric, args.verbose)
        analyzer.analyze_all_kernels()
        analyzer.extract_per_line_metrics()

        if not analyzer.line_data:
            print("\nWarning: No data extracted from source files!")

        # Export LCOV format
        analyzer.export_lcov(args.lcov)

        if analyzer.file_coverage:
            print(f"\n{'=' * 80}")
            print(f"COVERAGE SUMMARY BY FILE (Source Repository Only)")
            print(f"{'=' * 80}")
            for file_name in sorted(analyzer.file_coverage.keys()):
                cov = analyzer.file_coverage[file_name]
                print(
                    f"  {os.path.basename(file_name)}: {cov['coverage_pct']:.1f}% coverage ({cov['covered_lines']}/{cov['executable_lines']} executable lines)"
                )
            print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temporary files
        if analyzer:
            analyzer.cleanup()


if __name__ == "__main__":
    main()
