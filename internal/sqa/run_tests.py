# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import fcntl
import fnmatch
import os
import pty
import re
import select
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import click
import yaml

from python.runfiles import runfiles

from internal.sqa.test_cases.artifacts import Artifacts, ArtifactsConfig, sqa_test_artifacts
from internal.sqa.test_cases.commands import Command, CommandGroup
from internal.sqa.test_cases.datasets import Dataset, DatasetConfig, sqa_test_datasets
from internal.sqa.test_cases.resource import Resource
from internal.sqa.test_cases.test_cases import TestCaseConfig, generate_test_cases


def get_project_base_path() -> Path:
    return Path.cwd()


def process_output_line(
    line: str, prefix: str, stdout_lines: List[str], stderr_lines: List[str], is_stderr: bool = False
) -> None:
    """Process and print a single line of output from a subprocess, and record it for further use."""
    line = line.rstrip("\n")
    output_file = sys.stderr if is_stderr else sys.stdout

    print(f"{prefix}{line}", file=output_file)

    if is_stderr:
        stderr_lines.append(line)
    else:
        stdout_lines.append(line)
    output_file.flush()


def stream_output(proc: subprocess.Popen[str], prefix: str = "") -> Tuple[str, str]:
    """Stream process output in real-time while capturing it."""
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    # Read stdout and stderr in real-time
    while True:
        # Check if process is still running
        if proc.poll() is not None:
            break

        # Use select to check for available output (Unix-only)
        try:
            ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.1)

            for stream in ready:
                if stream == proc.stdout:
                    line = stream.readline()
                    if line:
                        process_output_line(line, prefix, stdout_lines, stderr_lines, is_stderr=False)
                elif stream == proc.stderr:
                    line = stream.readline()
                    if line:
                        process_output_line(line, prefix, stdout_lines, stderr_lines, is_stderr=True)
        except (OSError, select.error):
            # Fallback for systems without select
            break

    # Read any remaining output
    remaining_stdout, remaining_stderr = proc.communicate()
    if remaining_stdout:
        for line in remaining_stdout.splitlines():
            if line.strip():
                process_output_line(line, prefix, stdout_lines, stderr_lines, is_stderr=False)
    if remaining_stderr:
        for line in remaining_stderr.splitlines():
            if line.strip():
                process_output_line(line, prefix, stdout_lines, stderr_lines, is_stderr=True)

    return "\n".join(stdout_lines), "\n".join(stderr_lines)


def substitute_command_arguments(cmd: Command, substitutions: Dict[str, Any]) -> List[str]:
    """Handle argument substitution logic for commands."""
    substituted_args = []
    for arg in cmd.args:
        # Special handling for EXTRA_PARAMS - split into individual arguments
        if arg.strip() == "$(EXTRA_PARAMS)":
            extra_params = substitutions.get("EXTRA_PARAMS", "").strip()
            if extra_params:
                # If extra_params is a single flag (e.g., --no-obfuscated), pass as a single argument
                if extra_params.startswith("--") and " " not in extra_params:
                    substituted_args.append(extra_params)
                else:
                    try:
                        extra_args = shlex.split(extra_params)
                        substituted_args.extend(extra_args)
                    except ValueError:
                        # If shlex fails, split on spaces (fallback)
                        extra_args = extra_params.split()
                        substituted_args.extend(extra_args)
            # Skip the placeholder since it's been replaced
            continue

        # Regular substitution for other placeholders
        substituted_arg = arg
        for key, value in substitutions.items():
            if key == "EXTRA_PARAMS":
                continue  # Already handled above
            placeholder = f"$({key})"
            if placeholder in substituted_arg:
                if isinstance(value, list):
                    value_str = ",".join(str(v) for v in value)
                else:
                    value_str = str(value)
                substituted_arg = substituted_arg.replace(placeholder, value_str)

        # Quote arguments that contain spaces or are empty
        if substituted_arg and (" " in substituted_arg or substituted_arg == ""):
            substituted_arg = f'"{substituted_arg}"'

        substituted_args.append(substituted_arg)

    return substituted_args


def execute_with_pty(full_cmd: List[str]) -> subprocess.CompletedProcess[str]:
    """Execute command using PTY for interactive support."""
    print(f"[EXEC] {' '.join(full_cmd)}")

    # Use PTY for all commands to support interactive flags like Docker's -it
    # Create a pseudo-terminal pair
    parent_fd, child_fd = pty.openpty()

    try:
        # Start the process with the child fd as stdin/stdout/stderr
        proc = subprocess.Popen(
            full_cmd,
            stdin=child_fd,
            stdout=child_fd,
            stderr=child_fd,
        )

        # Close child fd in parent process
        os.close(child_fd)

        # Set parent fd to non-blocking mode
        fcntl.fcntl(parent_fd, fcntl.F_SETFL, os.O_NONBLOCK)

        output_data = []

        try:
            while True:
                # Check if process is still running
                if proc.poll() is not None:
                    break

                # Use select to check for available data
                ready, _, _ = select.select([parent_fd], [], [], 0.1)

                if parent_fd in ready:
                    try:
                        data = os.read(parent_fd, 1024).decode("utf-8", errors="replace")
                        if data:
                            # Print in real-time
                            print(data, end="", flush=True)
                            output_data.append(data)
                    except OSError:
                        # Parent fd closed or no more data
                        break

        except KeyboardInterrupt:
            print("\nReceived CTRL+C, forwarding to PTY...")
            try:
                # Forward CTRL+C character to the PTY (like pressing CTRL+C in terminal)
                ctrl_c_char = b"\x03"  # ASCII ETX (End of Text) character for CTRL+C
                os.write(parent_fd, ctrl_c_char)
                # Let the process handle it naturally
                proc.wait()
                print("Process terminated by CTRL+C")
            except (OSError, ProcessLookupError):
                # PTY closed or process already gone
                pass
            raise

        # Wait for process to complete
        proc.wait()

        # Read any remaining data
        try:
            while True:
                data = os.read(parent_fd, 1024).decode("utf-8", errors="replace")
                if not data:
                    break
                print(data, end="", flush=True)
                output_data.append(data)
        except OSError:
            pass

        # PTY combines stdout and stderr, so return output as stdout
        stdout = "".join(output_data)
        stderr = ""

    finally:
        # Clean up
        try:
            os.close(parent_fd)
        except OSError:
            pass

    return subprocess.CompletedProcess(full_cmd, proc.returncode, stdout, stderr)


def execute_command(
    cmd: Command,
    substitutions: Dict[str, Any],
    project_base_path: str,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute a single command with substitutions."""
    script_path = Path(project_base_path) / cmd.script

    # Substitute arguments using helper function
    substituted_args = substitute_command_arguments(cmd, substitutions)

    # Build full command
    full_cmd = [str(script_path)] + substituted_args

    # In dry-run mode, return a mock successful result
    if dry_run:
        print(f"[DRY-RUN] Would execute: {' '.join(full_cmd)}")
        return subprocess.CompletedProcess(full_cmd, 0, f"[DRY-RUN] Would execute: {' '.join(full_cmd)}", "")

    # Execute using PTY for interactive support
    return execute_with_pty(full_cmd)


def drain_pty_to_file(parent_fd: int, output_file: Any, stop_event: threading.Event) -> None:
    """Continuously drain PTY output to file in a background thread."""
    while not stop_event.is_set():
        try:
            ready, _, _ = select.select([parent_fd], [], [], 0.1)
            if parent_fd in ready:
                data = os.read(parent_fd, 4096)
                if data:
                    output_file.write(data)
                    output_file.flush()
        except OSError:
            # PTY closed or other OS-level issue, normal during process termination
            break


def execute_command_group(
    cmd_group: CommandGroup,
    substitutions: Dict[str, Any],
    project_base_path: str,
    dry_run: bool = False,
) -> List[subprocess.CompletedProcess[str]]:
    """Execute a group of commands with proper background/timing handling."""
    results = []
    background_processes: List[
        Tuple[Any, Any, int, threading.Event, threading.Thread]
    ] = []  # (process, output_file, parent_fd, stop_event, thread)

    for i, cmd in enumerate(cmd_group.commands):
        run_in_background = cmd_group.run_in_background[i] if i < len(cmd_group.run_in_background) else False
        wait_before = cmd_group.wait_before_commands_s[i] if i < len(cmd_group.wait_before_commands_s) else 0

        # Wait before starting this command (ex. to leave time for GRPC server to start before running the client)
        if wait_before > 0:
            print(f"[WAIT] Waiting {wait_before}s before starting command {i + 1}")
            if not dry_run:  # Skip actual waiting in dry-run mode
                time.sleep(wait_before)

        # For foreground commands, use execute_command
        if not run_in_background:
            result = execute_command(cmd, substitutions, project_base_path, dry_run)
            results.append(result)
            if result.returncode != 0:
                # Stop on first failure
                break
            continue

        # For background commands, prepare them manually
        script_path = Path(project_base_path) / cmd.script
        substituted_args = substitute_command_arguments(cmd, substitutions)
        full_cmd = [str(script_path)] + substituted_args
        print(f"[EXEC] {' '.join(full_cmd)} {'(background)' if run_in_background else ''}")

        # In dry-run mode, create a mock result for background commands
        if dry_run:
            print(f"[DRY-RUN] Would execute background: {' '.join(full_cmd)}")
            mock_result = subprocess.CompletedProcess(
                full_cmd, 0, f"[DRY-RUN] Would execute background: {' '.join(full_cmd)}", ""
            )
            results.append(mock_result)
            continue

        # Start background process with PTY and output redirected to temporary file
        # PTY prevents garbled output from Docker, temp file prevents pipe buffer overflow
        output_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False, suffix=f"_bg{i}.log")

        # Create a pseudo-terminal pair
        parent_fd, child_fd = pty.openpty()

        proc = subprocess.Popen(
            full_cmd,
            stdin=child_fd,
            stdout=child_fd,
            stderr=child_fd,
        )

        # Close child fd in parent process
        os.close(child_fd)

        # Set parent fd to non-blocking mode
        fcntl.fcntl(parent_fd, fcntl.F_SETFL, os.O_NONBLOCK)

        # Start thread to continuously drain PTY output to file
        stop_event = threading.Event()
        drain_thread = threading.Thread(
            target=drain_pty_to_file, args=(parent_fd, output_file, stop_event), daemon=True
        )
        drain_thread.start()

        background_processes.append((proc, output_file, parent_fd, stop_event, drain_thread))

    # Terminate background processes after foreground commands complete (skip in dry-run mode)
    if not dry_run:
        for i, (proc, output_file, parent_fd, stop_event, drain_thread) in enumerate(background_processes):
            had_to_kill = False

            if proc.poll() is None:  # Process is still running
                print(f"[BACKGROUND] Background process {i + 1} still running, waiting 5s for graceful exit...")

                # Give 5s grace period for the process to exit on its own
                try:
                    proc.wait(timeout=5)
                    print(f"[BACKGROUND] Background process {i + 1} exited during grace period")
                except subprocess.TimeoutExpired:
                    # Process still running after grace period - need to terminate it
                    print(f"[BACKGROUND] Background process {i + 1} did not exit, sending CTRL+C...")
                    had_to_kill = True

                    # Try graceful termination by sending CTRL+C to PTY
                    # This allows Docker to clean up containers properly
                    try:
                        ctrl_c_char = b"\x03"  # ASCII ETX (End of Text) character for CTRL+C
                        os.write(parent_fd, ctrl_c_char)
                        # Give it a moment to terminate gracefully
                        proc.wait(timeout=10)
                    except (subprocess.TimeoutExpired, OSError):
                        print(f"[BACKGROUND] Force killing process {i + 1}...")
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                        proc.wait()
            else:
                print(f"[BACKGROUND] Background process {i + 1} has already exited")

            # Stop the draining thread
            stop_event.set()
            drain_thread.join(timeout=1)

            # Close PTY and output file
            try:
                os.close(parent_fd)
            except OSError:
                pass
            output_file.close()

            # Read the output file
            with open(output_file.name, "rb") as f:
                output = f.read().decode("utf-8", errors="replace")

            # Clean up temp file
            try:
                os.unlink(output_file.name)
            except OSError:
                pass

            # Print captured output
            if output.strip():
                for line in output.strip().split("\n"):
                    print(f"[BG{i + 1}] {line}")

            # Treat it as an error if we had to kill the process
            # Otherwise, use the actual return code
            if had_to_kill:
                print(f"[ERROR] Background process {i + 1} had to be killed - expected graceful exit")
                returncode = 1  # Error: process didn't exit gracefully
            else:
                returncode = proc.returncode or 0
            result = subprocess.CompletedProcess(proc.args, returncode, output, "")
            results.append(result)

    return results


# Generic resource handling functions (shared between datasets and artifacts)


def copy_bazel_runfiles_resource(
    resource: Resource,
    dry_run: bool = False,
) -> bool:
    """Generic function to copy Bazel runfiles resources to local path.

    Args:
        resource: The resource object
        dry_run: If True, only print what would be done
    """
    if resource.bazel_target is None:
        print(f"[{resource.resource_type}] ERROR: {resource.name} has no bazel_target defined")
        return False

    # Extract the runfiles path from bazel_target
    runfiles_path = resource.get_runfiles_path()
    if not runfiles_path:
        print(f"[{resource.resource_type}] ERROR: {resource.name} has no file path documented in bazel_target")
        return False

    print(f"[{resource.resource_type}] {resource.name} uses Bazel runfiles")
    print(f"[{resource.resource_type}] Bazel target: {resource.bazel_target['target']}")
    print(f"[{resource.resource_type}] Runfiles path: {runfiles_path}")
    print(f"[{resource.resource_type}] Target location: {resource.local_path}")

    try:
        # Create runfiles instance and get the actual location
        r = runfiles.Create()
        if not r:
            print(f"[{resource.resource_type}] ERROR: Failed to create runfiles instance for {resource.name}")
            return False

        # Resolve the runfiles path
        resolved_path = r.Rlocation(runfiles_path)
        if not resolved_path:
            print(
                f"[{resource.resource_type}] ERROR: Could not find runfiles path '{runfiles_path}' for {resource.name}"
            )
            return False

        resolved_file = Path(resolved_path)

        # Get actual path using the resource's method
        actual_path = resource.get_actual_path_from_runfiles(resolved_file)

        if not actual_path.exists():
            print(f"[{resource.resource_type}] ERROR: Runfiles path does not exist: {actual_path}")
            return False

        print(f"[{resource.resource_type}] Found runfiles location: {actual_path}")

        # Check for dry run before making any file system changes
        if dry_run:
            print(f"[DRY-RUN] Would create parent directory: {resource.local_path.parent}")
            print(f"[DRY-RUN] Would copy files from: {actual_path} -> {resource.local_path}")
            return True

        # Create parent directory if it doesn't exist
        resource.local_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing directory if it exists
        if resource.local_path.exists():
            print(f"[{resource.resource_type}] Removing existing directory: {resource.local_path}")
            shutil.rmtree(resource.local_path)

        # Copy the entire directory tree, following symlinks to get actual files
        print(f"[{resource.resource_type}] Copying files: {actual_path} -> {resource.local_path}")
        shutil.copytree(actual_path, resource.local_path, symlinks=False, dirs_exist_ok=True)

        # Verify the copy was successful
        if resource.check_exists():
            print(f"[{resource.resource_type}] Successfully copied {resource.name} from Bazel runfiles")
            return True
        else:
            print(f"[{resource.resource_type}] ERROR: Files copied but required files not found for {resource.name}")
            return False

    except Exception as e:
        print(f"[{resource.resource_type}] ERROR: Failed to copy {resource.name}: {e}")
        return False


def download_resource_with_rclone(
    resource: Resource,
    dry_run: bool = False,
) -> bool:
    """Generic function to download or update a resource using rclone.

    Args:
        resource: The resource object
        dry_run: If True, only print what would be done
    """
    if resource.remote_path is None:
        print(f"[{resource.resource_type}] ERROR: {resource.name} has no remote_path defined")
        return False

    # Check current status
    if resource.check_exists():
        print(f"[{resource.resource_type}] {resource.name} found locally, updating...")
    else:
        print(f"[{resource.resource_type}] {resource.name} not found locally, downloading...")

    print(f"[{resource.resource_type}] Downloading {resource.name} from {resource.remote_path}")
    print(f"[{resource.resource_type}] Target location: {resource.local_path}")

    # Create parent directory if it doesn't exist
    if not dry_run:
        resource.local_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[DRY-RUN] Would create directory: {resource.local_path.parent}")

    # Build rclone command
    cmd = ["rclone", "sync", "-P", resource.remote_path, str(resource.local_path)]

    if isinstance(resource, Artifacts) and resource.remote_include_patterns:
        for pattern in resource.remote_include_patterns:
            cmd.extend(["--include", pattern])

    try:
        print(f"[{resource.resource_type}] Running: {' '.join(cmd)}")

        if dry_run:
            print(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
            print(f"[DRY-RUN] Would download {resource.name} to {resource.local_path}")
            return True

        # Run with real-time output for progress
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output in real-time
        if proc.stdout:
            for line in proc.stdout:
                if line.strip():
                    print(f"[{resource.resource_type}] {line.strip()}")

        proc.wait()

        if proc.returncode == 0:
            print(f"[{resource.resource_type}] Successfully downloaded {resource.name}")

            # Verify the download worked
            if resource.check_exists():
                return True
            else:
                print(
                    f"[{resource.resource_type}] ERROR: Download completed but required files not found for {resource.name}"
                )
                # List what was actually downloaded for debugging
                try:
                    if resource.local_path.exists():
                        files = list(resource.local_path.rglob("*"))[:10]
                        print(f"[{resource.resource_type}] Found files: {[f.name for f in files]}")
                except:
                    pass
                return False
        else:
            print(
                f"[{resource.resource_type}] ERROR: Failed to download {resource.name} (exit code: {proc.returncode})"
            )
            return False

    except subprocess.TimeoutExpired:
        print(f"[{resource.resource_type}] ERROR: Download timed out for {resource.name}")
        return False
    except Exception as e:
        print(f"[{resource.resource_type}] ERROR: Exception during download of {resource.name}: {e}")
        return False


def ensure_resource_available(resource: Resource, skip_download: bool = False, dry_run: bool = False) -> bool:
    """Ensure a resource is available locally and up-to-date, downloading/updating if necessary."""
    if skip_download:
        if resource.check_exists():
            print(
                f"[{resource.resource_type}] {resource.name} found locally, skipping update (--skip-resource-download)"
            )
            return True
        else:
            print(
                f"[{resource.resource_type}] {resource.name} not found locally, skipping download (--skip-resource-download)"
            )
            return False

    if resource.bazel_target is not None:
        return copy_bazel_runfiles_resource(resource, dry_run)

    if resource.remote_path is not None:
        return download_resource_with_rclone(resource, dry_run)

    # No bazel target or remote path
    if resource.check_exists():
        print(f"[{resource.resource_type}] {resource.name} found locally")
        return True
    else:
        print(f"[{resource.resource_type}] ERROR: {resource.name} not found and no download method available")
        return False


def print_resources(resources: Mapping[str, Resource]) -> None:
    """Print resource information for datasets or artifacts."""
    if not resources:
        return

    # Get resource type from first item
    first_resource = next(iter(resources.values()))
    resource_type = type(first_resource).__name__
    resource_type_plural = resource_type if resource_type.endswith("s") else f"{resource_type}s"

    print(f"\n{resource_type_plural} for these tests:\n")
    for key, resource in resources.items():
        print(f"  {resource_type} {resource.name}")
        print(f"    Local path:  {resource.local_path}")
        if resource.bazel_target is not None:
            print(f"    Bazel target: {resource.bazel_target['target']}")
            # Print resource-specific paths
            for label, value in resource.get_bazel_target_display():
                print(f"    {label}: {value}")
        if resource.remote_path is not None:
            print(f"    Remote path: {resource.remote_path}")
    print()


def print_available_test_cases(all_test_cases: List[Any]) -> None:
    print("\nAvailable test cases:\n")
    for test_case in all_test_cases:
        print(f"  {test_case.name}")
    print()


def initialize_test_environment(
    dry_run: bool, force_obfuscation: Optional[str], grpc_port_base: int
) -> Tuple[Path, TestCaseConfig]:
    """Initialize test configuration and environment."""
    if dry_run:
        print("=" * 60)
        print("DRY-RUN MODE: Commands will be displayed but not executed")
        print("=" * 60)

    project_base_path = get_project_base_path()
    # Use Bazel's undeclared outputs dir for results when available, as it allows exposing outputs out of the sandbox
    # for "bazel test" execution.
    test_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    results_base = Path(test_outputs_dir) if test_outputs_dir else project_base_path / "results"
    dataset_base = project_base_path / "dataset"
    artifacts_base = project_base_path / "artifacts"
    print("Project path:", project_base_path)
    print("Results path:", results_base)
    print("Dataset path:", dataset_base)

    # Convert force_obfuscation to the expected format
    force_obfuscation_value = None
    if force_obfuscation is not None:
        force_obfuscation_value = "yes" if force_obfuscation == "on" else "no"

    # Initialize test configuration
    test_config = TestCaseConfig(
        results_base=str(results_base),
        dataset_config=DatasetConfig(local_path=dataset_base),
        artifacts_config=ArtifactsConfig(local_path=artifacts_base),
        grpc_port_base=grpc_port_base,
        force_obfuscation=force_obfuscation_value,
    )

    return project_base_path, test_config


def load_test_cases_and_resources(
    test_config: TestCaseConfig,
) -> Tuple[Optional[List[Any]], Optional[Dict[str, Dataset]], Optional[Dict[str, Artifacts]]]:
    """Load and display available test cases, datasets, and artifacts."""
    # Load test cases
    try:
        all_test_cases = generate_test_cases(test_config)

        # Display force obfuscation info if applied
        if test_config.force_obfuscation is not None:
            print(f"\nForce obfuscation applied: --obfuscation-{test_config.force_obfuscation}")

        # Display the final list of test cases
        print_available_test_cases(all_test_cases)

    except Exception as e:
        print(f"[FATAL] Failed to generate test cases: {e}")
        return None, None, None

    # Get all available datasets and configs
    try:
        datasets = sqa_test_datasets(test_config.dataset_config)
        print_resources(datasets)
    except Exception as e:
        print(f"[FATAL] Failed to load datasets: {e}")
        return None, None, None

    # Get all available artifacts
    try:
        artifacts = sqa_test_artifacts(test_config.artifacts_config)
        print_resources(artifacts)
    except Exception as e:
        print(f"[FATAL] Failed to load artifacts: {e}")
        return None, None, None

    return all_test_cases, datasets, artifacts


def collect_required_datasets(test_cases: List[Any]) -> Set[str]:
    """Collect required datasets from test cases."""
    required_datasets = set()

    for test_case in test_cases:
        # Skip test cases that don't have a dataset
        if test_case.dataset is not None:
            required_datasets.add(test_case.dataset.name)

    return required_datasets


def ensure_resources_available_batch(
    required_resources: Set[str],
    resources: Mapping[str, Resource],
    skip_download: bool,
    dry_run: bool,
) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
    """Ensure all required resources are available, downloading if necessary."""
    if not required_resources:
        return set(), set()

    # Get resource type from first item in the dictionary
    resource_type = type(next(iter(resources.values()))).__name__.upper() if resources else "RESOURCE"
    resource_type_lower = resource_type.lower()
    # Add 's' for pluralization only if it doesn't already end in 's'
    resource_type_plural = resource_type_lower if resource_type_lower.endswith("s") else f"{resource_type_lower}s"

    print(f"[{resource_type}] Checking {len(required_resources)} required {resource_type_plural}...")
    unavailable_resources = set()
    failed_downloads = set()

    for resource_name in required_resources:
        if resource_name not in resources:
            print(f"[{resource_type}] ERROR: Unknown resource '{resource_name}'")
            failed_downloads.add(resource_name)
            continue

        resource = resources[resource_name]
        print(f"[{resource_type}] Checking {resource}")
        if not ensure_resource_available(resource, skip_download=skip_download, dry_run=dry_run):
            if skip_download:
                unavailable_resources.add(resource_name)
            else:
                failed_downloads.add(resource_name)

    if failed_downloads:
        print(f"[FATAL] {len(failed_downloads)} {resource_type_plural} failed to download: {failed_downloads}")
        return None, None

    if unavailable_resources:
        print(
            f"[{resource_type}] WARNING: {len(unavailable_resources)} {resource_type_plural} are unavailable: {unavailable_resources}"
        )
        print(f"[{resource_type}] Tests requiring these {resource_type_plural} will be skipped")

    return unavailable_resources, failed_downloads


def collect_required_artifacts(test_cases: List[Any]) -> Set[str]:
    """Collect all unique artifact names required by the test cases."""
    required_artifacts = set()
    for test_case in test_cases:
        if (
            hasattr(test_case, "artifact_source")
            and test_case.artifact_source
            and test_case.artifact_source != "train_val"
        ):
            required_artifacts.add(test_case.artifact_source)

    return required_artifacts


def prepare_test_output_directory(output_dir: str) -> bool:
    """Prepare and clean the test output directory."""
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"[SETUP] Test output directory exists, cleaning: {output_dir}")
        try:
            shutil.rmtree(output_path)
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
            print(f"[SETUP] ERROR: Cannot clean output directory {output_path}")
            return False
    output_path.mkdir(parents=True, exist_ok=True)
    return True


def resolve_runfiles_executable_path(test_case: Any) -> Optional[str]:
    """Resolve the runfiles executable path using Rlocation.

    Args:
        test_case: Test case with commands containing bazel_executable information

    Returns:
        str: Resolved filesystem path to the executable, or None if resolution fails
    """
    if not hasattr(test_case, "commands") or not test_case.commands:
        print(f"[ERROR] Test case {test_case.name} does not have commands")
        return None

    # Get bazel_executable from the first command
    # Note: All commands in a test case should use the same executable (validated in sync_test_plan.py)
    first_command = test_case.commands[0]
    if hasattr(first_command, "commands") and first_command.commands:
        # It's a CommandGroup, get the first command from the group
        bazel_executable = first_command.commands[0].bazel_executable
    else:
        # It's a regular Command
        bazel_executable = first_command.bazel_executable

    if not bazel_executable:
        print(f"[ERROR] Command in test case {test_case.name} does not have bazel_executable information")
        return None

    runfiles_path = bazel_executable.get("path")
    if not runfiles_path:
        print(f"[ERROR] No executable path found in 'bazel_executable' for test case {test_case.name}")
        return None

    r = runfiles.Create()
    if not r:
        print("[ERROR] Could not create runfiles instance")
        return None

    resolved_executable_path = r.Rlocation(runfiles_path)
    if not resolved_executable_path:
        print(f"[ERROR] Could not resolve runfiles path: {runfiles_path}")
        return None

    print(f"[RUNFILES] Using executable: {resolved_executable_path}")
    return resolved_executable_path


def resolve_gif_tool_path() -> Optional[str]:
    """Resolve the GIF comparison creation tool from Bazel runfiles.

    Returns:
        str: Resolved filesystem path to the GIF creation tool, or None if resolution fails.
    """
    r = runfiles.Create()
    if not r:
        print("[ERROR] Could not create runfiles instance")
        return None

    # The runfiles path for the create_comparison_gif tool
    gif_tool_runfiles_path = "_main/internal/sqa/scripts/create_comparison_gif"
    resolved_path = r.Rlocation(gif_tool_runfiles_path)

    if not resolved_path or not Path(resolved_path).exists():
        print(
            f"[ERROR] Could not resolve GIF creation tool from runfiles: {gif_tool_runfiles_path}. "
            "Ensure //internal/sqa/scripts:create_comparison_gif is available as a data dependency."
        )
        return None

    print(f"[RUNFILES] Resolved GIF creation tool: {resolved_path}")
    return resolved_path


def build_substitutions_dict(
    tag: Optional[str],
    suffix: Optional[str],
    executable_path: Optional[str],
    gif_tool_path: str,
    output_dir: str,
    extra_params: str,
) -> Dict[str, Any]:
    """Build the substitutions dictionary for command execution."""
    # Handle both Docker tag and runfiles through EXTRA_PARAMS
    if executable_path:
        # In runfiles mode, add --runfiles to EXTRA_PARAMS
        new_param = f"--runfiles {executable_path}"
    else:
        # In Docker mode, add --tag to EXTRA_PARAMS
        new_param = f"--tag {tag}"

        # Add --suffix if provided (only valid with --tag)
        if suffix:
            new_param += f" --suffix {suffix}"

    # Combine with existing extra_params
    combined_extra_params = f"{extra_params} {new_param}".strip()

    return {
        "NRE_OUTPUT_DIR": output_dir,  # For render tests
        "GIF_TOOL": gif_tool_path,
        "EXTRA_PARAMS": combined_extra_params,
    }


def execute_test_commands(
    matching_test_case: Any, substitutions: Dict[str, Any], project_base_path: str, dry_run: bool
) -> bool:
    """Execute all commands for a test case."""
    test_success = True

    for idx, cmd_or_group in enumerate(matching_test_case.commands):
        # If this is the first command and it's nre_image_trainval, run as usual
        if idx == 0 and isinstance(cmd_or_group, Command) and "nre_image_trainval" in cmd_or_group.script:
            result = execute_command(
                cmd_or_group,
                substitutions,
                str(project_base_path),
                dry_run,
            )
            if result.returncode != 0:
                print(f"[FAIL] Command failed with code {result.returncode}")
                test_success = False
                break

            # After successful nre_image_trainval, find the random subdir
            if dry_run:
                # Use fake directory for dry-run mode
                nre_output_dir_for_grpc = str(Path(matching_test_case.results_dir) / "fake_random_subdir")
                print(f"[DRY-RUN] Using fake subdirectory: {nre_output_dir_for_grpc}")
            else:
                output_path = Path(matching_test_case.results_dir)
                random_dirs = [d for d in output_path.iterdir() if d.is_dir()]
                if not random_dirs:
                    print(
                        f"[FAIL] No random subdirectory found after nre_image_trainval in {matching_test_case.results_dir}"
                    )
                    test_success = False
                    break
                # Pick the most recently created subdir
                random_dirs.sort(key=lambda d: d.stat().st_ctime, reverse=True)
                random_dir = random_dirs[0]
                nre_output_dir_for_grpc = str(random_dir)
            # Update substitutions for grpc
            substitutions["NRE_OUTPUT_DIR"] = nre_output_dir_for_grpc
            continue  # Move to next command

        # For subsequent commands, if they are grpc, use the updated NRE_OUTPUT_DIR
        if isinstance(cmd_or_group, Command):
            result = execute_command(
                cmd_or_group,
                substitutions,
                str(project_base_path),
                dry_run,
            )
            if result.returncode != 0:
                print(f"[FAIL] Command failed with code {result.returncode}")
                test_success = False
                break
        elif isinstance(cmd_or_group, CommandGroup):
            results = execute_command_group(
                cmd_or_group,
                substitutions,
                str(project_base_path),
                dry_run,
            )
            for result in results:
                if result.returncode != 0:
                    print(f"[FAIL] Command in group failed with code {result.returncode}")
                    test_success = False
                    break
            if not test_success:
                break

    return test_success


def parse_timings_file(timings_file: Path) -> Dict[str, float]:
    """Parse timings.txt file and return a dict of step_name -> elapsed_seconds.

    Expected format per line:
    [2026-01-13_08-36-27] Elapsed time for train run: 00:02:31.703

    Returns:
        Dict mapping step names (e.g., "train", "val") to elapsed time in seconds.
    """
    timings: Dict[str, float] = {}

    if not timings_file.exists():
        return timings

    # Regex to parse: [timestamp] Elapsed time for <name> run: HH:MM:SS.mmm
    pattern = re.compile(r"\[[\d_:-]+\]\s+Elapsed time for (\w+) run:\s+(\d+):(\d+):(\d+)\.(\d+)")

    with open(timings_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step_name = match.group(1)
                hours = int(match.group(2))
                minutes = int(match.group(3))
                seconds = int(match.group(4))
                milliseconds = int(match.group(5))

                total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
                timings[step_name] = total_seconds

    return timings


def check_ci_runtime_limits(
    results_dir: str,
    ci_runtime_limits: Dict[str, int],
) -> bool:
    """Validate recorded runtimes against CI runtime limits.

    Args:
        results_dir: Path to the test results directory containing timings.txt
        ci_runtime_limits: Dict mapping step names to limit in seconds

    Returns:
        True if all limits are satisfied, False otherwise
    """
    if not ci_runtime_limits:
        return True

    timings_file = Path(results_dir) / "timings.txt"

    if not timings_file.exists():
        print(f"[RUNTIME] ERROR: timings.txt not found in {results_dir}")
        return False

    print(f"[RUNTIME] Validating CI runtime limits from: {timings_file}")

    # Parse the timings file
    recorded_timings = parse_timings_file(timings_file)

    # Validate each limit
    success = True
    for step_name, limit_seconds in ci_runtime_limits.items():
        if step_name not in recorded_timings:
            print(
                f"[RUNTIME] ERROR: CI runtime limit defined for '{step_name}' but no timing found in timings.txt. "
                f"Available steps: {list(recorded_timings.keys())}"
            )
            success = False
            continue

        recorded_seconds = recorded_timings[step_name]
        recorded_str = str(timedelta(seconds=recorded_seconds))
        limit_str = str(timedelta(seconds=limit_seconds))

        if recorded_seconds > limit_seconds:
            print(
                f"[RUNTIME] ERROR: EXCEEDED '{step_name}': "
                f"{recorded_str} ({recorded_seconds:.1f}s) > limit {limit_str} ({limit_seconds}s)"
            )
            success = False
        else:
            print(
                f"[RUNTIME] OK '{step_name}': "
                f"{recorded_str} ({recorded_seconds:.1f}s) <= limit {limit_str} ({limit_seconds}s)"
            )

    return success


def check_eval_psnr_thresholds(
    results_dir: str,
    eval_psnr_thresholds: Dict[str, float],
) -> bool:
    """Validate recorded PSNR metrics against thresholds.

    Args:
        results_dir: Path to the test results directory containing eval/rendering_metrics.yaml
        eval_psnr_thresholds: Dict mapping camera IDs to minimum PSNR values

    Returns:
        True if all thresholds are satisfied, False otherwise
    """

    def _get_by_key_from_nested_dict(d: dict, key_sequence_dotted: str) -> Any:
        """Get a value from a nested dict given a sequence of keys separated by dots or raise a KeyError."""
        keys = key_sequence_dotted.split(".")
        current_path = []
        for key in keys:
            current_path.append(key)
            if not isinstance(d, dict) or key not in d:
                raise KeyError(".".join(current_path))
            d = d[key]
        return d

    if not eval_psnr_thresholds:
        return True

    metrics_file_name = "rendering_metrics.yaml"
    metrics_file = Path(results_dir) / "eval" / metrics_file_name

    if not metrics_file.exists():
        print(f"[EVAL PSNR] ERROR: {metrics_file_name} not found at {metrics_file}")
        return False

    print(f"[EVAL PSNR] Validating PSNR thresholds from: {metrics_file}")

    # Parse the metrics file
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_data = yaml.safe_load(f)

    # Validate each threshold
    success = True
    for camera_id, min_psnr in eval_psnr_thresholds.items():
        try:
            key = f"per_sequence_metrics.static_camera_mask.{camera_id}.psnr.avg"
            actual_psnr = _get_by_key_from_nested_dict(metrics_data, key)
        except KeyError as e:
            print(f"[EVAL PSNR] ERROR: {e.args[0]} not found in {metrics_file_name} when trying to get {key}")
            return False

        if not isinstance(actual_psnr, (int, float)):
            print(f"[EVAL PSNR] ERROR: Invalid PSNR value for '{camera_id}': {actual_psnr} (expected numeric)")
            success = False
            continue

        if actual_psnr < min_psnr:
            print(f"[EVAL PSNR] ERROR: BELOW THRESHOLD '{camera_id}': {actual_psnr:.2f} < threshold {min_psnr:.2f}")
            success = False
        else:
            print(f"[EVAL PSNR] OK '{camera_id}': {actual_psnr:.2f} >= threshold {min_psnr:.2f}")

    return success


def setup_test_environment(
    test_case: Any,
    unavailable_datasets: Set[str],
) -> bool:
    """Set up test environment and validate prerequisites.

    Returns:
        bool: True if test should proceed, False if test should be skipped.
    """
    print("\n" + "=" * 80)
    print(f"EXECUTING TEST: {test_case.name}")
    print("-" * 80)

    # Skip if dataset is required but not available
    if test_case.dataset is not None:
        dataset_name = test_case.dataset.name
        if dataset_name in unavailable_datasets:
            print(f"SKIPPED: Dataset {dataset_name} is not available")
            print("=" * 80)
            return False  # Signal to skip

    return True  # Proceed with test


def human_readable_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    n: float = size_bytes
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            break
        n /= 1024
    return f"{int(n)} {unit}" if unit == "bytes" else f"{n:.2f} {unit}"


def replace_large_outputs_with_markers(dry_run: bool = False) -> None:
    """Replace large output files with marker files to avoid too much remote cache traffic.
       This replacement only happens when we detect execution under "bazel test" and use the undeclared outputs
       directory, otherwise outputs remain local and are kept in full.

    Args:
        dry_run: If True, only print what would be done
    """

    # Only replace large files under "bazel test"'s undeclared outputs directory.
    output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if not output_dir:
        return

    # Don't do the replacement if we notice that we're executing through "bazel run" instead of "bazel test".
    # In this case there is no caching.
    if os.environ.get("BUILD_WORKING_DIRECTORY"):
        return

    # Set size limit for test outputs to 50 MB.
    LARGE_OUTPUT_SIZE_LIMIT = 50 * 1024 * 1024

    output_path = Path(output_dir)
    if not output_path.exists():
        return

    for file_path in output_path.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            file_size = file_path.stat().st_size
            if file_size > LARGE_OUTPUT_SIZE_LIMIT:
                marker_content = f"File replaced with marker due to large size {human_readable_size(file_size)}\n"
                marker_path = file_path.with_suffix(file_path.suffix + ".marker.txt")

                if dry_run:
                    print(f"[DRY-RUN] Would replace large file: {file_path} ({human_readable_size(file_size)})")
                else:
                    # Remove the original file and create marker
                    file_path.unlink()
                    marker_path.write_text(marker_content)
                    print(
                        f"[CLEANUP] Replaced large output with marker: {file_path.name} ({human_readable_size(file_size)})"
                    )
        except (OSError, IOError) as e:
            print(f"[WARNING] Could not process {file_path}: {e}")


def find_matching_test_cases(test_cases: List[Any], pattern: str) -> List[Any]:
    """Find test cases matching the given pattern (supports wildcards)."""
    matching_cases = []
    for test_case in test_cases:
        if fnmatch.fnmatch(test_case.name, pattern):
            matching_cases.append(test_case)
    return matching_cases


def execute_single_test(
    test_case: Any,
    unavailable_datasets: Set[str],
    artifacts: Dict[str, Artifacts],
    unavailable_artifacts: Set[str],
    tag: str,
    suffix: Optional[str],
    runfiles: bool,
    extra_params: str,
    project_base_path: str,
    dry_run: bool,
    validate_ci_runtime_limits: bool = False,
) -> str:
    """Execute a single test case."""
    # Set up test environment and validate prerequisites
    should_proceed = setup_test_environment(test_case, unavailable_datasets)
    if not should_proceed:
        return "skipped"

    # Check if required artifact is available
    if hasattr(test_case, "artifact_source") and test_case.artifact_source and test_case.artifact_source != "train_val":
        if test_case.artifact_source in unavailable_artifacts:
            print(f"[SKIP] Test requires unavailable artifact: {test_case.artifact_source}")
            return "skipped"

    # Get output directory from test case
    output_dir = test_case.results_dir

    # Prepare output directory
    if not prepare_test_output_directory(output_dir):
        print(f"[FAIL] Test failed due to directory cleanup failure")
        return "failed"

    # Resolve executable path if in runfiles mode
    executable_path = None
    if runfiles:
        executable_path = resolve_runfiles_executable_path(test_case)
        if executable_path is None:
            return "failed"

    # Resolve GIF tool path
    gif_tool_path = resolve_gif_tool_path()
    if gif_tool_path is None:
        return "failed"

    # Build substitution dict
    substitutions = build_substitutions_dict(tag, suffix, executable_path, gif_tool_path, output_dir, extra_params)

    # Execute commands
    test_success = execute_test_commands(test_case, substitutions, project_base_path, dry_run)

    # Validate metrics if test succeeded (run all checks even if some fail)
    if test_success and not dry_run:
        validation_success = True

        # Validate CI runtime limits if requested
        if validate_ci_runtime_limits and test_case.ci_runtime_limits:
            if not check_ci_runtime_limits(output_dir, test_case.ci_runtime_limits):
                validation_success = False

        # Validate eval PSNR thresholds
        if test_case.eval_psnr_thresholds:
            if not check_eval_psnr_thresholds(output_dir, test_case.eval_psnr_thresholds):
                validation_success = False

        if not validation_success:
            test_success = False

    # Potentially replace large output files with markers to avoid too much remote cache traffic for Bazel "test" runs.
    replace_large_outputs_with_markers(dry_run)

    # Update overall tracking
    print("-" * 80)

    if test_success:
        print(f"SUCCESS: Test {test_case.name} completed successfully")
        print("=" * 80)
        return "success"
    else:
        print(f"FAILED: Test {test_case.name} failed")
        print("=" * 80)
        return "failed"


def print_final_summary(tests_run: int, tests_skipped: int, failed_tests: List[str]) -> int:
    """Print the final test execution summary."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Tests run: {tests_run}")
    print(f"Tests skipped: {tests_skipped}")
    print(f"Tests failed: {len(failed_tests)}")
    if len(failed_tests) == 0:
        print("RESULT: All tests completed successfully")
        print("=" * 80)
        return 0
    else:
        print("RESULT: The following tests failed:")
        for failed_test in failed_tests:
            print(f" - {failed_test}")
        print("=" * 80)
        return 1


def validate_tag_runfiles_exclusivity(ctx, param, value):
    """Validate that exactly one of --tag or --runfiles is provided."""
    # Check both existing params and the current parameter being processed
    tag_provided = bool(ctx.params.get("tag")) or (param.name == "tag" and value)
    runfiles_provided = bool(ctx.params.get("runfiles")) or (param.name == "runfiles" and value)

    if tag_provided and runfiles_provided:
        raise click.BadParameter("Cannot use both --tag and --runfiles options")
    if not tag_provided and not runfiles_provided:
        raise click.BadParameter("Either --tag or --runfiles must be provided")

    return value


@click.command()
@click.option(
    "--test-identifiers",
    help="Comma-separated list of test identifiers to execute, supports shell-style wildcards (if not provided, runs all available test cases)",
)
@click.option(
    "--tag",
    callback=validate_tag_runfiles_exclusivity,
    help="Docker image tag to use for the run (mutually exclusive with --runfiles)",
)
@click.option(
    "--suffix",
    help="Optional Docker image suffix (only valid with --tag, no suffix is used by default)",
)
@click.option(
    "--runfiles",
    is_flag=True,
    callback=validate_tag_runfiles_exclusivity,
    help="Use Bazel runfiles instead of Docker image (mutually exclusive with --tag), only supported for 'bazel test' execution",
)
@click.option("--extra-params", default="", help="Additional parameters to pass to scripts")
@click.option("--skip-resource-download", is_flag=True, help="Skip automatic dataset and artifacts downloading")
@click.option("--dry-run", is_flag=True, help="Show commands that would be executed without running them")
@click.option(
    "--force-obfuscation",
    type=click.Choice(["on", "off"]),
    help="Force the whole test plan to use the specified obfuscation setting",
)
@click.option(
    "--validate_ci_runtime_limits",
    is_flag=True,
    help="Validate recorded runtimes against ci_runtime_limit_* thresholds defined in test_plan.yml (requires --runfiles, limits are only tuned for use in GitLab CI)",
)
@click.option(
    "--grpc-port-base",
    type=int,
    default=8000,
    help="Base port for gRPC servers, final port = base + test_id",
)
def run_tests(
    test_identifiers: Optional[str],
    tag: str,
    suffix: Optional[str],
    runfiles: bool,
    extra_params: str,
    skip_resource_download: bool,
    dry_run: bool,
    force_obfuscation: Optional[str],
    validate_ci_runtime_limits: bool,
    grpc_port_base: int,
) -> int:
    """Run test plan or subset of test cases."""
    # Validate suffix and tag combination
    if suffix is not None and tag is None:
        raise click.BadParameter("--suffix can only be used with --tag")

    # Validate that --validate_ci_runtime_limits is only used with --runfiles
    if validate_ci_runtime_limits and not runfiles:
        raise click.BadParameter(
            "--validate_ci_runtime_limits requires --runfiles (Docker download time would affect runtime)"
        )

    try:
        # Initialize environment and configuration
        project_base_path, test_config = initialize_test_environment(dry_run, force_obfuscation, grpc_port_base)

        # Load test cases, datasets, and artifacts
        all_test_cases, datasets, artifacts = load_test_cases_and_resources(test_config)
        if all_test_cases is None or datasets is None or artifacts is None:
            sys.exit(1)

        # Resolve test cases to execute
        if test_identifiers:
            # Parse comma-separated test identifiers
            identifier_list = [id for id in test_identifiers.split(",")]

            # Filter test cases based on provided identifiers
            test_cases_to_run = []
            invalid_test_identifiers = []

            for identifier in identifier_list:
                matching_test_cases = find_matching_test_cases(all_test_cases, identifier)
                if matching_test_cases:
                    test_cases_to_run.extend(matching_test_cases)
                else:
                    invalid_test_identifiers.append(identifier)

            # Remove duplicates, possible with multiple input identifiers, while preserving order
            unique_dict = {tc.name: tc for tc in test_cases_to_run}
            test_cases_to_run = list(unique_dict.values())

            # Check for invalid test identifiers
            if invalid_test_identifiers:
                print(
                    f"[FATAL] {len(invalid_test_identifiers)} invalid test identifiers found: {invalid_test_identifiers}"
                )
                print("[INFO] These test identifiers are not available in the current test plan.")
                if force_obfuscation:
                    print(f"[INFO] Note: Test plan was filtered by --force-obfuscation {force_obfuscation}")
                sys.exit(1)
        else:
            # Run all available test cases
            print("[INFO] No test identifiers provided, running all available test cases...")
            test_cases_to_run = all_test_cases

        # Collect required datasets from test cases
        required_datasets = collect_required_datasets(test_cases_to_run)

        # Ensure datasets are available
        unavailable_datasets, failed_downloads = ensure_resources_available_batch(
            required_datasets, datasets, skip_resource_download, dry_run
        )
        if unavailable_datasets is None:
            sys.exit(1)

        # Collect required artifacts from test cases
        required_artifacts = collect_required_artifacts(test_cases_to_run)

        # Ensure artifacts are available
        unavailable_artifacts, failed_artifact_downloads = ensure_resources_available_batch(
            required_artifacts, artifacts, skip_resource_download, dry_run
        )
        if unavailable_artifacts is None:
            sys.exit(1)

        # Execute tests
        tests_run = 0
        tests_skipped = 0
        failed_tests = []

        for test_case in test_cases_to_run:
            try:
                result = execute_single_test(
                    test_case,
                    unavailable_datasets,
                    artifacts,
                    unavailable_artifacts,
                    tag,
                    suffix,
                    runfiles,
                    extra_params,
                    str(project_base_path),
                    dry_run,
                    validate_ci_runtime_limits,
                )

                if result == "success":
                    tests_run += 1
                elif result == "failed":
                    tests_run += 1
                    failed_tests.append(test_case.name)
                elif result == "skipped":
                    tests_skipped += 1

            except Exception as e:
                print("-" * 80)
                print(f"EXCEPTION: Exception while running test: {e}")
                print(f"TRACEBACK:\n{traceback.format_exc()}")
                tests_run += 1
                failed_tests.append(test_case.name)
                print("=" * 80)

        # Print final summary and exit with proper code
        exit_code = print_final_summary(tests_run, tests_skipped, failed_tests)
        sys.exit(exit_code)

    except Exception as e:
        print(f"[FATAL] Unhandled exception in main: {e}")
        print(f"[FATAL] Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
