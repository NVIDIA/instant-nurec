#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
GitLab CI Job Collector
=======================

This script queries the GitLab API to collect CI job data from multiple projects and saves it as JSON.
It collects comprehensive data including job results, runner information, and timing metrics.

If the output file already exists, new jobs run after it was created will be merged with the existing data.
All dates and times are treated as UTC.

Setup:
------
1. Install required Python packages:
   pip install requests python-dateutil

2. Authentication options (in priority order):
   a) --token parameter:
      - Go to GitLab > Preferences > Access Tokens
      - Create a token with read_api scope
      - Use with --token parameter

   b) .netrc file authentication:
      - Create or edit ~/.netrc file with your GitLab credentials:
        machine gitlab-master.nvidia.com
        login oauth2
        password YOUR_GITLAB_TOKEN
      - Make sure permissions are secure: chmod 600 ~/.netrc

   For extra projects that require a separate token (e.g. projects outside the
   primary token's visibility), use --extra-project-token PROJECT=TOKEN.
   The flag can be repeated for multiple projects.

Usage:
------
Basic usage:
  # Using token authentication
  python ci_job_collector.py --token YOUR_GITLAB_TOKEN

  # Using .netrc authentication (no token needed)
  python ci_job_collector.py

Additional options:
  # Specify number of days to look back in case no existing data is available
  python ci_job_collector.py --days 2

  # Collect all available jobs in case no existing data is available
  python ci_job_collector.py --days all

  # Filter by branch
  python ci_job_collector.py --branch main

  # Specify a different namespace (default is "nrs")
  python ci_job_collector.py --namespace other-group

  # Specify a single project instead of all projects in namespace
  python ci_job_collector.py --project nrs/specific-project

  # Specify a custom output file (default is ci_analytics/ci_jobs.json relative to repo root)
  python ci_job_collector.py --file custom_name.json

  # Also collect jobs from additional projects by path
  python ci_job_collector.py --extra-projects group1/project1,group2/project2

  # Container image sizes for :latest NRE images are collected from nvcr.io
  # (requires ~/.docker/config.json auth).

Output:
-------
The script generates ci_analytics/ci_jobs.json (raw job data for visualization). When
collecting from the NRS group (which includes the nrs/nre project), it also merges
test runtimes from the latest passing job (see TEST_RUNTIMES_JOB_NAME) on main into
ci_analytics/ci_test_runtimes.json, and queries/stores compressed sizes of the :latest
container images into ci_analytics/container_image_sizes.json.
"""

import argparse
import datetime
import json
import netrc
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile

from datetime import timezone
from pathlib import Path

from dateutil.relativedelta import relativedelta


# Job whose artifacts we use for test runtimes.
TEST_RUNTIMES_JOB_NAME = "serialized_tests"
TEST_RUNTIMES_PROJECT = "nrs/nre"  # path to the project that runs that job

# Container registry entries for :latest image size collection.
CONTAINER_REGISTRY = "nvcr.io"
CONTAINER_BASE_REPOSITORY = "nvidian/ct-toronto-ai/nre"
CONTAINER_IMAGE_SUFFIXES = ("_run", "_tools", "_nrm_run", "_obfuscated_run", "_obfuscated_tools", "_obfuscated_nrm_run")
CONTAINER_IMAGES = tuple(
    f"{CONTAINER_REGISTRY}/{CONTAINER_BASE_REPOSITORY}{s}:latest" for s in CONTAINER_IMAGE_SUFFIXES
)


def parse_args():
    parser = argparse.ArgumentParser(description="Query GitLab CI job data and save as JSON")
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the repo root by going up 4 levels from the script location
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
    default_output = os.path.join(repo_root, "ci_analytics", "ci_jobs.json")

    parser.add_argument("--token", help="GitLab API token (optional if credentials exist in ~/.netrc)")
    parser.add_argument("--url", default="https://gitlab-master.nvidia.com", help="GitLab URL")
    parser.add_argument("--group", default="nrs", help="GitLab group to collect projects from")
    parser.add_argument(
        "--project", help="Specific project ID or path (if provided, only this project will be processed)"
    )
    parser.add_argument(
        "--extra-projects",
        metavar="LIST",
        help="Comma-separated list of extra project IDs or paths to collect.",
    )
    parser.add_argument(
        "--extra-project-token",
        metavar="PROJECT=TOKEN",
        action="append",
        dest="extra_project_tokens",
        help=(
            "Per-project token for extra projects not accessible with the primary token. "
            "Format: PROJECT_PATH=TOKEN (e.g. dvl/sauron/sauron=glpat-xxx). "
            "Can be repeated for multiple projects."
        ),
    )
    parser.add_argument(
        "--days",
        default="7",
        help="Number of days to look back if no existing data (defaults to 7). Use 'all' to collect all available jobs.",
    )
    parser.add_argument(
        "--file",
        default=default_output,
        help="Output file path (defaults to ci_analytics/ci_jobs.json relative to repo root)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--branch", help="Filter by branch name")
    return parser.parse_args()


# Import shared HTTP utilities for safe response handling (CVE-2025-13836)
# Add parent scripts directory to path for standalone execution
import sys as _sys


_scripts_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _scripts_dir not in _sys.path:
    _sys.path.insert(0, _scripts_dir)
from stream_read import stream_read


def _get_netrc_authenticators(url, verbose=False):
    """Helper function to get authenticators from .netrc file for a given URL"""
    try:
        # Parse the hostname from the URL
        hostname = urllib.parse.urlparse(url).netloc
        if verbose:
            print(f"Checking for credentials for hostname: {hostname}")

        # Check if ~/.netrc exists
        netrc_path = Path.home() / ".netrc"
        if not netrc_path.exists():
            if verbose:
                print(f"~/.netrc file not found at {netrc_path}")
            return None

        # Check file permissions
        if verbose:
            import stat

            mode = os.stat(netrc_path).st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                print(f"Warning: ~/.netrc has insecure permissions. Use: chmod 600 {netrc_path}")

        # Get authenticators for the hostname
        try:
            nrc = netrc.netrc(str(netrc_path))
            authenticators = nrc.authenticators(hostname)

            if verbose:
                if authenticators:
                    login = authenticators[0]
                    print(f"Found credentials for {hostname} with login: {login}")
                else:
                    print(f"No credentials found for {hostname} in ~/.netrc")
                    hosts = [host for host in nrc.hosts.keys()]
                    print(f"Available hosts in .netrc: {hosts}")

            return authenticators
        except netrc.NetrcParseError as e:
            if verbose:
                print(f"Error parsing ~/.netrc file: {e}")
                print("Check the format of your .netrc file. It should look like:")
                print(f"machine {hostname}")
                print("login oauth2")
                print("password YOUR_GITLAB_TOKEN")
            return None

    except Exception as e:
        if verbose:
            print(f"Error checking ~/.netrc: {e}")
        return None


def check_netrc_credentials(url, verbose=False):
    """Check if credentials for the GitLab URL exist in ~/.netrc"""
    authenticators = _get_netrc_authenticators(url, verbose)
    return authenticators is not None


def get_token_from_netrc(url, verbose=False):
    """Extract GitLab token from .netrc file"""
    authenticators = _get_netrc_authenticators(url, verbose)
    if authenticators:
        # authenticators is a tuple of (login, account, password)
        _, _, password = authenticators
        return password
    return None


def get_gitlab_session(url, token=None, verbose=False):
    """Create a session with GitLab authentication"""
    # If no token provided, try to get it from .netrc
    if not token:
        token = get_token_from_netrc(url, verbose)

    if not token:
        print("Error: No GitLab authentication credentials found.")
        print("Please either:")
        print("  1. Provide a token with --token parameter, or")
        print("  2. Configure credentials in ~/.netrc file")
        print("\nTo configure ~/.netrc, add the following lines:")
        hostname = urllib.parse.urlparse(url).netloc
        print(f"machine {hostname}")
        print("login oauth2")
        print("password YOUR_GITLAB_TOKEN")
        print("\nAnd set proper permissions: chmod 600 ~/.netrc")
        exit(1)

    try:
        if verbose:
            print("Using token authentication")

        # Create a session with the token
        session = {"headers": {"PRIVATE-TOKEN": token}, "url": url}

        # Test the connection by listing a project
        try:
            response = _make_request(session, "/api/v4/projects", params={"per_page": 1})
            if response.getcode() != 200:
                raise Exception(f"Connection test failed with status {response.getcode()}")
            return session
        except Exception as e:
            if verbose:
                print(f"Connection test failed: {e}")
            raise

    except Exception as e:
        print(f"Error connecting to GitLab: {e}")
        exit(1)


def _make_request(session, path, params=None, method="GET"):
    """Make an HTTP request to the GitLab API"""
    url = session["url"] + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)

    request = urllib.request.Request(url, headers=session["headers"], method=method)
    try:
        return urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise Exception(f"Resource not found: {path}")
        elif e.code == 401:
            raise Exception("Authentication failed. Please check your token.")
        elif e.code == 403:
            raise Exception("Access denied. Please check your permissions.")
        else:
            raise Exception(f"HTTP error {e.code}: {e.reason}")


def find_project(session, url, project_name_or_id, verbose=False):
    """Find a project by name or ID"""
    try:
        # First try to get the project directly (works for IDs and exact paths)
        if verbose:
            print(f"Attempting to access project: {project_name_or_id}")

        try:
            response = _make_request(session, f"/api/v4/projects/{urllib.parse.quote(project_name_or_id, safe='')}")
            return json.loads(stream_read(response).decode())
        except Exception as e:
            if verbose:
                print(f"Direct access failed: {e}")

            # If that fails, try to search for it by name
            if verbose:
                print(f"Searching for project by name: {project_name_or_id}")

            # Search is case-insensitive and partial match
            response = _make_request(
                session, "/api/v4/projects", params={"search": project_name_or_id, "per_page": 100}
            )
            projects = json.loads(stream_read(response).decode())

            if not projects:
                if verbose:
                    print("No matching projects found")

                # Try to get some accessible projects to suggest
                response = _make_request(session, "/api/v4/projects", params={"per_page": 5})
                sample_projects = json.loads(stream_read(response).decode())
                if sample_projects:
                    print("\nSome accessible projects:")
                    for p in sample_projects:
                        print(f"  - {p['path_with_namespace']} (ID: {p['id']})")

                raise Exception(f"Project '{project_name_or_id}' not found")

            if verbose:
                print(f"Found {len(projects)} matching projects:")
                for p in projects:
                    print(f"  - {p['path_with_namespace']} (ID: {p['id']})")

            # Try to find an exact match first
            for p in projects:
                if p["path_with_namespace"].lower() == project_name_or_id.lower():
                    if verbose:
                        print(f"Using exact match: {p['path_with_namespace']} (ID: {p['id']})")
                    return p

            # If no exact match, use the first result
            if verbose:
                print(f"Using first match: {projects[0]['path_with_namespace']} (ID: {projects[0]['id']})")
            return projects[0]

    except Exception as e:
        raise Exception(f"Error finding project '{project_name_or_id}': {e}")


def get_group_projects(session, url, group, verbose=False):
    """Get all first-level projects in the specified group"""
    if verbose:
        print(f"Fetching projects in group: {group}")

    try:
        # Get the group
        if verbose:
            print(f"Getting group: '{group}'")

        response = _make_request(session, f"/api/v4/groups/{urllib.parse.quote(group, safe='')}")
        group_data = json.loads(stream_read(response).decode())

        if verbose:
            print(f"Found group with ID: {group_data['id']}, Name: {group_data['name']}")

        # Get projects directly from the group
        if verbose:
            print(f"Retrieving projects for group ID: {group_data['id']}")

        # include_subgroups=False ensures we only get first-level projects
        response = _make_request(
            session, f"/api/v4/groups/{group_data['id']}/projects", params={"include_subgroups": False, "per_page": 100}
        )
        projects = json.loads(stream_read(response).decode())

        if verbose:
            print(f"Found {len(projects)} projects in group {group}")
            if projects:
                print("Projects found:")
                for i, project in enumerate(projects[:10]):  # Show up to 10 projects
                    print(f"  {i + 1}. {project['path_with_namespace']} (ID: {project['id']})")

        return projects

    except Exception as e:
        print(f"Error fetching projects in group {group}: {e}")
        return []


def to_utc(dt):
    """Convert a datetime to UTC if it's not already in UTC"""
    if dt.tzinfo is not None and dt.tzinfo != timezone.utc:
        return dt.astimezone(timezone.utc)
    return dt


def _get_start_date_from_days(days, context_msg, verbose=False):
    """Helper function to get start date from days parameter"""
    if days == "all":
        start_date = datetime.datetime.min.replace(tzinfo=timezone.utc)
        if verbose:
            print(f"{context_msg}, collecting all available jobs")
    else:
        start_date = datetime.datetime.now(timezone.utc) - datetime.timedelta(days=int(days))
        if verbose:
            print(f"{context_msg}, using {days} days ago as start date")
    return start_date


def get_date_range(days, existing_data=None, verbose=False):
    """Get start date for the query in UTC"""
    # If existing data is available, use the last job's created_at date
    if existing_data:
        # Find the most recent created_at date in the existing data
        latest_date = None
        for job in existing_data:
            if job.get("created_at"):
                try:
                    job_date = datetime.datetime.fromisoformat(job["created_at"])
                    job_date = to_utc(job_date)

                    if latest_date is None or job_date > latest_date:
                        latest_date = job_date
                except (ValueError, TypeError) as e:
                    if verbose:
                        print(f"Warning: Could not parse date '{job.get('created_at')}': {e}")
                    # Skip jobs with invalid date format
                    continue

        if latest_date:
            # Use the date of the last job's created_at
            start_date = latest_date.replace(hour=0, minute=0, second=0)
            if verbose:
                print(f"Using last job's creation date as start date: {start_date.date().isoformat()}")
        else:
            # Fall back to default days if no valid dates found
            start_date = _get_start_date_from_days(days, "No valid dates in existing data", verbose)
    else:
        # No existing data, use the specified number of days
        start_date = _get_start_date_from_days(days, "No existing data", verbose)

    # Set to start of day in UTC
    start_date = start_date.replace(hour=0, minute=0, second=0)

    return start_date


def get_job_data(job, project):
    """Extract relevant data from a job object"""

    # Extract all available job data
    return {
        "id": job["id"],
        "pipeline_id": job["pipeline"]["id"],
        "name": job["name"],
        "commit_title": job["commit"]["title"],
        "status": job["status"],
        "stage": job["stage"],
        "ref": job["ref"],
        "tag": job.get("tag", False),
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "duration": job["duration"],  # runtime in seconds
        "queued_duration": job["queued_duration"],  # pending time in seconds
        "runner_description": job["runner"]["description"] if job.get("runner") else None,
        "runner_id": job["runner"]["id"] if job.get("runner") else None,
        "runner_type": job["runner"].get("runner_type") if job.get("runner") else None,
        "user": job["user"]["username"] if job.get("user") else None,
        "failure_reason": job.get("failure_reason"),
        "web_url": job.get("web_url"),
        "project_name": project["path_with_namespace"],
        "project_id": project["id"],
    }


def get_project_jobs(session, url, project, start_date, branch=None, verbose=False):
    """Get completed jobs for a project since the specified start date"""
    if verbose:
        print(f"Fetching completed jobs created since {start_date.isoformat()}")

    # Prepare job query parameters
    job_params = {
        "scope[]": ["success", "failed", "canceled"],  # Get all relevant job statuses
        "per_page": 100,  # Maximum allowed by GitLab API
    }

    if branch:
        job_params["ref"] = branch

    # Get jobs
    jobs = []
    page = 1
    while True:
        job_params["page"] = page
        response = _make_request(session, f"/api/v4/projects/{project['id']}/jobs", params=job_params)
        page_jobs = json.loads(stream_read(response).decode())

        if not page_jobs:
            break

        # Check if any job in this page is before our start date
        found_older_job = False
        for job in page_jobs:
            if not job.get("created_at"):
                continue
            try:
                job_date = datetime.datetime.fromisoformat(job["created_at"])
                job_date = to_utc(job_date)
                if job_date < start_date:
                    found_older_job = True
                    break
            except (ValueError, TypeError):
                continue

        # Add all jobs from this page
        jobs.extend(page_jobs)

        if verbose and page % 10 == 0:
            print(f"Loaded {len(jobs)} jobs so far...")

        # If we found a job older than our start date, stop fetching more pages
        if found_older_job:
            if verbose:
                print(f"Found job created before {start_date.isoformat()}, stopping API queries")
            break

        page += 1

    if verbose:
        print(f"Retrieved {len(jobs)} jobs from {page} pages")

    return jobs


def process_jobs(jobs, project, start_date, verbose=False):
    """Process jobs and extract job data"""
    jobs_data = []

    total_jobs = len(jobs)
    processed_jobs = 0
    filtered_jobs = 0
    incomplete_jobs = 0

    for job in jobs:
        processed_jobs += 1
        if processed_jobs % 10 == 0:
            print(f"Processing job {processed_jobs}/{total_jobs} (ID: {job['id']})")

        try:
            # Skip jobs that haven't finished
            if not job.get("finished_at"):
                incomplete_jobs += 1
                if verbose:
                    print(f"  Skipping job {job['id']} ({job['name']}) - not completed")
                continue

            # Filter by creation date
            try:
                job_date = datetime.datetime.fromisoformat(job["created_at"])
                job_date = to_utc(job_date)

                if job_date < start_date:
                    filtered_jobs += 1
                    continue
            except (ValueError, TypeError) as e:
                if verbose:
                    print(f"Warning: Could not parse date '{job.get('created_at')}': {e}")
                continue

            jobs_data.append(get_job_data(job, project))
        except Exception as e:
            print(f"Error processing job {job['id']}: {e}")

    if verbose:
        print(f"Skipped {incomplete_jobs} jobs that haven't completed")
        print(f"Filtered out {filtered_jobs} jobs created before start date")

    return jobs_data


def process_jobs_for_projects(session, url, projects, start_date, branch=None, verbose=False):
    """Process each project and get its jobs"""
    all_jobs_data = []

    # Process each project
    for i, project in enumerate(projects, 1):
        try:
            print(f"\nProcessing project {i}/{len(projects)}: {project['path_with_namespace']}")

            # Use the per-project session if one was assigned via --extra-projects-token
            effective_session = project.get("_session", session)

            # Get jobs for this project
            jobs = get_project_jobs(effective_session, url, project, start_date, branch, verbose)

            # Process jobs and extract job data
            jobs_data = process_jobs(jobs, project, start_date, verbose)

            print(f"Collected {len(jobs_data)} jobs from project {project['path_with_namespace']}")
            all_jobs_data.extend(jobs_data)

        except Exception as e:
            print(f"Error processing project {project['id']}: {e}")
            if verbose:
                import traceback

                traceback.print_exc()

    return all_jobs_data


def load_existing_data(file_path, verbose=False):
    """Load existing job data from file if it exists"""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if verbose:
                    print(f"Loading existing data from {file_path}")
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading existing data: {e}")
            print(f"Creating new data file (existing file may be corrupted)")
            return []
        except Exception as e:
            print(f"Error reading existing file: {e}")
            return []
    else:
        if verbose:
            print(f"No existing data file found at {file_path}")
        return []


def merge_job_data(existing_data, new_data, verbose=False):
    """Merge new job data with existing data, avoiding duplicates"""
    if not existing_data:
        return new_data

    # Create a set of existing job IDs for quick lookup
    existing_ids = {job["id"] for job in existing_data}

    # Count new jobs before merging
    new_job_count = 0

    # Add only new jobs to the existing data
    for job in new_data:
        if job["id"] not in existing_ids:
            existing_data.append(job)
            existing_ids.add(job["id"])
            new_job_count += 1

    if verbose:
        print(f"Added {new_job_count} new jobs to existing data")

    return existing_data


def save_data(jobs_data, file_path, verbose=False):
    """Save job data to JSON file"""
    try:
        # Ensure the directory exists if file_path contains directories
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            if verbose:
                print(f"Ensuring directory exists: {directory}")

        with open(file_path, "w", encoding="utf-8") as jsonfile:
            json.dump(jobs_data, jsonfile, indent=2, default=str)

        if verbose:
            print(f"Data saved to {file_path}")

        return file_path
    except Exception as e:
        print(f"Error saving data: {e}")
        return None


def get_latest_job_by_name(session, project_id, job_name, ref, verbose=False):
    """Return (job, pipeline) for the successful job with the given name from the most recent pipeline
    (by pipeline id). Iterates pipelines in id-desc order. Returns (None, None) if not found."""
    for job, pipeline in get_passing_jobs_by_name_since(
        session, project_id, job_name, ref, since_date=None, verbose=verbose
    ):
        return job, pipeline
    return None, None


def get_passing_jobs_by_name_since(session, project_id, job_name, ref, since_date=None, verbose=False):
    """Yield (job, pipeline) for each successful job with the given name on ref, newest pipelines first.
    Stops when pipeline created_at date (YYYY-MM-DD) is before since_date (if set), or when no more pipelines."""
    project_enc = urllib.parse.quote(str(project_id), safe="")
    pipeline_params = {"order_by": "id", "sort": "desc", "per_page": 100}
    if ref:
        pipeline_params["ref"] = ref
    pipeline_page = 1
    try:
        while True:
            pipeline_params["page"] = pipeline_page
            response = _make_request(
                session,
                f"/api/v4/projects/{project_enc}/pipelines",
                params=pipeline_params,
            )
            pipelines = json.loads(stream_read(response).decode())
            if not pipelines:
                return
            for pipeline in pipelines:
                created_at = (pipeline.get("created_at") or "")[:10]
                if since_date and created_at < since_date:
                    return
                pid = pipeline.get("id")
                job_response = _make_request(
                    session,
                    f"/api/v4/projects/{project_enc}/pipelines/{pid}/jobs",
                    params={"per_page": 100, "scope": "success"},
                )
                jobs = json.loads(stream_read(job_response).decode())
                for job in jobs:
                    if job.get("name") == job_name:
                        yield job, pipeline
                        break
            pipeline_page += 1
    except Exception as e:
        if verbose:
            print(f"Error fetching jobs {job_name}: {e}")


def download_job_artifacts(session, url, project_id, job_id, dest_dir, verbose=False):
    """Download job artifacts zip and extract into dest_dir. Returns dest_dir on success."""
    project_enc = urllib.parse.quote(str(project_id), safe="")
    artifact_url = f"{url}/api/v4/projects/{project_enc}/jobs/{job_id}/artifacts"
    request = urllib.request.Request(artifact_url, headers=session["headers"])
    try:
        with urllib.request.urlopen(request) as response:
            zip_path = os.path.join(dest_dir, "artifacts.zip")
            with open(zip_path, "wb") as f:
                f.write(stream_read(response))
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
            if verbose:
                print(f"Extracted artifacts to {dest_dir}")
            return dest_dir
    except Exception as e:
        if verbose:
            print(f"Error downloading artifacts: {e}")
        raise


def _parse_bep_duration_seconds(d):
    """Parse BEP duration string (e.g. '123.456s'). Returns float seconds or None."""
    if not isinstance(d, str) or not d.endswith("s"):
        return None
    try:
        return float(d[:-1])
    except ValueError:
        return None


def extract_test_runtimes_from_bep(telemetry_root, verbose=False):
    """
    Parse BEP (Build Event Protocol) JSON from bazel-telemetry to get test runtimes and timeouts.
    Returns list of dicts: [{"target": "//pkg:name", "runtime_seconds": float, "timeout_seconds": int|null}, ...].
    For tests with more than one shard, each shard is emitted as a separate entry with target "label (shard N)".
    """
    telemetry_path = Path(telemetry_root)
    if not telemetry_path.is_dir():
        return []

    bep_paths = sorted(telemetry_path.glob("bep_*test.json"))
    if not bep_paths:
        if verbose:
            print("No bep_*test.json found in telemetry dir")
        return []

    events = []
    for bep_path in bep_paths:
        try:
            with open(bep_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            if verbose:
                print(f"Skip BEP file {bep_path}: {e}")

    # From targetCompleted: label -> number of test shards (testResult refs in children), and label -> timeout
    label_total_shards = {}
    label_timeout_seconds = {}
    for event in events:
        id_obj = event.get("id") or {}
        if "targetCompleted" not in id_obj:
            continue
        comp_id = id_obj["targetCompleted"]
        label = (comp_id.get("label") or "").strip()
        if not label:
            continue
        children = event.get("children") or []
        shard_count = sum(1 for c in children if isinstance(c, dict) and "testResult" in c)
        if shard_count > 0:
            label_total_shards[label] = shard_count
        payload = event.get("completed") or {}
        dur = payload.get("testTimeout")
        if dur is not None:
            sec = _parse_bep_duration_seconds(dur)
            if sec is not None:
                label_timeout_seconds[label] = round(sec)

    # Key by (label, shard) so multi-shard tests don't overwrite; value has "target" for display
    results = {}
    for event in events:
        id_obj = event.get("id") or {}
        if "testResult" not in id_obj or "testResult" not in event:
            continue
        test_id = id_obj["testResult"]
        label = (test_id.get("label") or "").strip()
        if not label:
            continue
        shard = test_id.get("shard", 1)
        total_shards = label_total_shards.get(label, 1)
        if total_shards > 1:
            target = f"{label} (shard {shard})"
        else:
            target = label
        key = (label, shard)
        payload = event.get("testResult") or {}
        dur = payload.get("testAttemptDuration")
        sec = _parse_bep_duration_seconds(dur)
        if sec is not None:
            if key not in results:
                results[key] = {"target": target, "runtime_seconds": None, "timeout_seconds": None}
            results[key]["runtime_seconds"] = round(sec, 2)
            if label in label_timeout_seconds:
                results[key]["timeout_seconds"] = label_timeout_seconds[label]

    return sorted(
        (r for r in results.values() if r["runtime_seconds"] is not None),
        key=lambda x: x["target"],
    )


def load_test_runtimes(file_path, verbose=False):
    """Load test runtimes JSON. Format: { \"runs\": [ { pipeline_id, date, job_id, tests } ] }."""
    if not os.path.exists(file_path):
        return {"runs": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "runs" not in data:
            data = {"runs": []}
        if verbose:
            print(f"Loaded {len(data['runs'])} runs from {file_path}")
        return data
    except (json.JSONDecodeError, OSError) as e:
        if verbose:
            print(f"Could not load {file_path}: {e}")
        return {"runs": []}


def merge_test_runtimes(existing, new_run, verbose=False):
    """Append new_run to existing['runs'] if its pipeline_id is not already present."""
    seen = {r["pipeline_id"] for r in existing["runs"]}
    if new_run["pipeline_id"] in seen:
        if verbose:
            print(f"Pipeline {new_run['pipeline_id']} already in test runtimes, skipping")
        return existing
    existing["runs"].append(new_run)
    existing["runs"].sort(key=lambda r: (r["date"], r["pipeline_id"]))
    return existing


def collect_test_runtimes(session, url, repo_root, verbose=False, project_id=None, start_date=None):
    """
    Find all passing runtime measurement jobs (see TEST_RUNTIMES_JOB_NAME) on main since the latest run already
    in the JSON, or since start_date when no data exists yet. Download artifacts, extract test
    runtimes from BEP in bazel-telemetry, and merge into ci_analytics/ci_test_runtimes.json.
    start_date is the same date used for job collection (from --days); used only when there are no existing runs.
    """
    if not project_id:
        if verbose:
            print("Skipping test runtimes collection (the job used to measure runtimes is not in collected projects)")
        return
    output_path = os.path.join(repo_root, "ci_analytics", "ci_test_runtimes.json")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    existing = load_test_runtimes(output_path, verbose=verbose)
    seen = {r["pipeline_id"] for r in existing["runs"]}
    if existing["runs"]:
        since_date = max(r["date"] for r in existing["runs"])
    else:
        since_date = start_date.date().isoformat() if start_date else None

    print()
    added = 0
    for job, pipeline in get_passing_jobs_by_name_since(
        session,
        project_id,
        TEST_RUNTIMES_JOB_NAME,
        ref="main",
        since_date=since_date,
        verbose=verbose,
    ):
        pipeline_id = pipeline.get("id")
        if pipeline_id in seen:
            continue
        created_at = (pipeline or {}).get("created_at") or ""
        try:
            dt = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            date_str = dt.astimezone(timezone.utc).date().isoformat()
        except (ValueError, TypeError):
            date_str = created_at[:10] if len(created_at) >= 10 else ""
        with tempfile.TemporaryDirectory(prefix="ci_test_runtimes_artifacts_") as tmp:
            download_job_artifacts(session, url, project_id, job["id"], tmp, verbose=verbose)
            telemetry_dir = os.path.join(tmp, "bazel-telemetry")
            tests = extract_test_runtimes_from_bep(telemetry_dir, verbose=verbose)
            if not tests and verbose:
                print("No test runtimes found in BEP (bazel-telemetry)")
        new_run = {
            "pipeline_id": pipeline_id,
            "date": date_str,
            "job_id": job["id"],
            "tests": tests,
        }
        existing = merge_test_runtimes(existing, new_run, verbose=verbose)
        seen.add(pipeline_id)
        added += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"Added {added} run(s) to test runtimes")
    print(f"Total runs in dataset: {len(existing['runs'])}")
    print(f"Data saved to: {output_path}")


def collect_container_image_sizes(repo_root, images, verbose=False):
    """
    Query compressed sizes of :latest container images via Python script
    and save to ci_analytics/container_image_sizes.json.
    """
    if not images:
        if verbose:
            print("Skipping container image sizes (no images configured)")
        return
    images_arg = ",".join(images)
    script_path = os.path.join(
        repo_root,
        "internal",
        "scripts",
        "ci",
        "image_analysis",
        "registry_image_sizes.py",
    )
    cmd = ["python3", script_path, "--images", images_arg, "--json"]
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        if verbose:
            print(f"Container image sizes skipped: {e}")
        return
    if result.returncode != 0:
        if verbose:
            print(f"Container image sizes script failed (exit {result.returncode}): {result.stderr or result.stdout}")
        return
    try:
        sizes = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Container image sizes parse error: {e}")
        return
    output_path = os.path.join(repo_root, "ci_analytics", "container_image_sizes.json")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now(timezone.utc)
    date_str = now.date().isoformat()  # UTC date (YYYY-MM-DD)

    # Merge into history: at most one data point per day (UTC)
    history = []
    if os.path.isfile(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                history = existing
        except (OSError, json.JSONDecodeError):
            pass
    history = [h for h in history if h.get("date") != date_str]
    history.append({"date": date_str, "images": sizes})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Container image sizes saved to: {output_path}")


def main():
    args = parse_args()

    # Connect to GitLab
    session = get_gitlab_session(args.url, args.token, args.verbose)

    # Build per-project sessions from --extra-projects-token PROJECT=TOKEN entries
    extra_sessions = {}
    for spec in args.extra_project_tokens or []:
        path, sep, token = spec.partition("=")
        if not sep or not token:
            print(f"Error: --extra-projects-token value '{spec}' must be in PROJECT=TOKEN format")
            exit(1)
        extra_sessions[path.strip()] = get_gitlab_session(args.url, token.strip(), args.verbose)

    # Set up the output file path
    output_file_path = args.file

    # Load existing data if available
    existing_data = load_existing_data(output_file_path, args.verbose)
    existing_count = len(existing_data)
    if existing_count > 0:
        print(f"Loaded {existing_count} existing jobs from {output_file_path}")

    # Get start date (in UTC)
    start_date = get_date_range(args.days, existing_data, args.verbose)
    if args.days == "all":
        print("Collecting all available CI jobs")
    else:
        print(f"Collecting CI jobs since {start_date.isoformat()} UTC")

    # Determine which projects to process
    nre_project_id = None
    if args.project:
        # Process a single project
        try:
            project = find_project(session, args.url, args.project, args.verbose)
            projects = [project]
            if project.get("path_with_namespace") == TEST_RUNTIMES_PROJECT:
                nre_project_id = project["id"]
            print(f"Using single project: {project['path_with_namespace']} (ID: {project['id']})")
        except Exception as e:
            print(f"Error: {e}")
            print("\nPossible solutions:")
            print("1. Check if the project name/path is correct")
            print("2. Verify you have access to this project")
            print("3. Try using the project ID number instead of the name")
            exit(1)
    else:
        # Process all first-level projects in the group
        projects = get_group_projects(session, args.url, args.group, args.verbose)
        if not projects:
            print(f"No projects found in group: {args.group}")
            exit(1)
        print(f"Found {len(projects)} projects in group {args.group}")

        # Add extra projects to the collection set
        seen_ids = {p["id"] for p in projects}
        extra_specs = []
        if args.extra_projects:
            extra_specs = [p.strip() for p in args.extra_projects.split(",") if p.strip()]
        for extra in extra_specs:
            project = None
            per_project_session = extra_sessions.get(extra.strip())
            try:
                project = find_project(session, args.url, extra.strip(), args.verbose)
            except Exception as primary_err:
                if per_project_session is not None:
                    if args.verbose:
                        print(f"Primary token failed for '{extra}' ({primary_err}), trying per-project token...")
                    try:
                        project = find_project(per_project_session, args.url, extra.strip(), args.verbose)
                        project["_session"] = per_project_session
                    except Exception as e:
                        print(f"Error: Could not add extra project '{extra}': {e}")
                        exit(1)
                else:
                    print(f"Error: Could not add extra project '{extra}': {primary_err}")
                    exit(1)
            if project is not None and project["id"] not in seen_ids:
                projects.append(project)
                seen_ids.add(project["id"])
                print(f"Added extra project: {project['path_with_namespace']} (ID: {project['id']})")

        nre_project_id = next(
            (p["id"] for p in projects if p.get("path_with_namespace") == TEST_RUNTIMES_PROJECT),
            None,
        )

    # Process all projects and collect job data
    new_jobs_data = process_jobs_for_projects(session, args.url, projects, start_date, args.branch, args.verbose)

    # Merge new data with existing data
    if new_jobs_data:
        print(f"\nCollected {len(new_jobs_data)} jobs from the GitLab API")
        merged_data = merge_job_data(existing_data, new_jobs_data, args.verbose)

        # Save the merged data
        save_data(merged_data, output_file_path, args.verbose)

        # Report results
        new_count = len(merged_data) - existing_count
        print(f"Added {new_count} new jobs to the dataset")
        print(f"Total jobs in dataset: {len(merged_data)}")
        print(f"Data saved to: {output_file_path}")
    else:
        print("\nNo new job data found since the last collection")
        if existing_count > 0:
            print(f"Existing dataset contains {existing_count} jobs")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
    collect_test_runtimes(
        session, args.url, repo_root, verbose=args.verbose, project_id=nre_project_id, start_date=start_date
    )
    collect_container_image_sizes(repo_root, CONTAINER_IMAGES, verbose=args.verbose)


if __name__ == "__main__":
    main()
