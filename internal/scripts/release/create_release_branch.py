# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import re
import sys

from typing import Tuple

import click
import git

from internal.scripts.release.utils import (
    GitError,
    VersionError,
    create_gitlab_branch,
    get_current_version,
    get_gitlab_project_info,
    get_gitlab_token,
    validate_release_branch_format,
    validate_version_sequence,
    version_update_commit_message,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def find_version_update_commit(repo: git.Repo, specified_version: str, current_version: str) -> Tuple[str, str]:
    """
    Find the most recent version update commit.

    Returns:
        Tuple of (commit_hash, old_version)
    """
    # Look for commits with "Change: Update NuRec version" in the message
    try:
        commits = list(
            repo.iter_commits(
                grep=version_update_commit_message(specified_version, current_version), max_count=1, no_merges=True
            )
        )

        if not commits:
            raise GitError(
                "Could not find a version update commit\n"
                f"Looking for commits with '{version_update_commit_message(specified_version, current_version)}' in the message"
            )

        commit = commits[0]
        commit_hash = commit.hexsha
        commit_message = commit.message.strip()

        logger.info(f"Found version update commit ({commit_hash}) message: {commit_message}")

        # Extract old version from message like "Change: Update NuRec version 25.09 -> 25.10"
        match = re.search(r"Update NuRec version (\d+\.\d+) ->", commit_message)
        if not match:
            raise VersionError(
                "Could not extract old version from commit message\n"
                "Expected format: 'Change: Update NuRec version X.Y -> A.B'"
            )

        old_version = match.group(1)
        return commit_hash, old_version

    except git.exc.GitError as e:
        raise GitError(f"Git error while searching for commits: {e}")


def confirm_action(release_branch: str, parent_commit: str) -> bool:
    """Ask user for confirmation."""
    print()
    print("This will:")
    print("  1. Locate the version update commit on main")
    print(f"  2. Use its parent commit as the release base: {parent_commit}")
    print(f"  3. Create release branch on GitLab: {release_branch} (no local branch)")
    print()

    try:
        response = input("Continue? [y/N]: ").strip().lower()
        return response in ["y", "yes"]
    except KeyboardInterrupt:
        print("\n❌ Aborted")
        return False


@click.command(context_settings={"show_default": True})
@click.option("-b", "--release-branch", required=True, help="Release branch in format release/YY.MM (required)")
@click.option("-y", "--yes", is_flag=True, help="Auto-confirm without prompting")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing git commands")
@click.option("--version-file", help="Path to VERSION_FILE", default="bazel/version/VERSION_FILE")
@click.option("--git-repo", help="Path to git repository", default=os.environ.get("BUILD_WORKSPACE_DIRECTORY"))
def create_release_branch(release_branch, yes, dry_run, version_file, git_repo):
    """Create release branch from the commit before the latest version update."""

    try:
        # Initialize repository
        repo = git.Repo(git_repo)

        # Ensure we're on main and up to date
        logger.info("Ensuring we're on main branch and up to date...")
        if not dry_run:
            origin = repo.remote("origin")
            origin.fetch("main:main")
            repo.heads.main.checkout()
        else:
            logger.info("[DRY-RUN] Would fetch origin main:main")
            logger.info("[DRY-RUN] Would checkout main")
        logger.info("Locating version update commit and base release commit...")

        # Validate release branch format (now required)
        validate_release_branch_format(release_branch)

        # Extract version from release branch
        release_version = release_branch.replace("release/", "")

        # Read current version from VERSION_FILE
        logger.info(f"Using version file: {version_file}")
        current_version = get_current_version(version_file)
        logger.info(f"Current version is {current_version}")

        # Release version should be one smaller than current version from main branch
        validate_version_sequence(release_version, current_version)
        logger.info(f"Validated version {release_version} for release branch {release_branch}")

        # Find the version update commit that updated TO the current version
        logger.info(f"Searching for commit: {version_update_commit_message(release_version, current_version)}")
        version_update_commit, old_version = find_version_update_commit(repo, release_version, current_version)

        # Get the commit just before the version update commit
        commit = repo.commit(version_update_commit)
        parent_commit = commit.parents[0].hexsha
        logger.info(f"Parent commit (release branch target): {parent_commit}")

        logger.info(f"Prepared release branch {release_branch} at base commit {parent_commit}")

        # Ask for confirmation unless auto-confirm is enabled
        if not yes:
            if not confirm_action(release_branch, parent_commit):
                logger.info("❌ Aborted")
                return 1

        # Create the release branch via GitLab API
        logger.info("Creating release branch on GitLab via API...")
        if not dry_run:
            gitlab_base_url, project_path = get_gitlab_project_info(repo)
            token = get_gitlab_token(gitlab_base_url)
            create_gitlab_branch(
                gitlab_base_url=gitlab_base_url,
                project_path=project_path,
                token=token,
                branch_name=release_branch,
                ref=parent_commit,
            )
        else:
            logger.info(f"[DRY-RUN] Would create GitLab branch: {release_branch} from base commit: {parent_commit}")

        if dry_run:
            logger.info("🔍 DRY RUN: All operations completed (no actual changes made)!")
            logger.info(f"Would have created GitLab branch: {release_branch}")
            logger.info(f"Would have based it on commit: {parent_commit}")
            logger.info("No local branch would be created.")
        else:
            logger.info("✅ Successfully created release branch!")
            logger.info(f"Branch: {release_branch}")
            logger.info(f"Based on commit: {parent_commit} (just before version update)")
            logger.info(f"Old version: {old_version}")

        return 0

    except (VersionError, GitError) as e:
        logger.error(f"❌ Error: {e}")
        return 1
    except git.exc.GitError as e:  # Add GitPython exceptions
        logger.error(f"❌ Git error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(create_release_branch())
