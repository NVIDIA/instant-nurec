# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from pathlib import Path

import click
import git

from internal.scripts.release.utils import (
    GitError,
    VersionError,
    from_version_to_major_minor,
    get_current_version,
    get_gitlab_project_info,
    get_gitlab_token,
    validate_version_format,
    version_increment,
    version_update_commit_message,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def update_version_file(
    current_version: str,
    new_version: str,
    dry_run: bool = False,
    version_file_path: str = "bazel/version/VERSION_FILE",
) -> None:
    """Update VERSION_FILE with new version."""
    version_file = Path(version_file_path)

    if not version_file.exists():
        raise VersionError(f"VERSION_FILE not found at {version_file}")

    file_content = version_file.read_text()

    # Extract major and minor from current and new versions
    current_major, current_minor = from_version_to_major_minor(current_version)
    new_major, new_minor = from_version_to_major_minor(new_version)

    # Replace VERSION_MAJOR and VERSION_MINOR
    content = re.sub(f"^VERSION_MAJOR={current_major}$", f"VERSION_MAJOR={new_major}", file_content, flags=re.MULTILINE)
    content = re.sub(f"^VERSION_MINOR={current_minor}$", f"VERSION_MINOR={new_minor}", content, flags=re.MULTILINE)

    if dry_run:
        logger.info(f"[DRY-RUN] Would update {version_file}:")
        logger.info(f"From:\n{reindent(file_content)}")
        logger.info(f"To:\n{reindent(content)}")
    else:
        version_file.write_text(content)
        logger.info(f"Updated {version_file}:")
        logger.info(reindent(content))


def get_clean_username() -> str:
    """Get current user without 'local-' prefix."""
    user = os.environ.get("USER", "unknown")
    return user.replace("local-", "")


def confirm_action(branch_name: str, current_version: str, new_version: str, mr_only: bool = False) -> bool:
    """Ask user for confirmation."""
    print()
    if mr_only:
        print("This will:")
        print(f"  1. Create a merge request from {branch_name} to main")
        print(f"     (version update to {new_version})")
    else:
        print("This will:")
        print(f"  1. Create branch: {branch_name}")
        print(f"  2. Update version from {current_version} to {new_version}")
        print("  3. Commit and push the changes")
        print("  4. Create a merge request")
    print()

    try:
        response = input("Continue? [y/N]: ").strip().lower()
        return response in ["y", "yes"]
    except KeyboardInterrupt:
        print("\n❌ Aborted")
        return False


def reindent(s, numSpaces=4):
    """Reindent a (multi-line)string with a given number of spaces."""
    return "\n".join([(numSpaces * " ") + str.lstrip(line) for line in s.split("\n")])


def create_gitlab_merge_request(
    gitlab_base_url: str,
    project_path: str,
    token: str,
    source_branch: str,
    target_branch: str,
    title: str,
    labels: str,
) -> None:
    """Create a GitLab merge request via API."""
    project_id = urllib.parse.quote(project_path, safe="")
    api_url = f"{gitlab_base_url}/api/v4/projects/{project_id}/merge_requests"
    payload = urllib.parse.urlencode(
        {
            "source_branch": source_branch,
            "target_branch": target_branch,
            "title": title,
            "labels": labels,
        }
    ).encode("utf-8")

    request = urllib.request.Request(api_url, data=payload, method="POST")
    request.add_header("PRIVATE-TOKEN", token)
    request.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 409:
            logger.info("Merge request already exists for this branch; skipping creation.")
            return
        error_body = e.read().decode("utf-8")
        raise GitError(f"Failed to create merge request via GitLab API (HTTP {e.code}). {error_body}") from e
    except urllib.error.URLError as e:
        raise GitError(f"Failed to create merge request via GitLab API: {e}") from e

    try:
        data = json.loads(body)
        mr_url = data.get("web_url")
        if mr_url:
            logger.info(f"Merge request created: {mr_url}")
    except json.JSONDecodeError:
        logger.info("Merge request created, but response could not be parsed.")


@click.command(context_settings={"show_default": True})
@click.option(
    "-v",
    "--new-version",
    required=False,
    help="New version in YY.MM format (optional; defaults to current version + 1)",
)
@click.option("-y", "--yes", is_flag=True, help="Auto-confirm without prompting")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing git commands")
@click.option("--version-file", help="Path to VERSION_FILE", default="bazel/version/VERSION_FILE")
@click.option("--git-repo", help="Path to nurec git repository", default=os.environ.get("BUILD_WORKSPACE_DIRECTORY"))
@click.option(
    "--mr-only",
    is_flag=True,
    help="Only create the merge request (skip branch creation, version update, commit, and push)",
)
def update_version_mr(new_version, yes, dry_run, version_file, git_repo, mr_only):
    """Update NuRec version and create merge request."""
    try:
        # Initialize repository
        repo = git.Repo(git_repo)

        # Ensure we're on main and up to date
        if repo.head.is_detached:
            raise GitError("Repository is in a detached HEAD state. Checkout a branch and retry.")

        if dry_run:
            logger.info("🔍 DRY RUN: Showing what would be done...")

        current_version = ""
        if mr_only:
            if not new_version:
                raise click.UsageError("--new-version is required when using --mr-only")
            validate_version_format(new_version)
            logger.info("MR-only mode: skipping branch creation, version update, and push.")
            logger.info(f"New version: {new_version}")
        else:
            logger.info("Ensuring we are on `main` it's up to date...")

            if not dry_run:
                origin = repo.remote("origin")
                origin.fetch("main")
                if "main" not in repo.heads:
                    if "origin/main" not in repo.refs:
                        raise GitError("Could not find origin/main to create local main.")
                    repo.create_head("main", repo.refs["origin/main"])
                repo.heads.main.checkout()
                try:
                    repo.git.merge("--ff-only", "origin/main")
                except git.exc.GitCommandError as e:
                    raise GitError(
                        "Fast-forward merge failed. Your local main diverged from origin/main. "
                        "Please rebase or reset main and try again."
                    ) from e
            else:
                logger.info("[DRY-RUN] Would fetch origin main")
                logger.info("[DRY-RUN] Would checkout main")
                logger.info("[DRY-RUN] Would fast-forward main to origin/main")

            # Read current version
            logger.info(f"Using version file: {version_file}")
            current_version = get_current_version(version_file)
            logger.info(f"Current version: {current_version}")

            # Determine new version
            if new_version:
                validate_version_format(new_version)
                current_major, current_minor = from_version_to_major_minor(current_version)
                new_major, new_minor = from_version_to_major_minor(new_version)
                if (new_major, new_minor) <= (current_major, current_minor):
                    raise VersionError(
                        f"New version ({new_version}) must be later than current version ({current_version})"
                    )
                logger.info(f"New version: {new_version}")
            else:
                new_version = version_increment(current_version)
                logger.info(f"New version (auto-increment): {new_version}")

        # Create branch name
        branch_name = f"dev/{get_clean_username()}/update_version_to_{new_version.replace('.', '')}"
        logger.info(f"Preparing branch: {branch_name} off main")

        # Ask for confirmation unless auto-confirm is enabled
        if not yes:
            current_ver_display = current_version if not mr_only else ""
            if not confirm_action(branch_name, current_ver_display, new_version, mr_only=mr_only):
                logger.info("❌ Aborted")
                return 1

        if not mr_only:
            # Create and checkout new branch (or reuse existing one)
            branch_exists_local = branch_name in repo.heads
            remote_branch_exists = False
            if not dry_run:
                remote_branch_exists = bool(repo.git.ls_remote("--heads", "origin", branch_name).strip())
            else:
                logger.info(f"[DRY-RUN] Would check if branch exists locally/remotely: {branch_name}")

            if not dry_run:
                if branch_exists_local:
                    logger.info(f"Using existing local branch: {branch_name}")
                    repo.heads[branch_name].checkout()
                elif remote_branch_exists:
                    logger.info(f"Using existing remote branch: {branch_name}")
                    repo.git.fetch("origin", f"{branch_name}:{branch_name}")
                    repo.heads[branch_name].set_tracking_branch(repo.remote("origin").refs[branch_name])
                    repo.heads[branch_name].checkout()
                else:
                    logger.info(f"Creating new branch: {branch_name}")
                    new_branch = repo.create_head(branch_name, repo.heads.main)
                    new_branch.checkout()
            else:
                logger.info(f"[DRY-RUN] Would create or checkout branch: {branch_name}")

            # Update VERSION_FILE if needed
            branch_version = get_current_version(version_file)
            if branch_version == new_version:
                logger.info(f"Version already updated in {version_file}: {branch_version}")
            else:
                update_version_file(current_version, new_version, dry_run=dry_run, version_file_path=version_file)

            # Commit changes and publish branch
            logger.info("Committing changes and pushing branch to origin...")
            commit_message = version_update_commit_message(current_version, new_version)

            if not dry_run:
                commit_log = repo.git.log("--pretty=%B", "--", version_file)
                commit_exists = commit_message in commit_log
                version_file_dirty = repo.is_dirty(path=version_file)

                if version_file_dirty:
                    repo.index.add([version_file])
                    repo.index.commit(commit_message)
                else:
                    if branch_version == new_version and commit_exists:
                        logger.info("Version commit already exists; skipping commit.")
                    elif branch_version == new_version and not commit_exists:
                        raise GitError(
                            "Version file already updated but expected commit not found. "
                            "Please verify branch history and retry."
                        )
                    else:
                        raise GitError("VERSION_FILE was not updated as expected.")

                origin = repo.remote("origin")
                if remote_branch_exists:
                    origin.push(branch_name)
                else:
                    origin.push(branch_name, set_upstream=True)
            else:
                logger.info(f"[DRY-RUN] Would add file: {version_file}")
                logger.info(f"[DRY-RUN] Would commit with message: {commit_message}")
                logger.info(f"[DRY-RUN] Would push branch: {branch_name}")

        # Create MR
        logger.info("Creating merge request...")
        if mr_only:
            mr_title = f"Change: Update NuRec version to {new_version}"
        else:
            mr_title = version_update_commit_message(current_version, new_version)

        if not dry_run:
            gitlab_base_url, project_path = get_gitlab_project_info(repo)
            token = get_gitlab_token(gitlab_base_url)
            create_gitlab_merge_request(
                gitlab_base_url=gitlab_base_url,
                project_path=project_path,
                token=token,
                source_branch=branch_name,
                target_branch="main",
                title=mr_title,
                labels="release activities",
            )
        else:
            logger.info(f"[DRY-RUN] Would create MR with title: {mr_title}")

        if dry_run:
            logger.info("🔍 DRY RUN: All operations completed (no actual changes made)!")
            if mr_only:
                logger.info(f"Would have created MR: {branch_name} -> main")
            else:
                logger.info(f"Would have created branch: {branch_name}")
                logger.info(f"Would have updated version: {current_version} to {new_version}")
        else:
            logger.info("✅ Successfully created merge request!")
            logger.info(f"Branch: {branch_name}")

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
    sys.exit(update_version_mr())
