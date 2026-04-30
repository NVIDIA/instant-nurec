# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Script to create JIRA Key Result issues from a text file of Acceptance Criteria.
Each Key Result will be linked to the Epic with a "blocks" relationship.

Usage:
    python create_jira_key_results.py <ac_file> <initiative_url>

Requirements:
    - .netrc file with JIRA credentials (machine, login, password)
    - JIRA instance with "Key Result" issue type
    - Epic/Initiative must exist in JIRA

Example .netrc entry:
    machine your-jira-instance.com
    login your-username
    password your-api-token
"""

import argparse
import netrc
import re
import sys
import urllib.parse

from typing import List, Tuple

import requests


class JiraKeyResultCreator:
    def __init__(self, jira_base_url: str, username: str, api_token: str):
        """Initialize JIRA client with authentication."""
        self.jira_base_url = jira_base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (username, api_token)
        self.session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    def extract_issue_key_from_url(self, initiative_url: str) -> str:
        """Extract JIRA issue key from full URL."""
        # Match patterns like /browse/PROJ-123 or /projects/PROJ/issues/PROJ-123
        patterns = [r"/browse/([A-Z]+-\d+)", r"/issues/([A-Z]+-\d+)", r"selectedIssue=([A-Z]+-\d+)"]

        for pattern in patterns:
            match = re.search(pattern, initiative_url)
            if match:
                return match.group(1)

        raise ValueError(f"Could not extract issue key from URL: {initiative_url}")

    def get_issue_project_key(self, issue_key: str) -> str:
        """Get the project key for an issue."""
        url = f"{self.jira_base_url}/rest/api/2/issue/{issue_key}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()["fields"]["project"]["key"]

    def strip_enumeration(self, text: str) -> str:
        """Strip enumeration patterns from AC text."""
        # Remove patterns like "1.", "1)", "- ", "* ", "a.", "i.", etc.
        patterns = [
            r"^\s*\d+\.\s*",  # "1. "
            r"^\s*\d+\)\s*",  # "1) "
            r"^\s*[a-z]\.\s*",  # "a. "
            r"^\s*[ivx]+\.\s*",  # "i. ", "ii. ", etc.
            r"^\s*[-*•]\s*",  # "- ", "* ", "• "
            r"^\s*\([a-z]\)\s*",  # "(a) "
            r"^\s*\(\d+\)\s*",  # "(1) "
        ]

        cleaned_text = text.strip()
        for pattern in patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

        return cleaned_text.strip()

    def read_acceptance_criteria(self, file_path: str) -> List[str]:
        """Read and parse acceptance criteria from file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {file_path}")

        # Strip enumeration and filter out empty lines
        acs = []
        for line in lines:
            cleaned = self.strip_enumeration(line)
            if cleaned:  # Only add non-empty lines
                acs.append(cleaned)

        return acs

    def create_key_result(self, summary: str, project_key: str) -> str:
        """Create a Key Result issue in JIRA."""
        issue_data = {
            "fields": {"project": {"key": project_key}, "summary": summary, "issuetype": {"name": "Key Result"}}
        }

        url = f"{self.jira_base_url}/rest/api/2/issue"
        response = self.session.post(url, json=issue_data)

        if response.status_code == 201:
            issue_key = response.json()["key"]
            issue_url = f"{self.jira_base_url}/browse/{issue_key}"
            print(f"✓ Created Key Result: {issue_key}")
            print(f"  Summary: {summary}")
            print(f"  Link: {issue_url}")
            return issue_key
        else:
            error_msg = f"Failed to create issue: {response.status_code} - {response.text}"
            print(f"✗ Error creating Key Result for: {summary}")
            print(f"  {error_msg}")
            raise requests.RequestException(error_msg)

    def link_issue_to_epic(self, key_result_key: str, epic_key: str) -> bool:
        """Link Key Result to Epic with 'blocks' relationship."""
        link_data = {
            "type": {"name": "Blocks"},
            "outwardIssue": {"key": epic_key},
            "inwardIssue": {"key": key_result_key},
        }

        url = f"{self.jira_base_url}/rest/api/2/issueLink"
        response = self.session.post(url, json=link_data)

        if response.status_code == 201:
            print(f"  ✓ Linked to Epic: {epic_key} (Epic is blocked by {key_result_key})")
            return True
        else:
            print(f"  ✗ Failed to link to Epic: {response.status_code} - {response.text}")
            return False

    def create_key_results_from_file(self, ac_file: str, initiative_url: str) -> Tuple[List[str], str]:
        """Create Key Results from AC file and return created issue keys and initiative key."""
        print(f"Reading acceptance criteria from: {ac_file}")
        acs = self.read_acceptance_criteria(ac_file)
        print(f"Found {len(acs)} acceptance criteria")

        print(f"Extracting initiative key from: {initiative_url}")
        initiative_key = self.extract_issue_key_from_url(initiative_url)
        print(f"Initiative key: {initiative_key}")

        print("Getting project information...")
        project_key = self.get_issue_project_key(initiative_key)
        print(f"Project key: {project_key}")

        print(f"\nCreating {len(acs)} Key Result issues...")
        created_issues = []

        for i, ac in enumerate(acs, 1):
            try:
                print(f"\n[{i}/{len(acs)}] Creating Key Result...")
                issue_key = self.create_key_result(ac, project_key)

                # Link to Epic with "blocks" relationship
                print(f"  Linking {issue_key} to Epic {initiative_key}...")
                link_success = self.link_issue_to_epic(issue_key, initiative_key)

                created_issues.append(issue_key)
                if not link_success:
                    print(f"  ⚠️  Key Result created but linking failed")

            except Exception as e:
                print(f"✗ Failed to create Key Result {i}: {e}")
                continue

        print(f"\n🎉 Successfully created {len(created_issues)} out of {len(acs)} Key Result issues!")
        print(f"    Each Key Result is linked to Epic {initiative_key} with 'blocks' relationship.")
        return created_issues, initiative_key


def get_jira_credentials(jira_url: str) -> Tuple[str, str]:
    """Get JIRA credentials from .netrc file."""
    try:
        parsed_url = urllib.parse.urlparse(jira_url)
        hostname = parsed_url.netloc

        netrc_auth = netrc.netrc()
        auth_info = netrc_auth.authenticators(hostname)

        if not auth_info:
            raise ValueError(f"No credentials found in .netrc for {hostname}")

        username, _, password = auth_info
        return username, password

    except FileNotFoundError:
        raise FileNotFoundError("Could not find .netrc file in home directory")
    except Exception as e:
        raise ValueError(f"Error reading .netrc: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Create JIRA Key Result issues from Acceptance Criteria file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_jira_key_results.py initiative-requirements.txt "https://jira.company.com/browse/INIT-123"

Note: Requires .netrc file with JIRA credentials:
    machine jirasw.nvidia.com
      login your-username (without @nvidia.com) 
      password your-api-token

The script creates Key Result issues and links them to the Epic with "blocks" relationship.
This means the Epic is blocked by each Key Result.
        """,
    )

    parser.add_argument("ac_file", help="Text file containing acceptance criteria (one per line)")
    parser.add_argument(
        "initiative_url", help="Full JIRA URL to the Epic/Initiative that will be blocked by the Key Results"
    )

    args = parser.parse_args()

    try:
        # Extract JIRA base URL from initiative URL
        parsed_url = urllib.parse.urlparse(args.initiative_url)
        jira_base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Get credentials from .netrc
        username, api_token = get_jira_credentials(args.initiative_url)

        # Create JIRA client and process file
        creator = JiraKeyResultCreator(jira_base_url, username, api_token)
        created_issues, initiative_key = creator.create_key_results_from_file(args.ac_file, args.initiative_url)
        if created_issues:
            print(f"\n🎉 Process completed! Created {len(created_issues)} Key Result issues:")
            for issue_key in created_issues:
                issue_url = f"{jira_base_url}/browse/{issue_key}"
                print(f"  • {issue_key}: {issue_url}")
            print(f"\nAll Key Results are linked to Epic {initiative_key} with 'blocks' relationship.")
            print(f"Epic view: {jira_base_url}/browse/{initiative_key}")
        else:
            print(f"\n❌ No issues were created successfully.")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
