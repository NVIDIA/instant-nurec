# Release Management Scripts

This directory contains Python scripts for managing NRE releases. The scripts automate version updates, release branch creation, and image retagging for release candidates.

## Release Process

### 1. Update Version and Create Merge Request

First, create a version update MR to increment the version number:

```bash
# Auto-increment to next version
bazel run //internal/scripts/release:update-version-mr

# Update to specific version (optional override)
bazel run //internal/scripts/release:update-version-mr -- --new-version 25.09

# Short form with auto-confirm
bazel run //internal/scripts/release:update-version-mr -- -v 25.09 -y

# Dry-run (no git commands are executed)
bazel run //internal/scripts/release:update-version-mr -- --dry-run

# Specify version file path if needed
bazel run //internal/scripts/release:update-version-mr -- --version-file bazel/version/VERSION_FILE

# Specify git repository path if needed
bazel run //internal/scripts/release:update-version-mr -- --git-repo /path/to/repo
```

This script will:

- Auto-increment the version (or validate a provided version is later than current)
- Create a new branch with the version update
- Commit the changes and push the branch
- Create a merge request with the "release activities" label

### 2. Merge the Version Update MR

**Important**: The version update MR must be merged before proceeding to the next step. Wait for the MR to be reviewed and merged into the `main` branch.

### 3. Create Release Branch

After the version update MR is merged, create the release branch:

```bash
# Create release branch (required format: release/YY.MM)
bazel run //internal/scripts/release:create-release-branch -- --release-branch release/25.09

# Short form with auto-confirm
bazel run //internal/scripts/release:create-release-branch -- -b release/25.09 -y

# Dry-run (no git commands are executed)
bazel run //internal/scripts/release:create-release-branch -- -b release/25.09 --dry-run

# Specify version file path if needed
bazel run //internal/scripts/release:create-release-branch -- -b release/25.09 --version-file bazel/version/VERSION_FILE

# Specify git repository path if needed
bazel run //internal/scripts/release:create-release-branch -- -b release/25.09 --git-repo /path/to/repo
```

This script will:

- Validate the release version is current version - 1
- Find the version update commit that updated TO the current version
- Use the parent commit as the release base
- Create the release branch on GitLab (no local branch)

The release branch will be based on the state of the code before the version was incremented, ensuring it contains the correct version for the release.

## Script Options

### update-version-mr

- `-v, --new-version VERSION`: Target version in YY.MM format (optional; defaults to current version + 1)
- `-y, --yes`: Auto-confirm without prompts
- `--dry-run`: Show what would be done without executing git commands
- `--version-file PATH`: Path to VERSION_FILE (default: bazel/version/VERSION_FILE)
- `--git-repo PATH`: Path to git repository (default: BUILD_WORKSPACE_DIRECTORY)

### create-release-branch

- `-b, --release-branch BRANCH`: Release branch name in format `release/YY.MM` (required)
- `-y, --yes`: Auto-confirm without prompts
- `--dry-run`: Show what would be done without executing git commands
- `--version-file PATH`: Path to VERSION_FILE (default: bazel/version/VERSION_FILE)
- `--git-repo PATH`: Path to git repository (default: BUILD_WORKSPACE_DIRECTORY)

### retag-release-images

- `-t, --source-tag TAG`: Source image tag (required, e.g., 25.10.5-abc12345)
- `-n, --rc-number NUMBER`: Release candidate number (default: 1)
- `-y, --yes`: Auto-confirm all pushes without prompting
- `--dry-run`: Show what would be done without executing docker commands

### generate-sbom

- `--image IMAGE`: Docker image reference to scan (required)
- `--out PATH`: Output SPDX JSON path (required)
- `--dry-run`: Show what would be done without executing
- `--use-netrc`: Use .netrc credentials for the image registry
- `--netrc-path PATH`: Path to .netrc with registry credentials (default: `~/.netrc`)
- `--registry-host HOST`: Override registry host used for .netrc lookup

### download-new-sources

- `--base-image IMAGE`: Base Docker image reference (required if generating SBOMs)
- `--release-image IMAGE`: Release Docker image reference (required if generating SBOMs)
- `--base-sbom PATH`: Base SPDX SBOM path (optional if generating SBOMs)
- `--release-sbom PATH`: Release SPDX SBOM path (optional if generating SBOMs)
- `--output-dir PATH`: Output directory for source downloads (required)
- `--generate-sbom`: Force SBOM generation even if SBOM paths are provided
- `--dry-run`: Show what would be done without executing downloads
- `--continue-on-error`: Continue when a package download fails
- `--use-netrc`: Use .netrc credentials for image registries
- `--netrc-path PATH`: Path to .netrc with registry credentials (default: `~/.netrc`)
- `--registry-host HOST`: Override registry host used for .netrc lookup

## Version Format

All versions follow the `YY.MM` format where:

- `YY`: Two-digit year (e.g., 25 for 2025)
- `MM`: Two-digit month with leading zero (01-12)

Examples: `25.09`, `25.12`, `26.01`

## Automated Processes

Release candidate image retagging can be automated by GitLab CI when RC tags (e.g., `25.10-rc1`) are created, or run manually using the `retag-release-images` script.

## SBOM Diff Tool

### Overview

The SBOM Diff Tool compares two SBOM files in CycloneDX format and generates a CSV report containing:

- **New packages**: Packages that exist in the modified SBOM but not in the base SBOM (compared by package name only)
- **Version changes**: Existing packages that have different versions between the two SBOMs (optional)

### Usage

#### Basic Usage (New Packages Only)

```bash
bazel run //internal/scripts/release:sbom-diff -- <base_sbom.json> <modified_sbom.json> <output.csv>
```

This will generate a CSV file containing only new packages found in the modified SBOM.

#### With Version Change Detection

```bash
bazel run //internal/scripts/release:sbom-diff -- --identify-version-changes <base_sbom.json> <modified_sbom.json> <output.csv>
```

This will generate a CSV file containing both new packages and version changes.

## SBOM Source Downloads

Generate SPDX SBOMs for a base and release image, diff them, and download source archives
for packages that appear only in the release image.

### Generate SBOMs (standalone)

```bash
bazel run //internal/scripts/release:generate-sbom -- --image <image> --out /path/to/output.spdx.json
```

### Download sources (generate SBOMs automatically)

```bash
bazel run //internal/scripts/release:download-new-sources -- \
  --base-image <base-image> \
  --release-image <release-image> \
  --output-dir /path/to/downloads
```

### Download sources (use existing SBOMs)

```bash
bazel run //internal/scripts/release:download-new-sources -- \
  --base-sbom /path/to/base.spdx.json \
  --release-sbom /path/to/release.spdx.json \
  --output-dir /path/to/downloads
```

#### Help

```bash
bazel run //internal/scripts/release:sbom-diff -- --help
```

### Arguments

- `base_sbom`: Path to the base SBOM file (JSON format)
- `modified_sbom`: Path to the modified SBOM file (JSON format)
- `output_csv`: Path where the output CSV file will be written
- `--identify-version-changes`: Optional flag to include version changes in the output

### Output Format

The generated CSV file contains the following columns:

| Column                                       | Description                                | Status                        |
| -------------------------------------------- | ------------------------------------------ | ----------------------------- |
| Package / Component Name                     | Name of the package extracted from PURL    | Populated                     |
| Version                                      | Version of the package extracted from PURL | Populated                     |
| License                                      | Package license information                | Empty (for future use)        |
| Link to Component's License                  | URL to license text                        | Empty (for future use)        |
| Method of Distribution                       | How the package is distributed             | Empty (for future use)        |
| Usage Method with NV proprietary code        | Integration details                        | Empty (for future use)        |
| Comments                                     | Additional notes                           | Populated for version changes |
| Location where component was downloaded from | Download source                            | Empty (for future use)        |
| Link to internal IT Controlled Repository    | Internal repository link                   | Empty (for future use)        |
| OSRB Bug ID                                  | Open Source Review Board ID                | Empty (for future use)        |

#### Output Structure

1. **New Packages Section**: Lists packages that don't exist in the base SBOM (by name)
2. **Empty Separator Row**: Only present if version changes are included (contains only commas)
3. **Version Changes Section**: Lists packages with different versions (format: "old_version -> new_version")

## SBOM License Management

This section describes tools for extracting, detecting, and concatenating license information from SBOM files.

### Overview

The license management workflow involves two main scripts:

1. **update-license-overrides**: Automatically detects licenses for packages with unknown licenses by querying PyPI, Ubuntu repositories, and GitHub APIs.
2. **fetch-licenses**: Fetches license texts and generates a concatenated license file and summary report.

### Workflow

```bash
# 1. First, update license overrides for packages with unknown licenses
bazel run //internal/scripts/release:update-license-overrides -- sbom.spdx.json

# 2. Then, fetch all licenses and generate the concatenated file
bazel run //internal/scripts/release:fetch-licenses -- sbom.spdx.json
```

### update-license-overrides

Automatically detects licenses for packages listed as `NOASSERTION` or `LicenseRef-*` in the SBOM by querying external sources.

#### Usage

```bash
# Basic usage
bazel run //internal/scripts/release:update-license-overrides -- sbom.dvl.inference.spdx.json

# With GitHub token for higher rate limits (5000 req/hour vs 60)
bazel run //internal/scripts/release:update-license-overrides -- --github-token <token> sbom.dvl.inference.spdx.json

# Using environment variable for token
export GITHUB_TOKEN=<token>
bazel run //internal/scripts/release:update-license-overrides -- sbom.dvl.inference.spdx.json

# Limit GitHub API requests
bazel run //internal/scripts/release:update-license-overrides -- --max-github-requests 100 sbom.dvl.inference.spdx.json

# Dry run (show what would be updated without writing)
bazel run //internal/scripts/release:update-license-overrides -- --dry-run sbom.dvl.inference.spdx.json
```

If the GitHub request limit is exceeded, the script will prompt the user for further action. It is generally failure-tolerant and will make a
best-effort attempt to complete the `license_overrides.toml` file. If possible, it will populate the `licenses/` directory with some packages'
unconventional software licenses for later use with `release:fetch-licenses`.

#### Options

| Option                    | Description                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| `spdx_file`               | Path to SPDX JSON file with package information (required)       |
| `--dry-run`               | Show what would be updated without writing to file               |
| `--github-token TOKEN`    | GitHub API token for higher rate limits                          |
| `--max-github-requests N` | Maximum GitHub API requests to make (default: unlimited)         |
| `-y, --yes`               | Non-interactive mode: auto-continue on rate limits/auth failures |

#### How It Works

The script queries multiple sources in parallel:

- **PyPI**: For Python packages, queries the PyPI API for license metadata
- **Ubuntu/Debian**: For system packages, fetches copyright files from changelogs.ubuntu.com and metadata.ftp-master.debian.org
- **GitHub**: For Go modules and packages with GitHub URLs, queries the GitHub License API

Results are saved to `license_overrides.toml` in two sections:

- `[user_overrides]`: Manual overrides that take priority (preserved across runs)
- `[auto_overrides]`: Automatically detected licenses (updated each run)

### fetch-licenses

Fetches license texts for all packages and generates output files.

#### Usage

```bash
# Basic usage (outputs to temp directory)
bazel run //internal/scripts/release:fetch-licenses -- sbom.dvl.inference.spdx.json

# Specify output directory
bazel run //internal/scripts/release:fetch-licenses -- sbom.dvl.inference.spdx.json ./output

# Continue even if some downloads fail
bazel run //internal/scripts/release:fetch-licenses -- --continue-on-error sbom.dvl.inference.spdx.json
```

#### Options

| Option                | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `input_file`          | Path to SPDX JSON file with package information (required) |
| `output_dir`          | Output directory (optional, defaults to temp directory)    |
| `--continue-on-error` | Continue processing even if some downloads fail            |

#### Output Files

The script generates:

- **`all_licenses_concatenated.txt`**: All license texts concatenated into a single file
- **`licenses_summary.txt`**: Summary table of all packages and their licenses
- **`licenses/`**: Directory containing individual license files

#### License Categories

The summary reports licenses in three categories:

- **Open source**: Apache, MIT, BSD, GPL, LGPL, ISC, Python, Zlib, etc.
- **Proprietary**: NVIDIA, commercial licenses
- **Unknown**: Licenses that could not be detected or categorized

### license_overrides.toml

Manual and automatic license overrides are stored in this TOML file.

#### Structure

```toml
[user_overrides]
# Manual overrides - these take priority and are preserved across runs
"package-name" = "MIT"
"another-package" = "Apache"

[auto_overrides]
# Automatically detected licenses - updated by update-license-overrides
"detected-package" = "BSD"
```

#### Adding Manual Overrides

If a package's license cannot be detected automatically, add it to the `[user_overrides]` section:

```toml
[user_overrides]
"my-package" = "MIT"
```

The script will report inconsistencies if a detected license differs from a user override.

### Custom License Files

For packages with non-standard licenses (marked as "OTHER"), the actual license text is saved to `internal/scripts/release/licenses/` with the naming convention:

```
<package-name>-<version>-LICENSE-OTHER.txt
```

These files are automatically used by `fetch-licenses` when generating the concatenated output.
