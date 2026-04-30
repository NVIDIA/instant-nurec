<!-- Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved. -->

# Guide to Updating Python Versions and Patches in rules_python

This guide explains how to update Python versions and patches in our Bazel
rules_python setup. Our Python versions are sourced from [Astral's
python-build-standalone](https://github.com/astral-sh/python-build-standalone)
project, which provides pre-built Python binaries for multiple platforms.

## Overview

### What is python-build-standalone?

Astral's python-build-standalone project provides:

- Pre-compiled Python binaries for multiple platforms and architectures
- Consistent builds across different operating systems
- Optimized Python builds with various compile-time flags
- Regular updates including security patches

### How rules_python uses python-build-standalone

Instead of downloading Python from python.org, rules_python can be configured to
use binaries from python-build-standalone. This provides:

- Better cross-platform consistency
- Faster downloads (pre-compiled binaries)
- Support for more platforms (e.g., musl libc, ARM architectures)

## Directory Structure

```
bazel/rules_python/
├── get_python_hashes.sh                    # General script to fetch SHA256 hashes
├── rules_python.patch                      # Patches for rules_python
├── BUILD.bazel                             # Build configuration
└── update_python_versions.md               # This guide
```

## Step-by-Step Guide to Update Python Versions

### 1. Find Available Python Versions

Visit the
[python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases)
releases page to find available Python versions.

Each release is tagged with a date (e.g., `20250604`) and contains multiple
Python versions.

### 2. Use the Hash Fetching Script

We provide a general-purpose script `get_python_hashes.sh` that accepts the
Python version and release date as arguments:

```bash
# Usage
./get_python_hashes.sh <python_version> <release_date>

# Example for Python 3.11.13 from release 20250604
./get_python_hashes.sh 3.11.13 20250604
```

The script will:

- Fetch SHA256 hashes for all supported platforms
- Display the hashes in a readable format
- Output a pre-formatted `versions.bzl` entry at the end

### 3. Run the Script to Get Hashes

First, make sure the script is executable:

```bash
chmod +x get_python_hashes.sh
```

Then run it with your desired Python version and release date:

```bash
# Example for Python 3.12.8 from release 20250115
./get_python_hashes.sh 3.12.8 20250115

# Example for Python 3.11.13 from release 20250604
./get_python_hashes.sh 3.11.13 20250604
```

The script will output:

1. Individual platform hashes for verification
2. A formatted entry ready to be added to `versions.bzl`

### 4. Update rules_python.patch

The patch file `rules_python.patch` modifies `python/versions.bzl` in
`@rules_python` to add new Python versions. The structure for each version is:

```starlark
"X.Y.Z": {
    "url": "YYYYMMDD/cpython-{python_version}+YYYYMMDD-{platform}-{build}.tar.gz",
    "sha256": {
        "aarch64-apple-darwin": "hash_here",
        "aarch64-unknown-linux-gnu": "hash_here",
        "ppc64le-unknown-linux-gnu": "hash_here",
        "riscv64-unknown-linux-gnu": "hash_here",
        "s390x-unknown-linux-gnu": "hash_here",
        "x86_64-apple-darwin": "hash_here",
        "x86_64-pc-windows-msvc": "hash_here",
        "x86_64-unknown-linux-gnu": "hash_here",
        "x86_64-unknown-linux-musl": "hash_here",
    },
    "strip_prefix": "python",
},
```
