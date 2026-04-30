#!/usr/bin/env python3

# This script is written as a python replacement for the existing bash script
# NRE/bazel/version/version_string.sh, and is expected to return the same
# results. The intended usecase is to generate a version string during the
# windows build of NREND, where is not possible to run a bash script.

import os
import pathlib
import subprocess


def readVersionFile(fn):
    """Parse version file, extract key-value pairs and return as dict."""
    kv = {}
    with open(fn, "r") as fp:
        for line in fp:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                kv[key] = value
    return kv


def runCommand(cmd):
    """Run shell command and return output as a string."""
    return subprocess.check_output(cmd).decode().strip()


def isGitTreeDirty():
    """Check whether the local git work tree is dirty."""
    try:
        subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], check=True)
        return False
    except subprocess.CalledProcessError:
        return True


# Path to version file containing NRE major/minor versions
versionPath = pathlib.Path(__file__).resolve().parent
versionFile = os.path.join(versionPath, "../../../../bazel/version/VERSION_FILE")

# Get the last commit hash where the version file was changed
versionHash = runCommand(["git", "log", "-n", "1", "--format=format:%H", "--", versionFile])

# Count the number of merge commits since the last commit and use this as patch version
patchNumber = runCommand(["git", "rev-list", "--count", f"{versionHash}..HEAD", "--merges"])

# Get git commit hash of current HEAD
commitShaShort = runCommand(["git", "rev-parse", "--short=8", "HEAD"])

# Get suffix to indicate local work tree is dirty
dirtySuffix = "+dirty" if isGitTreeDirty() else ""

# Read key-value pairs from version file and retrieve major and minor numbers
kv = readVersionFile(versionFile)
versionMajor = kv["VERSION_MAJOR"]
versionMinor = kv["VERSION_MINOR"]

# Output final version string
print(f"{versionMajor}.{versionMinor}.{patchNumber}-{commitShaShort}{dirtySuffix}")
