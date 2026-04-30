# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import hashlib
import logging
import os
import re
import subprocess

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import shortuuid


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def parse_wandb_sweep_string(sweep_str: str) -> tuple[str, str, str]:
    """Parse a wandb sweep string <entity>/<project>/<sweep_id> into <entity>, <project>, <sweep_id>"""
    result = sweep_str.split(sep="/")
    assert len(result) == 3
    return result[0], result[1], result[2]


def parse_image_name(image_str: str) -> tuple[str, str, str, Optional[str], Optional[str]]:
    """Parse a docker image name of the form <registry>/<path>/<name>[:<tag>][@<digest>] into <registry>, <path>, <name>, <tag>, <digest>"""
    assert len(image_str) > 0
    result = image_str.split(sep="@")
    if len(result) > 2:
        raise DockerImageNameError("multiple @ in docker image name")
    digest = None if len(result) == 1 else result[1]
    result = result[0].split(sep=":")
    if len(result) > 2:
        raise DockerImageNameError("multiple : in docker image name, excluding the digest")
    tag = None if len(result) < 2 or result[1] == "" else result[1]
    result = result[0].split(sep="/")
    if len(result) < 3 or result[0] == "" or result[1] == "" or result[2] == "":
        raise DockerImageNameError("invalid docker image path")
    name = result[-1]
    registry = result[0]
    path = "/".join(result[1:-1])
    check = f"{registry}/{path}/{name}"
    if tag is not None:
        check += f":{tag}"
    if digest is not None:
        check += f"@{digest}"
    assert check == image_str
    return registry, path, name, tag, digest


def search_and_replace(lst: List[str], search_replace: Dict[str, str]) -> List[str]:
    """Substitute all occurrences of strings in a list of strings"""
    out = lst.copy()
    for idx, item in enumerate(out):
        if not isinstance(item, str):
            continue
        for pattern, replace in search_replace.items():
            item = re.sub(pattern, replace, item)
        out[idx] = item
    return out


def run_command_as_subprocess(cmd: list[str], env: Optional[dict] = None, cwd: Optional[Path] = None) -> None:
    """Run a command via subprocess and capture its output line by line"""
    # Using Popen in order to capture/monitor live progress output from the called process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        encoding="utf-8",
        env=env,
        bufsize=1,
        cwd=cwd,
    )
    retval: Optional[int] = None
    while True:
        stdout = process.stdout
        assert stdout is not None
        output = stdout.readline()
        retval = process.poll()
        if output == "" and retval is not None:
            break
        if output:
            log.info(output.strip())

    assert retval is not None
    if retval != 0:
        raise SubprocessError("Subprocess failed")


class ArgError(Exception):
    def __init__(self, message):
        super().__init__(message)


class DockerImageNameError(Exception):
    def __init__(self, message):
        super().__init__(message)


class SubprocessError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ConfError(Exception):
    def __init__(self, message):
        super().__init__(message)


def mount_dict_to_string(mount_dict: dict[str, str]) -> str:
    """Build mount string from dictionary"""
    mount_str = []
    # NOTE The order of the mounts is important, later mounts will overwrite earlier mounts. The idea is
    #      we want to respect user specified mounts first, thus we append them in the end.
    mount_str.append("/lustre:/lustre")
    for k, v in mount_dict.items():
        mount_str.append(k + ":" + v)
    return ",".join(mount_str)


def write_run_bash(filename: str, exp_cmds: list[str], pre_cmds: list[str] = [], post_cmds: list[str] = []) -> None:
    """
    Create a bash file, include all the pre_cmd
    and given args;

    args:
        filename: string, name the file to write
        exp_cmds: list of main commands to run
        pre_cmds: list of commands used to setup the env
        post_cmds: list of commands run after training finished
    """
    log.info(f"Write bash file: {filename}")
    os.makedirs(Path(filename).parent, exist_ok=True)

    with open(filename, "w") as f:
        for e in pre_cmds:
            f.write(e + "\n")

        for e in exp_cmds:
            f.write(e + "\n")

        for e in post_cmds:
            f.write(e + "\n")


def parse_command_config(cmd: str) -> Optional[str]:
    """Parse a command of the form ... bazelisk run //:run -- --config-name=... and return the config name"""
    pattern = r".*(?:bazelisk|bazel)\s+run\s+[^\s]+\s+--\s+--config-name[=\s]+(\S+).*"
    match = re.search(pattern, cmd)
    if match:
        config_name = match.group(1)
        # return the basename of the config name without the path and extension
        return os.path.basename(config_name).split(".")[0]
    else:
        return None


def parse_command_dataset(cmd: str) -> Optional[str]:
    """Parse a command of the form ... bazelisk run //:run -- ... dataset.path=... and return the dataset path"""
    pattern = r".*(?:bazelisk|bazel)\s+run\s+[^\s]+\s+--\s+.*dataset\.path[=\s]+(\S+).*"
    match = re.search(pattern, cmd)
    if match:
        dataset_path = match.group(1)
        # return the basename of the dataset path without the path and extension
        return os.path.basename(dataset_path).split(".")[0]
    else:
        return None


def generate_runstamp(command: str, include_config: bool = True, include_dataset: bool = True) -> tuple[str, str]:
    """Generate a unique runstamp"""
    timestamp = datetime.now().strftime("%m_%d_%H%M")
    uuid = shortuuid.uuid()
    run_stamp = f"{timestamp}_{uuid}"
    # 1. first, check if we can extract the config and dataset from the command
    config = parse_command_config(command)
    dataset = parse_command_dataset(command)
    # 2. combine the config and dataset into the runstamp
    if include_config and config is not None:
        run_stamp = os.path.join(config, run_stamp)
    if include_dataset and dataset is not None:
        run_stamp = os.path.join(dataset, run_stamp)
    return run_stamp, uuid


def get_formatted_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def convert_time_to_timeout(time_str: str, offset_minutes: int = 0) -> str:
    """Convert a time string of the form HH:MM[:SS] to a timeout string of the form MMm"""
    time_split = [int(i) for i in time_str.split(":")]
    if len(time_split) == 2:
        seconds = 0
        hours, minutes = time_split
    elif len(time_split) == 3:
        hours, minutes, seconds = time_split
    else:
        raise ValueError(f"Invalid time string: {time_str}")

    # Convert everything to minutes and round up if there are seconds
    total_minutes = hours * 60 + minutes
    if seconds > 0:
        total_minutes += 1  # Round up to next minute

    total_minutes += offset_minutes

    return f"{total_minutes}m"
