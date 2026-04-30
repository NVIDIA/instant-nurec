#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""S3 monitor implementation that uses s3cmd CLI to query and download objects.

Relies on a configured ~/.s3cfg that already contains the endpoint, access key,
secret key, etc.  Works with NVIDIA CSS or any S3-compatible service.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import subprocess

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class S3Object:
    key: str
    size: int
    last_modified: str  # ISO string from s3cmd output


class S3CmdMonitor:
    """Find new / changed metrics.yaml objects via s3cmd."""

    def __init__(
        self,
        bucket: str,
        state_file: str = "/tmp/s3_monitor_state.json",
        temp_dir: str = "/tmp/s3_metrics",
        file_patterns: Optional[List[str]] = None,
    ) -> None:
        # Ensure temp dir exists early; s3cmd is provided via Bazel python deps and invoked via `python -m s3cmd`

        self.bucket = bucket
        self.state_path = Path(state_file)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.patterns = file_patterns or ["metrics.yaml", "metrics.yml"]

        self.state: Dict[str, Dict] = self._load_state()

    # ---------------------------------------------------------------------
    # Low-level helpers
    # ---------------------------------------------------------------------
    def _run(self, *args: str) -> str:
        cmd = ["s3cmd", *args]
        logger.debug("Running %s", " ".join(cmd))
        try:
            out = subprocess.check_output(cmd, text=True)
            return out
        except subprocess.CalledProcessError as exc:
            logger.error("s3cmd failed (%s): %s", exc.returncode, exc.output)
            raise

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Dict]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except Exception as e:
                logger.warning("Failed to load state file: %s", e)
        return {}

    def _save_state(self) -> None:
        # Ensure parent directory exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))
        logger.info("Wrote state: %s", self.state_path)

    # ------------------------------------------------------------------
    # Listing & filtering
    # ------------------------------------------------------------------
    def _any_key_matches_pattern(self, key: str) -> bool:
        # Match against the basename so patterns like "metrics.yaml" work for keys with prefixes
        name = Path(key).name
        return any(fnmatch.fnmatch(name, pat) for pat in self.patterns)

    def list_objects(self) -> List[S3Object]:
        out = self._run("ls", "--recursive", f"s3://{self.bucket}")
        objs: List[S3Object] = []
        for line in out.splitlines():
            # format: 2025-08-07 12:34  123456   s3://bucket/key
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 3)  # Split on whitespace, max 4 parts
            if len(parts) < 4:
                logger.warning("Unexpected s3cmd output format: %s", line)
                continue
            try:
                size = int(parts[2])
            except ValueError:
                logger.warning("Invalid size in s3cmd output: %s", parts[2])
                continue
            s3_path = parts[3]
            if not s3_path.startswith(f"s3://{self.bucket}/"):
                logger.warning("Unexpected S3 path format: %s", s3_path)
                continue
            key = s3_path[len(f"s3://{self.bucket}/") :]
            last_mod = f"{parts[0]} {parts[1]}"
            objs.append(S3Object(key=key, size=size, last_modified=last_mod))
        return objs

    def is_new_or_changed(self, obj: S3Object) -> bool:
        entry = self.state.get(obj.key)
        if not entry:
            return True
        return entry.get("size") != obj.size or entry.get("last_modified") != obj.last_modified

    # ------------------------------------------------------------------
    # Download & process
    # ------------------------------------------------------------------
    def download(self, obj: S3Object) -> Path:
        dest = self.temp_dir / obj.key.replace("/", "_")
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._run("get", f"s3://{self.bucket}/{obj.key}", str(dest))
        return dest

    def monitor_once(self, upload_callback: Optional[Callable] = None, dry_run: bool = False) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        objs = [o for o in self.list_objects() if self._any_key_matches_pattern(o.key) and self.is_new_or_changed(o)]
        logger.info("Found %d new/changed metrics files", len(objs))

        for obj in objs:
            if dry_run:
                logger.info(f"[DRY RUN] Would download and process: {obj.key}")
                results[obj.key] = True
            else:
                tmp = self.download(obj)
                success = upload_callback(tmp) if upload_callback else True
                results[obj.key] = success
                if success:
                    self.state[obj.key] = {
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                        "processed_at": datetime.utcnow().isoformat(),
                        "upload_success": True,
                    }
                    self._save_state()
                tmp.unlink(missing_ok=True)
        return results

    def get_statistics(self) -> Dict:
        """Get statistics about processed files."""
        total = len(self.state)
        successful = sum(1 for item in self.state.values() if item.get("upload_success", False))
        failed = total - successful

        return {
            "total_processed": total,
            "successful_uploads": successful,
            "failed_uploads": failed,
            "state_file": str(self.state_path),
        }

    def reset_and_save_state(self) -> None:
        """Reset the processed files state and persist an empty state file.

        This is useful for the first time we run the script, or when we want to start fresh (upload everything that exists under the bucket/path).
        """
        self.state = {}
        self._save_state()
        logger.info("Reset processed files state (persisted empty state)")

    def scan_and_mark_all(self) -> Dict[str, bool]:
        """Scan bucket and mark all metrics files as processed without downloading/uploading."""
        results: Dict[str, bool] = {}

        # List all objects and filter for metrics files
        objs = [o for o in self.list_objects() if self._any_key_matches_pattern(o.key)]
        logger.info("Found %d total metrics files in bucket", len(objs))

        # Mark each file as processed without downloading
        for obj in objs:
            # Check if it's new or changed
            is_new = self.is_new_or_changed(obj)

            # Mark as processed
            self.state[obj.key] = {
                "size": obj.size,
                "last_modified": obj.last_modified,
                "processed_at": datetime.utcnow().isoformat(),
                "upload_success": False,  # Not actually uploaded
                "scan_only": True,  # Marked via scan-only mode
            }
            results[obj.key] = True

            if is_new:
                logger.info(f"Marked as processed (new/changed): {obj.key}")
            else:
                logger.debug(f"Already in state (unchanged): {obj.key}")

        # Save the updated state
        self._save_state()
        logger.info(f"Saved state for {len(results)} files")

        return results
