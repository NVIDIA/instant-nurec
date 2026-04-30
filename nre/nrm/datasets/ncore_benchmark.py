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
Benchmark script for NCore NRM dataset loading.

Subclasses NCoreNRMDataset to add timing hooks on getitem_allow_exceptions
and its critical internal methods (_get_loaders_and_sensors, _get_rig_trajectory,
_load_data_batch, _compute_cuboid_tracks). Reports per-call timings when num_workers=0 and
always reports dataloader iteration speed.

Usage (with Bazel):
  bazel run //nre/nrm/datasets:ncore_benchmark -- --config-name=configs/nrm/apps/celsius_ci.yaml \\
    dataset.train.ncore_json_list_path=/path/to/train.lst --num-workers=4 --num-batches=20
"""

from __future__ import annotations

import itertools
import logging
import os
import sys
import time

from collections import defaultdict
from typing import Any
from urllib.parse import unquote, urlparse


MB = 1024 * 1024

import click
import numpy as np
import torch

from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from nre.nrm.config.dataset import BaseNCoreNRMDatasetConfig, NRMEpochSplitConfig, NRMMixedDatasetConfig
from nre.nrm.config.nrm import parse_typed_nrm_config
from nre.nrm.datasets.nrm_base import BaseNRMIndexableDataset
from nre.nrm.datasets.nrm_ncore import NCoreNRMDataset
from nre.utils.batch import NRMDataBatch


logger = logging.getLogger(__name__)


class _S3RequestLogHandler(logging.StreamHandler):
    """Logs only S3 request path and Range header (one line per request)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.name != "botocore.endpoint":
                return
            msg = record.getMessage()
            if "Sending http request" not in msg or not getattr(record, "args", ()):
                return
            args = record.args
            if not args or len(args) < 1:
                return
            request = args[0]
            if not hasattr(request, "url") or not hasattr(request, "headers"):
                return
            url = getattr(request, "url", "") or ""
            parsed = urlparse(url)
            path = unquote(parsed.path).lstrip("/") or "(root)"
            stem = os.path.splitext(os.path.basename(path))[0] or path
            headers = getattr(request, "headers", {}) or {}
            range_val = ""
            for k, v in headers.items():
                if k and str(k).lower() == "range" and v is not None:
                    range_val = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    break
            if range_val and range_val.strip().lower().startswith("bytes="):
                part = range_val.strip()[6:].strip()
                if "-" in part:
                    a, b = part.split("-", 1)
                    try:
                        start_b, end_b = int(a.strip()), int(b.strip())
                        size_b = end_b - start_b + 1
                        start_mb = start_b / MB
                        end_mb = end_b / MB
                        size_mb = size_b / MB
                        range_str = f"Range: {start_mb:.2f}-{end_mb:.2f} MB ({size_mb:.2f} MB)"
                    except ValueError:
                        range_str = f"Range: {range_val}"
                else:
                    range_str = f"Range: {range_val}"
            else:
                range_str = f"Range: {range_val}" if range_val else ""
            line = f"S3 {stem}  {range_str}" if range_str else f"S3 {stem}"
            if self.stream:
                self.stream.write(line + "\n")
                self.flush()
        except Exception:
            pass


def _get_train_dataset_config(config: Any) -> BaseNCoreNRMDatasetConfig:
    """Resolve train split to a single NCore dataset config."""
    train_cfg = config.dataset.train
    if isinstance(train_cfg, NRMEpochSplitConfig):
        train_cfg = train_cfg.last_milestone()
    if isinstance(train_cfg, NRMMixedDatasetConfig):
        mixture_names = list(train_cfg.mixture.keys())
        mixture_sample_ratios = [train_cfg.mixture[name].sample_ratio for name in mixture_names]
        logger.info(
            f"Benchmarking mixed dataset with components: {mixture_names}, sample ratios: {mixture_sample_ratios}."
        )
        # Choose the component with the highest sample ratio.
        max_sample_ratio = max(mixture_sample_ratios)
        max_sample_ratio_index = mixture_sample_ratios.index(max_sample_ratio)
        mixture_name = mixture_names[max_sample_ratio_index]
        logger.info(f"Choosing component {mixture_name} with sample ratio {max_sample_ratio}.")
        train_cfg = train_cfg.mixture[mixture_name].config
    if not isinstance(train_cfg, BaseNCoreNRMDatasetConfig):
        raise ValueError(
            "Benchmark only supports train split with nrm-ncore (or nrm-websocket-ncore) dataset. "
            f"Got config with name={getattr(train_cfg, 'name', None)}."
        )
    return train_cfg


class NCoreNRMBenchmarkDataset(NCoreNRMDataset):
    """
    NCoreNRMDataset subclass that records timings for getitem_allow_exceptions
    and the critical internal methods: _get_loaders_and_sensors, _get_rig_trajectory,
    _load_data_batch, _compute_cuboid_tracks. Timings are stored in self._timing_records
    (list of dicts). Only populated when __getitem__ runs in this process (i.e. when
    num_workers=0 or in worker processes); the main process does not see worker
    timings.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._timing_records: list[dict[str, float]] = []

    def getitem_allow_exceptions(self, batch_idx: int, rng: np.random.Generator) -> NRMDataBatch:
        self._timing_records.append({})  # one record per getitem for inner timings
        t0 = time.perf_counter()
        out = super().getitem_allow_exceptions(batch_idx, rng)
        total_s = time.perf_counter() - t0
        self._timing_records[-1]["getitem_total_s"] = total_s
        return out

    def _get_loaders_and_sensors(self, ncore_json_path: Any) -> Any:
        t0 = time.perf_counter()
        out = super()._get_loaders_and_sensors(ncore_json_path)
        elapsed = time.perf_counter() - t0
        self._append_timing("get_loaders_and_sensors_s", elapsed)
        return out

    def _get_rig_trajectory(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = super()._get_rig_trajectory(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        self._append_timing("get_rig_trajectory_s", elapsed)
        return out

    def _load_data_batch(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = super()._load_data_batch(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        self._append_timing("load_data_batch_s", elapsed)
        return out

    def _compute_cuboid_tracks(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = super()._compute_cuboid_tracks(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        self._append_timing("compute_cuboid_tracks_s", elapsed)
        return out

    def _append_timing(self, key: str, value: float) -> None:
        if not self._timing_records:
            self._timing_records.append({})
        self._timing_records[-1][key] = self._timing_records[-1].get(key, 0.0) + value


def _aggregate_timings(records: list[dict[str, float]]) -> dict[str, list[float]]:
    """Aggregate per-call timings into lists per key."""
    agg: dict[str, list[float]] = defaultdict(list)
    for r in records:
        for k, v in r.items():
            agg[k].append(v)
    return dict(agg)


# Preferred order for timing summary: component breakdown first, total last.
_TIMING_KEY_ORDER = (
    "get_loaders_and_sensors_s",
    "get_rig_trajectory_s",
    "load_data_batch_s",
    "compute_cuboid_tracks_s",
    "getitem_total_s",
)


def _summarize_timings(agg: dict[str, list[float]]) -> None:
    """Log summary statistics for each timing key. getitem_total_s is printed last."""
    key_order = [k for k in _TIMING_KEY_ORDER if k in agg]
    key_order += sorted(k for k in agg if k not in _TIMING_KEY_ORDER)
    for key in key_order:
        vals = agg[key]
        n = len(vals)
        total = sum(vals)
        mean = total / n if n else 0.0
        logger.info(
            "  %s: n=%d total=%.3fs mean=%.2fms min=%.2fms max=%.2fms",
            key,
            n,
            total,
            mean * 1000,
            min(vals) * 1000,
            max(vals) * 1000,
        )


@click.command()
@click.option(
    "--config-name",
    type=str,
    required=True,
    help="Hydra config name (e.g. configs/nrm/apps/celsius_ci.yaml).",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    show_default=True,
    help="Number of dataloader workers. Per-getitem timings are only collected when 0.",
)
@click.option(
    "--num-batches",
    type=int,
    default=10,
    show_default=True,
    help="Number of batches to iterate for iteration speed measurement.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size (default: from config system.train_batch_size).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging.",
)
@click.option(
    "--log-s3-requests",
    is_flag=True,
    help="Log each S3 request as one line: path and Range header (to stderr).",
)
@click.option(
    "--s3-block-size-mb",
    type=int,
    default=None,
    help="Override S3 block size in MB for range requests (default: from config).",
)
@click.option(
    "--s3-cache-type",
    type=str,
    default=None,
    help="Override S3 cache type for range requests (default: from config).",
)
@click.argument("hydra_args", nargs=-1)
def main(
    config_name: str,
    num_workers: int,
    num_batches: int,
    batch_size: int | None,
    verbose: bool,
    log_s3_requests: bool,
    s3_block_size_mb: int | None,
    s3_cache_type: str | None,
    hydra_args: tuple[str, ...],
) -> None:
    """Benchmark NCore NRM dataset and dataloader iteration speed."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    if log_s3_requests:
        # Log only S3 path and Range header (one line per request) to stderr.
        # Stop propagation so the root logger does not print full DEBUG messages.
        s3_handler = _S3RequestLogHandler(sys.stderr)
        s3_handler.setLevel(logging.DEBUG)
        botocore_endpoint = logging.getLogger("botocore.endpoint")
        botocore_endpoint.setLevel(logging.DEBUG)
        botocore_endpoint.propagate = False
        botocore_endpoint.addHandler(s3_handler)

    # Load NRM config and resolve train dataset config
    config = parse_typed_nrm_config(
        config_name=config_name,
        hydra_args=hydra_args,
        config_dir="./configs",
    )
    train_dataset_config = _get_train_dataset_config(config)
    if s3_block_size_mb is not None:
        train_dataset_config.s3_block_size_mb = s3_block_size_mb
    if s3_cache_type is not None:
        train_dataset_config.s3_cache_type = s3_cache_type

    # Build benchmark dataset (same config as NCoreNRMDataset)
    dataset: BaseNRMIndexableDataset = NCoreNRMBenchmarkDataset(train_dataset_config, split="train")
    dataset.set_epoch(0)
    if "PL_GLOBAL_SEED" not in os.environ:
        os.environ["PL_GLOBAL_SEED"] = str(42)
    dataset.set_rng_epoch(0)

    batch_size_val = batch_size if batch_size is not None else config.system.train_batch_size

    # Dataloader
    batch_sampler = BatchSampler(
        RandomSampler(dataset, generator=torch.Generator().manual_seed(42)),
        batch_size=batch_size_val,
        drop_last=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=NRMDataBatch.collate_fn,
        pin_memory=False,
        persistent_workers=False,
    )

    # Run benchmark (islice so exactly num_batches are loaded, not num_batches+1).
    # Measure time-to-first-batch (warmup) and sustained throughput (batches 2..N) separately.
    total_samples = 0
    first_batch_samples = 0
    t_start = time.perf_counter()
    t_first_batch: float | None = None
    i = -1
    for i, batch in enumerate(itertools.islice(dataloader, num_batches)):
        n = len(batch.meta) if batch.meta else batch_size_val
        total_samples += n
        if i == 0:
            t_first_batch = time.perf_counter() - t_start
            first_batch_samples = n
    batches_done = i + 1 if num_batches > 0 else 0
    t_elapsed = time.perf_counter() - t_start

    # Iteration speed: overall + first-batch (warmup) and sustained (batches 2..N)
    batches_per_sec = batches_done / t_elapsed if t_elapsed > 0 else 0.0
    samples_per_sec = total_samples / t_elapsed if t_elapsed > 0 else 0.0
    logger.info("=== Dataloader iteration ===")
    logger.info("  Batches: %d in %.3fs", batches_done, t_elapsed)
    logger.info("  Throughput (overall): %.2f batches/s, %.2f samples/s", batches_per_sec, samples_per_sec)
    if t_first_batch is not None:
        logger.info("  Time to first batch: %.3fs", t_first_batch)
    if batches_done >= 2:
        sustained_batches = batches_done - 1
        sustained_time = t_elapsed - (t_first_batch or 0.0)
        if sustained_time > 0:
            sustained_samples = total_samples - first_batch_samples
            sustained_batches_per_sec = sustained_batches / sustained_time
            sustained_samples_per_sec = sustained_samples / sustained_time
            logger.info(
                "  Sustained (batches 2..%d): %.2f batches/s, %.2f samples/s",
                batches_done,
                sustained_batches_per_sec,
                sustained_samples_per_sec,
            )

    # Per-call timings (only when num_workers=0, from main process dataset)
    if num_workers == 0 and isinstance(dataset, NCoreNRMBenchmarkDataset) and dataset._timing_records:
        agg = _aggregate_timings(dataset._timing_records)
        logger.info("=== Per-getitem timings (num_workers=0) ===")
        _summarize_timings(agg)
    elif num_workers > 0:
        logger.info("=== Per-getitem timings ===")
        logger.info("  (Skipped: set --num-workers=0 to collect per-call timings.)")


if __name__ == "__main__":
    main()
