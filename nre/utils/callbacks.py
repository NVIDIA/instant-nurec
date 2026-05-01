# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3: predict-only standalone keeps two symbols from the
# original callback grab-bag: TQDMProgressBar (re-exported for the progress
# bar callback) and make_logger (returns DummyLogger for predict — the
# Wandb / TensorBoard branches were training-only).

from __future__ import annotations

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers.logger import DummyLogger, Logger

from nre.config.logger import LoggerConfigType


__all__ = ["TQDMProgressBar", "make_logger"]


def make_logger(_config: LoggerConfigType) -> Logger:
    return DummyLogger()
