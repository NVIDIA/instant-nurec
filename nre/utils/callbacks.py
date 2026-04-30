# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3: predict-only standalone keeps three symbols from the
# original callback grab-bag: PreemptionInterruptException (an exception
# class run.py catches), TQDMProgressBar (re-exported for the progress
# bar callback), and make_logger (returns DummyLogger for predict — the
# Wandb / TensorBoard branches were training-only).

from __future__ import annotations

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers.logger import DummyLogger, Logger

from nre.config.logger import DummyLoggerConfig, LoggerConfigType


__all__ = ["PreemptionInterruptException", "TQDMProgressBar", "make_logger"]


class PreemptionInterruptException(Exception):
    def __init__(self, message: str = "Preempted") -> None:
        super().__init__(message)
        self.message = message


def make_logger(config: LoggerConfigType) -> Logger:
    if isinstance(config, DummyLoggerConfig):
        return DummyLogger()
    raise TypeError(
        f"Predict-only standalone supports DummyLoggerConfig only; got {type(config).__name__}."
    )
