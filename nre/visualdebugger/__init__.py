# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3 stub: the polyscope-backed visual debugger has no role in
# the standalone predict pipeline. Production code calls
# `get_visualdebugger()` and conditionally invokes the returned object's
# methods; we return a NullDebugger that swallows everything so the call sites
# don't need editing. The real subtree was deleted; this 30-LOC stub replaces
# nre/internal/visualdebugger entirely.

from __future__ import annotations

from typing import Any


class _NullDebugger:
    """No-op visual debugger used in place of the polyscope-backed one."""

    def __getattr__(self, _name: str):
        return self._noop

    @staticmethod
    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None


_INSTANCE = _NullDebugger()


def get_visualdebugger() -> _NullDebugger:
    return _INSTANCE


VisualDebugger = _NullDebugger
