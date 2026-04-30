# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import Iterator, Optional, Sized, Union

import torch

from nre.utils.types import NovelViewOverrides


class ParameterizedSequentialSampler(torch.utils.data.Sampler[tuple[int, Optional[NovelViewOverrides]]]):
    """Efficient sampler yielding (index, novel_view_overrides) tuples.

    Supports both sequential and shuffled iteration over a dataset with optional
    novel view pose overrides. Optimized for minimal memory allocation.

    Args:
        data_source: Dataset to sample from
        novel_view_overrides: Optional pose overrides for novel view sampling
        shuffle: If True, shuffle indices each epoch; if False, iterate sequentially
    """

    def __init__(
        self,
        data_source: Sized,
        novel_view_overrides: Optional[NovelViewOverrides] = None,
        shuffle: bool = False,
    ) -> None:
        self._n = len(data_source)
        self._overrides = novel_view_overrides
        self._shuffle = shuffle
        self._forced_idx: Optional[int] = None

    @property
    def novel_view_overrides(self) -> Optional[NovelViewOverrides]:
        """Get the current novel view overrides."""
        return self._overrides

    @novel_view_overrides.setter
    def novel_view_overrides(self, value: Optional[NovelViewOverrides]) -> None:
        """Set the novel view overrides for all yielded samples."""
        self._overrides = value

    @property
    def forced_index(self) -> Optional[int]:
        """Get the forced index if set."""
        return self._forced_idx

    @forced_index.setter
    def forced_index(self, value: Optional[int]) -> None:
        """Set a forced index to be yielded once on next iteration."""
        self._forced_idx = value

    def __iter__(self) -> Iterator[tuple[int, Optional[NovelViewOverrides]]]:
        # Handle forced index: yield once then revert to normal iteration
        if self._forced_idx is not None:
            yield (self._forced_idx, self._overrides)
            self._forced_idx = None
            return

        # Generate indices efficiently
        indices: Union[list[int], range]
        if self._shuffle:
            indices = torch.randperm(self._n).tolist()
        else:
            indices = range(self._n)

        # Yield indices with overrides
        for idx in indices:
            yield (idx, self._overrides)

    def __len__(self) -> int:
        return self._n
