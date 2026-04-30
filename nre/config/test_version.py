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

from datetime import datetime

import pytest

from nre.config.version import Version


def test_version_from_components():
    """Test the Version.from_components class method."""
    v = Version.from_components(1, 2, 3)
    assert v.version_major == 1
    assert v.version_minor == 2
    assert v.version_patch == 3
    assert v.git_commit_sha_short == "0000000"
    assert v.git_tree_dirty is False
    assert v.git_commit_date == datetime.fromtimestamp(0)
    assert v.version_string == "1.2.3-0000000"


def test_version_equality():
    """Test equality and inequality of Version objects."""
    assert Version.from_components(1, 2, 3) == Version.from_components(1, 2, 3)
    assert Version.from_components(1, 2, 3) != Version.from_components(1, 2, 4)
    assert Version.from_components(1, 2, 3) != Version.from_components(1, 3, 3)
    assert Version.from_components(1, 2, 3) != Version.from_components(2, 2, 3)
    assert Version.from_components(1, 2, 3) != "1.2.3"
    assert not (Version.from_components(1, 2, 3) == "1.2.3")


def test_version_less_than():
    """Test less than comparison for Version objects."""
    assert Version.from_components(1, 2, 3) < Version.from_components(1, 2, 4)
    assert Version.from_components(1, 2, 3) < Version.from_components(1, 3, 0)
    assert Version.from_components(1, 2, 3) < Version.from_components(2, 0, 0)
    assert not (Version.from_components(1, 2, 3) < Version.from_components(1, 2, 3))
    assert not (Version.from_components(2, 0, 0) < Version.from_components(1, 2, 3))


def test_version_less_than_or_equal():
    """Test less than or equal to comparison for Version objects."""
    assert Version.from_components(1, 2, 3) <= Version.from_components(1, 2, 3)
    assert Version.from_components(1, 2, 3) <= Version.from_components(1, 2, 4)
    assert Version.from_components(1, 2, 3) <= Version.from_components(1, 3, 0)
    assert Version.from_components(1, 2, 3) <= Version.from_components(2, 0, 0)
    assert not (Version.from_components(1, 3, 0) <= Version.from_components(1, 2, 3))


def test_version_greater_than():
    """Test greater than comparison for Version objects."""
    assert Version.from_components(1, 2, 4) > Version.from_components(1, 2, 3)
    assert Version.from_components(1, 3, 0) > Version.from_components(1, 2, 3)
    assert Version.from_components(2, 0, 0) > Version.from_components(1, 2, 3)
    assert not (Version.from_components(1, 2, 3) > Version.from_components(1, 2, 3))
    assert not (Version.from_components(1, 2, 3) > Version.from_components(2, 0, 0))


def test_version_greater_than_or_equal():
    """Test greater than or equal to comparison for Version objects."""
    assert Version.from_components(1, 2, 3) >= Version.from_components(1, 2, 3)
    assert Version.from_components(1, 2, 4) >= Version.from_components(1, 2, 3)
    assert Version.from_components(1, 3, 0) >= Version.from_components(1, 2, 3)
    assert Version.from_components(2, 0, 0) >= Version.from_components(1, 2, 3)
    assert not (Version.from_components(1, 2, 3) >= Version.from_components(1, 3, 0))


def test_version_comparison_only_on_semantic_version():
    """Test that comparison ignores non-semantic version components."""
    v1 = Version(
        version_major=1,
        version_minor=2,
        version_patch=3,
        git_commit_sha_short="def",
        version_build=20,
        version_extra="beta",
    )
    v2 = Version(
        version_major=1,
        version_minor=2,
        version_patch=3,
        git_commit_sha_short="abc",
        version_build=10,
        version_extra="alpha",
    )

    assert v1 >= v2
    assert v1 <= v2
    assert not (v1 > v2)
    assert not (v1 < v2)

    # For good measure, let's make sure that they are not equal, since the git has is different
    assert v1 != v2


def test_version_comparison_with_other_types():
    """Test that comparing a Version object with other types raises a TypeError."""
    with pytest.raises(TypeError):
        Version.from_components(1, 2, 3) < "1.2.3"
    with pytest.raises(TypeError):
        Version.from_components(1, 2, 3) > 123
    with pytest.raises(TypeError):
        Version.from_components(1, 2, 3) <= (1, 2, 3)
    with pytest.raises(TypeError):
        Version.from_components(1, 2, 3) >= None
