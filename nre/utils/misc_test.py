# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Dict, List

import numpy as np
import pytest

import nre.utils.misc as misc_module

from nre.utils.misc import list_of_dicts_to_dict_of_lists


class TestListOfDictsToDictOfLists:
    """Test cases for list_of_dicts_to_dict_of_lists function."""

    def test_basic_conversion(self):
        """Test basic conversion without any special flags."""
        list_of_dicts = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
            {"a": 7, "b": 8, "c": 9},
        ]
        expected = {
            "a": [1, 4, 7],
            "b": [2, 5, 8],
            "c": [3, 6, 9],
        }
        result = list_of_dicts_to_dict_of_lists(list_of_dicts)
        assert result == expected

    def test_empty_list(self):
        """Test with empty list of dictionaries."""
        list_of_dicts: List[Dict[str, Any]] = []
        result = list_of_dicts_to_dict_of_lists(list_of_dicts)
        assert result == {}

    def test_single_dict(self):
        """Test with a single dictionary."""
        list_of_dicts = [{"x": 10, "y": 20}]
        expected = {"x": [10], "y": [20]}
        result = list_of_dicts_to_dict_of_lists(list_of_dicts)
        assert result == expected

    def test_no_duplicates_flag(self):
        """Test with no_duplicates=True to remove duplicate values."""
        list_of_dicts = [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 1, "b": "x"},  # Duplicates
            {"a": 3, "b": "z"},
        ]
        expected = {
            "a": [1, 2, 3],  # Unique values only
            "b": ["x", "y", "z"],  # Unique values only
        }
        result = list_of_dicts_to_dict_of_lists(list_of_dicts, no_duplicates=True)

        # Since set ordering is not guaranteed, we need to check contents
        assert set(result["a"]) == set(expected["a"])
        assert set(result["b"]) == set(expected["b"])
        assert len(result["a"]) == 3
        assert len(result["b"]) == 3

    def test_singleton_flag_success(self):
        """Test with singleton=True when all values for each key are the same."""
        list_of_dicts = [
            {"type": "user", "status": "active"},
            {"type": "user", "status": "active"},
            {"type": "user", "status": "active"},
        ]
        expected = {
            "type": "user",  # Single value, not a list
            "status": "active",  # Single value, not a list
        }
        result = list_of_dicts_to_dict_of_lists(list_of_dicts, singleton=True)
        assert result == expected

    def test_singleton_flag_failure(self):
        """Test with singleton=True when values differ - should raise assertion error."""
        list_of_dicts = [
            {"type": "user", "status": "active"},
            {"type": "admin", "status": "active"},  # Different type
        ]
        with pytest.raises(AssertionError, match="List for key type has more than one unique element"):
            list_of_dicts_to_dict_of_lists(list_of_dicts, singleton=True)

    def test_mixed_data_types(self):
        """Test with mixed data types in values."""
        list_of_dicts = [
            {"num": 42, "text": "hello", "flag": True, "items": [1, 2]},
            {"num": 99, "text": "world", "flag": False, "items": [3, 4]},
        ]
        expected = {
            "num": [42, 99],
            "text": ["hello", "world"],
            "flag": [True, False],
            "items": [[1, 2], [3, 4]],
        }
        result = list_of_dicts_to_dict_of_lists(list_of_dicts)
        assert result == expected

    def test_different_keys_assertion_error(self):
        """Test that assertion error is raised when dictionaries have different keys."""
        list_of_dicts = [
            {"a": 1, "b": 2},
            {"a": 3, "c": 4},  # Missing 'b', has 'c' instead
        ]
        with pytest.raises(AssertionError, match="All dictionaries must have the same keys"):
            list_of_dicts_to_dict_of_lists(list_of_dicts)

    def test_missing_keys_assertion_error(self):
        """Test assertion error when some dictionaries are missing keys."""
        list_of_dicts = [
            {"x": 1, "y": 2, "z": 3},
            {"x": 4, "y": 5},  # Missing 'z'
        ]
        with pytest.raises(AssertionError, match="All dictionaries must have the same keys"):
            list_of_dicts_to_dict_of_lists(list_of_dicts)

    def test_no_duplicates_with_hashable_objects(self):
        """Test no_duplicates with hashable objects like tuples and strings."""
        list_of_dicts = [
            {"data": (1, 2), "name": "alice"},
            {"data": (3, 4), "name": "bob"},
            {"data": (1, 2), "name": "alice"},  # Duplicate
            {"data": (5, 6), "name": "charlie"},
        ]
        result = list_of_dicts_to_dict_of_lists(list_of_dicts, no_duplicates=True)

        # Check that we have unique values
        assert set(result["data"]) == {(1, 2), (3, 4), (5, 6)}
        assert set(result["name"]) == {"alice", "bob", "charlie"}
        assert len(result["data"]) == 3
        assert len(result["name"]) == 3

    def test_no_duplicates_with_numpy_arrays_error(self):
        """Test that no_duplicates raises TypeError when values contain numpy arrays (unhashable)."""
        list_of_dicts = [
            {"int_array": np.array([1, 2], dtype=np.int32), "name": "alice"},
            {"int_array": np.array([3, 4], dtype=np.int32), "name": "bob"},
            {"int_array": np.array([1, 2], dtype=np.int32), "name": "alice"},  # Duplicate
        ]
        # Should raise TypeError because numpy arrays are not hashable
        with pytest.raises(TypeError):
            list_of_dicts_to_dict_of_lists(list_of_dicts, no_duplicates=True)

    def test_with_numpy_arrays_no_flags(self):
        """Test that function works correctly with numpy arrays when no special flags are used."""
        int_arr1 = np.array([1, 2], dtype=np.int32)
        int_arr2 = np.array([3, 4], dtype=np.int32)
        float_arr1 = np.array([1.5, 2.5], dtype=np.float32)
        float_arr2 = np.array([3.5, 4.5], dtype=np.float32)
        double_arr1 = np.array([1.1, 2.2], dtype=np.float64)
        double_arr2 = np.array([3.3, 4.4], dtype=np.float64)

        list_of_dicts = [
            {"int_array": int_arr1, "float_array": float_arr1, "double_array": double_arr1},
            {"int_array": int_arr2, "float_array": float_arr2, "double_array": double_arr2},
        ]
        result = list_of_dicts_to_dict_of_lists(list_of_dicts)

        # Check that arrays are preserved correctly
        assert len(result["int_array"]) == 2
        assert len(result["float_array"]) == 2
        assert len(result["double_array"]) == 2

        # Check array contents and dtypes
        np.testing.assert_array_equal(result["int_array"][0], int_arr1)
        np.testing.assert_array_equal(result["int_array"][1], int_arr2)
        assert result["int_array"][0].dtype == np.int32
        assert result["int_array"][1].dtype == np.int32

        np.testing.assert_array_equal(result["float_array"][0], float_arr1)
        np.testing.assert_array_equal(result["float_array"][1], float_arr2)
        assert result["float_array"][0].dtype == np.float32
        assert result["float_array"][1].dtype == np.float32

        np.testing.assert_array_equal(result["double_array"][0], double_arr1)
        np.testing.assert_array_equal(result["double_array"][1], double_arr2)
        assert result["double_array"][0].dtype == np.float64
        assert result["double_array"][1].dtype == np.float64

    def test_singleton_with_none_values(self):
        """Test singleton flag with None values."""
        list_of_dicts = [
            {"value": None, "count": 0},
            {"value": None, "count": 0},
        ]
        expected = {
            "value": None,
            "count": 0,
        }
        result = list_of_dicts_to_dict_of_lists(list_of_dicts, singleton=True)
        assert result == expected

    def test_large_dataset(self):
        """Test with a larger dataset to ensure performance is reasonable."""
        list_of_dicts = [{"id": i, "group": i % 3, "value": f"item_{i}"} for i in range(1000)]
        result = list_of_dicts_to_dict_of_lists(list_of_dicts)

        assert len(result["id"]) == 1000
        assert len(result["group"]) == 1000
        assert len(result["value"]) == 1000
        assert result["id"][0] == 0
        assert result["id"][-1] == 999
        assert result["group"][0] == 0
        assert result["value"][0] == "item_0"

    def test_no_duplicates_with_unhashable_objects_error(self):
        """Test that no_duplicates raises TypeError when values contain unhashable objects like lists."""
        list_of_dicts = [
            {"data": [1, 2], "count": 1},
            {"data": [3, 4], "count": 2},
            {"data": [1, 2], "count": 1},  # Duplicate with unhashable list
        ]
        # Should raise TypeError because lists are not hashable and can't be put in a set
        with pytest.raises(TypeError):
            list_of_dicts_to_dict_of_lists(list_of_dicts, no_duplicates=True)

    def test_singleton_with_unhashable_objects_error(self):
        """Test that singleton raises TypeError when values contain unhashable objects like lists."""
        list_of_dicts = [
            {"data": [1, 2], "count": 5},
            {"data": [1, 2], "count": 5},  # Same unhashable list values
        ]
        # Should raise TypeError because lists are not hashable and can't be put in a set
        with pytest.raises(TypeError):
            list_of_dicts_to_dict_of_lists(list_of_dicts, singleton=True)

    def test_with_unhashable_objects_no_flags(self):
        """Test that function works correctly with unhashable objects when no special flags are used."""
        list_of_dicts = [
            {"data": [1, 2], "meta": {"id": 1}, "tags": ["a", "b"]},
            {"data": [3, 4], "meta": {"id": 2}, "tags": ["c", "d"]},
            {"data": [1, 2], "meta": {"id": 1}, "tags": ["a", "b"]},  # Duplicate
        ]
        result = list_of_dicts_to_dict_of_lists(list_of_dicts)

        expected = {
            "data": [[1, 2], [3, 4], [1, 2]],
            "meta": [{"id": 1}, {"id": 2}, {"id": 1}],
            "tags": [["a", "b"], ["c", "d"], ["a", "b"]],
        }
        assert result == expected


def _patch_distributed(monkeypatch, *, rank: int, world_size: int, broadcast_impl):
    monkeypatch.setattr(misc_module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(misc_module.torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(misc_module.torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(misc_module.torch.distributed, "broadcast_object_list", broadcast_impl)


class TestSyncObjectsOrRaise:
    def test_master_rank_returns_value_and_broadcasts_success_payload(self, monkeypatch):
        captured = {}

        def fake_broadcast(object_list, src):
            captured["src"] = src
            captured["payload"] = object_list[0]

        _patch_distributed(monkeypatch, rank=0, world_size=2, broadcast_impl=fake_broadcast)

        value = {"mesh": 123}
        assert misc_module.sync_objects_or_raise(lambda: value) == value
        assert captured["src"] == 0
        assert captured["payload"] == misc_module.SyncedSuccessPayload(value=value)

    def test_master_rank_broadcasts_failure_payload_before_reraising(self, monkeypatch):
        captured = {}

        def fake_broadcast(object_list, src):
            captured["src"] = src
            captured["payload"] = object_list[0]

        def boom():
            raise ValueError("boom")

        _patch_distributed(monkeypatch, rank=0, world_size=2, broadcast_impl=fake_broadcast)

        with pytest.raises(ValueError, match="boom"):
            misc_module.sync_objects_or_raise(boom)

        assert captured["src"] == 0
        assert isinstance(captured["payload"], misc_module.SyncedExceptionPayload)
        assert captured["payload"].ok is False
        assert captured["payload"].message == "boom"

    def test_master_rank_broadcasts_base_exception_payload_before_reraising(self, monkeypatch):
        class SentinelBaseException(BaseException):
            pass

        captured = {}

        def fake_broadcast(object_list, src):
            captured["src"] = src
            captured["payload"] = object_list[0]

        def boom():
            raise SentinelBaseException("cancel")

        _patch_distributed(monkeypatch, rank=0, world_size=2, broadcast_impl=fake_broadcast)

        with pytest.raises(SentinelBaseException, match="cancel"):
            misc_module.sync_objects_or_raise(boom)

        assert captured["src"] == 0
        assert isinstance(captured["payload"], misc_module.SyncedExceptionPayload)
        assert captured["payload"].ok is False
        assert captured["payload"].message == "cancel"

    def test_non_master_rank_reraises_master_failure(self, monkeypatch):
        failure_payload = misc_module.SyncedExceptionPayload(
            master_rank=0,
            message="boom",
            traceback="Traceback (most recent call last):\nValueError: boom\n",
        )

        def fake_broadcast(object_list, src):
            assert src == 0
            object_list[0] = failure_payload

        _patch_distributed(monkeypatch, rank=1, world_size=2, broadcast_impl=fake_broadcast)

        with pytest.raises(misc_module.SyncedCallError, match="Callable failed on rank 0: boom") as exc_info:
            misc_module.sync_objects_or_raise(lambda: pytest.fail("non-master rank should not execute the callable"))

        assert exc_info.value.__notes__ == [
            "Raised on rank 0 during sync_objects_or_raise().",
            "Traceback (most recent call last):\nValueError: boom\n",
        ]

    def test_non_master_rank_returns_success_payload(self, monkeypatch):
        success_payload = misc_module.SyncedSuccessPayload(value=[1, 2, 3])

        def fake_broadcast(object_list, src):
            assert src == 0
            object_list[0] = success_payload

        _patch_distributed(monkeypatch, rank=1, world_size=2, broadcast_impl=fake_broadcast)

        assert misc_module.sync_objects_or_raise(lambda: pytest.fail("callable should not run")) == [
            1,
            2,
            3,
        ]
