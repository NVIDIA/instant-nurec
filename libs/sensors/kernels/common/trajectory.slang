// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

// Trajectory interval utilities for sensor kernels.
// Provides binary search for finding intervals in control point trajectories.

#include <slang_utils/extensions.slang>

// ============================================================================
// Trajectory Interval Helper
// ============================================================================

// Result of finding an interval in a trajectory for interpolation
struct TrajectoryInterval {
    uint idx0;      // Index of interval start
    uint idx1;      // Index of interval end
    float alpha;    // Interpolation factor [0, 1]
};

// Find the trajectory interval containing the target time and compute interpolation alpha.
// Uses binary search for O(log K) complexity with K control points.
//
// Parameters:
//   target_time: Time to find interval for [0, 1]
//   control_times: TensorView<float> of control point times (K,)
//   control_count: Number of control points
//
// Returns:
//   TrajectoryInterval with indices and interpolation factor
//
TrajectoryInterval find_trajectory_interval(
    float target_time,
    TensorView<float> control_times,
    uint control_count
) {
    TrajectoryInterval result;
    result.idx0 = 0;
    result.idx1 = 0;
    result.alpha = 0.0;
    
    if (control_count == 0) {
        return result;
    }
    
    // Binary search for interval
    uint interval_idx = 0;
    if (control_count > 1) {
        uint left = 0;
        uint right = control_count - 1;
        while (right - left > 1) {
            uint mid = (left + right) / 2;
            float t_mid = control_times.load(mid);
            if (t_mid <= target_time) {
                left = mid;
            } else {
                right = mid;
            }
        }
        interval_idx = left;
    }
    
    result.idx0 = interval_idx;
    result.idx1 = min(interval_idx + 1, control_count - 1);
    
    float t0 = control_times.load(result.idx0);
    float t1 = control_times.load(result.idx1);
    result.alpha = (control_count == 1 || t1 == t0) ? 0.0 : (target_time - t0) / (t1 - t0);
    
    return result;
}
