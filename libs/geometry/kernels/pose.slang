// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "quaternion.slang"
#include "transform.slang"

// Core pose interface - all pose types implement this
// A pose is a static transformation (no temporal component)
interface IPose : IDifferentiable {
    // Transform a 3D point from source frame to target frame
    [Differentiable]
    float3 transform_point(float3 point);

    // Transform a direction vector (no translation)
    [Differentiable]
    float3 transform_direction(float3 direction);

    // Inverse transformation (target → source)
    [Differentiable]
    float3 inverse_transform_point(float3 point);

    [Differentiable]
    float3 inverse_transform_direction(float3 direction);
}

// SE(3) pose: translation + rotation (quaternion representation)
// Implements IPose interface for static transformations
// This is a thin wrapper around transform::SE3 that adds the IPose interface
// Note: Timestamps belong in Trajectories, not in Poses. A Pose is a purely static transformation.
struct SE3Pose : IPose {
    transform::SE3 se3;                          // Internal SE3 transformation

    // Accessor properties for convenience (forward to internal SE3)
    property float3 translation {
        get { return se3.trans.p; }
        set(newValue) { se3.trans.p = newValue; }
    }

    property float4 rotation {
        get { return se3.so3.unit_quaternion.q; }
        set(newValue) { se3.so3.unit_quaternion.q = newValue; }
    }

    // IPose interface implementation - forward to internal SE3
    [Differentiable]
    float3 transform_point(float3 point) {
        return (se3 * transform::Point(point)).p;
    }

    [Differentiable]
    float3 transform_direction(float3 direction) {
        return (se3.so3 * transform::Point(direction)).p;
    }

    [Differentiable]
    float3 inverse_transform_point(float3 point) {
        // T^(-1) * p = R^T * (p - t)
        transform::Point translated = transform::Point(point - se3.trans.p);
        return (se3.so3.inv() * translated).p;
    }

    [Differentiable]
    float3 inverse_transform_direction(float3 direction) {
        return (se3.so3.inv() * transform::Point(direction)).p;
    }

    // Utility constructors
    [Differentiable]
    static SE3Pose identity() {
        SE3Pose pose;
        pose.se3 = transform::SE3.identity();
        return pose;
    }

    [Differentiable]
    static SE3Pose from_translation_quaternion(float3 t, float4 q) {
        SE3Pose pose;
        pose.se3 = transform::SE3.from_vec(t, q);
        return pose;
    }

    // Get as 4x4 matrix (for compatibility)
    // Standard format: rotation in upper-left 3x3, translation in 4th column
    [Differentiable]
    float4x4 to_matrix() {
        return transform::convertSE3ToMatrix(se3);
    }
}

// Rigged SE3 pose: composition of sensor-to-rig + rig-to-world transforms
// Composition of sensor-to-rig + rig-to-world SE3 transforms
//
// NOTE: We use owned values instead of pointers due to a Slang compiler bug.
// Even non-generic structs with pointers that implement interfaces with 
// [Differentiable] methods trigger an internal compiler error.
struct RiggedSE3Pose : IPose {
    SE3Pose T_sensor_rig;                    // Static sensor → rig transform (owned)
    SE3Pose T_rig_world;                     // Rig → world transform (owned)

    // IPose interface implementation
    // Transform chain: sensor → rig → world
    [Differentiable]
    float3 transform_point(float3 point) {
        float3 p_rig = T_sensor_rig.transform_point(point);
        return T_rig_world.transform_point(p_rig);
    }

    // Transform direction through sensor → rig → world
    [Differentiable]
    float3 transform_direction(float3 direction) {
        float3 d_rig = T_sensor_rig.transform_direction(direction);
        return T_rig_world.transform_direction(d_rig);
    }

    // Inverse transform: world → rig → sensor
    [Differentiable]
    float3 inverse_transform_point(float3 point) {
        float3 p_rig = T_rig_world.inverse_transform_point(point);
        return T_sensor_rig.inverse_transform_point(p_rig);
    }

    [Differentiable]
    float3 inverse_transform_direction(float3 direction) {
        float3 d_rig = T_rig_world.inverse_transform_direction(direction);
        return T_sensor_rig.inverse_transform_direction(d_rig);
    }
}

// ============================================================================
// Trajectory Interfaces and Implementations
// ============================================================================

// Boundary mode for trajectory queries outside valid time range
enum BoundaryMode {
    CONSTANT = 0,      // Return boundary pose (first/last control pose) - safe default
    EXTRAPOLATE = 1,   // Extrapolate using boundary velocities - useful for predictions
    ERROR = 2          // Mark query as invalid via output flag - strict mode for debugging
}

// Trajectory interface for time-varying poses
// Uses float for time (can represent normalized [0,1] or absolute timestamps)
interface ITrajectory : IDifferentiable {
    // Get pose at specific time
    [Differentiable]
    IPose get_pose_at(float t);

    // Transform a point at a specific time
    [Differentiable]
    float3 transform_point_at(float3 point, float t);

    // Transform a direction at a specific time
    [Differentiable]
    float3 transform_direction_at(float3 direction, float t);
}

// Piecewise trajectory with float time values
// Uses linear interpolation for translations and SLERP for rotations
// Time can represent normalized [0,1] or absolute timestamps
//
// Requirements:
// - Minimum 1 control pose (can only return that pose)
// - Minimum 2 control poses for interpolation or EXTRAPOLATE boundary mode
struct PiecewiseTrajectory : ITrajectory {
    StructuredBuffer<SE3Pose> control_poses;     // (N,) control poses (N >= 1)
    StructuredBuffer<float> control_times;       // (N,) time values for control poses [0, 1]
    uint control_count;                          // Number of control poses
    
    // Helper function: Find the interval containing time t
    // Returns the index i such that control_times[i] <= t < control_times[i+1]
    uint find_interval(float t) {
        // Binary search for the interval
        uint left = 0;
        uint right = control_count - 1;
        
        while (right - left > 1) {
            uint mid = (left + right) / 2;
            if (control_times[mid] <= t) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return left;
    }
    
    // Get pose at specific time by interpolating between control poses
    // boundary_mode: Specifies how to handle queries outside valid time range
    // valid: Output flag - false if query is out of range and boundary_mode is ERROR,
    //        or if EXTRAPOLATE is requested but insufficient control poses (need >= 2)
    [Differentiable]
    SE3Pose get_pose_at(float t, BoundaryMode boundary_mode, out bool valid) {
        valid = true;
        
        // Handle insufficient control poses
        if (control_count == 0) {
            // No control poses - this should not happen in practice
            // Trajectory should always be constructed with at least 1 pose
            valid = false;
            return SE3Pose.identity();  // Return identity pose as safe fallback
        }
        if (control_count == 1) {
            // Only one control pose - can only return it (no interpolation possible)
            // EXTRAPOLATE mode requires at least 2 poses to compute velocity
            if (boundary_mode == BoundaryMode.EXTRAPOLATE) {
                valid = false;  // Cannot extrapolate with only 1 pose
            }
            return control_poses[0];
        }
        
        float t_min = control_times[0];
        float t_max = control_times[control_count - 1];
        
        // Check if out of bounds
        bool is_before = t < t_min;
        bool is_after = t > t_max;
        bool is_out_of_bounds = is_before || is_after;
        
        // Handle out of bounds based on boundary mode
        if (is_out_of_bounds) {
            if (boundary_mode == BoundaryMode.ERROR) {
                valid = false;
                // Return boundary pose but mark as invalid
                return is_before ? control_poses[0] : control_poses[control_count - 1];
            }
            else if (boundary_mode == BoundaryMode.CONSTANT) {
                // Return boundary pose
                return is_before ? control_poses[0] : control_poses[control_count - 1];
            }
            else if (boundary_mode == BoundaryMode.EXTRAPOLATE) {
                // Extrapolate using velocity from boundary interval
                if (is_before) {
                    // Extrapolate backward from first interval
                    SE3Pose p0 = control_poses[0];
                    SE3Pose p1 = control_poses[1];
                    
                    float t0 = control_times[0];
                    float t1 = control_times[1];
                    
                    float dt_interval = t1 - t0;
                    float dt_query = t0 - t;
                    
                    float alpha = -dt_query / dt_interval;
                    
                    float3 trans = lerp(p0.se3.trans.p, p1.se3.trans.p, alpha);
                    float4 rot = quaternion::slerp(p0.se3.so3.unit_quaternion.q, p1.se3.so3.unit_quaternion.q, alpha);
                    return SE3Pose.from_translation_quaternion(trans, rot);
                }
                else {
                    // Extrapolate forward from last interval
                    uint i = control_count - 2;
                    uint j = control_count - 1;
                    SE3Pose p0 = control_poses[i];
                    SE3Pose p1 = control_poses[j];
                    
                    float ti = control_times[i];
                    float tj = control_times[j];
                    
                    float dt_interval = tj - ti;
                    float dt_query = t - tj;
                    
                    float alpha = 1.0 + dt_query / dt_interval;
                    
                    float3 trans = lerp(p0.se3.trans.p, p1.se3.trans.p, alpha);
                    float4 rot = quaternion::slerp(p0.se3.so3.unit_quaternion.q, p1.se3.so3.unit_quaternion.q, alpha);
                    return SE3Pose.from_translation_quaternion(trans, rot);
                }
            }
        }
        
        // Within bounds - interpolate normally
        uint i = find_interval(t);
        uint j = i + 1;
        
        float t0 = control_times[i];
        float t1 = control_times[j];
        
        // Compute interpolation parameter
        float alpha = (t - t0) / (t1 - t0);
        
        SE3Pose p0 = control_poses[i];
        SE3Pose p1 = control_poses[j];
        
        // Linear interpolation for translation
        float3 trans = lerp(p0.se3.trans.p, p1.se3.trans.p, alpha);
        
        // SLERP for rotation
        float4 rot = quaternion::slerp(p0.se3.so3.unit_quaternion.q, p1.se3.so3.unit_quaternion.q, alpha);
        
        return SE3Pose.from_translation_quaternion(trans, rot);
    }
    
    // Convenience method: Get pose at specific time with default CONSTANT boundary mode
    [Differentiable]
    SE3Pose get_pose_at(float t) {
        bool valid;
        return get_pose_at(t, BoundaryMode.CONSTANT, valid);
    }
    
    // Transform a point at a specific time with boundary mode control
    [Differentiable]
    float3 transform_point_at(float3 point, float t, BoundaryMode boundary_mode, out bool valid) {
        SE3Pose pose = get_pose_at(t, boundary_mode, valid);
        return pose.transform_point(point);
    }
    
    // Transform a point at a specific time with default CONSTANT boundary mode
    [Differentiable]
    float3 transform_point_at(float3 point, float t) {
        SE3Pose pose = get_pose_at(t);
        return pose.transform_point(point);
    }
    
    // Transform a direction at a specific time with boundary mode control
    [Differentiable]
    float3 transform_direction_at(float3 direction, float t, BoundaryMode boundary_mode, out bool valid) {
        SE3Pose pose = get_pose_at(t, boundary_mode, valid);
        return pose.transform_direction(direction);
    }
    
    // Transform a direction at a specific time with default CONSTANT boundary mode
    [Differentiable]
    float3 transform_direction_at(float3 direction, float t) {
        SE3Pose pose = get_pose_at(t);
        return pose.transform_direction(direction);
    }
}

// Dynamic pose: time-varying pose over normalized time [0, 1]
// Used for rolling shutter, motion during exposure, short-duration pose variations
struct DynamicPose : ITrajectory {
    PiecewiseTrajectory trajectory;  // Normalized time [0, 1]
    
    // ITrajectory interface implementation
    [Differentiable]
    SE3Pose get_pose_at(float t) {
        return trajectory.get_pose_at(t);
    }
    
    [Differentiable]
    SE3Pose get_pose_at(float t, BoundaryMode boundary_mode, out bool valid) {
        return trajectory.get_pose_at(t, boundary_mode, valid);
    }
    
    [Differentiable]
    float3 transform_point_at(float3 point, float t) {
        return trajectory.transform_point_at(point, t);
    }
    
    [Differentiable]
    float3 transform_point_at(float3 point, float t, BoundaryMode boundary_mode, out bool valid) {
        return trajectory.transform_point_at(point, t, boundary_mode, valid);
    }
    
    [Differentiable]
    float3 transform_direction_at(float3 direction, float t) {
        return trajectory.transform_direction_at(direction, t);
    }
    
    [Differentiable]
    float3 transform_direction_at(float3 direction, float t, BoundaryMode boundary_mode, out bool valid) {
        return trajectory.transform_direction_at(direction, t, boundary_mode, valid);
    }
    
    // Additional method specific to DynamicPose (not part of ITrajectory interface)
    // Get mean pose over the normalized time range [0, 1]
    // Useful for rolling shutter approximations
    [Differentiable]
    SE3Pose mean() {
        return get_pose_at(0.5);  // Return pose at middle of time range
    }
}

// ============================================================================
// Top-level wrapper functions for slangpy compatibility
// ============================================================================
// Slangpy 0.38.1 cannot instantiate structs from Python directly.
// These wrapper functions create structs internally and expose their methods.

[Differentiable]
float3 se3pose_transform_point(float3 translation, float4 rotation, float3 point) {
    SE3Pose pose = SE3Pose.from_translation_quaternion(translation, rotation);
    return pose.transform_point(point);
}

[Differentiable]
float3 se3pose_transform_direction(float3 translation, float4 rotation, float3 direction) {
    SE3Pose pose = SE3Pose.from_translation_quaternion(translation, rotation);
    return pose.transform_direction(direction);
}

[Differentiable]
float3 se3pose_inverse_transform_point(float3 translation, float4 rotation, float3 point) {
    SE3Pose pose = SE3Pose.from_translation_quaternion(translation, rotation);
    return pose.inverse_transform_point(point);
}

[Differentiable]
float3 se3pose_inverse_transform_direction(float3 translation, float4 rotation, float3 direction) {
    SE3Pose pose = SE3Pose.from_translation_quaternion(translation, rotation);
    return pose.inverse_transform_direction(direction);
}

[Differentiable]
float4x4 se3pose_to_matrix(float3 translation, float4 rotation) {
    SE3Pose pose = SE3Pose.from_translation_quaternion(translation, rotation);
    return pose.to_matrix();
}

[Differentiable]
void se3pose_from_matrix(float4x4 matrix, out float3 translation, out float4 rotation) {
    translation = float3(matrix[0].w, matrix[1].w, matrix[2].w);
    rotation = quaternion::fromMatrix(float3x3(matrix[0].xyz, matrix[1].xyz, matrix[2].xyz));
}

[Differentiable]
float4x4 se3pose_to_inverse_matrix(float3 translation, float4 rotation, no_diff bool wxyz_format = false) {
    // Build SE3 inverse matrix: T^-1 = [R^T | -R^T * t; 0 0 0 1]
    float3x3 R = quaternion::toMatrix(quaternion::wxyzToXyzw(rotation, wxyz_format));
    float3x3 R_T = transpose(R);
    float3 neg_R_T_t = -float3(
        R_T[0][0] * translation.x + R_T[0][1] * translation.y + R_T[0][2] * translation.z,
        R_T[1][0] * translation.x + R_T[1][1] * translation.y + R_T[1][2] * translation.z,
        R_T[2][0] * translation.x + R_T[2][1] * translation.y + R_T[2][2] * translation.z
    );
    return float4x4(
        float4(R_T[0], neg_R_T_t.x),
        float4(R_T[1], neg_R_T_t.y),
        float4(R_T[2], neg_R_T_t.z),
        float4(0.0, 0.0, 0.0, 1.0)
    );
}
