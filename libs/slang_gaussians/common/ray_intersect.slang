// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

#include <slang_utils/extensions.slang>
#include <geometry/kernels/transform.slang>
#include <slang_gaussians/common/search.slang>

// This file contains the implementation of ray intersection with timed cuboids.
//
// It provides different intersection strategies with different benefits.
//
// It provides intersection handlers which compute the average timestamp of the
// rays intersecting with a cuboid track.

namespace ray_intersect {

    // This interface defines how to intersect a world ray with a timed cuboid.
interface IIntersectionStrategy {
    // Intersection when the pose needs to be interpolated.
    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float alpha,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_start_pose,
        transform::SE3 interpolation_end_pose,
    );

    // Intersection when the pose is directly available.
    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_pose,
    );
};

// This interface defines what to do when a ray intersects with a cuboid track.
//
// For the time being, the only thing needed is the ray timestamp, but we can
// pass more of other ray intersecting algorithm need things like the ray id, etc.
interface IIntersectionHandler {
    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    );
};

// This is the generic, non-differentiable implementation of the ray intersection algorithm.
void intersect_rays_with_timed_cuboids<
    IntersectionStrategy : IIntersectionStrategy,
    IntersectionHandler : IIntersectionHandler
    > (
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    IntersectionHandler intersection_handler,
) {
    let nb_rays = rays.size(0);
    let ray_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (ray_idx >= nb_rays) {
        return;
    }
    let nb_tracks = tracks_packinfo.size(0);
    let track_idx = cudaBlockIdx().y;
    if (track_idx >= nb_tracks) {
        return;
    }

    let track_start = tracks_packinfo.load(uint2(track_idx, 0));
    let track_length = tracks_packinfo.load(uint2(track_idx, 1));

    // The original CUDA implementation was using shared memory and copied the track
    // timestamps to it.  We couldn't see performance improvements by doing this and
    // it forced us to know the maximum track length in advance.  Therefore we just
    // read from global memory, which is likely to be cached in the most cases anyways.
    let track_timestamps = ImmutablePtr<int64_t>(tracks_timestamps.data_ptr_at(track_start));

    let ray_timestamp = rays_timestamps.load(ray_idx);

    if (track_length <= 0 || ray_timestamp < track_timestamps[0] || ray_timestamp > track_timestamps[track_length - 1]) {
        return;
    }

    typedef search::BinarySearch<int64_t, int32_t> SearchStrategy;
    let indices = search::find_surrounding_indices<SearchStrategy>(ray_timestamp, track_timestamps, track_length);

    let cuboid_dimensions = cuboids_dimensions.loadVec<3>(uint2(track_idx, 0));

    let ray_origin = rays.loadVec<3>(uint2(ray_idx, 0));
    let ray_direction = rays.loadVec<3>(uint2(ray_idx, 3));

    float2 t1t2;

    // We already checked that track is not empty, so we are guaranteed to have indices in range.
    if (indices._0 == indices._1) {
        let interpolation_pose = transform::SE3::from_vec(
            tracks_poses.loadVec<3>(uint2(track_start + indices._0, 0)),
            tracks_poses.loadVec<4>(uint2(track_start + indices._0, 3)),
        );
        t1t2 = IntersectionStrategy.intersect_world_ray_with_timed_cuboid(
            ray_origin,
            ray_direction,
            cuboid_dimensions,
            interpolation_pose,
        );
    }
    else {
        let interpolation_start_idx = indices._0;
        let interpolation_end_idx = indices._1;

        let interpolation_start_timestamp = track_timestamps[interpolation_start_idx];
        let interpolation_end_timestamp = track_timestamps[interpolation_end_idx];
        let interpolation_alpha = (ray_timestamp - interpolation_start_timestamp) / float(interpolation_end_timestamp - interpolation_start_timestamp);

        let interpolation_start_pose = transform::SE3::from_vec(
            tracks_poses.loadVec<3>(uint2(track_start + interpolation_start_idx, 0)),
            tracks_poses.loadVec<4>(uint2(track_start + interpolation_start_idx, 3)),
        );
        let interpolation_end_pose = transform::SE3::from_vec(
            tracks_poses.loadVec<3>(uint2(track_start + interpolation_end_idx, 0)),
            tracks_poses.loadVec<4>(uint2(track_start + interpolation_end_idx, 3)),
        );

        t1t2 = IntersectionStrategy.intersect_world_ray_with_timed_cuboid(
            ray_origin,
            ray_direction,
            interpolation_alpha,
            cuboid_dimensions,
            interpolation_start_pose,
            interpolation_end_pose
        );
    }

    // Forward processing of the intersection results to the handler.
    intersection_handler.handle_intersection(t1t2, track_idx, ray_timestamp);
}


// Intersect a ray with an axis-aligned bounding box.
// It takes the inverse of the ray direction to avoid computing it
// several times when intersecting the same ray with multiples objects.
static float2 ray_aabb_intersect(
    float3 ray_origin,
    float3 inv_ray_direction,
    float3 aabb_min,
    const float3 aabb_max
) {
    let t_min = (aabb_min - ray_origin) * inv_ray_direction;
    let t_max = (aabb_max - ray_origin) * inv_ray_direction;

    let _t1 = min(t_min, t_max);
    let _t2 = max(t_min, t_max);
    let t1  = max(max(_t1.x, _t1.y), _t1.z);
    let t2  = min(min(_t2.x, _t2.y), _t2.z);

    if (t1 > t2) {
        return float2(-1.0f);
    }
    return float2(t1, t2);
}


// This intersection strategy matches the original CUDA implementation.
// It uses quaternion slerp to interpolate the rotations and convert it
// to a rotation matrix before applying it to the ray.
//
// This approach is slightly faster than using the SE3 class.
struct QuatSlerpMatrixIntersectionStrategy : IIntersectionStrategy {
    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float alpha,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_start_pose,
        transform::SE3 interpolation_end_pose,
    ) {
        let ray_o_world = ray_origin_world;
        let ray_d_world = ray_direction_world;
        let pose_start = interpolation_start_pose.to_vec();
        let c_start = pose_start._0;
        let q_start = pose_start._1;
        let pose_end = interpolation_end_pose.to_vec();
        let c_end = pose_end._0;
        let q_end = pose_end._1;

        let c_raytime = (1.0f - alpha) * c_start + alpha * c_end;
        let R_raytime = transpose(quaternion::convertNormalizedQuatToMatrix(quaternion::slerp(q_start, q_end, alpha)));

        let ray_o_box = mul(R_raytime, ray_o_world - c_raytime);
        let ray_d_box = mul(R_raytime, ray_d_world);

        let aabb_max = cuboid_dimensions * 0.5;
        let aabb_min = -aabb_max;
        let t1t2 = ray_aabb_intersect(ray_o_box, 1.0f / ray_d_box, aabb_min, aabb_max);

        return t1t2;
    }

    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_pose,
    ) {
        let ray_o_world = ray_origin_world;
        let ray_d_world = ray_direction_world;
        let pose_raytime = interpolation_pose.to_vec();
        let c_raytime = pose_raytime._0;
        let q_raytime = transpose(quaternion::convertNormalizedQuatToMatrix(pose_raytime._1));

        let ray_o_box = mul(q_raytime, ray_o_world - c_raytime);
        let ray_d_box = mul(q_raytime, ray_d_world);

        let aabb_max = cuboid_dimensions * 0.5;
        let aabb_min = -aabb_max;
        let t1t2 = ray_aabb_intersect(ray_o_box, 1.0f / ray_d_box, aabb_min, aabb_max);

        return t1t2;
    }
};

// This intersection strategy uses Lie algebra to interpolate the rotation.
// It matches the rest of the tracks interpolation approach.
struct LieIntersectionStrategy : IIntersectionStrategy {
    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float alpha,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_start_pose,
        transform::SE3 interpolation_end_pose,
    ) {
        let ray_time_pose = transform::interpolate(interpolation_start_pose, interpolation_end_pose, alpha);
        return intersect_world_ray_with_timed_cuboid(
            ray_origin_world, ray_direction_world, cuboid_dimensions, ray_time_pose
        );
    }

    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_pose,
    ) {
        let ray_time_pose = interpolation_pose;
        let ray_time_translation = ray_time_pose.translation();
        let ray_time_rotation = ray_time_pose.so3.inv();

        let ray_origin_box = ray_time_rotation * (transform::Point(ray_origin_world) - ray_time_translation);
        let ray_direction_box = ray_time_rotation * transform::Point(ray_direction_world);

        let aabb_max = cuboid_dimensions * 0.5;
        let aabb_min = -aabb_max;

        let t1t2 = ray_aabb_intersect(ray_origin_box.p, 1.0f / ray_direction_box.p, aabb_min, aabb_max);

        return t1t2;
    }
};


namespace timestamps_estimation {

//
// The following intersection handlers are used to sum the timestamps of the rays
// intersecting with a cuboid track.
//
// They offer different performance trade-offs dependening on how much contention
// there is on the global memory, i.e. how many rays are intersecting with the same track.
//

// To avoid overflows, the intersection handlers add the difference between the ray timestamp
// and the frame timestamp.  By accumulating deltas instead of absolute values, we don't overflow
// in 64 bits.
//
// We could do addition on more than 64 bits or use min / max to be even more sure not to overflow,
// but this is enough for now.

// This handler simply atomically adds the delta to the result and increment the count
// using atomic additions to global memory.
struct AtomicAddIntersectionHandler : IIntersectionHandler {
    TensorView<int64_t> output_tracks_timestamps_sum_and_count;
    int64_t frame_timestamp;

    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    ) {
        if (t1t2.y > 0) {
            int64_t old_value;
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 0), ray_timestamp - frame_timestamp, old_value);
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 1), 1, old_value);
        }
    }
}

// This handler performs a wave reduction to sum the deltas and the count
// without resorting to atomic operations, potentially dividing the number of
// atomic operations by 32.
//
// However, when there is very low contention, this approach actually adds overhead
// over the naive one, but because it always performs the (fast) reduction, which
// would be slower than doing almost no atomic operations.
struct WaveReduceIntersectionHandler : IIntersectionHandler {
    TensorView<int64_t> output_tracks_timestamps_sum_and_count;
    int64_t frame_timestamp;

    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    ) {
        int64_t output_delta;
        int32_t count;
        if (t1t2.y > 0) {
            output_delta = ray_timestamp - frame_timestamp;
            count = 1;
        }
        else {
            output_delta = 0;
            count = 0;
        }
        output_delta = WaveActiveSum(output_delta);
        count = WaveActiveSum(count);
        if (WaveIsFirstLane() && count > 0) {
            int64_t old_value;
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 0), output_delta, old_value);
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 1), count, old_value);
        }
    }
}

// This handler uses shared memory to accumulate the deltas and the count
// so the atomic operations are performed on the shared memory instead of the
// global memory.
//
// This approach can also be slower than the naive one when there is very low
// contention, because we go to shared memory before going to global memory,
// which is slower than going directly to global memory when there are near-zero
// operations to do.
struct SharedAccumulatorsIntersectionHandler : IIntersectionHandler {
    TensorView<int64_t> output_tracks_timestamps_sum_and_count;
    int64_t frame_timestamp;

    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    ) {
        static groupshared int64_t shared_accumulators[2];
        if (cudaThreadIdx().x == 0) {
            shared_accumulators[0] = 0;
            shared_accumulators[1] = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        if (t1t2.y > 0) {
            InterlockedAdd(shared_accumulators[0], ray_timestamp - frame_timestamp);
            InterlockedAdd(shared_accumulators[1], 1);
        }
        GroupMemoryBarrierWithGroupSync();
        if (cudaThreadIdx().x == 0 && shared_accumulators[1] > 0) {
            int64_t old_value;
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 0), shared_accumulators[0], old_value);
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 1), shared_accumulators[1], old_value);
        }
    }
}

// This handler combines the wave reduction and the shared memory approach,
// performing warp reduction to avoid atomic operations, then using shared
// memory to do the reduction at the block level using fewer atomic operations
// and doing them on shared memory, resulting in a single atomic operation per
// block on global memory.
struct WaveReduceSharedAccumulatorsIntersectionHandler : IIntersectionHandler {
    TensorView<int64_t> output_tracks_timestamps_sum_and_count;
    int64_t frame_timestamp;

    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    ) {
        static groupshared int64_t shared_accumulators[2];
        if (cudaThreadIdx().x == 0) {
            shared_accumulators[0] = 0;
            shared_accumulators[1] = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        int64_t output_delta;
        int32_t count;
        if (t1t2.y > 0) {
            output_delta = ray_timestamp - frame_timestamp;
            count = 1;
        }
        else {
            output_delta = 0;
            count = 0;
        }
        output_delta = WaveActiveSum(output_delta);
        count = WaveActiveSum(count);
        if (WaveIsFirstLane() && count > 0) {
            InterlockedAdd(shared_accumulators[0], output_delta);
            InterlockedAdd(shared_accumulators[1], count);
        }
        GroupMemoryBarrierWithGroupSync();
        if (cudaThreadIdx().x == 0 && shared_accumulators[1] > 0) {
            int64_t old_value;
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 0), shared_accumulators[0], old_value);
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 1), shared_accumulators[1], old_value);
        }
    }
}

}; // namespace timestamps_estimation

}; // namespace ray_intersect
