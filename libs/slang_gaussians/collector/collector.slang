// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

// Gaussian Parameter Collection Kernels
//
// This module implements GPU kernels for collecting, transforming, and concatenating
// Gaussian parameters from multiple layers into unified output tensors. The kernels
// support various layer types with different activation functions and transformations.
//
// The module uses a task-based architecture where each operation (e.g., rotation
// normalization, scale exponentiation) is implemented as an ITask. Multiple tasks
// are fused into a single kernel via the run_tasks() template function, enabling
// efficient fused execution with minimal kernel launch overhead.

module collector;

#include <slang_gaussians/common/ray_intersect.slang>
#include <slang_gaussians/common/sigmoid.slang>

namespace utils
{

// Get the 2D index for a tensor with rows of N components.
[ForceInline]
uint2 get_index<let N : int>(no_diff uint index)
{
    const uint index_i = index / N;
    const uint index_j = index - index_i * N;
    return uint2(index_i, index_j);
}

// This function passes a value through, but only optionally allows the
// gradient to pass through.
[ForwardDerivative(__pass_forward)]
[BackwardDerivative(__pass_backward)]
[Differentiable]
[ForceInline]
__generic<T: IDifferentiable>
T pass(T x, no_diff bool pass_gradient) {
    return x;
}

[ForceInline]
__generic<T: IDifferentiable>
DifferentialPair<T> __pass_forward(
    DifferentialPair<T> x, no_diff bool pass_gradient
    ) {
    if (pass_gradient) {
        return x;
    }
    else {
        return diffPair(x.p, T.dzero());
    }
}

[ForceInline]
__generic<T: IDifferentiable>
void __pass_backward(
    inout DifferentialPair<T> x, no_diff bool pass_gradient, T.Differential dOut
    ) {
    if (!pass_gradient) {
        dOut = T.dzero();
    }
    x = diffPair(x.p, dOut);
}

}; // namespace utils.


// Interface for collection tasks that can be fused into a single kernel.
//
// Each task processes one aspect of Gaussian parameters (e.g., positions,
// rotations, densities). Multiple tasks implementing this interface can be
// batched together via run_tasks() for efficient parallel execution.
public interface ITask
{
    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset);
};

// This is the generic function that provides a specific interface of what parameters
// it will collect, but is generic and therefore can be bound to any implementation,
// with generic parameters for each implementation, providing each of the input /
// output buffers or data for each collection.
[Differentiable]
public void run_tasks<each T>(
    no_diff uint out_offset,
    no_diff uint count,
    expand each T args,
    )
    where T : ITask
{
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    expand (each args).run(tid, count, out_offset);
}


//
// Implementations for base layer.
//

// A couple of helpers for simple things like copy with single function applied to each value.
interface IUnaryFunction
{
    [Differentiable]
    static float apply(float x);
};

struct CopyFunction : IUnaryFunction
{
    [Differentiable]
    static float apply(float x) { return x; }
};

struct ExpFunction : IUnaryFunction
{
    [Differentiable]
    static float apply(float x) { return exp(x); }
};

struct SigmoidFunction : IUnaryFunction
{
    [Differentiable]
    static float apply(float x) { return sigmoid::standard(x); }
};

// Helper function to copy while accessing memory with smaller stride than
// if each element was accessing their own index.
//
// Note: their might still be something more optimal to do if there are less
// than `count` threads currently active, because this potentially cause
// current threads to jump by "count" when they count by jumping just by
// the number of active threads.
//
// OPTME: In the case of the pure copy, we could do vectorized float4 load.
// Manual backward for apply_to_buffer to avoid Slang's autodiff unrolling
// which causes register pressure blowup.
// OffsetDim: column offset added to the destination index (0 for simple copy).
[BackwardDerivative(apply_to_buffer_bwd)]
void apply_to_buffer<UnaryFunction, let N : int, let OffsetDim : int, A : IDiffTensorWrapper>(
    no_diff DiffTensorView<float, A> output, no_diff DiffTensorView<float, A> input,
    no_diff uint tid, no_diff uint count, no_diff uint out_offset
    ) where UnaryFunction : IUnaryFunction
{
    [ForceUnroll]
    for (int i = 0; i < N; ++i) {
        const uint2 index = utils::get_index<N>(i * count + tid);
        const uint2 out_index = uint2(index.x + out_offset, uint(OffsetDim) + index.y);
        output.primal.store(out_index, UnaryFunction.apply(input.primal.load(index)));
    }
}

void apply_to_buffer_bwd<UnaryFunction, let N : int, let OffsetDim : int, A : IDiffTensorWrapper>(
    no_diff DiffTensorView<float, A> output, no_diff DiffTensorView<float, A> input,
    no_diff uint tid, no_diff uint count, no_diff uint out_offset
    ) where UnaryFunction : IUnaryFunction
{
    [ForceUnroll]
    for (int i = 0; i < N; ++i) {
        const uint2 index = utils::get_index<N>(i * count + tid);
        const uint2 out_index = uint2(index.x + out_offset, uint(OffsetDim) + index.y);
        float d_out = output.diff.storeOnce_backward<float, 2>(out_index);
        // Chain rule: use Slang's autodiff for the unary function derivative.
        var dp = diffPair(input.primal.load(index));
        bwd_diff(UnaryFunction.apply)(dp, d_out);
        input.diff.loadOnce_backward<float, 2>(index, dp.d);
    }
};

// It would be nice to be able to define a generic implementation of ITask
// which would take a UnaryFunction and N as a parameter, and simply typedef to it,
// but SlangTorch does not export the parameters to it.
public struct Collector_Copy<let N : int> : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    public void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset)
    {
        apply_to_buffer<CopyFunction, N, 0>(output, input, tid, count, out_offset);
    }
};

// Positions.
// Use Collector_Copy

// Rotations.
public struct RotationsCollector_Normalize : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    public void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset)
    {
        const uint out_index = out_offset + tid;

        // OPTME: We could do vectorized float4 load.
        var rot = input.loadVecOnce<4>(uint2(tid,0));
        rot = quaternion::normalizeSafe(rot);
        output.storeVecOnce<4>(uint2(out_index,0), rot);
    }
};

// Scales.
public struct ScalesCollector_Exp : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    public void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset)
    {
        apply_to_buffer<ExpFunction, 3, 0>(output, input, tid, count, out_offset);
    }
};

// Densities.
public struct DensitiesCollector_Sigmoid : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    public void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset)
    {
        apply_to_buffer<SigmoidFunction, 1, 0>(output, input, tid, count, out_offset);
    }
};

// Extra signal.
// Use Collector_Copy

// Camera extra signal.
// Use Collector_Copy

// Lidar extra signal.
// Use Collector_Copy


//
// Implementations for spherical harmonics layer.
//

public struct SphericalFeaturesCollector_Copy<let AlbedoDim : int, let SpecularDim : int> : ITask
{
    DiffTensorView albedo;
    DiffTensorView specular;
    DiffTensorView output;

    [Differentiable]
    public void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset)
    {
        static const var nb_albedo_components = AlbedoDim;
        static const var nb_specular_components = SpecularDim;
        static const var nb_components = nb_albedo_components + nb_specular_components;

        [ForceUnroll]
        for (int i = 0; i < nb_components; ++i) {
            const var index = utils::get_index<nb_components>(i * count + tid);
            float value;
            if (index.y < nb_albedo_components) {
                const var in_index = index;
                value = albedo.loadOnce(in_index);
            }
            else {
                const var in_index = uint2(index.x, index.y - nb_albedo_components);
                value = specular.loadOnce(in_index);
            }
            const var out_index = uint2(index.x + out_offset, index.y);
            output.storeOnce(out_index, value);
        }
    }
};

// Interface for time embedding strategies.
// 
// Converts integer timestamps (microseconds) to normalized floating-point
// values suitable for Fourier features or hash grid inputs.
public interface ITimeEmbedding
{
    [Differentiable]
    public float embed(no_diff uint tid, no_diff int64_t timestamp);
};

public struct IndividualRemapTimeInputEmbedding : ITimeEmbedding
{
    no_diff TensorView<int32_t> instance_idx;
    no_diff TensorView<int64_t> timestamps_ranges;
    no_diff float remap_min;
    no_diff float remap_max;

    [Differentiable]
    [NoDiffThis]
    public float embed(no_diff uint tid, no_diff int64_t timestamp)
    {
        var instance_id = instance_idx[tid];
        var range_min = timestamps_ranges[uint2(instance_id, 0)];
        var range_max = timestamps_ranges[uint2(instance_id, 1)];
        // This could be done with a single fma if the ranges were recomputed differently.
        const float ratio = (timestamp - range_min) / float(range_max - range_min);
        const float time_embed = clamp(ratio, 0, 1) * (remap_max - remap_min) + remap_min;
        return time_embed;
    }
};

public struct HolisticRemapTimeInputEmbedding : ITimeEmbedding
{
    no_diff int64_t timestamps_us_min;
    no_diff int64_t timestamps_us_max;
    no_diff float remap_min;
    no_diff float remap_max;

    [Differentiable]
    [NoDiffThis]
    public float embed(no_diff uint tid, no_diff int64_t timestamp)
    {
        // This could be done with a single fma if the ranges were recomputed differently.
        const float ratio = (timestamp - timestamps_us_min) / float(timestamps_us_max - timestamps_us_min);
        const float time_embed = ratio * (remap_max - remap_min) + remap_min;
        return time_embed;
    }
};


// Individual step-function time embedding with per-instance ranges (n_dims=1 only)
// Based on NSC paper (https://arxiv.org/abs/2306.07970)
//
// Parameters:
//   u: [n_tracks, 1, n_steps] - learnable step positions
//   timestamps_ranges: [n_tracks, 2] - per-track [min, max] timestamps in microseconds
//   beta: stepness parameter (in microseconds, gets scaled by time range)
//
// Output: single float per Gaussian (mean over n_steps)
public struct IndividualStepTimeInputEmbedding<let NumSteps : int> : ITimeEmbedding
{
    no_diff TensorView<int32_t> instance_idx;       // [num_gaussians]
    no_diff TensorView<int64_t> timestamps_ranges;  // [num_tracks, 2]
    DiffTensorView u;                                // [num_tracks, 1, n_steps]
    no_diff TensorView<float> beta;                 // stepness in microseconds

    [Differentiable]
    public float embed(no_diff uint tid, no_diff int64_t timestamp)
    {
        let instance_id = instance_idx[tid];
        let range_min = timestamps_ranges[uint2(instance_id, 0)];
        let range_max = timestamps_ranges[uint2(instance_id, 1)];
        
        // Compute ratio
        let time_range = float(range_max - range_min);
        let ratio = float(timestamp - range_min) / time_range;
        
        // Compute beta_scaled = beta / time_range (matching Python exactly)
        // Then clamp |beta_scaled| to avoid division by zero
        let beta_scaled = beta[0] / time_range;
        let abs_beta = max(abs(beta_scaled), 1e-3f);
        
        // Compute mean over steps for n_dims=1
        // Python: output = ratio[:, None, None] - self.u[instance_idx]
        //         time_emb = output.mean(dim=-1)  # shape [N, n_dims]
        float sum = 0.0f;
        
        [ForceUnroll]
        for (int s = 0; s < NumSteps; ++s)
        {
            // Load u[instance_id, 0, s]
            // Using loadEx because multiple Gaussians share the same instance_id,
            // so gradients need to be accumulated atomically during backprop
            let u_val = u.loadEx(uint3(instance_id, 0, s));
            
            let diff = ratio - u_val;
            
            // Step function:
            //   if diff <= 0: 0.5 * exp(diff / abs_beta)
            //   else: 1 - 0.5 * exp(-diff / abs_beta)
            float step_val;
            if (diff <= 0.0f) {
                step_val = 0.5f * exp(diff / abs_beta);
            } else {
                step_val = 1.0f - 0.5f * exp(-diff / abs_beta);
            }
            sum += step_val;
        }
        
        // Mean over steps
        return sum / float(NumSteps);
    }
}

// Interface for fetching timestamps based on different query strategies.
public interface ITimestampsFetcher {
    int64_t fetch(no_diff uint track_id);
}

public struct TimestampsGlobalFetcher : ITimestampsFetcher {
    // Declare this member public so constructor can be generated automatically.
    public no_diff int64_t timestamp;

    int64_t fetch(no_diff uint track_id)
    {
        return timestamp;
    }
};

public struct TimestampsPerTrackFetcher : ITimestampsFetcher {
    TensorView<int64_t> timestamps;

    int64_t fetch(no_diff uint track_id)
    {
        return timestamps.load(track_id);
    }
};

// Instance embedding
public struct WeightedInstanceInputEmbedding<let EmbeddingDim : int>
{
    no_diff TensorView<int32_t> instance_idx;    // [N]
    DiffTensorView embedding_weights;            // [num_instances, embedding_dim]
    DiffTensorView output;                       // Output tensor
    
    // Embed to a specific output row starting at given column
    // col_start specifies where the embedding begins (e.g., 3 if after xyz)
    [Differentiable]
    public void embed_to_row(no_diff uint tid, no_diff uint output_row, no_diff uint col_start)
    {
        let instance_id = instance_idx[tid];
        [ForceUnroll]
        for (int i = 0; i < EmbeddingDim; ++i) {
            // Use loadEx() instead of loadOnce() because multiple samples may have the same
            // instance_id, so gradients need to be accumulated atomically during backprop.
            let emb_value = embedding_weights.loadEx(uint2(instance_id, i));
            
            // Use storeOnce() because each thread writes to a unique output location (unique output_row).
            output.storeOnce(uint2(output_row, col_start + i), emb_value);
        }
    }
};

[Differentiable]
void compute_spherical_features_fourier<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    TimestampsFetcher : ITimestampsFetcher,
    TimeEmbedding : ITimeEmbedding,
    >(
    no_diff uint tid, no_diff uint count, no_diff uint out_offset,
    TimestampsFetcher timestamps_fetcher,
    TimeEmbedding time_embedding,
    DiffTensorView albedo,
    DiffTensorView specular,
    DiffTensorView output,
    )
{
    // We do the Fourier transform in double-precision since it's sensitive.
    // We can trade-off quality for speed later if we prefer.
    typedef double scalar_t;

    const int64_t timestamp = timestamps_fetcher.fetch(tid);
    const scalar_t time_embed = (scalar_t)time_embedding.embed(tid, timestamp);

    scalar_t feature_albedo[AlbedoDim];
    [ForceUnroll]
    for (int i = 0; i < AlbedoDim; ++i) {
        feature_albedo[i] = 0;
    }

    [ForceUnroll]
    for (int j = 0; j < FourierDim; ++j) {
        static const scalar_t PI = constants::pi<scalar_t>();
        scalar_t value;
        if (j % 2 == 0) {
            const scalar_t x = PI * time_embed * j;
            value = cos(x);
        }
        else {
            const scalar_t x = PI * time_embed * (j + 1);
            value = sin(x);
        }

        [ForceUnroll]
        for (int i = 0; i < AlbedoDim; ++i) {
            feature_albedo[i] += value * (scalar_t)albedo.loadOnce(uint3(tid, j, i));
        }
    }

    const uint out_index = out_offset + tid;
    [ForceUnroll]
    for (int i = 0; i < AlbedoDim; ++i) {
        output.storeOnce(uint2(out_index, i), (float)feature_albedo[i]);
    }

    // Specular copy with manual backward.
    apply_to_buffer<CopyFunction, SpecularDim, AlbedoDim>(output, specular, tid, count, out_offset);
}

// slangtorch has problems with kernel parameters with generic parts.  If
// the same struct is instantiated multiple times with a different concrete
// type in the same compile unit, slangtorch will generate more than one
// functions with the same name.  To workaround this, we create 2 separate
// types, one per concrete embedding.
public struct SphericalFeaturesCollector_Fourier_Individual<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    > : ITask
{
    IndividualRemapTimeInputEmbedding time_embedding;
    no_diff int64_t timestamp;
    DiffTensorView albedo;
    DiffTensorView specular;
    DiffTensorView output;

    [Differentiable]
    public void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset)
    {
        let timestamps_fetcher = TimestampsGlobalFetcher(timestamp);
        compute_spherical_features_fourier<AlbedoDim, SpecularDim, FourierDim>(
            tid, count, out_offset, timestamps_fetcher, time_embedding, albedo, specular, output,
        );
    }
};

public struct SphericalFeaturesCollector_Fourier_Holistic<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    > : ITask
{
    HolisticRemapTimeInputEmbedding time_embedding;
    no_diff int64_t timestamp;
    DiffTensorView albedo;
    DiffTensorView specular;
    DiffTensorView output;

    [Differentiable]
    public void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset)
    {
        let timestamps_fetcher = TimestampsGlobalFetcher(timestamp);
        compute_spherical_features_fourier<AlbedoDim, SpecularDim, FourierDim>(
            tid, count, out_offset, timestamps_fetcher, time_embedding, albedo, specular, output,
        );
    }
};

public struct SphericalFeaturesCollector_Fourier_IndividualStep<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    let NumSteps : int,
    > : ITask
{
    IndividualStepTimeInputEmbedding<NumSteps> time_embedding;
    no_diff int64_t timestamp;
    DiffTensorView albedo;
    DiffTensorView specular;
    DiffTensorView output;

    [Differentiable]
    public void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset)
    {
        let timestamps_fetcher = TimestampsGlobalFetcher(timestamp);
        compute_spherical_features_fourier<AlbedoDim, SpecularDim, FourierDim>(
            tid, count, out_offset, timestamps_fetcher, time_embedding, albedo, specular, output,
        );
    }
};

//
// Implementations for dynamic rigid layer.
//

// Track calibration.
interface TracksCalib
{
    [Differentiable]
    transform::SE3 calib(no_diff uint pose_idx);
};

struct DirectTracksCalib : TracksCalib
{
    no_diff TensorView<bool> gradient_mask;
    no_diff TensorView<float> tracks_poses;
    DiffTensorView tracks_delta_q;
    DiffTensorView tracks_delta_t;

    [Differentiable]
    transform::SE3 calib(no_diff uint pose_idx)
    {
        const bool use_gradient = gradient_mask.load(pose_idx);
        const float3 current_tracks_delta_t = utils::pass(
            tracks_delta_t.loadVecOnce<3>(uint2(pose_idx, 0)), use_gradient
            );
        const float4 current_tracks_delta_q = quaternion::normalizeSafe(
            utils::pass(
                tracks_delta_q.loadVecOnce<4>(uint2(pose_idx, 0)), use_gradient
                )
            );

        const float3 tracks_t = no_diff tracks_poses.loadVec<3>(uint2(pose_idx, 0));
        float4 tracks_q = no_diff tracks_poses.loadVec<4>(uint2(pose_idx, 3));
        // OPTME: Tracks rotations should already be normalized.
        tracks_q = quaternion::normalizeSafe(tracks_q);

        // Tracks and calibration rotations are in xyzw order.
        // We can use them directly.
        const var delta = transform::SE3.from_vec(current_tracks_delta_t, current_tracks_delta_q);
        const var pose = transform::SE3.from_vec(tracks_t, tracks_q);
        return delta * pose;
    }
};

// We create a kernel entry point to be able to call the calibration.
//
// If we were not using the calibrated tracks to do ray intersection in order to
// estimate the best timestamp per track, we would not need to launch a separate
// kernel like this and we could just do the calibration at the same time as we
// to the interpolation, only on the poses we actually read.  But because ray
// intersection will read the same transform several times, we do one pass of
// calibration before ray intersection.
//
// These calibrated poses will be used for pose interpolation, since we already
// have them.
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void tracks_calib_direct(
    no_diff uint count,
    DirectTracksCalib calib,
    DiffTensorView output_poses,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    let pose = calib.calib(tid).to_vec();

    output_poses.storeVecOnce<3>(uint2(tid, 0), pose._0);
    output_poses.storeVecOnce<4>(uint2(tid, 3), pose._1);
}

// We create a kernel entry point to be able to call the tracks timestamps estimation.
//
// This kernel is not differentiable.
[CUDAKernel]
[AutoPyBindCUDA]
void estimate_tracks_timestamps(
    TensorView<int64_t> output_tracks_timestamps_sum_and_count,
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    no_diff int64_t frame_timestamp,
) {
    // We choose the intersection strategy which matches our tracks interpolation method.
    typedef ray_intersect::LieIntersectionStrategy IntersectionStrategy;

    // Intersection handler performance depends on how many concurrent threads (rays) are
    // intersecting with the same track.
    // We choose the intersection handler which has a slightly worst performance in
    // the low contention case, but scales the better in high contention to have
    // stable performance regardless of the contention level.
    // If this becomes relevant we can optimize later.
    typedef ray_intersect::timestamps_estimation::WaveReduceSharedAccumulatorsIntersectionHandler IntersectionHandler;

    ray_intersect::intersect_rays_with_timed_cuboids<IntersectionStrategy, IntersectionHandler>(
        rays,
        rays_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        cuboids_dimensions,
        IntersectionHandler(output_tracks_timestamps_sum_and_count, frame_timestamp),
    );
}

// Track interpolation.
interface IPoseFetcher {
    [Differentiable]
    [ForceInline]
    static transform::SE3 fetch(
        DiffTensorView tracks_poses,
        int32_t index_left, int32_t index_right,
        int64_t time_left, int64_t time_right, int64_t timestamp
        );
}

// Note that when we are doing track interpolation, we can use loadOnce()
// because each track will read a pose (or two) and no 2 tracks can read
// the same pose.  However, we still need to zero the gradients buffer
// because not all poses will be read (only one or two per track), therefore
// many poses will not get a gradient assigned by the kernel.
struct PoseInterpolator : IPoseFetcher {
    [Differentiable]
    [ForceInline]
    static transform::SE3 fetch(
        DiffTensorView tracks_poses,
        int32_t index_left, int32_t index_right,
        int64_t time_left, int64_t time_right, int64_t timestamp
        ) {
        let pose_left = transform.SE3.from_vec(
            tracks_poses.loadVecOnce<3>(uint2(index_left, 0)),
            tracks_poses.loadVecOnce<4>(uint2(index_left, 3)),
        );
        let pose_right = transform.SE3.from_vec(
            tracks_poses.loadVecOnce<3>(uint2(index_right, 0)),
            tracks_poses.loadVecOnce<4>(uint2(index_right, 3)),
        );

        let alpha = (timestamp - time_left) / float(time_right - time_left);

        let pose_final = transform::interpolate(pose_left, pose_right, alpha);

        return pose_final;
    }
};

struct PoseNearestNeighbor : IPoseFetcher {
    [Differentiable]
    [ForceInline]
    static transform::SE3 fetch(
        DiffTensorView tracks_poses,
        int32_t index_left, int32_t index_right,
        int64_t time_left, int64_t time_right, int64_t timestamp
        ) {
        let time_diff_left = abs(timestamp - time_left);
        let time_diff_right = abs(timestamp - time_right);
        int32_t index_nearest;
        if (time_diff_right < time_diff_left) {
            index_nearest = index_right;
        }
        else {
            index_nearest = index_left;
        }

        let pose_final = transform::SE3.from_vec(
            tracks_poses.loadVecOnce<3>(uint2(index_nearest, 0)),
            tracks_poses.loadVecOnce<4>(uint2(index_nearest, 3)),
        );

        return pose_final;
    }
};

public struct TimestampsEstimationFetcher : ITimestampsFetcher {
    TensorView<int64_t> tracks_timestamps_sum_and_count;
    int64_t default_timestamp;

    int64_t fetch(no_diff uint track_idx)
    {
        let count = tracks_timestamps_sum_and_count.load(uint2(track_idx, 1));
        if (count > 0) {
            return default_timestamp + tracks_timestamps_sum_and_count.load(uint2(track_idx, 0)) / count;
        }
        else {
            return default_timestamp;
        }
    }
}


// To avoid compiler errors in auto differentiation with pointer operations,
// we extract the search in a non differentiable function.
Tuple<int32_t, int32_t, bool> search_in_tracks(
    TensorView<int64_t> tracks_timestamps, int64_t timestamp, int32_t start_index, int32_t length
) {
    typedef search::BinarySearch<int64_t, int32_t> SearchStrategy;

    let data = ImmutablePtr<int64_t>(tracks_timestamps.data_ptr_at(start_index));

    return search::find_surrounding_indices<SearchStrategy>(timestamp, data, length);
}

[Differentiable]
void fetch_tracks<TimestampsFetcher: ITimestampsFetcher, PoseFetcher: IPoseFetcher>(
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsFetcher timestamps_fetcher,
    uint track_id,
) {
    let start_index = tracks_packinfo.load(uint2(track_id,0));
    let length = tracks_packinfo.load(uint2(track_id,1));

    let timestamp = timestamps_fetcher.fetch(track_id);

    let indices = search_in_tracks(tracks_timestamps, timestamp, start_index, length);

    // Store whether the timestamps where inside the track or not.
    output_inside.store(track_id, indices._2);

    // Store the final pose.
    transform::SE3 pose;
    if (indices._0 < 0) {
        // Empty track, return identity.
        pose = transform::SE3.identity();
    }
    else if (indices._0 == indices._1) {
        // Matching exactly a pose timestamp, return this pose.
        let index = start_index + indices._0;
        pose = transform::SE3.from_vec(
            tracks_poses.loadVecOnce<3>(uint2(index, 0)),
            tracks_poses.loadVecOnce<4>(uint2(index, 3)),
        );
    }
    else {
        // Between two poses.
        let index_left = start_index + indices._0;
        let index_right = start_index + indices._1;
        let time_left = tracks_timestamps.load(index_left);
        let time_right = tracks_timestamps.load(index_right);
        pose = PoseFetcher.fetch(tracks_poses, index_left, index_right, time_left, time_right, timestamp);
    }

    var pose_vec = pose.to_vec();
    output_poses.storeVecOnce<3>(uint2(track_id, 0), pose_vec._0);
    output_poses.storeVecOnce<4>(uint2(track_id, 3), pose_vec._1);
}

// We create kernel entry points to be able to call the interpolation.
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void interpolate_tracks_poses_global(
    no_diff uint count,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsGlobalFetcher timestamps,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    fetch_tracks<TimestampsGlobalFetcher, PoseInterpolator>(
        output_poses,
        output_inside,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        tid,
    );
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void interpolate_tracks_poses_per_track(
    no_diff uint count,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsPerTrackFetcher timestamps,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    fetch_tracks<TimestampsPerTrackFetcher, PoseInterpolator>(
        output_poses,
        output_inside,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        tid,
    );
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void interpolate_tracks_poses_with_estimation(
    no_diff uint count,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsEstimationFetcher timestamps,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    fetch_tracks<TimestampsEstimationFetcher, PoseInterpolator>(
        output_poses,
        output_inside,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        tid,
    );
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void nearest_neighbor_tracks_poses_global(
    no_diff uint nb_tracks,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsGlobalFetcher timestamps,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= nb_tracks) {
        return;
    }

    fetch_tracks<TimestampsGlobalFetcher, PoseNearestNeighbor>(
        output_poses,
        output_inside,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        tid,
    );
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void nearest_neighbor_tracks_poses_per_track(
    no_diff uint nb_tracks,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsPerTrackFetcher timestamps,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= nb_tracks) {
        return;
    }

    fetch_tracks<TimestampsPerTrackFetcher, PoseNearestNeighbor>(
        output_poses,
        output_inside,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        tid,
    );
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void nearest_neighbor_tracks_poses(
    no_diff uint nb_tracks,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsEstimationFetcher timestamps,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= nb_tracks) {
        return;
    }

    fetch_tracks<TimestampsEstimationFetcher, PoseNearestNeighbor>(
        output_poses,
        output_inside,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        tid,
    );
}


public struct SceneContractor
{
    static const float NORM_COMPARISON_EPS = 1e-7;

    no_diff TensorView<float> aabb_blb;
    no_diff TensorView<float> aabb_trf;
    no_diff float degree;
    no_diff bool is_merf;
    no_diff bool has_contraction;

    
    [ForceInline]
    float3 get_aabb_extent(no_diff uint instance_id)
    {
        float3 blb = this.aabb_blb.loadVec<3>(uint2(instance_id, 0));
        float3 trf = this.aabb_trf.loadVec<3>(uint2(instance_id, 0));
        return trf - blb;
    }
    
    [Differentiable]
    [ForceInline]
    float3 points_to_unit_cube(float3 point, no_diff uint instance_id)
    {
        float3 blb = this.aabb_blb.loadVec<3>(uint2(instance_id, 0));
        float3 extent = this.get_aabb_extent(instance_id);
        return (point - blb) / extent;
    }
    
    [Differentiable]
    [ForceInline]
    static float compute_norm(float3 point, no_diff float degree)
    {
        if (isinf(degree)) {
            // L-infinity norm: max(|x|, |y|, |z|)
            return max(max(abs(point.x), abs(point.y)), abs(point.z));
        }
        else {
            return pow(
                pow(abs(point.x), degree) + 
                pow(abs(point.y), degree) + 
                pow(abs(point.z), degree),
                1.0 / degree
            );
        }
    }
    
    [Differentiable]
    [ForceInline]
    static float3 merf_contract(float3 points, float norm)
    {
        float scale = max(norm, 1.0);
        bool3 is_largest_coord = (abs(points) >= norm - NORM_COMPARISON_EPS);
        
        float3 scaled_points = points / scale;
        float3 contracted = select(is_largest_coord, scaled_points * (2.0 - 1.0 / scale), scaled_points);

        return contracted;
    }
    
    [Differentiable]
    [ForceInline]
    static float3 standard_contract(float3 points, float norm)
    {
        if (norm < 1.0) {
            return points;
        }
        else {
            return (2.0 - 1.0 / norm) * (points / norm);
        }
    }
    
    [Differentiable]
    [ForceInline]
    [NoDiffThis]
    float3 to_contracted_space(float3 world_point, no_diff uint instance_id)
    {
        float3 points = this.points_to_unit_cube(world_point, instance_id);
        
        if (!this.has_contraction) {
            return points;
        }
        
        points = 2.0 * points - 1.0;
        
        float norm = SceneContractor.compute_norm(points, this.degree);
        
        float3 contracted_points;
        if (this.is_merf) {
            contracted_points = SceneContractor.merf_contract(points, norm);
        }
        else {
            contracted_points = SceneContractor.standard_contract(points, norm);
        }
        
        return contracted_points * 0.25 + 0.5;
    }
}

// Input embedding computation for hash grid inputs.
// Computes concatenated embeddings: [xyz (3) + instance_emb (InstanceEmbeddingDim) + time_emb (1)]
// When NbTimestamps > 1, outputs multiple full embeddings with different time values stacked along batch dimension:
//   Row 0 to N-1:     [xyz, instance, time=timestamps[0]]
//   Row N to 2N-1:    [xyz, instance, time=timestamps[1]]
//   Row 2N to 3N-1:   [xyz, instance, time=timestamps[2]]
//   ...
//   Output shape: [NbTimestamps*N, base_dim] (stacked along batch dimension, NOT feature dimension)
//   Column layout: [xyz(0-2), instance(3..3+InstanceEmbeddingDim), time(3+InstanceEmbeddingDim)]
// Common helper function to avoid code duplication
[Differentiable]
void compute_input_embedding_impl<
    let InstanceEmbeddingDim : int,
    let NbTimestamps : int
>(
    no_diff uint count,
    DiffTensorView xyzs,
    no_diff TensorView<int32_t> instance_idx,
    SceneContractor scene_contractor,
    WeightedInstanceInputEmbedding<InstanceEmbeddingDim> instance_embedding,
    no_diff int64_t timestamps[NbTimestamps],
    IndividualRemapTimeInputEmbedding time_embedding,
    DiffTensorView output,
    no_diff uint tid
)
{
    no_diff var instance_id = instance_idx[tid];
    float3 xyz = xyzs.loadVecOnce<3>(uint2(tid, 0));
    var contracted_xyz = scene_contractor.to_contracted_space(xyz, instance_id);
    contracted_xyz = clamp(contracted_xyz, float3(0.0), float3(1.0));
    
    // Column positions: xyz at 0-2, instance at 3, time at 3+InstanceEmbeddingDim
    const int instance_col = 3;  // After xyz
    const int time_col = instance_col + InstanceEmbeddingDim;
    
    // Loop bound is compile-time constant, fully unrolled
    [ForceUnroll]
    for (int variant = 0; variant < NbTimestamps; ++variant) {
        const int64_t timestamp = timestamps[variant];
        // Each variant writes to a different row: tid, tid+count, tid+2*count, ...
        const uint output_row = tid + variant * count;
        
        // xyz at columns 0-2
        [ForceUnroll]
        for (int i = 0; i < 3; ++i) {
            output.storeOnce(uint2(output_row, i), contracted_xyz[i]);
        }
        
        // instance embedding starting at column 3
        instance_embedding.embed_to_row(tid, output_row, instance_col);
        
        // time embedding after instance embedding
        var time_emb = time_embedding.embed(tid, timestamp);
        output.storeOnce(uint2(output_row, time_col), time_emb);
    }
}

// Kernel entry point for input embeddings
// Output shape: [count, base_dim] where base_dim = 3 + instance_dim + time_dim
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void prepare_input_embeddings(
    no_diff uint count,
    DiffTensorView xyzs,
    no_diff TensorView<int32_t> instance_idx,
    SceneContractor scene_contractor,
    WeightedInstanceInputEmbedding<1> instance_embedding,
    TimestampsGlobalFetcher timestamps_fetcher,
    IndividualRemapTimeInputEmbedding time_embedding,
    DiffTensorView output,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }
    
    let timestamp = timestamps_fetcher.fetch(tid);
    int64_t timestamps[1] = { timestamp };
    compute_input_embedding_impl</*InstanceEmbeddingDim=*/1, /*NbTimestamps=*/1>(
        count, xyzs, instance_idx, scene_contractor, instance_embedding, timestamps, time_embedding, output, tid);
}

// Kernel entry point for input embeddings - With timestamps delta version
// Output shape: [count * 3, base_dim] where:
//   Rows 0 to count-1:         embeddings at time t
//   Rows count to 2*count-1:   embeddings at time t-delta
//   Rows 2*count to 3*count-1: embeddings at time t+delta
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void prepare_input_embeddings_with_timestamps_delta(
    no_diff uint count,
    DiffTensorView xyzs,
    no_diff TensorView<int32_t> instance_idx,
    SceneContractor scene_contractor,
    WeightedInstanceInputEmbedding<1> instance_embedding,
    TimestampsGlobalFetcher timestamps_fetcher,
    IndividualRemapTimeInputEmbedding time_embedding,
    no_diff TensorView<int64_t> timestamps_delta,
    DiffTensorView output,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }
    
    let timestamp = timestamps_fetcher.fetch(tid);
    const int64_t delta = timestamps_delta[tid];
    int64_t timestamps[3] = { timestamp, timestamp - delta, timestamp + delta };
    compute_input_embedding_impl</*InstanceEmbeddingDim=*/1, /*NbTimestamps=*/3>(
        count, xyzs, instance_idx, scene_contractor, instance_embedding, timestamps, time_embedding, output, tid);
}


interface IModifier<let ModifierDim : int> : IDifferentiable {
    [Differentiable]
    vector<float, ModifierDim> apply(vector<float, ModifierDim> value);
};

struct Modifier_Copy<let ModifierDim : int> : IModifier<ModifierDim> {
    [Differentiable]
    [ForceInline]
    vector<float, ModifierDim> apply(vector<float, ModifierDim> value)
    {
        return value;
    }
};

struct Modifier_Add<let ModifierDim : int> : IModifier<ModifierDim> {
    vector<float, ModifierDim> add_value;

    [Differentiable]
    [ForceInline]
    vector<float, ModifierDim> apply(vector<float, ModifierDim> value)
    {
        return value + add_value;
    }
};

struct Modifier_Rotate : IModifier<4> {
    vector<float, 4> rotation;

    [Differentiable]
    [ForceInline]
    vector<float, 4> apply(vector<float, 4> value)
    {
        return quaternion::multiply(quaternion::normalizeSafe(rotation), value);
    }
};

// Positions, rotations and densities.
//
// We do positions and rotations together to do fewer reads to the poses to apply.
// We do densities too just to avoid re-reading the track id.
//
// It could be fewer GPU operations to compute the rotation matrices
// for each track during interpolation and use those.
// However, it would use more memory (9 floats instead of 4) and since
// we are typically memory bound, we trade the loads for a few more operations
// per gaussian.
// That being said, since there aren't that many tracks, maybe the loads would
// not be an issue so we could consider this other approach.
[Differentiable]
void collect_positions_rotations_densities_for_tracks<
    PositionsModifier : IModifier<3>,
    RotationsModifier : IModifier<4>,
>(
    no_diff uint tid, no_diff uint out_offset,
    DiffTensorView input_positions,
    DiffTensorView input_rotations,
    DiffTensorView input_densities,
    DiffTensorView output_positions,
    DiffTensorView output_rotations,
    DiffTensorView output_densities,
    DiffTensorView tracks_poses,
    no_diff TensorView<int32_t> tracks_ids,
    no_diff TensorView<bool> keep_mask,
    PositionsModifier positions_modifier,
    RotationsModifier rotations_modifier,
) {
    let track_idx = tracks_ids.load(tid);
    let keep = keep_mask.load(track_idx);

    // Read the pose.
    const var pose = transform.SE3.from_vec(
        utils::pass(tracks_poses.loadVecEx<3>(uint2(track_idx, 0)), keep),
        utils::pass(tracks_poses.loadVecEx<4>(uint2(track_idx, 3)), keep),
    );

    // Read the position to transform.
    float3 position = input_positions.loadVecOnce<3>(uint2(tid, 0));
    position = positions_modifier.apply(position);

    // Read the rotation to transform.
    // It is stored in wxyz, but the current library expects xyzw.
    // wxyz -> xyzw => (0,1,2,3) -> (1,2,3,0) => We read yzwx
    float4 rotation = input_rotations.loadVecOnce<4>(uint2(tid, 0)).yzwx;
    rotation = quaternion::normalizeSafe(rotation);
    rotation = rotations_modifier.apply(rotation);

    position = (pose * transform::Point(position)).p;
    rotation = (pose.so3 * transform::SO3.from_vec(rotation)).to_vec();

    const uint out_index = out_offset + tid;

    output_positions.storeVecOnce<3>(uint2(out_index, 0), position);

    // We need to store the rotation in wxyz.
    // xyzw -> wxyz => (0,1,2,3) -> (3,0,1,2) => We write wxyz
    output_rotations.storeVecOnce<4>(uint2(out_index, 0), rotation.wxyz);

    // Densities.
    float density;
    if (keep) {
        density = input_densities.loadOnce(uint2(tid, 0));
        density = SigmoidFunction.apply(density);
    }
    else {
        // No gradient will be written back for this index.
        density = 0.0f;
    }
    output_densities.storeOnce(uint2(out_index, 0), density);
}

public struct PositionsRotationsDensitiesCollector_Tracks : ITask
{
    DiffTensorView input_positions;
    DiffTensorView input_rotations;
    DiffTensorView input_densities;
    DiffTensorView output_positions;
    DiffTensorView output_rotations;
    DiffTensorView output_densities;
    DiffTensorView tracks_poses;
    no_diff TensorView<int32_t> tracks_ids;
    no_diff TensorView<bool> keep_mask;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset)
    {
        collect_positions_rotations_densities_for_tracks(
            tid, out_offset,
            input_positions, input_rotations, input_densities,
            output_positions, output_rotations, output_densities,
            tracks_poses, tracks_ids, keep_mask,
            Modifier_Copy<3>(), Modifier_Copy<4>(),
        );
    }
};

// These types are not really used by the rigid layer, but they are still created
// temporarily before they are replaced by the PositionsRotationsDensitiesCollector_Tracks.
// So we make sure they are exported by slangtorch.
[CUDAKernel]
[AutoPyBindCUDA]
void trigger_type_export(
    Collector_Copy<3> positions_collector,
    RotationsCollector_Normalize rotations_collector,
    DensitiesCollector_Sigmoid densities_collector,
    PositionsRotationsDensitiesCollector_Tracks tracks_collector,
) {
}

// Kernel to zero the gradients for the disabled tracks.
// Note that this part could simply be written in CUDA.
void zero_at_index(
    no_diff uint index,
    no_diff TensorView<float> buffer,
) {
    // OPTME: We could know the size of the buffer at compile time.
    let size_to_zero = buffer.size(1);
    for (int i = 0; i < size_to_zero; ++i) {
        buffer.store(uint2(index, i), 0.0f);
    }
}

[CUDAKernel]
[AutoPyBindCUDA]
void zero_gradients_for_disabled_tracks(
    TensorView<int32_t> tracks_ids,
    TensorView<bool> keep_mask,
    TensorView<float> positions_grad,
    TensorView<float> rotations_grad,
    TensorView<float> scales_grad,
    TensorView<float> densities_grad,
    TensorView<float> extra_signal_grad,
    TensorView<float> camera_extra_signal_grad,
    TensorView<float> lidar_extra_signal_grad,
    TensorView<float> features_albedo_grad,
    TensorView<float> features_specular_grad,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= tracks_ids.size(0)) {
        return;
    }

    let track_id = tracks_ids.load(tid);
    let keep = keep_mask.load(track_id);
    if (keep) {
        return;
    }

    zero_at_index(tid, positions_grad);
    zero_at_index(tid, rotations_grad);
    zero_at_index(tid, scales_grad);
    zero_at_index(tid, densities_grad);
    zero_at_index(tid, extra_signal_grad);
    zero_at_index(tid, camera_extra_signal_grad);
    zero_at_index(tid, lidar_extra_signal_grad);
    zero_at_index(tid, features_albedo_grad);
    zero_at_index(tid, features_specular_grad);
}


//
// Implementations for dynamic rigid layer.
//

public struct PositionsRotationsDensitiesCollector_Deformable : ITask
{
    DiffTensorView input_positions;
    DiffTensorView input_rotations;
    DiffTensorView input_densities;
    DiffTensorView output_positions;
    DiffTensorView output_rotations;
    DiffTensorView output_densities;
    DiffTensorView tracks_poses;
    no_diff TensorView<int32_t> tracks_ids;
    no_diff TensorView<bool> keep_mask;
    DiffTensorView deform_positions;
    DiffTensorView deform_rotations;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint out_offset)
    {
        const float3 deform_position = deform_positions.loadVecOnce<3>(uint2(tid, 0));
        float4 deform_rotation = deform_rotations.loadVecOnce<4>(uint2(tid, 0));
        // The deformation rotations are offsets from the identity rotation,
        // so we add 1 to the w component to get the full rotation offset.
        deform_rotation.w += 1.0f;

        collect_positions_rotations_densities_for_tracks(
            tid, out_offset,
            input_positions, input_rotations, input_densities,
            output_positions, output_rotations, output_densities,
            tracks_poses, tracks_ids, keep_mask,
            Modifier_Add<3>(deform_position), Modifier_Rotate(deform_rotation),
        );
    }
};
