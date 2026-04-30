// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/kernels/cuda/common/rayPayload.cuh>

// NB :constant parameters : pipelineParams

extern "C" __global__ void __raygen__rg() {

    const nrend::RenderParameters& renderParameters = *reinterpret_cast<const nrend::RenderParameters*>(
        &pipelineParams.renderParametersArray);

#if !TGRTTracer_Backward
    // if constexpr (!TGRTTracer::Backward) {
    auto ray = initializeRay<TGRTTracer::TRayPayload, true>(
        renderParameters,
        pipelineParams.wordlRayOriginPtr,
        pipelineParams.worldRayDirectionPtr,
        pipelineParams.worldRayTimestampCudaPtr,
        pipelineParams.instanceIdPtr,
        pipelineParams.worldHitDistancePtr);

    TGRTTracer::raygen(pipelineParams.traversableHandle,
                       renderParameters,
                       ray,
                       reinterpret_cast<float*>(pipelineParams.sceneDataPtr),
                       {pipelineParams.parameterMemoryHandles});

    TGRTModel::eval(renderParameters,
                    ray,
                    {pipelineParams.parameterMemoryHandles},
                    pipelineParams.sensorsIdsPtr);

    // NB : finalize ray is not differentiable (has to be no-op when used in a differentiable renderer)
    finalizeRay<SRGBModel, SRGBOutput, Differentiable>(
        ray,
        renderParameters,
        pipelineParams.wordlRayOriginPtr,
        pipelineParams.instanceIdPtr,
        pipelineParams.worldHitDistancePtr,
        pipelineParams.worldHitNormalPtr,
        reinterpret_cast<tcnn::vec<TGRTTracer::TRayPayload::BaseFeatDim + 1>*>(pipelineParams.radianceDensityPtr),
        pipelineParams.extendedFeaturesPtr,
        pipelineParams.worldHitDistanceBoundsPtr);

#else
    //} else {

    auto ray = initializeBackwardRay<TGRTTracer::TRayPayloadBackward, true>(
        renderParameters,
        pipelineParams.wordlRayOriginPtr,
        pipelineParams.worldRayDirectionPtr,
        pipelineParams.worldRayTimestampCudaPtr,
        pipelineParams.instanceIdPtr,
        pipelineParams.worldHitDistancePtr,
        pipelineParams.worldHitDistanceGradientPtr,
        pipelineParams.worldHitNormalPtr,
        pipelineParams.worldHitNormalGradientPtr,
        reinterpret_cast<const tcnn::vec<TGRTTracer::TRayPayload::BaseFeatDim + 1>*>(pipelineParams.radianceDensityPtr),
        reinterpret_cast<const tcnn::vec<TGRTTracer::TRayPayload::BaseFeatDim + 1>*>(pipelineParams.radianceDensityGradientPtr),
        pipelineParams.extendedFeaturesPtr,
        pipelineParams.extendedFeaturesGradientPtr,
        pipelineParams.worldHitDistanceBoundsPtr);

    TGRTModel::evalBackward(renderParameters,
                            ray,
                            {pipelineParams.parameterMemoryHandles},
                            {pipelineParams.parameterGradientMemoryHandles},
                            pipelineParams.sensorsIdsPtr);

    TGRTTracer::raygen(pipelineParams.traversableHandle,
                       renderParameters,
                       ray,
                       nullptr, //< sceneDataPtr - not used in backward pass
                       {pipelineParams.parameterMemoryHandles},
                       {pipelineParams.parameterGradientMemoryHandles});

    finalizeBackwardRay<TGRTTracer::TRayPayloadBackward>(ray,
                                                         renderParameters,
                                                         pipelineParams.wordlRayOriginGradientPtr,
                                                         pipelineParams.worldRayDirectionGradientPtr);
    //}
#endif
}

extern "C" __global__ void __intersection__is() {
    TGRTTracer::intersect({pipelineParams.parameterMemoryHandles});
}

extern "C" __global__ void __anyhit__ah() {
    TGRTTracer::anyhit({pipelineParams.parameterMemoryHandles});
}
