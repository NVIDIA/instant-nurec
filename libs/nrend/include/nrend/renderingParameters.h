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

#include <nrend/modelParameters.h>
#include <nrend/tracksParameters.h>

#include <cstddef>
#include <cstdint>

namespace nrend {

struct MsgPackData {
    const char* dataPtr = nullptr;
    size_t dataSz       = 0;
};

struct RenderingParameters {

    // renderer hints are used to select the renderer to instantiate
    // when no rendering settings are provided
    enum RendererHints : uint32_t {
        RendererDefault,
        RendererFastest,
        RendererFast,
        RendererQuality,
        RendererHighestQuality,
        RendererFastQuality,
        RendererQualityFast
    };
    RendererHints rendererHint = RendererDefault;

    enum OptFlags : uint32_t {
        OptNone                           = 0,
        OptDisableFeatures                = 1 << 1,  ///< disable all features
        OptDisableNormals                 = 1 << 2,  ///< disable normals computation
        OptDisableExtendedFeatures        = 1 << 3,  ///< disable extended features
        OptDisableSensorExtendedFeatures  = 1 << 4,  ///< disable sensor specific extended features
        OptDifferentiable                 = 1 << 5,  ///< disable differentiable rendering
        OptDisableRayGradients            = 1 << 6,  ///< disable ray gradients computation
        OptLinearRGB                      = 1 << 7,  ///< enable linear RGB rendering
        OptNREReferential                 = 1 << 8,  ///< for legacy renderers (set in NRE, unset in external renderer)
        OptDisableBackground              = 1 << 9,  ///< disable background rendering
        OptDisablePostProcessings         = 1 << 10, ///< disable post-processing rendering
        OptEnableParticleCumulatedWeights = 1 << 11, ///< enable cumulated weights computation
        OptEnableParticleVisibility       = 1 << 12, ///< enable particle visibility computation
        OptDefault                        = OptNone
    };

    OptFlags opts = OptDefault;
    TrackInstancesUIdsSpan trackInstancesStrUIds;
};

struct RenderingFeaturesLayout {
    /// Base features are output along with the density (eg : radiance).
    int baseFeaturesDim = 3;
    /// Extended features are optionnal features (eg : normals, raydrop, intensity, uncertainty, semantic, semantic_logits, dinov2_feats, etc)
    /// To enable extended features, set OptExtendedFeatures flag at renderer creation (extendedFeaturesDim will be 0 in this case)
    int extendedFeaturesDim       = 0;
    int sensorExtendedFeaturesDim = 0;
    /// TODO : add extended features layout
    bool computeNormals      = true;  //< if false normals are not computed
    bool computeRayGradients = false; //< if false ray gradients are not computed
};

// Layout of the scene data output by the renderer
struct RenderingSceneDataLayout {
    struct Span {
        int32_t offset = -1;
        int32_t count  = 0;
    };
    Span cumulatedWeights; //< span of the cumulated weights in the scene data
    Span visibility;       //< span of the visibility mask in the scene data

    inline int32_t count() const {
        return cumulatedWeights.count + visibility.count;
    }
};

} // namespace nrend