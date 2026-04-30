// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#pragma once

#include "particleFeatures.slang"

// ==================================================================================================================
// CUDA ENTRY POINTS FOR PARTICLE FEATURES
// ==================================================================================================================
//
// This file defines macros that generate CUDA device export functions for particle feature operations.
// These macros provide standardized interfaces for particle feature integration, backward propagation,
// and buffer operations that can be called from CUDA kernels.
//

// ------------------------------------------------------------------------------------------------------------------
// COMMON PARTICLE FEATURES ENTRY POINTS
// ------------------------------------------------------------------------------------------------------------------
/**
 * @brief Generates common CUDA entry points for particle feature operations
 * @param ExportPrefix The prefix to add to the exported function names
 * @param ParticleType The particle type to generate entry points for (e.g., Vector3, ShVector3)
 *
 */
#define ParticleFeaturesCommonEntryPoints(ExportPrefix, ParticleType)                                                     \
                                                                                                                          \
    /* Forward integration: integrates particle features with given weight */                                             \
    [CudaDeviceExport] inline void ExportPrefix##IntegrateFwd(                                                            \
        in float weight,                                                                                                  \
        in particleFeatures.ParticleType.FeatureArray features,                                                           \
        inout particleFeatures.ParticleType.FeatureArray integratedFeatures) {                                            \
        particleFeatures.ParticleType.integrate<false>(weight, features, integratedFeatures);                             \
    }                                                                                                                     \
                                                                                                                          \
    /* Backward integration: simple backward pass without alpha blending */                                               \
    [CudaDeviceExport] inline void ExportPrefix##IntegrateBwd(                                                            \
        in float weight,                                                                                                  \
        in particleFeatures.ParticleType.FeatureArray features,                                                           \
        inout particleFeatures.ParticleType.FeatureArray integratedFeatures) {                                            \
        particleFeatures.ParticleType.integrate<false>(weight, features, integratedFeatures);                             \
    }                                                                                                                     \
                                                                                                                          \
    /* Backward integration with alpha blending and gradient propagation */                                               \
    [CudaDeviceExport] inline void ExportPrefix##IntegrateBwd(                                                            \
        in float alpha,                                                                                                   \
        inout float alphaGrad,                                                                                            \
        in particleFeatures.ParticleType.FeatureArray features,                                                           \
        inout particleFeatures.ParticleType.FeatureArray integratedFeatures,                                              \
        inout particleFeatures.ParticleType.FeatureArray integratedFeaturesGrad,                                          \
        inout particleFeatures.ParticleType.FeatureArray featuresGrad) {                                                  \
        if (alpha > 0.0f) {                                                                                               \
            /* Set up differential pairs for automatic differentiation */                                                 \
            DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, 0.f);                                      \
            DifferentialPair<particleFeatures.ParticleType.FeatureArray> featuresDiff =                                   \
                DifferentialPair<particleFeatures.ParticleType.FeatureArray>(features, featuresGrad);                     \
                                                                                                                          \
            /* Apply alpha blending: result = (integrated - features * alpha) / (1 - alpha) */                            \
            const float weight = 1.0f / (1.0f - alpha);                                                                   \
            [ForceUnroll] for (int i = 0; i < particleFeatures.ParticleType.Dim; ++i) {                                   \
                integratedFeatures[i] = (integratedFeatures[i] - features[i] * alpha) * weight;                           \
            }                                                                                                             \
                                                                                                                          \
            DifferentialPair<particleFeatures.ParticleType.FeatureArray> integratedFeaturesDiff =                         \
                DifferentialPair<particleFeatures.ParticleType.FeatureArray>(integratedFeatures, integratedFeaturesGrad); \
                                                                                                                          \
            /* Backward differentiation through the integration operation */                                              \
            bwd_diff(particleFeatures.ParticleType.integrate<true>)(                                                      \
                alphaDiff,                                                                                                \
                featuresDiff,                                                                                             \
                integratedFeaturesDiff);                                                                                  \
                                                                                                                          \
            /* Extract gradients from differential pairs */                                                               \
            integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();                                            \
            featuresGrad           = featuresDiff.getDifferential();                                                      \
            alphaGrad += alphaDiff.getDifferential();                                                                     \
        }                                                                                                                 \
    }

// ------------------------------------------------------------------------------------------------------------------
// VECTOR PARTICLE FEATURES ENTRY POINTS
// ------------------------------------------------------------------------------------------------------------------
/**
 * @brief Generates CUDA entry points for vector-based particle features
 * @param ExportPrefix The prefix to add to the exported function names
 * @param ParticleType The vector particle type to generate entry points for
 *
 * Vector particle features store feature data directly without directional encoding.
 */
#define ParticleFeaturesVectorEntryPoints(ExportPrefix, ParticleType)                                                     \
                                                                                                                          \
    /* Fetch vector features from buffer */                                                                               \
    [CudaDeviceExport] inline void ExportPrefix##FromBuffer(                                                              \
        in uint32_t particleIdx,                                                                                          \
        particleFeatures.ParticleType.ParametersBuffer parametersBuffer,                                                  \
        in float3 incidentDirection,                                                                                      \
        out particleFeatures.ParticleType.FeatureVector features) {                                                       \
        features = particleFeatures.ParticleType.fetchFromBuffer(particleIdx, parametersBuffer);                          \
    }                                                                                                                     \
                                                                                                                          \
    /* Backward integration to buffer: propagates gradients back to parameter buffer */                                   \
    [CudaDeviceExport] void ExportPrefix##IntegrateBwdToBuffer(                                                           \
        in float3 incidentDirection,                                                                                      \
        inout float3 incidentDirectionGrad,                                                                               \
        in float alpha,                                                                                                   \
        inout float alphaGrad,                                                                                            \
        in uint32_t particleIdx,                                                                                          \
        particleFeatures.ParticleType.ParametersBuffer parametersBuffer,                                                  \
        in particleFeatures.ParticleType.FeatureArray features,                                                           \
        inout particleFeatures.ParticleType.FeatureArray integratedFeatures,                                              \
        inout particleFeatures.ParticleType.FeatureArray integratedFeaturesGrad) {                                        \
        if (alpha > 0.0f) {                                                                                               \
            /* Set up differential pair for alpha */                                                                      \
            DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, 0.f);                                      \
                                                                                                                          \
            /* Apply alpha blending */                                                                                    \
            const float weight = 1.0f / (1.0f - alpha);                                                                   \
            [ForceUnroll] for (int i = 0; i < particleFeatures.ParticleType.Dim; ++i) {                                   \
                integratedFeatures[i] = (integratedFeatures[i] - features[i] * alpha) * weight;                           \
            }                                                                                                             \
                                                                                                                          \
            DifferentialPair<particleFeatures.ParticleType.FeatureArray> integratedFeaturesDiff =                         \
                DifferentialPair<particleFeatures.ParticleType.FeatureArray>(integratedFeatures, integratedFeaturesGrad); \
                                                                                                                          \
            /* Backward differentiation through buffer integration */                                                     \
            bwd_diff(particleFeatures.ParticleType.integrateFromBuffer<true>)(                                            \
                alphaDiff,                                                                                                \
                particleIdx,                                                                                              \
                parametersBuffer,                                                                                         \
                integratedFeaturesDiff);                                                                                  \
                                                                                                                          \
            /* Extract gradients */                                                                                       \
            integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();                                            \
            alphaGrad += alphaDiff.getDifferential();                                                                     \
        }                                                                                                                 \
    }                                                                                                                     \
                                                                                                                          \
    /* Sample backward integration to buffer: handles sampling-specific backward pass */                                  \
    [CudaDeviceExport] void ExportPrefix##IntegrateSampleBwdToBuffer(                                                     \
        in float3 incidentDirection,                                                                                      \
        inout float3 incidentDirectionGrad,                                                                               \
        in float alpha,                                                                                                   \
        inout float alphaGrad,                                                                                            \
        in uint32_t particleIdx,                                                                                          \
        particleFeatures.ParticleType.ParametersBuffer parametersBuffer,                                                  \
        in particleFeatures.ParticleType.FeatureArray features,                                                           \
        inout particleFeatures.ParticleType.FeatureArray integratedFeaturesTotal,                                         \
        inout particleFeatures.ParticleType.FeatureArray integratedFeaturesGrad) {                                        \
        if (alpha > 0.0f) {                                                                                               \
            /* Set up differential pair for alpha */                                                                      \
            DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, 0.f);                                      \
                                                                                                                          \
            /* Calculate integrated features from total */                                                                \
            particleFeatures.ParticleType.FeatureArray integratedFeatures;                                                \
            [ForceUnroll] for (int i = 0; i < particleFeatures.ParticleType.Dim; ++i) {                                   \
                integratedFeatures[i] = integratedFeaturesTotal[i] - features[i] * alpha;                                 \
            }                                                                                                             \
                                                                                                                          \
            DifferentialPair<particleFeatures.ParticleType.FeatureArray> integratedFeaturesDiff =                         \
                DifferentialPair<particleFeatures.ParticleType.FeatureArray>(integratedFeatures, integratedFeaturesGrad); \
                                                                                                                          \
            /* Backward differentiation through buffer integration (non-blending mode) */                                 \
            bwd_diff(particleFeatures.ParticleType.integrateFromBuffer<false>)(                                           \
                alphaDiff,                                                                                                \
                particleIdx,                                                                                              \
                parametersBuffer,                                                                                         \
                integratedFeaturesDiff);                                                                                  \
                                                                                                                          \
            /* Extract gradients */                                                                                       \
            integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();                                            \
            alphaGrad += alphaDiff.getDifferential();                                                                     \
        }                                                                                                                 \
    }                                                                                                                     \
                                                                                                                          \
    /* Backward propagation to buffer: propagates feature gradients back to parameter buffer */                           \
    [CudaDeviceExport] void ExportPrefix##BwdToBuffer(                                                                    \
        in uint32_t particleIdx,                                                                                          \
        particleFeatures.ParticleType.ParametersBuffer parametersBuffer,                                                  \
        in particleFeatures.ParticleType.FeatureArray featuresGrad,                                                       \
        in float3 incidentDirection,                                                                                      \
        inout float3 incidentDirectionGrad) {                                                                             \
        /* Backward differentiation through buffer fetch operation */                                                     \
        bwd_diff(particleFeatures.ParticleType.fetchFromBuffer)(                                                          \
            particleIdx,                                                                                                  \
            parametersBuffer,                                                                                             \
            featuresGrad);                                                                                                \
    }                                                                                                                     \
    /* Include all common entry points */                                                                                 \
    ParticleFeaturesCommonEntryPoints(ExportPrefix, ParticleType)

// ------------------------------------------------------------------------------------------------------------------
// SPHERICAL HARMONICS PARTICLE FEATURES ENTRY POINTS
// ------------------------------------------------------------------------------------------------------------------
/**
 * @brief Generates CUDA entry points for spherical harmonics-based particle features
 * @param ExportPrefix The prefix to add to the exported function names
 * @param ParticleType The spherical harmonics particle type to generate entry points for
 *
 * Spherical harmonics particle features encode directional information using spherical harmonics.
 * Features are decoded based on the incident direction.
 */
#define ParticleFeaturesSphericalHarmonicsEntryPoints(ExportPrefix, ParticleType)                                         \
                                                                                                                          \
    /* Fetch and decode spherical harmonics features from buffer */                                                       \
    [CudaDeviceExport] inline void ExportPrefix##FromBuffer(                                                              \
        in uint32_t particleIdx,                                                                                          \
        particleFeatures.ParticleType.ParametersBuffer parametersBuffer,                                                  \
        in float3 incidentDirection,                                                                                      \
        out particleFeatures.ParticleType.FeatureVector features) {                                                       \
        features = particleFeatures.ParticleType.fetchFromBuffer(particleIdx, parametersBuffer, incidentDirection);       \
    }                                                                                                                     \
                                                                                                                          \
    /* Backward integration to buffer: propagates gradients back to parameter buffer */                                   \
    [CudaDeviceExport] void ExportPrefix##IntegrateBwdToBuffer(                                                           \
        in float3 incidentDirection,                                                                                      \
        inout float3 incidentDirectionGrad,                                                                               \
        in float alpha,                                                                                                   \
        inout float alphaGrad,                                                                                            \
        in uint32_t particleIdx,                                                                                          \
        particleFeatures.ParticleType.ParametersBuffer parametersBuffer,                                                  \
        in particleFeatures.ParticleType.FeatureArray features,                                                           \
        inout particleFeatures.ParticleType.FeatureArray integratedFeatures,                                              \
        inout particleFeatures.ParticleType.FeatureArray integratedFeaturesGrad) {                                        \
        if (alpha > 0.0f) {                                                                                               \
            DifferentialPair<float3> incidentDirectionDiff = DifferentialPair<float3>(incidentDirection, float3(0.f));    \
            DifferentialPair<float> alphaDiff              = DifferentialPair<float>(alpha, 0.f);                         \
                                                                                                                          \
            const float weight = 1.0f / (1.0f - alpha);                                                                   \
            [ForceUnroll] for (int i = 0; i < particleFeatures.ParticleType.Dim; ++i) {                                   \
                integratedFeatures[i] = (integratedFeatures[i] - features[i] * alpha) * weight;                           \
            }                                                                                                             \
            DifferentialPair<particleFeatures.ParticleType.FeatureArray> integratedFeaturesDiff =                         \
                DifferentialPair<particleFeatures.ParticleType.FeatureArray>(integratedFeatures, integratedFeaturesGrad); \
                                                                                                                          \
            bwd_diff(particleFeatures.ParticleType.integrateFromBuffer<true>)(                                            \
                incidentDirectionDiff,                                                                                    \
                alphaDiff,                                                                                                \
                particleIdx,                                                                                              \
                parametersBuffer,                                                                                         \
                integratedFeaturesDiff);                                                                                  \
                                                                                                                          \
            integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();                                            \
            alphaGrad += alphaDiff.getDifferential();                                                                     \
            incidentDirectionGrad += incidentDirectionDiff.getDifferential();                                             \
        }                                                                                                                 \
    }                                                                                                                     \
                                                                                                                          \
    /* Sample backward integration to buffer: handles sampling-specific backward pass */                                  \
    [CudaDeviceExport] void ExportPrefix##IntegrateSampleBwdToBuffer(                                                     \
        in float3 incidentDirection,                                                                                      \
        inout float3 incidentDirectionGrad,                                                                               \
        in float alpha,                                                                                                   \
        inout float alphaGrad,                                                                                            \
        in uint32_t particleIdx,                                                                                          \
        particleFeatures.ParticleType.ParametersBuffer parametersBuffer,                                                  \
        in particleFeatures.ParticleType.FeatureArray features,                                                           \
        inout particleFeatures.ParticleType.FeatureArray integratedFeaturesTotal,                                         \
        inout particleFeatures.ParticleType.FeatureArray integratedFeaturesGrad) {                                        \
        if (alpha > 0.0f) {                                                                                               \
            DifferentialPair<float3> incidentDirectionDiff = DifferentialPair<float3>(incidentDirection, float3(0.f));    \
            DifferentialPair<float> alphaDiff              = DifferentialPair<float>(alpha, 0.f);                         \
                                                                                                                          \
            particleFeatures.ParticleType.FeatureArray integratedFeatures;                                                \
            [ForceUnroll] for (int i = 0; i < particleFeatures.ParticleType.Dim; ++i) {                                   \
                integratedFeatures[i] = integratedFeaturesTotal[i] - features[i] * alpha;                                 \
            }                                                                                                             \
            DifferentialPair<particleFeatures.ParticleType.FeatureArray> integratedFeaturesDiff =                         \
                DifferentialPair<particleFeatures.ParticleType.FeatureArray>(integratedFeatures, integratedFeaturesGrad); \
                                                                                                                          \
            bwd_diff(particleFeatures.ParticleType.integrateFromBuffer<false>)(                                           \
                incidentDirectionDiff,                                                                                    \
                alphaDiff,                                                                                                \
                particleIdx,                                                                                              \
                parametersBuffer,                                                                                         \
                integratedFeaturesDiff);                                                                                  \
                                                                                                                          \
            integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();                                            \
            alphaGrad += alphaDiff.getDifferential();                                                                     \
            incidentDirectionGrad += incidentDirectionDiff.getDifferential();                                             \
        }                                                                                                                 \
    }                                                                                                                     \
                                                                                                                          \
    /* Backward propagation to buffer: propagates feature gradients back to parameter buffer */                           \
    [CudaDeviceExport] void ExportPrefix##BwdToBuffer(                                                                    \
        in uint32_t particleIdx,                                                                                          \
        particleFeatures.ParticleType.ParametersBuffer parametersBuffer,                                                  \
        in particleFeatures.ParticleType.FeatureArray featuresGrad,                                                       \
        in float3 incidentDirection,                                                                                      \
        inout float3 incidentDirectionGrad) {                                                                             \
        /* Backward differentiation through buffer fetch operation */                                                     \
        DifferentialPair<float3> incidentDirectionDiff = DifferentialPair<float3>(incidentDirection, float3(0.f));        \
                                                                                                                          \
        bwd_diff(particleFeatures.ParticleType.fetchFromBuffer)(                                                          \
            particleIdx,                                                                                                  \
            parametersBuffer,                                                                                             \
            incidentDirectionDiff,                                                                                        \
            featuresGrad);                                                                                                \
                                                                                                                          \
        incidentDirectionGrad += incidentDirectionDiff.getDifferential();                                                 \
    }                                                                                                                     \
                                                                                                                          \
    /* Include all common entry points */                                                                                 \
    ParticleFeaturesCommonEntryPoints(ExportPrefix, ParticleType)

// ------------------------------------------------------------------------------------------------------------------
// STANDALONE TESTING CONFIGURATION
// ------------------------------------------------------------------------------------------------------------------
#ifdef STANDALONE
/**
 * @brief Standalone configuration for testing particle feature entry points
 *
 * When STANDALONE is defined, this section:
 * 1. Defines common particle type aliases for testing
 * 2. Instantiates entry points for basic particle types
 *
 * This is primarily used for isolated testing and validation.
 */
namespace particleFeatures {
typealias Vector3   = Vector<3>;                                         // 3D vector particle
typealias Vector4   = Vector<4>;                                         // 4D vector particle
typealias ShVector3 = ShVector<3, 3, (3 + 1) * (3 + 1)>; // 3D spherical harmonics particle (degree 3)
} // namespace particleFeatures

// Instantiate entry points for vector particles
ParticleFeaturesVectorEntryPoints(particleExtendedFeatures, Vector3);
ParticleFeaturesVectorEntryPoints(particleCameraExtendedFeatures, Vector4);

// Instantiate entry points for spherical harmonics particles
ParticleFeaturesSphericalHarmonicsEntryPoints(particleLidarExtendedFeatures, ShVector3);

#endif // STANDALONE