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

#include <nrend/renderer/renderParameters.h>

#include <nrend/kernels/cuda/primitives/particleAsPrimitivesUtils.cuh>

#include <optix.h>

namespace nrend {

struct TParticlePrimitiveDummyParams {
    static constexpr bool DensityScaleClamping = true;
};

template <typename TPrimitive, typename TParams>
struct TriangleMeshParticlePrimitive {

#ifdef __CUDACC__
    template <typename TParticle>
    static inline __device__ void eval(const uint32_t numParticles,
                                       void* __restrict__ primitiveData,
                                       void* __restrict__ primitiveExtendedData,
                                       OptixTraversableHandle,
                                       nrend::MemoryHandles parameters) {
        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleIdx < numParticles) {
            TParticle particles;
            particles.initializeDensity(parameters);

            const typename TParticle::DensityParameters densityParameters = particles.fetchDensityParameters(particleIdx);

            const float kernelScale = particles.canonicalScale(densityParameters, TParams::DensityScaleClamping);

            const uint32_t firstVertexIdx = TPrimitive::NumVertices * particleIdx;
            tcnn::vec3* primitiveVertices = reinterpret_cast<tcnn::vec3*>(primitiveData);
#pragma unroll
            for (int i = 0; i < TPrimitive::NumVertices; ++i) {
                primitiveVertices[firstVertexIdx + i] = particles.fromCanonical(densityParameters,
                                                                                tcnn::vec3{TPrimitive::Vertices[i * 3 + 0],
                                                                                           TPrimitive::Vertices[i * 3 + 1],
                                                                                           TPrimitive::Vertices[i * 3 + 2]} *
                                                                                    kernelScale);
            }

            const uint32_t firstFaceIdx = TPrimitive::NumFaces * particleIdx;
            tcnn::uvec3* primitiveFaces = reinterpret_cast<tcnn::uvec3*>(primitiveExtendedData);
#pragma unroll
            for (int i = 0; i < TPrimitive::NumFaces; ++i) {
                primitiveFaces[firstFaceIdx + i] = tcnn::uvec3{TPrimitive::FaceVertexIndices[i * 3 + 0],
                                                               TPrimitive::FaceVertexIndices[i * 3 + 1],
                                                               TPrimitive::FaceVertexIndices[i * 3 + 2]} +
                                                   static_cast<uint32_t>(firstVertexIdx);
            }
        }
    }
#endif
};

template <typename TParams = TParticlePrimitiveDummyParams>
struct TetrahedraParticlePrimitive : public TriangleMeshParticlePrimitive<TetrahedraParticlePrimitive<TParams>, TParams> {
    // Regular tetraHedra
    //     1              Y
    //    / \            |__X
    //   / 2 \            \
    //  0-----3            Z
    //
    // let :
    // r : radius of the inscribed circle = 1
    // s : edge length = sqrt(24) ~ 4.89
    // h : height = sqrt(2/3) * s = sqrt(2/3) * sqrt(24) = sqrt(16) = 4
    // q : h - r = 4 -1 = 3
    // V : volume = s^3 / (6*sqrt(2)) = 13.856406460551014
    static constexpr float tetraHedraInRadius     = 1;
    static constexpr float tetraHedraEdge         = 4.898979485566356;  // sqrt(24)
    static constexpr float tetraHedraHeight       = 4;                  // tetraHedraEdge * sqrt(2/3)
    static constexpr float tetraHedraFaceHeight   = 4.242640687119285;  //  tetraHedraEdge * sqrt(3) / 2
    static constexpr float tetraHedraFaceInRadius = 1.4142135623730951; //  tetraHedraEdge * sqrt(3) / 6 = sqrt(2)

    static constexpr int NumVertices                 = 4;
    static constexpr float Vertices[NumVertices * 3] = {
        -0.5 * tetraHedraEdge, -tetraHedraFaceInRadius, -1,
        0, tetraHedraFaceHeight - tetraHedraFaceInRadius, -1,
        0, 0, tetraHedraHeight - tetraHedraInRadius,
        0.5 * tetraHedraEdge, -tetraHedraFaceInRadius, -1};

    static constexpr int NumFaces                             = 4;
    static constexpr uint32_t FaceVertexIndices[NumFaces * 3] = {
        0, 2, 1,
        0, 3, 2,
        0, 1, 3,
        1, 2, 3};
};

template <typename TParams = TParticlePrimitiveDummyParams>
struct DiamondParticlePrimitive : public TriangleMeshParticlePrimitive<DiamondParticlePrimitive<TParams>, TParams> {
    // Triangular diamond
    //
    //              0
    //             / \
    //            2-3-4
    //             \ /
    //              1
    //
    // let :
    // r : radius of the inscribed circle = 1
    // s : edge length = 6 * r / sqrt(3)
    // h : height = sqrt(2/3) * s
    // V : volume = 2 * s^3 / (6*sqrt(2)) = 9.797958971132713
    static constexpr float diamondEdge       = 3.464101615137755;  // 6 / sqrt(3)
    static constexpr float diamondHeight     = 2.8284271247461903; // tetraHedraEdge * sqrt(2/3) = 2 * sqrt(2)
    static constexpr float diamondFaceHeight = 3;                  //  tetraHedraEdge * sqrt(3) / 2

    static constexpr int NumVertices                 = 5;
    static constexpr float Vertices[NumVertices * 3] = {
        0, diamondHeight, 0,
        0, -diamondHeight, 0,
        -0.5 * diamondEdge, 0, -1,
        0, 0, diamondFaceHeight - 1,
        0.5 * diamondEdge, 0, -1};

    static constexpr int NumFaces                             = 6;
    static constexpr uint32_t FaceVertexIndices[NumFaces * 3] = {
        0, 2, 3,
        0, 4, 2,
        0, 3, 4,
        1, 3, 2,
        1, 2, 4,
        1, 4, 3};
};

template <typename TParams = TParticlePrimitiveDummyParams>
struct OctahedronParticlePrimitive : public TriangleMeshParticlePrimitive<OctahedronParticlePrimitive<TParams>, TParams> {
    // Regular octahedron
    //
    // let
    // r : radius of the inscribed circle = 1
    // s : edge length = 6 / sqrt(6) = 2.4494897427831783
    // h : height = sqrt(2/3) * s = 2
    // V : volume = 2 * s^2 * h / 3 = 8.0
    //
    static constexpr float octaHedraDiag = 1.7320508075688774; // s / sqrt(2)

    static constexpr int NumVertices                 = 6;
    static constexpr float Vertices[NumVertices * 3] = {
        0, 0, -octaHedraDiag,
        0, octaHedraDiag, 0,
        -octaHedraDiag, 0, 0,
        0, -octaHedraDiag, 0,
        octaHedraDiag, 0, 0,
        0, 0, octaHedraDiag};

    static constexpr int NumFaces                             = 8;
    static constexpr uint32_t FaceVertexIndices[NumFaces * 3] = {
        2, 1, 0,
        1, 4, 0,
        4, 3, 0,
        3, 2, 0,
        4, 1, 5,
        3, 4, 5,
        2, 3, 5,
        1, 2, 5};
};

template <typename TParams = TParticlePrimitiveDummyParams>
struct IcosahedronParticlePrimitive : public TriangleMeshParticlePrimitive<IcosahedronParticlePrimitive<TParams>, TParams> {
    // regular icosahedron
    //
    // let :
    // phi : golden ratio = (1 + sqrt(5)) / 2
    // r : radius of the inscribed circle = 1 = (phi^2 * s) / ( 2 * sqrt(3))
    // s : edge length = ( 2 * sqrt(3) ) / phi^2
    // V : volume = (5/12) * ( 3 + sqrt(5) ) * s^3 = 8.0
    static constexpr float goldenRatio       = 1.618033988749895;
    static constexpr float edge              = 1.323169076499215;
    static constexpr float scale             = 0.5 * edge;
    static constexpr float scaledGoldenRatio = scale * goldenRatio;

    static constexpr int NumVertices                 = 12;
    static constexpr float Vertices[NumVertices * 3] = {
        -scale, scaledGoldenRatio, 0.f,
        scale, scaledGoldenRatio, 0.f,
        0.f, scale, -scaledGoldenRatio,
        -scaledGoldenRatio, 0.f, -scale,
        -scaledGoldenRatio, 0.f, scale,
        0.f, scale, scaledGoldenRatio,
        scaledGoldenRatio, 0.f, scale,
        0.f, -scale, scaledGoldenRatio,
        -scale, -scaledGoldenRatio, 0.f,
        0.f, -scale, -scaledGoldenRatio,
        scaledGoldenRatio, 0.f, -scale,
        scale, -scaledGoldenRatio, 0.f};

    static constexpr int NumFaces                             = 20;
    static constexpr uint32_t FaceVertexIndices[NumFaces * 3] = {
        0, 1, 2,
        0, 2, 3,
        0, 3, 4,
        0, 4, 5,
        0, 5, 1,
        6, 1, 5,
        6, 5, 7,
        6, 7, 11,
        6, 11, 10,
        6, 10, 1,
        8, 4, 3,
        8, 3, 9,
        8, 9, 11,
        8, 11, 7,
        8, 7, 4,
        9, 3, 2,
        9, 2, 10,
        9, 10, 11,
        5, 4, 7,
        1, 10, 2};
};

template <typename TParams = TParticlePrimitiveDummyParams>
struct RhombusParticlePrimitive : public TriangleMeshParticlePrimitive<RhombusParticlePrimitive<TParams>, TParams> {
    // rhombus surface element
    //
    //          /\
        //          --
    //          \/
    //
    static constexpr float rhombusDiag = 1.4142135623730951; // sqrt(2)

    static constexpr int NumVertices                 = 4;
    static constexpr float Vertices[NumVertices * 3] = {rhombusDiag, 0, 0,
                                                        -rhombusDiag, 0, 0,
                                                        0, rhombusDiag, 0,
                                                        0, -rhombusDiag, 0};

    static constexpr int NumFaces                             = 2;
    static constexpr uint32_t FaceVertexIndices[NumFaces * 3] = {
        0, 1, 2,
        0, 1, 3};
};

template <typename TParams = TParticlePrimitiveDummyParams>
struct TrisurfelParticlePrimitive : public TriangleMeshParticlePrimitive<TrisurfelParticlePrimitive<TParams>, TParams> {
    // triangle surface element
    //
    //          /\
    //          --
    //
    static constexpr float CosPiOver6 = 0.8660254037844387; // cos(pi/6)
    static constexpr float SinPiOver6 = 0.5;                // sin(pi/6)

    static constexpr int NumVertices                 = 4;
    static constexpr float Vertices[NumVertices * 3] = {
        -CosPiOver6, -SinPiOver6, 0,
        0, 1.0f, 0,
        CosPiOver6, -SinPiOver6, 0};

    static constexpr int NumFaces                             = 2;
    static constexpr uint32_t FaceVertexIndices[NumFaces * 3] = {0, 1, 2};
};

template <typename TParams = TParticlePrimitiveDummyParams>
struct SphereParticlePrimitive {

#ifdef __CUDACC__
    template <typename TParticle>
    static inline __device__ void eval(const uint32_t numParticles,
                                       void* __restrict__ primitiveData,
                                       void* __restrict__ primitiveExtendedData,
                                       OptixTraversableHandle,
                                       nrend::MemoryHandles parameters) {

        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleIdx < numParticles) {
            TParticle particles;
            particles.initializeDensity(parameters);

            const typename TParticle::DensityParameters densityParameters = particles.fetchDensityParameters(particleIdx);
            const float kernelScale                                       = particles.canonicalScale(densityParameters, TParams::DensityScaleClamping);

            reinterpret_cast<tcnn::vec3*>(primitiveData)[particleIdx]    = densityParameters.position(densityParameters);
            reinterpret_cast<float*>(primitiveExtendedData)[particleIdx] = fmaxf(densityKernel.scale.x, fmaxf(densityKernel.scale.y, densityKernel.scale.z)) * kernelScale;
        }
    }
#endif
};

template <typename TParams = TParticlePrimitiveDummyParams>
struct AabbParticlePrimitive {

    static constexpr int NumVertices                 = 8;
    static constexpr float Vertices[NumVertices * 3] = {
        -1, -1, -1,
        -1, -1, 1,
        -1, 1, -1,
        -1, 1, 1,
        1, -1, -1,
        1, -1, 1,
        1, 1, -1,
        1, 1, 1};

#ifdef __CUDACC__
    template <typename TParticle>
    static inline __device__ void eval(const uint32_t numParticles,
                                       void* __restrict__ primitiveData,
                                       void* __restrict__,
                                       OptixTraversableHandle asHandle,
                                       nrend::MemoryHandles parameters) {

        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleIdx < numParticles) {
            TParticle particles;
            particles.initializeDensity(parameters);

            const typename TParticle::DensityParameters densityParameters = particles.fetchDensityParameters(particleIdx);
            const tcnn::vec3 position                                     = particles.position(densityParameters);
            const tcnn::mat3 rotation                                     = particles.rotation(densityParameters);
            const float kernelScale                                       = particles.canonicalScale(densityParameters, TParams::DensityScaleClamping);

            OptixAabb aabb;
#pragma unroll
            for (int i = 0; i < NumVertices; ++i) {
                const auto vrt = particles.fromCanonical(densityParameters,
                                                         tcnn::vec3{Vertices[i * 3 + 0],
                                                                    Vertices[i * 3 + 1],
                                                                    Vertices[i * 3 + 2]} *
                                                             kernelScale);
                if (i == 0) {
                    aabb.minX = vrt.x;
                    aabb.minY = vrt.y;
                    aabb.minZ = vrt.z;
                    aabb.maxX = vrt.x;
                    aabb.maxY = vrt.y;
                    aabb.maxZ = vrt.z;
                } else {
                    aabb.minX = fminf(aabb.minX, vrt.x);
                    aabb.minY = fminf(aabb.minY, vrt.y);
                    aabb.minZ = fminf(aabb.minZ, vrt.z);
                    aabb.maxX = fmaxf(aabb.maxX, vrt.x);
                    aabb.maxY = fmaxf(aabb.maxY, vrt.y);
                    aabb.maxZ = fmaxf(aabb.maxZ, vrt.z);
                }
            }
            reinterpret_cast<OptixAabb*>(primitiveData)[particleIdx] = aabb;
        }
    }
#endif
};

template <typename TParams                 = TParticlePrimitiveDummyParams,
          OptixInstanceFlags InstanceFlags = OPTIX_INSTANCE_FLAG_NONE>
struct TransformedAabbParticlePrimitive {

#ifdef __CUDACC__
    template <typename TParticle>
    static inline __device__ void eval(const uint32_t numParticles,
                                       void* __restrict__ primitiveData,
                                       void* __restrict__,
                                       OptixTraversableHandle asHandle,
                                       nrend::MemoryHandles parameters) {

        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleIdx < numParticles) {
            TParticle particles;
            particles.initializeDensity(parameters);

            const typename TParticle::DensityParameters densityParameters = particles.fetchDensityParameters(particleIdx);
            const tcnn::vec3 position                                     = particles.position(densityParameters);
            const tcnn::mat3 rotation                                     = particles.rotation(densityParameters);
            const float kernelScale                                       = particles.canonicalScale(densityParameters, TParams::DensityScaleClamping);
            const tcnn::vec3 scale                                        = particles.scale(densityParameters) * kernelScale;
            const float opacity                                           = particles.opacity(densityParameters);

            OptixInstance instance;
            instance.instanceId        = opacityAsInstanceId(opacity);
            instance.sbtOffset         = 0;
            instance.visibilityMask    = 255;
            instance.flags             = InstanceFlags;
            instance.traversableHandle = asHandle;
            instance.transform[0]      = rotation[0].x * scale.x;
            instance.transform[1]      = rotation[1].x * scale.y;
            instance.transform[2]      = rotation[2].x * scale.z;
            instance.transform[3]      = position.x;
            instance.transform[4]      = rotation[0].y * scale.x;
            instance.transform[5]      = rotation[1].y * scale.y;
            instance.transform[6]      = rotation[2].y * scale.z;
            instance.transform[7]      = position.y;
            instance.transform[8]      = rotation[0].z * scale.x;
            instance.transform[9]      = rotation[1].z * scale.y;
            instance.transform[10]     = rotation[2].z * scale.z;
            instance.transform[11]     = position.z;

            reinterpret_cast<OptixInstance*>(primitiveData)[particleIdx] = instance;
        }
    };
#endif
};

} // namespace nrend
