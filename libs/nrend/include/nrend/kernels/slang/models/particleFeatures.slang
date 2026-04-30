// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#pragma once

#include "../common/sphericalHarmonics.slang"

namespace particleFeatures 
{

#ifdef STANDALONE
typealias ParametersType = half;
const static bool Differentiable = true;
#endif

struct Vector<let FeaturesDim: int> {

    const static int Dim = FeaturesDim;

    typealias FeatureArray = float[Dim];
    typealias FeatureVector = float[Dim]; // vector<float, Dim>; not supported when Dim>4

    struct ParametersBuffer
    {
        ImmutablePtr<ParametersType> _dataPtr;
        ParametersType* _gradPtr;
        bool exclusiveGradient; //< true if the gradient maybe updated without atomics
    };

    [BackwardDifferentiable][ForceInline]
    static FeatureVector fetchFromBuffer(no_diff uint32_t particleIdx,
                                         no_diff ParametersBuffer buffer) 
    {
        FeatureVector features;
        const uint32_t particleOffset = particleIdx * Dim;
        [ForceUnroll]
        for (int i = 0; i < Dim; ++i) {
            features[i] = float(buffer._dataPtr[particleOffset + i]);
        }
        return features;
    }

    [BackwardDerivativeOf(fetchFromBuffer)][ForceInline] 
    static void fetchFromBufferBwd(no_diff uint32_t particleIdx,
                            no_diff ParametersBuffer buffer,
                            FeatureVector grad)
    {
        if (Differentiable) {
            const uint32_t particleOffset = particleIdx * Dim;
            if (buffer.exclusiveGradient) {
                [ForceUnroll] for (int i = 0; i < Dim; ++i) {
                    buffer._gradPtr[particleOffset + i] += (ParametersType)(grad[i]);
                }
            }
            else {
                [ForceUnroll] for (int i = 0; i < Dim; ++i) {
                    InterlockedAdd(buffer._gradPtr[particleOffset + i], (ParametersType)(grad[i]));
                }
            }
        }
    }

    [BackwardDifferentiable][ForceInline]
    static void integrate<let backToFront : bool>(float weight,
                                        in FeatureVector features,
                                        inout FeatureVector integratedFeatures)
    {
        if (weight > 0.0f)
        {
            if (backToFront)
            {
                [ForceUnroll] for (int i = 0; i < Dim; ++i) {
                    integratedFeatures[i] = lerp(integratedFeatures[i], features[i], weight);
                }
            }
            else
            {
                [ForceUnroll] for (int i = 0; i < Dim; ++i) {
                    integratedFeatures[i] += features[i] * weight;
                }
            }
        }
    }

    [BackwardDifferentiable][ForceInline]
    static void integrateFromBuffer<let backToFront : bool>(float weight,
                                                    no_diff uint32_t particleIdx,
                                                    no_diff ParametersBuffer parametersBuffer,
                                                    inout FeatureVector integratedFeatures) 
    {
        if (weight > 0.0f)
        {
            integrate<backToFront>(weight, fetchFromBuffer(particleIdx, parametersBuffer), integratedFeatures);
        }
    }
} // struct ExtendedFeatures

struct ShVector<let FeaturesDim: int, let FeaturesShDegree: int, let NumShCoefficients: int> {
    
    const static int Dim = FeaturesDim;
    const static int ShDegree = FeaturesShDegree;

    typealias FeatureArray = float[Dim];
    typealias FeatureVector = float[Dim]; // vector<float, Dim>; not supported when Dim>4

    //static const int NumShCoefficients = (ShDegree + 1) * (ShDegree + 1);
    static const int ParametersDim = Dim * NumShCoefficients;
    typealias ParametersArray = float[ParametersDim];

    struct ParametersBuffer
    {
#ifdef STANDALONE
        Ptr<ParametersType, Access.Read> _dataPtr;
        Ptr<ParametersType, Access.ReadWrite> _gradPtr;
#else
        ImmutablePtr<ParametersType> _dataPtr;
        ParametersType* _gradPtr;
#endif
        bool exclusiveGradient; //< true if the gradient maybe updated without atomics
    };

    [BackwardDifferentiable] [ForceInline]
    static ParametersArray fetchParametersFromBuffer(no_diff uint32_t particleIdx,
                                                     no_diff ParametersBuffer buffer) 
    {
        ParametersArray parameters;
        const uint32_t particleOffset = particleIdx * ParametersDim;
        [ForceUnroll] for (int i = 0; i < ParametersDim; ++i) {
            parameters[i] = (float)(buffer._dataPtr[particleOffset + i]);
        }
        return parameters;
    }

    [BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline]
    static void fetchParametersFromBufferBwd(no_diff uint32_t particleIdx,
                                    no_diff ParametersBuffer buffer,
                                    ParametersArray grad)
    {
        if (Differentiable)
        {
            const uint32_t particleOffset = particleIdx * ParametersDim;
            if (buffer.exclusiveGradient) {
                [ForceUnroll] for (int i = 0; i < ParametersDim; ++i) {
                    buffer._gradPtr[particleOffset + i] += (ParametersType)(grad[i]);
                }
            }
            else {
                [ForceUnroll] for (int i = 0; i < ParametersDim; ++i) {
                    InterlockedAdd(buffer._gradPtr[particleOffset + i], (ParametersType)(grad[i]));
                }
            }
        }
    }

    [BackwardDifferentiable] [ForceInline]
    static FeatureVector fetchFromBuffer(no_diff uint32_t particleIdx,
                                         no_diff ParametersBuffer buffer,
                                         in float3 direction)
    {
        const ParametersArray parameters = fetchParametersFromBuffer(particleIdx, buffer);
        FeatureVector features;
        sphericalHarmonics.decode<Dim, NumShCoefficients, false>(ShDegree, parameters, direction, features);
        return features;
    }

    [BackwardDifferentiable][ForceInline]
    static void integrate<let backToFront : bool>(float weight,
                                        in FeatureVector features,
                                        inout FeatureVector integratedFeatures)
    {
        if (weight > 0.0f)
        {
            if (backToFront)
            {
                [ForceUnroll] for (int i = 0; i < Dim; ++i) {
                    integratedFeatures[i] = lerp(integratedFeatures[i], features[i], weight);
                }
            }
            else
            {
                [ForceUnroll] for (int i = 0; i < Dim; ++i) {
                    integratedFeatures[i] += features[i] * weight;
                }
            }
        }
    }

    [BackwardDifferentiable][ForceInline]
    static void integrateFromBuffer<let backToFront : bool>(in float3 direction,
                                                    float weight,
                                                    no_diff uint32_t particleIdx,
                                                    no_diff ParametersBuffer parametersBuffer,
                                                    inout FeatureVector integratedFeatures) 
    {
        if (weight > 0.0f)
        {
            integrate<backToFront>(weight, fetchFromBuffer(particleIdx, parametersBuffer, direction), integratedFeatures);
        }
    }
} // struct ShParticle

} // namespace particleFeatures
