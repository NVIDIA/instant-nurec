// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include "../common/sphericalHarmonics.slang"

namespace shRadiativeParticle
{

struct ParametersBuffer
{
    ImmutablePtr<ShRadianceCoefficientsType> _dataPtr;
    ShRadianceCoefficientsType *_gradPtr;
    bool exclusiveGradient; //< true if the gradient maybe updated without atomics
};

struct CommonParameters
{
    ParametersBuffer parametersBuffer;
    int sphDegree;
};

struct Parameters : IDifferentiable
{
    vector<float, Dim> shCoefficients[RadianceMaxNumShCoefficients];
};

[BackwardDifferentiable][ForceInline]
Parameters fetchParametersFromBuffer(no_diff uint32_t particleIdx,
                                     no_diff ParametersBuffer parametersBuffer) 
{
    Parameters parameters;
    const uint32_t particleOffset = particleIdx * RadianceMaxNumShCoefficients * Dim;
    [ForceUnroll] for (int i = 0; i < RadianceMaxNumShCoefficients; ++i) {
        [ForceUnroll] for (int j = 0; j < Dim; ++j) {
            parameters.shCoefficients[i][j] = (float)(parametersBuffer._dataPtr[particleOffset + i*Dim + j]);
        }
    }
    return parameters;
}

[BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline] 
void fetchParametersFromBufferBwd(no_diff uint32_t particleIdx,
                                  no_diff ParametersBuffer parametersBuffer,
                                  Parameters parametersGrad) 
{
    if (Differentiable) {
        const uint32_t particleOffset = particleIdx * RadianceMaxNumShCoefficients * Dim;
        if (parametersBuffer.exclusiveGradient) {
            [ForceUnroll] for (int i = 0; i < RadianceMaxNumShCoefficients; ++i) {
                const vector<float, Dim> coeffs = parametersGrad.shCoefficients[i];
                [ForceUnroll] for (int j = 0; j < Dim; ++j) {
                    parametersBuffer._gradPtr[particleOffset + i*Dim + j] += (ShRadianceCoefficientsType)(coeffs[j]);
                }
            }
        } else {
            [ForceUnroll] for (int i = 0; i < RadianceMaxNumShCoefficients; ++i) {
                const vector<float, Dim> coeffs = parametersGrad.shCoefficients[i];
                [ForceUnroll] for (int j = 0; j < Dim; ++j) {
                    InterlockedAdd(parametersBuffer._gradPtr[particleOffset + i*Dim + j], (ShRadianceCoefficientsType)(coeffs[j]));
                }
            }
        }
    }
}

[BackwardDifferentiable] [ForceInline]
vector<float, Dim> radianceFromBuffer(no_diff uint32_t particleIdx,
                                      in float3 incidentDirection,
                                      no_diff uint32_t sphDegree,
                                      no_diff ParametersBuffer parametersBuffer)
{
    if (RadianceSphO0 && RadianceMaxNumShCoefficients == 1) {
        // Direct radiance from coefficient (no SH basis scaling)
        return max(fetchParametersFromBuffer(particleIdx, parametersBuffer).shCoefficients[0], 
                   vector<float, Dim>(0.0f));
    } else {
        return sphericalHarmonics.decode<Dim, RadianceMaxNumShCoefficients, true>(
                sphDegree,
                fetchParametersFromBuffer(particleIdx, parametersBuffer).shCoefficients,
                incidentDirection);
    }
}

[BackwardDifferentiable][ForceInline]
void integrateRadiance<let BackToFront : bool>(float weight,
                                               in vector<float, Dim> radiance,
                                               inout vector<float, Dim> integratedRadiance)
{
    if (weight > 0.0f)
    {
        if (BackToFront)
        {
            integratedRadiance = lerp(integratedRadiance, radiance, weight);
        }
        else
        {
            integratedRadiance += radiance * weight;
        }
    }
}

[BackwardDifferentiable][ForceInline]
void integrateRadianceFromParameters<let BackToFront : bool>(float3 incidentDirection,
                                                             no_diff uint32_t sphDegree,
                                                             float weight,
                                                             Parameters parameters,
                                                             inout vector<float, Dim> integratedRadiance) 
{
    if (weight > 0.0f) 
    {
        integrateRadiance<BackToFront>(
            weight,
            (RadianceSphO0 && RadianceMaxNumShCoefficients==1) ? 
                max(parameters.shCoefficients[0], vector <float, Dim>(0.0f)) : 
                sphericalHarmonics.decode<Dim, RadianceMaxNumShCoefficients, true>(
                    sphDegree,
                    parameters.shCoefficients,
                    incidentDirection),
            integratedRadiance
        );
    }
}

[BackwardDifferentiable][ForceInline]
void integrateRadianceFromBuffer<let BackToFront : bool>(float3 incidentDirection,
                                                         no_diff uint32_t sphDegree,
                                                         float weight,
                                                         no_diff uint32_t particleIdx,
                                                         no_diff ParametersBuffer parametersBuffer,
                                                         inout vector<float, Dim> integratedRadiance)
{
    integrateRadianceFromParameters<BackToFront>(incidentDirection,
                                                 sphDegree,
                                                 weight,
                                                 fetchParametersFromBuffer(particleIdx, parametersBuffer),
                                                 integratedRadiance);
}

} // namespace shRadiativeParticle

// ------------------------------------------------------------------------------------------------------------------
// Entry points

[CudaDeviceExport]
inline vector<float, shRadiativeParticle.Dim> particleFeaturesFromBuffer(in uint32_t particleIdx,
                                         shRadiativeParticle.CommonParameters commonParameters,
                                         in float3 incidentDirection)
{
    if (shRadiativeParticle.RadianceSphO0 && shRadiativeParticle.RadianceMaxNumShCoefficients == 1) {
        // Direct radiance from coefficient (no SH basis scaling)
        return max(shRadiativeParticle.fetchParametersFromBuffer(particleIdx, commonParameters.parametersBuffer).shCoefficients[0], 
                   vector<float, shRadiativeParticle.Dim>(0.0f));
    }
    else {
        return sphericalHarmonics.decode<shRadiativeParticle.Dim, shRadiativeParticle.RadianceMaxNumShCoefficients, true>(
                commonParameters.sphDegree,
                shRadiativeParticle.fetchParametersFromBuffer(particleIdx, commonParameters.parametersBuffer).shCoefficients,
                incidentDirection);
    }
}

[CudaDeviceExport]
inline void particleFeaturesIntegrateFwd(in float weight,
                                         in vector<float, shRadiativeParticle.Dim> features,
                                         inout vector<float, shRadiativeParticle.Dim> integratedFeatures)
{
    shRadiativeParticle.integrateRadiance<false>(
        weight,
        features,
        integratedFeatures
    );
}

[CudaDeviceExport] inline void particleFeaturesIntegrateFwdFromBuffer(in float3 incidentDirection,
                                                   in float weight,
                                                   in uint32_t particleIdx,
                                                   shRadiativeParticle.CommonParameters commonParameters,
                                                   inout vector<float, shRadiativeParticle.Dim> integratedFeatures)
{
    shRadiativeParticle.integrateRadianceFromBuffer<false>(
        incidentDirection,
        commonParameters.sphDegree,
        weight,
        particleIdx,
        commonParameters.parametersBuffer,
        integratedFeatures);
}

[CudaDeviceExport] void particleFeaturesIntegrateBwd(
    in float alpha,
    inout float alphaGrad,
    in vector<float, shRadiativeParticle.Dim> features,
    inout vector<float, shRadiativeParticle.Dim> featuresGrad,
    inout vector<float, shRadiativeParticle.Dim> integratedFeatures,
    inout vector<float, shRadiativeParticle.Dim> integratedFeaturesGrad)
{
    if(alpha > 0.0f)
    {
        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, 0.f);
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> featuresDiff =
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(features, vector<float, shRadiativeParticle.Dim>(0.f));

        const float weight = 1.0f / (1.0f - alpha);
        integratedFeatures = (integratedFeatures - features * alpha) * weight;
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff = 
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedFeatures, integratedFeaturesGrad);

        bwd_diff(shRadiativeParticle.integrateRadiance<true>)(
            alphaDiff,
            featuresDiff,
            integratedFeaturesDiff);

        alphaGrad += alphaDiff.getDifferential();
        featuresGrad += featuresDiff.getDifferential();
        integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();
    }
}

[CudaDeviceExport] void particleFeaturesIntegrateBwdToBuffer(
    in float3 incidentDirection,
    in bool incidentDirectionGradEnabled,
    inout float3 incidentDirectionGrad,
    in float alpha,
    inout float alphaGrad,
    in uint32_t particleIdx,
    shRadiativeParticle.CommonParameters commonParameters,
    in vector<float, shRadiativeParticle.Dim> features,
    inout vector<float, shRadiativeParticle.Dim> integratedFeatures,
    inout vector<float, shRadiativeParticle.Dim> integratedFeaturesGrad) 
{
    if (alpha > 0.0f)
    {
        DifferentialPair<float3> incidentDirectionDiff = DifferentialPair<float3>(incidentDirection, float3(0.f));
        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, 0.f);
        
        const float weight = 1.0f / (1.0f - alpha);
        integratedFeatures = (integratedFeatures - features * alpha) * weight;
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff = 
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedFeatures, integratedFeaturesGrad);

        bwd_diff(shRadiativeParticle.integrateRadianceFromBuffer<true>)(
            incidentDirectionDiff,
            commonParameters.sphDegree,
            alphaDiff,
            particleIdx,
            commonParameters.parametersBuffer,
            integratedFeaturesDiff);

        integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();
        alphaGrad += alphaDiff.getDifferential();
        if (incidentDirectionGradEnabled) {
            incidentDirectionGrad += incidentDirectionDiff.getDifferential();
        }
    }
}

[CudaDeviceExport] void particleFeaturesIntegrateSampleBwdToBuffer(
    in float3 incidentDirection,
    in bool incidentDirectionGradEnabled,
    inout float3 incidentDirectionGrad,
    in float alpha,
    inout float alphaGrad,
    in uint32_t particleIdx,
    shRadiativeParticle.CommonParameters commonParameters,
    in vector<float, shRadiativeParticle.Dim> features,
    inout vector<float, shRadiativeParticle.Dim> integratedFeatures,
    inout vector<float, shRadiativeParticle.Dim> integratedFeaturesGrad)
{
    if (alpha > 0.0f)
    {
        DifferentialPair<float3> incidentDirectionDiff = DifferentialPair<float3>(incidentDirection, float3(0.f));
        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, 0.f);

        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff =
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>((integratedFeatures - features * alpha), integratedFeaturesGrad);

        bwd_diff(shRadiativeParticle.integrateRadianceFromBuffer<false>)(
            incidentDirectionDiff,
            commonParameters.sphDegree,
            alphaDiff,
            particleIdx,
            commonParameters.parametersBuffer,
            integratedFeaturesDiff);

        alphaGrad += alphaDiff.getDifferential();
        if (incidentDirectionGradEnabled) {
            incidentDirectionGrad += incidentDirectionDiff.getDifferential();
        }
    }
}

[CudaDeviceExport] void particleFeaturesBwdToBuffer(
    in uint32_t particleIdx,
    shRadiativeParticle.CommonParameters commonParameters,
    in vector<float, shRadiativeParticle.Dim> featuresGrad,
    in float3 incidentDirection,
    inout float3 incidentDirectionGrad
)
{
    DifferentialPair<float3> incidentDirectionDiff = DifferentialPair<float3>(incidentDirection, float3(0.f));

    bwd_diff(shRadiativeParticle.radianceFromBuffer)(
        particleIdx,
        incidentDirectionDiff,
        commonParameters.sphDegree,
        commonParameters.parametersBuffer,
        featuresGrad);

    incidentDirectionGrad += incidentDirectionDiff.getDifferential(); 
}