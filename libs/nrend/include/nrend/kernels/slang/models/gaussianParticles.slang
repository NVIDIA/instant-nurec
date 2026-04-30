// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include "../common/cameras.slang"
#include "../common/transforms.slang"
#include "../common/utils.slang"

namespace gaussianParticle {

struct RawParameters : IDifferentiable {
    float3 position;
    float density;
    float4 quaternion;
    float3 scale;
    float padding;
};

struct RawParametersBuffer {
    ImmutablePtr<RawParameters> _dataPtr;
    RawParameters *_gradPtr;
    bool exclusiveGradient; //< true if the gradient maybe updated without atomics
};

struct CommonParameters {
    RawParametersBuffer parametersBuffer;
};

[BackwardDifferentiable][ForceInline] RawParameters fetchParametersFromBuffer(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer) {
    RawParameters rawParameters = parametersBuffer._dataPtr[particleIdx];
    if (Surfel) {
        rawParameters.scale.z = 1e-06f;
    }
    return rawParameters;
}

[BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline] void fetchParametersFromBufferBwd(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    RawParameters rawParameters) {
    if (parametersBuffer.exclusiveGradient) {
        parametersBuffer._gradPtr[particleIdx].density += rawParameters.density;
        parametersBuffer._gradPtr[particleIdx].position += rawParameters.position;
        parametersBuffer._gradPtr[particleIdx].quaternion += rawParameters.quaternion;
        if (!Surfel) {
            parametersBuffer._gradPtr[particleIdx].scale += rawParameters.scale;
        } else {
            parametersBuffer._gradPtr[particleIdx].scale.xy += rawParameters.scale.xy;
        }
    } else {
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].density, rawParameters.density);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.x, rawParameters.position.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.y, rawParameters.position.y);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.z, rawParameters.position.z);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.x, rawParameters.quaternion.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.y, rawParameters.quaternion.y);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.z, rawParameters.quaternion.z);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.w, rawParameters.quaternion.w);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.x, rawParameters.scale.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.y, rawParameters.scale.y);
        if (!Surfel) {
            InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.z, rawParameters.scale.z);
        }
    }

}

struct Parameters : IDifferentiable {
    float3 position;
    float3 scale;
    float3x3 rotationT;
    float density;
};

[BackwardDifferentiable] [ForceInline] Parameters parametersFromRaw(
    in gaussianParticle.RawParameters rawParameters) {
    return {
        rawParameters.position,
        rawParameters.scale,
        transforms.rotationMatrixTranspose(rawParameters.quaternion),
        rawParameters.density};
}

[BackwardDifferentiable][ForceInline] Parameters fetchParameters(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer) {
    return parametersFromRaw(fetchParametersFromBuffer(particleIdx, parametersBuffer));
}

[BackwardDifferentiable][ForceInline] void cannonicalRay(
    in float3 rayOrigin,
    in float3 rayDirection,
    in Parameters parameters,
    out float3 particleRayOrigin,
    out float3 particleRayDirection, ) {
    const float3 giscl  = float3(1.0f) / parameters.scale;
    const float3 gposc  = (rayOrigin - parameters.position);
    const float3 gposcr = mul(parameters.rotationT, gposc);
    particleRayOrigin   = giscl * gposcr;

    const float3 rayDirR = mul(parameters.rotationT, rayDirection);
    const float3 grdu    = giscl * rayDirR;
    particleRayDirection = utils.safe_normalize(grdu);
}

[BackwardDifferentiable][ForceInline] float canonicalRayMinSquaredDistance<let Surfel : bool>(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection)
{
    // NB : function is called only for intersecting rays
    float3 gcrod;
    if (Surfel) {
        // if ray z direction is null, the ray origin is on the plane z = 0
        gcrod = abs(canonicalRayDirection.z) > 1e-06 ? canonicalRayOrigin + canonicalRayDirection * -canonicalRayOrigin.z / canonicalRayDirection.z : canonicalRayOrigin;
    } else {
        gcrod = cross(canonicalRayDirection, canonicalRayOrigin);
    }
    return dot(gcrod, gcrod);
}

[BackwardDifferentiable][ForceInline] float canonicalRayMaxKernelResponse<let KernelDegree : int, let Surfel : bool>(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection) {
    const float grayDist = canonicalRayMinSquaredDistance<Surfel>(canonicalRayOrigin, canonicalRayDirection);

    /// generalized gaussian of degree n : scaling is s = -4.5/3^n
    switch (KernelDegree) {
    case 8: // Zenzizenzizenzic
    {
        /*static const*/ float s = -0.000685871056241;
        const float grayDistSq = grayDist * grayDist;
        return exp(s * grayDistSq * grayDistSq) ;
    }
    case 5: // Quintic
    {
        /*static const*/ float s = -0.0185185185185;
        return exp(s * grayDist * grayDist * sqrt(grayDist));
    }
    case 4: // Tesseractic
    {
        /*static const*/ float s = -0.0555555555556;
        return exp(s * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        /*static const*/ float s = -0.166666666667;
        return exp(s * grayDist * sqrt(grayDist));
    }
    case 1: // Laplacian
    {
        /*static const*/ float s = -1.5f;
        return exp(s * sqrt(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487;
        return max(1 + s * sqrt(grayDist), 0.f);
    }
    default: // Quadratic
    {
        /*static const*/ float s = -0.5f;
        return exp(s * grayDist);
    }
    }
}

[ForceInline] float canonicalScale<let KernelDegree : int>(in float kernelDegree, in float density, in bool densityScaleClamping) {

    const float responseModulation = densityScaleClamping ? density : 1.f;
    const float minResponse = clamp(MinParticleKernelDensity / responseModulation, 0.0001f, 0.999f);

    // bump kernel
    if (KernelDegree < 0) {
        const float k = abs(KernelDegree);
        const float s = 1.0f / pow(KernelScale, k);
        const float ks = pow((1.f / (log(minResponse) - 1.f) + 1.f) / s, 1.f / k);
        return ks;
    }

    // linear kernel
    else if (KernelDegree == 0) {
        return ((1.0f - minResponse) / KernelScale) / -0.329630334487f;
    }

    // (generalized) gaussian kernel
    else if ((KernelDegree > 0) && (KernelDegree <= 9)) {
        /// generalized gaussian of degree b : scaling a = a1 / a2 = -0.5s^2 / s^b
        /// e^{a*|x|^b}
        ///
        /// find distance r (>0) st e^{a*r^b} = minResponse
        const float KernelScaleDensity = -0.5f * KernelScale * KernelScale;
        return pow(log(minResponse)/KernelScaleDensity, 1.0f / KernelDegree) * KernelScale;
    }

    // default
    return KernelScale;
}

[BackwardDifferentiable][ForceInline] float canonicalRayDistance(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection,
    float3 scale) {
    const float3 grds = scale * canonicalRayDirection * dot(canonicalRayDirection, -1 * canonicalRayOrigin);
    return sqrt(dot(grds, grds));
}

[BackwardDifferentiable][ForceInline] float3 canonicalRayNormal(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection,
    float3 scale,
    float3x3 rotationT) {
    
    // TODO : unify the computation of normals
    float3 surfelNm = float3(0, 0, 1); // fetchSurfelNm<MOGTRACING_PRIMITIVE_TYPE>(primId % params.gPrimNumTri);
    // resolve direction ambiguities
    if (dot(surfelNm, canonicalRayDirection) > 0) {
        surfelNm *= -1.0f;
    }
    return utils.safe_normalize(mul(surfelNm, rotationT));
}

[BackwardDifferentiable][ForceInline]
bool hit<let ValidatedHit : bool, let KnownHitDistance : bool>(
    float3 rayOrigin,
    float3 rayDirection,
    no_diff float raySpread,
    Parameters parameters,
    out float alpha,
    inout float hitDistance,
    no_diff bool enableNormal,
    inout float3 normal) {

    float3 canonicalRayOrigin;
    float3 canonicalRayDirection;
    cannonicalRay(
        rayOrigin,
        rayDirection,
        parameters,
        canonicalRayOrigin,
        canonicalRayDirection);

    float maxResponse;

    if (RaySpreadFilterEnabled) {

        static_assert(KernelDegree == 2, "ray-spread filter is only supported for quadratic kernels");

        // FIXME : to be optimized

        if (!KnownHitDistance) {
            hitDistance = canonicalRayDistance(canonicalRayOrigin, canonicalRayDirection, parameters.scale);
        }
        
        const float3 x = rayOrigin + rayDirection * hitDistance - parameters.position;
        const float3 y = mul(parameters.rotationT, x);
        const float3 s = parameters.scale;
        float v = raySpread * hitDistance; // => depth along the ray / focal
        v = v * v;

        // https://arxiv.org/abs/2504.12811

        // xᵀ(RSSᵀRᵀ + vI)⁻¹x = xᵀ(R(S² + vI) Rᵀ)⁻¹x                                                          // Since RSSᵀRᵀ = R(S²)Rᵀ and vI = R(vI)Rᵀ
        //     = xᵀR(S² + vI)⁻¹Rᵀx                                                                            // (RARᵀ)⁻¹ = RA⁻¹Rᵀ for orthogonal R
        //     = yᵀ(S² + vI)⁻¹y                                                                               // Since y = Rᵀx
        //     = Σᵢ(yᵢ² / (sᵢ² + v))                                                                          // Since S² and vI are diagonal
        // Compute yᵀ(S² + vI)⁻¹y = Σᵢ(yᵢ² / (sᵢ² + v))
        const float graydistFiltered = y.x * y.x / (s.x * s.x + v) + 
                                       y.y * y.y / (s.y * s.y + v) + 
                                       y.z * y.z / (s.z * s.z + v);
        maxResponse = exp(-0.5f * graydistFiltered);
        
        // Compute the ratio between the projected area of the Gaussian without and with ray spread
        // For a ray direction d and covariance matrix Σ = RSSᵀRᵀ, the projected area is proportional to √(dᵀΣd)
        // Without ray spread: Σ = RSSᵀRᵀ
        // With ray spread: Σ = RSSᵀRᵀ + VVᵀ = R(SSᵀ + vI)Rᵀ
        // We compute the ratio √(dᵀRSSᵀRᵀd) / √(dᵀR(SSᵀ + vI)Rᵀd)
        float3 sp = s * s + v;  // diagonal of SSᵀ + vI
        sp = float3(sp.y*sp.z, sp.x*sp.z, sp.x*sp.y);  // cofactors of SSᵀ + vI
        float3 ssq = s * s;  // diagonal of SSᵀ
        ssq = float3(ssq.y * ssq.z, ssq.x * ssq.z, ssq.x * ssq.y);  // cofactors of SSᵀ
        float3 dp = mul(parameters.rotationT, rayDirection);  // Rᵀd
        dp = dp * dp;  // squared components of Rᵀd
        const float ratio = sqrt(dot(dp,ssq)/dot(dp,sp));  // final ratio

        maxResponse *= ratio;
    }
    else
    {
        maxResponse = canonicalRayMaxKernelResponse<KernelDegree, Surfel>(
            canonicalRayOrigin,
            canonicalRayDirection
        );
    }

    alpha = min(MaxParticleAlpha, maxResponse * parameters.density); // needed for the stability of the backward pass
    // FIXME : check if ValidatedHit is sufficient to not recompute everything during the backward pass
    const bool acceptHit = ValidatedHit ? true : ((maxResponse > MinParticleKernelDensity) && (alpha > MinParticleAlpha));
    if (acceptHit)
    {
        if (!RaySpreadFilterEnabled && !KnownHitDistance) {
            hitDistance = canonicalRayDistance(canonicalRayOrigin, canonicalRayDirection, parameters.scale);
        }

        if (enableNormal)
        {
            normal = canonicalRayNormal(canonicalRayOrigin, canonicalRayDirection, parameters.scale, parameters.rotationT);
        }
    }
    return acceptHit;
}

[BackwardDifferentiable][ForceInline]
float integrateHit<let BackToFront : bool, let Sampling : bool>(
    in float alpha,
    inout float transmittance,
    in float hitDistance,
    inout float integratedhitDistance,
    no_diff bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal)
{
   const float weight = BackToFront || Sampling ? alpha : alpha * transmittance;

   if (BackToFront)
   {
       integratedhitDistance = lerp(integratedhitDistance, hitDistance, alpha);
       if (enableNormal)
       {
           integratedNormal = lerp(integratedNormal, normal, alpha);
       }
   }
   else 
   {
       integratedhitDistance += hitDistance * weight;
       if (enableNormal) 
       {
            integratedNormal += normal * weight;
       }
   }

   transmittance = Sampling ? transmittance - alpha : transmittance * (1.f - alpha);

   return weight; 
}

[BackwardDifferentiable] [ForceInline]
float processHitFromBuffer<let BackToFront : bool, let ValidatedHit : bool>(
    float3 rayOrigin,
    float3 rayDirection,
    no_diff float raySpread,
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    inout float transmittance,
    inout float integratedhitDistance,
    no_diff bool enableNormal,
    inout float3 integratedNormal)
{
    float alpha = 0.0f;
    float hitDistance;
    float3 normal;
    const bool acceptHit = hit<ValidatedHit, false>(
        rayOrigin,
        rayDirection,
        raySpread,
        fetchParameters(particleIdx, parametersBuffer),
        alpha,
        hitDistance,
        enableNormal,
        normal);
    if (acceptHit)
    {
        return integrateHit<BackToFront, false>(alpha, 
                                                transmittance, 
                                                hitDistance, 
                                                integratedhitDistance, 
                                                enableNormal, 
                                                normal, 
                                                integratedNormal);
    }
    return 0.0f;
}

[BackwardDifferentiable] [ForceInline]
float processSampleHitFromBuffer<let ValidatedHit : bool>(
    float3 rayOrigin,
    float3 rayDirection,
    no_diff float raySpread,
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    no_diff float sampleWeight,
    inout float transmittance,
    inout float integratedhitDistance,
    no_diff bool enableNormal,
    inout float3 integratedNormal)
{
    float alpha = 0.0f;
    float hitDistance;
    float3 normal;
    const bool acceptHit = hit<ValidatedHit, false>(
        rayOrigin,
        rayDirection,
        raySpread,
        fetchParameters(particleIdx, parametersBuffer),
        alpha,
        hitDistance,
        enableNormal,
        normal);
    if (acceptHit)
    {
        return integrateHit<false, true>(alpha * sampleWeight,
                                         transmittance,
                                         hitDistance,
                                         integratedhitDistance,
                                         enableNormal,
                                         normal,
                                         integratedNormal);
    }
    return 0.0f;
}

[BackwardDifferentiable] [ForceInline]
float processHitFromParameters<let BackToFront : bool, let ValidatedHit: bool>(
    float3 rayOrigin,
    float3 rayDirection,
    no_diff float raySpread,
    no_diff uint32_t particleIdx,
    inout gaussianParticle.Parameters parameters,
    inout float transmittance,
    inout float integratedhitDistance,
    no_diff bool enableNormal,
    inout float3 integratedNormal)
{
    float alpha = 0.0f;
    float hitDistance;
    float3 normal;
    const bool acceptHit = hit<ValidatedHit, false>(rayOrigin,
                                                    rayDirection,
                                                    raySpread,
                                                    parameters,
                                                    alpha,
                                                    hitDistance,
                                                    enableNormal,
                                                    normal);
    if (acceptHit)
    {
        return integrateHit<BackToFront, false>(alpha,
                                                transmittance,
                                                hitDistance,
                                                integratedhitDistance,
                                                enableNormal,
                                                normal,
                                                integratedNormal);
    }
    return 0.0f;
}

[BackwardDifferentiable][ForceInline] float3x3 computeCovariance(
    in Parameters parameters) {
    // Σ = RSS^TR^T
    const float3x3 StRt = float3x3(parameters.scale.x * parameters.rotationT[0],
                                  parameters.scale.y * parameters.rotationT[1],
                                  parameters.scale.z * parameters.rotationT[2]);
    return mul(transpose(StRt), StRt);
}

[BackwardDifferentiable] [ForceInline]
float3 incidentDirectionFromParameters(
    Parameters parameters,
    no_diff float3 sourcePosition
)
{
    return utils.safe_normalize(parameters.position - sourcePosition);
}

[BackwardDifferentiable][ForceInline]
float3 incidentDirectionFromBuffer(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    no_diff float3 sourcePosition
)
{
    return incidentDirectionFromParameters(
        fetchParameters(particleIdx, parametersBuffer),
        sourcePosition
    );
}

[BackwardDifferentiable][ForceInline]
bool canonicalRayHitDistance(
    float3 canonicalRayOrigin,
    float3 canonicalUnormalizedRayDirection,
    float minHitDistance,
    float maxHitDistance,
    out float hitDistance
)
{
    const float numerator = -dot(canonicalRayOrigin, canonicalUnormalizedRayDirection);
    const float denominator = rcp(dot(canonicalUnormalizedRayDirection, canonicalUnormalizedRayDirection));
    hitDistance = numerator * denominator;
    return (hitDistance > minHitDistance) && (hitDistance < maxHitDistance);
}

[BackwardDifferentiable][ForceInline]
bool conicOpacity(
    no_diff in float covarianceDilation, 
    in float3 projectedCovariance,
    in float density,
    out float4 conicOpacity)
{
    const float3 dilatedCovariance = float3(projectedCovariance.x + covarianceDilation,
                                            projectedCovariance.y,
                                            projectedCovariance.z + covarianceDilation);
    const float dilatedCovDet = dilatedCovariance.x * dilatedCovariance.z - dilatedCovariance.y * dilatedCovariance.y;
    if (dilatedCovDet == 0.0f) {
        return false;
    }
    conicOpacity.xyz = float3(dilatedCovariance.z, -dilatedCovariance.y, dilatedCovariance.x) / dilatedCovDet;

    // see Yu et al. in "Mip-Splatting: Alias-free 3D Gaussian Splatting" https://github.com/autonomousvision/mip-splatting
    const float covDet = projectedCovariance.x * projectedCovariance.z - projectedCovariance.y * projectedCovariance.y;
    const float convolutionFactor = sqrt(covDet / dilatedCovDet);
    conicOpacity.w = density * convolutionFactor;

    return conicOpacity.w >= MinParticleAlpha;
}

[BackwardDifferentiable][ForceInline]
bool conicHit(
    no_diff float2 rayProjectedPosition,
    in float2 projectedCenter,
    in float4 projectedConicOpacity,
    out float alpha)
{
    const float2 centeredPos = rayProjectedPosition - projectedCenter;
    const float power = -0.5 * (projectedConicOpacity.x * centeredPos.x * centeredPos.x + projectedConicOpacity.z * centeredPos.y * centeredPos.y) -
                        projectedConicOpacity.y * centeredPos.x * centeredPos.y;
    if (power > 0.) {
        return false;
    }
    const float maxResponse = exp(power);
    alpha = min(MaxParticleAlpha, maxResponse * projectedConicOpacity.w); // needed for the stability of the backward pass
    return ((maxResponse > MinParticleKernelDensity) && (alpha > MinParticleAlpha));
}

[BackwardDifferentiable][ForceInline]
float processHitFromConic<let BackToFront : bool>(
    no_diff float2 rayProjectedPosition,
    in float2 projectedCenter,
    in float4 projectedConicOpacity,
    in float hitDistance,
    inout float transmittance,
    inout float integratedhitDistance)
{
    float alpha = 0.0f;
    if (conicHit(rayProjectedPosition, projectedCenter, projectedConicOpacity, alpha))
    {
        float3 normal, integratedNormal;
        return integrateHit<BackToFront, false>(alpha,
                                                transmittance,
                                                hitDistance,
                                                integratedhitDistance,
                                                false,
                                                normal,
                                                integratedNormal);
    }
    return 0.0f;
}

[BackwardDifferentiable] [ForceInline]
bool perspectiveConicProjectionFromParameters(
    in gaussianParticle.Parameters parameters,
    no_diff in float2 resolution,
    no_diff in float2 nearFarClipDistances,
    no_diff in float2 focalLength,
    no_diff in float2 principalPoint,
    no_diff in float4x3 sensorViewT,
    out float2 projectedCenter,
    out float3 projectedCovariance,
    out float projectedOpacity) {

    if (parameters.density < MinParticleAlpha) {
        return false;
    }

    float2x2 projectedCovarianceMatrix;
    if (!cameras.perspectiveGaussianCovarianceProjection(resolution,
                                                         nearFarClipDistances,
                                                         focalLength,
                                                         principalPoint,
                                                         parameters.position,
                                                         gaussianParticle.computeCovariance(parameters),
                                                         sensorViewT,
                                                         projectedCenter,
                                                         projectedCovarianceMatrix)) {
        return false;
    }

    projectedOpacity = parameters.density;

    projectedCovariance.x = projectedCovarianceMatrix[0][0];
    projectedCovariance.y = projectedCovarianceMatrix[0][1];
    projectedCovariance.z = projectedCovarianceMatrix[1][1];

    return true;
}

[BackwardDifferentiable][ForceInline]
bool perspectiveConicProjectionFromBuffer(
    no_diff in uint32_t particleIdx,
    no_diff in RawParametersBuffer parametersBuffer,
    no_diff in float2 resolution,
    no_diff in float2 nearFarClipDistances,
    no_diff in float2 focalLength,
    no_diff in float2 principalPoint,
    no_diff in float4x3 sensorViewT,
    no_diff in float3 sensorPosition,
    no_diff in float covarianceDilation,
    out float2 projectedCenter,
    out float4 projectedConicOpacity,
    out float hitDistance
)
{
    Parameters parameters = fetchParameters(particleIdx, parametersBuffer);
    
    float opacity;  
    float3 projectedCovariance;
    if (!perspectiveConicProjectionFromParameters(parameters,
                                                  resolution,
                                                  nearFarClipDistances,
                                                  focalLength,
                                                  principalPoint,
                                                  sensorViewT,
                                                  projectedCenter,
                                                  projectedCovariance,
                                                  opacity)) {
        return false;
    }

    if (!conicOpacity(covarianceDilation, projectedCovariance, opacity, projectedConicOpacity)) {
        return false;
    }

    hitDistance = length(incidentDirectionFromParameters(parameters, sensorPosition));

    return true;
    
}

} // namespace gaussianParticle

// ------------------------------------------------------------------------------------------------------------------
// Entry points

[CudaDeviceExport]
inline gaussianParticle.Parameters particleDensityParametersFromRaw(
    in gaussianParticle.RawParameters rawParameters)
{
    return gaussianParticle.parametersFromRaw(rawParameters);
}

[CudaDeviceExport]
inline gaussianParticle.RawParameters particleDensityFetchParametersRaw(
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters)
{
    return gaussianParticle.fetchParametersFromBuffer(particleIdx, 
                                                      commonParameters.parametersBuffer);
}

[CudaDeviceExport]
inline void particleDensityFetchParametersRawBwd(
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    gaussianParticle.RawParameters rawParameters)
{
    gaussianParticle.fetchParametersFromBufferBwd(particleIdx, 
                                                  commonParameters.parametersBuffer, 
                                                  rawParameters);
}

[CudaDeviceExport] 
inline gaussianParticle.Parameters particleDensityParameters(
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters) 
{
    return gaussianParticle.fetchParameters(
        particleIdx,
    commonParameters.parametersBuffer);
}

[CudaDeviceExport]
inline float particleCanonicalScale(gaussianParticle.Parameters parameters, bool densityScaleClamping)
{
    return gaussianParticle.canonicalScale<gaussianParticle.KernelDegree>(
        gaussianParticle.KernelDegree, // TODO : make this a parameter
        parameters.density,
        densityScaleClamping
    );
}

[CudaDeviceExport]
inline float3 particleFromCanonical(gaussianParticle.Parameters parameters, in float3 canonicalPoint)
{
    return parameters.position + mul(parameters.scale * canonicalPoint, parameters.rotationT);
}

[CudaDeviceExport]
inline bool particleDensityHit(
    float3 rayOrigin,
    float3 rayDirection,
    float raySpread,
    gaussianParticle.Parameters parameters,
    out float alpha,
    out float hitDistance,
    bool enableNormal,
    out float3 normal)
{
    return gaussianParticle.hit<false, false>(rayOrigin,
                                              rayDirection,
                                              raySpread,
                                              parameters,
                                              alpha,
                                              hitDistance,
                                              enableNormal,
                                              normal);
}

[CudaDeviceExport]
inline bool particleDensityHitAlpha(
    float3 rayOrigin,
    float3 rayDirection,
    float raySpread,
    gaussianParticle.Parameters parameters,
    out float alpha,
    inout float hitDistance,
    bool enableNormal,
    out float3 normal)
{
    return gaussianParticle.hit<false, true>(rayOrigin,
                                             rayDirection,
                                             raySpread,
                                             parameters,
                                             alpha,
                                             hitDistance,
                                             enableNormal,
                                             normal);
}

[CudaDeviceExport]
inline bool particleDensityHitNormal(
    float3 rayOrigin,
    float3 rayDirection,
    float raySpread,
    gaussianParticle.Parameters parameters,
    in float alpha,
    in float hitDistance,
    bool enableNormal,
    out float3 normal)
{
    return gaussianParticle.hit<true, true>(rayOrigin,
                                             rayDirection,
                                             raySpread,
                                             parameters,
                                             alpha,
                                             hitDistance,
                                             enableNormal,
                                             normal);
}

[CudaDeviceExport] inline float particleDensityIntegrateHit(
    in float alpha,
    inout float transmittance,
    in float hitDistance,
    inout float integratedhitDistance,
    in bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal) 
{
    return gaussianParticle.integrateHit<false, false>(
        alpha,
        transmittance,
        hitDistance,
        integratedhitDistance,
        enableNormal,
        normal,
        integratedNormal);
}

[CudaDeviceExport] inline float particleDensityIntegrateSampleHit(
    in float alpha,
    inout float transmittance,
    in float hitDistance,
    inout float integratedhitDistance,
    in bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal)
{
    return gaussianParticle.integrateHit<false, true>(
        alpha,
        transmittance,
        hitDistance,
        integratedhitDistance,
        enableNormal,
        normal,
        integratedNormal);
}

[CudaDeviceExport]
inline float particleDensityProcessHitFwdFromBuffer(
    float3 rayOrigin,
    float3 rayDirection,
    float raySpread,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    inout float transmittance,
    inout float integratedhitDistance,
    in bool enableNormal,
    inout float3 integratedNormal) 
{
    return gaussianParticle.processHitFromBuffer<false, false>(
        rayOrigin,
        rayDirection,
        raySpread,
        particleIdx,
        commonParameters.parametersBuffer,
        transmittance,
        integratedhitDistance,
        enableNormal,
        integratedNormal);
}

[CudaDeviceExport]
void particleDensityProcessHitBwdToBuffer(
    float3 rayOrigin,
    float3 rayDirection,
    bool enableRayGradients,
    inout float3 rayOriginGrad,
    inout float3 rayDirectionGrad,
    float raySpread,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    in float alpha,
    in float alphaGrad,
    inout float transmittance,
    inout float transmittanceGrad,
    in float hitDistance,
    inout float integratedhitDistance,
    inout float integratedhitDistanceGrad,
    bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal,
    inout float3 integratedNormalGrad) 
{
    if (alpha > 0.0f)
    {
        const float weight = 1.0f / (1.0f - alpha);
        
        DifferentialPair<float3> rayOriginDiff = DifferentialPair<float3>(rayOrigin, float3(0.f));
        DifferentialPair<float3> rayDirectionDiff = DifferentialPair<float3>(rayDirection, float3(0.f));

        transmittance *= weight;
        DifferentialPair<float> transmittanceDiff = DifferentialPair<float>(transmittance, transmittanceGrad);
        
        integratedhitDistance = (integratedhitDistance - hitDistance * alpha) * weight;
        DifferentialPair<float> integratedhitDistanceDiff = DifferentialPair<float>(integratedhitDistance, integratedhitDistanceGrad);

        DifferentialPair<float3> integratedNormalDiff;
        if (enableNormal)
        {
            integratedNormal = (integratedNormal - normal * alpha) * weight;
            integratedNormalDiff = DifferentialPair<float3>(integratedNormal, integratedNormalGrad);
        }
        else
        {
            integratedNormalDiff = DifferentialPair<float3>(float3(0), float3(0));
        }

        bwd_diff(gaussianParticle.processHitFromBuffer<true, true>)(
            rayOriginDiff,
            rayDirectionDiff,
            raySpread,
            particleIdx,
            commonParameters.parametersBuffer,
            transmittanceDiff,
            integratedhitDistanceDiff,
            enableNormal,
            integratedNormalDiff,
            alphaGrad);

        transmittanceGrad = transmittanceDiff.getDifferential();
        integratedhitDistanceGrad = integratedhitDistanceDiff.getDifferential();
        if (enableNormal)
        {
            integratedNormalGrad = integratedNormalDiff.getDifferential();
        }

        if (enableRayGradients)
        {
            rayOriginGrad += rayOriginDiff.getDifferential();
            rayDirectionGrad += rayDirectionDiff.getDifferential();
        }
    }
}

[CudaDeviceExport]
void particleDensityProcessSampleHitBwdToBuffer(
    float3 rayOrigin,
    float3 rayDirection,
    bool enableRayGradients,
    inout float3 rayOriginGrad,
    inout float3 rayDirectionGrad,
    float raySpread,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    in float alpha,
    in float alphaGrad,
    inout float transmittance,
    inout float transmittanceGrad,
    in float hitDistance,
    inout float integratedhitDistance,
    inout float integratedhitDistanceGrad,
    bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal,
    inout float3 integratedNormalGrad)
{
    if (alpha > 0.0f)
    {
        DifferentialPair<float3> rayOriginDiff = DifferentialPair<float3>(rayOrigin, float3(0.f));
        DifferentialPair<float3> rayDirectionDiff = DifferentialPair<float3>(rayDirection, float3(0.f));

        transmittance += alpha;
        DifferentialPair<float> transmittanceDiff = DifferentialPair<float>(transmittance, transmittanceGrad);

        integratedhitDistance -= hitDistance * alpha;
        DifferentialPair<float> integratedhitDistanceDiff = DifferentialPair<float>(integratedhitDistance, integratedhitDistanceGrad);

        DifferentialPair<float3> integratedNormalDiff;
        if (enableNormal)
        {
            integratedNormal -= normal * alpha;
            integratedNormalDiff = DifferentialPair<float3>(integratedNormal, integratedNormalGrad);
        }
        else
        {
            integratedNormalDiff = DifferentialPair<float3>(float3(0), float3(0));
        }

        // compute the sample weight
        float sampleAlpha = 0.0f;
        float sampleHitDistance;
        float3 sampleNormal;
        gaussianParticle.hit<true, true>(
            rayOrigin,
            rayDirection,
            raySpread,
            gaussianParticle.fetchParameters(particleIdx, commonParameters.parametersBuffer),
            sampleAlpha,
            sampleHitDistance,
            enableNormal,
            sampleNormal);

        bwd_diff(gaussianParticle.processSampleHitFromBuffer<true>)(
            rayOriginDiff,
            rayDirectionDiff,
            raySpread,
            particleIdx,
            commonParameters.parametersBuffer,
            alpha / sampleAlpha, // sample weight
            transmittanceDiff,
            integratedhitDistanceDiff,
            enableNormal,
            integratedNormalDiff,
            alphaGrad);

        transmittanceGrad = transmittanceDiff.getDifferential();
        integratedhitDistanceGrad = integratedhitDistanceDiff.getDifferential();
        if (enableNormal)
        {
            integratedNormalGrad = integratedNormalDiff.getDifferential();
        }

        if (enableRayGradients)
        {
            rayOriginGrad += rayOriginDiff.getDifferential();
            rayDirectionGrad += rayDirectionDiff.getDifferential();
        }
    }
}

[CudaDeviceExport]
void particleDensityProcessHitBwdToRawParameters(
    float3 rayOrigin,
    float3 rayDirection,
    bool enableRayGradients,
    inout float3 rayOriginGrad,
    inout float3 rayDirectionGrad,
    float raySpread,
    uint32_t particleIdx,
    in gaussianParticle.Parameters parameters,
    in float4 quaternion,
    inout gaussianParticle.RawParameters rawParametersGrad,
    in float alpha,
    in float alphaGrad,
    inout float transmittance,
    inout float transmittanceGrad,
    in float hitDistance,
    inout float integratedhitDistance,
    inout float integratedhitDistanceGrad,
    bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal,
    inout float3 integratedNormalGrad)
{
    if (alpha > 0.0f)
    {
        const float weight = 1.0f / (1.0f - alpha);

        DifferentialPair<gaussianParticle.Parameters> parametersDiff =
            DifferentialPair<gaussianParticle.Parameters>(parameters, gaussianParticle.Parameters(float3(0), float3(0), float3x3(0), 0));

        DifferentialPair<float3> rayOriginDiff = DifferentialPair<float3>(rayOrigin, float3(0.f));
        DifferentialPair<float3> rayDirectionDiff = DifferentialPair<float3>(rayDirection, float3(0.f));

        transmittance *= weight;
        DifferentialPair<float> transmittanceDiff = DifferentialPair<float>(transmittance, transmittanceGrad);

        integratedhitDistance = (integratedhitDistance - hitDistance * alpha) * weight;
        DifferentialPair<float> integratedhitDistanceDiff = DifferentialPair<float>(integratedhitDistance, integratedhitDistanceGrad);

        DifferentialPair<float3> integratedNormalDiff;
        if (enableNormal)
        {
            integratedNormal = (integratedNormal - normal * alpha) * weight;
            integratedNormalDiff = DifferentialPair<float3>(integratedNormal, integratedNormalGrad);
        }
        else
        {
            integratedNormalDiff = DifferentialPair<float3>(float3(0), float3(0));
        }

        bwd_diff(gaussianParticle.processHitFromParameters<true, true>)(
            rayOriginDiff,
            rayDirectionDiff,
            raySpread,
            particleIdx,
            parametersDiff,
            transmittanceDiff,
            integratedhitDistanceDiff,
            enableNormal,
            integratedNormalDiff,
            alphaGrad);

        transmittanceGrad = transmittanceDiff.getDifferential();
        integratedhitDistanceGrad = integratedhitDistanceDiff.getDifferential();
        if (enableNormal)
        {
            integratedNormalGrad = integratedNormalDiff.getDifferential();
        }

        if (enableRayGradients)
        {
            rayOriginGrad += rayOriginDiff.getDifferential();
            rayDirectionGrad += rayDirectionDiff.getDifferential();
        }

        DifferentialPair<gaussianParticle.RawParameters> rawParametersDiff =
            DifferentialPair<gaussianParticle.RawParameters>(
                gaussianParticle.RawParameters(parameters.position, parameters.density, quaternion, parameters.scale, 0),
                gaussianParticle.RawParameters(float3(0), float(0), float4(0), float3(0), 0)
        );

        bwd_diff(gaussianParticle.parametersFromRaw)(
            rawParametersDiff,
            parametersDiff.getDifferential());

        rawParametersGrad.position += rawParametersDiff.getDifferential().position;
        rawParametersGrad.density += rawParametersDiff.getDifferential().density;
        rawParametersGrad.quaternion += rawParametersDiff.getDifferential().quaternion;
        rawParametersGrad.scale += rawParametersDiff.getDifferential().scale;
        rawParametersGrad.padding += rawParametersDiff.getDifferential().padding;
    }
}

[CudaDeviceExport]
bool particleDensityRayHit(
    float3 rayOrigin,
    float3 rayDirection,
    int32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    float minHitDistance,
    float maxHitDistance,
    out float hitDistance,
    bool enableNormal,
    out float3 normal
)
{
    gaussianParticle.Parameters parameters = gaussianParticle.fetchParameters(particleIdx, commonParameters.parametersBuffer);

    float3 canonicalRayOrigin;
    float3 canonicalRayDirection;
    gaussianParticle.cannonicalRay(
        rayOrigin,
        rayDirection,
        parameters,
        canonicalRayOrigin,
        canonicalRayDirection);

    const float maxResponse = gaussianParticle.canonicalRayMaxKernelResponse<gaussianParticle.KernelDegree, gaussianParticle.Surfel>(
        canonicalRayOrigin,
        canonicalRayDirection);

    // distance to the gaussian center projection on the ray
    hitDistance = gaussianParticle.canonicalRayDistance(canonicalRayOrigin, canonicalRayDirection, parameters.scale);

    if (enableNormal)
    {
        normal = gaussianParticle.canonicalRayNormal(canonicalRayOrigin, canonicalRayDirection, parameters.scale, parameters.rotationT);
    }

    return (hitDistance > minHitDistance) &&
           (hitDistance < maxHitDistance) &&
           (gaussianParticle.canonicalRayMinSquaredDistance<gaussianParticle.Surfel>(
                canonicalRayOrigin, canonicalRayDirection) < (gaussianParticle.KernelScale * gaussianParticle.KernelScale));
}

[CudaDeviceExport]
bool particleDensityCanonicalRayHitDistance(
    float3 canonicalRayOrigin,
    float3 canonicalUnormalizedRayDirection,
    float minHitDistance,
    float maxHitDistance,
    out float hitDistance,
)
{
    return gaussianParticle.canonicalRayHitDistance(canonicalRayOrigin, canonicalUnormalizedRayDirection, minHitDistance, maxHitDistance, hitDistance);
}

[CudaDeviceExport]
bool particleDensityCanonicalRayHitAlpha(
    float3 canonicalRayOrigin,
    float3 canonicalUnormalizedRayDirection,
    float minHitDistance,
    float maxHitDistance,
    out float alpha,
    out float hitDistance,
    float particleDensity,
    bool densityScaleClamping
)
{
    if (gaussianParticle.canonicalRayHitDistance(canonicalRayOrigin, canonicalUnormalizedRayDirection, minHitDistance, maxHitDistance, hitDistance))
    {
        const float canonicalScale = gaussianParticle.canonicalScale<gaussianParticle.KernelDegree>(gaussianParticle.KernelDegree, // TODO : make this a parameter
                                                                                                    particleDensity,
                                                                                                    densityScaleClamping);
        const float maxResponse = gaussianParticle.canonicalRayMaxKernelResponse<gaussianParticle.KernelDegree, gaussianParticle.Surfel>(
            canonicalRayOrigin * canonicalScale, utils.safe_normalize(canonicalUnormalizedRayDirection));
        alpha = min(gaussianParticle.MaxParticleAlpha, maxResponse * particleDensity); 
        return ((maxResponse > gaussianParticle.MinParticleKernelDensity) && (alpha > gaussianParticle.MinParticleAlpha));
    }
    return false;
}

[CudaDeviceExport] float3 particleDensityIncidentDirection(
    in gaussianParticle.Parameters parameters,
    in float3 sourcePosition
)
{
    return gaussianParticle.incidentDirectionFromParameters(parameters, sourcePosition);
}

[CudaDeviceExport] void particleDensityIncidentDirectionBwdToBuffer(
    in uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    in float3 sourcePosition,
    in float3 incidentDirectionGrad
)
{
    bwd_diff(gaussianParticle.incidentDirectionFromBuffer)(
        particleIdx, 
        commonParameters.parametersBuffer, 
        sourcePosition, 
        incidentDirectionGrad
    );
}

[CudaDeviceExport] bool particleDensityPerspectiveConicProjection(
    in gaussianParticle.Parameters parameters,
    no_diff in float2 resolution,
    no_diff in float2 nearFarClipDistances,
    in float2 focalLength,
    in float2 principalPoint,
    in float4x3 sensorViewT,
    in float3 sensorPosition,
    inout float3 incidentRay,
    out float2 projectedCenter,
    out float3 projectedCovariance,
    out float projectedOpacity) {

    if (!gaussianParticle.perspectiveConicProjectionFromParameters(parameters,
                                                                   resolution,
                                                                   nearFarClipDistances,
                                                                   focalLength,
                                                                   principalPoint,
                                                                   sensorViewT,
                                                                   projectedCenter,
                                                                   projectedCovariance,
                                                                   projectedOpacity))
    {
        return false;
    }

    incidentRay = parameters.position - sensorPosition;

    return true;
}

[CudaDeviceExport] bool particleDensityConicHit(
    in float2 rayProjectedPosition,
    in float2 projectedCenter,
    in float4 projectedConicOpacity,
    out float alpha
)
{
    return gaussianParticle.conicHit(rayProjectedPosition, 
                                     projectedCenter, 
                                     projectedConicOpacity,
                                     alpha);
}

[CudaDeviceExport] void particleDensityProcessHitBwdToConic(
    in float2 rayProjectedPosition,
    in float alpha,
    in float alphaGrad,
    inout float transmittance,
    inout float transmittanceGrad,
    in float hitDistance,
    inout float hitDistanceGrad,
    inout float integratedhitDistance,
    inout float integratedhitDistanceGrad,
    in float2 projectedCenter,
    inout float2 projectedCenterGrad,
    in float4 projectedConicOpacity,
    inout float4 projectedConicOpacityGrad,
)
{
    if (alpha > 0.0f)
    {
        const float weight = 1.0f / (1.0f - alpha);

        transmittance *= weight;
        DifferentialPair<float> transmittanceDiff = DifferentialPair<float>(transmittance, transmittanceGrad);

        integratedhitDistance = (integratedhitDistance - hitDistance * alpha) * weight;
        DifferentialPair<float> integratedhitDistanceDiff = DifferentialPair<float>(integratedhitDistance, integratedhitDistanceGrad);

        DifferentialPair<float2> projectedCenterDiff = DifferentialPair<float2>(projectedCenter, float2(0.f));
        DifferentialPair<float4> projectedConicOpacityDiff = DifferentialPair<float4>(projectedConicOpacity, float4(0.f));
        DifferentialPair<float> hitDistanceDiff = DifferentialPair<float>(hitDistance, 0.f);

        bwd_diff(gaussianParticle.processHitFromConic<true>)(
            rayProjectedPosition,
            projectedCenterDiff,
            projectedConicOpacityDiff,
            hitDistanceDiff,
            transmittanceDiff,
            integratedhitDistanceDiff,
            alphaGrad);

        hitDistanceGrad += hitDistanceDiff.getDifferential();
        transmittanceGrad = transmittanceDiff.getDifferential();
        integratedhitDistanceGrad = integratedhitDistanceDiff.getDifferential();
        projectedCenterGrad += projectedCenterDiff.getDifferential();
        projectedConicOpacityGrad += projectedConicOpacityDiff.getDifferential();  
    }
}

[CudaDeviceExport] void particleDensityPerspectiveConicProjectionBwdToBuffer(
    in uint32_t particleIdx,
    in gaussianParticle.CommonParameters commonParameters,
    in float2 resolution,
    in float2 nearFarClipDistances,
    in float2 focalLength,
    in float2 principalPoint,
    in float4x3 sensorViewT,
    in float3 sensorPosition,
    in float covarianceDilation,
    in float2 projectedCenterGrad,
    in float4 projectedConicOpacityGrad,
    in float hitDistanceGrad
)
{
    bwd_diff(gaussianParticle.perspectiveConicProjectionFromBuffer)(
        particleIdx,
        commonParameters.parametersBuffer,
        resolution,
        nearFarClipDistances,
        focalLength,
        principalPoint,
        sensorViewT,
        sensorPosition,
        covarianceDilation,
        projectedCenterGrad,
        projectedConicOpacityGrad,
        hitDistanceGrad
    );
}