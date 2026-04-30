// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/kernels/cuda/primitives/particleASPrimitives.cuh>
#include <nrend/models/nreParticlesModel.h>
#include <nrend/renderer/grtOptixRenderParameters.h>
#include <nrend/renderer/grtOptixRenderer.h>
#include <nrend/utils/deviceLaunchesLogger.h>
using namespace tcnn;

namespace {

using namespace nrend;

// TODO : per-stream n context cache
struct GRTOptixRenderForwardContext final : public NRenderer::ForwardContext {

    ScopedCudaBuffer parameterMemoryHandles;
    std::vector<KernelBindedTransientMemory> transientParameters;

    OptixAccelerationStructure gas;
    OptixAccelerationStructure unitAABB;

    ScopedCudaBuffer worldHitDistanceBounds;

    uint32_t numParticles = 0u;

    // clang-format off
    GRTOptixRenderForwardContext(int cudaDeviceIndex, cudaStream_t cudaStream, const Logger& logger)
        : parameterMemoryHandles(reinterpret_cast<uint64_t>(cudaStream)),
          gas(cudaDeviceIndex, cudaStream, logger),
          unitAABB(cudaDeviceIndex, cudaStream, logger),
          worldHitDistanceBounds(reinterpret_cast<uint64_t>(cudaStream)) {}
    // clang-format on
    ~GRTOptixRenderForwardContext() = default;

    inline Status updateParameterMemoryHandlesBuffer(
        const KernelMemoryPtrVec& parameterMemoryPtrVec,
        const Logger& logger) {
        std::vector<uint64_t> parameterMemoryHandlesVec(parameterMemoryPtrVec.size());
        std::transform(parameterMemoryPtrVec.begin(), parameterMemoryPtrVec.end(), parameterMemoryHandlesVec.begin(),
                       [](const KernelMemoryPtr& ptr) { return ptr ? ptr->handle() : 0; });
        for (const auto& transientParameter : transientParameters) {
            RETURN_ERROR_IF((transientParameter.memoryBindingIndex == KernelMemoryBindings::InvalidMemoryIndex) || !transientParameter.memory,
                            logger,
                            ErrorCode::BadInput,
                            "GRTOptixRenderer : cannot update parameter memory handles buffer, transient parameter memory binding index is invalid.");
            parameterMemoryHandlesVec[transientParameter.memoryBindingIndex] = transientParameter.memory->handle();
        }
        return parameterMemoryHandles.setFromHost(parameterMemoryHandlesVec.data(), parameterMemoryHandlesVec.size() * sizeof(uint64_t), logger);
    }
};

} // namespace

nrend::GRTOptixRenderer::GRTOptixRenderer(const nlohmann::json& rendererState, const Logger& logger, bool defaultSettings)
    : GRUTRenderer(rendererState, logger) {
    if (!defaultSettings) {
        initializeSettings(rendererState, logger);
    }
}

void nrend::GRTOptixRenderer::initializeSettings(const nlohmann::json& rendererState, const Logger& logger) {
    if (!rendererState.empty()) {
        initializeOutputSettings(rendererState, logger);
        if (rendererState.contains("pipeline")) {
            const auto& pipelineSettingsConfig = rendererState["pipeline"];
            const std::string pipelineTypeStr  = pipelineSettingsConfig.value<std::string>("type", "reference");
            m_settings.pipelineType            = pipelineTypeFromStr(pipelineTypeStr);
            LOG_DEBUG(m_logger, "GRTOptixRenderer : pipeline type : %s", pipelineTypeStr.c_str());
            const std::string pipelineTypeFwdStr = pipelineSettingsConfig.value<std::string>("fwd_type", "reference");
            m_settings.pipelineTypeFwd           = pipelineTypeFromStr(pipelineTypeFwdStr);
            LOG_DEBUG(m_logger, "GRTOptixRenderer : pipeline type fwd : %s", pipelineTypeFwdStr.c_str());
            const std::string pipelineTypeBwdStr = pipelineSettingsConfig.value<std::string>("bwd_type", "reference");
            m_settings.pipelineTypeBwd           = pipelineTypeFromStr(pipelineTypeBwdStr);
            LOG_DEBUG(m_logger, "GRTOptixRenderer : pipeline type bwd : %s", pipelineTypeBwdStr.c_str());
            m_settings.pipelineKBufferSize = pipelineSettingsConfig.value("k_buffer_size", m_settings.pipelineKBufferSize);
            LOG_DEBUG(m_logger, "GRTOptixRenderer : pipeline k buffer size : %u", m_settings.pipelineKBufferSize);
            m_settings.pipelineNumSamples = pipelineSettingsConfig.value("num_samples", m_settings.pipelineNumSamples);
            LOG_DEBUG(m_logger, "GRTOptixRenderer : pipeline num samples : %u", m_settings.pipelineNumSamples);
        }
        if (rendererState.contains("primitives")) {
            const auto& primitiveSettingsConfig      = rendererState["primitives"];
            m_settings.primitiveType                 = primitiveTypeFromStr(primitiveSettingsConfig.value<std::string>("type", ""));
            m_settings.primitiveDensityScaleClamping = primitiveSettingsConfig.value("density_scale", m_settings.primitiveDensityScaleClamping);
        }
        if (rendererState.contains("culling")) {
            const auto& cullingSettingsConfig = rendererState["culling"];
            m_settings.nearClipDistance       = cullingSettingsConfig.value("near_clip_distance", m_settings.nearClipDistance);

            // Load separate camera and lidar far clip distances with backward compatibility
            if (cullingSettingsConfig.contains("far_clip_distance_camera") || cullingSettingsConfig.contains("far_clip_distance_lidar")) {
                m_settings.farClipDistanceCamera = cullingSettingsConfig.value("far_clip_distance_camera", m_settings.farClipDistanceCamera);
                m_settings.farClipDistanceLidar  = cullingSettingsConfig.value("far_clip_distance_lidar", m_settings.farClipDistanceLidar);
            } else if (cullingSettingsConfig.contains("far_clip_distance")) {
                // Backward compatibility: use the same value for both camera and lidar
                float farClipDistance            = cullingSettingsConfig.value("far_clip_distance", std::numeric_limits<float>::max());
                m_settings.farClipDistanceCamera = farClipDistance;
                m_settings.farClipDistanceLidar  = farClipDistance;
            }
        }
    }
    m_particlesPrimitiveInfos = particlePrimitiveInfos(m_settings.primitiveType);
}

nrend::GRTOptixRenderer::~GRTOptixRenderer() {
}

nrend::GRTOptixRenderer::ParticlesPrimitiveInfos nrend::GRTOptixRenderer::particlePrimitiveInfos(Settings::PrimitiveType primitiveType) {

    ParticlesPrimitiveInfos infos;

    switch (primitiveType) {
    case Settings::Tetrahedra:
        infos.type                    = OptixAccelerationStructure::InputType::TriangleMesh;
        infos.dataSizePerParticles    = TetrahedraParticlePrimitive<>::NumVertices * sizeof(float) * 3;
        infos.extDataSizePerParticles = TetrahedraParticlePrimitive<>::NumFaces * sizeof(uint32_t) * 3;
        infos.numIndicesPerParticles  = TetrahedraParticlePrimitive<>::NumFaces;
        break;
    case Settings::Diamond:
        infos.type                    = OptixAccelerationStructure::InputType::TriangleMesh;
        infos.dataSizePerParticles    = DiamondParticlePrimitive<>::NumVertices * sizeof(float) * 3;
        infos.extDataSizePerParticles = DiamondParticlePrimitive<>::NumFaces * sizeof(uint32_t) * 3;
        infos.numIndicesPerParticles  = DiamondParticlePrimitive<>::NumFaces;
        break;
    case Settings::Octahedron:
        infos.type                    = OptixAccelerationStructure::InputType::TriangleMesh;
        infos.dataSizePerParticles    = OctahedronParticlePrimitive<>::NumVertices * sizeof(float) * 3;
        infos.extDataSizePerParticles = OctahedronParticlePrimitive<>::NumFaces * sizeof(uint32_t) * 3;
        infos.numIndicesPerParticles  = OctahedronParticlePrimitive<>::NumFaces;
        break;
    case Settings::Icosahedron:
        infos.type                    = OptixAccelerationStructure::InputType::TriangleMesh;
        infos.dataSizePerParticles    = IcosahedronParticlePrimitive<>::NumVertices * sizeof(float) * 3;
        infos.extDataSizePerParticles = IcosahedronParticlePrimitive<>::NumFaces * sizeof(uint32_t) * 3;
        infos.numIndicesPerParticles  = IcosahedronParticlePrimitive<>::NumFaces;
        break;
    case Settings::Aabb:
        infos.type                    = OptixAccelerationStructure::InputType::Custom;
        infos.dataSizePerParticles    = sizeof(OptixAabb);
        infos.extDataSizePerParticles = 0;
        infos.numIndicesPerParticles  = 1;
        break;
    case Settings::Trisurfel:
        infos.type                    = OptixAccelerationStructure::InputType::TriangleMesh;
        infos.dataSizePerParticles    = TrisurfelParticlePrimitive<>::NumVertices * sizeof(float) * 3;
        infos.extDataSizePerParticles = TrisurfelParticlePrimitive<>::NumFaces * sizeof(uint32_t) * 3;
        infos.numIndicesPerParticles  = TrisurfelParticlePrimitive<>::NumFaces;
        break;
    case Settings::Rhombus:
        infos.type                    = OptixAccelerationStructure::InputType::TriangleMesh;
        infos.dataSizePerParticles    = RhombusParticlePrimitive<>::NumVertices * sizeof(float) * 3;
        infos.extDataSizePerParticles = RhombusParticlePrimitive<>::NumFaces * sizeof(uint32_t) * 3;
        infos.numIndicesPerParticles  = RhombusParticlePrimitive<>::NumFaces;
        break;
    case Settings::Sphere:
        infos.type                    = OptixAccelerationStructure::InputType::Sphere;
        infos.dataSizePerParticles    = sizeof(float) * 3; // center
        infos.extDataSizePerParticles = sizeof(float);     // radius
        infos.numIndicesPerParticles  = 1;
        break;
    default: // Settings::TransformedAabb:
        infos.type                    = OptixAccelerationStructure::InputType::Instance;
        infos.dataSizePerParticles    = sizeof(OptixInstance);
        infos.extDataSizePerParticles = 0;
        infos.numIndicesPerParticles  = 1;
        break;
    }

    return infos;
}

std::string nrend::GRTOptixRenderer::particlePrimitiveDefinition(Settings::PrimitiveType primitiveType,
                                                                 bool densityScaleClamping) {

    std::string particlePrimitiveClassAlias;
    switch (primitiveType) {
    case Settings::Tetrahedra:
        particlePrimitiveClassAlias = "nrend::TetrahedraParticlePrimitive<TGRTParticlePrimitiveParams>";
        break;
    case Settings::Diamond:
        particlePrimitiveClassAlias = "nrend::DiamondParticlePrimitive<TGRTParticlePrimitiveParams>";
        break;
    case Settings::Octahedron:
        particlePrimitiveClassAlias = "nrend::OctahedronParticlePrimitive<TGRTParticlePrimitiveParams>";
        break;
    case Settings::Icosahedron:
        particlePrimitiveClassAlias = "nrend::IcosahedronParticlePrimitive<TGRTParticlePrimitiveParams>";
        break;
    case Settings::Aabb:
        particlePrimitiveClassAlias = "nrend::AabbParticlePrimitive<TGRTParticlePrimitiveParams>";
        break;
    case Settings::Trisurfel:
        particlePrimitiveClassAlias = "nrend::TrisurfelParticlePrimitive<TGRTParticlePrimitiveParams>";
        break;
    case Settings::Rhombus:
        particlePrimitiveClassAlias = "nrend::RhombusParticlePrimitive<TGRTParticlePrimitiveParams>";
        break;
    case Settings::Sphere:
        particlePrimitiveClassAlias = "nrend::SphereParticlePrimitive<TGRTParticlePrimitiveParams>";
        break;
    default: // Settings::TransformedAabb:
        particlePrimitiveClassAlias = "nrend::TransformedAabbParticlePrimitive<TGRTParticlePrimitiveParams, OPTIX_INSTANCE_FLAG_NONE>";
        break;
    }

    return fmt::format(R"(
        #include <nrend/kernels/cuda/primitives/particleASPrimitives.cuh>

        struct TGRTParticlePrimitiveParams {{
            static constexpr bool DensityScaleClamping = {DensityScaleClamping};
        }};

        using TGRTParticlePrimitive = {GRTParticlePrimitiveClassAlias};
    )",
                       fmt::arg("GRTParticlePrimitiveClassAlias", particlePrimitiveClassAlias),
                       fmt::arg("DensityScaleClamping", densityScaleClamping));
}

nrend::Status nrend::GRTOptixRenderer::setupOptixPipelineTracer(bool backward,
                                                                bool lidar,
                                                                OptixPipelineOptions& optixPipelineOptions,
                                                                std::string& optixPipelineTracerSourceCode,
                                                                KernelOpts kernelOpts,
                                                                const Logger& logger) const {
    optixPipelineOptions.raygenEntryPointName = "__raygen__rg";
    // AnyHit is needed when the KHitBufferSize is not empty.
    // (Note that in rejectionSampling the KHitBuffer is populated in the intersection shader)
    if ((m_settings.pipelineKBufferSize >= 1) && (m_settings.pipelineType != Settings::RejectionSampling)) {
        optixPipelineOptions.anyHitEntryPointName = "__anyhit__ah";
    }
    if (m_settings.primitiveType == Settings::Aabb || m_settings.primitiveType == Settings::TransformedAabb) {
        optixPipelineOptions.intersectionEntryPointName = "__intersection__is";
    }
    optixPipelineOptions.parametersVariableName = "pipelineParams";

    optixPipelineOptions.flags = OptixPipelineOptions::Flags::None;
    if (m_settings.primitiveType == Settings::TransformedAabb) {
        optixPipelineOptions.flags |= OptixPipelineOptions::Flags::AllowSingleLevelInstancing;
    } else {
        optixPipelineOptions.flags |= OptixPipelineOptions::Flags::AllowSingleGas;
    }
    if (m_settings.primitiveType == Settings::Aabb || m_settings.primitiveType == Settings::TransformedAabb) {
        optixPipelineOptions.flags |= OptixPipelineOptions::Flags::EnableCustomPrimitives;
    } else if (m_settings.primitiveType == Settings::Sphere) {
        optixPipelineOptions.flags |= OptixPipelineOptions::Flags::EnableSpherePrimitives;
    } else {
        optixPipelineOptions.flags |= OptixPipelineOptions::Flags::EnableTrianglePrimitives;
    }

    std::string optixPipelineTracerDefinition;
    const Settings::PipelineType pipelineType = backward ? m_settings.pipelineTypeBwd : (m_optFlags & KernelOpts::Differentiable) ? m_settings.pipelineTypeFwd
                                                                                                                                  : m_settings.pipelineType;
    if ((pipelineType == Settings::Reference) || (pipelineType == Settings::ReferenceInstance) || (pipelineType == Settings::RejectionSampling)) {

        // pipeline specific kernel definitions
        std::string pipelineTracerStruct;
        std::string pipelineKernelDefinitionFile;
        switch (pipelineType) {
        case Settings::Reference:
            pipelineTracerStruct         = "GRTReferenceOptixTracer";
            pipelineKernelDefinitionFile = "referenceOptixTracer";
            // reference pipeline use 2 value per-k-buffer entry : the particle index and the hit distance
            optixPipelineOptions.numPayloadValues   = m_settings.pipelineKBufferSize * 2;
            optixPipelineOptions.numAttributeValues = 0;
            break;
        case Settings::ReferenceInstance:
            pipelineTracerStruct         = "GRTReferenceInstanceOptixTracer";
            pipelineKernelDefinitionFile = "referenceInstanceOptixTracer";
            // reference instance pipeline use 3 value per-k-buffer entry : the particle index, the hit distance and the computed alpha
            optixPipelineOptions.numPayloadValues = m_settings.pipelineKBufferSize * 3;
            // reference instance pipeline use the first attribute value to pass the computed hit point alpha
            optixPipelineOptions.numAttributeValues = 1;
            break;
        case Settings::RejectionSampling:
            pipelineTracerStruct                    = "GRTRejectionSamplingOptixTracer";
            pipelineKernelDefinitionFile            = "rejectionSamplingOptixTracer";
            optixPipelineOptions.numPayloadValues   = m_settings.pipelineKBufferSize * 2 + 1;
            optixPipelineOptions.numAttributeValues = 0;
            break;
        default:
            RETURN_ERROR(logger, ErrorCode::BadInput, "GRTOptixRenderer : invalid pipeline type.");
        }

        const uint32_t optixTraceRayFlags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                                            (m_settings.pipelineKBufferSize < 1 ? OPTIX_RAY_FLAG_DISABLE_ANYHIT : OPTIX_RAY_FLAG_NONE) |
                                            (m_settings.pipelineType == Settings::RejectionSampling ? OPTIX_RAY_FLAG_DISABLE_ANYHIT : OPTIX_RAY_FLAG_NONE) |
                                            (m_settings.primitiveType == Settings::Trisurfel ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES);

        const OutputSettings& outputSettings    = lidar ? m_lidarOutputSettings : m_cameraOutputSettings;
        const bool enableFeatures               = !(kernelOpts & KernelOpts::DisableFeatures) && outputSettings.enableFeatures;
        const bool enableExtendedFeatures       = !(kernelOpts & KernelOpts::DisableExtendedFeatures) && outputSettings.enableExtendedFeatures;
        const bool enableSensorExtendedFeatures = !(kernelOpts & KernelOpts::DisableSensorExtendedFeatures) && outputSettings.enableSensorExtendedFeatures;
        const bool enableNormals                = !(kernelOpts & KernelOpts::DisableNormals) && outputSettings.enableNormals;
        const bool enableRayGradients           = !(kernelOpts & KernelOpts::DisableRayGradients) && outputSettings.enableRayGradients;

        std::string optixPipelineTracerDefinitionTemplate = R"(
            struct TGRTReferenceOptixTracerParams
            {{
                static constexpr int KHitBufferSize             = {KHitBufferSize};
                static constexpr int NumSamples                 = {NumSamples};
                static constexpr bool InstancePrimitive         = {InstancePrimitive};
                static constexpr bool InstanceIdAsOpacity       = {InstanceIdAsOpacity};
                static constexpr bool DensityScaleClamping      = {DensityScaleClamping};
                static constexpr uint32_t IndicesPerPrimitive   = {IndicesPerPrimitive};
                static constexpr uint32_t OptixTraceRayFlags    = {OptixTraceRayFlags};
                static constexpr float NearDistance             = {NearDistance};
                static constexpr float FarDistance              = {FarDistance};
                static constexpr uint32_t SceneDataDim           = {SceneDataDim};
                static constexpr int32_t SceneDataWeightsOffset = {SceneDataWeightsOffset};
            }};

            #define GRTReferenceOptixTracer_KHitBufferSize {KHitBufferSize}

            #include <nrend/kernels/optix/tracers/{PipelineKernelDefinitionFile}.cuh>

            using TGRTTracer = {PipelineTracerStruct}<{GRTModelClassAlias}::Particles, 
                                                       TGRTReferenceOptixTracerParams, 
                                                       {EnableFeatures}, 
                                                       {EnableExtendedFeatures}, 
                                                       {EnableCameraExtendedFeatures}, 
                                                       {EnableLidarExtendedFeatures}, 
                                                       {EnableNormals}, 
                                                       {EnableRayGradients}, 
                                                       {Backward}>;
        )";
        optixPipelineTracerDefinition =
            fmt::format(optixPipelineTracerDefinitionTemplate,
                        fmt::arg("KHitBufferSize", m_settings.pipelineKBufferSize),
                        fmt::arg("NumSamples", m_settings.pipelineNumSamples),
                        fmt::arg("InstancePrimitive", m_settings.primitiveType == Settings::TransformedAabb),
                        fmt::arg("InstanceIdAsOpacity", true), // Debug : set to false to check the effect of the quantization
                        fmt::arg("DensityScaleClamping", m_settings.primitiveDensityScaleClamping),
                        fmt::arg("IndicesPerPrimitive", m_particlesPrimitiveInfos.numIndicesPerParticles),
                        fmt::arg("OptixTraceRayFlags", optixTraceRayFlags),
                        fmt::arg("NearDistance", m_settings.nearClipDistance),
                        fmt::arg("FarDistance", lidar ? m_settings.farClipDistanceLidar : m_settings.farClipDistanceCamera),
                        fmt::arg("SceneDataDim", m_renderingSceneDataLayout.count()),
                        fmt::arg("SceneDataWeightsOffset", m_renderingSceneDataLayout.cumulatedWeights.offset),
                        fmt::arg("GRTModelClassAlias", m_modelPtr->cudaCallPrefix()),
                        fmt::arg("EnableFeatures", enableFeatures),
                        fmt::arg("EnableExtendedFeatures", enableExtendedFeatures),
                        fmt::arg("EnableCameraExtendedFeatures", lidar ? false : enableSensorExtendedFeatures),
                        fmt::arg("EnableLidarExtendedFeatures", lidar ? enableSensorExtendedFeatures : false),
                        fmt::arg("EnableNormals", enableNormals),
                        fmt::arg("EnableRayGradients", enableRayGradients),
                        fmt::arg("Backward", backward),
                        fmt::arg("PipelineTracerStruct", pipelineTracerStruct),
                        fmt::arg("PipelineKernelDefinitionFile", pipelineKernelDefinitionFile));
    } else {
        RETURN_ERROR(logger, ErrorCode::BadInput, "GRTOptixRenderer : invalid pipeline type.");
    }

    const std::string optixPipelineSourceCodeTemplate = R"(
            using TGRTModel = {GRTModelClassAlias};

            {GRTTracerDefinition};

            #include <nrend/renderer/grtOptixRenderParameters.h>

            extern "C" {{
            __constant__ {TParametersType} {ParametersVariableName};
            }};

            static constexpr bool SRGBModel = {SRGBModel};
            static constexpr bool SRGBOutput = {SRGBOutput};
            static constexpr bool Differentiable = {Differentiable};
            #define TGRTTracer_Backward {Backward}

            #include <nrend/kernels/optix/pipelines/grtOptixPipeline.cuh>            
        )";

    optixPipelineTracerSourceCode =
        fmt::format(optixPipelineSourceCodeTemplate,
                    fmt::arg("GRTModelClassAlias", m_modelPtr->cudaCallPrefix()),
                    fmt::arg("GRTTracerDefinition", optixPipelineTracerDefinition),
                    fmt::arg("TParametersType", backward ? "nrend::GRTOptixRenderBackwardParameters" : "nrend::GRTOptixRenderParameters"),
                    fmt::arg("ParametersVariableName", optixPipelineOptions.parametersVariableName),
                    fmt::arg("SRGBModel", true),
                    fmt::arg("SRGBOutput", !(m_optFlags & KernelOpts::LinearRGB)),
                    fmt::arg("Differentiable", m_optFlags & KernelOpts::Differentiable),
                    fmt::arg("Backward", backward));

    return Status();
}

nrend::Status nrend::GRTOptixRenderer::registerKernelDefinitions(
    const KernelMemoryBindings& memoryBindings,
    const KernelSourceCodeTable& sourceCodeTable,
    const KernelDefinitionsTable& kernelDefinitionsTable,
    KernelOpts kernelOpts,
    const Logger& logger) const {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRTOptixRenderer : cannot register resource, not initialized.");
    }

    RETURN_ERROR_IF(m_settings.pipelineType >= Settings::NumPipelineTypes, m_logger, ErrorCode::BadInput, "GRTOptixRenderer : pipeline type is not set.");
    RETURN_ERROR_IF(m_settings.pipelineTypeFwd >= Settings::NumPipelineTypes, m_logger, ErrorCode::BadInput, "GRTOptixRenderer : forward pipeline type is not set.");
    RETURN_ERROR_IF(m_settings.pipelineTypeBwd >= Settings::NumPipelineTypes, m_logger, ErrorCode::BadInput, "GRTOptixRenderer : backward pipeline type is not set.");
    RETURN_ERROR_IF(m_settings.primitiveType >= Settings::NumPrimitiveTypes, m_logger, ErrorCode::BadInput, "GRTOptixRenderer : primitive type is not set.");

    if (kernelOpts != m_optFlags) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GRTOptixRenderer : inconsistent render options.");
    }

    if (!m_modelPtr) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRTOptixRenderer : cannot register resource, uninitialized model.");
    }
    CHECK_STATUS_RETURN(m_modelPtr->registerKernelResources(memoryBindings, sourceCodeTable, kernelOpts, logger));

    // static acceleration structure is only supported for static models at inference time
    INREParticlesModel* particlesModelPtr = dynamic_cast<INREParticlesModel*>(m_modelPtr.get());
    m_staticAS                            = !(kernelOpts & KernelOpts::Differentiable) && particlesModelPtr && !particlesModelPtr->isDynamic();

    // generate cuda kernels
    {
        const std::string kernelSourceCodeTemplate = R"(
    
        using TGRTModel = {GRTModelClassAlias};
        
        {GRTParticlePrimitiveDefinition}

        #include <nrend/kernels/cuda/renderers/grtOptixRenderer.cuh>
        
        )";

        std::vector<const char*> entryPointNames(NumKernelEntryPoints);
        entryPointNames[PreProcessParticlesKernelEntryPoint]     = "preProcessParticles";
        entryPointNames[BuildParticlesPrimitiveKernelEntryPoint] = "buildParticlePrimitives";

        m_kernelIndex = kernelDefinitionsTable.registerKernel({KernelDefinition::CudaKernel,
                                                               CudaKernelOptions{entryPointNames},
                                                               fmt::format(kernelSourceCodeTemplate,
                                                                           fmt::arg("GRTParticlePrimitiveDefinition",
                                                                                    particlePrimitiveDefinition(m_settings.primitiveType, m_settings.primitiveDensityScaleClamping)),
                                                                           fmt::arg("GRTModelClassAlias", m_modelPtr->cudaCallPrefix()))});
    }

    // generate optix forward pipeline
    {
        OptixPipelineOptions optixPipelineOptions;
        std::string optixPipelineTracerSourceCode;
        CHECK_STATUS_RETURN(setupOptixPipelineTracer(
            false,
            false,
            optixPipelineOptions,
            optixPipelineTracerSourceCode,
            kernelOpts,
            logger));

        m_renderPipelineIndex = kernelDefinitionsTable.registerKernel({KernelDefinition::OptixPipeline,
                                                                       optixPipelineOptions,
                                                                       optixPipelineTracerSourceCode});
    }

    // generate optix backward pipeline
    if (m_optFlags & KernelOpts::Differentiable) {

        OptixPipelineOptions optixPipelineOptions;
        std::string optixPipelineTracerSourceCode;
        CHECK_STATUS_RETURN(setupOptixPipelineTracer(
            true,
            false,
            optixPipelineOptions,
            optixPipelineTracerSourceCode,
            kernelOpts,
            logger));

        m_renderPipelineBwdIndex = kernelDefinitionsTable.registerKernel({KernelDefinition::OptixPipeline,
                                                                          optixPipelineOptions,
                                                                          optixPipelineTracerSourceCode});
    }

    // generate optix forward lidar pipeline
    {
        OptixPipelineOptions optixPipelineOptions;
        std::string optixPipelineTracerSourceCode;
        CHECK_STATUS_RETURN(setupOptixPipelineTracer(
            false,
            true,
            optixPipelineOptions,
            optixPipelineTracerSourceCode,
            kernelOpts,
            logger));

        m_renderLidarPipelineIndex = kernelDefinitionsTable.registerKernel({KernelDefinition::OptixPipeline,
                                                                            optixPipelineOptions,
                                                                            optixPipelineTracerSourceCode});
    }

    // generate optix backward lidar pipeline
    if (m_optFlags & KernelOpts::Differentiable) {

        OptixPipelineOptions optixPipelineOptions;
        std::string optixPipelineTracerSourceCode;
        CHECK_STATUS_RETURN(setupOptixPipelineTracer(
            true,
            true,
            optixPipelineOptions,
            optixPipelineTracerSourceCode,
            kernelOpts,
            logger));

        m_renderLidarPipelineBwdIndex = kernelDefinitionsTable.registerKernel({KernelDefinition::OptixPipeline,
                                                                               optixPipelineOptions,
                                                                               optixPipelineTracerSourceCode});
    }

    return Status();
}

nrend::Status nrend::GRTOptixRenderer::renderForward(const RenderParameters& params,
                                                     const vec3* wordlRayOriginCudaPtr,
                                                     const vec3* worldRayDirectionCudaPtr,
                                                     const TTimestamp* worldRayTimestampCudaPtr,
                                                     const ivec2* sensorsIdsPtr,
                                                     const ivec2* activeTrackInstancesIdsCudaPtr,
                                                     const TTrackInstancePose* activeTrackInstancesStartPoseCudaPtr,
                                                     const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                                     uint32_t* instanceIdCudaPtr,
                                                     float* worldHitDistanceCudaPtr,
                                                     vec3* worldHitNormalCudaPtr,
                                                     vec4* radianceDensityCudaPtr,
                                                     void* extendedFeaturesCudaPtr,
                                                     void* sceneDataCudaPtr,
                                                     ForwardContext** forwardContext,
                                                     int cudaDeviceIndex,
                                                     cudaStream_t cudaStream) const {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRTOptixRenderer : cannot render forward, not initialized.");
    }

    if (forwardContext) {
        *forwardContext = nullptr;
    }

    // prepare the cuda resources required for rendering
    auto cudaResource = m_cudaKernelResources->prepare(this, m_optFlags, cudaStream, cudaDeviceIndex, m_logger);
    if (!cudaResource) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRTOptixRenderer : cannot get cuda resource on the device %d.", cudaDeviceIndex);
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("render");

    // FIXME : resources concurrent management
    // if (cudaResource->resourcesUseCount() > 2) {
    //     RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot render, cuda resource on device %d is already in use.", cudaDeviceIndex);
    // }
    std::unique_ptr<GRTOptixRenderForwardContext> grtForwardContextPtr = std::make_unique<GRTOptixRenderForwardContext>(cudaDeviceIndex, cudaStream, m_logger);

    INREParticlesModel* particlesModelPtr = dynamic_cast<INREParticlesModel*>(m_modelPtr.get());
    RETURN_ERROR_IF(!particlesModelPtr, m_logger, ErrorCode::InvalidResource, "GRTOptixRenderer : cannot render, model is not an INREParticlesModel.");

    const KernelMemoryPtrVec* parameterMemoryPtrVec = cudaResource->memoryPtrVec(KernelMemoryBindings::BindingsFlag::Parameters);
    if (!parameterMemoryPtrVec) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource,
                     "GRTOptixRenderer : cannot render, no parameter memory on device %d.", cudaDeviceIndex);
    }

    uint32_t numParticlesToPreProcess = 0;
    {
        const auto prepapreProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/prepare-particles"};
        CHECK_STATUS_RETURN(particlesModelPtr->prepareParticlesParameters(params.numActiveTrackInstances,
                                                                          activeTrackInstancesIdsCudaPtr,
                                                                          *parameterMemoryPtrVec,
                                                                          grtForwardContextPtr->transientParameters,
                                                                          grtForwardContextPtr->numParticles,
                                                                          numParticlesToPreProcess,
                                                                          cudaStream,
                                                                          m_logger));
    }
    if (grtForwardContextPtr->numParticles == 0) {
        LOG_WARN(m_logger, "GRTOptixRenderer : no particles to render.");
        return Status();
    }

    CHECK_STATUS_RETURN(grtForwardContextPtr->updateParameterMemoryHandlesBuffer(*parameterMemoryPtrVec, m_logger));

    if (numParticlesToPreProcess > 0) {

        const auto preprocessProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/preprocess-particles"};

        CHECK_STATUS_RETURN(cudaResource->launchCudaKernel(
            m_kernelIndex,
            PreProcessParticlesKernelEntryPoint,
            div_round_up<int>(numParticlesToPreProcess, 512), 512, 0, cudaStream, m_logger,
            numParticlesToPreProcess,
            params.numActiveTrackInstances,
            activeTrackInstancesIdsCudaPtr,
            params.sensorState.startTimestamp + (params.sensorState.endTimestamp - params.sensorState.startTimestamp) / 2,
            activeTrackInstancesStartPoseCudaPtr,
            activeTrackInstancesEndPoseCudaPtr,
            grtForwardContextPtr->parameterMemoryHandles.data()));
    }

    const uint32_t numParticles = grtForwardContextPtr->numParticles;
    // build the acceleration structure
    if (!m_staticAS || (m_staticPerDevicesAS.size() <= static_cast<size_t>(cudaDeviceIndex)) || !m_staticPerDevicesAS[cudaDeviceIndex]) {

        // create the static acceleration structures for the device
        if (m_staticAS) {
            m_staticPerDevicesAS.resize(cudaDeviceIndex + 1);
            m_staticPerDevicesAS[cudaDeviceIndex] = std::make_unique<std::vector<OptixAccelerationStructure>>();
            m_staticPerDevicesAS[cudaDeviceIndex]->reserve(2);
            m_staticPerDevicesAS[cudaDeviceIndex]->emplace_back(cudaDeviceIndex, cudaStream, m_logger);
            m_staticPerDevicesAS[cudaDeviceIndex]->emplace_back(cudaDeviceIndex, cudaStream, m_logger);
        }

        const auto buildASProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/build-as"};

        ScopedCudaBuffer particlesPrimitiveDataBuffer(reinterpret_cast<uint64_t>(cudaStream));
        ScopedCudaBuffer particlesPrimitiveExtDataBuffer(reinterpret_cast<uint64_t>(cudaStream));

        {
            const auto buildASPrimitivesProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/build-as/primitives"};

            CHECK_STATUS_RETURN(particlesPrimitiveDataBuffer.enlarge(numParticles * m_particlesPrimitiveInfos.dataSizePerParticles, m_logger));

            if (m_particlesPrimitiveInfos.extDataSizePerParticles > 0) {
                CHECK_STATUS_RETURN(particlesPrimitiveExtDataBuffer.enlarge(numParticles * m_particlesPrimitiveInfos.extDataSizePerParticles, m_logger));
            }

            OptixTraversableHandle asHandle = 0;
            if (m_settings.primitiveType == Settings::TransformedAabb) {
                CHECK_STATUS_RETURN(OptixAccelerationStructure::buildUnitAABBInstanceAS(m_staticAS ? m_staticPerDevicesAS[cudaDeviceIndex]->at(1) : grtForwardContextPtr->unitAABB, m_logger));
                asHandle = m_staticAS ? m_staticPerDevicesAS[cudaDeviceIndex]->at(1).handle : grtForwardContextPtr->unitAABB.handle;
            }

            CHECK_STATUS_RETURN(cudaResource->launchCudaKernel(
                m_kernelIndex,
                BuildParticlesPrimitiveKernelEntryPoint,
                div_round_up<int>(numParticles, 1024), 1024, 0, cudaStream, m_logger,
                numParticles,
                particlesPrimitiveDataBuffer.data(),
                particlesPrimitiveExtDataBuffer.data(),
                asHandle,
                grtForwardContextPtr->parameterMemoryHandles.data()));
        }

        {
            const auto buildASBuildProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/build-as/build"};

            CHECK_STATUS_RETURN(OptixAccelerationStructure::buildAS(m_staticAS ? m_staticPerDevicesAS[cudaDeviceIndex]->front() : grtForwardContextPtr->gas,
                                                                    m_particlesPrimitiveInfos.type,
                                                                    m_particlesPrimitiveInfos.dataSizePerParticles * numParticles,
                                                                    reinterpret_cast<CUdeviceptr>(particlesPrimitiveDataBuffer.data()),
                                                                    m_particlesPrimitiveInfos.extDataSizePerParticles * numParticles,
                                                                    reinterpret_cast<CUdeviceptr>(particlesPrimitiveExtDataBuffer.data()),
                                                                    m_particlesPrimitiveInfos.buildASFlag,
                                                                    m_logger));
        }
    }

    {
        const auto renderProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/render"};

        if (m_optFlags & KernelOpts::Differentiable) {
            CHECK_STATUS_RETURN(grtForwardContextPtr->worldHitDistanceBounds.enlarge(
                params.frameTileResolution.x * params.frameTileResolution.y * sizeof(tcnn::vec2), m_logger));
        }

        GRTOptixRenderParameters hostRenderParameters;
        hostRenderParameters.traversableHandle         = m_staticAS ? m_staticPerDevicesAS[cudaDeviceIndex]->front().handle : grtForwardContextPtr->gas.handle;
        hostRenderParameters.renderParametersArray     = *reinterpret_cast<const RenderParametersArray*>(&params);
        hostRenderParameters.wordlRayOriginPtr         = wordlRayOriginCudaPtr;
        hostRenderParameters.worldRayDirectionPtr      = worldRayDirectionCudaPtr;
        hostRenderParameters.worldRayTimestampCudaPtr  = worldRayTimestampCudaPtr;
        hostRenderParameters.sensorsIdsPtr             = sensorsIdsPtr;
        hostRenderParameters.instanceIdPtr             = instanceIdCudaPtr;
        hostRenderParameters.worldHitDistancePtr       = worldHitDistanceCudaPtr;
        hostRenderParameters.worldHitDistanceBoundsPtr = grtForwardContextPtr->worldHitDistanceBounds.ptr<tcnn::vec2>();
        hostRenderParameters.worldHitNormalPtr         = worldHitNormalCudaPtr;
        hostRenderParameters.radianceDensityPtr        = radianceDensityCudaPtr;
        hostRenderParameters.extendedFeaturesPtr       = extendedFeaturesCudaPtr;
        hostRenderParameters.sceneDataPtr              = sceneDataCudaPtr;
        hostRenderParameters.parameterMemoryHandles    = grtForwardContextPtr->parameterMemoryHandles.ptr<uint64_t>();

        ScopedCudaBuffer deviceRenderParameters(reinterpret_cast<uint64_t>(cudaStream));
        CHECK_STATUS_RETURN(deviceRenderParameters.setFromHost(reinterpret_cast<const void*>(&hostRenderParameters),
                                                               sizeof(GRTOptixRenderParameters), m_logger));

        // clang-format off
        CHECK_STATUS_RETURN(cudaResource->launchOptixPipeline(
            sensorIsLidar(params.sensorModel) ? m_renderLidarPipelineIndex : m_renderPipelineIndex,
            dim3{static_cast<uint32_t>(params.frameTileResolution.x), static_cast<uint32_t>(params.frameTileResolution.y), 1u},
            cudaStream,
            m_logger,
            deviceRenderParameters.ptr<GRTOptixRenderParameters>()));
        // clang-format on
    }

    // setup the context for backward pass
    if (forwardContext && (m_optFlags & KernelOpts::Differentiable)) {
        *forwardContext = dynamic_cast<ForwardContext*>(grtForwardContextPtr.release());
    }

    return Status();
}

nrend::Status nrend::GRTOptixRenderer::renderBackward(const RenderParameters& params,
                                                      const vec3* wordlRayOriginCudaPtr,
                                                      const vec3* worldRayDirectionCudaPtr,
                                                      const nrend::TTimestamp* worldRayTimestampCudaPtr,
                                                      const ivec2* sensorsIdsPtr,
                                                      const ivec2* activeTrackInstancesIdsCudaPtr,
                                                      const TTrackInstancePose* activeTrackInstancesStartPoseCudaPtr,
                                                      const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                                      uint32_t* instanceIdCudaPtr,
                                                      const float* worldHitDistanceCudaPtr,
                                                      const float* worldHitDistanceGradientCudaPtr,
                                                      const vec3* worldHitNormalCudaPtr,
                                                      const vec3* worldHitNormalGradientCudaPtr,
                                                      const vec4* radianceDensityCudaPtr,
                                                      const vec4* radianceDensityGradientCudaPtr,
                                                      const void* extendedFeaturesCudaPtr,
                                                      const void* extendedFeaturesGradientCudaPtr,
                                                      vec3* wordlRayOriginGradientCudaPtr,
                                                      vec3* worldRayDirectionGradientCudaPtr,
                                                      ForwardContext* forwardContextPtr,
                                                      int cudaDeviceIndex,
                                                      cudaStream_t cudaStream) const {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRTOptixRenderer : cannot render backward, not initialized.");
    }

    if ((!(m_optFlags & KernelOpts::Differentiable))) {
        RETURN_ERROR(m_logger, ErrorCode::Runtime, "GRTOptixRenderer : cannot call renderBackward, not initialized as differentiable.");
    }

    auto cudaResource = m_cudaKernelResources->prepare(this, m_optFlags, cudaStream, cudaDeviceIndex, m_logger);
    if (!cudaResource) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRTOptixRenderer : cannot get cuda resource on the device %d.", cudaDeviceIndex);
    }
    // FIXME : resources concurrent management
    // if (cudaResource->resourcesUseCount() > 3) {
    //     RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot render backward, cuda resource on device %d, already in use.", cudaDeviceIndex);
    // }
    const GRTOptixRenderForwardContext* grtForwardContextPtr = dynamic_cast<const GRTOptixRenderForwardContext*>(forwardContextPtr);
    if (!grtForwardContextPtr || (grtForwardContextPtr->parameterMemoryHandles.processQueueHandle() != reinterpret_cast<uint64_t>(cudaStream))) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GRTOptixRenderer : cannot render backward, invalid forward context on device %d.", cudaDeviceIndex);
    }

    const uint32_t numParticles = grtForwardContextPtr->numParticles;
    if (numParticles == 0) {
        LOG_WARN(m_logger, "GRTOptixRenderer : not backpropagating, no particles.");
        return Status();
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("render-backward");

    {
        const auto renderBackwardProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render-backward/render"};

        GRTOptixRenderBackwardParameters hostRenderParameters;
        hostRenderParameters.traversableHandle              = m_staticAS ? m_staticPerDevicesAS[cudaDeviceIndex]->front().handle : grtForwardContextPtr->gas.handle;
        hostRenderParameters.renderParametersArray          = *reinterpret_cast<const RenderParametersArray*>(&params);
        hostRenderParameters.wordlRayOriginPtr              = wordlRayOriginCudaPtr;
        hostRenderParameters.worldRayDirectionPtr           = worldRayDirectionCudaPtr;
        hostRenderParameters.worldRayTimestampCudaPtr       = worldRayTimestampCudaPtr;
        hostRenderParameters.sensorsIdsPtr                  = sensorsIdsPtr;
        hostRenderParameters.instanceIdPtr                  = instanceIdCudaPtr;
        hostRenderParameters.worldHitDistancePtr            = worldHitDistanceCudaPtr;
        hostRenderParameters.worldHitDistanceBoundsPtr      = grtForwardContextPtr->worldHitDistanceBounds.ptr<tcnn::vec2>();
        hostRenderParameters.worldHitNormalPtr              = worldHitNormalCudaPtr;
        hostRenderParameters.radianceDensityPtr             = radianceDensityCudaPtr;
        hostRenderParameters.extendedFeaturesPtr            = extendedFeaturesCudaPtr;
        hostRenderParameters.parameterMemoryHandles         = grtForwardContextPtr->parameterMemoryHandles.ptr<uint64_t>();
        hostRenderParameters.worldHitDistanceGradientPtr    = worldHitDistanceGradientCudaPtr;
        hostRenderParameters.worldHitNormalGradientPtr      = worldHitNormalGradientCudaPtr;
        hostRenderParameters.radianceDensityGradientPtr     = radianceDensityGradientCudaPtr;
        hostRenderParameters.extendedFeaturesGradientPtr    = extendedFeaturesGradientCudaPtr;
        hostRenderParameters.wordlRayOriginGradientPtr      = wordlRayOriginGradientCudaPtr;
        hostRenderParameters.worldRayDirectionGradientPtr   = worldRayDirectionGradientCudaPtr;
        hostRenderParameters.parameterGradientMemoryHandles = cudaResource->memoryHandlesPtr(KernelMemoryBindings::BindingsFlag::ParameterGradients);

        ScopedCudaBuffer deviceRenderParameters(reinterpret_cast<uint64_t>(cudaStream));
        CHECK_STATUS_RETURN(deviceRenderParameters.setFromHost(reinterpret_cast<const void*>(&hostRenderParameters),
                                                               sizeof(GRTOptixRenderBackwardParameters), m_logger));

        // clang-format off
        CHECK_STATUS_RETURN(cudaResource->launchOptixPipeline(
            sensorIsLidar(params.sensorModel) ? m_renderLidarPipelineBwdIndex : m_renderPipelineBwdIndex,
            dim3{static_cast<uint32_t>(params.frameTileResolution.x), static_cast<uint32_t>(params.frameTileResolution.y), 1u},
            cudaStream,
            m_logger,
            deviceRenderParameters.ptr<GRTOptixRenderBackwardParameters>()));
        // clang-format on
    }

    return Status();
}
