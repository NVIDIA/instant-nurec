// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/models/nreModel.h>
#include <nrend/models/nreNeRFModel.h>
#include <nrend/models/nreTraceableCompositeModel.h>
#include <nrend/renderer/nreRenderer.h>

bool nrend::NRERenderer::supportVersion(const ModelVersion& version,
                                        RenderingParameters::RendererHints /*rendererHint*/,
                                        RenderingParameters::OptFlags renderFlags) const {
    return !(renderFlags & RenderingParameters::OptDifferentiable) &&
           version.is("nre") &&
           (version.isInstance(NRENeRFModel::name) || version.isInstance(NRETraceableCompositeModel::name)) &&
           (version.number() >= minVersionNumber) &&
           (version.number() < maxVersionNumber);
}

nrend::NRERenderer::NRERenderer(const nlohmann::json& rendererState, const Logger& logger)
    : NRendererImplementation(rendererState, logger) {
}

nrend::NRERenderer::~NRERenderer() {
}

nrend::Status nrend::NRERenderer::initialize(const ModelVersion& version,
                                             const nlohmann::json& modelState,
                                             const RenderingParameters& renderParams) {
    if (!supportVersion(version, renderParams.rendererHint, renderParams.opts)) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "NRERenderer : unsupported model version %s.", version.str().c_str());
    }

    if (renderParams.opts & RenderingParameters::OptDifferentiable) {
        RETURN_ERROR(m_logger, ErrorCode::NotImplemented, "NRERenderer : differentiable model not implemented.");
    }

    m_modelVersion = version;
    m_optFlags     = static_cast<KernelOpts>((renderParams.opts & RenderingParameters::OptDifferentiable ? KernelOpts::Differentiable : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptLinearRGB ? KernelOpts::LinearRGB : KernelOpts::None));

    if (!modelState.contains("nre_data")) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "NRERenderer : cannot create renderer from JSON : no nre_data header.");
    }

    if (modelState["nre_data"].contains("config") && modelState["nre_data"].contains("state_dict")) {
        m_modelPtr.reset(
            NREModel::createFromJSON(modelState["nre_data"]["config"], m_logger, modelState["nre_data"]["state_dict"], "."));
        if (!m_modelPtr) {
            RETURN_ERROR(m_logger, ErrorCode::BadInput, "NRERenderer : cannot create renderer from JSON.");
        } else if (renderParams.trackInstancesStrUIds.size) {
            m_modelPtr->initializeTrackInstances(renderParams.trackInstancesStrUIds, m_logger);
        }
    } else {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "NRERenderer : cannot create renderer from JSON : no config or state_dict entries.");
    }

    return Status();
}

nrend::Status nrend::NRERenderer::registerKernelDefinitions(
    const KernelMemoryBindings& memoryBindings,
    const KernelSourceCodeTable& sourceCodeTable,
    const KernelDefinitionsTable& kernelDefinitionsTable,
    KernelOpts kernelOpts,
    const Logger& logger) const {

    if (kernelOpts != m_optFlags) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "NRERenderer : inconsistent differentiable state.");
    }

    Status status;
    if (m_modelPtr) {
        status = m_modelPtr->registerKernelResources(memoryBindings, sourceCodeTable, kernelOpts, logger);
    }
    if (!status) {
        return status;
    }

    const std::string sourceCodeTemplate = R"(
        static constexpr bool SRGBModel = {SRGBModel};
        static constexpr bool SRGBOutput = {SRGBOutput};

        using TNREModel = {NRENeRFClassAlias};
        
        #include <nrend/kernels/cuda/renderers/nreRenderer.cuh>
    )";

    m_renderKernelIndex = kernelDefinitionsTable.registerKernel({KernelDefinition::CudaKernel,
                                                                 CudaKernelOptions{{"render"}},
                                                                 fmt::format(sourceCodeTemplate,
                                                                             fmt::arg("SRGBModel", true),
                                                                             fmt::arg("NRENeRFClassAlias", m_modelPtr->cudaCallPrefix()),
                                                                             fmt::arg("SRGBOutput", !(m_optFlags & KernelOpts::LinearRGB)))});

    return status;
}

nrend::Status nrend::NRERenderer::processKernelMemory(
    const KernelMemoryBindings& memoryBindings,
    KernelMemoryBindings::BindingsFlag bindingsFlag,
    const std::vector<std::unique_ptr<KernelMemory>>& memory,
    ProcessMemoryFlag processFlag,
    uint64_t processQueueHandle,
    const Logger& logger) const {

    Status status;
    if (m_modelPtr) {
        status = m_modelPtr->processKernelMemory(memoryBindings, bindingsFlag, memory, processFlag, processQueueHandle, logger);
    }
    return status;
}

nrend::Status nrend::NRERenderer::renderForward(const RenderParameters& params,
                                                const tcnn::vec3* wordlRayOriginCudaPtr,
                                                const tcnn::vec3* worldRayDirectionCudaPtr,
                                                const TTimestamp* worldRayTimestampCudaPtr,
                                                const tcnn::ivec2* sensorsIdsPtr,
                                                const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
                                                const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                                                const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                                uint32_t* instanceIdCudaPtr,
                                                float* worldHitDistanceCudaPtr,
                                                tcnn::vec3* worldHitNormalCudaPtr,
                                                tcnn::vec4* radianceDensityCudaPtr,
                                                void* /*extendedFeaturesCudaPtr*/,
                                                void* /*sceneDataCudaPtr*/,
                                                ForwardContext** forwardContext,
                                                int cudaDeviceIndex,
                                                cudaStream_t cudaStream) const {

    if (forwardContext) {
        *forwardContext = nullptr;
    }

    auto cudaResource = m_cudaKernelResources.prepare(this, m_optFlags, cudaStream, cudaDeviceIndex, m_logger);
    if (!cudaResource) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "NRERenderer : cannot get cuda resource on the device %d.", cudaDeviceIndex);
    }

    if (worldHitNormalCudaPtr) {
        LOG_WARN(m_logger, "NRERenderer : does not support world hit normal output.");
    }

    const dim3 threads = {8, 16, 1};
    const dim3 blocks  = {tcnn::div_round_up((uint32_t)params.frameTileResolution.x, threads.x),
                          tcnn::div_round_up((uint32_t)params.frameTileResolution.y, threads.y), 1};

    return cudaResource->launchCudaKernel(m_renderKernelIndex, 0,
                                          blocks, threads, 0, cudaStream, m_logger,
                                          params,
                                          wordlRayOriginCudaPtr,
                                          worldRayDirectionCudaPtr,
                                          worldRayTimestampCudaPtr,
                                          sensorsIdsPtr,
                                          instanceIdCudaPtr,
                                          worldHitDistanceCudaPtr,
                                          worldHitNormalCudaPtr,
                                          radianceDensityCudaPtr,
                                          activeTrackInstancesIdsCudaPtr,
                                          activeTrackInstancesPoseCudaPtr,
                                          activeTrackInstancesEndPoseCudaPtr,
                                          cudaResource->memoryHandlesPtr(KernelMemoryBindings::BindingsFlag::Parameters));
}
