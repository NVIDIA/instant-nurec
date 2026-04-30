// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <json/json.hpp>
#include <nrend/kernelResources/rtcKernelConfig.h>
#include <nrend/models/skyNetwork.h>
#include <nrend/renderer/ngpOccupancyGrid.cuh>
#include <nrend/renderer/nreLegacyRenderer.h>
#include <nrend/utils/cuda/cudaCommon.h>
#include <nrend/utils/cuda/cudaRtcKernel.h>
#include <nrend/utils/spinMutex.h>

#include <neural-graphics-primitives/nerf_network.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/encodings/spherical_harmonics.h>
#include <tiny-cuda-nn/reduce_sum.h>
#include <tiny-cuda-nn/vec_json.h>

using namespace nrend;

namespace {

struct NRENeRF final {

    using NerfNetwork = ngp::NerfNetwork<tcnn::network_precision_t>;
    // network data
    uint32_t nPosDims;
    uint32_t nDirDims;
    uint32_t nExtraDims;
    nlohmann::json encodingConfig;
    nlohmann::json dirEncodingConfig;
    nlohmann::json networkConfig;
    nlohmann::json rgbNetworkConfig;
    bool linear_rgb = false; // true if the output of the rgb network is in linear space
    float per_level_scale;
    uint32_t base_grid_resolution;
    uint32_t n_features_per_level;
    uint32_t n_levels;
    float density_scale = 1.0f;

    // Sky-MLP
    using SkyNetwork = nrend::SkyNetwork<tcnn::network_precision_t>;
    bool has_sky_mlp = false;
    nlohmann::json skyEncodingConfig;
    nlohmann::json skyNetworkConfig;

    // options are mutable since they maybe switched off in prepare/execute
    mutable uint32_t optFlags = RenderingParameters::OptDefault;
    nrend::Logger logger;

    tcnn::BoundingBox render_aabb;
    int aabb_scale            = 1;
    float scale               = 1.0f;
    tcnn::vec3 offset         = {0.0f, 0.0f, 0.0f};
    float cone_angle_constant = 0.0f;
    float min_step_size       = -1.0f;

    // network parameters
    std::vector<char> params_buffer_host;
    NGPDensityGrid densityGrid;
    std::vector<char> sky_params_buffer_host;

    mutable SpinMutex buffersMutex;

    // Per-device readonly buffers used to store invariant model informations
    struct DeviceStaticBuffers {
        std::atomic<bool> initialized;
        tcnn::GPUMemory<char> params_buffer;
        tcnn::GPUMemory<uint8_t> occupancyGridDeviceBuffers;
        std::unique_ptr<NerfNetwork> network;
        std::unique_ptr<SkyNetwork> skyNetwork;
        tcnn::GPUMemory<char> sky_params_buffer;
        std::unique_ptr<nrend::CudaRtcKernel> fusedRenderKernel;

        DeviceStaticBuffers()
            : initialized(false) {
        }
        DeviceStaticBuffers(DeviceStaticBuffers&& other)
            : DeviceStaticBuffers() {
            other.~DeviceStaticBuffers();
        }
    };
    mutable std::vector<DeviceStaticBuffers> deviceStaticBuffers;
};

bool prepareForDevice(const NRENeRF* nrengPtr, uint32_t cudaDeviceIndex) {
    if (!nrengPtr) {
        return false;
    }

    std::unique_lock<SpinMutex> lock(nrengPtr->buffersMutex);
    NRENeRF::DeviceStaticBuffers& deviceStaticBuffers = nrengPtr->deviceStaticBuffers[cudaDeviceIndex];
    if (deviceStaticBuffers.initialized) {
        return true;
    }

    CudaCheckDeviceGuard cudaDeviceGuard(cudaDeviceIndex);
    if (!cudaDeviceGuard.check()) {
        LOG_ERROR(nrengPtr->logger, "NRE INGP Engine : cannot set device index %d.", cudaDeviceIndex);
        return false;
    }

    try {
        deviceStaticBuffers.network.reset(new NRENeRF::NerfNetwork(
            nrengPtr->nPosDims, nrengPtr->nDirDims, nrengPtr->nExtraDims, nrengPtr->nPosDims,
            ///< variable of NerfCoordinate. HACKY
            nrengPtr->encodingConfig, nrengPtr->dirEncodingConfig, nrengPtr->networkConfig, nrengPtr->rgbNetworkConfig));
        if (nrengPtr->has_sky_mlp) {
            deviceStaticBuffers.skyNetwork.reset(new NRENeRF::SkyNetwork(
                3, nrengPtr->nExtraDims, 3, nrengPtr->skyEncodingConfig, nrengPtr->skyNetworkConfig));
        }
    } catch (const std::exception& e) {
        LOG_WARN(nrengPtr->logger, "NRE INGP Engine : cannot create the network model : %s", e.what());
        return false;
    }

    cudaStream_t deviceStream = 0;
    if (deviceStaticBuffers.params_buffer.size() == 0) {
        const size_t n_params = deviceStaticBuffers.network->n_params();
        if (n_params != (nrengPtr->params_buffer_host.size() / sizeof(tcnn::network_precision_t))) {
            LOG_ERROR(nrengPtr->logger,
                      "NRE INGP Engine : loaded NeRF model has an invalid number of parameters. (%zu x %zu != %zu)",
                      n_params, sizeof(tcnn::network_precision_t), nrengPtr->params_buffer_host.size());
            return false;
        }
        deviceStaticBuffers.params_buffer.resize(sizeof(tcnn::network_precision_t) * n_params);
        CUDA_CHECK(nrengPtr->logger,
                   cudaMemcpyAsync(deviceStaticBuffers.params_buffer.data(), nrengPtr->params_buffer_host.data(),
                                   sizeof(tcnn::network_precision_t) * n_params, cudaMemcpyHostToDevice, deviceStream));
        deviceStaticBuffers.network->set_params(
            nullptr, reinterpret_cast<tcnn::network_precision_t*>(deviceStaticBuffers.params_buffer.data()), nullptr);
    }
    if (deviceStaticBuffers.occupancyGridDeviceBuffers.size() == 0) {
        deviceStaticBuffers.occupancyGridDeviceBuffers.enlarge_and_copy_from_host(nrengPtr->densityGrid.density_grid_host);
    }
    if (nrengPtr->has_sky_mlp && (deviceStaticBuffers.sky_params_buffer.size() == 0)) {
        const size_t n_params = deviceStaticBuffers.skyNetwork->n_params();
        if (n_params != (nrengPtr->sky_params_buffer_host.size() / sizeof(tcnn::network_precision_t))) {
            LOG_ERROR(nrengPtr->logger,
                      "NREnd : loaded sky background model has an invalid number of parameters. (%zu x %zu != %zu)",
                      n_params, sizeof(tcnn::network_precision_t),
                      nrengPtr->sky_params_buffer_host.size());
            return false;
        }
        deviceStaticBuffers.sky_params_buffer.resize(sizeof(tcnn::network_precision_t) * n_params);
        CUDA_CHECK(nrengPtr->logger,
                   cudaMemcpyAsync(deviceStaticBuffers.sky_params_buffer.data(), nrengPtr->sky_params_buffer_host.data(),
                                   sizeof(tcnn::network_precision_t) * n_params, cudaMemcpyHostToDevice, deviceStream));
        deviceStaticBuffers.skyNetwork->set_params(
            nullptr, reinterpret_cast<tcnn::network_precision_t*>(deviceStaticBuffers.sky_params_buffer.data()), nullptr);
    }

    const std::string codeTemplate = R"(
        {NeRFModelBody}

        static constexpr bool ApplyBackgroundModel = {ApplyBackgroundModel};

        {BackgroundModeBody}

        static constexpr bool SRGBModel = {SRGBModel};
        static constexpr bool SRGBOutput = {SRGBOutput};
        static constexpr uint32_t N_EXTRA_DIMS = {N_EXTRA_DIMS};
        static constexpr uint32_t DensityGridMaxCascade = {DensityGridMaxCascade};
        static constexpr float M_AABB_SCALE = {M_AABB_SCALE};
        static constexpr float M_CONE_ANGLE = {M_CONE_ANGLE};
        static constexpr float M_STEP_SIZE = {M_STEP_SIZE};
        static constexpr float densityScale = {densityScale};

        #include <nrend/kernels/cuda/renderers/nreDefault.cuh>
    )";

    const std::string kernelStr = fmt::format(
        codeTemplate,
        fmt::arg("NeRFModelBody", deviceStaticBuffers.network->generate_device_function("eval_nerf")),
        fmt::arg("ApplyBackgroundModel", nrengPtr->has_sky_mlp),
        fmt::arg(
            "BackgroundModeBody",
            nrengPtr->has_sky_mlp ? deviceStaticBuffers.skyNetwork->generate_device_function("eval_background") : "inline __device__ vec3 eval_background(const vec3&, const network_precision_t*) {return vec3(0.0f); }"),
        fmt::arg("SRGBModel", !nrengPtr->linear_rgb),
        fmt::arg("SRGBOutput", !(nrengPtr->optFlags & nrend::RenderingParameters::OptLinearRGB)),
        fmt::arg("N_EXTRA_DIMS", nrengPtr->nExtraDims),
        fmt::arg("DensityGridMaxCascade", nrengPtr->densityGrid.max_cascade),
        fmt::arg("M_AABB_SCALE", nrengPtr->aabb_scale), fmt::arg("M_CONE_ANGLE", nrengPtr->cone_angle_constant),
        fmt::arg("M_STEP_SIZE", nrengPtr->min_step_size), fmt::arg("densityScale", nrengPtr->density_scale));

    Status status;
    deviceStaticBuffers.fusedRenderKernel =
        std::make_unique<nrend::CudaRtcKernel>(
            nrend::CudaKernelOptions{{"render"}},
            kernelStr,
            nrend::RtcKernelConfig::includeDirectories(),
            nrend::RtcKernelConfig::cacheDirectory(),
            nrend::RtcKernelConfig::extraIncludes(),
            nrengPtr->logger,
            status);
    if (!status) {
        return false;
    }

    deviceStaticBuffers.network->convert_params_to_jit_layout(deviceStream, true);
    if (deviceStaticBuffers.skyNetwork) {
        deviceStaticBuffers.skyNetwork->convert_params_to_jit_layout(deviceStream, true);
    }

    cudaStreamSynchronize(deviceStream);
    deviceStaticBuffers.initialized = true;

    return true;
}

} // anonymous namespace

nrend::NRELegacyRenderer::NRELegacyRenderer(const nlohmann::json& rendererState, const Logger& logger)
    : NRendererImplementation(rendererState, logger) {}

nrend::Status nrend::NRELegacyRenderer::initialize(const ModelVersion& version,
                                                   const nlohmann::json& config,
                                                   const RenderingParameters& renderParams) {
    if (!supportVersion(version, renderParams.rendererHint, renderParams.opts)) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "NRELegacyRenderer : unsupported model version %s.", version.str().c_str());
    }

    m_modelVersion = version;

    NRENeRF* rendererPtr = reinterpret_cast<NRENeRF*>(m_impl);
    if (!rendererPtr) {
        rendererPtr = new NRENeRF();
        if (!rendererPtr) {
            RETURN_ERROR(m_logger, ErrorCode::OutOfMemory, "NRELegacyRenderer : cannot allocate renderer implementation.");
        }
        m_impl = reinterpret_cast<void*>(rendererPtr);
    }

    rendererPtr->optFlags = renderParams.opts;
    rendererPtr->logger   = m_logger;

    const json& network_config      = config["network"];
    const json& dir_encoding_config = config["dir_encoding"];
    const json& rgb_network_config  = config["rgb_network"];
    json encoding_config            = config["encoding"];

    const uint32_t n_pos_dims = 3;
    const uint32_t n_dir_dims = 3;
    uint32_t n_extra_dims     = 0u; //< TODO : check if has_light_dirs should be supported

    {
        if (config["snapshot"].contains("density_grid")) {
            assert(ngp::NERF_GRIDSIZE() == config["snapshot"]["density_grid"]["grid_resolution"]);
            rendererPtr->densityGrid.max_cascade = static_cast<int>(config["snapshot"]["density_grid"]["cascades"]) - 1;
            rendererPtr->aabb_scale              = 1 << rendererPtr->densityGrid.max_cascade;
            rendererPtr->cone_angle_constant     = config["snapshot"]["density_grid"]["exp_step_factor"];
            rendererPtr->min_step_size           = config["snapshot"]["density_grid"]["min_step_size"];
            const json::binary_t& cpu_data       = config["snapshot"]["occupancy_grid_binary"];
            assert(ngp::NERF_GRIDSIZE() * ngp::NERF_GRIDSIZE() * ngp::NERF_GRIDSIZE() *
                       (rendererPtr->densityGrid.max_cascade + 1) * sizeof(uint8_t) ==
                   cpu_data.size() * 8);
            rendererPtr->densityGrid.density_grid_host.resize(cpu_data.size());
            std::memcpy(rendererPtr->densityGrid.density_grid_host.data(), cpu_data.data(), cpu_data.size());
        }
        if (config["snapshot"].contains("appearance_embedding_dims")) {
            n_extra_dims = config["snapshot"].value("appearance_embedding_dims", 0);
        }
        tcnn::BoundingBox aabb;
        aabb.min                    = config["snapshot"]["world_aabb"].at("min");
        aabb.max                    = config["snapshot"]["world_aabb"].at("max");
        const tcnn::vec3 aabbExtent = aabb.diag();
        rendererPtr->render_aabb    = tcnn::BoundingBox{tcnn::vec3(0.5f), tcnn::vec3(0.5f)};
        rendererPtr->render_aabb.inflate(0.5f * std::min<float>(1 << (ngp::NERF_CASCADES() - 1), rendererPtr->aabb_scale));
        const float aabbRelativeScale = tcnn::length(rendererPtr->render_aabb.diag()) / tcnn::length(aabbExtent);
        // scale from world -> ngp aabb scale
        rendererPtr->scale = config["snapshot"]["render_transform"].value("scale", 1.0f) * aabbRelativeScale;
        // offset from world -> NRE
        const auto offset = rendererPtr->optFlags & RenderingParameters::OptNREReferential ? std::array<float, 3>{0, 0, 0} : config["snapshot"]["render_transform"]["offset"].get<std::array<float, 3>>();
        // offset from world -> ngp + (0.5,0.5,0.5)
        rendererPtr->offset = tcnn::vec3(offset[0], offset[1], offset[2]) * rendererPtr->scale + vec3(0.5f);
        rendererPtr->min_step_size *= aabbRelativeScale;
        rendererPtr->density_scale = 1.0f / (aabbRelativeScale * tcnn::max(aabbExtent));
    }

    // Automatically determine certain parameters if we're dealing with the (hash)grid encoding
    rendererPtr->n_features_per_level = 2u;
    rendererPtr->n_levels             = 0;
    if (tcnn::to_lower(encoding_config.value("otype", "OneBlob")).find("grid") != std::string::npos) {
        encoding_config["n_pos_dims"] = n_pos_dims;

        rendererPtr->n_features_per_level = encoding_config.value("n_features_per_level", 2u);

        if (encoding_config.contains("n_features") && encoding_config["n_features"] > 0) {
            rendererPtr->n_levels = (uint32_t)encoding_config["n_features"] / rendererPtr->n_features_per_level;
        } else {
            rendererPtr->n_levels = encoding_config.value("n_levels", 16u);
        }

        const uint32_t log2_hashmap_size = encoding_config.value("log2_hashmap_size", 15);

        uint32_t base_grid_resolution = encoding_config.value("base_resolution", 0);
        if (!base_grid_resolution) {
            base_grid_resolution               = 1u << ((log2_hashmap_size) / n_pos_dims);
            encoding_config["base_resolution"] = base_grid_resolution;
        }

        const float desired_resolution = static_cast<int>(encoding_config["max_resolution"]);

        // Automatically determine suitable per_level_scale
        float per_level_scale = encoding_config.value("per_level_scale", 0.0f);
        if (per_level_scale <= 0.0f && rendererPtr->n_levels > 1) {
            per_level_scale                    = std::exp(std::log(desired_resolution / base_grid_resolution) / (rendererPtr->n_levels - 1));
            encoding_config["per_level_scale"] = per_level_scale;
        }

        if (LoggerParameters::Debug <= rendererPtr->logger.level()) {
            std::cout << "Renderer : loaded gridEncoding : " << std::endl;
            std::cout << " Nmin=" << base_grid_resolution << " b=" << per_level_scale
                      << " F=" << rendererPtr->n_features_per_level << " T=2^" << log2_hashmap_size
                      << " L=" << rendererPtr->n_levels << std::endl;
        }
    }

    rendererPtr->nPosDims          = n_pos_dims;
    rendererPtr->nDirDims          = n_dir_dims;
    rendererPtr->nExtraDims        = n_extra_dims;
    rendererPtr->encodingConfig    = encoding_config;
    rendererPtr->dirEncodingConfig = dir_encoding_config;
    rendererPtr->networkConfig     = network_config;
    rendererPtr->rgbNetworkConfig  = rgb_network_config;

    rendererPtr->per_level_scale      = encoding_config["per_level_scale"];
    rendererPtr->base_grid_resolution = encoding_config["base_resolution"];

    if (config["snapshot"].contains("params_binary")) {
        const json::binary_t& cpu_data = config["snapshot"]["params_binary"];
        rendererPtr->params_buffer_host.resize(cpu_data.size());
        std::memcpy(rendererPtr->params_buffer_host.data(), cpu_data.data(), cpu_data.size());
    }

    // Load sky-mlp
    rendererPtr->has_sky_mlp = config.contains("sky_model");
    if (rendererPtr->has_sky_mlp) {
        rendererPtr->skyEncodingConfig = config["sky_model"]["encoding"];
        rendererPtr->skyNetworkConfig  = config["sky_model"]["mlp"];

        if (config["sky_model"].contains("params_binary")) {
            const json::binary_t& cpu_data = config["sky_model"]["params_binary"];
            rendererPtr->sky_params_buffer_host.resize(cpu_data.size());
            std::memcpy(rendererPtr->sky_params_buffer_host.data(), cpu_data.data(), cpu_data.size());
        }
    }

    if (LoggerParameters::Debug <= rendererPtr->logger.level()) {
        std::cout << "Renderer : loaded model : " << m_modelVersion.model() << "::" << m_modelVersion.modelInstance() << " @ "
                  << m_modelVersion.str() << std::endl;

        std::cout << "Scale: " << rendererPtr->scale << std::endl;
        std::cout << "Offset: (" << rendererPtr->offset.x << "," << rendererPtr->offset.y << ","
                  << rendererPtr->offset.z << ")" << std::endl;

        std::cout << "NParams: "
                  << "N/A(n_params)"
                  << "  (ExtraDims: " << rendererPtr->nExtraDims << " ["
                  << n_extra_dims << "] )" << std::endl;

        std::cout << "Density model: " << n_pos_dims << "--[" << std::string(encoding_config["otype"]) << "]-->"
                  << "N/A(encoding::output_width)"
                  << "--[" << std::string(network_config["otype"])
                  << "(neurons=" << (int)network_config["n_neurons"]
                  << ",layers=" << ((int)network_config["n_hidden_layers"] + 2) << ")"
                  << "]-->" << 1
                  << " | scale=" << rendererPtr->density_scale << std::endl;

        std::cout << "Color model:   " << n_dir_dims << "--[" << std::string(dir_encoding_config["otype"]) << "]-->"
                  << "N/A(dir_encoding::output_width)"
                  << "+" << network_config.value("n_output_dims", 16u) << "--["
                  << std::string(rgb_network_config["otype"]) << "(neurons=" << (int)rgb_network_config["n_neurons"]
                  << ",layers=" << ((int)rgb_network_config["n_hidden_layers"] + 2) << ")"
                  << "]-->" << 3 << std::endl;

        std::cout << "AABB Scale:    " << rendererPtr->aabb_scale << std::endl;
        std::cout << "Render AABB:   [" << rendererPtr->render_aabb.min[0] << "," << rendererPtr->render_aabb.min[1]
                  << "," << rendererPtr->render_aabb.min[2] << "] [" << rendererPtr->render_aabb.max[0] << ","
                  << rendererPtr->render_aabb.max[1] << "," << rendererPtr->render_aabb.max[2] << "]" << std::endl;
        std::cout << "Cone angle:    " << rendererPtr->cone_angle_constant << std::endl;
        std::cout << "Min step size:    " << rendererPtr->min_step_size << std::endl;

        if (rendererPtr->has_sky_mlp) {
            std::cout << "Sky model:  "
                      << "--[" << std::string(rendererPtr->skyEncodingConfig["otype"]) << "]-->"
                      << "N/A(encoding::output_width)"
                      << "--[" << std::string(rendererPtr->skyNetworkConfig["otype"])
                      << "(neurons=" << (int)rendererPtr->skyNetworkConfig["n_neurons"]
                      << ",layers=" << ((int)rendererPtr->skyNetworkConfig["n_hidden_layers"] + 2) << ")"
                      << "]-->"
                      << std::endl;
        }

        std::cout << std::endl
                  << std::flush;
    }

    int numCudaDevices = 0;
    CUDA_CHECK(rendererPtr->logger, cudaGetDeviceCount(&numCudaDevices));
    rendererPtr->deviceStaticBuffers.resize(numCudaDevices);

    // check for all potential invalid value
    if (rendererPtr->scale <= 0.0f) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "NRELegacyRenderer : scene scale is null.");
    }

    return Status();
}

void nrend::NRELegacyRenderer::release() {
    if (m_impl) {
        delete reinterpret_cast<NRENeRF*>(m_impl);
        m_impl = nullptr;
    }
}

nrend::Status nrend::NRELegacyRenderer::renderForward(const RenderParameters& params,
                                                      const tcnn::vec3* wordlRayOriginCudaPtr,
                                                      const tcnn::vec3* worldRayDirectionCudaPtr,
                                                      const TTimestamp* worldRayTimestampCudaPtr,
                                                      const tcnn::ivec2* /*sensorsIdsPtr*/,
                                                      const tcnn::ivec2* /*activeTrackInstancesIdsCudaPtr*/,
                                                      const TTrackInstancePose* /*activeTrackInstancesPoseCudaPtr*/,
                                                      const TTrackInstancePose* /*activeTrackInstancesEndPoseCudaPtr*/,
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

    const NRENeRF* nrengPtr = reinterpret_cast<const NRENeRF*>(m_impl);
    if (!nrengPtr) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "NRELegacyRenderer : renderer is not initialized.");
    }

    if ((cudaDeviceIndex < 0) || (cudaDeviceIndex >= nrengPtr->deviceStaticBuffers.size()) || !prepareForDevice(nrengPtr, cudaDeviceIndex)) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "NRELegacyRenderer : cannot get cuda resource on the device %d.", cudaDeviceIndex);
    }

    NRENeRF::DeviceStaticBuffers& staticBuffers = nrengPtr->deviceStaticBuffers[cudaDeviceIndex];
    if (!staticBuffers.fusedRenderKernel) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "NRELegacyRenderer : cannot get cuda resource on the device %d.", cudaDeviceIndex);
    }

    if (worldHitNormalCudaPtr) {
        LOG_WARN(m_logger, "NRELegacyRenderer : does not support world hit normal output.");
    }

    RenderParameters kernelParams = params;
    kernelParams.worldToObjectTransform =
        kernelParams.worldToObjectTransform * (tcnn::mat4x4::identity() * nrengPtr->scale);
    kernelParams.worldToObjectTransform[3] += nrengPtr->offset;
    kernelParams.objectAABB.min         = kernelParams.objectAABB.min * nrengPtr->scale + nrengPtr->offset;
    kernelParams.objectAABB.max         = kernelParams.objectAABB.max * nrengPtr->scale + nrengPtr->offset;
    kernelParams.objectToWorldTransform = tcnn::inverse(tcnn::mat4x4(kernelParams.worldToObjectTransform));
    kernelParams.objectAABB             = kernelParams.objectAABB.intersection(nrengPtr->render_aabb);

    const dim3 threads = {8, 16, 1};
    const dim3 blocks  = {tcnn::div_round_up((uint32_t)kernelParams.frameTileResolution.x, threads.x),
                          tcnn::div_round_up((uint32_t)kernelParams.frameTileResolution.y, threads.y), 1};

    return staticBuffers.fusedRenderKernel->launch(
        0, blocks, threads, 0, cudaStream, m_logger, kernelParams,
        wordlRayOriginCudaPtr, worldRayDirectionCudaPtr, worldRayTimestampCudaPtr,
        instanceIdCudaPtr, worldHitDistanceCudaPtr, worldHitNormalCudaPtr, radianceDensityCudaPtr,
        staticBuffers.occupancyGridDeviceBuffers.data(), staticBuffers.network->inference_params(),
        nrengPtr->has_sky_mlp ? staticBuffers.skyNetwork->inference_params() : nullptr);
}
