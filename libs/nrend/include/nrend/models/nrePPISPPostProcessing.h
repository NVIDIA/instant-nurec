// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <nrend/models/nreModel.h>
#include <sstream>
#include <vector>

namespace nrend {

// Expecting number of channels, vignetting, CRF and homography parameters (cf. nre/models/ppisp.py)
constexpr int NUM_CHANNELS                  = 3;
constexpr int NUM_VIGNETTING_OPTICAL_CENTER = 2;
constexpr int NUM_VIGNETTING_ALPHA_TERMS    = 3;
constexpr int NUM_VIGNETTING_PARAMS         = NUM_VIGNETTING_OPTICAL_CENTER + NUM_VIGNETTING_ALPHA_TERMS;
constexpr int NUM_CRF_PARAMS                = 7;
constexpr int NUM_HOMOGRAPHY_PARAMS         = 8;

class NREPPISPPostProcessing : public NREModel {
public:
    static constexpr char name[] = "ppisp-post-processing";

    // Total frames considers a unique frame index across all cameras
    int m_totalFrames = 0;
    int m_numCameras  = 0;

    // Frame-based parameters: exposure (1) + homography (8) = 9 per frame
    TStateDictTensor m_frameParams; // shape: [m_totalFrames, 9]
    // Sensor-channel parameters: optical_center(2)+alpha(3)+CRF(7) = 12 per sensor-channel
    TStateDictTensor m_sensorParams; // shape: [m_numCameras * NUM_CHANNELS, 12]

    NREPPISPPostProcessing(const nlohmann::json& config,
                           const Logger& logger,
                           const nlohmann::json& stateDict,
                           const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {

        // Read exposure (per frame, 1 value)
        TStateDictTensor exposureT;
        exposureT.key = prefix + "ppisp.exposure_params";
        if (!readStateDictTensor(stateDict, exposureT)) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : missing tensor parameter <%s> in state dict", exposureT.key.c_str());
        }

        // Read homography (per frame, 8 values)
        TStateDictTensor homoT;
        homoT.key = prefix + "ppisp.color_params";
        if (!readStateDictTensor(stateDict, homoT)) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : missing tensor parameter <%s> in state dict", homoT.key.c_str());
        }

        // Read total number of frames from exposure and check shapes and sizes of exposure and homography
        m_totalFrames = exposureT.shape.size() > 0 ? exposureT.shape[0] : 0;

        if (m_totalFrames == 0) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : number of frames is zero as per exposure tensor shape.");
        }
        if (homoT.shape.size() > 0 && homoT.shape[0] != m_totalFrames) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : inconsistent number of frames in exposure (%d) and homography (%d).",
                      m_totalFrames, homoT.shape[0]);
        }
        if (homoT.shape.size() != 2) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : expected homography shape dimension to be 2 and got %d.",
                      homoT.shape.size());
        }
        if (homoT.shape[1] != NUM_HOMOGRAPHY_PARAMS) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : expected homography shape to be [num_frames, 8] and got [%d, %d].",
                      homoT.shape[0], homoT.shape[1]);
        }

        // Initialize frameParams
        m_frameParams.key   = prefix + "ppisp.frame_params";
        m_frameParams.shape = {m_totalFrames, 1 + NUM_HOMOGRAPHY_PARAMS};
        m_frameParams.buffer.resize(m_totalFrames * (1 + NUM_HOMOGRAPHY_PARAMS) * sizeof(__half));
        { // Pack exposure and homography into frameParams
            auto* frameBuf = reinterpret_cast<__half*>(m_frameParams.buffer.data());
            auto* expPtr   = reinterpret_cast<const __half*>(exposureT.buffer.data());
            auto* homoPtr  = reinterpret_cast<const __half*>(homoT.buffer.data());
            for (int f = 0; f < m_totalFrames; ++f) {
                int base = f * (1 + NUM_HOMOGRAPHY_PARAMS);
                // exposure
                frameBuf[base + 0] = expPtr[f];
                // homography
                for (int i = 0; i < NUM_HOMOGRAPHY_PARAMS; ++i) {
                    frameBuf[base + 1 + i] = homoPtr[f * NUM_HOMOGRAPHY_PARAMS + i];
                }
            }
        }

        // Read vignetting (per camera, per channel, 5 values)
        TStateDictTensor vigT;
        vigT.key = prefix + "ppisp.vignetting_params";
        if (!readStateDictTensor(stateDict, vigT)) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : missing tensor parameter <%s> in state dict", vigT.key.c_str());
        }

        // Read CRF (per camera, per channel, 7 values)
        TStateDictTensor crfT;
        crfT.key = prefix + "ppisp.crf_params";
        if (!readStateDictTensor(stateDict, crfT)) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : missing tensor parameter <%s> in state dict", crfT.key.c_str());
        }

        // Read cameras and channels from vignetting and check shapes and sizes of vignetting and CRF
        m_numCameras = vigT.shape.size() > 0 ? vigT.shape[0] : 0;

        int numChannels  = vigT.shape.size() > 1 ? vigT.shape[1] : 0;
        int numVigParams = vigT.shape.size() > 2 ? vigT.shape[2] : 0;
        int numCRFParams = crfT.shape.size() > 2 ? crfT.shape[2] : 0;

        if (m_numCameras == 0) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : number of cameras is zero as per vignetting tensor shape.");
        }
        if (crfT.shape.size() > 0 && crfT.shape[0] != m_numCameras) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : inconsistent number of cameras in vignetting (%d) and CRF (%d).",
                      m_numCameras, crfT.shape[0]);
        }
        if (crfT.shape.size() > 1 && crfT.shape[1] != numChannels) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : inconsistent number of channels in vignetting (%d) and CRF (%d).",
                      numChannels, crfT.shape[1]);
        }
        if (numChannels != NUM_CHANNELS) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : expected number of channels is %d and got %d.",
                      NUM_CHANNELS, numChannels);
        }
        if (numVigParams != NUM_VIGNETTING_PARAMS) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : expected number of vignetting params is %d and got %d.",
                      NUM_VIGNETTING_PARAMS, numVigParams);
        }
        if (numCRFParams != NUM_CRF_PARAMS) {
            LOG_ERROR(logger, "NREPPISPPostProcessing : expected number of CRF params is %d and got %d.",
                      NUM_CRF_PARAMS, numCRFParams);
        }

        // Initialize sensorParams and pack vignetting and CRF into a single buffer per sensor-channel
        m_sensorParams.key   = prefix + "ppisp.sensor_params";
        m_sensorParams.shape = {m_numCameras * NUM_CHANNELS, NUM_VIGNETTING_PARAMS + NUM_CRF_PARAMS};
        m_sensorParams.buffer.resize(m_numCameras * NUM_CHANNELS * (NUM_VIGNETTING_PARAMS + NUM_CRF_PARAMS) * sizeof(__half));
        { // Pack vignetting and CRF into sensorParams
            auto* sensorBuf = reinterpret_cast<__half*>(m_sensorParams.buffer.data());
            auto* vigPtr    = reinterpret_cast<const __half*>(vigT.buffer.data());
            auto* crfPtr    = reinterpret_cast<const __half*>(crfT.buffer.data());
            for (int c = 0; c < m_numCameras; ++c) {
                for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
                    int sensorIdx = c * NUM_CHANNELS + ch;
                    int base      = sensorIdx * (NUM_VIGNETTING_PARAMS + NUM_CRF_PARAMS);
                    // Pack vignetting parameters
                    for (int i = 0; i < NUM_VIGNETTING_PARAMS; ++i) {
                        int vigIdx          = sensorIdx * NUM_VIGNETTING_PARAMS + i;
                        sensorBuf[base + i] = vigPtr[vigIdx];
                    }
                    // Pack CRF parameters
                    for (int j = 0; j < NUM_CRF_PARAMS; ++j) {
                        int crfIdx                                  = sensorIdx * NUM_CRF_PARAMS + j;
                        sensorBuf[base + NUM_VIGNETTING_PARAMS + j] = crfPtr[crfIdx];
                    }
                }
            }
        }
    }

    virtual ~NREPPISPPostProcessing() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger& logger) const override {
        Status status;

        // Register packed frame and sensor parameter buffers
        status = memoryBindings.registerMemory(
            KernelMemoryBindings::BindingsFlag::Parameters,
            m_frameParams.key,
            KernelMemoryType::Buffer,
            logger);
        CHECK_STATUS_RETURN(status);
        status = memoryBindings.registerMemory(
            KernelMemoryBindings::BindingsFlag::Parameters,
            m_sensorParams.key,
            KernelMemoryType::Buffer,
            logger);
        CHECK_STATUS_RETURN(status);

        // Inject binding indices into CUDA kernel via alias struct
        // Compute frame and sensor buffer binding indices
        const int frameBufIdx = memoryBindings.registeredMemoryIndex(
            KernelMemoryBindings::BindingsFlag::Parameters,
            m_frameParams.key);
        const int sensorBufIdx = memoryBindings.registeredMemoryIndex(
            KernelMemoryBindings::BindingsFlag::Parameters,
            m_sensorParams.key);

        const std::string sourceCode = fmt::format(R"(
#include <nrend/kernels/cuda/models/nrePPISPPostProcessing.cuh>

struct {alias}Params {{
    static constexpr int NUM_CHANNELS = {NUM_CHANNELS};
    static constexpr int NUM_VIGNETTING_OPTICAL_CENTER = {NUM_VIGNETTING_OPTICAL_CENTER};
    static constexpr int NUM_VIGNETTING_ALPHA_TERMS = {NUM_VIGNETTING_ALPHA_TERMS};
    static constexpr int NUM_CRF_PARAMS = {NUM_CRF_PARAMS};
    static constexpr int NUM_HOMOGRAPHY_PARAMS = {NUM_HOMOGRAPHY_PARAMS};
    static constexpr int TOTAL_FRAMES = {TOTAL_FRAMES};
    static constexpr int NUM_CAMERAS = {NUM_CAMERAS};
    static constexpr int FrameParamsBufferIndex = {FrameParamsBufferIndex};
    static constexpr int SensorParamsBufferIndex = {SensorParamsBufferIndex};
}};

using {alias} = NREPPISPPostProcessing<{alias}Params>;

)",
                                                   fmt::arg("alias", cudaCallPrefix()),
                                                   fmt::arg("NUM_CHANNELS", NUM_CHANNELS),
                                                   fmt::arg("NUM_VIGNETTING_OPTICAL_CENTER", NUM_VIGNETTING_OPTICAL_CENTER),
                                                   fmt::arg("NUM_VIGNETTING_ALPHA_TERMS", NUM_VIGNETTING_ALPHA_TERMS),
                                                   fmt::arg("NUM_CRF_PARAMS", NUM_CRF_PARAMS),
                                                   fmt::arg("NUM_HOMOGRAPHY_PARAMS", NUM_HOMOGRAPHY_PARAMS),
                                                   fmt::arg("TOTAL_FRAMES", m_totalFrames),
                                                   fmt::arg("NUM_CAMERAS", m_numCameras),
                                                   fmt::arg("FrameParamsBufferIndex", frameBufIdx),
                                                   fmt::arg("SensorParamsBufferIndex", sensorBufIdx));

        sourceCodeTable.registerKernel(KernelSourceCodeTable::Cuda, sourceCode);

        return Status();
    }

    virtual Status processKernelMemory_(
        const KernelMemoryBindings& memoryBindings,
        KernelMemoryBindings::BindingsFlag bindingsFlag,
        const std::vector<std::unique_ptr<KernelMemory>>& memory,
        ProcessMemoryFlag processFlag,
        uint64_t processQueueHandle,
        const Logger& logger) const override {

        Status status;
        if ((processFlag != ProcessMemoryFlag::Initialization) ||
            (bindingsFlag != KernelMemoryBindings::BindingsFlag::Parameters)) {
            return status;
        }

        // Upload packed buffers
        int idx = memoryBindings.registeredMemoryIndex(bindingsFlag, m_frameParams.key);
        CHECK_STATUS_RETURN(
            memory[idx]->setFromHost(m_frameParams.buffer.data(), m_frameParams.buffer.size(), processQueueHandle, logger));

        idx = memoryBindings.registeredMemoryIndex(bindingsFlag, m_sensorParams.key);
        CHECK_STATUS_RETURN(
            memory[idx]->setFromHost(m_sensorParams.buffer.data(), m_sensorParams.buffer.size(), processQueueHandle, logger));

        return status;
    }
};

} // namespace nrend
