// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/kernelResources/rtcKernelConfig.h>
#include <nrend/renderer/grtOptixRejectionRenderer.h>
#include <nrend/renderer/grtOptixRejectionS0Renderer.h>
#include <nrend/renderer/grtOptixRenderer.h>
#include <nrend/renderer/gsRenderer.h>
#include <nrend/renderer/gutK0Renderer.h>
#include <nrend/renderer/gutRenderer.h>
#include <nrend/renderer/ngpRenderer.h>
#include <nrend/renderer/nreLegacyRenderer.h>
#include <nrend/renderer/nreRenderer.h>
#include <nrend/renderer/renderer.h>
#include <nrend/utils/cuda/cudaCommon.h>
#include <nrend/utils/registrar.h>

#include <json/json.hpp>

#include <type_traits>
#include <vector>

namespace {

struct NRendererRegistrar : nrend::IRegistrar<NRendererRegistrar, nrend::NRendererImplementation> {

    static nrend::NRendererImplementation* create(const nlohmann::json& rendererState,
                                                  const nrend::RenderingParameters& renderParams,
                                                  nrend::ModelVersion modelVersion,
                                                  const nlohmann::json& modelState,
                                                  const nrend::Logger& logger) {
        nrend::NRendererImplementation* rendererPtr = createFromJSON(rendererState, logger);
        return initialize(rendererPtr, modelVersion, modelState, renderParams);
    }

    static nrend::NRendererImplementation* createDefault(const nrend::RenderingParameters& renderParams,
                                                         nrend::ModelVersion modelVersion,
                                                         const nlohmann::json& modelState,
                                                         const nrend::Logger& logger) {

        nrend::NRendererImplementation* rendererPtr = createFirstValid(
            logger,
            [&modelVersion, &renderParams](nrend::NRendererImplementation* ptr) { return ptr && ptr->supportVersion(modelVersion, renderParams.rendererHint, renderParams.opts); });
        return initialize(rendererPtr, modelVersion, modelState, renderParams);
    }

    static RegisterInstantiatorMap s_registeredInstantiators;

private:
    static inline nrend::NRendererImplementation* initialize(nrend::NRendererImplementation* rendererPtr, nrend::ModelVersion modelVersion,
                                                             const nlohmann::json& modelState, const nrend::RenderingParameters& renderParams) {
        if (rendererPtr) {
            if (!rendererPtr->initialize(modelVersion, modelState, renderParams)) {
                delete rendererPtr;
                return nullptr;
            }
        }
        return rendererPtr;
    }
};

NRendererRegistrar::RegisterInstantiatorMap NRendererRegistrar::s_registeredInstantiators;

} // namespace

namespace nrend {
REGISTER_IMPLEMENTATION_EXT(NRendererRegistrar, NGPRenderer);
REGISTER_IMPLEMENTATION_EXT(NRendererRegistrar, NRELegacyRenderer);
REGISTER_IMPLEMENTATION_EXT(NRendererRegistrar, NRERenderer);
REGISTER_IMPLEMENTATION_EXT(NRendererRegistrar, GUTRenderer);
REGISTER_IMPLEMENTATION_EXT(NRendererRegistrar, GUTK0Renderer);
REGISTER_IMPLEMENTATION_EXT(NRendererRegistrar, GRTOptixRenderer);
REGISTER_IMPLEMENTATION_EXT(NRendererRegistrar, GRTOptixRejectionRenderer);
REGISTER_IMPLEMENTATION_EXT(NRendererRegistrar, GRTOptixRejectionS0Renderer);
REGISTER_IMPLEMENTATION_EXT(NRendererRegistrar, GSRenderer);
} // namespace nrend

nrend::NRenderer* nrend::NRenderer::loadFromMsgPackData(MsgPackData modelData,
                                                        MsgPackData rendererData,
                                                        const RenderingParameters& renderParams,
                                                        Logger logger) {
    nlohmann::json modelState       = nlohmann::json::from_msgpack(modelData.dataPtr, modelData.dataPtr + modelData.dataSz);
    const ModelVersion modelVersion = ModelVersion{modelState};

    nrend::NRenderer* rendererPtr = nullptr;
    nlohmann::json rendererState;
    if (rendererData.dataSz && rendererData.dataPtr) {
        rendererState = nlohmann::json::from_msgpack(rendererData.dataPtr, rendererData.dataPtr + rendererData.dataSz);
    }
    // fallback to renderer config in the model state only if render hint is default
    else if ((renderParams.rendererHint == RenderingParameters::RendererDefault) &&
             modelState.contains("nre_data") &&
             modelState["nre_data"].contains("config") &&
             modelState["nre_data"]["config"].contains("renderer")) {
        rendererState = modelState["nre_data"]["config"]["renderer"];
    }
    if (!rendererState.empty()) {
        rendererPtr = NRendererRegistrar::create(rendererState, renderParams, modelVersion, modelState, logger);
    } else {
        rendererPtr = NRendererRegistrar::createDefault(renderParams, modelVersion, modelState, logger);
    }
    if (rendererPtr) {
        LOG_INFO(logger, "Model %s::%s => %s opened", modelVersion.model().c_str(), modelVersion.modelInstance().c_str(), modelVersion.extStr().c_str());
    } else {
        LOG_ERROR(logger, "Cannot open model %s::%s => %s", modelVersion.model().c_str(),
                  modelVersion.modelInstance().c_str(), modelVersion.str().c_str());
    }
    return rendererPtr;
}
