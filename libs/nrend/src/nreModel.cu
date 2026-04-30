// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/models/nreAccelerationStructure.h>
#include <nrend/models/nreAppearanceEmbedding.h>
#include <nrend/models/nreBackground.h>
#include <nrend/models/nreDynamicShGaussianModel.h>
#include <nrend/models/nreFeatureVolume.h>
#include <nrend/models/nreGaussiansCompositeModel.h>
#include <nrend/models/nreGaussiansPrimitiveModel.h>
#include <nrend/models/nreGeometry.h>
#include <nrend/models/nreInputEmbedding.h>
#include <nrend/models/nreModel.h>
#include <nrend/models/nreNeRFModel.h>
#include <nrend/models/nrePPISPPostProcessing.h>
#include <nrend/models/nrePostProcessings.h>
#include <nrend/models/nreShGaussianModel.h>
#include <nrend/models/nreTexture.h>
#include <nrend/models/nreTraceableCompositeModel.h>

namespace nrend {

NREModel::RegisterInstantiatorMap NREModel::s_registeredInstantiators;

REGISTER_IMPLEMENTATION(NRESkipBackground);
REGISTER_IMPLEMENTATION(NREColorBackground);
REGISTER_IMPLEMENTATION(NREEnvMapBackground);
REGISTER_IMPLEMENTATION(NRESkyMLPBackground);

REGISTER_IMPLEMENTATION(NREDenseObjectAccStructure);
REGISTER_IMPLEMENTATION(NRENeRFAccelerationStructure);

REGISTER_IMPLEMENTATION(NRESkipAppearanceEmbedding);
REGISTER_IMPLEMENTATION(NREDefaultAppearanceEmbedding);
REGISTER_IMPLEMENTATION(NREGloAppearanceEmbedding);
REGISTER_IMPLEMENTATION(NREWeightedInstanceInputEmbedding);
REGISTER_IMPLEMENTATION(NREIndividualRemapTimeInputEmbedding);

REGISTER_IMPLEMENTATION(NREHashGridFeatureVolume);
REGISTER_IMPLEMENTATION(NREHashGridObjectFeatureVolume);

REGISTER_IMPLEMENTATION(NRESkipGeometry);

REGISTER_IMPLEMENTATION(NREFullyFusedTexture);

REGISTER_IMPLEMENTATION(NRENeRFModel);
REGISTER_IMPLEMENTATION(NRETraceableCompositeModel);

REGISTER_IMPLEMENTATION(NRESHGaussianModel);
REGISTER_IMPLEMENTATION(NRERigidSHGaussianModel);
REGISTER_IMPLEMENTATION(NREDeformableSHGaussianModel);
REGISTER_IMPLEMENTATION(NREGaussiansPrimitiveModel);
REGISTER_IMPLEMENTATION(NREGaussiansCompositeModel);

REGISTER_IMPLEMENTATION(NREPostProcessings);
REGISTER_IMPLEMENTATION(NRESkipPostProcessing);
REGISTER_IMPLEMENTATION(NRECameraBilateralGridPostProcessing);
REGISTER_IMPLEMENTATION(NREFrameBilateralGridPostProcessing);
REGISTER_IMPLEMENTATION(NREPPISPPostProcessing);

} // namespace nrend