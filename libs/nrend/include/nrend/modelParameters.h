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

#include <cstddef>

namespace nrend {

struct ParameterDefinition {
    size_t size   = 0;
    void* dataPtr = nullptr;
    enum Type {
        Buffer,
        Value,
        Undefined
    } type = Undefined;
};

struct NamedParameterDefinition {
    const char* name = "";
    ParameterDefinition definition;
};

struct NamedParameterDefinitionsSpan {
    size_t size = 0;
    NamedParameterDefinition* data;
};

}; // namespace nrend