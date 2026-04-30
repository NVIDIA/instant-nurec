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

namespace nrend {

enum class ErrorCode {
    None,            ///< No error == success
    InvalidResource, ///< A resource is not valid
    BadInput,        ///< An argument has an unexpected value
    OutOfMemory,     ///< Out of memory (allocation) error
    NotImplemented,  ///< Calling of function that is not implemented
    Runtime,         ///< Generic runtime error
    Num              ///< Number of valid error codes
};

#define NREND_SUCCESS(errorCode) (static_cast<nrend::ErrorCode>(errorCode) == nrend::ErrorCode::None)
#define NREND_FAILED(errorCode) (static_cast<nrend::ErrorCode>(errorCode) != nrend::ErrorCode::None)

} // namespace nrend