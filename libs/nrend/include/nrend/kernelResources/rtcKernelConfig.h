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

#include <string>
#include <vector>

namespace nrend {

struct RtcKernelConfig {

    static void setCacheDirectory(const std::string& dir);
    static void setIncludeDirectory(const std::string& dir,
                                    bool append = false,
                                    bool extra  = false);

    static const std::string& cacheDirectory();
    static const std::vector<std::string>& includeDirectories();
    static std::vector<std::pair<std::string, const char*>> extraIncludes();

private:
    static std::vector<std::string> _includeDirectories;
    static std::vector<std::string> _extraIncludeDirectories;
    static bool _extraIncludeDirectoriesDirty;
    static std::string _cacheDirectory;
    static std::vector<std::pair<std::string, std::string>> _extraIncludes;
};

} // namespace nrend
