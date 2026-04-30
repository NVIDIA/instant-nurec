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

#include <fstream>

#ifdef TCNN_CMRC
#include <cmrc/cmrc.hpp>
#include <fmt/core.h>
CMRC_DECLARE(tcnn);
CMRC_DECLARE(nrend);
namespace cmrc {
class embedded_filesystem;
};
namespace {
void fetchCMRMCFiles(const cmrc::embedded_filesystem& fs, const std::string& dir, std::vector<std::pair<std::string, const char*>>& result) {
    for (auto&& entry : fs.iterate_directory(dir)) {
        auto fn = dir.empty() ? entry.filename() : fmt::format("{}/{}", dir, entry.filename());
        if (entry.is_file()) {
            result.emplace_back(fn, fs.open(fn).begin());
        } else if (entry.is_directory()) {
            fetchCMRMCFiles(fs, fn, result);
        }
    }
}
}; // namespace
#elif defined(TCNN_PMRC)
#include <nrend_pmrc_resource.hpp>
#else
#include <filesystem>
#include <queue>
#endif

std::vector<std::string> nrend::RtcKernelConfig::_includeDirectories;
std::vector<std::string> nrend::RtcKernelConfig::_extraIncludeDirectories;
bool nrend::RtcKernelConfig::_extraIncludeDirectoriesDirty = false;
std::string nrend::RtcKernelConfig::_cacheDirectory;
std::vector<std::pair<std::string, std::string>> nrend::RtcKernelConfig::_extraIncludes;

const std::string& nrend::RtcKernelConfig::cacheDirectory() {
    return _cacheDirectory;
}

void nrend::RtcKernelConfig::setCacheDirectory(const std::string& dir) {
    _cacheDirectory = dir;
}

const std::vector<std::string>& nrend::RtcKernelConfig::includeDirectories() {
    return _includeDirectories;
}

void nrend::RtcKernelConfig::setIncludeDirectory(const std::string& dir, bool append, bool extra) {
    std::vector<std::string>& dirs = extra ? _extraIncludeDirectories : _includeDirectories;
    if (append) {
        dirs.push_back(dir);
    } else {
        dirs = {dir};
    }
    _extraIncludeDirectoriesDirty = _extraIncludeDirectoriesDirty || extra;
}

#if !defined(TCNN_CMRC) && !defined(TCNN_PMRC)
static std::vector<std::string> findFilesRecursive(const std::string& root) {
    std::vector<std::string> files;
    std::queue<std::string> dirs;

    dirs.push(root);

    while (dirs.empty() == false) {
        const std::string d = dirs.front();
        dirs.pop();

        for (const auto& entry : std::filesystem::directory_iterator(d)) {
            if (entry.is_directory()) {
                dirs.push(entry.path().string());
            } else if (entry.is_regular_file()) {
                files.push_back(entry.path().string());
            }
        }
    }

    return files;
}
#endif

std::vector<std::pair<std::string, const char*>> nrend::RtcKernelConfig::extraIncludes() {
#ifdef TCNN_CMRC
    std::vector<std::pair<std::string, const char*>> _extraIncludes;
    // add tcnn cmrc files
    fetchCMRMCFiles(cmrc::tcnn::get_filesystem(), "", _extraIncludes);
    // add nrend cmrc files
    fetchCMRMCFiles(cmrc::nrend::get_filesystem(), "", _extraIncludes);
    return _extraIncludes;
#elif defined(TCNN_PMRC)
    // return nrend pmrc files (includes tcnn files)
    return nrend_pmrc_resource::get_resources();
#else
    // fetch files from disk
    namespace fs = std::filesystem;
    if (_extraIncludeDirectoriesDirty) {
        _extraIncludeDirectoriesDirty = false;
        _extraIncludes.clear();
        auto readFileContent = [](const fs::path& filePath) {
            std::ifstream ifs(filePath);
            return std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
        };
        for (const std::string& dir : _extraIncludeDirectories) {
            const std::vector<std::string> files = findFilesRecursive(dir);
            for (const std::string& fn : files) {
                const std::string subpath = fn.substr(dir.length() + 1);
                _extraIncludes.push_back(std::make_pair(subpath, readFileContent(fn)));
            }
        }
    }
    std::vector<std::pair<std::string, const char*>> extraIncludesStr;
    extraIncludesStr.reserve(_extraIncludes.size());
    for (const auto& inc : _extraIncludes) {
        extraIncludesStr.emplace_back(inc.first, inc.second.c_str());
    }
    return extraIncludesStr;
#endif
}
