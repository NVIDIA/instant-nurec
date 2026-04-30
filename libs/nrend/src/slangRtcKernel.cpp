// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/utils/slang/slangRtcKernel.h>

#include <slang-com-ptr.h>

#include <obfuscation.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace {

inline std::string dbgDumpCodeToTmpFile(const std::string& basename, const std::string& code) {
    std::filesystem::path codeDumpPath = std::filesystem::temp_directory_path() / std::filesystem::path(basename);
    std::ofstream codeDump(codeDumpPath.string());
    codeDump << code;
    return codeDumpPath.string();
}

// Check if coverage mode is enabled via CUDA_COVERAGE_MODE=1
// When enabled, uses reduced optimization for better source correlation
bool isCoverageModeEnabled() {
    const char* env = std::getenv("CUDA_COVERAGE_MODE");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

class DecodingFileSystem final : public ISlangFileSystem {
    class DecodedFileBlob final : public ISlangBlob {
        std::string m_str;
        std::atomic<uint32_t> m_refCount{0};

    private:
        ISlangBlob* getInterface(const SlangUUID& guid) {
            if (guid == ISlangUnknown::getTypeGuid() || guid == ISlangBlob::getTypeGuid()) {
                return static_cast<ISlangBlob*>(this);
            }
            return nullptr;
        }

    public:
        DecodedFileBlob(obfuscation::Decoder& decoder, const char* content) {
            m_str = decoder.decode(content);
        }
        ~DecodedFileBlob() = default;
        SLANG_NO_THROW const void* SLANG_MCALL getBufferPointer() override { return reinterpret_cast<const void*>(m_str.c_str()); }
        SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() override { return m_str.length(); }

        SLANG_IUNKNOWN_ALL;
    };

    std::atomic<uint32_t> m_refCount{0};
    const std::vector<std::string>& m_includeDirectories;
    const std::vector<std::pair<std::string, const char*>>& m_extraIncludes;
    obfuscation::Decoder m_decoder;

private:
    ISlangFileSystem* getInterface(const SlangUUID& guid) {
        if (guid == ISlangUnknown::getTypeGuid() || guid == ISlangFileSystem::getTypeGuid()) {
            return static_cast<ISlangFileSystem*>(this);
        }
        return nullptr;
    }

public:
    DecodingFileSystem(
        const std::vector<std::string>& includeDirectories,
        const std::vector<std::pair<std::string, const char*>>& extraIncludes)
        : m_includeDirectories(includeDirectories), m_extraIncludes(extraIncludes) {
    }

    SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& guid) override { return getInterface(guid); }
    SLANG_IUNKNOWN_ALL;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(
        char const* path,
        ISlangBlob** outBlob) {

        const std::string pathStr = path;

        // search the extra includes
        for (auto extraInclude : m_extraIncludes) {
            if (extraInclude.first == pathStr) {
                *outBlob = new DecodedFileBlob(m_decoder, extraInclude.second);
                if (*outBlob) {
                    (*outBlob)->addRef();
                    return 0;
                }
                return -1;
            }
        }

        // search the local filesystem
        {
            namespace fs = std::filesystem;
            for (const auto& includeDirectory : m_includeDirectories) {
                const fs::path absolutePath = fs::path(includeDirectory) / path;
                if (fs::exists(absolutePath)) {
                    std::ifstream ifs(absolutePath);
                    std::string content = std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
                    *outBlob            = new DecodedFileBlob(m_decoder, content.c_str());
                    if (*outBlob) {
                        (*outBlob)->addRef();
                        return 0;
                    }
                    return -1;
                }
            }
        }

        // failure
        return -1;
    }
};

inline SlangCompileTarget intermediateTargetToSlang(nrend::SlangRtcKernel::IntermediateTarget intermediateTarget) {
    switch (intermediateTarget) {
    case nrend::SlangRtcKernel::IntermediateTarget::Cuda: return SLANG_CUDA_SOURCE;
    case nrend::SlangRtcKernel::IntermediateTarget::PTX: return SLANG_PTX;
    default: return SLANG_TARGET_UNKNOWN;
    }
}
} // namespace

nrend::Status nrend::SlangRtcKernel::generateIntermediateTarget(IntermediateTarget target,
                                                                const std::string& kernelCode,
                                                                const std::vector<std::string>& includeDirs,
                                                                const std::string& /*cacheDir*/,
                                                                const std::vector<std::pair<std::string, const char*>>& extraIncludes,
                                                                std::string& targetCode,
                                                                const Logger& logger) {

    using namespace slang;

    Slang::ComPtr<IGlobalSession> globalSession;
    SlangResult slangStatus = createGlobalSession(globalSession.writeRef());
    if (SLANG_FAILED(slangStatus)) {
        RETURN_ERROR(logger, ErrorCode::Runtime, "Slang : cannot create global session.");
    }

    TargetDesc targetDesc        = {};
    targetDesc.format            = intermediateTargetToSlang(target);
    targetDesc.floatingPointMode = SLANG_FLOATING_POINT_MODE_FAST;

    SessionDesc sessionDesc = {};
    sessionDesc.targets     = &targetDesc;
    sessionDesc.targetCount = 1;
    // NB : enforce row-major layout for matrices (slang cuda target does not support column-major)
    // https://github.com/shader-slang/slang/blob/master/docs/user-guide/a1-01-matrix-layout.md
    sessionDesc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;

    // search paths are handled by the filesystem
    sessionDesc.searchPaths     = nullptr;
    sessionDesc.searchPathCount = 0;

    // create the decoding file system
    {
        auto decodingFileSystem = Slang::ComPtr<ISlangFileSystem>(new DecodingFileSystem(includeDirs, extraIncludes));
        sessionDesc.fileSystem  = decodingFileSystem.detach();
    }

    sessionDesc.preprocessorMacros     = nullptr;
    sessionDesc.preprocessorMacroCount = 0;

    std::array<CompilerOptionEntry, 2> compilerOptionEntries;

    // Coverage mode: enable line directives and disable optimization for NCU source correlation
    // Production mode: no line directives, high optimization
    const bool coverageMode = isCoverageModeEnabled();

    compilerOptionEntries[0].name            = CompilerOptionName::LineDirectiveMode;
    compilerOptionEntries[0].value           = {};
    compilerOptionEntries[0].value.intValue0 = coverageMode
                                                   ? SlangLineDirectiveMode::SLANG_LINE_DIRECTIVE_MODE_STANDARD
                                                   : SlangLineDirectiveMode::SLANG_LINE_DIRECTIVE_MODE_NONE;

    compilerOptionEntries[1].name            = CompilerOptionName::Optimization;
    compilerOptionEntries[1].value           = {};
    compilerOptionEntries[1].value.intValue0 = coverageMode
                                                   ? SlangOptimizationLevel::SLANG_OPTIMIZATION_LEVEL_NONE
                                                   : SlangOptimizationLevel::SLANG_OPTIMIZATION_LEVEL_HIGH;

    sessionDesc.compilerOptionEntries    = compilerOptionEntries.data();
    sessionDesc.compilerOptionEntryCount = compilerOptionEntries.size();

    Slang::ComPtr<ISession> session;
    slangStatus = globalSession->createSession(sessionDesc, session.writeRef());
    if (SLANG_FAILED(slangStatus)) {
        RETURN_ERROR(logger, ErrorCode::Runtime, "Slang : cannot create session.");
    }

    Slang::ComPtr<IBlob> diagnostics;
    IModule* modulePtr = session->loadModuleFromSourceString(
        "nrend",
        "nrend.slang",
        kernelCode.c_str(),
        diagnostics.writeRef());
    if (modulePtr == nullptr) {
        if (logger.level() >= LoggerParameters::Debug) {
            LOG_DEBUG(logger, "SlangRTC : error compiling kernel :");
            std::cout << dbgDumpCodeToTmpFile("nrend.slang", kernelCode) << std::endl;
            if (diagnostics) {
                std::cout << std::endl
                          << reinterpret_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
            }
        }
        RETURN_ERROR(logger, ErrorCode::Runtime, "Slang@%s : cannot compile kernel code.", spGetBuildTagString());
    } else if (logger.level() >= LoggerParameters::Debug) {
        LOG_DEBUG(logger, "Slang@%s : compiling slang kernel :", spGetBuildTagString());
        std::cout << dbgDumpCodeToTmpFile("nrend.slang", kernelCode) << std::endl;
        if (diagnostics != nullptr) {
            std::cout << reinterpret_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
        }
    }

    Slang::ComPtr<IComponentType> linkedProgram;
    slangStatus = modulePtr->link(linkedProgram.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(slangStatus)) {
        if (logger.level() >= LoggerParameters::Debug) {
            LOG_DEBUG(logger, "Slang : error linking slang kernel :");
            if (diagnostics) {
                std::cerr << std::endl
                          << reinterpret_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
            }
        }
        RETURN_ERROR(logger, ErrorCode::Runtime, "Slang : cannot link kernel code.");
    } else if ((logger.level() >= LoggerParameters::Debug) && (diagnostics != nullptr)) {
        LOG_DEBUG(logger, "Slang : linking slang kernel :");
        std::cout << reinterpret_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
    }

    Slang::ComPtr<slang::IBlob> targetCodeBuffer;
    slangStatus = linkedProgram->getTargetCode(0, targetCodeBuffer.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(slangStatus)) {
        if (logger.level() >= LoggerParameters::Debug) {
            LOG_DEBUG(logger, "Slang : error getting the target code :");
            std::cerr << kernelCode << std::endl;
            if (diagnostics) {
                std::cerr << std::endl
                          << reinterpret_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
            }
        }
        RETURN_ERROR(logger, ErrorCode::Runtime, "Slang : cannot get target code.");
    }

    targetCode.resize(targetCodeBuffer->getBufferSize());
    std::memcpy(targetCode.data(), targetCodeBuffer->getBufferPointer(), targetCodeBuffer->getBufferSize());

    return Status();
}
