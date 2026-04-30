// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

/** This code has been adapted from tcnn/rtc_kernel.h
 *  @author Thomas Müller, NVIDIA
 */

#include <nrend/utils/cuda/cudaRtcKernel.h>

#include <nvrtc.h>
#include <tiny-cuda-nn/common_host.h>

#include <obfuscation.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
// Check if coverage mode is enabled via CUDA_COVERAGE_MODE=1
// When enabled, uses reduced optimization for better source correlation
bool isCoverageModeEnabled() {
    const char* env = std::getenv("CUDA_COVERAGE_MODE");
    return env && std::strcmp(env, "1") == 0;
}

// Get RTC optimization level from environment variable CUDA_RTC_OPT_LEVEL
// CUDA_COVERAGE_MODE=1: set opt level to 0 for source correlation
// Else by default returns -1, for Prod compilation
int getRtcOptLevelInt() {
    const char* env = std::getenv("CUDA_RTC_OPT_LEVEL");
    if (env) {
        int level = std::atoi(env);
        if (level >= 0 && level <= 3) {
            return level;
        }
    }
    // In coverage mode, default to 0;
    return isCoverageModeEnabled() ? 0 : -1;
}
} // namespace

#define NVRTC_CHECK_RETURN(result, logger)                                                                             \
    do {                                                                                                               \
        nvrtcResult _result = result;                                                                                  \
        if (_result != NVRTC_SUCCESS) {                                                                                \
            _SET_ERROR(logger, ErrorCode::Runtime, FILE_LINE " " #result " failed: %s", nvrtcGetErrorString(_result)); \
            return ___status;                                                                                          \
        }                                                                                                              \
    } while (0)

#define CU_RES_CHECK_SET(result, logger, status)                                                                    \
    do {                                                                                                            \
        CUresult _result = result;                                                                                  \
        if (_result != CUDA_SUCCESS) {                                                                              \
            const char* cuResCheckMsg;                                                                              \
            cuGetErrorName(_result, &cuResCheckMsg);                                                                \
            _SET_ERROR(logger, ErrorCode::Runtime, FILE_LINE " " #result " failed: %d %s", _result, cuResCheckMsg); \
            status = ___status;                                                                                     \
        }                                                                                                           \
    } while (0)

/*
 * Declare the NVRTC API function here, since "nvrtc_internal.h" is not part of
 * shipping NVRTC toolkits
 */

enum NVRTC_EXTENSION {
    /* address of the decoder callback, which is expected
     * to have the signature "char * (*decodeFn)(const void *in, void *payload)"
     * 'in' is the encoded string representation.
     * 'payload' is a user specified pointer value. This can be optionally specified
     *  with the 'NVRTC_EXT_DECODE_HANDLER_USER_PAYLOAD' extension, othwrwise will be
     * nullptr.
     * The decoder callback is expected to return a pointer to a null-terminated
     * C string that is the decoded representation of the encoded argument.
     */
    NVRTC_EXT_DECODE_HANDLER_CALLBACK = 1,

    /* user provided 'void *' value, that is passed back to the decode callback
     * function specified with extension 'NVRTC_EXT_DECODE_HANDLER_CALLBACK'.
     * */
    NVRTC_EXT_DECODE_HANDLER_USER_PAYLOAD = 2
};

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
nvrtcResult __nvrtcCPEx(nvrtcProgram* prog,
                        const char* src,
                        const char* name,
                        int numHeaders,
                        const char* const* headers,
                        const char* const* includeNames,
                        int numExtensions,
                        NVRTC_EXTENSION* extensions,
                        void** extensionArgs);
#ifdef __cplusplus
}
#endif /* __cplusplus */

namespace {
inline std::string dbgDumpCodeToTmpFile(const std::string& basename, const std::string& code) {
    std::filesystem::path codeDumpPath = std::filesystem::temp_directory_path() / std::filesystem::path(basename);
    std::ofstream codeDump(codeDumpPath.string());
    codeDump << code;
    return codeDumpPath.string();
}
} // namespace

nrend::Status nrend::CudaRtcKernel::generatePTX(std::vector<const char*> kernelNames,
                                                const std::string& kernel_code,
                                                const std::vector<std::string>& include_dirs,
                                                const std::string& cache_dir,
                                                const std::vector<std::pair<std::string, const char*>>& extra_includes,
                                                const std::vector<std::string>& options,
                                                std::vector<char>& ptxBuffer,
                                                std::vector<std::string>& loweredKernelNames,
                                                const Logger& logger) {

    RETURN_ERROR_IF(kernelNames.empty(), logger, ErrorCode::BadInput, "CudaRtcKernel : no kernel names provided.");
    // use the first kernel name as the name of the module
    const std::string name = kernelNames[0];

    std::vector<std::string> opts = options;
    if (!include_dirs.empty()) {
        for (auto include_dir : include_dirs) {
            opts.emplace_back(fmt::format("-I{}", include_dir));
        }
    }

    const auto codeTemplate        = R"(/*
        * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
        *
        * NVIDIA CORPORATION and its licensors retain all intellectual property
        * and proprietary rights in and to this software, related documentation
        * and any modifications thereto.  Any use, reproduction, disclosure or
        * distribution of this software and related documentation without an express
        * license agreement from NVIDIA CORPORATION is strictly prohibited.
        * 
        * {KERNEL_NAME}
        * 
        * {OPTS}
        */
        #include <tiny-cuda-nn/ministd.h>
        {PREAMBLE}

        {KERNEL_CODE}
    )";
    const std::string codeInstance = fmt::format(codeTemplate,
                                                 fmt::arg("KERNEL_NAME", name),
                                                 fmt::arg("PREAMBLE", tcnn::generate_device_code_preamble()),
                                                 fmt::arg("OPTS", tcnn::join(opts, "\n")),
                                                 fmt::arg("KERNEL_CODE", kernel_code));

    size_t code_hash = tcnn::hash_combine(0, codeInstance);

    std::vector<const char*> headers = {
        // First define headers that we wish to ignore (because they won't be available
        // at runtime). Instead, we will include tiny-cuda-nn/ministd.h, which implements
        // the small subset of the STL that we actually need.
        "algorithm",
        "cassert",
        "cmath",
        "cstddef",
        "cstdint",
        "cuda.h",
        "limits",
        "type_traits",
        "initializer_list",
    };
    std::vector<const char*> headers_content(headers.size(), "");

    // Next, we add all headers that come bundled with tcnn as well as those we received in the constructor.
    // We combine each header's hash with that of our code to obtain a unique fingerprint that can be used
    // for caching.
    std::vector<std::pair<std::string, const char*>> includes;
    includes.insert(std::end(includes), std::begin(extra_includes), std::end(extra_includes));
    for (const auto& entry : includes) {
        headers.emplace_back(entry.first.c_str());
        headers_content.emplace_back(entry.second);
        code_hash = tcnn::hash_combine(code_hash, std::string{entry.second});
    }

    std::string code_hash_string = fmt::format("{:016x}", code_hash);
    std::string filename         = fmt::format("{}.{}.cu", name, code_hash_string);

    const bool use_cache = !cache_dir.empty() && (logger.level() < LoggerParameters::Debug);

    std::string cached_code_filename = fmt::format("{}/{}", cache_dir, filename);
    std::string cached_ptx_filename  = fmt::format("{}/{}.{}.ptx", cache_dir, name, code_hash_string);

    // Decoder class that manages memory
    obfuscation::Decoder decoder;

    if (use_cache) {
        // Check if we've cached the resulting PTX last time around and load it if so.
        std::ifstream f{cached_ptx_filename};
        if (f) {
            // Get the size of the cached PTX file
            f.seekg(0, std::ios::end);
            size_t fullSize = f.tellg();
            f.seekg(0);
            // The first line of the cached PTX contains a comment of the form
            //  `//lowered_kernel_name=<value>` which we need to parse.
            std::string firstLine;
            std::getline(f, firstLine);
            auto s = tcnn::split(firstLine, "=");
            if (s.size() == 2 && s[0].find("lowered_kernel_name") != std::string::npos) {
                loweredKernelNames = tcnn::split(s[1], ",");
                // Read the remaining PTX code
                size_t encodedSize = fullSize - f.tellg();
                std::vector<char> encodedStr(encodedSize, '\0');
                f.read(encodedStr.data(), encodedSize);
                // decode the obfuscated PTX file
                const char* decodedPtr = decoder.decode(encodedStr.data());
                ptxBuffer              = std::vector<char>(decodedPtr, decodedPtr + strlen(decodedPtr) + 1);
                LOG_DEBUG(logger, "CudaRTC : loaded PTX from cache %s", cached_ptx_filename.c_str());
            }
        }
    }

    // If we haven't loaded PTX from cache, compile the program
    if (ptxBuffer.empty()) {

        NVRTC_EXTENSION extensions[] = {
            NVRTC_EXT_DECODE_HANDLER_CALLBACK,
            NVRTC_EXT_DECODE_HANDLER_USER_PAYLOAD};

        auto decode = [](const void* in, void* decoder) {
            return reinterpret_cast<obfuscation::Decoder*>(decoder)->decode((const char*)in);
        };

        const char* (*fp)(const void*, void* payload) = decode;
        void* extensionArgs[]                         = {(void*)fp, (void*)&decoder};

        // In coverage mode: persist source to a stable path so NCU can resolve it
        std::string nvrtc_program_name = filename;
        if (isCoverageModeEnabled()) {
            std::error_code ec;
            std::filesystem::create_directories(std::filesystem::path(cached_code_filename).parent_path(), ec);
            std::ofstream persisted_source{cached_code_filename};
            if (persisted_source) {
                persisted_source << codeInstance;
                nvrtc_program_name = cached_code_filename;
            }
        }

        nvrtcProgram prog;
        NVRTC_CHECK_RETURN(__nvrtcCPEx(
                               &prog,
                               codeInstance.c_str(),
                               nvrtc_program_name.c_str(),
                               headers.size(),
                               headers_content.data(),
                               headers.data(),
                               sizeof(extensions) / sizeof(extensions[0]),
                               extensions,
                               extensionArgs),
                           logger);

        tcnn::ScopeGuard destroyProgGuard{[&prog]() { nvrtcDestroyProgram(&prog); }};

        for (const auto& kernelName : kernelNames) {
            NVRTC_CHECK_RETURN(nvrtcAddNameExpression(prog, kernelName), logger);
        }

        std::vector<const char*> opts_c_str;
        for (const auto& opt : opts) {
            opts_c_str.emplace_back(opt.c_str());
        }

        nvrtcResult compile_result = nvrtcCompileProgram(prog, opts_c_str.size(), opts_c_str.data());
        if (logger.level() >= LoggerParameters::Debug) {
            size_t log_size;
            NVRTC_CHECK_RETURN(nvrtcGetProgramLogSize(prog, &log_size), logger);
            std::vector<char> log(log_size + 1, '\0');
            NVRTC_CHECK_RETURN(nvrtcGetProgramLog(prog, log.data()), logger);
            LOG_DEBUG(logger, "CudaRTC : compiling kernel %s :", filename.c_str());
            std::cerr << dbgDumpCodeToTmpFile(filename, codeInstance) << std::endl;
            std::cerr << log.data() << std::endl;
        }
        if (compile_result != NVRTC_SUCCESS) {
            RETURN_ERROR(logger, ErrorCode::Runtime, "JIT: compiling %s failed.", filename.c_str());
        }

        loweredKernelNames.reserve(kernelNames.size());
        for (const auto& kernelName : kernelNames) {
            const char* lowered_kernel_name_cstr;
            NVRTC_CHECK_RETURN(nvrtcGetLoweredName(prog, kernelName, &lowered_kernel_name_cstr), logger);
            loweredKernelNames.push_back(lowered_kernel_name_cstr);
        }

        size_t ptx_size;
        NVRTC_CHECK_RETURN(nvrtcGetPTXSize(prog, &ptx_size), logger);
        ptxBuffer.resize(ptx_size, '\0');
        NVRTC_CHECK_RETURN(nvrtcGetPTX(prog, ptxBuffer.data()), logger);

        if (use_cache) {
            std::ofstream f{cached_ptx_filename};
            if (f) {
                f << fmt::format("//lowered_kernel_name={}\n", tcnn::join(loweredKernelNames, ","));
                std::vector<char> encodedData = obfuscation::encode(ptxBuffer.data(), ptxBuffer.size());
                f.write(encodedData.data(), encodedData.size());
                LOG_DEBUG(logger, "CudaRTC : cached PTX to %s", cached_ptx_filename.c_str());
            }
        }
    }

    return Status();
}

// If a cache dir is provided, compilation artifacts will be cached in there and re-loaded upon program restart. Improves the user experience.
nrend::CudaRtcKernel::CudaRtcKernel(
    const CudaKernelOptions& kernelOptions,
    const std::string& kernel_code,
    const std::vector<std::string>& include_dirs,
    const std::string& cache_dir,
    const std::vector<std::pair<std::string, const char*>>& extra_includes,
    const Logger& logger,
    Status& status) {
    auto start_time = std::chrono::steady_clock::now();

    status = Status();

    std::vector<char> ptxBuffer;
    std::vector<std::string>& loweredKernelNames = m_loweredKernelNames;

    uint32_t cc                   = tcnn::cuda_supported_compute_capability();
    int optLevel                  = getRtcOptLevelInt();
    bool coverageMode             = (optLevel >= 0); // Coverage mode if explicit opt level set
    std::vector<std::string> opts = {
        fmt::format("--gpu-architecture=compute_{}", cc),
        fmt::format("-DTCNN_MIN_GPU_ARCH={}", cc),
        "--std=c++17",
        "--use_fast_math",
    };
    if (coverageMode) {
        // Add debug info and control PTX assembler optimization
        opts.push_back("-lineinfo");
        opts.push_back(fmt::format("--ptxas-options=-O{}", optLevel));
    } else {
        opts.push_back("--extra-device-vectorization");
    }

    if (logger.level() < LoggerParameters::Debug) {
        opts.push_back("-DNDEBUG"); // Disable asserts in release mode
    }

    status = generatePTX(kernelOptions.entryPointNames, kernel_code, include_dirs, cache_dir, extra_includes, opts, ptxBuffer, loweredKernelNames, logger);
    if (!status) {
        return;
    }

    // Load PTX - use JIT optimization control only in coverage mode
    CUresult cuResult;
    if (coverageMode) {
        CUjit_option jitOptions[] = {CU_JIT_OPTIMIZATION_LEVEL};
        void* jitOptionValues[]   = {reinterpret_cast<void*>(static_cast<uintptr_t>(optLevel))};
        cuResult                  = cuModuleLoadDataEx(&m_module, ptxBuffer.data(), 1, jitOptions, jitOptionValues);
    } else {
        // Prod mode: let driver choose optimization
        cuResult = cuModuleLoadDataEx(&m_module, ptxBuffer.data(), 0, nullptr, nullptr);
    }
    if (cuResult != CUDA_SUCCESS) {
        CU_RES_CHECK_SET(cuResult, logger, status);
        return;
    }

    m_kernels.reserve(loweredKernelNames.size());
    for (const auto& loweredKernelName : loweredKernelNames) {
        CUfunction kernel;
        cuResult = cuModuleGetFunction(&kernel, m_module, loweredKernelName.c_str());
        if (cuResult != CUDA_SUCCESS) {
            clear();
            CU_RES_CHECK_SET(cuResult, logger, status);
            return;
        }
        m_kernels.push_back(kernel);
    }

    const float compilation_duration_seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
    LOG_INFO(logger, "JIT: compiled %s in %.2fs", tcnn::join(loweredKernelNames, ",").c_str(), compilation_duration_seconds);
}

CUfunction nrend::CudaRtcKernel::getKernelFunction(uint32_t entryPointIndex, const Logger& logger) const {
    if (entryPointIndex >= m_kernels.size()) {
        LOG_ERROR(logger, "CudaRtcKernel : invalid entry point index %u.", entryPointIndex);
        return nullptr;
    }

    return m_kernels[entryPointIndex];
}

nrend::Status nrend::CudaRtcKernel::setKernelCacheConfig(uint32_t entryPointIndex, CUfunc_cache cacheConfig, const Logger& logger) {
    if (entryPointIndex >= m_kernels.size()) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "CudaRtcKernel : invalid entry point index %u.", entryPointIndex);
    }

    CUresult result = cuFuncSetCacheConfig(m_kernels[entryPointIndex], cacheConfig);
    if (result != CUDA_SUCCESS) {
        const char* errorMsg;
        cuGetErrorName(result, &errorMsg);
        RETURN_ERROR(logger, ErrorCode::Runtime,
                     "CudaRtcKernel : cuFuncSetCacheConfig failed: %s", errorMsg);
    }

    return Status();
}

nrend::CudaRtcKernel::~CudaRtcKernel() {
    clear();
}

void nrend::CudaRtcKernel::clear() {
    if (m_module) {
        cuModuleUnload(m_module);
        m_module = {};
    }
}

nrend::Status nrend::CudaRtcKernel::set(CUfunction kernel, CUfunction_attribute attrib, int value, const Logger& logger) {
    CUresult result = cuFuncSetAttribute(kernel, attrib, value);
    if (result != CUDA_SUCCESS) {
        const char* msg;
        cuGetErrorName(result, &msg);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
        RETURN_ERROR(logger, ErrorCode::Runtime, "CudaRtcKernel : failed setting function attribute (%d) : %s", static_cast<int>(result), msg);
#pragma GCC diagnostic pop
    }
    return Status();
}
