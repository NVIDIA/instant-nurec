# Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.26)

include(FetchContent)

project(
    nrend
	DESCRIPTION "Neural Renderer"
	LANGUAGES C CXX CUDA
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(NREND_EXTERNAL_FMT "" CACHE STRING "If non-empty, the `fmt` target is supplied externally with the given name.")

option(NREND_OBFUSCATE_HEADERS "Encode headers with obfuscator" ON)
option(NREND_SEPARATE_DEBUG_INFO "Strip and separate debug info from output library" OFF)
set(NREND_SYMBOL_INSTALL_DIR "lib" CACHE STRING "Subdirectory where symbols are installed")

set(NREND_OBFUSCATOR_PATH "" CACHE STRING "Optional path to obfuscator compiled for host, useful when cross-compiling")

message(STATUS "NREND kernel obfuscation: ${NREND_OBFUSCATE_HEADERS}")
message(STATUS "NREND symbol install dir: ${NREND_SYMBOL_INSTALL_DIR}")

find_package(CUDAToolkit REQUIRED)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_LINK_STATIC_TOOLKIT ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

include(FetchContent)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set the build directory inside NREND dependencies, specified by: Dockerfile_nrend_deps.build; build_windows.bat
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(NREND_DEPS_BUILD_DIR "build")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(NREND_DEPS_BUILD_DIR "build-cross")
endif()

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT MSVC)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
endif()

###############################################################################
# CUDA compiler setup
###############################################################################

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
endif()
list(APPEND CUDA_NVCC_FLAGS "-std=c++17")
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF_OPERATORS__")
list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF_CONVERSIONS__")
list(APPEND CUDA_NVCC_FLAGS "-U__CUDA_NO_HALF2_OPERATORS__")

list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

string(REPLACE "-virtual" "" MIN_GPU_ARCH "${MIN_GPU_ARCH}")

message(STATUS "Targeting CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (MIN_GPU_ARCH LESS_EQUAL 70)
	message(WARNING
		"Fully fused MLPs do not support GPU architectures of 70 or less. "
		"Falling back to CUTLASS MLPs. Remove GPU architectures 70 and lower "
		"to allow maximum performance"
	)
endif()

###############################################################################
# Dependencies
###############################################################################

# get fmt and json from tiny-cuda-nn
if (NOT DEFINED NREND_TCNN_DIR)
	set(NREND_TCNN_DIR "deps/tiny_cuda_nn")
endif()
if (NOT DEFINED NREND_FMT_DIR)
	set(NREND_FMT_DIR "${NREND_TCNN_DIR}/dependencies/fmt")
endif()
if (NOT DEFINED NREND_JSON_DIR)
	set(NREND_JSON_DIR "${NREND_TCNN_DIR}/dependencies")
endif()

if (NOT DEFINED SLANG_DIR)
    set(SLANG_DIR "deps/slang/")
endif()

file(TO_CMAKE_PATH ${NREND_TCNN_DIR} NREND_TCNN_DIR)
file(TO_CMAKE_PATH ${NREND_FMT_DIR} NREND_FMT_DIR)
file(TO_CMAKE_PATH ${NREND_JSON_DIR} NREND_JSON_DIR)
file(TO_CMAKE_PATH ${SLANG_DIR} SLANG_DIR)

# Optix
FetchContent_Declare(
  optix-dev
  URL      https://github.com/NVIDIA/optix-dev/archive/refs/tags/v8.0.0.tar.gz
  URL_HASH SHA256=b32e74c9f5c13549ff3a9760076271b5b6ec28f93fe6a8dd0bde74d7e5c58e05
)
FetchContent_MakeAvailable(optix-dev)
set(OPTIX_DEV_DIR ${optix-dev_SOURCE_DIR})

# for experimental::filesystem
if (NOT MSVC)
list(APPEND NREND_LIBRARIES stdc++fs)
endif()

# do we need this?
find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

###############################################################################
# Run-Time Resources
###############################################################################

if (NREND_OBFUSCATE_HEADERS)
    if (NREND_OBFUSCATOR_PATH STREQUAL "")
	    add_executable(obfuscator ${NREND_TCNN_DIR}/src/obfuscation/obfuscation.cpp)
	    if (NOT MSVC)
		    target_link_libraries(obfuscator PUBLIC stdc++fs)
	    endif()
        set(NREND_OBFUSCATOR_PATH ${CMAKE_CURRENT_BINARY_DIR})
        set(OBFUSCATOR_TARGET obfuscator)
    endif()

    message(STATUS "NREND obfuscator path: ${NREND_OBFUSCATOR_PATH}")

	file(GLOB_RECURSE NREND_UNOBFUSCATED_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/kernels/*")
	list(APPEND NREND_UNOBFUSCATED_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/renderer/renderParameters.h"
	                                       "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/renderer/grtOptixRenderParameters.h"
	                                       "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/renderer/gutRendererParameters.h"
	                                       "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/sensors/sensorsModel.h"
	                                       "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/sensors/sensors.h"
	                                       "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/utils/nreVec.h"
	                                       "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/utils/nrePose.h")

	list(TRANSFORM NREND_UNOBFUSCATED_KERNELS REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/obfuscated/include/" OUTPUT_VARIABLE NREND_KERNELS)
	list(TRANSFORM NREND_UNOBFUSCATED_KERNELS REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/include" "include" OUTPUT_VARIABLE RELATIVE_NREND_KERNELS)

	add_custom_command(
		OUTPUT ${NREND_KERNELS}
		COMMAND ${NREND_OBFUSCATOR_PATH}/obfuscator "${CMAKE_CURRENT_BINARY_DIR}/obfuscated" ${RELATIVE_NREND_KERNELS}
		DEPENDS ${OBFUSCATOR_TARGET} ${NREND_TCNN_DIR}/src/obfuscation/obfuscation.cpp ${NREND_TCNN_DIR}/src/obfuscation/obfuscation.h ${NREND_UNOBFUSCATED_KERNELS}
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	)
else()
	file(GLOB_RECURSE NREND_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/kernels/*")
    list(APPEND NREND_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/sensors/sensorsModel.h"
                              "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/sensors/sensors.h"
                              "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/utils/nreVec.h"
                              "${CMAKE_CURRENT_SOURCE_DIR}/include/nrend/utils/nrePose.h")
endif()

if(EXISTS "${NREND_TCNN_DIR}/dependencies/cmrc/CMakeRC.cmake")
    include("${NREND_TCNN_DIR}/dependencies/cmrc/CMakeRC.cmake")
    list(APPEND NREND_DEFINITIONS -DTCNN_CMRC -DTCNN_PARAMS_UNALIGNED)

    cmrc_add_resource_library(nrend_pmrc_resource NAMESPACE nrend)
    if (NREND_OBFUSCATE_HEADERS)
        cmrc_add_resources(nrend_pmrc_resource WHENCE "${CMAKE_CURRENT_BINARY_DIR}/obfuscated/include" ${NREND_KERNELS})
    else()
        cmrc_add_resources(nrend_pmrc_resource WHENCE "${CMAKE_CURRENT_SOURCE_DIR}/include" ${NREND_KERNELS})
    endif()

    # adding optix headers to the resources
    file(GLOB_RECURSE OPTIX_HEADERS "${OPTIX_DEV_DIR}/include/*")
    cmrc_add_resources(nrend_pmrc_resource WHENCE "${OPTIX_DEV_DIR}/include" ${OPTIX_HEADERS})

    list(APPEND NREND_LIBRARIES nrend_pmrc_resource)
else()
    message(FATAL_ERROR "cmrc module not found at ${NREND_TCNN_DIR}/dependencies/cmrc, aborting build")
endif()

###############################################################################
# Program
###############################################################################

list(APPEND NREND_DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})
list(APPEND NREND_INCLUDE_DIRECTORIES
    "include"
    "src/obfuscation/"
    "${NREND_TCNN_DIR}/include"
    "${NREND_FMT_DIR}/include"
	"${NREND_JSON_DIR}"
    "${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/${CMAKE_BUILD_TYPE}/include"
    "${OPTIX_DEV_DIR}/include"
)
if (NOT MSVC)
	list(APPEND NREND_LIBRARIES ${CMAKE_DL_LIBS})
endif()
list(APPEND NREND_SOURCES
    src/cudaBuffer.cpp
    src/cudaTexture.cpp
    src/cudaRtcKernel.cpp
    src/rtcKernelConfig.cpp
    src/slangRtcKernel.cpp
    src/cudaKernelResources.cpp
    src/renderer.cpp
    src/nreRenderer.cpp
    src/status.cpp
    src/obfuscation/obfuscation.cpp
    src/optixAccelerationStructure.cpp
    src/optixRtcPipeline.cpp
    src/iNRenderer.cpp
    src/ngpOccupancyGrid.cu
    src/grtOptixRenderer.cpp
    src/grutRenderer.cpp
    src/gutRenderer.cu
    src/ngpRenderer.cu
    src/nreShGaussianModel.cu
    src/nreGaussianCompositeModel.cu
    src/nreDynamicShGaussianModel.cu
    src/nreModel.cu
    src/nreLegacyRenderer.cu
)

# TODO: When Slang STATIC is fixed (to not need to add those .a files below) remove .a files below

add_library(nrend SHARED ${NREND_SOURCES})
set_target_properties(nrend PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(nrend PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
set_target_properties(nrend PROPERTIES
    PDB_NAME nrend
    COMPILE_PDB_NAME nrend
    COMPILE_PDB_OUTPUT_DIR ${PROJECT_BINARY_DIR})
target_compile_definitions(nrend PUBLIC ${NREND_DEFINITIONS})
target_compile_definitions(nrend PRIVATE SLANG_STATIC)
target_compile_options(nrend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(nrend PUBLIC ${NREND_INCLUDE_DIRECTORIES})
target_link_directories(nrend PUBLIC ${NREND_LINK_DIRECTORIES})
target_link_libraries(nrend
    PUBLIC
        ${NREND_LIBRARIES}
    PRIVATE
        CUDA::cuda_driver
        CUDA::cudart_static
        CUDA::nvrtc_static)
if (MSVC)
    target_link_libraries(nrend
        PRIVATE
            ${NREND_TCNN_DIR}/${NREND_DEPS_BUILD_DIR}/dependencies/fmt/fmt.lib
            ${NREND_TCNN_DIR}/${NREND_DEPS_BUILD_DIR}/tiny-cuda-nn.lib
            ${NREND_TCNN_DIR}/${NREND_DEPS_BUILD_DIR}/tiny-cuda-nn-resources.lib
            ${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/${CMAKE_BUILD_TYPE}/lib/slang-compiler.lib
            ${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/${CMAKE_BUILD_TYPE}/lib/compiler-core.lib
            ${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/${CMAKE_BUILD_TYPE}/lib/core.lib
            ${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/external/miniz/miniz.lib
            ${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/external/lz4/build/cmake/lz4.lib)
else()
    target_link_libraries(nrend
        PRIVATE
            ${NREND_TCNN_DIR}/${NREND_DEPS_BUILD_DIR}/dependencies/fmt/libfmt.a
            ${NREND_TCNN_DIR}/${NREND_DEPS_BUILD_DIR}/libtiny-cuda-nn.a
            ${NREND_TCNN_DIR}/${NREND_DEPS_BUILD_DIR}/libtiny-cuda-nn-resources.a
            ${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/${CMAKE_BUILD_TYPE}/lib/libslang.a
            ${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/${CMAKE_BUILD_TYPE}/lib/libcompiler-core.a
            ${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/${CMAKE_BUILD_TYPE}/lib/libcore.a
            ${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/external/miniz/libminiz.a
            ${SLANG_DIR}/${NREND_DEPS_BUILD_DIR}/external/lz4/build/cmake/liblz4.a)
endif()

if (NREND_SEPARATE_DEBUG_INFO AND UNIX AND (CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo"))
    add_custom_command(TARGET nrend POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} --only-keep-debug $<TARGET_FILE:nrend> $<TARGET_FILE:nrend>.debug
        COMMAND ${CMAKE_STRIP} --strip-debug --strip-unneeded $<TARGET_FILE:nrend>
        COMMAND ${CMAKE_OBJCOPY} --add-gnu-debuglink=$<TARGET_FILE:nrend>.debug $<TARGET_FILE:nrend>
        COMMENT "Separating debug symbols for NREND")
endif()

file(GLOB NREND_PUBLIC_HEADERS "include/nrend/*.h")

install(FILES ${NREND_PUBLIC_HEADERS}
        DESTINATION include/nrend)
install(TARGETS nrend
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
if (MSVC)
    install(FILES $<TARGET_PDB_FILE:nrend>
            DESTINATION ${NREND_SYMBOL_INSTALL_DIR}
            CONFIGURATIONS Debug RelWithDebInfo)
endif()
if (NREND_SEPARATE_DEBUG_INFO AND UNIX)
    install(FILES $<TARGET_FILE:nrend>.debug
            DESTINATION ${NREND_SYMBOL_INSTALL_DIR}
            CONFIGURATIONS Debug RelWithDebInfo)
endif()
