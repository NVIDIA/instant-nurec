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

// Enable Tracy profiling if TRACY_ENABLE is defined
#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#else
// Define empty macros when Tracy is disabled
#define ZoneScoped
#define ZoneScopedN(name)
#define ZoneScopedC(color)
#define ZoneScopedNC(name, color)
#define ZoneText(text, size)
#define ZoneName(text, size)
#define FrameMark
#define FrameMarkNamed(name)
#define FrameMarkStart(name)
#define FrameMarkEnd(name)
#define TracyAlloc(ptr, size)
#define TracyFree(ptr)
#define TracyAllocS(ptr, size, depth)
#define TracyFreeS(ptr, depth)
#define TracyAllocN(ptr, size, name)
#define TracyFreeN(ptr, name)
#define TracyAllocNS(ptr, size, depth, name)
#define TracyFreeNS(ptr, depth, name)
#define TracyMessage(txt, size)
#define TracyMessageL(txt)
#define TracyAppInfo(txt, size)
#define TracyValue(name, value)
#define TracyValueS(name, value, depth)
#define TracyPlot(name, val)
#define TracyPlotConfig(name, type, step, fill, color)
#endif

namespace nre {
namespace tracy_utils {

/**
 * TracyProfiler class provides a wrapper around Tracy profiler functionality
 * with support for on-demand profiling and file saving
 */
class TracyProfiler {
public:
    static TracyProfiler& getInstance();

    // Initialize Tracy profiler
    void initialize(bool enabled = false);

    // Mark frame boundaries
    void markFrame(const char* name = nullptr);

    // Send messages to Tracy
    void message(const std::string& text);

    // Plot values
    void plot(int plotType, double value);

private:
    TracyProfiler();
    ~TracyProfiler();
    TracyProfiler(const TracyProfiler&)            = delete;
    TracyProfiler& operator=(const TracyProfiler&) = delete;
};

/**
 * RAII helper for profiling a scope
 */
class ScopedTracyZone {
public:
    explicit ScopedTracyZone(const char* name, uint32_t color = 0);
    ~ScopedTracyZone();

    void setText(const char* text);
    void setName(const char* name);

private:
#ifdef TRACY_ENABLE
    TracyCZoneCtx zone;
#else
    void* zone;
#endif
};

// Helper macros for easy Tracy integration
// Use __LINE__ to create unique variable names and avoid conflicts in nested usage
#define NRE_TRACY_ZONE() nre::tracy_utils::ScopedTracyZone TRACY_CONCAT(_tracy_zone_, __LINE__)(__FUNCTION__)
#define NRE_TRACY_ZONE_N(name) nre::tracy_utils::ScopedTracyZone TRACY_CONCAT(_tracy_zone_, __LINE__)(name)
#define NRE_TRACY_ZONE_C(color) nre::tracy_utils::ScopedTracyZone TRACY_CONCAT(_tracy_zone_, __LINE__)(__FUNCTION__, color)
#define NRE_TRACY_ZONE_NC(name, color) nre::tracy_utils::ScopedTracyZone TRACY_CONCAT(_tracy_zone_, __LINE__)(name, color)

// Helper macro for token concatenation
#define TRACY_CONCAT(a, b) TRACY_CONCAT_INNER(a, b)
#define TRACY_CONCAT_INNER(a, b) a##b

// Common colors for profiling zones
namespace TracyColors {
constexpr uint32_t Red     = 0xFF0000;
constexpr uint32_t Green   = 0x00FF00;
constexpr uint32_t Blue    = 0x0000FF;
constexpr uint32_t Yellow  = 0xFFFF00;
constexpr uint32_t Magenta = 0xFF00FF;
constexpr uint32_t Cyan    = 0x00FFFF;
constexpr uint32_t Orange  = 0xFF8800;
constexpr uint32_t Purple  = 0x8800FF;
} // namespace TracyColors

} // namespace tracy_utils
} // namespace nre
