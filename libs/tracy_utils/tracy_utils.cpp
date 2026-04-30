// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "tracy_utils.h"
#include "tracy_plot_types.h"

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#endif

#include <cstring>
#include <iostream>

namespace nre {
namespace tracy_utils {

TracyProfiler& TracyProfiler::getInstance() {
    static TracyProfiler instance;
    return instance;
}

TracyProfiler::TracyProfiler() = default;

TracyProfiler::~TracyProfiler() = default;

void TracyProfiler::initialize(bool enabled) {
    // Tracy initialization is handled automatically by Tracy macros

#ifdef TRACY_ENABLE
    if (enabled) {
        // For TRACY_ON_DEMAND, we need to trigger Tracy to start
        // This is done by using Tracy macros which will initialize the profiler
        {
            ZoneScopedN("TracyInit"); // Create a scoped zone to trigger initialization
        }

        // Also create a frame mark to ensure full initialization
        FrameMark;

        // Send a message to ensure the connection is established
        TracyMessage("Tracy profiler started", 23);
    }
#else
    if (enabled) {
        std::cerr << "Warning: Tracy profiler requested but not compiled with TRACY_ENABLE" << std::endl;
    }
#endif
}

void TracyProfiler::markFrame(const char* name) {
#ifdef TRACY_ENABLE
    if (name) {
        FrameMarkNamed(name);
    } else {
        FrameMark;
    }
#endif
}

void TracyProfiler::message(const std::string& text) {
#ifdef TRACY_ENABLE
    TracyMessage(text.c_str(), text.size());
#endif
}

void TracyProfiler::plot(int plotType, double value) {
#ifdef TRACY_ENABLE
    if (plotType >= 0 && plotType < PLOT_TYPE_COUNT) {
        TracyPlot(PLOT_NAMES[plotType], value);
    } else {
        TracyPlot("unknown_plot", value);
    }
#endif
}

// ScopedTracyZone implementation
ScopedTracyZone::ScopedTracyZone(const char* name, uint32_t color) {
#ifdef TRACY_ENABLE
    // For runtime zone names, allocate the source location
    // Use the zone name as the function name for proper flame graph display
    const char* funcName = name ? name : "Zone";
    uint64_t srcLoc      = ___tracy_alloc_srcloc(
        (uint32_t)__LINE__,
        __FILE__,
        strnlen(__FILE__, 4096),
        funcName, // Use zone name as function name for flame graph
        strnlen(funcName, 256),
        color);
    zone = ___tracy_emit_zone_begin_alloc(srcLoc, true);

    // Also set the zone name for timeline display
    if (zone.active && name) {
        ___tracy_emit_zone_name(zone, name, strnlen(name, 256));
    }
#else
    zone = nullptr;
#endif
}

ScopedTracyZone::~ScopedTracyZone() {
#ifdef TRACY_ENABLE
    if (zone.active) {
        ___tracy_emit_zone_end(zone);
    }
#endif
}

void ScopedTracyZone::setText(const char* text) {
#ifdef TRACY_ENABLE
    if (zone.active && text) {
        ___tracy_emit_zone_text(zone, text, strnlen(text, 4096));
    }
#endif
}

void ScopedTracyZone::setName(const char* name) {
#ifdef TRACY_ENABLE
    if (zone.active && name) {
        ___tracy_emit_zone_name(zone, name, strnlen(name, 256));
    }
#endif
}

} // namespace tracy_utils
} // namespace nre
