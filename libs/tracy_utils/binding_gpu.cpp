// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "tracy_plot_types.h"
#include "tracy_utils.h"
#ifdef TRACY_HAS_GPU
#include "tracy_utils_cuda.cuh"
#include <cuda_runtime.h>
#endif

namespace py = pybind11;

namespace nre {
namespace tracy_utils {

class PyScopedTracyZone {
public:
    explicit PyScopedTracyZone(const std::string& name, uint32_t color = 0)
        : zone(name.c_str(), color) {}

    void set_text(const std::string& text) {
        zone.setText(text.c_str());
    }

    void set_name(const std::string& name) {
        zone.setName(name.c_str());
    }

    void __enter__() {}

    void __exit__(py::object exc_type, py::object exc_value, py::object traceback) {}

private:
    ScopedTracyZone zone;
};

PYBIND11_MODULE(tracy_utils_gpu_py, m) {
    m.doc() = "Tracy profiler Python bindings (GPU variant)";

    // Check if GPU profiling should be disabled
#ifdef TRACY_HAS_GPU
    const char* disable_gpu = std::getenv("TRACY_NO_GPU");
    if (disable_gpu && std::string(disable_gpu) == "1") {
        std::cerr << "[Tracy] GPU profiling disabled via TRACY_NO_GPU=1" << std::endl;
    }
#endif

    // Plot type enum
    py::enum_<PlotType>(m, "PlotType")
        .value("GPU_MEM_ALLOCATED_MB", PLOT_GPU_MEM_ALLOCATED_MB)
        .value("GPU_MEM_RESERVED_MB", PLOT_GPU_MEM_RESERVED_MB)
        .value("GPU_MEM_DURING_RENDER_MB", PLOT_GPU_MEM_DURING_RENDER_MB)
        .value("CPU_MEMORY_MB", PLOT_CPU_MEMORY_MB)
        .export_values();

    // TracyProfiler functions (not exposing the class directly to avoid destructor issues)
    m.def(
        "initialize", [](bool enabled) {
            TracyProfiler::getInstance().initialize(enabled);
        },
        py::arg("enabled") = false, "Initialize Tracy profiler");

    m.def(
        "is_available", []() {
#ifdef TRACY_ENABLE
            return true;
#else
            return false;
#endif
        },
        "Check if Tracy is available");

    // Debug function to check defines
    m.def(
        "debug_defines", []() {
            std::string result = "Defines: ";
#ifdef TRACY_ENABLE
            result += "TRACY_ENABLE=1 ";
#else
        result += "TRACY_ENABLE=0 ";
#endif
#ifdef TRACY_HAS_GPU
            result += "TRACY_HAS_GPU=1 ";
#else
        result += "TRACY_HAS_GPU=0 ";
#endif
#ifdef TRACY_NO_GPU
            result += "TRACY_NO_GPU=1 ";
#else
        result += "TRACY_NO_GPU=0 ";
#endif
            return result;
        },
        "Debug function to show compile-time defines");
    m.def(
        "is_connected", []() {
#ifdef TRACY_ENABLE
            return tracy::GetProfiler().IsConnected();
#else
            return false;
#endif
        },
        "Check if Tracy is connected to a profiler");
    m.def(
        "mark_frame", [](const char* name) {
            TracyProfiler::getInstance().markFrame(name);
        },
        py::arg("name") = nullptr, "Mark frame boundary");
    m.def(
        "message", [](const std::string& text) {
            TracyProfiler::getInstance().message(text);
        },
        py::arg("text"), "Send message to Tracy");
    m.def(
        "plot", [](PlotType plotType, double value) {
            TracyProfiler::getInstance().plot(static_cast<int>(plotType), value);
        },
        py::arg("plot_type"), py::arg("value"), "Plot a value in Tracy");

    // Scoped zone for context manager
    py::class_<PyScopedTracyZone>(m, "ScopedTracyZone")
        .def(py::init<const std::string&, uint32_t>(),
             py::arg("name"), py::arg("color") = 0,
             "Create a scoped Tracy zone")
        .def("set_text", &PyScopedTracyZone::set_text,
             py::arg("text"),
             "Set zone text")
        .def("set_name", &PyScopedTracyZone::set_name,
             py::arg("name"),
             "Set zone name")
        .def("__enter__", &PyScopedTracyZone::__enter__)
        .def("__exit__", &PyScopedTracyZone::__exit__);

    // Colors
    py::module_ colors     = m.def_submodule("colors", "Tracy zone colors");
    colors.attr("RED")     = py::int_(TracyColors::Red);
    colors.attr("GREEN")   = py::int_(TracyColors::Green);
    colors.attr("BLUE")    = py::int_(TracyColors::Blue);
    colors.attr("YELLOW")  = py::int_(TracyColors::Yellow);
    colors.attr("MAGENTA") = py::int_(TracyColors::Magenta);
    colors.attr("CYAN")    = py::int_(TracyColors::Cyan);
    colors.attr("ORANGE")  = py::int_(TracyColors::Orange);
    colors.attr("PURPLE")  = py::int_(TracyColors::Purple);

    // GPU context functions
    m.def(
        "initialize_gpu_context", [](const std::string& name, int stream) {
#ifdef TRACY_HAS_GPU
            const char* disable_gpu = std::getenv("TRACY_NO_GPU");
            if (!disable_gpu || std::string(disable_gpu) != "1") {
                // Convert integer stream ID to cudaStream_t
                // Note: Stream 0 is the default stream. Other values should be valid
                // CUDA stream handles cast to integers (use with caution)
                cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
                initializeGlobalGpuContext(name.c_str(), cuda_stream);
            }
#endif
        },
        py::arg("name") = "CUDA", py::arg("stream") = 0, "Initialize GPU context for Tracy profiling. Stream 0 is the default stream, other values should be valid CUDA stream handles.");

    m.def(
        "destroy_gpu_context", []() {
#ifdef TRACY_HAS_GPU
            destroyGlobalGpuContext();
#endif
        },
        "Destroy GPU context");

    // GPU collection functions - use different names to avoid overload conflicts
    m.def(
        "collect_gpu", []() {
#ifdef TRACY_HAS_GPU
            // Get the global GPU context and collect
            auto* ctx = getGlobalGpuContext(0);
            if (ctx) {
                ctx->collect();
            }
#endif
        },
        "Collect GPU profiling data from default stream");

    m.def(
        "collect_gpu_stream", [](intptr_t stream_id) {
#ifdef TRACY_HAS_GPU
            // Convert integer stream ID to cudaStream_t
            // Note: Stream 0 is the default stream. Other values should be valid
            // CUDA stream handles cast to integers (use with caution)
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_id));
            auto* ctx                = getGlobalGpuContext(cuda_stream);
            if (ctx) {
                ctx->collect();
            }
#endif
        },
        py::arg("stream"), "Collect GPU profiling data from specific stream. Stream 0 is the default stream, other values should be valid CUDA stream handles.");

    m.def(
        "collect_all_gpu", []() {
#ifdef TRACY_HAS_GPU
            collectAllGpuContexts();
#endif
        },
        "Collect GPU profiling data from all streams");

    m.def(
        "is_gpu_profiling_available", []() {
#ifdef TRACY_HAS_GPU
            const char* disable_gpu = std::getenv("TRACY_NO_GPU");
            return !disable_gpu || std::string(disable_gpu) != "1";
#else
            return false;
#endif
        },
        "Check if GPU profiling support was compiled in");

    // Note: We don't expose get_profiler to avoid destructor issues with singleton
}

} // namespace tracy_utils
} // namespace nre
