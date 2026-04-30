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

#include "tracy_plot_types.h"
#include "tracy_utils.h"

namespace py = pybind11;

namespace nre {
namespace tracy_utils {

class PyScopedTracyZone {
public:
    explicit PyScopedTracyZone(const std::string& name, uint32_t color = 0) {}
    void set_text(const std::string& text) {}
    void set_name(const std::string& name) {}
    void __enter__() {}
    void __exit__(py::object exc_type, py::object exc_value, py::object traceback) {}
};

PYBIND11_MODULE(tracy_utils_disabled_py, m) {
    m.doc() = "Tracy profiler Python bindings (disabled variant)";

    // Plot type enum (stub)
    py::enum_<PlotType>(m, "PlotType")
        .value("GPU_MEM_ALLOCATED_MB", PLOT_GPU_MEM_ALLOCATED_MB)
        .value("GPU_MEM_RESERVED_MB", PLOT_GPU_MEM_RESERVED_MB)
        .value("GPU_MEM_DURING_RENDER_MB", PLOT_GPU_MEM_DURING_RENDER_MB)
        .value("CPU_MEMORY_MB", PLOT_CPU_MEMORY_MB)
        .export_values();

    // TracyProfiler functions (all no-ops)
    m.def(
        "initialize", [](bool enabled) { /* no-op */ },
        py::arg("enabled") = false, "Initialize Tracy profiler");
    m.def(
        "is_available", []() { return false; }, "Check if Tracy is available");

    // Debug function to check defines
    m.def(
        "debug_defines", []() {
            return std::string("Defines: TRACY_ENABLE=0 TRACY_HAS_GPU=0 TRACY_NO_GPU=0");
        },
        "Debug function to show compile-time defines");
    m.def(
        "is_connected", []() { return false; }, "Check if Tracy is connected to a profiler");
    m.def(
        "mark_frame", [](const char* name) { /* no-op */ },
        py::arg("name") = nullptr, "Mark frame boundary");
    m.def(
        "message", [](const std::string& text) { /* no-op */ },
        py::arg("text"), "Send message to Tracy");
    m.def(
        "plot", [](PlotType plotType, double value) { /* no-op */ },
        py::arg("plot_type"), py::arg("value"), "Plot a value in Tracy");

    // Scoped zone for context manager (stub)
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

    // Colors (stub)
    py::module_ colors     = m.def_submodule("colors", "Tracy zone colors");
    colors.attr("RED")     = py::int_(0);
    colors.attr("GREEN")   = py::int_(0);
    colors.attr("BLUE")    = py::int_(0);
    colors.attr("YELLOW")  = py::int_(0);
    colors.attr("MAGENTA") = py::int_(0);
    colors.attr("CYAN")    = py::int_(0);
    colors.attr("ORANGE")  = py::int_(0);
    colors.attr("PURPLE")  = py::int_(0);

    // GPU context functions (all no-ops)
    m.def(
        "initialize_gpu_context", [](const std::string& name, int stream) { /* no-op */ },
        py::arg("name") = "CUDA", py::arg("stream") = 0,
        "Initialize GPU context for Tracy profiling");
    m.def(
        "destroy_gpu_context", []() { /* no-op */ }, "Destroy GPU context");
    // GPU collection functions - use different names to avoid overload conflicts
    m.def(
        "collect_gpu", []() { /* no-op */ }, "Collect GPU profiling data from default stream");
    m.def(
        "collect_gpu_stream", [](intptr_t stream_id) { /* no-op */ },
        py::arg("stream"), "Collect GPU profiling data from specific stream");
    m.def(
        "collect_all_gpu", []() { /* no-op */ }, "Collect GPU profiling data from all streams");
    m.def(
        "is_gpu_profiling_available", []() { return false; },
        "Check if GPU profiling support was compiled in");
}

} // namespace tracy_utils
} // namespace nre
