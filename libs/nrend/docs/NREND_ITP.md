# NREND Integration Test Plan

---

## 1. Introduction

### 1.1. Purpose

This document provides a set of integration test cases for the NREND library. Its purpose is to guide developers and quality assurance (QA) teams in verifying that a new version of the NREND library has been successfully integrated into their host application.

### 1.2. Scope

The scope of this document is limited to API-level integration testing. It covers the verification of the public C++ API to ensure that the library is correctly linked, configured, and performs its core functions within the target environment.

These tests are not exhaustive functional tests of every NREND feature but are designed to serve as a "smoke test" to confirm that the integration is sound and that basic rendering pipelines are operational.

### 1.3. Audience

This document is intended for software developers and QA engineers who are responsible for integrating the NREND library into a larger software system, such as a machine learning framework or a sensor simulation platform. Users are expected to have access to their application's source code and build system.

### 1.4. References

- **Package Repository:** Released NREND packages are available on the Omniverse Artifactory at <https://omnipackages.nvidia.com/packages/artifactory/nrend>.
- **Release Notes:** <https://docs.google.com/document/d/1b4XrIl7aURbQKE0IgHqJPMFpHq9R_e1Hb7pR3Opy3aQ/edit?usp=sharing>
- **Software Requirements Document (SRD):** <https://gitlab-master.nvidia.com/nrs/nre/-/blob/main/libs/nrend/docs/NREND_SRD.md>

---

## 2. Test Environment and Prerequisites

Before running the integration tests, please ensure your environment meets the following requirements.

### 2.1. Hardware

- An NVIDIA GPU with CUDA Compute Capability 7.5 (Turing) or higher. ([SRD] REQ-NFR-PORT-001)

### 2.2. Software

- **Operating System:**
  - Linux (Ubuntu 20.04 or later, x86_64 or aarch64)
  - Windows (x86_64)
- **NVIDIA Dependencies:**
  - NVIDIA CUDA Toolkit (refer to NREND release notes for the required version).
  - NVIDIA display driver that supports the required CUDA Toolkit version.
- **Compiler:** A C++17 compliant compiler.
- **NREND Library:** The specific version of the `nrend` shared library (`libnrend.so` or `nrend.dll`) and the corresponding C++ header files must be available to your application's build system. The packages can be downloaded from the [Omniverse Artifactory](https://omnipackages.nvidia.com/packages/artifactory/nrend).

**Note on Platforms:** NREND is distributed in separate packages for each supported platform (Windows x86_64, Linux x86_64, Linux aarch64). You must select the package that matches the platform you are testing on.

### 2.3. Test Assets

It is assumed that the user has access to a simple, valid scene model (e.g., a 3D Gaussian Splatting model in MsgPack format) to use as input for the rendering tests.

---

## 3. Integration Test Cases

The following test cases should be executed to validate the NREND integration.

### 3.1. Category: API Lifecycle and Configuration

#### **ITP-LC-001: Renderer Creation and Destruction**

- **Description:** This test verifies that a renderer instance can be created and subsequently destroyed without errors. It is the most fundamental check of a successful library linkage.
- **SRD Reference(s):** REQ-FR-CORE-006, REQ-NFR-REL-001
- **Steps:**
  1. In the host application, call `nrend::ErrorCode status = nrend::create(...)` with a minimal valid configuration.
  2. Check that the returned status is `nrend::ErrorCode::SUCCESS` and the renderer handle is not null.
  3. Call `nrend::ErrorCode status = nrend::destroy(...)` with the obtained renderer handle.
  4. Check that the returned status is `nrend::ErrorCode::SUCCESS`.
- **Expected Results:**
  - All API calls return `nrend::ErrorCode::SUCCESS`.
  - No crashes or memory access violations occur.
  - (Optional) Running with memory analysis tools (like Valgrind on Linux) should not report memory leaks from this sequence.

#### **ITP-LC-002: Logging Callback Registration**

- **Description:** Verifies that the library can be initialized with a user-provided logging callback.
- **SRD Reference(s):** REQ-FR-API-002
- **Steps:**
  1. Define a logging function in the host application that matches the `nrend::LoggerParameters::Callback` signature.
  2. Call `nrend::create(...)` with a valid configuration and pass a `LoggerParameters` struct with the callback pointer to trigger log messages.
- **Expected Results:**
  - The user-defined logging function is called by the NREND library with messages such as "Model ... opened" (on success) or "Cannot open model ..." (on failure).

#### **ITP-LC-003: RTC Kernel Caching**

- **Description:** Verifies that the Runtime Compilation (RTC) cache directory can be set and is populated by the library.
- **SRD Reference(s):** REQ-FR-CORE-005
- **Steps:**
  1. Choose a path for the RTC cache directory. Ensure this directory is empty or does not exist before the test.
  2. Call `nrend::setRTCCacheDirectory()` with the chosen path. This function should be called before any renderer is created.
  3. Create a renderer instance and perform at least one `nrend::render()` call to ensure RTC kernels are compiled.
  4. After the render call completes and the renderer is destroyed, inspect the filesystem at the specified cache path.
- **Expected Results:**
  - The `setRTCCacheDirectory` call returns `nrend::ErrorCode::SUCCESS`.
  - The specified cache directory is created if it did not exist.
  - The directory contains files, which are the cached RTC kernels.

---

### 3.2. Category: Core Rendering Functionality

#### **ITP-CORE-001: Basic Forward Rendering**

- **Description:** This test verifies that the `render` function can be called successfully and that it produces output.
- **SRD Reference(s):** REQ-FR-CORE-001, REQ-FR-CORE-008, REQ-FR-OUT-001, REQ-FR-OUT-002
- **Steps:**
  1. Create and configure a renderer instance.
  2. Load a simple scene model (e.g., 3D Gaussian Splatting).
  3. Allocate CUDA device memory for output buffers (radiance, depth).
  4. Create a CUDA stream.
  5. Call the `nrend::render(...)` function with the appropriate parameters, including the CUDA stream and device pointers to the output buffers.
  6. Synchronize the CUDA stream.
  7. Copy the output buffers from GPU to CPU for verification.
- **Expected Results:**
  - The `render` call returns `nrend::ErrorCode::SUCCESS`.
  - The process does not crash.
  - The data copied back from the GPU contains non-zero, plausible values for radiance and depth. The background should not be all black, and depth should not be all zero, assuming the camera is pointed at the model.

#### **ITP-CORE-002: Asynchronous Execution**

- **Description:** Verifies that API calls are non-blocking and are correctly enqueued on the provided CUDA stream.
- **SRD Reference(s):** REQ-FR-CORE-003
- **Steps:**
  1. Create a CUDA stream.
  2. Immediately before calling `nrend::render(...)`, record a CPU timestamp.
  3. Call `nrend::render(...)` on the specified CUDA stream.
  4. Immediately after the call returns, record another CPU timestamp.
  5. Synchronize the CUDA stream using `cudaStreamSynchronize`.
  6. Record a final CPU timestamp after synchronization completes.
- **Expected Results:**
  - The duration between the first two timestamps (the API call itself) should be very short, as the call should return without waiting for the GPU to finish.
  - The duration between the second and third timestamps (the synchronization) should be significantly longer, representing the actual GPU render time.

#### **ITP-CORE-003: Render Compositing**

- **Description:** This test verifies that the renderer composites its output over the existing contents of the output buffers.
- **SRD Reference(s):** REQ-FR-CORE-009
- **Steps:**
  1. Create a renderer instance and a test model.
  2. Allocate a CUDA device buffer for the radiance output.
  3. Initialize the entire radiance buffer to a solid, non-black color (e.g., red).
  4. Call `nrend::render()` to render a small object that is expected to cover only a portion of the frame.
  5. Copy the radiance buffer from the GPU to the CPU for inspection.
- **Expected Results:**
  - The `render` call returns `nrend::ErrorCode::SUCCESS`.
  - Pixels in the output image that correspond to the rendered object should show the object's appearance.
  - Pixels in the output image that are _not_ covered by the object should retain their initial value (the solid red color). They should not be cleared to a default background color.

---

### 3.3. Category: Scene and Model Handling

#### **ITP-SCENE-001: 3D Gaussian Splatting Model Rendering**

- **Description:** Verifies a scene with a 3D Gaussian Splatting model can be loaded and rendered.
- **SRD Reference(s):** REQ-FR-SCENE-001, REQ-FR-SCENE-006
- **Steps:**
  1. Configure the renderer to use a 3D Gaussian Splatting model.
  2. Load the model parameters from a MsgPack file.
  3. Render a frame.
- **Expected Results:**
  - The call succeeds.
  - The rendered output visually matches the expected appearance of the test model. If a "golden image" is available, the output should be numerically close to it ([SRD] REQ-NFR-REL-003).

#### **ITP-SCENE-002: Object Transformation**

- **Description:** Verifies that the per-frame object transformation is applied correctly.
- **SRD Reference(s):** REQ-FR-CTRL-002
- **Steps:**
  1. Render a frame of a test model with an identity transform. Save the image or a checksum.
  2. Render a second frame of the same model, but provide a `worldToObjectTransform` that translates or rotates the model.
- **Expected Results:**
  - The model appears in a different position or orientation in the second frame compared to the first, confirming the transform was applied.

#### **ITP-SCENE-003: Selective Tracked Object Rendering**

- **Description:** Verifies that a single object can be rendered from a model containing multiple tracked objects.
- **SRD Reference(s):** REQ-FR-CTRL-009
- **Steps:**
  1. Use a test model that contains at least two distinct tracked objects.
  2. Call `nrend::render` and provide parameters via CUDA device pointers to render only one of the tracked objects (e.g., by specifying a single active instance ID and its corresponding pose).
- **Expected Results:**
  - The rendered image contains only the single, specified tracked object. Other objects from the source model are not visible.

#### **ITP-SCENE-004: Instanced Rendering**

- **Description:** Verifies that a single source object can be rendered multiple times (instanced) with different poses in the same frame.
- **SRD Reference(s):** REQ-FR-CTRL-009
- **Steps:**
  1. Use a test model with at least one tracked object.
  2. Call `nrend::render` and provide parameters to render two instances of the same source object. This involves providing two different poses in the poses buffer and identifying the same source object ID for both instances.
- **Expected Results:**
  - The rendered image shows the same object appearing in two different locations, corresponding to the two unique poses provided for the instances.

---

### 3.4. Category: Per-Frame Controls

#### **ITP-CTRL-001: Tiled Rendering**

- **Description:** Verifies that the tiled rendering parameters are respected.
- **SRD Reference(s):** REQ-FR-CTRL-001
- **Steps:**
  1. Render a full frame at a specific resolution (e.g., 128x128).
  2. Render a second frame at the same full resolution, but specify a render tile (e.g., `frameTileOffset` = (0, 0), `frameTileResolution` = (64, 64)). Initialize the output buffer with a known pattern before the call.
- **Expected Results:**
  - In the second render, only the top-left 64x64 quadrant of the output buffer is modified. The rest of the buffer remains untouched. The content of the modified tile should match the corresponding quadrant from the full-frame render.

#### **ITP-CTRL-002: Custom Ray Rendering**

- **Description:** Verifies that the renderer can accept user-defined rays instead of using a sensor model.
- **SRD Reference(s):** REQ-FR-CTRL-010
- **Steps:**
  1. Allocate and populate CUDA device buffers for ray origins and directions.
  2. Call `nrend::render` with pointers to these custom ray buffers.
- **Expected Results:**
  - The render call succeeds and produces a valid image based on the provided rays.

---

## 4. Acceptance Criteria

The integration is considered successful if:

- The host application compiles and links against the NREND shared library without errors.
- All applicable test cases listed in Section 3 pass successfully.
- The rendered output is visually correct for the given test scenes, with no obvious rendering artifacts. Specifically check for:
  - Strong or odd color shifts.
  - Ringing or ghosting artifacts around objects.
  - Incorrect Z-depth compositing (e.g., objects appearing in front of others when they should be behind).
  - Flickering pixels or unstable noise.
  - Moiré patterns or other aliasing artifacts.
- No crashes, assertions, or critical errors from the NREND library are observed during the test runs.
- For users concerned with performance, it is recommended to run a baseline performance test to compare against previous integrations and ensure no major regressions have occurred ([SRD] REQ-NFR-PERF-001).

---

## 5. Issue Reporting

If any integration test fails, please report the issue to the NREND development team. Include the following information in your report:

- NREND library version.
- Host application details.
- Operating System and version.
- GPU model and NVIDIA Driver version.
- Detailed steps to reproduce the failure.
- All log output from the NREND library.
- A minimal, self-contained code example that reproduces the issue, if possible.
