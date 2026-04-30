# NREND Software Requirements Document

---

## 1. Introduction

### 1.1. Purpose

This document specifies the software requirements for the NREND library. It is intended for project managers, software developers, and quality assurance (QA) teams involved in the development, integration, and testing of the library. It provides a detailed description of the library's functionality, constraints, and interfaces to serve as a foundation for development and verification.

### 1.2. Scope

The NREND library is a high-performance, GPU-accelerated C++ library for differentiable neural rendering. Its primary scope is to generate realistic, physically-based sensor data by rendering 3D scenes represented by advanced neural models (e.g., NeRF, 3D Gaussian Splatting). It is designed to be integrated as a core component within larger machine learning and simulation frameworks.

The scope of this document covers the library's C++ API, its core rendering capabilities, the supported scene and sensor models, and its performance and platform requirements. It does not cover the specifics of the Python bindings or the internal implementation details of the underlying deep learning frameworks.

### 1.3. Overview

NREND provides a bridge between neural 3D scene representations and the 2D sensor data required to train and validate modern AI perception models. Its key features include a differentiable rendering pipeline for gradient-based optimization, support for state-of-the-art neural models, comprehensive sensor simulation capabilities, and a high-performance, asynchronous, GPU-native architecture. The library is configured and controlled through a C-style API, with data and parameters passed via efficient binary formats and direct GPU device pointers.

Compared to renderers implemented in high-level languages like Python with PyTorch, NREND's highly optimized, compiled C++/CUDA architecture offers a significant performance advantage. This speed is not merely an incremental improvement; it enables use cases that are often infeasible with slower renderers. The performance of NREND allows for real-time rendering applications, drastically accelerates machine learning model training cycles, and, as a result, can lead to a significant reduction in computational costs.

---

## 2. Overall Description

### 2.1. Product Perspective

NREND is a specialized software component, not a standalone application. It is designed to be a dependency for larger systems, such as:

- **Machine Learning Training Frameworks:** Integrated into frameworks like PyTorch or TensorFlow, it acts as the differentiable rendering layer that allows for end-to-end training of neural scene representations from image data.
- **Sensor Simulation Platforms:** Used within simulators for autonomous vehicles or robotics to generate high-fidelity, synthetic data for training and validating perception algorithms.
- **3D Content Creation Tools:** Incorporated into digital content creation (DCC) tools to provide advanced, neural-based rendering and visualization capabilities.

### 2.2. Product Functions

The major functions of the NREND library are:

- **Renderer Lifecycle Management:** Creating, configuring, and destroying renderer instances through the C++ API.
- **Differentiable Rendering:** Performing a forward rendering pass to generate sensor data (`render`) and a backward pass to compute gradients for training (`renderBackward`).
- **Model Parameterization:** Dynamically loading, updating, and managing the parameters of neural models on the GPU.
- **Asynchronous Execution:** All GPU operations are enqueued on user-provided CUDA streams, enabling non-blocking execution and efficient pipelining with other compute tasks.

### 2.3. User Characteristics

The intended users of the NREND library include:

- **Machine Learning Engineers/Researchers:** Individuals who require a high-performance differentiable renderer to develop and train novel neural scene representations.
- **Simulation Engineers:** Developers who need to generate large-scale, high-fidelity synthetic sensor data for training and testing perception systems.
- **C++ Developers:** Programmers responsible for integrating the NREND library into larger C++ software applications and frameworks.
  Users are expected to have expertise in C++, CUDA programming, and an understanding of 3D graphics and deep learning concepts.

### 2.4. General Constraints

- **Hardware Constraint:** The library requires an NVIDIA GPU that supports a specific minimum CUDA compute capability.
- **Software Constraint:** Development and execution depend on the NVIDIA CUDA toolkit, a C++17 compliant compiler, and potentially other third-party libraries defined by the build system.
- **Platform Constraint:** The library must be compilable and fully functional on its target operating systems (e.g., Linux, Windows).

---

## 3. Specific Requirements

### 3.1. Functional Requirements

#### 3.1.1. Core Rendering Engine

- **REQ-FR-CORE-001:** The library **shall** provide a forward rendering function (`render`) that generates sensor-realistic views of a scene, producing outputs such as radiance, depth, and instance IDs.
- **REQ-FR-CORE-002:** All computationally intensive rendering and data processing operations **shall** be executed on a target NVIDIA GPU via the CUDA programming model.
- **REQ-FR-CORE-003:** The library **shall** operate asynchronously. All GPU command and memory transfer operations **shall** be enqueued onto a user-specified CUDA stream (`DeviceQueueHandle`).
- **REQ-FR-CORE-004:** The library **shall** utilize Runtime Compilation (RTC) of CUDA C++ kernels to generate specialized, high-performance code paths based on the specific model, sensor, and rendering configuration at runtime.
- **REQ-FR-CORE-005:** To improve startup performance, the library **shall** support caching of runtime-compiled kernels. The user **shall** be able to specify a directory for the RTC cache via the `setRTCCacheDirectory` function.
- **REQ-FR-CORE-006:** The library **shall** provide functions for the full lifecycle management of a renderer instance, including `create` and `destroy`.
- **REQ-FR-CORE-007:** The library **shall** accept initial model and renderer configuration via the MsgPack binary serialization format.
- **REQ-FR-CORE-008:** For core rendering loops, the library **shall** accept all per-ray input data and write all per-pixel output data to user-provided CUDA device pointers to eliminate CPU-GPU data transfer bottlenecks.
- **REQ-FR-CORE-009:** The output buffers provided to the render function (e.g., for radiance, depth) **shall** be treated as both input and output. This allows the rendering process to selectively update pixels and composite new results over existing buffer content.

#### 3.1.2. Scene Representation Models

- **REQ-FR-SCENE-001:** The library **shall** support rendering of scenes represented by 3D Gaussian Splatting models. This includes modeling view-dependent appearance using Spherical Harmonics.
- **REQ-FR-SCENE-002:** The library **shall** support rendering of dynamic scenes where individual model primitives (e.g., Gaussians) have time-dependent properties.
- **REQ-FR-SCENE-003:** The library **shall** support rendering of implicit neural scene representations, including Neural Radiance Fields (NeRF).
- **REQ-FR-SCENE-004:** The library **shall** support the creation and rendering of composite scenes that combine multiple, potentially different, model types into a single renderable entity.
- **REQ-FR-SCENE-005:** The library **shall** support the use of dedicated background models. This includes, but is not limited to: a solid color background, a textured sky environment map (e.g., HDR cubemap or equirectangular map), and a network-based sky model.
- **REQ-FR-SCENE-006:** The data models for scene representations **shall** be compatible with the schema of assets generated by the NuRec (neural reconstruction engine) software. The library consumes this data via the MsgPack format; it does not directly parse USDZ files.
- **REQ-FR-SCENE-007:** The library **shall** provide backward compatibility for loading and rendering scene models created with previous versions of the software, ensuring that older assets remain usable with new library updates.

#### 3.1.3. Per-Frame Rendering Controls

The library **shall** allow users to control the rendering of each frame via parameters passed to the `render` function. This includes the following capabilities, controlled via the `RenderParameters` struct and direct function arguments:

- **REQ-FR-CTRL-001:** The ability to specify the resolution of the full output frame (`frameResolution`) as well as the specific sub-region (tile) to be rendered (`frameTileOffset`, `frameTileResolution`), enabling tiled rendering.
- **REQ-FR-CTRL-002:** The ability to provide world-to-object and object-to-world transformation matrices (`worldToObjectTransform`, `objectToWorldTransform`) to place the model correctly in the scene for rendering.
- **REQ-FR-CTRL-003:** The ability to define a 3D axis-aligned bounding box (`objectAABB`) that defines the spatial extent of the scene to be rendered.
- **REQ-FR-CTRL-004:** The ability to specify a complete sensor model (`sensorModel`) and its state, including pose and timestamp information (`sensorState`), for the frame.
- **REQ-FR-CTRL-005:** The ability to provide a color correction matrix (`colorCorrectionMatrix`) to be applied to the final radiance output.
- **REQ-FR-CTRL-006:** The ability to specify a list of object instance IDs (`objectInstanceIds`) that should be ignored during rendering, allowing for selective visibility.
- **REQ-FR-CTRL-007:** The ability to define a hit transmittance threshold (`hitTransmittance`) that determines when a ray is considered to have hit an opaque surface.
- **REQ-FR-CTRL-008:** The ability to provide a unique identifier for the frame being rendered (`id`).
- **REQ-FR-CTRL-009:** Via direct CUDA device pointer arguments to the `render` function, the library **shall** support instanced rendering of tracked objects from the source model. This includes the ability to specify which object instances are active for a frame and to provide a unique pose (including start and end poses for motion) for each active instance, allowing a single source object to be rendered multiple times in different locations within the same frame.
- **REQ-FR-CTRL-010:** As an alternative to using the built-in sensor models, the library **shall** support providing user-defined rays directly to the `render` function. This includes providing CUDA device pointers to buffers for ray origins, directions, and timestamps.

#### 3.1.4. Rendering Outputs

The library **shall** write the results of the rendering process into the following user-provided CUDA device buffers:

- **REQ-FR-OUT-001:** A buffer for radiance and opacity (`radianceDensityCudaPtr`), which represents the primary color output of the renderer.
- **REQ-FR-OUT-002:** A buffer for world-space hit distance (`worldHitDistanceCudaPtr`), which represents a depth map.
- **REQ-FR-OUT-003:** A buffer for instance IDs (`instanceIdCudaPtr`), which can be used for semantic or instance segmentation.
- **REQ-FR-OUT-004:** A buffer for arbitrary, model-defined "extended features" (`extendedFeaturesCudaPtr`). The layout for this buffer is defined by the model and can be queried from the API.

#### 3.1.5. Sensor Simulation

- **REQ-FR-SENSOR-001:** The library **shall** support a variety of camera projection models, including standard perspective, orthographic, fisheye (OpenCV), and F-Theta.
- **REQ-FR-SENSOR-002:** The library **shall** support the simulation of complex lens distortions, including radial and tangential distortions (OpenCV pinhole model) and bivariate polynomial windshield distortion models.
- **REQ-FR-SENSOR-003:** The library **shall** support the simulation of spinning Lidar sensors, including detailed, physically-based projection models that account for spinning direction and row offsets (e.g., for Hesai Lidars).
- **REQ-FR-SENSOR-004:** The library **shall** support the simulation of different camera shutter mechanisms, including Global Shutter and Rolling Shutter (top-to-bottom, left-to-right, etc.).

#### 3.1.6. API Conventions and Diagnostics

- **REQ-FR-API-001:** The library **shall** expose its public interface via a stable Application Binary Interface (ABI). This is achieved by exposing functions as static methods from a C++ struct, ensuring binary compatibility across different compiler versions and client applications.
- **REQ-FR-API-002:** The library **shall** provide a configurable logging mechanism. The user **shall** be able to specify a callback function (`LoggerParameters::Callback`) during initialization to process log messages from the library.
- **REQ-FR-API-003:** For diagnostic and debugging purposes, the library **shall** provide a function (`devicesMemoryUsage`) to query the total GPU memory currently in use by the library.

#### 3.1.7. Internal-Use API

This section describes functions that are part of the public API but are intended for specialized internal use cases (e.g., by the NuRec software for model training) and are not intended for general use.

- **REQ-FR-TRAIN-001:** The library **shall** provide a differentiable rendering function (`renderBackward`) that computes gradients of the rendered output with respect to scene parameters and ray properties, enabling gradient-based optimization during model training.
- **REQ-FR-TRAIN-002:** The library **shall** provide a function (`updateModelParameters`) to update model parameters (e.g., network weights) on the GPU from either host or device memory during a model training loop.
- **REQ-FR-TRAIN-003:** To support integration with external frameworks that generate RTC kernels (e.g., PyTorch), the library **shall** provide a function (`setRTCIncludeDirectory`) to specify additional include paths for the runtime compiler.

### 3.2. Non-Functional Requirements

#### 3.2.1. Performance

- **REQ-NFR-PERF-001:** The library **shall** meet defined performance targets for rendering throughput (e.g., megapixel-rays per second) for a set of benchmark scenes and sensor configurations. Performance targets **shall** be specified for each supported major GPU architecture (e.g., Turing, Ampere, Hopper). (Specific targets TBD).
- **REQ-NFR-PERF-002:** The library's GPU memory consumption **shall not** exceed specified limits for a set of benchmark scenes. These limits **shall** be defined per supported GPU architecture. (Specific targets TBD).

#### 3.2.2. Reliability

- **REQ-NFR-REL-001:** All public API functions **shall** report success or failure using a consistent set of enumerated error codes (`nrend::ErrorCode`).
- **REQ-NFR-REL-002:** For a fixed set of inputs, model parameters, hardware, and software versions, the rendered output **shall** be numerically stable and repeatable. While bit-for-bit determinism may not be guaranteed due to floating-point arithmetic, the difference between two runs **shall** be within a defined tolerance (delta).
- **REQ-NFR-REL-003:** To ensure functional parity and correctness, the library's rendering output **shall** match the output of the reference Python-based PyTorch renderer used in the NuRec software suite, within a defined numerical tolerance.

#### 3.2.3. Usability

- **REQ-NFR-USE-001:** The public API header files **shall** be clearly documented using a Doxygen-compatible format, explaining all functions, parameters, and data structures.
- **REQ-NFR-USE-002:** The library **shall** be delivered with a set of example applications demonstrating its key features and API integration patterns.

#### 3.2.4. Portability

- **REQ-NFR-PORT-001:** The library **shall** support NVIDIA GPU architectures of compute capability 7.5 (Turing) and higher.
- **REQ-NFR-PORT-002:** The library **shall** be buildable and fully functional on the following target platforms: Linux (x86_64), Linux (aarch64), and Windows (x86_64).
- **REQ-NFR-PORT-003:** To ensure forward compatibility, the Linux version of the library **shall** be built on Ubuntu 20.04, compatible with that version and newer distributions.

#### 3.2.5. Security

- **REQ-NFR-SEC-001:** The library **shall** perform validation of input configuration files (e.g., MsgPack data) to prevent parsing-related security vulnerabilities like buffer overflows.
- **REQ-NFR-SEC-002:** The library **shall** validate parameters passed to the API to prevent out-of-bounds memory access or other undefined behavior.

#### 3.2.6. Build and Linkage

- **REQ-NFR-BL-001:** The library **shall** be compiled as a shared library (`.so` for Linux, `.dll` for Windows).
- **REQ-NFR-BL-002:** The final output binary file **shall** be named `nrend` (e.g., `libnrend.so`, `nrend.dll`).
- **REQ-NFR-BL-003:** The library **shall** be built against the version of the NVIDIA CUDA Toolkit required by the target Omniverse release. (Note: The current requirement for Omniverse is CUDA 11.8).
- **REQ-NFR-BL-004:** All third-party software dependencies, excluding core system and CUDA libraries, **shall** be statically linked into the final binary.
- **REQ-NFR-BL-005:** On the Windows platform, the Microsoft Visual C++ (MSVC) runtime library **shall** be statically linked.

### 3.3. Interface Requirements

#### 3.3.1. Hardware Interfaces

- The library interfaces directly with NVIDIA GPU hardware via the CUDA driver and runtime APIs. It requires an NVIDIA GPU with one of the following compute capabilities: 7.5 (Turing), 8.0 (Ampere), 8.6 (Ampere), 8.9 (Ada Lovelace), or 9.0 (Hopper).

#### 3.3.2. Software Interfaces

- **CUDA Toolkit:** The library **shall** depend on a specific version of the NVIDIA CUDA Toolkit.
- **C++ Compiler:** The library **shall** require a C++ compiler that is compliant with the C++17 standard.
- **Third-Party Libraries:** The library will have dependencies on other libraries (e.g., tiny-cuda-nn). These dependencies must be managed by the project's build system.

---

## 4. Packaging and Deployment

### 4.1. Release Package

The NREND library release package **shall** be created using the Omniverse Packman tool and published to the Omniverse Artifactory at https://omnipackages.nvidia.com/. The structure is defined by the Packman specification, detailed at https://gitlab-master.nvidia.com/omniverse/repo/repo_man.

Key assets included in the release package **shall** include:

- **Binaries:** Compiled shared libraries (`.so`, `.dll`).
- **Headers:** All public C++ header files required to use the library.
- **Version:** A file indicating the package version number.

The following related assets **shall** be published to separate locations:

- **Debug Symbols:** The corresponding debug symbols **shall** be published to the Omniverse Symbolserver.

### 4.2. Versioning

The library's versioning scheme **shall** follow the format: `<major>.<minor>.<commits>-<githash>`. This scheme is consistent with the versioning used by the NuRec (neural reconstruction engine) software.

- **major:** Major version number, incremented for significant incompatible API changes.
- **minor:** Minor version number, incremented for new features and compatible API changes.
- **commits:** The number of git merge commits since the last minor version update.
- **githash:** The abbreviated git commit hash of the build.
  For example: `0.2.703-771a5a2d`.
