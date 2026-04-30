# Dependencies Architecture

**Directory Path:** `deps/`

---

## 1. Overview

The `deps/` directory contains all third-party dependencies and their Bazel build configurations for the NRE project. This directory serves as the central location for managing external packages, build files, and dependency resolution configurations.

### 1.1 Purpose

The `deps/` directory provides:

1. **Centralized Dependency Management**: Single location for all external dependencies
2. **Build Configuration**: Custom Bazel BUILD files for third-party repositories
3. **Python Package Management**: Requirements files and pip compilation rules
4. **JavaScript/TypeScript Tools**: npm packages for formatting and development tools
5. **Docker Build Configurations**: Dockerfiles for various build and runtime environments

### 1.2 Design Principles

1. **Flat Structure**: All dependencies live at the same level (`deps/`) rather than nested hierarchies
2. **Explicit Build Files**: Custom `.BUILD` files that override or supplement upstream build configurations
3. **Multi-Platform Support**: Separate configurations for x86_64 and aarch64 architectures
4. **Reproducible Builds**: Locked versions and SHA-256 hashes for all dependencies
5. **Separation of Concerns**: Each subdirectory manages a specific category of dependencies

---

## 2. What Belongs in `deps/`?

### 2.1 Types of Dependencies

**✅ Should be in `deps/`:**

1. **Third-party C++ libraries** requiring custom Bazel BUILD files
   - Examples: Slang, tiny_cuda_nn, numpyeigen
2. **Python package management files**

   - Requirements files (`.in`, `.txt`)
   - Pip compilation rules
   - Wheel patching documentation

3. **JavaScript/TypeScript tooling**

   - npm packages for development tools
   - Code formatters, linters, plugins

4. **Docker build configurations**

   - Dockerfiles for various environments
   - Must reference dependency paths

5. **Custom/patched external repositories**
   - Forks of upstream projects
   - Internal NVIDIA packages

**❌ Should NOT be in `deps/`:**

1. **Bazel rule definitions** → `bazel/` directory
2. **Project source code** → `libs/`, `nre/`, etc.
3. **Test data** → Defined in `MODULE.bazel` as `http_archive` (e.g., `test_data_ncore`)
4. **Pretrained models** → Defined in `MODULE.bazel` as `http_archive` (e.g., `pretrained_models_repo`)
5. **Build tools** → Bazel modules (defined in `MODULE.bazel` with `bazel_dep`)

### 2.2 Adding New Dependencies

#### For Python Packages:

1. Add to `deps/python/requirements_3_11_common.in`
2. Run `bazel run //deps/python:update_all_requirements` (this also updates the top-level `uv.lock` for security scans)
3. Commit both `.in` and `.txt` files, and `uv.lock` if it changed

#### For C++ Libraries:

1. Create `deps/<library_name>/` directory
2. Add custom `<library_name>.BUILD` file
3. Create `deps/<library_name>/BUILD.bazel` with aliases
4. Add repository definition to `MODULE.bazel`:
   ```python
   new_git_repository(
       name = "library_name",
       build_file = "//deps/library_name:library_name.BUILD",
       commit = "...",
       remote = "https://...",
   )
   ```

#### For npm Packages:

1. Edit `deps/npm/package.json`
2. Run `pnpm install` to update `pnpm-lock.yaml`
3. Update `MODULE.bazel` if needed
4. Commit both files

#### For Docker Images:

1. Add Dockerfile to `deps/docker/`
2. Ensure it references correct dependency paths
3. Document build instructions in comments
4. Update CI/CD pipelines if used for builds

### 2.3 Dependency Version Management

**Pinning Strategy:**

- **Python**: Pin specific versions with hashes in `.txt` files
- **C++**: Pin to specific Git commit SHAs
- **npm**: Use lockfile (`pnpm-lock.yaml`) for deterministic installs
- **Docker**: Reference specific base image tags

**Updating Dependencies:**

1. Update version in source (`.in` file, `MODULE.bazel`, `package.json`)
2. Regenerate lockfiles
3. Test locally: `bazel test ...`
4. Update SHA-256 hashes for `http_archive` entries
5. Commit changes and run CI

---

## 3. Directory Structure

```text
deps/
├── docker/              # Docker build configurations
├── mmseg_repo/          # MMSegmentation inference library
├── npm/                 # JavaScript/TypeScript tooling (prettier, etc.)
├── numpyeigen/          # Eigen bindings for NumPy
├── numpyeigen_pybind11/ # Pybind11 fork for numpyeigen
├── python/              # Python package requirements and pip rules
├── slang/               # Slang shader compiler
└── tiny_cuda_nn/        # Tiny CUDA Neural Network library
```

### 3.1 Directory Purposes

| Directory              | Purpose                                             | Type       |
| ---------------------- | --------------------------------------------------- | ---------- |
| `docker/`              | Container images for build and runtime environments | Docker     |
| `mmseg_repo/`          | Semantic segmentation models and inference          | Python/C++ |
| `npm/`                 | Code formatting tools (prettier, plugins)           | JavaScript |
| `numpyeigen/`          | High-performance NumPy-Eigen interop                | C++/Python |
| `numpyeigen_pybind11/` | Custom pybind11 fork for numpyeigen                 | C++/Python |
| `python/`              | Python package management and pip rules             | Python     |
| `slang/`               | GPU shader compiler for neural rendering            | C++        |
| `tiny_cuda_nn/`        | Fast CUDA neural network primitives                 | CUDA/C++   |

---

## 4. Python Dependencies (`deps/python/`)

### 4.1 Files

- **`requirements_3_11_common.in`**: Common Python requirements across architectures
- **`requirements_3_11_x86_64.in`**: x86_64-specific requirements
- **`requirements_3_11_x86_64.txt`**: Locked requirements with hashes (x86_64)
- **`requirements_3_11_aarch64.in`**: ARM64-specific requirements
- **`requirements_3_11_aarch64.txt`**: Locked requirements with hashes (ARM64)
- **`requirements_3_11_internal_x86_64.in`**: Internal-only packages
- **`requirements_3_11_internal_x86_64.txt`**: Locked internal requirements
- **`BUILD.bazel`**: Bazel pip_compile rules for requirement generation
- **`how_to_patch_python_wheels.md`**: Guide for patching Python packages

### 4.2 Python Package Management

The project uses `rules_uv` and `pip_compile` for deterministic Python dependency resolution:

```python
pip_compile(
    name = "requirements_3_11_x86_64",
    args = ["--generate-hashes", "--emit-index-url"],
    python_platform = "x86_64-manylinux_2_31",
    requirements_in = "requirements_3_11_x86_64.in",
    requirements_txt = "requirements_3_11_x86_64.txt",
)
```

### 4.3 Package Index Sources

- **PyPI**: `https://pypi.org/simple`
- **PyTorch**: `https://download.pytorch.org/whl/cu128`
- **NVIDIA Internal**: `https://gitlab-master.nvidia.com/api/v4/projects/...`
- **NVIDIA Public**: `https://pypi.nvidia.com`

### 4.4 Changing Python Dependencies

To add or update Python packages:

1. Edit the appropriate `.in` file:

   - `requirements_3_11_common.in` for cross-platform packages
   - `requirements_3_11_x86_64.in` for x86_64-only packages
   - `requirements_3_11_aarch64.in` for ARM64-only packages
   - `requirements_3_11_internal_x86_64.in` for internal packages

2. Regenerate the `.txt` files and the uv lock file:

   ```bash
   # Update all requirements files and top-level uv.lock (for security scans, e.g. Black Duck)
   bazel run //deps/python:update_all_requirements

   # Or update individually
   bazel run //deps/python:requirements_3_11_x86_64.update
   bazel run //deps/python:requirements_3_11_aarch64.update
   bazel run //deps/python:requirements_3_11_internal_x86_64.update
   bazel run //:generate_lock   # uv.lock only
   ```

3. Commit both the `.in` and `.txt` files, and `uv.lock` if it changed

### 4.5 Patching Python Wheels

When upstream packages need modifications:

1. Download and extract the wheel
2. Apply patches to the source
3. Update version with git hash suffix (e.g., `0.2.23+nre4`)
4. Repack and upload to internal registry
5. Reference in `requirements_3_11_common.in` with `--hash=sha256:...`

See `deps/python/how_to_patch_python_wheels.md` for detailed instructions.

---

## 5. C++ Dependencies

### 5.1 Slang Shader Compiler (`deps/slang/`)

**Repository**: `https://gitlab-master.nvidia.com/nrs/nre_external/slang`  
**Build File**: `deps/slang/slang.BUILD`  
**Purpose**: Real-time shader compilation and GPU kernel execution

The Slang dependency provides:

- **`slangc`**: Shader compiler executable
- **`slang_headers`**: C++ header files
- **`lib_slang_compiler`**: Compiler shared library
- **`lib_slang_rtc`**: Runtime compilation library

Platform-specific builds are selected via Bazel aliases:

```python
alias(
    name = "slangc",
    actual = select({
        "@platforms//cpu:aarch64": "@slang_aarch64//:slangc",
        "//conditions:default": "@slang_x86_64//:slangc",
    }),
)
```

#### Slang Usage in NRE

The project provides a custom Bazel rule `slangtorch_library` (defined in `bazel/slang/defs.bzl`) for compiling Slang modules into PyTorch C++ CUDA extensions:

```python
load("//bazel/slang:defs.bzl", "slangtorch_library")

slangtorch_library(
    name = "my_kernels",
    srcs = ["kernels.slang"],
    deps = ["//libs/slang_utils:extensions_slang"],
)
```

This rule automatically:

1. Compiles Slang source to CUDA code using `slangc`
2. Builds CUDA and C++ libraries
3. Creates a PyTorch-compatible extension

**Libraries using Slang:**

- `libs/geometry/` - Quaternion operations and geometric transformations
- `libs/slang_gaussians/` - Gaussian splatting kernels
- `nre/models/post_processings/ppisp/` - Post-processing ISP pipeline
- `libs/slang_utils/` - Common utilities and extensions

### 5.2 Tiny CUDA Neural Network (`deps/tiny_cuda_nn/`)

**Repository**: `https://gitlab-master.nvidia.com/nrs/nre_external/tiny-cuda-nn`  
**Build File**: `deps/tiny_cuda_nn/tcnn_obfuscation.BUILD`  
**Purpose**: Fast CUDA implementations of neural network primitives

This is an **obfuscated build** - the source is compiled into binary form and the original structure is hidden. The project uses `tinycudann` Python bindings which wrap this C++ library.

### 5.3 NumPyEigen (`deps/numpyeigen/`)

**Source**: Forked from `https://github.com/fwilliams/numpyeigen`  
**Build File**: `deps/numpyeigen/numpyeigen.BUILD`  
**Purpose**: Zero-copy interface between NumPy and Eigen

Provides high-performance C++ bindings for passing NumPy arrays to Eigen-based C++ code without copying data.

Dependencies:

- Eigen 3.4.0 (header-only linear algebra library)
- Custom pybind11 fork (`deps/numpyeigen_pybind11/`)

### 5.4 MMSegmentation (`deps/mmseg_repo/`)

**Repository**: `https://gitlab-master.nvidia.com/nrs/nre_external/mmseginference`  
**Build File**: `deps/mmseg_repo/mmseg_repo.BUILD`  
**Purpose**: Semantic segmentation models and inference

Used for scene understanding and object detection in neural reconstruction pipelines.

---

## 7. Docker Configurations (`deps/docker/`)

### 7.1 Dockerfiles

| File                   | Purpose                                    |
| ---------------------- | ------------------------------------------ |
| `Dockerfile-dev.build` | Development environment with build tools   |
| `Dockerfile-run.build` | Runtime environment (smaller, production)  |
| `Dockerfile-AH`        | Asset Harvester specialized environment    |
| `Dockerfile-pytorch`   | PyTorch-specific build base                |
| `Dockerfile-tcnn`      | Tiny CUDA Neural Network build environment |
| `Dockerfile-wheels`    | Python wheel building environment          |

### 7.2 Build Instructions

Docker images are built using:

```bash
# Example: Build development environment
docker build -f deps/docker/Dockerfile-dev.build -t nre-dev .

# Example: Build runtime environment
docker build -f deps/docker/Dockerfile-run.build -t nre-run .
```

These Dockerfiles reference dependency paths and are used for:

- CI/CD pipelines
- Consistent development environments
- Production deployments

---

## 8. Integration with `MODULE.bazel`

### 8.1 External Repository Definitions

The `MODULE.bazel` file at the project root defines how external dependencies are fetched and built:

```python
# Slang compiler
http_archive(
    name = "slang_x86_64",
    build_file = "//deps/slang:slang.BUILD",
    sha256 = "...",
    urls = ["https://..."],
)

# Tiny CUDA Neural Network
new_git_repository(
    name = "tiny_cuda_nn",
    build_file = "//deps/tiny_cuda_nn:tcnn_obfuscation.BUILD",
    commit = "a9da7ef6b649445c514a49c2351652f05b0ca463",
    remote = "https://gitlab-master.nvidia.com/nrs/nre_external/tiny-cuda-nn",
)

# Python pip dependencies
pip.parse(
    hub_name = "nre_pip_deps",
    python_version = "3.11",
    requirements_by_platform = {
        "//deps/python:requirements_3_11_aarch64.txt": "linux_aarch64",
        "//deps/python:requirements_3_11_x86_64.txt": "linux_x86_64",
    },
)

# npm dependencies
npm.npm_translate_lock(
    name = "npm",
    npmrc = "//deps/npm:.npmrc",
    pnpm_lock = "//deps/npm:pnpm-lock.yaml",
)
```

### 8.2 Dependency Types

1. **`http_archive`**: Download and extract tar.gz files (with SHA-256 verification)
2. **`git_repository`**: Clone Git repositories at specific commits
3. **`new_git_repository`**: Clone Git repositories with custom BUILD files
4. **`http_file`**: Download individual files

### 8.3 Custom BUILD Files

Dependencies in `deps/` provide custom BUILD files that:

- Expose specific targets for use in the project
- Override upstream build configurations
- Add platform-specific logic
- Integrate with Bazel's dependency graph
