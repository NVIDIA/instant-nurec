# NREND CI Documentation

## Overview

This document describes the Continuous Integration (CI) pipeline for building and publishing the NREND (NVIDIA Rendering Engine) package. The pipeline is hosted on GitLab at `https://gitlab-master.nvidia.com/omniverse/nrend` and builds packages for multiple platforms:

- Linux x86_64
- Linux aarch64 (ARM64)
- Windows x86_64

## Pipeline Architecture

### Repository Structure

The CI pipeline configuration is defined in `.gitlab-ci.yml` and uses a multi-stage approach to build and publish packages across different platforms.

### Key Components

1. **Source Repository**: The pipeline pulls the NRE source code from `https://gitlab-master.nvidia.com/nrs/nre.git`
2. **Docker Images**: Uses containerized build environments for consistent builds
3. **Package Publishing**: Publishes to NVIDIA's internal package repository using omni-pkgpublish

## Docker Images

The NREND CI pipeline uses several specialized Docker images to provide consistent build environments across different platforms and build stages.

### Base Image: `nrend/nre`

**Purpose**: Primary development and base environment for the pipeline
**Source**: Built from `deps/docker/Dockerfile-dev.build` in the NRE repository
**Base**: `nvidia/cuda:12.4.1-devel-ubuntu22.04`

For legacy reasons this image re-uses the dev image from NuRec. It could be trimmed down, as its not used for compilation but just simple functionality of basic commands for some of the pipeline jobs.

### Linux Dependencies Image: `nrend/nrend-deps`

**Purpose**: Pre-built dependencies for Linux NREND builds
**Source**: Built from `libs/nrend/Dockerfile_nrend_deps.build` by the `image-linux` job
**Base**: `nvidia/cuda:11.8.0-devel-ubuntu20.04`

This image provides the build environment for the linux builds. It also provides the prebuilt versions of the TCNN and Slang libraries, which are compiled during docker image creation and embedded directly into the docker image.

Versioning of the image is driven from the `NREND_DEPS_VERSION` variable in the `nrend-deps.yml` file, so that Slang and TCNN dependencies are keep at the expected version for the pipeline build.

**Usage**:

- Linux build jobs (`build-linux`)
- Provides pre-compiled dependencies to speed up builds
- Supports both x86_64 and aarch64 package generation

### Windows Builder Image: `nrend/builder`

**Purpose**: Windows-specific build environment
**Source**: Built from `libs/nrend/ci/windows/Dockerfile.builder` by the `image-windows` job using the `docker_update_nrend_deps.sh` script
**Base**: `mcr.microsoft.com/windows/servercore:ltsc2022`

This image provides the build environment for the windows builds.

**Usage**:

- Windows build jobs (`build-windows`)
- Windows-specific compilation and packaging

## Pipeline Configuration

### Variables

Key environment variables used in the pipeline:

- `NRE_REPO`: Repository URL with read-only access token
- `NRE_BRANCH`: Branch of NRE to use (default: "main")
- `NREND_DEPS_TAG`: Version tag for NREND dependencies Docker image

## Pipeline Stages

### Stage 1: Prepare

This stage builds the necessary Docker images for the build environment. For linux this includes building the TCNN and Slang dependencies as they are embedded into the docker image. This stage only needs to be run if either the build environment changes, or TCNN and Slang versions change.

### Stage 2: Build

This stage compiles the NREND packages for each target platform. For windows, this stage also builds the TCNN and Slang dependencies.

#### Linux Build (`build-linux`)

- **Trigger**: Scheduled/Pipeline execution
- **Requirements**:
  - Linux OS
  - x86_64 architecture
- **Build Environment**: Uses NREND dependencies Docker image
- **Process**:
  1. Clones NRE repository
  2. Runs `build_linux.sh` script
  3. Builds packages for both x86_64 and aarch64 architectures
- **Artifacts**:
  - `nre/libs/nrend/package_x86_64/`
  - `nre/libs/nrend/package_aarch64/`

#### Windows Build (`build-windows`)

- **Trigger**: Scheduled/Pipeline execution
- **Requirements**:
  - Windows OS
  - x86_64 architecture
  - Docker support
- **Build Environment**: Uses Windows builder Docker image
- **Process**:
  1. Clones NRE repository
  2. Runs `ci\windows\build_deps.bat`
  3. Runs `ci\windows\build_nrend.bat`
- **Artifacts**:
  - `nre/libs/nrend/_build/packages/`

### Stage 3: Deploy

This stage takes the package artifacts built by the `build` stage and publishes them to the Omiverse Artifactory. The gitlab CI config for this is pulled in directly from the external gitlab project https://gitlab-master.nvidia.com/omniverse/sectools/omni-pkgpublish.

The Omniverse Atifactory URL is https://omnipackages.nvidia.com/packages/artifactory/nrend.

## Pipeline Triggers

The NREND CI pipeline can be triggered in three different ways, each serving different purposes in the development and release workflow.

### 1. Manual Trigger (Default)

**Purpose**: On-demand builds for development, testing, and custom builds
**Access**: GitLab UI pipeline execution

**Process**:

- Navigate to the NREND GitLab project pipeline section
- Click "Run Pipeline"
- **Required Input**: Must specify the `NRE_BRANCH` variable
  - This controls which branch of the NuRec repository will be built
  - Common values: `main`, `develop`, or specific feature branches
- **Optional Variables**: Can override other pipeline variables as needed
- **Scope**: Runs all pipeline stages (prepare, build, deploy) based on job configuration

**Use Cases**:

- Building specific feature branches for testing
- Creating custom builds with modified dependencies
- Manual releases outside of the normal schedule
- Debugging pipeline issues with specific configurations

**Benefits**:

- Full control over source branch and build parameters
- Immediate feedback for development work
- Ability to test pipeline changes safely

### 2. Scheduled Nightly Build

**Purpose**: Regular automated builds for continuous validation
**Schedule**: Nightly execution (exact time configurable)

**Process**:

- Automatically triggered by GitLab's scheduled pipeline feature
- Uses default `NRE_BRANCH` value (typically `main`)
- **Scope**: Runs only the **build** stage
  - Executes `build-linux` and `build-windows` jobs
  - **Does NOT** publish packages to Omniverse Artifactory
- Generates build artifacts for verification

**Use Cases**:

- Continuous validation of the main branch
- Early detection of build failures
- Maintaining up-to-date build artifacts
- Performance monitoring and build time tracking

**Benefits**:

- Automated quality assurance
- No manual intervention required
- Consistent build validation
- Build artifact availability for testing

### 3. NuRec Release Trigger

**Purpose**: Automated package publishing for official releases
**Source**: Triggered externally from the NuRec repository pipeline

**Process**:

- **Trigger Source**: NuRec repository's `publish_nrend` job
- **Activation**: Automatically runs when a new release tag is created in the NuRec repository
- **Scope**: Runs only the **build** and **publish** stages
- **Target**: Publishes packages to Omniverse Artifactory for distribution

**Workflow**:

1. Developer creates a release tag in the NuRec repository
2. NuRec's CI pipeline detects the new tag
3. NuRec's `publish_nrend` job triggers the NREND pipeline
4. NREND pipeline builds packages from the tagged release
5. Packages are automatically published to Omniverse Artifactory
6. Packages become available for downstream consumers

**Use Cases**:

- Official product releases
- Automated distribution of stable builds
- Integration with NuRec release process
- Ensuring consistent versioning between NuRec and NREND packages

**Benefits**:

- Fully automated release process
- Tight integration with NuRec versioning
- Immediate availability of release packages
- Reduced manual intervention for releases

### Important Note: Image Stage Jobs

**Manual Execution Required**: For all trigger types (manual, scheduled, and NuRec release), the **image stage jobs are never executed automatically**. These jobs must be manually triggered by developers when needed.

**Image Stage Jobs**:

- `image-linux`: Builds the `nrend/nrend-deps` Docker image with TCNN and Slang dependencies
- `image-windows`: Builds the `nrend/builder` Windows Docker image

**Why Manual Execution**:

- **Build Time**: The Linux image job in particular takes a very long time to execute due to:
  - Cross-compilation setup for both x86_64 and aarch64 architectures
  - Building TCNN for multiple CUDA architectures (75, 80, 86, 89, 90)
  - Building Slang with cross-compilation support
  - Large dependency downloads and compilation
- **Frequency**: Image updates are only needed when:
  - Build environment requirements change
  - TCNN or Slang dependency versions are updated
  - Base Docker images need security updates
- **Resource Usage**: Avoid unnecessary resource consumption on GitLab runners

**When to Trigger Image Jobs**:

- After updating `NREND_DEPS_VERSION` in `nrend-deps.yml`
- When modifying `Dockerfile_nrend_deps.build` (linux) or `Dockerfile.builder` (windows)
- When base CUDA or OS images need updates
- When troubleshooting build environment issues

**Developer Workflow**:

1. Trigger image jobs manually when dependency versions change
2. Wait for image build completion (especially for Linux - can take 1+ hours)
3. Subsequent build jobs will use the updated images automatically
4. Image builds are cached and reused until manually updated again

## Windows CI Infrastructure

The Windows builds for NREND are executed on a dedicated Windows virtual machine that requires specific setup and maintenance procedures.

### Machine Specifications

**Hostname**: `hqdvnrebuild01.nvidia.com`
**Operating System**: Windows Server 2022
**Infrastructure**: NVIDIA IT-hosted virtual machine

### Storage Configuration

The virtual machine uses a dual-partition setup to optimize performance and storage management:

- **C: Drive**: Operating system and system applications
- **D: Drive**: Data storage and CI-related applications (330GB capacity)
  - GitLab Runner installation and workspace
  - Docker data directory
  - Build artifacts and temporary files

### Remote Access

**Connection Method**: Windows Remote Desktop Protocol (RDP)
**Authentication**: Requires DUO multi-factor authentication
**Access Requirements**:

- NVIDIA network access or VPN connection
- Valid NVIDIA credentials with DUO setup
- RDP client software

**_NOTE_** The server is being moved from DUO to Silverfort for login authentication process, switchover is scheduled for 2025-07-06.
See: https://nvidia.service-now.com/esc?id=kb_article&sysparm_article=KB0029645

**Connection Process**:

1. Connect to `hqdvnrebuild01.nvidia.com` via RDP client
2. Authenticate with NVIDIA credentials
3. Complete DUO multi-factor authentication
4. Access Windows Server 2022 desktop environment

### GitLab Runner Setup

**Installation Location**: `D:\gitlab\`
**Components**:

- `gitlab-runner.exe`: Main executable
- `config.toml`: Configuration file with custom resource limits

**Resource Configuration**:
The configuration file has been customized to provide adequate resources for NREND builds:

```toml
cpus = "4"
memory = "12GB"
```

**Default Limitations**: Without this customization, GitLab Runner defaults to 1 CPU and 1GB memory, which is insufficient for NREND compilation.

**Service Management**:

- **Service Name**: "gitlab-runner"
- **Startup**: Automatically starts during system boot
- **Execution**: Runs as a background system service

**Administrative Commands**:
To interact with GitLab Runner, use an Administrator command prompt:

```cmd
cd D:\gitlab
gitlab-runner.exe help          # View available commands
gitlab-runner.exe status        # Check runner status
gitlab-runner.exe restart       # Restart the runner
gitlab-runner.exe list          # List registered runners
```

### Docker Engine Setup

**Installation Location**: `C:\ProgramData\docker\` (system files)
**Data Directory**: `D:\docker\` (configured via daemon.json)

**Configuration Customization**:
Docker's data directory has been redirected to the D: drive via:
**File**: `C:\ProgramData\docker\config\daemon.json`
**Purpose**: Store Docker images, containers, and data on the larger D: partition

```json
"data-root": "D:\\docker"
```

**Service Management**:

- **Service Name**: "docker"
- **Startup**: Automatically starts during system boot
- **Execution**: Runs as a background system service

**Command Line Usage**:
Docker commands work identically to Linux environments:

```cmd
docker --version               # Check Docker version
docker images                  # List available images
docker ps                      # List running containers
docker system df               # Check disk usage
docker system prune            # Clean up unused resources
```

### Maintenance Procedures

#### Routine Maintenance

**Disk Space Monitoring**:

- Monitor D: drive usage (330GB capacity)
- Clean up old build artifacts and Docker images
- Use `docker system prune` to remove unused Docker resources

**Service Health Checks**:

```cmd
# Check GitLab Runner status
net start | findstr gitlab-runner

# Check Docker Engine status
net start | findstr docker

# View services via Windows Services app
services.msc
```

**Log File Locations**:

- **GitLab Runner Logs**: Available via `gitlab-runner` commands
- **Docker Logs**: Windows Event Viewer or Docker CLI commands
- **System Logs**: Windows Event Viewer

#### Troubleshooting

**GitLab Runner Issues**:

1. Check service status: `sc query gitlab-runner`
2. Restart service: `sc stop gitlab-runner && sc start gitlab-runner`
3. Check configuration: Review `D:\gitlab\config.toml`
4. Test connection: `gitlab-runner verify`

**Docker Issues**:

1. Check service status: `sc query docker`
2. Restart Docker: `net stop docker && net start docker`
3. Check daemon configuration: Review `C:\ProgramData\docker\config\daemon.json`
4. Test Docker: `docker run hello-world`

**Resource Issues**:

1. Monitor CPU and memory usage via Task Manager
2. Check D: drive space: `dir D:\ /-c`
3. Clean Docker resources: `docker system prune -a`
4. Review running processes for resource conflicts

#### Docker Volume Permission Issues on Windows

An intermittent issue can occur on Windows where files in Docker volumes become locked, causing GitLab jobs to fail with file permission errors.

**Symptoms**:

- GitLab CI jobs fail with permission denied errors
- Unable to access or modify files in build directories
- Job fails during preparation when cleaning build directory

**Resolution Steps**:

1. **Check existing Docker volumes**:

   ```cmd
   docker volume ls
   ```

2. **Attempt volume deletion**:

   ```cmd
   docker volume prune
   docker volume rm <volume-name>
   ```

3. **If deletion fails, manual intervention required**:

   a. **Identify the problematic volume** (example name: `runner-3sqasprsj-project-168047-concurrent-0-cache-2c121c608c4de07906679419a11b5806`)

   b. **Run a new container mapping the volume**:

   ```cmd
   docker run --rm -it -v runner-3sqasprsj-project-168047-concurrent-0-cache-2c121c608c4de07906679419a11b5806:C:\builds gitlab-master.nvidia.com:5005/omniverse/nrend/builder:windows-x86_64
   ```

   c. **Inside the container, execute these commands**:

   **Option 1: Using Command Prompt (cmd, run as Administrator)**:

   ```cmd
   # Take ownership of the files
   takeown /f "C:\builds\omniverse" /r /d y

   # Grant full permissions to current user
   icacls "C:\builds\omniverse" /grant "%username%:F" /t

   # Remove the directory and all contents
   rmdir /s /q "C:\builds\omniverse"
   ```

   **Option 2: Using PowerShell (run as Adminstrator)**:

   ```powershell
   # Take ownership of the files
   takeown /f "C:\builds\omniverse" /r /d Y

   # Grant full permissions to current user
   icacls "C:\builds\omniverse" /grant "$($env:UserName):F" /t

   # Remove the directory and all contents
      Remove-Item "C:\builds\omniverse" -Recurse -Force
   ```

   d. **Exit the container** (it will automatically be removed due to `--rm` flag)

   e. **Remove the now-empty volume**:

   ```cmd
   docker volume rm <volume-name>
   ```

4. **Retry the failed GitLab CI job**

**Note**: Volume names are dynamically generated by GitLab Runner and will vary between jobs. Use `docker volume ls` to identify the correct volume name for your specific case.
