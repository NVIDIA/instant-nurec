# NRE Profiling Scripts

This directory contains profiler-specific scripts for different profiling backends.

## Prerequisites

**For GPU profiling (Tracy GPU mode and NSight Systems):**

- CUDA toolkit must be installed
- CUPTI (CUDA Profiling Tools Interface) must be available

The build system automatically detects CUPTI. If needed, set the path explicitly:

```bash
export CUPTI_PATH=/usr/local/cuda/extras/CUPTI
# or
export CUDA_HOME=/usr/local/cuda
```

See the main README.md for detailed CUPTI configuration instructions.

## Directory Structure

```
scripts/profilers/
├── tracy/                          # Tracy profiler scripts
│   └── run_with_tracy.sh          # Run with Tracy profiling (CPU/GPU)
├── nsys/                           # NVIDIA NSight Systems scripts
│   └── run_with_nsys.sh           # Run with NSys profiling
└── README.md                       # This file
```

## Tracy Profiling

Tracy provides real-time profiling with a GUI that connects to your running application.

### Usage:

```bash
# CPU profiling only
./scripts/profilers/tracy/run_with_tracy.sh serve-grpc

# CPU + GPU profiling
./scripts/profilers/tracy/run_with_tracy.sh --gpu serve-grpc
```

### What You Get:

- **Profiling Zones**: Visual timeline in Tracy GUI with colored zones
- **Silent Operation**: No text timing spam (uses `--timing-verbosity=NONE`)
- **Real-time**: Live profiling data as your application runs

### Requirements:

- Tracy GUI running and listening on port 8086
- For GPU profiling: CUDA toolkit with CUPTI libraries

## NSys Profiling

NSys captures profiles to files that you view afterwards.

### Usage:

```bash
# Capture a profile
./scripts/profilers/nsys/run_with_nsys.sh serve-grpc

# Custom output file
./scripts/profilers/nsys/run_with_nsys.sh -o my_profile.nsys serve-grpc

# Just run without capturing (for testing)
./scripts/profilers/nsys/run_with_nsys.sh --no-capture serve-grpc

# View the captured profile
nsys-ui my_profile.nsys
```

### What You Get:

- **Profiling Zones**: Visual timeline in NSys GUI with NVTX markers (no colors due to PyTorch limitation)
- **Silent Operation**: No text timing spam (uses `--timing-verbosity=NONE`)
- **File-based**: Capture to file for later analysis

### Requirements:

- NVIDIA NSight Systems installed
- `nsys` command available in PATH

## Unified Profiling API

Both profilers use the same unified API in the code using `@ScopedTimer`:

```python
from nre.utils.profiling import ScopedTimer, ProfileColor

# As a decorator (recommended for entire functions)
@ScopedTimer("my_function", color=ProfileColor.GREEN)
def my_function():
    # Function code to profile and time
    pass

# As a context manager (for code blocks)
with ScopedTimer("my_operation", color=ProfileColor.BLUE):
    # Code block to profile and time
    pass
```

The profiler backend is selected via command-line flags:

- `--profiling-backend=tracy` → Tracy backend (CPU/GPU profiling with real-time GUI)
- `--profiling-backend=nvtx` → NVTX/NSys backend (file-based profiling)
- `--enable-timing` → Enables the timing system (required for profiling zones)
- `--timing-verbosity=NONE` → Disables text timing output (keeps only visual zones)

**Note**: The scripts automatically pass these flags and use specialized Bazel targets (`//:run_with_tracy*` for Tracy, `//:run` for NSys).

## Advanced Usage

You can customize the profiling behavior using CLI flags:

```bash
# Enable text timing output along with Tracy profiling
./scripts/profilers/tracy/run_with_tracy.sh serve-grpc --timing-verbosity=BASIC

# Run without profiling zones but with text timing only
bazel run //:run -- serve-grpc \
  --profiling-backend=none \
  --enable-timing \
  --timing-verbosity=BASIC

# NSys profiling with detailed timing output
./scripts/profilers/nsys/run_with_nsys.sh serve-grpc --timing-verbosity=DETAILED
```

## Available CLI Options

- `--profiling-backend=none/tracy/nvtx` - Select profiling backend
- `--enable-timing` - Enable/disable the timing system
- `--timing-verbosity=NONE/SUMMARY/BASIC/DETAILS` - Control timing output detail
