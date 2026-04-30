#!/bin/bash
# Run NRE with Tracy profiling enabled

set -e

# Parse command line arguments
GPU_PROFILING=false
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_PROFILING=true
            shift
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            # Not a flag, assume it's the start of the command
            break
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: $0 [OPTIONS] <command> [args...]"
    echo ""
    echo "Run NRE with Tracy profiling enabled"
    echo ""
    echo "Options:"
    echo "  --gpu                Enable GPU profiling (requires CUPTI)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 serve-grpc --host 127.0.0.1"
    echo "  $0 --gpu serve-grpc"
    echo ""
    echo "Notes:"
    echo "  - Tracy GUI should be running and listening on port 8086"
    echo "  - Script automatically uses appropriate Bazel target"
    echo "  - GPU profiling requires CUDA toolkit with CUPTI libraries"
    exit 0
fi

# Check if command was provided
if [ $# -eq 0 ]; then
    echo "ERROR: No command provided"
    echo "Use --help for usage information"
    exit 1
fi

echo "🔍 Starting Tracy profiling..."

# Setup GPU profiling if requested
if [ "$GPU_PROFILING" = true ]; then
    echo "🎮 GPU profiling enabled"
    
    # Find and set CUPTI path for both runtime and build system
    CUPTI_ROOT=""
    
    # Check if CUPTI_PATH is already set
    if [ -n "$CUPTI_PATH" ]; then
        CUPTI_ROOT="$CUPTI_PATH"
        echo "Using CUPTI_PATH from environment: $CUPTI_ROOT"
    else
        # Try to find CUPTI installation
        for base_path in /usr/local/cuda/extras/CUPTI /usr/local/cuda-*/extras/CUPTI /opt/cuda/extras/CUPTI; do
            if [ -d "$base_path" ] && [ -d "$base_path/include" ]; then
                CUPTI_ROOT="$base_path"
                break
            fi
        done
        
        # Also check CUDA_HOME if set
        if [ -z "$CUPTI_ROOT" ] && [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/extras/CUPTI" ]; then
            CUPTI_ROOT="$CUDA_HOME/extras/CUPTI"
        fi
    fi

    if [ -z "$CUPTI_ROOT" ]; then
        echo "⚠️  WARNING: CUPTI not found. GPU profiling will not work."
        echo "   Set CUPTI_PATH environment variable or install CUDA toolkit."
        echo "   Example: export CUPTI_PATH=/usr/local/cuda/extras/CUPTI"
    else
        # Set CUPTI_PATH for Bazel build system
        export CUPTI_PATH="$CUPTI_ROOT"
        
        # Set LD_LIBRARY_PATH for runtime
        for lib_dir in lib64 lib; do
            if [ -d "$CUPTI_ROOT/$lib_dir" ]; then
                export LD_LIBRARY_PATH="$CUPTI_ROOT/$lib_dir:$LD_LIBRARY_PATH"
                break
            fi
        done
        
        echo "✅ CUPTI found: $CUPTI_ROOT"
    fi

else
    echo "💻 CPU-only profiling"
fi

echo "Command: $*"
echo ""
echo "Tracy should be listening on port 8086"
echo "   Connect with Tracy GUI to see real-time profiling data"
echo ""

# Use the appropriate Tracy-enabled Bazel target (they have different dependencies)
if [ "$GPU_PROFILING" = true ]; then
    BAZEL_TARGET="//:run_with_tracy_gpu"
    PROFILE_TYPE="(with GPU profiling)"
else
    BAZEL_TARGET="//:run_with_tracy"
    PROFILE_TYPE="(CPU-only profiling)"
fi

# Build the command  
BAZEL_CMD="bazel run $BAZEL_TARGET -- $* --profiling-backend=TRACY --enable-timing"

echo "Building and running: $BAZEL_CMD $PROFILE_TYPE"

# Run the Tracy-enabled target (environment variables are set by the Bazel target)
exec bazel run "$BAZEL_TARGET" -- "$@" --profiling-backend=TRACY --enable-timing
