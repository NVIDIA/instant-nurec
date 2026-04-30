#!/bin/bash
# Run NRE with NVIDIA NSight Systems (NSys) profiling

set -e

# Default output filename
OUTPUT_FILE="nre_profile_$(date +%Y%m%d_%H%M%S).nsys"
TRACE_OPTIONS="cuda,nvtx"
SHOW_OUTPUT=true
CAPTURE_MODE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --trace)
            TRACE_OPTIONS="$2"
            shift 2
            ;;
        --quiet)
            SHOW_OUTPUT=false
            shift
            ;;
        --no-capture)
            CAPTURE_MODE=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] <command> [args...]"
            echo ""
            echo "Run NRE with NVIDIA NSight Systems profiling"
            echo ""
            echo "Options:"
            echo "  -o, --output FILE    Output file name (default: nre_profile_TIMESTAMP.nsys)"
            echo "  --trace APIS         APIs to trace (default: cuda,nvtx)"
            echo "  --quiet              Don't show nsys output"
            echo "  --no-capture         Just run without NSys capture (for testing)"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 serve-grpc --host 127.0.0.1"
            echo "  $0 -o my_profile.nsys serve-grpc"
            echo "  $0 --trace cuda,nvtx serve-grpc"
            echo "  $0 --no-capture serve-grpc    # Just run without profiling"
            exit 0
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

# Check if nsys is available (only if capturing)
if [ "$CAPTURE_MODE" = true ] && ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys command not found"
    echo "Please install NVIDIA NSight Systems:"
    echo "  https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# Check if command was provided
if [ $# -eq 0 ]; then
    echo "ERROR: No command provided"
    echo "Use --help for usage information"
    exit 1
fi

echo "🔍 Starting NSys profiling..."

if [ "$CAPTURE_MODE" = true ]; then
    # Ensure output directory exists
    OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
    mkdir -p "$OUTPUT_DIR"
    
    echo "📁 Output file: $OUTPUT_FILE"
    echo "Trace APIs: $TRACE_OPTIONS"
else
    echo "Running without capture (--no-capture mode)"
fi

echo "Command: $*"


echo ""

# Use the regular Bazel target with NSYS environment variable
BAZEL_TARGET="//:run"

if [ "$CAPTURE_MODE" = true ]; then
    # Build nsys command
    NSYS_CMD="nsys profile"
    NSYS_CMD="$NSYS_CMD --trace=$TRACE_OPTIONS"
    NSYS_CMD="$NSYS_CMD --output=$OUTPUT_FILE"
    NSYS_CMD="$NSYS_CMD --force-overwrite=true"
    NSYS_CMD="$NSYS_CMD --sample=none"  # Disable CPU sampling to reduce overhead

    # Add quiet flag if requested
    if [ "$SHOW_OUTPUT" = false ]; then
        NSYS_CMD="$NSYS_CMD --quiet"
    fi

    # Build the command
    BAZEL_CMD="bazel run $BAZEL_TARGET -- $* --profiling-backend=NVTX --enable-timing"
    
    # Run with NSys profiling
    echo "Building and running: $NSYS_CMD $BAZEL_CMD"
    echo ""
    echo "📝 Note: Press Ctrl+C to stop profiling and generate the profile file"
    echo ""

    # Run NSys profiling (NSys handles Ctrl+C gracefully)
    $NSYS_CMD $BAZEL_CMD
    
    echo ""
    
    # Check if profile was generated (NSys often exits with non-zero on Ctrl+C but still generates profile)
    if [ -f "$OUTPUT_FILE" ] || [ -f "${OUTPUT_FILE}.nsys-rep" ]; then
        echo "✅ Profiling completed successfully!"
        
        # Check which file was actually generated
        if [ -f "${OUTPUT_FILE}.nsys-rep" ]; then
            ACTUAL_FILE="${OUTPUT_FILE}.nsys-rep"
            echo "📊 Profile saved to: $ACTUAL_FILE"
        else
            ACTUAL_FILE="$OUTPUT_FILE"
            echo "📊 Profile saved to: $ACTUAL_FILE"
        fi
        
        echo ""
        echo "To view the profile:"
        echo "  nsys-ui $ACTUAL_FILE"
        echo ""
        echo "Or convert to other formats:"
        echo "  nsys stats $ACTUAL_FILE                    # Text summary"
        echo "  nsys export --type=sqlite $ACTUAL_FILE    # SQLite database"
    else
        echo "❌ Profiling failed - no profile file generated!"
        exit 1
    fi
else
    # Build the command
    BAZEL_CMD="bazel run $BAZEL_TARGET -- $* --profiling-backend=nvtx --enable-timing"
    
    # Just run without NSys
    echo "🔨 Building and running: $BAZEL_CMD"
    echo ""
    exec bazel run "$BAZEL_TARGET" -- "$@" --profiling-backend=nvtx --enable-timing
fi
