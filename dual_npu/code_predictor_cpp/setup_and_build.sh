#!/bin/bash
# Setup, build and optionally run C++ Code Predictor server
# Run on CM3588: bash setup_and_build.sh [--run]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ORT_VERSION="1.24.1"
ORT_INCLUDE="$SCRIPT_DIR/ort_include"
WEIGHTS_DIR="/root/tts-rknn/code_predictor/weights_npy"

echo "=== Step 1: Download ONNX Runtime C++ headers ==="
if [ ! -f "$ORT_INCLUDE/onnxruntime_cxx_api.h" ]; then
    mkdir -p "$ORT_INCLUDE"
    BASE="https://raw.githubusercontent.com/microsoft/onnxruntime/v${ORT_VERSION}/include/onnxruntime/core/session"
    for header in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h onnxruntime_run_options_config_keys.h onnxruntime_session_options_config_keys.h onnxruntime_ep_c_api.h; do
        echo "  Downloading $header..."
        wget -q -O "$ORT_INCLUDE/$header" "$BASE/$header" || \
        curl -sL -o "$ORT_INCLUDE/$header" "$BASE/$header"
    done
    echo "  Headers saved to $ORT_INCLUDE/"
else
    echo "  Headers already present"
fi

echo ""
echo "=== Step 2: Create libonnxruntime.so symlink ==="
ORT_SO="/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so"
if [ ! -f "$ORT_SO" ]; then
    ORT_SO_FULL=$(find /usr/local/lib/python3.11/dist-packages/onnxruntime/capi/ -name "libonnxruntime.so.*" | head -1)
    if [ -n "$ORT_SO_FULL" ]; then
        ln -sf "$ORT_SO_FULL" "$ORT_SO"
        echo "  Symlink: $ORT_SO -> $ORT_SO_FULL"
    else
        echo "  ERROR: libonnxruntime.so not found!"
        exit 1
    fi
else
    echo "  Symlink already exists"
fi

echo ""
echo "=== Step 3: Extract weights from .npz ==="
if [ ! -f "$WEIGHTS_DIR/lm_head_0.npy" ]; then
    python3 "$SCRIPT_DIR/extract_weights.py"
else
    echo "  Weights already extracted in $WEIGHTS_DIR"
fi

echo ""
echo "=== Step 4: Build ==="
mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
echo "  Built: $SCRIPT_DIR/build/code_predictor_server"

echo ""
echo "=== Done ==="
ls -lh "$SCRIPT_DIR/build/code_predictor_server"

if [ "$1" = "--run" ]; then
    echo ""
    echo "=== Running ==="
    # Pin to A76 cores (4-7) like the Python version
    exec taskset -c 4-7 "$SCRIPT_DIR/build/code_predictor_server" "$@"
fi
