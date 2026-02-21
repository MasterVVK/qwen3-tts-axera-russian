#!/bin/bash
# Launch Qwen3-TTS in multi-server mode:
#   Talker:         llama.cpp CPU A76 (GGUF Q4_K_M, embedding mode)
#   Code Predictor: GGML Q4_0 (~55ms) | ONNX C++ (~240ms) | ONNX Python (~248ms)
#   Vocoder:        RKNN NPU or ONNX CPU
#
# Usage:
#   bash launch_qwen3_tts.sh "Привет, как дела?"
#   bash launch_qwen3_tts.sh                          # Default text
#   bash launch_qwen3_tts.sh --daemon                 # Daemon mode
#
# Architecture (vs CosyVoice3):
#   CosyVoice3:  LLM -> tokens -> Flow DiT (ODE) -> mel -> HiFT -> audio
#   Qwen3-TTS:   Talker -> code_0 -> Code Predictor (15 groups) -> Vocoder -> audio

set -e

export HF_HUB_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Configuration ---
BASE_DIR="${BASE_DIR:-/root/tts-rknn}"

# Talker (llama.cpp)
GGUF_MODEL="${GGUF_MODEL:-${BASE_DIR}/qwen3_tts_talker_q4_k_m.gguf}"
GGUF_THREADS="${GGUF_THREADS:-4}"

# Embeddings
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-${BASE_DIR}/embeddings}"

# Code Predictor
CP_DIR="${CP_DIR:-${BASE_DIR}/code_predictor}"
CP_THREADS="${CP_THREADS:-3}"

# GGML Code Predictor (4.8x faster than ONNX)
CP_GGML_BINARY="${CP_GGML_BINARY:-/root/qwen3-tts.cpp/build/code_pred_server}"
CP_GGML_MODEL="${CP_GGML_MODEL:-/root/qwen3-tts.cpp/models/qwen3-tts-0.6b-q4_0.gguf}"

# Vocoder (ONNX FP32 — RKNN Q8 has unusable quality due to Snake activation)
VOCODER_MODEL="${VOCODER_MODEL:-${BASE_DIR}/vocoder/vocoder_traced_64.onnx}"

# Sockets
TALKER_SOCKET="/tmp/qwen3_talker.sock"
CP_SOCKET="/tmp/qwen3_cp.sock"
VOC_SOCKET="/tmp/qwen3_voc.sock"

# Generation params
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-50}"
MAX_TOKENS="${MAX_TOKENS:-200}"
LANGUAGE="${LANGUAGE:-russian}"

# --- Parse arguments ---
DAEMON_MODE=false
TEXT=""
for arg in "$@"; do
    if [ "$arg" = "--daemon" ]; then
        DAEMON_MODE=true
    else
        TEXT="$arg"
    fi
done

if [ "$DAEMON_MODE" = false ] && [ -z "$TEXT" ]; then
    TEXT="Привет, как дела? Сегодня хорошая погода для прогулки."
fi

# --- Functions ---
cleanup() {
    echo ""
    echo "Shutting down..."
    for pid_var in TALKER_PID CP_PID VOC_PID; do
        pid="${!pid_var}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null || true
            echo "  Stopped PID $pid ($pid_var)"
        fi
    done
    rm -f "$TALKER_SOCKET" "$CP_SOCKET" "$VOC_SOCKET"
}
trap cleanup EXIT

wait_for_socket() {
    local sock_path="$1"
    local pid="$2"
    local name="$3"
    local timeout="${4:-60}"

    for i in $(seq 1 $((timeout * 2))); do
        if [ -S "$sock_path" ]; then
            echo "  $name ready (PID $pid)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $name process died"
            return 1
        fi
        sleep 0.5
    done
    echo "ERROR: $name socket not created after ${timeout}s"
    return 1
}

# --- Pre-flight checks ---
echo "=== Qwen3-TTS Multi-Server Pipeline ==="
echo "Talker:         llama.cpp CPU A76 (GGUF, ~30-35 tok/s)"
if [ "${USE_GGML_CP:-1}" = "1" ] && [ -x "$CP_GGML_BINARY" ] && [ -f "$CP_GGML_MODEL" ]; then
    echo "Code Predictor: GGML Q4_0 (4 A76 cores, ~55ms/step) ***"
elif [ "${USE_CPP_CP:-1}" = "1" ] && [ -x "${SCRIPT_DIR}/code_predictor_cpp/build/code_predictor_server" ]; then
    echo "Code Predictor: C++ ONNX Runtime (3-thread, batch prefill, ~240ms/step)"
else
    echo "Code Predictor: Python ONNX Runtime (3-thread, ~248ms/step)"
fi
VOC_NAME="$(basename $VOCODER_MODEL)"
if [[ "$VOC_NAME" == *.rknn ]]; then
    echo "Vocoder:        $VOC_NAME (RKNN NPU, ~2.85s/64-tok chunk)"
else
    echo "Vocoder:        $VOC_NAME (ONNX CPU, ~5.0s/64-tok chunk)"
fi
echo ""

if [ ! -f "$GGUF_MODEL" ]; then
    echo "ERROR: GGUF model not found: $GGUF_MODEL"
    echo "  Convert: python3 scripts/convert_talker_gguf.py"
    exit 1
fi

# --- Step 1: Start Talker Server ---
echo "Starting Talker Server (A76 cores 4-7)..."
rm -f "$TALKER_SOCKET"

OPENBLAS_NUM_THREADS=1 taskset -c 4-7 python3 -u "${SCRIPT_DIR}/llamacpp_talker_server.py" \
    --model "$GGUF_MODEL" \
    --embeddings "$EMBEDDINGS_DIR" \
    --socket "$TALKER_SOCKET" \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --max_tokens "$MAX_TOKENS" \
    --threads "$GGUF_THREADS" &
TALKER_PID=$!

wait_for_socket "$TALKER_SOCKET" "$TALKER_PID" "Talker" 60 || exit 1

# --- Step 2: Start Code Predictor Server ---
CP_CPP="${SCRIPT_DIR}/code_predictor_cpp/build/code_predictor_server"
if [ "${USE_GGML_CP:-1}" = "1" ] && [ -x "$CP_GGML_BINARY" ] && [ -f "$CP_GGML_MODEL" ]; then
    echo "Starting Code Predictor Server [GGML Q4_0] (A76 cores 4-7)..."
    rm -f "$CP_SOCKET"
    taskset -c 4-7 "$CP_GGML_BINARY" \
        --model "$CP_GGML_MODEL" \
        --socket "$CP_SOCKET" \
        --temperature 0.1 \
        --top_k "$TOP_K" &
    CP_PID=$!
elif [ "${USE_CPP_CP:-1}" = "1" ] && [ -x "$CP_CPP" ]; then
    echo "Starting Code Predictor Server [C++ ONNX] (A76 cores 4-7)..."
    rm -f "$CP_SOCKET"
    taskset -c 4-7 "$CP_CPP" \
        --model "${CP_DIR}/code_predictor_decode_step.onnx" \
        --weights "${CP_DIR}/weights_npy" \
        --codec_emb "${EMBEDDINGS_DIR}/codec_embedding.npy" \
        --socket "$CP_SOCKET" \
        --threads "$CP_THREADS" &
    CP_PID=$!
else
    echo "Starting Code Predictor Server [Python ONNX] (A76 cores 4-7)..."
    rm -f "$CP_SOCKET"
    OPENBLAS_NUM_THREADS=1 taskset -c 4-7 python3 -u "${SCRIPT_DIR}/code_predictor_server.py" \
        --model_dir "$CP_DIR" \
        --embeddings_dir "$EMBEDDINGS_DIR" \
        --socket "$CP_SOCKET" \
        --threads "$CP_THREADS" &
    CP_PID=$!
fi

wait_for_socket "$CP_SOCKET" "$CP_PID" "Code Predictor" 30 || exit 1

# --- Step 3: Start Vocoder Server ---
echo "Starting Vocoder Server..."
rm -f "$VOC_SOCKET"

# Vocoder on A76 cores (runs after generation, no contention)
OPENBLAS_NUM_THREADS=1 taskset -c 4-7 python3 -u "${SCRIPT_DIR}/vocoder_server.py" \
    --model "$VOCODER_MODEL" \
    --socket "$VOC_SOCKET" &
VOC_PID=$!

wait_for_socket "$VOC_SOCKET" "$VOC_PID" "Vocoder" 30 || exit 1

# --- Step 4: Run client ---
echo ""

if [ "$DAEMON_MODE" = true ]; then
    echo "Daemon mode: all servers running."
    echo "Use: python3 tts_client.py \"text to speak\""
    echo ""
    echo "Press Ctrl+C to stop."
    wait
else
    echo "Running TTS: \"$TEXT\""
    echo ""

    python3 "${SCRIPT_DIR}/tts_client.py" \
        "$TEXT" \
        --language "$LANGUAGE" \
        --output output.wav \
        --talker_socket "$TALKER_SOCKET" \
        --cp_socket "$CP_SOCKET" \
        --voc_socket "$VOC_SOCKET" \
        --embeddings_dir "$EMBEDDINGS_DIR" \
        --cp_dir "$CP_DIR"

    echo ""
    echo "Output: output.wav"
fi
