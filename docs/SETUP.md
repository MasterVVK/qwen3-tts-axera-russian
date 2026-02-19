# Setup Guide

## Hardware

- **CM3588**: RK3588 (4xA76 + 4xA55), 32GB RAM
- **Optional**: M5Stack AI-8850 (AX650N NPU) via PCIe M.2

## Software Requirements

### On CM3588

```bash
# Python 3.10+
pip3 install numpy onnxruntime transformers

# llama.cpp (build from source)
cd /root
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Build llama_wrapper.so
cd /root/qwen3-tts-axera/dual_npu
gcc -shared -fPIC -O2 -o llama_wrapper.so llama_wrapper.c \
    -I/root/llama.cpp/include -I/root/llama.cpp/ggml/include \
    -L/root/llama.cpp/build/bin \
    -lllama -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,/root/llama.cpp/build/bin

# RKNN Lite (for vocoder)
pip3 install rknn-toolkit-lite2
```

### On host (for model conversion)

```bash
pip install torch safetensors transformers qwen-tts
pip install onnx onnxruntime onnxsim
```

## Deploy Models

```bash
ssh root@cm3588 "mkdir -p /root/tts-rknn/{embeddings,code_predictor}"

scp qwen3_tts_talker_q4_k_m.gguf root@cm3588:/root/tts-rknn/
scp -r embeddings/* root@cm3588:/root/tts-rknn/embeddings/
scp code_predictor/* root@cm3588:/root/tts-rknn/code_predictor/
scp vocoder_traced_64_sim_q8.rknn root@cm3588:/root/tts-rknn/
```

## CPU Core Affinity

Critical for performance on RK3588:
- **A76 cores (4-7)**: llama.cpp talker + code predictor
- **A55 cores (0-3)**: Available for other tasks
- `OPENBLAS_NUM_THREADS=1` prevents numpy stealing A76 cores

The launch script handles this automatically via `taskset`.
