# Model Conversion Guide

## Overview

Qwen3-TTS requires converting 4 components:

| Component | Source | Target | Tool |
|---|---|---|---|
| Talker LLM | safetensors | GGUF Q4_K_M | llama.cpp convert_hf_to_gguf.py |
| Embeddings | safetensors | numpy .npy | extract_embeddings.py |
| Code Predictor | safetensors | ONNX + numpy | export scripts |
| Vocoder | PyTorch | RKNN INT8 | RKNN Toolkit |

## Step-by-Step

### 1. Download base model

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

### 2. Extract talker as Qwen3

```bash
python3 scripts/extract_talker_as_qwen3.py
# Output: ./qwen3_tts_talker_as_qwen3/
```

### 3. Convert talker to GGUF

```bash
python3 scripts/convert_talker_gguf.py \
    --model_dir ./qwen3_tts_talker_as_qwen3 \
    --output qwen3_tts_talker_q4_k_m.gguf \
    --quantize q4_k_m
# Output: ~469 MB
```

### 4. Extract embeddings

```bash
python3 scripts/extract_embeddings.py
# Output: ./qwen3_tts_embeddings/ (~1.2 GB)
```

### 5. Export code predictor

```bash
# Numpy weights (required)
python3 scripts/export_code_predictor_weights.py
# Output: code_predictor_weights.npz (~240 MB)

# ONNX model (required for ONNX backend)
python3 scripts/export_code_predictor_onnx.py
# Output: code_predictor_decode_step.onnx (~314 MB)
```

### 6. Export and convert vocoder

```bash
# Export ONNX
python3 scripts/export_vocoder_traced.py --codes_length 64 --simplify --remove_isnan

# Convert to RKNN
python3 scripts/convert_vocoder_rknn.py --model vocoder_traced_64.onnx --quantize
```

### 7. Deploy to CM3588

```bash
ssh root@cm3588 "mkdir -p /root/tts-rknn/{embeddings,code_predictor}"

scp qwen3_tts_talker_q4_k_m.gguf root@cm3588:/root/tts-rknn/
scp -r qwen3_tts_embeddings/* root@cm3588:/root/tts-rknn/embeddings/
scp code_predictor_weights.npz root@cm3588:/root/tts-rknn/code_predictor/
scp code_predictor_decode_step.onnx root@cm3588:/root/tts-rknn/code_predictor/
scp vocoder_traced_64_sim_q8.rknn root@cm3588:/root/tts-rknn/

# Copy scripts
scp -r dual_npu/ root@cm3588:/root/qwen3-tts-axera/
```

### 8. Build llama_wrapper.so on CM3588

```bash
cd /root/qwen3-tts-axera/dual_npu
gcc -shared -fPIC -O2 -o llama_wrapper.so llama_wrapper.c \
    -I/root/llama.cpp/include -I/root/llama.cpp/ggml/include \
    -L/root/llama.cpp/build/bin \
    -lllama -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,/root/llama.cpp/build/bin
```

### 9. Test

```bash
bash dual_npu/launch_qwen3_tts.sh "Test synthesis"
```
