# Qwen3-TTS on CM3588 — RTF 2.0x (llama.cpp + GGML)

High-performance C++/Python pipeline for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) on FriendlyElec CM3588 (RK3588).

**Achieved RTF 2.0x** — the fastest Qwen3-TTS on edge hardware (vs RTF 5.5x baseline with RKLLM).

## Architecture

```
                   ┌─────────────────────┐
Text ──────────────┤  Tokenizer + Embed  ├──→ embeddings
                   └────────┬────────────┘
                            │
         ┌──────────────────▼──────────────────┐
         │    llamacpp_talker_server.py         │
         │    llama.cpp GGUF Q4_K_M (A76)      │
         │    ~30-35 tok/s, ~30ms/step          │
         └──────────────────┬──────────────────┘
                            │ unix socket
         ┌──────────────────▼──────────────────┐
         │    code_pred_server (GGML Q4_0)     │
         │    4.8x faster than ONNX            │
         │    ~55ms/step                        │
         └──────────────────┬──────────────────┘
                            │ unix socket
         ┌──────────────────▼──────────────────┐
         │    vocoder_server.py                 │
         │    ONNX FP32 (A76, 4 threads)       │
         │    overlap-crossfade for chunks      │
         └──────────────────────────────────────┘
```

Three independent servers communicate via Unix sockets. Each can be restarted independently.

## Performance

Tested on FriendlyElec CM3588 (RK3588 4xA76+4xA55, 32GB RAM):

| Component | Time | Notes |
|---|---|---|
| Talker (llama.cpp A76) | ~30ms/step | GGUF Q4_K_M, 469 MB |
| Code Predictor (GGML Q4_0) | ~55ms/step | **4.8x faster than ONNX** |
| Vocoder (ONNX FP32) | ~5.0s/64-tok | Overlap-crossfade for chunks |
| **Total RTF** | | **2.0x** |

### Optimization Journey

| Version | Code Predictor | Vocoder | RTF |
|---|---|---|---|
| Baseline (RKLLM talker) | ONNX 1-thread (374ms) | RKNN Q8 | 5.5x |
| + llama.cpp talker | ONNX 1-thread (374ms) | RKNN Q8 | 5.5x |
| + 2-thread ONNX CP | ONNX 2-thread (265ms) | RKNN Q8 | 4.4x |
| + GGML Q4_0 CP | **GGML (55ms)** | RKNN Q8 | 2.2x |
| + ONNX FP32 vocoder | GGML (55ms) | **ONNX FP32** | **2.0x** |

### Why ONNX FP32 Vocoder (not RKNN Q8)?

The vocoder uses **Snake activation** (`x + sin²(α·x)/α`) which is fundamentally incompatible with quantization:
- **RKNN Q8**: 1.75x faster but audible noise (SNR 9.5 dB, corr 0.94)
- **ONNX INT8**: 3x *slower* + poor quality (SNR 4.2 dB)
- **AX650N NPU**: Conv1D dilation=9 not supported
- **MNN GPU**: Shape broadcast error in SineGen

All quantization paths exhausted — FP32 is the only option for acceptable quality.

## Quick Start

```bash
# Single-shot synthesis
bash dual_npu/launch_qwen3_tts.sh "Привет, как дела?"

# Daemon mode (persistent servers)
bash dual_npu/launch_qwen3_tts.sh --daemon

# Client (while daemon is running)
python3 dual_npu/tts_client.py "Text to speak" --language russian

# With streaming vocoder (overlap with generation)
python3 dual_npu/tts_client.py "Long text..." --language russian --streaming
```

## Code Predictor Variants

The pipeline supports three code predictor backends (auto-selected):

| Backend | Speed | Quality | Notes |
|---|---|---|---|
| **GGML Q4_0** (default) | 55ms/step | Best | Requires [qwen3-tts.cpp](https://github.com/nickovchinnikov/qwen3-tts.cpp) |
| C++ ONNX Runtime | 240ms/step | Good | Built-in, `dual_npu/code_predictor_cpp/` |
| Python ONNX Runtime | 248ms/step | Good | Fallback, no compilation needed |

## Model Preparation

1. Extract talker as Qwen3: `python3 scripts/extract_talker_as_qwen3.py`
2. Convert to GGUF: `python3 scripts/convert_talker_gguf.py`
3. Extract embeddings: `python3 scripts/extract_embeddings.py`
4. Export code predictor: `python3 scripts/export_code_predictor_weights.py`
5. Export vocoder: `python3 scripts/export_vocoder_traced.py`
6. (Optional) Convert GGML code predictor: see [qwen3-tts.cpp](https://github.com/nickovchinnikov/qwen3-tts.cpp)

See [docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md) for details.

## Model Files

```
/root/tts-rknn/
├── qwen3_tts_talker_q4_k_m.gguf        # Talker LLM (469 MB)
├── embeddings/
│   ├── text_embedding.npy               # [151936, 2048]
│   ├── text_projection_*.npy            # MLP weights
│   ├── codec_embedding.npy              # [3072, 1024]
│   └── codec_head.npy                   # [3072, 1024]
├── code_predictor/
│   ├── code_predictor_weights.npz       # All weights (240 MB)
│   └── code_predictor_decode_step.onnx  # ONNX model (314 MB)
└── vocoder/
    └── vocoder_traced_64.onnx           # Vocoder FP32 (93 MB)
```

## Build llama_wrapper.so

```bash
cd dual_npu
gcc -shared -fPIC -O2 -o llama_wrapper.so llama_wrapper.c \
    -I/root/llama.cpp/include -I/root/llama.cpp/ggml/include \
    -L/root/llama.cpp/build/bin \
    -lllama -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,/root/llama.cpp/build/bin
```

## Comparison with CosyVoice3

| | Qwen3-TTS (this project) | [CosyVoice3](https://github.com/nickovchinnikov/cosyvoice3-axera-russian) |
|---|---|---|
| RTF | 2.0x | **1.26x** |
| Architecture | Talker → CP → Vocoder | LLM → Flow DiT → HiFT |
| NPU (AX650N) | Not used | Flow DiT on NPU |
| Audio quality | Good | Good |
| Token rate | 12 Hz | 25 Hz |

CosyVoice3 achieves better RTF because its Flow DiT (MatMul+Attention) runs on AX650N NPU, freeing CPU for LLM and HiFT. Qwen3-TTS has no NPU-friendly component — the vocoder uses Snake activation which is incompatible with all NPU/quantization paths.

## Supported Languages

chinese, english, german, russian, french, japanese, korean

## License

Apache-2.0. Model weights: [Qwen3-TTS license](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
