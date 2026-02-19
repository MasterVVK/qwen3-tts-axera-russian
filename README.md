# Qwen3-TTS on CM3588 (llama.cpp + AX650N)

Hybrid C++/Python pipeline for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) on CM3588 with llama.cpp for the talker LLM and optional AX650N NPU acceleration.

## Architecture

```
                   ┌─────────────────────┐
Text ──────────────┤  Tokenizer + Embed  ├──→ embeddings
                   └────────┬────────────┘
                            │
         ┌──────────────────▼──────────────────┐
         │    llamacpp_talker_server.py         │
         │    (llama.cpp GGUF Q4_K_M, A76)     │
         │    → hidden_state + code_0           │
         └──────────────────┬──────────────────┘
                            │ socket
         ┌──────────────────▼──────────────────┐
         │    code_predictor_server.py          │
         │    (ONNX Runtime, 1-thread)          │
         │    → codes 1-15                      │
         └──────────────────┬──────────────────┘
                            │ socket
         ┌──────────────────▼──────────────────┐
         │    vocoder_server.py                 │
         │    (RKNN NPU or AX650N)              │
         │    → audio PCM 24kHz                 │
         └──────────────────────────────────────┘
```

## Key Advantage: llama.cpp

The talker is a 28-layer Qwen3 transformer. llama.cpp provides:
- **~30-35 tok/s** on A76 cores (vs ~17 tok/s with RKLLM)
- Q4_K_M quantization (469 MB vs 887 MB)
- Custom embedding mode via `llama_wrapper.c`
- KV-cache persistence for repeated speakers

## Expected Performance

| Component | Time/step | Notes |
|---|---|---|
| Talker (llama.cpp A76) | ~30ms | 2x faster than RKLLM |
| Code Predictor (ONNX 1t) | ~374ms | Bottleneck (86% of time) |
| Vocoder (RKNN INT8) | ~6s total | End of pipeline |
| **Expected RTF** | | **~5.6x** (vs 6.2x with RKLLM) |

With 4-thread ORT for code predictor (possible without RKLLM interference):

| Component | Time/step | Notes |
|---|---|---|
| Talker (llama.cpp A76) | ~30ms | |
| Code Predictor (ONNX 4t) | ~200ms | No RKLLM cache thrashing |
| **Expected RTF** | | **~3.2x** |

## Quick Start

```bash
# Single-shot
bash dual_npu/launch_qwen3_tts.sh "Hello, how are you?"

# Daemon mode (persistent servers)
bash dual_npu/launch_qwen3_tts.sh --daemon

# Client
python3 dual_npu/tts_client.py "Text to speak" --language russian
```

## Model Preparation

1. Extract talker as Qwen3: `python3 scripts/extract_talker_as_qwen3.py`
2. Convert to GGUF: `python3 scripts/convert_talker_gguf.py`
3. Extract embeddings: `python3 scripts/extract_embeddings.py`
4. Export code predictor: `python3 scripts/export_code_predictor_weights.py`
5. Export vocoder: `python3 scripts/export_vocoder_traced.py`
6. Convert vocoder to RKNN: `python3 scripts/convert_vocoder_rknn.py`

See [docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md) for details.

## Build llama_wrapper.so

```bash
cd dual_npu
gcc -shared -fPIC -O2 -o llama_wrapper.so llama_wrapper.c \
    -I/root/llama.cpp/include -I/root/llama.cpp/ggml/include \
    -L/root/llama.cpp/build/bin \
    -lllama -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,/root/llama.cpp/build/bin
```

## License

Apache-2.0. Model weights: [Qwen3-TTS license](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
