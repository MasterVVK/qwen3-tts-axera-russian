# Benchmarks

All measurements on FriendlyElec CM3588 (RK3588 4xA76+4xA55, 32GB RAM).

## Comparison: RKLLM vs llama.cpp Talker

| Metric | RKLLM (NPU) | llama.cpp (CPU A76) |
|---|---|---|
| Model size | 887 MB (w8a8) | ~469 MB (Q4_K_M) |
| Decode speed | ~17 tok/s | ~30-35 tok/s |
| Prefill (20 tok) | ~640ms | ~300ms |
| Cache thrashing | Yes (blocks ORT 4t) | No (separate from NPU) |

## Pipeline Configurations

### Current (RKLLM, from qwen3-tts-rknn-russian)

| Talker | Code Predictor | Vocoder | Per-step | RTF |
|--------|---------------|---------|----------|-----|
| RKLLM w8a8 60ms | ONNX 1-thread 374ms | RKNN INT8 | 434ms | **6.2x** |

### Expected (llama.cpp)

| Talker | Code Predictor | Vocoder | Per-step | RTF |
|--------|---------------|---------|----------|-----|
| llama.cpp 30ms | ONNX 1-thread 374ms | RKNN INT8 | ~404ms | **~5.6x** |
| llama.cpp 30ms | ONNX 4-thread ~200ms | RKNN INT8 | ~230ms | **~3.2x** |

*Note: 4-thread ORT should work better without RKLLM NPU contention.*

## Memory Usage

| Component | RAM |
|-----------|-----|
| llama.cpp talker (GGUF Q4_K_M) | ~600 MB |
| text_embedding.npy | 1.2 GB |
| code_predictor (weights + ONNX) | ~550 MB |
| RKNN vocoder | ~150 MB |
| **Total** | **~2.5 GB** |

*Lower than RKLLM version (~2.8 GB) due to Q4_K_M quantization.*
