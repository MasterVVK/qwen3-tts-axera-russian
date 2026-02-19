# Architecture

## Qwen3-TTS vs CosyVoice3

```
CosyVoice3:
  LLM (Qwen2) → speech tokens → Flow DiT (ODE solver) → mel → HiFT vocoder → audio
  896-dim        6561 vocab      AX650N NPU              80-dim  ConvNet         24kHz

Qwen3-TTS:
  Talker (Qwen3) → code_0 → Code Predictor (15 groups) → Vocoder (ConvNet) → audio
  1024-dim         2048 vocab  5 layers × 15 steps        16 codebooks         24kHz
```

Key differences:
1. **No Flow DiT**: Qwen3-TTS uses discrete codec tokens, not continuous mel
2. **Code Predictor bottleneck**: 15 sequential transformer passes per token (86% of time)
3. **Simpler vocoder**: Direct codec-to-audio ConvNet (no ODE solver)

## Multi-Server Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    tts_client.py                        │
│  Orchestrates the pipeline, handles feedback loop       │
└────┬──────────────────┬────────────────────┬────────────┘
     │ Unix socket      │ Unix socket        │ Unix socket
     ▼                  ▼                    ▼
┌────────────┐   ┌──────────────┐   ┌──────────────┐
│  Talker    │   │Code Predictor│   │   Vocoder    │
│ llama.cpp  │   │  ONNX RT     │   │  RKNN NPU   │
│  A76 cores │   │  A76 cores   │   │  or AX650N  │
│  GGUF Q4KM │   │  1-thread    │   │  INT8       │
└────────────┘   └──────────────┘   └──────────────┘
```

### Why Multi-Server?

1. **CPU affinity**: Each server pinned to specific cores
2. **Independent lifecycle**: Restart one without killing others
3. **KV-cache persistence**: Talker keeps state between requests
4. **Future**: Code predictor could move to AX650N NPU

### Protocol

**Talker Server** (bidirectional):
```
Client → Server: JSON {"text": "...", "language": "russian"}
Server → Client: [code_0 int32] [hidden float32[1024]] per step
Client → Server: [feedback float32[1024]] per step
Server → Client: [-1 int32] when done
```

**Code Predictor Server** (request-response):
```
Client → Server: [hidden float32[1024]] [code_0 int32]
Server → Client: [15 × int32 codes]
```

**Vocoder Server** (batch):
```
Client → Server: [n_tokens int32] [codes int64[n×16]]
Server → Client: [n_samples int32] [audio int16[n_samples]]
```

## llama.cpp Integration

### Embedding Mode

Standard llama.cpp returns logits. We need hidden states. Solution:

1. Build llama.cpp with `--embeddings` support
2. `llama_wrapper.c` wraps the batch API for custom embedding input
3. `llama_set_embeddings(ctx, true)` enables last-layer extraction
4. `llama_get_embeddings_ith(ctx, last_token)` returns hidden state

### Sampling code_0

Hidden state from llama.cpp is used for code_0 sampling in Python:
```python
logits = hidden_state @ codec_head.T  # [1024] @ [2048, 1024].T → [2048]
code_0 = top_k_sample(logits[:2048], temperature=0.8, top_k=50)
```

### KV-Cache Persistence

Same mechanism as CosyVoice3:
- `wrapper_state_save_file()` saves full KV state to disk
- `wrapper_state_load_file()` restores it
- Keyed by MD5 hash of prefix embeddings
- Saves ~1-2s prefill time for repeated speaker+text combinations

## Performance Analysis

### Bottleneck: Code Predictor

The code predictor runs 15 sequential steps per generated token:
```
For each token:
  Step 0: prefill [hidden, code_0_embed] → group_0
  Steps 1-14: decode with KV-cache → groups 1-14

Each step: 5-layer transformer, 1024-dim, GQA 16/8 heads
Total: 15 × 25ms = 374ms/token (ONNX 1-thread)
```

This is **86% of per-token time** (374ms out of 434ms).

### Acceleration Options

1. **4-thread ONNX**: May reduce to ~200ms without RKLLM interference
2. **AX650N for code predictor**: Convert 5-layer transformer to axmodel
3. **Batch multiple groups**: Requires model restructuring
