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
         │    RKNN FP16 (NPU, 3 cores)          │
         │    stateful streaming (no seams)     │
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

Measured on a stock RK3588 board (4×A76 pinned, no Axera NPU, Russian text): RTF 3.9-4.4x as
shipped. The code predictor then cost 199 ms per 15-step request; after removing the
`ggml_cast(ffn_down, F32)` nodes from `qwen3-tts.cpp/src/tts_transformer.cpp` (the CPU backend
takes F16/Q8_0 weights in `mul_mat` directly — a cast node dequantizes 12.6 MB per layer per
step, 75 times per token) it is 64 ms, generation goes 4.5 → 11.5 tok/s and RTF lands at
**2.6x**. Moving the vocoder to the RK3588 NPU as **RKNN FP16** (lossless, SNR 56.8 dB — see
below) takes streaming to **RTF 1.8x** (generation 8.9 tok/s concurrent with vocoding, first
chunk ~3 s).

### Vocoder: FP32 ONNX on CPU vs RKNN on NPU

The vocoder uses **Snake activation** (`x + sin²(α·x)/α`), which is fundamentally incompatible with *quantization*:
- **RKNN Q8**: 1.75x faster but audible noise (SNR 9.5 dB with random calibration, 13.4 dB with real recorded requests)
- **ONNX INT8**: 3x *slower* + poor quality (SNR 4.2 dB)
- **AX650N NPU**: Conv1D dilation=9 not supported
- **MNN GPU**: Shape broadcast error in SineGen

**RKNN FP16 (`w16a16`) is not quantized and measures lossless** (SNR 56.8 dB, corr 1.0000 vs the
FP32 reference) at 1.34x the 4-core CPU speed — and it runs on the NPU, so it no longer steals
A76 cores from the code predictor. That is why streaming now wins. Deploy:
`python3 scripts/convert_vocoder_rknn.py --model vocoder_traced_64.onnx` (no `--quantize`), point
`VOCODER_MODEL` at the result, and export `PYTHONPATH=dual_npu/rknnlite_shim` (the shim provides
the `RKNNLite` API over `librknnrt.so` for board Pythons newer than the lite2 wheels).

### Vocoder chunk seams

The WavTokenizer decoder is **causal**, so a token voiced at a chunk start with zero padding is a
*different waveform* than the same token voiced mid-sequence (measured: corr 0.445, SNR 0.3 dB).
The original 16-token zero-pad overlap crossfade therefore blended two unaligned signals — an
audible "double voice" at every boundary. Chunking now works in layers, each measured against a
single 256-token ONNX run of the same codes:

- **Real context, not padding**: prepend the previous chunk's tail tokens (48 for the 112-slot
  model; short tail chunks fill every free slot with context) and trim their audio. Boundary SNR:
  4.1 dB @16 ctx → 9.9 @32 → 19.1 @48 → 19.8 @60 (saturates; the receptive field is far wider).
- **Seams in pauses**: adjacent segments cannot be crossfaded (sum of 19 dB-decorrelated waves =
  comb-filter crackle) and dropping speech is audible, so the seam *moves*: the longest contiguous
  quiet run (≥150 ms, 20 ms-RMS < 25% of window RMS) in the last 3 s of a chunk becomes its end,
  and `stitch()` fade-drops only silence there (equal-power cos/sin windows). Mid-phrase fallback:
  20 ms equal-power micro-blend at the quietest point near the seam.
- **Warmup-truncate** (CosyVoice-style overlap): each chunk decodes 2 tokens past its seam, the
  next starts at the seam, and the shared region — which contains the contextless conv transient
  at the next chunk's head — is dropped in the pause or blended at a loud seam. Seam
  max|diff| vs the single-run reference: 97–108 m → 0.1–9.5 m.

End to end (RKNN FP16, 112-slot): global SNR to the single-run reference 26–32 dB, seam-local
25–57 dB; the residual is FP16 re-decode relaxation after each seam. The exact fix is the
stateful export below — it has no seams at all, and this scheme stays as the fallback for
boards without the NPU path.

### Vocoder: stateful streaming (no chunks)

The WavTokenizer v2 decoder is *fully causal*, so its whole history can live in the model's own
I/O instead of being re-fed per chunk. `scripts/export_vocoder_stateful.py` exports one streaming
step: every causal conv's tail (`(k-1)*d` inputs, `ceil((k-1)/s)` for the transposed convs) and the
pre-transformer's sliding-window KV (71 keys × 8 layers × 2 × 16 heads × 64) are packed into **one
flat float32 cache tensor — 1303424 floats, 5.2 MB — which is both an input and an output**:

```
audio_codes [1,16,64] int64 ─┐
freqs_in    [64,32]   fp32  ─┤→ step →  audio_values [1,122880]  (64 tokens = 5.12 s of 24 kHz)
cache_valid [1]       fp32  ─┤          cache_out   [1303424] fp32
cache_in    [1303424] fp32  ─┘
```

Fixed shapes everywhere, so it converts like any other vocoder model:

```bash
python3 scripts/export_vocoder_stateful.py --codes-length 64 \
    --output models/vocoder_stateful_64.onnx
python3 scripts/convert_vocoder_rknn.py --model models/vocoder_stateful_64.onnx --stateful \
    --output models/vocoder_stateful_64_fp16.rknn
# board: VOCODER_MODEL=/root/tts-rknn/vocoder/vocoder_stateful_64_fp16.rknn (name must contain
#        'stateful' — that is what selects the streaming path in server and client)
```

**Optional, cuts TTFB:** export a second, small-B step and let it voice the *first* group of every
utterance. A step pays the same fixed 5.2 MB cache round trip whatever B is, so 16 tokens cost
almost the same as 64 — and 16 tokens is 1.28 s of audio the listener gets 7 s earlier:

```bash
python3 scripts/export_vocoder_stateful.py --codes-length 16 \
    --output models/vocoder_stateful_16.onnx
python3 scripts/convert_vocoder_rknn.py --model models/vocoder_stateful_16.onnx --stateful \
    --output models/vocoder_stateful_16_fp16.rknn
# board: VOCODER_MODEL=.../vocoder_stateful_64b_fp16.rknn \
#        VOCODER_MODEL_FIRST=.../vocoder_stateful_16_fp16.rknn
```

`cache_numel` does not depend on B (conv tails + the 71-key window), so the B=16 step's `cache_out`
feeds the B=64 step directly: mixed `[16,64,64,64]` vs a single full decode measures **110.4 dB**,
max diff 1e-05. Without `VOCODER_MODEL_FIRST` the server keeps the single-model behaviour.

Protocol (same socket, no new opcodes): the server **buffers** request tokens and steps only on
full 64-token groups, so a request boundary in the middle of a group never zero-pads (padding
writes silence into the caches — that alone cost 25 dB end to end). A `n_tokens=0` request
**flushes**: it voices the leftover < 64 tokens and returns their audio, then zeroes the state. The
client sends that reset once at the start of an utterance (empty flush) and once after the last
chunk, and concatenates the pieces — no stitch, no overlap, no seam.

The NPU fp16 build forced four non-obvious choices, each measured:

- **RoPE angles are host inputs, cos/sin are in-graph.** Feeding cos/sin as inputs explodes the
  fp16 pipeline (q rms 0.68 → 120). Raw angles `pos*inv_freq` reach ~1e4, where fp16 ulp is 0.5,
  so the host reduces them mod 2π (values ≤ 6.28, ulp 0.004).
- **The attention mask is a graph constant at −100**, on the *relative* grid (`qp-kp`), so it is
  step-index free. −3e4 leaks through the NPU softmax (attention goes uniform); and an absolute
  mask fed as an *input* together with in-graph cos/sin segfaults the board runtime.
- **Cache validity is a scalar input** — and it is a **count**, `cache_valid = min(pos, 71)`, not a
  boolean. A boolean makes every step after the first treat the `Lc - B` never-written window slots
  as real keys: a zero key scores 0 in the softmax, not −inf, so it steals mass from the real keys
  (measured stepwise SNR 48.8 dB at B=64, 10.2 dB at B=16). The graph turns the count into a
  per-slot additive bias with Sub/Clip/Mul. Deriving validity from the cache data instead (an
  Abs/ReduceSum/Clip chain over the 1.3 MB cache) costs 30 dB.
- **KV history joins through constant selection matmuls** (`A@cache_k + P@k`), not Concat: the
  RKNN Concat path corrupts sliding-window attention (9.5 dB vs 56.8 dB).

Measured on the board, same 200-token Russian text, `--streaming`, vs a single 256-token ONNX FP32
run of the *same* codes:

| Vocoder | Global SNR | Per-64-token zones | Generation | Vocoder wait | RTF |
|---|---|---|---|---|---|
| chunked seams (112-slot) | 26.4 dB | 57.5 / 21.1 / 29.7 | 6.5 tok/s | 6.0 s | 2.6x |
| **stateful streaming** | **48.0 dB** | 56.2 / 44.5 / 51.1 / 53.4 | 7.5 tok/s | 0.0 s | **1.9x** |

(the stateful row's SNR is from the pre-fix boolean-validity model, and re-measuring that same
model file against a torch-fp32 full decode of the same 144 recorded codes gives global **25.1 dB**
with zones 55.6 / 21.3 / 23.8 — the decay starts exactly at the second step, which is where phantom
keys first exist. The fixed exports measure **54.3 dB** (zones 55.6 / 53.4 / 51.7, flat) and the
shipped B=16 + B=64 path **53.9 dB** (55.0 / 53.5 / 50.7): the small first group costs 0.4 dB, i.e.
nothing. All three float32 in-process, no int16 round trip, 3 steps of 64 in 10.8 s vs 3 steps
(16+64+64) in 8.1 s.)

The stateful run never drops below 44 dB anywhere in the utterance (the chunked scheme loses 30 dB
past its first seam), and it stops stealing A76 cores from the code predictor, so generation itself
gets faster. The fp32 self-check of the export (stepwise vs full decode, `--check-only`) is 111.4 dB
at B=64 and 103.7 dB at B=16 (onnxruntime re-run of the exported graph: 111.1 / 103.1) — the
residual is the kv-selection matmuls' fp32 accumulation noise, not a semantic error. (Before the
cache-validity count fix the same check read 48.8 dB, which is why its threshold is 40 dB: the
export aborts *before* `torch.onnx.export` and a stale ONNX is left for the converter.)

### Optimization Journey

| Version | Code Predictor | Vocoder | RTF |
|---|---|---|---|
| Baseline (RKLLM talker) | ONNX 1-thread (374ms) | RKNN Q8 | 5.5x |
| + llama.cpp talker | ONNX 1-thread (374ms) | RKNN Q8 | 5.5x |
| + 2-thread ONNX CP | ONNX 2-thread (265ms) | RKNN Q8 | 4.4x |
| + GGML Q4_0 CP | **GGML (55ms)** | RKNN Q8 | 2.2x |
| + ONNX FP32 vocoder | GGML (55ms) | **ONNX FP32** | **2.0x** |
| + ffn_down cast fix | GGML Q8_0 (64ms/15 steps) | ONNX FP32 | 2.6x (3.0x streaming) |
| + RKNN FP16 vocoder on NPU | GGML Q8_0 (64ms/15 steps) | **RKNN FP16 (112-tok chunks)** | 1.8x |
| + stateful streaming vocoder | GGML Q8_0 (64ms/15 steps) | **RKNN FP16 stateful (no chunks)** | **1.9x** |
| + 16-token first group | GGML Q8_0 (64ms/15 steps) | **RKNN FP16 stateful (B=16 + B=64)** | **1.7x** |

The last four rows are measured on this board (Russian text, 4×A76; the cast-fix row is
non-streaming, the other three `--streaming`); the CP times in the rows above are the
upstream-reported ones. The 1.8x and 1.9x rows are different texts (137 / 200 tokens) — the
stateful path's own A/B against the chunked-seam vocoder on one fixed 200-token text is
2.6x → 1.9x, and the 1.7x row is a 443-token text.

Streaming TTFB on the stateful path is **≈9.3 s** with a single B=64 export: the first 64-token
chunk is submitted at +5.7 s and the stateful vocoder step takes 3.6 s (19–23 s with the FP32-CPU
vocoder). With the B=16 first-group model above it is **2.4 s** (measured: first audio at +2.4 s,
1.28 s of it, on a 948-char Russian request, 4×A76, both models FP16 on the NPU). The code
predictor — 63% of per-token CPU time — is the one component the NPU does *not* help: the stateful
step graph
(`scripts/export_code_predictor_stateful.py` + `scripts/convert_code_predictor_rknn.py`, verified
15/15 codebooks and 2.8e-05 step-0 logit diff against an fp64 reference) runs at **23.2 ms/step on
the three NPU cores against 4.3 ms/step on 4×A76** with GGML Q8_0 — 5.4x slower. 157 MB of fp16
weights in 23 ms is 6.8 GB/s, 4x under the board's measured DRAM read ceiling (27.3 GB/s streaming
4 KB blocks over 4×A76, 17.2 GB/s over 4×A55), so the cost is per-op overhead: ~75
tiny S=1 GEMV ops plus 1.3 MB of KV cache across the runtime boundary on every call. The NPU pays
off on real GEMMs (the vocoder's 64-token chunk), not on 16 sequential one-token steps. The `.rkllm`
route is closed by the same arithmetic: the CPU path reads the ~72 MB of Q8_0 weights it actually
touches per step in 4.24 ms = 17 GB/s, i.e. 62% of the measured ceiling, while RKLLM's measured
effective weight-read rate on this board is ~15 GB/s.

Long utterances are no longer truncated. The talker context is `N_CTX` (default 2048; `N_BATCH`
stays separate and small — the ggml compute buffer is sized by n_batch: 1243 MiB at 2048 vs
311 MiB at 512, on the cores the code predictor shares) and the per-request budget is
`min(MAX_TOKENS, N_CTX − prefix − 1)`, so a long text stops at the model's own EOS. Measured on the
board, one 948-char Russian request (784-790 tokens, 62-63 s of audio): non-streaming 9.6 tok/s,
123 s wall, **RTF 2.0x**; `--streaming` 6.9 tok/s, vocoder wait 0.0 s, 117 s wall, **RTF 1.9x**.
Streaming buys TTFB, not throughput: the stateful vocoder's CPU-side work (10.6 MB of cache and
audio marshalled in and out of the RKNN runtime per 64-token step) shares the four A76 cores with
the code predictor and costs it ~28% of its rate. Pinning the vocoder to the A55 cluster
(`VOC_CORES=0-3`) is a measured loss — 5.3 tok/s, RTF 3.9x — that cluster is too slow for the
marshalling (17.2 vs 27.3 GB/s).

The client does not guess which vocoder protocol the server speaks: it probes it with a 0-token
request at the start of every utterance (a stateful server answers it, a chunked-seam server closes
the connection). Inferring it from the client's own `VOCODER_MODEL` environment silently mis-ran the
stateful server through the chunked path — 48 context tokens prepended, then trimmed and
seam-stitched against audio the server never chunked — which dropped 26% of the utterance (795
tokens → 47.04 s of audio instead of 63.6 s) with no error.

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
5b. (Recommended, NPU) Export the **stateful streaming** vocoder and convert it:
   `python3 scripts/export_vocoder_stateful.py --codes-length 64 --output models/vocoder_stateful_64.onnx`
   then `python3 scripts/convert_vocoder_rknn.py --model models/vocoder_stateful_64.onnx --stateful`
   — see [Vocoder: stateful streaming](#vocoder-stateful-streaming-no-chunks).
5c. (Recommended, TTFB) Repeat 5b with `--codes-length 16` and pass the result as
   `VOCODER_MODEL_FIRST` — the first 16 tokens of every utterance then get voiced by the small
   model: first audio 9.3 s → 2.4 s.
6. (Optional) Convert GGML code predictor: see [qwen3-tts.cpp](https://github.com/nickovchinnikov/qwen3-tts.cpp)
7. (Recommended) Shrink + quantize the GGML code-predictor model:
   `python3 scripts/quantize_code_predictor_gguf.py --in <f16.gguf> --out <cp-q8_0.gguf>` and point
   `CP_GGML_MODEL` at it. Keeps only the tensors the code predictor reads and Q8_0-quantizes the
   matmul weights: 1745 MB → 183 MB, code-predictor RSS 1764 → 217 MB on RK3588, speed unchanged.

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
    ├── vocoder_traced_64.onnx           # Vocoder FP32 (93 MB)
    └── vocoder_stateful_64_fp16.rknn    # Vocoder FP16, stateful streaming step (248 MB)
        (+ vocoder_stateful_16_fp16.rknn  # the same step at B=16, voices the first group)
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

CosyVoice3 achieves better RTF because its Flow DiT (MatMul+Attention) runs on AX650N NPU, freeing CPU for LLM and HiFT. Qwen3-TTS has no NPU-friendly component *for quantization* — the vocoder's Snake activation is what kills Q8/INT8 — but the unquantized RKNN FP16 path is lossless, so the vocoder does run on the RKNN NPU (see above).

## Supported Languages

chinese, english, german, russian, french, japanese, korean

## License

Apache-2.0. Model weights: [Qwen3-TTS license](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
