# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Qwen3-TTS (12Hz 0.6B Base) running on a FriendlyElec CM3588 (RK3588, 4×A76 + 4×A55), with an
optional Axera AX650N NPU (M5Stack AI-8850, PCIe M.2). Current headline number: **RTF 2.0x**.
Sibling of the CosyVoice3 project (`nickovchinnikov/cosyvoice3-axera-russian`); several paths and
the AX650N runner header are adapted from it.

`README.md` is the English source of truth; `README_RU.md` and `README_ZH.md` are translations of
it — update all three when user-facing behaviour changes.

## Host reality

Everything under `dual_npu/` and `cpp/` targets the **board** (aarch64, `/root/...` paths hardcoded,
`taskset -c 4-7`, `llama.cpp` built at `/root/llama.cpp`, ONNX Runtime at
`/usr/local/lib/python3.11/dist-packages/onnxruntime/capi`). None of it runs on this x86_64 WSL2
box. Only `scripts/` (the converters) are host-side, and they need `torch`, `safetensors`,
`transformers` and the `qwen-tts` package plus a downloaded `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

No tests, no CI, no package manifest, no linter config. Verification = run the pipeline on the
board and listen to `output.wav` / read the RTF line it prints.

## Commands

```bash
# Board, one-shot (starts all three servers, runs client, tears them down on EXIT)
bash dual_npu/launch_qwen3_tts.sh "Привет, как дела?"

# Board, persistent servers (Ctrl+C stops)
bash dual_npu/launch_qwen3_tts.sh --daemon
python3 dual_npu/tts_client.py "Text" --language russian            # one request
python3 dual_npu/tts_client.py "Long text" --language russian --streaming

# Build the llama.cpp ctypes shim (required by the talker server)
cd dual_npu && gcc -shared -fPIC -O2 -o llama_wrapper.so llama_wrapper.c \
    -I/root/llama.cpp/include -I/root/llama.cpp/ggml/include -L/root/llama.cpp/build/bin \
    -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,/root/llama.cpp/build/bin

# Build the C++ ONNX code predictor (downloads ORT headers, symlinks the pip .so, builds)
bash dual_npu/code_predictor_cpp/setup_and_build.sh

# Host, model preparation (env: QWEN3_TTS_MODEL, OUTPUT_DIR)
python3 scripts/extract_talker_as_qwen3.py && python3 scripts/convert_talker_gguf.py
python3 scripts/extract_embeddings.py
python3 scripts/export_code_predictor_weights.py && python3 scripts/export_code_predictor_onnx.py
python3 scripts/export_vocoder_traced.py --codes_length 64 --simplify --remove_isnan
```

Servers are standalone: each takes `--socket` and can be started by hand; the launch script's
`BASE_DIR` (default `/root/tts-rknn`), `GGUF_MODEL`, `CP_DIR`, `VOCODER_MODEL`, `EMBEDDINGS_DIR`,
`CP_THREADS`, `GGUF_THREADS`, `TEMPERATURE`, `TOP_K`, `MAX_TOKENS`, `LANGUAGE` are all env-overridable.

## Architecture

Three independent Unix-socket servers plus one orchestrating client. No shared memory, no Python
package — each server is a script with its own `serve()` accept loop.

```
tts_client.py ──unix socket──> llamacpp_talker_server.py   llama.cpp GGUF Q4_K_M, A76, embedding mode
              ──unix socket──> code_pred_server (GGML | C++ ONNX | Python ONNX)
              ──unix socket──> vocoder_server.py           ONNX FP32 CPU (RKNN NPU path also present)
```

The **feedback loop** is the thing that makes this Qwen3-TTS and not a plain LLM: the talker emits
one `code_0` + hidden state per step, then *blocks* waiting for the client to send back a 1024-float
feedback embedding. The client computes it as `codec_embedding[code_0] + Σᵢ codec_emb_i[codes[i]] +
tts_pad_embed` and writes it to the talker, which runs one forward pass with it as the input
embedding. So the client, not the talker, owns the 16-codebook sum — the CP backends each keep
their own copy of the per-group embeddings and the client reloads them from
`code_predictor_weights.npz` (`codec_emb_0..14`).

The talker runs llama.cpp in **embedding mode** (`llama_wrapper.c` → `llama_cpp_bindings.py`):
custom float32 embeddings go in, last-layer hidden states come out; there are no logits. `code_0`
is sampled in Python as `hidden @ codec_head.T` masked to the 2048 audio tokens plus EOS 2150, with
a repetition penalty and an adaptive EOS boost keyed to `len(tokens) / (3 × n_text_tokens)`.

Text side is a **dual-stream prefix** built in `_build_prefix()`: role tokens, then
`tts_pad + codec[nothink, think_bos, think_eos]`, then `tts_bos + codec_pad`, then each text token
embedded through the 2048→2048→1024 text projection MLP and summed with `codec_pad`, then
`tts_pad + codec_bos`. Getting this ordering right is what commit "Fix dual-stream prefix" bought.
The prefix is the KV-cache key (md5 of the prefix bytes → `/tmp/qwen3_kv_<hash>.bin` +
`qwen3_hidden_<hash>.npy`), so a prefix change silently invalidates every warm cache.

Code predictor = 5-layer, 1024-dim, GQA 16/8 transformer run **15 sequential steps per token**
(step 0 prefills `[hidden, code_0_embed]`, steps 1-14 decode with KV cache). This is ~86% of
per-token time in the ONNX path and is the reason three CP backends exist.

### Fixed-shape contract

`VOC_CHUNK_SIZE = 64` (new tokens per chunk) + `VOC_CTX = 48` (real context tokens prepended per
chunk) must sum to the vocoder model's fixed input size — `VOC_SLOTS = 112` for
`vocoder_traced_112_fp16.rknn` (`--codes_length 112` export). `SAMPLES_PER_TOKEN = 1920` =
24000 Hz / 12.5 Hz. The RKNN branch of `vocoder_server.py` infers the slot count from the
*filename* (`_(\d+)` in basename). Multi-chunk audio is seam-stitched, not crossfaded: prepend
real context, move each seam into the longest quiet run (≥150 ms) of the chunk, fade-drop there,
20 ms equal-power micro-blend on loud seams, and 2-token warmup-truncate overlap (the decoder is
causal — zero-pad chunk starts are a *different waveform*, corr 0.445 / SNR 0.3 dB, so any
zero-pad overlap crossfade = "double voice"). Streaming mode submits each chunk to the vocoder on
a background thread so it overlaps generation.

### Stateful streaming contract (the seam-free path, now the default on the NPU)

`scripts/export_vocoder_stateful.py` exports the decoder as one **streaming step with explicit
state**: 4 inputs `audio_codes [1,16,B] int64` / `freqs_in [B,32] fp32` / `cache_valid [1] fp32` /
`cache_in [1303424] fp32`, 2 outputs `audio_values` / `cache_out`. The flat cache is every causal
conv's tail plus the pre-transformer's sliding-window KV (8 layers × 2 × 16 heads × 71 × 64), in
the order of `StatefulDecoder.specs` — change `B` (`--codes-length`) and you must reconvert the
RKNN model, whose input sizes `convert_vocoder_rknn.py --stateful` hardcodes. `cache_numel` does
**not** depend on B, so a B=16 step's `cache_out` feeds a B=64 step directly (measured: mixed
`[16,64,64,64]` vs a single full decode = 110.4 dB, max diff 1e-05) — that is the whole basis of
the two-tier first chunk. `vocoder_server.py` and `tts_client.py` both switch on the substring
`stateful` in the model filename (`VOCODER_MODEL`), and the server takes `B` from the same
`_(\d+)` filename regex.

Protocol: the server **buffers** request tokens and steps only on full `B`-token groups — a request
boundary in the middle of a group must never be zero-padded, padding writes silence into the caches
(measured: 25 dB end to end). `n_tokens=0` is a **flush**: it voices the leftover tail, *returns its
audio*, then zeroes `cache`/`pos`/`buf`. The client sends a flush at utterance start (empty) and
again after the vocoder threads join, and concatenates stateful audio without `stitch()`. A client
that exits without the final flush leaves tokens in `buf`; the next utterance's start flush voices
them, so state never leaks across utterances.

The four NPU-fp16-forced export choices are load-bearing and each is measured (see the README
section): RoPE angles as host inputs reduced mod 2π with cos/sin computed in-graph (cos/sin as
inputs explodes the pipeline; raw angles hit fp16 ulp 0.5 at pos ~1e3), the sliding-window mask as
a graph constant at −100 on the relative grid (−3e4 leaks through the NPU softmax; an absolute mask
as an *input* plus in-graph cos/sin SEGFAULTs the board runtime), cache validity as a scalar input
carrying the **count** of real slots `min(pos, 71)` — a boolean leaks the never-written slots into
the softmax (a zero key scores 0, not −inf) and costs 62 dB, while deriving the count from the cache
data (a data-dependent Abs/ReduceSum/Clip chain) costs 30 dB — and KV history joined by constant
selection matmuls instead of Concat (the RKNN Concat path corrupts sliding-window attention:
9.5 dB vs 56.8).

### Why the vocoder is FP32 on CPU (superseded: FP16 on NPU is lossless)

Snake activation (`x + sin²(αx)/α`) breaks every *quantized* path: RKNN Q8 is 1.75x faster but
audibly noisy (SNR 9.5 dB random-cal / 13.4 dB real-cal), ONNX INT8 is slower *and* worse
(SNR 4.2 dB), AX650N can't do Conv1D dilation=9, MNN GPU hits a SineGen broadcast error. RKNN
**FP16 (w16a16) is not quantized** and measures lossless (SNR 56.8 dB) at 1.34x the CPU speed —
see "Fixed locally". Don't re-quantize the vocoder to Q8/INT8 without re-measuring SNR.

## Code predictor backends

`launch_qwen3_tts.sh` auto-selects, in order, and each is a drop-in on the same socket protocol:

| Backend | Selected when | Speed |
|---|---|---|
| GGML Q4_0 | `USE_GGML_CP=1` (default) and `/root/qwen3-tts.cpp/build/code_pred_server` + its model exist | ~55ms/step |
| C++ ONNX RT | `USE_CPP_CP=1` (default) and `code_predictor_cpp/build/code_predictor_server` exists | ~240ms/step |
| Python ONNX RT | otherwise | ~248ms/step |

`dual_npu/code_predictor_ggml/code_pred_server.cpp` is a **copy kept for reference** — the binary
that actually runs is built out-of-tree inside the external `qwen3-tts.cpp` repo against
`tts_transformer.h`. Editing it here changes nothing on the board.

## Wire protocol (all little-endian, raw bytes, no framing magic)

- Talker: client sends `[u32 len][JSON {"text","language"}]`; server streams per token
  `[i32 code_0][float32[1024] hidden]`, client replies `[float32[1024] feedback]`, ends with
  `[i32 -1]` done / `[i32 -2]` error.
- Code predictor: `[float32[1024] hidden][i32 code_0]` → `[i32[15] codes]`. One fresh connection
  per token; no length header (the Python server's own docstring claims a 4-byte header — the code
  and all three clients read raw 4096 bytes; trust the code).
- Vocoder: `[i32 n_tokens][int64[n×16]]` → `[i32 n_samples][int16[n_samples]]`.

## Traps

- `TTS_PAD_TOKEN_ID = 151671` (and the other TTS/codec special IDs) is duplicated in
  `tts_client.py` and `llamacpp_talker_server.py`; the feedback embedding is computed in the client
  and consumed in the talker, so the two must agree.
- The CP uses the **talker's** `codec_embedding.npy` for the `code_0` lookup, not its own group-0
  embedding — matches the official model; changing it degrades quality.
- `OPENBLAS_NUM_THREADS=1` on the Python servers is load-bearing: numpy otherwise grabs the A76
  cores that llama.cpp and the CP are pinned to.
- `cpp/src/runner/LLM_Qwen3TTS.hpp` is an experimental AX650N talker runner — a lone header with no
  build system in-tree, including `ax_model_runner/`, `ax_cmm_utils.hpp`, `axcl_manager.h` and
  `bfloat16.hpp` from the external Axera SDK/CosyVoice3 tree. It cannot be compiled here; treat it
  as design-in-progress, not as live code.
- All model artifacts are gitignored (`*.gguf *.onnx *.rknn *.npy *.npz *.axmodel *.so`); they live
  in `/root/tts-rknn/` on the board and total ~2.5 GB at runtime.
- `HF_HUB_OFFLINE=1` is exported by the launch script and the tokenizer loads with
  `local_files_only=True` — the Qwen3-TTS tokenizer must already be in the HF cache on the board.

## Verified on a real RK3588 (deployed + measured, 2026-10-09)

74 Russian chars → 6.3-6.8 s audio, `taskset -c 4-7`, GGML CP backend:

| Mode | tokens | generation | wall | audio | RTF | CPU-sec/audio |
|---|---|---|---|---|---|---|
| non-streaming | 85 | 17.8 s (4.8 tok/s) | 26.6 s | 6.80 s | 3.9x | 14.7 |
| `--streaming` | 79 | 22.2 s (3.6 tok/s) | 26.6 s | 6.32 s | 4.2x | 16.3 |

Per-server CPU/RSS for one run: CP 62.9 s / 1764 MB, vocoder 30.0 s / 839 MB, talker 7.3 s /
2154 MB, client 2.5 s / 1253 MB → ~6.0 GB resident. First audio chunk ≈ 19-23 s. Streaming does
not help RTF: all three servers are pinned to the same 4 A76 cores, so the vocoder steals them
from the CP (4.8 → 3.6 tok/s). The headline "RTF 2.0x" presumes ~30 tok/s generation; the real
ceiling is the code predictor at 15 sequential steps/token = 209 ms/token (63% of all CPU).

What does not work as shipped:

- The default CP backend's binary is built by nothing in-tree (see above); the Python ONNX backend
  is dead on arrival — the exporter emits `code_predictor_core.onnx`
  (`hidden_states`,`position_ids`, no KV cache) while `code_predictor_server.py` loads
  `code_predictor_decode_step.onnx` (`hidden`,`position`,`past_k_i`,`past_v_i`) →
  `NO_SUCHFILE` at startup. Only the GGML rung of the three-rung fallback chain works.
- No producer exists for `qwen3-tts-0.6b-q4_0.gguf`: `llama-quantize` says
  `failed to quantize: unknown model architecture: 'qwen3-tts'` and `qwen3-tts.cpp` ships no
  quantizer. f16 GGUF works (1.83 GB file, 1.76 GB RSS).
- `export_code_predictor_onnx.py` at `opset_version=17` dies with
  `Assertion ... No Adapter To Version $17 for Pad` on current torch/onnx; opset 18 exports clean
  (max diff 1.2e-5). The same assertion inside onnxsim breaks `export_vocoder_traced.py --simplify`.
- `scripts/setup_pipeline_models.py` hardcodes `GGUF_PY_PATH = Path("/root/llama.cpp/gguf-py")`.
- The launch script calls bare `python3` — with a venv you must prepend it to `PATH`.

## Fixed locally (this checkout)

- `scripts/quantize_code_predictor_gguf.py` (new) — the missing producer: keeps only the tensors
  the CP path reads (`code_pred.*` + `talker.codec_embd`) and Q8_0-quantizes the matmul weights.
  1745 MB f16 → 183 MB, CP server RSS **1764 → 217 MB**, pipeline RSS 6.0 → 4.5 GB, speed
  unchanged. Embedding tables stay f16 on purpose: `lookup_single_embedding_row` slices rows by
  `nb[1]` and ggml CPU `get_rows` has no quantized support.
  Measured: Q8_0 does **not** speed the CP up (184 → ~200 ms per 15-step request) — the CP is
  graph-build/alloc/dispatch-bound at ~10 GB/s effective, not weight-bandwidth-bound.
- `code_pred_server --threads N` (with `TTSTransformer::set_n_threads` in `qwen3-tts.cpp`) and
  `launch_qwen3_tts.sh` env knobs `PYTHON`, `CP_GGML_THREADS`, `TALKER_CORES`/`CP_CORES`/`VOC_CORES`.
  Measured: `--threads 4` vs ggml's default = no difference; pinning the FP32 vocoder to the A55
  cluster costs **4.2x** (8.8 → 36.9 s per 80 tokens), so the core knobs default to `4-7`
  (they pay off only with the RKNN vocoder, or on a board with more big cores than servers).
- `code_predictor_server.py` now auto-detects the graph: `code_predictor_decode_step.onnx` →
  KV-cache path (unchanged), `code_predictor_core.onnx` → prefix-recompute path (O(n²) over 16
  positions, no KV cache). Verified end-to-end on the board.
- `export_code_predictor_onnx.py`: `opset_version` 17 → 18 (same fix applied in
  `qwen3-tts-rknn-russian`, the files are byte-identical).
- `qwen3-tts.cpp/scripts/convert_{tts,tokenizer}_to_gguf.py`: `GGUF_PY_PATH` is now
  `os.environ.get("GGUF_PY_PATH", "/root/llama.cpp/gguf-py/gguf")` — the old default pointed at
  the *repo* dir, which is not importable (`gguf-py/gguf` is the package).
- **`qwen3-tts.cpp/src/tts_transformer.cpp`: removed the five `ggml_cast(ffn_down, F32)` nodes.**
  The biggest single win in the whole pipeline, and the reason the README's "RTF 2.0x" was
  unreachable: the cast node dequantizes the whole 1024×3072 `ffn_down` matrix (12.6 MB) on *every*
  graph compute — 5 layers × 15 steps = 75 times per generated token in the CP — and then runs
  F32×F32 `mul_mat`, which on aarch64 has no quantized GEMV kernel. `ggml_mul_mat` takes F16/Q8_0
  `src0` directly (vec_dot path). Measured on the board, same Q8 CP model, `--threads 4`:
  **199 → 64 ms per 15-step request (3.1x)**, generation **4.5 → 11.5 tok/s**, non-streaming
  **RTF 4.4 → 2.6x** (5.68 s audio in 15.0 s wall), streaming 16.6 s / 3.0x, first vocoder chunk
  submitted at 6.6 s instead of ~19-23 s. Numerics: recorded 70 real (hidden, code_0) requests
  through a socket proxy, replayed greedy against all three backends — patched-vs-baseline codebook
  agreement 98/97/97/95/92/87/82/81/74/72/70/67/62/57/52 %, baseline-vs-Python-FP32-ONNX reference
  98/95/92/87/82/80/75/75/71/67/68/61/57/47/44 %. The patch is *closer* to the baseline than the
  baseline is to the FP32 reference; the decay along codebooks is the usual autoregressive
  amplification, not a regression. After this the FP32 ONNX **vocoder is the bottleneck**
  (8.7-9.0 s CPU per 70 tokens ≈ RTF 1.6x on its own), not the CP.
- `code_predictor_server.py`: `_sample()` honours `--temperature 0` as greedy argmax (it divided by
  `max(temperature, 1e-6)` → inf/nan), which makes the Python backend comparable to the GGML ones.
- `scripts/quantize_code_predictor_gguf.py`: working Q4_0 packing (ggml packs nibbles as
  `x[i] | x[i+16] << 4` — split halves, not interleaved pairs; the first Q4_0 build had this wrong
  and scored 0% codebook match) and a `--lmhead <type>` override. Measured verdict: **Q4_0 is a dead
  end for the CP** — greedy replay of 70 recorded requests vs the Q8_0 baseline gives
  78/51/41/24/12/10/10/1/... % codebook agreement (Q8_0-vs-baseline: 98/97/97/95/92/87/82/81/...)
  and only 64 → 49.5 ms per 15-step request (1.29x, not the 1.78x the byte ratio promises — the
  f16 embedding tables and KV traffic don't shrink). Keeping lm_head at Q8_0 (`--lmhead Q8_0`)
  changes nothing (81/52/45/22) — the transformer layers are what Q4_0 breaks. Q8_0 stays the sweet
  spot.
- **Vocoder on the NPU (RKNN FP16) — the "Snake activation makes quantization impossible" claim was
  never tested for FP16.** Measured on the board, 64-token chunk, 10 runs: ONNX FP32 (4×A76)
  4407 ms; **RKNN FP16 (w16a16, NPU cores 0_1_2) 3287 ms, SNR 56.8 dB / corr 1.0000 vs the FP32
  reference — numerically lossless**; RKNN Q8 (w8a8 per-channel) 2572 ms, SNR 13.4 dB with real
  calibration data vs 9.5 dB with the random calibration the old Q8 model used. End-to-end, same
  137-token Russian text: streaming **RTF 2.5 → 1.8x** (generation 5.8 → 8.9 tok/s — the vocoder no
  longer steals A76 cores from the CP), non-streaming 2.6 → 2.0x. Deploy:
  `python3 scripts/convert_vocoder_rknn.py --model vocoder_traced_64.onnx` (no `--quantize`), scp the
  242 MB `.rknn` to the board, start the daemon with `VOCODER_MODEL=.../vocoder_traced_64_fp16.rknn`
  and `PYTHONPATH=dual_npu/rknnlite_shim` — the board's Python is 3.14 and no
  `rknn_toolkit_lite2` wheel exists for it, so `dual_npu/rknnlite_shim/` is a ctypes shim over
  `librknnrt.so` implementing the `RKNNLite` surface the server uses. `convert_vocoder_rknn.py
  --quantize --dataset <file>` now accepts real recorded vocoder requests for calibration
  (`dual_npu/vocproxy.py`-style recording: `[i32 n][int64 n*16]` per request, chunked to 64).
- **Stateful streaming vocoder (seam-free) — the chunk-seam relaxation is gone, not tuned.**
  `scripts/export_vocoder_stateful.py` + `convert_vocoder_rknn.py --stateful` + the buffer/flush
  protocol in `vocoder_server.py`/`tts_client.py` (contract above) replace chunking entirely.
  Measured on the board, same 200-token Russian text, `--streaming`, vs a single 256-token ONNX
  FP32 run of the same codes: chunked seams (112-slot) global SNR **26.4 dB** with zones
  57.5/21.1/29.7, generation 6.5 tok/s, vocoder wait 6.0 s, RTF 2.6x; stateful **48.0 dB** with
  zones 56.2/44.5/51.1/53.4, generation 7.5 tok/s, vocoder wait 0.0 s, **RTF 1.9x**. The stateful
  run never drops below 44 dB anywhere, i.e. no seam-shaped relaxation at all, and the vocoder no
  longer competes with the CP for the A76 cores. The export's own fp32 self-check (stepwise vs
  full decode) is 111.4 dB at B=64 / 103.7 dB at B=16 (onnxruntime re-run: 111.1 / 103.1) — the
  residual is the kv-selection matmuls' fp32 accumulation noise; the export's check threshold is
  40 dB, do not raise it back to 80 (it aborts *before* `torch.onnx.export`, which then leaves a
  stale ONNX for the converter to load). Before the cache-validity-count fix the same check read
  48.8 dB, so the board numbers in this bullet are from a model 62 dB below its own tolerance.
  Re-measured on the board after the fix, same 144 recorded codes, float32 in-process, reference =
  torch-fp32 full decode: pre-fix `vocoder_stateful_64_fp16.rknn` **25.1 dB** (zones
  55.6/21.3/23.8 — decay starts at the second step, where phantom keys first exist), fixed
  `vocoder_stateful_64b_fp16.rknn` **54.3 dB** (55.6/53.4/51.7, flat), shipped
  B=16+B=64 **53.9 dB** (55.0/53.5/50.7) — the small first group costs 0.4 dB. Scripts
  `../probes/snr_board.py` + `../probes/rec_codes.py` (records real codes through the live
  talker+CP) and `../probes/ref_decode.py` (the fp32 reference), dumps in `../samples-board/`.
- **Phantom KV slots: `cache_valid` must be a COUNT, not a flag** (see the contract above). The
  deployed NPU vocoder scored 48.8 dB (B=64) / 10.2 dB (B=16) stepwise because a boolean flag told
  every step after the first that all 71 window slots held real keys. Fixed in the exporter
  (per-slot bias from `Lc - n_valid - slot_idx`, `-100.0 * phantom` — the inverted sign scores
  5.4 dB, real bug caught by the self-check) and in `vocoder_server.py` (`min(pos, kv_len)`).
- **Two-tier first chunk: TTFB 9.3 → 2.4 s.** A stateful step pays the same fixed 5.2 MB cache
  round trip at any B, so the client sends the first 16 tokens as soon as they exist
  (`VOC_FIRST_CHUNK`, stateful only — a chunked-seam server zero-pads a short request) and the
  server voices that group on a second RKNN context (`--model_first` / `VOCODER_MODEL_FIRST`,
  `vocoder_stateful_16_fp16.rknn`). Measured on the board, 948-char Russian text, `--streaming`,
  4×A76: first audio **+2.4 s** (1.28 s of it), 443 tokens → 35.44 s audio (complete),
  7.6 tok/s, RTF 1.7x. The client prints that line itself (`<- First audio: ... +X.X s`).
  Click audit of the shipped path (all negative, `../probes/click{,2,3}.py`,
  `../probes/phase_test.py`, `../probes/pad_test.py`): `mix16` and `new64b` have identical
  per-token diff envelopes
  (the B=16→B=64 handoff adds no localized artifact); the flush path's zero padding costs
  **0.0 dB** (tails of 1/14/44 tokens vs no padding, same zones); no clipping (max |a| 0.95-0.97,
  zero samples >0.9999); zero sample-domain discontinuities above the click threshold in the
  shipped files, same signature as the single-B=64 control; and the 12.5 Hz phase-locked texture
  (peak 1.70x median at phase 1073) is present in the fp32 full decode too, i.e. it is the
  decoder's own lattice, not streaming. If a listener reports clicks, suspect the **pre-fix**
  samples (`../samples-board/ab_stream_*.wav`, the 25 dB model), not this path.
- **Code predictor on the NPU — built, measured, rejected (do not rebuild the backend).**
  `scripts/export_code_predictor_stateful.py` (one stateful decode step, KV cache in graph I/O:
  14 inputs / 11 outputs, ONE graph for all 16 positions because the per-token sequence is exactly
  16 long, so the cache length is a compile-time constant; the step index enters as data —
  `freqs_in` RoPE angles mod 2π, `sel` one-hot cache write index, `mask` additive −100) plus
  `scripts/convert_code_predictor_rknn.py` (FP16, 152.6 MB) do work: the export scores 15/15
  codebooks on every input and 2.8e-05 max step-0 logit diff against an independent numpy-fp64
  re-run of the same transformer. The board probe (`/root/tts-rknn/cp/cp_npu_probe.py`) says
  **23.2 ms/step on NPU cores 0_1_2 vs 4.3 ms/step for the CPU GGML Q8_0 CP — 5.4x slower**, so the
  CP stays on the CPU. Not bandwidth: 157 MB fp16 / 23 ms = 6.8 GB/s, 4x under the board's measured
  27.3 GB/s ceiling; the step is ~75 tiny S=1 GEMV ops plus 1.3 MB of KV across the runtime boundary
  per call, and the NPU matrix units only pay off on real GEMMs (the vocoder's 64-token chunk).
  Compiler flags are not the lever either: `qwen3-tts.cpp` builds Release with `-march=native`
  (gcc 15.2 aarch64) and the binary already contains `udot/umlal/fcvtl`.
  The `.rkllm` variant (whole 5-layer predictor as an RKLLM model, 15 steps on the NPU) is closed by
  the same inequality, no build needed: the CPU path reads the ~72 MB of Q8_0 weights it touches per
  step in 4.24 ms = 17 GB/s, 62% of the measured ceiling, and RKLLM's measured
  effective weight-read rate on this board is ~15 GB/s (0.5B W8A8 at 16 tok/s, per
  `cosyvoice3-rknn-russian/docs/BENCHMARKS.md`), so a 78 MB W8A8 CP would be ~5.2 ms/step — slower
  before quality is even considered, and W8A8 on a token-generating transformer is that repo's
  documented dead end (truncation/repeats/language mixing).
  Export traps, all measured: `torch.onnx.export(..., dynamo=False)` is required (the
  torch.export decomposition set emits Reshape/Unsqueeze/Slice patterns that kill
  rknn-toolkit2's `graph_optimizer.fuse_ops` with `KeyError: 'unsqueeze_8_rs'`), the graph must be
  matmul-only (no Concat/Expand/Unsqueeze), and the GQA selection matmul emits `[B,L,H,D]` — with
  `L_KV == N_HEADS == 16` a missing `.transpose(1,2)` is shape-invisible and numerically fatal
  (1-pos hidden diff 15.6, 0/15 codebooks).
- Streaming TTFB on the stateful path, measured: client start → first 64-token chunk submitted
  +5.7 s, stateful vocoder step 3.6 s → **first audio ≈ 9.3 s** (was 19-23 s on the FP32-CPU
  vocoder, 6.6 s to chunk submission after the cast fix). Cut to 2.4 s by the two-tier first chunk
  above.
- **Long utterances: `n_ctx` 512 → `N_CTX` 2048, `n_batch` decoupled, per-request token budget.**
  `n_ctx=512` + `MAX_TOKENS=200` truncated a long text mid-phrase (no forced EOS — the talker just
  sends `SENTINEL_DONE`), while the model's adaptive EOS wants ~3× the text-token count (forced at
  6×). Keep `N_BATCH` (512) separate from `N_CTX`: the wrapper feeds it to `params.n_batch/n_ubatch`,
  which sizes the ggml compute buffer — `sched_reserve` reports 1243 MiB at n_batch=2048 vs 311 MiB
  at 512, resident on the four A76 cores the CP shares. Measured on the board with the protocol
  probe below in place, one 948-char Russian request (784-790 tokens, 62-63 s audio): non-streaming
  9.6 tok/s / 123 s wall / RTF 2.0x, `--streaming` 6.9 tok/s / vocoder wait 0.0 s / 117 s /
  **RTF 1.9x**. Streaming buys TTFB, not throughput.
- **Per-token budget, measured with `-DQWEN3_TTS_TIMING=ON` on the board:** talker 24.9 ms
  (llama.cpp decode + Python sampling, grows 29 → 37.5 ms as the KV cache fills), CP **63.2 ms**
  (flat with length), client 0.25 ms → 88 ms/token = 11.3 tok/s with the vocoder idle. Streaming
  costs 28% of that (6.9 tok/s) because the stateful vocoder's CPU-side marshalling shares the
  A76s; `VOC_CORES=0-3` (A55) is a measured loss, 5.3 tok/s / RTF 3.9x. Board DRAM read ceiling
  measured with a NEON stream probe: **27.3 GB/s on 4×A76, 17.2 GB/s on 4×A55**, 16.8 on one A76.
  The CP reads ~72 MB of Q8_0 weights per step in 4.24 ms = 17 GB/s = 62% of that ceiling, and its
  own phase split is `build 1.3 + alloc 1.7 + compute 63.6 + data 1.0` ms — 94% compute, so graph
  reuse across the 15 steps is worth ≤6% and quantization below Q8_0 is the only byte-level lever
  (Q6_K ≈ 0.80x bytes; Q4_0 measured 1.29x speed at unusable codebook agreement).
- **The client probes the vocoder protocol; it must not guess it.** `tts_client.py` sends a
  0-token request at the start of each utterance: a stateful server answers it (and flushes), a
  chunked-seam server closes the connection. `VOC_STATEFUL = 'stateful' in
  os.environ['VOCODER_MODEL']` — the old inference — silently ran a stateful server through the
  chunked path whenever the client was launched without that env var (daemon mode + a bare client
  shell), prepending 48 context tokens and seam-stitching audio the server never chunked:
  **795 tokens → 47.04 s of audio instead of 63.6 s (26% of the utterance gone)**, RTF 2.9x vs 1.9x,
  no error. Any client-side behaviour that depends on which vocoder is running belongs on the probe.
- **`CP_GGML_MODEL`'s old default pointed at a filename nothing produces**
  (`qwen3-tts-0.6b-q4_0.gguf`; the quantizer writes `qwen3-tts-0.6b-cp-q8_0.gguf`), and the CP
  backend probe is a bare `[ -f ]`, so the whole pipeline silently ran the Python ONNX CP:
  **390 ms vs 63 ms per 15-step request (6.1x)**, 6.6 → 2.2 tok/s end to end, with no log line
  beyond the startup banner. If generation is ~2 tok/s, check the banner's "Code Predictor:" line
  first. `cleanup()` in `launch_qwen3_tts.sh` also now escalates SIGTERM → SIGKILL after 3 s:
  the GGML CP blocks in `accept()` and its handler only flips a flag, so idle instances survived
  teardown and piled up.
