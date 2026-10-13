#!/usr/bin/env python3
"""
Produce the code-predictor GGUF that `dual_npu/launch_qwen3_tts.sh` wants.

The GGML code predictor (`code_pred_server`) runs 15 sequential forward passes per
generated token and is memory-bandwidth bound on the A76 cores: it reads the 5 CP
transformer layers + one lm_head on every step. The f16 GGUF that
`qwen3-tts.cpp/scripts/convert_hf_to_gguf.py` produces carries the whole model
(talker text_embd alone is 594 MB), so the CP server pages in 1.75 GB of weights it
never touches, and `llama-quantize` cannot help (`unknown model architecture:
'qwen3-tts'`). This script is that missing producer:

  1. keep only the tensors the CP path reads (`code_pred.*` + `talker.codec_embd`),
  2. Q8_0-quantize the matmul weights (attention/FFN/lm_head),
  3. keep the embedding tables f16 — the loader slices embedding rows with
     `ggml_backend_tensor_get` on `nb[1]` and ggml's CPU `get_rows` has no quantized
     support, so quantizing them is not a size win, it is a crash,
  4. copy every KV metadata entry verbatim.

Usage:
  python3 scripts/quantize_code_predictor_gguf.py \
      --in ../qwen3-tts.cpp/models/qwen3-tts-0.6b-f16.gguf \
      --out /root/tts-rknn/code_predictor/qwen3-tts-0.6b-cp-q8_0.gguf
"""

import argparse
import os
import struct
import sys

import numpy as np

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType

QK8_0 = 32
Q8_0_BLOCK_BYTES = 2 + QK8_0  # fp16 scale + 32 int8
Q4_0_BLOCK_BYTES = 2 + QK8_0 // 2  # fp16 scale + 16 bytes of packed nibbles

# Read by the code predictor path: 5 transformer layers, 15 per-codebook embeddings
# and 15 lm_heads, plus the talker's codec embedding table (code_0 lookup).
CP_PREFIX = "code_pred."
CP_EXTRA = ("talker.codec_embd.weight",)
# get_rows / row-slice path, must stay dequantized.
ROW_SLICED = "codec_embd"


def quantize_q8_0(x: np.ndarray) -> np.ndarray:
    """Quantize a flat f16/f32 weight to ggml Q8_0 blocks (32 values along ne[0])."""
    flat = x.astype(np.float32).reshape(-1)
    if flat.size % QK8_0 != 0:
        raise ValueError(f"ne[0] not divisible by {QK8_0}: {flat.size}")
    blocks = flat.reshape(-1, QK8_0)
    amax = np.abs(blocks).max(axis=1)
    scale = (amax / 127.0).astype(np.float16)
    # Divide by the fp16-rounded scale: that is the value the dequant path multiplies by.
    q = np.clip(np.rint(blocks / scale.astype(np.float32)[:, None]), -127, 127)
    out = np.empty((blocks.shape[0], Q8_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, :2] = scale.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.astype(np.int8).view(np.uint8)
    return out.reshape(-1)


def quantize_q4_0(x: np.ndarray) -> np.ndarray:
    """Quantize a flat f16/f32 weight to ggml Q4_0 blocks (32 values, nibbles low-first)."""
    flat = x.astype(np.float32).reshape(-1)
    if flat.size % QK8_0 != 0:
        raise ValueError(f"ne[0] not divisible by {QK8_0}: {flat.size}")
    blocks = flat.reshape(-1, QK8_0)
    amax = np.abs(blocks).max(axis=1)
    scale = (amax / 8.0).astype(np.float16)
    q = np.clip(np.rint(blocks / scale.astype(np.float32)[:, None]) + 8.0, 0, 15)
    q = q.astype(np.uint8)
    out = np.empty((blocks.shape[0], Q4_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, :2] = scale.view(np.uint8).reshape(-1, 2)
    # ggml packs x[i] low / x[i + 16] high — split halves, not interleaved pairs.
    out[:, 2:] = q[:, :QK8_0 // 2] | (q[:, QK8_0 // 2:] << 4)
    return out.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--dtype", default="Q8_0", choices=["Q8_0", "Q4_0", "F16"],
                    help="target type for the matmul weights (Q4_0: 4x smaller, "
                         "audible CP degradation is likely — re-measure SNR/codes)")
    ap.add_argument("--lmhead", default=None, choices=["Q8_0", "Q4_0", "F16"],
                    help="type for the code-predictor lm_head heads (default: --dtype). "
                         "lm_head feeds the argmax over 2048 codes per step and is the "
                         "most quantization-sensitive part of the CP.")
    args = ap.parse_args()

    reader = GGUFReader(args.src)
    arch = reader.fields["general.architecture"].contents
    writer = GGUFWriter(args.dst, arch)

    n_kv = 0
    qtype = int(GGMLQuantizationType[args.dtype])
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue  # reader-synthesized header fields, not real KV entries
        types = field.types
        if key == "general.file_type":
            # record the type the matmul weights actually got
            writer.add_key_value(key, qtype, GGUFValueType.UINT32)
        else:
            writer.add_key_value(key, field.contents(), types[0],
                                 types[1] if len(types) > 1 else None)
        n_kv += 1

    kept = quantized = 0
    kept_bytes = total_bytes = 0
    for t in reader.tensors:
        total_bytes += t.data.nbytes
        if not (t.name.startswith(CP_PREFIX) or t.name in CP_EXTRA):
            continue
        shape = tuple(int(v) for v in t.shape)   # ggml order: shape[0] == ne[0] (contiguous)
        x = t.data
        quantizable = (
            args.dtype != "F16"
            and len(shape) == 2
            and shape[0] % QK8_0 == 0
            and ROW_SLICED not in t.name
        )
        if quantizable:
            # gguf-py wants numpy order (rows, bytes-per-row) for quantized payloads
            dt = args.lmhead if (args.lmhead and "lm_head" in t.name) else args.dtype
            if dt == "Q4_0":
                packed = quantize_q4_0(x)
                row_bytes = shape[0] // QK8_0 * Q4_0_BLOCK_BYTES
            else:
                packed = quantize_q8_0(x)
                row_bytes = shape[0] // QK8_0 * Q8_0_BLOCK_BYTES
            writer.add_tensor(t.name, packed,
                              raw_shape=(shape[1], row_bytes),
                              raw_dtype=GGMLQuantizationType[dt])
            quantized += 1
        else:
            writer.add_tensor(t.name, np.ascontiguousarray(x), raw_shape=shape[::-1])
        kept += 1
        kept_bytes += t.data.nbytes

    if kept == 0:
        sys.exit(f"ERROR: no {CP_PREFIX}* tensors in {args.src} — is this the "
                 f"transformer GGUF from convert_hf_to_gguf.py?")

    writer.write_header_to_file(path=args.dst)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"metadata: {n_kv} KV entries")
    print(f"tensors:  kept {kept} of {len(reader.tensors)} "
          f"({quantized} -> {args.dtype}, {kept - quantized} f16)")
    print(f"source tensor bytes: {total_bytes / 1048576:.1f} MB, "
          f"kept (pre-quant): {kept_bytes / 1048576:.1f} MB")
    print(f"out: {args.dst} ({os.path.getsize(args.dst) / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
