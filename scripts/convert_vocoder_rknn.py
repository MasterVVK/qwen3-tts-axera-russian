#!/usr/bin/env python3
"""
Convert Qwen3-TTS vocoder ONNX model to RKNN for RK3588 NPU.

The vocoder is a ConvNet that converts codec tokens to audio waveforms.
INT8 quantization provides 2x speedup with minimal quality loss.

Requirements:
  pip install rknn-toolkit2  # On x86_64 host

Usage:
  # FP16 (no quantization)
  python3 scripts/convert_vocoder_rknn.py --model vocoder_traced_64.onnx

  # INT8 quantization (recommended)
  python3 scripts/convert_vocoder_rknn.py --model vocoder_traced_64.onnx --quantize

  # Custom output
  python3 scripts/convert_vocoder_rknn.py --model vocoder_traced_64.onnx --output vocoder_q8.rknn
"""

import os
import sys
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Convert vocoder ONNX to RKNN")
    parser.add_argument("--model", required=True, help="Input ONNX model path")
    parser.add_argument("--output", default=None, help="Output RKNN path")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply INT8 quantization")
    parser.add_argument("--target", default="rk3588",
                        choices=["rk3588", "rk3576"])
    parser.add_argument("--codes_length", type=int, default=None,
                        help="Number of codec tokens (auto-detect from filename)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: {args.model} not found")
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.model))[0]
        suffix = "_q8" if args.quantize else "_fp16"
        args.output = f"{base}{suffix}.rknn"

    # Auto-detect codes_length from filename
    if args.codes_length is None:
        for n in ["64", "256"]:
            if n in os.path.basename(args.model):
                args.codes_length = int(n)
                break
        if args.codes_length is None:
            args.codes_length = 64

    print(f"=== Converting Vocoder ONNX -> RKNN ===")
    print(f"Input:        {args.model}")
    print(f"Output:       {args.output}")
    print(f"Codes length: {args.codes_length}")
    print(f"Quantize:     {args.quantize}")
    print(f"Target:       {args.target}")

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)

    print("\n1. Configuring RKNN...")
    rknn.config(target_platform=args.target, optimization_level=1)

    print("\n2. Loading ONNX model...")
    ret = rknn.load_onnx(
        model=args.model,
        inputs=['audio_codes'],
        input_size_list=[[1, args.codes_length, 16]],
    )
    if ret != 0:
        print(f"ERROR: Failed to load ONNX: {ret}")
        sys.exit(1)

    print("\n3. Building RKNN model...")
    if args.quantize:
        # Generate calibration data
        n_samples = 20
        dataset_path = "/tmp/vocoder_calib_data.txt"
        calib_dir = "/tmp/vocoder_calib"
        os.makedirs(calib_dir, exist_ok=True)

        with open(dataset_path, "w") as f:
            for i in range(n_samples):
                npy_path = os.path.join(calib_dir, f"sample_{i}.npy")
                data = np.random.randint(0, 2048,
                    (1, args.codes_length, 16), dtype=np.int64)
                np.save(npy_path, data)
                f.write(f"{npy_path}\n")

        ret = rknn.build(do_quantization=True, dataset=dataset_path)
    else:
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f"ERROR: Failed to build: {ret}")
        sys.exit(1)

    print(f"\n4. Exporting to {args.output}...")
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print(f"ERROR: Failed to export: {ret}")
        sys.exit(1)

    rknn.release()

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\n=== Done ===")
    print(f"Output: {args.output} ({size_mb:.1f} MB)")
    print(f"\nDeploy: scp {args.output} root@cm3588:/root/tts-rknn/")


if __name__ == "__main__":
    main()
