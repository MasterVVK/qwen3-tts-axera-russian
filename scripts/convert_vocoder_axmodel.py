#!/usr/bin/env python3
"""
Convert Qwen3-TTS vocoder ONNX to AX650N axmodel format.

Experimental: The vocoder is a ConvNet that may benefit from
the AX650N's 24 TOPS vs RK3588's 6 TOPS.

Requirements:
  - Pulsar2 toolkit (AX650N model compiler)
  - Vocoder ONNX model (from export_vocoder_traced.py)

Usage:
  python3 scripts/convert_vocoder_axmodel.py \
      --model vocoder_traced_64.onnx \
      --output vocoder_64.axmodel
"""

import os
import sys
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Convert vocoder to AX650N axmodel")
    parser.add_argument("--model", required=True, help="Input ONNX model")
    parser.add_argument("--output", default=None, help="Output axmodel path")
    parser.add_argument("--config", default=None, help="Pulsar2 config JSON")
    parser.add_argument("--codes_length", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: {args.model} not found")
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"{base}.axmodel"

    if args.codes_length is None:
        for n in ["64", "256"]:
            if n in os.path.basename(args.model):
                args.codes_length = int(n)
                break
        if args.codes_length is None:
            args.codes_length = 64

    print(f"=== Converting Vocoder to AX650N axmodel ===")
    print(f"Input:        {args.model}")
    print(f"Output:       {args.output}")
    print(f"Codes length: {args.codes_length}")

    # Generate Pulsar2 config if not provided
    if args.config is None:
        config = {
            "model_type": "ONNX",
            "npu_mode": "NPU1",
            "input_processors": [
                {
                    "tensor_name": "audio_codes",
                    "src_format": "AutoColorSpace",
                    "src_dtype": "int64",
                    "tensor_layout": "NHWC",
                }
            ],
            "compiler": {
                "check": 0,
                "debug": 0,
                "optlevel": 3,
            }
        }
        config_path = "/tmp/vocoder_pulsar2_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        args.config = config_path
        print(f"Generated config: {config_path}")

    print(f"\nTo convert with Pulsar2:")
    print(f"  pulsar2 build \\")
    print(f"    --input {args.model} \\")
    print(f"    --output {args.output} \\")
    print(f"    --config {args.config} \\")
    print(f"    --target_hardware AX650")
    print(f"\nNote: This is experimental. Fallback to RKNN vocoder if quality degrades.")


if __name__ == "__main__":
    main()
