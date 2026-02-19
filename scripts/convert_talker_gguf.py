#!/usr/bin/env python3
"""
Convert Qwen3-TTS talker to GGUF format for llama.cpp.

The talker has already been extracted as a standard Qwen3 model
(via extract_talker_as_qwen3.py). This script wraps the llama.cpp
convert_hf_to_gguf.py conversion.

Requirements:
  - llama.cpp repository with convert_hf_to_gguf.py
  - Extracted talker model (qwen3_tts_talker_as_qwen3/)

Usage:
  # On x86_64 host (NAS)
  python3 scripts/convert_talker_gguf.py

  # Custom paths
  python3 scripts/convert_talker_gguf.py \
      --model_dir ./qwen3_tts_talker_as_qwen3 \
      --llama_cpp_dir /path/to/llama.cpp \
      --output qwen3_tts_talker_q4_k_m.gguf \
      --quantize q4_k_m
"""

import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS talker to GGUF")
    parser.add_argument("--model_dir", default="./qwen3_tts_talker_as_qwen3",
                        help="Path to extracted Qwen3 model directory")
    parser.add_argument("--llama_cpp_dir", default=None,
                        help="Path to llama.cpp repo (auto-detect)")
    parser.add_argument("--output", default="qwen3_tts_talker_q4_k_m.gguf",
                        help="Output GGUF file path")
    parser.add_argument("--quantize", default="q4_k_m",
                        choices=["f32", "f16", "q8_0", "q4_k_m", "q4_0", "q5_k_m"],
                        help="Quantization type (default: q4_k_m)")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"ERROR: Model directory not found: {args.model_dir}")
        print("  Run: python3 scripts/extract_talker_as_qwen3.py")
        sys.exit(1)

    # Find llama.cpp
    if args.llama_cpp_dir is None:
        for path in ["/root/llama.cpp", os.path.expanduser("~/llama.cpp"),
                      "/home/user/NAS/llama.cpp"]:
            if os.path.exists(os.path.join(path, "convert_hf_to_gguf.py")):
                args.llama_cpp_dir = path
                break
        if args.llama_cpp_dir is None:
            print("ERROR: llama.cpp not found. Specify --llama_cpp_dir")
            sys.exit(1)

    convert_script = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        print(f"ERROR: {convert_script} not found")
        sys.exit(1)

    print(f"=== Converting Qwen3-TTS Talker to GGUF ===")
    print(f"Model:     {args.model_dir}")
    print(f"Output:    {args.output}")
    print(f"Quantize:  {args.quantize}")
    print(f"llama.cpp: {args.llama_cpp_dir}")
    print()

    cmd = [
        sys.executable, convert_script,
        args.model_dir,
        "--outfile", args.output,
        "--outtype", args.quantize,
    ]

    print(f"Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=args.llama_cpp_dir)

    if ret.returncode != 0:
        print(f"\nERROR: Conversion failed with code {ret.returncode}")
        sys.exit(1)

    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"\n=== Done ===")
        print(f"Output: {args.output} ({size_mb:.1f} MB)")
        print(f"\nDeploy: scp {args.output} root@cm3588:/root/tts-rknn/")
    else:
        print(f"WARNING: Output file not found at {args.output}")
        print("  Check if the file was created in a different directory")


if __name__ == "__main__":
    main()
