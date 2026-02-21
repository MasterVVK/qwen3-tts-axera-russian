#!/usr/bin/env python3
"""Extract code_predictor_weights.npz to individual .npy files for C++ loading."""
import numpy as np
import os, sys

npz_path = sys.argv[1] if len(sys.argv) > 1 else "/root/tts-rknn/code_predictor/code_predictor_weights.npz"
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(npz_path), "weights_npy")

os.makedirs(out_dir, exist_ok=True)
data = np.load(npz_path)

print(f"Extracting {npz_path} -> {out_dir}/")
for key in sorted(data.files):
    arr = data[key]
    out_path = os.path.join(out_dir, f"{key}.npy")
    np.save(out_path, arr.astype(np.float32))
    print(f"  {key}: {arr.shape} {arr.dtype} -> {out_path}")

print(f"\nDone: {len(data.files)} arrays extracted")
