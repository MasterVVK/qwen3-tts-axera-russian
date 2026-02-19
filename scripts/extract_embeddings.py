#!/usr/bin/env python3
"""
Extract text/codec embeddings and projection weights from Qwen3-TTS.

These components run on CPU (numpy) before feeding to RKLLM:
1. text_embedding: [151936, 2048] - text token lookup
2. text_projection: MLP(2048 -> 2048 -> 1024) - project to hidden dim
3. codec_embedding: [3072, 1024] - codec token lookup
4. codec_head: [3072, 1024] - logits for sampling
5. code_predictor weights: 5 layers + embeddings/heads for 15 groups

Requirements:
  pip install safetensors numpy

Usage:
  python3 scripts/extract_embeddings.py

  # Custom paths:
  QWEN3_TTS_MODEL=/path/to/model OUTPUT_DIR=./embeddings python3 scripts/extract_embeddings.py
"""

import os
import numpy as np
from safetensors.torch import load_file

SRC_MODEL = os.environ.get(
    "QWEN3_TTS_MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/"
        "snapshots/c27fe8aa05b732b1376d0f6a1e522fbccb84abbd"
    )
)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./qwen3_tts_embeddings")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Source: {SRC_MODEL}")
    print(f"Output: {OUTPUT_DIR}")

    src_path = os.path.join(SRC_MODEL, "model.safetensors")
    print(f"\nLoading {src_path}...")
    weights = load_file(src_path)

    # 1. Text embedding
    text_emb = weights["talker.model.text_embedding.weight"].float().numpy()
    np.save(os.path.join(OUTPUT_DIR, "text_embedding.npy"), text_emb)
    print(f"text_embedding: {text_emb.shape} ({text_emb.nbytes/1024/1024:.1f} MB)")

    # 2. Text projection MLP
    for k in ["talker.text_projection.linear_fc1.weight",
              "talker.text_projection.linear_fc1.bias",
              "talker.text_projection.linear_fc2.weight",
              "talker.text_projection.linear_fc2.bias"]:
        short_name = k.split("talker.text_projection.")[-1].replace(".", "_")
        arr = weights[k].float().numpy()
        np.save(os.path.join(OUTPUT_DIR, f"text_projection_{short_name}.npy"), arr)
        print(f"text_projection.{short_name}: {arr.shape}")

    # 3. Codec embedding
    codec_emb = weights["talker.model.codec_embedding.weight"].float().numpy()
    np.save(os.path.join(OUTPUT_DIR, "codec_embedding.npy"), codec_emb)
    print(f"codec_embedding: {codec_emb.shape} ({codec_emb.nbytes/1024/1024:.1f} MB)")

    # 4. Codec head
    codec_head = weights["talker.codec_head.weight"].float().numpy()
    np.save(os.path.join(OUTPUT_DIR, "codec_head.npy"), codec_head)
    print(f"codec_head: {codec_head.shape} ({codec_head.nbytes/1024/1024:.1f} MB)")

    # 5. Code predictor weights
    cp_dir = os.path.join(OUTPUT_DIR, "code_predictor")
    os.makedirs(cp_dir, exist_ok=True)

    # Embeddings and heads
    for i in range(15):
        k_emb = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if k_emb in weights:
            np.save(os.path.join(cp_dir, f"codec_embedding_{i}.npy"),
                    weights[k_emb].float().numpy())
        k_head = f"talker.code_predictor.lm_head.{i}.weight"
        if k_head in weights:
            np.save(os.path.join(cp_dir, f"lm_head_{i}.npy"),
                    weights[k_head].float().numpy())
    print(f"code_predictor: 15 embedding + 15 lm_head groups saved")

    # Transformer layers
    cp_layers = {}
    for key, tensor in weights.items():
        if key.startswith("talker.code_predictor.model.layers."):
            short_key = key.replace("talker.code_predictor.model.", "")
            cp_layers[short_key] = tensor.float().numpy()
        elif key == "talker.code_predictor.model.norm.weight":
            cp_layers["norm.weight"] = tensor.float().numpy()

    np.savez_compressed(os.path.join(cp_dir, "transformer_layers.npz"), **cp_layers)
    print(f"code_predictor transformer: {len(cp_layers)} tensors saved")

    # Summary
    total_size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(OUTPUT_DIR)
        for f in files
    )
    print(f"\n=== Embeddings extracted ===")
    print(f"Total size: {total_size/1024/1024:.1f} MB")
    print(f"Directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
