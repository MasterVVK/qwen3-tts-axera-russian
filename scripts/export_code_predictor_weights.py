#!/usr/bin/env python3
"""
Export ALL code predictor weights as numpy for pure-numpy inference.

Extracts:
  - 5 transformer layers (Q/K/V/O projections, QK-norm, MLP, LayerNorm)
  - Final RMSNorm
  - 15 codec embeddings + 15 lm_heads

Output: code_predictor_weights.npz (~240 MB)

Requirements:
  pip install safetensors numpy

Usage:
  python3 scripts/export_code_predictor_weights.py
"""

import os
import numpy as np
from safetensors.torch import load_file

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    model_dir = os.environ.get(
        "QWEN3_TTS_MODEL",
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/"
            "snapshots/c27fe8aa05b732b1376d0f6a1e522fbccb84abbd"
        )
    )
    output_dir = os.environ.get("OUTPUT_DIR", "./code_predictor_onnx")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Source: {model_dir}")
    print(f"Output: {output_dir}")

    print("\nLoading safetensors...")
    weights = load_file(os.path.join(model_dir, "model.safetensors"))

    prefix = "talker.code_predictor."
    cp_weights = {k[len(prefix):]: v.cpu().float().numpy()
                  for k, v in weights.items() if k.startswith(prefix)}
    print(f"Found {len(cp_weights)} code predictor tensors")

    np_weights = {}
    num_layers = 5

    for i in range(num_layers):
        lp = f"model.layers.{i}."
        np_weights[f"layer_{i}_input_ln"] = cp_weights[lp + "input_layernorm.weight"]
        np_weights[f"layer_{i}_q_proj"] = cp_weights[lp + "self_attn.q_proj.weight"]
        np_weights[f"layer_{i}_k_proj"] = cp_weights[lp + "self_attn.k_proj.weight"]
        np_weights[f"layer_{i}_v_proj"] = cp_weights[lp + "self_attn.v_proj.weight"]
        np_weights[f"layer_{i}_o_proj"] = cp_weights[lp + "self_attn.o_proj.weight"]
        np_weights[f"layer_{i}_q_norm"] = cp_weights[lp + "self_attn.q_norm.weight"]
        np_weights[f"layer_{i}_k_norm"] = cp_weights[lp + "self_attn.k_norm.weight"]
        np_weights[f"layer_{i}_post_ln"] = cp_weights[lp + "post_attention_layernorm.weight"]
        np_weights[f"layer_{i}_gate_proj"] = cp_weights[lp + "mlp.gate_proj.weight"]
        np_weights[f"layer_{i}_up_proj"] = cp_weights[lp + "mlp.up_proj.weight"]
        np_weights[f"layer_{i}_down_proj"] = cp_weights[lp + "mlp.down_proj.weight"]

        if i == 0:
            for name in ["input_ln", "q_proj", "k_proj"]:
                w = np_weights[f"layer_0_{name}"]
                print(f"  layer_0_{name}: {w.shape} {w.dtype}")

    np_weights["final_norm"] = cp_weights["model.norm.weight"]

    for i in range(15):
        np_weights[f"codec_emb_{i}"] = cp_weights[f"model.codec_embedding.{i}.weight"]
        np_weights[f"lm_head_{i}"] = cp_weights[f"lm_head.{i}.weight"]

    npz_path = os.path.join(output_dir, "code_predictor_weights.npz")
    print(f"\nSaving {len(np_weights)} arrays to {npz_path}...")
    np.savez(npz_path, **np_weights)
    print(f"Saved: {os.path.getsize(npz_path)/1024/1024:.1f} MB")

    # Verify
    loaded = np.load(npz_path)
    print(f"\nVerification: loaded {len(loaded.files)} arrays")
    for name in ["layer_0_q_proj", "final_norm", "codec_emb_0", "lm_head_14"]:
        print(f"  {name}: {loaded[name].shape} {loaded[name].dtype}")

    print("\nDone!")


if __name__ == "__main__":
    main()
