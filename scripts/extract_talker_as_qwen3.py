#!/usr/bin/env python3
"""
Extract Qwen3-TTS talker weights and repackage as standard Qwen3 model.

The talker is a 28-layer Qwen3 transformer. We extract its weights and
repackage them in standard Qwen3ForCausalLM format so that RKLLM toolkit
can convert them.

The codec embedding (3072 tokens) is expanded to full text vocab (151936)
to satisfy RKLLM validation. Only first 3072 entries are real.

Requirements:
  pip install safetensors torch transformers

Usage:
  python3 scripts/extract_talker_as_qwen3.py

  # Custom paths:
  QWEN3_TTS_MODEL=/path/to/model OUTPUT_DIR=./talker python3 scripts/extract_talker_as_qwen3.py
"""

import os
import json
import shutil
import torch
from safetensors.torch import load_file, save_file

SRC_MODEL = os.environ.get(
    "QWEN3_TTS_MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/"
        "snapshots/c27fe8aa05b732b1376d0f6a1e522fbccb84abbd"
    )
)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./qwen3_tts_talker_as_qwen3")
TEXT_VOCAB_SIZE = 151936


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Source: {SRC_MODEL}")
    print(f"Output: {OUTPUT_DIR}")

    src_path = os.path.join(SRC_MODEL, "model.safetensors")
    print(f"\nLoading {src_path}...")
    weights = load_file(src_path)

    new_weights = {}
    skipped_prefixes = set()

    for key, tensor in weights.items():
        if key.startswith("talker.model.layers."):
            new_key = key.replace("talker.model.layers.", "model.layers.")
            new_weights[new_key] = tensor
        elif key == "talker.model.codec_embedding.weight":
            codec_emb = tensor
            hidden_dim = codec_emb.shape[1]
            expanded = torch.zeros(TEXT_VOCAB_SIZE, hidden_dim, dtype=codec_emb.dtype)
            expanded[:codec_emb.shape[0]] = codec_emb
            new_weights["model.embed_tokens.weight"] = expanded
            print(f"  embed_tokens: expanded {list(codec_emb.shape)} -> {list(expanded.shape)}")
        elif key == "talker.model.norm.weight":
            new_weights["model.norm.weight"] = tensor
        elif key == "talker.codec_head.weight":
            head = tensor
            expanded_head = torch.zeros(TEXT_VOCAB_SIZE, head.shape[1], dtype=head.dtype)
            expanded_head[:head.shape[0]] = head
            new_weights["lm_head.weight"] = expanded_head
            print(f"  lm_head: expanded {list(head.shape)} -> {list(expanded_head.shape)}")
        else:
            prefix = key.split(".")[0]
            skipped_prefixes.add(prefix)

    print(f"\nMapped {len(new_weights)} tensors")
    print(f"Skipped prefixes: {sorted(skipped_prefixes)}")

    layer_nums = sorted(set(
        int(k.split(".")[2]) for k in new_weights if k.startswith("model.layers.")
    ))
    print(f"Layers: {len(layer_nums)} ({min(layer_nums)}..{max(layer_nums)})")

    out_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    print(f"\nSaving {out_path}...")
    save_file(new_weights, out_path)
    print(f"Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "max_position_embeddings": 32768,
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": TEXT_VOCAB_SIZE,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "use_cache": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "torch_dtype": "float32",
        "transformers_version": "4.57.3",
    }

    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved {config_path}")

    for fname in ["tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = os.path.join(SRC_MODEL, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUTPUT_DIR, fname))
            print(f"Copied {fname}")

    print(f"\n=== Ready for RKLLM conversion ===")
    print(f"  python3 scripts/convert_talker_rkllm.py --model_dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
