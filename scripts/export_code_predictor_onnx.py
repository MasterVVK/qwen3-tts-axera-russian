#!/usr/bin/env python3
"""
Export Qwen3-TTS code predictor transformer core to ONNX.

The code predictor has 5 transformer layers + RMSNorm shared across 15 groups.
This script exports the transformer core; embedding lookup and lm_head are
handled on CPU (numpy).

Also exports decode-step ONNX with KV-cache for efficient autoregressive inference.

Requirements:
  pip install torch safetensors onnx onnxruntime
  pip install qwen-tts  # For model classes

Usage:
  python3 scripts/export_code_predictor_onnx.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class CodePredictorCore(nn.Module):
    """Wraps the transformer layers + norm from the code predictor."""

    def __init__(self, model):
        super().__init__()
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.rotary_emb = model.model.rotary_emb
        self.small_to_mtp_projection = model.small_to_mtp_projection

    def forward(self, hidden_states, position_ids=None):
        hidden_states = self.small_to_mtp_projection(hidden_states)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            layer_out = layer(hidden_states, position_embeddings=position_embeddings)
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        return self.norm(hidden_states)


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

    print("\nLoading model weights...")
    weights = load_file(os.path.join(model_dir, "model.safetensors"))

    with open(os.path.join(model_dir, "config.json")) as f:
        full_config = json.load(f)

    talker_config_dict = full_config.get("talker_config", {})
    cp_config_dict = talker_config_dict.get("code_predictor_config", {})

    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    )

    talker_cfg = Qwen3TTSTalkerConfig(**talker_config_dict)
    cp_config = Qwen3TTSTalkerCodePredictorConfig(**cp_config_dict)

    print(f"Code predictor: {cp_config.num_hidden_layers} layers, "
          f"hidden={cp_config.hidden_size}, groups={cp_config.num_code_groups}")

    cp_state = {k.replace("talker.code_predictor.", ""): v
                for k, v in weights.items() if k.startswith("talker.code_predictor.")}

    model = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(cp_config, talker_cfg)
    model.load_state_dict(cp_state, strict=False)
    model.eval()

    # Extract embeddings and heads
    num_groups = cp_config.num_code_groups - 1  # 15
    print(f"\nExtracting {num_groups} codec embeddings and lm_heads...")
    for i in range(num_groups):
        emb_w = model.model.codec_embedding[i].weight.data.cpu().numpy()
        np.save(os.path.join(output_dir, f"codec_embedding_{i}.npy"), emb_w)
        head_w = model.lm_head[i].weight.data.cpu().numpy()
        np.save(os.path.join(output_dir, f"lm_head_{i}.npy"), head_w)

    # Create and export core model
    print("\nCreating transformer core wrapper...")
    core = CodePredictorCore(model)
    core.eval()

    dummy_input = torch.randn(1, 2, cp_config.hidden_size)
    dummy_pos = torch.arange(2).unsqueeze(0)

    with torch.no_grad():
        test_out = core(dummy_input, dummy_pos)
    print(f"Test forward: {dummy_input.shape} -> {test_out.shape}")

    onnx_path = os.path.join(output_dir, "code_predictor_core.onnx")
    print(f"\nExporting to ONNX...")
    torch.onnx.export(
        core, (dummy_input, dummy_pos), onnx_path,
        input_names=["hidden_states", "position_ids"],
        output_names=["output"],
        dynamic_axes={
            "hidden_states": {0: "batch", 1: "seq_len"},
            "position_ids": {0: "batch", 1: "seq_len"},
            "output": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"ONNX saved: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

    # Verify
    print("\nVerifying with ONNX Runtime...")
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    ort_out = session.run(None, {
        "hidden_states": dummy_input.numpy(),
        "position_ids": dummy_pos.numpy(),
    })
    diff = np.abs(test_out.numpy() - ort_out[0]).max()
    print(f"Max diff PyTorch vs ONNX: {diff:.6f}")

    # Save config
    cp_info = {
        "hidden_size": cp_config.hidden_size,
        "num_layers": cp_config.num_hidden_layers,
        "num_groups": num_groups,
        "vocab_size": cp_config.vocab_size,
        "num_attention_heads": cp_config.num_attention_heads,
        "num_kv_heads": cp_config.num_key_value_heads,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cp_info, f, indent=2)

    print(f"\nDone! Files in {output_dir}:")
    for fn in sorted(os.listdir(output_dir)):
        sz = os.path.getsize(os.path.join(output_dir, fn))
        print(f"  {fn}: {sz/1024/1024:.1f} MB" if sz > 1024*1024 else f"  {fn}: {sz/1024:.1f} KB")


if __name__ == "__main__":
    main()
