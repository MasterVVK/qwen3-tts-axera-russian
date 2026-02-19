#!/usr/bin/env python3
"""
Export Qwen3-TTS vocoder to ONNX with proper tracing for fixed sequence length.

Traces the vocoder model with actual input tensors, producing an ONNX model
where all internal computations match the target shape.

Requirements:
  pip install torch onnx onnxruntime
  pip install qwen-tts  # For model classes

Usage:
  # Export for 64 tokens
  python3 scripts/export_vocoder_traced.py --codes_length 64

  # Export for 256 tokens
  python3 scripts/export_vocoder_traced.py --codes_length 256

  # With simplification
  python3 scripts/export_vocoder_traced.py --codes_length 64 --simplify --remove_isnan
"""

import os
import argparse
import torch
import numpy as np

MODEL_DIR = os.environ.get(
    "QWEN3_TTS_MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/"
        "snapshots/c27fe8aa05b732b1376d0f6a1e522fbccb84abbd"
    )
)
TOKENIZER_DIR = os.path.join(MODEL_DIR, "speech_tokenizer")


class VocoderWrapper(torch.nn.Module):
    """Wrapper: [batch, seq_len, 16] -> audio + lengths."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.total_upsample = decoder.total_upsample

    def forward(self, audio_codes):
        codes = audio_codes.permute(0, 2, 1).long()
        wav = self.decoder(codes)
        audio_values = wav.squeeze(1)
        lengths = torch.tensor([audio_codes.shape[1] * self.total_upsample],
                               dtype=torch.int64)
        return audio_values, lengths


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS vocoder to ONNX")
    parser.add_argument("--codes_length", type=int, default=64,
                        help="Fixed number of codec tokens (default: 64)")
    parser.add_argument("--output", default=None,
                        help="Output ONNX path (auto-generated if not specified)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--remove_isnan", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"vocoder_traced_{args.codes_length}.onnx"

    print(f"=== Vocoder ONNX Export ===")
    print(f"Codes length: {args.codes_length}")
    print(f"Output:       {args.output}")

    print("\n1. Loading model...")
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )
    full_model = Qwen3TTSTokenizerV2Model.from_pretrained(TOKENIZER_DIR)
    full_model.eval()
    decoder = full_model.decoder
    print(f"   Decoder loaded. total_upsample={decoder.total_upsample}")

    wrapper = VocoderWrapper(decoder)
    wrapper.eval()

    dummy_input = torch.randint(0, 2048, (1, args.codes_length, 16), dtype=torch.int64)

    print("\n2. Testing forward pass...")
    with torch.no_grad():
        audio, lengths = wrapper(dummy_input)
    print(f"   Audio: {audio.shape}, Lengths: {lengths}")

    print(f"\n3. Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        wrapper, (dummy_input,), args.output,
        input_names=["audio_codes"],
        output_names=["audio_values", "lengths"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"   Saved: {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")

    # Verify
    print("\n4. Verifying ONNX...")
    import onnx
    model = onnx.load(args.output)
    onnx.checker.check_model(model)

    if args.remove_isnan:
        print("\n5. Removing IsNaN ops...")
        clean_nodes = []
        skip_outputs = set()
        for node in model.graph.node:
            if node.op_type == "IsNaN":
                skip_outputs.add(node.output[0])
            elif node.op_type == "Where" and node.input[0] in skip_outputs:
                clean_nodes.append(onnx.helper.make_node(
                    "Identity", inputs=[node.input[2]], outputs=node.output,
                ))
            else:
                clean_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(clean_nodes)

    if args.simplify:
        print("\n6. Running onnxsim...")
        import onnxsim
        model, check = onnxsim.simplify(model)
        print(f"   onnxsim: {'OK' if check else 'WARNING'}")

    if args.remove_isnan or args.simplify:
        onnx.save(model, args.output)

    # ORT verification
    print("\n7. ONNX Runtime verification...")
    import onnxruntime as ort
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    test_input = np.random.randint(0, 2048, (1, args.codes_length, 16)).astype(np.int64)
    ort_out = sess.run(None, {"audio_codes": test_input})
    print(f"   Output: {ort_out[0].shape}")

    print(f"\n=== Done ===")
    print(f"Next: python3 scripts/convert_vocoder_rknn.py --model {args.output}")


if __name__ == "__main__":
    main()
