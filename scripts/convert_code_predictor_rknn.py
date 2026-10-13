#!/usr/bin/env python3
"""
Convert the stateful code-predictor step graph
(scripts/export_code_predictor_stateful.py) to RKNN for RK3588/RK3576.

The graph is fully static: 14 inputs (hidden, RoPE angles, cache write-select,
attention mask, and past_k_i/past_v_i for i=0..4) and 11 outputs, so RKNN gets
explicit shapes and the 15-step per-token loop is 15 calls of ONE model.

FP16 (w16a16, no --quantize) is the default and the only mode measured so far:
the vocoder showed RKNN FP16 is numerically lossless (SNR 56.8 dB) while Q8/INT8
of these audio nets is not (9.5-13.4 dB). INT8 is wired up (rknn-toolkit2 2.x
`quantized_dtype='w8a8', quantized_method='channel'`) for a per-layer probe, not
yet validated.

Calibration dataset format (rknn-toolkit2 multi-input): one line per sample, the
14 input .npy paths in input order, space-separated.

Usage:
  python3 scripts/convert_code_predictor_rknn.py \
      --model code_predictor_onnx/code_predictor_decode_step.onnx
"""

import argparse
import os
import sys

N_LAYERS = 5
N_KV_HEADS = 8
L_KV = 16
HEAD_DIM = 128
HIDDEN = 1024


def graph_inputs():
    """(name, shape) in the exact order export_code_predictor_stateful.py wrote."""
    ins = [("hidden", [1, 1, HIDDEN]),
           ("freqs_in", [1, HEAD_DIM // 2]),
           ("sel", [L_KV]),
           ("mask", [1, 1, 1, L_KV])]
    for i in range(N_LAYERS):
        ins.append((f"past_k_{i}", [1, N_KV_HEADS, L_KV, HEAD_DIM]))
        ins.append((f"past_v_{i}", [1, N_KV_HEADS, L_KV, HEAD_DIM]))
    return ins


def main():
    ap = argparse.ArgumentParser(description="Convert stateful CP step to RKNN")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--target", default="rk3588", choices=["rk3588", "rk3576"])
    ap.add_argument("--quantize", action="store_true", help="INT8 (unmeasured)")
    ap.add_argument("--dataset", default=None, help="calibration list (see docstring)")
    ap.add_argument("--opt-level", type=int, default=1)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: {args.model} not found")
        sys.exit(1)
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"{base}{'_q8' if args.quantize else '_fp16'}.rknn"

    ins = graph_inputs()
    print(f"=== Converting stateful CP step ONNX -> RKNN ({args.target}) ===")
    print(f"Input:  {args.model}")
    print(f"Output: {args.output}")
    print(f"Inputs: {len(ins)}  quantize: {args.quantize}")

    from rknn.api import RKNN
    rknn = RKNN(verbose=True)
    # fp16 path: float_dtype defaults to float16 and build(do_quantization=False)
    # int8 path: w8a8 with per-channel scales (the RKNN 2.x default method),
    # calibrated on real recorded CP traffic
    rknn.config(target_platform=args.target,
                optimization_level=args.opt_level,
                quantized_dtype="w8a8", quantized_method="channel")

    if rknn.load_onnx(model=args.model, inputs=[n for n, _ in ins],
                      input_size_list=[s for _, s in ins]) != 0:
        print("ERROR: load_onnx failed")
        sys.exit(1)

    if args.quantize:
        if not args.dataset:
            print("ERROR: --quantize needs --dataset (real recorded CP traffic); "
                  "random calibration data is how the vocoder Q8 model got SNR 9.5 dB")
            sys.exit(1)
        ret = rknn.build(do_quantization=True, dataset=args.dataset)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"ERROR: build failed: {ret}")
        sys.exit(1)

    if rknn.export_rknn(args.output) != 0:
        print("ERROR: export_rknn failed")
        sys.exit(1)
    rknn.release()
    print(f"\nDone: {args.output} "
          f"({os.path.getsize(args.output)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
