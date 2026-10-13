#!/usr/bin/env python3
"""Board probe: how fast is ONE stateful code-predictor decode step on the NPU?

The CP is 15 sequential steps per generated token and ~63% of per-token CPU
time (64 ms per 15-step request on 4xA76 with the Q8_0 GGML backend, i.e.
~4.3 ms/step). This probe measures the same step on the NPU so the CP-on-NPU
idea can be accepted or killed with a number instead of a build.

Runs N steps of the RKNN model (KV cache carried in graph I/O, exactly the
export_code_predictor_stateful.py contract) and prints ms/step plus the output
norms (NaN/inf check).

Measured verdict (RK3588, NPU cores 0_1_2): 23.2 ms/step vs 4.3 ms/step on 4xA76 with
the GGML Q8_0 model -- 5.4x SLOWER, so the CP stays on the CPU. 157 MB fp16 / 23 ms is
6.8 GB/s, ~6x under the board's ~40 GB/s ceiling, i.e. the cost is per-op overhead on
~75 tiny S=1 GEMV ops plus 1.3 MB of KV across the runtime boundary, not bandwidth.

Usage on the board (no `-I`: it implies `-E`, which drops PYTHONPATH, and the shim is found
through PYTHONPATH):
  PYTHONPATH=/root/rknnlite_shim /root/tts-venv/bin/python cp_npu_probe.py \
      --model /root/tts-rknn/cp/cp_step_fp16.rknn \
      --weights /root/tts-rknn/code_predictor/code_predictor_weights.npz --steps 16
"""
import argparse
import math
import time

import numpy as np

L_KV, NL, KVH, D, H = 16, 5, 8, 128, 1024
THETA = 1000000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cores", default="0_1_2")
    args = ap.parse_args()

    from rknnlite.api import RKNNLite
    r = RKNNLite()
    assert r.load_rknn(args.model) == 0, "load_rknn"
    core = getattr(RKNNLite, f"NPU_CORE_{args.cores}")
    t0 = time.time()
    assert r.init_runtime(core_mask=core) == 0, "init_runtime"
    print(f"init_runtime: {time.time()-t0:.1f}s")

    w = np.load(args.weights)
    inv = 1.0 / (THETA ** (np.arange(0, D, 2) / D))
    emb = [w[f"codec_emb_{i}"].astype(np.float32) for i in range(15)]
    lm = [w[f"lm_head_{i}"].astype(np.float32) for i in range(15)]
    rng = np.random.default_rng(0)
    codes = []

    kv = [np.zeros((1, KVH, L_KV, D), np.float32) for _ in range(2 * NL)]
    t_best = None
    for rep in range(args.reps):
        kv = [np.zeros((1, KVH, L_KV, D), np.float32) for _ in range(2 * NL)]
        codes = []
        t0 = time.time()
        for pos in range(args.steps):
            if pos == 0:
                vec = (rng.standard_normal(H) / 32).astype(np.float32)
            elif pos == 1:
                vec = w["codec_emb_0"][7].astype(np.float32)
            else:
                vec = emb[pos - 2][codes[-1]].astype(np.float32)
            feed = [vec.reshape(1, 1, H).astype(np.float16),
                    np.remainder(np.outer([pos], inv), 2 * math.pi)
                      .astype(np.float16),
                    np.eye(L_KV, dtype=np.float16)[pos],
                    np.where(np.arange(L_KV) <= pos, 0.0, -100.0)
                      .reshape(1, 1, 1, L_KV).astype(np.float16)] + \
                   [k.astype(np.float16) for k in kv]
            out = r.inference(inputs=feed)
            kv = [o.astype(np.float32) for o in out[1:]]
            if pos:
                lg = out[0].astype(np.float32).reshape(1, 1, H)[0, -1] @ lm[pos - 1].T
                codes.append(int(np.argmax(lg)))
        dt = (time.time() - t0) / args.steps
        t_best = dt if t_best is None else min(t_best, dt)
        print(f"rep {rep}: {dt*1000:.1f} ms/step ({args.steps} steps) "
              f"codes {codes[:5]}... hidden rms "
              f"{float(np.sqrt((out[0].astype(np.float32)**2).mean())):.3f} "
              f"cache rms {float(np.sqrt((kv[0]**2).mean())):.3f}")
    print(f"\nBEST {t_best*1000:.2f} ms/step -> {t_best*15*1000:.0f} ms per token "
          f"(CPU GGML Q8_0 baseline: 4.3 ms/step, 64 ms/token)")
    r.release()


if __name__ == "__main__":
    main()
