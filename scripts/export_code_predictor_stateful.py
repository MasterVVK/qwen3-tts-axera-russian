#!/usr/bin/env python3
"""
Export the Qwen3-TTS code predictor (CP) as a STATEFUL streaming step graph:
one fixed-shape decode step with the KV cache as explicit graph I/O, so the
whole 15-step per-token loop can run on a runtime that has no dynamic shapes
(RKNN/NPU). Embedding lookup and lm_head sampling stay outside the graph
(numpy), exactly as dual_npu/code_predictor_server.py does them.

The CP is 5 layers, hidden 1024, GQA 16 q-heads / 8 kv-heads, head_dim 128,
RoPE theta 1e6, all full attention, no sliding window. One generated token is
16 positions: the talker hidden vector at pos 0, the code_0 embedding at pos 1,
then the 14 previously sampled codebook embeddings at pos 2..15. So the KV cache
has a compile-time maximum of 16 slots — the graph is fully static.

One graph serves every step. The step index enters as data, not as shape:
  freqs_in [1,64]  RoPE angles pos*inv_freq reduced mod 2*pi on the host
                   (cos/sin computed in-graph; cos/sin as inputs explodes the
                   RKNN fp16 pipeline — measured on the vocoder export)
  sel      [16]    one-hot write index: cache_k_out = cache_k + sel * k_new
                   (a broadcast Mul+Add, i.e. the constant-selection trick that
                   replaced Concat in the vocoder export, with the selection
                   vector supplied by the host instead of baked in)
  mask     [1,1,1,16] additive attention mask, 0.0 for written slots and -100
                   for the not-yet-written ones (a -3e4 mask leaks through the
                   NPU softmax; the data-dependent chain that would derive it
                   from the cache content costs ~30 dB, so it is an input)
  past_k_i/past_v_i [1,8,16,128]  KV cache in (zeros on the first step)
  past_*_i outputs                KV cache out

GQA head repeat is `cat([k, k], dim=1)`-style Concat, not Expand: the RKNN
Expand path is a known problem (cosyvoice3-rknn-russian/scripts/export_dec.py).

Usage:
  python3 scripts/export_code_predictor_stateful.py \
      --weights code_predictor_onnx/code_predictor_weights.npz \
      --output code_predictor_onnx/code_predictor_decode_step.onnx --check 20
"""

import argparse
import json
import math
import os

import numpy as np
import torch

L_KV = 16          # 16 positions per generated token: cache is always full size
N_LAYERS = 5
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 1024
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6
N_GROUPS = 15      # codebooks 0..14 predicted by the CP


def rms_norm(x, w, eps=RMS_EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rot_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class StatefulCodePredictor(torch.nn.Module):
    """One CP decode step: 1 embedding + KV cache in -> hidden + KV cache out."""

    def __init__(self, w):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (
            ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float()
                           / HEAD_DIM)), persistent=False)
        # GQA selection: q-head h reads kv-head h // (16/8)
        self.register_buffer("gqa_sel", torch.eye(
            N_KV_HEADS).repeat_interleave(N_HEADS // N_KV_HEADS, dim=0),
            persistent=False)
        for i in range(N_LAYERS):
            for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                      "up_proj", "down_proj", "input_ln", "post_ln",
                      "q_norm", "k_norm"):
                self.register_parameter(
                    f"l{i}_{n}",
                    torch.nn.Parameter(torch.from_numpy(w[f"layer_{i}_{n}"].copy()),
                                       requires_grad=False))
        self.register_parameter(
            "final_norm",
            torch.nn.Parameter(torch.from_numpy(w["final_norm"].copy()),
                               requires_grad=False))
        self.scaling = HEAD_DIM ** -0.5

    def forward(self, hidden, freqs, sel, mask, *past):
        """hidden [1,1,H], freqs [1,HEAD_DIM/2], sel [L_KV], mask [1,1,1,L_KV],
        past = past_k_0, past_v_0, ..., past_k_4, past_v_4 -> [1,KV,L_KV,D]"""
        emb = torch.cat((freqs, freqs), dim=-1)          # [1, HEAD_DIM]
        cos, sin = emb.cos(), emb.sin()
        x = hidden
        new_past = []
        for i in range(N_LAYERS):
            P = lambda n, i=i: getattr(self, f"l{i}_{n}")
            h = rms_norm(x, P("input_ln"))
            B, S, _ = h.shape
            q = rms_norm(torch.matmul(h, P("q_proj").T).view(B, S, N_HEADS, HEAD_DIM),
                         P("q_norm")).transpose(1, 2)     # [B,H,S,D]
            k = rms_norm(torch.matmul(h, P("k_proj").T).view(B, S, N_KV_HEADS, HEAD_DIM),
                         P("k_norm")).transpose(1, 2)     # [B,KV,S,D]
            v = torch.matmul(h, P("v_proj").T).view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
            q = q * cos + rot_half(q) * sin              # cos [1,D] broadcasts
            k = k * cos + rot_half(k) * sin              # over B,S(=1),KV

            pk, pv = past[2 * i], past[2 * i + 1]
            # cache write: outer(sel, k_new) as a matmul, no Reshape of sel
            selm = sel.view(1, 1, L_KV, 1)
            nk = pk + torch.matmul(selm, k)              # [B,KV,L_KV,D]
            nv = pv + torch.matmul(selm, v)
            new_past += [nk, nv]

            # GQA selection as a constant one-hot matmul (q-head h reads kv-head
            # h//2) — the same trick that replaced Concat in the vocoder stateful
            # export: Unsqueeze/Concat/Expand chains make rknn-toolkit2's
            # graph_optimizer die in fuse_ops. The matmul emits [B,L,H,D]
            # (batch dims are B and L_KV, the matrix rows are the q-heads), so
            # transpose back to [B,H,L,D]. L_KV == N_HEADS == 16 here, so a
            # missing transpose is shape-invisible and numerically fatal.
            kf = torch.matmul(self.gqa_sel.view(1, N_HEADS, N_KV_HEADS),
                              nk.transpose(1, 2)).transpose(1, 2)
            vf = torch.matmul(self.gqa_sel.view(1, N_HEADS, N_KV_HEADS),
                              nv.transpose(1, 2)).transpose(1, 2)
            s = torch.matmul(q, kf.transpose(2, 3)) * self.scaling + mask
            a = torch.softmax(s, dim=-1)
            o = torch.matmul(a, vf).transpose(1, 2).reshape(B, S, N_HEADS * HEAD_DIM)
            x = x + torch.matmul(o, P("o_proj").T)
            h = rms_norm(x, P("post_ln"))
            x = x + torch.matmul(
                torch.nn.functional.silu(torch.matmul(h, P("gate_proj").T))
                * torch.matmul(h, P("up_proj").T), P("down_proj").T)
        return (rms_norm(x, self.final_norm),) + tuple(new_past)


def rope_freqs(pos, inv_freq):
    """Host-side RoPE angles, reduced mod 2*pi (small angles keep fp16 accurate)."""
    return np.remainder(np.outer(np.atleast_1d(pos).astype(np.float64),
                                 np.asarray(inv_freq).astype(np.float64)),
                        2 * math.pi).astype(np.float32)


def run_stateful(sess, w, hidden, code_0_embed, n_steps=15, greedy=True):
    """Replay the whole per-token loop through the stateful graph (fp32 CPU).

    16 graph calls (positions 0..15): pos 0 = talker hidden (output not sampled),
    pos 1 = code_0 embedding -> codebook 0, pos p = codec_emb[p-2] of the code
    sampled at p-2 -> codebook p-1. Exactly the server's decode_step sequence.
    """
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
    kv = {}
    for i in range(N_LAYERS):
        kv[f"past_k_{i}"] = np.zeros((1, N_KV_HEADS, L_KV, HEAD_DIM), np.float32)
        kv[f"past_v_{i}"] = np.zeros((1, N_KV_HEADS, L_KV, HEAD_DIM), np.float32)
    lm = [w[f"lm_head_{i}"].astype(np.float32) for i in range(N_GROUPS)]
    emb = [w[f"codec_emb_{i}"].astype(np.float32) for i in range(N_GROUPS)]
    codes, logits = [], []
    for pos in range(n_steps + 1):
        if pos == 0:
            vec = hidden.reshape(1, 1, HIDDEN)
        elif pos == 1:
            vec = code_0_embed.reshape(1, 1, HIDDEN)
        else:
            vec = emb[pos - 2][codes[pos - 2]].reshape(1, 1, HIDDEN)
        feed = {"hidden": vec,
                "freqs_in": rope_freqs(np.array([pos]), inv_freq),
                "sel": np.eye(L_KV, dtype=np.float32)[pos],
                "mask": np.where(np.arange(L_KV) <= pos, 0.0, -100.0)
                        .reshape(1, 1, 1, L_KV).astype(np.float32)}
        feed.update(kv)
        out = sess.run(None, feed)
        kv = {}
        for i in range(N_LAYERS):
            kv[f"past_k_{i}"] = out[1 + 2 * i]
            kv[f"past_v_{i}"] = out[2 + 2 * i]
        if pos == 0:
            continue
        lg = out[0][0, -1] @ lm[pos - 1].T
        logits.append(lg)
        codes.append(int(np.argmax(lg)) if greedy else 0)
    return codes, logits


def _np_rms(x, w, eps=RMS_EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _np_softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _np_rot(x):
    d = x.shape[-1] // 2
    return np.concatenate((-x[..., d:], x[..., :d]), axis=-1)


def run_reference(w, hidden, code_0_embed, n_steps=15):
    """Reference: the same transformer re-run from scratch on the growing
    prefix, in plain numpy, causal, fp32 — an independent implementation of the
    HF code predictor (the core.onnx export is NOT a valid reference: it is
    exported with attention_mask=None, i.e. bidirectional over the prefix)."""
    inv = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
    lm = [w[f"lm_head_{i}"].astype(np.float64) for i in range(N_GROUPS)]
    emb = [w[f"codec_emb_{i}"].astype(np.float32) for i in range(N_GROUPS)]
    seq = [hidden.reshape(HIDDEN), code_0_embed.reshape(HIDDEN)]
    codes, logits = [], []
    for step in range(n_steps):
        n = len(seq)
        x = np.stack(seq).astype(np.float64)[None]
        ang = np.outer(np.arange(n), inv)
        cos, sin = np.cos(np.concatenate((ang, ang), -1))[None, None], \
            np.sin(np.concatenate((ang, ang), -1))[None, None]
        causal = np.where(np.arange(n)[None, :] <= np.arange(n)[:, None],
                          0.0, -100.0)[None, None]
        for i in range(N_LAYERS):
            g = lambda nm: w[f"layer_{i}_{nm}"].astype(np.float64)
            h = _np_rms(x, g("input_ln"))
            q = _np_rms((h @ g("q_proj").T).reshape(1, n, N_HEADS, HEAD_DIM),
                        g("q_norm")).transpose(0, 2, 1, 3)
            k = _np_rms((h @ g("k_proj").T).reshape(1, n, N_KV_HEADS, HEAD_DIM),
                        g("k_norm")).transpose(0, 2, 1, 3)
            v = (h @ g("v_proj").T).reshape(1, n, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            q = q * cos + _np_rot(q) * sin
            k = k * cos + _np_rot(k) * sin
            kf = np.repeat(k, N_HEADS // N_KV_HEADS, axis=1)
            vf = np.repeat(v, N_HEADS // N_KV_HEADS, axis=1)
            a = _np_softmax(q @ kf.transpose(0, 1, 3, 2) * (HEAD_DIM ** -0.5)
                           + causal)
            o = a @ vf
            x = x + o.transpose(0, 2, 1, 3).reshape(1, n, N_HEADS * HEAD_DIM) @ g("o_proj").T
            h = _np_rms(x, g("post_ln"))
            z = h @ g("gate_proj").T
            x = x + (z / (1.0 + np.exp(-z))
                     * (h @ g("up_proj").T)) @ g("down_proj").T
        lg = _np_rms(x, w["final_norm"].astype(np.float64))[0, -1] @ lm[step].T
        logits.append(lg.astype(np.float32))
        tok = int(np.argmax(lg))
        codes.append(tok)
        if step + 1 < n_steps:
            seq.append(emb[step][tok].reshape(HIDDEN))
    return codes, logits


def main():
    ap = argparse.ArgumentParser(description="Stateful (KV-as-I/O) CP ONNX export")
    ap.add_argument("--weights", required=True,
                    help="code_predictor_weights.npz from export_code_predictor_onnx.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--check", type=int, default=0,
                    help="replay N random inputs through both graphs and compare")
    args = ap.parse_args()

    w = np.load(args.weights)
    model = StatefulCodePredictor(w)
    model.eval()

    inv_freq = model.inv_freq
    dummy = {
        "hidden": np.zeros((1, 1, HIDDEN), np.float32),
        "freqs_in": rope_freqs(np.array([1]), inv_freq),
        "sel": np.eye(L_KV, dtype=np.float32)[1],
        "mask": np.where(np.arange(L_KV) <= 1, 0.0, -100.0)
                  .reshape(1, 1, 1, L_KV).astype(np.float32),
    }
    inputs = list(dummy)
    sizes = [list(v.shape) for v in dummy.values()]
    for i in range(N_LAYERS):
        inputs += [f"past_k_{i}", f"past_v_{i}"]
        sizes += [[1, N_KV_HEADS, L_KV, HEAD_DIM]] * 2
    feeds = list(dummy.values()) + [np.zeros((1, N_KV_HEADS, L_KV, HEAD_DIM), np.float32)
                                    for _ in range(2 * N_LAYERS)]
    outputs = ["output"] + sum([[f"new_k_{i}", f"new_v_{i}"] for i in range(N_LAYERS)], [])

    print(f"Exporting {args.output}: {len(inputs)} inputs, {len(outputs)} outputs")
    # dynamo=False: the legacy TorchScript exporter. torch.export's decomposition
    # set emits Reshape/Unsqueeze/Slice patterns that make rknn-toolkit2's
    # graph_optimizer die with `KeyError: 'unsqueeze_8_rs'` in fuse_ops.
    torch.onnx.export(
        model, tuple(torch.from_numpy(f) for f in feeds), args.output,
        input_names=inputs, output_names=outputs, opset_version=18,
        do_constant_folding=True, dynamo=False,
    )
    print(f"  {os.path.getsize(args.output)/1e6:.1f} MB (+ external data)")

    if not args.check:
        return

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count()
    s_state = ort.InferenceSession(args.output, sess_options=so,
                                   providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    agree = np.zeros(N_GROUPS, int)
    max_ld = 0.0
    for n in range(args.check):
        hidden = rng.standard_normal(HIDDEN).astype(np.float32)
        hidden /= np.linalg.norm(hidden)
        c0 = int(rng.integers(0, 2048))
        e0 = w["codec_emb_0"][c0].astype(np.float32)
        cs, ls = run_stateful(s_state, w, hidden, e0)
        cr, lr = run_reference(w, hidden, e0)
        agree += np.array(cs) == np.array(cr)
        ld = float(np.abs(ls[0] - lr[0]).max())
        max_ld = max(max_ld, ld)
        print(f"  [{n+1}/{args.check}] codebooks {sum(np.array(cs)==np.array(cr))}/15"
              f" step-0 logit diff {ld:.2e}        ", end="\r")
    print()
    print("per-codebook agreement:", " ".join(f"{a}/{args.check}" for a in agree))
    print(f"max abs step-0 logit diff: {max_ld:.2e}")
    # Gate on the first five codebooks being exact. The CP's layer-3 attention
    # scores reach ~1300 (measured), so its softmax is a hard argmax and a
    # 1e-6 fp32/fp64 difference flips a near-tie on ~2 of 8 random inputs;
    # from there the 15-step autoregressive cascade amplifies it. The repo's own
    # GGML-vs-Python-FP32 A/B measures the same decay (98/97/.../52 %), so
    # late-codebook disagreement is the model's conditioning, not this graph.
    ok = agree[:5].min() >= args.check and agree.mean() >= 0.6 * args.check
    print("CHECK", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
