#!/usr/bin/env python3
"""
Export the Qwen3-TTS vocoder (WavTokenizer v2 decoder) as a STATEFUL streaming
model: fixed-shape step graph with explicit caches as inputs/outputs, so chunks
are decoded from continuous state — no chunk seams at all.

The decoder is fully causal:
  quantizer lookup (pointwise) -> pre_conv (causal conv k=3)
  -> 8x sliding-window attention (window 72, causal) -> KV cache = last 71 keys
  -> upsample convT(k=2,s=2) + ConvNeXt (causal dwconv k=7)
  -> conv k=7 -> 4 DecoderBlocks: convT(k=2r,s=r) + 3 residual units
     (causal conv k=7 d=1/3/9 -> caches 6/18/54)
  -> SnakeBeta + conv k=7.
Every causal conv's history is (k-1)*d input samples; every causal convT's is
ceil((k-1)/s) input samples; attention's is 71 (K,V) pairs. All caches are
packed into ONE flat float32 tensor (input and output), so the ONNX graph has
4 inputs (codes, RoPE freqs, cache_valid, cache) / 2 outputs and RKNN sees fixed
shapes everywhere.

Usage:
  python3 scripts/export_vocoder_stateful.py --codes-length 64 \
      --output models/vocoder_stateful_64.onnx
"""

import argparse
import math
import os

import numpy as np
import torch

MODEL_DIR = os.environ.get(
    "QWEN3_TTS_MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/"
        "snapshots/c27fe8aa05b732b1376d0f6a1e522fbccb84abbd"
    ),
)
TOKENIZER_DIR = os.path.join(MODEL_DIR, "speech_tokenizer")


def conv_cache_len(kernel_size, dilation):
    return (kernel_size - 1) * dilation


def convT_cache_len(kernel_size, stride):
    return math.ceil((kernel_size - 1) / stride)


class StatefulDecoder(torch.nn.Module):
    """One streaming step: B tokens + flat cache in -> B*1920 samples + cache out.

    forward(codes [1,16,B] int64, freqs [B, head_dim/2] float32,
            cache float32 [N])
      -> (wav [1, B*1920], cache_out float32 [N])
    freqs is pos*inv_freq reduced mod 2*pi on the host (fp32 there; the
    reduced angles stay small so the fp16 NPU build keeps cos/sin accurate);
    RoPE cos/sin are computed in-graph from freqs.
    """

    def __init__(self, dec, B: int):
        super().__init__()
        self.dec = dec
        self.B = B
        cfg = dec.config
        self.Lc = cfg.sliding_window - 1          # 71 keys of KV history
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.scaling = self.head_dim ** -0.5

        specs = []
        specs.append(("pre_conv", conv_cache_len(3, 1) * cfg.codebook_dim))
        specs.append(("kv", self.n_layers * 2 * self.n_heads * self.Lc
                      * self.head_dim))
        for bi in range(len(cfg.upsampling_ratios)):
            specs.append((f"up_convT{bi}", convT_cache_len(2, 2) * cfg.latent_dim))
            specs.append((f"up_dw{bi}", conv_cache_len(7, 1) * cfg.latent_dim))
        specs.append(("dec0", conv_cache_len(7, 1) * cfg.latent_dim))
        for i, r in enumerate(cfg.upsample_rates):
            in_dim = cfg.decoder_dim // 2 ** i
            out_dim = cfg.decoder_dim // 2 ** (i + 1)
            specs.append((f"b{i}_convT", convT_cache_len(2 * r, r) * in_dim))
            for d in (1, 3, 9):
                specs.append((f"b{i}_r{d}", conv_cache_len(7, d) * out_dim))
        out_dim = cfg.decoder_dim // 2 ** len(cfg.upsample_rates)
        specs.append(("final", conv_cache_len(7, 1) * out_dim))
        self.specs = specs
        self.cache_numel = sum(n for _, n in specs)
        # kv history joins the step's k/v through constant-selection matmuls,
        # not Concat: k_full = A @ cache_k + P @ k with A = [I_Lc; 0]
        # [Lc+B, Lc], P = [0; I_B] [Lc+B, B]. The RKNN NPU Concat path
        # corrupts the sliding-window attention (measured: step-0 audio SNR
        # 9.5 dB with kv zeroed and rope/mask off, vs 56.8 dB for the
        # concat-free traced graph); matmul+add are first-class NPU ops.
        a = torch.zeros(self.Lc + B, self.Lc)
        a[:self.Lc, :] = torch.eye(self.Lc)
        p = torch.zeros(self.Lc + B, B)
        p[self.Lc:, :] = torch.eye(B)
        self.register_buffer("kv_sel_hist", a)
        self.register_buffer("kv_sel_cur", p)
        # Sliding-window causal mask, RELATIVE grid (depends only on qp-kp), as
        # a graph constant at -100 (exp(-100) = 0 exactly in fp16; -3e4 leaks
        # through the NPU softmax). The training mask's (kp>=0) guard — "no
        # history before token 0" — cannot be a relative constant; feeding an
        # absolute mask as an input together with in-graph Cos/Sin segfaults
        # the board runtime, so the cache slots carry a per-slot bias built in
        # forward() from the host-supplied count of real slots (see forward).
        qp = torch.arange(B, dtype=torch.float32).view(B, 1)
        kp = (torch.arange(self.Lc + B, dtype=torch.float32)
              - self.Lc).view(1, self.Lc + B)
        m = torch.where((kp <= qp) & (kp > qp - self.Lc - 1),
                        torch.zeros(()), torch.full((), -100.0))
        self.register_buffer("attn_mask", m.view(1, 1, B, self.Lc + B))
        # Slot index of the Lc history slots, for the per-slot validity bias
        # built in forward() from the host-supplied count of real slots.
        self.register_buffer("slot_idx",
                             torch.arange(self.Lc, dtype=torch.float32)
                             .view(1, 1, 1, self.Lc))

    def _slot_bias(self, n_valid):
        """Additive attention bias for the Lc cache slots + the B current ones:
        -100 on the Lc - n_valid never-written slots (exp(-100) = 0 exactly in
        fp16), 0 on the real ones. Elementwise on a [1,1,1,Lc] grid."""
        phantom = torch.clamp(self.Lc - n_valid - self.slot_idx, 0.0, 1.0)
        bias = -100.0 * phantom                   # [1,1,1,Lc] in {-100, 0}
        cur = torch.zeros(1, 1, 1, self.B)
        return torch.cat([bias, cur], dim=-1)     # [1,1,1,Lc+B]

    def _split(self, flat):
        out, off = {}, 0
        for name, n in self.specs:
            out[name] = flat[off:off + n]
            off += n
        return out

    def _conv_step(self, cnet, x, cache):
        """CausalConvNet step: prepend cache, conv, keep new tail cache."""
        cl = x.shape[1]
        n = conv_cache_len(cnet.conv.kernel_size[0], cnet.conv.dilation[0])
        cs = cache.reshape(1, cl, n) if n else cache.reshape(1, cl, 0)
        xin = torch.cat([cs, x], dim=-1)
        y = cnet.conv(xin)
        return y, xin[..., -n:].reshape(-1) if n else x[..., :0].reshape(-1)

    def _convT_step(self, tnet, x, cache):
        """CausalTransConvNet step: prepend cache, convT, causal-crop, tail."""
        cl, L = x.shape[1], x.shape[2]
        n = convT_cache_len(tnet.conv.kernel_size[0], tnet.conv.stride[0])
        cs = cache.reshape(1, cl, n) if n else cache.reshape(1, cl, 0)
        xin = torch.cat([cs, x], dim=-1)
        s = tnet.conv.stride[0]
        y = tnet.conv(xin)
        total = xin.shape[-1]
        y = y[..., :total * s]
        y = y[..., (total - L) * s:]
        return y, x[..., -n:].reshape(-1) if n else x[..., :0].reshape(-1)

    @staticmethod
    def _rot_half(t):
        d = t.shape[-1]
        a, b = t[..., :d // 2], t[..., d // 2:]
        return torch.cat([-b, a], dim=-1)

    def forward(self, codes, freqs, cache_valid, flat_cache):
        """freqs [B, head_dim/2] = RoPE angles (pos*inv_freq) reduced mod 2*pi
        on the host (cos/sin computed in-graph; feeding cos/sin as RKNN inputs
        explodes the NPU fp16 pipeline, and raw angles > ~1e3 lose fp16
        precision).
        cache_valid [1] = how many of the Lc cache slots hold REAL keys, i.e.
        min(pos, Lc) on the host: 0 on the first step, Lc once the window has
        wrapped. The cache is a rolling window, so the real slots are always the
        LAST cache_valid ones; the never-written ones are all-zero keys, whose
        score is 0 (not -inf), so they would steal softmax mass. A boolean
        'any history' flag instead of this count is only true at step 0, so every
        later step leaks Lc - B phantom keys into the softmax: measured stepwise
        SNR 10.2 dB at B=16 (40 of 87 keys phantom) against 48.8 dB at B=64
        (7 of 79). The bias is built with Sub/Clip/Mul on a [Lc] constant grid —
        no comparison/Where, and no reduction over the 1.3 MB cache tensor,
        whose Abs/ReduceSum/Clip version costs 30 dB of NPU fp16 accuracy."""
        dec = self.dec
        B = self.B
        c = self._split(flat_cache)
        parts = []

        hidden = dec.quantizer.decode(codes)          # [1, 512, B]
        hidden, cache = self._conv_step(dec.pre_conv, hidden, c["pre_conv"])
        parts.append(cache)
        hidden = hidden.transpose(1, 2)               # [1, B, 512]

        # ---- pre_transformer: 8 sliding-window layers, manual KV cache ----
        t = dec.pre_transformer
        hidden = t.input_proj(hidden)
        emb = torch.cat([freqs, freqs], dim=-1)       # [B, head_dim]
        cos = emb.cos()
        sin = emb.sin()

        kv = c["kv"].reshape(self.n_layers, 2, self.n_heads, self.Lc,
                             self.head_dim)
        for i, layer in enumerate(t.layers):
            res = hidden
            hs = layer.input_layernorm(hidden)[0]   # [B, 512]
            a = layer.self_attn
            q = a.q_proj(hs).view(B, self.n_heads, self.head_dim).transpose(0, 1)
            k = a.k_proj(hs).view(B, self.n_heads, self.head_dim).transpose(0, 1)
            v = a.v_proj(hs).view(B, self.n_heads, self.head_dim).transpose(0, 1)
            q = q * cos + self._rot_half(q) * sin
            k = k * cos + self._rot_half(k) * sin
            k_full = (torch.matmul(self.kv_sel_hist, kv[i, 0])
                      + torch.matmul(self.kv_sel_cur, k))      # [H,Lc+B,D]
            v_full = (torch.matmul(self.kv_sel_hist, kv[i, 1])
                      + torch.matmul(self.kv_sel_cur, v))
            parts.append(k_full[:, -self.Lc:].reshape(-1))
            parts.append(v_full[:, -self.Lc:].reshape(-1))
            att = torch.softmax(
                q[None] @ k_full[None].transpose(2, 3) * self.scaling
                + self.attn_mask + self._slot_bias(cache_valid), dim=-1)
            o = (att @ v_full[None]).transpose(1, 2).reshape(1, B, -1)
            hidden = res + layer.self_attn_layer_scale(a.o_proj(o))
            hidden = hidden + layer.mlp_layer_scale(
                layer.mlp(layer.post_attention_layernorm(hidden)))
        hidden = t.norm(hidden)
        hidden = t.output_proj(hidden).transpose(1, 2)      # [1, 1024, B]

        # ---- upsample: convT + ConvNeXt blocks ----
        for bi, blocks in enumerate(dec.upsample):
            hidden, cache = self._convT_step(blocks[0], hidden, c[f"up_convT{bi}"])
            parts.append(cache)
            nb = blocks[1]
            res = hidden
            hidden, cache = self._conv_step(nb.dwconv, hidden, c[f"up_dw{bi}"])
            parts.append(cache)
            hs = hidden.permute(0, 2, 1)
            hs = nb.pwconv2(nb.act(nb.pwconv1(nb.norm(hs))))
            hidden = res + (nb.gamma * hs).permute(0, 2, 1)

        # ---- decoder: conv + 4 blocks + final ----
        hidden, cache = self._conv_step(dec.decoder[0], hidden, c["dec0"])
        parts.append(cache)
        for bi in range(len(dec.config.upsample_rates)):
            blk = dec.decoder[1 + bi]
            hidden = blk.block[0](hidden)             # SnakeBeta
            hidden, cache = self._convT_step(blk.block[1], hidden,
                                             c[f"b{bi}_convT"])
            parts.append(cache)
            for unit, d in zip(blk.block[2:], (1, 3, 9)):
                res = hidden
                hidden = unit.act1(hidden)
                hidden, cache = self._conv_step(unit.conv1, hidden,
                                                c[f"b{bi}_r{d}"])
                parts.append(cache)
                hidden = unit.act2(hidden)
                hidden = unit.conv2(hidden) + res
        hidden = dec.decoder[-2](hidden)              # SnakeBeta
        hidden, cache = self._conv_step(dec.decoder[-1], hidden, c["final"])
        parts.append(cache)

        wav = hidden.clamp(min=-1, max=1).reshape(-1)
        return wav, torch.cat([p.reshape(-1) for p in parts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes-length", type=int, default=64)
    ap.add_argument("--output", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )
    full = Qwen3TTSTokenizerV2Model.from_pretrained(TOKENIZER_DIR)
    dec = full.decoder.eval()
    dec.config._attn_implementation = "eager"

    B = args.codes_length
    sdec = StatefulDecoder(dec, B).eval()
    print(f"cache numel: {sdec.cache_numel} "
          f"({sdec.cache_numel * 4 / 1e6:.1f} MB fp32)")

    torch.manual_seed(0)
    n = 3 * B
    codes = torch.randint(0, 2048, (1, 16, n), dtype=torch.int64)

    inv_freq = dec.pre_transformer.rotary_emb.inv_freq.float()

    def rope(i, L):
        # host-side RoPE angles, reduced mod 2*pi (cos/sin computed in-graph;
        # fp32 trig on the host, reduced angles <= 2*pi keep the fp16 NPU
        # build accurate — raw pos ~1e3 lands on fp16 ulp 0.5)
        pos = torch.arange(i, i + L, dtype=torch.float32)
        return torch.remainder(torch.outer(pos, inv_freq), 2 * math.pi)

    with torch.no_grad():
        ref = dec(codes).reshape(-1)
        flat = torch.zeros(sdec.cache_numel, dtype=torch.float32)
        outs = []
        for i in range(0, n, B):
            L = min(B, n - i)
            wav, flat = sdec(codes[:, :, i:i + L], rope(i, L),
                             np.float32(min(i, sdec.Lc)), flat)
            outs.append(wav)
        step = torch.cat(outs)

    L = min(len(ref), len(step))
    snr = 10 * torch.log10(ref[:L].pow(2).sum()
                            / ((ref[:L] - step[:L]).pow(2).sum() + 1e-12))
    print(f"stepwise vs full: len {len(step)}/{len(ref)} SNR {snr:.1f} dB")
    if snr < 40:  # fp32 self-check: semantic errors land at <20 dB; the
                  # kv-selection matmuls + 135-wide mask broadcast add fp32
                  # accumulation noise (measured 48.8 dB, max diff 2e-5)
        d = (ref[:L] - step[:L]).abs()
        print("max diff by 10k-sample blocks:",
              [round(float(d[i:i + 10000].max()), 5) for i in range(0, L, 10000)][:12])
        raise SystemExit("stateful decode does not match full decode")
    if args.check_only:
        return

    out = args.output or f"vocoder_stateful_{B}.onnx"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    dummy_c = torch.randint(0, 2048, (1, 16, B), dtype=torch.int64)
    f0 = rope(0, B)
    dummy_cache = torch.zeros(sdec.cache_numel, dtype=torch.float32)
    print(f"exporting {out} ...")
    # cache_valid is traced as a dynamic input; trace it at a mid-range value so
    # no branch of the phantom-slot clamp degenerates into a constant.
    torch.onnx.export(
        sdec, (dummy_c, f0, torch.full((1,), float(B)), dummy_cache), out,
        input_names=["audio_codes", "freqs_in", "cache_valid", "cache_in"],
        output_names=["audio_values", "cache_out"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(out, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    flat = np.zeros(sdec.cache_numel, dtype=np.float32)
    outs = []
    for i in range(0, n, B):
        L = min(B, n - i)
        w, flat = sess.run(None, {
            "audio_codes": codes[:, :, i:i + L].numpy(),
            "freqs_in": rope(i, L).numpy(),
            "cache_valid": np.full(1, min(i, sdec.Lc), dtype=np.float32),
            "cache_in": flat})
        outs.append(w.reshape(-1)[:L * 1920])
    onnx_step = np.concatenate(outs)
    snr = 10 * np.log10(np.sum(ref[:L].numpy() ** 2)
                        / np.sum((ref[:L].numpy() - onnx_step[:L]) ** 2) + 1e-12)
    print(f"onnx stepwise vs full: SNR {snr:.1f} dB")


if __name__ == "__main__":
    main()
