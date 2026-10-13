#!/usr/bin/env python3
"""
Vocoder Server for Qwen3-TTS.

Converts codec tokens [n_tokens, 16] to audio using RKNN NPU or ONNX CPU.
Processes in chunks for long sequences.

Protocol (Unix socket):
  Client -> Server: [4 bytes: n_tokens int32]
                    [n_tokens * 16 * 8 bytes: codes int64]
  Server -> Client: [4 bytes: n_samples int32]
                    [n_samples * 2 bytes: audio int16]

Usage on CM3588:
  python3 vocoder_server.py \
      --model /root/tts-rknn/vocoder_traced_64_sim_q8.rknn \
      --socket /tmp/qwen3_voc.sock
"""

import os
import re
import sys
import socket
import struct
import argparse
import time
import signal
import numpy as np

SAMPLE_RATE = 24000
SAMPLES_PER_TOKEN = 1920
# flat conv/KV cache size of the stateful export; must match
# scripts/export_vocoder_stateful.py StatefulDecoder.cache_numel
STATEFUL_CACHE_NUMEL = 1303424
STATEFUL_HEAD_DIM = 64
STATEFUL_KV_LEN = 71


def find_pause(x, min_len=3600, win=480):
    """Center index of the longest run of 20 ms-RMS below 25% of the window
    RMS, or None. A plain argmin finds stop-consonant closures (0 ms wide)
    and the seam then eats a consonant; requiring a long quiet run finds a
    real pause."""
    if len(x) < 2 * win:
        return None
    r = np.sqrt(np.convolve(np.square(x), np.ones(win, dtype=np.float32) / win,
                            mode='same'))
    thr = 0.25 * float(np.sqrt(np.mean(np.square(x)))) + 1e-9
    quiet = (r < thr).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet, [0]))))
    runs = edges[1::2] - edges[::2]
    if len(runs) == 0:
        return None
    i = int(np.argmax(runs))
    if runs[i] < min_len:
        return None
    return int(edges[i] + runs[i] // 2)


def stitch(a, b, Wmax=SAMPLE_RATE, K=480, Fd=960, Fm=480, G=2400,
           overlap=0):
    """Join two audio segments. `overlap` samples of b repeat the tail of a
    (warmup-truncate: the next chunk starts a few tokens before a's audio
    ends, so the contextless conv transient at b's head is inside the
    shared region and gets dropped/blended away).

    The segments are temporally adjacent (no shared content), so a long
    crossfade beats (waveforms do not align — context conditioning leaves
    ~19 dB mismatch at the join) and a fixed fade-drop is audible when it
    eats speech. Strategy:
      1. Search a real pause (minimum 20 ms-RMS point) within 1 s of the
         seam in each segment. If what would be dropped there is quiet
         (<25% of the local RMS), fade-drop-stitch inside the pause.
      2. Otherwise (seam mid-phrase) do a 20 ms equal-power micro-blend
         centered on the quietest sample within +-50 ms of the seam:
         amplitude-continuous, energy-preserving (linear windows dip in the
         middle = audible crackle), no samples dropped.
    """
    W = min(Wmax, len(a) // 2, len(b) // 2)
    if W < K + Fd:
        return np.concatenate([a, b])

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0

    def quietest(x):
        e = np.convolve(x * x, np.ones(K, dtype=np.float32) / K, mode='same')
        return int(np.argmax(e))

    m = len(a) - W + quietest(a[-W:])
    n = quietest(b[:W])
    drop = np.concatenate([a[m:], b[:n]])
    local = max(rms(a[-W:]), rms(b[:W]), 1e-9)
    if rms(drop) < 0.25 * local:
        t = np.linspace(0.0, 1.0, Fd, dtype=np.float32)
        fo = np.cos(t * np.pi / 2)
        return np.concatenate([
            a[:m], a[m:m + Fd] * fo, b[n:n + Fd] * (1.0 - fo), b[n + Fd:]])

    L = len(a)
    bo = b[overlap:] if len(b) > overlap + 2 * Fm else b
    out = np.concatenate([a, bo])
    c0 = max(Fm, L - G // 2)
    c1 = min(len(out) - Fm, L + G // 2)
    seg = out[c0 - Fm:c1 + Fm]
    e = np.convolve(seg * seg, np.ones(120, dtype=np.float32) / 120,
                    mode='same')
    c = min(max(c0 - Fm + int(np.argmax(e)), L - Fm + 10), L + Fm - 10)
    t = np.linspace(0.0, 1.0, 2 * Fm, dtype=np.float32)
    w = np.cos(t * np.pi / 2)
    ae = np.zeros(2 * Fm, dtype=np.float32)
    be = np.zeros(2 * Fm, dtype=np.float32)
    a1 = min(len(a), c + Fm)
    ae[:a1 - (c - Fm)] = a[c - Fm:a1]
    b0 = max(0, c - L - Fm)
    b1 = min(len(bo), c - L + Fm)
    k0 = 2 * Fm - (b1 - b0)
    be[k0:k0 + (b1 - b0)] = bo[b0:b1]
    out[c - Fm:c + Fm] = ae * w + be * (1.0 - w)
    return out


class VocoderServer:
    def __init__(self, model_path, socket_path="/tmp/qwen3_voc.sock",
                 first_model_path=None):
        self.socket_path = socket_path
        self.is_onnx = model_path.endswith('.onnx')
        self.stateful = 'stateful' in os.path.basename(model_path)

        if self.is_onnx:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(model_path, sess_options=opts,
                                              providers=['CPUExecutionProvider'])
            inp_shape = self.sess.get_inputs()[0].shape
            self.max_tokens = inp_shape[1]
            print(f"Vocoder: ONNX CPU (4 threads), max_tokens={self.max_tokens}")
        else:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(model_path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN vocoder: {ret}")
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                raise RuntimeError(f"Failed to init RKNN vocoder: {ret}")
            m = re.search(r"_(\d+)", os.path.basename(model_path))
            self.max_tokens = int(m.group(1)) if m else 256
            print(f"Vocoder: RKNN NPU, max_tokens={self.max_tokens}")

        if self.stateful:
            # streaming state: conv caches + sliding-window KV, carried across
            # requests of one utterance; reset (n_tokens=0) flushes the
            # leftover < max_tokens tokens and starts a new utterance
            self.cache = np.zeros(STATEFUL_CACHE_NUMEL, dtype=np.float32)
            self.pos = 0   # absolute token index of the next step
            # tokens not yet voiced (the model steps in fixed max_tokens
            # groups; a request boundary in the middle of a group must NOT
            # zero-pad — padding poisons the caches — it must buffer)
            self.buf = np.zeros((0, 16), dtype=np.int64)
            # RoPE angles are host-side model inputs (cos/sin computed
            # in-graph: cos/sin inputs explode on the NPU; the attention mask
            # is in-graph too); inv_freq from rope_theta
            self.inv_freq = np.exp2(
                -np.log2(10000.0)
                * np.arange(STATEFUL_HEAD_DIM // 2, dtype=np.float32)
                / (STATEFUL_HEAD_DIM // 2))
            self.kv_len = STATEFUL_KV_LEN
            print("Vocoder: stateful streaming mode (no chunk seams)")

            # First-step model (TTFB): the cache layout is B-independent (conv
            # tails + a fixed 71-key KV window), so a small-B export can voice
            # the first group and hand its cache straight to the big one. The
            # first chunk is what the listener waits for: 64 tokens = 5.12 s of
            # audio at ~11 tok/s, 16 tokens = 1.28 s.
            self.first = None
            if first_model_path and self.stateful and not self.is_onnx:
                net = RKNNLite()
                if net.load_rknn(first_model_path) != 0:
                    raise RuntimeError(
                        f"Failed to load first-step RKNN vocoder: {first_model_path}")
                if net.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
                    raise RuntimeError("Failed to init first-step RKNN vocoder")
                m = re.search(r"_(\d+)", os.path.basename(first_model_path))
                self.first = (net, int(m.group(1)) if m else 16)
                print(f"Vocoder: first step uses {self.first[1]} tokens "
                      f"({self.first[1] * 1920 / 24000:.2f} s of audio)")

        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._running = False

    def _inference_chunk(self, padded):
        if self.is_onnx:
            return self.sess.run(None, {'audio_codes': padded})[0].flatten()
        else:
            return self.rknn.inference(inputs=[padded])[0].flatten()

    def _inference_stateful(self, padded, pos, net=None):
        L = padded.shape[2]
        p = pos + np.arange(L, dtype=np.float32)
        # RoPE angles reduced mod 2*pi (fp32 trig on host; the graph computes
        # cos/sin — feeding cos/sin as RKNN inputs explodes the NPU fp16
        # pipeline, and raw angles > ~1e3 lose fp16 precision: ulp(1000)=0.5).
        # The sliding-window mask is in-graph (constant window +
        # cache-validity bias), no mask input.
        freqs = np.remainder(np.outer(p, self.inv_freq),
                             2 * np.pi).astype(np.float32)
        # how many cache slots hold real keys (the rolling window fills from the
        # right); the graph masks the rest. A boolean 'any history' here leaks
        # Lc - B never-written zero keys into the softmax: measured stepwise SNR
        # 48.8 dB at B=64 and 10.2 dB at B=16, vs 111.4 / 103.7 dB with the count.
        cv = np.full(1, min(pos, self.kv_len), dtype=np.float32)
        if self.is_onnx:
            w, c = self.sess.run(None, {
                'audio_codes': padded, 'freqs_in': freqs,
                'cache_valid': cv, 'cache_in': self.cache})
        else:
            # w16a16 build stores fp32 inputs as fp16
            w, c = (net or self.rknn).inference(
                inputs=[padded, freqs.astype(np.float16),
                        cv.astype(np.float16),
                        self.cache.astype(np.float16)])
        self.cache = c
        return w.flatten()

    def synthesize(self, codes_array):
        n_tokens = len(codes_array)

        # Stateful streaming: the whole decoder is one continuous causal run —
        # conv caches and sliding-window KV live in the cache tensor across
        # requests. Requests are buffered so every model step carries exactly
        # max_tokens REAL tokens (zero-padding mid-utterance would write
        # silence into the caches); the leftover tail is voiced by the reset
        # request. No chunk boundaries at all.
        if self.stateful:
            self.buf = np.concatenate([self.buf, codes_array])
            out = []
            while True:
                # The first group of an utterance runs on the small model: it is
                # the audio the listener waits for, and a step costs the same
                # fixed cache traffic (5.2 MB in and out) whatever B is, so
                # voicing 16 tokens early is nearly free.
                n = (self.first[1] if self.first and self.pos == 0
                     else self.max_tokens)
                if len(self.buf) < n:
                    break
                seg = self.buf[:n]
                self.buf = self.buf[n:]
                padded = np.zeros((1, 16, n), dtype=np.int64)
                padded[0] = seg[:, :16].T
                out.append(self._inference_stateful(
                    padded, self.pos,
                    self.first[0] if n != self.max_tokens else None))
                self.pos += n
            return (np.concatenate(out) if out
                    else np.array([], dtype=np.float32))

        # Single chunk — no overlap needed
        if n_tokens <= self.max_tokens:
            padded = np.zeros((1, self.max_tokens, 16), dtype=np.int64)
            padded[0, :n_tokens, :] = codes_array[:, :16]
            audio = self._inference_chunk(padded)
            return audio[:n_tokens * SAMPLES_PER_TOKEN]

        # Multi-chunk. The WavTokenizer decoder is causal, so the same tokens
        # voiced at a chunk start with zero padding are a *different waveform*
        # than voiced mid-sequence with real context (measured: corr 0.445,
        # SNR 0.3 dB). The old zero-pad 16-token overlap crossfade therefore
        # blended two unaligned signals — audible "double voice" at every
        # boundary. Feed the previous chunk's tail as real context instead.
        # Boundary SNR (first 0.1 s of the new region vs single full run,
        # RKNN FP16): CTX=16 -> 4.1 dB (audible), CTX=48 -> 18.6 dB,
        # CTX=60 -> 20.2 dB (saturating).
        #
        # Even at 19 dB the two waveforms are not phase-aligned, so *any*
        # sum at a mid-phrase seam combs (audible crackle) and any drop of
        # speech is audible as a dropout. So move the seam to a pause:
        # after each chunk, find its quietest 100 ms window in the last 1.5 s
        # and start the next chunk there; stitch() then drops only silence.
        CTX = max(16, self.max_tokens - 64)  # tokens of real context per chunk
        N_NEW = self.max_tokens - CTX        # new tokens voiced per chunk
        result = np.array([], dtype=np.float32)
        pos = 0
        while pos < n_tokens:
            take = codes_array[pos:pos + N_NEW]
            # fill every slot the chunk does not need with real context: a
            # short (tail) chunk gets 94 tokens of context instead of 48,
            # halving the conditioning deficit at the last seam
            ctx = codes_array[max(0, pos - (self.max_tokens - len(take))):pos]
            seg = np.concatenate([ctx, take])
            padded = np.zeros((1, self.max_tokens, 16), dtype=np.int64)
            padded[0, :len(seg), :] = seg[:, :16]
            audio = self._inference_chunk(padded)[
                len(ctx) * SAMPLES_PER_TOKEN:len(seg) * SAMPLES_PER_TOKEN].copy()
            seam = pos + len(take)
            if pos + len(take) < n_tokens:
                Ws = int(3.0 * SAMPLE_RATE)
                if len(audio) > Ws + 2400:
                    k = find_pause(audio[-Ws:])
                    if k is not None:
                        seam = pos + len(take) - (Ws - k) // SAMPLES_PER_TOKEN
                        seam = max(pos + 8, min(seam, pos + len(take) - 2))
                # warmup-truncate: keep 2 tokens past the seam (this chunk
                # decodes them mid-region, clean) so the next chunk can
                # drop its contextless head transient inside the overlap
                audio = audio[:(seam - pos) * SAMPLES_PER_TOKEN
                              + 2 * SAMPLES_PER_TOKEN]
            result = (stitch(result, audio, overlap=2 * SAMPLES_PER_TOKEN)
                      if len(result) else audio)
            pos = seam if seam > pos else n_tokens

        return result

    def serve(self):
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(1)
        sock.settimeout(1.0)
        os.chmod(self.socket_path, 0o666)

        print(f"\nVocoder Server listening on {self.socket_path}")

        while self._running:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue

            try:
                # Read n_tokens
                header = conn.recv(4)
                if len(header) < 4:
                    conn.close()
                    continue
                n_tokens = struct.unpack("<i", header)[0]

                if n_tokens == 0 and self.stateful:
                    # utterance boundary: voice the buffered tail, then drop
                    # streaming state
                    tail = []
                    if len(self.buf):
                        seg = self.buf
                        self.buf = np.zeros((0, 16), dtype=np.int64)
                        padded = np.zeros((1, 16, self.max_tokens),
                                          dtype=np.int64)
                        padded[0, :, :len(seg)] = seg[:, :16].T
                        tail.append(self._inference_stateful(
                            padded, self.pos)[:len(seg) * SAMPLES_PER_TOKEN])
                    audio = (np.concatenate(tail) if tail
                             else np.array([], dtype=np.float32))
                    self.cache = np.zeros(STATEFUL_CACHE_NUMEL,
                                          dtype=np.float32)
                    self.pos = 0
                    audio_int16 = np.clip(audio * 32767, -32768,
                                          32767).astype(np.int16)
                    conn.sendall(struct.pack("<i", len(audio_int16)))
                    conn.sendall(audio_int16.tobytes())
                    conn.close()
                    continue

                if n_tokens <= 0 or n_tokens > 10000:
                    conn.close()
                    continue

                # Read codes [n_tokens, 16] int64
                data_size = n_tokens * 16 * 8
                codes_data = b""
                while len(codes_data) < data_size:
                    chunk = conn.recv(min(65536, data_size - len(codes_data)))
                    if not chunk:
                        break
                    codes_data += chunk

                if len(codes_data) < data_size:
                    conn.close()
                    continue

                codes_array = np.frombuffer(codes_data, dtype=np.int64).reshape(n_tokens, 16)

                # Synthesize
                t0 = time.time()
                audio = self.synthesize(codes_array)
                dt = time.time() - t0
                print(f"  Vocoder: {n_tokens} tokens -> {len(audio)} samples ({dt:.2f}s)")

                # Send audio as int16
                audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                n_samples = len(audio_int16)
                conn.sendall(struct.pack("<i", n_samples))
                conn.sendall(audio_int16.tobytes())

            except Exception as e:
                print(f"  Vocoder Error: {e}")
                if self.stateful:
                    # a half-processed request must not leak buffered tokens or
                    # caches into the next utterance: its start flush would
                    # voice the stale tokens (and the client discards that
                    # audio), and the new utterance would inherit their history
                    self.buf = np.zeros((0, 16), dtype=np.int64)
                    self.cache = np.zeros(STATEFUL_CACHE_NUMEL, dtype=np.float32)
                    self.pos = 0
            finally:
                conn.close()

        sock.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        if not self.is_onnx:
            self.rknn.release()
            if self.first:
                self.first[0].release()
        print("Vocoder Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Vocoder Server")
    parser.add_argument("--model", required=True, help="Vocoder model (.rknn or .onnx)")
    parser.add_argument("--socket", default="/tmp/qwen3_voc.sock")
    parser.add_argument("--model_first", default=None,
                        help="Stateful model with a small B for the first group of "
                             "each utterance (TTFB); same cache layout, any B")
    args = parser.parse_args()

    server = VocoderServer(model_path=args.model, socket_path=args.socket,
                           first_model_path=args.model_first)
    server.serve()


if __name__ == "__main__":
    main()
