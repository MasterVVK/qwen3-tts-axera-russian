#!/usr/bin/env python3
"""
Qwen3-TTS Client — orchestrates talker, code predictor, and vocoder servers.

Connects to all three servers via Unix sockets and runs the full pipeline:
1. Send text to talker -> receive hidden_state + code_0
2. Send hidden_state + code_0 to code predictor -> receive codes 1-15
3. Compute feedback embedding, send back to talker
4. Repeat until EOS
5. Send all codes to vocoder -> receive audio

Usage:
  python3 tts_client.py "Hello, how are you?"
  python3 tts_client.py --text "Привет" --language russian --output output.wav
"""

import os
import sys
import socket
import struct
import json
import argparse
import time
import threading
import numpy as np
import wave as wavmod

HIDDEN_SIZE = 1024
SAMPLE_RATE = 24000
SAMPLES_PER_TOKEN = 1920


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


VOC_CHUNK_SIZE = 64  # new tokens per streaming vocoder chunk
VOC_FIRST_CHUNK = 16 # tokens in the FIRST streaming chunk (TTFB; see below)
VOC_CTX = 48         # tokens of real context prepended per chunk (model = 112)
VOC_SLOTS = 112      # fixed vocoder model input size
# The vocoder protocol (stateful streaming vs chunked seams) is probed against the
# running server in synthesize(), NOT read from this process's environment: a client
# that guesses 'chunked' against a stateful server prepends 48 context tokens, then
# trims and seam-stitches audio the server never chunked — measured 26% of the
# utterance silently dropped (795 tokens -> 47.04 s of audio instead of 63.6 s).


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


SENTINEL_DONE = -1
SENTINEL_ERROR = -2


class Qwen3TTSClient:
    def __init__(self, talker_socket="/tmp/qwen3_talker.sock",
                 cp_socket="/tmp/qwen3_cp.sock",
                 voc_socket="/tmp/qwen3_voc.sock",
                 embeddings_dir=None):
        self.talker_socket = talker_socket
        self.cp_socket = cp_socket
        self.voc_socket = voc_socket
        # vocoder protocol, probed once per utterance in synthesize()
        self.stateful = None
        self.t0 = time.time()

        # Load codec_embedding for feedback computation
        if embeddings_dir:
            self.codec_embedding = np.load(
                os.path.join(embeddings_dir, "codec_embedding.npy")
            )
        else:
            self.codec_embedding = None

        # Load code predictor codec embeddings for feedback
        self.cp_codec_embeddings = None

        # Pre-compute tts_pad_embed for non-streaming feedback
        self.tts_pad_embed = None
        if embeddings_dir:
            text_embedding = np.load(os.path.join(embeddings_dir, "text_embedding.npy"))
            fc1_w = np.load(os.path.join(embeddings_dir, "text_projection_linear_fc1_weight.npy"))
            fc1_b = np.load(os.path.join(embeddings_dir, "text_projection_linear_fc1_bias.npy"))
            fc2_w = np.load(os.path.join(embeddings_dir, "text_projection_linear_fc2_weight.npy"))
            fc2_b = np.load(os.path.join(embeddings_dir, "text_projection_linear_fc2_bias.npy"))
            TTS_PAD_TOKEN_ID = 151671
            raw = text_embedding[TTS_PAD_TOKEN_ID:TTS_PAD_TOKEN_ID+1]  # [1, 2048]
            h = raw @ fc1_w.T + fc1_b
            h = h * (1.0 / (1.0 + np.exp(-h)))  # SiLU
            self.tts_pad_embed = (h @ fc2_w.T + fc2_b).flatten().astype(np.float32)  # [1024]
            del text_embedding, fc1_w, fc1_b, fc2_w, fc2_b  # free memory

    def load_cp_embeddings(self, cp_dir):
        """Load code predictor codec embeddings for feedback sum."""
        w = np.load(os.path.join(cp_dir, "code_predictor_weights.npz"))
        self.cp_codec_embeddings = [
            w[f"codec_emb_{i}"].astype(np.float32) for i in range(15)
        ]

    def _vocoder_reset(self):
        """Stateful vocoder: n_tokens=0 request flushes the server's buffered
        tail (tokens not filling a full model step) and drops the streaming
        state. Returns (answered, audio): a chunked-seam server closes the
        connection on a 0-token request instead of answering it, so `answered`
        doubles as the stateful-mode probe. At an utterance start the buffer is
        empty, so the probe is also the start flush."""
        empty = np.array([], dtype=np.int16)
        try:
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.connect(self.voc_socket)
            c.sendall(struct.pack("<i", 0))
            header = b""
            while len(header) < 4:
                d = c.recv(4 - len(header))
                if not d:
                    c.close()
                    return False, empty
                header += d
            n_samples = struct.unpack("<i", header)[0]
            data = b""
            while len(data) < n_samples * 2:
                d = c.recv(min(65536, n_samples * 2 - len(data)))
                if not d:
                    break
                data += d
            c.close()
            return True, np.frombuffer(data, dtype=np.int16)
        except Exception as e:
            print(f"  Vocoder reset error: {e}")
            return False, empty

    def _vocoder_chunk(self, codes_list, chunk_idx, results, ctx=None):
        """Process one vocoder chunk in background thread (RKNN NPU).

        ctx: last VOC_CTX tokens of the previous chunk, fed before this chunk
        as real causal context (the vocoder model is max_tokens = VOC_CHUNK_SIZE
        + VOC_CTX); its audio is trimmed off the response.
        """
        try:
            if self.stateful:
                ctx = None
                if chunk_idx == 0:
                    self._vocoder_reset()
            codes_array = np.array(codes_list, dtype=np.int64)
            n_ctx = 0
            if ctx:
                codes_array = np.concatenate(
                    [np.array(ctx, dtype=np.int64), codes_array])
                n_ctx = len(ctx)
            n_tok = len(codes_array)

            voc_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            voc_conn.connect(self.voc_socket)
            voc_conn.sendall(struct.pack("<i", n_tok) + codes_array.tobytes())

            # Receive audio
            header = b""
            while len(header) < 4:
                d = voc_conn.recv(4 - len(header))
                if not d:
                    break
                header += d
            n_samples = struct.unpack("<i", header)[0]

            audio_data = b""
            while len(audio_data) < n_samples * 2:
                d = voc_conn.recv(min(65536, n_samples * 2 - len(audio_data)))
                if not d:
                    break
                audio_data += d
            voc_conn.close()

            audio = np.frombuffer(audio_data, dtype=np.int16)
            if n_ctx:
                audio = audio[n_ctx * SAMPLES_PER_TOKEN:]
            results[chunk_idx] = audio
            if chunk_idx == 0:
                # TTFB: the moment the listener would hear anything.
                print(f"  <- First audio: {len(audio) / SAMPLE_RATE:.2f} s at "
                      f"+{time.time() - self.t0:.1f} s")
        except Exception as e:
            print(f"  Vocoder chunk {chunk_idx} error: {e}")
            results[chunk_idx] = np.array([], dtype=np.int16)

    def synthesize(self, text, language="russian", output="output.wav",
                   streaming=False):
        """Run full TTS pipeline via sockets.

        With streaming=True, vocoder chunks are submitted to RKNN NPU
        as soon as VOC_CHUNK_SIZE tokens accumulate, overlapping with
        CPU generation.
        """
        print(f"Text: '{text}'")
        print(f"Language: {language}")

        t_start = time.time()
        self.t0 = t_start

        # Ask the vocoder server which protocol it speaks (stateful streaming or
        # chunked seams) before generating anything. On a stateful server this
        # request is also the utterance-start flush.
        self.stateful = self._vocoder_reset()[0]
        print(f"Vocoder protocol: {'stateful streaming' if self.stateful else 'chunked seams'}")

        # Connect to talker
        talker_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        talker_conn.connect(self.talker_socket)

        # Send text
        msg = json.dumps({"text": text, "language": language}).encode()
        talker_conn.sendall(struct.pack("<I", len(msg)))
        talker_conn.sendall(msg)

        all_codes = []
        pending_codes = []  # Buffer for streaming vocoder
        prev_tail = None   # last VOC_CTX tokens of the previous submitted chunk
        next_start = 0     # token index where the next streaming chunk starts
        last_start = 0     # token index where the last submitted chunk started
        voc_threads = []
        voc_results = {}
        voc_chunk_idx = 0
        n_tokens = 0
        hidden_buf_size = HIDDEN_SIZE * 4
        cp_codes_size = 15 * 4  # 15 int32 = 60 bytes

        # Pre-allocate feedback buffer
        feedback_buf = np.zeros(HIDDEN_SIZE, dtype=np.float32)

        while True:
            # Receive code_0 from talker
            code_data = talker_conn.recv(4)
            if len(code_data) < 4:
                break
            code_0 = struct.unpack("<i", code_data)[0]

            if code_0 == SENTINEL_DONE:
                print(f"  Talker done after {n_tokens} tokens")
                break
            if code_0 == SENTINEL_ERROR:
                print("  Talker error!")
                break

            # Receive hidden_state from talker (one shot)
            hidden_data = b""
            while len(hidden_data) < hidden_buf_size:
                chunk = talker_conn.recv(hidden_buf_size - len(hidden_data))
                if not chunk:
                    break
                hidden_data += chunk

            hidden_state = np.frombuffer(hidden_data, dtype=np.float32).copy()

            # Send to code predictor (new connection per request)
            cp_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            cp_conn.connect(self.cp_socket)
            cp_conn.sendall(hidden_state.tobytes() + struct.pack("<i", code_0))

            # Receive 15 codes as single block (60 bytes)
            cp_data = b""
            while len(cp_data) < cp_codes_size:
                chunk = cp_conn.recv(cp_codes_size - len(cp_data))
                if not chunk:
                    break
                cp_data += chunk
            cp_conn.close()
            codes_1_15 = list(struct.unpack("<15i", cp_data))

            token_codes = [code_0] + codes_1_15
            all_codes.append(token_codes)
            pending_codes.append(token_codes)
            n_tokens += 1

            # Streaming vocoder: submit chunk when buffer full. Chunk k+1
            # starts at the quietest point in the last 1.5 s of chunk k's
            # audio (adaptive seam), so stitch() joins chunks inside a
            # pause instead of mid-phrase.
            # The FIRST streaming chunk is small: it is the audio the listener
            # waits for, and a stateful step costs the same fixed cache traffic
            # (5.2 MB in and out) whatever B is, so voicing 16 tokens early is
            # nearly free. Stateful only — a chunked-seam server pads a short
            # request to its full slot count, and a zero-padded chunk start is a
            # different waveform (the "double voice" defect).
            chunk_n = (VOC_FIRST_CHUNK
                       if (self.stateful and voc_chunk_idx == 0)
                       else VOC_CHUNK_SIZE)
            if streaming and len(all_codes) - next_start >= chunk_n:
                if voc_threads and not self.stateful:
                    voc_threads[-1].join()
                    a = voc_results.get(voc_chunk_idx - 1)
                    if a is not None and len(a) > int(3.0 * SAMPLE_RATE) + 2400:
                        Ws = int(3.0 * SAMPLE_RATE)
                        k = find_pause(a[-Ws:])
                        if k is None:
                            k = len(a)
                        seam = last_start + (len(a) - Ws + k) // SAMPLES_PER_TOKEN
                        seam = max(last_start + 8,
                                   min(seam, last_start + len(a) // SAMPLES_PER_TOKEN - 2))
                        # warmup-truncate: keep 2 tokens past the seam so the
                        # next chunk drops its contextless head transient in
                        # the shared overlap
                        voc_results[voc_chunk_idx - 1] = a[:(seam - last_start + 2) * SAMPLES_PER_TOKEN]
                        next_start = seam
                take = all_codes[next_start:next_start + chunk_n]
                prev_tail = all_codes[max(0, next_start - VOC_CTX):next_start]
                t = threading.Thread(
                    target=self._vocoder_chunk,
                    args=(take, voc_chunk_idx, voc_results, prev_tail))
                t.start()
                voc_threads.append(t)
                print(f"  -> Vocoder chunk {voc_chunk_idx} ({len(take)} tokens @ {next_start}) submitted")
                last_start = next_start
                next_start += chunk_n
                voc_chunk_idx += 1

            # Feedback: code_0=talker embed, codes 1-15=CP per-group embeds
            np.copyto(feedback_buf, self.codec_embedding[code_0])
            if self.cp_codec_embeddings is not None:
                for gi, tok in enumerate(codes_1_15):
                    feedback_buf += self.cp_codec_embeddings[gi][tok]
            else:
                for tok in codes_1_15:
                    feedback_buf += self.codec_embedding[tok]
            if self.tts_pad_embed is not None:
                feedback_buf += self.tts_pad_embed

            # Send feedback to talker
            talker_conn.sendall(feedback_buf.tobytes())

            if n_tokens % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  [{n_tokens}] {n_tokens/elapsed:.1f} tok/s")

        talker_conn.close()

        if n_tokens == 0:
            print("No tokens generated!")
            return

        t_gen = time.time() - t_start
        print(f"\nGenerated {n_tokens} tokens in {t_gen:.1f}s ({n_tokens/t_gen:.1f} tok/s)")

        # Submit remaining tokens to vocoder
        if streaming and voc_threads:
            # Streaming: seam-shift the remainder too, then submit the tail
            voc_threads[-1].join()
            a = voc_results.get(voc_chunk_idx - 1)
            if (not self.stateful and a is not None and next_start < n_tokens
                    and len(a) > int(3.0 * SAMPLE_RATE) + 2400):
                Ws = int(3.0 * SAMPLE_RATE)
                k = find_pause(a[-Ws:])
                if k is None:
                    k = len(a)
                seam = last_start + (len(a) - Ws + k) // SAMPLES_PER_TOKEN
                seam = max(last_start + 8,
                           min(seam, last_start + len(a) // SAMPLES_PER_TOKEN - 2))
                voc_results[voc_chunk_idx - 1] = a[:(seam - last_start + 2) * SAMPLES_PER_TOKEN]
                next_start = seam
            if next_start < n_tokens:
                take = all_codes[next_start:]
                # fill free model slots with real context (tail chunk is
                # short, so it gets more context than a full chunk)
                ctx_n = max(VOC_CTX, VOC_SLOTS - len(take))
                t = threading.Thread(
                    target=self._vocoder_chunk,
                    args=(take, voc_chunk_idx, voc_results,
                          all_codes[max(0, next_start - ctx_n):next_start]))
                t.start()
                voc_threads.append(t)
                print(f"  -> Vocoder chunk {voc_chunk_idx} ({len(take)} tokens @ {next_start}) submitted")
        elif pending_codes:
            # Non-streaming or single chunk: process directly
            t_voc = time.time()
            self._vocoder_chunk(pending_codes, 0, voc_results)
            # consumed slot 0: the stateful tail flush below must land in the
            # next slot, not overwrite this audio
            voc_chunk_idx = 1
            print(f"Vocoder: {time.time() - t_voc:.1f}s ({len(pending_codes)} tokens)")

        # Wait for all vocoder threads
        if voc_threads:
            t_wait = time.time()
            for t in voc_threads:
                t.join()
            print(f"Vocoder wait: {time.time() - t_wait:.1f}s ({len(voc_threads)} chunks)")

        # Stateful: the server buffers tokens to full model steps; the reset
        # flushes and returns the tail (a request boundary mid-step must not
        # be zero-padded — padding writes silence into the caches).
        if self.stateful:
            ok, tail = self._vocoder_reset()
            if len(tail):
                voc_results[voc_chunk_idx] = tail
                voc_chunk_idx += 1

        dump = os.environ.get("TTS_DUMP_CODES")
        if dump:
            np.save(dump, np.array(all_codes, dtype=np.int64))

        # Concatenate audio in order, stitching chunk joins with the same
        # pause/micro-blend rule as the vocoder server
        audio_chunks = []
        for i in range(voc_chunk_idx):
            if i in voc_results and len(voc_results[i]) > 0:
                audio_chunks.append(voc_results[i].astype(np.float32))

        if not audio_chunks:
            print("No audio generated!")
            return

        audio_f32 = audio_chunks[0]
        for b in audio_chunks[1:]:
            if self.stateful:
                audio_f32 = np.concatenate([audio_f32, b])
            else:
                audio_f32 = stitch(audio_f32, b,
                                   overlap=2 * SAMPLES_PER_TOKEN)
        audio_int16 = np.clip(audio_f32, -32768, 32767).astype(np.int16)

        # Save WAV
        with wavmod.open(output, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        audio_dur = len(audio_int16) / SAMPLE_RATE
        total = time.time() - t_start
        print(f"\nAudio: {audio_dur:.2f}s, saved to {output}")
        print(f"Total: {total:.1f}s (RTF={total/audio_dur:.1f}x)")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Client")
    parser.add_argument("text", nargs="?", default=None)
    parser.add_argument("--text", dest="text_flag", default=None)
    parser.add_argument("--language", default="russian")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--talker_socket", default="/tmp/qwen3_talker.sock")
    parser.add_argument("--cp_socket", default="/tmp/qwen3_cp.sock")
    parser.add_argument("--voc_socket", default="/tmp/qwen3_voc.sock")
    parser.add_argument("--embeddings_dir", default="/root/tts-rknn/embeddings")
    parser.add_argument("--cp_dir", default="/root/tts-rknn/code_predictor")
    parser.add_argument("--streaming", action="store_true",
                        help="Enable streaming vocoder (overlap with generation)")
    args = parser.parse_args()

    text = args.text or args.text_flag
    if not text:
        text = "Привет, как дела? Сегодня хорошая погода для прогулки."

    client = Qwen3TTSClient(
        talker_socket=args.talker_socket,
        cp_socket=args.cp_socket,
        voc_socket=args.voc_socket,
        embeddings_dir=args.embeddings_dir,
    )
    client.load_cp_embeddings(args.cp_dir)
    client.synthesize(text, args.language, args.output, streaming=args.streaming)


if __name__ == "__main__":
    main()
