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
VOC_CHUNK_SIZE = 64  # Must match RKNN model's fixed input size
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

    def _vocoder_chunk(self, codes_list, chunk_idx, results):
        """Process one vocoder chunk in background thread (RKNN NPU)."""
        try:
            codes_array = np.array(codes_list, dtype=np.int64)
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

            results[chunk_idx] = np.frombuffer(audio_data, dtype=np.int16)
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

        # Connect to talker
        talker_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        talker_conn.connect(self.talker_socket)

        # Send text
        msg = json.dumps({"text": text, "language": language}).encode()
        talker_conn.sendall(struct.pack("<I", len(msg)))
        talker_conn.sendall(msg)

        all_codes = []
        pending_codes = []  # Buffer for streaming vocoder
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

            # Streaming vocoder: submit chunk when buffer full
            if streaming and len(pending_codes) >= VOC_CHUNK_SIZE:
                t = threading.Thread(
                    target=self._vocoder_chunk,
                    args=(pending_codes[:], voc_chunk_idx, voc_results))
                t.start()
                voc_threads.append(t)
                print(f"  -> Vocoder chunk {voc_chunk_idx} ({len(pending_codes)} tokens) submitted")
                voc_chunk_idx += 1
                pending_codes = []

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
        if pending_codes:
            if streaming and voc_threads:
                # Streaming: submit remainder as background thread
                t = threading.Thread(
                    target=self._vocoder_chunk,
                    args=(pending_codes[:], voc_chunk_idx, voc_results))
                t.start()
                voc_threads.append(t)
                print(f"  -> Vocoder chunk {voc_chunk_idx} ({len(pending_codes)} tokens) submitted")
            else:
                # Non-streaming or single chunk: process directly
                t_voc = time.time()
                self._vocoder_chunk(pending_codes, 0, voc_results)
                print(f"Vocoder: {time.time() - t_voc:.1f}s ({len(pending_codes)} tokens)")

        # Wait for all vocoder threads
        if voc_threads:
            t_wait = time.time()
            for t in voc_threads:
                t.join()
            print(f"Vocoder wait: {time.time() - t_wait:.1f}s ({len(voc_threads)} chunks)")

        # Concatenate audio in order
        audio_chunks = []
        for i in range(max(len(voc_threads), 1)):
            if i in voc_results and len(voc_results[i]) > 0:
                audio_chunks.append(voc_results[i])

        if not audio_chunks:
            print("No audio generated!")
            return

        audio_int16 = np.concatenate(audio_chunks)

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
