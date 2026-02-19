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
import numpy as np
import wave as wavmod

HIDDEN_SIZE = 1024
SAMPLE_RATE = 24000
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

    def synthesize(self, text, language="russian", output="output.wav"):
        """Run full TTS pipeline via sockets."""
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
        n_tokens = 0

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

            # Receive hidden_state from talker
            hidden_data = b""
            while len(hidden_data) < HIDDEN_SIZE * 4:
                chunk = talker_conn.recv(HIDDEN_SIZE * 4 - len(hidden_data))
                if not chunk:
                    break
                hidden_data += chunk

            hidden_state = np.frombuffer(hidden_data, dtype=np.float32).copy()

            # Send to code predictor
            cp_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            cp_conn.connect(self.cp_socket)
            cp_conn.sendall(hidden_state.tobytes())
            cp_conn.sendall(struct.pack("<i", code_0))

            # Receive 15 codes
            codes_1_15 = []
            for _ in range(15):
                cd = cp_conn.recv(4)
                if len(cd) < 4:
                    codes_1_15.append(0)
                else:
                    codes_1_15.append(struct.unpack("<i", cd)[0])
            cp_conn.close()

            all_codes.append([code_0] + codes_1_15)
            n_tokens += 1

            # Feedback: code_0=talker embed, codes 1-15=CP per-group embeds
            if self.codec_embedding is not None:
                sum_embed = self.codec_embedding[code_0].copy()  # talker embed
                if self.cp_codec_embeddings is not None:
                    for gi, tok in enumerate(codes_1_15):
                        sum_embed += self.cp_codec_embeddings[gi][tok]  # CP embed
                else:
                    for tok in codes_1_15:
                        sum_embed += self.codec_embedding[tok]  # fallback: talker embed
                # Non-streaming: add constant tts_pad_embed (all text already in prefix)
                if self.tts_pad_embed is not None:
                    sum_embed += self.tts_pad_embed
                feedback = sum_embed.reshape(1, HIDDEN_SIZE).astype(np.float32)
            else:
                raise RuntimeError("codec_embedding required for feedback")

            # Send feedback to talker
            talker_conn.sendall(feedback.tobytes())

            if n_tokens % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  [{n_tokens}] {n_tokens/elapsed:.1f} tok/s")

        talker_conn.close()

        if n_tokens == 0:
            print("No tokens generated!")
            return

        t_gen = time.time() - t_start
        print(f"\nGenerated {n_tokens} tokens in {t_gen:.1f}s ({n_tokens/t_gen:.1f} tok/s)")

        # Send to vocoder
        codes_array = np.array(all_codes, dtype=np.int64)
        print(f"\nVocoder: {codes_array.shape}")

        voc_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        voc_conn.connect(self.voc_socket)

        t_voc = time.time()
        voc_conn.sendall(struct.pack("<i", n_tokens))
        voc_conn.sendall(codes_array.tobytes())

        # Receive audio
        header = voc_conn.recv(4)
        n_samples = struct.unpack("<i", header)[0]

        audio_data = b""
        while len(audio_data) < n_samples * 2:
            chunk = voc_conn.recv(min(65536, n_samples * 2 - len(audio_data)))
            if not chunk:
                break
            audio_data += chunk
        voc_conn.close()

        t_voc = time.time() - t_voc
        print(f"Vocoder: {t_voc:.1f}s")

        # Save WAV
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
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
    client.synthesize(text, args.language, args.output)


if __name__ == "__main__":
    main()
