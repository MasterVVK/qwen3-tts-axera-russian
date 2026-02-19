#!/usr/bin/env python3
"""
llama.cpp Talker Server for Qwen3-TTS.

Runs Qwen3-TTS talker LLM on CPU (A76 cores, llama.cpp GGUF Q4_K_M)
in embedding mode. Returns hidden states for code_0 sampling and
code predictor input.

Architecture:
  Text -> Tokenizer -> TextEmbedder -> llama.cpp (embedding mode)
       -> hidden_state + code_0 -> socket

Protocol (Unix socket):
  Client -> Server: [4 bytes msg_len] [JSON: {"text":"...", "language":"russian"}]
  Server -> Client: per generated token:
      [4 bytes int32: code_0]
      [4096 bytes float32: hidden_state[1024]]
  End:
      [4 bytes int32: -1] (EOS/done)
      [4 bytes int32: -2] (error)

Usage on CM3588:
  taskset -c 4-7 python3 llamacpp_talker_server.py \
      --model /root/tts-rknn/qwen3_tts_talker_q4_k_m.gguf \
      --embeddings /root/tts-rknn/embeddings \
      --socket /tmp/qwen3_talker.sock
"""

import os
import sys
import socket
import struct
import json
import argparse
import time
import signal
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_cpp_bindings import LlamaCppModel

HIDDEN_SIZE = 1024
CODEC_BOS_ID = 2149
CODEC_EOS_ID = 2150
CODEC_PAD_ID = 2148
CODEC_NOTHINK_ID = 2155

CODEC_LANG_IDS = {
    "chinese": 2055, "english": 2050, "german": 2053,
    "russian": 2069, "french": 2061, "japanese": 2058,
    "korean": 2065,
}

SENTINEL_DONE = -1
SENTINEL_ERROR = -2


class Qwen3TTSTalkerServer:
    def __init__(self, model_path, embeddings_dir,
                 socket_path="/tmp/qwen3_talker.sock",
                 temperature=0.8, top_k=50,
                 max_tokens=200, n_threads=4,
                 kv_cache_dir="/tmp"):
        self.socket_path = socket_path
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.kv_cache_dir = kv_cache_dir

        # Load embeddings
        print("Loading embeddings...")
        self.text_embedding = np.load(os.path.join(embeddings_dir, "text_embedding.npy"))
        self.codec_embedding = np.load(os.path.join(embeddings_dir, "codec_embedding.npy"))
        self.codec_head = np.load(os.path.join(embeddings_dir, "codec_head.npy"))
        self.proj_fc1_w = np.load(os.path.join(embeddings_dir, "text_projection_linear_fc1_weight.npy"))
        self.proj_fc1_b = np.load(os.path.join(embeddings_dir, "text_projection_linear_fc1_bias.npy"))
        self.proj_fc2_w = np.load(os.path.join(embeddings_dir, "text_projection_linear_fc2_weight.npy"))
        self.proj_fc2_b = np.load(os.path.join(embeddings_dir, "text_projection_linear_fc2_bias.npy"))
        print(f"  text_embedding: {self.text_embedding.shape}")
        print(f"  codec_embedding: {self.codec_embedding.shape}")
        print(f"  codec_head: {self.codec_head.shape}")

        # Load tokenizer
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True
        )

        # Load llama.cpp model
        print(f"Loading llama.cpp model: {model_path}")
        self.llm = LlamaCppModel(model_path, n_ctx=512, n_threads=n_threads)
        print(f"  Model loaded. n_embd={self.llm.n_embd}")

        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        self._running = False

    def _embed_text(self, token_ids):
        embeds = self.text_embedding[token_ids]
        h = embeds @ self.proj_fc1_w.T + self.proj_fc1_b
        h = h * (1.0 / (1.0 + np.exp(-h)))  # SiLU
        return (h @ self.proj_fc2_w.T + self.proj_fc2_b).astype(np.float32)

    def _build_prefix(self, text_token_ids, language="russian"):
        text_embeds = self._embed_text(text_token_ids)
        lang_id = CODEC_LANG_IDS.get(language, CODEC_LANG_IDS["russian"])
        special_ids = [CODEC_PAD_ID, lang_id, CODEC_NOTHINK_ID, CODEC_BOS_ID]
        codec_embeds = self.codec_embedding[special_ids]
        return np.concatenate([text_embeds, codec_embeds], axis=0).astype(np.float32)

    def _sample_token(self, hidden_state):
        logits = hidden_state @ self.codec_head.T
        logits = logits[:2048]
        top_indices = np.argsort(logits)[-self.top_k:]
        top_logits = logits[top_indices]
        probs = np.exp((top_logits - top_logits.max()) / max(self.temperature, 1e-6))
        probs /= probs.sum()
        return int(top_indices[np.random.choice(len(top_indices), p=probs)])

    def _prefix_hash(self, prefix):
        return hashlib.md5(prefix.tobytes()).hexdigest()[:16]

    def _generate_streaming(self, conn, text, language="russian"):
        text_token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        print(f"  Text tokens: {len(text_token_ids)}")

        prefix = self._build_prefix(text_token_ids, language)
        prefix_hash = self._prefix_hash(prefix)
        kv_path = os.path.join(self.kv_cache_dir, f"qwen3_kv_{prefix_hash}.bin")
        hidden_path = os.path.join(self.kv_cache_dir, f"qwen3_hidden_{prefix_hash}.npy")

        print(f"  Prefix: {prefix.shape[0]} tokens, hash={prefix_hash}")

        t0 = time.time()
        cache_hit = False

        # Try KV cache
        if os.path.exists(kv_path) and os.path.exists(hidden_path):
            try:
                ret = self.llm.state_load(kv_path)
                if ret == 0:
                    hidden = np.load(hidden_path)
                    self.llm.pos = prefix.shape[0]
                    cache_hit = True
                    t_prefill = time.time() - t0
                    print(f"  KV CACHE HIT: {t_prefill:.3f}s")
            except Exception as e:
                print(f"  KV cache error: {e}")

        if not cache_hit:
            hidden = self.llm.get_hidden(prefix, keep_history=0)
            t_prefill = time.time() - t0
            print(f"  Prefill: {t_prefill:.3f}s ({prefix.shape[0]/t_prefill:.0f} tok/s)")
            try:
                self.llm.state_save(kv_path)
                np.save(hidden_path, hidden)
            except Exception as e:
                print(f"  KV cache save error: {e}")

        # Generate tokens
        t_gen_start = time.time()
        out_tokens = 0

        for i in range(self.max_tokens):
            code_0 = self._sample_token(hidden)

            if code_0 == CODEC_EOS_ID or code_0 >= 2048:
                print(f"  EOS at step {i} (token={code_0})")
                break

            # Send code_0 + hidden_state to client
            try:
                conn.sendall(struct.pack("<i", code_0))
                conn.sendall(hidden.astype(np.float32).tobytes())
            except (BrokenPipeError, ConnectionResetError):
                print("  Client disconnected")
                return out_tokens

            out_tokens += 1

            # Wait for feedback embedding from client
            try:
                feedback_data = b""
                while len(feedback_data) < HIDDEN_SIZE * 4:
                    chunk = conn.recv(HIDDEN_SIZE * 4 - len(feedback_data))
                    if not chunk:
                        print("  Client closed connection")
                        return out_tokens
                    feedback_data += chunk
                feedback_embed = np.frombuffer(feedback_data, dtype=np.float32).reshape(1, HIDDEN_SIZE)
            except Exception as e:
                print(f"  Feedback error: {e}")
                break

            # Forward pass with feedback
            hidden = self.llm.get_hidden(feedback_embed, keep_history=1)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_gen_start
                rate = (i + 1) / elapsed
                print(f"  [{i+1}] rate={rate:.1f} tok/s")

        # Send done
        try:
            conn.sendall(struct.pack("<i", SENTINEL_DONE))
        except (BrokenPipeError, ConnectionResetError):
            pass

        elapsed = time.time() - t0
        if out_tokens > 0:
            gen_time = time.time() - t_gen_start
            print(f"  Generated {out_tokens} tokens in {gen_time:.1f}s "
                  f"({out_tokens/gen_time:.1f} tok/s)")
        return out_tokens

    def serve(self):
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(1)
        sock.settimeout(1.0)
        os.chmod(self.socket_path, 0o666)

        print(f"\nQwen3-TTS Talker Server listening on {self.socket_path}")
        print("Waiting for connections...\n")

        request_count = 0
        while self._running:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue

            request_count += 1
            print(f"--- Request #{request_count} ---")

            try:
                raw_len = conn.recv(4)
                if len(raw_len) < 4:
                    conn.close()
                    continue

                msg_len = struct.unpack("<I", raw_len)[0]
                if msg_len > 65536:
                    conn.sendall(struct.pack("<i", SENTINEL_ERROR))
                    conn.close()
                    continue

                raw_msg = b""
                while len(raw_msg) < msg_len:
                    chunk = conn.recv(msg_len - len(raw_msg))
                    if not chunk:
                        break
                    raw_msg += chunk

                msg = json.loads(raw_msg.decode())
                text = msg.get("text", "")
                language = msg.get("language", "russian")

                print(f"  Text: '{text[:60]}{'...' if len(text)>60 else ''}'")
                print(f"  Language: {language}")

                self._generate_streaming(conn, text, language)

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    conn.sendall(struct.pack("<i", SENTINEL_ERROR))
                except:
                    pass
            finally:
                conn.close()

        sock.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        print("Server stopped.")
        self.llm.destroy()


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Talker Server (llama.cpp)")
    parser.add_argument("--model", required=True, help="Path to .gguf model")
    parser.add_argument("--embeddings", required=True, help="Dir with embedding .npy files")
    parser.add_argument("--socket", default="/tmp/qwen3_talker.sock")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--kv_cache_dir", default="/tmp")
    args = parser.parse_args()

    server = Qwen3TTSTalkerServer(
        model_path=args.model,
        embeddings_dir=args.embeddings,
        socket_path=args.socket,
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n_threads=args.threads,
        kv_cache_dir=args.kv_cache_dir,
    )
    server.serve()


if __name__ == "__main__":
    main()
