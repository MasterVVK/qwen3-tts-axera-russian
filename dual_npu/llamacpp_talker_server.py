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
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157

# TTS special tokens (text vocabulary IDs for text_embedding lookup)
TTS_PAD_TOKEN_ID = 151671
TTS_BOS_TOKEN_ID = 151672  # <tts_text_bos>
TTS_EOS_TOKEN_ID = 151673  # <tts_text_eod>
IM_START_TOKEN_ID = 151644  # <|im_start|>

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

        # Pre-compute TTS special embeddings (dual-stream text side)
        special_ids = np.array([TTS_PAD_TOKEN_ID, TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID])
        special_embeds = self._embed_text(special_ids)
        self.tts_pad_embed = special_embeds[0]  # [1024]
        self.tts_bos_embed = special_embeds[1]  # [1024]
        self.tts_eos_embed = special_embeds[2]  # [1024]
        print(f"  TTS special embeds computed (pad/bos/eos)")

        # Load tokenizer
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True,
            local_files_only=True
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
        """Build dual-stream prefix matching official Qwen3-TTS.

        Structure (text_stream + codec_stream summed at each position):
          [role_0, role_1, role_2]                     -- embed_text only
          [pad+nothink, pad+think_bos, pad+think_eos]  -- tts_pad + codec
          [bos+pad]                                     -- tts_bos + codec_pad
          [text(t0)+pad, ..., text(tN)+pad, eos+pad]    -- text + codec_pad
          [pad+bos]                                     -- tts_pad + codec_bos
        """
        # Role tokens: <|im_start|> assistant \n (pure text stream)
        role_ids = np.array([IM_START_TOKEN_ID, 77091, 198])
        role_embeds = self._embed_text(role_ids)  # [3, 1024]

        # Codec prefix: tts_pad + codec_special [nothink, think_bos, think_eos]
        codec_prefix = self.codec_embedding[
            [CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID]
        ]  # [3, 1024]
        text_for_codec = np.stack([self.tts_pad_embed] * 3)  # [3, 1024]
        dual_codec = text_for_codec + codec_prefix  # [3, 1024]

        # Transition: tts_bos + codec_pad
        transition = (self.tts_bos_embed + self.codec_embedding[CODEC_PAD_ID])[np.newaxis]  # [1, 1024]

        # Text tokens + tts_eos: text_proj(token) + codec_pad
        text_embeds = self._embed_text(text_token_ids)  # [N, 1024]
        text_plus_eos = np.concatenate(
            [text_embeds, self.tts_eos_embed[np.newaxis]], axis=0
        )  # [N+1, 1024]
        codec_pad_tile = np.tile(
            self.codec_embedding[CODEC_PAD_ID], (len(text_token_ids) + 1, 1)
        )  # [N+1, 1024]
        dual_text = text_plus_eos + codec_pad_tile  # [N+1, 1024]

        # Final: tts_pad + codec_bos
        final = (self.tts_pad_embed + self.codec_embedding[CODEC_BOS_ID])[np.newaxis]  # [1, 1024]

        prefix = np.concatenate(
            [role_embeds, dual_codec, transition, dual_text, final], axis=0
        ).astype(np.float32)
        return prefix

    def _sample_token(self, hidden_state, past_tokens=None, n_text_tokens=0):
        """Sample codec token. Allows audio (0-2047) + EOS (2150), masks rest."""
        logits = hidden_state @ self.codec_head.T  # [3072]

        # Mask special tokens except EOS: suppress 2048-2149 and 2151+
        logits[2048:CODEC_EOS_ID] = -1e10
        if CODEC_EOS_ID + 1 < len(logits):
            logits[CODEC_EOS_ID + 1:] = -1e10

        # Adaptive EOS boost (compensates GGUF underweighting EOS logit)
        if past_tokens is not None and n_text_tokens > 0:
            expected_len = n_text_tokens * 3
            progress = len(past_tokens) / expected_len if expected_len > 0 else 0
            if progress > 0.8:
                boost = min((progress - 0.8) / 0.7, 1.0) * 15.0
                logits[CODEC_EOS_ID] += boost
            if progress > 2.0:
                return CODEC_EOS_ID  # force EOS

        # Repetition penalty (window=30, deduplicated)
        if past_tokens:
            for t in set(past_tokens[-30:]):
                if 0 <= t < len(logits):
                    if logits[t] > 0:
                        logits[t] /= 1.2
                    else:
                        logits[t] *= 1.2

        # Top-K sampling
        top_indices = np.argsort(logits)[-self.top_k:]
        top_logits = logits[top_indices]
        scaled = top_logits / max(self.temperature, 1e-6)
        probs = np.exp(scaled - scaled.max())
        probs /= probs.sum()

        # Top-p (nucleus) filtering
        sorted_idx = np.argsort(-probs)
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cumsum, 0.95) + 1
        keep = sorted_idx[:cutoff]
        probs_filtered = probs[keep]
        probs_filtered /= probs_filtered.sum()
        chosen = keep[np.random.choice(len(keep), p=probs_filtered)]
        return int(top_indices[chosen])

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
        past_tokens = []
        n_text_tokens = len(text_token_ids)

        for i in range(self.max_tokens):
            code_0 = self._sample_token(hidden, past_tokens=past_tokens,
                                         n_text_tokens=n_text_tokens)

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
            past_tokens.append(code_0)

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
