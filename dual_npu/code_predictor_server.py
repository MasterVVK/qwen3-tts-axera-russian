#!/usr/bin/env python3
"""
Code Predictor Server for Qwen3-TTS.

Runs the 5-layer code predictor transformer via ONNX Runtime to predict
codec groups 1-15 from hidden state + code_0 embedding.

Protocol (Unix socket):
  Client -> Server: [4 bytes: hidden_size*4] [hidden_state float32]
                    [4 bytes: code_0 int32]
  Server -> Client: [60 bytes: 15 int32 codes]

Usage on CM3588:
  taskset -c 4-7 python3 code_predictor_server.py \
      --model_dir /root/tts-rknn/code_predictor \
      --embeddings_dir /root/tts-rknn/embeddings \
      --socket /tmp/qwen3_cp.sock
"""

import os
import sys
import socket
import struct
import argparse
import time
import signal
import numpy as np

HIDDEN_SIZE = 1024


class CodePredictorServer:
    def __init__(self, model_dir, embeddings_dir,
                 socket_path="/tmp/qwen3_cp.sock",
                 temperature=0.1, top_k=50, n_threads=1,
                 batch_prefill=False):
        self.socket_path = socket_path
        self.temperature = temperature
        self.top_k = top_k
        self.num_groups = 15
        self.batch_prefill = batch_prefill

        # Load weights
        print("Loading code predictor weights...")
        weights_path = os.path.join(model_dir, "code_predictor_weights.npz")
        w = np.load(weights_path)
        self.codec_embeddings = [w[f"codec_emb_{i}"].astype(np.float32) for i in range(self.num_groups)]
        self.lm_heads = [w[f"lm_head_{i}"].astype(np.float32) for i in range(self.num_groups)]

        # Load codec embedding for code_0 lookup
        self.codec_embedding = np.load(os.path.join(embeddings_dir, "codec_embedding.npy"))

        # Load ONNX model
        print("Loading ONNX code predictor...")
        import onnxruntime as ort
        onnx_path = os.path.join(model_dir, "code_predictor_decode_step.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = n_threads
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=['CPUExecutionProvider']
        )

        self.num_layers = 5
        self.head_dim = w["layer_0_q_proj"].shape[0] // 16
        self.num_kv_heads = 8

        print(f"  ONNX loaded: {self.num_layers} layers, head_dim={self.head_dim}")

        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._running = False

    def _ort_step(self, hidden, position, kv_caches):
        feed = {"hidden": hidden, "position": np.array([position], dtype=np.int64)}
        feed.update(kv_caches)
        out = self.sess.run(None, feed)
        new_kv = {}
        for i in range(self.num_layers):
            new_kv[f"past_k_{i}"] = out[1 + i * 2]
            new_kv[f"past_v_{i}"] = out[2 + i * 2]
        return out[0], new_kv

    def _sample(self, logits):
        top_indices = np.argpartition(logits, -self.top_k)[-self.top_k:]
        top_logits = logits[top_indices]
        probs = np.exp((top_logits - top_logits.max()) / max(self.temperature, 1e-6))
        probs /= probs.sum()
        return int(top_indices[np.random.choice(len(top_indices), p=probs)])

    def predict(self, hidden_state, code_0):
        """Predict groups 1-15 from hidden_state and code_0."""
        H = HIDDEN_SIZE
        # Use TALKER's codec embedding for code_0 (matches official model)
        code_0_embed = self.codec_embedding[code_0]

        # Init empty KV caches
        kv = {}
        for i in range(self.num_layers):
            kv[f"past_k_{i}"] = np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32)
            kv[f"past_v_{i}"] = np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32)

        if self.batch_prefill:
            # Batch prefill: positions 0+1 together (~15ms savings per token)
            h0 = hidden_state.flatten()[:H].astype(np.float32)
            h1 = code_0_embed.flatten()[:H].astype(np.float32)
            h_batch = np.stack([h0, h1]).reshape(1, 2, H)
            feed = {"hidden": h_batch, "position": np.array([0, 1], dtype=np.int64)}
            feed.update(kv)
            out = self.sess.run(None, feed)
            hidden = out[0]
            kv = {}
            for i in range(self.num_layers):
                kv[f"past_k_{i}"] = out[1 + i * 2]
                kv[f"past_v_{i}"] = out[2 + i * 2]
        else:
            # Sequential prefill: numerically exact
            h = hidden_state.flatten()[:H].reshape(1, 1, H).astype(np.float32)
            hidden, kv = self._ort_step(h, 0, kv)
            h = code_0_embed.flatten()[:H].reshape(1, 1, H).astype(np.float32)
            hidden, kv = self._ort_step(h, 1, kv)

        # Sample group 0
        predicted_tokens = []
        logits = hidden[0, -1] @ self.lm_heads[0].T
        token = self._sample(logits)
        predicted_tokens.append(token)

        # Decode groups 1-14
        for step in range(1, self.num_groups):
            embed = self.codec_embeddings[step - 1][token].reshape(1, 1, H)
            hidden, kv = self._ort_step(embed, step + 1, kv)
            logits = hidden[0, -1] @ self.lm_heads[step].T
            token = self._sample(logits)
            predicted_tokens.append(token)

        return predicted_tokens

    def serve(self):
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(1)
        sock.settimeout(1.0)
        os.chmod(self.socket_path, 0o666)

        print(f"\nCode Predictor Server listening on {self.socket_path}")

        while self._running:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue

            try:
                # Read hidden_state
                hidden_data = b""
                while len(hidden_data) < HIDDEN_SIZE * 4:
                    chunk = conn.recv(HIDDEN_SIZE * 4 - len(hidden_data))
                    if not chunk:
                        break
                    hidden_data += chunk
                if len(hidden_data) < HIDDEN_SIZE * 4:
                    conn.close()
                    continue

                hidden_state = np.frombuffer(hidden_data, dtype=np.float32)

                # Read code_0
                code_data = conn.recv(4)
                if len(code_data) < 4:
                    conn.close()
                    continue
                code_0 = struct.unpack("<i", code_data)[0]

                # Predict
                t0 = time.time()
                codes = self.predict(hidden_state, code_0)
                dt = time.time() - t0

                # Send back 15 codes
                for c in codes:
                    conn.sendall(struct.pack("<i", c))

            except Exception as e:
                print(f"  CP Error: {e}")
            finally:
                conn.close()

        sock.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        print("Code Predictor Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Code Predictor Server")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--socket", default="/tmp/qwen3_cp.sock")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--batch_prefill", action="store_true",
                        help="Batch 2-token prefill (~15ms faster but approximate)")
    args = parser.parse_args()

    server = CodePredictorServer(
        model_dir=args.model_dir,
        embeddings_dir=args.embeddings_dir,
        socket_path=args.socket,
        temperature=args.temperature,
        top_k=args.top_k,
        n_threads=args.threads,
        batch_prefill=args.batch_prefill,
    )
    server.serve()


if __name__ == "__main__":
    main()
