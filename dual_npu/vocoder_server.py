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
import sys
import socket
import struct
import argparse
import time
import signal
import numpy as np

SAMPLE_RATE = 24000
SAMPLES_PER_TOKEN = 1920


class VocoderServer:
    def __init__(self, model_path, socket_path="/tmp/qwen3_voc.sock"):
        self.socket_path = socket_path
        self.is_onnx = model_path.endswith('.onnx')

        if self.is_onnx:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            inp_shape = self.sess.get_inputs()[0].shape
            self.max_tokens = inp_shape[1]
            print(f"Vocoder: ONNX CPU, max_tokens={self.max_tokens}")
        else:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(model_path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN vocoder: {ret}")
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                raise RuntimeError(f"Failed to init RKNN vocoder: {ret}")
            self.max_tokens = 64 if "64" in os.path.basename(model_path) else 256
            print(f"Vocoder: RKNN NPU, max_tokens={self.max_tokens}")

        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._running = False

    def synthesize(self, codes_array):
        n_tokens = len(codes_array)
        audio_chunks = []
        chunk_start = 0

        while chunk_start < n_tokens:
            chunk_end = min(chunk_start + self.max_tokens, n_tokens)
            chunk_len = chunk_end - chunk_start
            padded = np.zeros((1, self.max_tokens, 16), dtype=np.int64)
            padded[0, :chunk_len, :] = codes_array[chunk_start:chunk_end, :16]

            if self.is_onnx:
                outputs = self.sess.run(None, {'audio_codes': padded})
            else:
                outputs = self.rknn.inference(inputs=[padded])

            audio_chunk = outputs[0].flatten()
            actual_samples = int(chunk_len * SAMPLES_PER_TOKEN)
            audio_chunks.append(audio_chunk[:actual_samples])
            chunk_start = chunk_end

        if not audio_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(audio_chunks)

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
            finally:
                conn.close()

        sock.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        if not self.is_onnx:
            self.rknn.release()
        print("Vocoder Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Vocoder Server")
    parser.add_argument("--model", required=True, help="Vocoder model (.rknn or .onnx)")
    parser.add_argument("--socket", default="/tmp/qwen3_voc.sock")
    args = parser.parse_args()

    server = VocoderServer(model_path=args.model, socket_path=args.socket)
    server.serve()


if __name__ == "__main__":
    main()
