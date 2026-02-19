"""
Minimal ctypes bindings for llama.cpp via llama_wrapper.so.

Uses a thin C wrapper to avoid struct-by-value ABI issues on aarch64.
All struct handling happens in C; Python only passes simple types and pointers.

Provides API for Qwen3-TTS talker:
  - Load GGUF model
  - Feed custom embeddings (codec + text)
  - Extract hidden states (embeddings from last layer)
  - KV cache management for autoregressive generation
"""

import ctypes
import os
import numpy as np

_WRAPPER_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_wrapper.so"),
    "/root/cosyvoice3-build/dual_npu/llama_wrapper.so",
]


def _load_wrapper():
    # Load llama.cpp dependencies globally first
    lib_dir = "/root/llama.cpp/build/bin"
    for dep in ["libggml-base.so", "libggml-cpu.so", "libggml.so", "libllama.so"]:
        dep_path = os.path.join(lib_dir, dep)
        if os.path.exists(dep_path):
            ctypes.CDLL(dep_path, mode=ctypes.RTLD_GLOBAL)

    for path in _WRAPPER_PATHS:
        if os.path.exists(path):
            return ctypes.CDLL(path)
    raise RuntimeError(f"llama_wrapper.so not found in {_WRAPPER_PATHS}")


_lib = _load_wrapper()

# Function signatures
_lib.wrapper_backend_init.argtypes = []
_lib.wrapper_backend_init.restype = None

_lib.wrapper_backend_free.argtypes = []
_lib.wrapper_backend_free.restype = None

_lib.wrapper_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.wrapper_load_model.restype = ctypes.c_void_p

_lib.wrapper_free_model.argtypes = [ctypes.c_void_p]
_lib.wrapper_free_model.restype = None

_lib.wrapper_model_n_embd.argtypes = [ctypes.c_void_p]
_lib.wrapper_model_n_embd.restype = ctypes.c_int

_lib.wrapper_create_context.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_lib.wrapper_create_context.restype = ctypes.c_void_p

_lib.wrapper_free_context.argtypes = [ctypes.c_void_p]
_lib.wrapper_free_context.restype = None

_lib.wrapper_kv_clear.argtypes = [ctypes.c_void_p]
_lib.wrapper_kv_clear.restype = None

_lib.wrapper_decode_embd.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
_lib.wrapper_decode_embd.restype = ctypes.c_int

_lib.wrapper_state_get_size.argtypes = [ctypes.c_void_p]
_lib.wrapper_state_get_size.restype = ctypes.c_size_t

_lib.wrapper_state_save_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.wrapper_state_save_file.restype = ctypes.c_int

_lib.wrapper_state_load_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.wrapper_state_load_file.restype = ctypes.c_int


class LlamaCppModel:
    """High-level wrapper for llama.cpp inference with custom embeddings.

    Used for Qwen3-TTS talker in embedding mode: feeds codec/text embeddings
    and extracts hidden states (not logits).
    """

    def __init__(self, model_path, n_ctx=512, n_threads=4):
        _lib.wrapper_backend_init()

        self.model = _lib.wrapper_load_model(model_path.encode(), 0)
        if not self.model:
            raise RuntimeError(f"Failed to load model: {model_path}")

        self.n_embd = _lib.wrapper_model_n_embd(self.model)
        assert self.n_embd > 0, f"n_embd={self.n_embd}"

        self.ctx = _lib.wrapper_create_context(
            self.model, n_ctx, n_ctx, n_threads, 1  # embeddings=1
        )
        if not self.ctx:
            raise RuntimeError("Failed to create llama context")

        self._pos = 0
        self._hidden_buf = np.zeros(self.n_embd, dtype=np.float32)

        print(f"LlamaCppModel ready: n_embd={self.n_embd}, n_ctx={n_ctx}, threads={n_threads}")

    def get_hidden(self, embeddings, keep_history=0):
        """Run forward pass with custom embeddings, return last hidden state.

        Args:
            embeddings: [n_tokens, n_embd] or [n_embd] float32
            keep_history: 0=clear KV (prefill), 1=append (decode step)
        Returns:
            [n_embd] float32 hidden state of last token
        """
        if keep_history == 0:
            _lib.wrapper_kv_clear(self.ctx)
            self._pos = 0

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n_tokens = embeddings.shape[0]
        assert embeddings.shape[1] == self.n_embd, \
            f"Dim mismatch: {embeddings.shape[1]} vs {self.n_embd}"

        embd_ptr = embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = self._hidden_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = _lib.wrapper_decode_embd(
            self.ctx, embd_ptr, n_tokens, self.n_embd, self._pos, out_ptr
        )
        if ret != 0:
            raise RuntimeError(f"wrapper_decode_embd failed: {ret}")

        self._pos += n_tokens
        return self._hidden_buf.copy()

    def clear_kv(self):
        _lib.wrapper_kv_clear(self.ctx)
        self._pos = 0

    def state_get_size(self):
        return _lib.wrapper_state_get_size(self.ctx)

    def state_save(self, path):
        ret = _lib.wrapper_state_save_file(self.ctx, path.encode())
        if ret == 0:
            print(f"  KV state saved: {path} (pos={self._pos})")
        return ret

    def state_load(self, path):
        ret = _lib.wrapper_state_load_file(self.ctx, path.encode())
        if ret == 0:
            print(f"  KV state loaded: {path}")
        return ret

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    def destroy(self):
        if self.ctx:
            _lib.wrapper_free_context(self.ctx)
            self.ctx = None
        if self.model:
            _lib.wrapper_free_model(self.model)
            self.model = None
        _lib.wrapper_backend_free()
