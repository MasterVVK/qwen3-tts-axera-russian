# Minimal rknnlite.api shim over librknnrt.so via ctypes.
# Board python has no rknn_toolkit_lite2 wheel (py3.14); this covers exactly
# the RKNNLite surface vocoder_server.py uses.
import ctypes

import numpy as np

_lib = ctypes.CDLL("librknnrt.so")
_u64, _u32, _u8, _i32, _f32 = (ctypes.c_uint64, ctypes.c_uint32,
                                ctypes.c_uint8, ctypes.c_int32, ctypes.c_float)


class _IO(ctypes.Structure):
    _fields_ = [("n_input", _u32), ("n_output", _u32)]


class _Attr(ctypes.Structure):
    _fields_ = [("index", _u32), ("n_dims", _u32),
                ("dims", _u32 * 16), ("name", ctypes.c_char * 256),
                ("n_elems", _u32), ("size", _u32),
                ("fmt", _i32), ("type", _i32), ("qnt_type", _i32),
                ("fl", ctypes.c_int8), ("zp", _i32), ("scale", _f32),
                ("w_stride", _u32), ("size_with_stride", _u32),
                ("pass_through", _u8), ("h_stride", _u32)]


class _Input(ctypes.Structure):
    _fields_ = [("index", _u32), ("buf", ctypes.c_void_p), ("size", _u32),
                ("pass_through", _u8), ("type", _i32), ("fmt", _i32)]


class _Output(ctypes.Structure):
    _fields_ = [("want_float", _u8), ("is_prealloc", _u8),
                ("index", _u32), ("buf", ctypes.c_void_p), ("size", _u32)]


def _sig(fn, args, res=ctypes.c_int):
    fn.argtypes = args
    fn.restype = res


_sig(_lib.rknn_init, [ctypes.POINTER(_u64), ctypes.c_void_p, _u32, _u32,
                      ctypes.c_void_p])
_sig(_lib.rknn_destroy, [_u64])
_sig(_lib.rknn_query, [_u64, ctypes.c_int, ctypes.c_void_p, _u32])
_sig(_lib.rknn_inputs_set, [_u64, _u32, ctypes.POINTER(_Input)])
_sig(_lib.rknn_set_core_mask, [_u64, ctypes.c_int])
_sig(_lib.rknn_run, [_u64, ctypes.c_void_p])
_sig(_lib.rknn_outputs_get, [_u64, _u32, ctypes.POINTER(_Output),
                             ctypes.c_void_p])
_sig(_lib.rknn_outputs_release, [_u64, _u32, ctypes.POINTER(_Output)])

_TENSOR_FLOAT32 = 0
_TENSOR_FLOAT16 = 1
_TENSOR_INT64 = 8
_TYPE_BY_DTYPE = {np.int64: _TENSOR_INT64, np.float32: _TENSOR_FLOAT32,
                  np.float16: _TENSOR_FLOAT16}


class RKNNLite:
    NPU_CORE_AUTO = 0
    NPU_CORE_0 = 1
    NPU_CORE_1 = 2
    NPU_CORE_2 = 4
    NPU_CORE_0_1_2 = 7
    NPU_CORE_ALL = 0xffff

    def __init__(self):
        self.ctx = _u64(0)
        self._buf = None
        self._n_out = 0

    def load_rknn(self, path):
        data = open(path, "rb").read()
        self._buf = ctypes.create_string_buffer(data, len(data))
        return _lib.rknn_init(ctypes.byref(self.ctx), self._buf, len(data),
                              0, None)

    def init_runtime(self, target=None, core_mask=0, **kw):
        r = _lib.rknn_set_core_mask(self.ctx, core_mask)
        io = _IO()
        r |= _lib.rknn_query(self.ctx, 0, ctypes.byref(io),
                             ctypes.sizeof(io))
        self._n_out = io.n_output
        return r

    def inference(self, inputs):
        n_in = len(inputs)
        inp = (_Input * n_in)()
        keep = []
        for i, t in enumerate(inputs):
            t = np.asarray(t)
            dtype = (np.int64 if t.dtype == np.int64
                     else np.float16 if t.dtype == np.float16 else np.float32)
            x = np.ascontiguousarray(t, dtype=dtype)
            keep.append(x)
            inp[i] = _Input(index=i, buf=x.ctypes.data, size=x.nbytes,
                            pass_through=1, type=_TYPE_BY_DTYPE[dtype], fmt=0)
        r = _lib.rknn_inputs_set(self.ctx, n_in, inp)
        if r:
            raise RuntimeError(f"rknn_inputs_set: {r}")
        r = _lib.rknn_run(self.ctx, None)
        if r:
            raise RuntimeError(f"rknn_run: {r}")
        outs = (_Output * self._n_out)()
        for i in range(self._n_out):
            outs[i].index = i
            outs[i].want_float = 1
        r = _lib.rknn_outputs_get(self.ctx, self._n_out, outs, None)
        if r:
            raise RuntimeError(f"rknn_outputs_get: {r}")
        res = []
        attr = _Attr()
        for i in range(self._n_out):
            attr.index = i
            _lib.rknn_query(self.ctx, 2, ctypes.byref(attr),
                            ctypes.sizeof(attr))
            ptr = ctypes.cast(outs[i].buf, ctypes.POINTER(_f32))
            res.append(np.ctypeslib.as_array(ptr, shape=(attr.n_elems,)).copy())
        _lib.rknn_outputs_release(self.ctx, self._n_out, outs)
        return res

    def release(self):
        if self.ctx.value:
            _lib.rknn_destroy(self.ctx)
            self.ctx = _u64(0)
