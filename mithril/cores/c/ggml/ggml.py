import ctypes


GGML_MAX_DIMS = 4
GGML_MAX_OP_PARAMS = 64
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64





class ggml_tensor(ctypes.Structure):
    pass



ggml_tensor._fields_ = [
    ("type", ctypes.c_int),
    ("backend", ctypes.c_int),
    ("buffer", ctypes.c_void_p),
    ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
    ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
    ("op", ctypes.c_int),
    (
        "op_params",
        ctypes.c_int32 * (GGML_MAX_OP_PARAMS // ctypes.sizeof(ctypes.c_int32)),
    ),
    ("flags", ctypes.c_int),
    ("grad", ctypes.POINTER(ggml_tensor)),
    ("src", ctypes.POINTER(ggml_tensor) * GGML_MAX_SRC),
    ("perf_runs", ctypes.c_int),
    ("perf_cycles", ctypes.c_int64),
    ("perf_time_us", ctypes.c_int64),
    ("view_src", ctypes.POINTER(ggml_tensor)),
    ("view_offs", ctypes.c_size_t),
    ("data", ctypes.c_void_p),
    ("name", ctypes.c_char * GGML_MAX_NAME),
    ("extra", ctypes.c_void_p),
    ("padding", ctypes.c_char * 8),
]


GGML_TENSOR_SIZE = ctypes.sizeof(ggml_tensor)

ggml_tensor_p = ctypes._Pointer[ggml_tensor]