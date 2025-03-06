# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["ggml_struct"]

import ctypes

GGML_MAX_DIMS = 4
GGML_MAX_OP_PARAMS = 64
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64


class ggml_struct(ctypes.Structure):  # noqa: N801
    """n-dimensional tensor

    Attributes:
        type (int): ggml_type
        buffer (ctypes.pointer[ggml_backend_buffer]): pointer to backend buffer
        ne (ctypes.Array[ctypes.c_int64]): number of elements in each dimension
        nb (ctypes.Array[ctypes.c_size_t]): stride in bytes for each dimension
        op (int): ggml operation
        op_params (ctypes.Array[ctypes.c_int32]): `GGML_MAX_OP_PARAMS`-length array of
        operation parameters
        flags (int): tensor flags
        grad (ggml_struct_p): reference to gradient tensor
        src (ctypes.Array[ggml_struct_p]): `GGML_MAX_SRC`-length array of source tensors
        perf_runs (int): number of performance runs
        perf_cycles (int): number of cycles
        perf_time_us (int): time in microseconds
        view_src (ggml_struct_p): pointer to tensor if this tensor is a view, None if
        the tensor is not a view
        view_offs (ctypes.c_size_t): offset into the data pointer of the view tensor
        data (ctypes.c_void_p): reference to raw tensor data
        name (bytes): name of tensor
        extra (ctypes.c_void_p): extra data (e.g. for CUDA)
    """


ggml_struct._fields_ = [
    ("type", ctypes.c_int),
    ("buffer", ctypes.c_void_p),
    ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
    ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
    ("op", ctypes.c_int),
    (
        "op_params",
        ctypes.c_int32 * (GGML_MAX_OP_PARAMS // ctypes.sizeof(ctypes.c_int32)),
    ),
    ("flags", ctypes.c_int),
    ("src", ctypes.POINTER(ggml_struct) * GGML_MAX_SRC),
    ("view_src", ctypes.POINTER(ggml_struct)),
    ("view_offs", ctypes.c_size_t),
    ("data", ctypes.c_void_p),
    ("name", ctypes.c_char * GGML_MAX_NAME),
    ("extra", ctypes.c_void_p),
    ("padding", ctypes.c_char * 8),
]
