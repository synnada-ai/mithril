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

import ctypes
import os
from typing import Any

import numpy as np

from .... import types
from ....common import BiMap
from ....cores.c.array import PyArray
from ....cores.c.ggml import ops
from ....cores.c.ggml.ggml_core import ggml_struct
from ....cores.c.raw_c import array
from ...backend import Backend
from ...utils import process_shape
from ..c_backend.utils import from_numpy
from . import utils

__all__ = ["GGMLBackend"]

dtype_map: BiMap[str, Any] = BiMap(
    {
        "float32": np.float32,
    }
)


class GGMLBackend(Backend[PyArray]):
    backend_type = "ggml"
    SRC_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "cores", "c", "ggml"
    )
    CODEGEN_CONFIG = utils.CODEGEN_CONFIG

    def __init__(self) -> None:
        self._device = "cpu"
        self.primitive_function_dict = ops.primitive_func_dict
        self.dtype_map = dtype_map
        self.registered_primitives = {}
        self.array_creation_funcs: list[str] = []

    @property
    def is_manualgrad(self) -> bool:
        return True

    @property
    def precision(self) -> int:
        return 32

    def set_seed(self, seed: int) -> None:
        raise NotImplementedError("set_seed is not supported in GGML Backend")

    def get_backend_array_type(self) -> type[PyArray]:
        return PyArray

    def get_struct_cls(self) -> type[ctypes.Structure]:
        return ggml_struct

    def to_numpy(self, array: PyArray) -> np.ndarray[Any, Any]:
        return np.ctypeslib.as_array(
            ctypes.cast(array.arr.data, ctypes.POINTER(ctypes.c_float)),
            shape=(array.shape),
        )

    def array(
        self, input: np.ndarray[Any, Any], *, dtype: types.Dtype | None = None
    ) -> PyArray:
        assert dtype is None, "dtype is not supported in CBackend"
        input = input.astype(np.float32)
        data_ptr = ctypes.cast(from_numpy(input).arr.data, ctypes.c_void_p)
        return PyArray(ggml_struct(data=data_ptr), input.shape)

    def ones(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: types.Dtype | None = None,
    ) -> PyArray:
        assert dtype is None, "dtype is not supported in GGML Backend"
        _shape = process_shape(shape)
        data_ptr = ctypes.cast(array.ones(_shape).arr.data, ctypes.c_void_p)
        return PyArray(ggml_struct(data=data_ptr), _shape)

    def zeros(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: types.Dtype | None = None,
    ) -> PyArray:
        assert dtype is None, "dtype is not supported in GGML Backend"
        _shape = process_shape(shape)
        data_ptr = ctypes.cast(array.zeros(_shape).arr.data, ctypes.c_void_p)
        return PyArray(ggml_struct(data=data_ptr), _shape)

    def empty(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: types.Dtype | None = None,
    ) -> PyArray:
        assert dtype is None, "dtype is not supported in GGML Backend"
        _shape = process_shape(shape)
        data_ptr = ctypes.cast(array.empty(_shape).arr.data, ctypes.c_void_p)
        return PyArray(ggml_struct(data=data_ptr), _shape)
