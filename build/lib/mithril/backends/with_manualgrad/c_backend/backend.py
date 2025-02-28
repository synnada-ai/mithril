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

import os
from typing import Any

import numpy as np

from .... import types
from ....cores.c import array
from ....cores.c.array import PyArray
from ...backend import Backend
from ...utils import process_shape
from . import utils

__all__ = ["CBackend"]


class CBackend(Backend[PyArray]):
    backend_type = "c"
    SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "cores", "c")

    def __init__(self) -> None:
        self._device = "cpu"
        self.primitive_function_dict = {}

    @property
    def is_manualgrad(self) -> bool:
        return True

    @property
    def precision(self) -> int:
        return 32

    def set_seed(self, seed: int) -> None:
        pass

    def get_backend_array_type(self) -> type[PyArray]:
        return PyArray

    def empty(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: types.Dtype | None = None,
    ) -> PyArray:
        assert dtype is None, "dtype is not supported in CBackend"
        _shape = process_shape(shape)
        return array.empty(_shape)

    def ones(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: types.Dtype | None = None,
    ) -> PyArray:
        assert dtype is None, "dtype is not supported in CBackend"
        _shape = process_shape(shape)
        return array.ones(_shape)

    def zeros(
        self,
        *shape: int | tuple[int, ...] | list[int],
        dtype: types.Dtype | None = None,
    ) -> PyArray:
        _shape = process_shape(shape)
        return array.zeros(_shape)

    def zeros_like(
        self, input: PyArray, *, dtype: types.Dtype | None = None
    ) -> PyArray:
        assert dtype is None, "dtype is not supported in CBackend"
        return self.array(np.zeros(input.shape, dtype=np.float32))

    def to_numpy(self, array: PyArray) -> np.ndarray[Any, Any]:
        return utils.to_numpy(array)

    def array(
        self, input: np.ndarray[Any, Any], *, dtype: types.Dtype | None = None
    ) -> PyArray:
        assert dtype is None, "dtype is not supported in CBackend"
        input = input.astype(np.float32)
        return utils.from_numpy(input)
