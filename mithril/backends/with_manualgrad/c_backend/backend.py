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

import numpy as np

from .... import core
from ...backend import Backend
from ...utils import process_shape
from .src import array, utils
from .src.array import PyArray

__all__ = ["CBackend"]


class CBackend(Backend[PyArray]):
    type = "c"
    SRC_PATH = "mithril/backends/with_manualgrad/c_backend/src"

    def __init__(self):
        self._precision = 32
        self._device = "cpu"

    @property
    def is_manualgrad(self):
        return True

    def set_seed(self, seed: int):
        pass

    def get_backend_array_type(self):
        return PyArray

    def empty(
        self, *shape: int | tuple[int, ...] | list[int], dtype: core.Dtype | None = None
    ) -> PyArray:
        assert dtype is None, "dtype is not supported in CBackend"
        _shape = process_shape(shape)
        return array.empty(_shape)

    def ones(
        self, *shape: int | tuple[int, ...] | list[int], dtype: core.Dtype | None = None
    ) -> PyArray:
        assert dtype is None, "dtype is not supported in CBackend"
        _shape = process_shape(shape)
        return array.ones(_shape)

    def zeros(
        self, *shape: int | tuple[int, ...] | list[int], dtype: core.Dtype | None = None
    ) -> PyArray:
        _shape = process_shape(shape)
        return array.zeros(_shape)

    def zeros_like(self, array, dtype: core.Dtype | None = None) -> PyArray:
        assert dtype is None, "dtype is not supported in CBackend"
        return self.array(np.zeros(array.shape, dtype=np.float32))

    def to_numpy(self, array: PyArray) -> np.ndarray:
        return utils.to_numpy(array)

    def array(self, np_array: np.ndarray, dtype: core.Dtype | None = None) -> PyArray:
        assert dtype is None, "dtype is not supported in CBackend"
        np_array = np_array.astype(np.float32)
        return utils.from_numpy(np_array)
