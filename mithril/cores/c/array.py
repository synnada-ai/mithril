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
from collections.abc import Sequence

import numpy as np


class PyArray:
    def __init__(self, arr: ctypes.Structure, shape: tuple[int, ...] | list[int]):
        # TODO: PyArray need to store strides

        self.arr = arr
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.ndim = len(shape)

    # TODO: Implement __del__ method for deleting the struct
    # def __del__(self):
    #     lib.delete_struct(self.arr)

    @property
    def dtype(self) -> type:
        return np.float32

    @property
    def data(self) -> Sequence[int | Sequence[int | Sequence[int]]]:
        total_elements = 1
        for dim in self.shape:
            total_elements *= dim

        # Convert the array into a Python list
        data_ptr = ctypes.cast(self.arr.data, ctypes.POINTER(ctypes.c_float))
        data_list = [data_ptr[i] for i in range(total_elements)]

        # Reshape the flat list based on the shape
        def reshape(
            data: Sequence[int], shape: tuple[int, ...]
        ) -> Sequence[int | Sequence[int | Sequence[int]]]:
            if len(shape) == 1:
                return data

            size = shape[0]
            return [
                reshape(data[i * size : (i + 1) * size], shape[1:])
                for i in range(len(data) // size)
            ]

        return reshape(data_list, self.shape)

    def __repr__(self):
        return f"array({self.data})"

    def __str__(self):
        return f"PyArray(shape={self.shape})\n{self.data}"
