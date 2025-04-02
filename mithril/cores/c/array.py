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
from numbers import Real

from .ggml.ggml_core import ggml_struct
from .raw_c.definitions import Array, lib


class PyArray:
    def __init__(self, arr: ctypes.Structure, shape: tuple[int, ...] | list[int]):
        # TODO: PyArray need to store strides

        self.arr = arr
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.ndim = len(shape)
        self.name = self.arr.__class__.__name__

    # TODO: Implement __del__ method for deleting the struct
    # def __del__(self):
    #     self.lib.delete_struct(self.arr)

    @property
    def dtype(self) -> type:
        return ctypes.c_float

    @property
    def data(self) -> Sequence[int | Sequence[int | Sequence[int]]]:
        total_elements = 1
        for dim in self.shape:
            total_elements *= dim

        # Convert the array into a Python list
        data_ptr = ctypes.cast(self.arr.data, ctypes.POINTER(ctypes.c_float))
        data_list = [data_ptr[i] for i in range(total_elements)]

        def reshape(data: list[float], shape: tuple[int, ...]) -> list:
            if len(shape) == 1:
                return data
            slice_size = 1
            for d in shape[1:]:
                slice_size *= d
            return [
                reshape(data[i * slice_size : (i + 1) * slice_size], shape[1:])
                for i in range(shape[0])
            ]

        return reshape(data_list, self.shape)

    def __repr__(self):
        return f"array({self.data})"

    def __str__(self):
        return f"PyArray(shape={self.shape})\n{self.data}"

    def __add__(self, other):
        fn = (lib.scalar_add, lib.add)[isinstance(other, PyArray)]
        return binary_op(self, other, fn)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        fn = (lib.scalar_multiply, lib.multiplication)[isinstance(other, PyArray)]
        return binary_op(self, other, fn)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        fn = (lib.scalar_subtract, lib.subtract)[isinstance(other, PyArray)]
        return binary_op(self, other, fn)

    def __rsub__(self, other):
        if isinstance(other, Real):
            return other + (-1 * self)
        else:
            return NotImplemented


def binary_op(left, right, op):
    if isinstance(right, PyArray):
        shape = left.shape if left.ndim >= right.ndim else right.shape
        other_ptr, _ = _get_array_ptr(right)
    else:
        # Scalar addition
        shape = left.shape
        other_ptr = ctypes.c_float(float(right))

    c_shape = (ctypes.c_int * len(shape))(*shape)
    result = lib.create_empty_struct(len(c_shape), c_shape)
    self_ptr, _ = _get_array_ptr(left)
    op(result, self_ptr, other_ptr)
    return _create_result(result, shape, left.name)


def _create_temp_array(pyarray):
    arr = pyarray.arr
    ndim = pyarray.ndim
    shape = pyarray.shape
    strides = [1] if ndim == 1 else [shape[1], 1]
    c_shape_array = (ctypes.c_int * ndim)(*shape)
    c_strides_array = (ctypes.c_int * ndim)(*strides)
    size = 1
    for size_ in shape:
        size *= size_
    return Array(
        data=ctypes.cast(arr.data, ctypes.POINTER(ctypes.c_float)),
        shape=ctypes.cast(c_shape_array, ctypes.POINTER(ctypes.c_int)),
        strides=ctypes.cast(c_strides_array, ctypes.POINTER(ctypes.c_int)),
        ndim=ndim,
        size=size,
    )


def _get_array_ptr(arr):
    if arr.name == "Array":
        ptr = ctypes.cast(ctypes.byref(arr.arr), ctypes.POINTER(Array))
        return ptr, None
    else:
        temp_array = _create_temp_array(arr)
        return ctypes.byref(temp_array), temp_array


def _create_result(result_struct, shape, name):
    if name == "Array":
        return PyArray(result_struct.contents, shape)
    else:
        data_ptr = ctypes.cast(result_struct.contents.data, ctypes.c_void_p)
        return PyArray(ggml_struct(data=data_ptr), shape)
