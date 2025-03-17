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
from .rawc_definitions import Array, lib
import ctypes


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
            return [reshape(data[i * slice_size:(i + 1) * slice_size], shape[1:])
                    for i in range(shape[0])]
        return reshape(data_list, self.shape)

    def __repr__(self):
        return f"array({self.data})"

    def __str__(self):
        return f"PyArray(shape={self.shape})\n{self.data}"
    
    # Element-wise addition
    def __add__(self, other):
        if isinstance(other, PyArray):
            if self.ndim > other.ndim:
                ndim = self.ndim
                shape = tuple(self.shape[i] for i in range(ndim))
            else:
                ndim = other.ndim
                shape = tuple(other.shape[i] for i in range(ndim))
            c_shape = (ctypes.c_int * len(shape))(*shape)
            result = lib.create_empty_struct(len(c_shape), c_shape)
            self_ptr = ctypes.cast(ctypes.byref(self.arr), ctypes.POINTER(Array))
            other_ptr = ctypes.cast(ctypes.byref(other.arr), ctypes.POINTER(Array))
            lib.add(result,  self_ptr,  other_ptr)
            return PyArray(result.contents, shape)
        elif isinstance(other, Real):
            # Scalar addition
            c_shape = (ctypes.c_int * len(self.shape))(*self.shape)
            result = lib.create_empty_struct(len(c_shape), c_shape)
            self_ptr = ctypes.cast(ctypes.byref(self.arr), ctypes.POINTER(Array))
            lib.scalar_add(result, self_ptr, ctypes.c_float(float(other)))
            return PyArray(result.contents, self.shape)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)
    
    # Element-wise and scalar multiplication
    def __mul__(self, other):
        if isinstance(other, PyArray):
            shape = self.shape if self.ndim >= other.ndim else other.shape
            c_shape = (ctypes.c_int * len(shape))(*shape)
            result = lib.create_empty_struct(len(c_shape), c_shape)
            self_ptr = ctypes.cast(ctypes.byref(self.arr), ctypes.POINTER(Array))
            other_ptr = ctypes.cast(ctypes.byref(other.arr), ctypes.POINTER(Array))
            lib.multiplication(result, self_ptr, other_ptr)
            return PyArray(result.contents, shape)
        elif isinstance(other, Real):
            c_shape = (ctypes.c_int * len(self.shape))(*self.shape)
            result = lib.create_empty_struct(len(c_shape), c_shape)
            self_ptr = ctypes.cast(ctypes.byref(self.arr), ctypes.POINTER(Array))
            lib.scalar_multiply(result, self_ptr, ctypes.c_float(float(other)))
            return PyArray(result.contents, self.shape)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)
    
    # Element-wise and scalar subtraction
    def __sub__(self, other):
        if isinstance(other, PyArray):
            shape = self.shape if self.ndim >= other.ndim else other.shape
            c_shape = (ctypes.c_int * len(shape))(*shape)
            result = lib.create_empty_struct(len(c_shape), c_shape)
            self_ptr = ctypes.cast(ctypes.byref(self.arr), ctypes.POINTER(Array))
            other_ptr = ctypes.cast(ctypes.byref(other.arr), ctypes.POINTER(Array))
            lib.subtract(result, self_ptr, other_ptr)
            return PyArray(result.contents, shape)
        elif isinstance(other, Real):
            c_shape = (ctypes.c_int * len(self.shape))(*self.shape)
            result = lib.create_empty_struct(len(c_shape), c_shape)
            self_ptr = ctypes.cast(ctypes.byref(self.arr), ctypes.POINTER(Array))
            lib.scalar_subtract(result, self_ptr, ctypes.c_float(float(other)))
            return PyArray(result.contents, self.shape)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Real):
            # scalar - array = scalar + (-1 * array)
            return other + (-1 * self)
        else:
            return NotImplemented
