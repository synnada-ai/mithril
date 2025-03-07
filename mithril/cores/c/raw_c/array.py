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

from ..array import PyArray

current_file_path = os.path.abspath(__file__)

lib = ctypes.CDLL(os.path.join(os.path.dirname(current_file_path), "libmithrilc.so"))


class Array(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_int),
    ]


lib.create_struct.restype = ctypes.POINTER(Array)
lib.create_struct.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

lib.create_empty_struct.restype = ctypes.POINTER(Array)
lib.create_empty_struct.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

lib.create_full_struct.restype = ctypes.POINTER(Array)
lib.create_full_struct.argtypes = [
    ctypes.c_float,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

lib.delete_struct.argtypes = [ctypes.POINTER(Array)]


def to_c_int_array(lst: list[int] | tuple[int, ...]) -> ctypes.Array[ctypes.c_int]:
    return (ctypes.c_int * len(lst))(*lst)


def to_c_float_array(
    arr: list[float] | tuple[float, ...] | np.ndarray[Any, Any],
) -> ctypes.Array[ctypes.c_float]:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def empty(shape: tuple[int, ...] | list[int]):
    c_shape = to_c_int_array(shape)
    arr = lib.create_empty_struct(len(shape), c_shape).contents
    return PyArray(arr, shape)


def ones(shape: tuple[int, ...] | list[int]):
    c_shape = to_c_int_array(shape)
    arr = lib.create_full_struct(1.0, len(shape), c_shape).contents
    return PyArray(arr, shape)


def zeros(shape: tuple[int, ...] | list[int]):
    c_shape = to_c_int_array(shape)
    arr = lib.create_full_struct(0.0, len(shape), c_shape).contents
    return PyArray(arr, shape)
