import ctypes
import os
from ....cores.c.array import PyArray
from ....cores.c.raw_c.array import (
    Array,
    zeros,
    lib
)
from ....backends.with_manualgrad.c_backend.backend import array
from ....backends.with_manualgrad.c_backend.utils import from_numpy
from ....cores.c.ggml.ggml_core import ggml_struct
import numpy as np

__all__ = [
    "add",
    "multiplication"
]

def add(
    left: PyArray,
    right: PyArray
) -> PyArray:
    # In C backend, output is given as first input
    output = zeros(left.shape)
    left_c_1_ptr = ctypes.cast(left.arr.data, ctypes.POINTER(ctypes.c_float))
    left_c = PyArray(Array(data=left_c_1_ptr), left.shape)
    right_c_1_ptr = ctypes.cast(right.arr.data, ctypes.POINTER(ctypes.c_float))
    right_c = PyArray(Array(data=right_c_1_ptr), right.shape)
    lib.add(ctypes.byref(output.arr), ctypes.byref(left_c.arr), ctypes.byref(right_c.arr))
    _shape = output.shape
    data_ptr = ctypes.cast(output.arr.data, ctypes.c_void_p)
    return PyArray(ggml_struct(data=data_ptr), _shape)

def multiplication(
    left: PyArray,
    right: PyArray
) -> PyArray:
    # In C backend, output is given as first input
    output = zeros(left.shape)
    left_c_1_ptr = ctypes.cast(left.arr.data, ctypes.POINTER(ctypes.c_float))
    left_c = PyArray(Array(data=left_c_1_ptr), left.shape)
    right_c_1_ptr = ctypes.cast(right.arr.data, ctypes.POINTER(ctypes.c_float))
    right_c = PyArray(Array(data=right_c_1_ptr), right.shape)
    lib.multiplication(ctypes.byref(output.arr), ctypes.byref(left_c.arr), ctypes.byref(right_c.arr))
    _shape = output.shape
    data_ptr = ctypes.cast(output.arr.data, ctypes.c_void_p)
    return PyArray(ggml_struct(data=data_ptr), _shape)



primitive_func_dict = {key: fn for key, fn in globals().items() if callable(fn)}