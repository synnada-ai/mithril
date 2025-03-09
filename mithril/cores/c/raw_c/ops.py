import ctypes
import os
from ....cores.c.array import PyArray
from ....cores.c.raw_c.array import (
    Array,
    zeros,
    lib
)
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
    lib.add(ctypes.byref(output.arr), ctypes.byref(left.arr), ctypes.byref(right.arr))
    return output

def multiplication(
    left: PyArray,
    right: PyArray
) -> PyArray:
    # In C backend, output is given as first input
    output = zeros(left.shape)
    lib.multiplication(ctypes.byref(output.arr), ctypes.byref(left.arr), ctypes.byref(right.arr))
    return output

primitive_func_dict = {key: fn for key, fn in globals().items() if callable(fn)}