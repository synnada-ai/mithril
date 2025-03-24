import ctypes
import os

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

lib.scalar_add.restype = None
lib.scalar_add.argtypes = [ctypes.POINTER(Array), ctypes.POINTER(Array), ctypes.c_float]

lib.scalar_multiply.restype = None
lib.scalar_multiply.argtypes = [
    ctypes.POINTER(Array),
    ctypes.POINTER(Array),
    ctypes.c_float,
]

lib.multiplication.restype = None
lib.multiplication.argtypes = [
    ctypes.POINTER(Array),
    ctypes.POINTER(Array),
    ctypes.POINTER(Array),
]

lib.subtract.restype = None
lib.subtract.argtypes = [
    ctypes.POINTER(Array),
    ctypes.POINTER(Array),
    ctypes.POINTER(Array),
]

lib.scalar_subtract.restype = None
lib.scalar_subtract.argtypes = [
    ctypes.POINTER(Array),
    ctypes.POINTER(Array),
    ctypes.c_float,
]
