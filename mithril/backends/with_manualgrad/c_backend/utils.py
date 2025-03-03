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

from typing import Any

import numpy as np

from ....common import CGenConfig
from ....cores.c.raw_c.array import (
    Array,
    PyArray,
    lib,
    to_c_float_array,
    to_c_int_array,
)

CODEGEN_CONFIG = CGenConfig()

# File configs
CODEGEN_CONFIG.HEADER_NAME = "cbackend.h"

# Array configs
CODEGEN_CONFIG.ARRAY_NAME = "Array"

# Function configs
CODEGEN_CONFIG.RETURN_OUTPUT = False
CODEGEN_CONFIG.USE_OUTPUT_AS_INPUT = True

# Memory Management configs
CODEGEN_CONFIG.ALLOCATE_INTERNALS = True


def to_numpy(array: PyArray) -> np.ndarray[Any, Any]:
    return np.ctypeslib.as_array(array.arr.contents.data, shape=(array.shape))


def from_numpy(array: np.ndarray[Any, Any]) -> PyArray:
    shape = array.shape
    ndim = len(shape)

    c_shape = to_c_int_array(shape)
    c_data = to_c_float_array(array)  # type: ignore
    arr: Array = lib.create_struct(c_data, ndim, c_shape)
    return PyArray(arr, shape)
