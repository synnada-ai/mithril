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

from .array import Array, PyArray, lib, to_c_float_array, to_c_int_array


def to_numpy(array: PyArray):
    return np.ctypeslib.as_array(array.arr.contents.data, shape=(array.shape))


def from_numpy(array: np.ndarray):
    shape = array.shape
    ndim = len(shape)

    c_shape = to_c_int_array(shape)
    c_data = to_c_float_array(array)
    arr: Array = lib.create_struct(c_data, ndim, c_shape)
    return PyArray(arr, shape)  # type: ignore
