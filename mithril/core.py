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


from __future__ import annotations

import builtins as py_builtins
import enum
from enum import Enum
from typing import Any, Generic, TypeGuard, TypeVar


class Constant(Enum):
    EPSILON = 0
    LEFT_EPSILON = 1
    MIN_POSITIVE_NORMAL = 2
    MIN_POSITIVE_SUBNORMAL = 3
    STABLE_RECIPROCAL_THRESHOLD = 4


constant_type_table = {
    Constant.EPSILON: float,
    Constant.LEFT_EPSILON: float,
    Constant.MIN_POSITIVE_NORMAL: float,
    Constant.MIN_POSITIVE_SUBNORMAL: float,
    Constant.STABLE_RECIPROCAL_THRESHOLD: float,
}

epsilon_table: dict[py_builtins.int, dict[Constant, py_builtins.float | None]] = {
    64: {
        Constant.EPSILON: 2 ** (-52),
        Constant.LEFT_EPSILON: 2
        ** (
            -53
        ),  # Machine epsilon to the left of unity (i.e. 1 - epsilon / 2 != 1.0). # noqa E501
        Constant.MIN_POSITIVE_NORMAL: 2.2250738585072014e-308,  # Minimum positive normal floating-point value with a "finite reciprocal". Can be obtained using np.finfo(np.float64).tiny. # noqa E501
        Constant.MIN_POSITIVE_SUBNORMAL: 5e-324,  # Can be obtained using np.nexafter(0, 1.0). # noqa E501
        Constant.STABLE_RECIPROCAL_THRESHOLD: 1.4916681462400413e-153,
    },
    32: {
        Constant.EPSILON: 1.1920929e-07,  # We don't prefer to write as 2 ** (-23) because it can cause problems on 64 bits environments. # noqa E501
        Constant.LEFT_EPSILON: 5.9604645e-08,  # Machine epsilon to the left of unity (i.e. 1 - epsilon / 2 != 1.0). # noqa E501
        Constant.MIN_POSITIVE_NORMAL: 1.1754944e-38,  # Minimum positive normal floating-point value with a "finite reciprocal". Can be obtained using np.finfo(np.float32).tiny. # noqa E501
        Constant.MIN_POSITIVE_SUBNORMAL: 1e-45,  # Can be obtained using np.nexafter(0, 1.0, dtype = np.float32). # noqa E501
        Constant.STABLE_RECIPROCAL_THRESHOLD: 1.0842021951e-19,
    },
    16: {
        Constant.EPSILON: 0.000977,  # We don't prefer to write as 2 ** (-10) because it can cause problems on 64 bits environments. # noqa E501
        Constant.LEFT_EPSILON: 0.0004885,  # Machine epsilon to the left of unity (i.e. 1 - epsilon / 2 != 1.0). # noqa E501
        Constant.MIN_POSITIVE_NORMAL: 6.104e-05,  # Minimum positive normal floating-point value with a "finite reciprocal". Can be obtained using np.finfo(np.float16).tiny. # noqa E501
        Constant.MIN_POSITIVE_SUBNORMAL: 6e-08,  # Can be obtained using np.nexafter(0, 1.0, dtype = np.float16). # noqa E501
        Constant.STABLE_RECIPROCAL_THRESHOLD: 0.007812809993849843,
    },
}


class Dtype(enum.IntEnum):  # noqa N801
    int8 = 0
    int16 = 1
    int32 = 2
    int64 = 3
    float16 = 4
    float32 = 5
    float64 = 6
    bool = 7


int16: Dtype = Dtype.int16
short = int16
int32: Dtype = Dtype.int32
int = int32
int64: Dtype = Dtype.int64
long = int64
float16: Dtype = Dtype.float16
half = float16
float32: Dtype = Dtype.float32
float = float32
float64: Dtype = Dtype.float64
double = float64
bool: Dtype = Dtype.bool

data_types: list[type] = []

try:
    from numpy import ndarray

    data_types.append(ndarray)
except ImportError:
    pass

try:
    from jax import Array

    data_types.append(Array)
except ImportError:
    pass

try:
    from torch import Tensor

    data_types.append(Tensor)
except ImportError:
    pass

try:
    from mlx.core import array

    data_types.append(array)
except ImportError:
    pass

try:
    from mithril.backends.with_manualgrad.c_backend.src.array import PyArray

    data_types.append(PyArray)
except ImportError:
    pass


DataType = TypeVar("DataType", "ndarray", "Array", "Tensor", "PyArray", "array")


class GenericDataType(Generic[DataType]):
    @staticmethod
    def is_tensor_type(t: Any) -> TypeGuard[DataType]:
        return isinstance(t, tuple(data_types))
