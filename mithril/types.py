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

from .cores.core import DataType, GenericDataType, data_types

__all__ = [
    "DataType",
    "GenericDataType",
    "Constant",
    "Dtype",
    "uint8",
    "int8",
    "int16",
    "short",
    "int32",
    "int",
    "int64",
    "long",
    "float16",
    "bfloat16",
    "float32",
    "float",
    "float64",
    "double",
    "bool",
    "constant_type_table",
    "epsilon_table",
    "data_types",
]


class Constant(Enum):
    EPSILON = 0
    LEFT_EPSILON = 1
    MIN_POSITIVE_NORMAL = 2
    MIN_POSITIVE_SUBNORMAL = 3
    STABLE_RECIPROCAL_THRESHOLD = 4


constant_type_table = {
    Constant.EPSILON: py_builtins.float,
    Constant.LEFT_EPSILON: py_builtins.float,
    Constant.MIN_POSITIVE_NORMAL: py_builtins.float,
    Constant.MIN_POSITIVE_SUBNORMAL: py_builtins.float,
    Constant.STABLE_RECIPROCAL_THRESHOLD: py_builtins.float,
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
    uint8 = 0
    int8 = 1
    int16 = 2
    int32 = 3
    int64 = 4
    float16 = 5
    bfloat16 = 6
    float32 = 7
    float64 = 8
    bool = 9


uint8: Dtype = Dtype.uint8
int8: Dtype = Dtype.int8
int16: Dtype = Dtype.int16
short = int16
int32: Dtype = Dtype.int32
int = int32
int64: Dtype = Dtype.int64
long = int64
float16: Dtype = Dtype.float16
half = float16
bfloat16: Dtype = Dtype.bfloat16
float32: Dtype = Dtype.float32
float = float32
float64: Dtype = Dtype.float64
double = float64
bool: Dtype = Dtype.bool
