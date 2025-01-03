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

import enum
from collections.abc import Sequence

from ..utils.type_utils import is_tuple_int

NestedFloatOrIntOrBoolList = float | int | bool | list["NestedFloatOrIntOrBoolList"]


def process_shape(
    shape: tuple[int | tuple[int, ...] | list[int], ...],
) -> tuple[int, ...] | list[int]:
    _shape: list[int] | tuple[int, ...]
    if is_tuple_int(shape):
        _shape = shape
    elif len(shape) == 1 and isinstance(shape[0], Sequence):
        _shape = shape[0]
    else:
        for item in shape:
            if not isinstance(item, int):
                raise TypeError(
                    f"Provided shape {shape} is not supported."
                    " Failed to handle the object at pos {idx+1}"
                )

    return _shape


class DtypeBits(enum.IntEnum):
    bool = 8
    int8 = 8
    int16 = 16
    int32 = 32
    int64 = 64
    float16 = 16
    bfloat16 = 16
    float32 = 32
    float64 = 64


class DtypeSubTypes(enum.Enum):
    bool = "bool"
    int8 = "int"
    int16 = "int"
    int32 = "int"
    int64 = "int"
    float16 = "float"
    bfloat16 = "bfloat"
    float32 = "float"
    float64 = "float"
