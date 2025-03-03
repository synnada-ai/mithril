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

import builtins
import ctypes

from ..array import PyArray

NestedList = list[float | "NestedList"]

lib: ctypes.CDLL

def to_c_int_array(lst: list[int] | tuple[int, ...]) -> ctypes.Array[ctypes.c_int]: ...
def to_c_float_array(
    arr: list[float] | tuple[float, ...],
) -> ctypes.Array[ctypes.c_float]: ...

class Array(ctypes.Structure): ...

def empty(shape: tuple[builtins.int, ...] | list[builtins.int]) -> PyArray: ...
def ones(shape: tuple[builtins.int, ...] | list[builtins.int]) -> PyArray: ...
def zeros(shape: tuple[builtins.int, ...] | list[builtins.int]) -> PyArray: ...
