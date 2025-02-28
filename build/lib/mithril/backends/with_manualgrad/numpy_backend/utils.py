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

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .... import types
from ....common import find_dominant_type
from ....cores.python.numpy.utils import dtype_map
from ...utils import DtypeSubTypes

CODEGEN_CONFIG: dict[str, bool] = {
    "specify_device": False,
}
ArrayType = np.ndarray


def find_label_indices(
    input_array: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    return (np.arange(len(input_array)), input_array.T)


def handle_data_precision(
    data: np.ndarray[Any, Any], precision: int
) -> np.ndarray[Any, Any]:
    if isinstance(data, float | int):
        return data
    _dtype = data.dtype
    # Do not make any changes to boolean types.
    if _dtype != np.bool_:
        if np.issubdtype(_dtype, np.integer) and _dtype != getattr(
            np, f"int{precision}"
        ):
            data = data.astype(f"int{precision}")
        elif np.issubdtype(_dtype, np.floating) and _dtype != getattr(
            np, f"float{precision}"
        ):
            data = data.astype(f"float{precision}")
    return data


def handle_data_dtype(
    data: np.ndarray[Any, Any], dtype: types.Dtype | int
) -> np.ndarray[Any, Any]:
    dtype = types.Dtype(dtype)

    if data.dtype != dtype_map[dtype.name]:
        return data.astype(dtype_map[dtype.name])
    return data


def get_type(
    input: int | float | bool | Sequence[int | float | bool | Sequence[Any]],
    precision: int,
) -> np.dtype[Any]:
    type = find_dominant_type(input).__name__
    if type == "bool":
        return np.bool_  # type: ignore

    return getattr(np, type + str(precision))


def verify_shapes(
    inputs: tuple[np.ndarray[Any, Any], ...],
    idx: int,
    non_differentiables: Iterable[int] | None = None,
) -> None:
    if idx >= len(inputs):
        raise Exception(f"Gradient is not defined for the input at index {idx}!")
    if non_differentiables is not None and idx in non_differentiables:
        raise Exception(f"Given key at index {idx} is not differentiable!")


def determine_dtype(
    input: Any, dtype: types.Dtype | None, default_dtype: types.Dtype, precision: int
) -> str:
    if isinstance(dtype, types.Dtype):
        return dtype.name

    if isinstance(input, (np.ndarray | np.generic)):
        dtype_name = "".join(char for char in str(input.dtype) if not char.isdigit())
    else:
        dtype_name = find_dominant_type(input).__name__

    if dtype_name == "float":
        dtype_name = DtypeSubTypes[default_dtype.name].value

    return dtype_name + str(precision) if dtype_name != "bool" else "bool"
