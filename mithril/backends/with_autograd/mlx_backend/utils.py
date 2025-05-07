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

from collections.abc import Callable, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from .... import types
from ....common import PythonGenConfig, find_dominant_type
from ....cores.python.mlx.utils import dtype_map
from ...utils import DtypeSubTypes

CODEGEN_CONFIG = PythonGenConfig(SPECIFY_DEVICE=True)
JITABLE: dict[str, Callable[..., bool]] = {
    "default": lambda *args, **kwargs: True,
}

ArrayType = mx.array


def get_available_devices() -> list[str]:
    # For now available devices static
    return ["cpu", "mps"]


def get_device(device: str) -> mx.Device:
    if device == "mps":
        device = "gpu"
    return mx.Device(getattr(mx, device), 0)


def handle_data_precision(data: mx.array, precision: int) -> mx.array:
    _dtype = data.dtype
    # Do not make any changes to boolean types.
    if _dtype != mx.bool_:  # type: ignore
        if "int" in str(_dtype) and _dtype != getattr(mx, f"int{precision}"):
            data = data.astype(getattr(mx, f"int{precision}"))
        elif "float" in str(_dtype) and _dtype != getattr(mx, f"float{precision}"):
            data = data.astype(getattr(mx, f"float{precision}"))
    return data


def handle_data_dtype(data: mx.array, dtype: types.Dtype | int) -> mx.array:
    dtype = types.Dtype(dtype)

    if data.dtype != dtype_map[dtype.name]:
        return data.astype(dtype_map[dtype.name])
    return data


def determine_dtype(
    input: Any, dtype: types.Dtype | None, default_type: types.Dtype, precision: int
) -> str:
    if isinstance(dtype, types.Dtype):
        return dtype.name

    if isinstance(input, mx.array):
        dtype_name = "".join(
            char for char in input.dtype.__str__().split(".")[-1] if not char.isdigit()
        )
    elif isinstance(input, (np.ndarray | np.generic)):
        dtype_name = "".join(char for char in str(input.dtype) if not char.isdigit())
    else:
        dtype_name = find_dominant_type(input).__name__

    if dtype_name == "float":
        dtype_name = DtypeSubTypes[default_type.name].value

    return dtype_name + str(precision) if dtype_name != "bool" else "bool"


def get_type(input: int | float | bool | Sequence[Any], precision: int) -> mx.Dtype:
    type = find_dominant_type(input).__name__
    if type == "bool":
        return mx.bool_  # type: ignore

    return getattr(mx, type + str(precision))
