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

from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np

from .... import types
from ....common import PythonGenConfig, find_dominant_type
from ....cores.python.jax.utils import dtype_map
from ...utils import DtypeSubTypes

CODEGEN_CONFIG = PythonGenConfig(SPECIFY_DEVICE=True)
JITABLE: dict[str, Callable[..., bool]] = {
    "default": lambda *args, **kwargs: True,
    "tensor_to_list": lambda *args, **kwargs: False,
    "item": lambda *args, **kwargs: False,
    "unique": lambda *args, **kwargs: False,
}


ArrayType = jax.Array


def handle_data_precision(data: ArrayType, precision: int) -> ArrayType:
    _dtype = data.dtype
    # Do not make any changes to boolean types.
    if _dtype != jax.numpy.bool_:
        if jax.numpy.issubdtype(_dtype, jax.numpy.integer) and _dtype != getattr(
            jax.numpy, f"int{precision}"
        ):
            data = data.astype(f"int{precision}")
        elif jax.numpy.issubdtype(_dtype, jax.numpy.floating) and _dtype != getattr(
            jax.numpy, f"float{precision}"
        ):
            data = data.astype(f"float{precision}")
    return data


def handle_data_dtype(data: jax.Array, dtype: types.Dtype | int) -> jax.Array:
    dtype = types.Dtype(dtype)

    if data.dtype != dtype_map[dtype.name]:
        return data.astype(dtype_map[dtype.name])
    return data


def get_type(
    input: int | float | bool | Sequence[Any], precision: int
) -> jax.numpy.dtype[Any]:
    type = find_dominant_type(input).__name__
    if type == "bool":
        return jax.numpy.bool_

    return getattr(jax.numpy, type + str(precision))


def determine_dtype(
    input: Any, dtype: types.Dtype | None, default_dtype: types.Dtype, precision: int
) -> str:
    if isinstance(dtype, types.Dtype):
        return dtype.name

    if isinstance(input, jax.Array):
        dtype_name = "".join(
            char for char in input.dtype.__str__() if not char.isdigit()
        )
    elif isinstance(input, (np.ndarray | np.generic)):
        dtype_name = "".join(char for char in str(input.dtype) if not char.isdigit())
    else:
        dtype_name = find_dominant_type(input).__name__

    if dtype_name == "float":
        dtype_name = DtypeSubTypes[default_dtype.name].value

    return dtype_name + str(precision) if dtype_name != "bool" else "bool"


# JITABLITY RULES


def is_tensor_slice_jitable(
    *input_types: tuple[bool, type | tuple[type, ...]],
) -> bool:
    # Tensor slice operation with boolean types is not supported in JAX.
    for is_tensor, dtypes in input_types:
        is_bool_exists = bool in dtypes if isinstance(dtypes, tuple) else bool is dtypes
        if not is_tensor and is_bool_exists:
            return False
    return True


JITABLE["primitive_slice"] = is_tensor_slice_jitable
