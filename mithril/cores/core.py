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


from typing import Any, Generic, TypeGuard, TypeVar

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
    from mithril.cores.c.raw_c.array import PyArray

    data_types.append(PyArray)
except ImportError:
    pass


DataType = TypeVar(
    "DataType", "ndarray[Any, Any]", "Array", "Tensor", "PyArray", "array"
)


class GenericDataType(Generic[DataType]):
    @staticmethod
    def is_tensor_type(t: Any) -> TypeGuard[DataType]:
        return isinstance(t, tuple(data_types))
