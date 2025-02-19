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

from collections.abc import Sequence
from functools import reduce
from types import GenericAlias
from typing import Any, Generic, TypeGuard

from ...backends.backend import Backend
from ...core import Constant, DataType, Dtype, data_types, epsilon_table
from ...utils.utils import BiMap
from ..common import (
    TBD,
    AllValueType,
    DataEvalType,
    IOHyperEdge,
    MainValueType,
    ScalarValueType,
    Tensor,
    ToBeDetermined,
    Updates,
)


class StaticDataStore(Generic[DataType]):
    def __init__(
        self,
        backend: Backend[DataType],
        inference: bool,
    ) -> None:
        self._all_data: dict[str, IOHyperEdge] = dict()
        self.data_memo: dict[int, IOHyperEdge] = dict()
        self.backend: Backend[DataType] = backend
        self.inference = inference
        self.intermediate_non_differentiables: BiMap[str, IOHyperEdge] = BiMap()
        self._runtime_static_keys: set[str] = set()
        self._unused_keys: set[str] = set()
        # Final tensor values of data store.
        # TODO: Constant types are not allowed in data_values but DataEvalType
        # includes it.
        self.data_values: DataEvalType[DataType] = dict()
        self.random_seeds: dict[str, int] = dict()

    @property
    def all_data(self) -> dict[str, IOHyperEdge]:
        return self._all_data

    @property
    def cached_data(self) -> DataEvalType[DataType]:
        return self.data_values

    @property
    def runtime_static_keys(self) -> set[str]:
        return self._runtime_static_keys

    @property
    def all_static_keys(self) -> set[str]:
        return self._runtime_static_keys | self.data_values.keys()

    @property
    def unused_keys(self) -> set[str]:
        return self._unused_keys

    @staticmethod
    def is_scalar_type(t: Any) -> TypeGuard[MainValueType]:
        if isinstance(t, tuple(data_types)):
            return False
        elif isinstance(t, list | tuple):
            return all(StaticDataStore.is_scalar_type(value) for value in t)
        elif isinstance(t, dict):
            return all(StaticDataStore.is_scalar_type(value) for value in t.values())
        return True

    @staticmethod
    def is_tensor_type(t: Any) -> TypeGuard[DataType]:
        return isinstance(t, tuple(data_types))

    def remove_key_from_store(
        self, key: str, label_as_unused: bool = True, hard_remove: bool = False
    ) -> None:
        if key in self.data_values:
            self.data_values.pop(key)  # type: ignore

        self._runtime_static_keys.discard(key)
        if key in self.intermediate_non_differentiables:
            self.intermediate_non_differentiables.pop(key)

        if key in self.random_seeds:
            self.random_seeds.pop(key)

        if label_as_unused:
            self._unused_keys.add(key)

        if hard_remove and key in self._all_data:
            self._all_data.pop(key)

    def update_cached_data(self, updated_data: Updates) -> set[str]:
        # If any data value is found by shape inference algorithms
        # transfer this data in cached_data.
        transferred_keys: set[str] = set()
        updated_inter_data = (
            updated_data.value_updates
            & self.intermediate_non_differentiables.inverse.keys()
        )
        for data in updated_inter_data:
            key = self.intermediate_non_differentiables.inverse[data]
            if key in self.data_values or data.value is not TBD:
                if key in self.data_values:
                    raise KeyError(
                        f"'{key}' key can not be an intermediate and cached key "
                        "at the same time!"
                    )
                if key not in self.data_values:
                    self._set_data_value(key, data)
                transferred_keys.add(key)
        for key in transferred_keys:
            self.intermediate_non_differentiables.pop(key)
        return transferred_keys

    def find_value_type(
        self, value: DataType | int | float | bool | Sequence[Any] | dict[str, Any]
    ) -> type | GenericAlias | type[Tensor[int | float | bool]]:
        if self.is_tensor_type(value):
            # Find tensor value type.
            tensor_type = self._infer_tensor_value_type(value)
            return Tensor[tensor_type]  # type: ignore
        elif isinstance(value, list | tuple):
            result = [self.find_value_type(v) for v in value]
            if isinstance(value, tuple):
                return tuple[*result]  # type: ignore
            else:
                return list[reduce(lambda x, y: x | y, result)]  # type: ignore
        elif isinstance(value, dict):
            value_types = [self.find_value_type(v) for v in value.values()]
            return dict[str, reduce(lambda x, y: x | y, value_types)]  # type: ignore
        else:
            return type(value)

    def convert_phys_value_to_logical(
        self, value: DataType | int | float | bool | Sequence[Any] | dict[str, Any]
    ) -> AllValueType | Tensor[int | float | bool]:
        if self.is_tensor_type(value):
            # Find tensor value type.
            tensor_type = self._infer_tensor_value_type(value)
            tensor_shape = value.shape
            tensor: Tensor[int | float | bool] = Tensor(type=tensor_type)
            tensor.shape.set_values(tensor_shape)
            return tensor
        elif isinstance(value, list | tuple):
            result = [self.convert_phys_value_to_logical(v) for v in value]
            if isinstance(value, tuple):
                return tuple(result)
            return result
        elif isinstance(value, dict):
            return {k: self.convert_phys_value_to_logical(v) for k, v in value.items()}
        return value

    def _convert_to_physical_value(
        self, data: AllValueType
    ) -> DataType | ScalarValueType | str:
        if isinstance(data, Tensor) and data.value is not TBD:
            data = self.backend.array(
                self._convert_to_physical_value(data.value)
                if isinstance(data.value, Constant)
                else data.value
            )
        elif isinstance(data, Tensor):
            raise ValueError("Tensor value is not set!")
        elif isinstance(data, list | tuple):
            result: list[Any] | tuple[Any, ...] = [
                self._convert_to_physical_value(d) for d in data
            ]
            if isinstance(data, tuple):
                result = tuple(result)
            data = result
        elif isinstance(data, dict):
            data = {k: self._convert_to_physical_value(v) for k, v in data.items()}
        elif isinstance(data, Constant):
            data = epsilon_table[self.backend.precision][data]
        elif isinstance(data, Dtype):
            data = getattr(self.backend, data.name)
        return data

    def _set_data_value(self, key: str, data: IOHyperEdge) -> None:
        value: AllValueType = data._value
        assert not isinstance(value, ToBeDetermined)
        try:
            phys_value = self._convert_to_physical_value(value)
            self.data_values[key] = phys_value
        except Exception as e:
            if str(e) == "Tensor value is not set!":
                return
            raise
        # # If value is a constant, get its corresponding value for
        # # the backend.
        # if isinstance(value, Constant):
        #     value = epsilon_table[self.backend.precision][value]

        # if data.is_tensor:
        #     value = self.backend.array(value)
        # elif isinstance(value, Dtype):
        #     value = getattr(self.backend, value.name)

        # self.data_values[key] = value  # type: ignore

    # Add constant values of given models __call__ to constant_keys if any.
    # TODO: merge convert_data_to_physical with _set_data_value
    @staticmethod
    def convert_data_to_physical(
        value: AllValueType, backend: Backend[DataType]
    ) -> DataType | AllValueType:
        match value:
            case Constant():
                value = epsilon_table[backend.precision][value]
            case Dtype():
                value = getattr(backend, value.name)
            case Tensor():
                value = backend.array(
                    StaticDataStore.convert_data_to_physical(value.value, backend)
                )
            case _:
                value = value
        return value

    def _infer_tensor_value_type(
        self, value: DataType
    ) -> type[bool] | type[int] | type[float]:
        val_type: type[bool] | type[int] | type[float]
        data_dtype = str(value.dtype)
        # Check value type is OK, and update type accordinly.
        if "bool" in data_dtype:
            val_type = bool
        elif "int" in data_dtype:
            val_type = int
        elif "float" in data_dtype:
            val_type = float
        else:
            raise TypeError(
                f"Given type ({data_dtype}) is not supported. "
                "Only float, int or bool types are accepted."
            )
        return val_type

    def set_random_seed_keys(self, seed_keys: set[str]) -> None:
        for key in seed_keys:
            if self.all_data[key].value == TBD:
                self.random_seeds[key] = 0
            else:
                value = self.all_data[key].value
                assert isinstance(value, int)
                self.random_seeds[key] = value

    def set_random_seed_values(self, **seed_mapping: int) -> None:
        for key, value in seed_mapping.items():
            if key not in self.random_seeds:
                raise KeyError(f"'{key}' key is not a random seed key!")
            self.random_seeds[key] = value
