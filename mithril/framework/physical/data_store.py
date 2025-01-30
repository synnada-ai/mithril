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

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Generic, TypeGuard

from ...backends.backend import Backend
from ...core import Constant, DataType, Dtype, data_types, epsilon_table
from ...utils.func_utils import is_make_array_required, prepare_function_args
from ...utils.utils import BiMap
from ..common import (
    TBD,
    AllValueType,
    ConstraintSolver,
    DataEvalType,
    IOHyperEdge,
    MainValueInstance,
    MainValueType,
    Tensor,
    ToBeDetermined,
    Updates,
    is_type_adjustment_required,
)
from ..logical.model import Connection
from .flat_graph import FlatGraph


class StaticDataStore(Generic[DataType]):
    def __init__(
        self,
        graph: FlatGraph[DataType],
        backend: Backend[DataType],
        inference: bool,
        solver: ConstraintSolver,
        memo: dict[int, IOHyperEdge] | None = None,
    ) -> None:
        if memo is None:
            memo = {}

        self._all_data: dict[str, IOHyperEdge] = dict()
        self.data_memo: dict[int, IOHyperEdge] = dict()
        self.graph: FlatGraph[DataType] = graph
        self.backend: Backend[DataType] = backend
        self.inference = inference
        self.intermediate_non_differentiables: BiMap[str, IOHyperEdge] = BiMap()
        self._runtime_static_keys: set[str] = set()
        self._unused_keys: set[str] = set()
        # Final tensor values of data store.
        # TODO: Constant types are not allowed in data_values but DataEvalType
        # includes it.
        self.data_values: DataEvalType[DataType] = dict()
        self.constraint_solver: ConstraintSolver = deepcopy(solver, memo=memo)
        self._random_seeds: dict[str, int] = dict()

    @property
    def all_data(self) -> dict[str, IOHyperEdge]:
        return self._all_data

    @property
    def cached_data(self) -> DataEvalType[DataType]:
        return self.data_values

    @property
    def random_seeds(self) -> dict[str, int]:
        return self._random_seeds

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
        return not isinstance(t, tuple(data_types))

    def remove_keys_from_store(self, keys: set[str]) -> None:
        for key in keys:
            hard_remove = key not in self.graph.output_keys
            self.remove_key_from_store(
                key, label_as_unused=False, hard_remove=hard_remove
            )

    def remove_key_from_store(
        self, key: str, label_as_unused: bool = True, hard_remove: bool = False
    ) -> None:
        if key in self.data_values:
            self.data_values.pop(key)  # type: ignore

        self._runtime_static_keys.discard(key)
        if key in self.intermediate_non_differentiables:
            self.intermediate_non_differentiables.pop(key)

        if key in self._random_seeds:
            self._random_seeds.pop(key)

        if label_as_unused:
            self._unused_keys.add(key)
            self._clear_constraints(key)

        # Finally delete data value or data of removed key from all_data attribute
        # taking hard_remove flag into account.
        if hard_remove:
            self._all_data.pop(key)
            self._clear_constraints(key)

    def _clear_constraints(self, key: str) -> None:
        if key not in self._all_data:
            return

        shape_constraints = self._all_data[key].shape_constraints
        type_constraints = self._all_data[key].type_constraints
        for source_key in self.graph.get_source_keys(key):
            if source_key in self._all_data:
                self._all_data[source_key].shape_constraints -= shape_constraints
                self._all_data[source_key].type_constraints -= type_constraints

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

    def _set_data_value(self, key: str, data: IOHyperEdge) -> None:
        value: DataType | AllValueType = data.value
        assert not isinstance(value, ToBeDetermined)
        # If value is a constant, get its corresponding value for
        # the backend.
        if isinstance(value, Constant):
            value = epsilon_table[self.backend.precision][value]

        if data.edge_type is Tensor:
            value = self.backend.array(value)
        elif isinstance(value, Dtype):
            value = getattr(self.backend, value.name)
        self.data_values[key] = value  # type: ignore

    def infer_unused_keys(self, key: str) -> None:
        # Infers unused keys when "key" is set as static.
        output_keys = self.graph.output_keys
        queue = set(self.graph.get_source_keys(key, True))
        while queue:
            source_key = queue.pop()
            all_static_keys = self.all_static_keys
            if source_key not in self.unused_keys and all(
                [
                    item in all_static_keys | self.unused_keys
                    for item in self.graph.get_target_keys(source_key)
                ]
            ):
                if source_key not in output_keys and set(
                    self.graph.get_target_keys(source_key, True)
                ).issubset(self._unused_keys | self.cached_data.keys()):
                    self.remove_key_from_store(source_key)

                queue |= set(
                    self.graph.get_source_keys(source_key, True)
                    if source_key in self.graph.connections
                    else []
                )

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

    def set_shapes(
        self,
        shapes: Mapping[str, Sequence[int | None]]
        | Mapping[Connection, Sequence[int | None]]
        | Mapping[str | Connection, Sequence[int | None]],
    ) -> None:
        updates = Updates()
        for key, value in shapes.items():
            if isinstance(key, Connection):
                key = key.key
            assert isinstance(key, str)
            if (data := self._all_data[key]).edge_type is not Tensor:
                raise ValueError("Non-tensor data can not have shape!")
            assert data.shape is not None
            updates |= data.shape.set_values(value)
        self.constraint_solver(updates)
        # Some intermediate values may be calculated, update cached data.
        new_statics = self.update_cached_data(updates)
        for key in new_statics:
            self.infer_unused_keys(key)

    def update_data(self, data: dict[str, IOHyperEdge]) -> None:
        if data.keys() & self._all_data.keys():
            raise Exception("Some keys are already in data store!")
        self._all_data |= data
        for key, value in data.items():
            if not value.is_non_diff:
                continue

            # Distribute non-differentiable keys into 3 attributes using
            # type of values. If a key has a definite value, add it into cached_data.
            # If key ends with "_cache", backend is manual and it does not appear as
            # an input to the model, then directly add it into the cached_data because
            # these keys are internally created cache keys for the corresponding
            # primitive functions.
            if (
                self.backend.is_manualgrad
                and key.endswith("_cache")
                and key not in self.graph.input_keys
            ) or (key in self.graph.input_keys and value.value is not TBD):
                self._set_data_value(key, value)
            elif key in self.graph.input_keys:
                self._runtime_static_keys.add(key)
            else:
                if value.value is not TBD:
                    self._set_data_value(key, value)
                else:
                    self.intermediate_non_differentiables[key] = value

    def set_static_keys(
        self,
        static_keys: dict[str, DataType | MainValueType],
    ) -> Updates:
        updates = Updates()
        for key, value in static_keys.items():
            if key not in self.graph.input_keys:
                raise KeyError(
                    "Requires static key to be in the input keys of the model!"
                )
            if (self._all_data[key].edge_type is Tensor) and not isinstance(
                value, ToBeDetermined | self.backend.get_backend_array_type()
            ):
                raise ValueError(
                    "Requires given arrays to be of same type with given backend!"
                )
            _, _updates = self.add_static_data(key, value)
            updates |= _updates
        return updates

    def add_static_data(
        self, key: str, value: DataType | MainValueType
    ) -> tuple[set[str], Updates]:
        updates = Updates()
        updated_keys = {key}
        if key in self.data_values:
            raise ValueError(
                f"Statically given key: {key} has been already set as static "
                "with a value!"
            )
        if key in self.unused_keys:
            raise ValueError(
                f"Given '{key}' key is unused for the model, no need to provide "
                "data for it."
            )
        else:
            if (data := self._all_data.get(key, None)) is None:
                raise KeyError(f"'{key}' key not found in model!")

            if self.is_scalar_type(value):  # TODO: Is this check really required?
                updates |= data.set_value(value)
            else:
                assert not isinstance(value, MainValueInstance | ToBeDetermined)
                # Find type of tensor and set.
                val_type = self._infer_tensor_value_type(value)
                updates |= data.set_type(Tensor[val_type])  # type: ignore
                assert data.shape is not None
                # Find shape of tensor and set.
                shape = list(value.shape)
                updates |= data.shape.set_values(shape)
            self.data_values[key] = value  # type: ignore
            self.intermediate_non_differentiables.pop(key, None)
            if (
                key not in self.intermediate_non_differentiables
                and key in self.runtime_static_keys
            ):
                self._runtime_static_keys.remove(key)
        # Finally update cached_data, infer unused keys and
        # return newly added static keys.
        self.constraint_solver(updates)
        statics = self.update_cached_data(updates) | updated_keys
        for static in statics:
            self.infer_unused_keys(static)

        return statics, updates

    def infer_static_keys(self) -> Updates:
        """Infers the static keys and calculates
        the static values during the inference.
        """
        statics = self.data_values
        queue = set(statics.keys())
        updates = Updates()
        while queue:
            key = queue.pop()
            if (key not in self.graph.all_source_keys) or key in self.unused_keys:
                continue

            for value in self.graph.get_target_keys(key):
                # Value is already in statics or unused keys, then skip.
                if value in (statics.keys() | self.unused_keys):
                    continue

                value_mapping = self.graph.get_source_keys(value)

                # To infer a model, all of its input keys should be in statics.
                if not set(value_mapping).issubset(statics.keys()):
                    continue

                model = self.graph.get_model(value)

                # TODO: Move this outside of while loop
                # after CBackend is completely implemented.
                fn_dict = (
                    self.backend.primitive_function_dict
                    | self.backend.registered_primitives
                )

                static_value: DataType | MainValueType
                assert model.formula_key is not None
                fn = fn_dict[model.formula_key]

                # Orginize args and kwargs
                local_input_keys = list(model.input_keys)
                if self.backend.is_manualgrad:
                    local_input_keys.append("cache")
                inputs = {
                    key: value
                    for key, value in zip(local_input_keys, value_mapping, strict=False)
                }
                args_dict, kwargs_dict = prepare_function_args(
                    self.data_values,
                    fn,
                    inputs,
                    self.backend.array_creation_funcs,
                    False,
                )
                args = [
                    self.data_values[arg_key]
                    for arg_keys in args_dict.values()
                    for arg_key in arg_keys
                ]
                kwargs = {
                    key: self.data_values[value] for key, value in kwargs_dict.items()
                }

                # If function needs backend specific args
                if model.formula_key in self.backend.array_creation_funcs:
                    kwargs["default_dtype"] = self.backend._dtype.name
                    if self.backend.codegen_config["specify_device"]:
                        kwargs["device"] = self.backend.get_device()

                static_value = fn(*args, **kwargs)

                # Check astype needed
                if self.backend.is_manualgrad and is_type_adjustment_required(
                    self.all_data, value_mapping
                ):
                    static_value = self.backend.array(static_value)

                if self.backend.is_manualgrad:
                    data = self._all_data[value]
                    if is_make_array_required(data):
                        static_value = self.backend.array(static_value)

                _queue, _updates = self.add_static_data(value, static_value)
                queue |= _queue
                updates |= _updates
        return updates

    def set_random_seed_keys(self, seed_keys: set[str]) -> None:
        for key in seed_keys:
            if self.all_data[key].value == TBD:
                self._random_seeds[key] = 0
            else:
                value = self.all_data[key].value
                assert isinstance(value, int)
                self._random_seeds[key] = value

    def set_random_seed_values(self, **seed_mapping: int) -> None:
        for key, value in seed_mapping.items():
            if key not in self._random_seeds:
                raise KeyError(f"'{key}' key is not a random seed key!")
            self._random_seeds[key] = value
