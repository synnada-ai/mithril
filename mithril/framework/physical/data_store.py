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
from typing import Any, TypeGuard

from ...backends.backend import Backend
from ...core import DataType, data_types
from ...utils.func_utils import is_make_array_required, prepare_function_args
from ...utils.utils import BiMap
from ..common import (
    TBD,
    Connection,
    ConstraintSolver,
    GenericDataType,
    MainValueType,
    Scalar,
    Tensor,
    ToBeDetermined,
    Updates,
    is_type_adjustment_required,
)
from .flat_graph import FlatGraph


class StaticDataStore(GenericDataType[DataType]):
    def __init__(
        self,
        graph: FlatGraph,
        backend: Backend,
        inference: bool,
        solver: ConstraintSolver,
        memo: dict | None = None,
    ) -> None:
        if memo is None:
            memo = {}

        self.is_materialized = False
        self._all_data: dict[str, Tensor | Scalar] = dict()
        self.data_memo: dict[int, Tensor | Scalar] = dict()
        self.graph = graph
        self.backend: Backend[DataType] = backend
        self.inference = inference
        self._cached_data: dict[str, Tensor | Scalar] = dict()
        self._intermediate_non_differentiables: BiMap[str, Tensor | Scalar] = BiMap()
        self._runtime_static_keys: set[str] = set()
        self._unused_keys: set[str] = set()
        # Final tensor values of data store.
        self.data_values: dict[str, DataType | MainValueType] = {}
        self.constraint_solver: ConstraintSolver = deepcopy(solver, memo=memo)

    @property
    def all_data(self):
        return self._all_data

    @property
    def cached_data(self):
        return self._cached_data

    @property
    def runtime_static_keys(self) -> set[str]:
        return self._runtime_static_keys

    @property
    def all_static_keys(self) -> set[str]:
        return self._runtime_static_keys | self._cached_data.keys()

    @property
    def unused_keys(self) -> set[str]:
        return self._unused_keys

    @staticmethod
    def is_scalar_type(t: Any) -> TypeGuard[MainValueType]:
        return not isinstance(t, tuple(data_types))

    def remove_keys_from_store(self, keys: set[str]):
        keys -= self.graph.alias_map.keys()
        for key in keys:
            self._remove_key_from_store(key, label_as_unused=False, hard_remove=True)

    def _remove_key_from_store(
        self, key: str, label_as_unused: bool = True, hard_remove: bool = False
    ):
        if key in self._cached_data:
            self._cached_data.pop(key)
        self._runtime_static_keys.discard(key)
        if key in self._intermediate_non_differentiables:
            self._intermediate_non_differentiables.pop(key)

        if label_as_unused:
            self._unused_keys.add(key)
            self._clear_constraints(key)

        # Finally delete data value or data of removed key from all_data attribute
        # taking hard_remove flag into account.
        if hard_remove:
            self._all_data.pop(key)
            self._clear_constraints(key)
        else:
            data = self.all_data[key]
            if isinstance(data, Tensor):
                data.value = None

    def _clear_constraints(self, key: str):
        if key not in self._all_data:
            return

        shape_constraints = self._all_data[key].shape_constraints
        type_constraints = self._all_data[key].type_constraints
        for source_key in self.graph.get_source_keys(key):
            if source_key in self._all_data:
                self._all_data[source_key].shape_constraints -= shape_constraints
                self._all_data[source_key].type_constraints -= type_constraints

    def _update_cached_data(self, updated_data: Updates) -> set[str]:
        # If any data value is found by shape inference algorithms
        # transfer this data in cached_data.
        transferred_keys = set()
        updated_inter_data = (
            updated_data.value_updates
            & self._intermediate_non_differentiables.inverse.keys()
        )
        # updated_inter_data = self._intermediate_non_differentiables.inverse.keys()
        # for key, data in self._intermediate_non_differentiables.items():
        for data in updated_inter_data:
            key = self._intermediate_non_differentiables.inverse[data]
            if data.value is not TBD:
                if key in self._cached_data:
                    raise KeyError(
                        f"'{key}' key can not be an intermediate and cached key "
                        "at the same time!"
                    )
                self._cached_data[key] = data
                transferred_keys.add(key)
        for key in transferred_keys:
            self._intermediate_non_differentiables.pop(key)
        return transferred_keys

    def _infer_unused_keys(self, key: str):
        # Infers unused keys when "key" is set as static.
        output_keys = self.graph.output_keys
        queue = set(self.graph.get_source_keys(key))
        while queue:
            source_key = queue.pop()
            all_static_keys = self.all_static_keys
            if source_key not in self.unused_keys and all(
                [
                    item in all_static_keys | self.unused_keys
                    for item in self.graph.get_target_keys(source_key)
                ]
            ):
                if source_key not in output_keys:
                    # Given source key can have an alias. also its alias shoud not
                    # be in output_keys
                    for value in self.graph.alias_map.values():
                        if value == source_key:
                            break
                    else:
                        if set(
                            self.graph.get_target_keys(source_key, include_aliases=True)
                        ).issubset(self._unused_keys | self.cached_data.keys()):
                            self._remove_key_from_store(source_key)

                queue |= set(
                    self.graph.get_source_keys(source_key)
                    if source_key in self.graph.connections
                    else []
                )

    def _materialize_cached_data(self):
        # Simply assigns final (real) values to the keys.
        if not self.data_values:
            self.data_values = {
                key: data.value for key, data in self._cached_data.items()
            }
            self.is_materialized = True
        else:
            raise Exception("Can not materialize cached data multiple times.")

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
            if isinstance(data := self._all_data[key], Scalar):
                raise ValueError("Scalar data can not have shape!")
            updates |= data.shape.set_values(value)
        self.constraint_solver(updates)
        # Some intermediate values may be calculated, update cached data.
        new_statics = self._update_cached_data(updates)
        for key in new_statics:
            self._infer_unused_keys(key)

    def update_data(self, data: dict[str, Tensor | Scalar]):
        if data.keys() & self._all_data.keys():
            raise Exception("Some keys are already in data store!")
        self._all_data |= data
        for key, value in data.items():
            if not value.is_non_diff:
                continue

            # Distribute non-differentiable keys into 3 attributes using
            # type of values. If a key has a definite value, add it into cached_data.
            # If key ends with "_cache", backend is manual and it does not apper as
            # and input to the model, then directly add it to th cached_data because
            # these keys are internally created cache keys for the corresponding
            # primitive functions.
            if (
                self.backend.is_manualgrad
                and key.endswith("_cache")
                and key not in self.graph.input_keys
            ) or (key in self.graph.input_keys and value.value is not TBD):
                self._cached_data[key] = value
            elif key in self.graph.input_keys:
                self._runtime_static_keys.add(key)
            else:
                if value.value is not TBD:
                    self._cached_data[key] = value
                else:
                    self._intermediate_non_differentiables[key] = value

    def set_static_keys(
        self,
        static_keys: dict[str, DataType]
        | dict[str, MainValueType]
        | dict[str, DataType | MainValueType],
    ) -> Updates:
        updates = Updates()
        for key, value in static_keys.items():
            if key not in self.graph.input_keys:
                raise KeyError(
                    "Requires static key to be in the input keys of the model!"
                )
            if not (
                isinstance(self._all_data[key], Scalar)
                or isinstance(
                    value, ToBeDetermined | self.backend.get_backend_array_type()
                )
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
        if self.is_materialized:
            raise Exception(
                "DataStore materialized, can not add any other static data."
            )
        if key in self._cached_data:
            raise ValueError(
                f"Statically given key: {key} has been already set as static "
                "with a value!"
            )
        if key in self.unused_keys:
            raise ValueError(
                f"Given '{key}' key is unused for the model, no need to provide "
                "data for it."
            )
        elif value is TBD:
            self._runtime_static_keys |= {key}
        else:
            if (data := self._all_data.get(key, None)) is None:
                raise KeyError(f"'{key}' key not found in model!")

            if isinstance(data, Tensor) and self.is_tensor_type(value):
                updates |= data.set_value(value)

            elif isinstance(data, Scalar) and self.is_scalar_type(value):
                updates |= data.set_value(value)  #
            else:
                raise ValueError(
                    f"Given value type: {type(value)} does not match with "
                    f"the type of data: {type(data)}!"
                )
            if key not in self._intermediate_non_differentiables:
                self._cached_data[key] = data
                for al_key, al_value in self.graph.alias_map.items():
                    if key == al_value:
                        self._cached_data[al_key] = data

        # Finally update cached_data, infer unused keys and
        # return newly added static keys.
        self.constraint_solver(updates)
        statics = self._update_cached_data(updates) | updated_keys
        for static in statics:
            self._infer_unused_keys(static)

        return statics, updates

    def infer_static_keys(self) -> Updates:
        """Infers the static keys and calculates
        the static values during the inference.
        """
        statics = self.cached_data
        queue = set(statics.keys())
        updates = Updates()
        while queue:
            key = queue.pop()
            if (
                key in self.graph.all_source_keys
                or key in self.graph.alias_map.values()
            ) and key not in self.unused_keys:
                for value in self.graph.get_target_keys(key, include_aliases=True):
                    if value not in (statics.keys() | self.unused_keys):
                        value_mapping = self.graph.get_source_keys(value)
                        if set(value_mapping).issubset(statics.keys()):
                            model = self.graph.get_model(value)
                            # TODO: Move this outside of while loop
                            # after CBackend is completely implemented.
                            fn_dict = (
                                self.backend.primitive_function_dict
                                | self.backend.registered_primitives
                            )
                            static_value: DataType | MainValueType
                            if model is None:
                                static_value = statics[value_mapping[0]].value
                            else:
                                fn = fn_dict[model.formula_key]

                                # Orginize args and kwargs
                                local_input_keys = list(model._input_keys)
                                if self.backend.is_manualgrad:
                                    local_input_keys.append("cache")
                                inputs = {
                                    key: value
                                    for key, value in zip(
                                        local_input_keys, value_mapping, strict=False
                                    )
                                }
                                args_dict, kwargs_dict = prepare_function_args(
                                    self.all_data,
                                    fn,
                                    inputs,
                                    self.backend.array_creation_funcs,
                                    False,
                                )
                                args = [
                                    statics[arg_key].value
                                    for arg_keys in args_dict.values()
                                    for arg_key in arg_keys
                                ]
                                kwargs = {
                                    key: statics[value].value
                                    for key, value in kwargs_dict.items()
                                }

                                # If function needs backend specific args
                                if (
                                    model.formula_key
                                    in self.backend.array_creation_funcs
                                ):
                                    kwargs["precision"] = self.backend.precision
                                    if not self.backend.is_manualgrad:
                                        kwargs["device"] = self.backend._device

                                static_value = fn(*args, **kwargs)

                                # Check astype needed
                                if (
                                    self.backend.is_manualgrad
                                    and is_type_adjustment_required(
                                        self.all_data, value_mapping
                                    )
                                ):
                                    static_value = self.backend.array(static_value)

                            if self.backend.is_manualgrad:
                                data = self._all_data[value]
                                # _temp_shape = data._temp_shape
                                # _temp_shape = next(iter(data.shape.reprs))
                                if is_make_array_required(data):
                                    static_value = self.backend.array(static_value)

                            # queue |= self.add_static_data(value, static_value)
                            _queue, _updates = self.add_static_data(value, static_value)
                            queue |= _queue
                            updates |= _updates
        # Finalize cached_data.
        self._materialize_cached_data()
        return updates
