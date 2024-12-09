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

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, TypeGuard

from ...backends.backend import Backend
from ...core import DataType, data_types
from ...utils.func_utils import is_make_array_required
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

key_map_type = dict[str, str]


class StaticDataStore(GenericDataType[DataType]):
    def __init__(
        self,
        graph: FlatGraph[DataType],
        backend: Backend[DataType],
        inference: bool,
        solver: ConstraintSolver,
        memo: dict[int, Tensor[DataType] | Scalar] | None = None,
    ) -> None:
        if memo is None:
            memo = {}

        self._all_data: dict[str, Tensor[DataType] | Scalar] = dict()
        self.data_memo: dict[int, Tensor[DataType] | Scalar] = dict()
        self.graph: FlatGraph[DataType] = graph
        self.backend: Backend[DataType] = backend
        self.inference = inference
        self._cached_data: dict[str, Tensor[DataType] | Scalar] = dict()
        self._intermediate_non_differentiables: BiMap[
            str, Tensor[DataType] | Scalar
        ] = BiMap()
        self._runtime_static_keys: set[str] = set()
        self._unused_keys: set[str] = set()
        # Final tensor values of data store.
        self.data_values: dict[str, DataType | MainValueType | str] = dict()
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
        keys -= set(self.graph.output_keys)
        for key in keys:
            self._remove_key_from_store(key, label_as_unused=False, hard_remove=True)

    def _remove_key_from_store(
        self, key: str, label_as_unused: bool = True, hard_remove: bool = False
    ):
        if key in self._cached_data:
            self._cached_data.pop(key)
            self.data_values.pop(key)
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
        transferred_keys: set[str] = set()
        updated_inter_data = (
            updated_data.value_updates
            & self._intermediate_non_differentiables.inverse.keys()
        )
        for data in updated_inter_data:
            key = self._intermediate_non_differentiables.inverse[data]
            if self.get_value(key) is not TBD:
                if key in self._cached_data:
                    raise KeyError(
                        f"'{key}' key can not be an intermediate and cached key "
                        "at the same time!"
                    )
                self._cached_data[key] = data
                if key not in self.data_values:
                    assert not isinstance(data.value, ToBeDetermined)
                    self.data_values[key] = data.value
                transferred_keys.add(key)
        for key in transferred_keys:
            self._intermediate_non_differentiables.pop(key)
        return transferred_keys

    def _infer_unused_keys(self, key: str):
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
                    self._remove_key_from_store(source_key)

                queue |= set(
                    self.graph.get_source_keys(source_key, True)
                    if source_key in self.graph.connections
                    else []
                )

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

    def update_data(self, data: dict[str, Tensor[DataType] | Scalar]):
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
                assert not isinstance(value.value, ToBeDetermined)
                self.data_values[key] = value.value
            elif key in self.graph.input_keys:
                self._runtime_static_keys.add(key)
            else:
                if value.value is not TBD:
                    self._cached_data[key] = value
                    assert not isinstance(value.value, ToBeDetermined)
                    self.data_values[key] = value.value
                else:
                    self._intermediate_non_differentiables[key] = value

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
        else:
            if (data := self._all_data.get(key, None)) is None:
                raise KeyError(f"'{key}' key not found in model!")
            # TODO: Mypy does not understand the type of data and value
            # if we dont't write if-else statement for Tensor and Scalar
            # Any fixes?.
            if isinstance(data, Tensor) and self.is_tensor_type(value):
                # TODO: Do not set value to Tensor if value is DataType. Update here
                # after Tensor and Scalar classes are merged to Edge.
                updates |= data.set_value(value)
                # Temporarily remove value from tensor and add to tensor_values!
                self.data_values[key] = value
                data.value = TBD
            elif isinstance(data, Scalar) and self.is_scalar_type(value):
                updates |= data.set_value(value)  #
                self.data_values[key] = value
            else:
                raise ValueError(
                    f"Given value type: {type(value)} does not match with "
                    f"the type of data: {type(data)}!"
                )
            if key not in self._intermediate_non_differentiables:
                if key in self.runtime_static_keys:
                    self._runtime_static_keys.remove(key)
                self._cached_data[key] = data
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

                fn = fn_dict[model.formula_key]

                # Orginize args and kwargs
                local_input_keys = list(model._input_keys)
                if self.backend.is_manualgrad:
                    local_input_keys.append("cache")
                inputs = {
                    key: value
                    for key, value in zip(local_input_keys, value_mapping, strict=False)
                }
                args_dict, kwargs_dict = self.prepare_function_args(
                    fn,
                    inputs,
                    self.backend.array_creation_funcs,
                    False,
                )
                args = [
                    self.get_value(arg_key)
                    for arg_keys in args_dict.values()
                    for arg_key in arg_keys
                ]
                kwargs = {
                    key: self.get_value(value) for key, value in kwargs_dict.items()
                }

                # If function needs backend specific args
                if model.formula_key in self.backend.array_creation_funcs:
                    kwargs["precision"] = self.backend.precision
                    if not self.backend.is_manualgrad:
                        kwargs["device"] = self.backend._device

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

    def get_value(self, key: str) -> DataType | MainValueType | ToBeDetermined | str:
        return self.data_values.get(key, self._all_data[key].value)

    def prepare_function_args(
        self,
        function: Callable[..., Any],
        inputs: key_map_type,
        array_creation_funcs: list[str],
        reduce_with_defaults: bool = True,
    ) -> tuple[dict[str, list[str]], key_map_type]:
        formula_key = function.__name__
        code_obj = function.__code__

        fn_args_dict = {
            arg_name: False for arg_name in code_obj.co_varnames[: code_obj.co_argcount]
        }

        # Check if variadic positional argument exists
        if code_obj.co_flags & 0x04 == 0x04:
            idx = code_obj.co_argcount + code_obj.co_kwonlyargcount
            fn_args_dict[code_obj.co_varnames[idx]] = True

        fn_kwarg_keys = list(
            code_obj.co_varnames[
                code_obj.co_argcount : code_obj.co_argcount + code_obj.co_kwonlyargcount
            ]
        )

        # Array creation functions requires device and
        # precision to properly create tensor
        if formula_key in array_creation_funcs:
            # Remove precision and device from kwarg_keys
            # we will partially provide them
            fn_args_dict.pop("precision", None)
            if "precision" in fn_kwarg_keys:
                fn_kwarg_keys.remove("precision")

            fn_args_dict.pop("device", None)
            if "device" in fn_kwarg_keys:
                fn_kwarg_keys.remove("device")

        # Prepare arguments
        fn_kwarg_dict, removed_kwarg_dict = self.create_kwarg_dict(
            fn_kwarg_keys, function, inputs, reduce_with_defaults
        )
        fn_args_mapping = self.reorganize_args(
            fn_args_dict,
            set(fn_kwarg_dict.values()) | set(removed_kwarg_dict.values()),
            function,
            inputs,
            reduce_with_defaults,
        )

        return fn_args_mapping, fn_kwarg_dict

    def create_kwarg_dict(
        self,
        kwarg_keys: list[str],
        function: Callable,
        inputs: key_map_type,
        reduce_with_defaults: bool,
    ) -> tuple[key_map_type, key_map_type]:
        kwarg_keys_dict: key_map_type = {
            kwarg_key: inputs[kwarg_key] for kwarg_key in kwarg_keys
        }
        removed_kwargs_dict: key_map_type = {}

        kwdefaults = function.__kwdefaults__

        if kwdefaults is not None and reduce_with_defaults:
            for key, value in kwdefaults.items():
                # provided_value = data[kwarg_keys_dict[key]].value
                provided_value = self.get_value(kwarg_keys_dict[key])
                if value == provided_value and type(value) is type(provided_value):
                    removed_kwargs_dict[key] = kwarg_keys_dict[key]
                    kwarg_keys_dict.pop(key)

        return kwarg_keys_dict, removed_kwargs_dict

    def reorganize_args(
        self,
        arg_keys: dict[str, bool],
        kwarg_keys: list[str] | set[str],
        function: Callable,
        inputs: key_map_type,
        reduce_with_defaults: bool,
    ) -> dict[str, list[str]]:
        defaults = function.__defaults__
        formula_key = function.__name__

        local_input_keys = list(inputs.keys())
        inputs = deepcopy(inputs)
        organized_arguments: dict[str, list[str]] = {}

        for idx, (name, is_variadic) in enumerate(arg_keys.items()):
            if "cache" in name:
                # TODO: Refactor here
                provided_value = self.get_value(inputs[name])
                if (
                    reduce_with_defaults
                    and idx == len(arg_keys) - 1
                    and defaults
                    and provided_value == defaults[-1]
                ):
                    continue

                outer_names = [inputs[name]]
                inputs.pop(name)

            # If the argument variadic, then it takes rest of the inputs
            elif is_variadic:
                outer_names = [
                    input for input in inputs.values() if input not in kwarg_keys
                ]

            elif name not in local_input_keys:
                raise RuntimeError(
                    f"Primitive '{formula_key}' input keys:'{local_input_keys}' and"
                    f" backend function input keys: '{arg_keys.keys()}' "
                    "are not matching!"
                )

            else:
                outer_names = [inputs[name]]
                inputs.pop(name)

            organized_arguments[name] = outer_names

        return organized_arguments
