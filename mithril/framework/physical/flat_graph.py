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

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import mithril as ml

from ...common import BiMap, PythonGenConfig
from ...cores.c.array import PyArray
from ...types import DataType, GenericDataType
from ...utils.func_utils import is_make_array_required, prepare_function_args
from ...utils.utils import OrderedSet
from ..common import (
    TBD,
    AllValueType,
    ConstraintSolver,
    DataEvalType,
    IOHyperEdge,
    MainValueType,
    StateKey,
    ToBeDetermined,
    Updates,
    UpdateType,
    ValueType,
    any_differentiable,
    is_type_adjustment_required,
    is_valued,
)
from ..logical.model import Connection
from ..logical.operator import Operator
from ..logical.operators import BufferOp
from .data_store import StaticDataStore


class GConnection:
    """Represents a connection between models in a flat graph.
    Attributes:
        key (str): A global identifier for this connection.
        source_keys (list[str]): List of source keys from which this connection
            originates
        target_keys (list[str]): List of target keys to which this connection points.
        connections (set[Connection]): Set of connected connections.

    Note:
        Every connection has an operator, except the input connections.
    """

    def __init__(
        self,
        key: str,
        op: Operator | None,
        source_keys: list[str],
        target_keys: list[str],
    ) -> None:
        self.key = key  # Global key
        self.op = op  # Only global input keys do not have an operator
        self.source_keys = source_keys  # Global source keys
        self.target_keys = target_keys  # Global target keys

    @property
    def local_source_keys(self) -> list[str]:
        if self.op is None:
            return []

        return list(self.op.input_keys)


class FlatGraph(GenericDataType[DataType]):
    # input -> outputkeys
    def __init__(
        self,
        input_keys: set[str],
        output_keys: set[str],
        backend: ml.Backend[DataType],
        solver: ConstraintSolver,
        state_keys: list[StateKey],
        memo: dict[int, IOHyperEdge] | None = None,
    ) -> None:
        if memo is None:
            memo = {}
        self.state_keys = state_keys
        self.backend: ml.Backend[DataType] = backend
        self.model_table: dict[Operator, GConnection] = {}
        self.connections: dict[
            str, GConnection
        ] = {}  # Assumed connections added in topological order.
        self._all_source_keys: set[str] = set()
        self._all_target_keys: set[str] = set(output_keys)
        self._all_keys: set[str] = set()

        self._input_keys = input_keys

        # Output dictionary used for mapping output keys to their corresponding keys
        self.output_dict: dict[str, str] = {key: key for key in sorted(output_keys)}

        # Temporary connection info used for updating connections when
        # a model is removed
        self._temp_connection_info: dict[GConnection, GConnection] = {}

        # Utility tables used for pruning duplicate connections
        self.unique_model_table: dict[str, GConnection] = {}
        self.value_table: dict[str, DataType | ValueType] = {}

        self.data_store: StaticDataStore[DataType] = StaticDataStore(backend)
        self.constraint_solver: ConstraintSolver = deepcopy(solver, memo=memo)
        self.multi_node_keys: dict[str, list[str]] = {}

    @property
    def hanging_keys(self) -> set[str]:
        hanging_keys = (self.all_target_keys - self.all_source_keys) | set(
            self.connections.keys()
        ) - self.all_target_keys - self.all_source_keys

        return hanging_keys - set(self.output_dict.values())

    @property
    def input_keys(self) -> set[str]:
        return set(self._input_keys)

    @property
    def output_keys(self) -> set[str]:
        return set(self.output_dict.keys())

    @property
    def all_keys(self) -> set[str]:
        return (
            set(self.connections.keys())
            | set(self.output_dict.keys())
            | set(self.output_dict.values())
        )

    @property
    def all_static_keys(self) -> set[str]:
        return self.data_store.all_static_keys

    @property
    def runtime_static_keys(self) -> set[str]:
        return self.data_store.runtime_static_keys

    @property
    def unused_keys(self) -> set[str]:
        return self.data_store.unused_keys

    @property
    def data_memo(self) -> dict[int, IOHyperEdge]:
        return self.data_store.data_memo

    @property
    def all_data(self) -> dict[str, IOHyperEdge]:
        return self.data_store.all_data

    @property
    def cached_data(self) -> DataEvalType[DataType]:
        return self.data_store.cached_data

    @property
    def intermediate_non_differentiables(self) -> BiMap[str, IOHyperEdge]:
        return self.data_store.intermediate_non_differentiables

    @property
    def topological_order(self) -> OrderedSet[str]:
        # Traverse the model table in topological order
        topological_order: OrderedSet[str] = OrderedSet()
        keys_to_visit = list(sorted(self.all_source_keys - self.all_target_keys))
        visited: set[str] = set()

        while keys_to_visit:
            key = keys_to_visit.pop()
            if key in visited:
                continue

            visited.add(key)

            # Visit all target keys of the current key
            for target_key in self.get_target_keys(key):
                if target_key in visited:
                    continue

                # Numpy backend uses cache keys for internal operations.
                # So, we need to exclude the cache keys from the source keys.
                source_keys = self.get_source_keys(target_key)
                if self.backend.is_manualgrad and source_keys[-1].endswith("_cache"):
                    source_keys = source_keys[:-1]

                # If all source keys of the target key are visited,
                # then add the target key to the topological order.
                if set(source_keys).issubset(visited):
                    keys_to_visit.append(target_key)
                    topological_order.add(target_key)

        return topological_order

    @property
    def all_target_keys(self) -> set[str]:
        return self._all_target_keys

    @property
    def all_source_keys(self) -> set[str]:
        return self._all_source_keys

    @property
    def all_models(self) -> list[Operator]:
        return list(self.model_table.keys())

    def is_key_static(self, key: str) -> bool:
        return key in self.runtime_static_keys or key in self.data_store.data_values

    def get_op(self, key: str) -> Operator:
        conn = self.connections.get(key, None)
        if conn is None or conn.op is None:
            raise ValueError(f"Model with key {key} not found")
        else:
            return conn.op

    def update_cached_data(self, updates: Updates) -> set[str]:
        return self.data_store.update_cached_data(updates)

    def add_value(self, model: Operator, keys: dict[str, str]) -> None:
        output_key = keys[Operator.output_key]
        # If output/input is of list, tuple or dict type, find indexes of
        # corresponding inputs/outputs when it is flattened to a list.
        # These indices will be used for gradient calculations.
        if self.backend.is_manualgrad:
            all_conns = model.conns.all
            out = all_conns[Operator.output_key]
            out_tensors = out.metadata.tensors
            # Check if any tensor in output also is an input to the operator
            # model.
            for tensor in out_tensors:
                for key, outer_key in keys.items():
                    # Skip output key and cache key.
                    if key in (Operator.output_key, "cache"):
                        continue
                    # Get the index of the tensor in the output.
                    key_conn = all_conns[key]
                    if (
                        key_tensors := key_conn.metadata.tensors
                    ) and tensor in key_tensors:
                        if outer_key in self.multi_node_keys:
                            # If outer_key already exists in multi_node_keys,
                            # then find the index of the tensor in the outer_key and
                            # find corresponding sub_key in corresponding list.
                            sub_index = key_tensors.index(tensor)
                            sub_key = self.multi_node_keys[outer_key][sub_index]
                            self.multi_node_keys.setdefault(output_key, []).append(
                                sub_key
                            )
                        else:
                            self.multi_node_keys.setdefault(output_key, []).append(
                                outer_key
                            )

        # Create output connection of the new Connection.
        out_conn = GConnection(output_key, model, [], [])
        self.model_table[model] = out_conn
        self.connections[output_key] = out_conn
        self._all_target_keys.add(output_key)

        # Create input connections
        for inner_key, outer_key in keys.items():
            self._all_keys.add(outer_key)
            if inner_key == Operator.output_key:
                continue

            conn = self.connections.get(outer_key, None)
            # New input
            if conn is None:
                conn = GConnection(outer_key, None, [], [])
                self.connections[outer_key] = conn

            out_conn.source_keys.append(conn.key)
            conn.target_keys.append(out_conn.key)
            self._all_source_keys.add(conn.key)

    # This method is used to insert an operator into the graph at a specific position
    # The output of inserted operator will replace the output of the previous operator
    def insert_operator_before(
        self,
        new_op: Operator,
        keys: dict[str, str],
        base_op: Operator,
        inserted_key: str,
    ) -> None:
        output_key = keys[Operator.output_key]

        if base_op not in self.model_table:
            raise ValueError(
                f"Base operator `{base_op.formula_key}` must already be in the graph"
            )

        if inserted_key not in keys.values():
            raise ValueError(
                f"Inserted key `{inserted_key}` must be in the keys dictionary"
            )

        if output_key in self.all_keys:
            raise ValueError(f"Output key `{output_key}` must not be in the graph")

        # Add the new operator to the graph
        self.add_value(new_op, keys)

        # Disconnect the inserted key from the base operator
        base_op_out_conn = self.model_table[base_op]
        replaced_idxs = [
            i
            for i, key in enumerate(base_op_out_conn.source_keys)
            if key == inserted_key
        ]

        for idx in replaced_idxs:
            base_op_out_conn.source_keys[idx] = output_key

        self.connections[inserted_key].target_keys.remove(base_op_out_conn.key)
        self.connections[output_key].target_keys.append(base_op_out_conn.key)
        self.all_source_keys.add(output_key)

    def insert_operator_after(
        self,
        new_op: Operator,
        keys: dict[str, str],
        new_source_key: str,
        base_op: Operator,
        target_key: str,
    ) -> None:
        if base_op not in self.model_table:
            raise ValueError(
                f"Base operator `{base_op.formula_key}` must already be in the graph"
            )

        if target_key not in keys.values():
            raise ValueError(
                f"Inserted key `{target_key}` must be in the keys dictionary"
            )

        if new_source_key in self.all_keys:
            raise ValueError(
                f"Source key `{new_source_key}` must be in the keys dictionary"
            )

        # Rename the base operator output key to the source key
        base_op_out_conn = self.model_table[base_op]
        base_op_out_conn.key = new_source_key

        # Cache the target keys and source keys of the inserted key
        cache_target_keys = self.connections[target_key].target_keys
        cache_source_keys = self.connections[target_key].source_keys

        # Add the new operator to the graph
        self.add_value(new_op, keys)
        self.connections[target_key].target_keys = cache_target_keys

        # Update target and source keys
        self.connections[new_source_key].target_keys = [target_key]
        self.connections[new_source_key].source_keys = cache_source_keys
        self.all_target_keys.add(target_key)

    def _collapse_model_keys(self, output_key: str, new_reference_key: str) -> None:
        # If a model removed, the models that uses the output of the removed model
        # should be updated with the new reference key.
        for key, value in self._temp_connection_info.items():
            if value == output_key:
                self._temp_connection_info[key] = self.connections[new_reference_key]

        for key_str, value_str in self.output_dict.items():
            if value_str == output_key:
                self.output_dict[key_str] = new_reference_key

        # Update the multi_node_keys with the new reference key if it is used.
        if output_key in self.multi_node_keys:
            self.multi_node_keys[new_reference_key] = self.multi_node_keys.pop(
                output_key
            )
        else:
            for _key, _value in self.multi_node_keys.items():
                if output_key in _value:
                    _value[_value.index(output_key)] = new_reference_key

    def _update_output_keys(self, output_key: str, new_reference_key: str) -> bool:
        if output_key not in self.output_dict:
            return False

        self.output_dict[output_key] = new_reference_key
        return True

    def get_connection(self, key: str) -> GConnection | None:
        return self.connections.get(key)

    def get_source_keys(self, key: str, include_outputs: bool = False) -> list[str]:
        source_keys: list[str] = []
        if key in self.connections:
            source_keys += self.connections[key].source_keys

        if include_outputs and key in self.output_dict:
            val = self.output_dict[key]
            # Key cannot be source of itself!
            if val != key:
                source_keys.append(self.output_dict[key])

        return source_keys

    def get_target_keys(self, key: str, include_outputs: bool = False) -> list[str]:
        target_keys = (
            list(self.connections[key].target_keys) if key in self.connections else []
        )

        if include_outputs and key in self.output_dict.values():
            target_keys += [
                out_key
                for out_key, ref_key in self.output_dict.items()
                if ref_key == key and out_key != key
            ]

        return target_keys

    def prune_duplicate_operation(
        self,
        op: Operator,
        data: dict[str, IOHyperEdge],
        constant_keys: Mapping[str, DataType | MainValueType],
    ) -> None:
        conn = self.model_table[op]
        key = conn.key

        # The connection is already calculated
        if key in self.data_store.data_values:
            # Unlink source connections
            for source_key in list(conn.source_keys):
                src_conn = self.connections[source_key]
                src_conn.target_keys.remove(key)

                self._update_conn_info(src_conn)

                while source_key in conn.source_keys:
                    conn.source_keys.remove(source_key)

            # Clear connection
            conn.op = None
            self.model_table.pop(op)

            if len(conn.source_keys) != 0:
                raise RuntimeError(f"Source keys of {key} must be empty")

            if key in self._all_target_keys:
                self._all_target_keys.remove(key)

            return

        # Nuke buffer
        if isinstance(op, BufferOp):
            input_key = conn.source_keys[0]
            input_conn = self.connections[input_key]
            input_conn = self._temp_connection_info.get(input_conn, input_conn)
            output_conn = self.connections[key]

            self._update_output_keys(key, input_key)
            self._temp_connection_info[output_conn] = input_conn

            # Update Output conn target conns source keys
            for target_key in output_conn.target_keys:
                target_conn = self.connections[target_key]
                idx = target_conn.source_keys.index(key)
                target_conn.source_keys[idx] = input_key

            # Update input conn target keys
            input_conn.target_keys += output_conn.target_keys

            self._remove_conn(output_conn)
        else:
            # Check duplicate
            source_conn = self._is_duplicate(conn, data, constant_keys)
            if source_conn is None:
                return

            pruned_key = key
            source_key = source_conn.key

            ## Update Data Memo
            pruned_data = self.all_data[pruned_key]
            remained_data = self.all_data[source_key]

            # Match shapes
            updates = remained_data.match(pruned_data)

            # Finally prune the connection
            self._prune_connection(conn, source_conn)
            self._all_source_keys.add(source_conn.key)

            self.data_store.update_cached_data(updates)
            self.constraint_solver(updates)

    def _is_duplicate(
        self,
        conn: GConnection,
        data: dict[str, IOHyperEdge],
        constant_keys: Mapping[str, DataType | MainValueType],
    ) -> GConnection | None:
        # Model id is a unique key for unique operation
        model_id: list[str] = []
        for key in conn.source_keys:
            # We do not consider output and cache keys, when determining model id.
            if key == "output" or "cache" in key:
                continue

            # Extract value from data or static_keys
            value: DataType | AllValueType
            if (_data := data.get(key)) is not None and _data.is_valued:
                value = _data.value
            else:
                value = constant_keys.get(key, TBD)

            # If connection is valued, then compare values.
            if is_valued(value):  # type: ignore
                for value_key, ref_value in self.value_table.items():
                    if type(ref_value) is not type(value):
                        is_equal: bool = False
                    # Check tensors are equal
                    elif self.is_tensor_type(ref_value) and self.is_tensor_type(value):
                        is_equal = (
                            not isinstance(ref_value, PyArray)
                            and not isinstance(value, PyArray)
                        ) and (
                            id(ref_value) == id(value)
                            or ref_value.shape == value.shape  # type: ignore
                            and (ref_value == value).all().item()  # type: ignore
                        )
                    elif (
                        isinstance(ref_value, tuple)
                        and isinstance(value, tuple)
                        or isinstance(ref_value, list)
                        and isinstance(value, list)
                    ):
                        # TODO: Should we handle dict type values too?
                        if len(ref_value) != len(value):
                            is_equal = False
                        else:
                            array_type = self.backend.get_backend_array_type()
                            is_equal = all(
                                type(a) is type(b) and self.backend.all(a == b)
                                if isinstance(a, array_type)
                                else a == b
                                for a, b in zip(ref_value, value, strict=True)
                            )
                    else:
                        is_equal = ref_value == value  # type: ignore

                    if is_equal:
                        model_id.append(value_key)
                        break

                else:
                    # TODO: Remove type ignore after combining Tensor and Scalar
                    self.value_table[str(len(self.value_table))] = value  #  type: ignore
                    model_id.append(str(len(self.value_table) - 1))
            else:
                model_id.append(key)

        if conn.op is None:
            raise RuntimeError(f"Connection {conn.key} must have an operator")

        final_model_id = "-".join(model_id) + f"-{conn.op.formula_key}"

        if final_model_id in self.unique_model_table:
            return self.unique_model_table[final_model_id]

        self.unique_model_table[final_model_id] = conn
        return None

    def _prune_connection(self, conn: GConnection, source_conn: GConnection) -> None:
        self._collapse_model_keys(conn.key, source_conn.key)

        # Update target keys of connections
        for target_key in conn.target_keys:
            # if target_key not in source_conn.target_keys:
            source_conn.target_keys.append(target_key)

        # The source key of the conn's target conns should be updated to
        # the source_conn's key
        for target_key in conn.target_keys:
            target_conn = self.connections[target_key]
            for idx, source_key in enumerate(list(target_conn.source_keys)):
                if source_key == conn.key:
                    target_conn.source_keys[idx] = source_conn.key

        self._remove_conn(conn)
        self.all_source_keys.add(source_conn.key)

    def _update_conn_info(self, conn: GConnection) -> None:
        if len(conn.target_keys) == 0 and conn.key in self._all_source_keys:
            self._all_source_keys.remove(conn.key)

        if len(conn.source_keys) == 0 and conn.key in self._all_target_keys:
            self._all_target_keys.remove(conn.key)

    def _remove_conn(self, conn: GConnection) -> None:
        if conn.key in self.connections and conn.key not in self.output_dict.values():
            self.remove_key_from_store(conn.key, hard_remove=True)

        self.connections.pop(conn.key, None)

        # Remove connection from source connections
        for source_key in conn.source_keys:
            source_conn = self.connections[source_key]
            if conn.key in source_conn.target_keys:
                source_conn.target_keys.remove(conn.key)

            self._update_conn_info(source_conn)

        for target_key in conn.target_keys:
            target_conn = self.connections[target_key]
            if conn.key in target_conn.source_keys:
                target_conn.source_keys.remove(conn.key)

            self._update_conn_info(target_conn)

        if conn.key in self._all_source_keys:
            self._all_source_keys.remove(conn.key)

        if conn.key in self._all_target_keys:
            self._all_target_keys.remove(conn.key)

        if conn.op in self.model_table:
            self.model_table.pop(conn.op)

    def remove_key(self, key: str) -> None:
        if key in self.output_dict:
            self.output_dict.pop(key)
        conn = self.get_connection(key)
        if conn is not None:
            self._remove_conn(conn)

    def infer_ignore_step(
        self, key: str, keys: set[str], queue: set[str], from_source: bool
    ) -> None:
        forward_key_fn: Callable[[str, bool], list[str]]
        if from_source:
            forward_key_fn = self.get_target_keys
            backward_key_fn = self.get_source_keys
            if key not in self.all_source_keys:
                return
        else:
            forward_key_fn = self.get_source_keys
            backward_key_fn = self.get_target_keys

            if key not in self.all_target_keys and key not in self.output_dict:
                return

        for value in forward_key_fn(key, include_outputs=True):
            if value not in keys:
                value_mapping = backward_key_fn(value, include_outputs=True)
                if set(value_mapping).issubset(keys) and (
                    value not in self.output_keys
                ):
                    keys.add(value)
                    queue.add(value)

    def infer_static_keys(self) -> Updates:
        """Infers the static keys and calculates
        the static values during the inference.
        """
        static_keys = set(self.data_store.data_values.keys())
        queue = set(static_keys)
        updates = Updates()
        state_inputs = {_key.in_key for _key in self.state_keys}
        state_outputs = {_key.out_key for _key in self.state_keys}
        while queue:
            key = queue.pop()
            if (
                (key not in self.all_source_keys)
                or key in self.unused_keys
                or key in state_inputs
            ):
                continue

            for value in self.get_target_keys(key):
                # Value is already in statics or unused keys, then skip.
                if (
                    value in self.data_store.data_values
                    or value in self.unused_keys
                    or value in state_outputs
                ):
                    continue

                value_mapping = self.get_source_keys(value)

                # To infer a model, all of its input keys should be in statics.
                if not all(key in self.data_store.data_values for key in value_mapping):
                    continue

                op = self.get_op(value)

                # TODO: Move this outside of while loop
                # after CBackend is completely implemented.
                fn_dict = (
                    self.backend.op_function_dict | self.backend.registered_primitives
                )

                static_value: DataType | MainValueType

                fn = fn_dict[op.formula_key]

                # Orginize args and kwargs
                local_input_keys = list(op.input_keys)
                if self.backend.is_manualgrad:
                    local_input_keys.append("cache")
                inputs = {
                    key: value
                    for key, value in zip(local_input_keys, value_mapping, strict=False)
                }
                args_dict, kwargs_dict = prepare_function_args(
                    self.data_store.data_values,
                    fn,
                    inputs,
                    self.backend.array_creation_funcs,
                    False,
                )
                args = [
                    self.data_store.data_values[arg_key]
                    for arg_keys in args_dict.values()
                    for arg_key in arg_keys
                ]
                kwargs = {
                    key: self.data_store.data_values[value]
                    for key, value in kwargs_dict.items()
                }

                # If function needs backend specific args
                if op.formula_key in self.backend.array_creation_funcs:
                    kwargs["default_dtype"] = self.backend._dtype.name
                    # TODO: Add support for C backends
                    if not isinstance(self.backend.CODEGEN_CONFIG, PythonGenConfig):
                        raise NotImplementedError("C backend is not supported yet!")
                    if self.backend.CODEGEN_CONFIG.SPECIFY_DEVICE:
                        kwargs["device"] = self.backend.get_device()

                static_value = fn(*args, **kwargs)

                # Check astype needed
                if (
                    self.backend.is_manualgrad
                    and is_type_adjustment_required(self.all_data, value_mapping)
                    and isinstance(static_value, self.backend.get_backend_array_type())
                ):
                    static_value = self.backend.array(static_value)

                if self.backend.is_manualgrad:
                    data = self.data_store._all_data[value]
                    if is_make_array_required(data):
                        static_value = self.backend.array(static_value)

                _queue, _updates = self.add_static_data(value, static_value)
                self.prune_duplicate_operation(op, self.all_data, self.cached_data)

                queue |= _queue
                updates |= _updates
        return updates

    def set_static_keys(
        self,
        static_keys: dict[
            str, DataType | int | float | bool | Sequence[Any] | dict[str, Any]
        ],
    ) -> Updates:
        updates = Updates()
        for key, value in static_keys.items():
            if key not in self.input_keys:
                raise KeyError(
                    "Requires static key to be in the input keys of the model!"
                )
            if self.data_store._all_data[key].is_tensor and not isinstance(
                value, ToBeDetermined | self.backend.get_backend_array_type()
            ):
                raise ValueError(
                    "Requires given arrays to be of same type with given backend!"
                )
            _, _updates = self.add_static_data(key, value)
            updates |= _updates
        return updates

    def add_static_data(
        self,
        key: str,
        value: DataType | int | float | bool | Sequence[Any] | dict[str, Any],
    ) -> tuple[set[str], Updates]:
        updates = Updates()
        updated_keys = {key}
        if key in self.cached_data:
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
            if (data := self.all_data.get(key, None)) is None:
                raise KeyError(f"'{key}' key not found in model!")

            if self.data_store.is_scalar_type(
                value
            ):  # TODO: Is this check really required?
                # If value is a dtype, convert into correseponding logical dtype.
                if (
                    value.__hash__ is not None
                    and value in self.backend.dtype_map.inverse
                ):
                    dtype_logical = self.backend.convert_to_logical(value)
                    updates |= data.set_type(type(dtype_logical))
                else:
                    updates |= data.set_value(value)
            else:
                # Convert value to logical representaiton and set accordingly.
                if key in self.runtime_static_keys:
                    x = self.data_store.convert_phys_value_to_logical(value)
                    updates |= data.set_value(x)

            self.cached_data[key] = value  # type: ignore
            self.intermediate_non_differentiables.pop(key, None)
            if (
                key not in self.intermediate_non_differentiables
                and key in self.runtime_static_keys
            ):
                self.data_store._runtime_static_keys.remove(key)
        # Finally update cached_data, infer unused keys and
        # return newly added static keys.
        self.constraint_solver(updates)
        statics = self.update_cached_data(updates) | updated_keys

        return statics, updates

    def infer_unused_keys(self, key: str) -> None:
        # Infers unused keys when "key" is set as static.
        output_keys = self.output_keys
        queue = set(self.get_source_keys(key, True))
        while queue:
            source_key = queue.pop()
            all_static_keys = self.data_store.all_static_keys
            if source_key not in self.data_store.unused_keys and all(
                [
                    item in all_static_keys | self.data_store.unused_keys
                    for item in self.get_target_keys(source_key)
                ]
            ):
                if source_key not in output_keys and set(
                    self.get_target_keys(source_key, True)
                ).issubset(
                    self.data_store._unused_keys | self.data_store.cached_data.keys()
                ):
                    self.data_store.remove_key_from_store(source_key)

                queue |= set(
                    self.get_source_keys(source_key, True)
                    if source_key in self.connections
                    else []
                )

    def infer_ignore(
        self,
        weak_keys: set[str],
        output_keys: set[str],
        strict_keys: set[str] | None = None,
        update_graph: bool = True,
    ) -> tuple[set[str], set[str]]:
        """
        Infers the keys which will be ignored


        Parameters
        ----------
        keys : set[str]
            output keys that will be ignored,
            it must be given from user during compilation

        output_keys: tuple[str, ...]
            output keys of the model

        Returns
        -------
        tuple[Callable, Callable]
            _description_


        Returns
        -------
        tuple[set[str], tuple[str, ...]]
            Returns keys that will be ignored during ignore keys inference algorithm
            also returns updated output_keys in a tuple
        """
        if strict_keys is None:
            strict_keys = set()

        # Remove non_leaf ignored keys from output keys and ignored keys
        # e.g. Logistic Regression output (logits) is also an input to probs_out
        # in this case logits_out will become an internal key.
        keys = weak_keys | strict_keys
        non_leaf_keys = {
            key
            for key in weak_keys
            if key in self.all_source_keys and key in output_keys
        }
        # Internal keys will be removed from output_keys but also they will
        # be removed from current ignored keys.
        keys -= non_leaf_keys
        output_keys -= non_leaf_keys

        queue = keys.copy()
        while queue:
            key = queue.pop()
            # try forward inference (check if any inference is possible
            # from inputs to outputs)
            self.infer_ignore_step(key, keys, queue, from_source=True)
            # try bacward inference (check if any inference possible
            # from outputs to inputs)
            self.infer_ignore_step(key, keys, queue, from_source=False)

            if update_graph:
                self.remove_key(key)
                output_keys.discard(key)
                self._input_keys.discard(key)

        return keys, output_keys

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
            if not (data := self.data_store._all_data[key]).is_tensor:
                raise ValueError("Non-tensor data can not have shape!")
            assert data.shape is not None
            updates |= data.shape.set_values(value)
        self.constraint_solver(updates)
        # Some intermediate values may be calculated, update cached data.
        self.data_store.update_cached_data(updates)

    def update_data(self, data: dict[str, IOHyperEdge]) -> None:
        if data.keys() & self.data_store._all_data.keys():
            raise Exception("Some keys are already in data store!")
        self.data_store._all_data |= data
        state_keys = set()
        for item in self.state_keys:
            state_keys.add(item.in_key)
            state_keys.add(item.out_key)
        for key, value in data.items():
            if key not in state_keys:
                is_diff = any_differentiable(value._value)
                if value.initial_valued and not is_diff:
                    raise ValueError("Initial valued data should be differentiable!")
                elif is_diff:
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
                and key not in self.input_keys
            ) or (key in self._input_keys and value.is_valued):
                self.data_store._set_data_value(key, value)
            elif key in self._input_keys:
                self.data_store._runtime_static_keys.add(key)
            else:
                if value.is_valued:
                    self.data_store._set_data_value(key, value)
                else:
                    self.data_store.intermediate_non_differentiables[key] = value

    def graph_update(self) -> None:
        # Currently only GGML needs graph update!
        for out_conn in list(self.model_table.values()):
            out_key = out_conn.key
            op = self.get_op(out_key)

            if op.formula_key not in self.backend.CODEGEN_CONFIG.IMPLICIT_BROADCAST_OPS:
                continue

            # NOTE: This broadcasting behaviour only works properly in GGML.

            # GGML backend does not support implicit broadcasting when the
            # left shape needs to be broadcasted to the right shape.
            # Therefore, we need to add a broadcast_to operator to the graph.
            source_keys = self.get_source_keys(out_key)
            left_key = source_keys[0]
            left_shape: list[int] = self.get_key_shape(left_key)
            output_shape: list[int] = self.get_key_shape(out_key)
            assert isinstance(
                left_shape, list
            ), f"`{left_key}` is not specified with shape!"
            assert isinstance(
                output_shape, list
            ), f"`{out_key}` is not specified with shape!"

            # If left shape is not the same as output shape, we need to add
            # a broadcast_to operator.
            if left_shape != output_shape:
                key = "broadcast_to_shape"  # TODO fix this key
                shape_out_key = "broadcast_to_shape_output"
                left_data = self.all_data[left_key]
                self.update_data(
                    {
                        key: IOHyperEdge(tuple[int, ...], tuple(output_shape)),
                        shape_out_key: IOHyperEdge(left_data._type),
                    }
                )
                _mappings = {
                    "input": left_key,
                    "shape": "broadcast_to_shape",
                    "output": shape_out_key,
                }
                kwargs = {
                    "input": left_data,
                    "shape": self.all_data[key],
                    "output": self.all_data[shape_out_key],
                }
                self.insert_operator_before(
                    Operator("broadcast_to", name="Broadcast_to", **kwargs),
                    _mappings,
                    op,
                    left_key,
                )

    def get_key_shape(self, key: str) -> list[int]:
        return self.data_store.get_key_shape(key)

    def get_next_unique_key(self, prefix: str) -> str:
        i = 0
        while f"{prefix}_{i}" in self._all_keys:
            i += 1

        self._all_keys.add(f"{prefix}_{i}")
        return f"{prefix}_{i}"

    def remove_key_from_store(
        self, key: str, label_as_unused: bool = True, hard_remove: bool = False
    ) -> None:
        self.data_store.remove_key_from_store(key, label_as_unused, hard_remove)

        if hard_remove:
            self._clear_constraints(key)

    def _clear_constraints(self, key: str) -> None:
        if key not in self.all_data:
            return

        shape_constraints = self.all_data[key].constraints[UpdateType.SHAPE]
        type_constraints = self.all_data[key].constraints[UpdateType.TYPE]
        value_constraints = self.all_data[key].constraints[UpdateType.VALUE]
        for source_key in self.get_source_keys(key):
            if source_key in self.all_data:
                self.all_data[source_key].constraints[UpdateType.SHAPE] -= (
                    shape_constraints
                )
                self.all_data[source_key].constraints[UpdateType.TYPE] -= (
                    type_constraints
                )
                self.all_data[source_key].constraints[UpdateType.VALUE] -= (
                    value_constraints
                )
