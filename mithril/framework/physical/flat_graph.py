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

from collections.abc import Callable, KeysView, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

import mithril as ml

from ...core import DataType, GenericDataType
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
    UpdateType,
    ValueType,
    is_type_adjustment_required,
)
from ..logical.model import Connection
from ..logical.operator import Operator
from ..logical.operators import BufferOp
from .data_store import StaticDataStore


@dataclass
class GConnection:
    """Represents a connection between nodes in a flat graph.
    Attributes:
        node (Node | None): The node associated with this connection.
        key (str): A global identifier for this connection.
        source_keys (list[str]): List of source keys from which this connection
            originates
        target_keys (list[str]): List of target keys to which this connection points.
        connections (set[Connection]): Set of connected connections.

    Note:
        Every connection is belong to a node, except the input connections.
    """

    node: Node | None
    key: str
    source_keys: list[str]
    target_keys: list[str]
    connections: set[GConnection]

    def __hash__(self) -> int:
        return hash(id(self))


@dataclass
class Node:
    """A node representing a primitive model and its connections in the graph.

    Attributes:
        model (Operator): The primitive model associated with this node.
        connections (dict[str, Connection]): A dictionary mapping connection names
            to Connection objects.

    Note:
        The key "output" in the connections dictionary is reserved for the output
        connection of the node.
    """

    model: Operator
    connections: dict[str, GConnection]

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)

    def __repr__(self) -> str:
        return f"{self.model}"


class FlatGraph(GenericDataType[DataType]):
    # input -> outputkeys
    def __init__(
        self,
        input_keys: set[str],
        output_keys: set[str],
        backend: ml.Backend[DataType],
        solver: ConstraintSolver,
        memo: dict[int, IOHyperEdge] | None = None,
        inference: bool = False,
    ) -> None:
        if memo is None:
            memo = {}

        self.backend: ml.Backend[DataType] = backend
        self.nodes: dict[Operator, Node] = {}
        self.connections: dict[
            str, GConnection
        ] = {}  # Assumed connections added in topological order.
        self._all_source_keys: set[str] = set()
        self._all_target_keys: set[str] = set(output_keys)

        self._topological_order: list[str] = []
        self._input_keys = input_keys
        self.random_keys: set[str] = set()

        self.output_dict: dict[str, str] = {key: key for key in output_keys}
        self._temp_connection_info: dict[GConnection, GConnection] = {}

        self.unique_model_table: dict[str, GConnection] = {}
        self.value_table: dict[str, DataType | ValueType] = {}

        self.data_store: StaticDataStore[DataType] = StaticDataStore(backend, inference)

        self.constraint_solver: ConstraintSolver = deepcopy(solver, memo=memo)

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
    def random_seeds(self) -> dict[str, int]:
        return self.data_store.random_seeds

    def set_random_seed_keys(self, seed_keys: set[str]) -> None:
        self.data_store.set_random_seed_keys(seed_keys)

    def set_random_seed_values(self, **seed_mapping: int) -> None:
        self.data_store.set_random_seed_values(**seed_mapping)

    def update_cached_data(self, updates: Updates) -> set[str]:
        return self.data_store.update_cached_data(updates)

    def add_value(self, model: Operator, keys: dict[str, str]) -> None:
        output_key = keys[Operator.output_key]

        if model.random_keys:
            self.random_keys |= {keys[key] for key in model.random_keys}

        node = Node(model, {})

        # Create output connection of the new Node.
        out_conn = GConnection(node, output_key, [], [], set())

        self.connections[output_key] = out_conn
        node.connections[Operator.output_key] = out_conn

        # Create input connections
        for inner_key, outer_key in keys.items():
            if inner_key == Operator.output_key:
                continue

            conn = self.connections.get(outer_key, None)
            # New input
            if conn is None:
                conn = GConnection(None, outer_key, [], [], set())
                self.connections[outer_key] = conn

            self._all_source_keys.add(conn.key)
            conn.connections.add(out_conn)
            node.connections[inner_key] = conn

        self.nodes[model] = node

        self._all_target_keys.add(output_key)
        self._topological_order.append(node.connections[Operator.output_key].key)

        for conn in node.connections.values():
            self._update_connection_keys(conn)

        self._update_all_source_keys()
        self._update_all_target_keys()

    def collapse_model_keys(self, output_key: str, new_reference_key: str) -> None:
        # If a model removed, the models that uses the output of the removed model
        # should be updated with the new reference key.
        for key, value in self._temp_connection_info.items():
            if value == output_key:
                self._temp_connection_info[key] = self.connections[new_reference_key]

        for key_str, value_str in self.output_dict.items():
            if value_str == output_key:
                self.output_dict[key_str] = new_reference_key

    def update_output_keys(self, output_key: str, new_reference_key: str) -> bool:
        if output_key not in self.output_dict:
            return False

        self.output_dict[output_key] = new_reference_key
        return True

    @property
    def topological_order(self) -> list[str]:
        return self._topological_order

    @property
    def all_target_keys(self) -> set[str]:
        return self._all_target_keys

    @property
    def all_source_keys(self) -> set[str]:
        return self._all_source_keys

    def _update_topological_order(self) -> None:
        self._topological_order = [
            node.connections[Operator.output_key].key for node in self.nodes.values()
        ]

    def _update_all_source_keys(self) -> None:
        self._all_source_keys = {
            conn.key
            for item in self.nodes.values()
            for key, conn in item.connections.items()
            if key != "output"
        }

    def _update_all_target_keys(self) -> None:
        self._all_target_keys = {
            conn.key
            for item in self.nodes.values()
            for key, conn in item.connections.items()
            if key == "output"
        }

    def _update_connection_keys(self, connection: GConnection) -> None:
        source_keys: list[str] = []
        target_keys: list[str] = []

        if connection.node is not None:
            for inner_key, conn in connection.node.connections.items():
                if inner_key == Operator.output_key:
                    continue
                key = conn.key
                source_keys.append(key)

        def get_target_keys(connection: GConnection) -> list[str]:
            target_keys: list[str] = []
            for conn in connection.connections:
                target_keys.append(conn.key)

            return target_keys

        target_keys += get_target_keys(connection)
        if (
            connection.node is not None
            and connection.key != connection.node.connections[Operator.output_key].key
            and connection.node.connections[Operator.output_key].key in self.connections
        ):
            target_keys.append(connection.node.connections[Operator.output_key].key)

        # Make sure connection key registered all_source and all_target keys
        if len(target_keys) > 0:
            self._all_source_keys.add(connection.key)
        if len(source_keys) > 0:
            self._all_target_keys.add(connection.key)

        connection.target_keys = list(target_keys)
        connection.source_keys = list(source_keys)

    def get_model(self, key: str) -> Operator:
        conn = self.connections.get(key, None)
        if conn is None or conn.node is None:
            raise ValueError(f"Model not found for key: {key}")

        return conn.node.model

    def get_model_out_key(self, model: Operator) -> str | None:
        node = self.nodes.get(model, None)
        if node is None:
            return None
        return node.connections[Operator.output_key].key

    def get_model_outer_key(self, model: Operator, inner_key: str) -> str:
        return self.nodes[model].connections[inner_key].key

    def get_model_connections(self, model: Operator):  # type: ignore
        return self.nodes[model].connections.values()

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

    def prune_duplicate_nodes(
        self,
        data: dict[str, IOHyperEdge],
        constant_keys: Mapping[str, DataType | MainValueType],
    ) -> None:
        reverse_data_memo = {value: key for key, value in self.data_memo.items()}

        updates = Updates()

        for node in list(self.nodes.values()):
            if node.connections["output"].key in self.data_store.data_values:
                self._remove_node(node)
                continue

            if isinstance(node.model, BufferOp):
                input_conn = node.connections["input"]
                input_conn = self._temp_connection_info.get(input_conn, input_conn)
                output_conn = node.connections["output"]
                self.update_output_keys(output_conn.key, input_conn.key)
                self._temp_connection_info[output_conn] = input_conn

                input_conn.connections.discard(output_conn)
                input_conn.connections |= output_conn.connections

                # Update target conn source keys
                for target_conn in list(output_conn.connections):
                    if target_conn.node is None:
                        continue

                    for key, target_conn_source in target_conn.node.connections.items():
                        if target_conn_source == output_conn:
                            target_conn.node.connections[key] = input_conn
                            self._update_connection_keys(target_conn)

                    output_conn.connections.discard(target_conn)

                self._update_connection_keys(input_conn)
                self._update_connection_keys(output_conn)
                self._remove_node(node)
                continue

            # Check duplicate
            conn = self._is_duplicate(node, data, constant_keys)
            if conn is not None:
                pruned_key = node.connections["output"].key
                source_key = conn.key

                ## Update Data Memo
                pruned_data = self.all_data[pruned_key]
                remained_data = self.all_data[source_key]

                # find the occurrence of pruned data in data memo and replace it with
                # remained data
                logical_id = reverse_data_memo[pruned_data]
                self.data_memo[logical_id] = remained_data

                # Match shapes
                updates |= remained_data.match(pruned_data)

                # Finally prune the node
                self._prune_node(node, conn)

        self.data_store.update_cached_data(updates)
        self.constraint_solver(updates)

    def _is_duplicate(
        self,
        node: Node,
        data: dict[str, IOHyperEdge],
        constant_keys: Mapping[str, DataType | MainValueType],
    ) -> GConnection | None:
        # Model id is a unique key for unique operation
        model_id: list[str] = []
        for key, conn in node.connections.items():
            # We do not consider output and cache keys, when determining model id.
            if key == "output" or "cache" in key:
                continue

            # Extract value from data or static_keys
            value: DataType | AllValueType
            if conn.key in data and data[conn.key].value is not TBD:
                value = data[conn.key].value
            else:
                value = constant_keys.get(conn.key, TBD)

            # If connection is valued, then compare values.
            if not isinstance(value, ToBeDetermined):
                for value_key, ref_value in self.value_table.items():
                    if type(ref_value) is not type(value):
                        is_equal: bool = False
                    # Check tensors are equal
                    elif self.is_tensor_type(ref_value) and self.is_tensor_type(value):
                        is_equal = (
                            id(ref_value) == id(value)
                            or ref_value.shape == value.shape  # type: ignore
                            and (ref_value == value).all().item()  # type: ignore
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
                model_id.append(conn.key)

        final_model_id = "-".join(model_id) + f"-{node.model.formula_key}"

        if final_model_id in self.unique_model_table:
            return self.unique_model_table[final_model_id]

        self.unique_model_table[final_model_id] = node.connections["output"]
        return None

    def _prune_node(self, node: Node, conn: GConnection) -> None:
        self.collapse_model_keys(node.connections["output"].key, conn.key)

        # Update source and target keys of node connections
        for item1 in node.connections["output"].connections:
            if item1.node is None:
                continue

            for key, item2 in item1.node.connections.items():
                if item2 == node.connections["output"]:
                    item1.node.connections[key] = conn
                    self._update_connection_keys(item1)

        conn.connections |= node.connections["output"].connections

        # Remove node connections
        for conn_ in node.connections.values():
            if conn_.node == node:
                self._remove_conn(conn_)
            else:
                for item in list(conn_.connections):
                    if item.node == node:
                        conn_.connections.discard(item)
                    self._update_connection_keys(item)
            self._update_connection_keys(conn_)

        if (
            key := node.connections[Operator.output_key].key
        ) not in self.output_keys and key in self._all_target_keys:
            self._all_target_keys.remove(key)

        self.nodes.pop(node.model)

        self._update_connection_keys(conn)
        self._update_all_source_keys()
        self._update_all_target_keys()
        self._update_topological_order()

    def _remove_node(self, node: Node) -> None:
        connections = set(node.connections.values())
        output_conn = node.connections[Operator.output_key]

        # To remove node, node should not be used any other nodes or
        # Output of this node is already cached, so we can remove this node.
        if len(output_conn.connections) == 0 or output_conn.key in self.all_data:
            for conn in connections - {output_conn}:
                conn.connections -= connections

                self._update_connection_keys(conn)

                if len(conn.target_keys) == 0 and len(conn.source_keys) == 0:
                    # Connection is not used by any other connections
                    self._remove_conn(conn)

            if (
                len(output_conn.connections) == 0
                and output_conn.key not in self.output_dict.values()
            ):
                self._remove_conn(output_conn)

            else:
                output_conn.node = None

            self.nodes.pop(node.model)
            node.connections.clear()

        else:
            raise ValueError(
                "Node can not be removed, because it is used by other nodes!"
            )

        self._update_all_source_keys()
        self._update_all_target_keys()
        self._update_topological_order()

    def _remove_conn(self, conn: GConnection) -> None:
        if conn.key in self.connections and conn.key not in self.output_dict.values():
            self.remove_key_from_store(conn.key, hard_remove=True)

        self.connections.pop(conn.key, None)

        # Remove connection from other connections
        if conn.node is not None:
            for conn_ in conn.node.connections.values():
                if conn.key in conn_.target_keys:
                    conn_.target_keys.remove(conn.key)

        if conn.key in self._all_source_keys:
            self._all_source_keys.remove(conn.key)

        if conn.key in self._all_target_keys:
            self._all_target_keys.remove(conn.key)

    def remove_key(self, key: str) -> None:
        if key in self.output_dict:
            self.output_dict.pop(key)

        if (conn := self.get_connection(key)) is not None and conn.node is not None:
            self._remove_node(conn.node)
        elif conn is not None:
            self._remove_conn(conn)

    def infer_ignore_step(
        self, key: str, keys: set[str], queue: set[str], from_source: bool
    ) -> None:
        forward_key_fn: Callable[[str, bool], list[str]]
        if from_source:
            forward_key_fn = self.get_target_keys
            backward_key_fn = self.get_source_keys
            all_keys = self.all_source_keys
        else:
            forward_key_fn = self.get_source_keys
            backward_key_fn = self.get_target_keys

            all_keys = self.all_target_keys | self.output_dict.keys()

        if key in all_keys:
            for value in forward_key_fn(key, include_outputs=True):
                if value not in keys:
                    value_mapping = backward_key_fn(value, include_outputs=True)
                    if set(value_mapping).issubset(keys) and (
                        value not in self.output_keys
                    ):
                        keys.add(value)
                        queue.add(value)

    def get_models(self) -> KeysView[Operator]:
        return self.nodes.keys()

    def infer_static_keys(self) -> Updates:
        """Infers the static keys and calculates
        the static values during the inference.
        """
        statics = self.data_store.data_values
        queue = set(statics.keys())
        updates = Updates()
        while queue:
            key = queue.pop()
            if (key not in self.all_source_keys) or key in self.unused_keys:
                continue

            for value in self.get_target_keys(key):
                # Value is already in statics or unused keys, then skip.
                if value in (statics.keys() | self.unused_keys):
                    continue

                value_mapping = self.get_source_keys(value)

                # To infer a model, all of its input keys should be in statics.
                if not set(value_mapping).issubset(statics.keys()):
                    continue

                model = self.get_model(value)

                # TODO: Move this outside of while loop
                # after CBackend is completely implemented.
                fn_dict = (
                    self.backend.primitive_function_dict
                    | self.backend.registered_primitives
                )

                static_value: DataType | MainValueType

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
                    data = self.data_store._all_data[value]
                    if is_make_array_required(data):
                        static_value = self.backend.array(static_value)

                _queue, _updates = self.add_static_data(value, static_value)
                queue |= _queue
                updates |= _updates
        return updates

    def set_static_keys(
        self,
        static_keys: dict[str, DataType | MainValueType],
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
        self, key: str, value: DataType | MainValueType
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
                updates |= data.set_value(value)
            else:
                assert not isinstance(value, MainValueInstance | ToBeDetermined)
                # Find type of tensor and set.
                val_type = self.data_store._infer_tensor_value_type(value)
                updates |= data.set_type(Tensor[val_type])  # type: ignore
                assert data.shape is not None
                # Find shape of tensor and set.
                shape = list(value.shape)
                updates |= data.shape.set_values(shape)
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
                and key not in self.input_keys
            ) or (key in self.input_keys and value.value is not TBD):
                self.data_store._set_data_value(key, value)
            elif key in self.input_keys:
                self.data_store._runtime_static_keys.add(key)
            else:
                if value.value is not TBD:
                    self.data_store._set_data_value(key, value)
                else:
                    self.data_store.intermediate_non_differentiables[key] = value

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
