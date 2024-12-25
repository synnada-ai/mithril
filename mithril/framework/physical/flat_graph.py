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

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...core import DataType, GenericDataType
from ..common import (
    TBD,
    AllValueType,
    IOHyperEdge,
    MainValueType,
    ToBeDetermined,
    ValueType,
)
from ..logical import Buffer
from ..logical.primitive import PrimitiveModel


@dataclass
class Connection:
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
    connections: set[Connection]

    def __hash__(self):
        return hash(id(self))


@dataclass
class Node:
    """A node representing a primitive model and its connections in the graph.

    Attributes:
        model (PrimitiveModel): The primitive model associated with this node.
        connections (dict[str, Connection]): A dictionary mapping connection names
            to Connection objects.

    Note:
        The key "output" in the connections dictionary is reserved for the output
        connection of the node.
    """

    model: PrimitiveModel
    connections: dict[str, Connection]

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)

    def __repr__(self) -> str:
        return f"{self.model}"


class FlatGraph(GenericDataType[DataType]):
    # input -> outputkeys
    def __init__(self, input_keys: set[str], output_keys: set[str]) -> None:
        self.nodes: dict[PrimitiveModel, Node] = {}
        self.connections: dict[
            str, Connection
        ] = {}  # Assumed connections added in topological order.
        self._all_source_keys: set[str] = set()
        self._all_target_keys: set[str] = set(output_keys)

        self._topological_order: list[str] = []
        self._input_keys = input_keys

        self.output_dict: dict[str, str] = {key: key for key in output_keys}
        self._temp_connection_info: dict[str, str] = {}

        self.unique_model_table: dict[str, Connection] = {}
        self.value_table: dict[str, DataType | ValueType] = {}

    @property
    def hanging_keys(self):
        hanging_keys = (self.all_target_keys - self.all_source_keys) | set(
            self.connections.keys()
        ) - self.all_target_keys - self.all_source_keys

        return hanging_keys - set(self.output_dict.values())

    @property
    def input_keys(self):
        return set(self._input_keys)

    @property
    def output_keys(self):
        return set(self.output_dict.keys())

    @property
    def all_keys(self):
        return (
            set(self.connections.keys())
            | set(self.output_dict.keys())
            | set(self.output_dict.values())
        )

    def add_value(self, model: PrimitiveModel, keys: dict[str, str]):
        output_key = keys[PrimitiveModel.output_key]
        keys = {
            key: self._temp_connection_info.get(value, value)
            for key, value in keys.items()
        }

        # Buffer primitives are not added to the graph
        if isinstance(model, Buffer):
            self.update_output_keys(keys["output"], keys["input"])
            self._temp_connection_info[keys["output"]] = keys["input"]

            if keys["input"] in self.connections:
                self._update_connection_keys(self.connections[keys["input"]])

        else:
            node = Node(model, {})

            # Create output connection of the new Node.
            out_conn = Connection(node, output_key, [], [], set())

            self.connections[output_key] = out_conn
            node.connections[PrimitiveModel.output_key] = out_conn

            # Create input connections
            for inner_key, outer_key in keys.items():
                if inner_key == PrimitiveModel.output_key:
                    continue

                conn = self.connections.get(outer_key, None)
                # New input
                if conn is None:
                    conn = Connection(None, outer_key, [], [], set())
                    self.connections[outer_key] = conn

                self._all_source_keys.add(conn.key)
                conn.connections.add(out_conn)
                node.connections[inner_key] = conn

            self.nodes[model] = node

            self._all_target_keys.add(output_key)
            self._topological_order.append(
                node.connections[PrimitiveModel.output_key].key
            )

            for conn in node.connections.values():
                self._update_connection_keys(conn)

        self._update_all_source_keys()
        self._update_all_target_keys()

    def collapse_model_keys(self, output_key: str, new_reference_key: str):
        # If a model removed, the models that uses the output of the removed model
        # should be updated with the new reference key.
        for key, value in self._temp_connection_info.items():
            if value == output_key:
                self._temp_connection_info[key] = new_reference_key

        for key, value in self.output_dict.items():
            if value == output_key:
                self.output_dict[key] = new_reference_key

    def update_output_keys(self, output_key: str, new_reference_key: str) -> bool:
        if output_key not in self.output_dict:
            return False

        self.output_dict[output_key] = new_reference_key
        return True

    @property
    def topological_order(self):
        return self._topological_order

    @property
    def all_target_keys(self) -> set[str]:
        return self._all_target_keys

    @property
    def all_source_keys(self) -> set[str]:
        return self._all_source_keys

    def _update_topological_order(self):
        self._topological_order = [
            node.connections[PrimitiveModel.output_key].key
            for node in self.nodes.values()
            if node.model is not None
            or node.connections[PrimitiveModel.output_key].key in self.output_keys
        ]

    def _update_all_source_keys(self):
        self._all_source_keys = {
            conn.key
            for item in self.nodes.values()
            for key, conn in item.connections.items()
            if key != "output"
        }

    def _update_all_target_keys(self):
        self._all_target_keys = {
            conn.key
            for item in self.nodes.values()
            for key, conn in item.connections.items()
            if key == "output"
        }

    def _update_connection_keys(self, connection: Connection):
        source_keys: list[str] = []
        target_keys: list[str] = []

        if connection.node is not None:
            for inner_key, conn in connection.node.connections.items():
                if inner_key == PrimitiveModel.output_key:
                    continue
                key = conn.key
                source_keys.append(key)

        def get_target_keys(connection: Connection):
            target_keys: list[str] = []
            for conn in connection.connections:
                target_keys.append(conn.key)

            return target_keys

        target_keys += get_target_keys(connection)
        if (
            connection.node is not None
            and connection.key
            != connection.node.connections[PrimitiveModel.output_key].key
            and connection.node.connections[PrimitiveModel.output_key].key
            in self.connections
        ):
            target_keys.append(
                connection.node.connections[PrimitiveModel.output_key].key
            )

        # Make sure connection key registered all_source and all_target keys
        if len(target_keys) > 0:
            self._all_source_keys.add(connection.key)
        if len(source_keys) > 0:
            self._all_target_keys.add(connection.key)

        connection.target_keys = list(target_keys)
        connection.source_keys = list(source_keys)

    def get_model(self, key) -> PrimitiveModel:
        conn = self.connections.get(key, None)
        if conn is None or conn.node is None:
            raise ValueError(f"Model not found for key: {key}")

        return conn.node.model

    def get_model_out_key(self, model: PrimitiveModel):
        node = self.nodes.get(model, None)
        if node is None:
            return None
        return node.connections[PrimitiveModel.output_key].key

    def get_model_outer_key(self, model: PrimitiveModel, inner_key: str):
        return self.nodes[model].connections[inner_key].key

    def get_model_connections(self, model: PrimitiveModel):
        return self.nodes[model].connections.values()

    def get_connection(self, key: str):
        return self.connections.get(key, None)

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
    ) -> dict[str, str]:
        pruned_keys: dict[str, str] = {}
        for node in list(self.nodes.values()):
            conn = self._is_duplicate(node, data, constant_keys)
            if conn is None:
                continue

            key = node.connections["output"].key
            self._prune_node(node, conn)
            pruned_keys[key] = conn.key

        return pruned_keys

    def _is_duplicate(
        self,
        node: Node,
        data: dict[str, IOHyperEdge],
        constant_keys: Mapping[str, DataType | MainValueType],
    ):
        if node.model is None:
            return

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
                            or ref_value.shape == value.shape
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

        final_model_id = "-".join(model_id) + f"-{node.model._formula_key}"

        if final_model_id in self.unique_model_table:
            return self.unique_model_table[final_model_id]

        self.unique_model_table[final_model_id] = node.connections["output"]

    def _prune_node(self, node: Node, conn: Connection):
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
            key := node.connections[PrimitiveModel.output_key].key
        ) not in self.output_keys and key in self._all_target_keys:
            self._all_target_keys.remove(key)

        if node.model is not None:
            self.nodes.pop(node.model)

        self._update_connection_keys(conn)
        self._update_all_source_keys()
        self._update_all_target_keys()
        self._update_topological_order()

    def _remove_node(self, node: Node):
        connections = set(node.connections.values())
        output_conn = node.connections[PrimitiveModel.output_key]

        # To remove node, node should not be used any other nodes!
        if len(output_conn.connections) == 0:
            for conn in connections - {output_conn}:
                conn.connections -= connections

                self._update_connection_keys(conn)

            self._remove_conn(output_conn)
            if node.model is not None:
                self.nodes.pop(node.model)

        self._update_topological_order()

    def _remove_conn(self, conn: Connection):
        self.connections.pop(conn.key, None)

        # Remove connection from other connections
        if conn.node is not None:
            for conn_ in conn.node.connections.values():
                if conn.key in conn_.target_keys:
                    conn_.target_keys.remove(conn.key)

        if conn.key in self._all_source_keys:  # and conn.key not in self.alias_map:
            self._all_source_keys.remove(conn.key)

        if conn.key in self._all_target_keys:  # and conn.key not in self.alias_map:
            self._all_target_keys.remove(conn.key)

    def remove_key(self, key: str):
        if key in self.output_dict:
            self.output_dict.pop(key)

        if (conn := self.get_connection(key)) is not None and conn.node is not None:
            self._remove_node(conn.node)
        elif conn is not None:
            self._remove_conn(conn)

    def infer_ignore_step(
        self, key: str, keys: set[str], queue: set[str], from_source: bool
    ):
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

    def get_models(self):
        return self.nodes.keys()
