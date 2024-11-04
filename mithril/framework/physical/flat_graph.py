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

from dataclasses import dataclass

from ...core import DataType
from ..common import TBD, GenericDataType, MainValueType, Scalar, Tensor, ValueType
from ..logical.primitive import PrimitiveModel


class FlatGraph(GenericDataType[DataType]):
    @dataclass
    class Connection:
        node: FlatGraph.Node | None
        key: str
        source_keys: list[str]
        target_keys: list[str]
        connections: set[FlatGraph.Connection]

        def __hash__(self):
            return hash(id(self))

    @dataclass
    class Node:
        model: PrimitiveModel
        connections: dict[str, FlatGraph.Connection]

        def __hash__(self) -> int:
            return hash(id(self))

        def __eq__(self, other) -> bool:
            return id(self) == id(other)

        def __repr__(self) -> str:
            return f"{self.model}"

    # input -> outputkeys
    def __init__(self, input_keys: set[str], output_keys: set[str]) -> None:
        self.nodes: dict[PrimitiveModel, FlatGraph.Node] = {}
        self.connections: dict[
            str, FlatGraph.Connection
        ] = {}  # Assumed connections added in topological order.
        self._all_source_keys: set[str] = set()
        self._all_target_keys: set[str] = set()
        self.alias_map: dict[str, str] = {}

        self._topological_order: list[str] = []
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.unique_model_table: dict[str, FlatGraph.Connection] = {}
        self.value_table: dict[str, DataType | ValueType] = {}

    def add_value(self, model: PrimitiveModel, keys: dict) -> bool:
        connections = {}
        node = FlatGraph.Node(model, {})

        # Add output of the Node
        output_key = keys[PrimitiveModel.output_key]
        cycle_occured = output_key in self.connections

        out_conn = FlatGraph.Connection(node, output_key, [], [], set())
        target_conns = set()

        # Model addition order is wrong, therefore a cycle occured.
        # Output of this model is created by another model input, remove
        # that connection and recreate connection as this model output.
        if cycle_occured:
            for item in self.nodes.values():
                output_conn = self.connections.get(output_key)
                if output_conn is None:
                    continue

                if output_conn in item.connections.values():
                    target_conns.add(item.connections[PrimitiveModel.output_key])

                    idx = list(item.connections.values()).index(output_conn)
                    key = list(item.connections.keys())[idx]
                    item.connections[key] = out_conn

                    self._remove_conn(output_conn)

        out_conn.connections |= target_conns

        self.connections[output_key] = out_conn
        connections[PrimitiveModel.output_key] = out_conn
        self._all_target_keys.add(output_key)

        # Add input connections
        for inner_key, outer_key in keys.items():
            if inner_key == PrimitiveModel.output_key:
                continue

            conn = self.connections.get(outer_key, None)
            # New input
            if conn is None:
                conn = FlatGraph.Connection(None, outer_key, [], [], set())
                self.connections[outer_key] = conn

            self._all_source_keys.add(conn.key)
            conn.connections.add(out_conn)
            connections[inner_key] = conn

        self.nodes[model] = node
        node.connections = connections

        self._topological_order.append(node.connections[PrimitiveModel.output_key].key)

        for conn in connections.values():
            self._update_connection_keys(conn)

        return cycle_occured

    @property
    def topological_order(self):
        return self._topological_order

    @property
    def all_target_keys(self):
        return self._all_target_keys

    @property
    def all_source_keys(self):
        return self._all_source_keys

    def _reorder_connections(self):
        queue = list(self.input_keys)
        visited_keys = []

        while queue:
            key = queue.pop()
            if key in visited_keys:
                continue

            visited_keys.append(key)
            # TODO: Cyclic extension bug is solved temporarily
            # (see test_cyclic_extension in test_scripts.py)
            # find a better solution for this.
            new_target_keys = self.get_target_keys(key)
            for target_key in new_target_keys:
                source_keys = self.get_source_keys(target_key)

                node = self.connections[target_key].node
                local_keys = []
                if node is not None:
                    local_keys = list(node.connections.keys())

                if "cache" in local_keys:
                    source_keys.pop(local_keys.index("cache") - 1)
                if set(source_keys).issubset(visited_keys):
                    queue.append(target_key)

        for key in self.input_keys:
            visited_keys.remove(key)

        nodes = {}
        for key in visited_keys:
            model = self.get_model(key)
            if model is None:
                continue
            nodes[model] = self.nodes[model]

        # If graph is not completed do not reorder nodes!
        if len(nodes) == len(self.nodes):
            self.nodes = nodes
            self._update_topological_order()

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
        self._all_source_keys |= set(self.alias_map.values())

    def _update_all_target_keys(self):
        self._all_target_keys = {
            conn.key
            for item in self.nodes.values()
            for key, conn in item.connections.items()
            if key == "output"
        }
        self._all_target_keys |= set(self.alias_map.keys())

    def _update_connection_keys(self, connection: FlatGraph.Connection):
        source_keys = []
        target_keys = []

        if connection.node is not None:
            for inner_key, conn in connection.node.connections.items():
                if inner_key == PrimitiveModel.output_key:
                    continue
                key = conn.key
                while key in self.alias_map:
                    key = self.alias_map[key]
                source_keys.append(key)

        def get_target_keys(connection: FlatGraph.Connection):
            target_keys = []
            for conn in connection.connections:
                if conn.key in self.alias_map:
                    if conn.key in self.output_keys:
                        target_keys.append(conn.key)
                    target_keys += get_target_keys(conn)
                else:
                    target_keys.append(conn.key)
            target_keys += [
                key for key, value in self.alias_map.items() if value == connection.key
            ]

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

    def get_alias_key(self, key):
        while key in self.alias_map:
            key = self.alias_map[key]
        return key

    def get_model(self, key) -> None | PrimitiveModel:  # O(1)
        conn = self.connections.get(key, None)
        if conn is None or conn.node is None:
            return None
        return conn.node.model

    def get_model_out_key(self, model):
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

    def get_source_keys(self, key: str, include_aliases: bool = False):
        source_keys = (
            self.connections[key].source_keys
            if key in self.connections
            else self.alias_map.get(key, [])
        )
        source_keys = [source_keys] if isinstance(source_keys, str) else source_keys
        return source_keys

    def get_target_keys(self, key: str, include_aliases: bool = False):
        target_keys = (
            list(self.connections[key].target_keys) if key in self.connections else []
        )

        if include_aliases:
            alias_values = list(self.alias_map.values())
            alias_keys = list(self.alias_map.keys())
            used_as_source = set(target_keys).intersection(alias_values)
            target_keys += [
                alias_keys[alias_values.index(key)] for key in used_as_source
            ]

        return target_keys

    def prune_duplicate_nodes(
        self,
        data: dict[str, Tensor | Scalar],
        static_keys: dict[str, DataType]
        | dict[str, MainValueType]
        | dict[str, DataType | MainValueType],
    ):
        pruned_keys = {}
        for node in list(self.nodes.values()):
            conn = self._is_duplicate(node, data, static_keys)
            if conn is None:
                continue

            key = node.connections["output"].key
            self.alias_map[key] = conn.key
            self._prune_node(node, conn)
            pruned_keys[key] = conn.key

        return pruned_keys

    def _is_duplicate(
        self,
        node: FlatGraph.Node,
        data: dict[str, Tensor | Scalar],
        static_keys: dict[str, DataType]
        | dict[str, MainValueType]
        | dict[str, DataType | MainValueType],
    ):
        if node.model is None:
            return

        # Model id is a unique key for unique operation
        model_id = []
        for key, conn in node.connections.items():
            # We do not consider output and cache keys, when determining model id.
            if key == "output" or "cache" in key:
                continue

            # Extract value from data or static_keys
            value: DataType | MainValueType | None
            if conn.key in data and data[conn.key].value is not None:
                value = data[conn.key].value
            else:
                value = static_keys.get(conn.key)

            # For scalars None is also a value
            is_valued = (
                (conn.key in data and isinstance(data[conn.key], Scalar))
                or value is not None
            ) and value is not TBD

            # If connection is valued, then compare values.
            if is_valued:
                for value_key, ref_value in self.value_table.items():
                    if type(ref_value) is not type(value):
                        is_equal = False
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
                    self.value_table[str(len(self.value_table))] = value
                    model_id.append(str(len(self.value_table) - 1))
            else:
                model_id.append(conn.key)

        final_model_id = "-".join(model_id) + f"-{node.model.formula_key}"

        if final_model_id in self.unique_model_table:
            return self.unique_model_table[final_model_id]
        elif node.model.formula_key == "buffer":
            return node.connections["input"]

        self.unique_model_table[final_model_id] = node.connections["output"]

    def _prune_node(self, node: FlatGraph.Node, conn: FlatGraph.Connection):
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
        self._update_topological_order()

    def _remove_node(self, node: FlatGraph.Node):
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

    def _remove_conn(self, conn: FlatGraph.Connection):
        self.connections.pop(conn.key, None)

        # Remove connection from other connections
        if conn.node is not None:
            for conn_ in conn.node.connections.values():
                if (
                    conn.key in conn_.target_keys
                    and self.alias_map.get(conn.key) != conn_.key
                ):
                    conn_.target_keys.remove(conn.key)

        # Clear alias map
        if conn.key in self.alias_map and conn.key not in self.output_keys:
            for key, value in self.alias_map.items():
                if value == conn.key:
                    self.alias_map[key] = self.alias_map[conn.key]

            if len(self.get_target_keys(conn.key, include_aliases=True)) == 0:
                self.alias_map.pop(conn.key)

        if conn.key in self._all_source_keys and conn.key not in self.alias_map:
            self._all_source_keys.remove(conn.key)

        if conn.key in self._all_target_keys and conn.key not in self.alias_map:
            self._all_target_keys.remove(conn.key)

    def remove_key(self, key: str):
        if key in self.alias_map:
            self.alias_map.pop(key)
        elif (conn := self.get_connection(key)) is not None and conn.node is not None:
            self._remove_node(conn.node)
        elif conn is not None:
            self._remove_conn(conn)

    def get_models(self):
        return self.nodes.keys()
