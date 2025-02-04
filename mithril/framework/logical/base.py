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

import abc
from collections.abc import KeysView, Mapping
from dataclasses import dataclass
from itertools import chain
from types import UnionType
from typing import Any

from ...utils.utils import OrderedSet
from ..common import (
    TBD,
    AssignedConstraintType,
    Connection,
    ConnectionData,
    Connections,
    ConnectionType,
    Constraint,
    ConstraintFunctionType,
    ConstraintSolver,
    IOHyperEdge,
    MainValueType,
    ScalarType,
    ShapeNode,
    ShapesType,
    ShapeTemplateType,
    ShapeType,
    Tensor,
    ToBeDetermined,
    UniadicRecord,
    Updates,
    UpdateType,
    Variadic,
    create_shape_repr,
    get_shapes,
)
from ..constraints import constraint_type_map

__all__ = ["BaseModel", "ExtendInfo"]


@dataclass
class ExtendInfo:
    _model: BaseModel
    _connections: dict[str, ConnectionType]

    def __post_init__(self) -> None:
        external_keys = (
            set(self._model.external_keys)
            | {item.key for item in self._model.conns.couts}
            | {item.key for item in self._model.conns.cins}
        )

        for key in self._connections:
            if key not in external_keys:
                raise KeyError(f"Key '{key}' is not a valid key for the model!")

    @property
    def model(self) -> BaseModel:
        return self._model

    @property
    def connections(self) -> dict[str, ConnectionType]:
        return self._connections


class BaseModel(abc.ABC):
    # Disposable models only used once for entire training session.
    # This attribute is only use for manual backends' code generation.

    # TODO: This can be checked from backend's gradient function dict???
    disposable: bool = False
    # TODO: factory_args should be instance variable not class!
    factory_args: dict[str, Any] = {}

    def __call__(self, **kwargs: ConnectionType) -> ExtendInfo:
        return ExtendInfo(self, kwargs)

    def __init__(self, name: str | None = None, enforce_jit: bool = True) -> None:
        self.parent: BaseModel | None = (
            None  # TODO: maybe set it only to PrimitiveModel / Model.
        )
        self.assigned_shapes: list[ShapesType] = []
        self.assigned_types: dict[
            str,
            type | UnionType | ScalarType | Tensor[int | float | bool],
        ] = {}
        self.assigned_constraints: list[AssignedConstraintType] = []
        self.conns = Connections()
        self.frozen_attributes: list[str] = []
        self.dependency_map = DependencyMap(self.conns)
        self.name = name
        self._enforce_jit = enforce_jit
        self._jittable = True
        self.constraint_solver: ConstraintSolver = ConstraintSolver()
        self.safe_shapes: dict[str, ShapeTemplateType] = {}
        self.is_frozen = False

    @abc.abstractmethod
    def summary(
        self,
        shapes: bool = True,
        types: bool = False,
        symbolic: bool = False,
        name: str | None = None,
        alternative_shapes: bool = False,
        uni_cache: dict[UniadicRecord, str] | None = None,
        var_cache: dict[Variadic, str] | None = None,
    ) -> None:
        raise NotImplementedError("Implement summary method!")

    @property
    def enforce_jit(self) -> bool:
        return self._enforce_jit

    @enforce_jit.setter
    def enforce_jit(self, value: bool) -> None:
        self._enforce_jit = value

    @property
    def jittable(self) -> bool:
        return self._jittable

    @property
    def shapes(
        self,
    ) -> Mapping[str, ShapeTemplateType | list[ShapeTemplateType] | None]:
        return self.get_shapes()

    @property
    def external_keys(self) -> KeysView[str]:
        return self.conns.io_keys

    @property
    def input_keys(self) -> KeysView[str]:
        return self.conns.input_keys

    @property
    def _all_keys(self) -> KeysView[str]:
        return self.conns.all.keys()

    @property
    def output_keys(self) -> list[str]:
        output_keys = list(self.conns.output_keys)
        return output_keys

    def check_extendability(self) -> None:
        # Check possible errors before the extension.
        if self.parent is not None:
            raise AttributeError("Submodel of a model could not be extended!")

    def _get_outermost_parent(self) -> BaseModel:
        model = self
        while model.parent is not None:
            model = model.parent
        return model

    def generate_keys(
        self,
        symbolic: bool = True,
        include_internals: bool = True,
        include_outputs: bool = False,
    ) -> dict[str, str]:
        return {}

    def __setattr__(self, name: str, value: Any) -> None:
        # You need to be careful here to avoid infinite recursion
        if (
            getattr(self, "frozen_attributes", None) is not None
            and name in self.frozen_attributes
        ):
            # To avoid infinite recursion, use the base class's __setattr__ method
            raise AttributeError(
                f"{name} attribute of {self.__class__.__name__} type object is frozen."
            )
        else:
            super().__setattr__(name, value)

    def _freeze(self) -> None:
        self.is_frozen = True

    @abc.abstractmethod
    def extract_connection_info(
        self,
        name_mappings: dict[BaseModel, str],
        data_to_key_map: dict[IOHyperEdge, list[str]] | None = None,
        data_memo: Mapping[int, IOHyperEdge] | None = None,
    ) -> dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]]:
        raise NotImplementedError("Implement extract_connection_info method!")

    def _create_connection(
        self, metadata: IOHyperEdge, key: str, is_key_autogenerated: bool
    ) -> Connection:
        # Check if the key is already exist in the connections object.
        if self.conns.get_connection(key) is not None:
            raise KeyError("Connection with name " + key + " already exists!")

        # Create connection object with given metadata, key
        # and auto-generation status of the key.
        con = Connection(
            key=key,
            metadata=metadata,
            is_key_autogenerated=is_key_autogenerated,
        )
        # Add ConnectionData to the connections object.
        self.conns.add(con.data)
        return con

    def create_connection(self, metadata: IOHyperEdge, key: str) -> ConnectionData:
        con = self._create_connection(metadata, key, False)
        # Set key_origin into metadata
        metadata.key_origin = key
        setattr(self, key, con)
        return con.data

    def _set_shapes(
        self,
        shapes: ShapesType,
        trace: bool = False,
        updates: Updates | None = None,
        **kwargs: ShapeTemplateType,
    ) -> None:
        # Initialize assigned shapes dictionary to store assigned shapes.
        assigned_shapes: dict[str, ShapeTemplateType] = {}

        if updates is None:
            updates = Updates()

        model = self._get_outermost_parent()
        used_keys: dict[str | int, ShapeType] = {}
        shape_nodes: dict[str | Connection, tuple[ShapeNode, str]] = {}
        # TODO: Can this be refactored to use a single loop?
        for key, shape in chain(shapes.items(), kwargs.items()):
            metadata = self.conns.extract_metadata(key)
            given_repr = create_shape_repr(shape, model.constraint_solver, used_keys)
            # Get inner string representation of the metadata and save
            # use this name in order to merge .
            conn = self.conns.get_con_by_metadata(metadata)
            assert conn is not None
            inner_key = conn.key
            shape_nodes[key] = (given_repr.node, inner_key)
            assigned_shapes[inner_key] = shape
        # Apply updates to the shape nodes.
        for key in chain(shapes, kwargs):
            node, _inner_key = shape_nodes[key]
            if (metadata := self.conns.get_data(_inner_key)).is_polymorphic:
                # If edge_type is not defined yet, set it to Tensor since
                # shape is provided.
                updates |= metadata.set_type(Tensor[int | float | bool])
            shape_node = self.conns.get_shape_node(_inner_key)
            assert shape_node is not None
            updates |= shape_node.merge(node)

        if trace:
            self.assigned_shapes.append(assigned_shapes)

        model.constraint_solver(updates)

    def _set_value(
        self,
        key: ConnectionData,
        value: MainValueType | Tensor[int | float | bool] | str,
    ) -> Updates:
        """
        Set value for the given connection.

        Args:
            key (str | Connection): Connection key or Connection object to set value.
            value (MainValueType): Value to set for the given connection.

        Raises:
            KeyError: If the provided key is not a valid IO key.
        """

        if key.key not in self.conns.input_keys:
            raise ValueError("Values of internal and output keys cannot be set.")
        if value != TBD:
            self.conns.cins.discard(key)
        # Data is scalar, set the value directly.
        return key.metadata.set_value(value)

    def set_shapes(
        self, config: ShapesType | None = None, **kwargs: ShapeTemplateType
    ) -> None:
        if config is None:
            config = {}
        self._set_shapes(config, trace=True, updates=None, **kwargs)

    def set_values(
        self,
        config: Mapping[
            str | Connection, Tensor[int | float | bool] | MainValueType | str
        ]
        | Mapping[Connection, Tensor[int | float | bool] | MainValueType | str]
        | Mapping[str, Tensor[int | float | bool] | MainValueType | str]
        | None = None,
        **kwargs: Tensor[int | float | bool] | MainValueType | str,
    ) -> None:
        """
        Set multiple values in the model.
        This method updates values in the outermost model by traversing up the
        parent hierarchy. It performs metadata extraction on self, validity checks
        and updates on the parent model. Finally, it solves constraints
        with the updated values.

        Args:
            config (dict[str | Connection, MainValueType]): A dictionary where
            keys are either strings or Connection objects, and values are
            of type MainValueType.
            **kwargs (MainValueType): Key-value pairs where keys are string names
            of connections present in this model.

        Raises:
            KeyError: If a valid key or Connection is not provided in the values
            dictionary.
        """
        if config is None:
            config = {}
        # Make all value updates in the outermost model.s
        model = self._get_outermost_parent()
        updates = Updates()
        for key, value in chain(config.items(), kwargs.items()):
            # Perform metadata extraction process on self.
            metadata = self.conns.extract_metadata(key)
            # Perform validity check and updates on model.
            if (conn_data := model.conns.get_con_by_metadata(metadata)) is None:
                raise KeyError("Requires valid key or Connection to set values!")
            updates |= model._set_value(conn_data, value)

        # Solve constraints with the updated values.
        model.constraint_solver(updates)

    def set_types(
        self,
        config: Mapping[
            str | Connection,
            type | UnionType | ScalarType | type[Tensor[int | float | bool]],
        ]
        | Mapping[
            Connection,
            type | UnionType | ScalarType | type[Tensor[int | float | bool]],
        ]
        | Mapping[
            str,
            type | UnionType | ScalarType | type[Tensor[int | float | bool]],
        ]
        | None = None,
        **kwargs: type | UnionType | ScalarType | type[Tensor[int | float | bool]],
    ) -> None:
        """
        Set types of any connection in the Model

        This method updates types in given connections.
        connections can be given either as Connection or their string
        equivalent. Giving a valid type for given connections, this method
        will update the connections's types and thereafter runs the
        constraints to update affected connections' types.

        Args:
            values (dict[str | Connection, MainValueType]): A dictionary where
            keys are either strings or Connection objects, and values are
            of type of type or UnionType objects.

        """
        if config is None:
            config = {}
        # Initialize assigned shapes dictionary to store assigned shapes.
        assigned_types: dict[
            str,
            type | UnionType | ScalarType | Tensor[int | float | bool],
        ] = {}

        # Get the outermost parent as all the updates will happen here.
        model = self._get_outermost_parent()
        updates = Updates()
        for key, key_type in chain(config.items(), kwargs.items()):
            metadata = self.conns.extract_metadata(key)
            conn = self.conns.get_con_by_metadata(metadata)
            assert conn is not None
            inner_key = conn.key
            if key_type is Tensor:
                key_type = Tensor[int | float | bool]
            assigned_types[inner_key] = key_type
            updates |= metadata.set_type(key_type)
        # Store assigned types in the model.
        self.assigned_types |= assigned_types
        # Run the constraints for updating affected connections.
        model.constraint_solver(updates)

    def get_shapes(
        self,
        uni_keys: dict[UniadicRecord, str] | None = None,
        var_keys: dict[Variadic, str] | None = None,
        symbolic: bool = True,
        verbose: bool = False,
    ) -> Mapping[str, ShapeTemplateType | list[ShapeTemplateType] | None]:
        return get_shapes(
            data_dict={key: value.metadata for key, value in self.conns.all.items()},
            uniadic_keys=uni_keys,
            varadic_keys=var_keys,
            symbolic=symbolic,
            verbose=verbose,
            key_mappings=self.generate_keys(include_outputs=True),
        )

    def _add_constraint(
        self,
        fn: ConstraintFunctionType,
        keys: list[str],
        types: list[UpdateType] | None = None,
        dependencies: set[Constraint] | None = None,
    ) -> Constraint:
        all_conns = self.conns.all
        hyper_edges = [all_conns[key].metadata for key in keys]

        if dependencies is None:
            dependencies = set()
        unresolved_dependencies = (
            dependencies & self.constraint_solver.constraint_map.keys()
        )
        if types is None:
            types = constraint_type_map.get(fn, [UpdateType.SHAPE, UpdateType.VALUE])
        constr = Constraint(fn=fn, types=types)
        constr.add_dependencies(*unresolved_dependencies)

        self.constraint_solver.constraint_map[constr] = hyper_edges
        for hyper_edge in hyper_edges:
            hyper_edge.add_constraint(constr)

        self.constraint_solver.solver_loop({constr})
        return constr

    def add_constraint(
        self,
        fn: ConstraintFunctionType,
        keys: list[str],
        type: list[UpdateType] | None = None,
        dependencies: set[Constraint] | None = None,
    ) -> Constraint:
        self.assigned_constraints.append({"fn": fn.__name__, "keys": keys})
        return self._add_constraint(fn, keys, type, dependencies)

    @property
    def cin(self) -> Connection:
        if (cin_len := len(self.conns.cins)) != 1:
            raise KeyError(
                f"Currently, there exists {cin_len} canonical inputs, model "
                "should have exactly one canonical input!"
            )
        return next(iter(self.conns.cins)).conn

    @property
    def cout(self) -> Connection:
        if (cout_len := len(self.conns.couts)) != 1:
            raise KeyError(
                f"Currently, there exists {cout_len} canonical outputs, model "
                "should have exactly one canonical output!"
            )

        return next(iter(self.conns.couts)).conn

    def set_cin(self, *connections: str | Connection, safe: bool = True) -> None:
        self.conns.cins = set()
        for given_conn in connections:
            conn = self.conns.get_extracted_connection(given_conn)

            is_valued = conn.metadata.value is not TBD
            if conn not in self.dependency_map.local_input_dependency_map:
                raise ValueError(
                    "To set a connection as canonical input, connection must be an "
                    "input connection!"
                )
            elif is_valued:
                if safe:
                    raise ValueError(
                        "To set a connection as canonical input, "
                        "connection must be unvalued!"
                    )
            else:
                self.conns.cins.add(conn)

    def set_cout(self, *connections: str | Connection, safe: bool = True) -> None:
        self.conns.couts = set()
        for given_conn in connections:
            conn = self.conns.get_extracted_connection(given_conn)
            is_valued = conn.metadata.value is not TBD
            if conn not in self.dependency_map.local_output_dependency_map or is_valued:
                if safe:
                    raise ValueError(
                        "To set a connection as canonical output, "
                        "connection must be an output connection!"
                    )
            else:
                self.conns.couts.add(conn)

    def _match_hyper_edges(self, left: IOHyperEdge, right: IOHyperEdge) -> Updates:
        # if type(left.data) is not type(right.data):
        l_type = left.edge_type
        r_type = right.edge_type
        if ((l_type is Tensor) ^ (r_type is Tensor)) and (
            ToBeDetermined not in (l_type, r_type)
        ):
            raise TypeError(
                "Types of connections are not consistent. Check connection types!"
            )

        right_connections = self.conns.connections_dict.pop(right, set())
        self.conns.connections_dict[left] |= right_connections

        for conns_obj in right_connections:
            conns = conns_obj.metadata_dict.pop(right, None)

            if conns is None:
                raise KeyError(
                    "Requires merged connection's metadata to contain its "
                    "Connections object."
                )

            if left not in conns_obj.metadata_dict:
                conns_obj.metadata_dict[left] = conns
            elif left in conns_obj.metadata_dict:
                conns_obj.metadata_dict[left] |= conns

            for conn in conns:
                conn.metadata = left

        # Update IOHyperEdge's in constraint solver.
        self.constraint_solver.update_constraint_map(left, right)
        # Match data of each IOHyperEdge's.
        updates = left.match(right)
        return updates

    def get_models_in_topological_order(self) -> list[BaseModel]:
        dependency_map = self.dependency_map.local_output_dependency_map
        graph = {
            info[0]: OrderedSet(
                [dependency_map[spec][0] for spec in info[1] if spec in dependency_map]
            )
            for info in dependency_map.values()
        }
        top_order: list[BaseModel] = list()
        visited: set[BaseModel] = set()
        for model in graph:
            if model not in top_order:
                BaseModel._reverse_dfs(model, graph, top_order, visited)
        return top_order

    @staticmethod
    def _reverse_dfs(
        node: BaseModel,
        graph: dict[BaseModel, OrderedSet[BaseModel]],
        top_order: list[BaseModel],
        visited: set[BaseModel],
    ) -> None:
        visited.add(node)
        for m in graph[node]:
            if m not in visited:
                BaseModel._reverse_dfs(m, graph, top_order, visited)
        top_order.append(node)


class DependencyMap:
    """
    Depedency Map stores relations between connections and models
    TODO: Write doc
    """

    def __init__(self, connections: Connections) -> None:
        self.conns = connections
        # Stores relation between input_keys to dependent output connections
        self._global_input_dependency_map: dict[
            ConnectionData, OrderedSet[ConnectionData]
        ] = {}
        # Stores relation between output_keys to dependent input connections
        self._global_output_dependency_map: dict[
            ConnectionData, OrderedSet[ConnectionData]
        ] = {}

        # Caches relations during extend to avoid traverse whole graph
        self._global_input_dependency_map_cache: dict[
            ConnectionData, OrderedSet[ConnectionData]
        ] = {}
        self._global_output_dependency_map_cache: dict[
            ConnectionData, OrderedSet[ConnectionData]
        ] = {}
        # Stores releation between local input keys to dependent local
        # output connections
        self.local_input_dependency_map: dict[
            ConnectionData, list[tuple[BaseModel, OrderedSet[ConnectionData]]]
        ] = {}
        # Stores releation between local output keys to dependent local
        # input connections
        self.local_output_dependency_map: dict[
            ConnectionData, tuple[BaseModel, OrderedSet[ConnectionData]]
        ] = {}

    # Add new model to dependency map, model_dag is created in extend
    def add_model_dag(
        self, model: BaseModel, model_dag: dict[str, ConnectionData]
    ) -> None:
        updated_conns: OrderedSet[ConnectionData] = OrderedSet()
        for local_key, conn in model_dag.items():
            if local_key in model.conns.input_keys:
                specs: OrderedSet[ConnectionData] = OrderedSet(
                    [
                        model_dag[conn.key]
                        for conn in model.dependency_map.get_dependent_output_conns(
                            local_key
                        )
                        if model_dag.get(conn.key) is not None
                    ]
                )
                self.local_input_dependency_map.setdefault(conn, []).append(
                    (model, specs)
                )
                updated_conns.add(conn)
            else:
                specs = OrderedSet(
                    [
                        model_dag[conn.key]
                        for conn in model.dependency_map.get_dependent_input_conns(
                            local_key
                        )
                        if model_dag.get(conn.key) is not None
                    ]
                )
                self.local_output_dependency_map[conn] = (model, specs)

                updated_conns.add(conn)
                self.cache_internal_references(conn, specs)

            if self.look_for_cyclic_connection(conn, specs):
                raise KeyError(
                    f"There exists a cyclic subgraph between {conn.key} key and "
                    f"{[spec.key for spec in specs]} key(s)!"
                )

        self.update_globals(updated_conns)

    # Caches extended connections to avoid traverse
    def cache_internal_references(
        self, output_conn: ConnectionData, dependent_conns: OrderedSet[ConnectionData]
    ) -> None:
        # Be sure all input and output keys has cache entry
        for conn in self.conns.input_connections:
            self._global_input_dependency_map_cache.setdefault(conn, OrderedSet())

        for conn in self.conns.output_connections:
            self._global_output_dependency_map_cache.setdefault(conn, OrderedSet())

        all_dependent_input_conns: OrderedSet[ConnectionData] = OrderedSet()

        for input_conn in dependent_conns:
            # Extend from output, update global input dependency map with new
            # output conn
            for dependent_conn in self._global_output_dependency_map_cache.get(
                input_conn, OrderedSet()
            ):
                all_dependent_input_conns.add(dependent_conn)

                # If not extend from input add
                if output_conn not in self._global_input_dependency_map_cache:
                    self._global_input_dependency_map_cache[dependent_conn].add(
                        output_conn
                    )

            # Input is not internal
            if input_conn in self.conns.input_connections:
                all_dependent_input_conns.add(input_conn)

                # If not extend from input add
                if output_conn not in self._global_input_dependency_map_cache:
                    self._global_input_dependency_map_cache[input_conn].add(output_conn)

        # Extend from Input
        if output_conn in self._global_input_dependency_map_cache:
            dependent_output_conns = self._global_input_dependency_map_cache.pop(
                output_conn
            )

            # Update global_input_dependencies
            for dependent_input_conn in all_dependent_input_conns:
                if dependent_input_conn in self._global_input_dependency_map_cache:
                    self._global_input_dependency_map_cache[
                        dependent_input_conn
                    ].discard(output_conn)
                    self._global_input_dependency_map_cache[dependent_input_conn] |= (
                        dependent_output_conns
                    )

            # Update global_output_dependencies
            for dependent_output_conn in dependent_output_conns:
                if dependent_output_conn in self._global_output_dependency_map_cache:
                    self._global_output_dependency_map_cache[
                        dependent_output_conn
                    ].discard(output_conn)
                    self._global_output_dependency_map_cache[dependent_output_conn] |= (
                        all_dependent_input_conns
                    )

        else:
            self._global_output_dependency_map_cache.setdefault(
                output_conn, OrderedSet()
            )
            self._global_output_dependency_map_cache[output_conn] |= (
                all_dependent_input_conns
            )

    # Caches given input connection for later usage
    def cache_conn_input_dependency(self, conn: ConnectionData) -> None:
        if conn not in self._global_input_dependency_map_cache:
            dependents = self.get_output_key_dependency(conn.key)
            self._global_input_dependency_map_cache[conn] = dependents

    # Caches given output connection for later usage
    def cache_conn_output_dependency(self, conn: ConnectionData) -> None:
        if conn not in self._global_output_dependency_map_cache:
            dependents = self.get_input_key_dependency(conn.key)
            self._global_output_dependency_map_cache[conn] = dependents

    # Returns dependent input keys of given output key
    def get_dependent_input_conns(self, key: str) -> OrderedSet[ConnectionData]:
        if (given_conn := self.conns.get_connection(key)) is None:
            raise KeyError("Given key does not belong to the Model!")
        dependent_conns: OrderedSet[ConnectionData] = OrderedSet()
        if key in self.conns.output_keys:
            dependent_conns = self._global_output_dependency_map[given_conn]
        elif (
            conn := self.conns.get_connection(key)
        ) in self._global_output_dependency_map_cache:
            return self._global_output_dependency_map_cache[conn]
        else:
            return self.get_output_key_dependency(key)

        return dependent_conns

    # Return dependent output keys of given input key
    def get_dependent_output_conns(self, key: str) -> OrderedSet[ConnectionData]:
        if (given_conn := self.conns.get_connection(key)) is None:
            raise KeyError("Given key does not belong to the Model!")
        dependent_conns: OrderedSet[ConnectionData] = OrderedSet()
        if key in self.conns.input_keys:
            dependent_conns = self._global_input_dependency_map[given_conn]
        elif (
            conn := self.conns.get_connection(key)
        ) in self._global_input_dependency_map_cache:
            return self._global_input_dependency_map_cache[conn]
        else:
            return self.get_input_key_dependency(key)

        return dependent_conns

    # Update dependecy map
    def update_all_keys(self) -> None:
        # This method is used in freeze, because in freeze dependencies changed
        # without updating dependency map.
        self.update_globals(
            OrderedSet(self.conns.input_connections)
            | OrderedSet(self.conns.output_connections)
        )

    # Get dependent output connections if given input connection is cached
    # else returns None
    def _get_from_input_cache(self, conn: ConnectionData) -> OrderedSet[ConnectionData]:
        dependent_conns = self._global_input_dependency_map_cache.get(
            conn, OrderedSet()
        )
        dependent_conns = OrderedSet(
            [
                dependent_conn
                for dependent_conn in dependent_conns
                if dependent_conn.key in self.conns.output_keys
            ]
        )
        return dependent_conns

    # Get dependent input connections if given output connection is cached
    # else returns None
    def _get_from_output_cache(
        self, conn: ConnectionData
    ) -> OrderedSet[ConnectionData]:
        dependent_conns = self._global_output_dependency_map_cache.get(
            conn, OrderedSet()
        )
        dependent_conns = OrderedSet(
            [
                dependent_conn
                for dependent_conn in dependent_conns
                if dependent_conn.key in self.conns.input_keys
            ]
        )
        return dependent_conns

    # Update global dependency maps wrt given connections
    def update_globals(self, updated_conns: OrderedSet[ConnectionData]) -> None:
        for input_conn in self.conns.input_connections:
            self._global_input_dependency_map.setdefault(input_conn, OrderedSet())

        for output_conn in self.conns.output_connections:
            self._global_output_dependency_map.setdefault(output_conn, OrderedSet())

        visited_keys: OrderedSet[ConnectionData] = OrderedSet()
        while updated_conns:
            conn = updated_conns.pop()
            if conn in visited_keys:
                continue

            visited_keys.add(conn)
            dependent_conns: OrderedSet[ConnectionData] = OrderedSet()

            if conn.key in self.conns.input_keys:
                # New global input key
                dependent_conns = self._get_from_input_cache(conn)

                for dependent_conn in dependent_conns:
                    self._global_input_dependency_map[conn].add(dependent_conn)

            elif conn.key in self.conns.output_keys:
                # New global output key
                dependent_conns = self._get_from_output_cache(conn)

                for dependent_conn in dependent_conns:
                    self._global_output_dependency_map[conn].add(dependent_conn)

            else:
                # Key must be overriden, remove from dependecy map
                if conn in self._global_input_dependency_map:
                    dependent_conns = self._global_input_dependency_map.pop(
                        conn, OrderedSet()
                    )
                    for dependent_conn in dependent_conns:
                        self._global_output_dependency_map[dependent_conn].remove(conn)

                dependent_conns = OrderedSet()
            updated_conns |= OrderedSet(dependent_conns)

    # Retrieve dependent output connection keys given input key by traversing the graph.
    def get_input_key_dependency(self, key: str) -> OrderedSet[ConnectionData]:
        if (given_conn := self.conns.get_connection(key)) is None:
            raise KeyError("Given key does not belong to the Model!")
        # If there already exists any input keys, add them.
        specs = OrderedSet(
            [
                key
                for item in self.local_input_dependency_map[given_conn]
                for key in item[1]
                if key in self.conns.output_keys
            ]
        )

        # Make search from intermediate keys to the input keys.
        key_stack = OrderedSet(
            [
                spec
                for item in self.local_input_dependency_map[given_conn]
                for spec in item[1]
                if spec not in specs
            ]
        )
        while key_stack:
            conn_data = key_stack.pop()
            if conn_data.key in self.conns.output_keys:
                specs.add(conn_data)
            # key_stack.update(self.dependency_map.get(key.key, OrderedSet()))
            # TODO: add test checking the while
            key_stack |= (
                OrderedSet(
                    [
                        spec
                        for item in self.local_input_dependency_map[conn_data]
                        for spec in item[1]
                    ]
                )
                if conn_data in self.local_input_dependency_map
                else OrderedSet()
            )
        return specs

    # Retrieve dependent input connection keys given output key by traversing the graph.
    def get_output_key_dependency(self, key: str) -> OrderedSet[ConnectionData]:
        if (given_conn := self.conns.get_connection(key)) is None:
            raise KeyError("Given key does not belong to the Model!")

        # If there already exists any input keys, add them
        specs = OrderedSet(
            [
                key
                for key in self.local_output_dependency_map[given_conn][1]
                if key in self.conns.input_keys
            ]
        )
        # Make search from intermediate keys to the input keys.
        key_stack = OrderedSet(
            [
                spec
                for spec in self.local_output_dependency_map[given_conn][1]
                if spec not in specs
            ]
        )
        while key_stack:
            conn_data = key_stack.pop()
            if conn_data.key in self.conns.input_keys:
                specs.add(conn_data)
            # key_stack.update(self.dependency_map.get(key.key, OrderedSet()))
            # TODO: add test checking the while
            key_stack |= (
                self.local_output_dependency_map[conn_data][1]
                if conn_data in self.local_output_dependency_map
                else OrderedSet()
            )
        return specs

    # Check if cycle occured in graph
    def look_for_cyclic_connection(
        self, target_conn: ConnectionData, specs: OrderedSet[ConnectionData]
    ) -> bool:
        if target_conn in (conns := OrderedSet([conn for conn in specs])):
            return True
        else:
            for conn in conns:
                if conn in self.local_output_dependency_map:
                    return self.look_for_cyclic_connection(
                        target_conn, self.local_output_dependency_map[conn][1]
                    )
            return False

    def merge_global_connections(
        self, conn1: ConnectionData, conn2: ConnectionData
    ) -> None:
        conn1_global_out_dependency = self._global_output_dependency_map.get(conn1)
        conn2_global_out_dependency = self._global_output_dependency_map.pop(
            conn2, None
        )
        # If conn1 and conn2 cache exists, add all conn2 dependencies to conn1
        if (
            conn1_global_out_dependency is not None
            and conn2_global_out_dependency is not None
        ):
            for dependent_conn in conn2_global_out_dependency:
                conn1_global_out_dependency.add(dependent_conn)

                self._global_input_dependency_map[dependent_conn].remove(conn2)
                self._global_input_dependency_map[dependent_conn].add(conn1)
        # If conn1 is not, but conn2 cache exists, move all conn2 dependencies to conn1
        elif conn2_global_out_dependency is not None:
            self._global_output_dependency_map[conn1] = conn2_global_out_dependency

            for dependent_conn in conn2_global_out_dependency:
                self._global_input_dependency_map[dependent_conn].remove(conn2)
                self._global_input_dependency_map[dependent_conn].add(conn1)

        conn1_global_in_dependency = self._global_input_dependency_map.get(conn1)
        conn2_global_in_dependency = self._global_input_dependency_map.pop(conn2, None)
        # If conn1 and conn2 cache exists, add all conn2 dependencies to conn1
        if (
            conn1_global_in_dependency is not None
            and conn2_global_in_dependency is not None
        ):
            for dependent_conn in conn2_global_in_dependency:
                conn1_global_in_dependency.add(dependent_conn)

                self._global_output_dependency_map[dependent_conn].remove(conn2)
                self._global_output_dependency_map[dependent_conn].add(conn1)
        # If conn1 is not, but conn2 cache exists, move all conn2 dependencies to conn1
        elif conn2_global_in_dependency is not None:
            self._global_input_dependency_map[conn1] = conn2_global_in_dependency

            for dependent_conn in conn2_global_in_dependency:
                self._global_output_dependency_map[dependent_conn].remove(conn2)
                self._global_output_dependency_map[dependent_conn].add(conn1)

    def merge_global_caches(self, conn1: ConnectionData, conn2: ConnectionData) -> None:
        conn1_global_out_cache = self._global_output_dependency_map_cache.get(conn1)
        conn2_global_out_cache = self._global_output_dependency_map_cache.pop(
            conn2, None
        )

        # If conn1 and conn2 cache exists, add all conn2 dependencies to conn1
        if conn1_global_out_cache is not None and conn2_global_out_cache is not None:
            for dependent_conn in conn2_global_out_cache:
                conn1_global_out_cache.add(dependent_conn)

                self._global_input_dependency_map_cache[dependent_conn].remove(conn2)
                self._global_input_dependency_map_cache[dependent_conn].add(conn1)

        # If conn1 is not, but conn2 cache exists, move all conn2 dependencies to conn1
        elif conn2_global_out_cache is not None:
            self._global_output_dependency_map_cache[conn1] = conn2_global_out_cache

            for dependent_conn in conn2_global_out_cache:
                self._global_input_dependency_map_cache[dependent_conn].remove(conn2)
                self._global_input_dependency_map_cache[dependent_conn].add(conn1)

        conn1_global_in_cache = self._global_input_dependency_map_cache.get(conn1)
        conn2_global_in_cache = self._global_input_dependency_map_cache.pop(conn2, None)

        # If conn1 and conn2 cache exists, add all conn2 dependencies to conn1
        if conn1_global_in_cache is not None and conn2_global_in_cache is not None:
            for dependent_conn in conn2_global_in_cache:
                conn1_global_in_cache.add(dependent_conn)

                self._global_output_dependency_map_cache[dependent_conn].remove(conn2)
                self._global_output_dependency_map_cache[dependent_conn].add(conn1)

        # If conn1 is not, but conn2 cache exists, move all conn2 dependencies to conn1
        elif conn2_global_in_cache is not None:
            self._global_input_dependency_map_cache[conn1] = conn2_global_in_cache

            for dependent_conn in conn2_global_in_cache:
                self._global_output_dependency_map_cache[dependent_conn].remove(conn2)
                self._global_output_dependency_map_cache[dependent_conn].add(conn1)
