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
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from types import UnionType
from typing import Any

from ...utils.utils import OrderedSet
from ..common import (
    NOT_AVAILABLE,
    NOT_GIVEN,
    TBD,
    Connect,
    Connection,
    ConnectionData,
    Connections,
    ConnectionType,
    Constraint,
    ConstraintFunctionType,
    ConstraintSolver,
    ExtendTemplate,
    IOHyperEdge,
    IOKey,
    MainValueInstance,
    MainValueType,
    MyTensor,
    NestedListType,
    NotAvailable,
    ShapeNode,
    ShapesType,
    ShapeTemplateType,
    ShapeType,
    TensorValueType,
    ToBeDetermined,
    UniadicRecord,
    Updates,
    UpdateType,
    Variadic,
    _get_shapes,
    _ShapesType,
    create_shape_repr,
    values_equal,
)
from ..constraints import post_process_map, type_constraints

__all__ = ["BaseModel", "ExtendInfo"]


@dataclass
class ExtendInfo:
    _model: BaseModel
    _connections: dict[str, ConnectionType]

    def __post_init__(self):
        external_keys = set(self._model.external_keys)
        if self._model.canonical_input is not NOT_AVAILABLE:
            external_keys.add(self._model.canonical_input.key)
        if self._model.canonical_output is not NOT_AVAILABLE:
            external_keys.add(self._model.canonical_output.key)

        for key in self._connections:
            if key not in external_keys:
                raise KeyError(f"Key '{key}' is not a valid key for the model!")


class BaseModel(abc.ABC):
    # Disposable models only used once for entire training session.
    # This attribute is only use for manual backends' code generation.

    # TODO: This can be checked from backend's gradient function dict???
    disposable: bool = False
    # TODO: factory_args should be instance variable not class!
    factory_args: dict[str, Any] = {}

    def __call__(self, **kwargs: ConnectionType) -> ExtendInfo:
        for key, val in self.factory_inputs.items():
            if val is not TBD:
                if key not in kwargs or (con := kwargs[key]) is NOT_GIVEN:
                    kwargs[key] = val  # type: ignore
                    continue
                match con:
                    case Connection():
                        kwargs[key] = Connect(con, key=IOKey(value=val, expose=False))
                        # TODO: Maybe we could check con's value if matches with val
                    case item if isinstance(item, MainValueInstance) and con != val:
                        raise ValueError(
                            f"Given value {con} for local key: '{key}' "
                            f"has already being set to {val}!"
                        )
                    case str():
                        kwargs[key] = IOKey(con, value=val, expose=False)
                    case IOKey():
                        if con._value is not TBD and not values_equal(con._value, val):
                            raise ValueError(
                                f"Given IOKey for local key: '{key}' is not valid!"
                            )
                        else:
                            kwargs[key] = IOKey(
                                name=con._name,
                                value=val,
                                shape=con._shape,
                                type=con._type,
                                expose=con._expose,
                            )
                    case Connect():
                        if (io_key := con.key) is not None:
                            if io_key._value is not TBD and not values_equal(
                                io_key._value, val
                            ):
                                raise ValueError(
                                    "Given IOKey in Connect for "
                                    f"local key: '{key}' is not valid!"
                                )
                            else:
                                io_key._value = val
                        else:
                            io_key = IOKey(value=val, expose=False)
                            kwargs[key] = Connect(*con.connections, key=io_key)
                    case ExtendTemplate():
                        raise ValueError(
                            "Multi-write detected for a valued "
                            f"local key: '{key}' is not valid!"
                        )
        return ExtendInfo(self, kwargs)

    def __init__(self, name: str | None = None, enforce_jit: bool = True) -> None:
        self.parent: BaseModel | None = (
            None  # TODO: maybe set it only to PrimitiveModel / Model.
        )
        self.factory_inputs: dict[
            str, TensorValueType | MainValueType | ToBeDetermined
        ] = {}
        self.assigned_shapes: list[ShapesType] = []
        self.assigned_constraints: list[dict[str, str | list[str]]] = []
        self.conns = Connections()
        self.frozen_attributes: list[str] = []
        self.dependency_map = DependencyMap(self.conns)
        self._canonical_input: ConnectionData | NotAvailable = NOT_AVAILABLE
        self._canonical_output: ConnectionData | NotAvailable = NOT_AVAILABLE
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
    def shapes(self) -> _ShapesType:
        return self.get_shapes()

    @property
    def external_keys(self):
        return self.conns.io_keys

    @property
    def _input_keys(self):
        return self.conns.input_keys

    @property
    def _all_keys(self):
        return self.conns.all.keys()

    @property
    def output_keys(self):
        output_keys = list(self.conns.output_keys)
        if (
            self.canonical_output is not NOT_AVAILABLE
            and self.canonical_output.key not in output_keys
        ):
            output_keys.append("#canonical_output")
        return output_keys

    def check_extendability(self):
        # Check possible errors before the extension.
        if self.parent is not None:
            raise AttributeError("Submodel of a model could not be extended!")

    def _get_outermost_parent(self):
        model = self
        while model.parent is not None:
            model = model.parent
        return model

    def _generate_keys(
        self,
        symbolic: bool = True,
        include_internals: bool = True,
        include_outputs: bool = False,
    ) -> dict[str, str]:
        return {}

    def __setattr__(self, name: str, value: Any):
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
            shape_node = self.conns.get_shape_node(_inner_key)
            assert shape_node is not None
            updates |= shape_node.merge(node)

        if trace:
            self.assigned_shapes.append(assigned_shapes)

        model.constraint_solver(updates)

    def _set_value(self, key: ConnectionData, value: MainValueType | str) -> Updates:
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
        # Data is scalar, set the value directly.
        return key.metadata.set_value(value)  # type: ignore

    def set_shapes(
        self, config: ShapesType | None = None, **kwargs: ShapeTemplateType
    ) -> None:
        if config is None:
            config = {}
        self._set_shapes(config, trace=True, updates=None, **kwargs)

    def set_values(
        self,
        config: Mapping[str | Connection, MyTensor | MainValueType | str]
        | Mapping[Connection, MyTensor | MainValueType | str]
        | Mapping[str, MyTensor | MainValueType | str]
        | None = None,
        **kwargs: MyTensor | MainValueType | str,
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
        # TODO: Currently Setting values in fozen models are prevented only for Tensors.
        # Scalar and Tensors should not be operated differently. This should be fixed.
        for key in chain(config, kwargs):
            metadata = self.conns.extract_metadata(key)
            if metadata.edge_type is MyTensor and model.is_frozen:
                conn_data = model.conns.get_con_by_metadata(metadata)
                assert conn_data is not None
                raise ValueError(
                    f"Model is frozen, can not set the key: {conn_data.key}!"
                )

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
        config: Mapping[str | Connection, type | UnionType | NestedListType]
        | Mapping[Connection, type | UnionType | NestedListType]
        | Mapping[str, type | UnionType | NestedListType]
        | None = None,
        **kwargs: type | UnionType | NestedListType,
    ):
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

        # Get the outermost parent as all the updates will happen here.
        model = self._get_outermost_parent()
        updates = Updates()
        for key, key_type in chain(config.items(), kwargs.items()):
            metadata = self.conns.extract_metadata(key)
            updates |= metadata.set_type(key_type)  # type: ignore
        # Run the constraints for updating affected connections.
        model.constraint_solver(updates)

    def get_shapes(
        self,
        uni_keys: dict[UniadicRecord, str] | None = None,
        var_keys: dict[Variadic, str] | None = None,
        symbolic: bool = True,
        verbose: bool = False,
    ) -> _ShapesType:
        return _get_shapes(
            data_dict={key: value.metadata for key, value in self.conns.all.items()},
            uniadic_keys=uni_keys,
            varadic_keys=var_keys,
            symbolic=symbolic,
            verbose=verbose,
            key_mappings=self._generate_keys(include_outputs=True),
        )

    def _set_constraint(
        self,
        fn: ConstraintFunctionType,
        keys: list[str],
        post_processes: set[ConstraintFunctionType] | None = None,
        type: UpdateType | None = None,
    ):
        all_conns = self.conns.all
        hyper_edges = [all_conns[key].metadata for key in keys]
        if type is None:
            # TODO: separate type_constraints and shape constraints into two files under
            # constraints folder. Then, check if fn is not in any of those types set
            # _type to None. If _type and type are both None or one is UpdateType.SHAPE
            # while other one is UpdateType.Type, raise Exception!
            type = UpdateType.TYPE if fn in type_constraints else UpdateType.SHAPE
        constr = Constraint(fn=fn, type=type)
        self.constraint_solver.constraint_map[constr] = hyper_edges
        for hyper_edge in hyper_edges:
            hyper_edge.add_constraint(constr)

        # Get union of all given and default post processes for the given
        # constraint and update post_processes field.
        if post_processes is None:
            post_processes = set()
        all_post_processes = post_processes | post_process_map.get(fn, set())
        for post_fn in all_post_processes:
            constr.add_post_process(post_fn)

        # _, updates = constr([hyper_edge.data for hyper_edge in hyper_edges])
        _, updates = constr(hyper_edges)
        self.constraint_solver(updates)

    def set_constraint(
        self,
        fn: ConstraintFunctionType,
        keys: list[str],
        post_processes: set[ConstraintFunctionType] | None = None,
        type: UpdateType = UpdateType.SHAPE,
    ) -> None:
        self.assigned_constraints.append({"fn": fn.__name__, "keys": keys})
        self._set_constraint(fn, keys, post_processes, type=type)

    @property
    def canonical_input(self) -> Connection | NotAvailable:
        if isinstance(self._canonical_input, ConnectionData):
            return self._canonical_input.conn
        else:
            return NOT_AVAILABLE

    @property
    def canonical_output(self) -> Connection | NotAvailable:
        if isinstance(self._canonical_output, NotAvailable):
            return self._canonical_output
        else:
            return self._canonical_output.conn

    def set_canonical_input(self, given_conn: str | Connection):
        if isinstance(given_conn, str):
            conn = self.conns.all.get(given_conn)
            if conn is None:
                raise ValueError("Provided 'key' is not belong to the model!")
        else:
            conn = given_conn.data

        conn = self.conns.get_con_by_metadata(conn.metadata)

        if conn not in self.dependency_map._local_input_dependency_map:
            raise ValueError(
                "To set a connection as canonical input, connection must be an "
                "input connection!"
            )

        self._canonical_input = conn

    def set_canonical_output(self, given_conn: str | Connection):
        if isinstance(given_conn, str):
            conn = self.conns.all.get(given_conn)
            if conn is None:
                raise ValueError("Provided 'key' is not belong to the model!")
        else:
            conn = given_conn.data

        conn = self.conns.get_con_by_metadata(conn.metadata)

        if conn not in self.dependency_map._local_output_dependency_map:
            raise ValueError(
                "To set a connection as canonical output, connection must be an "
                "output connection!"
            )

        self._canonical_output = conn

    def _match_hyper_edges(self, left: IOHyperEdge, right: IOHyperEdge) -> Updates:
        # if type(left.data) is not type(right.data):
        if (left.edge_type is MyTensor) ^ (right.edge_type is MyTensor):
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
        updates = left.match(right)  # type: ignore
        return updates

    def get_models_in_topological_order(self):
        dependency_map = self.dependency_map._local_output_dependency_map
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
    ):
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
        self._local_input_dependency_map: dict[
            ConnectionData, list[tuple[BaseModel, OrderedSet[ConnectionData]]]
        ] = {}
        # Stores releation between local output keys to dependent local
        # input connections
        self._local_output_dependency_map: dict[
            ConnectionData, tuple[BaseModel, OrderedSet[ConnectionData]]
        ] = {}

    # Add new model to dependency map, model_dag is created in extend
    def add_model_dag(self, model: BaseModel, model_dag: dict[str, ConnectionData]):
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
                self._local_input_dependency_map.setdefault(conn, []).append(
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
                self._local_output_dependency_map[conn] = (model, specs)

                updated_conns.add(conn)
                self._cache_internal_references(conn, specs)

            if self.look_for_cyclic_connection(conn, specs):
                raise Exception(
                    f"There exists a cyclic subgraph between {conn.key} key and "
                    f"{[spec.key for spec in specs]} key(s)!"
                )

        self._update_globals(updated_conns)

    # Caches extended connections to avoid traverse
    def _cache_internal_references(
        self, output_conn: ConnectionData, dependent_conns: OrderedSet[ConnectionData]
    ):
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
    def cache_conn_input_dependency(self, conn: ConnectionData):
        if conn not in self._global_input_dependency_map_cache:
            dependents = self.get_output_key_dependency(conn.key)
            self._global_input_dependency_map_cache[conn] = dependents

    # Caches given output connection for later usage
    def cache_conn_output_dependency(self, conn: ConnectionData):
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
    def update_all_keys(self):
        # This method is used in freeze, because in freeze dependencies changed
        # without updating dependency map.
        self._update_globals(
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
    def _get_from_output_cache(self, conn: ConnectionData):
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
    def _update_globals(self, updated_conns: OrderedSet[ConnectionData]):
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
    def get_input_key_dependency(self, key: str):
        if (given_conn := self.conns.get_connection(key)) is None:
            raise KeyError("Given key does not belong to the Model!")
        # If there already exists any input keys, add them.
        specs = OrderedSet(
            [
                key
                for item in self._local_input_dependency_map[given_conn]
                for key in item[1]
                if key in self.conns.output_keys
            ]
        )

        # Make search from intermediate keys to the input keys.
        key_stack = OrderedSet(
            [
                spec
                for item in self._local_input_dependency_map[given_conn]
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
                        for item in self._local_input_dependency_map[conn_data]
                        for spec in item[1]
                    ]
                )
                if conn_data in self._local_input_dependency_map
                else OrderedSet()
            )
        return specs

    # Retrieve dependent input connection keys given output key by traversing the graph.
    def get_output_key_dependency(self, key: str):
        if (given_conn := self.conns.get_connection(key)) is None:
            raise KeyError("Given key does not belong to the Model!")

        # If there already exists any input keys, add them
        specs = OrderedSet(
            [
                key
                for key in self._local_output_dependency_map[given_conn][1]
                if key in self.conns.input_keys
            ]
        )
        # Make search from intermediate keys to the input keys.
        key_stack = OrderedSet(
            [
                spec
                for spec in self._local_output_dependency_map[given_conn][1]
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
                self._local_output_dependency_map[conn_data][1]
                if conn_data in self._local_output_dependency_map
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
                if conn in self._local_output_dependency_map:
                    return self.look_for_cyclic_connection(
                        target_conn, self._local_output_dependency_map[conn][1]
                    )
            return False

    def merge_global_connections(self, conn1: ConnectionData, conn2: ConnectionData):
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

    def merge_global_caches(self, conn1: ConnectionData, conn2: ConnectionData):
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
