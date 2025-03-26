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

from collections.abc import KeysView, Mapping, ValuesView
from itertools import chain
from types import UnionType
from typing import Any, get_origin

from ...utils.utils import OrderedSet
from ..common import (
    NOT_GIVEN,
    TBD,
    AssignedConstraintType,
    Constraint,
    ConstraintFunctionType,
    ConstraintSolver,
    IOHyperEdge,
    KeyType,
    MainValueInstance,
    MainValueType,
    NullConnection,
    ScalarType,
    ScalarValueType,
    ShapeNode,
    ShapeTemplateType,
    ShapeType,
    StateValue,
    Tensor,
    ToBeDetermined,
    UniadicRecord,
    Updates,
    UpdateType,
    Variadic,
    create_shape_repr,
    find_intersection_type,
    find_type,
    get_shapes,
)
from ..constraints import constraint_type_map

__all__ = ["BaseModel", "BaseKey", "ConnectionData", "ConnectionDataType"]


StateValueType = StateValue | MainValueInstance | NullConnection


class ConnectionData:
    # TODO: This class updated as mutable. Update docstrings accordingly!
    """Immutable dataclass object which holds model instance, key
    and I/O info. It is immutable because once a model's input-output
    names are defined, changing them is not allowed for proper DAG
    connections.
    """

    def __init__(
        self,
        name: str | None = None,
        value: Tensor[int | float | bool]
        | ScalarValueType
        | ToBeDetermined
        | str = TBD,
        shape: ShapeTemplateType | None = None,
        type: UnionType
        | type
        | type[Tensor[int | float | bool]]
        | ScalarType
        | None = None,
        expose: bool | None = None,
        differentiable: bool | None = None,
        interval: list[float | int] | None = None,
    ) -> None:
        # If shape is provided, type should be Tensor.
        if shape is not None:
            if type in (Tensor, None):
                type = Tensor[int | float | bool]
            elif get_origin(type) is not Tensor:
                raise TypeError("Shape can not be provided for a non-tensor type!")
        elif type is Tensor:
            # Convert to generic Tensor type if Tensor type is provided.
            type = Tensor[int | float | bool]

        if differentiable:
            if type is not None:
                if (type := find_intersection_type(type, Tensor[float])) is None:
                    raise TypeError(
                        "Differentiable connection should be Tensor[float] type!"
                    )
            else:
                type = Tensor[float]
            if value is TBD:
                value = Tensor(differentiable=True)
            elif isinstance(value, Tensor):
                value.differentiable = True
            else:
                raise ValueError(
                    "Differentiable connection value should be Tensor type!"
                )

        self._name = name
        self._expose = expose
        # TODO: Shape should not be [] also!
        if (
            value is not TBD
            and not isinstance(value, Tensor)
            and shape is not None
            and shape != []
        ):
            raise ValueError(
                f"Scalar values are shapeless, shape should be None or []. "
                f"Got {shape}."
            )

        if value is not TBD and type is not None:
            value_type = find_type(value)
            if find_intersection_type(value_type, type) is None:
                raise TypeError(
                    "type of the given value and given type does not match. Given "
                    f"type is {type} while type of value is {value_type}"
                )
        self.value_shape = shape
        if type is None:
            type = ToBeDetermined
        self.metadata = IOHyperEdge(
            key_origin=name,
            type=type,
            value=value,
            interval=interval,
        )
        self.model: BaseModel | None = None

    @property
    def is_exposed(self) -> bool | None:
        # TODO: get this from self.model model is set.
        if self.model is not None:
            m = self.model._get_outermost_parent()
            con = m.conns.get_extracted_connection(self)
            self._expose = None
            return m.conns.get_type(con) in (KeyType.INPUT, KeyType.OUTPUT)
        return self._expose

    @property
    def key(self) -> str:
        assert self._name is not None, "Connection name should be provided!"
        return self._name

    def set_key(self, key: str | None) -> None:
        self._name = key

    def get_key(self) -> str | None:
        return self._name

    @property
    def is_autogenerated(self) -> bool:
        return self.key.startswith("$")

    def __hash__(self) -> int:
        return hash(id(self))

    def set_differentiability(self, differentiable: bool = True) -> Updates:
        return self.metadata.set_differentiability(differentiable)


BaseKey = ConnectionData

ConnectionDataType = (
    str
    | MainValueType
    | NullConnection
    | BaseKey
    | ConnectionData
    | Tensor[int | float | bool]
)

ConnectionDataInstanceType = (
    str | MainValueInstance | NullConnection | BaseKey | ConnectionData | Tensor  # type: ignore
)


class Connections:
    """This class maintains all the connections and their operations in model.
    _input_dict, _output_dict and _internal_dict stores added / updated connections
    and metadata_dict contains all metadata of the connections in _input_dict,
    _output_dict and _internal_dict. Metadata dict is updated in case of metadata
    merger.
    """

    def __init__(self) -> None:
        self._connection_dict: dict[KeyType, dict[str, ConnectionData]] = {
            key_type: {} for key_type in KeyType
        }

        self.metadata_dict: dict[IOHyperEdge, set[ConnectionData]] = {}
        self.connections_dict: dict[IOHyperEdge, set[Connections]] = {}
        self.cins: set[ConnectionData] = set()
        self.couts: set[ConnectionData] = set()

    @property
    def input_keys(self) -> KeysView[str]:
        return self._connection_dict[KeyType.INPUT].keys()

    @property
    def input_connections(self) -> ValuesView[ConnectionData]:
        return self._connection_dict[KeyType.INPUT].values()

    @property
    def output_keys(self) -> KeysView[str]:
        return self._connection_dict[KeyType.OUTPUT].keys()

    @property
    def output_connections(self) -> ValuesView[ConnectionData]:
        return self._connection_dict[KeyType.OUTPUT].values()

    @property
    def internal_keys(self) -> KeysView[str]:
        return self._connection_dict[KeyType.INTERNAL].keys()

    @property
    def internal_connections(self) -> ValuesView[ConnectionData]:
        return self._connection_dict[KeyType.INTERNAL].values()

    @property
    def latent_output_keys(self) -> KeysView[str]:
        return self._connection_dict[KeyType.LATENT_OUTPUT].keys()

    @property
    def latent_output_connections(self) -> ValuesView[ConnectionData]:
        return self._connection_dict[KeyType.LATENT_OUTPUT].values()

    @property
    def latent_input_keys(self) -> KeysView[str]:
        return self._connection_dict[KeyType.LATENT_INPUT].keys()

    @property
    def latent_input_connections(self) -> ValuesView[ConnectionData]:
        return self._connection_dict[KeyType.LATENT_INPUT].values()

    @property
    def all(self) -> dict[str, ConnectionData]:
        return (
            self._connection_dict[KeyType.INTERNAL]
            | self._connection_dict[KeyType.INPUT]
            | self._connection_dict[KeyType.OUTPUT]
            | self._connection_dict[KeyType.LATENT_INPUT]
            | self._connection_dict[KeyType.LATENT_OUTPUT]
        )

    @property
    def io_keys(self) -> KeysView[str]:
        return (
            self._connection_dict[KeyType.INPUT] | self._connection_dict[KeyType.OUTPUT]
        ).keys()

    def add(
        self,
        connection: ConnectionData,
    ) -> None:
        metadata = connection.metadata
        self.metadata_dict.setdefault(metadata, set()).add(connection)
        self.connections_dict.setdefault(metadata, set()).add(self)

    def set_connection_type(
        self,
        connection: ConnectionData,
        con_type: KeyType,
        safe: bool = True,
    ) -> None:
        if safe and con_type == KeyType.OUTPUT and connection.is_autogenerated:
            raise KeyError("Connection without a name cannot be set as output")
        key = connection.key
        if con_type == KeyType.INTERNAL:
            self.couts.discard(connection)
        if con_type != KeyType.INPUT:
            self.cins.discard(connection)
        for _type in KeyType:
            if _type == con_type:
                self._connection_dict[_type][key] = connection
            else:
                self._connection_dict[_type].pop(key, None)

    def remove_connection(self, connection: ConnectionData) -> None:
        for _type in KeyType:
            self._connection_dict[_type].pop(connection.key, None)
            self.cins.discard(connection)
            self.couts.discard(connection)

    def get_data(self, key: str) -> IOHyperEdge:
        return self.get_metadata(key)

    def get_type(self, key: ConnectionData) -> KeyType:
        con_data = self.get_extracted_connection(key)
        for _key_type in KeyType:
            key_dict = self._connection_dict[_key_type]
            if key_dict.get(con_data.key) is not None:
                return _key_type
        raise ValueError("No matching key type found!")

    def get_non_diff_keys(self) -> set[str]:
        return {
            key for key, conn in self.all.items() if not conn.metadata.differentiable
        }

    def get_connection(self, key: str) -> ConnectionData | None:
        for value in self._connection_dict.values():
            if (connection := value.get(key)) is not None:
                return connection
        return None

    def get_con_by_metadata(self, key: IOHyperEdge) -> ConnectionData | None:
        conns = self.metadata_dict.get(key)
        if conns is not None:
            return next(iter(conns))
        return conns

    def get_cons_by_metadata(self, key: IOHyperEdge) -> set[ConnectionData] | None:
        return self.metadata_dict.get(key)

    def get_metadata(self, key: str) -> IOHyperEdge:
        if (con := self.get_connection(key)) is not None:
            return con.metadata
        raise KeyError(f"Key '{key}' is not found in connections.")

    def get_key_origin(self, key: str) -> str | None:
        return self.get_metadata(key).key_origin

    def get_shape_node(self, key: str) -> ShapeNode:
        edge = self.get_metadata(key)
        if not edge.is_tensor:
            raise ValueError("'Only Tensor type connections has shape!'")
        assert edge.shape is not None
        return edge.shape

    def set_value(self, con: ConnectionData, value: MainValueType) -> None:
        self.get_data(con.key).set_value(value)

    def extract_metadata(self, key: str | ConnectionData) -> IOHyperEdge:
        if isinstance(key, ConnectionData):
            # Extract the key from the Connection object.
            metadata = key.metadata
        else:
            metadata = self.get_metadata(key)
        return metadata

    def get_extracted_connection(self, key: str | ConnectionData) -> ConnectionData:
        if (result := self.get_con_by_metadata(self.extract_metadata(key))) is None:
            raise KeyError("Connection is not found!")
        return result


class BaseModel:
    # Disposable models only used once for entire training session.
    # This attribute is only use for manual backends' code generation.

    # TODO: This can be checked from backend's gradient function dict???
    disposable: bool = False
    # TODO: factory_args should be instance variable not class!
    factory_args: dict[str, Any] = {}

    def __init__(
        self,
        name: str | None = None,
        formula_key: str | None = None,
        enforce_jit: bool = True,
    ) -> None:
        self.dag: dict[BaseModel, dict[str, ConnectionData]] = {}
        self._formula_key: str | None = formula_key
        # TODO: maybe set it only to Operator / Model.
        self.parent: BaseModel | None = None
        self.assigned_shapes: list[dict[ConnectionData, ShapeTemplateType]] = []
        self.assigned_types: dict[
            ConnectionData,
            type | UnionType | ScalarType | type[Tensor[int | float | bool]],
        ] = {}
        self.assigned_differentiabilities: dict[ConnectionData, bool] = {}
        self.assigned_constraints: list[AssignedConstraintType] = []
        self.assigned_cins: set[ConnectionData] = set()
        self.assigned_couts: set[ConnectionData] = set()
        self.conns = Connections()
        self.frozen_attributes: list[str] = []
        self.dependency_map = DependencyMap(self.conns)
        self.name = name
        self._enforce_jit = enforce_jit
        self._jittable = True
        self.constraint_solver: ConstraintSolver = ConstraintSolver()
        self.safe_shapes: dict[str, ShapeTemplateType] = {}
        self.is_frozen = False
        self.inter_key_count = 0
        self.extract = False
        # Temporarily created provisional model when needed.
        self.provisional_model: None | BaseModel = None
        self.provisional_source: bool | BaseModel = False
        # NOTE: If model is provisional and it does have a source model
        # provisional_source keeps the source model.
        # If model is provisional and it does not have a source model,
        # provisional_source is True without model information.
        self.state_connections: dict[
            ConnectionData, tuple[ConnectionData, StateValueType]
        ] = {}

    @property
    def formula_key(self) -> str | None:
        return self._formula_key

    def create_key_name(self) -> str:
        self.inter_key_count += 1
        return "$" + str(self.inter_key_count)

    def expose_keys(
        self, *args: str | ConnectionData, **kwargs: str | ConnectionData
    ) -> None:
        if self.parent is not None:
            raise Exception("Child model's outputs cannot be set.")
        # Convert all args and kwargs to tuple.
        # Convert all args and kwargs to tuple.
        pairs = tuple([(None, arg) for arg in args]) + tuple(kwargs.items())
        connections = []
        for pair in pairs:
            new_name, name = pair
            metadata = self.conns.extract_metadata(name)

            # Check the connection is valid.
            if (conn_data := self.conns.get_con_by_metadata(metadata)) is None:
                raise KeyError("Requires valid key or Connection to set output!")

            key_type = self.conns.get_type(conn_data)
            is_input = key_type in {KeyType.INPUT, KeyType.LATENT_INPUT}

            # Autogenerated keys can not be set directly as output without a name.
            if new_name is None and conn_data.is_autogenerated:
                raise KeyError(
                    "Autogenerated keys can only be exposed if"
                    " a name is provided for the connection as keyworded argument."
                )

            if new_name is not None:  # Non-named connections.
                if new_name in self.conns.all:
                    raise KeyError(f"Key '{new_name}' is already used!")
                self.rename_key(conn_data, new_name)
            new_type = (KeyType.OUTPUT, KeyType.INPUT)[is_input]
            if new_type != key_type:
                self.conns.set_connection_type(conn_data, new_type)
                connections.append(conn_data)
        self.dependency_map.update_globals(OrderedSet(connections))

    def bind_state_keys(
        self,
        input: ConnectionData | str,
        output: ConnectionData | str,
        initial_value: StateValueType = NOT_GIVEN,
    ) -> None:
        if self.is_frozen:
            raise AttributeError("Frozen model's bind_state_keys is not allowed!")
        # Get connections.
        in_con = self.conns.get_extracted_connection(input)
        out_con = self.conns.get_extracted_connection(output)
        if self.conns.get_type(in_con) not in {KeyType.INPUT, KeyType.LATENT_INPUT}:
            raise KeyError("Input connection should be an input key!")
        if self.conns.get_type(out_con) in {KeyType.INPUT, KeyType.LATENT_INPUT}:
            raise KeyError("Output connection should be an output key!")
        for _out, (_in, _) in self.state_connections.items():
            if _in.metadata is in_con.metadata or _out.metadata is out_con.metadata:
                raise KeyError("Binded connections could not be binded again!")

        # Set connection types to latent.
        self.conns.set_connection_type(in_con, KeyType.LATENT_INPUT)
        self.conns.couts.discard(out_con)
        if self.conns.get_type(out_con) == KeyType.OUTPUT:
            self.conns.set_connection_type(out_con, KeyType.LATENT_OUTPUT)

        updates = Updates()
        # Set differentiability of input connection to False.
        updates |= in_con.set_differentiability(False)
        # Merge types.
        updates |= in_con.metadata.set_type(out_con.metadata._type)
        updates |= out_con.metadata.set_type(in_con.metadata._type)
        if in_con.metadata.is_tensor:
            # Merge shapes if connections are Tensors.
            assert isinstance(in_con.metadata._value, Tensor)
            assert isinstance(out_con.metadata._value, Tensor)
            updates |= in_con.metadata._value.match_shapes(
                out_con.metadata._value.shape
            )
        self.constraint_solver(updates)

        # Save state connections.
        self.state_connections[out_con] = (in_con, initial_value)

    def _check_multi_write(
        self,
        local_input: bool,
        local_connection: ConnectionData,
        connection: ConnectionData,
    ) -> None:
        conn_is_output = (
            self.dependency_map.local_output_dependency_map.get(connection, None)
            is not None
        )
        if local_connection.key in self.conns.all and connection.key in self.conns.all:
            local_conn_is_output = (
                self.dependency_map.local_output_dependency_map.get(
                    local_connection, None
                )
                is not None
            )
            if (
                conn_is_output
                and local_conn_is_output
                and local_connection.key != connection.key
            ):
                # Check if 2 connections are part of main model. If it is the case,
                # We expect at least one of them is not an input of the main model,
                # otherwise condition is Multi-write error
                raise Exception(
                    "Given connections are both output connections. Multi-write error!"
                )

        local_val = local_connection.metadata.value
        global_val = connection.metadata.value

        if conn_is_output and not local_input:
            # Check if 2 connections are both output of any models.
            raise Exception(
                "Given connections are both output connections. Multi-write error!"
            )
        elif (
            local_input
            and local_connection.metadata.is_valued
            and conn_is_output
            and global_val != local_val
        ):
            raise ValueError(
                "An input of the extending model tries to write "
                "to an output connection in the extended model. "
                "Multi-write error!"
            )
        elif (
            not local_input
            and connection.metadata.is_valued
            and local_val != global_val
        ):
            raise ValueError(
                "A valued connection of the extended model tries to write "
                "to an output connection of the extending model. "
                "Multi-write error!"
            )

    def _add_connection(
        self,
        model: BaseModel,
        local_key: str,
        given_connection: ConnectionDataType,
        updates: Updates,
        trace: bool,
    ) -> tuple[ConnectionData, Updates]:
        is_input = local_key in model.input_keys
        local_connection = model.conns.get_connection(local_key)
        assert local_connection is not None, "Connection is not found!"
        edge = local_connection.metadata
        is_not_valued = not edge.is_valued
        set_diff = None

        d_map = self.dependency_map.local_output_dependency_map

        con_obj = None
        set_value: (
            ToBeDetermined
            | str
            | ScalarValueType
            | Tensor[int | float | bool]
            | NullConnection
        ) = NOT_GIVEN
        set_type: type[Tensor[int | float | bool]] | ScalarType = ToBeDetermined
        is_new_connection = False

        match given_connection:
            case NullConnection():
                given_connection = self._create_connection(edge)
            case str():
                if (_conn := self.conns.get_connection(given_connection)) is None:
                    _conn = self._create_connection(edge, given_connection)
                given_connection = _conn
            case _ if isinstance(given_connection, MainValueInstance | Tensor):
                if local_connection in model.dependency_map.local_output_dependency_map:
                    raise KeyError(
                        f"{local_key} key is an output of the model, "
                        "output values could not be set in extend."
                    )
                set_value = given_connection
                given_connection = self._create_connection(edge, None)
        assert isinstance(given_connection, ConnectionData)

        if (
            given_connection.metadata.differentiable is not None
            and given_connection.metadata.differentiable != edge.differentiable
        ):
            set_diff = given_connection.metadata.differentiable

        # Connection is given as a Connection object.
        if (
            con_obj := self.conns.get_con_by_metadata(given_connection.metadata)
        ) is None:
            if given_connection.model is not None:
                raise KeyError("Requires accessible connection to be processed!")
            is_new_connection = True
            expose = given_connection.is_exposed
            outer_key = given_connection.get_key()
            if set_value is NOT_GIVEN:
                set_type = given_connection.metadata.edge_type
            if set_value is NOT_GIVEN and given_connection.metadata.value is not TBD:
                set_value = given_connection.metadata._value
            if outer_key is not None:
                con_obj = self.conns.get_connection(outer_key)
            if outer_key is None or con_obj is None:
                if expose is None and is_input and is_not_valued:
                    expose = True
                # NOTE: This is the previous version that creates new connection
                # even if the connection is already created and provided.
                # con_obj = self._create_connection(edge, outer_key)
                # self.attach_connection(con_obj)
                given_connection.model = self
                self.attach_connection(given_connection)
                con_obj = given_connection
            if (
                expose is False
                and is_input
                and set_value is NOT_GIVEN
                and edge.value is TBD
                and (con_obj is None or con_obj not in d_map)
            ):
                raise ValueError(
                    "Expose flag cannot be false when "
                    "no value is provided for input keys!"
                )

            # Set value or type if given.
            if not isinstance(set_value, NullConnection):
                updates |= con_obj.metadata.set_value(set_value)
            elif set_type is not ToBeDetermined:
                # Skip tracing if the local connection's type is already
                # set to the given type.
                trace &= set_type != local_connection.metadata.edge_type
                model._set_types({local_connection: set_type}, trace=trace)

            # Set differentiability if given.
            if set_diff is not None:
                # No need to trace differentiability for valued and
                # existing connections.
                trace &= is_new_connection and not given_connection.metadata.is_valued
                model._set_differentiability({local_connection: set_diff}, trace)

        else:
            if given_connection in model.conns.all.values():
                raise ValueError(
                    f"Given connection '{given_connection.key}' should not belong "
                    "to the extending model!"
                )

            outer_key = con_obj.key
            expose = outer_key in self.conns.output_keys and not is_input

        # Name "input" can only be used for input connections.
        is_key_name_input = con_obj is not None and (con_key := con_obj.key) == "input"
        if not is_input and (outer_key == "input" or is_key_name_input):
            raise KeyError(
                "The key 'input' is a reserved key which could not be used for "
                "internal keys."
            )

        # Inherit submodel connections dict
        self.conns.connections_dict.setdefault(edge, set())
        self.conns.connections_dict[edge] |= model.conns.connections_dict.pop(
            edge, set()
        )

        # Check multi-write error for con_obj.
        self._check_multi_write(is_input, local_connection, con_obj)
        # If match required, perform.
        if con_obj.metadata != edge:
            local_key_origin = edge.key_origin
            updates |= self._match_hyper_edges(con_obj.metadata, edge)
            # If local_connection is an output of the model,
            # update con_obj "key_origin" with local_connection's key_origin.
            if (
                not is_input
                and not is_new_connection
                and outer_key not in self.conns.output_keys
                or con_obj.metadata.key_origin is None
            ):
                con_obj.metadata.key_origin = local_key_origin

        unexposed = not (expose or (is_input and con_key in self.conns.io_keys))
        if unexposed:
            if is_input:
                key_type = (KeyType.LATENT_INPUT, KeyType.INTERNAL)[con_obj in d_map]
            else:
                key_type = (KeyType.LATENT_OUTPUT, KeyType.INTERNAL)[con_obj in d_map]
        else:
            key_type = (KeyType.OUTPUT, KeyType.INPUT)[
                is_input and con_obj not in d_map
            ]
        if con_obj in d_map:
            self.conns.couts.discard(con_obj)
        self.conns.set_connection_type(con_obj, key_type)
        # Update Canonicals
        if (
            local_connection in model.conns.cins
            and con_obj in self.conns.input_connections
            and not con_obj.metadata.is_valued
        ):
            self.conns.cins.add(con_obj)

        if local_connection in model.conns.couts and (
            con_obj not in self.dependency_map.local_input_dependency_map
            or con_obj in self.conns.output_connections
        ):
            self.conns.couts.add(con_obj)

        return con_obj, updates

    def rename_key(self, connection: ConnectionData, key: str) -> None:
        # TODO: raise if frozen model's rename_key is called?
        con_data = self.conns.get_extracted_connection(connection)
        key_type = self.conns.get_type(con_data)
        key_dict = self.conns._connection_dict[key_type]
        key_dict[key] = key_dict.pop(con_data.key)
        # Update con_data key
        con_data.set_key(key)
        con_data.metadata.key_origin = key

    def merge_connections(
        self, *connections: str | ConnectionData, name: str | None = None
    ) -> None:
        """
        Merge multiple ConnectionData objects into one.

        This method takes multiple ConnectionData objects and merges them into a single
        ConnectionData object. The first ConnectionData object in the list is used as
        the base, and subsequent ConnectionData objects are merged into it. The method
        ensures that the merged ConnectionData object maintains consistency and updates
        the dependency maps accordingly.

        Args:
            *connections (ConnectionData): Multiple ConnectionData objects to be merged.

        Returns:
            Updates: An Updates object containing the changes made during the merge.
        """
        # TODO: raise if frozen model's merge_connections is called?
        # TODO: raise error if two named keys are merged without naming.
        # TODO: Delete named attributes after merge.
        if len(connections) >= 2:
            updates = Updates()
            conn1 = self.conns.get_extracted_connection(connections[0])

            for conn in connections[1:]:
                conn2 = self.conns.get_extracted_connection(conn)
                d_map = self.dependency_map.local_output_dependency_map
                if conn2 in d_map:
                    if conn1 in d_map:
                        raise KeyError(
                            "IOKey object can not have more than one output "
                            "connection. Multi-write error!"
                        )
                    conn1, conn2 = conn2, conn1
                updates |= self._merge_connections(conn1, conn2, name=name)
            self.constraint_solver(updates)

    def _merge_connections(
        self,
        connection1: ConnectionData,
        connection2: ConnectionData,
        name: str | None = None,
    ) -> Updates:
        # This method is used if there is 2 Connection objects to represent same Edge.
        # In this case, connection2 is updated with connection1's data and it is removed
        # from dag, dependency_map, self attribute (if exists) and Connections object.

        # TODO: Check multi-write error for Connect type.

        conn1 = self.conns.get_con_by_metadata(connection1.metadata)
        conn2 = self.conns.get_con_by_metadata(connection2.metadata)

        if conn1 is None or conn2 is None or conn1 == conn2:
            return Updates()

        # Remove conn2 from connections dict
        con1_key = conn1.key

        if connection2 in self.conns.output_connections:
            if con1_key not in self.conns.output_keys:
                self.conns.set_connection_type(connection1, KeyType.OUTPUT)
            if con1_key in self.input_keys:
                self.conns.set_connection_type(conn1, KeyType.INTERNAL)
        elif conn2 in self.conns.internal_connections and con1_key in self.input_keys:
            self.conns.set_connection_type(conn1, KeyType.INTERNAL)

        # Switch all connection2 objects with connection1 object in current dag.
        for m, m_info in self.dag.items():
            local_conns = m.conns.get_cons_by_metadata(conn2.metadata)
            if local_conns is None:
                continue

            for local_conn in local_conns:
                if m_info.get(local_conn.key) is not None:
                    self.dag[m][local_conn.key] = conn1

        d_out = self.dependency_map.local_output_dependency_map
        d_in = self.dependency_map.local_input_dependency_map
        # Update dependecy map, we need to update only local maps
        for o_conn, key_info in d_out.items():
            if conn2 in key_info[1]:
                d_out[o_conn][1].remove(conn2)
                d_out[o_conn][1].add(conn1)

        if conn2 in d_out:
            d_out[conn1] = d_out.pop(conn2)

        if conn2 in d_in:
            old_dependencies = d_in.pop(conn2)
            d_in.setdefault(conn1, old_dependencies)
            for dependecy in old_dependencies:
                if dependecy not in d_in[conn1]:
                    d_in[conn1].append(dependecy)

        self.dependency_map.merge_global_connections(conn1, conn2)
        self.dependency_map.merge_global_caches(conn1, conn2)
        updates = self._match_hyper_edges(conn1.metadata, conn2.metadata)

        self.conns.remove_connection(conn2)
        if name is None:
            if not conn1.is_autogenerated and not conn2.is_autogenerated:
                raise KeyError(
                    "Requires a connection to have only one unique key "
                    "name but encountered more!"
                )
            elif conn2.is_autogenerated and not conn1.is_autogenerated:
                name = conn1.key
            elif conn1.is_autogenerated and not conn2.is_autogenerated:
                name = conn2.key
        if name is not None:
            self.rename_key(conn1, name)
            # TODO: Deleted connection's 'key' attribute is not updated.
            # Consider updating it.

        # Update assigned attributes with conn2 to conn1.
        self._update_assigned_attributes(conn1, conn2)

        return updates

    def extend(
        self,
        model: BaseModel | BaseModel,
        trace: bool = True,
        /,
        **kwargs: ConnectionDataType,
    ) -> None:
        # Check possible errors before the extension.
        model.check_extendability()
        if self.parent is not None:
            raise AttributeError("Child model could not be re-extended!")
        if self == model:
            raise KeyError("Model can not extend with itself!")
        if self._enforce_jit and not model.jittable:
            raise Exception(
                "Model with enforced Jit can not be extended by a non-jittable model! \
                            Jit can be unforced by setting enforce_jit = False"
            )
        if model.name is not None:
            # TODO: We could store model names in a set to check if it is unique.
            for m in self.dag:
                if m.name == model.name:
                    raise KeyError(f"Model already has a submodel named {model.name}.")

        model.parent = self
        # Freeze the model.
        model._freeze()

        updates = Updates()
        self.state_connections |= model.state_connections

        shape_info: dict[str, ShapeTemplateType] = {}
        submodel_dag: dict[str, ConnectionData] = {}
        updates = self.constraint_solver.match(model.constraint_solver)
        external_keys = list(model.external_keys)
        external_keys += [
            item.key for item in model.conns.couts if item.key not in external_keys
        ]

        for local_key in external_keys:
            value = kwargs.get(local_key, NOT_GIVEN)
            if isinstance(value, BaseKey):
                if value.value_shape is not None:
                    shape_info |= {local_key: value.value_shape}

                expose = value._expose
                name = value.get_key()
                # TODO: We should not operate different if _connections is given. Fix
                # this and also fix corresponding tests and dict conversions with
                # "connect".
                if expose is None and (
                    name is None or self.conns.get_connection(name) is None
                ):
                    value._expose = True

            con_obj, _updates = self._add_connection(
                model, local_key, value, updates, trace
            )
            updates |= _updates
            submodel_dag[local_key] = con_obj
            if tensors := con_obj.metadata.tensors:
                # assert isinstance(con_obj.metadata._value, Tensor)
                updates.shape_updates |= tensors

        # Replace shape info keys, which are local keys, with global equivalents.
        shape_info = {
            submodel_dag[key].key: template for key, template in shape_info.items()
        }

        # Insert to self dag as a FrozenDict.""
        # Since we update dag in merge_connections, we could not use FrozenDict.
        self.dag[model] = model_dag = submodel_dag

        self.dependency_map.add_model_dag(model, model_dag)

        # Set given shapes.
        self._set_shapes(**shape_info)  # TODO: Should "trace" be set to True?.
        self.constraint_solver(updates)

        model.constraint_solver.clear()
        model.conns.connections_dict = {}

        # Update jittablity by using model's jittablity.
        self._jittable &= model.jittable

    @staticmethod
    def _update_key_name(
        new_key: str,
        underscored_keys: set[str],
        raw_keys: dict[str, list[str]],
        key_mappings: dict[str, str],
        key_origin: str,
        input_set: set[str],
    ) -> tuple[str, str]:
        # Add underscore if generated key name exists in input keys
        key_prefix = "_"
        # check any of key_prefix + raw_keys[key_origin] in input keys
        flag = True
        while flag:
            flag = False
            for item in raw_keys[key_origin]:
                if key_prefix + key_mappings[item] in input_set | set(
                    key_mappings.values()
                ):
                    key_prefix += "_"
                    flag = True

        new_key = key_prefix + new_key
        underscored_keys.add(key_origin)
        # Update same origin key names that has been previously added.
        for raw_key in raw_keys[key_origin]:
            key_mappings[raw_key] = key_prefix + key_mappings[raw_key]
        raw_keys[key_prefix + key_origin] = raw_keys.pop(key_origin)
        key_origin = key_prefix + key_origin
        return new_key, key_origin

    def generate_keys(
        self,
        symbolic: bool = True,
        include_internals: bool = True,
        include_outputs: bool = False,
    ) -> dict[str, str]:
        if self.dag == {}:
            return {}
        key_mappings: dict[str, str] = {}
        raw_keys: dict[str, list[str]] = {}
        underscored_keys = set[str]()

        if include_outputs:
            input_set = set(self.external_keys)
            keys = "external_keys"
        else:
            input_set = set(self.input_keys)
            keys = "input_keys"

        sorted_inputs = [
            self.dag[m][key].key
            for m in self.get_models_in_topological_order()
            for key in getattr(m, keys)
            if self.dag[m][key].key in input_set
        ]
        # TODO: remove duplicate loop traverse
        for key in sorted_inputs:
            new_key = key
            if key[0] != "$":
                continue
            # TODO: Discuss if we want to generate input key only
            # if len(self.conns.cins) == 1, or we could name all canonical inputs.
            if (
                len(self.conns.cins) == 1
                and key == self.cin.key
                and "input" not in self.input_keys
            ):
                # Handle canonical input
                new_key = "input"
            else:
                key_origin = self.conns.get_key_origin(key)
                assert key_origin is not None
                # Add prefix until key_origin not in underscored_keys and input_keys.
                while (
                    key_origin in (underscored_keys | self.input_keys)
                    or key_origin == "input"
                ):
                    key_origin = "_" + key_origin

                raw_keys.setdefault(key_origin, [])
                key_idx = len(raw_keys[key_origin])
                if key_idx == 0:
                    # Set key origin as is for the initial key.
                    key_suffix = ""
                else:
                    key_suffix = "_" + str(key_idx)
                    if key_idx == 1:
                        # Update initial key if same key origin is encountered
                        # (add index to initial key).
                        raw_key = raw_keys[key_origin][0]
                        key_mappings[raw_key] = key_mappings[raw_key] + "_0"
                        if key_mappings[raw_key] in self.input_keys:
                            new_key, key_origin = self._update_key_name(
                                new_key,
                                underscored_keys,
                                raw_keys,
                                key_mappings,
                                key_origin,
                                set(self.input_keys),
                            )

                new_key = key_origin + key_suffix
                if new_key in self.input_keys:
                    new_key, key_origin = self._update_key_name(
                        new_key,
                        underscored_keys,
                        raw_keys,
                        key_mappings,
                        key_origin,
                        set(self.input_keys),
                    )
                raw_keys[key_origin].append(key)
            key_mappings[key] = new_key

        if include_internals:
            sorted_models = self.get_models_in_topological_order()
            internal_key_mappings: dict[str, str] = {}
            for idx, m in enumerate(sorted_models):
                for key in m.external_keys:
                    outer_conn = self.dag[m][key]
                    outer_key = outer_conn.key
                    if outer_key[0] == "$":
                        # if key is autogenerated, generate a name for the key
                        model_name = m.class_name
                        key_origin = outer_conn.metadata.key_origin
                        assert key_origin is not None

                        generated_name = (
                            "_" + model_name + "_" + str(idx) + "_" + key_origin
                        )

                        # if key is an output key, directly write it
                        # to the internal_key_mappings
                        # or
                        # if key is an input key, first check if the key
                        # is already in internal_key mappings to avoid
                        # overwrite
                        write_to_internal_key_mappings = (
                            key in m.conns.output_keys
                            or internal_key_mappings.get(
                                outer_key, key_mappings.get(outer_key)
                            )
                            is None
                        )

                        while (
                            generated_name in internal_key_mappings.values()
                            and write_to_internal_key_mappings
                        ):
                            assert key_origin is not None
                            key_origin = "_" + key_origin
                            generated_name = (
                                "_" + model_name + "_" + str(idx) + "_" + key_origin
                            )

                        if write_to_internal_key_mappings:
                            internal_key_mappings[outer_key] = generated_name

            key_mappings = internal_key_mappings | key_mappings
        if symbolic:
            key_mappings = {key: "$" + value for key, value in key_mappings.items()}
        return key_mappings

    def get_unique_submodel_names(self) -> dict[BaseModel, str]:
        name_mapping: dict[BaseModel, str] = {}
        existing_names: set[str] = set()
        model_type_dict: dict[str, list[BaseModel]] = {}

        # First, assign existing names and track used names.
        # Also save unnamed models to model_type_dict.
        for model in self.dag:
            if model.name:
                name_mapping[model] = model.name
                existing_names.add(model.name)
            else:
                model_type_dict.setdefault(model.class_name, []).append(model)

        # Iterate over different model types among unnamed models.
        for model_type, model_list in model_type_dict.items():
            counter = 0
            # Iterate over same class model objects to name them.
            for i, model in enumerate(model_list):
                if len(model_list) == 1:
                    # If there is only one model of a type, do not increment counter.
                    counter -= 1
                    name = model_type
                else:
                    name = f"{model_type}_{counter + i}"
                while name in existing_names:
                    counter += 1  # counter is incremented until a unique name is found.
                    name = f"{model_type}_{counter + i}"
                name_mapping[model] = name
                existing_names.add(name)
        return name_mapping

    def _freeze(self) -> None:
        for cout in self.conns.couts:
            self.conns.set_connection_type(cout, KeyType.OUTPUT, safe=False)
        self.dependency_map.update_all_keys()

        # Name unnamed submodels before freezing considering the insertion order.
        model_names = self.get_unique_submodel_names()
        for m in self.dag:
            if m.name is None:
                m.name = model_names[m]

        if self.formula_key is not None:
            # Must be convertable to primitive.
            assert len(self.conns.output_keys) == 1, (
                "Logical models have altenative primitive implementation must "
                "have only 1 output."
            )
        # super()._freeze()
        self.is_frozen = True

    @staticmethod
    def _reverse_dfs(
        node: BaseModel,
        graph: dict[BaseModel, OrderedSet[BaseModel]],
        topo_order: list[BaseModel],
        visited: set[BaseModel],
    ) -> None:
        visited.add(node)
        for child in graph.get(node, OrderedSet()):
            if child not in visited:
                BaseModel._reverse_dfs(child, graph, topo_order, visited)
        topo_order.append(node)

    def get_models_in_topological_order(
        self, start: BaseModel | None = None
    ) -> list[BaseModel]:
        """
        Get topological order of submodels based on dependency. If a start model
        is provided, only the models reachable from it (including itself) are
        returned. Otherwise, all models in the dependency graph are returned.
        """
        # Build graph: each parent model maps to an OrderedSet of its child models.
        dep_map = self.dependency_map.local_output_dependency_map
        graph: dict[BaseModel, OrderedSet[BaseModel]] = {}
        for _, (parent_model, child_conns) in dep_map.items():
            # Ensure parent_model is in graph even if it has no children.
            graph.setdefault(parent_model, OrderedSet())
            for child_conn in child_conns:
                if child_conn in dep_map:
                    child_model = dep_map[child_conn][0]
                    graph.setdefault(parent_model, OrderedSet()).add(child_model)

        topo_order: list[BaseModel] = []
        visited: set[BaseModel] = set()

        if start is not None:
            # Perform DFS starting from the given model.
            BaseModel._reverse_dfs(start, graph, topo_order, visited)
        else:
            # Perform DFS for all models.
            for model in graph:
                if model not in visited:
                    BaseModel._reverse_dfs(model, graph, topo_order, visited)
        # Reverse the list to get standard topological order.
        return topo_order

    # TODO: Summary should be isolated from the model.
    def extract_connection_info(
        self,
        name_mappings: dict[BaseModel, str],
        data_to_key_map: dict[IOHyperEdge, list[str]] | None = None,
        data_memo: Mapping[int, IOHyperEdge] | None = None,
    ) -> dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]]:
        conn_info: dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]] = {}
        if self.input_keys:
            if data_to_key_map is None:
                data_to_key_map = {}
            if data_memo is None:
                data_memo = {}
            model_key_map: dict[BaseModel, dict[str, str]] = {}

            # handle the case when model is constructed with += operation. In that case,
            # directly take canonical output as the output_key.

            # TODO: We may expose all canonical outputs for summary instead of
            # checking only len(self.conns.couts) == 1.
            output_keys = (
                ([self.cout.key] if len(self.conns.couts) == 1 else [])
                if not self.conns.output_keys
                else self.conns.output_keys
            )
            # extract key mappings and data map of outer model
            key_mappings = self.generate_keys(
                include_internals=False, include_outputs=True
            )
            data_map = {key: conn.metadata for key, conn in self.conns.all.items()}

            # Sort in topological order
            sorted_models = self.get_models_in_topological_order()

            for model in sorted_models:
                model_name = name_mappings[model]
                m_info = self.dag[model]
                # set default structure of conn_info and shape_info
                conns = conn_info.setdefault(model_name, ({}, {}))
                # include input keys with Tensor value
                input_keys = tuple(model.input_keys)
                # Generate sub_model key_map and data map
                model_key_map[model] = m_key_mappings = model.generate_keys(
                    include_internals=False, include_outputs=True
                )
                m_data_map = {
                    key: conn.metadata for key, conn in model.conns.all.items()
                }
                for inner_key in input_keys + tuple(model.conns.output_keys):
                    # Find the data of the key, if data memo is given, extract its
                    # copied version and extract the shapes
                    key_data = data_memo.get(
                        id(m_data_map[inner_key]), m_data_map[inner_key]
                    )

                    # Find inner and outer keys. Also find their updated version based
                    # on their key mappings
                    updated_inner_key = m_key_mappings.get(inner_key, inner_key)
                    outer_conn = m_info[inner_key]
                    outer_key = outer_conn.key
                    updated_outer_key = data_to_key_map.get(
                        key_data, [key_mappings.get(outer_key, outer_key)]
                    )

                    # take and setdefault connection list in which update will be done
                    conn = conns[inner_key in model.conns.output_keys].setdefault(
                        updated_inner_key, []
                    )
                    if inner_key not in input_keys:
                        continue

                    if key_data.is_valued:
                        val = key_data.value
                        conn.append(str(val))

                    elif outer_key in self.input_keys:
                        # If outer_key in input_keys of overall model, it means
                        # the input key is overall input to the model. Do the
                        # updates accordingly
                        input_name = ["'" + key + "'" for key in updated_outer_key]
                        conn.extend(input_name)
                    else:
                        # if input_key is not in self.input_keys, that means this
                        # input key connected to a model and it is an internal
                        # connection. Find the connected model and do the intializations
                        con_model = self.dependency_map.local_output_dependency_map[
                            outer_conn
                        ][0]
                        con_generated_keys = model_key_map.setdefault(
                            con_model,
                            con_model.generate_keys(
                                include_internals=False, include_outputs=True
                            ),
                        )
                        conn_info.setdefault(name_mappings[con_model], ({}, {}))
                        model_conn = m_info[inner_key]
                        con = con_model.conns.get_con_by_metadata(model_conn.metadata)
                        assert con is not None, "Connection is not found"
                        con_key = con.key

                        con_key = con_generated_keys.get(con_key, con_key)
                        # Since being internal key means having two sided connection,
                        # Two updates on conn_info dict needs to be done. one for
                        # model's input key and other for connected_model's output
                        # key. do the updates accordingly.
                        conn_info[model_name][0].setdefault(
                            updated_inner_key, []
                        ).append(name_mappings[con_model] + "." + con_key)
                        conn_info[name_mappings[con_model]][1].setdefault(
                            con_key, []
                        ).append(model_name + "." + updated_inner_key)

            for outer_key in output_keys:
                # Lastly, traverse through output keys of the overall model
                # Find the connected model, and find the inner key by finding
                # the metadata
                metadata = self.conns.get_metadata(outer_key)
                outer_out_conn = self.conns.get_connection(outer_key)

                assert metadata is not None, "Metadata is not found!"
                assert outer_out_conn is not None, "Connection is not found"

                model = self.dependency_map.local_output_dependency_map[outer_out_conn][
                    0
                ]
                other_conn = model.conns.get_con_by_metadata(metadata)
                assert other_conn is not None, "Connection is not found"

                inner_key = other_conn.key
                updated_inner_key = model_key_map[model].get(inner_key, inner_key)
                key_data = data_memo.get(id(data_map[outer_key]), data_map[outer_key])
                updated_outer_key = data_to_key_map.get(
                    key_data, [key_mappings.get(outer_key, outer_key)]
                )
                if updated_outer_key[0][0] == "$":
                    # There is only possibilty of outer key is found to be with $ sign.
                    # That is, if model is constructed with += operator. In that case,
                    # canonical output will be external key even if it is not named by
                    #  user. Therefore, handle the case with dicrectly writing $output
                    updated_outer_key = ["$output"]
                model_name = name_mappings[model]
                conn_info[model_name][1][updated_inner_key].extend(
                    ["'" + key + "'" for key in updated_outer_key]
                )

        return conn_info

    @property
    def grad_formula(self) -> str:
        if self.formula_key is None:
            raise AttributeError("Model has no formula key!")
        return self.formula_key + "_grad"

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

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

    def _create_key_name(self) -> str:
        self.inter_key_count += 1
        return "$" + str(self.inter_key_count)

    def attach_connection(self, con: ConnectionData) -> None:
        metadata = con.metadata
        # If key is not provided, create a new key name and
        # label it as auto-generated.
        key = con.get_key()
        if key is not None and self.conns.get_connection(key) is not None:
            raise KeyError("Connection with name " + key + " already exists!")
        if is_autogenerated := key is None:
            key = self._create_key_name()
            con.set_key(key)

        con.model = self
        if not is_autogenerated:
            # Set key_origin into metadata
            metadata.key_origin = key

        self.conns.add(con)
        if not con.is_autogenerated:
            assert key is not None
            setattr(self, key, con)

    def _create_connection(
        self, metadata: IOHyperEdge, key: str | None = None
    ) -> ConnectionData:
        connection = ConnectionData(key)
        connection.metadata = metadata
        return connection

    def _set_differentiability(
        self,
        config: dict[ConnectionData, bool] | None = None,
        trace: bool = False,
        /,
        **kwargs: bool,
    ) -> None:
        updates = Updates()
        if config is None:
            config = {}

        for key, value in chain(config.items(), kwargs.items()):
            if isinstance(key, str):
                if key not in self.conns.all:
                    raise KeyError(f"Connection {key} is not found in the model.")

                conn_data = self.conns.all[key]
                updates |= conn_data.set_differentiability(value)
            elif isinstance(key, ConnectionData):
                if key not in self.conns.all.values():
                    raise KeyError(f"Connection {key} is not found in the model.")
                conn_data = key
                updates |= conn_data.set_differentiability(value)

            if trace:
                self.assigned_differentiabilities[conn_data] = value

        model = self._get_outermost_parent()
        model.constraint_solver(updates)

    def set_differentiability(
        self, config: dict[ConnectionData, bool] | None = None, /, **kwargs: bool
    ) -> None:
        self._set_differentiability(config, True, **kwargs)

    def _set_shapes(
        self,
        shapes: Mapping[ConnectionData, ShapeTemplateType] | None = None,
        trace: bool = False,
        /,
        **kwargs: ShapeTemplateType,
    ) -> None:
        # Initialize assigned shapes dictionary to store assigned shapes.
        assigned_shapes: dict[ConnectionData, ShapeTemplateType] = {}
        updates = Updates()
        if shapes is None:
            shapes = {}

        model = self._get_outermost_parent()
        used_keys: dict[str | int, ShapeType] = {}
        shape_nodes: dict[str | ConnectionData, tuple[ShapeNode, str]] = {}
        # TODO: Can this be refactored to use a single loop?
        for key, shape in chain(shapes.items(), kwargs.items()):
            assert isinstance(key, str | ConnectionData)
            metadata = self.conns.extract_metadata(key)
            given_repr = create_shape_repr(shape, model.constraint_solver, used_keys)
            # Get inner string representation of the metadata and save
            # use this name in order to merge .
            conn = self.conns.get_con_by_metadata(metadata)
            assert conn is not None
            inner_key = conn.key
            shape_nodes[key] = (given_repr.node, inner_key)
            # In order to store assigned shapes, we need to store corresponding model
            # and index of the connection for that model.
            assigned_shapes[conn] = shape

        # Apply updates to the shape nodes.
        for key in chain(shapes, kwargs):
            assert isinstance(key, str | ConnectionData)
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

    def set_shapes(
        self,
        shapes: Mapping[ConnectionData, ShapeTemplateType] | None = None,
        /,
        **kwargs: ShapeTemplateType,
    ) -> None:
        self._set_shapes(shapes, True, **kwargs)

    def _set_types(
        self,
        config: Mapping[
            ConnectionData,
            type | UnionType | ScalarType | type[Tensor[int | float | bool]],
        ]
        | None = None,
        trace: bool = False,
        **kwargs: type | UnionType | ScalarType | type[Tensor[int | float | bool]],
    ) -> None:  # Initialize assigned shapes dictionary to store assigned shapes.
        if config is None:
            config = {}
        # Get the outermost parent as all the updates will happen here.
        model = self._get_outermost_parent()
        updates = Updates()
        for key, key_type in chain(config.items(), kwargs.items()):
            assert isinstance(key, str | ConnectionData)
            metadata = self.conns.extract_metadata(key)
            conn = self.conns.get_con_by_metadata(metadata)
            assert conn is not None
            updates |= metadata.set_type(key_type)
            if trace:
                # Store assigned types in the model.
                if key_type is Tensor:
                    key_type = Tensor[int | float | bool]
                self.assigned_types[conn] = key_type
        # Run the constraints for updating affected connections.
        model.constraint_solver(updates)

    def set_types(
        self,
        config: Mapping[
            ConnectionData,
            type | UnionType | ScalarType | type[Tensor[int | float | bool]],
        ]
        | None = None,
        /,
        **kwargs: type | UnionType | ScalarType | type[Tensor[int | float | bool]],
    ) -> None:  # Initialize assigned shapes dictionary to store assigned shapes.
        return self._set_types(config, True, **kwargs)

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

    def set_values(
        self,
        config: Mapping[
            ConnectionData, Tensor[int | float | bool] | MainValueType | str
        ]
        | None = None,
        /,
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
            assert isinstance(key, str | ConnectionData)
            metadata = self.conns.extract_metadata(key)
            # Perform validity check and updates on model.
            if (conn_data := model.conns.get_con_by_metadata(metadata)) is None:
                raise KeyError("Requires valid key or Connection to set values!")
            updates |= model._set_value(conn_data, value)

        # Solve constraints with the updated values.
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
        trace: bool = False,
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

        if trace:
            self.assigned_constraints.append({"fn": fn.__name__, "keys": keys})

        return constr

    def add_constraint(
        self,
        fn: ConstraintFunctionType,
        keys: list[str],
        type: list[UpdateType] | None = None,
        dependencies: set[Constraint] | None = None,
    ) -> Constraint:
        return self._add_constraint(fn, keys, type, dependencies, True)

    @property
    def cin(self) -> ConnectionData:
        if (cin_len := len(self.conns.cins)) != 1:
            raise KeyError(
                f"Currently, there exists {cin_len} canonical inputs, model "
                "should have exactly one canonical input!"
            )
        return next(iter(self.conns.cins))

    @property
    def cout(self) -> ConnectionData:
        if (cout_len := len(self.conns.couts)) != 1:
            raise KeyError(
                f"Currently, there exists {cout_len} canonical outputs, model "
                "should have exactly one canonical output!"
            )

        return next(iter(self.conns.couts))

    def set_cin(self, *connections: str | ConnectionData, safe: bool = True) -> None:
        self._set_cin(*connections, safe=safe, trace=True)

    def _set_cin(
        self, *connections: str | ConnectionData, safe: bool = True, trace: bool = False
    ) -> None:
        self.conns.cins = set()
        for given_conn in connections:
            conn = self.conns.get_extracted_connection(given_conn)

            is_valued = conn.metadata.is_valued
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
            if trace:
                self.assigned_cins.add(conn)

    def set_cout(self, *connections: str | ConnectionData, safe: bool = True) -> None:
        self._set_cout(*connections, safe=safe, trace=True)

    def _set_cout(
        self, *connections: str | ConnectionData, safe: bool = True, trace: bool = False
    ) -> None:
        self.conns.couts = set()
        for given_conn in connections:
            conn = self.conns.get_extracted_connection(given_conn)
            is_valued = conn.metadata.is_valued
            if conn not in self.dependency_map.local_output_dependency_map or is_valued:
                if safe:
                    raise ValueError(
                        "To set a connection as canonical output, "
                        "connection must be an output connection!"
                    )
            else:
                self.conns.couts.add(conn)
            if trace:
                self.assigned_couts.add(conn)

    def _match_hyper_edges(self, left: IOHyperEdge, right: IOHyperEdge) -> Updates:
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
        return left.match(right)

    def _update_assigned_attributes(
        self, new: ConnectionData, old: ConnectionData
    ) -> None:
        """
        Update assigned attributes by replacing occurrences of the old ConnectionData
        with the new ConnectionData.

        This method updates the following attributes:
        - assigned_shapes: Replaces old ConnectionData with new ConnectionData
          in the assigned shapes.
        - assigned_types: Replaces old ConnectionData with new ConnectionData
          in the assigned types.
        - assigned_differentiabilities: Replaces old ConnectionData with new
          ConnectionData in the assigned differentiabilities.
        - assigned_canonicals: Updates the 'cins' and 'couts' sets by removing the
          old ConnectionData and adding the new ConnectionData.

        Args:
            new (ConnectionData): The new ConnectionData to replace the old one.
            old (ConnectionData): The old ConnectionData to be replaced.
        """
        for shape_info in self.assigned_shapes:
            if old in shape_info:
                shape_info[new] = shape_info.pop(old)
        if old in self.assigned_types:
            self.assigned_types[new] = self.assigned_types.pop(old)

        if old in self.assigned_differentiabilities:
            self.assigned_differentiabilities[new] = (
                self.assigned_differentiabilities.pop(old)
            )
        # Assigned canonicals
        if old in self.assigned_cins:
            self.assigned_cins.remove(old)
            self.assigned_cins.add(new)
        elif old in self.assigned_couts:
            self.assigned_couts.remove(old)
            self.assigned_couts.add(new)


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
        for conn in self.conns.latent_input_connections:
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
            if (
                input_conn in self.conns.input_connections
                or input_conn in self.conns.latent_input_connections
            ):
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
