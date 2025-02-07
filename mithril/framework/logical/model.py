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

from collections.abc import KeysView, Mapping, Sequence
from dataclasses import dataclass
from types import EllipsisType, UnionType
from typing import Any, Self

from ...core import Dtype as CoreDtype
from ...utils.utils import find_dominant_type
from ..common import (
    NOT_GIVEN,
    TBD,
    BaseKey,
    ConnectionData,
    ConnectionDataType,
    IOHyperEdge,
    MainValueInstance,
    MainValueType,
    NullConnection,
    ScalarType,
    ScalarValueType,
    ShapeTemplateType,
    Tensor,
    ToBeDetermined,
    UniadicRecord,
    Variadic,
    get_summary,
    get_summary_shapes,
    get_summary_types,
)

# from .base import BaseModel, ConnectionDataType
from .base import BaseModel
from .operator import Operator
from .operators import (
    AbsoluteOp,
    AddOp,
    CastOp,
    CosineOp,
    DivideOp,
    DtypeOp,
    EqualOp,
    ExponentialOp,
    FloorDivideOp,
    GreaterEqualOp,
    GreaterOp,
    IndexerOp,
    ItemOp,
    LengthOp,
    LessEqualOp,
    LessOp,
    LogicalAndOp,
    LogicalNotOp,
    LogicalOrOp,
    LogicalXOrOp,
    MatrixMultiplyOp,
    MaxOp,
    MeanOp,
    MinOp,
    MinusOp,
    MultiplyOp,
    NotEqualOp,
    PowerOp,
    ProdOp,
    ReshapeOp,
    ShapeOp,
    ShiftLeftOp,
    ShiftRightOp,
    SineOp,
    SizeOp,
    SliceOp,
    SplitOp,
    SqrtOp,
    SubtractOp,
    SumOp,
    ToListOp,
    ToTensorOp,
    ToTupleOp,
    TransposeOp,
    VarianceOp,
)

__all__ = [
    "Connection",
    "IOKey",
    "ExtendTemplate",
    "ExtendInfo",
    "Model",
    "ConnectionType",
    "ConnectionInstanceType",
    "TemplateConnectionType",
    "define_unique_names",
]


class TemplateBase:
    def __getitem__(
        self,
        key: slice
        | int
        | EllipsisType
        | tuple[slice | int | None | EllipsisType | TemplateBase, ...]
        | IOKey
        | TemplateBase
        | None,
    ) -> ExtendTemplate:
        match key:
            case slice():
                slice_output = ExtendTemplate(
                    connections=[key.start, key.stop, key.step], model=SliceOp
                )
                output = ExtendTemplate(
                    connections=[self, slice_output], model=IndexerOp
                )

            case int() | EllipsisType() | None:
                output = ExtendTemplate(connections=[self, key], model=IndexerOp)

            case tuple():
                connections: list[TemplateBase | int | None | EllipsisType] = []
                for item in key:
                    if isinstance(item, slice):
                        slice_output = ExtendTemplate(
                            connections=[item.start, item.stop, item.step],
                            model=SliceOp,
                        )
                        connections.append(slice_output)
                    else:
                        connections.append(item)
                tuple_template = ExtendTemplate(
                    connections=connections,
                    model=ToTupleOp,
                    defaults={"n": len(key)},
                )
                output = ExtendTemplate(
                    connections=[self, tuple_template], model=IndexerOp
                )
        return output

    def __add__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=AddOp)

    def __radd__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=AddOp)

    def __sub__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=SubtractOp)

    def __rsub__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=SubtractOp)

    def __mul__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=MultiplyOp)

    def __rmul__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=MultiplyOp)

    def __truediv__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=DivideOp)

    def __rtruediv__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=DivideOp)

    def __floordiv__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=FloorDivideOp)

    def __rfloordiv__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=FloorDivideOp)

    def __pow__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(
            connections=[self, other], model=PowerOp, defaults={"robust": False}
        )

    def __rpow__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(
            connections=[other, self], model=PowerOp, defaults={"robust": False}
        )

    def __matmul__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=MatrixMultiplyOp)

    def __gt__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=GreaterOp)

    def __rgt__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=GreaterOp)

    def __ge__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=GreaterEqualOp)

    def __rge__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=GreaterEqualOp)

    def __lt__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=LessOp)

    def __rlt__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=LessOp)

    def __le__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=LessEqualOp)

    def __rle__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=LessEqualOp)

    def __eq__(self, other: object) -> ExtendTemplate:  # type: ignore[override]
        if isinstance(
            other, int | float | bool | list | Connection | IOKey | tuple | Tensor
        ):
            return ExtendTemplate(connections=[self, other], model=EqualOp)
        else:
            raise ValueError("Unsupported type for equality operation.")

    def __req__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=EqualOp)

    def __ne__(self, other: object) -> ExtendTemplate:  # type: ignore[override]
        if isinstance(
            other, int | float | bool | list | Connection | IOKey | tuple | Tensor
        ):
            return ExtendTemplate(connections=[self, other], model=NotEqualOp)
        else:
            raise ValueError("Unsupported type for equality operation.")

    def __rne__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=NotEqualOp)

    def __and__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=LogicalAndOp)

    def __rand__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=LogicalAndOp)

    def __or__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=LogicalOrOp)

    def __ror__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=LogicalOrOp)

    def __xor__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=LogicalXOrOp)

    def __rxor__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=LogicalXOrOp)

    def __lshift__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=ShiftLeftOp)

    def __rlshift__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=ShiftLeftOp)

    def __rshift__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, other], model=ShiftRightOp)

    def __rrshift__(self, other: TemplateConnectionType) -> ExtendTemplate:
        return ExtendTemplate(connections=[other, self], model=ShiftRightOp)

    def __invert__(self) -> ExtendTemplate:
        return ExtendTemplate(connections=[self], model=LogicalNotOp)

    def __neg__(self) -> ExtendTemplate:
        return ExtendTemplate(connections=[self], model=MinusOp)

    def abs(self) -> ExtendTemplate:
        return ExtendTemplate(connections=[self], model=AbsoluteOp)

    def len(self) -> ExtendTemplate:
        return ExtendTemplate(connections=[self], model=LengthOp)

    @property
    def shape(self) -> ExtendTemplate:
        return ExtendTemplate(connections=[self], model=ShapeOp)

    def reshape(
        self, shape: tuple[int | TemplateBase, ...] | TemplateBase
    ) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, shape], model=ReshapeOp)

    def size(
        self, dim: int | tuple[int, ...] | TemplateBase | None = None
    ) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, dim], model=SizeOp)

    def tensor(self) -> ExtendTemplate:
        return ExtendTemplate(
            connections=[self], model=ToTensorOp, defaults={"dtype": None}
        )

    def mean(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
    ) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, axis, keepdim], model=MeanOp)

    def sum(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
    ) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, axis, keepdim], model=SumOp)

    def max(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
    ) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, axis, keepdim], model=MaxOp)

    def min(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
    ) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, axis, keepdim], model=MinOp)

    def prod(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
    ) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, axis, keepdim], model=ProdOp)

    def var(
        self,
        axis: int | tuple[int, ...] | TemplateBase | None = None,
        keepdim: bool = False,
        correction: float | None = 0.0,
    ) -> ExtendTemplate:
        return ExtendTemplate(
            connections=[self, axis, keepdim, correction], model=VarianceOp
        )

    def sqrt(self) -> ExtendTemplate:
        return ExtendTemplate(
            connections=[self], model=SqrtOp, defaults={"robust": False}
        )

    def exp(self) -> ExtendTemplate:
        return ExtendTemplate(connections=[self], model=ExponentialOp)

    def transpose(
        self, axes: tuple[int, ...] | TemplateBase | None = None
    ) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, axes], model=TransposeOp)

    def split(self, split_size: int, axis: int) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, split_size, axis], model=SplitOp)

    def item(self) -> ExtendTemplate:
        return ExtendTemplate(connections=[self], model=ItemOp)

    def cast(self, dtype: CoreDtype | None = None) -> ExtendTemplate:
        return ExtendTemplate(connections=[self, dtype], model=CastOp)

    def dtype(self) -> ExtendTemplate:
        return ExtendTemplate(connections=[self], model=DtypeOp)

    def sin(self) -> ExtendTemplate:
        return ExtendTemplate(connections=[self], model=SineOp)

    def cos(self) -> ExtendTemplate:
        return ExtendTemplate(connections=[self], model=CosineOp)


class Connection(TemplateBase):
    def __init__(self, data: ConnectionData) -> None:
        self.data = data

    @property
    def key(self) -> str:
        return self.data.key

    @property
    def metadata(self) -> IOHyperEdge:
        return self.data.metadata

    def set_differentiable(self, differentiable: bool = True) -> None:
        self.data.set_differentiable(differentiable)

    def __hash__(self) -> int:
        return hash(id(self))


class IOKey(BaseKey, TemplateBase):
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
        interval: list[float | int] | None = None,
        connections: set[Connection | str] | None = None,
    ) -> None:
        _connections: set[ConnectionData | str] = {
            con.data if isinstance(con, Connection) else con
            for con in connections or set()
        }

        super().__init__(
            name=name,
            value=value,
            shape=shape,
            type=type,
            expose=expose,
            interval=interval,
            connections=_connections,
        )


class ExtendTemplate(TemplateBase):
    output_connection: ConnectionData | None

    def __init__(
        self,
        connections: Sequence[TemplateConnectionType],
        model: type[BaseModel],
        defaults: dict[str, Any] | None = None,
    ) -> None:
        for connection in connections:
            if isinstance(connection, str):
                raise ValueError(
                    "In extend template operations, 'str' is not a valid type."
                )

        self.connections = connections
        self.model = model

        if defaults is None:
            defaults = {}
        self.defaults = defaults
        self.output_connection = None


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


TemplateConnectionType = (
    TemplateBase
    | int
    | float
    | list[int | float]
    | EllipsisType
    | tuple[slice | int | None | EllipsisType | TemplateBase, ...]
    | None
    | Tensor[int | float | bool]
)

ConnectionType = (
    str
    | MainValueType
    | ExtendTemplate
    | NullConnection
    | IOKey
    | Connection
    | Tensor[int | float | bool]
)

ConnectionInstanceType = (
    str
    | MainValueInstance
    | ExtendTemplate
    | NullConnection
    | IOKey
    | Connection
    | Tensor  # type: ignore
)


class Model(BaseModel):
    def __init__(
        self,
        name: str | None = None,
        formula_key: str | None = None,
        enforce_jit: bool = True,
    ) -> None:
        super().__init__(name, formula_key, enforce_jit)
        self.connection_map: dict[ConnectionData, Connection] = {}

    def __call__(self, **kwargs: ConnectionType) -> ExtendInfo:
        return ExtendInfo(self, kwargs)

    def _create_connection(
        self, metadata: IOHyperEdge, key: str | None = None
    ) -> ConnectionData:
        con_data = super()._create_connection(metadata, key)
        con = Connection(con_data)
        self.connection_map[con_data] = con
        if not con_data.is_key_autogenerated:
            assert key is not None
            setattr(self, key, con)
        return con_data

    # TODO: Refactor _prepare_keys / _unroll_template relation.
    def _prepare_keys(
        self,
        model: BaseModel,
        key: str,
        connection: ConnectionDataType | ConnectionType,
    ) -> BaseKey:
        local_connection = model.conns.get_connection(key)
        assert local_connection is not None, "Connection is not found!"
        _connection: BaseKey | ConnectionData | MainValueInstance | NullConnection | str
        match connection:
            case Connection():
                _connection = connection.data
            case ExtendTemplate():
                # Unroll ExtendTemplate
                con_data = self._unroll_template(connection)
                _connection = BaseKey(connections={con_data}, expose=False)
            case _ if isinstance(
                connection, MainValueInstance | Tensor
            ) and not isinstance(connection, str):
                # find_dominant_type returns the dominant type in a container.
                # If a container has a value of type Connection or ExtendTemplate
                # we add necessary models.
                types = [ConnectionData, ExtendTemplate, Connection, IOKey]
                if (
                    isinstance(connection, tuple | list)
                    and find_dominant_type(connection, False) in types
                ):
                    _model = ToTupleOp if isinstance(connection, tuple) else ToListOp
                    et = ExtendTemplate(connection, _model, {"n": len(connection)})
                    con_data = self._unroll_template(et)
                    _connection = BaseKey(connections={con_data}, expose=False)

                else:
                    assert isinstance(connection, MainValueInstance | Tensor)
                    _connection = BaseKey(value=connection)
            case IOKey():
                expose = connection.expose
                name = connection.name
                # TODO: This check should be removed: conn.connections==set()
                # We should not operate different if _connections is given. Fix this and
                # also fix corresponding tests and dict conversions with "connect".
                if (
                    expose is None
                    and (name is None or self.conns.get_connection(name) is None)
                    and connection.connections == set()
                ):
                    expose = True
                _connection = BaseKey(
                    name=name,
                    expose=expose,
                    connections=connection.connections,
                    type=connection.type,
                    shape=connection.value_shape,
                    value=connection.value,
                )
            case _:
                _connection = connection  # type: ignore
        return super()._prepare_keys(model, key, _connection)

    def _get_conn_data(self, conn: str | ConnectionData) -> ConnectionData:
        if isinstance(conn, str):
            _conn = self.conns.get_connection(conn)
        else:
            _conn = self.conns.get_con_by_metadata(conn.metadata)
        assert isinstance(_conn, ConnectionData)
        return _conn

    def update_key_name(self, connection: ConnectionData, key: str) -> None:
        super().update_key_name(connection, key)
        con_data = self.conns.get_extracted_connection(connection)
        conn = self.connection_map[con_data]
        setattr(self, key, conn)

    def _unroll_template(self, template: ExtendTemplate) -> ConnectionData:
        if template.output_connection is None:
            # Initialize all default init arguments of model as TBD other
            # than the keys in template.defaults, in order to provide
            # given connections to the model after it is created.
            # If we don't do that, it will throw error because of
            # re-setting a Tensor or Scalar value again in extend.
            # TODO: Remove all TBD if default init arguments will be moved to call!!!
            code = template.model.__init__.__code__

            # "self" argument is common for all models, Exclude it by
            # starting co_varnames from 1st index.
            default_args = code.co_varnames[1 : code.co_argcount]
            default_args_dict = {key: TBD for key in default_args} | template.defaults
            default_args_dict.pop("name", None)

            # TODO: Reconsider type ignore!
            model: Operator = template.model(**default_args_dict)  # type: ignore
            keys = {
                local_key: self._prepare_keys(model, local_key, outer_con)  # type: ignore
                for local_key, outer_con in zip(
                    model.input_keys, template.connections, strict=False
                )
            }
            self.extend(model, **keys)

            template.output_connection = model.conns.get_connection("output")
            assert template.output_connection is not None
        return template.output_connection

    def _extend(
        self, model: BaseModel, kwargs: dict[str, ConnectionType] | None = None
    ) -> Self:
        if kwargs is None:
            kwargs = {}
        if self.is_frozen:
            raise AttributeError("Model is frozen and can not be extended!")

        for key, value in kwargs.items():
            _value = value.name if isinstance(value, IOKey) else value

            if isinstance(_value, str) and _value == "":
                if key in model.input_keys:
                    _value = NOT_GIVEN
                else:
                    raise KeyError(
                        "Empty string is not a valid for output connections!"
                    )

                if isinstance(value, IOKey):
                    value.name = None
                else:
                    kwargs[key] = _value

        self.extend(model, **kwargs)  # type: ignore
        return self

    @property
    def cout(self) -> Connection:
        return self.connection_map[self._cout]

    @property
    def cin(self) -> Connection:
        return self.connection_map[self._cin]

    def set_cin(self, *connections: str | Connection, safe: bool = True) -> None:
        data: list[str | ConnectionData] = [
            item if isinstance(item, str) else item.data for item in connections
        ]
        self._set_cin(*data, safe=safe)

    def set_cout(self, *connections: str | Connection, safe: bool = True) -> None:
        data: list[str | ConnectionData] = [
            item if isinstance(item, str) else item.data for item in connections
        ]
        self._set_cout(*data, safe=safe)

    def __add__(self, info: ExtendInfo | Model) -> Self:
        # TODO: Check if info is a valid info for canonical connections.
        # TODO: Add canonical connection information to info.
        if isinstance(info, BaseModel):
            info = info()
        model, kwargs = info.model, info.connections
        given_keys = {key for key, val in kwargs.items() if val is not NullConnection()}
        available_cin = {item.key for item in model.conns.cins} - given_keys
        if len(self.dag) > 0:
            if len(model.conns.cins) == 0:
                raise KeyError(
                    "No existing canonical input is found "
                    "to extension model! Use |= operator."
                )
            if len(available_cin) > 1:
                raise KeyError(
                    "Multiple canonical inputs are not allowed! Use |= operator."
                )
            if len(available_cin) == 1:
                kwargs[next(iter(available_cin))] = self.cout
        return self._extend(model, kwargs)

    __iadd__ = __add__

    def __or__(self, info: ExtendInfo | Model) -> Self:
        # TODO: Check if info is a valid info for extend.
        if isinstance(info, Model):
            info = info()
        return self._extend(info.model, info.connections)

    __ior__ = __or__

    ShapeType = (
        Mapping[str | Connection, ShapeTemplateType]
        | Mapping[str, ShapeTemplateType]
        | Mapping[Connection, ShapeTemplateType]
    )

    def set_shapes(
        self, config: ShapeType | None = None, **kwargs: ShapeTemplateType
    ) -> None:
        if config is None:
            config = {}
        _config: dict[str | ConnectionData, ShapeTemplateType] = {
            key.data if isinstance(key, Connection) else key: value
            for key, value in config.items()
        }
        self._set_shapes(_config, trace=True, updates=None, **kwargs)

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
        if config is None:
            config = {}
        _config: dict[
            str | ConnectionData, Tensor[int | float | bool] | MainValueType | str
        ] = {
            key if isinstance(key, str) else key.data: value
            for key, value in config.items()
        }
        self._set_values(_config, **kwargs)

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
        _config: dict[
            str | ConnectionData,
            type | UnionType | ScalarType | type[Tensor[int | float | bool]],
        ] = {
            key if isinstance(key, str) else key.data: value
            for key, value in config.items()
        }
        self._set_types(_config, **kwargs)

    def set_outputs(self, *args: str | Connection, **kwargs: str | Connection) -> None:
        _args: list[str | ConnectionData] = [
            key if isinstance(key, str) else key.data for key in args
        ]
        _kwargs: dict[str, str | ConnectionData] = {
            key: value if isinstance(value, str) else value.data
            for key, value in kwargs.items()
        }
        self._set_outputs(*_args, **_kwargs)

    # TODO: Update summary, this should work same with both
    # Logical Model and Operator
    def summary(
        self,
        shapes: bool = True,
        types: bool = False,
        symbolic: bool = False,
        name: str | None = None,
        alternative_shapes: bool = False,
        uni_cache: dict[UniadicRecord, str] | None = None,
        var_cache: dict[Variadic, str] | None = None,
        depth: int = 0,
    ) -> None:
        if uni_cache is None:
            uni_cache = {}
        if var_cache is None:
            var_cache = {}

        type_info: dict[str, tuple[dict[str, str], dict[str, str]]] | None = None
        shape_info = None
        # extract relevant information about summary
        name_mappings = self.get_unique_submodel_names()

        # extract model topology
        conn_info = self.extract_connection_info(name_mappings)

        model_shapes = {
            sub_model_name: sub_model.get_shapes(
                uni_cache, var_cache, symbolic, alternative_shapes
            )
            for sub_model, sub_model_name in name_mappings.items()
        }
        if shapes:
            # extract model shapes
            shape_info = get_summary_shapes(model_shapes, conn_info)

        if types:
            # extract model types
            type_info = get_summary_types(name_mappings)

        # TODO: Remove name argument from summary method
        if not name and (name := self.name) is None:
            name = self.class_name

        # construct the table based on relevant information
        table = get_summary(
            conns=conn_info,
            name=name,
            shape=shape_info,  # type: ignore
            types=type_info,
        )

        table.compile()
        table.display()

        if depth > 0:
            for model, model_name in name_mappings.items():
                kwargs: dict[str, Any] = {
                    "depth": depth - 1,
                    "shapes": shapes,
                    "symbolic": symbolic,
                    "alternative_shapes": alternative_shapes,
                    "name": model_name,
                    "uni_cache": uni_cache,
                    "var_cache": var_cache,
                    "types": types,
                }
                if isinstance(model, Operator):
                    kwargs.pop("depth")
                model.summary(**kwargs)  # type: ignore


def define_unique_names(
    models: list[BaseModel] | KeysView[BaseModel],
) -> dict[BaseModel, str]:
    # TODO: Move this to Physical model (currently it is only used there)
    # TODO: Also add short-naming logic to this function
    model_name_dict = {}
    single_model_dict = {}
    model_count_dict: dict[str, int] = {}

    for model in models:
        class_name = model.name or model.class_name
        if model_count_dict.setdefault(class_name, 0) == 0:
            single_model_dict[class_name] = model
        else:
            single_model_dict.pop(class_name, None)
        model_name_dict[model] = (
            str(class_name) + "_" + str(model_count_dict[class_name])
        )
        model_count_dict[class_name] += 1

    for m in single_model_dict.values():
        model_name_dict[m] = m.name or str(m.class_name)
    return model_name_dict
