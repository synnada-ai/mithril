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

from collections.abc import KeysView, Mapping
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, Self

from ...common import find_dominant_type
from ...types import Dtype as CoreDtype
from ...utils.utils import constant_fn
from ..common import (
    NOT_GIVEN,
    TBD,
    IOHyperEdge,
    MainValueInstance,
    MainValueType,
    NullConnection,
    ShapeTemplateType,
    Tensor,
    UniadicRecord,
    Variadic,
    get_summary,
    get_summary_shapes,
    get_summary_types,
)

# from .base import BaseModel, ConnectionDataType
from .base import BaseModel, ConnectionData
from .operator import Operator
from .operators import (
    AbsoluteOp,
    AddOp,
    AtLeast1DOp,
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
    "ExtendInfo",
    "Model",
    "ConnectionType",
    "ConnectionInstanceType",
    "define_unique_names",
]


def create_extracted_model(
    connections: list[ConnectionType],
    model: type[Operator],
    defaults: dict[str, Any] | None = None,
) -> Connection:
    if defaults is None:
        defaults = {}
    code = model.__init__.__code__

    # "self" argument is common for all models, Exclude it by
    # starting co_varnames from 1st index.
    default_args = code.co_varnames[1 : code.co_argcount]
    default_args_dict = {key: TBD for key in default_args} | defaults
    default_args_dict.pop("name", None)

    op: Operator = model(**default_args_dict)

    # All connections' model field is [None] or [not None but a frozen model] ->
    #    -> Create a new model with extract = True
    # The must be maximum 1 connection's model field is not None and also not frozen
    # (All connections with a model field contains extract=True is okey)
    #    -> Add new models (both coming from connections and the maion Operation)
    #       directly into an already existing model.
    # There exists more than 1 connection's model field is not None and also not frozen
    #    -> Raise an error.

    # Find a main model and all new models to be added to main model
    main_model = None
    new_models = []
    for c in connections:
        if isinstance(c, str):
            raise ValueError("Connection key is not allowed in connections!")
        # TODO: check if _get_outermost_parent contains c.metadata
        if isinstance(c, ConnectionData) and c.model is not None:
            m = c.model._get_outermost_parent()
            if not m.is_frozen and not m.extract:
                if main_model is not None and main_model is not m:
                    raise ValueError(
                        "Multiple non-frozen active models found in connections!"
                    )
                main_model = m
            elif m not in new_models:
                new_models.append(m)

    # Select a main model from new models if main_model is None
    if main_model is None:
        for m in new_models:
            if not m.is_frozen:
                main_model = m
                break
        if main_model is not None:
            new_models.remove(main_model)

    # Create a main model if main_model remains None.
    # Create "ExtractModel" which means a model whose submodels will be
    # extracted during an extension instead of ExtractModel itself.
    if main_model is None:
        main_model = Model()
        main_model.extract = True

    assert isinstance(main_model, Model)
    # Extend main model with all the new models to be added.
    for m in new_models:
        updates = main_model.constraint_solver.match(m.constraint_solver)
        main_model.constraint_solver(updates)
        if m.extract:
            assert isinstance(m, Model)
            main_model.extend_extracted_model(m)
        else:
            main_model._extend(m)
    keys = {key: con for key, con in zip(op.input_keys, connections, strict=False)}
    # Extend main_model with given Operator.
    main_model._extend(op, keys)
    output = main_model.conns.get_extracted_connection(op.cout)
    assert isinstance(output, Connection)
    return output


class Connection(ConnectionData):
    def __hash__(self) -> int:
        return hash(id(self))

    def __getitem__(
        self,
        key: slice
        | int
        | EllipsisType
        | tuple[slice | int | None | EllipsisType | Connection, ...]
        | IOKey
        | Connection
        | None,
    ) -> Connection:
        match key:
            case slice():
                slice_output = create_extracted_model(
                    connections=[key.start, key.stop, key.step], model=SliceOp
                )
                output = create_extracted_model(
                    connections=[self, slice_output], model=IndexerOp
                )
            case int() | EllipsisType() | None:
                output = create_extracted_model(
                    connections=[self, key], model=IndexerOp
                )
            case tuple():
                connections: list[Connection | int | None | EllipsisType] = []
                for item in key:
                    if isinstance(item, slice):
                        slice_output = create_extracted_model(
                            connections=[item.start, item.stop, item.step],
                            model=SliceOp,
                        )
                        connections.append(slice_output)
                    else:
                        connections.append(item)
                tuple_template = create_extracted_model(
                    connections=connections,  # type: ignore
                    model=ToTupleOp,
                    defaults={"n": len(key)},
                )
                output = create_extracted_model(
                    connections=[self, tuple_template], model=IndexerOp
                )
        return output

    def __add__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=AddOp)

    def __radd__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=AddOp)

    def __sub__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=SubtractOp)

    def __rsub__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=SubtractOp)

    def __mul__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=MultiplyOp)

    def __rmul__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=MultiplyOp)

    def __truediv__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=DivideOp)

    def __rtruediv__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=DivideOp)

    def __floordiv__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=FloorDivideOp)

    def __rfloordiv__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=FloorDivideOp)

    def __pow__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(
            connections=[self, other], model=PowerOp, defaults={"robust": False}
        )

    def __rpow__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(
            connections=[other, self], model=PowerOp, defaults={"robust": False}
        )

    def __matmul__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=MatrixMultiplyOp)

    def __gt__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=GreaterOp)

    def __rgt__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=GreaterOp)

    def __ge__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=GreaterEqualOp)

    def __rge__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=GreaterEqualOp)

    def __lt__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=LessOp)

    def __rlt__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=LessOp)

    def __le__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=LessEqualOp)

    def __rle__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=LessEqualOp)

    def eq(self, other: object) -> Connection:
        if isinstance(
            other, int | float | bool | list | Connection | IOKey | tuple | Tensor
        ):
            return create_extracted_model(connections=[self, other], model=EqualOp)
        else:
            raise ValueError("Unsupported type for equality operation.")

    def ne(self, other: object) -> Connection:
        if isinstance(
            other, int | float | bool | list | Connection | IOKey | tuple | Tensor
        ):
            return create_extracted_model(connections=[self, other], model=NotEqualOp)
        else:
            raise ValueError("Unsupported type for equality operation.")

    def __and__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=LogicalAndOp)

    def __rand__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=LogicalAndOp)

    def __or__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=LogicalOrOp)

    def __ror__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=LogicalOrOp)

    def __xor__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=LogicalXOrOp)

    def __rxor__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=LogicalXOrOp)

    def __lshift__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=ShiftLeftOp)

    def __rlshift__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=ShiftLeftOp)

    def __rshift__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[self, other], model=ShiftRightOp)

    def __rrshift__(self, other: ConnectionType) -> Connection:
        return create_extracted_model(connections=[other, self], model=ShiftRightOp)

    def __invert__(self) -> Connection:
        return create_extracted_model(connections=[self], model=LogicalNotOp)

    def __neg__(self) -> Connection:
        return create_extracted_model(connections=[self], model=MinusOp)

    def abs(self) -> Connection:
        return create_extracted_model(connections=[self], model=AbsoluteOp)

    def len(self) -> Connection:
        return create_extracted_model(connections=[self], model=LengthOp)

    @property
    def shape(self) -> Connection:
        return create_extracted_model(connections=[self], model=ShapeOp)

    def reshape(self, shape: tuple[int | Connection, ...] | Connection) -> Connection:
        return create_extracted_model(connections=[self, shape], model=ReshapeOp)

    def size(self, dim: int | tuple[int, ...] | Connection | None = None) -> Connection:
        return create_extracted_model(connections=[self, dim], model=SizeOp)

    def tensor(self) -> Connection:
        return create_extracted_model(
            connections=[self], model=ToTensorOp, defaults={"dtype": None}
        )

    def mean(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
    ) -> Connection:
        return create_extracted_model(connections=[self, axis, keepdim], model=MeanOp)

    def sum(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
    ) -> Connection:
        return create_extracted_model(connections=[self, axis, keepdim], model=SumOp)

    def max(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
    ) -> Connection:
        return create_extracted_model(connections=[self, axis, keepdim], model=MaxOp)

    def min(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
    ) -> Connection:
        return create_extracted_model(connections=[self, axis, keepdim], model=MinOp)

    def prod(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
    ) -> Connection:
        return create_extracted_model(connections=[self, axis, keepdim], model=ProdOp)

    def var(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
        correction: float | None = 0.0,
    ) -> Connection:
        return create_extracted_model(
            connections=[self, axis, keepdim, correction], model=VarianceOp
        )

    def sqrt(self) -> Connection:
        return create_extracted_model(
            connections=[self], model=SqrtOp, defaults={"robust": False}
        )

    def exp(self) -> Connection:
        return create_extracted_model(connections=[self], model=ExponentialOp)

    def transpose(self, axes: tuple[int, ...] | Connection | None = None) -> Connection:
        return create_extracted_model(connections=[self, axes], model=TransposeOp)

    def split(self, split_size: int, axis: int) -> Connection:
        return create_extracted_model(
            connections=[self, split_size, axis], model=SplitOp
        )

    def item(self) -> Connection:
        return create_extracted_model(connections=[self], model=ItemOp)

    def cast(self, dtype: Connection | CoreDtype | None = None) -> Connection:
        return create_extracted_model(connections=[self, dtype], model=CastOp)

    def dtype(self) -> Connection:
        return create_extracted_model(connections=[self], model=DtypeOp)

    def sin(self) -> Connection:
        return create_extracted_model(connections=[self], model=SineOp)

    def cos(self) -> Connection:
        return create_extracted_model(connections=[self], model=CosineOp)

    def atleast_1d(self) -> Connection:
        return create_extracted_model(connections=[self], model=AtLeast1DOp)


IOKey = Connection


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


ConnectionType = (
    str
    | MainValueType
    | NullConnection
    | IOKey
    | ConnectionData
    | Tensor[int | float | bool]
)

ConnectionInstanceType = (
    str | MainValueInstance | NullConnection | IOKey | Connection | Tensor  # type: ignore
)


class Model(BaseModel):
    def __call__(self, **kwargs: ConnectionType) -> ExtendInfo:
        return ExtendInfo(self, kwargs)

    def _create_connection(
        self, metadata: IOHyperEdge, key: str | None = None
    ) -> Connection:
        connection = Connection(key)
        connection.metadata = metadata
        return connection

    def _get_conn_data(self, conn: str | ConnectionData) -> ConnectionData:
        if isinstance(conn, str):
            _conn = self.conns.get_connection(conn)
        else:
            _conn = self.conns.get_con_by_metadata(conn.metadata)
        assert isinstance(_conn, ConnectionData)
        return _conn

    def rename_key(self, connection: ConnectionData, key: str) -> None:
        super().rename_key(connection, key)
        conn = self.conns.get_extracted_connection(connection)
        setattr(self, key, conn)

    def _unroll_template(self, template: ConnectionType) -> ConnectionType:
        types = [ConnectionData, Connection, Connection, IOKey, Tensor]
        # Add tuple / list models if template is a tuple / list.
        if (
            isinstance(template, tuple | list)
            and find_dominant_type(template, False, constant_fn) in types
        ):
            _model: type[Operator] = (ToListOp, ToTupleOp)[isinstance(template, tuple)]
            conns = [self._unroll_template(item) for item in template]
            template = create_extracted_model(conns, _model, {"n": len(template)})
        # Extend model with submodels of ExtractModel if connection's model
        # has extract=True.
        if (
            isinstance(template, ConnectionData)
            and template.model is not None
            and (extract_m := template.model).extract
        ):
            assert isinstance(extract_m, Model)
            self.extend_extracted_model(extract_m)
        return template

    def extend_extracted_model(self, model: Model) -> None:
        # Extend model with submodels of ExtractModel.
        # Match and update constraint solver of the model.
        updates = self.constraint_solver.match(model.constraint_solver)
        self.constraint_solver(updates)
        for sub_m in model.get_models_in_topological_order():
            if sub_m not in self.dag:
                sub_m.parent = None
                conns: dict[str, ConnectionData] = {}
                # Update all connections of the submodel.
                for con in sub_m.conns.all.values():
                    if (
                        not model.conns.get_con_by_metadata(
                            con.metadata
                        ).is_autogenerated  # type: ignore
                        and self.conns.get_con_by_metadata(con.metadata) is None
                    ):
                        _con = model.conns.get_con_by_metadata(con.metadata)
                        assert _con is not None
                        _con.model = None
                        conns[con.key] = _con
                    # Update connections_dict and metadata_dict of Connections class.
                    sub_m.conns.connections_dict.setdefault(con.metadata, set())
                    sub_m.conns.connections_dict[con.metadata] |= (
                        model.conns.connections_dict[con.metadata]
                    )

                    sub_m.conns.metadata_dict.setdefault(con.metadata, set())
                    sub_m.conns.metadata_dict[con.metadata] |= (
                        model.conns.metadata_dict[con.metadata]
                    )
                # Extend the model with submodel.
                self.extend(sub_m, **conns)

    @property
    def cout(self) -> Connection:
        cout = super().cout
        assert isinstance(cout, Connection)
        return cout

    @property
    def cin(self) -> Connection:
        cin = super().cin
        assert isinstance(cin, Connection)
        return cin

    def _extend(
        self,
        model: BaseModel,
        kwargs: dict[str, ConnectionType] | None = None,
        trace: bool = True,
    ) -> Self:
        if kwargs is None:
            kwargs = {}
        if self.is_frozen:
            raise AttributeError("Model is frozen and can not be extended!")

        for key, value in kwargs.items():
            _value = value.get_key() if isinstance(value, ConnectionData) else value

            if isinstance(_value, str) and _value == "":
                if key in model.input_keys:
                    _value = NOT_GIVEN
                else:
                    raise KeyError(
                        "Empty string is not a valid for output connections!"
                    )

                if isinstance(value, IOKey):
                    value.set_key(None)
                else:
                    kwargs[key] = _value
            kwargs[key] = self._unroll_template(kwargs[key])

        self.extend(model, trace, **kwargs)
        return self

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
            if len(available_cin) != 1:
                raise KeyError(
                    "Submodel must have single available canonical input! "
                    "Set canonical input or use |= operator."
                )
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


#                       Flowchart for Canonical Logic
# +-----------------------------------------------------------------------------------+
# +------------+           +------------+
# |     |=     |           |     +=     |
# +------------+           +------------+
#        |                       |
#        |                       v
#        |             +-------------------------------------+
#        |             |  Check Parent Model has single cout |
#        |             +-------------------------------------+
#        |                       |  (Valid)
#        |                       v
#        |             +---------------------------------+
#        |             |  Check Child Model has single   |
#        |             |  available cin                  |
#        |             +---------------------------------+
#        |                       |  (Valid)
#        |                       v
#        |       +--------------------------------------------+
#        |       |   Add connection info:                     |
#        |       |    Parent Model (cout) -> Child Model (cin)|
#        |       +--------------------------------------------+
#        |                       |
#        v                       v
#    +---------------------------------------+
#    |  Extend Model with given connections  |
#    +---------------------------------------+
#                                |
#                                |
#                                v
#                  (Iterate over all connections)
# +------------------------------------------------------------------------------+
#                                |
#                                v
#      +----------------------------------------------------+
#      |  Update Canonical state of the Connection         |
#      +----------------------------------------------------+
#                                |
#                                v
# +-----------------------------------------------------------------------------+
# | Update Canonical state of the Connection                                    |
# |                                                                             |
# |       +------------------------------------------+                          |
# |       | Child / parent model cin stays as input? |                          |
# |       +------------------------------------------+                          |
# |          | Yes                | No                                          |
# |          v                    v                                             |
# |  +--------------------------+  +------------------------------------+       |
# |  | Add connection to Parent |  | Discard connection from Parent     |       |
# |  | Model's cin set          |  | Model's cin set                    |       |
# |  +--------------------------+  +------------------------------------+       |
# |                                           |                                 |
# |                                           v                                 |
# |                        +--------------------------------------------+       |
# |                        | Child / Parent Model cout stays as output? |       |
# |                        +--------------------------------------------+       |
# |                           | Yes                | No                         |
# |                           v                    v                            |
# |  +----------------------- --+  +------------------------------------+       |
# |  | Add connection to Parent |  | Discard connection from Parent     |       |
# |  |  Model's cout set        |  | Model's cout set                   |       |
# |  +--------------------------+  +------------------------------------+       |
# +------------------------------------------------------------------------------+
# +-----------------------------------------------------------------------------------+
