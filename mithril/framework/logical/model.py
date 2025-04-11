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
from types import EllipsisType
from typing import Any, Self

from ...common import contains_given_type
from ...types import Dtype as CoreDtype
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
    VariableSequenceType,
    Variadic,
    get_summary,
    get_summary_shapes,
    get_summary_types,
)
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
    MultiplyOp,
    NegateOp,
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


class Connection(ConnectionData):
    def __hash__(self) -> int:
        return hash(id(self))

    def __getitem__(
        self,
        key: slice
        | int
        | EllipsisType
        | tuple[
            slice
            | int
            | None
            | EllipsisType
            | Connection
            | VariableSequenceType[int]
            | Tensor[int],
            ...,
        ]
        | Connection
        | Tensor[int]
        | VariableSequenceType[int]
        | None,
    ) -> Connection:
        match key:
            case slice():
                slice_output = create_provisional_model(
                    connections=[key.start, key.stop, key.step], model=SliceOp
                )
                output = create_provisional_model(
                    connections=[self, slice_output], model=IndexerOp
                )

            case tuple():
                connections: list[
                    Connection
                    | int
                    | None
                    | EllipsisType
                    | VariableSequenceType[int]
                    | IOKey
                    | Tensor[int]
                ] = []
                for item in key:
                    if isinstance(item, slice):
                        slice_output = create_provisional_model(
                            connections=[item.start, item.stop, item.step],
                            model=SliceOp,
                        )
                        connections.append(slice_output)
                    else:
                        connections.append(item)
                tuple_template = create_provisional_model(
                    connections=connections,  # type: ignore
                    model=ToTupleOp,
                    defaults={"n": len(key)},
                )
                output = create_provisional_model(
                    connections=[self, tuple_template], model=IndexerOp
                )

            case int() | EllipsisType() | None | Tensor() | Sequence():
                output = create_provisional_model(
                    connections=[self, key],  # type: ignore
                    model=IndexerOp,
                )
        return output

    def __add__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=AddOp)

    def __radd__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=AddOp)

    def __sub__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=SubtractOp)

    def __rsub__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=SubtractOp)

    def __mul__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=MultiplyOp)

    def __rmul__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=MultiplyOp)

    def __truediv__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=DivideOp)

    def __rtruediv__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=DivideOp)

    def __floordiv__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=FloorDivideOp)

    def __rfloordiv__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=FloorDivideOp)

    def __pow__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(
            connections=[self, other], model=PowerOp, defaults={"robust": False}
        )

    def __rpow__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(
            connections=[other, self], model=PowerOp, defaults={"robust": False}
        )

    def __matmul__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(
            connections=[self, other], model=MatrixMultiplyOp
        )

    def __gt__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=GreaterOp)

    def __rgt__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=GreaterOp)

    def __ge__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=GreaterEqualOp)

    def __rge__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=GreaterEqualOp)

    def __lt__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=LessOp)

    def __rlt__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=LessOp)

    def __le__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=LessEqualOp)

    def __rle__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=LessEqualOp)

    def eq(self, other: object) -> Connection:
        if isinstance(
            other, int | float | bool | list | Connection | IOKey | tuple | Tensor
        ):
            return create_provisional_model(connections=[self, other], model=EqualOp)
        else:
            raise ValueError("Unsupported type for equality operation.")

    def ne(self, other: object) -> Connection:
        if isinstance(
            other, int | float | bool | list | Connection | IOKey | tuple | Tensor
        ):
            return create_provisional_model(connections=[self, other], model=NotEqualOp)
        else:
            raise ValueError("Unsupported type for equality operation.")

    def __and__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=LogicalAndOp)

    def __rand__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=LogicalAndOp)

    def __or__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=LogicalOrOp)

    def __ror__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=LogicalOrOp)

    def __xor__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=LogicalXOrOp)

    def __rxor__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=LogicalXOrOp)

    def __lshift__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=ShiftLeftOp)

    def __rlshift__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=ShiftLeftOp)

    def __rshift__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[self, other], model=ShiftRightOp)

    def __rrshift__(self, other: TemplateConnectionType) -> Connection:
        return create_provisional_model(connections=[other, self], model=ShiftRightOp)

    def __invert__(self) -> Connection:
        return create_provisional_model(connections=[self], model=LogicalNotOp)

    def __neg__(self) -> Connection:
        return create_provisional_model(connections=[self], model=NegateOp)

    def abs(self) -> Connection:
        return create_provisional_model(connections=[self], model=AbsoluteOp)

    def len(self) -> Connection:
        return create_provisional_model(connections=[self], model=LengthOp)

    @property
    def shape(self) -> Connection:
        return create_provisional_model(connections=[self], model=ShapeOp)

    def reshape(self, shape: tuple[int | Connection, ...] | Connection) -> Connection:
        return create_provisional_model(connections=[self, shape], model=ReshapeOp)

    def size(self, dim: int | tuple[int, ...] | Connection | None = None) -> Connection:
        return create_provisional_model(connections=[self, dim], model=SizeOp)

    def tensor(self) -> Connection:
        return create_provisional_model(
            connections=[self], model=ToTensorOp, defaults={"dtype": None}
        )

    def mean(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
    ) -> Connection:
        return create_provisional_model(connections=[self, axis, keepdim], model=MeanOp)

    def sum(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
    ) -> Connection:
        return create_provisional_model(connections=[self, axis, keepdim], model=SumOp)

    def max(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
    ) -> Connection:
        return create_provisional_model(connections=[self, axis, keepdim], model=MaxOp)

    def min(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
    ) -> Connection:
        return create_provisional_model(connections=[self, axis, keepdim], model=MinOp)

    def prod(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
    ) -> Connection:
        return create_provisional_model(connections=[self, axis, keepdim], model=ProdOp)

    def var(
        self,
        axis: int | tuple[int, ...] | Connection | None = None,
        keepdim: bool = False,
        correction: float | None = 0.0,
    ) -> Connection:
        return create_provisional_model(
            connections=[self, axis, keepdim, correction], model=VarianceOp
        )

    def sqrt(self) -> Connection:
        return create_provisional_model(
            connections=[self], model=SqrtOp, defaults={"robust": False}
        )

    def exp(self) -> Connection:
        return create_provisional_model(connections=[self], model=ExponentialOp)

    def transpose(self, axes: tuple[int, ...] | Connection | None = None) -> Connection:
        return create_provisional_model(connections=[self, axes], model=TransposeOp)

    @property
    def T(self) -> Connection:  # noqa: N802
        return create_provisional_model(
            connections=[self], model=TransposeOp, defaults={"axes": None}
        )

    def split(self, split_size: int, axis: int) -> Connection:
        return create_provisional_model(
            connections=[self, split_size, axis], model=SplitOp
        )

    def item(self) -> Connection:
        return create_provisional_model(connections=[self], model=ItemOp)

    def cast(self, dtype: Connection | CoreDtype | None = None) -> Connection:
        return create_provisional_model(connections=[self, dtype], model=CastOp)

    def dtype(self) -> Connection:
        return create_provisional_model(connections=[self], model=DtypeOp)

    def sin(self) -> Connection:
        return create_provisional_model(connections=[self], model=SineOp)

    def cos(self) -> Connection:
        return create_provisional_model(connections=[self], model=CosineOp)

    def atleast_1d(self) -> Connection:
        return create_provisional_model(connections=[self], model=AtLeast1DOp)


IOKey = Connection


@dataclass
class ExtendInfo:
    _model: BaseModel
    _connections: Mapping[
        str, ConnectionType | MainValueType | Tensor[int | float | bool]
    ]

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
    def connections(
        self,
    ) -> Mapping[str, ConnectionType | MainValueType | Tensor[int | float | bool]]:
        return self._connections


ConnectionType = str | NullConnection | IOKey | ConnectionData
TemplateConnectionType = (
    Connection
    | int
    | float
    | list[int | float]
    | EllipsisType
    | tuple[slice | int | None | EllipsisType | Connection, ...]
    | None
    | Tensor[int | float | bool]
    | VariableSequenceType[int]
)

ConnectionInstanceType = (
    str | MainValueInstance | NullConnection | IOKey | Connection | Tensor  # type: ignore
)

UnrollTriggerTypes = ConnectionData | Connection | IOKey | Tensor  # type: ignore


class Model(BaseModel):
    def __call__(
        self, **kwargs: ConnectionType | MainValueType | Tensor[int | float | bool]
    ) -> ExtendInfo:
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

    def _bind_provisional_model(self, provisional_model: BaseModel) -> None:
        """
        Binds a provisional model to the main model by synchronizing their
        configurations and combining their constraint solvers.

        This method is called when a provisional model needs to be integrated
        with the main model. It performs the following operations:
            - Sets the provisional model's source to the main model.
            - Transfers main model settings, such as just-in-time enforcement,
                to the provisional model.
            - Matches and updates the constraint solvers from both models,
                ensuring that the provisional model's constraint solver is
                updated with the main model's matched solver.

        Parameters:
                provisional_model (BaseModel): The provisional model that
                is to be bound to the main model.
        """
        provisional_model.provisional_source = self
        self.provisional_model = provisional_model
        provisional_model.enforce_jit = self.enforce_jit
        updates = self.constraint_solver.match(provisional_model.constraint_solver)
        self.constraint_solver(updates)
        provisional_model.constraint_solver = self.constraint_solver

    def _extend_op_model(
        self,
        connections: list[TemplateConnectionType | ConnectionData],
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

        keys = {key: con for key, con in zip(op.input_keys, connections, strict=False)}
        # Extend main_model with given Operator.
        self._extend(op, keys)
        output = self.conns.get_extracted_connection(op.cout)
        assert isinstance(output, Connection)
        return output

    def _unroll_template(
        self, template: TemplateConnectionType | ConnectionData
    ) -> TemplateConnectionType | ConnectionType:
        if isinstance(template, tuple | list) and contains_given_type(
            template, UnrollTriggerTypes
        ):
            _model: type[Operator] = (ToListOp, ToTupleOp)[isinstance(template, tuple)]

            conns = [self._unroll_template(item) for item in template]  # type: ignore
            template = self._extend_op_model(conns, _model, {"n": len(template)})  # type: ignore
        # If template is a connection and its model is provisional,
        # extend self with submodels of provisional model.
        elif (
            isinstance(template, ConnectionData)
            and template.model is not None
            and (extract_m := template.model).provisional_source
            and extract_m is not self
        ):
            assert isinstance(extract_m, Model)
            p_model = extract_m.provisional_source
            if (
                isinstance(p_model, BaseModel)
                and self is not p_model._get_outermost_parent()
            ):
                raise ValueError(
                    "Provisional source model is not the same as the current model!"
                )
            self.extend_extracted_model(extract_m, template)
            if (_tmp := self.conns.get_con_by_metadata(template.metadata)) is not None:
                template = _tmp
        return template

    def extend_extracted_model(self, model: Model, start_con: ConnectionData) -> None:
        """
        Extends the main (parent) model by extracting and incorporating submodels from
        a provisional child model based on a provided connection.

        This method is invoked during the extension process when a model extends a
        parent model and the given connections contain provisional models.
        The workflow is as follows:
        1. Checks whether provisional submodels should be used by assessing the
            existence of a provisional model.
        2. Identifies the starting connection in the given model by matching the
            metadata of the provided start connection.
        3. Determines the dependent submodels by tracing from the start node using
            the local dependency map, arranging them in topological order.
        4. Extends the parent model by appending the obtained submodels through
            the _extend_with_submodels method, and removes these submodels from
            the child model's DAG.
        5. Merges any remaining provisional submodels from the child model into
            the parent's provisional model, and cleans up provisional references
            to ensure no stale or empty provisional models remain.

        Args:
             model (Model): The provisional model from which the required
                submodels are extracted.
             start_con (ConnectionData): The connection data used as the
                starting point to identify dependent submodels.

        Returns:
             None
        """
        # Extend model with submodels of provisional Model.
        use_sub_provisional = False
        if self.provisional_model is None:
            use_sub_provisional = True
            self._bind_provisional_model(model)

        con = model.conns.get_con_by_metadata(start_con.metadata)
        submodels = []
        assert con is not None
        start_m = model.dependency_map.local_output_dependency_map.get(con)
        if start_m is None:
            con.model = None
        else:
            submodels = model.get_models_in_topological_order(start_m[0])
            self._extend_with_submodels(model, submodels)
            # Remove submodels from model.dag
            for m in submodels:
                model.dag.pop(m, None)

        # Merge provisional models
        if (
            isinstance(model.provisional_source, BaseModel)
            and not use_sub_provisional
            and self.provisional_model is not model
        ):
            submodels = [
                sub_m
                for sub_m in model.dag
                if sub_m not in submodels and sub_m not in self.dag
            ]
            assert isinstance(self.provisional_model, BaseModel)
            self.provisional_model._extend_with_submodels(
                model, submodels, replicate=False
            )
            if isinstance(source := model.provisional_source, BaseModel):
                source.provisional_model = None
            model.provisional_source = True
        # TODO: while removing the child model's provisional model,
        # make sure that all objects are deleted and no references are left.
        # Then remove the provisional model from the child model.

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
        kwargs: Mapping[
            str, ConnectionType | MainValueType | Tensor[int | float | bool]
        ]
        | None = None,
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
                    kwargs[key] = _value  # type: ignore
            kwargs[key] = self._unroll_template(kwargs[key])  # type: ignore

        mp = model.provisional_model
        if mp is not None and self.provisional_model is None:
            mp = model.provisional_model
            provisional_model = Model()
            self._bind_provisional_model(provisional_model)

        # Merge provisional models
        if mp is not None and mp is not self.provisional_model:
            submodels = [sub_m for sub_m in mp.dag if sub_m not in self.dag]
            assert isinstance(self.provisional_model, BaseModel)
            self.provisional_model._extend_with_submodels(mp, submodels, False)
            if isinstance(source := mp.provisional_source, BaseModel):
                source.provisional_model = None
            mp.provisional_source = False

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
            kwargs[next(iter(available_cin))] = self.cout  # type: ignore
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


def create_provisional_model(
    connections: list[TemplateConnectionType],
    model: type[Operator],
    defaults: dict[str, Any] | None = None,
) -> Connection:
    """
    Create a provisional model for connection-based operations (e.g. +, abs(), etc.).
    If any connection contains an associated model, that is considered the main model.
    If there exits a main model the provisional model is linked with the main model by:
      - Setting the provisional model as the main_model's provisional_model field.
      - Setting the main model as the provisional model's provisional_source field.
    When the actual extend operation is performed, only the corresponding submodels
    (tracked by topological order) are extracted from the provisional model using the
    _extend_with_submodels method.
    """
    # Find a main model and all new models to be added to main model
    provisional_model: BaseModel | None = None
    main_model: Model | None = None
    new_models: list[BaseModel] = []
    for c in connections:
        if isinstance(c, str):
            raise ValueError(
                "Strings are not allowed to be used in Connection Operations!"
            )
        if isinstance(c, ConnectionData) and c.model is not None:
            m = c.model
            if isinstance(m.provisional_source, BaseModel):
                m = m.provisional_source
            m = m._get_outermost_parent()

            if not m.is_frozen and m.provisional_source is False:
                if main_model is not None and main_model is not m:
                    raise ValueError(
                        "Multiple non-frozen active models found in connections!"
                    )
                assert isinstance(m, Model)
                main_model = m
            elif m not in new_models:
                if provisional_model is None and m.provisional_source is not False:
                    provisional_model = m
                new_models.append(m)

    if main_model is not None and main_model.provisional_model:
        # If main_model has provisional, then set it as provisional_model.
        provisional_model = main_model.provisional_model

    if provisional_model is None:
        provisional_model = Model()
        if main_model is not None:
            # If main_model is not None, then set created provisional to main_model.
            main_model._bind_provisional_model(provisional_model)
        else:
            provisional_model.provisional_source = True

    assert isinstance(provisional_model, Model)
    _connections: list[TemplateConnectionType | ConnectionData] = []
    # Iterate over connections, if connection is coming from main model, create
    # a new connection with same edge, otherwise use connections as is.
    for c in connections:
        if (
            isinstance(c, ConnectionData)
            and c.model is not None
            and c.model._get_outermost_parent() is main_model
        ):
            _c = main_model.conns.get_con_by_metadata(c.metadata)
            assert isinstance(_c, ConnectionData)
            con = provisional_model.conns.get_con_by_metadata(_c.metadata)
            if con is None:
                con = _c._replicate()
            _connections.append(con)
        else:
            _connections.append(c)

    # Merge provisional models
    for m in new_models:
        # Add all new models to provisional_model.
        if m is provisional_model:
            continue
        if m.provisional_source:
            updates = provisional_model.constraint_solver.match(m.constraint_solver)
            provisional_model.constraint_solver(updates)
            assert isinstance(m, Model)
            provisional_model._extend_with_submodels(m, replicate=False)
        else:
            provisional_model._extend(m)
    return provisional_model._extend_op_model(_connections, model, defaults)


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
