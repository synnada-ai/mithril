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

from collections.abc import Mapping, Sequence
from types import EllipsisType, NoneType, UnionType
from typing import Any

from ... import core
from ..common import (
    NOT_GIVEN,
    TBD,
    BaseKey,
    Connection,
    ConnectionType,
    Constraint,
    ScalarValueType,
    ShapeTemplateType,
    Tensor,
    TensorToListType,
    TensorValueType,
    ToBeDetermined,
)
from ..constraints import (
    bcast,
    bcast_error_check,
    bcast_mat_mul_check,
    bcast_matrix_mult,
    buffer_constraint,
    divide_type_constraint,
    edge_type_constraint,
    floor_divide_type_constraint,
    general_tensor_type_constraint,
    indexer_constraints,
    indexer_initial_type_constraint,
    indexer_type_constraint,
    item_constraints,
    reduce_constraints,
    reduce_type_constraint,
    relational_operator_type_constraint,
    reshape_constraints,
    reverse_constraints,
    shape_constraints,
    size_constraints,
    slice_constraints,
    split_constraints,
    tensor_to_list_constraints,
    tensor_to_list_type_constraint,
    to_list_constraints,
    to_tensor_constraints,
    to_tuple_constraints,
)
from .base import ExtendInfo
from .primitive import PrimitiveModel

__all__ = [
    "PrimitiveModel",
    "Buffer",
    "ToTuple",
    "Power",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "FloorDivide",
    "Minus",
    "MatrixMultiply",
    "Shape",
    "Reshape",
    "Length",
    "Size",
    "Exponential",
    "Item",
    "Indexer",
    "ToTensor",
    "ToList",
    "TensorToList",
    "Mean",
    "Sum",
    "Max",
    "Min",
    "Prod",
    "Variance",
    "Absolute",
    "Equal",
    "NotEqual",
    "Greater",
    "GreaterEqual",
    "Less",
    "LessEqual",
    "LogicalNot",
    "LogicalOr",
    "LogicalAnd",
    "LogicalXOr",
    "ShiftLeft",
    "ShiftRight",
    "ArgMax",
    "ArgMin",
    "Cast",
    "Transpose",
    "Sqrt",
    "Split",
    "Slice",
    "Dtype",
    "Sine",
    "Cosine",
    "Minimum",
    "Maximum",
]

ConstantType = float | int | core.Constant


class Buffer(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="buffer",
            name=name,
            output=BaseKey(),
            input=BaseKey(value=input),
        )

        self._add_constraint(
            fn=buffer_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class ToTuple(PrimitiveModel):
    def __init__(
        self,
        n: int,
        *,
        name: str | None = None,
        **kwargs: Tensor[Any] | ScalarValueType | ToBeDetermined,
    ) -> None:
        self.factory_args = {"n": n}
        key_definitions = {
            "output": BaseKey(
                type=tuple[
                    int | float | bool | list | tuple | slice | EllipsisType | None, ...  # type: ignore
                ]
            )
        }
        key_definitions |= {
            f"input{idx+1}": BaseKey(
                type=int | float | bool | list | tuple | slice | EllipsisType | None,
                value=kwargs.get(f"input{idx+1}", TBD),
            )
            for idx in range(n)
        }

        super().__init__(formula_key="to_tuple", name=name, **key_definitions)
        self._add_constraint(
            fn=to_tuple_constraints,
            keys=[PrimitiveModel.output_key] + [key for key in self.input_keys],
        )


class ArithmeticOperation(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(
        self,
        formula_key: str,
        left: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key=formula_key,
            name=name,
            output=BaseKey(),
            left=BaseKey(value=left),
            right=BaseKey(value=right),
        )

        edge_constraint = self._add_constraint(
            fn=edge_type_constraint,
            keys=[PrimitiveModel.output_key, "left", "right"],
        )

        self._add_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "left", "right"],
            dependencies={edge_constraint},
        )

        bcast_constraint = self._add_constraint(
            fn=bcast,
            keys=[PrimitiveModel.output_key, "left", "right"],
            dependencies={edge_constraint},
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[PrimitiveModel.output_key, "left", "right"],
            dependencies={bcast_constraint},
        )

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, output=output)


class Power(PrimitiveModel):
    base: Connection
    exponent: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        base: Tensor[Any] | int | float | ToBeDetermined = TBD,
        exponent: Tensor[Any] | int | float | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.robust = robust
        self.factory_args = {"robust": robust}
        assert isinstance(robust, bool), "Robust must be a boolean value!"

        if robust:
            super().__init__(
                formula_key="robust_power",
                name=name,
                output=BaseKey(shape=[("out", ...)], type=Tensor),
                base=BaseKey(shape=[("base", ...)], type=Tensor, value=base),
                exponent=BaseKey(shape=[("exp", ...)], type=Tensor, value=exponent),
                threshold=BaseKey(shape=[], type=Tensor),
            )

            constrs: set[Constraint] = set()

        else:
            super().__init__(
                formula_key="power",
                name=name,
                output=BaseKey(),
                base=BaseKey(value=base),
                exponent=BaseKey(value=exponent),
            )
            edge_constraint = self._add_constraint(
                fn=edge_type_constraint,
                keys=[PrimitiveModel.output_key, "base", "exponent"],
            )
            constrs = {edge_constraint}

        self._add_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "base", "exponent"],
            dependencies=constrs,
        )

        bcast_constraint = self._add_constraint(
            fn=bcast,
            keys=[PrimitiveModel.output_key, "base", "exponent"],
            dependencies=constrs,
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[PrimitiveModel.output_key, "base", "exponent"],
            dependencies={bcast_constraint},
        )

    def __call__(  # type: ignore[override]
        self,
        base: ConnectionType = NOT_GIVEN,
        exponent: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        name: str | None = None,
        threshold: ConnectionType = core.Constant.MIN_POSITIVE_NORMAL,
    ) -> ExtendInfo:
        kwargs = {"base": base, "exponent": exponent, "output": output}
        default = (
            isinstance(threshold, core.Constant)
            and threshold == core.Constant.MIN_POSITIVE_NORMAL
        )
        if self.robust:
            # NOTE: Since we can not provide Tensor objects as default
            # arguments, we need to convert default value.
            if default:
                threshold = Tensor(threshold)  # type: ignore
            kwargs["threshold"] = threshold
        elif not default:
            raise ValueError("Threshold cannot be specified when robust mode is off")

        return super().__call__(**kwargs)


class Add(ArithmeticOperation):
    def __init__(
        self,
        left: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="add", name=name, left=left, right=right)


class Subtract(ArithmeticOperation):
    def __init__(
        self,
        left: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="subtract", name=name, left=left, right=right)


class Multiply(ArithmeticOperation):
    def __init__(
        self,
        left: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="multiplication", name=name, left=left, right=right
        )


class Minimum(ArithmeticOperation):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="minimum", name=name, left=left, right=right)


class Maximum(ArithmeticOperation):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="maximum", name=name, left=left, right=right)


class Divide(PrimitiveModel):
    numerator: Connection
    denominator: Connection
    output: Connection

    def __init__(
        self,
        numerator: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        denominator: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="divide",
            name=name,
            output=BaseKey(),
            numerator=BaseKey(value=numerator),
            denominator=BaseKey(value=denominator),
        )
        edge_constraint = self._add_constraint(
            fn=edge_type_constraint,
            keys=[PrimitiveModel.output_key, "numerator", "denominator"],
        )

        self._add_constraint(
            fn=divide_type_constraint,
            keys=[PrimitiveModel.output_key, "numerator", "denominator"],
            dependencies={edge_constraint},
        )

        self._add_constraint(
            fn=bcast,
            keys=[PrimitiveModel.output_key, "numerator", "denominator"],
            dependencies={edge_constraint},
        )

    def __call__(  # type: ignore[override]
        self,
        numerator: ConnectionType = NOT_GIVEN,
        denominator: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            numerator=numerator, denominator=denominator, output=output
        )


class FloorDivide(PrimitiveModel):
    numerator: Connection
    denominator: Connection
    output: Connection

    # TODO: Torch does not accept bool type inputs while JAX and other accepts!
    def __init__(
        self,
        numerator: Tensor[Any] | ToBeDetermined = TBD,
        denominator: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="floor_divide",
            name=name,
            output=BaseKey(shape=[("Var_out", ...)], type=Tensor),
            numerator=BaseKey(shape=[("Var_1", ...)], type=Tensor, value=numerator),
            denominator=BaseKey(shape=[("Var_2", ...)], type=Tensor, value=denominator),
        )

        bcast_constraint = self._add_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "numerator", "denominator"]
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[PrimitiveModel.output_key, "numerator", "denominator"],
            dependencies={bcast_constraint},
        )

        self._add_constraint(
            fn=floor_divide_type_constraint,
            keys=[PrimitiveModel.output_key, "numerator", "denominator"],
        )

    def __call__(  # type: ignore[override]
        self,
        numerator: ConnectionType = NOT_GIVEN,
        denominator: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            numerator=numerator, denominator=denominator, output=output
        )


class MatrixMultiply(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="matrix_multiplication",
            name=name,
            output=BaseKey(shape=[("Var3", ...), "x", "z"], type=Tensor),
            left=BaseKey(shape=[("Var1", ...), "x", "y"], type=Tensor, value=left),
            right=BaseKey(shape=[("Var2", ...), "y", "z"], type=Tensor, value=right),
        )
        bcast_constraint = self._add_constraint(
            fn=bcast_matrix_mult, keys=[PrimitiveModel.output_key, "left", "right"]
        )

        self._add_constraint(
            fn=bcast_mat_mul_check,
            keys=[PrimitiveModel.output_key, "left", "right"],
            dependencies={bcast_constraint},
        )

        self._add_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "left", "right"],
        )

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, output=output)


class Shape(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="shape",
            name=name,
            output=BaseKey(type=tuple[int, ...]),
            input=BaseKey(shape=[("input", ...)], type=Tensor, value=input),
        )
        self._add_constraint(fn=shape_constraints, keys=["output", "input"])

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Reshape(PrimitiveModel):
    input: Connection
    shape: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int | None, ...] | list[int] | ToBeDetermined = TBD,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        output_shape_map: ShapeTemplateType
        if isinstance(shape, ToBeDetermined):
            output_shape_map = [("output", ...)]
        else:
            output_shape_map = [key if key != -1 else None for key in shape]

        super().__init__(
            formula_key="reshape",
            name=name,
            output=BaseKey(shape=output_shape_map, type=Tensor),
            input=BaseKey(shape=[("input", ...)], type=Tensor, value=input),
            shape=BaseKey(type=tuple[int | None, ...] | list[int | None], value=shape),
        )
        self._add_constraint(fn=reshape_constraints, keys=["output", "input", "shape"])

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shape: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shape=shape, output=output)


class Length(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="length",
            name=name,
            output=BaseKey(type=int),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Cast(PrimitiveModel):
    input: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self, dtype: core.Dtype | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="cast",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor),
            input=BaseKey(shape=[("Var", ...)], type=Tensor),
            dtype=BaseKey(type=core.Dtype, value=dtype),
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dtype=dtype, output=output)


class Dtype(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="dtype",
            name=name,
            output=BaseKey(type=core.Dtype),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Size(PrimitiveModel):
    input: Connection
    dim: Connection
    output: Connection

    def __init__(
        self,
        dim: int | tuple[int, ...] | None | ToBeDetermined = None,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"dim": dim}
        super().__init__(
            formula_key="size",
            name=name,
            output=BaseKey(type=int | tuple[int, ...]),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
            dim=BaseKey(type=int | tuple[int, ...] | None, value=dim),
        )
        self._add_constraint(fn=size_constraints, keys=["output", "input", "dim"])

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dim: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dim=dim, output=output)


class Item(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="item",
            name=name,
            output=BaseKey(type=int | float),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )
        self._add_constraint(
            fn=item_constraints, keys=[PrimitiveModel.output_key, "input"]
        )

        self._jittable = False

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class ToTensor(PrimitiveModel):
    input: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        input: TensorValueType | ToBeDetermined = TBD,
        dtype: core.Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="to_tensor",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor),
            input=BaseKey(type=TensorValueType, value=input),
            dtype=BaseKey(type=core.Dtype | None, value=dtype),
        )

        self._add_constraint(
            fn=to_tensor_constraints, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dtype=dtype, output=output)


class ToList(PrimitiveModel):
    output: Connection

    def __init__(
        self,
        n: int,
        *,
        name: str | None = None,
        **kwargs: ScalarValueType | ToBeDetermined,
    ) -> None:
        self.factory_args = {"n": n}
        key_definitions = {}
        key_definitions["output"] = BaseKey(
            type=list[int | float | bool | list | tuple]  # type: ignore
        )
        key_definitions |= {
            f"input{idx+1}": BaseKey(
                type=int | float | bool | list | tuple,
                value=kwargs.get(f"input{idx+1}", TBD),
            )
            for idx in range(n)
        }

        super().__init__(formula_key="to_list", name=name, **key_definitions)

        self._add_constraint(
            fn=to_list_constraints,
            keys=[PrimitiveModel.output_key] + [key for key in self.input_keys],
        )


class TensorToList(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="tensor_to_list",
            name=name,
            output=BaseKey(type=TensorToListType),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )
        self._add_constraint(
            fn=tensor_to_list_constraints, keys=[PrimitiveModel.output_key, "input"]
        )
        self._add_constraint(
            fn=tensor_to_list_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

        self._jittable = False

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Reduce(PrimitiveModel):
    input: Connection
    axis: Connection
    keepdim: Connection
    output: Connection

    def __init__(
        self,
        formula_key: str,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
        **kwargs: BaseKey,
    ) -> None:
        # TODO: Handle axis type for conditional cases below.
        self.factory_args = {"axis": axis, "keepdim": keepdim}
        axis_type: UnionType | type
        if isinstance(axis, tuple):
            axis_arg = [int for _ in range(len(axis))]
            axis_type = tuple[*axis_arg]  # type: ignore
        elif axis is TBD:
            axis_type = int | tuple[int, ...] | NoneType | list[int]
        elif isinstance(axis, int | NoneType):
            axis_type = int | NoneType
        else:
            raise ValueError("Requires valid axis type!")

        init_kwargs: dict[str, BaseKey] = {
            "output": BaseKey(shape=[("Var_out", ...)], type=Tensor),
            "input": BaseKey(shape=[("Var_in", ...)], type=Tensor, value=input),
            "axis": BaseKey(type=axis_type, value=axis),
            "keepdim": BaseKey(type=bool, value=keepdim),
        }
        super().__init__(formula_key=formula_key, name=name, **(init_kwargs | kwargs))

        self._add_constraint(
            fn=reduce_constraints,
            keys=[PrimitiveModel.output_key, "input", "axis", "keepdim"],
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axis: ConnectionType = NOT_GIVEN,
        keepdim: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, axis=axis, keepdim=keepdim, output=output)


class Mean(Reduce):
    # TODO: Torch expects float input for mean reduction, JAX accepts all types.
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_mean",
            name=name,
            axis=axis,
            keepdim=keepdim,
            input=input,
            output=BaseKey(shape=[("Var_out", ...)], type=Tensor[float]),
        )


class Sum(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_sum", name=name, axis=axis, keepdim=keepdim, input=input
        )
        self._add_constraint(
            fn=reduce_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class Max(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_max", name=name, axis=axis, keepdim=keepdim, input=input
        )
        self._add_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class ArgMax(Reduce):
    def __init__(
        self,
        axis: int | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            "reduce_argmax",
            name=name,
            axis=axis,
            keepdim=keepdim,
            input=input,
            # axis = Scalar(axis_type, axis), # TODO: Change axis type to int
            output=BaseKey(shape=[("Var_out", ...)], type=Tensor[int]),
        )


class Min(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_min", name=name, axis=axis, keepdim=keepdim, input=input
        )
        self._add_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class ArgMin(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_argmin",
            name=name,
            axis=axis,
            keepdim=keepdim,
            input=input,
            # axis = Scalar(axis_type, axis), # TODO: Change axis type to int
            output=BaseKey(shape=[("Var_out", ...)], type=Tensor[int]),
        )


class Prod(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_prod",
            name=name,
            axis=axis,
            keepdim=keepdim,
            input=input,
        )
        self._add_constraint(
            fn=reduce_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class Variance(Reduce):
    correction: Connection

    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        correction: int | float | None = 0.0,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="variance",
            name=name,
            axis=axis,
            keepdim=keepdim,
            input=input,
            correction=BaseKey(type=float | int | None, value=correction),
            output=BaseKey(shape=[("Var_out", ...)], type=Tensor[float]),
        )
        self.factory_args = {"axis": axis, "correction": correction, "keepdim": keepdim}
        # TODO: Should we remove axis, correction and keepdim from factory_args?

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axis: ConnectionType = NOT_GIVEN,
        keepdim: ConnectionType = NOT_GIVEN,
        correction: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super(Reduce, self).__call__(
            input=input,
            axis=axis,
            keepdim=keepdim,
            correction=correction,
            output=output,
        )


class SingleInputOperation(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        formula_key: str,
        polymorphic_constraint: bool = True,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
        **kwargs: BaseKey,
    ) -> None:
        default_kwargs = dict(
            output=BaseKey(shape=[("Var", ...)], type=Tensor),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )
        # Finalize kwargs.
        new_kwargs: Mapping[str, BaseKey] = default_kwargs | kwargs
        super().__init__(formula_key, name=name, **new_kwargs)

        if polymorphic_constraint:
            self._add_constraint(
                fn=general_tensor_type_constraint,
                keys=[PrimitiveModel.output_key, "input"],
            )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Absolute(SingleInputOperation):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(formula_key="abs", name=name, input=input)


class Minus(SingleInputOperation):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(formula_key="minus", name=name, input=input)


class Exponential(SingleInputOperation):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="exp",
            name=name,
            polymorphic_constraint=False,
            input=input,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
        )


class Sqrt(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        cutoff: Tensor[Any] | ToBeDetermined = TBD,
        name: str | None = None,
    ) -> None:
        self.robust = robust
        self.factory_args = {"robust": robust}

        if robust:
            super().__init__(
                formula_key="robust_sqrt",
                name=name,
                output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
                input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
                cutoff=BaseKey(shape=[], type=Tensor, value=cutoff),
            )
        else:
            super().__init__(
                formula_key="sqrt",
                name=name,
                output=BaseKey(shape=[("Var", ...)], type=Tensor),
                input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
            )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        cutoff: ConnectionType = core.Constant.MIN_POSITIVE_NORMAL,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output}

        default = (
            isinstance(cutoff, core.Constant)
            and cutoff == core.Constant.MIN_POSITIVE_NORMAL
        )
        if self.robust:
            if default:
                # NOTE: Since we can not provide Tensor objects as default
                # arguments, we need to convert default value.
                cutoff = Tensor(cutoff)  # type: ignore
            kwargs["cutoff"] = cutoff
        elif not default:
            raise ValueError("Cutoff cannot be specified when robust mode is off")

        return super().__call__(**kwargs)


class RelationalOperators(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(
        self,
        formula_key: str,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key=formula_key,
            name=name,
            output=BaseKey(),
            left=BaseKey(value=left),
            right=BaseKey(value=right),
        )

        edge_constraint = self._add_constraint(
            edge_type_constraint,
            ["output", "left", "right"],
        )

        self._add_constraint(
            relational_operator_type_constraint,
            ["output", "left", "right"],
            dependencies={edge_constraint},
        )

        self._add_constraint(
            bcast,
            ["output", "left", "right"],
            dependencies={edge_constraint},
        )

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, output=output)


class Greater(RelationalOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="greater", name=name, left=left, right=right)


class Less(RelationalOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="less", name=name, left=left, right=right)


class Equal(RelationalOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="equal", name=name, left=left, right=right)


class NotEqual(RelationalOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="not_equal", name=name, left=left, right=right)


class LessEqual(RelationalOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="less_equal", name=name, left=left, right=right)


class GreaterEqual(RelationalOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="greater_equal", name=name, left=left, right=right)


class LogicalNot(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="logical_not",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[bool]),
            input=BaseKey(shape=[("Var", ...)], type=Tensor[bool], value=input),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class BitwiseOperators(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(
        self,
        formula_key: str,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key=formula_key,
            name=name,
            output=BaseKey(shape=[("Var1", ...)], type=Tensor[bool]),
            left=BaseKey(shape=[("Var2", ...)], type=Tensor[bool], value=left),
            right=BaseKey(shape=[("Var3", ...)], type=Tensor[bool], value=right),
        )
        self._add_constraint(bcast, ["output", "left", "right"])

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, output=output)


class LogicalAnd(BitwiseOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="logical_and", name=name, left=left, right=right)


class LogicalOr(BitwiseOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="logical_or", name=name, left=left, right=right)


class LogicalXOr(BitwiseOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="logical_xor", name=name, left=left, right=right)
        self.factory_args = {"left": left, "right": right}


class ShiftLeft(PrimitiveModel):
    input: Connection
    shift: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[Any] | ToBeDetermined = TBD,
        shift: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="shift_left",
            name=name,
            output=BaseKey(shape=[("Var3", ...)], type=Tensor[int]),
            input=BaseKey(shape=[("Var1", ...)], type=Tensor[int], value=input),
            shift=BaseKey(shape=[("Var2", ...)], type=Tensor[int], value=shift),
        )

        self._add_constraint(bcast, ["output", "input", "shift"])

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shift: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shift=shift, output=output)


class ShiftRight(PrimitiveModel):
    input: Connection
    shift: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[Any] | ToBeDetermined = TBD,
        shift: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="shift_right",
            name=name,
            output=BaseKey(shape=[("Var3", ...)], type=Tensor),
            input=BaseKey(shape=[("Var1", ...)], type=Tensor, value=input),
            shift=BaseKey(shape=[("Var2", ...)], type=Tensor, value=shift),
        )

        self._add_constraint(bcast, ["output", "input", "shift"])

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shift: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shift=shift, output=output)


class Transpose(PrimitiveModel):
    # NOTE: Consider if axes type list[int] is conventionally True since it is generally
    # used tuple[int] in these type of cases
    input: Connection
    axes: Connection
    output: Connection

    def __init__(
        self,
        axes: int | list[int] | tuple[int, ...] | None | ToBeDetermined = None,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axes": axes}

        if axes is None:
            super().__init__(
                formula_key="transpose",
                name=name,
                output=BaseKey(shape=[("Var_out", ...)], type=Tensor),
                input=BaseKey(shape=[("Var_in", ...)], type=Tensor, value=input),
                axes=BaseKey(type=NoneType, value=axes),
            )
            self._add_constraint(
                fn=reverse_constraints, keys=["output", "input", "axes"]
            )

        elif isinstance(axes, int | Sequence):
            axes = [axes] if isinstance(axes, int) else axes
            input_shapes = [f"x_{i}" for i in range(len(axes))]
            output_shapes = [input_shapes[i] for i in axes]
            super().__init__(
                formula_key="transpose",
                name=name,
                output=BaseKey(shape=output_shapes, type=Tensor),
                input=BaseKey(shape=input_shapes, type=Tensor, value=input),
                axes=BaseKey(type=int | tuple[int, ...], value=axes),
            )

        elif axes is TBD:
            super().__init__(
                formula_key="transpose",
                name=name,
                output=BaseKey(shape=[("Var_out", ...)], type=Tensor),
                input=BaseKey(shape=[("Var_in", ...)], type=Tensor, value=input),
                axes=BaseKey(type=int | tuple[int, ...] | None, value=axes),
            )
            self._add_constraint(
                fn=reverse_constraints, keys=["output", "input", "axes"]
            )

        self._add_constraint(
            fn=general_tensor_type_constraint, keys=["output", "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axes: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, axes=axes, output=output)


class Split(PrimitiveModel):
    split_size: Connection
    axis: Connection
    input: Connection
    output: Connection

    def __init__(
        self,
        split_size: int,  # TODO: should we add default for split_size?
        axis: int = 0,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ):
        super().__init__(
            formula_key="split",
            name=name,
            output=BaseKey(shape=[("Var2", ...)], type=Tensor),
            input=BaseKey(shape=[("Var1", ...)], type=Tensor, value=input),
            split_size=BaseKey(type=int, value=split_size),
            axis=BaseKey(type=int, value=axis),
        )

        self._add_constraint(
            fn=split_constraints, keys=["output", "input", "split_size", "axis"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        split_size: ConnectionType = NOT_GIVEN,
        axis: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input, split_size=split_size, axis=axis, output=output
        )


class Slice(PrimitiveModel):
    start: Connection
    stop: Connection
    step: Connection
    output: Connection

    def __init__(
        self,
        start: int | None | ToBeDetermined = TBD,
        stop: int | None | ToBeDetermined = TBD,
        step: int | None | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ):
        super().__init__(
            formula_key="primitive_slice",
            name=name,
            output=BaseKey(type=slice),
            start=BaseKey(type=int | None, value=start),
            stop=BaseKey(type=int | None, value=stop),
            step=BaseKey(type=int | None, value=step),
        )
        self._add_constraint(
            fn=slice_constraints, keys=["output", "start", "stop", "step"]
        )

    def __call__(  # type: ignore[override]
        self,
        start: ConnectionType = NOT_GIVEN,
        stop: ConnectionType = NOT_GIVEN,
        step: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(start=start, stop=stop, step=step, output=output)


class Indexer(PrimitiveModel):
    input: Connection
    index: Connection
    output: Connection

    def __init__(
        self,
        index: int | ToBeDetermined = TBD,
        input: Tensor[Any] | Sequence[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="indexer",
            name=name,
            output=BaseKey(),
            input=BaseKey(value=input),
            index=BaseKey(
                type=int
                | slice
                | EllipsisType
                | None
                | tuple[int | slice | EllipsisType | None, ...],
                value=index,
            ),
        )

        edge_constraints = self._add_constraint(
            fn=edge_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

        indexer_initial_constraints = self._add_constraint(
            fn=indexer_initial_type_constraint,
            keys=[PrimitiveModel.output_key, "input", "index"],
            dependencies={edge_constraints},
        )

        self._add_constraint(
            fn=indexer_constraints,
            keys=[PrimitiveModel.output_key, "input", "index"],
            dependencies={indexer_initial_constraints},
        )

        self._add_constraint(
            fn=indexer_type_constraint,
            keys=[PrimitiveModel.output_key, "input", "index"],
            dependencies={indexer_initial_constraints},
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        index: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, index=index, output=output)


class Sine(SingleInputOperation):
    def __init__(
        self,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="sin",
            name=name,
            polymorphic_constraint=False,
            input=input,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
        )


class Cosine(SingleInputOperation):
    def __init__(
        self,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="cos",
            name=name,
            polymorphic_constraint=False,
            input=input,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
        )
