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

from ... import core
from ...core import Constant, Dtype
from ..common import (
    NOT_GIVEN,
    TBD,
    Connection,
    ConnectionType,
    Scalar,
    ShapeTemplateType,
    TensorType,
    ToBeDetermined,
)
from ..constraints import (
    bcast,
    bcast_matrix_mult,
    floor_divide_type_constraint,
    general_tensor_type_constraint,
    item_constraints,
    reduce_constraints,
    reduce_type_constraint,
    reshape_constraints,
    reverse_constraints,
    scalar_item_constraints,
    scalar_item_type_constraint,
    scalar_slice_type_constraint,
    shape_constraints,
    size_constraints,
    tensor_item_constraints,
    tensor_slice_constraints,
    tensor_to_list_constraints,
    tensor_to_list_type_constraint,
    to_list_constraints,
    to_tensor_constraints,
    to_tuple_constraints,
)
from ..utils import NestedListType
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
    "PrimitiveSlice",
    "Item",
    "ScalarItem",
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
    "TensorItem",
    "TensorSlice",
    "ArgMax",
    "ArgMin",
    "Cast",
    "Transpose",
    "Sqrt",
]
ConstantType = float | int | Constant


class Buffer(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            output=TensorType([("Var", ...)]),
            input=TensorType([("Var", ...)]),
        )

        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


ToTupleOutputType = tuple[int | float | bool | list | tuple, ...]


class ToTuple(PrimitiveModel):
    def __init__(self, n: int) -> None:
        self.factory_args = {"n": n}
        key_definitions = {}
        key_definitions["output"] = Scalar(ToTupleOutputType)
        key_definitions |= {
            f"input{idx+1}": Scalar(int | float | bool | list | tuple)
            for idx in range(n)
        }

        super().__init__(formula_key="to_tuple", **key_definitions)
        self._set_constraint(
            fn=to_tuple_constraints,
            keys=[PrimitiveModel.output_key] + [key for key in self._input_keys],
        )


class ArithmeticOperation(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(self, formula_key: str) -> None:
        super().__init__(
            formula_key=formula_key,
            output=TensorType([("Var_out", ...)]),
            left=TensorType([("Var_1", ...)]),
            right=TensorType([("Var_2", ...)]),
        )

        self._set_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "left", "right"]
        )
        self._set_constraint(
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


class Power(PrimitiveModel):
    base: Connection
    exponent: Connection
    threshold: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        threshold: ConstantType | ToBeDetermined = Constant.MIN_POSITIVE_NORMAL,
    ) -> None:
        self.factory_args = {"threshold": threshold, "robust": robust}
        assert isinstance(robust, bool), "Robust must be a boolean value!"

        if robust:
            super().__init__(
                formula_key="robust_power",
                output=TensorType([("Var_out", ...)]),
                base=TensorType([("Var_1", ...)]),
                exponent=TensorType([("Var_2", ...)]),
                threshold=TensorType([], ConstantType, threshold),
            )

        else:
            if threshold != Constant.MIN_POSITIVE_NORMAL:
                raise KeyError(
                    "Threshold cannot be specified \
                               when robust mode is off"
                )

            super().__init__(
                formula_key="power",
                output=TensorType([("Var_out", ...)]),
                base=TensorType([("Var_1", ...)]),
                exponent=TensorType([("Var_2", ...)]),
            )

        self._set_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "base", "exponent"]
        )
        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "base", "exponent"],
        )

    def __call__(  # type: ignore[override]
        self,
        base: ConnectionType = NOT_GIVEN,
        exponent: ConnectionType = NOT_GIVEN,
        threshold: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"base": base, "exponent": exponent, "output": output}

        if "threshold" in self._input_keys:
            kwargs["threshold"] = threshold
        elif threshold != NOT_GIVEN:
            raise ValueError("Threshold cannot be specified when robust mode is off")

        return super().__call__(**kwargs)


class Add(ArithmeticOperation):
    def __init__(self) -> None:
        super().__init__(formula_key="add")


class Subtract(ArithmeticOperation):
    def __init__(self) -> None:
        super().__init__(formula_key="subtract")


class Multiply(ArithmeticOperation):
    def __init__(self) -> None:
        super().__init__(formula_key="multiplication")


class Divide(PrimitiveModel):
    numerator: Connection
    denominator: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="divide",
            output=TensorType([("Var_out", ...)], float),
            numerator=TensorType([("Var_1", ...)]),
            denominator=TensorType([("Var_2", ...)]),
        )
        self._set_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "numerator", "denominator"]
        )
        # TODO: Needs any type constraint??

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
    def __init__(self) -> None:
        super().__init__(
            formula_key="floor_divide",
            output=TensorType([("Var_out", ...)]),
            numerator=TensorType([("Var_1", ...)]),
            denominator=TensorType([("Var_2", ...)]),
        )

        self._set_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "numerator", "denominator"]
        )
        self._set_constraint(
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

    def __init__(self) -> None:
        super().__init__(
            formula_key="matrix_multiplication",
            output=TensorType([("Var3", ...), "x", "z"]),
            left=TensorType([("Var1", ...), "x", "y"]),
            right=TensorType([("Var2", ...), "y", "z"]),
        )

        self._set_constraint(
            fn=bcast_matrix_mult, keys=[PrimitiveModel.output_key, "left", "right"]
        )
        self._set_constraint(
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

    def __init__(self) -> None:
        super().__init__(
            formula_key="shape",
            output=Scalar(tuple[int, ...]),
            input=TensorType([("input", ...)]),
        )
        self._set_constraint(fn=shape_constraints, keys=["output", "input"])

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Reshape(PrimitiveModel):
    input: Connection
    shape: Connection
    output: Connection

    def __init__(
        self, shape: tuple[int | None, ...] | list[int] | ToBeDetermined = TBD
    ) -> None:
        output_shape_map: ShapeTemplateType
        if isinstance(shape, ToBeDetermined):
            output_shape_map = [("output", ...)]
        else:
            output_shape_map = [key if key != -1 else None for key in shape]

        super().__init__(
            formula_key="reshape",
            output=TensorType(output_shape_map),
            input=TensorType([("input", ...)]),
            shape=Scalar(tuple[int | None, ...] | list[int | None], shape),
        )
        self._set_constraint(fn=reshape_constraints, keys=["output", "input", "shape"])

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

    def __init__(self) -> None:
        super().__init__(
            formula_key="length", output=Scalar(int), input=TensorType([("Var", ...)])
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Cast(PrimitiveModel):
    input: Connection
    dtype: Connection
    output: Connection

    def __init__(self, dtype: Dtype | ToBeDetermined) -> None:
        super().__init__(
            formula_key="astype",
            output=TensorType([("Var", ...)]),
            input=TensorType([("Var", ...)]),
            dtype=Scalar(Dtype, dtype),
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dtype=dtype, output=output)


class DType(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="dtype",
            output=Scalar(core.Dtype),
            input=TensorType([("Var", ...)]),
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
        self, dim: int | tuple[int, ...] | None | ToBeDetermined = None
    ) -> None:
        self.factory_args = {"dim": dim}
        super().__init__(
            formula_key="size",
            output=Scalar(int | tuple[int, ...]),
            input=TensorType([("Var", ...)]),
            dim=Scalar(int | tuple[int, ...] | None, dim),
        )
        self._set_constraint(fn=size_constraints, keys=["output", "input", "dim"])

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dim: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dim=dim, output=output)


class PrimitiveSlice(PrimitiveModel):
    input: Connection
    start: Connection
    stop: Connection
    step: Connection
    output: Connection

    def __init__(
        self,
        start: int | None | ToBeDetermined = None,
        stop: int | None | ToBeDetermined = None,
        step: int | None | ToBeDetermined = None,
    ) -> None:
        self.factory_args = {"start": start, "stop": stop, "step": step}
        super().__init__(
            formula_key="sequence_slice",
            output=Scalar(tuple[int | float | bool, ...] | list[int | float | bool]),
            input=Scalar(tuple[int | float | bool, ...] | list[int | float | bool]),
            start=Scalar(int | None, start),
            stop=Scalar(int | None, stop),
            step=Scalar(int | None, step),
        )

        self._set_constraint(
            fn=scalar_slice_type_constraint,
            keys=[PrimitiveModel.output_key, "input", "start", "stop", "step"],
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        start: ConnectionType = NOT_GIVEN,
        stop: ConnectionType = NOT_GIVEN,
        step: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input, start=start, stop=stop, step=step, output=output
        )


class TensorSlice(PrimitiveModel):
    input: Connection
    start: Connection
    stop: Connection
    step: Connection
    output: Connection

    def __init__(
        self,
        start: int | None | ToBeDetermined = None,
        stop: int | None | ToBeDetermined = None,
        step: int | None | ToBeDetermined = None,
    ) -> None:
        self.factory_args = {"start": start, "stop": stop, "step": step}
        super().__init__(
            formula_key="tensor_slice",
            output=TensorType(["a", ("Var1", ...)]),
            input=TensorType(["b", ("Var1", ...)]),
            start=Scalar(int | None, start),
            stop=Scalar(int | None, stop),
            step=Scalar(int | None, step),
        )

        self._set_constraint(
            fn=tensor_slice_constraints,
            keys=[PrimitiveModel.output_key, "input", "start", "stop", "step"],
        )
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        start: ConnectionType = NOT_GIVEN,
        stop: ConnectionType = NOT_GIVEN,
        step: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input, start=start, stop=stop, step=step, output=output
        )


class Item(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="item",
            output=Scalar(int | float),
            input=TensorType([("Var", ...)]),
        )

        self._set_constraint(
            fn=item_constraints, keys=[PrimitiveModel.output_key, "input"]
        )

        self._jittable = False

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class ScalarItem(PrimitiveModel):
    input: Connection
    index: Connection
    output: Connection

    def __init__(self, index: int | ToBeDetermined = TBD) -> None:
        super().__init__(
            formula_key="scalar_item",
            output=Scalar(int | float | list | tuple),
            input=Scalar(list | tuple),
            index=Scalar(int, index),
        )

        self._set_constraint(
            fn=scalar_item_constraints,
            keys=[PrimitiveModel.output_key, "input", "index"],
        )
        self._set_constraint(
            fn=scalar_item_type_constraint,
            keys=[PrimitiveModel.output_key, "input", "index"],
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        index: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, index=index, output=output)


class TensorItem(PrimitiveModel):
    input: Connection
    index: Connection
    output: Connection

    def __init__(self, index: int | ToBeDetermined = TBD) -> None:
        super().__init__(
            formula_key="tensor_item",
            output=TensorType([("Var2", ...)]),
            input=TensorType([("Var1", ...)]),
            index=Scalar(
                int
                | slice
                | EllipsisType
                | None
                | tuple[int | slice | EllipsisType | None, ...],
                index,
            ),
        )

        self._set_constraint(
            fn=tensor_item_constraints,
            keys=[PrimitiveModel.output_key, "input", "index"],
        )
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        index: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, index=index, output=output)


class ToTensor(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="to_tensor",
            output=TensorType([("Var", ...)]),
            input=Scalar(int | float | list | tuple),
        )

        self._set_constraint(
            fn=to_tensor_constraints, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class ToList(PrimitiveModel):
    output: Connection

    def __init__(self, n: int) -> None:
        self.factory_args = {"n": n}
        key_definitions = {}
        key_definitions["output"] = Scalar(list[int | float | bool | list | tuple])
        key_definitions |= {
            f"input{idx+1}": Scalar(int | float | bool | list | tuple)
            for idx in range(n)
        }

        super().__init__(formula_key="to_list", **key_definitions)

        self._set_constraint(
            fn=to_list_constraints,
            keys=[PrimitiveModel.output_key] + [key for key in self._input_keys],
        )


class TensorToList(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="tensor_to_list",
            output=Scalar(NestedListType(int | float | bool)),
            input=TensorType([("Var", ...)]),
        )

        self._set_constraint(
            fn=tensor_to_list_constraints, keys=[PrimitiveModel.output_key, "input"]
        )
        self._set_constraint(
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
        **kwargs: TensorType | Scalar,
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

        init_kwargs: dict[str, TensorType | Scalar] = {
            "output": TensorType([("Var_out", ...)]),
            "input": TensorType([("Var_in", ...)]),
            "axis": Scalar(axis_type, axis),
            "keepdim": Scalar(bool, keepdim),
        }

        super().__init__(formula_key=formula_key, **(init_kwargs | kwargs))

        self._set_constraint(
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
    ) -> None:
        super().__init__(
            formula_key="reduce_mean",
            axis=axis,
            keepdim=keepdim,
            output=TensorType([("Var_out", ...)], float),
        )


class Sum(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
    ) -> None:
        super().__init__(formula_key="reduce_sum", axis=axis, keepdim=keepdim)

        self._set_constraint(
            fn=reduce_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class Max(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
    ) -> None:
        super().__init__(formula_key="reduce_max", axis=axis, keepdim=keepdim)

        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class ArgMax(Reduce):
    def __init__(
        self,
        axis: int | None | ToBeDetermined = None,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            "reduce_argmax",
            axis,
            keepdim,
            # axis = Scalar(axis_type, axis), # TODO: Change axis type to int
            output=TensorType([("Var_out", ...)], int),
        )


class Min(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
    ) -> None:
        super().__init__(formula_key="reduce_min", axis=axis, keepdim=keepdim)

        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class ArgMin(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            formula_key="reduce_argmin",
            axis=axis,
            keepdim=keepdim,
            # axis = Scalar(axis_type, axis), # TODO: Change axis type to int
            output=TensorType([("Var_out", ...)], int),
        )


class Prod(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
    ) -> None:
        super().__init__(formula_key="reduce_prod", axis=axis, keepdim=keepdim)

        self._set_constraint(
            fn=reduce_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class Variance(Reduce):
    correction: Connection

    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        correction: int | float | None = 0.0,
    ) -> None:
        super().__init__(
            formula_key="variance",
            axis=axis,
            keepdim=keepdim,
            correction=Scalar(float | int | None, correction),
            output=TensorType([("Var_out", ...)], float),
        )
        self.factory_args = {"axis": axis, "correction": correction, "keepdim": keepdim}

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
        formula_key,
        polymorphic_constraint: bool = True,
        **kwargs: TensorType | Scalar,
    ) -> None:
        default_kwargs = dict(
            output=TensorType([("Var", ...)]), input=TensorType([("Var", ...)])
        )
        # Finalize kwargs.
        new_kwargs: Mapping = default_kwargs | kwargs
        super().__init__(formula_key, **new_kwargs)

        if polymorphic_constraint:
            self._set_constraint(
                fn=general_tensor_type_constraint,
                keys=[PrimitiveModel.output_key, "input"],
            )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Absolute(SingleInputOperation):
    def __init__(self) -> None:
        super().__init__(formula_key="abs")


class Minus(SingleInputOperation):
    def __init__(self) -> None:
        super().__init__(formula_key="minus")


class Sqrt(PrimitiveModel):
    input: Connection
    cutoff: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        cutoff: ConstantType | ToBeDetermined = Constant.MIN_POSITIVE_NORMAL,
    ) -> None:
        self.factory_args = {"robust": robust, "cutoff": cutoff}

        if robust:
            if isinstance(cutoff, str) and cutoff != Constant.MIN_POSITIVE_NORMAL:
                raise ValueError(f"cutoff can only be set to 'min_positive_normal' \
                                 in string format, got {cutoff}")

            super().__init__(
                formula_key="robust_sqrt",
                output=TensorType([("Var", ...)], float),
                input=TensorType([("Var", ...)]),
                cutoff=TensorType([], ConstantType, cutoff),
            )
        else:
            super().__init__(
                formula_key="sqrt",
                output=TensorType([("Var", ...)], float),
                input=TensorType([("Var", ...)]),
            )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        cutoff: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output}

        if self.formula_key == "sqrt" and cutoff != NOT_GIVEN:
            raise ValueError(
                "Sqrt does not accept cutoff argument \
                             when initialized with robust = False."
            )

        if self.formula_key == "robust_sqrt":
            kwargs["cutoff"] = cutoff

        return super().__call__(**kwargs)


class RelationalOperators(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(self, formula_key: str) -> None:
        super().__init__(
            formula_key=formula_key,
            output=TensorType([("Var1", ...)], bool),
            left=TensorType([("Var2", ...)]),
            right=TensorType([("Var3", ...)]),
        )

        self._set_constraint(bcast, ["output", "left", "right"])

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, output=output)


class Greater(RelationalOperators):
    def __init__(self) -> None:
        super().__init__("greater")


class Less(RelationalOperators):
    def __init__(self) -> None:
        super().__init__("less")


class Equal(RelationalOperators):
    def __init__(self) -> None:
        super().__init__("equal")


class NotEqual(RelationalOperators):
    def __init__(self) -> None:
        super().__init__("not_equal")


class LessEqual(RelationalOperators):
    def __init__(self) -> None:
        super().__init__("less_equal")


class GreaterEqual(RelationalOperators):
    def __init__(self) -> None:
        super().__init__("greater_equal")


class LogicalNot(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="logical_not",
            output=TensorType([("Var", ...)], bool),
            input=TensorType([("Var", ...)], bool),
        )

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class BitwiseOperators(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(self, formula_key: str) -> None:
        super().__init__(
            formula_key=formula_key,
            output=TensorType([("Var1", ...)], bool),
            left=TensorType([("Var2", ...)], bool),
            right=TensorType([("Var3", ...)], bool),
        )
        self._set_constraint(bcast, ["output", "left", "right"])

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, output=output)


class LogicalAnd(BitwiseOperators):
    def __init__(self) -> None:
        super().__init__("logical_and")


class LogicalOr(BitwiseOperators):
    def __init__(self) -> None:
        super().__init__("logical_or")


class LogicalXOr(BitwiseOperators):
    def __init__(self) -> None:
        super().__init__("logical_xor")


class ShiftLeft(PrimitiveModel):
    input: Connection
    shift: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="shift_left",
            output=TensorType([("Var3", ...)], int),
            input=TensorType([("Var1", ...)], int),
            shift=TensorType([("Var2", ...)], int),
        )

        self._set_constraint(bcast, ["output", "input", "shift"])

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

    def __init__(self) -> None:
        super().__init__(
            formula_key="shift_right",
            output=TensorType([("Var3", ...)], int),
            input=TensorType([("Var1", ...)], int),
            shift=TensorType([("Var2", ...)], int),
        )

        self._set_constraint(bcast, ["output", "input", "shift"])

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
        self, axes: int | list[int] | tuple[int, ...] | None | ToBeDetermined = None
    ) -> None:
        self.factory_args = {"axes": axes}

        if axes is None:
            super().__init__(
                formula_key="transpose",
                output=TensorType([("Var_out", ...)]),
                input=TensorType([("Var_in", ...)]),
                axes=Scalar(NoneType, axes),
            )
            self._set_constraint(
                fn=reverse_constraints, keys=["output", "input", "axes"]
            )

        elif isinstance(axes, int | Sequence):
            axes = [axes] if isinstance(axes, int) else axes
            input_shapes = [f"x_{i}" for i in range(len(axes))]
            output_shapes = [input_shapes[i] for i in axes]
            super().__init__(
                formula_key="transpose",
                output=TensorType(output_shapes),
                input=TensorType(input_shapes),
                axes=Scalar(int | tuple[int, ...], axes),
            )

        elif axes is TBD:
            super().__init__(
                formula_key="transpose",
                output=TensorType([("Var_out", ...)]),
                input=TensorType([("Var_in", ...)]),
                axes=Scalar(int | tuple[int, ...] | None, axes),
            )
            self._set_constraint(
                fn=reverse_constraints, keys=["output", "input", "axes"]
            )

        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axes: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, axes=axes, output=output)
