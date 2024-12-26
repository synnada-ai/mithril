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
    GenericTensorType,
    IOKey,
    MyTensor,
    ShapeTemplateType,
    TensorValueType,
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
    split_constraints,
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
    "Exponential",
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
    "Split",
]
ConstantType = float | int | Constant


class Buffer(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="buffer",
            name=name,
            output=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
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
        name: str | None = None,
        **kwargs: TensorValueType | ToBeDetermined,
    ) -> None:
        self.factory_args = {"n": n}
        key_definitions = {
            "output": IOKey(type=tuple[int | float | bool | list | tuple, ...])
        }
        key_definitions |= {
            f"input{idx+1}": IOKey(type=int | float | bool | list | tuple)
            for idx in range(n)
        }
        self.factory_inputs = kwargs  # type: ignore

        super().__init__(formula_key="to_tuple", name=name, **key_definitions)
        self._set_constraint(
            fn=to_tuple_constraints,
            keys=[PrimitiveModel.output_key] + [key for key in self._input_keys],
        )


class ArithmeticOperation(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(self, formula_key: str, name: str | None = None) -> None:
        super().__init__(
            formula_key=formula_key,
            name=name,
            output=IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
            left=IOKey(shape=[("Var_1", ...)], type=GenericTensorType),
            right=IOKey(shape=[("Var_2", ...)], type=GenericTensorType),
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
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        robust: bool = False,
        base: TensorValueType | ToBeDetermined = TBD,
        exponent: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.robust = robust
        self.factory_args = {"robust": robust}
        self.factory_inputs = {"base": base, "exponent": exponent}
        assert isinstance(robust, bool), "Robust must be a boolean value!"

        if robust:
            super().__init__(
                formula_key="robust_power",
                name=name,
                output=IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
                base=IOKey(shape=[("Var_1", ...)], type=GenericTensorType),
                exponent=IOKey(shape=[("Var_2", ...)], type=GenericTensorType),
                threshold=IOKey(shape=[], type=GenericTensorType),
            )
            self.threshold.set_differentiable(False)  # type: ignore
        else:
            super().__init__(
                formula_key="power",
                output=IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
                base=IOKey(shape=[("Var_1", ...)], type=GenericTensorType),
                exponent=IOKey(shape=[("Var_2", ...)], type=GenericTensorType),
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
        output: ConnectionType = NOT_GIVEN,
        *,
        threshold: ConnectionType = Constant.MIN_POSITIVE_NORMAL,
    ) -> ExtendInfo:
        kwargs = {"base": base, "exponent": exponent, "output": output}
        is_constant = isinstance(threshold, Constant)
        if self.robust:
            kwargs["threshold"] = threshold
        elif not (is_constant and threshold == Constant.MIN_POSITIVE_NORMAL):
            raise ValueError("Threshold cannot be specified when robust mode is off")

        return super().__call__(**kwargs)


class Add(ArithmeticOperation):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="add", name=name)
        self.factory_inputs = {"left": left, "right": right}


class Subtract(ArithmeticOperation):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="subtract", name=name)
        self.factory_inputs = {"left": left, "right": right}


class Multiply(ArithmeticOperation):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="multiplication", name=name)
        self.factory_inputs = {"left": left, "right": right}


class Divide(PrimitiveModel):
    numerator: Connection
    denominator: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        numerator: TensorValueType | ToBeDetermined = TBD,
        denominator: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="divide",
            name=name,
            output=IOKey(shape=[("Var_out", ...)], type=MyTensor[float]),
            numerator=IOKey(shape=[("Var_1", ...)], type=GenericTensorType),
            denominator=IOKey(shape=[("Var_2", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"numerator": numerator, "denominator": denominator}
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
    def __init__(
        self,
        name: str | None = None,
        numerator: TensorValueType | ToBeDetermined = TBD,
        denominator: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="floor_divide",
            name=name,
            output=IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
            numerator=IOKey(shape=[("Var_1", ...)], type=GenericTensorType),
            denominator=IOKey(shape=[("Var_2", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"numerator": numerator, "denominator": denominator}

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

    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="matrix_multiplication",
            name=name,
            output=IOKey(shape=[("Var3", ...), "x", "z"], type=GenericTensorType),
            left=IOKey(shape=[("Var1", ...), "x", "y"], type=GenericTensorType),
            right=IOKey(shape=[("Var2", ...), "y", "z"], type=GenericTensorType),
        )
        self.factory_inputs = {"left": left, "right": right}
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

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="shape",
            name=name,
            output=IOKey(shape=[], type=tuple[int, ...]),
            input=IOKey(shape=[("input", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}
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
        self,
        name: str | None = None,
        shape: tuple[int | None, ...] | list[int] | ToBeDetermined = TBD,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        output_shape_map: ShapeTemplateType
        if isinstance(shape, ToBeDetermined):
            output_shape_map = [("output", ...)]
        else:
            output_shape_map = [key if key != -1 else None for key in shape]

        super().__init__(
            formula_key="reshape",
            name=name,
            output=IOKey(shape=output_shape_map, type=GenericTensorType),
            input=IOKey(shape=[("input", ...)], type=GenericTensorType),
            shape=IOKey(type=tuple[int | None, ...] | list[int | None], value=shape),
        )
        self.factory_inputs = {"input": input, "shape": shape}
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

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="length",
            name=name,
            output=IOKey(type=int),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Cast(PrimitiveModel):
    input: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, dtype: Dtype | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="astype",
            name=name,
            output=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            dtype=IOKey(type=Dtype),
        )
        self.factory_inputs = {"dtype": dtype}

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

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="dtype",
            name=name,
            output=IOKey(type=core.Dtype),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}

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
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"dim": dim}
        super().__init__(
            formula_key="size",
            name=name,
            output=IOKey(type=int | tuple[int, ...]),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            dim=IOKey(type=int | tuple[int, ...] | None, value=dim),
        )
        self.factory_inputs = {"input": input}
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
        name: str | None = None,
        start: int | None | ToBeDetermined = None,
        stop: int | None | ToBeDetermined = None,
        step: int | None | ToBeDetermined = None,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"start": start, "stop": stop, "step": step}
        super().__init__(
            formula_key="sequence_slice",
            name=name,
            output=IOKey(
                type=tuple[int | float | bool, ...] | list[int | float | bool]
            ),
            input=IOKey(type=tuple[int | float | bool, ...] | list[int | float | bool]),
            start=IOKey(type=int | None, value=start),
            stop=IOKey(type=int | None, value=stop),
            step=IOKey(type=int | None, value=step),
        )

        self.factory_inputs = {"input": input}
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
        name: str | None = None,
        start: int | None | ToBeDetermined = None,
        stop: int | None | ToBeDetermined = None,
        step: int | None | ToBeDetermined = None,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"start": start, "stop": stop, "step": step}
        super().__init__(
            formula_key="tensor_slice",
            name=name,
            output=IOKey(shape=["a", ("Var1", ...)], type=GenericTensorType),
            input=IOKey(shape=["b", ("Var1", ...)], type=GenericTensorType),
            start=IOKey(type=int | None, value=start),
            stop=IOKey(type=int | None, value=stop),
            step=IOKey(type=int | None, value=step),
        )
        self.factory_inputs = {"input": input}

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

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="item",
            name=name,
            output=IOKey(type=int | float),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}
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

    def __init__(
        self,
        name: str | None = None,
        index: int | ToBeDetermined = TBD,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="scalar_item",
            name=name,
            output=IOKey(type=int | float | list | tuple),
            input=IOKey(type=list | tuple),
            index=IOKey(type=int, value=index),
        )
        self.factory_inputs = {"input": input, "index": index}

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

    def __init__(
        self,
        name: str | None = None,
        index: int | ToBeDetermined = TBD,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="tensor_item",
            name=name,
            output=IOKey(shape=[("Var2", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var1", ...)], type=GenericTensorType),
            index=IOKey(
                type=int
                | slice
                | EllipsisType
                | None
                | tuple[int | slice | EllipsisType | None, ...],
                value=index,
            ),
        )
        self.factory_inputs = {"input": input, "index": index}

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

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="to_tensor",
            name=name,
            output=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            input=IOKey(type=int | float | list | tuple),
        )

        self._set_constraint(
            fn=to_tensor_constraints, keys=[PrimitiveModel.output_key, "input"]
        )
        self.factory_inputs = {"input": input}

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class ToList(PrimitiveModel):
    output: Connection

    def __init__(self, n: int, name: str | None = None, **kwargs) -> None:
        self.factory_args = {"n": n}
        key_definitions = {}
        key_definitions["output"] = IOKey(type=list[int | float | bool | list | tuple])
        key_definitions |= {
            f"input{idx+1}": IOKey(type=int | float | bool | list | tuple)
            for idx in range(n)
        }
        self.factory_inputs = kwargs

        super().__init__(formula_key="to_list", name=name, **key_definitions)

        self._set_constraint(
            fn=to_list_constraints,
            keys=[PrimitiveModel.output_key] + [key for key in self._input_keys],
        )


class TensorToList(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="tensor_to_list",
            name=name,
            output=IOKey(type=NestedListType(int | float | bool)),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input}
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
        name: str | None = None,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        **kwargs: IOKey,
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

        init_kwargs: dict[str, IOKey] = {
            "output": IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
            "input": IOKey(shape=[("Var_in", ...)], type=GenericTensorType),
            "axis": IOKey(type=axis_type, value=axis),
            "keepdim": IOKey(type=bool, value=keepdim),
        }
        super().__init__(formula_key=formula_key, name=name, **(init_kwargs | kwargs))

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
        name: str | None = None,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="reduce_mean",
            name=name,
            axis=axis,
            keepdim=keepdim,
            output=IOKey(shape=[("Var_out", ...)], type=MyTensor[float]),
        )
        self.factory_inputs = {"input": input, "axis": axis, "keepdim": keepdim}
        # self.factory_inputs = {"input": input}


class Sum(Reduce):
    def __init__(
        self,
        name: str | None = None,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="reduce_sum", name=name, axis=axis, keepdim=keepdim
        )
        self.factory_inputs = {"input": input, "axis": axis, "keepdim": keepdim}

        self._set_constraint(
            fn=reduce_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class Max(Reduce):
    def __init__(
        self,
        name: str | None = None,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="reduce_max", name=name, axis=axis, keepdim=keepdim
        )
        self.factory_inputs = {"input": input, "axis": axis, "keepdim": keepdim}
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class ArgMax(Reduce):
    def __init__(
        self,
        name: str | None = None,
        axis: int | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            "reduce_argmax",
            name=name,
            axis=axis,
            keepdim=keepdim,
            # axis = Scalar(axis_type, axis), # TODO: Change axis type to int
            output=IOKey(shape=[("Var_out", ...)], type=MyTensor[int]),
        )
        self.factory_inputs = {"input": input, "axis": axis, "keepdim": keepdim}


class Min(Reduce):
    def __init__(
        self,
        name: str | None = None,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="reduce_min", name=name, axis=axis, keepdim=keepdim
        )

        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )
        self.factory_inputs = {"input": input, "axis": axis, "keepdim": keepdim}


class ArgMin(Reduce):
    def __init__(
        self,
        name: str | None = None,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="reduce_argmin",
            name=name,
            axis=axis,
            keepdim=keepdim,
            # axis = Scalar(axis_type, axis), # TODO: Change axis type to int
            output=IOKey(shape=[("Var_out", ...)], type=MyTensor[int]),
        )
        self.factory_inputs = {"input": input, "axis": axis, "keepdim": keepdim}


class Prod(Reduce):
    def __init__(
        self,
        name: str | None = None,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="reduce_prod", name=name, axis=axis, keepdim=keepdim
        )
        self.factory_inputs = {"input": input, "axis": axis, "keepdim": keepdim}
        self._set_constraint(
            fn=reduce_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class Variance(Reduce):
    correction: Connection

    def __init__(
        self,
        name: str | None = None,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool = False,
        correction: int | float | None = 0.0,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="variance",
            name=name,
            axis=axis,
            keepdim=keepdim,
            correction=IOKey(type=float | int | None, value=correction),
            output=IOKey(shape=[("Var_out", ...)], type=MyTensor[float]),
        )
        self.factory_args = {"axis": axis, "correction": correction, "keepdim": keepdim}
        # TODO: Should we remove axis, correction and keepdim from factory_args?
        self.factory_inputs = {
            "input": input,
            "axis": axis,
            "keepdim": keepdim,
            "correction": correction,
        }

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
        name: str | None = None,
        **kwargs: IOKey,
    ) -> None:
        default_kwargs = dict(
            output=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
        )
        # Finalize kwargs.
        new_kwargs: Mapping[str, IOKey] = default_kwargs | kwargs
        super().__init__(formula_key, name=name, **new_kwargs)

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
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(formula_key="abs", name=name)
        self.factory_inputs = {"input": input}


class Minus(SingleInputOperation):
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(formula_key="minus", name=name)
        self.factory_inputs = {"input": input}


class Exponential(SingleInputOperation):
    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="exp",
            name=name,
            polymorphic_constraint=False,
            output=IOKey(shape=[("Var", ...)], type=MyTensor[float]),
        )
        self.factory_inputs = {"input": input}


class Sqrt(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        *,
        cutoff: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.robust = robust
        self.factory_args = {"robust": robust}

        if robust:
            super().__init__(
                formula_key="robust_sqrt",
                name=name,
                output=IOKey(shape=[("Var", ...)], type=MyTensor[float]),
                input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
                cutoff=IOKey(shape=[], type=GenericTensorType),
            )
            self.factory_inputs = {"input": input, "cutoff": cutoff}
        else:
            super().__init__(
                formula_key="sqrt",
                output=IOKey(shape=[("Var", ...)], type=GenericTensorType),
                input=IOKey(shape=[("Var", ...)], type=GenericTensorType),
            )
            self.factory_inputs = {"input": input}

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        cutoff: ConnectionType = Constant.MIN_POSITIVE_NORMAL,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output}

        is_constant = isinstance(cutoff, Constant)
        if self.robust:
            kwargs["cutoff"] = cutoff
        elif not (is_constant and cutoff == Constant.MIN_POSITIVE_NORMAL):
            raise ValueError("Cutoff cannot be specified when robust mode is off")

        return super().__call__(**kwargs)


class RelationalOperators(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(self, formula_key: str, name: str | None = None) -> None:
        super().__init__(
            formula_key=formula_key,
            name=name,
            output=IOKey(shape=[("Var1", ...)], type=MyTensor[bool]),
            left=IOKey(shape=[("Var2", ...)], type=GenericTensorType),
            right=IOKey(shape=[("Var3", ...)], type=GenericTensorType),
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
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="greater", name=name)
        self.factory_inputs = {"left": left, "right": right}


class Less(RelationalOperators):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="less", name=name)
        self.factory_inputs = {"left": left, "right": right}


class Equal(RelationalOperators):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="equal", name=name)
        self.factory_inputs = {"left": left, "right": right}


class NotEqual(RelationalOperators):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="not_equal", name=name)
        self.factory_inputs = {"left": left, "right": right}


class LessEqual(RelationalOperators):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="less_equal", name=name)
        self.factory_inputs = {"left": left, "right": right}


class GreaterEqual(RelationalOperators):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="greater_equal", name=name)
        self.factory_inputs = {"left": left, "right": right}


class LogicalNot(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, name: str | None = None, input: TensorValueType | ToBeDetermined = TBD
    ) -> None:
        super().__init__(
            formula_key="logical_not",
            name=name,
            output=IOKey(shape=[("Var", ...)], type=MyTensor[bool]),
            input=IOKey(shape=[("Var", ...)], type=MyTensor[bool]),
        )
        self.factory_inputs = {"input": input}

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
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key=formula_key,
            name=name,
            output=IOKey(shape=[("Var1", ...)], type=MyTensor[bool]),
            left=IOKey(shape=[("Var2", ...)], type=MyTensor[bool]),
            right=IOKey(shape=[("Var3", ...)], type=MyTensor[bool]),
        )
        self.factory_inputs = {"left": left, "right": right}
        self._set_constraint(bcast, ["output", "left", "right"])

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
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="logical_and", name=name)
        self.factory_inputs = {"left": left, "right": right}


class LogicalOr(BitwiseOperators):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="logical_or", name=name)
        self.factory_inputs = {"left": left, "right": right}


class LogicalXOr(BitwiseOperators):
    def __init__(
        self,
        name: str | None = None,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="logical_xor", name=name)
        self.factory_inputs = {"left": left, "right": right}
        self.factory_args = {"left": left, "right": right}


class ShiftLeft(PrimitiveModel):
    input: Connection
    shift: Connection
    output: Connection

    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        shift: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="shift_left",
            name=name,
            output=IOKey(shape=[("Var3", ...)], type=MyTensor[int]),
            input=IOKey(shape=[("Var1", ...)], type=MyTensor[int]),
            shift=IOKey(shape=[("Var2", ...)], type=MyTensor[int]),
        )
        self.factory_inputs = {"input": input, "shift": shift}

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

    def __init__(
        self,
        name: str | None = None,
        input: TensorValueType | ToBeDetermined = TBD,
        shift: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="shift_right",
            name=name,
            output=IOKey(shape=[("Var3", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var1", ...)], type=GenericTensorType),
            shift=IOKey(shape=[("Var2", ...)], type=GenericTensorType),
        )
        self.factory_inputs = {"input": input, "shift": shift}

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
        self,
        name: str | None = None,
        axes: int | list[int] | tuple[int, ...] | None | ToBeDetermined = None,
        input: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        self.factory_args = {"axes": axes}

        if axes is None:
            super().__init__(
                formula_key="transpose",
                name=name,
                output=IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
                input=IOKey(shape=[("Var_in", ...)], type=GenericTensorType),
                axes=IOKey(type=NoneType, value=axes),
            )
            self.factory_inputs = {"input": input, "axes": axes}
            self._set_constraint(
                fn=reverse_constraints, keys=["output", "input", "axes"]
            )

        elif isinstance(axes, int | Sequence):
            axes = [axes] if isinstance(axes, int) else axes
            input_shapes = [f"x_{i}" for i in range(len(axes))]
            output_shapes = [input_shapes[i] for i in axes]
            super().__init__(
                formula_key="transpose",
                name=name,
                output=IOKey(shape=output_shapes, type=GenericTensorType),
                input=IOKey(shape=input_shapes, type=GenericTensorType),
                axes=IOKey(type=int | tuple[int, ...], value=axes),
            )

        elif axes is TBD:
            super().__init__(
                formula_key="transpose",
                name=name,
                output=IOKey(shape=[("Var_out", ...)], type=GenericTensorType),
                input=IOKey(shape=[("Var_in", ...)], type=GenericTensorType),
                axes=IOKey(type=int | tuple[int, ...] | None, value=axes),
            )
            self._set_constraint(
                fn=reverse_constraints, keys=["output", "input", "axes"]
            )

        self._set_constraint(
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
        name: str | None = None,
        axis: int = 0,
        input: TensorValueType | ToBeDetermined = TBD,
    ):
        super().__init__(
            formula_key="split",
            name=name,
            output=IOKey(shape=[("Var2", ...)], type=GenericTensorType),
            input=IOKey(shape=[("Var1", ...)], type=GenericTensorType),
            split_size=IOKey(type=int, value=split_size),
            axis=IOKey(type=int, value=axis),
        )
        self.factory_inputs = {"input": input, "split_size": split_size, "axis": axis}

        self._set_constraint(
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
