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
from functools import partial
from types import EllipsisType, NoneType, UnionType
from typing import Any

from ... import core
from ..common import (
    TBD,
    BaseKey,
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
    general_forward_constraint,
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
from .operator import Operator

__all__ = [
    "Operator",
    "BufferOp",
    "ToTupleOp",
    "PowerOp",
    "AddOp",
    "SubtractOp",
    "MultiplyOp",
    "DivideOp",
    "FloorDivideOp",
    "MinusOp",
    "MatrixMultiplyOp",
    "ShapeOp",
    "ReshapeOp",
    "LengthOp",
    "SizeOp",
    "ExponentialOp",
    "ItemOp",
    "IndexerOp",
    "ToTensorOp",
    "ToListOp",
    "TensorToListOp",
    "MeanOp",
    "SumOp",
    "MaxOp",
    "MinOp",
    "ProdOp",
    "VarianceOp",
    "AbsoluteOp",
    "EqualOp",
    "NotEqualOp",
    "GreaterOp",
    "GreaterEqualOp",
    "LessOp",
    "LessEqualOp",
    "LogicalNotOp",
    "LogicalOrOp",
    "LogicalAndOp",
    "LogicalXOrOp",
    "ShiftLeftOp",
    "ShiftRightOp",
    "ArgMaxOp",
    "ArgMinOp",
    "CastOp",
    "TransposeOp",
    "SqrtOp",
    "SplitOp",
    "SliceOp",
    "DtypeOp",
    "SineOp",
    "CosineOp",
    "MinimumOp",
    "MaximumOp",
]

ConstantType = float | int | core.Constant


class BufferOp(Operator):
    _model_name: str = "Buffer"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="buffer",
            output=BaseKey(),
            input=BaseKey(value=input),
        )

        self._add_constraint(fn=buffer_constraint, keys=[Operator.output_key, "input"])


class ToTupleOp(Operator):
    _model_name: str = "ToTuple"

    def __init__(
        self,
        n: int,
        **kwargs: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined,
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

        super().__init__(formula_key="to_tuple", name=None, **key_definitions)
        self._add_constraint(
            fn=to_tuple_constraints,
            keys=[Operator.output_key] + [key for key in self.input_keys],
        )
        self._set_cin()


class ArithmeticOp(Operator):
    _model_name: str = "Arithmetic"

    def __init__(
        self,
        formula_key: str,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
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
            keys=[Operator.output_key, "left", "right"],
        )

        self._add_constraint(
            fn=general_tensor_type_constraint,
            keys=[Operator.output_key, "left", "right"],
            dependencies={edge_constraint},
        )

        bcast_constraint = self._add_constraint(
            fn=bcast,
            keys=[Operator.output_key, "left", "right"],
            dependencies={edge_constraint},
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[Operator.output_key, "left", "right"],
            dependencies={bcast_constraint},
        )
        self.edge_constraint = edge_constraint


class PowerOp(Operator):
    _model_name: str = "Power"

    def __init__(
        self,
        robust: bool = False,
        base: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        exponent: Tensor[int | float | bool]
        | int
        | float
        | bool
        | ToBeDetermined = TBD,
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
                output=BaseKey(shape=[("out", ...)], type=Tensor[int | float]),
                base=BaseKey(shape=[("base", ...)], type=Tensor, value=base),
                exponent=BaseKey(shape=[("exp", ...)], type=Tensor, value=exponent),
                threshold=BaseKey(shape=[], type=Tensor),
            )

            constrs: set[Constraint] = set()

        else:
            super().__init__(
                formula_key="power",
                name=name,
                output=BaseKey(type=Tensor[int | float] | int | float),
                base=BaseKey(
                    type=Tensor[int | float | bool] | int | float | bool, value=base
                ),
                exponent=BaseKey(
                    type=Tensor[int | float | bool] | int | float | bool, value=exponent
                ),
            )
            edge_constraint = self._add_constraint(
                fn=edge_type_constraint,
                keys=[Operator.output_key, "base", "exponent"],
            )
            constrs = {edge_constraint}

        self._add_constraint(
            fn=general_tensor_type_constraint,
            keys=[Operator.output_key, "base", "exponent"],
            dependencies=constrs,
        )

        bcast_constraint = self._add_constraint(
            fn=bcast,
            keys=[Operator.output_key, "base", "exponent"],
            dependencies=constrs,
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[Operator.output_key, "base", "exponent"],
            dependencies={bcast_constraint},
        )

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x**y),
            keys=[Operator.output_key, "base", "exponent"],
            dependencies=constrs,
        )

        constrs = constrs


class AddOp(ArithmeticOp):
    _model_name: str = "Add"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="add", name=name, left=left, right=right)

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x + y),
            keys=[Operator.output_key, "left", "right"],
            dependencies={self.edge_constraint},
        )
        self._set_cin("left", "right", safe=False)


class SubtractOp(ArithmeticOp):
    _model_name: str = "Subtract"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="subtract", name=name, left=left, right=right)

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x - y),
            keys=[Operator.output_key, "left", "right"],
            dependencies={self.edge_constraint},
        )


class MultiplyOp(ArithmeticOp):
    _model_name: str = "Multiply"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="multiplication", name=name, left=left, right=right
        )

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x * y),
            keys=[Operator.output_key, "left", "right"],
            dependencies={self.edge_constraint},
        )
        self._set_cin("left", "right", safe=False)


class MinimumOp(ArithmeticOp):
    _model_name: str = "Minimum"

    def __init__(
        self,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="minimum", left=left, right=right)
        self._set_cin("left", "right", safe=False)


class MaximumOp(ArithmeticOp):
    _model_name: str = "Maximum"

    def __init__(
        self,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="maximum", left=left, right=right)
        self._set_cin("left", "right", safe=False)


class DivideOp(Operator):
    _model_name: str = "Divide"

    def __init__(
        self,
        numerator: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        denominator: Tensor[int | float | bool]
        | ScalarValueType
        | ToBeDetermined = TBD,
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
            keys=[Operator.output_key, "numerator", "denominator"],
        )

        self._add_constraint(
            fn=divide_type_constraint,
            keys=[Operator.output_key, "numerator", "denominator"],
            dependencies={edge_constraint},
        )

        bcast_constraint = self._add_constraint(
            fn=bcast,
            keys=[Operator.output_key, "numerator", "denominator"],
            dependencies={edge_constraint},
        )

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x / y),
            keys=[Operator.output_key, "numerator", "denominator"],
            dependencies={edge_constraint},
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[Operator.output_key, "numerator", "denominator"],
            dependencies={bcast_constraint},
        )


class FloorDivideOp(Operator):
    _model_name: str = "FloorDivide"

    def __init__(
        self,
        numerator: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        denominator: Tensor[int | float | bool]
        | ScalarValueType
        | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="floor_divide",
            name=name,
            output=BaseKey(type=Tensor[int | float] | int | float),
            numerator=BaseKey(value=numerator),
            denominator=BaseKey(value=denominator),
        )
        edge_constraint = self._add_constraint(
            fn=edge_type_constraint,
            keys=[Operator.output_key, "numerator", "denominator"],
        )

        self._add_constraint(
            fn=floor_divide_type_constraint,
            keys=[Operator.output_key, "numerator", "denominator"],
            dependencies={edge_constraint},
        )

        bcast_constraint = self._add_constraint(
            fn=bcast,
            keys=[Operator.output_key, "numerator", "denominator"],
            dependencies={edge_constraint},
        )

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x // y),
            keys=[Operator.output_key, "numerator", "denominator"],
            dependencies={edge_constraint},
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[Operator.output_key, "numerator", "denominator"],
            dependencies={bcast_constraint},
        )


class MatrixMultiplyOp(Operator):
    _model_name: str = "MatrixMultiply"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ToBeDetermined = TBD,
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
            fn=bcast_matrix_mult, keys=[Operator.output_key, "left", "right"]
        )

        self._add_constraint(
            fn=bcast_mat_mul_check,
            keys=[Operator.output_key, "left", "right"],
            dependencies={bcast_constraint},
        )

        self._add_constraint(
            fn=general_tensor_type_constraint,
            keys=[Operator.output_key, "left", "right"],
        )


class ShapeOp(Operator):
    _model_name: str = "Shape"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="shape",
            name=name,
            output=BaseKey(type=tuple[int, ...]),
            input=BaseKey(shape=[("input", ...)], type=Tensor, value=input),
        )
        self._add_constraint(fn=shape_constraints, keys=["output", "input"])


class ReshapeOp(Operator):
    _model_name: str = "Reshape"

    def __init__(
        self,
        shape: tuple[int | None, ...] | list[int] | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class LengthOp(Operator):
    _model_name: str = "Length"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="length",
            name=name,
            output=BaseKey(type=int),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )


class CastOp(Operator):
    _model_name: str = "Cast"

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


class DtypeOp(Operator):
    _model_name: str = "Dtype"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="dtype",
            name=name,
            output=BaseKey(type=core.Dtype),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )


class SizeOp(Operator):
    _model_name: str = "Size"

    def __init__(
        self,
        dim: int | tuple[int, ...] | None | ToBeDetermined = None,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class ItemOp(Operator):
    _model_name: str = "Item"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="item",
            name=name,
            output=BaseKey(type=int | float),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )
        self._add_constraint(fn=item_constraints, keys=[Operator.output_key, "input"])

        self._jittable = False


class ToTensorOp(Operator):
    _model_name: str = "ToTensor"

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
            fn=to_tensor_constraints, keys=[Operator.output_key, "input"]
        )


class ToListOp(Operator):
    _model_name: str = "ToList"

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
            keys=[Operator.output_key] + [key for key in self.input_keys],
        )
        self._set_cin()


class TensorToListOp(Operator):
    _model_name: str = "TensorToList"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="tensor_to_list",
            name=name,
            output=BaseKey(type=TensorToListType),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )
        self._add_constraint(
            fn=tensor_to_list_constraints, keys=[Operator.output_key, "input"]
        )
        self._add_constraint(
            fn=tensor_to_list_type_constraint, keys=[Operator.output_key, "input"]
        )

        self._jittable = False


class ReduceOp(Operator):
    _model_name: str = "Reduce"

    def __init__(
        self,
        formula_key: str,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
        **kwargs: BaseKey,
    ) -> None:
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
            keys=[Operator.output_key, "input", "axis", "keepdim"],
        )


class MeanOp(ReduceOp):
    _model_name: str = "Mean"

    # TODO: Torch expects float input for mean reduction, JAX accepts all types.
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class SumOp(ReduceOp):
    _model_name: str = "Sum"

    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_sum", name=name, axis=axis, keepdim=keepdim, input=input
        )
        self._add_constraint(
            fn=reduce_type_constraint, keys=[Operator.output_key, "input"]
        )


class MaxOp(ReduceOp):
    _model_name: str = "Max"

    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_max", name=name, axis=axis, keepdim=keepdim, input=input
        )
        self._add_constraint(
            fn=general_tensor_type_constraint, keys=[Operator.output_key, "input"]
        )


class ArgMaxOp(ReduceOp):
    _model_name: str = "ArgMax"

    def __init__(
        self,
        axis: int | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            "reduce_argmax",
            name=name,
            axis=axis,
            keepdim=keepdim,
            input=input,
            # axis = Scalar(axis_type, axis), # TODO: Change axis type to int
            output=BaseKey(shape=[("Var_out", ...)], type=Tensor[int]),
        )


class MinOp(ReduceOp):
    _model_name: str = "Min"

    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_min", name=name, axis=axis, keepdim=keepdim, input=input
        )
        self._add_constraint(
            fn=general_tensor_type_constraint, keys=[Operator.output_key, "input"]
        )


class ArgMinOp(ReduceOp):
    _model_name: str = "ArgMin"

    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_argmin",
            name=name,
            axis=axis,
            keepdim=keepdim,
            input=input,
            # axis = Scalar(axis_type, axis), # TODO: Change axis type to int
            output=BaseKey(shape=[("Var_out", ...)], type=Tensor[int]),
        )


class ProdOp(ReduceOp):
    _model_name: str = "Prod"

    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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
            fn=reduce_type_constraint, keys=[Operator.output_key, "input"]
        )


class VarianceOp(ReduceOp):
    _model_name: str = "Variance"

    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        correction: int | float | None = 0.0,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class SingleInputOperationOp(Operator):
    _model_name: str = "SingleInputOperation"

    def __init__(
        self,
        formula_key: str,
        polymorphic_constraint: bool = True,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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
        super().__init__(formula_key=formula_key, name=name, **new_kwargs)

        if polymorphic_constraint:
            self._add_constraint(
                fn=general_tensor_type_constraint,
                keys=[Operator.output_key, "input"],
            )


class AbsoluteOp(SingleInputOperationOp):
    _model_name: str = "Absolute"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="abs", name=name, input=input)


class MinusOp(SingleInputOperationOp):
    _model_name: str = "Minus"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="minus", name=name, input=input)


class ExponentialOp(SingleInputOperationOp):
    _model_name: str = "Exponential"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="exp",
            name=name,
            polymorphic_constraint=False,
            input=input,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
        )


class SqrtOp(Operator):
    _model_name: str = "Sqrt"

    def __init__(
        self,
        robust: bool = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        cutoff: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class RelationalOperatorsOp(Operator):
    _model_name: str = "RelationalOperators"

    def __init__(
        self,
        formula_key: str,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
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

        bcast_constraint = self._add_constraint(
            bcast,
            ["output", "left", "right"],
            dependencies={edge_constraint},
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[Operator.output_key, "left", "right"],
            dependencies={bcast_constraint},
        )
        self.edge_constraint = edge_constraint


class GreaterOp(RelationalOperatorsOp):
    _model_name: str = "Greater"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="greater", name=name, left=left, right=right)

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x > y),
            keys=[Operator.output_key, "left", "right"],
            dependencies={self.edge_constraint},
        )


class LessOp(RelationalOperatorsOp):
    _model_name: str = "Less"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="less", name=name, left=left, right=right)

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x < y),
            keys=[Operator.output_key, "left", "right"],
            dependencies={self.edge_constraint},
        )


class EqualOp(RelationalOperatorsOp):
    _model_name: str = "Equal"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="equal", name=name, left=left, right=right)

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x == y),
            keys=[Operator.output_key, "left", "right"],
            dependencies={self.edge_constraint},
        )
        self._set_cin("left", "right", safe=False)


class NotEqualOp(RelationalOperatorsOp):
    _model_name: str = "NotEqual"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="not_equal", name=name, left=left, right=right)

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x != y),
            keys=[Operator.output_key, "left", "right"],
            dependencies={self.edge_constraint},
        )
        self._set_cin("left", "right", safe=False)


class LessEqualOp(RelationalOperatorsOp):
    _model_name: str = "LessEqual"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="less_equal", name=name, left=left, right=right)

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x <= y),
            keys=[Operator.output_key, "left", "right"],
            dependencies={self.edge_constraint},
        )


class GreaterEqualOp(RelationalOperatorsOp):
    _model_name: str = "GreaterEqual"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="greater_equal", name=name, left=left, right=right)

        self._add_constraint(
            partial(general_forward_constraint, callable=lambda x, y: x >= y),
            keys=[Operator.output_key, "left", "right"],
            dependencies={self.edge_constraint},
        )


class LogicalNotOp(Operator):
    _model_name: str = "LogicalNot"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="logical_not",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[bool]),
            input=BaseKey(shape=[("Var", ...)], type=Tensor[bool], value=input),
        )


class BitwiseOperatorsOp(Operator):
    _model_name: str = "BitwiseOperators"

    def __init__(
        self,
        formula_key: str,
        left: Tensor[int | float | bool] | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class LogicalAndOp(BitwiseOperatorsOp):
    _model_name: str = "LogicalAnd"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="logical_and", name=name, left=left, right=right)
        self._set_cin("left", "right", safe=False)


class LogicalOrOp(BitwiseOperatorsOp):
    _model_name: str = "LogicalOr"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="logical_or", name=name, left=left, right=right)
        self._set_cin("left", "right", safe=False)


class LogicalXOrOp(BitwiseOperatorsOp):
    _model_name: str = "LogicalXOr"

    def __init__(
        self,
        left: Tensor[int | float | bool] | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="logical_xor", name=name, left=left, right=right)
        self.factory_args = {"left": left, "right": right}
        self._set_cin("left", "right", safe=False)


class ShiftLeftOp(Operator):
    _model_name: str = "ShiftLeft"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        shift: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class ShiftRightOp(Operator):
    _model_name: str = "ShiftRight"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        shift: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class TransposeOp(Operator):
    _model_name: str = "Transpose"

    def __init__(
        self,
        axes: int | list[int] | tuple[int, ...] | None | ToBeDetermined = None,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class SplitOp(Operator):
    _model_name: str = "Split"

    def __init__(
        self,
        split_size: int,  # TODO: should we add default for split_size?
        axis: int = 0,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class SliceOp(Operator):
    _model_name: str = "Slice"

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
        self._set_cin()


class IndexerOp(Operator):
    _model_name: str = "Indexer"

    def __init__(
        self,
        index: int | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | Sequence[Any] | ToBeDetermined = TBD,
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
            fn=edge_type_constraint, keys=[Operator.output_key, "input"]
        )

        indexer_initial_constraints = self._add_constraint(
            fn=indexer_initial_type_constraint,
            keys=[Operator.output_key, "input", "index"],
            dependencies={edge_constraints},
        )

        self._add_constraint(
            fn=indexer_constraints,
            keys=[Operator.output_key, "input", "index"],
            dependencies={indexer_initial_constraints},
        )

        self._add_constraint(
            fn=indexer_type_constraint,
            keys=[Operator.output_key, "input", "index"],
            dependencies={indexer_initial_constraints},
        )


class SineOp(SingleInputOperationOp):
    _model_name: str = "Sine"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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


class CosineOp(SingleInputOperationOp):
    _model_name: str = "Cosine"

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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
