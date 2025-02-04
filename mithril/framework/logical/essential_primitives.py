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
    TBD,
    BaseKey,
    ScalarValueType,
    ShapeTemplateType,
    Tensor,
    TensorToListType,
    TensorValueType,
    ToBeDetermined,
)
from ..constraints import (
    bcast,
    bcast_matrix_mult,
    bcast_power,
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
from .primitive import PrimitiveModel

__all__ = [
    "PrimitiveModel",
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


class BufferOp(PrimitiveModel):
    def __init__(
        self,
        input: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(
            formula_key="buffer",
            output=BaseKey(),
            input=BaseKey(value=input),
        )

        self._set_constraint(
            fn=buffer_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class ToTupleOp(PrimitiveModel):
    def __init__(
        self,
        n: int,
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

        super().__init__(formula_key="to_tuple", name=None, **key_definitions)
        self._set_constraint(
            fn=to_tuple_constraints,
            keys=[PrimitiveModel.output_key] + [key for key in self.input_keys],
        )


class ArithmeticOp(PrimitiveModel):
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

        self._set_constraint(
            fn=edge_type_constraint,
            keys=[PrimitiveModel.output_key, "left", "right"],
            post_processes={general_tensor_type_constraint, bcast},
        )


class PowerOp(PrimitiveModel):
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

            self._set_constraint(
                fn=edge_type_constraint,
                keys=[PrimitiveModel.output_key, "base", "exponent", "threshold"],
                post_processes={general_tensor_type_constraint, bcast_power},
            )
        else:
            super().__init__(
                formula_key="power",
                name=name,
                output=BaseKey(),
                base=BaseKey(value=base),
                exponent=BaseKey(value=exponent),
            )
            self._set_constraint(
                fn=edge_type_constraint,
                keys=[PrimitiveModel.output_key, "base", "exponent"],
                post_processes={general_tensor_type_constraint, bcast_power},
            )


class AddOp(ArithmeticOp):
    def __init__(
        self,
        left: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="add", name=name, left=left, right=right)


class SubtractOp(ArithmeticOp):
    def __init__(
        self,
        left: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="subtract", name=name, left=left, right=right)


class MultiplyOp(ArithmeticOp):
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


class MinimumOp(ArithmeticOp):
    def __init__(
        self,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="minimum", left=left, right=right)


class MaximumOp(ArithmeticOp):
    def __init__(
        self,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
    ) -> None:
        super().__init__(formula_key="maximum", left=left, right=right)


class DivideOp(PrimitiveModel):
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
        self._set_constraint(
            fn=edge_type_constraint,
            keys=[PrimitiveModel.output_key, "numerator", "denominator"],
            post_processes={divide_type_constraint, bcast},
        )


class FloorDivideOp(PrimitiveModel):
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

        self._set_constraint(
            fn=bcast, keys=[PrimitiveModel.output_key, "numerator", "denominator"]
        )
        self._set_constraint(
            fn=floor_divide_type_constraint,
            keys=[PrimitiveModel.output_key, "numerator", "denominator"],
        )


class MatrixMultiplyOp(PrimitiveModel):
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
        self._set_constraint(
            fn=bcast_matrix_mult, keys=[PrimitiveModel.output_key, "left", "right"]
        )
        self._set_constraint(
            fn=general_tensor_type_constraint,
            keys=[PrimitiveModel.output_key, "left", "right"],
        )


class ShapeOp(PrimitiveModel):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="shape",
            name=name,
            output=BaseKey(type=tuple[int, ...]),
            input=BaseKey(shape=[("input", ...)], type=Tensor, value=input),
        )
        self._set_constraint(fn=shape_constraints, keys=["output", "input"])


class ReshapeOp(PrimitiveModel):
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
        self._set_constraint(fn=reshape_constraints, keys=["output", "input", "shape"])


class LengthOp(PrimitiveModel):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="length",
            name=name,
            output=BaseKey(type=int),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )


class CastOp(PrimitiveModel):
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


class DtypeOp(PrimitiveModel):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="dtype",
            name=name,
            output=BaseKey(type=core.Dtype),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )


class SizeOp(PrimitiveModel):
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
        self._set_constraint(fn=size_constraints, keys=["output", "input", "dim"])


class ItemOp(PrimitiveModel):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="item",
            name=name,
            output=BaseKey(type=int | float),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )
        self._set_constraint(
            fn=item_constraints, keys=[PrimitiveModel.output_key, "input"]
        )

        self._jittable = False


class ToTensorOp(PrimitiveModel):
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

        self._set_constraint(
            fn=to_tensor_constraints, keys=[PrimitiveModel.output_key, "input"]
        )


class ToListOp(PrimitiveModel):
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

        self._set_constraint(
            fn=to_list_constraints,
            keys=[PrimitiveModel.output_key] + [key for key in self.input_keys],
        )


class TensorToListOp(PrimitiveModel):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="tensor_to_list",
            name=name,
            output=BaseKey(type=TensorToListType),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )
        self._set_constraint(
            fn=tensor_to_list_constraints, keys=[PrimitiveModel.output_key, "input"]
        )
        self._set_constraint(
            fn=tensor_to_list_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )

        self._jittable = False


class ReduceOp(PrimitiveModel):
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

        self._set_constraint(
            fn=reduce_constraints,
            keys=[PrimitiveModel.output_key, "input", "axis", "keepdim"],
        )


class MeanOp(ReduceOp):
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


class SumOp(ReduceOp):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_sum", name=name, axis=axis, keepdim=keepdim, input=input
        )
        self._set_constraint(
            fn=reduce_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class MaxOp(ReduceOp):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_max", name=name, axis=axis, keepdim=keepdim, input=input
        )
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class ArgMaxOp(ReduceOp):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
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


class MinOp(ReduceOp):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="reduce_min", name=name, axis=axis, keepdim=keepdim, input=input
        )
        self._set_constraint(
            fn=general_tensor_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class ArgMinOp(ReduceOp):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
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


class ProdOp(ReduceOp):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
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
        self._set_constraint(
            fn=reduce_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )


class VarianceOp(ReduceOp):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
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


class SingleInputOperationOp(PrimitiveModel):
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
        super().__init__(formula_key=formula_key, name=name, **new_kwargs)

        if polymorphic_constraint:
            self._set_constraint(
                fn=general_tensor_type_constraint,
                keys=[PrimitiveModel.output_key, "input"],
            )


class AbsoluteOp(SingleInputOperationOp):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(formula_key="abs", name=name, input=input)


class MinusOp(SingleInputOperationOp):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(formula_key="minus", name=name, input=input)


class ExponentialOp(SingleInputOperationOp):
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


class SqrtOp(PrimitiveModel):
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


class RelationalOperatorsOp(PrimitiveModel):
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

        self._set_constraint(
            edge_type_constraint,
            ["output", "left", "right"],
            post_processes={relational_operator_type_constraint, bcast},
        )


class GreaterOp(RelationalOperatorsOp):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="greater", name=name, left=left, right=right)


class LessOp(RelationalOperatorsOp):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="less", name=name, left=left, right=right)


class EqualOp(RelationalOperatorsOp):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="equal", name=name, left=left, right=right)


class NotEqualOp(RelationalOperatorsOp):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="not_equal", name=name, left=left, right=right)


class LessEqualOp(RelationalOperatorsOp):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="less_equal", name=name, left=left, right=right)


class GreaterEqualOp(RelationalOperatorsOp):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="greater_equal", name=name, left=left, right=right)


class LogicalNotOp(PrimitiveModel):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(
            formula_key="logical_not",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[bool]),
            input=BaseKey(shape=[("Var", ...)], type=Tensor[bool], value=input),
        )


class BitwiseOperatorsOp(PrimitiveModel):
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
        self._set_constraint(bcast, ["output", "left", "right"])


class LogicalAndOp(BitwiseOperatorsOp):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="logical_and", name=name, left=left, right=right)


class LogicalOrOp(BitwiseOperatorsOp):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="logical_or", name=name, left=left, right=right)


class LogicalXOrOp(BitwiseOperatorsOp):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="logical_xor", name=name, left=left, right=right)
        self.factory_args = {"left": left, "right": right}


class ShiftLeftOp(PrimitiveModel):
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
        self._set_constraint(bcast, ["output", "input", "shift"])


class ShiftRightOp(PrimitiveModel):
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
        self._set_constraint(bcast, ["output", "input", "shift"])


class TransposeOp(PrimitiveModel):
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
            self._set_constraint(
                fn=reverse_constraints, keys=["output", "input", "axes"]
            )

        self._set_constraint(
            fn=general_tensor_type_constraint, keys=["output", "input"]
        )


class SplitOp(PrimitiveModel):
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
        self._set_constraint(
            fn=split_constraints, keys=["output", "input", "split_size", "axis"]
        )


class SliceOp(PrimitiveModel):
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
        self._set_constraint(
            fn=slice_constraints, keys=["output", "start", "stop", "step"]
        )


class IndexerOp(PrimitiveModel):
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
        self._set_constraint(
            fn=edge_type_constraint, keys=[PrimitiveModel.output_key, "input"]
        )
        self._set_constraint(
            fn=indexer_initial_type_constraint,
            keys=[PrimitiveModel.output_key, "input", "index"],
            post_processes={indexer_type_constraint, indexer_constraints},
        )


class SineOp(SingleInputOperationOp):
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


class CosineOp(SingleInputOperationOp):
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
