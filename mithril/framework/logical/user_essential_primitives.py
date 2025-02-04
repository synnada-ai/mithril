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


from collections.abc import Sequence
from typing import Any

from ... import core

# from .essential_primitives import *
from ..common import (
    NOT_GIVEN,
    TBD,
    BaseKey,
    ScalarValueType,
    Tensor,
    TensorValueType,
    ToBeDetermined,
)
from .essential_primitives import (
    AbsoluteOp,
    AddOp,
    ArgMaxOp,
    ArgMinOp,
    BufferOp,
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
    MaximumOp,
    MaxOp,
    MeanOp,
    MinimumOp,
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
    TensorToListOp,
    ToListOp,
    ToTensorOp,
    ToTupleOp,
    TransposeOp,
    VarianceOp,
)
from .model import Connection, ConnectionType, ExtendInfo, IOKey, Model
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

from typing import overload

ConstantType = float | int | core.Constant


# TODO: merge this file to primitives file
class UserPrimitiveModel(Model):
    @overload
    def __init__(
        self,
        *,
        name: str | None = None,
        formula_key: str | None = None,
        **kwargs: BaseKey,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        name: str | None = None,
        model: PrimitiveModel | None = None,
    ) -> None: ...

    def __init__(  # type: ignore
        self,
        *,
        name: str | None = None,
        model: PrimitiveModel | None = None,
        formula_key: str | None = None,
        **kwargs: BaseKey,
    ) -> None:
        _kwargs: dict[str, ConnectionType]
        if not ((formula_key is None) ^ (model is None)):
            raise ValueError("Either formula_key or model must be provided")
        elif model is None:
            model = PrimitiveModel(
                formula_key=formula_key, name=self.__class__.__name__, **kwargs
            )
            _kwargs = {key: IOKey(key, expose=True) for key in kwargs}
        else:
            if kwargs != {}:
                raise ValueError("kwargs must be empty when model is provided")
            _kwargs = {key: IOKey(key, expose=True) for key in model.external_keys}
        super().__init__(name=name, enforce_jit=model._jittable)
        self._extend(model, _kwargs)


class Buffer(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=BufferOp(input=input))

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class ToTuple(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        n: int,
        *,
        name: str | None = None,
        **kwargs: Tensor[Any] | ScalarValueType | ToBeDetermined,
    ) -> None:
        super().__init__(name=name, model=ToTupleOp(n, **kwargs))


class ArithmeticOperation(UserPrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(
        self,
        model: PrimitiveModel,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=model)

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, output=output)


class Power(UserPrimitiveModel):
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
        m = PowerOp(robust=robust, base=base, exponent=exponent)
        super().__init__(name=name, model=m)

    def __call__(  # type: ignore[override]
        self,
        base: ConnectionType = NOT_GIVEN,
        exponent: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
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
        super().__init__(AddOp(left=left, right=right), name=name)


class Subtract(ArithmeticOperation):
    def __init__(
        self,
        left: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(SubtractOp(left=left, right=right), name=name)


class Multiply(ArithmeticOperation):
    def __init__(
        self,
        left: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        right: Tensor[Any] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(MultiplyOp(left=left, right=right), name=name)


class Minimum(ArithmeticOperation):
    def __init__(
        self,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(MinimumOp(left=left, right=right), name=name)


class Maximum(ArithmeticOperation):
    def __init__(
        self,
        left: TensorValueType | ToBeDetermined = TBD,
        right: TensorValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(MaximumOp(left=left, right=right), name=name)


class Divide(UserPrimitiveModel):
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
        m = DivideOp(numerator=numerator, denominator=denominator)
        super().__init__(name=name, model=m)

    def __call__(  # type: ignore[override]
        self,
        numerator: ConnectionType = NOT_GIVEN,
        denominator: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            numerator=numerator, denominator=denominator, output=output
        )


class FloorDivide(UserPrimitiveModel):
    numerator: Connection
    denominator: Connection
    output: Connection

    def __init__(
        self,
        numerator: Tensor[Any] | ToBeDetermined = TBD,
        denominator: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        m = FloorDivideOp(numerator=numerator, denominator=denominator)
        super().__init__(name=name, model=m)

    def __call__(  # type: ignore[override]
        self,
        numerator: ConnectionType = NOT_GIVEN,
        denominator: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            numerator=numerator, denominator=denominator, output=output
        )


class MatrixMultiply(UserPrimitiveModel):
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
        super().__init__(name=name, model=MatrixMultiplyOp(left=left, right=right))

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, output=output)


class Shape(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(name=name, model=ShapeOp(input=input))

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Reshape(UserPrimitiveModel):
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
        super().__init__(name=name, model=ReshapeOp(shape=shape, input=input))

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shape: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shape=shape, output=output)


class Length(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(name=name, model=LengthOp(input=input))

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Cast(UserPrimitiveModel):
    input: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self, dtype: core.Dtype | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(name=name, model=CastOp(dtype=dtype))

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dtype=dtype, output=output)


class Dtype(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(name=name, model=DtypeOp(input=input))

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Size(UserPrimitiveModel):
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
        super().__init__(name=name, model=SizeOp(input=input, dim=dim))

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dim: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dim=dim, output=output)


class Item(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(name=name, model=ItemOp(input=input))

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class ToTensor(UserPrimitiveModel):
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
        super().__init__(name=name, model=ToTensorOp(input=input, dtype=dtype))

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dtype=dtype, output=output)


class ToList(UserPrimitiveModel):
    output: Connection

    def __init__(
        self,
        n: int,
        *,
        name: str | None = None,
        **kwargs: ScalarValueType | ToBeDetermined,
    ) -> None:
        super().__init__(name=name, model=ToListOp(n, name=name, **kwargs))


class TensorToList(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        self._enforce_jit = False
        m = TensorToListOp(input=input)
        m._enforce_jit = False
        super().__init__(name=name, model=m)

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Reduce(UserPrimitiveModel):
    input: Connection
    axis: Connection
    keepdim: Connection
    output: Connection

    def __init__(self, model: PrimitiveModel, *, name: str | None = None) -> None:
        super().__init__(name=name, model=model)

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axis: ConnectionType = NOT_GIVEN,
        keepdim: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, axis=axis, keepdim=keepdim, output=output)


class Mean(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis, "keepdim": keepdim}
        super().__init__(MeanOp(axis=axis, keepdim=keepdim, input=input), name=name)


class Sum(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis, "keepdim": keepdim}
        super().__init__(SumOp(axis=axis, keepdim=keepdim, input=input), name=name)


class Max(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis, "keepdim": keepdim}
        super().__init__(MaxOp(axis=axis, keepdim=keepdim, input=input), name=name)


class ArgMax(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis, "keepdim": keepdim}
        super().__init__(ArgMaxOp(axis=axis, keepdim=keepdim, input=input), name=name)


class Min(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis, "keepdim": keepdim}
        super().__init__(MinOp(axis=axis, keepdim=keepdim, input=input), name=name)


class ArgMin(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis, "keepdim": keepdim}
        super().__init__(ArgMinOp(axis=axis, keepdim=keepdim, input=input), name=name)


class Prod(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis, "keepdim": keepdim}
        super().__init__(ProdOp(axis=axis, keepdim=keepdim, input=input), name=name)


class Variance(Reduce):
    correction: Connection

    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        correction: int | float | None = 0.0,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis, "keepdim": keepdim, "correction": correction}
        super().__init__(
            VarianceOp(axis=axis, keepdim=keepdim, input=input, correction=correction),
            name=name,
        )

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


class SingleInputModel(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Absolute(SingleInputModel):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(name=name, model=AbsoluteOp(input=input))


class Minus(SingleInputModel):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(name=name, model=MinusOp(input=input))


class Exponential(SingleInputModel):
    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(name=name, model=ExponentialOp(input=input))


class Sqrt(UserPrimitiveModel):
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
        m = SqrtOp(robust=robust, input=input, cutoff=cutoff)
        super().__init__(name=name, model=m)

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


class RelationalModel(UserPrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        left: ConnectionType = NOT_GIVEN,
        right: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(left=left, right=right, output=output)


class Greater(RelationalModel):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=GreaterOp(left=left, right=right))


class Less(RelationalModel):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LessOp(left=left, right=right))


class Equal(RelationalModel):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=EqualOp(left=left, right=right))


class NotEqual(RelationalModel):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=NotEqualOp(left=left, right=right))


class LessEqual(RelationalModel):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LessEqualOp(left=left, right=right))


class GreaterEqual(RelationalModel):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=GreaterEqualOp(left=left, right=right))


class LogicalNot(UserPrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self, input: Tensor[Any] | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(name=name, model=LogicalNotOp(input=input))

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class BitwiseOperators(UserPrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

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
        super().__init__(name=name, model=LogicalAndOp(left=left, right=right))


class LogicalOr(BitwiseOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LogicalOrOp(left=left, right=right))


class LogicalXOr(BitwiseOperators):
    def __init__(
        self,
        left: Tensor[Any] | ToBeDetermined = TBD,
        right: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LogicalXOrOp(left=left, right=right))


class ShiftLeft(UserPrimitiveModel):
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
        super().__init__(name=name, model=ShiftLeftOp(input=input, shift=shift))

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shift: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shift=shift, output=output)


class ShiftRight(UserPrimitiveModel):
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
        super().__init__(name=name, model=ShiftRightOp(input=input, shift=shift))

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shift: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shift=shift, output=output)


class Transpose(UserPrimitiveModel):
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
        super().__init__(name=name, model=TransposeOp(input=input, axes=axes))

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axes: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, axes=axes, output=output)


class Split(UserPrimitiveModel):
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
        m = SplitOp(split_size=split_size, axis=axis, input=input)
        super().__init__(name=name, model=m)

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


class Slice(UserPrimitiveModel):
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
        super().__init__(name=name, model=SliceOp(start=start, stop=stop, step=step))

    def __call__(  # type: ignore[override]
        self,
        start: ConnectionType = NOT_GIVEN,
        stop: ConnectionType = NOT_GIVEN,
        step: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(start=start, stop=stop, step=step, output=output)


class Indexer(UserPrimitiveModel):
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
        super().__init__(name=name, model=IndexerOp(input=input, index=index))

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        index: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, index=index, output=output)


class Sine(SingleInputModel):
    def __init__(
        self,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=SineOp(input=input))


class Cosine(SingleInputModel):
    def __init__(
        self,
        input: Tensor[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=CosineOp(input=input))
