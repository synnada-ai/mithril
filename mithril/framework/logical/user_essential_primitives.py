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


from ... import core
from ..common import NOT_GIVEN, Tensor
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

# from .essential_primitives import *
from .model import Connection, ConnectionType, ExtendInfo, Model
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


class Buffer(Model, BufferOp):
    input: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class ToTuple(Model, ToTupleOp):
    input: Connection
    output: Connection


class ArithmeticOperation(Model):
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


class Power(Model, PowerOp):
    base: Connection
    exponent: Connection
    output: Connection

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


class Add(ArithmeticOperation, AddOp):
    pass


class Subtract(ArithmeticOperation, SubtractOp):
    pass


class Multiply(ArithmeticOperation, MultiplyOp):
    pass


class Minimum(ArithmeticOperation, MinimumOp):
    pass


class Maximum(ArithmeticOperation, MaximumOp):
    pass


class Divide(Model, DivideOp):
    numerator: Connection
    denominator: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        numerator: ConnectionType = NOT_GIVEN,
        denominator: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            numerator=numerator, denominator=denominator, output=output
        )


class FloorDivide(Model, FloorDivideOp):
    numerator: Connection
    denominator: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        numerator: ConnectionType = NOT_GIVEN,
        denominator: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            numerator=numerator, denominator=denominator, output=output
        )


class MatrixMultiply(Model, MatrixMultiplyOp):
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


class Shape(Model, ShapeOp):
    input: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Reshape(Model, ReshapeOp):
    input: Connection
    shape: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shape: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shape=shape, output=output)


class Length(Model, LengthOp):
    input: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Cast(Model, CastOp):
    input: Connection
    dtype: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dtype=dtype, output=output)


class Dtype(Model, DtypeOp):
    input: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Size(Model, SizeOp):
    input: Connection
    dim: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dim: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dim=dim, output=output)


class Item(Model, ItemOp):
    input: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class ToTensor(Model, ToTensorOp):
    input: Connection
    dtype: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, dtype=dtype, output=output)


class ToList(Model, ToListOp):
    output: Connection


class TensorToList(Model, TensorToListOp):
    input: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class ReduceOp(Model):
    input: Connection
    axis: Connection
    keepdim: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axis: ConnectionType = NOT_GIVEN,
        keepdim: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, axis=axis, keepdim=keepdim, output=output)


class Mean(ReduceOp, MeanOp):
    pass


class Sum(ReduceOp, SumOp):
    pass


class Max(ReduceOp, MaxOp):
    pass


class ArgMax(ReduceOp, ArgMaxOp):
    pass


class Min(ReduceOp, MinOp):
    pass


class ArgMin(ReduceOp, ArgMinOp):
    pass


class Prod(ReduceOp, ProdOp):
    pass


class Variance(ReduceOp, VarianceOp):
    correction: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axis: ConnectionType = NOT_GIVEN,
        keepdim: ConnectionType = NOT_GIVEN,
        correction: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super(ReduceOp, self).__call__(
            input=input,
            axis=axis,
            keepdim=keepdim,
            correction=correction,
            output=output,
        )


class SingleInputOp(Model):
    input: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Absolute(SingleInputOp, AbsoluteOp):
    pass


class Minus(SingleInputOp, MinusOp):
    pass


class Exponential(SingleInputOp, ExponentialOp):
    pass


class Sqrt(Model, SqrtOp):
    input: Connection
    output: Connection

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


class RelationalOp(Model):
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


class Greater(RelationalOp, GreaterOp):
    pass


class Less(RelationalOp, LessOp):
    pass


class Equal(RelationalOp, EqualOp):
    pass


class NotEqual(RelationalOp, NotEqualOp):
    pass


class LessEqual(RelationalOp, LessEqualOp):
    pass


class GreaterEqual(RelationalOp, GreaterEqualOp):
    pass


class LogicalNot(Model, LogicalNotOp):
    input: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class BitwiseOperators(Model):
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


class LogicalAnd(BitwiseOperators, LogicalAndOp):
    pass


class LogicalOr(BitwiseOperators, LogicalOrOp):
    pass


class LogicalXOr(BitwiseOperators, LogicalXOrOp):
    pass


class ShiftLeft(Model, ShiftLeftOp):
    input: Connection
    shift: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shift: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shift=shift, output=output)


class ShiftRight(Model, ShiftRightOp):
    input: Connection
    shift: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        shift: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, shift=shift, output=output)


class Transpose(Model, TransposeOp):
    # NOTE: Consider if axes type list[int] is conventionally True since it is generally
    # used tuple[int] in these type of cases
    input: Connection
    axes: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        axes: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, axes=axes, output=output)


class Split(Model, SplitOp):
    split_size: Connection
    axis: Connection
    input: Connection
    output: Connection

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


class Slice(Model, SliceOp):
    start: Connection
    stop: Connection
    step: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        start: ConnectionType = NOT_GIVEN,
        stop: ConnectionType = NOT_GIVEN,
        step: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(start=start, stop=stop, step=step, output=output)


class Indexer(Model, IndexerOp):
    input: Connection
    index: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        index: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, index=index, output=output)


class SingleInputOperation(Model):
    input: Connection
    output: Connection

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)


class Sine(SingleInputOperation, SineOp):
    pass


class Cosine(SingleInputOperation, CosineOp):
    pass
