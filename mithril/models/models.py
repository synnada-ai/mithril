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

import math
from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy

from ..framework import Model
from ..framework.common import (
    NOT_GIVEN,
    TBD,
    Connection,
    ConnectionType,
    IOKey,
    ShapeTemplateType,
    ToBeDetermined,
)
from ..framework.logical.base import ExtendInfo
from ..framework.logical.essential_primitives import (
    Absolute,
    Add,
    ArgMax,
    Buffer,
    Cast,
    Divide,
    DType,
    Greater,
    Length,
    MatrixMultiply,
    Mean,
    Minus,
    Multiply,
    Power,
    Reshape,
    ScalarItem,
    Shape,
    Size,
    Sqrt,
    Subtract,
    Sum,
    TensorSlice,
    Transpose,
    Variance,
)
from ..framework.logical.primitive import BaseModel, PrimitiveModel
from ..utils.utils import PaddingType, convert_to_list, convert_to_tuple
from .primitives import (
    AUCCore,
    CartesianDifference,
    Cholesky,
    Concat,
    DistanceMatrix,
    Eigvalsh,
    Exponential,
    Eye,
    EyeComplement,
    GPRAlpha,
    GPRVOuter,
    KLDivergence,
    Log,
    NormModifier,
    PaddingConverter1D,
    PaddingConverter2D,
    PermuteTensor,
    PolynomialFeatures,
    PrimitiveConvolution1D,
    PrimitiveConvolution2D,
    PrimitiveMaxPool1D,
    PrimitiveMaxPool2D,
    Sigmoid,
    Sign,
    Square,
    Squeeze,
    StableReciprocal,
    StrideConverter,
    Tanh,
    TransposedDiagonal,
    Trapezoid,
    TsnePJoint,
    TupleConverter,
    Unique,
    Where,
)

__all__ = [
    "Linear",
    "ElementWiseAffine",
    "Layer",
    "LayerNorm",
    "GroupNorm",
    "L1",
    "L2",
    "QuadraticFormRegularizer",
    "RBFKernel",
    "PolynomialKernel",
    "KernelizedSVM",
    "LinearSVM",
    "LogisticRegression",
    "MLP",
    "Cell",
    "RNNCell",
    "LSTMCell",
    "RNN",
    "OneToMany",
    "ManyToOne",
    "EncoderDecoder",
    "EncoderDecoderInference",
    "EncoderDistanceMatrix",
    "PolynomialRegression",
    "MDSCore",
    "MDS",
    "TSNE",
    "GaussProcessRegressionCore",
    "GPRLoss",
    "F1",
    "Precision",
    "Recall",
    "MaxPool1D",
    "MaxPool2D",
    "Convolution1D",
    "Convolution2D",
    "LSTMCellBody",
    "Cast",
    "DType",
    "Metric",
    "Accuracy",
    "AUC",
    "SiLU",
]


class Pool1D(Model):
    input: Connection
    kernel_size: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    @property
    def pool_model(self) -> type[BaseModel]:
        raise NotImplementedError("Pool Model should be indicated!")

    def __init__(
        self,
        kernel_size: int | ToBeDetermined,
        stride: int | None | ToBeDetermined = None,
        padding: int | PaddingType | tuple[int, int] | ToBeDetermined = (0, 0),
        dilation: int | ToBeDetermined = 1,
    ) -> None:
        super().__init__()

        self.factory_args = {
            "kernel_size": convert_to_list(kernel_size),
            "stride": convert_to_list(stride),
            "padding": convert_to_list(padding),
            "dilation": convert_to_list(dilation),
        }
        pad = convert_to_tuple(padding) if isinstance(padding, list) else padding

        stride_conv = StrideConverter()
        pad_conv = PaddingConverter1D()

        self += stride_conv(
            input=IOKey(name="stride", value=stride),
            kernel_size=IOKey(name="kernel_size", value=kernel_size),
        )

        self += pad_conv(
            input=IOKey(name="padding", value=pad), kernel_size="kernel_size"
        )

        self += self.pool_model()(
            input="input",
            kernel_size="kernel_size",
            stride=stride_conv.output,
            padding=pad_conv.output,
            dilation=IOKey(name="dilation", value=dilation),
            output=IOKey(name="output"),
        )

        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        kernel_size: ConnectionType = NOT_GIVEN,
        stride: ConnectionType = NOT_GIVEN,
        padding: ConnectionType = NOT_GIVEN,
        dilation: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class MaxPool1D(Pool1D):
    @property
    def pool_model(self) -> type[PrimitiveModel]:
        return PrimitiveMaxPool1D


# TODO: Implement MinPool1D and AvgPool1D
# class MinPool1D(Pool1D):
#     @property
#     def pool_model(self):
#         return PrimitiveMinPool1D

# class AvgPool1D(Pool1D):
#     @property
#     def pool_model(self):
#         return PrimitiveAvgPool1D


class Pool2D(Model):
    input: Connection
    kernel_size: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    @property
    def pool_model(self) -> type[BaseModel]:
        raise NotImplementedError("Pool Model should be indicated!")

    def __init__(
        self,
        kernel_size: int | tuple[int, int] | ToBeDetermined,
        stride: int | None | tuple[int, int] | ToBeDetermined = None,
        padding: int | PaddingType | tuple[int, int] | ToBeDetermined = (0, 0),
        dilation: int | ToBeDetermined = 1,
    ) -> None:
        super().__init__()

        self.factory_args = {
            "kernel_size": convert_to_list(kernel_size),
            "stride": convert_to_list(stride),
            "padding": convert_to_list(padding),
            "dilation": convert_to_list(dilation),
        }

        pad = convert_to_tuple(padding) if isinstance(padding, list) else padding
        if isinstance(stride, list):
            stride = convert_to_tuple(stride)
        if isinstance(kernel_size, list):
            kernel_size = convert_to_tuple(kernel_size)

        kt_converter = TupleConverter()
        s_converter = StrideConverter()
        st_converter = TupleConverter()
        p_converter = PaddingConverter2D()
        pt_converter = TupleConverter()
        dt_converter = TupleConverter()

        self += kt_converter(input=IOKey(name="kernel_size", value=kernel_size))
        self += s_converter(
            input=IOKey(name="stride", value=stride), kernel_size=kt_converter.output
        )
        self += st_converter(input=s_converter.output)
        self += p_converter(
            input=IOKey(name="padding", value=pad), kernel_size=kt_converter.output
        )
        self += pt_converter(input=p_converter.output)
        self += dt_converter(input=IOKey(name="dilation", value=dilation))
        self += self.pool_model()(
            input="input",
            kernel_size=kt_converter.output,
            stride=st_converter.output,
            padding=pt_converter.output,
            dilation=dt_converter.output,
            output=IOKey(name="output"),
        )

        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        kernel_size: ConnectionType = NOT_GIVEN,
        stride: ConnectionType = NOT_GIVEN,
        padding: ConnectionType = NOT_GIVEN,
        dilation: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class MaxPool2D(Pool2D):
    @property
    def pool_model(self) -> type[PrimitiveModel]:
        return PrimitiveMaxPool2D


# TODO: Implement MinPool2D and AvgPool2D
# class MinPool2D(Pool2D):
#     @property
#     def pool_model(self):
#         return PrimitiveMinPool2D

# class AvgPool2D(Pool2D):
#     @property
#     def pool_model(self):
#         return PrimitiveAvgPool2D


class Convolution1D(Model):
    input: Connection
    kernel: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    def __init__(
        self,
        kernel_size: int | None = None,
        out_channels: int | None = None,
        stride: int | ToBeDetermined = 1,
        padding: int | PaddingType | tuple[int, int] | ToBeDetermined = 0,
        dilation: int | ToBeDetermined = 1,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        self.factory_args = {
            "kernel_size": convert_to_list(kernel_size),
            "out_channels": out_channels,
            "stride": convert_to_list(stride),
            "padding": convert_to_list(padding),
            "dilation": convert_to_list(dilation),
            "use_bias": use_bias,
        }

        pad = convert_to_tuple(padding) if isinstance(padding, list) else padding

        k_shp = Shape()
        p_converter = PaddingConverter1D()

        self += k_shp(
            input=IOKey(name="kernel", shape=[out_channels, "C_in", kernel_size])
        )
        self += p_converter(
            input=IOKey(name="padding", value=pad), kernel_size=k_shp.output[-1]
        )

        conv_connections: dict[str, ConnectionType] = {
            "output": IOKey(name="output"),
            "input": "input",
            "kernel": "kernel",
            "stride": IOKey(name="stride", value=stride),
            "padding": p_converter.output,
            "dilation": IOKey(name="dilation", value=dilation),
        }
        if use_bias:
            conv_connections["bias"] = "bias"

        self += PrimitiveConvolution1D(use_bias=use_bias)(**conv_connections)

        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        kernel: ConnectionType = NOT_GIVEN,
        stride: ConnectionType = NOT_GIVEN,
        padding: ConnectionType = NOT_GIVEN,
        dilation: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class Convolution2D(Model):
    input: Connection
    kernel: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    def __init__(
        self,
        kernel_size: int | tuple[int, int] | None = None,
        out_channels: int | None = None,
        stride: int | tuple[int, int] | ToBeDetermined = (1, 1),
        padding: int
        | PaddingType
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = (0, 0),
        dilation: int | tuple[int, int] | ToBeDetermined = (1, 1),
        use_bias=True,
    ) -> None:
        super().__init__()

        self.factory_args = {
            "kernel_size": convert_to_list(kernel_size),
            "out_channels": out_channels,
            "stride": convert_to_list(stride),
            "padding": convert_to_list(padding),
            "dilation": convert_to_list(dilation),
            "use_bias": use_bias,
        }

        if isinstance(kernel_size, int | None):
            k_size = (kernel_size, kernel_size)
        else:
            k_size = kernel_size

        if isinstance(stride, list):
            stride = convert_to_tuple(stride)
        pad = convert_to_tuple(padding) if isinstance(padding, list) else padding
        if isinstance(dilation, list):
            dilation = convert_to_tuple(dilation)

        k_shp = Shape()
        p_converter = PaddingConverter2D()
        st_converter = TupleConverter()
        pt_converter = TupleConverter()
        dt_converter = TupleConverter()

        self += k_shp(input=IOKey(name="kernel", shape=[out_channels, "C_in", *k_size]))
        self += p_converter(
            input=IOKey(name="padding", value=pad), kernel_size=k_shp.output[-2:]
        )
        self += st_converter(input=IOKey(name="stride", value=stride))
        self += pt_converter(input=p_converter.output)
        self += dt_converter(input=IOKey(name="dilation", value=dilation))

        conv_connections: dict[str, ConnectionType] = {
            "output": IOKey(name="output"),
            "input": "input",
            "kernel": "kernel",
            "stride": st_converter.output,
            "padding": pt_converter.output,
            "dilation": dt_converter.output,
        }
        if use_bias:
            conv_connections["bias"] = "bias"

        self += PrimitiveConvolution2D(use_bias=use_bias)(**conv_connections)

        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        kernel: ConnectionType = NOT_GIVEN,
        stride: ConnectionType = NOT_GIVEN,
        padding: ConnectionType = NOT_GIVEN,
        dilation: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class Linear(Model):
    output: Connection
    input: Connection
    w: Connection
    b: Connection

    def __init__(self, dimension: int | None = None, use_bias=True) -> None:
        super().__init__()
        self.factory_args = {"dimension": dimension, "use_bias": use_bias}
        dim: int | str = "d_out" if dimension is None else dimension
        shapes: dict[str, ShapeTemplateType] = {
            "input": ["N", ("Var_inter", ...), "d_in"],
            "w": ["d_in", dim],
            "output": ["N", ("Var_inter", ...), dim],
        }

        output_key = IOKey(name="output")
        mult = MatrixMultiply()

        if use_bias:
            self += mult(left="input", right="w")
            self += Add()(left=mult.output, right="b", output=output_key)
            shapes["b"] = [dim]
        else:
            self += mult(left="input", right="w", output=output_key)

        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        w: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        b: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input": input, "w": w, "output": output}

        if "b" not in self._input_keys and b != NOT_GIVEN:
            raise KeyError("b is not a valid input when 'use_bias' is False!")
        elif "b" in self._input_keys:
            kwargs["b"] = b

        return super().__call__(**kwargs)


class ElementWiseAffine(Model):
    def __init__(self) -> None:
        super().__init__()
        mult_model = Multiply()
        sum_model = Add()

        self += mult_model(left="input", right="w")
        self += sum_model(
            left=mult_model.output, right="b", output=IOKey(name="output")
        )

        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        w: ConnectionType = NOT_GIVEN,
        b: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            w=w,
            b=b,
            output=output,
        )


class Layer(Model):
    input: Connection
    w: Connection
    b: Connection
    output: Connection

    def __init__(self, activation: BaseModel, dimension: int | None = None) -> None:
        super().__init__()
        self.factory_args = {"activation": activation, "dimension": dimension}
        linear_model = Linear(dimension=dimension)
        self += linear_model(input="input", w="w", b="b")
        self += activation(input=linear_model.output, output=IOKey(name="output"))

        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        w: ConnectionType = NOT_GIVEN,
        b: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            w=w,
            b=b,
            output=output,
        )


class LayerNorm(Model):
    input: Connection
    output: Connection
    w: Connection
    b: Connection

    def __init__(self, use_scale=True, use_bias=True, eps=1e-5) -> None:
        super().__init__()
        self.factory_args = {"use_scale": use_scale, "use_bias": use_bias, "eps": eps}

        # Expects its input shape as [B, ..., d] d refers to normalized dimension
        mean = Mean(axis=-1, keepdim=True)
        numerator = Subtract()
        var = Variance(axis=-1, correction=0, keepdim=True)
        add = Add()
        denominator = Sqrt()

        self += mean(input="input")
        self += numerator(left="input", right=mean.output)
        self += var(input="input")
        self += add(left=var.output, right=eps)
        self += denominator(input=add.output)
        self += Divide()(numerator=numerator.output, denominator=denominator.output)

        self._set_shapes({"input": ["B", "C", "d"]})

        shapes: dict[str, ShapeTemplateType] = {
            "left": ["B", "C", "d"],
            "right": ["d"],
        }

        if use_scale:
            mult = Multiply()
            self += mult(left=self.canonical_output, right="w")
            mult._set_shapes(shapes)

        if use_bias:
            add = Add()
            self += add(left=self.canonical_output, right="b")
            add._set_shapes(shapes)

        self += Buffer()(input=self.canonical_output, output=IOKey(name="output"))

        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        w: ConnectionType = NOT_GIVEN,
        b: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output}

        if "w" not in self._input_keys and w != NOT_GIVEN:
            raise KeyError("w is not a valid input when 'use_scale' is False!")
        elif "w" in self._input_keys:
            kwargs["w"] = w

        if "b" not in self._input_keys and b != NOT_GIVEN:
            raise KeyError("b is not a valid input when 'use_bias' is False!")
        elif "b" in self._input_keys:
            kwargs["b"] = b

        return super().__call__(**kwargs)


class GroupNorm(Model):
    def __init__(
        self, num_groups: int = 32, use_scale=True, use_bias=True, eps=1e-5
    ) -> None:
        super().__init__()

        # Assumed input shape is [N, C, H, W]
        input = IOKey(name="input")

        input_shape = input.shape()
        B = input_shape[0]

        input = input.reshape((B, num_groups, -1))

        mean = input.mean(axis=-1, keepdim=True)
        var = input.var(axis=-1, keepdim=True)

        input = (input - mean) / (var + eps).sqrt()
        self += Reshape()(input=input, shape=input_shape)

        self._set_shapes({"input": ["B", "C", "H", "W"]})

        shapes: dict[str, ShapeTemplateType] = {
            "left": ["B", "C", "H", "W"],
            "right": [1, "C", 1, 1],
        }

        if use_scale:
            mult = Multiply()
            self += mult(left=self.canonical_output, right="weight")
            mult._set_shapes(shapes)

        if use_bias:
            add = Add()
            self += add(left=self.canonical_output, right="bias")
            add._set_shapes(shapes)

        self += Buffer()(input=self.canonical_output, output=IOKey(name="output"))

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        w: ConnectionType = NOT_GIVEN,
        b: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output}

        if "w" not in self._input_keys and w != NOT_GIVEN:
            raise KeyError("w is not a valid input when 'use_scale' is False!")
        elif "w" in self._input_keys:
            kwargs["w"] = w

        if "b" not in self._input_keys and b != NOT_GIVEN:
            raise KeyError("b is not a valid input when 'use_bias' is False!")
        elif "b" in self._input_keys:
            kwargs["b"] = b

        return super().__call__(**kwargs)


class L1(Model):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__()

        abs_model = Absolute()

        self += abs_model(input="input")
        self += Sum()(input=abs_model.output, output=IOKey(name="output"))

        self._freeze()

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            output=output,
        )


class L2(Model):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__()

        square = Square()
        sum = Sum()

        self += square(input="input")
        self += sum(input=square.output)
        self += Multiply()(left=sum.output, right=0.5, output=IOKey(name="output"))

        self._freeze()

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            output=output,
        )


class QuadraticFormRegularizer(Model):
    input: Connection
    kernel: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__()

        transpose_model = Transpose()
        dot_model1 = MatrixMultiply()
        dot_model2 = MatrixMultiply()

        self += transpose_model(input="input")
        self += dot_model1(left=transpose_model.output, right="kernel")
        self += dot_model2(left=dot_model1.output, right=transpose_model.input)
        self += Multiply()(
            left=dot_model2.output, right=0.5, output=IOKey(name="output")
        )
        shapes: dict[str, ShapeTemplateType] = {"input": ["N", 1], "kernel": ["N", "N"]}
        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        kernel: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            kernel=kernel,
            output=output,
        )


class RBFKernel(Model):
    input1: Connection
    input2: Connection
    l_scale: Connection
    sigma: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__()

        euclidean_model = CartesianDifference()
        square_model1 = Square()
        square_model2 = Square()
        sum_model = Sum(axis=2)
        mult_model1 = Multiply()
        div_model = Divide()
        exp_model = Exponential()
        mult_model2 = Multiply()
        l_square = Multiply()

        self += euclidean_model(left="input1", right="input2")
        self += square_model1(input=euclidean_model.output)
        self += sum_model(input=square_model1.output)
        self += mult_model1(left=sum_model.output, right=-0.5)
        self += square_model2(input="sigma")
        self += div_model(
            numerator=mult_model1.output, denominator=square_model2.output
        )
        self += exp_model(input=div_model.output)
        self += l_square(left="l_scale", right="l_scale")
        self += mult_model2(
            left=l_square.output, right=exp_model.output, output=IOKey(name="output")
        )

        self.set_canonical_input("input1")
        shapes: dict[str, ShapeTemplateType] = {
            "input1": ["N", "dim"],
            "input2": ["M", "dim"],
            "l_scale": [1],
            "sigma": [1],
            "output": ["N", "M"],
        }
        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        l_scale: ConnectionType = NOT_GIVEN,
        sigma: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input1=input1,
            input2=input2,
            l_scale=l_scale,
            sigma=sigma,
            output=output,
        )


class PolynomialKernel(Model):
    input1: Connection
    input2: Connection
    poly_coef: Connection
    degree: Connection
    output: Connection

    def __init__(self, robust: bool = True) -> None:
        super().__init__()

        transpose_model = Transpose()
        mult_model = MatrixMultiply()
        sum_model = Add()
        power_model = Power(robust=robust)  # TODO: Should it be usual Power or not???

        self += transpose_model(input="input2")
        self += mult_model(left="input1", right=transpose_model.output)
        self += sum_model(left=mult_model.output, right="poly_coef")
        self += power_model(
            base=sum_model.output, exponent="degree", output=IOKey(name="output")
        )

        self._set_shapes(
            {
                "input1": ["N", "d"],
                "input2": ["M", "d"],
                "poly_coef": [],
                "degree": [],
                "output": ["N", "M"],
            }
        )
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        poly_coef: ConnectionType = NOT_GIVEN,
        degree: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input1=input1,
            input2=input2,
            poly_coef=poly_coef,
            degree=degree,
            output=output,
        )


class KernelizedSVM(Model):
    input1: Connection
    input2: Connection
    w: Connection
    b: Connection
    output: Connection

    def __init__(self, kernel: BaseModel) -> None:
        if len(kernel._input_keys) < 2:
            raise KeyError("Kernel requires at least two inputs!")
        if len(kernel.conns.output_keys) != 1:
            raise KeyError("Kernel requires single output!")
        super().__init__()
        self.factory_args = {"kernel": kernel}

        linear_model = Linear()
        # Get kernel inputs from given model.
        kernel_input_args = {
            key: key
            for key in kernel._input_keys
            if not kernel.conns.is_key_non_diff(key)
        }
        (kernel_output_name,) = kernel.conns.output_keys  # NOTE: Assumes single output!
        kernel_output_args = {kernel_output_name: IOKey(name="kernel")}

        self += kernel(**kernel_input_args, **kernel_output_args)
        self += linear_model(
            input=kernel.canonical_output,
            w="w",
            b="b",
            output=IOKey(name="output"),
        )

        shapes: dict[str, ShapeTemplateType] = {
            "input1": ["N", "d_in"],
            "input2": ["M", "d_in"],
            "w": ["M", 1],
            "b": [1],
            "output": ["N", 1],
            "kernel": ["N", "M"],
        }
        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        w: ConnectionType = NOT_GIVEN,
        b: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input1=input1,
            input2=input2,
            w=w,
            b=b,
            output=output,
        )


class LinearSVM(Model):
    input: Connection
    w: Connection
    b: Connection
    output: Connection
    decision_output: Connection

    def __init__(
        self,
    ) -> None:
        super().__init__()

        linear_model = Linear(dimension=1)
        decision_model = Sign()

        self += linear_model(input="input", w="w", b="b", output=IOKey(name="output"))
        self += decision_model(
            input=linear_model.output, output=IOKey(name="decision_output")
        )

        self.set_canonical_output(linear_model.output)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        w: ConnectionType = NOT_GIVEN,
        b: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        decision_output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            w=w,
            b=b,
            output=output,
            decision_output=decision_output,
        )


class LogisticRegression(Model):
    input: Connection
    w: Connection
    b: Connection
    output: Connection
    probs_output: Connection

    def __init__(self) -> None:
        super().__init__()

        linear_model = Linear(dimension=1)
        sigmoid_model = Sigmoid()

        self += linear_model(input="input", w="w", b="b", output=IOKey(name="output"))
        self += sigmoid_model(
            input=linear_model.output, output=IOKey(name="probs_output")
        )

        self.set_canonical_output(linear_model.output)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        w: ConnectionType = NOT_GIVEN,
        b: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        probs_output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            w=w,
            b=b,
            output=output,
            probs_output=probs_output,
        )


class MLP(Model):
    input: Connection
    output: Connection

    def __init__(
        self,
        activations: list[BaseModel],
        dimensions: Sequence[int | None],
        input_name_templates: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.factory_args = {"activations": activations, "dimensions": dimensions}
        if len(activations) != len(dimensions):
            raise ValueError("Lengths of activations and dimensions must be equal!")
        assert len(activations) > 0, "At least one layer must be defined!"

        # Extract the keys to be used. Use "w" and "b" for default case.
        input_name_templates = input_name_templates or {}
        weight = input_name_templates.get("weight", "w")
        bias = input_name_templates.get("bias", "b")

        # Create first layer.
        prev_layer = Layer(activation=activations[0], dimension=dimensions[0])

        # Add first layer to the model in order to use as base for the
        # second model in the model extention loop.
        extend_kwargs: dict[str, ConnectionType] = {
            "input": "input",
            "w": weight + "0",
            "b": bias + "0",
        }
        if len(activations) == 1:
            extend_kwargs["output"] = IOKey(name="output")
        self += prev_layer(**extend_kwargs)

        # Add layers sequentially starting from second elements.
        for idx, (activation, dim) in enumerate(
            zip(activations[1:], dimensions[1:], strict=False)
        ):
            current_layer = Layer(activation=activation, dimension=dim)

            # Prepare the kwargs for the current layer.
            kwargs: dict[str, ConnectionType] = {
                "input": prev_layer.output,
                "w": f"{weight}{idx + 1}",
                "b": f"{bias}{idx + 1}",
            }

            # In order to make last layer output as model output we must name it.
            if idx == (
                len(activations) - 2
            ):  # Loop starts to iterate from second elemets, so it is -2.
                kwargs |= {"output": IOKey(name="output")}

            # Add current layer to the model.
            self += current_layer(**kwargs)
            prev_layer = current_layer

        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        **weights_biases: ConnectionType,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            output=output,
            **weights_biases,
        )


class Cell(Model):
    shared_keys: set[str]
    private_keys: set[str]
    state_keys: set[str]
    hidden_key: str

    input: Connection
    prev_hidden: Connection
    hidden: Connection
    hidden_compl: Connection
    output: Connection

    out_key = "output"

    @abstractmethod
    def __call__(self, **kwargs: ConnectionType) -> ExtendInfo:
        raise NotImplementedError("__call__ method not implemented!")


class RNNCell(Cell):
    input: Connection
    prev_hidden: Connection
    w_ih: Connection
    w_hh: Connection
    w_ho: Connection
    bias_h: Connection
    bias_o: Connection
    hidden: Connection
    hidden_compl: Connection
    output: Connection

    shared_keys = {"w_ih", "w_hh", "w_ho", "bias_h", "bias_o"}
    state_keys = {"hidden"}
    out_key = "output"
    # output_keys = {out, hidden_compl}

    def __init__(self) -> None:
        super().__init__()

        shape = Shape()
        scalar_item = ScalarItem()
        slice_model = TensorSlice(stop=TBD)
        mult_model_1 = MatrixMultiply()
        mult_model_2 = MatrixMultiply()
        mult_model_3 = MatrixMultiply()
        sum_model_1 = Add()
        sum_model_2 = Add()

        self += shape(input="input")
        self += scalar_item(input=shape.output, index=0)
        self += TensorSlice(start=TBD)(
            input="prev_hidden",
            start=scalar_item.output,
            output=IOKey(name="hidden_compl"),
        )
        self += slice_model(input="prev_hidden", stop=scalar_item.output)
        self += mult_model_1(left=slice_model.output, right="w_hh")
        self += mult_model_2(left="input", right="w_ih")
        self += sum_model_1(left=mult_model_1.output, right=mult_model_2.output)
        self += sum_model_2(left=sum_model_1.output, right="bias_h")
        self += Tanh()(input=sum_model_2.output, output=IOKey(name="hidden"))
        self += mult_model_3(left="hidden", right="w_ho")
        self += Add()(
            left=mult_model_3.output, right="bias_o", output=IOKey(name="output")
        )
        shapes: dict[str, ShapeTemplateType] = {
            "input": ["N", 1, "d_in"],
            "prev_hidden": ["M", 1, "d_hid"],
            "w_ih": ["d_in", "d_hid"],
            "w_hh": ["d_hid", "d_hid"],
            "w_ho": ["d_hid", "d_out"],
            "bias_h": ["d_hid"],
            "bias_o": ["d_out"],
        }
        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        prev_hidden: ConnectionType = NOT_GIVEN,
        w_ih: ConnectionType = NOT_GIVEN,
        w_hh: ConnectionType = NOT_GIVEN,
        w_ho: ConnectionType = NOT_GIVEN,
        bias_h: ConnectionType = NOT_GIVEN,
        bias_o: ConnectionType = NOT_GIVEN,
        hidden: ConnectionType = NOT_GIVEN,
        hidden_compl: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super(Cell, self).__call__(
            input=input,
            prev_hidden=prev_hidden,
            w_ih=w_ih,
            w_hh=w_hh,
            w_ho=w_ho,
            bias_h=bias_h,
            bias_o=bias_o,
            hidden=hidden,
            hidden_compl=hidden_compl,
            output=output,
        )


class LSTMCell(Cell):
    input: Connection
    prev_hidden: Connection
    prev_cell: Connection
    w_i: Connection
    w_f: Connection
    w_c: Connection
    w_o: Connection
    w_out: Connection
    bias_f: Connection
    bias_i: Connection
    bias_c: Connection
    bias_o: Connection
    bias_out: Connection
    hidden: Connection
    cell: Connection
    hidden_compl: Connection
    output: Connection

    shared_keys = {
        "w_f",
        "w_i",
        "w_o",
        "w_c",
        "w_out",
        "bias_f",
        "bias_i",
        "bias_o",
        "bias_c",
        "bias_out",
    }
    state_keys = {"hidden", "cell"}
    out_key = "output"

    def __init__(self) -> None:
        super().__init__()

        cell_body = LSTMCellBody()
        shape_model = Shape()
        scalar_item = ScalarItem(index=TBD)
        slice_model_1 = TensorSlice(stop=TBD)
        slice_model_2 = TensorSlice(stop=TBD)
        slice_model_3 = TensorSlice(start=TBD)
        slice_model_4 = TensorSlice(stop=TBD)

        self += shape_model(input="input")
        self += scalar_item(input=shape_model.output, index=0)

        # Forget gate processes.
        self += slice_model_1(input="prev_cell", stop=scalar_item.output)
        self += slice_model_2(input="prev_hidden", stop=scalar_item.output)

        body_kwargs: dict[str, ConnectionType] = {
            key: key for key in cell_body._input_keys if key[0] != "$"
        }
        body_kwargs["prev_cell"] = slice_model_1.output
        body_kwargs["prev_hidden"] = slice_model_2.output

        self += cell_body(**body_kwargs)

        self += slice_model_3(
            input=cell_body.output,
            start=scalar_item.output,
            output=IOKey(name="hidden"),
        )

        self += slice_model_4(
            input=cell_body.output, stop=scalar_item.output, output=IOKey(name="cell")
        )

        # Slice complement process.
        self += TensorSlice(start=TBD)(
            input="prev_hidden",
            start=scalar_item.output,
            output=IOKey(name="hidden_compl"),
        )
        # Final output.
        self += Linear()(
            input="hidden",
            w="w_out",
            b="bias_out",
            output=IOKey(name="output"),
        )
        shapes: dict[str, ShapeTemplateType] = {
            "input": ["N", 1, "d_in"],
            "prev_hidden": ["M", 1, "d_hid"],
            "prev_cell": ["M", 1, "d_hid"],
            "w_i": ["d_sum", "d_hid"],
            "w_f": ["d_sum", "d_hid"],
            "w_c": ["d_sum", "d_hid"],
            "w_o": ["d_sum", "d_hid"],
            "w_out": ["d_hid", "d_out"],
            "bias_f": ["d_hid"],
            "bias_i": ["d_hid"],
            "bias_c": ["d_hid"],
            "bias_o": ["d_hid"],
            "bias_out": ["d_out"],
            "hidden": ["N", 1, "d_hid"],
            "cell": ["N", 1, "d_hid"],
        }
        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        prev_hidden: ConnectionType = NOT_GIVEN,
        prev_cell: ConnectionType = NOT_GIVEN,
        w_i: ConnectionType = NOT_GIVEN,
        w_f: ConnectionType = NOT_GIVEN,
        w_c: ConnectionType = NOT_GIVEN,
        w_o: ConnectionType = NOT_GIVEN,
        w_out: ConnectionType = NOT_GIVEN,
        bias_f: ConnectionType = NOT_GIVEN,
        bias_i: ConnectionType = NOT_GIVEN,
        bias_c: ConnectionType = NOT_GIVEN,
        bias_o: ConnectionType = NOT_GIVEN,
        bias_out: ConnectionType = NOT_GIVEN,
        hidden: ConnectionType = NOT_GIVEN,
        cell: ConnectionType = NOT_GIVEN,
        hidden_compl: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super(Cell, self).__call__(
            input=input,
            prev_hidden=prev_hidden,
            prev_cell=prev_cell,
            w_i=w_i,
            w_f=w_f,
            w_c=w_c,
            w_o=w_o,
            w_out=w_out,
            bias_f=bias_f,
            bias_i=bias_i,
            bias_c=bias_c,
            bias_o=bias_o,
            bias_out=bias_out,
            hidden=hidden,
            cell=cell,
            hidden_compl=hidden_compl,
            output=output,
        )


class LSTMCellBody(Model):
    input: Connection
    prev_hidden: Connection
    prev_cell: Connection
    w_i: Connection
    w_f: Connection
    w_c: Connection
    w_o: Connection
    bias_f: Connection
    bias_i: Connection
    bias_c: Connection
    bias_o: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(formula_key="lstm_cell")

        matrix_concat_model = Concat(n=2, axis=-1)
        forward_lin = Linear()
        sigmoid_model_1 = Sigmoid()
        mult_model_1 = Multiply()
        input_lin = Linear()
        sigmoid_model_2 = Sigmoid()
        cell_lin = Linear()
        tanh_model_1 = Tanh()
        mult_model_2 = Multiply()
        sum_model_4 = Add()
        tanh_model_2 = Tanh()
        out_gate_lin = Linear()
        sigmoid_model_3 = Sigmoid()
        mult_model_3 = Multiply()

        self += matrix_concat_model(input1="input", input2="prev_hidden")
        self += forward_lin(input=matrix_concat_model.output, w="w_f", b="bias_f")
        self += sigmoid_model_1(input=forward_lin.output)
        self += mult_model_1(left="prev_cell", right=sigmoid_model_1.output)
        # Input gate processes.
        self += input_lin(input=matrix_concat_model.output, w="w_i", b="bias_i")
        self += sigmoid_model_2(input=input_lin.output)
        # Cell state gate processes.
        self += cell_lin(input=matrix_concat_model.output, w="w_c", b="bias_c")
        self += tanh_model_1(input=cell_lin.output)
        # Input-cell gate multiplication.
        self += mult_model_2(left=sigmoid_model_2.output, right=tanh_model_1.output)
        # Addition to cell state.
        self += sum_model_4(left=mult_model_1.output, right=mult_model_2.output)
        # Cell state to hidden state info.
        self += tanh_model_2(input=sum_model_4.output)
        # Output gate process.
        self += out_gate_lin(input=matrix_concat_model.output, w="w_o", b="bias_o")
        self += sigmoid_model_3(input=out_gate_lin.output)
        # Final hidden state.
        self += mult_model_3(left=tanh_model_2.output, right=sigmoid_model_3.output)
        self += Concat(n=2, axis=0)(
            input1=sum_model_4.output,
            input2=mult_model_3.output,
            output=IOKey(name="output"),
        )
        shapes: dict[str, ShapeTemplateType] = {
            "input": ["N", 1, "d_in"],
            "prev_hidden": ["N", 1, "d_hid"],
            "prev_cell": ["N", 1, "d_hid"],
            "w_i": ["d_sum", "d_hid"],
            "w_f": ["d_sum", "d_hid"],
            "w_c": ["d_sum", "d_hid"],
            "w_o": ["d_sum", "d_hid"],
            "bias_f": ["d_hid"],
            "bias_i": ["d_hid"],
            "bias_c": ["d_hid"],
            "bias_o": ["d_hid"],
        }

        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        prev_hidden: ConnectionType = NOT_GIVEN,
        prev_cell: ConnectionType = NOT_GIVEN,
        w_i: ConnectionType = NOT_GIVEN,
        w_f: ConnectionType = NOT_GIVEN,
        w_c: ConnectionType = NOT_GIVEN,
        w_o: ConnectionType = NOT_GIVEN,
        bias_f: ConnectionType = NOT_GIVEN,
        bias_i: ConnectionType = NOT_GIVEN,
        bias_c: ConnectionType = NOT_GIVEN,
        bias_o: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            prev_hidden=prev_hidden,
            prev_cell=prev_cell,
            w_i=w_i,
            w_f=w_f,
            w_c=w_c,
            w_o=w_o,
            bias_f=bias_f,
            bias_i=bias_i,
            bias_c=bias_c,
            bias_o=bias_o,
            output=output,
        )


class RNN(Model):
    def __init__(
        self,
        cell_type: Cell,
    ) -> None:
        self.cell_type = cell_type
        super().__init__()

    def __call__(self, **kwargs) -> ExtendInfo:  # type: ignore[override]
        raise NotImplementedError("__call__ method not implemented!")


class OneToMany(RNN):
    input: Connection

    def __init__(
        self, cell_type: Cell, max_sequence_length: int, teacher_forcing: bool = False
    ) -> None:
        super().__init__(cell_type=cell_type)
        cell = deepcopy(cell_type)
        prev_cell = cell

        shared_keys_kwargs = {key: key for key in cell_type.shared_keys}
        output_kwargs = {cell_type.out_key: IOKey(name="output0")}
        input_kwargs: dict[str, ConnectionType] = {"input": "input"}
        initial_state_kwargs = {
            f"prev_{key}": f"initial_{key}" for key in cell_type.state_keys
        }

        self += prev_cell(
            **(input_kwargs | shared_keys_kwargs | output_kwargs | initial_state_kwargs)
        )

        for idx in range(1, max_sequence_length):
            current_cell = deepcopy(cell_type)
            state_keys_kwargs = {
                f"prev_{key}": getattr(prev_cell, key) for key in cell_type.state_keys
            }
            # Create slicing model which filters unnecessary data for
            # current time step.
            shape_model = Shape()
            item_model = ScalarItem()
            slice_model = TensorSlice(stop=TBD)

            self += shape_model(input=f"target{idx}")
            self += item_model(input=shape_model.output, index=0)

            # # Create slicing model which filters unnecessary data for
            # # current time step.
            if teacher_forcing:
                # Teacher forcing apporach requires targets of  previous
                # time step as inputs to the current time step.
                slice_input_1 = f"target{idx - 1}"
            else:
                # When not using teacher forcing, simply take outputs
                # of previous time step as inputs to the current time step.
                slice_input_1 = getattr(prev_cell, prev_cell.out_key)

            self += slice_model(input=slice_input_1, stop=item_model.output)

            input_kwargs = {"input": slice_model.output}
            output_kwargs = {cell_type.out_key: IOKey(name=f"output{idx}")}

            self += current_cell(
                **(
                    input_kwargs
                    | shared_keys_kwargs
                    | state_keys_kwargs
                    | output_kwargs
                )
            )

            prev_cell = current_cell
        self._freeze()

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, **model_keys: ConnectionType
    ) -> ExtendInfo:
        return super(RNN, self).__call__(input=input, **model_keys)


class OneToManyInference(RNN):
    input: Connection

    def __init__(self, cell_type: Cell, max_sequence_length: int) -> None:
        super().__init__(cell_type=cell_type)
        cell = deepcopy(cell_type)
        prev_cell = cell

        shared_keys_kwargs = {key: key for key in cell_type.shared_keys}
        output_kwargs = {cell_type.out_key: IOKey(name="output0")}
        input_kwargs: dict[str, ConnectionType] = {"input": "input"}
        initial_state_kwargs = {
            f"prev_{key}": f"initial_{key}" for key in cell_type.state_keys
        }

        self += prev_cell(
            **(input_kwargs | shared_keys_kwargs | output_kwargs | initial_state_kwargs)
        )

        for idx in range(1, max_sequence_length):
            current_cell = deepcopy(cell_type)

            state_keys_kwargs = {
                f"prev_{key}": getattr(prev_cell, key) for key in cell_type.state_keys
            }
            input_kwargs = {"input": prev_cell.output}
            output_kwargs = {cell_type.out_key: IOKey(name=f"output{idx}")}

            self += current_cell(
                **(
                    input_kwargs
                    | shared_keys_kwargs
                    | state_keys_kwargs
                    | output_kwargs
                )
            )

            prev_cell = current_cell
        self._freeze()

    def __call__(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, **model_keys: ConnectionType
    ) -> ExtendInfo:
        return super(RNN, self).__call__(input=input, **model_keys)


class ManyToOne(RNN):
    hidden_concat: Connection

    def __init__(self, cell_type: Cell, max_sequence_length: int) -> None:
        super().__init__(cell_type)
        prev_cell = deepcopy(cell_type)

        concat_model = Concat(n=max_sequence_length)
        concat_input_kwargs: dict[str, ConnectionType] = {}
        shared_keys_kwargs = {key: key for key in cell_type.shared_keys}
        output_kwargs = {cell_type.out_key: IOKey(name="output0")}
        input_kwargs = {"input": "input0"}
        initial_state_kwargs = {
            f"prev_{key}": f"initial_{key}" for key in cell_type.state_keys
        }

        self += prev_cell(
            **(input_kwargs | shared_keys_kwargs | output_kwargs | initial_state_kwargs)
        )

        for idx in range(1, max_sequence_length):
            cur_cell = deepcopy(cell_type)
            state_keys_kwargs = {
                f"prev_{key}": getattr(prev_cell, key) for key in cell_type.state_keys
            }
            input_kwargs = {"input": f"input{idx}"}
            output_kwargs = {cell_type.out_key: IOKey(name=f"output{idx}")}

            # For the last cell, include hidden
            self += cur_cell(
                **(
                    input_kwargs
                    | shared_keys_kwargs
                    | state_keys_kwargs
                    | output_kwargs
                )
            )

            # # For the last cell, include hidden
            if idx < max_sequence_length - 1:
                concat_input_kwargs |= {
                    f"input{max_sequence_length - idx + 1}": cur_cell.hidden_compl
                }

            else:
                concat_input_kwargs |= {
                    "input2": cur_cell.hidden_compl,
                    "input1": cur_cell.hidden,
                }

            prev_cell = cur_cell

        # Add concat model with accumulated hidden states.
        self += concat_model(**concat_input_kwargs, output=IOKey(name="hidden_concat"))

        self._freeze()

    def __call__(  # type: ignore[override]
        self, hidden_concat: ConnectionType = NOT_GIVEN, **model_keys: ConnectionType
    ) -> ExtendInfo:
        return super(RNN, self).__call__(hidden_concat=hidden_concat, **model_keys)


class EncoderDecoder(Model):
    indices: Connection

    def __init__(
        self,
        cell_type: Cell,
        max_input_sequence_length: int,
        max_target_sequence_length: int,
        teacher_forcing=False,
    ) -> None:
        super().__init__()

        # Encoder Model
        encoder = ManyToOne(
            cell_type=cell_type, max_sequence_length=max_input_sequence_length
        )

        # Decoder Model
        decoder = OneToMany(
            cell_type=cell_type,
            max_sequence_length=max_target_sequence_length,
            teacher_forcing=teacher_forcing,
        )

        permutation_model = PermuteTensor()

        enc_input_mapping = {key: key for key in encoder._input_keys if "$" not in key}

        dec_input_mapping = {
            key: "decoder_" + key if "target" not in key else key
            for key in decoder._input_keys
            if "$" not in key and key != "initial_hidden"
        }

        dec_output_mapping = {key: IOKey(name=key) for key in decoder.conns.output_keys}

        self += encoder(**enc_input_mapping)
        self += permutation_model(input=encoder.hidden_concat, indices="indices")
        self += decoder(
            initial_hidden=permutation_model.output,
            **(dec_input_mapping | dec_output_mapping),
        )

        self._freeze()

    def __call__(  # type: ignore[override]
        self, indices: ConnectionType = NOT_GIVEN, **model_keys: ConnectionType
    ) -> ExtendInfo:
        return super().__call__(indices=indices, **model_keys)


class EncoderDecoderInference(Model):
    indices: Connection

    def __init__(
        self,
        cell_type: Cell,
        max_input_sequence_length: int,
        max_target_sequence_length: int,
    ) -> None:
        super().__init__()

        # Encoder Model
        encoder = ManyToOne(
            cell_type=cell_type, max_sequence_length=max_input_sequence_length
        )

        # Decoder Model
        decoder = OneToManyInference(
            cell_type=cell_type, max_sequence_length=max_target_sequence_length
        )

        enc_input_mapping = {key: key for key in encoder._input_keys if "$" not in key}

        dec_input_mapping = {
            key: "decoder_" + key if "target" not in key else key
            for key in decoder._input_keys
            if "$" not in key and key != "initial_hidden"
        }

        dec_output_mapping = {key: IOKey(name=key) for key in decoder.conns.output_keys}

        self += encoder(**enc_input_mapping)
        self += decoder(
            initial_hidden=encoder.hidden_concat,
            **(dec_input_mapping | dec_output_mapping),
        )

        self._freeze()

    def __call__(  # type: ignore[override]
        self, indices: ConnectionType = NOT_GIVEN, **model_keys: ConnectionType
    ) -> ExtendInfo:
        return super().__call__(indices=indices, **model_keys)


class EncoderDistanceMatrix(Model):
    input1: Connection
    input2: Connection
    norm: Connection
    output: Connection

    def __init__(self, get_final_distance: bool = True, robust: bool = True) -> None:
        super().__init__()
        self.factory_args = {"get_final_distance": get_final_distance, "robust": robust}

        dist_model = DistanceMatrix()
        modifier_model = NormModifier()

        if get_final_distance:
            reciprocal_model = Divide()
            power_model = Power(robust=robust)

            self += modifier_model(input="norm")
            self += dist_model(
                left="input1", right="input2", norm=modifier_model.output
            )
            self += reciprocal_model(numerator=1.0, denominator=modifier_model.output)
            self += power_model(
                base=dist_model.output,
                exponent=reciprocal_model.output,
                output=IOKey(name="output"),
            )

        else:
            self += modifier_model(input="norm")
            self += dist_model(
                left="input1",
                right="input2",
                norm=modifier_model.output,
                output=IOKey(name="output"),
            )

        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        norm: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input1=input1, input2=input2, norm=norm, output=output)


class PolynomialRegression(Model):
    input: Connection
    w: Connection
    b: Connection
    output: Connection

    def __init__(self, degree: int, dimension: int | None = None) -> None:
        super().__init__()
        self.factory_args = {"degree": degree, "dimension": dimension}

        linear_model = Linear(dimension=dimension)
        feature_model = PolynomialFeatures(degree=degree)

        self += feature_model(input="input")
        self += linear_model(
            input=feature_model.output, w="w", b="b", output=IOKey(name="output")
        )

        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        w: ConnectionType = NOT_GIVEN,
        b: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, w=w, b=b, output=output)


class MDSCore(Model):
    distances: Connection
    pred_distances: Connection
    norm: Connection
    output: Connection

    requires_norm: bool = True

    def __init__(self, exact_distances: bool = True, robust: bool = True) -> None:
        self.factory_args = {"exact_distances": exact_distances, "robust": robust}
        super().__init__()

        # Prepare models used in MDS.
        subtract_model = Subtract()
        abs_model = Absolute()
        norm_model = NormModifier()
        power_model_1 = Power(robust=robust)
        power_model_2 = Power(robust=robust)
        power_model_3 = Power(robust=robust)
        power_model_4 = Power(robust=robust)
        sum_model_1 = Sum()
        sum_model_2 = Sum()
        reciprocal_model_1 = StableReciprocal()
        reciprocal_model_2 = StableReciprocal()
        mult_model = Multiply()

        if exact_distances:
            self += norm_model(input="norm")
            self += reciprocal_model_1(input=norm_model.output)
            self += power_model_4(
                base="pred_distances", exponent=reciprocal_model_1.output
            )
            self += subtract_model(left="distances", right=power_model_4.output)
            self += abs_model(input=subtract_model.output)
            self += power_model_1(base=abs_model.output, exponent=norm_model.output)
            self += sum_model_1(input=power_model_1.output)
            self += power_model_2(base=self.distances, exponent=norm_model.output)
            self += sum_model_2(input=power_model_2.output)
            self += reciprocal_model_2(input=sum_model_2.output)
            self += mult_model(left=sum_model_1.output, right=reciprocal_model_2.output)
            self += power_model_3(
                base=mult_model.output,
                exponent=reciprocal_model_1.output,
                output=IOKey(name="output"),
            )

        else:
            self += norm_model(input="norm")
            self += reciprocal_model_1(input=norm_model.output)
            self += power_model_1(base="distances", exponent=reciprocal_model_1.output)
            self += power_model_4(
                base="pred_distances", exponent=reciprocal_model_1.output
            )
            self += subtract_model(
                left=power_model_1.output, right=power_model_4.output
            )
            self += abs_model(input=subtract_model.output)
            self += power_model_2(base=abs_model.output, exponent=norm_model.output)
            self += sum_model_1(input=power_model_2.output)
            self += sum_model_2(input=self.distances)
            self += reciprocal_model_2(input=sum_model_2.output)
            self += mult_model(left=sum_model_1.output, right=reciprocal_model_2.output)
            self += power_model_3(
                base=mult_model.output,
                exponent=reciprocal_model_1.output,
                output=IOKey(name="output"),
            )

        self._set_shapes({"distances": ["N", "N"], "pred_distances": ["N", "N"]})
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        distances: ConnectionType = NOT_GIVEN,
        pred_distances: ConnectionType = NOT_GIVEN,
        norm: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            distances=distances,
            pred_distances=pred_distances,
            norm=norm,
            output=output,
        )


class TSNECore(Model):
    distances: Connection
    pred_distances: Connection
    p_joint: Connection
    output: Connection

    requires_norm: bool = False

    def __init__(
        self,
        exact_distances: bool = True,
        calculate_p_joint: bool = False,
        perplexity: float = 20.0,
    ) -> None:
        super().__init__()
        self.factory_args = {
            "exact_distances": exact_distances,
            "perplexity": perplexity,
        }

        p_joint_model = TsnePJoint()
        divide_model_1 = Divide()
        divide_model_2 = Divide()
        sum_model_1 = Add()
        sum_model_2 = Sum()
        sum_model_3 = Sum()
        size_model = Size(dim=0)
        zero_diagonal_model = EyeComplement(N=TBD)
        mult_model = Multiply()
        kl_divergence_model = KLDivergence()

        # Always process with squared distances in TSNE calculations.
        if exact_distances:
            square_model = Square()
            self += square_model(input="distances")
            if calculate_p_joint:
                self += p_joint_model(
                    squared_distances=square_model.output, target_perplexity=perplexity
                )
            self += sum_model_1(left=1.0, right="pred_distances")
            self += divide_model_1(numerator=1.0, denominator=sum_model_1.output)
            self += size_model(input=getattr(self, "distances", "distances"))
            self += zero_diagonal_model(N=size_model.output)
            self += mult_model(
                left=divide_model_1.output, right=zero_diagonal_model.output
            )
            self += sum_model_2(input=mult_model.output)
            self += divide_model_2(
                numerator=mult_model.output, denominator=sum_model_2.output
            )
            self += kl_divergence_model(
                input=divide_model_2.output,
                target=p_joint_model.output if calculate_p_joint else "p_joint",
            )
            self += sum_model_3(
                input=kl_divergence_model.output, output=IOKey(name="output")
            )

        else:
            if calculate_p_joint:
                self += p_joint_model(
                    squared_distances="distances", target_perplexity=perplexity
                )
            self += sum_model_1(left=1.0, right="pred_distances")
            self += divide_model_1(numerator=1.0, denominator=sum_model_1.output)
            self += size_model(input=getattr(self, "distances", "distances"))
            self += zero_diagonal_model(N=size_model.output)
            self += mult_model(
                left=divide_model_1.output, right=zero_diagonal_model.output
            )
            self += sum_model_2(input=mult_model.output)
            self += divide_model_2(
                numerator=mult_model.output, denominator=sum_model_2.output
            )
            self += kl_divergence_model(
                input=divide_model_2.output,
                target=p_joint_model.output if calculate_p_joint else "p_joint",
            )
            self += sum_model_3(
                input=kl_divergence_model.output, output=IOKey(name="output")
            )

        self._set_shapes({"distances": ["N", "N"], "pred_distances": ["N", "N"]})
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        distances: ConnectionType = NOT_GIVEN,
        pred_distances: ConnectionType = NOT_GIVEN,
        p_joint: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "distances": distances,
            "pred_distances": pred_distances,
            "output": output,
        }

        if "p_joint" in self._input_keys:
            kwargs["p_joint"] = p_joint
        elif p_joint != NOT_GIVEN:
            raise ValueError("p_joint is only required when calculate_p_joint is True!")

        return super().__call__(**kwargs)


class DistanceEncoder(Model):
    input: Connection
    coords: Connection
    norm: Connection
    predicted_coords: Connection
    output: Connection

    ephemeral: bool = True

    def __init__(
        self, base_model: MDSCore | TSNECore, input_type: str = "distances"
    ) -> None:
        super().__init__()

        self.factory_args = {"base_model": base_model, "input_type": input_type}

        assert input_type in ["distances", "powered_distances", "points"]

        coords_distance_matrix = EncoderDistanceMatrix(get_final_distance=False)
        buffer_model = Buffer()

        # NOTE: We should assert a standard naming for inputs to
        #  the base model (i.e. "distances", "pred_distances")
        if input_type == "points":
            input_distance_matrix = EncoderDistanceMatrix(get_final_distance=False)
            self += input_distance_matrix(input1="input", input2="input", norm="norm")
            self += coords_distance_matrix(
                input1="coords", input2="coords", norm="norm"
            )

            base_kwargs: dict[str, ConnectionType] = {
                "distances": input_distance_matrix.output,
                "pred_distances": coords_distance_matrix.output,
                "output": IOKey(name="output"),
            }
            # Create inputs taking "requires_norm" attribute of base model class.
            if base_model.requires_norm:
                base_kwargs["norm"] = "norm"

            self += base_model(**base_kwargs)
            self += buffer_model(
                input=self.coords, output=IOKey(name="predicted_coords")
            )

        else:
            self += coords_distance_matrix(
                input1="coords", input2="coords", norm="norm"
            )

            # Create inputs taking "requires_norm" attribute of base model class.
            base_kwargs = {
                "distances": "input",
                "pred_distances": coords_distance_matrix.output,
                "output": IOKey(name="output"),
            }
            if base_model.requires_norm:
                base_kwargs["norm"] = "norm"

            self += base_model(**base_kwargs)
            self += buffer_model(
                input=self.coords, output=IOKey(name="predicted_coords")
            )

        # self._set_shapes(trace=False,
        #     input = ["N", "M"], # NOTE: Here "M" denotes input dim or
        #     sample size ("N") depending on input_type.
        #     coords = ["N", "d"]
        # )

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        coords: ConnectionType = NOT_GIVEN,
        norm: ConnectionType = NOT_GIVEN,
        predicted_coords: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "coords": coords,
            "norm": norm,
            "predicted_coords": predicted_coords,
            "output": output,
        }
        if "input" in self._input_keys:
            kwargs["input"] = input
        elif coords != NOT_GIVEN:
            raise ValueError("coords is only required when input_type is 'points'!")

        return super().__call__(**kwargs)


class MDS(DistanceEncoder):
    input: Connection
    coords: Connection
    norm: Connection
    predicted_coords: Connection
    output: Connection

    def __init__(self, prediction_dim: int, input_type="distances"):
        assert input_type in ["distances", "powered_distances", "points"]
        base_model = MDSCore(exact_distances=(input_type == "distances"))
        super().__init__(base_model=base_model, input_type=input_type)
        self.factory_args = {"prediction_dim": prediction_dim, "input_type": input_type}
        self._set_shapes({"coords": [None, prediction_dim]})
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        coords: ConnectionType = NOT_GIVEN,
        norm: ConnectionType = NOT_GIVEN,
        predicted_coords: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input": input,
            "norm": norm,
            "predicted_coords": predicted_coords,
            "output": output,
        }

        if "coords" in self._input_keys:
            kwargs["coords"] = coords
        elif coords != NOT_GIVEN:
            raise ValueError("coords is only required when input_type is 'points'!")

        return super().__call__(**kwargs)


class TSNE(DistanceEncoder):
    input: Connection
    norm: Connection
    predicted_coords: Connection
    output: Connection

    # TODO: TSNE norm is always 2. Should we handle this automatically?
    def __init__(
        self,
        prediction_dim: int,
        input_type="distances",
        preplexity=20.0,
        calculate_p_joint: bool = False,
    ):
        assert input_type in ["distances", "powered_distances", "points"]
        base_model = TSNECore(
            calculate_p_joint=calculate_p_joint,
            perplexity=preplexity,
            exact_distances=(input_type == "distances"),
        )
        super().__init__(base_model=base_model, input_type=input_type)
        self.factory_args = {
            "prediction_dim": prediction_dim,
            "input_type": input_type,
            "preplexity": preplexity,
        }

        self._set_shapes({"coords": [None, prediction_dim]})
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        norm: ConnectionType = NOT_GIVEN,
        predicted_coords: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            input=input,
            norm=norm,
            predicted_coords=predicted_coords,
            output=output,
        )


class GaussProcessRegressionCore(Model):
    label: Connection
    s: Connection
    k: Connection
    k_star: Connection
    mu: Connection
    loss: Connection
    prediction: Connection
    confidence: Connection

    def __init__(
        self,
    ) -> None:
        super().__init__()
        # Prepare models used in GPR.
        size_model = Size(dim=0)
        K_term_eye_model = Eye()
        K_term_mult_model = Multiply()
        K_term_model = Add()
        L_term_model = Cholesky()
        label_mu_diff_model = Subtract()
        alpha_model = GPRAlpha()
        gprloss_model = GPRLoss()
        pred_t_model = Transpose()
        pred_dot_model = MatrixMultiply()
        pred_model = Add()
        conf_v_outer_model = GPRVOuter()
        conf_sub_model = Subtract()
        conf_abs_model = Absolute()
        conf_diag_model = TransposedDiagonal()
        conf_model = Add()

        self += size_model(input="k")
        self += K_term_eye_model(N=size_model.output)
        self += K_term_mult_model(left="s", right=K_term_eye_model.output)
        self += K_term_model(left=self.k, right=K_term_mult_model.output)
        self += L_term_model(input=K_term_model.output)
        self += label_mu_diff_model(left="label", right="mu")
        self += alpha_model(
            label_mu_diff=label_mu_diff_model.output,
            L=L_term_model.output,
            K_term=K_term_model.output,
        )
        # Loss Model.
        self += gprloss_model(
            labels=self.label,
            mu=self.mu,
            L=L_term_model.output,
            K_term=K_term_model.output,
            alpha=alpha_model.output,
            output=IOKey(name="loss"),
        )
        # Prediction Pipeline.
        self += pred_t_model(input=self.k)
        self += pred_dot_model(left=pred_t_model.output, right=alpha_model.output)
        self += pred_model(
            left=self.mu, right=pred_dot_model.output, output=IOKey(name="prediction")
        )
        # Confidence Pipeline.
        self += conf_v_outer_model(
            K=self.k, L=L_term_model.output, K_term=K_term_model.output
        )
        self += conf_sub_model(left="k_star", right=conf_v_outer_model.output)
        self += conf_diag_model(input=conf_sub_model.output)
        self += conf_abs_model(input=conf_diag_model.output)
        self += conf_model(
            left=self.s, right=conf_abs_model.output, output=IOKey(name="confidence")
        )

        self.set_canonical_output(pred_model.output)
        shapes: dict[str, ShapeTemplateType] = {
            "label": ["N", 1],
            "s": [1],
            "k": ["N", "M"],
            "k_star": ["N", "M_test"],
            "mu": ["N", 1],
            "loss": [1],
            "prediction": ["N", 1],
            "confidence": ["N", 1],
        }
        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        label: ConnectionType = NOT_GIVEN,
        s: ConnectionType = NOT_GIVEN,
        k: ConnectionType = NOT_GIVEN,
        k_star: ConnectionType = NOT_GIVEN,
        mu: ConnectionType = NOT_GIVEN,
        loss: ConnectionType = NOT_GIVEN,
        prediction: ConnectionType = NOT_GIVEN,
        confidence: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            label=label,
            s=s,
            k=k,
            k_star=k_star,
            mu=mu,
            loss=loss,
            prediction=prediction,
            confidence=confidence,
        )


class GPRLoss(Model):
    labels: Connection
    mu: Connection
    L: Connection
    K_term: Connection
    alpha: Connection
    output: Connection

    def __init__(self, robust=False) -> None:
        super().__init__()
        self.factory_args = {"robust": robust}

        diff_model = Subtract()
        transpose_model = Transpose()
        dot_model = MatrixMultiply()
        squeeze_model = Squeeze()
        mult_model = Multiply()
        mult_model_2 = Multiply()
        eig_model = Eigvalsh()
        log_model = Log(robust=robust)
        sum_reduce_model = Sum()
        length_model = Length()
        sum_model_1 = Add()

        self += diff_model(left="labels", right="mu")
        self += transpose_model(input=diff_model.output)
        self += dot_model(left=transpose_model.output, right="alpha")
        self += squeeze_model(input=dot_model.output)
        self += mult_model(left=squeeze_model.output, right=0.5)
        self += eig_model(K_term="K_term", L="L")
        self += log_model(input=eig_model.output)
        self += sum_reduce_model(input=log_model.output)
        self += length_model(input=self.labels)
        self += mult_model_2(left=length_model.output, right=math.log(2 * math.pi) / 2)
        self += sum_model_1(left=mult_model.output, right=sum_reduce_model.output)
        self += Add()(
            left=sum_model_1.output,
            right=mult_model_2.output,
            output=IOKey(name="output"),
        )

        shapes: dict[str, ShapeTemplateType] = {
            "labels": ["N", 1],
            "mu": ["N", 1],
            "L": ["N", "N"],
            "K_term": ["N", "N"],
            "alpha": ["N", 1],
            "output": [1],
        }
        self._set_shapes(shapes)
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        labels: ConnectionType = NOT_GIVEN,
        mu: ConnectionType = NOT_GIVEN,
        L: ConnectionType = NOT_GIVEN,
        K_term: ConnectionType = NOT_GIVEN,
        alpha: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            labels=labels,
            mu=mu,
            L=L,
            K_term=K_term,
            alpha=alpha,
            output=output,
        )


class Metric(Model):
    pred: Connection
    label: Connection
    output: Connection
    pred_formatted: Connection
    label_formatted: Connection
    label_argmax: Connection
    pred_argmax: Connection
    greater_out: Connection
    pred_comp: Connection

    def __init__(
        self,
        threshold: float | None = None,
        is_binary: bool = False,
        is_pred_one_hot: bool = True,
        is_label_one_hot: bool = True,
    ) -> None:
        super().__init__()
        self.factory_args = {"threshold": threshold}

        assert (
            not is_binary or threshold is not None
        ), "Probs must be False if threshold is not None"

        pred: IOKey | Connection = IOKey(name="pred")
        label: IOKey | Connection = IOKey(name="label")

        if is_label_one_hot:
            self += ArgMax(axis=-1)(label, output="label_argmax")
            label = self.label_argmax

        if is_binary and is_pred_one_hot:
            self += ArgMax(axis=-1)(pred, output="pred_argmax")
            pred = self.pred_argmax
        elif is_binary and not is_pred_one_hot:
            self += Greater()(left=pred, right=threshold, output="greater_out")
            self += Where()(cond="greater_out", input1=1, input2=0, output="pred_comp")
            pred = self.pred_comp
        elif is_pred_one_hot:
            self += ArgMax(axis=-1)(pred, output="pred_argmax")
            pred = self.pred_argmax

        result = pred - label
        self += Buffer()(input=pred, output=IOKey("pred_formatted"))
        self += Buffer()(input=label, output=IOKey("label_formatted"))
        self += Buffer()(input=result, output=IOKey("output"))

    def __call__(  # type: ignore[override]
        self,
        pred: ConnectionType = NOT_GIVEN,
        label: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        pred_formatted: ConnectionType = NOT_GIVEN,
        label_formatted: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            pred=pred,
            label=label,
            output=output,
            pred_formatted=pred_formatted,
            label_formatted=label_formatted,
        )


class Accuracy(Model):
    pred: Connection
    label: Connection
    output: Connection
    metric_out: Connection
    pred_formatted: Connection
    label_formatted: Connection

    def __init__(
        self,
        threshold: float | None = None,
        is_binary: bool = False,
        is_pred_one_hot: bool = True,
        is_label_one_hot: bool = True,
    ):
        super().__init__()
        self += Metric(
            threshold=threshold,
            is_binary=is_binary,
            is_pred_one_hot=is_pred_one_hot,
            is_label_one_hot=is_label_one_hot,
        )("pred", "label", "metric_out", "pred_formatted", "label_formatted")

        true_predictions = self.metric_out == 0
        n_prediction = self.label_formatted.shape()[0]

        self += Sum()(input=true_predictions, output="n_true_predictions")
        self += Divide()(
            numerator="n_true_predictions",
            denominator=n_prediction,
            output=IOKey(name="output"),
        )

    def __call__(  # type: ignore[override]
        self,
        pred: ConnectionType = NOT_GIVEN,
        label: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            pred=pred,
            label=label,
            output=output,
        )


class Precision(Model):
    pred: Connection
    label: Connection
    output: Connection
    metric_out: Connection
    pred_formatted: Connection
    label_formatted: Connection
    n_true_positive: Connection
    n_false_positive: Connection
    n_classes: Connection

    def __init__(
        self,
        average: str = "micro",
        n_classes: int | None = None,
        threshold: float | None = None,
        is_binary: bool = False,
        is_pred_one_hot: bool = True,
        is_label_one_hot: bool = True,
    ) -> None:
        super().__init__()
        self.factory_args = {"threshold": threshold}

        assert average in [
            "micro",
            "macro",
            "weighted",
        ], "average must be one of ['micro', 'macro', 'weighted']"
        # assert (
        #     average not in ["weighted", "macro"] or n_classes is not None
        # ), "n_classes must be provided if average is 'weighted' or 'macro'"

        self += Metric(
            threshold=threshold,
            is_binary=is_binary,
            is_pred_one_hot=is_pred_one_hot,
            is_label_one_hot=is_label_one_hot,
        )("pred", "label", "metric_out", "pred_formatted", "label_formatted")

        if average == "micro":
            true_positive = self.metric_out == 0
            false_positive = self.metric_out != 0
            self += Sum()(input=true_positive, output="n_true_positive")
            self += Sum()(input=false_positive, output="n_false_positive")

            self += Buffer()(
                input=self.n_true_positive
                / (self.n_true_positive + self.n_false_positive),
                output=IOKey(name="output"),
            )

        if average == "macro":
            sum_precision = None
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'macro'"
            for idx in range(n_classes):
                class_idxs = self.label_formatted == idx
                true_positive = (self.metric_out == 0) & class_idxs
                false_positive = (self.pred_formatted == idx) & ~class_idxs

                self += Sum()(input=true_positive, output=f"true_positive_{idx}")
                self += Sum()(input=false_positive, output=f"false_positive_{idx}")
                denominator = getattr(self, f"true_positive_{idx}") + getattr(
                    self, f"false_positive_{idx}"
                )
                self += Where()(denominator == 0, 1, denominator, f"denominator_{idx}")
                self += Divide()(
                    numerator=f"true_positive_{idx}",
                    denominator=getattr(self, f"denominator_{idx}"),
                    output=f"precision_{idx}",
                )

                if sum_precision is None:
                    sum_precision = getattr(self, f"precision_{idx}")
                else:
                    sum_precision += getattr(self, f"precision_{idx}")

            self += Unique()(input=self.label_formatted, output="n_classes")

            self += Divide()(
                numerator=sum_precision,
                denominator=self.n_classes.shape()[0],
                output=IOKey(name="output"),
            )

        elif average == "weighted":
            precision = None
            n_element = self.label_formatted.shape()[0]
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'weighted'"
            for idx in range(n_classes):
                class_idxs = self.label_formatted == idx
                true_positive = (self.metric_out == 0) & class_idxs
                false_positive = (self.pred_formatted == idx) & ~class_idxs
                self += Sum()(input=class_idxs, output=f"n_class_{idx}")

                self += Sum()(input=true_positive, output=f"true_positive_{idx}")
                self += Sum()(input=false_positive, output=f"false_positive_{idx}")
                denominator = getattr(self, f"true_positive_{idx}") + getattr(
                    self, f"false_positive_{idx}"
                )
                self += Where()(denominator == 0, 1, denominator, f"denominator_{idx}")
                self += Divide()(
                    numerator=f"true_positive_{idx}",
                    denominator=(getattr(self, f"denominator_{idx}")),
                    output=f"precision_{idx}",
                )
                self += Divide()(
                    numerator=getattr(self, f"precision_{idx}")
                    * getattr(self, f"n_class_{idx}"),
                    denominator=n_element,
                    output=f"weighted_precision_{idx}",
                )

                if precision is None:
                    precision = getattr(self, f"weighted_precision_{idx}")
                else:
                    precision += getattr(self, f"weighted_precision_{idx}")

            self += Buffer()(input=precision, output=IOKey(name="output"))

    def __call__(  # type: ignore[override]
        self,
        pred: ConnectionType = NOT_GIVEN,
        label: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            pred=pred,
            label=label,
            output=output,
        )


class Recall(Model):
    pred: Connection
    label: Connection
    output: Connection
    metric_out: Connection
    pred_formatted: Connection
    label_formatted: Connection
    n_true_positive: Connection
    n_false_negative: Connection
    n_classes: Connection

    def __init__(
        self,
        average: str = "micro",
        n_classes: int | None = None,
        threshold: float | None = None,
        is_binary: bool = False,
        is_pred_one_hot: bool = True,
        is_label_one_hot: bool = True,
    ) -> None:
        super().__init__()
        self.factory_args = {"threshold": threshold}

        assert average in [
            "micro",
            "macro",
            "weighted",
        ], "average must be one of ['micro', 'macro', 'weighted']"
        # assert (
        #     average not in ["weighted", "macro"] or n_classes is not None
        # ), "n_classes must be provided if average is 'weighted' or 'macro'"

        self += Metric(
            threshold=threshold,
            is_binary=is_binary,
            is_pred_one_hot=is_pred_one_hot,
            is_label_one_hot=is_label_one_hot,
        )("pred", "label", "metric_out", "pred_formatted", "label_formatted")

        if average == "micro":
            true_positive = self.metric_out == 0
            false_negative = self.metric_out != 0
            self += Sum()(input=true_positive, output="n_true_positive")
            self += Sum()(input=false_negative, output="n_false_negative")

            self += Buffer()(
                input=self.n_true_positive
                / (self.n_true_positive + self.n_false_negative),
                output=IOKey(name="output"),
            )

        if average == "macro":
            sum_recall = None
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'macro'"
            for idx in range(n_classes):
                class_idxs = self.label_formatted == idx
                true_positive = (self.metric_out == 0) & class_idxs
                false_negative = (self.pred_formatted != idx) & class_idxs

                self += Sum()(input=true_positive, output=f"true_positive_{idx}")
                self += Sum()(input=false_negative, output=f"false_negative_{idx}")
                denominator = getattr(self, f"true_positive_{idx}") + getattr(
                    self, f"false_negative_{idx}"
                )
                self += Where()(denominator == 0, 1, denominator, f"denominator_{idx}")
                self += Divide()(
                    numerator=f"true_positive_{idx}",
                    denominator=getattr(self, f"denominator_{idx}"),
                    output=f"recall_{idx}",
                )

                if sum_recall is None:
                    sum_recall = getattr(self, f"recall_{idx}")
                else:
                    sum_recall += getattr(self, f"recall_{idx}")

            self += Unique()(input=self.label_formatted, output="n_classes")

            self += Divide()(
                numerator=sum_recall,
                denominator=self.n_classes.shape()[0],
                output=IOKey(name="output"),
            )

        elif average == "weighted":
            recall = None
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'weighted'"
            n_element = self.label_formatted.shape()[0]
            for idx in range(n_classes):
                class_idxs = self.label_formatted == idx
                true_positive = (self.metric_out == 0) & class_idxs
                false_negative = (self.pred_formatted != idx) & class_idxs
                self += Sum()(input=class_idxs, output=f"n_class_{idx}")

                self += Sum()(input=true_positive, output=f"true_positive_{idx}")
                self += Sum()(input=false_negative, output=f"false_negative_{idx}")
                denominator = getattr(self, f"true_positive_{idx}") + getattr(
                    self, f"false_negative_{idx}"
                )
                self += Where()(denominator == 0, 1, denominator, f"denominator_{idx}")
                self += Divide()(
                    numerator=f"true_positive_{idx}",
                    denominator=getattr(self, f"denominator_{idx}"),
                    output=f"recall_{idx}",
                )
                self += Divide()(
                    numerator=getattr(self, f"recall_{idx}")
                    * getattr(self, f"n_class_{idx}"),
                    denominator=n_element,
                    output=f"weighted_recall_{idx}",
                )

                if recall is None:
                    recall = getattr(self, f"weighted_recall_{idx}")
                else:
                    recall += getattr(self, f"weighted_recall_{idx}")

            self += Buffer()(input=recall, output=IOKey(name="output"))

    def __call__(  # type: ignore[override]
        self,
        pred: ConnectionType = NOT_GIVEN,
        label: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            pred=pred,
            label=label,
            output=output,
        )


class F1(Model):
    pred: Connection
    label: Connection
    output: Connection
    metric_out: Connection
    pred_formatted: Connection
    label_formatted: Connection
    n_true_positive: Connection
    n_false_positive: Connection
    n_classes: Connection

    def __init__(
        self,
        average: str = "micro",
        n_classes: int | None = None,
        threshold: float | None = None,
        is_binary: bool = False,
        is_pred_one_hot: bool = True,
        is_label_one_hot: bool = True,
    ) -> None:
        super().__init__()
        self.factory_args = {"threshold": threshold}

        assert average in [
            "micro",
            "macro",
            "weighted",
        ], "average must be one of ['micro', 'macro', 'weighted']"
        assert (
            average not in ["weighted", "macro"] or n_classes is not None
        ), "n_classes must be provided if average is 'weighted' or 'macro'"

        self += Metric(
            threshold=threshold,
            is_binary=is_binary,
            is_pred_one_hot=is_pred_one_hot,
            is_label_one_hot=is_label_one_hot,
        )("pred", "label", "metric_out", "pred_formatted", "label_formatted")

        if average == "micro":
            true_positive = self.metric_out == 0
            false_positive = self.metric_out != 0
            self += Sum()(input=true_positive, output="n_true_positive")
            self += Sum()(input=false_positive, output="n_false_positive")

            self += Buffer()(
                input=self.n_true_positive
                / (self.n_true_positive + self.n_false_positive),
                output=IOKey(name="output"),
            )

        if average == "macro":
            sum_precision = None
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'macro'"
            for idx in range(n_classes):
                class_idxs = self.label_formatted == idx
                true_positive = (self.metric_out == 0) & class_idxs
                false_negative = (self.pred_formatted != idx) & class_idxs
                false_positive = (self.pred_formatted == idx) & ~class_idxs

                self += Sum()(input=true_positive, output=f"true_positive_{idx}")
                self += Sum()(input=false_positive, output=f"false_positive_{idx}")
                self += Sum()(input=false_negative, output=f"false_negative_{idx}")
                denominator = getattr(self, f"true_positive_{idx}") + 0.5 * (
                    getattr(self, f"false_positive_{idx}")
                    + getattr(self, f"false_negative_{idx}")
                )
                self += Where()(denominator == 0, 1, denominator, f"denominator_{idx}")
                self += Divide()(
                    numerator=f"true_positive_{idx}",
                    denominator=getattr(self, f"denominator_{idx}"),
                    output=f"precision_{idx}",
                )

                if sum_precision is None:
                    sum_precision = getattr(self, f"precision_{idx}")
                else:
                    sum_precision += getattr(self, f"precision_{idx}")

            self += Unique()(input=self.label_formatted, output="n_classes")
            self += Divide()(
                numerator=sum_precision,
                denominator=self.n_classes.shape()[0],
                output=IOKey(name="output"),
            )

        elif average == "weighted":
            precision = None
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'weighted'"
            n_element = self.label_formatted.shape()[0]
            for idx in range(n_classes):
                class_idxs = self.label_formatted == idx
                true_positive = (self.metric_out == 0) & class_idxs
                false_negative = (self.pred_formatted != idx) & class_idxs
                false_positive = (self.pred_formatted == idx) & ~class_idxs
                self += Sum()(input=class_idxs, output=f"n_class_{idx}")

                self += Sum()(input=true_positive, output=f"true_positive_{idx}")
                self += Sum()(input=false_positive, output=f"false_positive_{idx}")
                self += Sum()(input=false_negative, output=f"false_negative_{idx}")
                denominator = getattr(self, f"true_positive_{idx}") + 0.5 * (
                    getattr(self, f"false_positive_{idx}")
                    + getattr(self, f"false_negative_{idx}")
                )
                self += Where()(denominator == 0, 1, denominator, f"denominator_{idx}")
                self += Divide()(
                    numerator=f"true_positive_{idx}",
                    denominator=getattr(self, f"denominator_{idx}"),
                    output=f"precision_{idx}",
                )
                self += Divide()(
                    numerator=getattr(self, f"precision_{idx}")
                    * getattr(self, f"n_class_{idx}"),
                    denominator=n_element,
                    output=f"weighted_precision_{idx}",
                )

                if precision is None:
                    precision = getattr(self, f"weighted_precision_{idx}")
                else:
                    precision += getattr(self, f"weighted_precision_{idx}")

            self += Buffer()(input=precision, output=IOKey(name="output"))

    def __call__(  # type: ignore[override]
        self,
        pred: ConnectionType = NOT_GIVEN,
        label: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            pred=pred,
            label=label,
            output=output,
        )


class AUC(Model):
    pred: Connection
    label: Connection
    output: Connection
    label_argmax: Connection

    def __init__(
        self,
        n_classes: int,
        is_label_one_hot: bool = True,
    ) -> None:
        super().__init__()

        assert n_classes > 0, ""
        assert isinstance(n_classes, int)

        label: IOKey | Connection = IOKey(name="label")
        pred: IOKey | Connection = IOKey(name="pred")

        if is_label_one_hot:
            self += ArgMax(axis=-1)(label, output="label_argmax")
            label = self.label_argmax

        auc_score = None
        for class_idx in range(n_classes):
            class_label = label == class_idx
            pred_class = pred[:, class_idx] if n_classes != 1 else pred

            self += AUCCore()(pred_class, class_label, f"auc_core_{class_idx}")
            self += Trapezoid()(
                y=getattr(self, f"auc_core_{class_idx}")[0],
                x=getattr(self, f"auc_core_{class_idx}")[1],
                output=IOKey(f"auc_class_{class_idx}"),
            )
            if auc_score is None:
                auc_score = getattr(self, f"auc_class_{class_idx}") / n_classes
            else:
                auc_score += getattr(self, f"auc_class_{class_idx}") / n_classes

        self += Buffer()(auc_score, IOKey("output"))

    def __call__(  # type: ignore[override]
        self,
        pred: ConnectionType = NOT_GIVEN,
        label: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(
            pred=pred,
            label=label,
            output=output,
        )


class SiLU(Model):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__()

        self += Minus()(input="input", output="minus")
        self += Exponential()(input="minus", output="exp")
        self += Add()(left=1, right="exp", output="add")
        self += Divide()(
            numerator="input", denominator="add", output=IOKey(name="output")
        )

        self._set_shapes({"input": [("Var", ...)], "output": [("Var", ...)]})
        self._freeze()

    def __call__(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().__call__(input=input, output=output)
