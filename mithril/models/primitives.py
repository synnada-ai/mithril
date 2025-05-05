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

import operator
from collections.abc import Sequence
from functools import partial
from types import EllipsisType, NoneType
from typing import Any

from .. import types
from ..common import PaddingType
from ..framework.common import (
    NOT_GIVEN,
    TBD,
    ScalarValueType,
    Tensor,
    TensorValueType,
    ToBeDetermined,
)
from ..framework.constraints import (
    arange_constraints,
    bcast,
    bcast_error_check,
    broadcast_to_constraints,
    concat_constraints,
    conv_1d_constraints,
    conv_2d_constraints,
    cross_entropy_constraint,
    distance_matrix_const,
    eye_constraints,
    flatten_constrains,
    general_type_constraint,
    pad_constraints,
    padding_1d_constraint,
    padding_2d_constraint,
    polynomial_features_constraints,
    pool_1d_constraints,
    pool_2d_constraints,
    randn_constraints,
    squeeze_constraints,
    stride_constraint,
    sum_fn,
    swap_axes_constraints,
    tuple_converter_constraint,
    where_constrains,
)
from ..framework.logical import Model
from ..framework.logical.base import BaseKey
from ..framework.logical.model import Connection, ConnectionType, ExtendInfo
from ..framework.logical.operator import Operator
from ..framework.logical.operators import (
    AbsoluteOp,
    AddOp,
    ArgMaxOp,
    ArgMinOp,
    AtLeast1DOp,
    BufferOp,
    CastOp,
    ConstantType,
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
from ..framework.logical.primitive import OperatorModel, PrimitiveModel
from ..types import Constant

__all__ = [
    "CustomPrimitiveModel",
    "Activation",
    "SquaredError",
    "AbsoluteError",
    "QuantileLoss",
    "CrossEntropy",
    "KLDivergence",
    "BinaryCrossEntropy",
    "HingeLoss",
    "QuadHingeLoss",
    "Sign",
    "Square",
    "Log",
    "StableReciprocal",
    "Relu",
    "LeakyRelu",
    "Sigmoid",
    "Softmax",
    "Softplus",
    "Tanh",
    "CartesianDifference",
    "Concat",
    "PermuteTensor",
    "PrimitiveConvolution1D",
    "PrimitiveConvolution2D",
    "Flatten",
    "PrimitiveMaxPool2D",
    "PrimitiveAvgPool2D",
    "PrimitiveMaxPool1D",
    "NormModifier",
    "DistanceMatrix",
    "PolynomialFeatures",
    "TsnePJoint",
    "EyeComplement",
    "Eye",
    "ZerosLike",
    "Cholesky",
    "GPRAlpha",
    "GPRVOuter",
    "TransposedDiagonal",
    "Eigvalsh",
    "Squeeze",
    "AUCCore",
    "Embedding",
    "PositionalEncoding",
    "SwapAxes",
    "Where",
    "Arange",
    "BroadcastTo",
    "Gelu",
    "PrimitiveUnion",
    "ScaledDotProduct",
    "IsNan",
    "NanToNum",
    "PaddingConverter1D",
    "PaddingConverter2D",
    "StrideConverter",
    "TupleConverter",
    "Unique",
    "Trapezoid",
    "Pad",
    "PrimitiveRandn",
    "PrimitiveRandInt",
    "Ones",
    "PrimitiveModel",
    "Buffer",
    "ToTuple",
    "Power",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "FloorDivide",
    "Negate",
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
    "Slice",
    "Dtype",
    "Sine",
    "Cosine",
    "Minimum",
    "Maximum",
    "AtLeast1D",
]


class CustomPrimitiveModel(PrimitiveModel):
    def __init__(
        self, formula_key: str, name: str | None = None, **kwargs: BaseKey
    ) -> None:
        self.factory_args = {"formula_key": formula_key} | kwargs
        super().__init__(formula_key=formula_key, name=name, **kwargs)


########################## Supervised Loss Types ##########################
class SupervisedLoss(PrimitiveModel):
    """Base class for supervised losses with one input and a target.
    Takes N-dimensional input and target and produces N-dimensional output.

    Parameters
    ----------
    Operator : _type_
        _description_
    """

    input: Connection
    target: Connection
    output: Connection

    def __init__(
        self,
        formula_key: str,
        polymorphic_constraint: bool = True,
        name: str | None = None,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        **kwargs: BaseKey,
    ) -> None:
        default_kwargs: dict[str, BaseKey] = {
            "output": BaseKey(type=Tensor[int | float | bool]),
            "input": BaseKey(type=Tensor[int | float | bool], value=input),
            "target": BaseKey(type=Tensor[int | float | bool], value=target),
        }
        # Finalize kwargs.
        kwargs = default_kwargs | kwargs
        super().__init__(formula_key=formula_key, name=name, **kwargs)

        # Set constraints.
        bcast_constraint = self._add_constraint(
            fn=bcast, keys=[Operator.output_key, "input", "target"]
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[Operator.output_key, "input", "target"],
            dependencies={bcast_constraint},
        )

        if polymorphic_constraint:
            self._add_constraint(
                fn=partial(general_type_constraint, fn=operator.add, is_bitwise=True),
                keys=[Operator.output_key, "input", "target"],
            )

        self.submodel.safe_shapes = {
            "output": ["N", ("Var", ...)],
            "input": ["N", ("Var", ...)],
            "target": ["N", ("Var", ...)],
        }

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        target: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, target=target, output=output)


class SquaredError(SupervisedLoss):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="squared_error", name=name, input=input, target=target
        )


class AbsoluteError(SupervisedLoss):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="absolute_error", name=name, input=input, target=target
        )


class HingeLoss(SupervisedLoss):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            polymorphic_constraint=False,
            formula_key="hinge_loss",
            name=name,
            output=BaseKey(shape=["N", ("Var", ...)], type=Tensor[float]),
            input=input,
            target=target,
        )


class QuadHingeLoss(SupervisedLoss):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            polymorphic_constraint=False,
            formula_key="quad_hinge_loss",
            name=name,
            output=BaseKey(shape=["N", ("Var", ...)], type=Tensor[float]),
            input=input,
            target=target,
        )


class QuantileLoss(PrimitiveModel):
    """
    Takes N-dimensional input and target and produces N-dimensional output.
    """

    input: Connection
    target: Connection
    quantile: Connection
    output: Connection

    def __init__(
        self,
        quantile: int | float | ToBeDetermined = 0.5,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"quantile": quantile}
        super().__init__(
            formula_key="quantile_loss",
            name=name,
            output=BaseKey(type=Tensor[int | float]),
            input=BaseKey(type=Tensor[int | float], value=input),
            target=BaseKey(type=Tensor[int | float], value=target),
            quantile=BaseKey(
                shape=[], type=Tensor[int | float], value=Tensor(quantile)
            ),
        )

        bcast_constraint = self._add_constraint(
            fn=bcast, keys=[Operator.output_key, "input", "target"]
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[Operator.output_key, "input", "target"],
            dependencies={bcast_constraint},
        )

        self._add_constraint(
            fn=partial(general_type_constraint, fn=sum_fn, is_bitwise=True),
            keys=[Operator.output_key, "input", "target", "quantile"],
        )

        self.submodel.safe_shapes = {
            "output": ["N", ("Var", ...)],
            "input": ["N", ("Var", ...)],
            "target": ["N", ("Var", ...)],
        }

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        target: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        quantile: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        output: ConnectionType | Tensor[int | float] = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input, target=target, quantile=quantile, output=output
        )


class CrossEntropy(PrimitiveModel):
    """
    If categorical = True:
        Takes N-dimensional input and (N-1)-dimensional target and
        produces (N-1)-dimensional output.
    else:
        Takes N-dimensional input and target and produces (N-1)-dimensional output.
    """

    input: Connection
    target: Connection
    weights: Connection
    threshold: Connection
    robust: Connection
    output: Connection

    def __init__(
        self,
        input_type: str = "logits",
        weights: list[float] | str = "",
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target: Tensor[int | float | bool] | ToBeDetermined = TBD,
        robust: bool | ToBeDetermined = False,
        threshold: ConstantType
        | ToBeDetermined
        | Tensor[float] = Constant.MIN_POSITIVE_NORMAL,
        categorical: bool | ToBeDetermined = True,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"input_type": input_type, "weights": weights}
        self.input_type = input_type

        weights_type: type = list[float]
        if isinstance(weights, str):
            if weights not in ("", "auto"):
                raise ValueError(
                    f"weights can only be set to 'auto' in string format, got {weights}"
                )
            final_weights: list[float] | bool = weights.lower() == "auto"
            weights_type = bool
        else:
            final_weights = weights

        kwargs: dict[str, BaseKey] = {
            "output": BaseKey(shape=["N", ("Var", ...)], type=Tensor[float]),
            "input": BaseKey(
                shape=["N", "C", ("Var", ...)],
                type=Tensor[int | float | bool],
                value=input,
            ),
            "target": BaseKey(
                shape=["N", ("VarTarget", ...)],
                type=Tensor[int | float | bool],
                value=target,
            ),
            "weights": BaseKey(type=weights_type, value=final_weights),
            "categorical": BaseKey(type=bool, value=categorical),
            "threshold": BaseKey(
                shape=[], type=Tensor[int | float | bool], value=threshold
            ),
            "robust": BaseKey(type=bool, value=robust),
        }

        if input_type == "logits":
            formula_key = "cross_entropy_with_logits"
        elif input_type == "probs":
            formula_key = "cross_entropy"
        elif input_type == "log_probs":
            formula_key = "cross_entropy_with_log_probs"
            kwargs.pop("threshold")
            kwargs.pop("robust")
        else:
            raise ValueError(
                f"Cross entropy does not support '{input_type}' input type. "
                " Available input types: 'logits', 'probs', and 'log_probs'."
            )

        super().__init__(formula_key=formula_key, name=name, **kwargs)

        self._add_constraint(
            fn=cross_entropy_constraint, keys=["categorical", "input", "target"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        target: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        weights: ConnectionType | list[float] = NOT_GIVEN,
        categorical: ConnectionType | bool = NOT_GIVEN,
        threshold: ConnectionType | Tensor[float] = NOT_GIVEN,
        robust: ConnectionType | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input": input,
            "target": target,
            "weights": weights,
            "categorical": categorical,
            "output": output,
        }
        if self.input_type == "log_probs":
            if threshold is not NOT_GIVEN:
                raise ValueError(
                    "Cross entropy with log probs does not accept threshold argument."
                )
            if robust is not NOT_GIVEN:
                raise ValueError(
                    "Cross entropy with log probs does not accept robust argument."
                )
        else:
            kwargs |= {"threshold": threshold, "robust": robust}

        return super().connect(**kwargs)


class KLDivergence(PrimitiveModel):
    """
    Takes N-dimensional input and target and produces N-dimensional output.
    """

    input: Connection
    target: Connection
    threshold: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        target: Tensor[int | float] | ToBeDetermined = TBD,
        threshold: ConstantType
        | Tensor[float]
        | ToBeDetermined = Constant.MIN_POSITIVE_NORMAL,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="kl_divergence",
            name=name,
            output=BaseKey(type=Tensor[float]),
            input=BaseKey(type=Tensor[int | float], value=input),
            target=BaseKey(type=Tensor[int | float], value=target),
            threshold=BaseKey(shape=[], type=Tensor[float], value=threshold),
        )

        self.submodel.safe_shapes = {
            "output": ["N", ("Var", ...)],
            "input": ["N", ("Var", ...)],
            "target": ["N", ("Var", ...)],
        }
        bcast_constraint = self._add_constraint(
            fn=bcast, keys=[Operator.output_key, "input", "target"]
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[Operator.output_key, "input", "target"],
            dependencies={bcast_constraint},
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        target: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        threshold: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input, target=target, threshold=threshold, output=output
        )


class BinaryCrossEntropy(PrimitiveModel):
    """
    Takes N-dimensional input and target and produces N-dimensional output.
    """

    input: Connection
    target: Connection
    pos_weight: Connection
    threshold: Connection
    robust: Connection
    output: Connection

    def __init__(
        self,
        input_type: str = "logits",
        pos_weight: float | str | ToBeDetermined = 1.0,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        target: Tensor[int | float] | ToBeDetermined = TBD,
        threshold: ConstantType
        | Tensor[float]
        | ToBeDetermined = Constant.MIN_POSITIVE_NORMAL,
        robust: bool | ToBeDetermined = False,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {
            "input_type": input_type,
            "pos_weight": pos_weight,
            "threshold": threshold,
            "robust": robust,
        }
        if isinstance(pos_weight, str):
            if pos_weight != "auto":
                raise ValueError(
                    "pos_weight can only be set to 'auto' in string format,"
                    " got {pos_weight}"
                )
            pos_weight = True

        pos_weight_type = (
            float | bool if pos_weight in (..., None) else type(pos_weight)
        )
        kwargs: dict[str, BaseKey] = {
            "output": BaseKey(shape=[("Var_out", ...)], type=Tensor[float]),
            "input": BaseKey(
                shape=[("Var_out", ...)], type=Tensor[int | float], value=input
            ),
            "target": BaseKey(
                shape=[("Var_out", ...)],
                type=Tensor[int | float],
                value=target,
            ),
            "pos_weight": BaseKey(type=pos_weight_type, value=pos_weight),
            "threshold": BaseKey(value=threshold),
            "robust": BaseKey(type=bool, value=robust),
        }

        if input_type == "logits":
            formula_key = "binary_cross_entropy_with_logits"
        elif input_type == "probs":
            formula_key = "binary_cross_entropy"
        else:
            raise ValueError(f"Binary Cross Entropy does not support \
                             '{input_type}' input type. Available    \
                             input types: 'logits' and 'probs'.")

        super().__init__(formula_key=formula_key, name=name, **kwargs)

        bcast_constraint = self._add_constraint(
            fn=bcast, keys=[Operator.output_key, "input", "target"]
        )

        self._add_constraint(
            fn=bcast_error_check,
            keys=[Operator.output_key, "input", "target"],
            dependencies={bcast_constraint},
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        target: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        pos_weight: ConnectionType | float = NOT_GIVEN,
        threshold: ConnectionType | Tensor[float] = NOT_GIVEN,
        robust: ConnectionType | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            target=target,
            pos_weight=pos_weight,
            threshold=threshold,
            robust=robust,
            output=output,
        )


class Log(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        threshold: ConstantType
        | Tensor[float]
        | ToBeDetermined = Constant.MIN_POSITIVE_NORMAL,
        name: str | None = None,
    ) -> None:
        self.robust = robust
        self.factory_args = {"robust": robust, "threshold": threshold}
        if threshold is not Constant.MIN_POSITIVE_NORMAL and not robust:
            raise ValueError(
                "Log does not accept threshold argument when robust is False."
            )

        if robust:
            super().__init__(
                formula_key="robust_log",
                name=name,
                output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
                input=BaseKey(
                    shape=[("Var", ...)], type=Tensor[int | float | bool], value=input
                ),
                threshold=BaseKey(shape=[], type=Tensor[float], value=threshold),
            )
        else:
            super().__init__(
                formula_key="log",
                name=name,
                output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
                input=BaseKey(
                    shape=[("Var", ...)], type=Tensor[int | float | bool], value=input
                ),
            )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        threshold: ConnectionType | Tensor[float] = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output}

        if not self.robust and threshold is not NOT_GIVEN:
            raise ValueError(
                "Log does not accept threshold argument when robust is False."
            )
        elif self.robust:
            kwargs["threshold"] = threshold

        return super().connect(**kwargs)


class StableReciprocal(PrimitiveModel):
    input: Connection
    threshold: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        threshold: ConstantType
        | Tensor[float]
        | ToBeDetermined = Constant.STABLE_RECIPROCAL_THRESHOLD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="stable_reciprocal",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
            input=BaseKey(shape=[("Var", ...)], type=Tensor[int | float], value=input),
            threshold=BaseKey(
                shape=[], type=Tensor[int | float | bool], value=threshold
            ),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        threshold: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, threshold=threshold, output=output)


class Sign(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="sign",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
            input=BaseKey(shape=[("Var", ...)], type=Tensor, value=input),
        )
        self._add_constraint(
            fn=general_type_constraint,
            keys=[Operator.output_key, "input"],
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Square(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="square",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
            input=BaseKey(
                shape=[("Var", ...)], type=Tensor[int | float | bool], value=input
            ),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


############################# Activation Types ##############################
class Activation(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        formula_key: str,
        polymorphic_constraint: bool = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
        **kwargs: BaseKey,
    ) -> None:
        # NOTE: Torch and JAX behave different for some activation functions.
        # For example JAX handles int type inputs for GELU or LeakyRelu while
        # Torch assumes only float inputs for these activations. Since JAX handles
        # more general case, default types are written taking this into account.
        default_kwargs: dict[str, BaseKey] = dict(
            input=BaseKey(shape=[("Var", ...)], type=Tensor[int | float], value=input),
            output=BaseKey(shape=[("Var", ...)], type=Tensor[int | float]),
        )
        # Finalize kwargs.
        kwargs = default_kwargs | kwargs
        super().__init__(name=name, formula_key=formula_key, **kwargs)

        if polymorphic_constraint:
            self._add_constraint(
                fn=general_type_constraint,
                keys=[Operator.output_key, "input"],
            )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Relu(Activation):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="relu",
            name=name,
            polymorphic_constraint=True,
            input=input,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[int | float]),
        )


class Gelu(Activation):
    def __init__(
        self,
        approximate: bool = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="gelu",
            approximate=BaseKey(value=approximate, type=bool),
            name=name,
            input=input,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        approximate: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return Model.connect(self, input=input, approximate=approximate, output=output)


class Sigmoid(Activation):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="sigmoid",
            name=name,
            input=input,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
        )


class Softmax(Activation):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        axis: int | None | ToBeDetermined = -1,
        *,
        name: str | None = None,
    ) -> None:
        axis_key = BaseKey(type=int | None, value=axis)
        super().__init__(
            formula_key="softmax",
            name=name,
            axis=axis_key,
            input=input,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        axis: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return Model.connect(self, input=input, axis=axis, output=output)


class Softplus(Activation):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="softplus", name=name, input=input)


class Tanh(Activation):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(formula_key="tanh", name=name, input=input)


class LeakyRelu(Activation):
    input: Connection
    output: Connection
    slope: Connection

    def __init__(
        self,
        slope: Tensor[int | float | bool] | int | float | ToBeDetermined = 0.01,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"slope": slope}
        super().__init__(
            formula_key="leaky_relu",
            name=name,
            input=input,
            slope=BaseKey(
                value=slope, type=int | float | bool | Tensor[int | float | bool]
            ),
            output=BaseKey(shape=[("Var", ...)], type=Tensor[float]),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        slope: ConnectionType | Tensor[int | float | bool] | int | float = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return Model.connect(self, input=input, slope=slope, output=output)


class StopGradient(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        # super().__init__(formula_key="stop_gradient", name=name)
        super().__init__(
            name=name,
            formula_key="stop_gradient",
            output=BaseKey(shape=[("Var", ...)], type=Tensor[int | float | bool]),
            input=BaseKey(
                shape=[("Var", ...)], type=Tensor[int | float | bool], value=input
            ),
        )

        self.add_constraint(
            fn=general_type_constraint,
            keys=[Operator.output_key, "input"],
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class CartesianDifference(PrimitiveModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(
        self,
        left: Tensor[int | float] | ToBeDetermined = TBD,
        right: Tensor[int | float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        # super().__init__(formula_key="cartesian_diff", name=name)
        super().__init__(
            name=name,
            formula_key="cartesian_diff",
            output=BaseKey(shape=["N", "M", "dim"], type=Tensor[int | float | bool]),
            left=BaseKey(shape=["N", "dim"], type=Tensor[int | float], value=left),
            right=BaseKey(shape=["M", "dim"], type=Tensor[int | float], value=right),
        )
        self._add_constraint(
            fn=partial(general_type_constraint, fn=operator.sub),
            keys=[Operator.output_key, "left", "right"],
        )

    def connect(  # type: ignore[override]
        self,
        left: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        right: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(left=left, right=right, output=output)


class Concat(PrimitiveModel):
    output: Connection
    axis: Connection

    def __init__(
        self,
        input: list[Tensor[int | float | bool]]
        | tuple[Tensor[int | float | bool], ...]
        | ToBeDetermined = TBD,
        axis: int | None | ToBeDetermined = 0,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis}

        super().__init__(
            formula_key="concat",
            name=name,
            output=BaseKey(type=Tensor[int | float | bool]),
            input=BaseKey(
                type=list[Tensor[int | float | bool]]
                | tuple[Tensor[int | float | bool], ...],
                value=input,
            ),
            axis=BaseKey(type=int | None, value=axis),
        )

        self._add_constraint(
            fn=concat_constraints, keys=[Operator.output_key, "input", "axis"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType
        | Sequence[ConnectionType]
        | Sequence[Tensor[int | float | bool]] = NOT_GIVEN,
        axis: ConnectionType | int | None = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, axis=axis, output=output)
        # TODO: Do we need to add general_tensor_type_constraint for Concat?
        # self._add_constraint(
        #     fn=general_tensor_type_constraint,
        #     keys=[Operator.output_key, "input"],
        # )


class PrimitiveUnion(PrimitiveModel):
    output: Connection

    def __init__(
        self,
        n: int = 1,
        *,
        name: str | None = None,
        **kwargs: int | float | tuple[int | float, ...] | ToBeDetermined,
    ) -> None:
        self.factory_args = {"n": n}
        input_definitions = {
            f"input{idx + 1}": BaseKey(
                type=int | float | tuple[int | float, ...],
                value=kwargs.get(f"input{idx + 1}", TBD),
            )
            for idx in range(n)
        }

        super().__init__(
            formula_key="union",
            name=name,
            output=BaseKey(type=tuple[int | float, ...]),
            **input_definitions,
        )


class PermuteTensor(PrimitiveModel):
    input: Connection
    indices: Connection
    output: Connection

    def __init__(
        self,
        indices: Tensor[int | float | bool] | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="permute_tensor",
            name=name,
            output=BaseKey(shape=["N", ("Var", ...)], type=Tensor[int | float | bool]),
            input=BaseKey(
                shape=["N", ("Var", ...)], type=Tensor[int | float | bool], value=input
            ),
            indices=BaseKey(shape=["N"], type=Tensor[int], value=indices),
        )

        self._add_constraint(
            fn=general_type_constraint, keys=[Operator.output_key, "input"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        indices: ConnectionType | Tensor[int] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, indices=indices, output=output)


class PrimitiveConvolution1D(PrimitiveModel):
    input: Connection
    weight: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection
    bias: Connection

    def __init__(
        self,
        use_bias: bool = True,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        weight: Tensor[int | float] | ToBeDetermined = TBD,
        stride: int | ToBeDetermined = TBD,
        padding: int | tuple[int, int] | ToBeDetermined = TBD,
        dilation: int | ToBeDetermined = TBD,
        *,
        bias: Tensor[int | float | bool] | ToBeDetermined = TBD,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"use_bias": use_bias}
        formula_key = "conv1d_bias"
        kwargs: dict[str, BaseKey] = {
            "output": BaseKey(
                shape=["N", "out_channels", "d_out"], type=Tensor[int | float]
            ),
            "input": BaseKey(
                shape=["N", "C_in", "d_in"], type=Tensor[int | float], value=input
            ),
            "weight": BaseKey(
                shape=["out_channels", "C_in", "kernel_size"],
                type=Tensor[int | float],
                value=weight,
            ),
            "bias": BaseKey(
                shape=[1, "out_channels", 1], type=Tensor[int | float], value=bias
            ),
            "stride": BaseKey(type=int, value=stride),
            "padding": BaseKey(type=int | tuple[int, int], value=padding),
            "dilation": BaseKey(type=int, value=dilation),
        }

        if not use_bias:
            formula_key = "conv1d"
            kwargs.pop("bias")

        super().__init__(formula_key=formula_key, name=name, **kwargs)

        self._add_constraint(
            fn=conv_1d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "weight"],
        )

        constraint_keys = ["input", "weight"]
        if use_bias:
            constraint_keys.append("bias")

            self._add_constraint(
                fn=partial(general_type_constraint, fn=sum_fn),
                keys=[Operator.output_key] + constraint_keys,
            )
        else:
            self._add_constraint(
                fn=partial(general_type_constraint, fn=operator.add),
                keys=[Operator.output_key] + constraint_keys,
            )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        weight: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        stride: ConnectionType | int = NOT_GIVEN,
        padding: ConnectionType
        | int
        | tuple[int | ConnectionType, int | ConnectionType] = NOT_GIVEN,
        dilation: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        bias: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input": input,
            "weight": weight,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "output": output,
        }

        if "bias" not in self.input_keys and bias != NOT_GIVEN:
            raise ValueError(f"Operator does not have 'bias' input. \
                             Got {bias} as bias argument!")
        elif "bias" in self.input_keys:
            kwargs |= {"bias": bias}

        return super().connect(**kwargs)


class PrimitiveConvolution2D(PrimitiveModel):
    input: Connection
    weight: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection
    bias: Connection

    def __init__(
        self,
        use_bias: bool = True,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        weight: Tensor[int | float] | ToBeDetermined = TBD,
        stride: int | tuple[int, int] | ToBeDetermined = TBD,
        padding: int
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        dilation: int | tuple[int, int] | ToBeDetermined = TBD,
        *,
        bias: Tensor[int | float] | ToBeDetermined = TBD,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"use_bias": use_bias}
        formula_key = "conv2d_bias"
        kwargs: dict[str, BaseKey] = {
            "output": BaseKey(
                shape=["N", "out_channels", "H_out", "W_out"], type=Tensor[int | float]
            ),
            "input": BaseKey(
                shape=["N", "C_in", "H", "W"], type=Tensor[int | float], value=input
            ),
            "weight": BaseKey(
                shape=["out_channels", "C_in", "kernel_size_0", "kernel_size_1"],
                type=Tensor[int | float],
                value=weight,
            ),
            "bias": BaseKey(
                shape=[1, "out_channels", 1, 1], type=Tensor[int | float], value=bias
            ),
            "stride": BaseKey(type=int | tuple[int, int], value=stride),
            "padding": BaseKey(
                type=int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
                value=padding,
            ),
            "dilation": BaseKey(type=int | tuple[int, int], value=dilation),
        }

        if not use_bias:
            formula_key = "conv2d"
            kwargs.pop("bias")

        super().__init__(formula_key=formula_key, name=name, **kwargs)

        self._add_constraint(
            fn=conv_2d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "weight"],
        )

        constraint_keys = ["input", "weight"]
        if use_bias:
            constraint_keys.append("bias")

            self._add_constraint(
                fn=partial(general_type_constraint, fn=sum_fn),
                keys=[Operator.output_key] + constraint_keys,
            )
        else:
            self._add_constraint(
                fn=partial(general_type_constraint, fn=operator.add),
                keys=[Operator.output_key] + constraint_keys,
            )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        weight: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        stride: ConnectionType
        | int
        | tuple[int | ConnectionType, int | ConnectionType] = NOT_GIVEN,
        padding: ConnectionType
        | int
        | tuple[int | ConnectionType, int | ConnectionType]
        | tuple[
            tuple[int | ConnectionType, int | ConnectionType],
            tuple[int | ConnectionType, int | ConnectionType],
        ] = NOT_GIVEN,
        dilation: ConnectionType
        | int
        | tuple[int | ConnectionType, int | ConnectionType] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        bias: ConnectionType | Tensor[int | float] = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "input": input,
            "weight": weight,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "output": output,
        }

        if "bias" not in self.input_keys and bias != NOT_GIVEN:
            raise ValueError(
                "Operator does not have 'bias' input." " Got {bias} as bias argument!"
            )
        elif "bias" in self.input_keys:
            kwargs |= {"bias": bias}
        return super().connect(**kwargs)


class Flatten(PrimitiveModel):
    input: Connection
    start_dim: Connection
    end_dim: Connection
    output: Connection

    def __init__(
        self,
        start_dim: int | ToBeDetermined = 0,
        end_dim: int | ToBeDetermined = -1,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"start_dim": start_dim, "end_dim": end_dim}

        key_definitions: dict[str, BaseKey] = {
            "output": BaseKey(type=Tensor[int | float | bool]),
            "input": BaseKey(type=Tensor[int | float | bool], value=input),
            "start_dim": BaseKey(type=int, value=start_dim),
            "end_dim": BaseKey(type=int, value=end_dim),
        }
        super().__init__(formula_key="flatten", name=name, **key_definitions)

        self._add_constraint(
            fn=flatten_constrains,
            keys=[Operator.output_key, "input", "start_dim", "end_dim"],
        )
        self._add_constraint(
            fn=general_type_constraint, keys=[Operator.output_key, "input"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        start_dim: ConnectionType | int = NOT_GIVEN,
        end_dim: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input, start_dim=start_dim, end_dim=end_dim, output=output
        )


class PrimitiveMaxPool1D(PrimitiveModel):
    input: Connection
    kernel_size: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        kernel_size: int | ToBeDetermined = TBD,
        stride: int | ToBeDetermined = TBD,
        padding: int | tuple[int, int] | ToBeDetermined = TBD,
        dilation: int | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="max_pool1d",
            name=name,
            output=BaseKey(
                shape=["N", ("C_in", ...), "W_out"], type=Tensor[int | float]
            ),
            input=BaseKey(
                shape=["N", ("C_in", ...), "W"], type=Tensor[int | float], value=input
            ),
            kernel_size=BaseKey(type=int, value=kernel_size),
            stride=BaseKey(type=int, value=stride),
            padding=BaseKey(type=tuple[int, int], value=padding),
            dilation=BaseKey(type=int, value=dilation),
        )
        self._add_constraint(
            fn=pool_1d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "kernel_size"],
        )
        # TODO: Torch does not accept any int type inputs but JAX implementation does.
        self._add_constraint(
            fn=general_type_constraint, keys=[Operator.output_key, "input"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        kernel_size: ConnectionType | int = NOT_GIVEN,
        stride: ConnectionType | int = NOT_GIVEN,
        padding: ConnectionType
        | int
        | tuple[int | ConnectionType, int | ConnectionType] = NOT_GIVEN,
        dilation: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class PaddingConverter1D(PrimitiveModel):
    input: Connection
    kernel_size: Connection
    output: Connection

    def __init__(
        self,
        kernel_size: int | ToBeDetermined = TBD,
        input: int | PaddingType | tuple[int, int] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="padding_converter_1d",
            name=name,
            output=BaseKey(type=tuple[int, int]),
            input=BaseKey(type=int | PaddingType | tuple[int, int], value=input),
            kernel_size=BaseKey(type=int, value=kernel_size),
        )

        self._add_constraint(
            fn=padding_1d_constraint,
            keys=[Operator.output_key, "input", "kernel_size"],
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType
        | int
        | PaddingType
        | tuple[int | ConnectionType, int | ConnectionType] = NOT_GIVEN,
        kernel_size: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, kernel_size=kernel_size, output=output)


class PaddingConverter2D(PrimitiveModel):
    input: Connection
    kernel_size: Connection
    output: Connection

    def __init__(
        self,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        input: int
        | PaddingType
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="padding_converter_2d",
            name=name,
            output=BaseKey(
                type=tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
            ),
            input=BaseKey(
                type=int
                | PaddingType
                | tuple[int, int]
                | tuple[tuple[int, int], tuple[int, int]],
                value=input,
            ),
            kernel_size=BaseKey(type=tuple[int, int], value=kernel_size),
        )

        self._add_constraint(
            fn=padding_2d_constraint,
            keys=[Operator.output_key, "input", "kernel_size"],
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType
        | int
        | PaddingType
        | tuple[int | ConnectionType, int | ConnectionType]
        | tuple[
            tuple[int | ConnectionType, int | ConnectionType],
            tuple[int | ConnectionType, int | ConnectionType],
        ] = NOT_GIVEN,
        kernel_size: ConnectionType
        | tuple[int | ConnectionType, int | ConnectionType] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, kernel_size=kernel_size, output=output)


class StrideConverter(PrimitiveModel):
    input: Connection
    kernel_size: Connection
    output: Connection

    def __init__(
        self,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        input: int | PaddingType | tuple[int, int] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="stride_converter",
            name=name,
            output=BaseKey(type=int | tuple[int, int]),
            input=BaseKey(type=int | PaddingType | tuple[int, int] | None, value=input),
            kernel_size=BaseKey(type=int | tuple[int, int], value=kernel_size),
        )
        self._add_constraint(
            fn=stride_constraint,
            keys=[Operator.output_key, "input", "kernel_size"],
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType
        | int
        | PaddingType
        | tuple[int | ConnectionType, int | ConnectionType]
        | None = NOT_GIVEN,
        kernel_size: ConnectionType
        | int
        | tuple[int | ConnectionType, int | ConnectionType] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, kernel_size=kernel_size, output=output)


class TupleConverter(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: int
        | PaddingType
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="tuple_converter",
            name=name,
            output=BaseKey(
                type=tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
            ),
            input=BaseKey(
                type=int
                | PaddingType
                | tuple[int, int]
                | tuple[tuple[int, int], tuple[int, int]]
                | None,
                value=input,
            ),
        )
        self._add_constraint(
            fn=tuple_converter_constraint, keys=[Operator.output_key, "input"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType
        | int
        | PaddingType
        | tuple[int | ConnectionType, int | ConnectionType]
        | tuple[
            tuple[int | ConnectionType, int | ConnectionType],
            tuple[int | ConnectionType, int | ConnectionType],
        ]
        | None,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class PrimitivePool2D(PrimitiveModel):
    input: Connection
    kernel_size: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        stride: int | tuple[int, int] | ToBeDetermined = TBD,
        padding: int
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        dilation: int | tuple[int, int] | ToBeDetermined = TBD,
        *,
        formula_key: str,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key=formula_key,
            name=name,
            output=BaseKey(
                shape=["N", ("C_in", ...), "H_out", "W_out"], type=Tensor[int | float]
            ),
            input=BaseKey(
                shape=["N", ("C_in", ...), "H", "W"],
                type=Tensor[int | float],
                value=input,
            ),
            kernel_size=BaseKey(type=tuple[int, int], value=kernel_size),
            stride=BaseKey(type=tuple[int, int], value=stride),
            padding=BaseKey(
                type=tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
                value=padding,
            ),
            dilation=BaseKey(type=tuple[int, int], value=dilation),
        )

        self._add_constraint(
            fn=pool_2d_constraints,
            keys=["output", "input", "stride", "padding", "dilation", "kernel_size"],
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        kernel_size: ConnectionType
        | tuple[int | ConnectionType, int | ConnectionType] = NOT_GIVEN,
        stride: ConnectionType
        | tuple[int | ConnectionType, int | ConnectionType] = NOT_GIVEN,
        padding: ConnectionType
        | tuple[int | ConnectionType, int | ConnectionType]
        | tuple[
            tuple[int | ConnectionType, int | ConnectionType],
            tuple[int | ConnectionType, int | ConnectionType],
        ] = NOT_GIVEN,
        dilation: ConnectionType
        | int
        | tuple[int | ConnectionType, int | ConnectionType] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class PrimitiveMaxPool2D(PrimitivePool2D):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        stride: int | tuple[int, int] | ToBeDetermined = TBD,
        padding: int
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = TBD,
        dilation: int | tuple[int, int] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="max_pool2d",
            name=name,
            input=input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


class PrimitiveAvgPool2D(PrimitivePool2D):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        kernel_size: int | tuple[int, int] | ToBeDetermined = TBD,
        stride: int | tuple[int, int] | ToBeDetermined = TBD,
        padding: int
        | tuple[int, int]
        | tuple[
            tuple[int, int],
            tuple[int, int],
        ]
        | ToBeDetermined = TBD,
        dilation: int | tuple[int, int] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            input=input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            formula_key="avg_pool2d",
            name=name,
        )


class NormModifier(PrimitiveModel):
    # TODO: Make this primitive logical.
    """A helper model that modifies norm input. It is used for mapping
    norm values from (`-inf`, `inf`) to the interval (`1.0`, `5.0`) using a
    periodic triangular function with period 8 as shown on the figure below.
    This helper model guarantees norm values to be in an acceptable and
    meaningful range.

    ```
    5 _ _ _ _ _ _ _ _ _ _
         \\      /\\      /
          \\    /  \\    /
           \\  /    \\  /
    1 _ _ _ \\/_ _ _ \\/_ _
        |   |   |   |
        |   |   |   |
        -3   1   5   9
    ```
    """

    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="norm_modifier",
            name=name,
            output=BaseKey(shape=[], type=Tensor[int | float | bool]),
            input=BaseKey(shape=[], type=Tensor[float], value=input),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class DistanceMatrix(PrimitiveModel):
    # TODO: Make this primitive logical.
    left: Connection
    right: Connection
    norm: Connection
    output: Connection

    # TODO: torch.cdist handles batches of matrices, for now we don't.
    def __init__(
        self,
        left: Tensor[int | float | bool] | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="distance_matrix",
            name=name,
            output=BaseKey(shape=["N", "M"], type=Tensor[int | float]),
            left=BaseKey(shape=["N", "d"], type=Tensor[int | float], value=left),
            right=BaseKey(shape=["M", "d"], type=Tensor[int | float], value=right),
            norm=BaseKey(shape=[], type=Tensor[int | float | bool]),
        )

        self._add_constraint(
            fn=partial(general_type_constraint, fn=distance_matrix_const),
            keys=[Operator.output_key, "left", "right", "norm"],
        )
        self._set_cin("left", "right", safe=False)

    def connect(  # type: ignore[override]
        self,
        left: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        right: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        norm: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        output: ConnectionType | Tensor[int | float] = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(left=left, right=right, norm=norm, output=output)


class PolynomialFeatures(PrimitiveModel):
    input: Connection
    degree: Connection
    output: Connection

    def __init__(
        self,
        degree: int | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="polynomial_features",
            name=name,
            output=BaseKey(shape=["N", "d_out"], type=Tensor[int | float]),
            input=BaseKey(shape=["N", "d_in"], type=Tensor[int | float], value=input),
            degree=BaseKey(type=int, value=degree),
        )

        self._add_constraint(
            fn=polynomial_features_constraints, keys=["output", "input", "degree"]
        )
        self._add_constraint(
            fn=general_type_constraint, keys=[Operator.output_key, "input"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        degree: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, degree=degree, output=output)


class TsnePJoint(PrimitiveModel):
    squared_distances: Connection
    target_perplexity: Connection
    threshold: Connection
    output: Connection

    disposable = True

    def __init__(
        self,
        squared_distances: Tensor[int | float | bool] | ToBeDetermined = TBD,
        target_perplexity: float | ToBeDetermined = TBD,
        threshold: ConstantType | Tensor[float] | ToBeDetermined = Constant.EPSILON,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="tsne_p_joint",
            name=name,
            output=BaseKey(shape=["N", "M"], type=Tensor[float]),
            squared_distances=BaseKey(
                shape=["N", "M"], type=Tensor[float], value=squared_distances
            ),
            target_perplexity=BaseKey(
                shape=[], type=Tensor[float], value=target_perplexity
            ),
            threshold=BaseKey(shape=[], type=Tensor[float], value=threshold),
        )

    def connect(  # type: ignore[override]
        self,
        squared_distances: ConnectionType | Tensor[float] = NOT_GIVEN,
        target_perplexity: ConnectionType | float = NOT_GIVEN,
        threshold: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            squared_distances=squared_distances,
            target_perplexity=target_perplexity,
            threshold=threshold,
            output=output,
        )


class EyeComplement(PrimitiveModel):
    # TODO: Add Dtype constraints to this model
    N: Connection
    M: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        N: int | ToBeDetermined = TBD,
        M: int | ToBeDetermined | None = None,
        dtype: types.Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="ones_with_zero_diag",
            name=name,
            output=BaseKey(shape=["N", "M"], type=Tensor[float]),
            N=BaseKey(type=int, value=N),
            M=BaseKey(type=int | None, value=M),
            dtype=BaseKey(type=types.Dtype | None, value=dtype),
        )
        self._add_constraint(fn=eye_constraints, keys=["output", "N", "M"])

    def connect(  # type: ignore[override]
        self,
        N: ConnectionType | int = NOT_GIVEN,
        M: ConnectionType | int = NOT_GIVEN,
        dtype: ConnectionType | types.Dtype = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(N=N, M=M, dtype=dtype, output=output)


class Eye(PrimitiveModel):
    N: Connection
    M: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        N: int | ToBeDetermined = TBD,
        M: int | ToBeDetermined | None = None,
        dtype: types.Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="eye",
            name=name,
            output=BaseKey(shape=["N", "M"], type=Tensor[float]),
            N=BaseKey(type=int, value=N),
            M=BaseKey(type=int | None, value=M),
            dtype=BaseKey(type=types.Dtype | None, value=dtype),
        )
        self._add_constraint(fn=eye_constraints, keys=["output", "N", "M"])

    def connect(  # type: ignore[override]
        self,
        N: ConnectionType | int = NOT_GIVEN,
        M: ConnectionType | int = NOT_GIVEN,
        dtype: ConnectionType | types.Dtype = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(N=N, M=M, dtype=dtype, output=output)


class Cholesky(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="cholesky",
            name=name,
            output=BaseKey(shape=["N", "N"], type=Tensor[float]),
            input=BaseKey(shape=["N", "N"], type=Tensor[float], value=input),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class GPRAlpha(PrimitiveModel):
    label_mu_diff: Connection
    L: Connection
    K_term: Connection
    output: Connection

    def __init__(
        self,
        label_mu_diff: Tensor[int | float | bool] | ToBeDetermined = TBD,
        L: Tensor[int | float | bool] | ToBeDetermined = TBD,
        K_term: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="gpr_alpha",
            name=name,
            output=BaseKey(shape=["N", 1], type=Tensor[float]),
            label_mu_diff=BaseKey(
                shape=["N", 1], type=Tensor[float], value=label_mu_diff
            ),
            L=BaseKey(shape=["N", "N"], type=Tensor[float], value=L),
            K_term=BaseKey(shape=["N", "N"], type=Tensor[float], value=K_term),
        )

    def connect(  # type: ignore[override]
        self,
        label_mu_diff: ConnectionType | Tensor[float] = NOT_GIVEN,
        L: ConnectionType | Tensor[float] = NOT_GIVEN,
        K_term: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            label_mu_diff=label_mu_diff, L=L, K_term=K_term, output=output
        )


class GPRVOuter(PrimitiveModel):
    K: Connection
    K_term: Connection
    L: Connection
    output: Connection

    def __init__(
        self,
        K: Tensor[float] | ToBeDetermined = TBD,
        K_term: Tensor[float] | ToBeDetermined = TBD,
        L: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="gpr_v_outer",
            name=name,
            output=BaseKey(shape=["N", "N"], type=Tensor[float]),
            K=BaseKey(shape=["N", "N"], type=Tensor[int | float | bool], value=K),
            K_term=BaseKey(
                shape=["N", "N"], type=Tensor[int | float | bool], value=K_term
            ),
            L=BaseKey(shape=["N", "N"], type=Tensor[int | float | bool], value=L),
        )

    def connect(  # type: ignore[override]
        self,
        K: ConnectionType | Tensor[float] = NOT_GIVEN,
        K_term: ConnectionType | Tensor[float] = NOT_GIVEN,
        L: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType | Tensor[float] = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(K=K, K_term=K_term, L=L, output=output)


class TransposedDiagonal(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="transposed_diag",
            name=name,
            output=BaseKey(shape=["N", 1], type=Tensor[int | float | bool]),
            input=BaseKey(
                shape=["N", "N"], type=Tensor[int | float | bool], value=input
            ),
        )

        self._add_constraint(
            fn=general_type_constraint, keys=[Operator.output_key, "input"]
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Arange(PrimitiveModel):
    start: Connection
    stop: Connection
    step: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        start: int | float | ToBeDetermined = 0,
        stop: int | float | ToBeDetermined = TBD,
        step: int | float | ToBeDetermined = 1,
        dtype: types.Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        # TODO: add Arange type constraint that handles all
        # start stop step dtype combinations
        all_defined = False
        if (
            not isinstance(start, ToBeDetermined)
            and not isinstance(stop, ToBeDetermined)
            and not isinstance(step, ToBeDetermined)
        ):
            all_defined = True
            val = (start - stop) / step
            # If val has decimal part take absolute of integer
            # part of it and add 1.
            # Else no decimal part, simply take absolute of val.
            val = abs(val) if int(val) == val else abs(int(val)) + 1
            val = int(val)
            output_shp: list[int | str] = [] if val == 0 else [val]
        else:
            output_shp = ["N"]

        super().__init__(
            formula_key="arange",
            name=name,
            output=BaseKey(shape=output_shp, type=Tensor[int | float]),
            start=BaseKey(type=int | float, value=start),
            stop=BaseKey(type=int | float, value=stop),
            step=BaseKey(type=int | float, value=step),
            dtype=BaseKey(type=types.Dtype | None, value=dtype),
        )
        self._set_cin("stop", safe=False)

        if not all_defined:
            self._add_constraint(
                fn=arange_constraints, keys=["output", "start", "stop", "step"]
            )

    def connect(  # type: ignore[override]
        self,
        start: ConnectionType | int | float = NOT_GIVEN,
        stop: ConnectionType | int | float = NOT_GIVEN,
        step: ConnectionType | int | float = NOT_GIVEN,
        dtype: ConnectionType | types.Dtype | None = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            start=start, stop=stop, step=step, dtype=dtype, output=output
        )


class PrimitiveRandn(PrimitiveModel):
    shape: Connection
    key: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int, ...] | ToBeDetermined = TBD,
        key: int | Tensor[int] | ToBeDetermined = TBD,
        dtype: types.Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="randn",
            name=name,
            output=BaseKey(type=Tensor[float]),
            shape=BaseKey(type=tuple[int, ...], value=shape),
            key=BaseKey(type=int | Tensor[int], value=key),
            dtype=BaseKey(type=types.Dtype | None, value=dtype),
        )

        self.add_constraint(randn_constraints, keys=["output", "shape"])

    def connect(  # type: ignore[override]
        self,
        shape: ConnectionType | tuple[int | ConnectionType, ...] = NOT_GIVEN,
        key: ConnectionType | int = NOT_GIVEN,
        dtype: ConnectionType | types.Dtype | None = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(shape=shape, key=key, dtype=dtype, output=output)


class PrimitiveRandInt(PrimitiveModel):
    shape: Connection
    key: Connection
    low: Connection
    high: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int, ...] | ToBeDetermined = TBD,
        key: int | Tensor[int] | ToBeDetermined = TBD,
        low: int | ToBeDetermined = TBD,
        high: int | ToBeDetermined = TBD,
        dtype: types.Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="randint",
            name=name,
            output=BaseKey(shape=[("output", ...)], type=Tensor[int]),
            shape=BaseKey(type=tuple[int, ...] | tuple[()], value=shape),
            key=BaseKey(type=int | Tensor[int], value=key),
            low=BaseKey(type=int, value=low),
            high=BaseKey(type=int, value=high),
            dtype=BaseKey(type=types.Dtype | None, value=dtype),
        )

        self.add_constraint(randn_constraints, keys=["output", "shape"])

    def connect(  # type: ignore[override]
        self,
        shape: ConnectionType = NOT_GIVEN,
        key: ConnectionType = NOT_GIVEN,
        low: ConnectionType = NOT_GIVEN,
        high: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            shape=shape, key=key, low=low, high=high, dtype=dtype, output=output
        )


class BroadcastTo(PrimitiveModel):
    input: Connection
    shape: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int, ...] | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="broadcast_to",
            name=name,
            output=BaseKey(type=Tensor[int | float | bool]),
            input=BaseKey(type=Tensor[int | float | bool], value=input),
            shape=BaseKey(type=tuple[int, ...], value=shape),
        )

        self.add_constraint(
            fn=broadcast_to_constraints, keys=["output", "shape", "input"]
        )
        self._add_constraint(
            fn=general_type_constraint, keys=[Operator.output_key, "input"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        shape: ConnectionType | tuple[int | ConnectionType, ...] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, shape=shape, output=output)


class Eigvalsh(PrimitiveModel):
    K_term: Connection
    L: Connection
    threshold: Connection
    output: Connection

    def __init__(
        self,
        K_term: Tensor[float] | ToBeDetermined = TBD,
        L: Tensor[float] | ToBeDetermined = TBD,
        threshold: ConstantType | Tensor[float] | ToBeDetermined = Constant.EPSILON,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="eigvalsh",
            name=name,
            output=BaseKey(shape=["N", 1], type=Tensor[float]),
            K_term=BaseKey(shape=["N", "N"], type=Tensor[float], value=K_term),
            L=BaseKey(shape=["N", "N"], type=Tensor[float], value=L),
            threshold=BaseKey(shape=[], type=Tensor[float], value=threshold),
        )

    def connect(  # type: ignore[override]
        self,
        K_term: ConnectionType = NOT_GIVEN,
        L: ConnectionType = NOT_GIVEN,
        threshold: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(K_term=K_term, L=L, threshold=threshold, output=output)


class Squeeze(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="squeeze",
            name=name,
            output=BaseKey(type=Tensor[int | float | bool]),
            input=BaseKey(type=Tensor[int | float | bool], value=input),
        )

        self._add_constraint(fn=squeeze_constraints, keys=["output", "input"])
        self._add_constraint(
            fn=general_type_constraint, keys=[Operator.output_key, "input"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class AUCCore(PrimitiveModel):
    input: Connection
    label: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[float] | ToBeDetermined = TBD,
        label: Tensor[bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="auc_core",
            name=name,
            output=BaseKey(shape=[2, "M"], type=Tensor[float]),
            input=BaseKey(shape=["N"], type=Tensor[float], value=input),
            label=BaseKey(shape=["N"], type=Tensor[bool], value=label),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[float] = NOT_GIVEN,
        label: ConnectionType | Tensor[bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, label=label, output=output)


class Embedding(PrimitiveModel):
    input: Connection
    weight: Connection
    output: Connection

    def __init__(
        self,
        num_embeddings: int | None = None,
        dim: int | None = None,
        input: Tensor[int] | ToBeDetermined = TBD,
        weight: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        out_dim: int | str = "dim" if dim is None else dim

        super().__init__(
            formula_key="primitive_embedding",
            name=name,
            output=BaseKey(shape=[("N1", ...), "d1", out_dim], type=Tensor[float]),
            input=BaseKey(shape=[("N1", ...), "d1"], type=Tensor[int], value=input),
            weight=BaseKey(
                shape=[num_embeddings, out_dim],
                type=Tensor[float],
                value=weight,
                differentiable=True,
            ),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int] = NOT_GIVEN,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, weight=weight, output=output)


class ScaledDotProduct(PrimitiveModel):
    query: Connection
    key: Connection
    value: Connection
    attn_mask: Connection
    dropout_p: Connection
    is_causal: Connection
    scale: Connection
    output: Connection

    def __init__(
        self,
        is_causal: bool | ToBeDetermined = True,
        scale: None | int | float | ToBeDetermined = None,
        dropout_p: float | ToBeDetermined = 0.0,
        use_attn_mask: bool = False,
        query: Tensor[int | float] | ToBeDetermined = TBD,
        key: Tensor[int | float] | ToBeDetermined = TBD,
        value: Tensor[int | float] | ToBeDetermined = TBD,
        attn_mask: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        # TODO: Reconsider how to get attn_mask, could it be A?
        assert (
            not isinstance(is_causal, bool) or not is_causal or not use_attn_mask
        ), "Causal attention is not support attn_mask!"
        assert isinstance(use_attn_mask, bool), "use_attn_mask must be a boolean value!"
        self.use_attn_mask = use_attn_mask

        formula_key = "scaled_dot_product_attention"
        kwargs: dict[str, BaseKey] = {
            "output": BaseKey(shape=[("Var", ...), "L", "O"], type=Tensor[float]),
            "query": BaseKey(
                shape=[("Var", ...), "L", "E"], type=Tensor[int | float], value=query
            ),
            "key": BaseKey(
                shape=[("Var", ...), "S", "E"], type=Tensor[int | float], value=key
            ),
            "value": BaseKey(
                shape=[("Var", ...), "S", "O"], type=Tensor[int | float], value=value
            ),
            "dropout_p": BaseKey(type=float, value=dropout_p),
            "attn_mask": BaseKey(type=NoneType, value=None),
            "is_causal": BaseKey(type=bool, value=is_causal),
            "scale": BaseKey(type=NoneType | int | float, value=scale),
        }

        if use_attn_mask:
            kwargs["attn_mask"] = BaseKey(
                shape=["L", "S"], type=Tensor, value=attn_mask
            )

        super().__init__(formula_key=formula_key, name=name, **kwargs)

    def connect(  # type: ignore[override]
        self,
        query: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        key: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        value: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        dropout_p: ConnectionType | float = NOT_GIVEN,
        is_causal: ConnectionType | bool = NOT_GIVEN,
        scale: ConnectionType | None | int | float = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        attn_mask: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        if (
            not self.use_attn_mask
            and attn_mask is not NOT_GIVEN
            and not isinstance(attn_mask, str)
            and isinstance(attn_mask, BaseKey)
            and attn_mask.metadata.value is not None  # TODO: Here will be updated!
        ):
            raise KeyError(
                "Operator does not have 'attn_mask' input." " Got attn_mask argument!"
            )

        return super().connect(
            query=query,
            key=key,
            value=value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            output=output,
            attn_mask=attn_mask,
        )


class PositionalEncoding(PrimitiveModel):
    input: Connection
    hidden_dim: Connection
    max_len: Connection
    output: Connection

    # TODO: Try to move to Logical composite models.
    def __init__(
        self,
        hidden_dim: int | ToBeDetermined,
        max_len: int | ToBeDetermined = 5000,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"hidden_dim": hidden_dim, "max_len": max_len}

        super().__init__(
            formula_key="positional_encoding",
            name=name,
            output=BaseKey(shape=[("N1", ...)], type=Tensor[float]),
            input=BaseKey(
                shape=[("N1", ...)], type=Tensor[int | float | bool], value=input
            ),
            hidden_dim=BaseKey(type=int, value=hidden_dim),
            max_len=BaseKey(type=int, value=max_len),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        hidden_dim: ConnectionType | int = NOT_GIVEN,
        max_len: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input, hidden_dim=hidden_dim, max_len=max_len, output=output
        )


class SwapAxes(PrimitiveModel):
    input: Connection
    axis1: Connection
    axis2: Connection
    output: Connection

    def __init__(
        self,
        axis1: int | ToBeDetermined,
        axis2: int | ToBeDetermined,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis1": axis1, "axis2": axis2}

        super().__init__(
            formula_key="swapaxes",
            name=name,
            output=BaseKey(type=Tensor[int | float | bool]),
            input=BaseKey(type=Tensor[int | float | bool], value=input),
            axis1=BaseKey(type=int, value=axis1),
            axis2=BaseKey(type=int, value=axis2),
        )

        self._add_constraint(
            fn=swap_axes_constraints, keys=["output", "input", "axis1", "axis2"]
        )
        self._add_constraint(
            fn=general_type_constraint, keys=[Operator.output_key, "input"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        axis1: ConnectionType | int = NOT_GIVEN,
        axis2: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, axis1=axis1, axis2=axis2, output=output)


class Where(PrimitiveModel):
    cond: Connection
    input1: Connection
    input2: Connection
    output: Connection

    def __init__(
        self,
        cond: Tensor[int | float | bool] | ToBeDetermined = TBD,
        input1: Tensor[int | float | bool] | ToBeDetermined = TBD,
        input2: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="where",
            name=name,
            output=BaseKey(type=Tensor[int | float | bool]),
            cond=BaseKey(type=Tensor[bool], value=cond),
            input1=BaseKey(type=Tensor[int | float | bool], value=input1),
            input2=BaseKey(type=Tensor[int | float | bool], value=input2),
        )

        self._add_constraint(
            fn=where_constrains, keys=["output", "cond", "input1", "input2"]
        )
        self._add_constraint(
            fn=partial(general_type_constraint, fn=operator.add, is_bitwise=True),
            keys=[Operator.output_key, "input1", "input2"],
        )
        self._set_cin("input1", safe=False)

    def connect(  # type: ignore[override]
        self,
        cond: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        input1: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        input2: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(cond=cond, input1=input1, input2=input2, output=output)


class IsNan(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="isnan",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[bool]),
            input=BaseKey(
                shape=[("Var", ...)], type=Tensor[int | float | bool], value=input
            ),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType | Tensor[bool] = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Unique(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="unique",
            name=name,
            input=BaseKey(type=Tensor[int | float | bool], value=input),
            output=BaseKey(type=Tensor[int | float | bool]),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Trapezoid(PrimitiveModel):
    y: Connection
    x: Connection
    output: Connection

    def __init__(
        self,
        x: Tensor[int | float | bool] | ToBeDetermined = TBD,
        y: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="trapezoid",
            name=name,
            output=BaseKey(shape=[], type=Tensor[int | float | bool]),
            y=BaseKey(shape=[("Var", ...)], type=Tensor[int | float | bool], value=y),
            x=BaseKey(shape=[("Var", ...)], type=Tensor[int | float | bool], value=x),
        )

    def connect(  # type: ignore[override]
        self,
        y: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        x: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(y=y, x=x, output=output)


class NanToNum(PrimitiveModel):
    input: Connection
    nan: Connection
    posinf: Connection
    neginf: Connection
    output: Connection

    def __init__(
        self,
        nan: float | ToBeDetermined = 0.0,
        posinf: float | None | ToBeDetermined = None,
        neginf: float | None | ToBeDetermined = None,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="nan_to_num",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[int | float | bool]),
            input=BaseKey(
                shape=[("Var", ...)], type=Tensor[int | float | bool], value=input
            ),
            nan=BaseKey(type=float, value=nan),
            posinf=BaseKey(type=float | None, value=posinf),
            neginf=BaseKey(type=float | None, value=neginf),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        nan: ConnectionType | float = NOT_GIVEN,
        posinf: ConnectionType | float | None = NOT_GIVEN,
        neginf: ConnectionType | float | None = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input, nan=nan, posinf=posinf, neginf=neginf, output=output
        )


class Pad(PrimitiveModel):
    input: Connection
    pad_width: Connection
    output: Connection

    def __init__(
        self,
        pad_width: tuple[tuple[int, int], ...] | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="pad",
            name=name,
            output=BaseKey(type=Tensor[int | float | bool]),
            input=BaseKey(type=Tensor[int | float | bool], value=input),
            pad_width=BaseKey(type=tuple[tuple[int, int], ...], value=pad_width),
        )

        self._add_constraint(
            fn=pad_constraints, keys=[Operator.output_key, "input", "pad_width"]
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        pad_width: ConnectionType
        | tuple[tuple[int | ConnectionType, int | ConnectionType], ...] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, pad_width=pad_width, output=output)


class ZerosLike(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="zeros_like",
            name=name,
            output=BaseKey(shape=[("Var", ...)], type=Tensor[int | float | bool]),
            input=BaseKey(
                shape=[("Var", ...)], type=Tensor[int | float | bool], value=input
            ),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Ones(PrimitiveModel):
    shape: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int, ...] | list[int] | ToBeDetermined = TBD,
        dtype: types.Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            formula_key="ones",
            name=name,
            output=BaseKey(type=Tensor),
            shape=BaseKey(type=tuple[int, ...] | list[int], value=shape),
            dtype=BaseKey(type=types.Dtype | None, value=dtype),
        )

        self._add_constraint(randn_constraints, keys=["output", "shape"])

    def connect(  # type: ignore[override]
        self,
        shape: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(shape=shape, dtype=dtype, output=output)


class Buffer(OperatorModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=BufferOp(input=input))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType
        | ScalarValueType
        | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class ToTuple(OperatorModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        n: int,
        *,
        name: str | None = None,
        **kwargs: Tensor[int | float | bool] | ScalarValueType | ToBeDetermined,
    ) -> None:
        super().__init__(name=name, model=ToTupleOp(n, **kwargs))


class ArithmeticOperation(OperatorModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(
        self,
        model: Operator,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=model)

    def connect(  # type: ignore[override]
        self,
        left: ConnectionType
        | Tensor[int | float | bool]
        | int
        | float
        | bool = NOT_GIVEN,
        right: ConnectionType
        | Tensor[int | float | bool]
        | int
        | float
        | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(left=left, right=right, output=output)


class Power(OperatorModel):
    base: Connection
    exponent: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        base: Tensor[int | float | bool] | int | float | ToBeDetermined = TBD,
        exponent: Tensor[int | float | bool] | int | float | ToBeDetermined = TBD,
        threshold: ConstantType
        | Tensor[float]
        | ToBeDetermined = types.Constant.MIN_POSITIVE_NORMAL,
        *,
        name: str | None = None,
    ) -> None:
        self.robust = robust
        m = PowerOp(robust=robust, base=base, exponent=exponent, threshold=threshold)
        super().__init__(name=name, model=m)

    def connect(  # type: ignore[override]
        self,
        base: ConnectionType | Tensor[int | float | bool] | int | float = NOT_GIVEN,
        exponent: ConnectionType | Tensor[int | float | bool] | int | float = NOT_GIVEN,
        output: ConnectionType | Tensor[float] = NOT_GIVEN,
        *,
        threshold: ConnectionType | Tensor[float] = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"base": base, "exponent": exponent, "output": output}
        if not self.robust and threshold is not NOT_GIVEN:
            raise ValueError(
                "Power does not accept threshold argument when robust is False."
            )
        elif self.robust:
            kwargs["threshold"] = threshold

        return super().connect(**kwargs)


class Add(ArithmeticOperation):
    def __init__(
        self,
        left: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(AddOp(left=left, right=right), name=name)


class Subtract(ArithmeticOperation):
    def __init__(
        self,
        left: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(SubtractOp(left=left, right=right), name=name)


class Multiply(ArithmeticOperation):
    def __init__(
        self,
        left: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(MultiplyOp(left=left, right=right), name=name)


class Minimum(ArithmeticOperation):
    def __init__(
        self,
        left: Tensor[int | float | bool] | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(MinimumOp(left=left, right=right), name=name)


class Maximum(ArithmeticOperation):
    def __init__(
        self,
        left: Tensor[int | float | bool] | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(MaximumOp(left=left, right=right), name=name)


class Divide(OperatorModel):
    numerator: Connection
    denominator: Connection
    output: Connection

    def __init__(
        self,
        numerator: Tensor[int | float | bool]
        | int
        | float
        | bool
        | ToBeDetermined = TBD,
        denominator: Tensor[int | float | bool]
        | int
        | float
        | bool
        | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        m = DivideOp(numerator=numerator, denominator=denominator)
        super().__init__(name=name, model=m)

    def connect(  # type: ignore[override]
        self,
        numerator: ConnectionType
        | Tensor[int | float | bool]
        | int
        | float
        | bool = NOT_GIVEN,
        denominator: ConnectionType
        | Tensor[int | float | bool]
        | int
        | float
        | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            numerator=numerator, denominator=denominator, output=output
        )


class FloorDivide(OperatorModel):
    numerator: Connection
    denominator: Connection
    output: Connection

    def __init__(
        self,
        numerator: Tensor[int | float | bool]
        | int
        | float
        | bool
        | ToBeDetermined = TBD,
        denominator: Tensor[int | float | bool]
        | int
        | float
        | bool
        | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        m = FloorDivideOp(numerator=numerator, denominator=denominator)
        super().__init__(name=name, model=m)

    def connect(  # type: ignore[override]
        self,
        numerator: ConnectionType
        | Tensor[int | float | bool]
        | int
        | float
        | bool = NOT_GIVEN,
        denominator: ConnectionType
        | Tensor[int | float | bool]
        | int
        | float
        | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            numerator=numerator, denominator=denominator, output=output
        )


class MatrixMultiply(OperatorModel):
    left: Connection
    right: Connection
    output: Connection

    def __init__(
        self,
        left: Tensor[int | float] | ToBeDetermined = TBD,
        right: Tensor[int | float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=MatrixMultiplyOp(left=left, right=right))

    def connect(  # type: ignore[override]
        self,
        left: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        right: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        output: ConnectionType | Tensor[int | float] = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(left=left, right=right, output=output)


class Shape(OperatorModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=ShapeOp(input=input))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Reshape(OperatorModel):
    input: Connection
    shape: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int | None, ...] | list[int] | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=ReshapeOp(shape=shape, input=input))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        shape: ConnectionType | Sequence[int | None | ConnectionType] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, shape=shape, output=output)


class Length(OperatorModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LengthOp(input=input))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Cast(OperatorModel):
    input: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self, dtype: types.Dtype | ToBeDetermined = TBD, *, name: str | None = None
    ) -> None:
        super().__init__(name=name, model=CastOp(dtype=dtype))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        dtype: ConnectionType | types.Dtype = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, dtype=dtype, output=output)


class Dtype(OperatorModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=DtypeOp(input=input))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Size(OperatorModel):
    input: Connection
    dim: Connection
    output: Connection

    def __init__(
        self,
        dim: int | tuple[int, ...] | None | ToBeDetermined = None,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=SizeOp(input=input, dim=dim))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        dim: ConnectionType | int | tuple[int | ConnectionType, ...] | None = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, dim=dim, output=output)


class Item(OperatorModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=ItemOp(input=input))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


ToTensorConnectionType = (
    ConnectionType | int | float | bool | Sequence["ToTensorConnectionType"]
)


class ToTensor(OperatorModel):
    input: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        input: TensorValueType | ToBeDetermined = TBD,
        dtype: types.Dtype | ToBeDetermined | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"dtype": dtype}
        super().__init__(name=name, model=ToTensorOp(input=input, dtype=dtype))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | ToTensorConnectionType = NOT_GIVEN,
        dtype: ConnectionType | types.Dtype = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, dtype=dtype, output=output)


class ToList(OperatorModel):
    output: Connection

    def __init__(
        self,
        n: int,
        *,
        name: str | None = None,
        **kwargs: ScalarValueType | ToBeDetermined,
    ) -> None:
        super().__init__(name=name, model=ToListOp(n, name=name, **kwargs))

    def connect(self, **kwargs: ConnectionType) -> ExtendInfo:  # type: ignore[override]
        return super().connect(**kwargs)


class TensorToList(OperatorModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self._enforce_jit = False
        m = TensorToListOp(input=input)
        m._enforce_jit = False
        super().__init__(name=name, model=m)

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Reduce(OperatorModel):
    input: Connection
    axis: Connection
    keepdim: Connection
    output: Connection

    def __init__(self, model: Operator, *, name: str | None = None) -> None:
        super().__init__(name=name, model=model)

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        axis: ConnectionType
        | int
        | tuple[int | ConnectionType, ...]
        | None = NOT_GIVEN,
        keepdim: ConnectionType | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, axis=axis, keepdim=keepdim, output=output)


class Mean(Reduce):
    def __init__(
        self,
        axis: int | tuple[int, ...] | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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
        input: Tensor[float] | ToBeDetermined = TBD,
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
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis, "keepdim": keepdim}
        super().__init__(MaxOp(axis=axis, keepdim=keepdim, input=input), name=name)


class ArgMax(Reduce):
    def __init__(
        self,
        axis: int | None | ToBeDetermined = None,
        keepdim: bool | ToBeDetermined = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
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
        input: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"axis": axis, "keepdim": keepdim, "correction": correction}
        super().__init__(
            VarianceOp(axis=axis, keepdim=keepdim, input=input, correction=correction),
            name=name,
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        axis: ConnectionType | int | tuple[int | ConnectionType, ...] = NOT_GIVEN,
        keepdim: ConnectionType | bool = NOT_GIVEN,
        correction: ConnectionType | float = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super(Reduce, self).connect(
            input=input,
            axis=axis,
            keepdim=keepdim,
            correction=correction,
            output=output,
        )


class SingleInputModel(OperatorModel):
    input: Connection
    output: Connection

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType
        | Tensor[int | float | bool]
        | int
        | float
        | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Absolute(SingleInputModel):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=AbsoluteOp(input=input))


class Negate(SingleInputModel):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=NegateOp(input=input))


class Exponential(SingleInputModel):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=ExponentialOp(input=input))


class Sqrt(OperatorModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        threshold: ConstantType
        | Tensor[float]
        | ToBeDetermined = types.Constant.MIN_POSITIVE_NORMAL,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"threshold": threshold, "robust": robust}
        self.robust = robust

        if threshold is not types.Constant.MIN_POSITIVE_NORMAL and not robust:
            raise ValueError("threshold must be specified in robust mode")
        # TODO: get rid of this tensor cast.
        thd = Tensor(threshold) if not isinstance(threshold, Tensor) else threshold
        m = SqrtOp(robust=robust, input=input, threshold=thd)
        super().__init__(name=name, model=m)

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        threshold: ConnectionType | Tensor[float] = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output}

        if not self.robust:
            if threshold is not NOT_GIVEN:
                raise ValueError("threshold must not be specified in robust mode off")
        else:
            kwargs["threshold"] = threshold

        return super().connect(**kwargs)


class RelationalModel(OperatorModel):
    left: Connection
    right: Connection
    output: Connection

    def connect(  # type: ignore[override]
        self,
        left: ConnectionType
        | Tensor[int | float | bool]
        | int
        | float
        | bool = NOT_GIVEN,
        right: ConnectionType
        | Tensor[int | float | bool]
        | int
        | float
        | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(left=left, right=right, output=output)


class Greater(RelationalModel):
    def __init__(
        self,
        left: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=GreaterOp(left=left, right=right))


class Less(RelationalModel):
    def __init__(
        self,
        left: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LessOp(left=left, right=right))


class Equal(RelationalModel):
    def __init__(
        self,
        left: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=EqualOp(left=left, right=right))


class NotEqual(RelationalModel):
    def __init__(
        self,
        left: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=NotEqualOp(left=left, right=right))


class LessEqual(RelationalModel):
    def __init__(
        self,
        left: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LessEqualOp(left=left, right=right))


class GreaterEqual(RelationalModel):
    def __init__(
        self,
        left: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        right: Tensor[int | float | bool] | int | float | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=GreaterEqualOp(left=left, right=right))


class LogicalNot(OperatorModel):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LogicalNotOp(input=input))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class BitwiseOperators(OperatorModel):
    left: Connection
    right: Connection
    output: Connection

    def connect(  # type: ignore[override]
        self,
        left: ConnectionType | Tensor[int | bool] | int | bool = NOT_GIVEN,
        right: ConnectionType | Tensor[int | bool] | int | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(left=left, right=right, output=output)


class LogicalAnd(BitwiseOperators):
    def __init__(
        self,
        left: Tensor[int | bool] | int | bool | ToBeDetermined = TBD,
        right: Tensor[int | bool] | int | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LogicalAndOp(left=left, right=right))


class LogicalOr(BitwiseOperators):
    def __init__(
        self,
        left: Tensor[int | bool] | int | bool | ToBeDetermined = TBD,
        right: Tensor[int | bool] | int | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LogicalOrOp(left=left, right=right))


class LogicalXOr(BitwiseOperators):
    def __init__(
        self,
        left: Tensor[int | bool] | int | bool | ToBeDetermined = TBD,
        right: Tensor[int | bool] | int | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=LogicalXOrOp(left=left, right=right))


class ShiftLeft(OperatorModel):
    input: Connection
    shift: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | bool] | int | bool | ToBeDetermined = TBD,
        shift: Tensor[int | bool] | int | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=ShiftLeftOp(input=input, shift=shift))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | bool] | int | bool = NOT_GIVEN,
        shift: ConnectionType | Tensor[int | bool] | int | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, shift=shift, output=output)


class ShiftRight(OperatorModel):
    input: Connection
    shift: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | bool] | int | bool | ToBeDetermined = TBD,
        shift: Tensor[int | bool] | int | bool | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=ShiftRightOp(input=input, shift=shift))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | bool] | int | bool = NOT_GIVEN,
        shift: ConnectionType | Tensor[int | bool] | int | bool = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, shift=shift, output=output)


class Transpose(OperatorModel):
    # NOTE: Consider if axes type list[int] is conventionally True since it is generally
    # used tuple[int] in these type of cases
    input: Connection
    axes: Connection
    output: Connection

    def __init__(
        self,
        axes: int | list[int] | tuple[int, ...] | None | ToBeDetermined = None,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=TransposeOp(input=input, axes=axes))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        axes: ConnectionType | int | Sequence[int | ConnectionType] | None = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, axes=axes, output=output)


class Slice(OperatorModel):
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

    def connect(  # type: ignore[override]
        self,
        start: ConnectionType | int | None = NOT_GIVEN,
        stop: ConnectionType | int | None = NOT_GIVEN,
        step: ConnectionType | int | None = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(start=start, stop=stop, step=step, output=output)


class Indexer(OperatorModel):
    input: Connection
    index: Connection
    output: Connection

    def __init__(
        self,
        index: int
        | slice
        | EllipsisType
        | None
        | tuple[int | slice | EllipsisType | None, ...]
        | ToBeDetermined = TBD,
        input: Tensor[int | float | bool] | Sequence[Any] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=IndexerOp(input=input, index=index))

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] | Sequence[Any] = NOT_GIVEN,
        index: ConnectionType
        | int
        | slice
        | EllipsisType
        | None
        | tuple[int | slice | EllipsisType | None | ConnectionType, ...] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, index=index, output=output)


class Sine(SingleInputModel):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=SineOp(input=input))


class Cosine(SingleInputModel):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=CosineOp(input=input))


class AtLeast1D(SingleInputModel):
    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, model=AtLeast1DOp(input=input))
